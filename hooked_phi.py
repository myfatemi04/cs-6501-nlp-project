import math

import torch
import torch.nn as nn
import transformers
from transformers import Cache, PhiModel
from transformers.models.phi.modeling_phi import apply_rotary_pos_emb, repeat_kv, PhiAttention, PhiDecoderLayer, PhiMLP
from typing import Optional, Tuple


def attach_hooks(phi: PhiModel, mlp_post_activation_hook, attn_weights_hook):
    MLP_LAYER_IDXS = {}
    ATTENTION_LAYER_IDXS = {}

    # In `PhiMLP`, `self` -> `phi_mlp`.
    def hooked_mlp_forward(phi_mlp: PhiMLP, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = phi_mlp.fc1(hidden_states)
        hidden_states = phi_mlp.activation_fn(hidden_states)

        ### HOOK INJECTION ###
        layer_idx = MLP_LAYER_IDXS[phi_mlp]

        if mlp_post_activation_hook is not None:
            hidden_states = mlp_post_activation_hook(hidden_states, layer_idx)

        hidden_states = phi_mlp.fc2(hidden_states)
        return hidden_states
    
    # From transformers.models.phi.modeling_phi#305
    # In `PhiAttention`, `self` -> `phi_attn`
    def hooked_attn_forward(
        phi_attn: PhiAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = phi_attn.q_proj(hidden_states)
        key_states = phi_attn.k_proj(hidden_states)
        value_states = phi_attn.v_proj(hidden_states)

        if phi_attn.qk_layernorm:
            query_states = phi_attn.q_layernorm(query_states)
            key_states = phi_attn.k_layernorm(key_states)

        query_states = query_states.view(bsz, q_len, phi_attn.num_heads, phi_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, phi_attn.num_key_value_heads, phi_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, phi_attn.num_key_value_heads, phi_attn.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if phi_attn.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {phi_attn.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, phi_attn.layer_idx)
        cos, sin = phi_attn.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : phi_attn.rotary_emb.dim],
            query_states[..., phi_attn.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : phi_attn.rotary_emb.dim],
            key_states[..., phi_attn.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": phi_attn.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, phi_attn.layer_idx, cache_kwargs) # type: ignore

        key_states = repeat_kv(key_states, phi_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, phi_attn.num_key_value_groups)

        # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        attn_weights = torch.matmul(
            query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
        ) / math.sqrt(phi_attn.head_dim)

        if attn_weights.size() != (bsz, phi_attn.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, phi_attn.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=phi_attn.attention_dropout, training=phi_attn.training)

        ### HOOK INJECTION: Modify attention weights ###
        layer_idx = ATTENTION_LAYER_IDXS[phi_attn]
        attn_weights = attn_weights_hook(attn_weights, layer_idx)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, phi_attn.num_heads, q_len, phi_attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, phi_attn.num_heads, q_len, phi_attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, phi_attn.hidden_size)

        attn_output = phi_attn.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value # type: ignore
    
    # Add hooks.
    for (layer_idx, decoder_layer) in enumerate(phi.layers):
        assert isinstance(decoder_layer, PhiDecoderLayer)

        mlp = decoder_layer.mlp
        attn = decoder_layer.self_attn

        assert isinstance(mlp, PhiMLP)
        assert isinstance(attn, PhiAttention)

        MLP_LAYER_IDXS[mlp] = layer_idx
        mlp.forward = hooked_mlp_forward.__get__(mlp, PhiMLP)

        ATTENTION_LAYER_IDXS[attn] = layer_idx
        attn.forward = hooked_attn_forward.__get__(attn, PhiAttention)

