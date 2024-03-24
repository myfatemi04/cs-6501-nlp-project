# Divide-and-conquer approach to ablation.
# This is called "neuron vis", but can really be used for anything involving a score assigned to each token.
# In this case, visualize logprob
from neuron_visualization import basic_neuron_vis, basic_neuron_vis_signed
from hooked_phi import attach_hooks, detach_hooks
import tqdm
import torch

def test():
    # Get granular nll estimates
    display(Markdown(f'# Calculating Neuron Responsibilities'))

    # 1. Get ground truth.
    detach_hooks(model.model)
    input_ids, input_and_completion_ids, ground_truth = generate(model, tokenizer, "5 times 8 equals", do_display=False, max_new_tokens=3)
    completion_ids = input_and_completion_ids[len(input_ids):]

    print(tokenizer.decode(input_and_completion_ids))

    # Calculate baseline logprobs.
    predictions_for_next_token = model(input_ids=input_and_completion_ids.unsqueeze(0))
    logits = predictions_for_next_token[0][0]
    logits_for_output_tokens = logits[len(input_ids) - 1:-1]
    logprobs_for_output_tokens = torch.log_softmax(logits_for_output_tokens, dim=-1)
    baseline_logprobs_for_sampled_output_tokens = logprobs_for_output_tokens[
        torch.arange(logprobs_for_output_tokens.shape[0]),
        completion_ids
    ]

    # 2. Ablate model. See which suddens suddenly become highly unlikely (by visualizing negative logprobs).

    branching_factors = [32, 32, 10]

def get_logprobs_for_completion(logits, tokens, last_n):
    logits_for_output_tokens = logits[len(tokens) - last_n - 1:-1]
    logprobs_for_output_tokens = torch.log_softmax(logits_for_output_tokens, dim=-1)
    # Select logprobs for specified last n tokens.
    completion_ids = tokens[-last_n:]
    logprobs_for_sampled_output_tokens = logprobs_for_output_tokens[
        torch.arange(logprobs_for_output_tokens.shape[0]),
        completion_ids
    ]
    return logprobs_for_sampled_output_tokens

def get_deflections(layer_id, start_neuron, end_neuron, branch_depth, tokens, last_n, baseline_activations=None):
    # Calculate and cache baseline activations.
    if baseline_activations is None:
        logits = model(input_ids=tokens.unsqueeze(0))[0][0]
        baseline_activations = get_logprobs_for_completion(logits, tokens, last_n)
    
    # For each layer in this coalition, simulate its removal.
    mask = torch.ones((model.config.num_hidden_layers, model.config.intermediate_size), dtype=torch.bool)
    if num_ablation_layers > 0:
        mask[layer_id, start_neuron, end_neuron] = False

    # Make the ablation.
    attach_hooks(model.model, ablate_neurons(mask))
    # Get predictions.
    logits = model(input_ids=tokens.unsqueeze(0))[0][0]
    # Detach model hooks.
    detach_hooks(model.model)
    
    ablated_activations = get_logprobs_for_completion(logits, tokens, last_n)
    
    deflections = ablated_activations - baseline_activations
    
    return deflections

def get_difference_gradient(model, prefix, expected, measured):
    # TODO: Add a kv cache.
    cache = []
    def hook(mlp_neurons, _layer_idx):
        # Can freeze the model while logging these gradients.
        mlp_neurons.requires_grad_(True)
        mlp_neurons.retain_grad()
        cache.append(mlp_neurons)
        return mlp_neurons
    
    detach_hooks(model.model)
    attach_hooks(model.model, hook)
    logits = model(input_ids=prefix)[0][0]
    detach_hooks(model.model)
    
    logprobs = torch.log_softmax(logits, dim=-1)
    difference = logprobs[-1][expected[0]] - logprobs[-1][measured[0]]
    difference.backward()
    
    gradients = [mlp_neurons.grad for mlp_neurons in cache]
    return gradients
