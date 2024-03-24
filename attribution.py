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

def ablate_neurons(mask):
    # assert mask.shape[0] == model.config.num_hidden_layers
    # assert mask.shape[1] == model.config.intermediate_size

    def hook(neurons, layer_idx):
        neurons[..., ~mask[layer_idx]] = 0
        return neurons

    return hook
    
def get_logprobs_for_completion(logits, tokens, last_n):
    logits_for_output_tokens = logits[len(tokens[0]) - last_n - 1:-1]
    logprobs_for_output_tokens = torch.log_softmax(logits_for_output_tokens, dim=-1)
    # Select logprobs for specified last n tokens.
    completion_ids = tokens[-last_n:]
    logprobs_for_sampled_output_tokens = logprobs_for_output_tokens[
        torch.arange(logprobs_for_output_tokens.shape[0]),
        completion_ids
    ][0, -last_n:]
    return logprobs_for_sampled_output_tokens

def get_deflections(model, layer_id, selection, tokens, last_n, baseline_activations):
    # For each layer in this coalition, simulate its removal.
    mask = torch.ones((model.config.num_hidden_layers, model.config.intermediate_size), dtype=torch.bool)
    mask[layer_id, selection] = False

    # Make the ablation.
    attach_hooks(model.model, ablate_neurons(mask))
    # Get predictions.
    logits = model(input_ids=tokens)[0][0]
    # Detach model hooks.
    detach_hooks(model.model)
    
    ablated_activations = get_logprobs_for_completion(logits, tokens, last_n)
    
    deflections = ablated_activations - baseline_activations
    
    return deflections

def search_neuron_buckets(model, layer_id, bucket_size, tokens, last_n, trials=32):
    logits = model(input_ids=tokens)[0][0]
    baseline_activations = get_logprobs_for_completion(logits, tokens, last_n)
    
    total_deflection = torch.zeros(10240, dtype=torch.float, device="cuda")
    
    for sampling_stage in range(trials):
        perm = torch.randperm(10240)
        print("Sampling stage", sampling_stage)
        for start_neuron in range(0, 10240, bucket_size):
            selection = perm[start_neuron:start_neuron + bucket_size]
            total_deflection[selection] += get_deflections(model, layer_id, selection, tokens, last_n, baseline_activations)
    
    return total_deflection / trials

def search_neuron_buckets_bucketified(model, layer_id, bucket_size, tokens, last_n):
    logits = model(input_ids=tokens)[0][0]
    baseline_activations = get_logprobs_for_completion(logits, tokens, last_n)
    
    total_deflection = torch.zeros(10240, dtype=torch.float, device="cuda")
    
    perm = torch.randperm(10240)
    for start_neuron in range(0, 10240, bucket_size):
        selection = torch.arange(start_neuron, start_neuron + bucket_size)
        total_deflection[selection] += get_deflections(model, layer_id, selection, tokens, last_n, baseline_activations)
    
    return total_deflection
    
    # cutoff_value = torch.quantile(total_deflection, 0.90)
    # neurons = torch.where(total_deflection > cutoff_value)[0]
    # return neurons

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
