# Adapted from: https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pprint
import pathlib

SAVE_DIR = pathlib.Path("./autoencoder-checkpoints")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Uses a straight-through estimator (STE) along with a configurable baseline to calculate sparse features.
class QSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        d_hidden = cfg["sparse_dims"]
        d_model = cfg["llm_dims"]
        self.dtype = dtype = DTYPES[cfg["enc_dtype"]]

        torch.manual_seed(cfg["seed"])

        self.topk = cfg["topk"]
        self.quantile = self.topk / d_model

        # subtract this before doing autoencoding
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(d_model, dtype=dtype))

        self.encode = nn.Linear(d_model, d_hidden, bias=True, dtype=dtype)
        self.decode = nn.Linear(d_hidden, d_model, bias=False, dtype=dtype)

        # kaiming initialization
        nn.init.kaiming_uniform_(self.encode.weight, a=0)
        nn.init.kaiming_uniform_(self.decode.weight, a=0)
    
    def forward(self, x):
        # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        # scores = self.encode((x - self.bias) / self.scale)
        scores = self.encode(x)
        scores = F.sigmoid(scores)

        # quantize the activations along feature dimension
        # add noise to ensure that features don't collapse
        # noise = torch.randn_like(scores)
        # noise_coeff = 0 # 1e-6
        quantiles = torch.quantile(scores.float().detach(), 1 - self.quantile, dim=-1, interpolation='lower')
        activations = (scores >= quantiles.unsqueeze(-1)).to(self.dtype)

        quantization_error = (scores - activations).pow(2).mean()

        # straight-through estimator trick: add `scores` and subtract a detached version of itself. the net result is `acts`,
        # but the gradient only goes to `scores`.
        x_reconstructed = self.decode(scores + (activations - scores).detach())
        # x_reconstructed = x_reconstructed * self.scale + self.bias

        reconstruction_error = (x_reconstructed.float() - x.float()).pow(2).mean()

        return x_reconstructed, scores, activations, quantization_error, reconstruction_error
