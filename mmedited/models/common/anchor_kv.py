"""Anchor-preserving selective local K/V refinement for CiaoSR.

Inspired by selective K/V aggregation, not a reproduction of SAT clustering.
No query coordinates, four-neighbor slots, or non-local features are changed.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class AnchorKVRefinement(nn.Module):
    """Shared 3x3 soft selection; bounded, zero-initialized K/V corrections.

    The two signed gates allow small interpolation/extrapolation, not just
    convex smoothing. They are global learned scalars, not confidence scores.
    Heavy descriptor aggregation is performed once on the LR grid.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.match_proj = nn.Conv2d(channels, min(32, channels), 1, bias=False)
        self.gate_k = nn.Parameter(torch.zeros(()))
        self.gate_v = nn.Parameter(torch.zeros(()))
        self.temperature = 0.2

    def forward(self, feature, descriptor):
        if feature.ndim != 4 or descriptor.ndim != 4:
            raise ValueError('feature and descriptor must be BCHW tensors')
        if feature.shape[0] != descriptor.shape[0] or feature.shape[-2:] != descriptor.shape[-2:]:
            raise ValueError('feature and descriptor must share batch/spatial sizes')
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            context = torch.amp.autocast(feature.device.type, enabled=False)
        elif feature.is_cuda:
            context = torch.cuda.amp.autocast(enabled=False)
        else:
            context = nullcontext()
        # Keep cosine/softmax/gates stable under mixed precision. Assumes
        # standard float32 parameters with autocast, not manual model.half().
        with context:
            _, _, h, w = feature.shape
            match = F.normalize(self.match_proj(feature.float()), dim=1, eps=1e-6)
            padded_match = F.pad(match, (1, 1, 1, 1))
            valid = F.pad(feature.new_ones((1, 1, h, w)), (1, 1, 1, 1))
            logits = []
            for dy in range(3):
                for dx in range(3):
                    neighbor = padded_match[:, :, dy:dy+h, dx:dx+w]
                    score = (match * neighbor).sum(1, keepdim=True) / self.temperature
                    mask = valid[:, :, dy:dy+h, dx:dx+w] > 0
                    logits.append(score.masked_fill(~mask, float('-inf')))
            weights = torch.cat(logits, dim=1).softmax(dim=1)
            source = descriptor.float()
            padded_source = F.pad(source, (1, 1, 1, 1))
            # Difference form is mathematically sum(a_ij U_j) - U_i.
            # It preserves constants exactly and avoids a BCHW*9 descriptor
            # unfold allocation. Invalid boundary positions receive zero weight.
            delta = torch.zeros_like(source)
            index = 0
            for dy in range(3):
                for dx in range(3):
                    neighbor = padded_source[:, :, dy:dy+h, dx:dx+w]
                    delta = delta + weights[:, index:index+1] * (neighbor - source)
                    index += 1
            key = source + 0.1 * self.gate_k.tanh() * delta
            value = source + 0.05 * self.gate_v.tanh() * delta
        return key.to(descriptor.dtype), value.to(descriptor.dtype)
