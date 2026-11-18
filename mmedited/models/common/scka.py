"""Geometry-aware selective K/V aggregation for CiaoSR.

Inspired by SAT (Tran et al., arXiv:2604.07994), specifically
basicsr/archs/sat_arch.py::cluster_and_merge and SAA. This is a new
implementation, not an exact reproduction. Retain upstream license notices
when distributing the parent CiaoSR/SAT projects.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveKVAttention(nn.Module):
    """Replace CrossScaleAttention; return [B, C * len(scale), H, W].

    Full LR queries attend to compressed multi-scale source tokens. The
    original CiaoSR decoder subsequently handles all continuous HR queries.
    Geometry is approximate: spatially bounded clusters retain a centroid
    and variance. Target cell conditions geometric bias, not token count.
    """

    def __init__(self, channel=64, scale=(2,), keep_ratio=0.25,
                 query_chunk=512):
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError('keep_ratio must be in (0, 1]')
        if not isinstance(query_chunk, int) or query_chunk < 1:
            raise ValueError('query_chunk must be a positive integer')
        self.scales = tuple(scale)
        if not self.scales or any(int(s) != s or s < 1 for s in self.scales):
            raise ValueError('scale must contain positive integers')
        self.keep_ratio = keep_ratio
        # Fixed local support: avoid aggressive long-distance merging.
        self.region_size = 8
        self.subsample_factor = 4
        self.query_chunk = query_chunk
        self.temperature = 0.2
        dim = max(8, channel // 2)
        self.q_proj = nn.Conv2d(channel, dim, 1)
        self.k_proj = nn.Linear(channel, dim)
        self.v_proj = nn.Linear(channel, channel)
        self.out_proj = nn.Conv2d(channel, channel, 1)
        # Trainable positive distance coefficient; gentle initial locality.
        self.geometry_strength = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def _grid(h, w, device):
        y = (torch.arange(h, device=device, dtype=torch.float32) + .5) * (2 / h) - 1
        x = (torch.arange(w, device=device, dtype=torch.float32) + .5) * (2 / w) - 1
        return torch.stack((y[:, None].expand(h, w), x[None, :].expand(h, w)), -1)

    def _merge(self, x, pos):
        """Hard density routing; weighted feature merging remains differentiable."""
        b, n, c = x.shape
        k = min(n, max(1, math.ceil(n * self.keep_ratio)))
        if k == n:
            return x, pos, torch.zeros_like(pos), x.new_ones(b, n, 1)
        # Deterministic stratified candidates also at inference; no randperm.
        with torch.no_grad():
            xn = F.normalize(x.detach(), dim=-1, eps=1e-6)
            count = min(n, max(2 * k, self.subsample_factor * k))
            idx = ((torch.arange(count, device=x.device) + .5) * n / count).long()
            candidates = xn[:, idx]
            sim = candidates @ candidates.transpose(1, 2)
            diag = torch.eye(count, device=x.device, dtype=torch.bool)[None]
            neighbors = sim.masked_fill(diag, -2).topk(min(5, count - 1), dim=-1).values
            density = ((neighbors.mean(-1) + 1) / 2).clamp_min(1e-6)
            order = torch.arange(count, device=x.device)
            higher = density[:, None, :] > density[:, :, None]
            ties = density[:, None, :] == density[:, :, None]
            higher = higher | (ties & (order[None, :] < order[:, None])[None])
            distance = (1 - sim).clamp_min(0)
            delta = distance.masked_fill(~higher, float('inf')).min(-1).values
            delta = torch.where(higher.any(-1), delta, distance.max(-1).values)
            center_idx = idx[(density * delta).topk(k, dim=-1).indices]
            centers = xn.gather(1, center_idx[..., None].expand(b, k, c))
            similarity = xn @ centers.transpose(1, 2)
            assignment = similarity.argmax(-1)
            # Every selected center owns itself, even for constant features.
            assignment.scatter_(1, center_idx, torch.arange(k, device=x.device)[None].expand(b, k))
        selected = x.gather(1, center_idx[..., None].expand(b, k, c))
        selected = selected.gather(1, assignment[..., None].expand(b, n, c))
        cosine = F.cosine_similarity(x, selected, dim=-1, eps=1e-6)
        weight = ((cosine - 1) / self.temperature).clamp(min=-30, max=0).exp()[..., None]

        def accumulate(values):
            out = values.new_zeros(b, k, values.shape[-1])
            return out.scatter_add(1, assignment[..., None].expand_as(values), values)

        denom = accumulate(weight).clamp_min(1e-8)
        merged = accumulate(x * weight) / denom
        mean_pos = accumulate(pos * weight) / denom
        variance = (accumulate(pos.square() * weight) / denom - mean_pos.square()).clamp_min(0)
        mass = accumulate(torch.ones_like(weight))
        # A convex mean avoids artificial feature-amplitude amplification.
        return merged, mean_pos, variance, mass

    def _bank(self, feature, positions=None):
        b, c, h, w = feature.shape
        grid = (self._grid(h, w, feature.device) if positions is None else positions)
        grid = grid[None].expand(b, h, w, 2)
        groups = []
        for top in range(0, h, self.region_size):
            for left in range(0, w, self.region_size):
                patch = feature[:, :, top:top+self.region_size, left:left+self.region_size]
                pos = grid[:, top:top+self.region_size, left:left+self.region_size].reshape(b, -1, 2)
                groups.append(self._merge(patch.flatten(2).transpose(1, 2), pos))
        return tuple(torch.cat([g[i] for g in groups], dim=1) for i in range(4))

    def forward(self, feature, cell=None):
        dtype = feature.dtype
        # Float32 clustering/softmax also under legacy MMEditing CUDA AMP.
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            context = torch.amp.autocast(feature.device.type, enabled=False)
        else:
            context = torch.cuda.amp.autocast(enabled=False)
        with context:
            return self._forward_float(feature.float(), cell).to(dtype)

    def _forward_float(self, feature, cell):
        b, c, h, w = feature.shape
        if cell is None or cell.ndim != 3 or cell.shape[0] != b or cell.shape[-1] != 2 or cell.shape[1] == 0:
            raise ValueError('cell must be [B, Q, 2] for a regular target pixel grid')
        # CiaoSR already assumes constant cell per image; heterogeneous batches work.
        if not torch.isfinite(cell).all() or (cell <= 0).any():
            raise ValueError('cell entries must be positive and finite')
        if not torch.allclose(cell, cell[:, :1].expand_as(cell)):
            raise ValueError('cell must be constant within each target image')
        target_cell = cell[:, :1].to(device=feature.device, dtype=torch.float32)
        q = self.q_proj(feature).flatten(2).transpose(1, 2)
        pos_q = self._grid(h, w, feature.device).reshape(1, h*w, 2)
        results = []
        for s in self.scales:
            sh, sw = max(1, math.ceil(h / s)), max(1, math.ceil(w / s))
            source = F.adaptive_avg_pool2d(feature, (sh, sw))
            # Adaptive pooling has nonuniform support for odd image sizes.
            # Pool coordinates with the same operator, rather than inventing
            # a uniformly spaced grid for those source tokens.
            source_pos = F.adaptive_avg_pool2d(
                pos_q.reshape(h, w, 2).permute(2, 0, 1)[None], (sh, sw)
            )[0].permute(1, 2, 0)
            tokens, centers, variance, mass = self._bank(source, source_pos)
            keys, values = self.k_proj(tokens), self.v_proj(tokens)
            # Area/variance prior: centroids of broad groups are less precise.
            base_cell = feature.new_tensor([2 / sh, 2 / sw]).view(1, 1, 2)
            denom = variance + base_cell.square() + target_cell.square()
            parts = []
            for start in range(0, h*w, self.query_chunk):
                stop = min(start + self.query_chunk, h*w)
                logits = q[:, start:stop] @ keys.transpose(1, 2) / math.sqrt(keys.shape[-1])
                offset = pos_q[:, start:stop, None] - centers[:, None]
                distance = (offset.square() / denom[:, None]).sum(-1)
                logits = logits - F.softplus(self.geometry_strength) * torch.log1p(distance)
                # Approximate multiplicity prior; exact only for identical
                # keys/values within each cluster, not a general error bound.
                logits = logits + mass.squeeze(-1).clamp_min(1).log()[:, None]
                parts.append(logits.softmax(-1) @ values)
            out = torch.cat(parts, dim=1).transpose(1, 2).reshape(b, c, h, w)
            results.append(self.out_proj(out))
        return torch.cat(results, dim=1)
