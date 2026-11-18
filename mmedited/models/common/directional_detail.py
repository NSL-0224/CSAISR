"""Fixed-neighbour directional reconstruction and relation distillation.

Neighbour order is (top-left, top-right, bottom-left, bottom-right), as
produced by the decoder's nested (y, x) sampling loop. These are feature
differences, NOT image derivatives or distance-normalized gradients.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DirectionalDetail(nn.Module):
    def __init__(self, channels, structure_dim=32, hidden_dim=64,
                 max_strength=0.1):
        super().__init__()
        if min(channels, structure_dim, hidden_dim) <= 0:
            raise ValueError('directional dimensions must be positive')
        if not math.isfinite(max_strength) or max_strength <= 0:
            raise ValueError('max_strength must be finite and positive')
        self.max_strength = float(max_strength)
        self.structure = nn.Sequential(
            nn.Conv2d(channels, structure_dim, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(structure_dim, structure_dim, 1))
        # Structure + four actual relative offsets (y,x) + LR-unit cell (y,x).
        self.coefficients = nn.Sequential(
            nn.Linear(structure_dim + 10, hidden_dim),
            nn.ReLU(inplace=False), nn.Linear(hidden_dim, 3))
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        self.register_buffer('basis', torch.tensor([
            [-1., 1., -1., 1.],   # horizontal difference
            [-1., -1., 1., 1.],   # vertical difference
            [1., -1., -1., 1.],   # mixed/cross difference, NOT diagonal derivative
        ]) / 4.)

    def forward(self, values, structure, offsets, cell):
        """values B,Q,4,D; structure B,Q,S; offsets B,Q,4,2; cell B,Q,2."""
        if values.shape[-2] != 4 or offsets.shape[-2:] != (4, 2):
            raise ValueError('directional detail requires exactly four neighbours')
        condition = torch.cat((structure, offsets.flatten(-2), cell), dim=-1)
        coefficient = self.max_strength * torch.tanh(self.coefficients(condition))
        differences = torch.einsum('kj,bqjd->bqkd', self.basis.to(values), values)
        return (coefficient.unsqueeze(-1) * differences).sum(-2)


def relation_distillation_loss(student, teacher, max_grid=8):
    """Compare spatial cosine relations; channel dimensions need not match.

    Both maps cover the SAME resized LR crop. Pooling is limited to the native
    teacher grid: interpolation must not invent teacher boundary resolution.
    Off-diagonal relations are used; the teacher is always detached.
    """
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError('student/teacher must be BCHW maps')
    if student.shape[0] != teacher.shape[0]:
        raise ValueError('student/teacher batch mismatch')
    if max_grid < 2:
        raise ValueError('max_grid must be at least 2')
    height = min(student.shape[-2], teacher.shape[-2], max_grid)
    width = min(student.shape[-1], teacher.shape[-1], max_grid)
    if height * width < 2:
        raise ValueError('teacher grid must contain at least two spatial tokens')
    def affinity(feature):
        pooled = F.adaptive_avg_pool2d(feature.float(), (height, width))
        tokens = F.normalize(pooled.flatten(2), dim=1, eps=1e-6)
        return tokens.transpose(1, 2) @ tokens
    source = affinity(student)
    target = affinity(teacher.detach().to(student.device))
    mask = ~torch.eye(height * width, device=student.device, dtype=torch.bool)
    return F.smooth_l1_loss(source[:, mask], target[:, mask])
