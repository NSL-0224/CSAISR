"""Actual restorer train_step with synthetic teacher; no real teacher/GPU claim.

Requires the CPU math test's RDN source plus CIAOSR_BASE_SOURCE pointing to
MMEditing 0.11 models/base.py (or inferred from the RDN source tree).
"""
import runpy
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from pathlib import Path
import os
import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
audit = runpy.run_path(str(ROOT / 'tests/test_directional_cpu.py'))
env = audit['env']
base_path = os.environ.get('CIAOSR_BASE_SOURCE')
if not base_path:
    base_path = str(Path(audit['rdn_source']).parents[2] / 'base.py')
env.update(ABCMeta=ABCMeta, abstractmethod=abstractmethod, OrderedDict=OrderedDict,
           numbers=numbers, math=math, psnr=None, ssim=None,
           auto_fp16=lambda **kwargs: lambda function: function)
audit['extract'](base_path, ['BaseModel'], env)


def build_backbone(cfg):
    cfg = dict(cfg)
    cls = cfg.pop('type')
    return cls(**cfg)


env.update(build_backbone=build_backbone, build_loss=lambda cfg: nn.L1Loss())
audit['extract'](ROOT / 'mmedited/models/restorers/basic_restorer.py', ['BasicRestorer'], env)
audit['extract'](ROOT / 'mmedited/models/restorers/ciaosr.py', ['CiaoSR'], env)
audit['extract'](ROOT / 'mmedited/models/restorers/ciaosr_directional.py', ['CiaoSRDirectional'], env)
# The decoder builders are looked up at execution time; dispatch both types.
original_builder = audit['build']
def dispatch(cfg):
    if isinstance(cfg['type'], str):
        return original_builder('RDN', cfg)
    return build_backbone(cfg)
env['build_backbone'] = dispatch


class SyntheticTeacher:
    """Only exercises RGB plumbing, schedule and gradients, not DINO quality."""
    def __init__(self):
        self.calls = 0
    def __call__(self, rgb):
        assert 0 <= rgb.min() <= rgb.max() <= 1
        self.calls += 1
        return F.adaptive_avg_pool2d(torch.cat([rgb, rgb.square()], dim=1), (4, 4))


cfg = audit['copy'].deepcopy(audit['cfg'])
cfg.update(type=audit['Updated'], directional=dict(enabled=True, structure_dim=6, hidden_dim=12))
restorer = env['CiaoSRDirectional'](
    generator=cfg, pixel_loss=dict(type='L1Loss'),
    teacher=dict(enabled=True, warmup=1, interval=2, ramp=1))
restorer.teacher_client = SyntheticTeacher()
optimizer = torch.optim.Adam(restorer.parameters(), lr=1e-4)
data = dict(lq=torch.rand(2, 3, 7, 9), gt=torch.rand(2, audit['query'].shape[1], 3),
            coord=audit['query'], cell=audit['cell'])
active = []
for _ in range(4):
    result = restorer.train_step(data, optimizer)
    active.append(result['log_vars']['dino_active'])
    assert 'loss_pix' in result['log_vars']
    assert 'loss_dino_relation' in result['log_vars']
assert active == [0., 1., 0., 1.] and restorer.teacher_client.calls == 2
assert restorer.guidance_step.item() == 4
assert not any('teacher' in key for key in restorer.state_dict())
assert restorer.generator.directional_detail.coefficients[-1].weight.abs().sum() > 0
print('PASS actual BaseModel/BasicRestorer/CiaoSR train_step: guided schedule, raw RGB, '
      'loss keys, counter, learning; synthetic teacher excluded from state_dict')
