"""CPU math audit of real decoder/MLP/RDN classes, without importing MMCV ops.

Install the SR dependencies or set CIAOSR_RDN_SOURCE to the original MMEditing
0.11 rdn.py. This does NOT test the MMCV runner, CUDA/DDP or real DINO weights.
"""
import ast
import copy
import importlib.util
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
torch.manual_seed(20261118)


def extract(path, names, env):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert set(names) == {n.name for n in nodes}
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)


def coords(shape, flatten=True):
    axes = [(torch.arange(s) + .5) * 2 / s - 1 for s in shape]
    result = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
    return result.reshape(-1, 2) if flatten else result


env = dict(torch=torch, nn=nn, F=F, math=math, nullcontext=nullcontext, make_coord=coords)
extract(ROOT / 'mmedited/models/common/directional_detail.py',
        ['DirectionalDetail', 'relation_distillation_loss'], env)
Detail, relation_loss = env['DirectionalDetail'], env['relation_distillation_loss']
detail = Detail(8, structure_dim=6, hidden_dim=12)
assert torch.equal(detail.basis.sum(-1), torch.zeros(3))
v = torch.randn(2, 7, 1, 11).expand(-1, -1, 4, -1)
s = torch.randn(2, 7, 6)
o, ce = torch.randn(2, 7, 4, 2), torch.ones(2, 7, 2) * .2
with torch.no_grad():
    detail.coefficients[-1].weight.normal_()
    detail.coefficients[-1].bias.normal_()
assert torch.equal(detail(v, s, o, ce), torch.zeros(2, 7, 11))
ramps = torch.tensor([0., 1., 0., 1.]).view(1, 1, 4, 1)
response = torch.einsum('kj,bqjd->bqkd', detail.basis, ramps)
torch.testing.assert_close(response.flatten(), torch.tensor([.5, 0., 0.]))
print('PASS stencil order, zero-sum kernels, constant cancellation, horizontal response')

student = torch.randn(2, 6, 8, 10, requires_grad=True)
teacher = torch.randn(2, 19, 4, 5, requires_grad=True)
loss = relation_loss(student, teacher, max_grid=4)
loss.backward()
assert student.grad is not None and student.grad.abs().sum() > 0
assert teacher.grad is None and torch.isfinite(student.grad).all()
torch.testing.assert_close(relation_loss(teacher.detach(), teacher.detach()), torch.tensor(0.))
print('PASS relation loss: different channel dimensions, student gradients, teacher detached')

rdn_source = os.environ.get('CIAOSR_RDN_SOURCE')
if not rdn_source:
    spec = importlib.util.find_spec('mmedit')
    if spec is None:
        raise RuntimeError('Install mmedit or set CIAOSR_RDN_SOURCE to its original rdn.py')
    rdn_source = str(Path(next(iter(spec.submodule_search_locations))) / 'models/backbones/sr_backbones/rdn.py')
extract(rdn_source, ['DenseLayer', 'RDB', 'RDN'], env)
extract(ROOT / 'mmedited/models/components/refiners/mlp_refiner.py', ['MLPRefiner'], env)
extract(ROOT / 'mmedited/models/common/arch_csnln.py',
        ['default_conv', 'BasicBlock', 'same_padding', 'reduce_sum',
         'extract_image_patches', 'CrossScaleAttention'], env)
extract(ROOT / 'mmedited/models/common/patch_kv.py', ['PatchKVAttention'], env)


def build(name, cfg):
    cfg = dict(cfg)
    cfg.pop('type')
    return env[name](**cfg)


env.update(build_backbone=lambda cfg: build('RDN', cfg),
           build_component=lambda cfg: build('MLPRefiner', cfg))
extract(ROOT / 'mmedited/models/backbones/sr_backbones/ciaosr_patch_kv_net.py',
        ['LocalImplicitSRNet', 'LocalImplicitSRRDN'], env)
Baseline = env['LocalImplicitSRRDN']
env['BaselineRDN'] = Baseline
extract(ROOT / 'mmedited/models/backbones/sr_backbones/ciaosr_directional_net.py',
        ['DirectionalImplicitSRRDN'], env)
Updated = env['DirectionalImplicitSRRDN']
cfg = dict(encoder=dict(type='RDN', in_channels=3, out_channels=3, mid_channels=8,
                       num_blocks=2, upscale_factor=4, num_layers=2, channel_growth=8),
           imnet_q=dict(type='MLPRefiner', in_dim=4, out_dim=3, hidden_list=[16, 16]),
           imnet_k=dict(type='MLPRefiner', in_dim=8, out_dim=8, hidden_list=[16, 16]),
           imnet_v=dict(type='MLPRefiner', in_dim=8, out_dim=8, hidden_list=[16, 16]),
           patch_kv=dict(keep_ratio=.5, geometry=False, reject_thresholds=(.5, .5)),
           eval_bsize=11)
baseline = Baseline(**copy.deepcopy(cfg))
disabled = Updated(**copy.deepcopy(cfg), directional=dict(enabled=False))
disabled.load_state_dict(baseline.state_dict(), strict=True)
model = Updated(**copy.deepcopy(cfg), directional=dict(enabled=True, structure_dim=6, hidden_dim=12))
incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
assert all(k.startswith('directional_detail.') for k in incompatible.missing_keys)
assert not incompatible.unexpected_keys
for shape in [(2, 3, 7, 9), (1, 3, 8, 8)]:
    image = torch.randn(shape)
    query = coords((shape[-2]*2, shape[-1]*2))[None][:, ::5].expand(shape[0], -1, -1)
    cell = torch.ones_like(query) * torch.tensor([1/shape[-2], 1/shape[-1]])
    with torch.no_grad():
        reference = baseline(image, query, cell)
        torch.testing.assert_close(disabled(image, query, cell), reference, atol=0, rtol=0)
        torch.testing.assert_close(model(image, query, cell), reference, atol=0, rtol=0)
print('PASS actual RDN+KVCA decoder: disabled/zero-init exact baseline, odd/even image sizes')

image = torch.randn(2, 3, 7, 9)
query = coords((14, 18))[None][:, ::7].expand(2, -1, -1)
cell = torch.ones_like(query) * torch.tensor([2/14, 2/18])
pred, structure = model(image, query, cell, return_aux=True)
objective = pred.square().mean() + .01*relation_loss(structure, torch.randn(2, 13, 4, 4))
objective.backward()
assert model.directional_detail.coefficients[-1].weight.grad.abs().sum() > 0
assert model.directional_detail.structure[0].weight.grad.abs().sum() > 0
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('PASS first-step gradients to detail output and teacher-guided structure branch')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for _ in range(3):
    optimizer.zero_grad()
    pred, structure = model(image, query, cell, return_aux=True)
    loss = pred.square().mean() + .01*relation_loss(structure, torch.randn(2, 13, 4, 4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
with torch.no_grad():
    direct = model(image, query, cell)
    chunked = model(image, query, cell, test_mode=True)
    torch.testing.assert_close(direct, chunked, atol=3e-6, rtol=3e-5)
    assert (direct - baseline(image, query, cell)).abs().max() > 0
print('PASS finite optimizer steps, learned branch effect, chunked/direct parity')

model.eval()
with torch.no_grad():
    pairs = []
    for upscale in [2., 2.5, 4., 8.]:
        ce = torch.full_like(cell, 2/(7*upscale))
        pairs.append(model(image, query, ce))
    assert all(torch.isfinite(p).all() for p in pairs)
    assert (pairs[0] - pairs[-1]).abs().max() > 0
print('PASS finite integer/noninteger/out-of-range cell queries (not dataset PSNR)')

extract(ROOT / 'tools/prepare_directional_checkpoint.py', ['expand_state'], env)
full_source = {'generator.' + k: v for k, v in baseline.state_dict().items()}
full_target = {'generator.' + k: v for k, v in model.state_dict().items()}
full_target['guidance_step'] = torch.zeros((), dtype=torch.long)
expanded, missing = env['expand_state'](full_source, full_target)
assert missing and all(torch.equal(expanded[k], value) for k, value in full_source.items())
try:
    bad = dict(full_source); bad.pop(next(iter(bad)))
    env['expand_state'](bad, full_target)
    raise AssertionError('missing backbone weights were silently accepted')
except ValueError:
    pass
print('PASS strict warm-start expansion and rejection of missing baseline parameters')

# Full 64-channel / 16-block RDN checks actual 576-query / 640-value dimensions.
full_cfg = copy.deepcopy(cfg)
full_cfg['encoder'].update(mid_channels=64, num_blocks=16, num_layers=8, channel_growth=64)
for name in ['imnet_q', 'imnet_k', 'imnet_v']:
    full_cfg[name]['hidden_list'] = [256]*4
full = Updated(**full_cfg, directional=dict(enabled=True)).eval()
with torch.no_grad():
    full_image = torch.randn(1, 3, 48, 48)
    full_query = coords((192, 192))[None][:, ::144]
    full_cell = torch.full_like(full_query, 2/192)
    full_pred = full(full_image, full_query, full_cell)
    assert full_pred.shape == (1, 256, 3) and torch.isfinite(full_pred).all()
print('PASS full RDN 64x16: 48x48 LR, 256 x4 queries, directional parameters:',
      sum(p.numel() for p in full.directional_detail.parameters()))
print('All CPU math audits passed; runner/CUDA/DDP/real teacher/PSNR are NOT validated.')
