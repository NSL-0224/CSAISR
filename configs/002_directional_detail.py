"""Directional branch only: first ablation, no DINO dependency."""
import os
from mmedited.models.backbones.sr_backbones.ciaosr_directional_net import DirectionalImplicitSRRDN
from mmedited.models.restorers.ciaosr_directional import CiaoSRDirectional

_base_ = './001_ciaosr_rdn_div2k_patch_kv.py'
exp_name = '002_directional_detail'
model = dict(
    type=CiaoSRDirectional,
    generator=dict(type=DirectionalImplicitSRRDN, local_size=2,
                   directional=dict(enabled=True, structure_dim=32,
                                    hidden_dim=64, max_strength=.1)),
    teacher=dict(enabled=False))
# These are short fine-tuning starting settings, not validated hyperparameters.
optimizers = dict(type='Adam', lr=1.e-5)
total_iters = 20000
lr_config = dict(policy='Step', by_epoch=False, step=[10000, 15000], gamma=.5)
checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1000, save_image=False, gpu_collect=True)
work_dir = './work_dirs/' + exp_name
load_from = os.environ.get('DIRECTIONAL_INIT_CHECKPOINT') or None
resume_from = None
test_checkpoint_path = work_dir + '/latest.pth'
