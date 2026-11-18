"""New decoder class with no extra branch: exact baseline control."""
import os
from mmedited.models.backbones.sr_backbones.ciaosr_directional_net import DirectionalImplicitSRRDN

_base_ = './004_baseline_finetune.py'
exp_name = '005_directional_disabled'
model = dict(generator=dict(type=DirectionalImplicitSRRDN,
                            directional=dict(enabled=False)))
work_dir = './work_dirs/' + exp_name
load_from = os.environ.get('CSASR_BASELINE_CHECKPOINT') or None
test_checkpoint_path = work_dir + '/latest.pth'
