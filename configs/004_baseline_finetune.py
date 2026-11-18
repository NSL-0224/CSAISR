"""Same 20k fine-tuning budget as 002/003, original decoder, no teacher."""
import os

_base_ = './001_ciaosr_rdn_div2k_patch_kv.py'
exp_name = '004_baseline_finetune'
optimizers = dict(type='Adam', lr=1.e-5)
total_iters = 20000
lr_config = dict(policy='Step', by_epoch=False, step=[10000, 15000], gamma=.5)
checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1000, save_image=False, gpu_collect=True)
work_dir = './work_dirs/' + exp_name
load_from = os.environ.get('CSASR_BASELINE_CHECKPOINT') or None
resume_from = None
test_checkpoint_path = work_dir + '/latest.pth'
