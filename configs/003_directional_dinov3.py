"""Same SR settings as 002; DINOv3 runs in a separate Python process."""
import os

_base_ = './002_directional_detail.py'
exp_name = '003_directional_dinov3'
model = dict(teacher=dict(
    enabled=True,
    repo_dir=os.environ.get('DINOV3_REPO', './third_party/dinov3'),
    weights=os.environ.get('DINOV3_WEIGHTS',
        './external_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
    python=os.environ.get('DINOV3_PYTHON') or None,
    device=os.environ.get('DINOV3_DEVICE', 'auto'),
    image_size=192, layer=11, micro_batch=1,
    loss_weight=.01, interval=4, warmup=500, ramp=1000, max_grid=8))
work_dir = './work_dirs/' + exp_name
test_checkpoint_path = work_dir + '/latest.pth'
