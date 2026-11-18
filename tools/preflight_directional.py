"""Read-only server preflight before spending GPU time."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from mmcv import Config
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--allow-scratch', action='store_true')
    parser.add_argument('--check-teacher', action='store_true')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    print('SR Python:', sys.executable)
    print('PyTorch:', torch.__version__, '; CUDA:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requires a working CUDA-enabled SR environment')
    if not cfg.get('load_from') and not args.allow_scratch:
        raise ValueError('Set the initial checkpoint environment variable; use --allow-scratch only intentionally')
    if cfg.get('load_from') and not Path(cfg.load_from).is_file():
        raise FileNotFoundError(cfg.load_from)
    for path in [cfg.data.train.dataset.gt_folder, cfg.data.val.gt_folder]:
        if not Path(path).is_dir():
            raise FileNotFoundError('Dataset directory missing: ' + path)
    teacher = cfg.model.get('teacher', {})
    if teacher.get('enabled', False):
        for path in [Path(teacher['weights']), Path(teacher['repo_dir']) / 'hubconf.py']:
            if not path.is_file():
                raise FileNotFoundError(str(path))
        if args.check_teacher:
            device = teacher.get('device', 'auto')
            if device == 'auto':
                device = 'cuda:0'
            subprocess.run([
                teacher.get('python') or sys.executable,
                str(Path(__file__).with_name('dinov3_teacher_worker.py')),
                '--repo', teacher['repo_dir'], '--weights', teacher['weights'],
                '--device', device, '--image-size', str(teacher.get('image_size', 192)),
                '--layer', str(teacher.get('layer', 11)), '--check-only'], check=True)
    print('Preflight passed. Work directory:', cfg.work_dir)
    print('Runner automatically resumes existing latest.pth; use a new work-dir for a fresh comparison.')


if __name__ == '__main__':
    main()
