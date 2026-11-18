"""Clone official source and optionally check a local checkpoint. No weights download."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(project / 'third_party/dinov3'))
    parser.add_argument('--ref', default=None, help='Optional official git tag/commit to pin')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', default=str(project / 'external_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'))
    args = parser.parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dinov3.git', str(repo)], check=True)
    elif not (repo / 'hubconf.py').is_file():
        raise ValueError('Existing directory is not a DINOv3 source tree: ' + str(repo))
    if args.ref:
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', args.ref], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    print('DINOv3 source revision:', revision, flush=True)
    print('Record this revision with your experiment. Dependencies must match official source.', flush=True)
    if args.check:
        subprocess.run([sys.executable, str(project / 'tools/dinov3_teacher_worker.py'),
                        '--repo', str(repo), '--weights', args.weights,
                        '--device', args.device, '--check-only'], check=True)


if __name__ == '__main__':
    main()
