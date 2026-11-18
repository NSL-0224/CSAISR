"""Expand baseline weights with ONLY the new initialized branch/counter.

Use the SR environment. Does not silently accept mismatched backbone weights.
No optimizer state is copied: this is fine-tuning, not resuming baseline training.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmcv import Config
from mmedit.models import build_model


def expand_state(source, target):
    source = {(k[7:] if k.startswith('module.') else k): v for k, v in source.items()}
    # Support a generator-only checkpoint as well as a restorer checkpoint.
    if source and not any(k.startswith('generator.') for k in source):
        source = {'generator.' + k: v for k, v in source.items()}
    unexpected = set(source) - set(target)
    if unexpected:
        raise ValueError('Unexpected checkpoint keys: ' + str(sorted(unexpected)[:20]))
    missing = set(target) - set(source)
    allowed = {k for k in missing if k.startswith('generator.directional_detail.')
               or k == 'guidance_step'}
    if missing != allowed:
        raise ValueError('Missing baseline keys: ' + str(sorted(missing - allowed)[:20]))
    for key, value in source.items():
        if target[key].shape != value.shape:
            raise ValueError('Shape mismatch for ' + key)
    result = dict(target)
    result.update(source)
    return result, sorted(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('output')
    parser.add_argument('--config', default='configs/002_directional_detail.py')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    # Load only your own/trusted training checkpoint; old MMCV uses this format.
    checkpoint = torch.load(args.baseline, map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint)
    state, initialized = expand_state(source, model.state_dict())
    model.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state,
                    meta=dict(source=str(Path(args.baseline).resolve()),
                              initialized=initialized, optimizer_reset=True)), output)
    print('Saved:', output)
    print('New initialized entries:', initialized)
    print('Original parameters copied exactly. Use as load_from, not resume_from.')


if __name__ == '__main__':
    main()
