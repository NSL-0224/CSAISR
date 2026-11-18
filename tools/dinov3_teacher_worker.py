"""Run with the DINOv3 Python environment, NOT necessarily the SR Python."""
import argparse
import contextlib
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmedited.models.common.dinov3_teacher import read_frame, write_frame


def load_model(repo, weights, device):
    # Official hub entry creates the architecture. Avoid URL loading entirely.
    with contextlib.redirect_stdout(sys.stderr):
        model = torch.hub.load(str(Path(repo).resolve()), 'dinov3_vitl16',
                               source='local', pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        model.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(model, rgb, device, image_size, layer, micro_batch):
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('invalid RGB input')
    if rgb.min() < -1e-4 or rgb.max() > 1.0001:
        raise ValueError('teacher input must be original [0,1] RGB, not SR-normalized input')
    outputs = []
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    for start in range(0, len(rgb), micro_batch):
        image = torch.from_numpy(rgb[start:start + micro_batch]).to(device)
        image = F.interpolate(image.float(), size=(image_size, image_size),
                              mode='bilinear', align_corners=False, antialias=True)
        image = (image - mean) / std
        with contextlib.redirect_stdout(sys.stderr):
            # Spatial tokens only; excludes CLS/storage tokens by official API.
            feature = model.get_intermediate_layers(image, n=[layer], reshape=True)[0]
        outputs.append(feature.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--image-size', type=int, default=192)
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    output = sys.stdout.buffer
    try:
        model = load_model(args.repo, args.weights, args.device)
        if args.check_only:
            features = extract_features(model, np.full((1, 3, 48, 48), .5, np.float32),
                                        args.device, args.image_size, args.layer, args.micro_batch)
            print('Loaded local ViT-L/16; spatial features:', features.shape,
                  '; frozen:', all(not p.requires_grad for p in model.parameters()))
            return
        write_frame(output, ready=np.array(True))
        while True:
            try:
                request = read_frame(sys.stdin.buffer)
            except EOFError:
                break
            features = extract_features(model, request['rgb'], args.device,
                                        args.image_size, args.layer, args.micro_batch)
            write_frame(output, features=features)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        if not args.check_only:
            write_frame(output, error=np.array(str(error)))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
