"""Persistent local DINOv3 worker, isolated from the legacy SR environment.

The worker uses only local source/weights. No image is sent over a network.
It is not an nn.Module: teacher weights never enter SR optimizers/checkpoints.
"""
import atexit
import io
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def write_frame(stream, **arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    stream.write(struct.pack('<Q', len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError('DINOv3 worker closed its pipe; inspect worker stderr')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def read_frame(stream):
    size = struct.unpack('<Q', _read_exact(stream, 8))[0]
    if size > 1024 ** 3:
        raise ValueError('invalid DINOv3 protocol frame size')
    with np.load(io.BytesIO(_read_exact(stream, size)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class DinoV3Teacher:
    def __init__(self, repo_dir, weights, python=None, device='auto',
                 image_size=192, layer=11, micro_batch=1, **unused):
        if unused:
            raise ValueError('unknown teacher settings: {}'.format(sorted(unused)))
        self.repo_dir = str(Path(repo_dir).expanduser().resolve())
        self.weights = str(Path(weights).expanduser().resolve())
        if not (Path(self.repo_dir) / 'hubconf.py').is_file():
            raise FileNotFoundError('Missing official DINOv3 source: ' + self.repo_dir)
        if not Path(self.weights).is_file():
            raise FileNotFoundError('Missing local DINOv3 weights: ' + self.weights)
        if image_size < 32 or image_size % 16 or not 0 <= layer < 24 or micro_batch < 1:
            raise ValueError('image_size must be a multiple of 16 >=32; layer 0..23; micro_batch >=1')
        self.python = python or sys.executable
        self.device = device
        self.image_size = int(image_size)
        self.layer = int(layer)
        self.micro_batch = int(micro_batch)
        self.process = None
        atexit.register(self.close)

    def _start(self, image_device):
        if self.process is not None:
            return
        project = Path(__file__).resolve().parents[3]
        device = str(image_device) if self.device == 'auto' else self.device
        command = [self.python, '-u', str(project / 'tools/dinov3_teacher_worker.py'),
                   '--repo', self.repo_dir, '--weights', self.weights,
                   '--device', device, '--image-size', str(self.image_size),
                   '--layer', str(self.layer), '--micro-batch', str(self.micro_batch)]
        options = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        if sys.platform == 'win32':
            options['creationflags'] = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen(command, **options)
        response = read_frame(self.process.stdout)
        if 'error' in response:
            self.close()
            raise RuntimeError(str(response['error']))
        if 'ready' not in response:
            self.close()
            raise RuntimeError('unexpected DINOv3 worker handshake')

    def __call__(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('teacher expects BCHW RGB in [0,1]')
        self._start(rgb.device)
        write_frame(self.process.stdin, rgb=rgb.detach().float().cpu().numpy())
        response = read_frame(self.process.stdout)
        if 'error' in response:
            raise RuntimeError('DINOv3 worker: ' + str(response['error']))
        feature = torch.from_numpy(response['features'].copy()).to(rgb.device)
        if not torch.isfinite(feature).all():
            raise FloatingPointError('non-finite DINOv3 features')
        return feature

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdout:
                process.stdout.close()
