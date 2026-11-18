"""Binary local-process transport audit using synthetic data, not DINOv3."""
import io
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmedited.models.common.dinov3_teacher import read_frame, write_frame

buffer = io.BytesIO()
source = np.random.default_rng(7).random((2, 3, 8, 8), dtype=np.float32)
write_frame(buffer, rgb=source)
buffer.seek(0)
np.testing.assert_array_equal(read_frame(buffer)['rgb'], source)
try:
    read_frame(io.BytesIO(b''))
    raise AssertionError('EOF was not detected')
except EOFError:
    pass

program = '''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from mmedited.models.common.dinov3_teacher import read_frame, write_frame
write_frame(sys.stdout.buffer, ready=np.array(True))
data = read_frame(sys.stdin.buffer)['rgb']
write_frame(sys.stdout.buffer, features=data.mean(axis=1, keepdims=True))
'''
process = subprocess.Popen([sys.executable, '-u', '-c', program, str(ROOT)],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    assert read_frame(process.stdout)['ready'].item()
    write_frame(process.stdin, rgb=source)
    np.testing.assert_array_equal(read_frame(process.stdout)['features'], source.mean(axis=1, keepdims=True))
    process.stdin.close()
    assert process.wait(timeout=15) == 0
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
    process.stdout.close()
print('PASS binary framing, exact array roundtrip, EOF detection and actual subprocess pipe')
