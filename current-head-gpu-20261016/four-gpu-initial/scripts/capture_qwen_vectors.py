"""Capture small actual tensors alongside unmodified Qwen state fingerprints."""

import os
from pathlib import Path
import sys

import torch

# Execute the committed tool with only evidence capture added at its snapshot boundary.
from tools.ssm import qwen_gdn_offload_training as training

fingerprint = training._fingerprint
vectors = {}

def capture(value):
    result = fingerprint(value)
    if isinstance(value, torch.Tensor) and value.ndim <= 1:
        key = (result['dtype'], tuple(result['shape']), result['sha256'])
        vectors[key] = value.detach().cpu().clone()
    return result

training._fingerprint = capture
training.main()
output = Path(sys.argv[sys.argv.index('--output') + 1]).parent
rank = int(os.environ['RANK'])
torch.save(vectors, output / f'vectors.rank{rank}.pt')
