"""Record the actual pinned runtime and whether its imports initialize CUDA."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch
import transformer_engine.pytorch.graph as te_graph
import flash_attn_2_cuda

package_names = ['torch', 'transformer-engine', 'triton', 'flash-linear-attention',
                 'fla-core', 'flash-attn', 'causal-conv1d', 'transformers']
result = dict(python=platform.python_version(), cuda=torch.version.cuda,
    packages={name: importlib.metadata.version(name) for name in package_names},
    source=os.environ['GDN_SOURCE'], torch_cuda_initialized=torch.cuda.is_initialized(),
    te_graph_sha256=hashlib.sha256(Path(te_graph.__file__).read_bytes()).hexdigest(),
    te_graph_official_url='https://raw.githubusercontent.com/NVIDIA/TransformerEngine/v2.20.2/transformer_engine/pytorch/graph.py',
    flash_attn_sm120_library_sha256=hashlib.sha256(Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest())
Path('/home/lwx/.cache/megatron-gdn-20261026/precision-graph-environment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
