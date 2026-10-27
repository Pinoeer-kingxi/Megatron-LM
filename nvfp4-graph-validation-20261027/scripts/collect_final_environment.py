"""Record actual qualification runtime versions and unmodified TE source hashes."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
import transformer_engine.pytorch.graph as graph
import transformer_engine.pytorch.module.linear as linear
import flash_attn_2_cuda

source = os.environ['GDN_SOURCE']
result = dict(python=platform.python_version(), cuda=torch.version.cuda,
    source=source, revision=subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip(),
    torch_cuda_initialized=torch.cuda.is_initialized(),
    packages={name: importlib.metadata.version(name) for name in
        ['torch', 'transformer-engine', 'triton', 'flash-linear-attention',
         'fla-core', 'flash-attn', 'causal-conv1d', 'transformers']},
    te_graph_sha256=hashlib.sha256(Path(graph.__file__).read_bytes()).hexdigest(),
    te_linear_sha256=hashlib.sha256(Path(linear.__file__).read_bytes()).hexdigest(),
    flash_attn_sm120_library_sha256=hashlib.sha256(Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest(),
    recipe_environment={name: os.environ.get(name) for name in
        ['NVTE_CPU_OFFLOAD_V1', 'NVTE_NVFP4_DISABLE_RHT', 'NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING']})
Path('/home/lwx/.cache/megatron-gdn-20261026/nvfp4-graph-environment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
