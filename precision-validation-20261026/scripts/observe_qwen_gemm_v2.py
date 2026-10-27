"""Observe real TE GEMM storage without replacing any numerical implementation."""
import importlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import torch

output, expected, target, *arguments = sys.argv[1:]
counts = Counter()
for module_name in ('linear', 'layernorm_linear', 'layernorm_mlp'):
    module = importlib.import_module('transformer_engine.pytorch.module.' + module_name)
    original = module.general_gemm

    def observe(*args, _original=original, **kwargs):
        left = args[0] if args else kwargs['A']
        right = args[1] if len(args) > 1 else kwargs['B']
        counts[type(left).__name__ + '/' + type(right).__name__] += 1
        return _original(*args, **kwargs)

    module.general_gemm = observe
sys.argv = [target, *arguments]
try:
    runpy.run_path(target, run_name='__main__')
finally:
    Path(output).write_text(json.dumps(dict(expected_storage=expected, gemm_operand_types=dict(counts)), indent=2) + '\n')
assert any(expected in storage for storage in counts), dict(counts)
