"""Record actual GEMM storage and Inductor kernel choices; delegate all numerical work."""
import importlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

output, expected, target, *arguments = sys.argv[1:]
counts = Counter()
kernels = {}
for name in ('linear', 'layernorm_linear', 'layernorm_mlp'):
    module = importlib.import_module('transformer_engine.pytorch.module.' + name)
    original = module.general_gemm

    def gemm(*args, _original=original, **kwargs):
        left = args[0] if args else kwargs['A']
        right = args[1] if len(args) > 1 else kwargs['B']
        counts[type(left).__name__ + '/' + type(right).__name__] += 1
        return _original(*args, **kwargs)

    module.general_gemm = gemm
original_run = CachingAutotuner.run


def run(self, *args, **kwargs):
    result = original_run(self, *args, **kwargs)
    if self.fn.__name__.startswith('l2norm_'):
        config = self.launchers[0].config
        key = self.fn.__name__ + ':' + str(self.filename)
        kernels[key] = dict(kwargs=config.kwargs, num_warps=config.num_warps,
                            num_stages=config.num_stages)
    return result


CachingAutotuner.run = run
sys.argv = [target, *arguments]
try:
    runpy.run_path(target, run_name='__main__')
finally:
    Path(output).write_text(json.dumps(dict(expected_storage=expected,
        gemm_operand_types=dict(counts), inductor_kernels=kernels), indent=2) + '\n')
assert any(expected in key for key in counts), dict(counts)
