import os
import runpy
import sys

import torch
import transformer_engine.pytorch.module.linear as linear

original = linear.general_gemm

def trace(*args, **kwargs):
    try:
        return original(*args, **kwargs)
    except RuntimeError:
        out = kwargs.get('out')
        print('TRACE GEMM FAILURE',
              'rank', os.environ.get('RANK'),
              'operands', [(str(getattr(x, 'shape', None)), type(x).__name__) for x in args[:2]],
              'out_shape', tuple(out.shape) if isinstance(out, torch.Tensor) else None,
              'out_stride', tuple(out.stride()) if isinstance(out, torch.Tensor) else None,
              'out_ptr_mod_256', out.data_ptr() % 256 if isinstance(out, torch.Tensor) else None,
              'dtype', kwargs.get('out_dtype'), 'accumulate', kwargs.get('accumulate'),
              flush=True)
        raise

linear.general_gemm = trace
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
