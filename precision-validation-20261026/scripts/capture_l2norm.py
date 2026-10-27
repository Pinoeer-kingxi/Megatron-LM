"""Capture one real Qwen L2-normalization backward for private numerical diagnosis."""
import json
import runpy
import sys
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

output,target,*arguments=sys.argv[1:]
saved=False
original=CachingAutotuner.run

def run(self,*args,**kwargs):
    global saved
    take=not saved and self.fn.__name__=='l2norm_bwd_kernel'
    if take:
        values=[arg.detach().cpu().clone() for arg in args[:3]]
    result=original(self,*args,**kwargs)
    if take:
        config=self.launchers[0].config
        Path(output).parent.mkdir(exist_ok=True)
        torch.save(dict(y=values[0],rstd=values[1],dy=values[2],actual_dx=args[3].detach().cpu(),
                        config=dict(BT=config.kwargs['BT'],num_warps=config.num_warps,num_stages=config.num_stages)),output)
        saved=True
    return result

CachingAutotuner.run=run
sys.argv=[target,*arguments]
runpy.run_path(target,run_name='__main__')
assert saved
