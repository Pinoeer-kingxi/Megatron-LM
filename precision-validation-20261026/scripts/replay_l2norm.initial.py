"""Replay the original FLA kernel at actually observed configs on identical captured inputs."""
import json
from pathlib import Path

import torch
import triton
from fla.modules.l2norm import l2norm_bwd_kernel

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
captured=torch.load(base/'l2norm-private/captured.pt',map_location='cuda',weights_only=True)
y,rstd,dy=(captured[name] for name in ('y','rstd','dy'))
tokens,width=y.shape
configs=[]
for label in ('seed0','seed1','seed0-repeat'):
    path=base/f'cache/inductor-gpu1-{label}/53/bcd6eab639b21c99958c953f21f90830bca4806a837864b7185e29d388299276.best_config'
    item=json.loads(path.read_text())
    config={name:item[name] for name in ('BT','num_warps','num_stages')}
    if config not in configs: configs.append(config)
if captured['config'] not in configs: configs.append(captured['config'])
reference=(dy.double()*rstd.double()[:,None]-(dy.double()*y.double()).sum(-1)[:,None]*y.double()*rstd.double()[:,None]).to(torch.bfloat16)
outputs=[]
rows=[]
for config in configs:
    repetitions=[]
    for _ in range(3):
        dx=torch.empty_like(y)
        l2norm_bwd_kernel.fn[(triton.cdiv(tokens,config['BT']),)](
            y,rstd,dy,dx,1e-6,tokens,D=width,BD=triton.next_power_of_2(width),
            NB=triton.cdiv(tokens,2048*32),**config)
        repetitions.append(dx)
    assert all(torch.equal(repetitions[0],other) for other in repetitions[1:])
    value=repetitions[0]
    error=(value.float()-reference.float()).abs()
    rows.append(dict(config=config,repeat_exact=True,max_absolute_vs_double_reference=float(error.max()),
                     different_from_double_reference=int((value!=reference).sum()),
                     matches_captured_inductor_dx=torch.equal(value,captured['actual_dx'])))
    outputs.append(value)
comparisons=[]
for i in range(len(outputs)):
    for j in range(i+1,len(outputs)):
        delta=(outputs[i].float()-outputs[j].float()).abs()
        comparisons.append(dict(left=i,right=j,different_elements=int((outputs[i]!=outputs[j]).sum()),max_absolute=float(delta.max())))
result=dict(shape=list(y.shape),dtype=str(y.dtype),reference_max_absolute=float(reference.float().abs().max()),
            variants=rows,comparisons=comparisons)
(base/'l2norm-configuration-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
