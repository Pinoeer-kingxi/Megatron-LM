"""Exercise stock TE graph replay without Megatron, GDN, DDP or offloading."""
import json
import sys

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

width = int(sys.argv[1])
recipe = NVFP4BlockScaling()
torch.manual_seed(31)
model = te.Linear(256, width, bias=False, params_dtype=torch.bfloat16, device='cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
sample = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        model(sample).float().square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize()
graph = te.make_graphed_callables(model, (sample,), num_warmup_iters=3,
    fp8_enabled=True, fp8_recipe=recipe, fp8_weight_caching=True)
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    value = torch.randn_like(sample, requires_grad=True)
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        loss = graph(value, is_first_microbatch=True).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(model.weight.grad).all()
    optimizer.step()
    print(json.dumps(dict(width=width, step=step, loss=float(loss.detach()))), flush=True)
