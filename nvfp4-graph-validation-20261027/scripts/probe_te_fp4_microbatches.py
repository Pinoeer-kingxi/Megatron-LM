"""Isolate cached NVFP4 weights across real captured microbatches, without Megatron."""
import json
import sys

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

kind, width, caching = sys.argv[1], int(sys.argv[2]), bool(int(sys.argv[3]))
torch.manual_seed(31)
recipe = NVFP4BlockScaling()
kwargs = dict(bias=False, params_dtype=torch.bfloat16, device='cuda')
if kind == 'norm':
    model = te.LayerNormLinear(256, width, normalization='RMSNorm',
                               zero_centered_gamma=True, **kwargs)
else:
    model = te.Linear(256, width, **kwargs)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
samples = tuple((torch.randn(128, 1, 256, device='cuda', dtype=torch.bfloat16,
                             requires_grad=True),) for _ in range(4))
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    for micro, (value,) in enumerate(samples):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            model(value, is_first_microbatch=(micro == 0) if caching else None).float().square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize()
graphs = te.make_graphed_callables((model,), samples, num_warmup_iters=3,
    fp8_enabled=True, fp8_recipe=recipe, fp8_weight_caching=caching,
    _order=[1, -1] * 4, _num_layers_per_chunk=[1], _reuse_graph_input_output_buffers=True)
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    for micro, graph in enumerate(graphs):
        value = torch.randn_like(samples[micro][0], requires_grad=True)
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = graph(value, is_first_microbatch=(micro == 0))
            loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters()), (step, micro)
        print(json.dumps(dict(kind=kind, width=width, caching=caching,
                              step=step, micro=micro, loss=float(loss.detach()))), flush=True)
    optimizer.step()
