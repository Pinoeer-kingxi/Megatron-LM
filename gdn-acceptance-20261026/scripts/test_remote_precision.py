"""Probe actual TE GEMM forward/backward support, independently of offloading."""
import argparse
import json
from pathlib import Path
import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, NVFP4BlockScaling

parser = argparse.ArgumentParser()
parser.add_argument('recipe', choices=['fp8-delayed', 'fp8-tensorwise', 'fp4-default', 'fp4-sm120'])
parser.add_argument('output', type=Path)
args = parser.parse_args()
recipes = {'fp8-delayed': DelayedScaling, 'fp8-tensorwise': Float8CurrentScaling,
           'fp4-default': NVFP4BlockScaling,
           'fp4-sm120': lambda: NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)}
recipe = recipes[args.recipe]()
torch.manual_seed(31)
model = te.Linear(512, 512, params_dtype=torch.bfloat16).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
samples = []
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    value = torch.randn(128, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        result = model(value)
        loss = result.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    samples.append(dict(step=step, loss=float(loss.detach()), input_grad_norm=float(value.grad.float().norm())))
torch.cuda.synchronize()
args.output.write_text(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
    te=transformer_engine.__version__, device=torch.cuda.get_device_name(),
    capability=torch.cuda.get_device_capability(), recipe=repr(recipe), samples=samples), indent=2)+'\n')
print(args.output.read_text())
