import sys
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

in_features, out_features, fused, offset = map(int, sys.argv[1:])
torch.manual_seed(42)
layer = te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16,
                  device='cuda', fuse_wgrad_accumulation=bool(fused))
if fused:
    storage = torch.zeros(layer.weight.numel() + offset, device="cuda", dtype=torch.float32)
    layer.weight.main_grad = storage[offset:].view_as(layer.weight)
    print("main_grad pointer modulo 256:", layer.weight.main_grad.data_ptr() % 256, flush=True)
inputs = torch.randn(128, in_features, device='cuda', dtype=torch.bfloat16, requires_grad=True)
with te.fp8_autocast(enabled=True, fp8_recipe=NVFP4BlockScaling()):
    output = layer(inputs, is_first_microbatch=None)
    output.float().square().mean().backward()
torch.cuda.synchronize()
print('PASS', in_features, out_features, fused, flush=True)
