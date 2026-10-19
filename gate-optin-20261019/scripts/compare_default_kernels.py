"""Compare actual SM86 instructions for the restored default atomic path."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from megatron.core.tuning import AutotunePolicy, install

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "default-kernel-check"
OUT.mkdir(exist_ok=False)
torch.use_deterministic_algorithms(False)
install(
    AutotunePolicy(
        mode="pinned",
        modules=("atomic_control_base", "atomic_control_candidate"),
        block_sizes=(("BLOCK_S", 32), ("BLOCK_H", 16)),
    )
)
config = json.loads(
    Path(
        "/home/songbinbin/gdn-offload-artifacts/models/Qwen3.5-0.8B/config.json"
    ).read_text()
)["text_config"]
key_heads = config["linear_num_key_heads"]
heads = config["linear_num_value_heads"]
dim = config["linear_key_head_dim"]
value_dim = config["linear_value_head_dim"]
channels = 2 * key_heads * dim + 2 * heads * value_dim + 2 * heads
projection = torch.randn(2048, 1, channels, device="cuda", dtype=torch.bfloat16)
parameter = torch.zeros(heads, device="cuda", dtype=torch.bfloat16)
gradient = torch.randn(1, 2048, heads, device="cuda")
output = torch.zeros_like(projection)
reports = {}
words = {}
for label, source in (("base", "upstream"), ("candidate", "source-final")):
    path = ROOT / source / "megatron/core/fusions/fused_pre_gated_delta_rule.py"
    name = "atomic_control_" + label
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._triton_g_beta_backward(
        projection,
        parameter,
        parameter,
        gradient,
        gradient,
        num_value_heads=heads,
        key_head_dim=dim,
        value_head_dim=value_dim,
        num_key_heads=key_heads,
        d_qkvzba_out=output,
    )
    torch.cuda.synchronize()
    tuner = module._g_beta_backward_kernel
    kernels = list(tuner.fn.device_caches[torch.cuda.current_device()][0].values())
    assert len(kernels) == 1
    kernel = kernels[0]
    binary = OUT / (label + ".cubin")
    binary.write_bytes(kernel.asm["cubin"])
    sass = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)], text=True
    )
    (OUT / (label + ".sass.log")).write_text(sass)
    words[label] = re.findall(r"/\*\s*(0x[0-9a-fA-F]{16})\s*\*/", sass)
    assert words[label], sass[:1000]
    reports[label] = {
        "source": source,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_config": str(tuner.best_config),
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "instruction_words": len(words[label]),
        "instruction_sha256": hashlib.sha256(
            "".join(words[label]).encode()
        ).hexdigest(),
    }
result = {
    "final_source": json.loads((ROOT / "final-source-identity.json").read_text())[
        "source"
    ],
    "baseline_source": "9131593673d037d360d252904f9198d36bc0c246",
    "sequence": 2048,
    "batch": 1,
    "value_heads": heads,
    "key_heads": key_heads,
    "key_dim": dim,
    "value_dim": value_dim,
    "deterministic_arithmetic": False,
    "arms": reports,
    "instruction_words_equal": words["base"] == words["candidate"],
}
(ROOT / "default-kernel-comparison.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
