"""Run remaining existing scope tests and the three native negative controls."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = Path("/home/songbinbin/gdn-offload-artifacts/review-20261021")
jobs = [
    ("feature-scopes", "feature", ["tests/unit_tests/ssm/test_gated_delta_net_offloading.py",
                                   "tests/unit_tests/ssm/test_gdn_activation_offload.py",
                                   "tests/unit_tests/ssm/test_qwen_gdn_offload_training.py"], 0),
    ("extensions-scopes", "extensions", ["tests/unit_tests/ssm/test_gated_delta_net_offloading.py", "-k", "gdn2"], 0),
    ("extensions-native", "extensions", ["tests/unit_tests/ssm/test_gdn_offload_training.py", "-k",
          "gdn2 and (dp or tp_sp or pp) and not tp_sp_cp and not tp_sp_pp and not pp_vpp"], 0),
]
for name in ("eval", "fraction", "progress"):
    jobs.append(("negative-final-" + name, "control-" + name,
                 ["tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py", "-k",
                  "after_evaluation_with_growing_sequences and 0.0"], 1))
for name, source, arguments, expected in jobs:
    result = subprocess.run([sys.executable, str(root / "scripts/run_job.py"), name,
                             str(sources / source), "2", *arguments])
    assert result.returncode == expected, (name, result.returncode, expected)
