"""Capture actual vector states and verify restored RNG at the native load boundary."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import torch

from tools.ssm import qwen_gdn_offload_training as training

output = Path(sys.argv[sys.argv.index("--output") + 1]).parent
rank = int(os.environ["RANK"])
fingerprint = training._fingerprint
vectors = {}
original_load = training.dist_checkpointing.load
original_restore = training._load_checkpoint
saved_rng = None


def capture(value):
    result = fingerprint(value)
    if isinstance(value, torch.Tensor) and value.ndim <= 1:
        vectors[(result["dtype"], tuple(result["shape"]), result["sha256"])] = value.detach().cpu().clone()
    return result


def observed_load(*args, **kwargs):
    global saved_rng
    state = original_load(*args, **kwargs)
    saved_rng = state["rng"]
    return state


def checked_restore(*args, **kwargs):
    import numpy as np

    step = original_restore(*args, **kwargs)
    actual = training._rng_state()
    assert actual["python"] == saved_rng["python"]
    assert actual["numpy"][0] == saved_rng["numpy"][0]
    assert np.array_equal(actual["numpy"][1], saved_rng["numpy"][1])
    assert actual["numpy"][2:] == saved_rng["numpy"][2:]
    for name in ("torch", "cuda"):
        assert torch.equal(actual[name], saved_rng[name]), name
    assert actual["tracker"].keys() == saved_rng["tracker"].keys()
    for name in actual["tracker"]:
        assert torch.equal(actual["tracker"][name], saved_rng["tracker"][name]), name
    report = {
        "next_step": step, "all_rng_states_exact": True,
        "torch": fingerprint(actual["torch"]), "cuda": fingerprint(actual["cuda"]),
        "tracker": {key: fingerprint(value) for key, value in actual["tracker"].items()},
        "python_sha256": hashlib.sha256(pickle.dumps(actual["python"])).hexdigest(),
        "numpy_sha256": hashlib.sha256(pickle.dumps(actual["numpy"])).hexdigest(),
    }
    (output / f"restored-rng.rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    return step


training._fingerprint = capture
training.dist_checkpointing.load = observed_load
training._load_checkpoint = checked_restore
training.main()
torch.save(vectors, output / f"vectors.rank{rank}.pt")
