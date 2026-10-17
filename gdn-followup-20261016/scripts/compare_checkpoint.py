"""Compare all fingerprints, inspecting actual tensors whenever bytes differ."""

import hashlib
import json
import math
from pathlib import Path

import torch

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("ranks", type=int, nargs="?", default=1)
configurations = ["f0-n0-o0", "f0-n0-o1", "f0-n1-o1", "f1-n0-o1", "f1-n1-o1"]
parser.add_argument("--config", action="append", choices=configurations)
args = parser.parse_args()
ROOT = args.root.resolve()
RTOL, ATOL = 1e-5, 1e-8


def read(path):
    return json.loads(path.read_text())


def raw_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def compare(expected, actual, left_vectors, right_vectors, path, rows):
    if isinstance(expected, dict) and set(expected) == {'shape', 'dtype', 'sha256'}:
        assert expected['shape'] == actual['shape'] and expected['dtype'] == actual['dtype'], path
        row = dict(path=path, exact=expected == actual)
        if not row['exact']:
            left = left_vectors[(expected['dtype'], tuple(expected['shape']), expected['sha256'])]
            right = right_vectors[(actual['dtype'], tuple(actual['shape']), actual['sha256'])]
            assert hashlib.sha256(raw_bytes(left)).hexdigest() == expected['sha256'], path
            assert hashlib.sha256(raw_bytes(right)).hexdigest() == actual['sha256'], path
            assert list(left.shape) == expected['shape'] and str(left.dtype) == expected['dtype']
            assert list(right.shape) == actual['shape'] and str(right.dtype) == actual['dtype']
            assert torch.isfinite(left).all() and torch.isfinite(right).all(), path
            delta = (right.double() - left.double()).abs()
            ratio = delta / (ATOL + RTOL * left.double().abs())
            row.update(expected=expected, actual=actual,
                       expected_values=left.tolist(), actual_values=right.tolist(),
                       expected_bytes_hex=raw_bytes(left).hex(), actual_bytes_hex=raw_bytes(right).hex(),
                       max_absolute_difference=delta.max().item() if delta.numel() else 0,
                       max_tolerance_ratio=ratio.max().item() if ratio.numel() else 0)
            rows.append(row)
            torch.testing.assert_close(right, left, rtol=RTOL, atol=ATOL, msg=path)
        else:
            rows.append(row)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], left_vectors, right_vectors, path + '/' + str(key), rows)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, left_vectors, right_vectors, path + '/' + str(index), rows)
    elif isinstance(expected, float):
        assert math.isfinite(expected) and math.isfinite(actual), path
        assert abs(actual - expected) <= ATOL + RTOL * abs(expected), path
        if expected != actual:
            rows.append(dict(path=path, scalar=True, exact=False, expected=expected, actual=actual,
                             max_absolute_difference=abs(actual - expected),
                             max_tolerance_ratio=abs(actual - expected)/(ATOL + RTOL * abs(expected))))
    else:
        assert expected == actual, path


summaries = []
for config in args.config or configurations:
    fused, norm, offload = [int(part[1:]) for part in config.split("-")]
    for rank in range(args.ranks):
        arms = {}
        for name in ("continuous", "prefix", "resumed"):
            directory = ROOT / f"{config}-{name}"
            result = read(directory / f"result.rank{rank}.json")
            assert result["gpu_training_run"] and result["pool_verified_before_reset"]
            assert result["rank"] == rank and result["text_layers"] == 24
            assert result["arguments"]["fused_pre_gdr"] == bool(fused)
            assert result["arguments"]["recompute_norm"] == bool(norm)
            assert result["arguments"]["fraction"] == (1.0 if offload else None)
            states = result["states"]
            steps = list(range(2, 6)) if name == "resumed" else list(range(6 if name == "continuous" else 2))
            assert [s["step"] for s in states] == steps
            for state in states:
                assert all(math.isfinite(x) for x in state["losses"])
                assert math.isfinite(state["grad_norm"]) and state["grad_norm"] > 0
                assert all(state[field] for field in ("grads", "weights", "optimizer"))
            vectors = torch.load(directory / f"vectors.rank{rank}.pt", map_location="cpu", weights_only=True)
            arms[name] = (result, vectors)
        rng = read(ROOT / f"{config}-resumed/restored-rng.rank{rank}.json")
        assert rng["all_rng_states_exact"] and rng["next_step"] == 2
        for name, start, end in (("prefix", 0, 2), ("resumed", 2, 6)):
            left = arms["continuous"][0]["states"][start:end]
            right = arms[name][0]["states"]
            if not fused:
                assert left == right, (config, rank, name)
            rows = []
            compare(left, right, arms["continuous"][1], arms[name][1], "", rows)
            changed = [r for r in rows if not r["exact"]]
            report = f"numeric-{config}-{name}-rank{rank}.json"
            (ROOT / report).write_text(json.dumps(rows, indent=2) + "\n")
            summaries.append(dict(configuration=config, rank=rank, arm=name, compared_steps=end-start,
                                  exact=left == right, close=True, rng_exact=True,
                                  tensor_leaves=sum(not r.get("scalar", False) for r in rows),
                                  differing_leaves=len(changed),
                                  max_absolute_difference=max((r["max_absolute_difference"] for r in changed), default=0),
                                  max_tolerance_ratio=max((r["max_tolerance_ratio"] for r in changed), default=0),
                                  report=report))
result = dict(status="passed", rtol=RTOL, atol=ATOL, comparisons=summaries,
              description="Native checkpoint restart: all losses, gradients, weights and complete Adam/master states; unfused byte exact, fused vectors numerically close and other tensors byte exact; all five RNG streams restored exactly.")
(ROOT / "checkpoint-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
