"""Quantify every differing fingerprint using captured, hash-verified tensor bytes."""

import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-5, 1e-8


def read(path):
    return json.loads(path.read_text())


def raw_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def compare(expected, actual, left_vectors, right_vectors, path, rows):
    if isinstance(expected, dict) and set(expected) == {"shape", "dtype", "sha256"}:
        assert expected["shape"] == actual["shape"] and expected["dtype"] == actual["dtype"], path
        row = dict(path=path, exact=expected == actual)
        if not row["exact"]:
            values = []
            for fingerprint, vectors in ((expected, left_vectors), (actual, right_vectors)):
                key = (fingerprint["dtype"], tuple(fingerprint["shape"]), fingerprint["sha256"])
                tensor = vectors[key]
                assert list(tensor.shape) == fingerprint["shape"]
                assert str(tensor.dtype) == fingerprint["dtype"]
                assert hashlib.sha256(raw_bytes(tensor)).hexdigest() == fingerprint["sha256"], path
                assert torch.isfinite(tensor).all(), path
                values.append(tensor)
            left, right = values
            delta = (right.double() - left.double()).abs()
            ratio = delta / (ATOL + RTOL * left.double().abs())
            row.update(
                expected=expected,
                actual=actual,
                expected_bytes_hex=raw_bytes(left).hex(),
                actual_bytes_hex=raw_bytes(right).hex(),
                max_absolute_difference=delta.max().item() if delta.numel() else 0,
                max_tolerance_ratio=ratio.max().item() if ratio.numel() else 0,
            )
            torch.testing.assert_close(right, left, rtol=RTOL, atol=ATOL, msg=path)
        rows.append(row)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            compare(
                expected[key], actual[key], left_vectors, right_vectors, path + "/" + str(key), rows
            )
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, left_vectors, right_vectors, path + "/" + str(index), rows)
    elif isinstance(expected, float):
        assert math.isfinite(expected) and math.isfinite(actual), path
        assert abs(actual - expected) <= ATOL + RTOL * abs(expected), path
        if expected != actual:
            rows.append(
                dict(
                    path=path,
                    scalar=True,
                    exact=False,
                    expected=expected,
                    actual=actual,
                    max_absolute_difference=abs(actual - expected),
                    max_tolerance_ratio=abs(actual - expected) / (ATOL + RTOL * abs(expected)),
                )
            )
    else:
        assert expected == actual, path


if __name__ == "__main__":
    summary = []
    for norm in (0, 1):
        prefix = f"four-qwen-captured-f1-n{norm}"
        for rank in range(4):
            baseline = ROOT / (prefix + "-disabled")
            left = read(baseline / f"result.rank{rank}.json")["states"]
            left_vectors = torch.load(
                baseline / f"vectors.rank{rank}.pt", map_location="cpu", weights_only=True
            )
            for arm in ("repeat-disabled", "offloaded"):
                directory = ROOT / (prefix + "-" + arm)
                right = read(directory / f"result.rank{rank}.json")["states"]
                right_vectors = torch.load(
                    directory / f"vectors.rank{rank}.pt", map_location="cpu", weights_only=True
                )
                rows = []
                compare(left, right, left_vectors, right_vectors, "", rows)
                (directory / f"comparison.rank{rank}.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
                changes = [row for row in rows if not row["exact"]]
                row = dict(
                    recompute_norm=bool(norm),
                    arm=arm,
                    rank=rank,
                    steps=3,
                    exact=left == right,
                    compared_tensor_leaves=sum(not row.get("scalar") for row in rows),
                    differing_leaves=len(changes),
                    close=True,
                    max_absolute_difference=max(
                        (x["max_absolute_difference"] for x in changes), default=0
                    ),
                    max_tolerance_ratio=max((x["max_tolerance_ratio"] for x in changes), default=0),
                )
                summary.append(row)
                print(json.dumps(row), flush=True)
    (ROOT / "captured-qwen-comparison.json").write_text(
        json.dumps(dict(rtol=RTOL, atol=ATOL, comparisons=summary), indent=2) + "\n"
    )
