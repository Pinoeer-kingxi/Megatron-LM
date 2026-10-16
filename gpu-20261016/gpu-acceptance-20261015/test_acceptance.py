"""CPU regression fixtures for acceptance checks; these are not GPU qualification."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_gpu_acceptance as runner
from verify_results import summarize_benchmarks, verify_pytest, verify_qwen, verify_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def rank_reports(self):
        for rank in range(2):
            write(
                self.root / f"rank-{rank}.json",
                {
                    "rank": rank,
                    "exit_code": 0,
                    "collected": 2,
                    "outcomes": [
                        {"nodeid": f"test_{i}", "when": "call", "outcome": "passed", "detail": ""}
                        for i in range(2)
                    ],
                },
            )

    def test_per_rank_counts_and_missing_rank(self):
        self.rank_reports()
        self.assertEqual(verify_pytest(self.root, 2, 2)["skipped"], 0)
        with self.assertRaises(ValueError):
            verify_pytest(self.root, 2, 3)
        (self.root / "rank-1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_pytest(self.root, 2, 2)

    def test_skips_do_not_pass_required_tests(self):
        self.rank_reports()
        path = self.root / "rank-1.json"
        report = json.loads(path.read_text())
        report["outcomes"][0].update(outcome="skipped", detail="missing dependency")
        write(path, report)
        with self.assertRaises(ValueError):
            verify_pytest(self.root, 2, 2)
        self.assertEqual(verify_pytest(self.root, 2, 2, allow_skips=True)["skipped"], 1)

    def test_teardown_failure_is_not_hidden_by_passed_calls(self):
        self.rank_reports()
        path = self.root / "rank-0.json"
        report = json.loads(path.read_text())
        report["outcomes"].append({"nodeid": "test_0", "when": "teardown", "outcome": "failed"})
        write(path, report)
        with self.assertRaises(ValueError):
            verify_pytest(self.root, 2, 2)

    def qwen_results(self):
        for fused in (0, 1):
            for norm in (0, 1):
                for offload in (0, 1):
                    for rank in range(4):
                        write(
                            self.root
                            / f"qwen-f{fused}-n{norm}-o{offload}"
                            / f"result.rank{rank}.json",
                            {
                                "rank": rank,
                                "gpu_training_run": True,
                                "pool_verified_before_reset": True,
                                "text_layers": 24,
                                "arguments": {
                                    "tp": 2,
                                    "sp": True,
                                    "cp": 1,
                                    "pp": 2,
                                    "vp": None,
                                    "warmup": 1,
                                    "iterations": 2,
                                    "microbatches": 4,
                                    "seq_length": 128,
                                    "check_state": True,
                                    "dry_run": False,
                                    "fused_pre_gdr": bool(fused),
                                    "recompute_norm": bool(norm),
                                    "fraction": 1.0 if offload else None,
                                },
                                "offload_status": {
                                    "selected_transfer_bytes": 1024 if offload else 0
                                },
                                "states": [
                                    {
                                        "step": step,
                                        "grad_norm": 1.0,
                                        "losses": [2.0] * (4 if rank >= 2 else 0),
                                        "grads": {"p": {"sha256": "gradient"}},
                                        "weights": {"p": {"sha256": "weight"}},
                                        "optimizer": {"state": {"sha256": "adam"}},
                                    }
                                    for step in range(3)
                                ],
                            },
                        )

    def test_all_qwen_states_and_gradient_difference(self):
        self.qwen_results()
        self.assertEqual(verify_qwen(self.root)["compared_rank_steps"], 48)
        path = self.root / "qwen-f1-n1-o1/result.rank3.json"
        result = json.loads(path.read_text())
        result["states"][2]["grads"]["p"]["sha256"] = "different-byte-hash"
        write(path, result)
        with self.assertRaisesRegex(ValueError, "grads differs"):
            verify_qwen(self.root)

    def test_preflight_and_missing_losses_cannot_qualify_training(self):
        self.qwen_results()
        path = self.root / "qwen-f0-n0-o0/result.rank2.json"
        original = json.loads(path.read_text())
        for key in ("gpu_training_run", "pool_verified_before_reset"):
            result = json.loads(json.dumps(original))
            result[key] = False
            write(path, result)
            with self.assertRaises(ValueError):
                verify_qwen(self.root)
        original["states"][0]["losses"] = []
        write(path, original)
        with self.assertRaisesRegex(ValueError, "microbatch losses"):
            verify_qwen(self.root)

    def benchmark_results(self):
        for batch in (1, 4):
            for length in (2048, 4096):
                for heads in (16, 32):
                    for repeat in range(3):
                        write(
                            self.root / f"gate-bench-{batch}-{length}-{heads}-{repeat}/result.json",
                            {
                                "gpu_uuid": "GPU-test",
                                "offloading": False,
                                "sources": {
                                    "baseline": {"revision": "old"},
                                    "candidate": {"revision": "new"},
                                },
                                "case": {
                                    "batch": batch,
                                    "sequence": length,
                                    "value_heads": heads,
                                    "numerical_reference": "passed",
                                    "samples": {
                                        label: [
                                            {
                                                "wall_ms_per_call": value,
                                                "extra_peak_allocated_bytes": 512,
                                            }
                                            for _ in range(3)
                                        ]
                                        for label, value in (("baseline", 2.0), ("candidate", 3.0))
                                    },
                                },
                            },
                        )

    def test_qwen_requires_real_transfers_and_finite_gradients(self):
        self.qwen_results()
        path = self.root / "qwen-f1-n0-o1/result.rank0.json"
        original = json.loads(path.read_text())
        result = json.loads(json.dumps(original))
        result["offload_status"]["selected_transfer_bytes"] = 0
        write(path, result)
        with self.assertRaisesRegex(ValueError, "no positive offload"):
            verify_qwen(self.root)
        original["states"][1]["grad_norm"] = float("inf")
        write(path, original)
        with self.assertRaisesRegex(ValueError, "gradient norm"):
            verify_qwen(self.root)

    def test_benchmark_reports_regression_without_hiding_it(self):
        self.benchmark_results()
        results = summarize_benchmarks(self.root, {"base": "old", "gate": "new"}, "GPU-test")
        self.assertEqual(len(results), 8)
        self.assertTrue(
            all(
                row["median_paired_runtime_change_pct"] == 50.0 and row["paired_samples"] == 9
                for row in results
            )
        )
        path = self.root / "gate-bench-1-2048-16-0/result.json"
        result = json.loads(path.read_text())
        result["case"]["samples"]["candidate"][0]["wall_ms_per_call"] = float("nan")
        write(path, result)
        with self.assertRaises(ValueError):
            summarize_benchmarks(self.root, {"base": "old", "gate": "new"}, "GPU-test")

    def test_cuda_and_nvidia_smi_uuid_formats_identify_the_same_device(self):
        self.benchmark_results()
        for path in self.root.glob("gate-bench-*/result.json"):
            result = json.loads(path.read_text())
            result["gpu_uuid"] = "test"
            write(path, result)
        results = summarize_benchmarks(self.root, {"base": "old", "gate": "new"}, "GPU-test")
        self.assertEqual(len(results), 8)
        path = self.root / "gate-bench-1-2048-16-0/result.json"
        result = json.loads(path.read_text())
        result["gpu_uuid"] = "another-device"
        write(path, result)
        with self.assertRaisesRegex(ValueError, "unexpected device"):
            summarize_benchmarks(self.root, {"base": "old", "gate": "new"}, "GPU-test")

    def test_job_matrix_and_fresh_qwen_processes(self):
        args = SimpleNamespace(
            stage="single",
            output=self.root,
            source_root=Path("/sources"),
            weights=Path("/weights"),
            data=Path("/data.jsonl"),
        )
        jobs = runner.make_jobs(args)
        self.assertEqual(len(jobs), 27)
        self.assertEqual(len({job["name"] for job in jobs}), 27)
        self.assertIn(str(self.root / "scripts/rank_pytest.py"), jobs[0]["command"])
        self.assertIn(str(self.root / "scripts/benchmark_gate.py"), jobs[-1]["command"])
        args.stage = "four"
        jobs = runner.make_jobs(args)
        self.assertEqual(len(jobs), 11)
        qwen = [job for job in jobs if job["name"].startswith("qwen-")]
        self.assertEqual(len(qwen), 8)
        self.assertEqual(sum("--fraction" in job["command"] for job in qwen), 4)
        self.assertTrue(all("--nproc-per-node=4" in job["command"] for job in qwen))
        self.assertTrue(
            all(str(self.root / "scripts/rank_env.py") in job["command"] for job in qwen)
        )
        self.assertNotIn(str(self.root / "scripts/rank_env.py"), jobs[0]["command"])

    def test_rank_cache_is_set_before_target_and_preserves_argv(self):
        target = self.root / "target.py"
        target.write_text(
            "import json, os, sys\n"
            "print(json.dumps([os.environ['TRITON_CACHE_DIR'], sys.argv, __name__]))\n"
        )
        for rank in (0, 2):
            output = subprocess.check_output(
                [sys.executable, "-S", str(runner.HERE / "rank_env.py"), str(target), "a b"],
                env={**os.environ, "RANK": str(rank), "TRITON_CACHE_DIR": str(self.root)},
                text=True,
            )
            self.assertEqual(
                json.loads(output),
                [str(self.root / f"rank-{rank}"), [str(target), "a b"], "__main__"],
            )

    def test_autotune_evidence_keeps_rank_ownership_and_actual_winner(self):
        for rank in (0, 1):
            write(
                self.root / f"rank-{rank}/key/kernel.autotune.json",
                {
                    "key": [128, "torch.bfloat16"],
                    "configs_timings": [[{"num_warps": 4}, [2.0]], [{"num_warps": 8}, [1.0]]],
                },
            )
        results = runner.autotune_cache_state(self.root)
        self.assertEqual(len(results), 2)
        self.assertEqual(results["rank-0/key/kernel.autotune.json"]["config"], {"num_warps": 8})
        self.assertEqual(
            results["rank-0/key/kernel.autotune.json"],
            results["rank-1/key/kernel.autotune.json"],
        )

    def test_exported_run_checksums_and_incomplete_plan(self):
        self.qwen_results()
        names = [path.name for path in sorted(self.root.glob("qwen-*"))]
        manifest = {
            "stage": "four",
            "sources": {key: runner.SOURCES[key] for key in ("base", "shared", "gate", "feature")},
            "status": "automated_checks_passed",
            "cleanup": {"exit_code": 0},
            "planned_jobs": names,
            "performance_review": "not measured",
            "gpus": [{}] * 4,
            "jobs": [
                {
                    "name": name,
                    "source": "feature",
                    "source_commit": runner.SOURCES["feature"],
                    "exit_code": 0,
                }
                for name in names
            ],
        }
        write(self.root / "manifest.json", manifest)
        checksums = "".join(
            runner.hash_file(path) + "  " + path.relative_to(self.root).as_posix() + "\n"
            for path in sorted(self.root.rglob("*.json"))
        )
        (self.root / "SHA256SUMS").write_text(checksums)
        self.assertEqual(verify_run(self.root)["verified_files"], 33)
        (self.root / "qwen-f0-n0-o0/result.rank0.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            verify_run(self.root)
        (self.root / "SHA256SUMS").write_text("")
        with self.assertRaisesRegex(ValueError, "must include manifest"):
            verify_run(self.root)
        manifest["jobs"].pop()
        write(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Some planned jobs"):
            verify_run(self.root)

    def test_busy_devices_are_rejected(self):
        with patch.object(runner.subprocess, "check_output", return_value="GPU-test, 1234\n"):
            with self.assertRaises(RuntimeError):
                runner.require_idle([{"uuid": "GPU-test"}])
            runner.require_idle([{"uuid": "GPU-another"}])

    def test_dirty_source_is_rejected(self):
        with patch.object(
            runner.subprocess,
            "check_output",
            side_effect=[runner.SOURCES["base"], " M changed.py\n"],
        ):
            with self.assertRaises(ValueError):
                runner.source_state(self.root)


if __name__ == "__main__":
    unittest.main()
