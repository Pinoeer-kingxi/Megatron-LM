"""Observe the committed Qwen runner without changing its training operations."""

import json
import logging
import math
from pathlib import Path

import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from tools.ssm import qwen_gdn_offload_training as qwen


original_finalize = qwen.finalize_model_grads
pool_observations = []


def observed_finalize(*args, **kwargs):
    original_finalize(*args, **kwargs)
    pool = PipelineOffloadManager.get_instance().cpu_tensor_pool
    stats = pool.get_pool_status()["global_stats"]
    assert stats["current_in_use"] == 0, stats
    pool_observations.append(stats)


class StepLog(logging.Handler):
    def emit(self, record):
        if record.msg != "Rank %d step %d: losses=%s grad_norm=%s selected_transfer_bytes=%d":
            return
        rank, step, losses, grad_norm, selected_bytes = record.args
        assert losses and all(math.isfinite(value) for value in losses)
        assert math.isfinite(grad_norm) and grad_norm > 0
        assert len(pool_observations) == step + 1
        row = {
            "rank": rank,
            "step": step,
            "losses": losses,
            "grad_norm": float(grad_norm),
            "selected_transfer_bytes": selected_bytes,
            "pool_before_schedule_reset": pool_observations[-1],
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        output = Path(__file__).resolve().parents[1] / "stability-steps.jsonl"
        with output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


qwen.finalize_model_grads = observed_finalize
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(StepLog())
qwen.main()
