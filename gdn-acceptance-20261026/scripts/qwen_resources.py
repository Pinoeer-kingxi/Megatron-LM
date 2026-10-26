"""Run the unchanged Qwen tool; observe retained pinned capacity at final cleanup."""

import json
import math
import os
import sys
from pathlib import Path

import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from tools.ssm import qwen_gdn_offload_training as training


observations = []
original_reset = PipelineOffloadManager.reset_instance


def record_reset():
    manager = PipelineOffloadManager.OFFLOAD_MGR
    if manager is not None:
        status = manager.cpu_tensor_pool.get_pool_status()
        capacity = sum(
            math.prod(pool["shape"]) * torch.tensor([], dtype=pool["dtype"]).element_size()
            * pool["total_count"]
            for pool in status["pools"].values()
        )
        observations.append(dict(stats=status["global_stats"], retained_pinned_bytes=capacity))
    original_reset()


if __name__ == "__main__":
    # Calls occur at initialization/cleanup, outside every measured optimizer step.
    PipelineOffloadManager.reset_instance = staticmethod(record_reset)
    training.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    path = output.with_name(f"resources.rank{os.environ['RANK']}.json")
    path.write_text(json.dumps(observations, indent=2) + "\n")
