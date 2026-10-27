"""Record pool ownership after the real gradient finalization in Qwen training."""
import json
import math
import os
import sys
from pathlib import Path

import torch
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from tools.ssm import qwen_gdn_offload_training as training

observations = []
finalize = training.finalize_model_grads


def record_finalize(*args, **kwargs):
    finalize(*args, **kwargs)
    status = PipelineOffloadManager.get_instance().cpu_tensor_pool.get_pool_status()
    assert status['global_stats']['current_in_use'] == 0
    retained = sum(
        math.prod(pool['shape']) * torch.tensor([], dtype=pool['dtype']).element_size()
        * pool['total_count'] for pool in status['pools'].values()
    )
    observations.append({'stats': status['global_stats'], 'retained_pinned_bytes': retained})


training.finalize_model_grads = record_finalize
training.main()
output = Path(sys.argv[sys.argv.index('--output') + 1])
output.with_name(f'pool.rank{os.environ["RANK"]}.json').write_text(json.dumps(observations, indent=2) + '\n')
