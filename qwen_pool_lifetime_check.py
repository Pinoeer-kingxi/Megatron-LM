"""Record native pool counts before reset, then validate after distributed teardown."""

import json
import os
import runpy
import sys
from pathlib import Path

from megatron.core.pipeline_parallel.fine_grained_activation_offload import OffloadTensorPool

counts = []
reset_pool = OffloadTensorPool.reset


def record_reset(pool):
    counts.append(pool.get_pool_status()['global_stats']['current_in_use'])
    reset_pool(pool)


OffloadTensorPool.reset = record_reset
runpy.run_path('/workspace/tools/ssm/qwen_gdn_offload_training.py', run_name='__main__')
assert counts and not any(counts), counts
output = Path(sys.argv[sys.argv.index('--output') + 1])
rank = int(os.environ['RANK'])
path = output.with_name(f'{output.stem}.rank{rank}.json')
result = json.loads(path.read_text())
assert len(counts) == (result['arguments']['warmup'] + result['arguments']['iterations'] + 2
                       if result['arguments']['fraction'] is not None else 2), counts
result['pool_in_use_before_resets'] = counts
path.write_text(json.dumps(result, indent=2) + '\n')
print(f'Rank {rank} pinned buffers before each native reset: {counts}', flush=True)
