import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

b = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = b / 'feature-merge-b6fa63'
revision = 'b6fa63ab8c2cb855f0368cd8455245c295cde234'
script = source / 'tools/ssm/qwen_gdn_offload_training.py'
checkpoint = b / 'qwen-checkpoint-b6fa63-gpu3'
assert not checkpoint.exists()
results = []
for arm in ('continuous', 'save', 'resume'):
    out = b / f'qwen-bf16-checkpoint-{arm}-b6fa63-gpu3'
    out.mkdir(exist_ok=False)
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    cmd = ['/usr/bin/python3', str(b / 'precision_graph_container_gpu3_constrained.py'), '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1', '--log-dir', str(out / 'ranks'), '--redirects', '3', str(b / 'rank_env_precision_graph.py'), str(b / 'verify_qwen_source.py'), str(script), '--weights', str(b / 'models/Qwen3.5-0.8B'), '--data', str(b / 'data/ks-item-sft.jsonl'), '--seq-length', '128', '--microbatches', '4', '--warmup', '1', '--iterations', '5' if arm == 'continuous' else '2', '--fraction', '1', '--recompute-norm', '--min-offloaded-tensor-size', '1024', '--check-state', '--output', str(out / 'result.json')]
    if arm == 'save':
        cmd += ['--save-checkpoint', str(checkpoint)]
    elif arm == 'resume':
        cmd += ['--load-checkpoint', str(checkpoint)]
    env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0', GDN_INDUCTOR_CACHE_DIR=str(b / 'cache/qwen-checkpoint-b6-inductor'), GDN_TRITON_CACHE_DIR=str(b / 'cache/qwen-checkpoint-b6-triton'))
    manifest = {'revision': revision, 'arm': arm, 'command': cmd, 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'started_unix': time.time(), 'gpu_uuid': 'GPU-dfee170b-6263-1438-4231-9e55c7f1c24d'}
    with (out / 'launcher.log').open('w') as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=proc.returncode, finished_unix=time.time(), revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(), dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(arm, proc.returncode, flush=True)
    assert proc.returncode == 0 and manifest['revision_after'] == revision and not manifest['dirty_after']
    data = json.loads((out / 'result.rank0.json').read_text())
    assert data['text_layers'] == 24 and len(data['states']) == (6 if arm == 'continuous' else 3)
    assert data['pool_verified_before_reset'] and data['offload_status']['selected_transfer_bytes'] > 0
    results.append(data['states'])
comparison = {'revision': revision, 'continuous_equals_save_resume': results[0] == results[1] + results[2], 'steps': len(results[0])}
(b / 'qwen-checkpoint-b6fa63-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison), flush=True)
assert comparison['continuous_equals_save_resume']
