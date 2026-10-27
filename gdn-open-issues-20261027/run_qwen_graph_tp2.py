import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

b = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = b / 'graph-with-mrope-c5264b6'
revision = 'c5264b66197040f4309edfdb22dc9bbaf62189b1'
script = b / 'benchmark_qwen_graph_tp2.py'
outputs = []
keys = ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer')
for arm in ('eager', 'graph-off', 'graph-on'):
    out = b / f'qwen-bf16-graph-tp2-{arm}-c5264b6-gpu12'
    out.mkdir(exist_ok=False)
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    cmd = ['/usr/bin/python3', str(b / 'precision_graph_container_gpu12_apex.py'), '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '--log-dir', str(out / 'ranks'), '--redirects', '3', str(b / 'rank_env_precision_graph.py'), str(b / 'verify_qwen_source.py'), str(script), '--model', 'qwen', '--weights', str(b / 'models/Qwen3.5-0.8B'), '--data', str(b / 'data/ks-item-sft.jsonl'), '--seq-length', '128', '--warmup', '1', '--iterations', '2', '--threshold', '1024', '--check-state', '--output', str(out / 'result.json')]
    if arm == 'eager':
        cmd.append('--no-graph')
    if arm == 'graph-on':
        cmd.append('--offload')
    env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0', GDN_INDUCTOR_CACHE_DIR=str(b / 'cache/qwen-graph-tp2-inductor'), GDN_TRITON_CACHE_DIR=str(b / 'cache/qwen-graph-tp2-triton'))
    manifest = {'revision': revision, 'arm': arm, 'command': cmd, 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'started_unix': time.time(), 'shared_gpus': True}
    with (out / 'launcher.log').open('w') as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=proc.returncode, finished_unix=time.time(), revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(), dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(arm, proc.returncode, flush=True)
    assert proc.returncode == 0 and manifest['revision_after'] == revision and not manifest['dirty_after']
    ranks = []
    for rank in (0, 1):
        data = json.loads((out / f'result.rank{rank}.json').read_text())
        assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else 24)
        assert all(x['selected_bytes'] > 0 for x in data['records'][3:]) == (arm == 'graph-on')
        ranks.append([{key: record[key] for key in keys} for record in data['records']])
    outputs.append(ranks)
comparison = {'revision': revision, 'graph_off_exact': [outputs[0][rank] == outputs[1][rank] for rank in (0, 1)], 'graph_on_exact': [outputs[0][rank] == outputs[2][rank] for rank in (0, 1)]}
(b / 'qwen-bf16-graph-tp2-c5264b6-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison), flush=True)
