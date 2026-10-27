import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

b = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = b / 'precision-with-mrope-dfaa633'
revision = 'dfaa6338c65edab118ba415fcc14c73241f503a6'
script = b / 'benchmark_fp8_qwen.py'
keys = ('input_sha256', 'loss', 'grad_norm', 'gradients', 'weights', 'optimizer')
summary = {}
for precision in ('fp8-delayed', 'fp8-tensorwise'):
    results = []
    for arm in ('eager', 'graph-off', 'graph-on'):
        out = b / f'qwen-{precision}-{arm}-dfaa633-gpu0'
        out.mkdir(exist_ok=False)
        assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
        assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
        cmd = ['/usr/bin/python3', str(b / 'precision_graph_container_gpu0_default.py'), '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1', '--log-dir', str(out / 'ranks'), '--redirects', '3', str(b / 'rank_env_precision_graph.py'), str(b / 'verify_qwen_source.py'), str(script), '--model', 'qwen', '--precision', precision, '--weights', str(b / 'models/Qwen3.5-0.8B'), '--data', str(b / 'data/ks-item-sft.jsonl'), '--seq-length', '128', '--warmup', '1', '--iterations', '2', '--threshold', '1024', '--check-state', '--output', str(out / 'result.json')]
        if arm == 'eager':
            cmd.append('--no-graph')
        if arm == 'graph-on':
            cmd.append('--offload')
        env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0', GDN_INDUCTOR_CACHE_DIR=str(b / 'cache/qwen-final-fp8-inductor'), GDN_TRITON_CACHE_DIR=str(b / 'cache/qwen-final-fp8-triton'))
        manifest = {'revision': revision, 'arm': arm, 'precision': precision, 'command': cmd, 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'started_unix': time.time(), 'gpu_uuid': 'GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a'}
        with (out / 'launcher.log').open('w') as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(exit_code=proc.returncode, finished_unix=time.time(), revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(), dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(precision, arm, proc.returncode, flush=True)
        assert proc.returncode == 0 and manifest['revision_after'] == revision and not manifest['dirty_after']
        data = json.loads((out / 'result.json').read_text())
        assert len(data['records']) == 6 and len(data['graph_counts']) == (0 if arm == 'eager' else 24)
        assert all((record['selected_bytes'] > 0) == (arm == 'graph-on') for record in data['records'][3:])
        assert data['pool_checks'] == 6 and all(record['pool']['current_in_use'] == 0 for record in data['records'])
        results.append(data['records'])
    summary[precision] = {arm: {key: [x[key] == y[key] for x, y in zip(results[0], data)] for key in keys} for arm, data in zip(('graph-off', 'graph-on'), results[1:])}
    summary[precision]['offload_vs_graph'] = {key: [x[key] == y[key] for x, y in zip(results[1], results[2])] for key in keys}
    (b / 'qwen-fp8-dfaa633-comparison.json').write_text(json.dumps({'revision': revision, 'results': summary}, indent=2) + '\n')
    print(precision, json.dumps(summary[precision]), flush=True)
