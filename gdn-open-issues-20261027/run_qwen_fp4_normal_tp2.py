import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

b = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = b / 'precision-with-mrope-dfaa633'
revision = 'dfaa6338c65edab118ba415fcc14c73241f503a6'
script = source / 'tools/ssm/qwen_gdn_offload_training.py'
records = []
for arm, override, offload in [('default', False, False), ('high-precision', True, False), ('high-precision-offload', True, True)]:
    out = b / f'qwen-fp4-normal-tp2-{arm}-dfaa633-gpu12'
    out.mkdir(exist_ok=False)
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    cmd = ['/usr/bin/python3', str(b / 'precision_graph_container_gpu12_apex.py'), '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '--log-dir', str(out / 'ranks'), '--redirects', '3', str(b / 'rank_env_precision_graph.py'), str(b / 'verify_qwen_source.py'), str(script), '--weights', str(b / 'models/Qwen3.5-0.8B'), '--data', str(b / 'data/ks-item-sft.jsonl'), '--precision', 'fp4', '--seq-length', '128', '--microbatches', '4', '--warmup', '1', '--iterations', '2', '--tp', '2', '--sp', '--min-offloaded-tensor-size', '1024', '--recompute-norm', '--check-state', '--output', str(out / 'result.json')]
    if offload:
        cmd += ['--fraction', '1']
    env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0', GDN_INDUCTOR_CACHE_DIR=str(b / 'cache/qwen-normal-tp2-inductor'), GDN_TRITON_CACHE_DIR=str(b / 'cache/qwen-normal-tp2-triton'))
    if override:
        env['NVTE_BACKWARD_OVERRIDE'] = 'high_precision'
    else:
        env.pop('NVTE_BACKWARD_OVERRIDE', None)
    manifest = {'revision': revision, 'arm': arm, 'command': cmd, 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'started_unix': time.time(), 'shared_gpus': True, 'rht_disabled': True, 'stochastic_rounding_disabled': True, 'backward_override': 'high_precision' if override else None}
    with (out / 'launcher.log').open('w') as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=proc.returncode, finished_unix=time.time(), revision_after=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(), dirty_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(arm, proc.returncode, flush=True)
    assert manifest['revision_after'] == revision and not manifest['dirty_after']
    if override:
        assert proc.returncode == 0
        states = []
        for rank in (0, 1):
            data = json.loads((out / f'result.rank{rank}.json').read_text())
            assert data['text_layers'] == 24 and len(data['states']) == 3 and data['pool_verified_before_reset']
            assert bool(data['offload_status']['selected_transfer_bytes']) == offload
            states.append(data['states'])
        records.append(states)
comparison = {'revision': revision, 'high_precision_offload_exact': [records[0][rank] == records[1][rank] for rank in (0, 1)]}
(b / 'qwen-fp4-normal-tp2-dfaa633-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison), flush=True)
