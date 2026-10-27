import json
import os
import subprocess
import time
from pathlib import Path

base = Path('/home/lwx/.cache/megatron-gdn-20261026')
source = base / 'feature-merge-b6fa63'
revision = 'b6fa63ab8c2cb855f0368cd8455245c295cde234'
python = '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python'
states = []
for arm in ('disabled', 'repeated-disabled', 'offloaded'):
    directory = base / f'qwen-merge-b6fa63-{arm}-gpu0'
    directory.mkdir()
    command = ['/usr/bin/python3', str(base/'precision_graph_container_gpu0.py'), python,
        '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
        '--log-dir', str(directory/'ranks'), '--redirects', '3', str(base/'rank_env_precision_graph.py'),
        str(base/'verify_qwen_source.py'), str(source/'tools/ssm/qwen_gdn_offload_training.py'),
        '--weights', str(base/'models/Qwen3.5-0.8B'), '--data', str(base/'data/ks-item-sft.jsonl'),
        '--seq-length', '128', '--microbatches', '4', '--warmup', '1', '--iterations', '2',
        '--min-offloaded-tensor-size', '1024', '--recompute-norm', '--check-state',
        '--output', str(directory/'result.json')]
    if arm == 'offloaded':
        command += ['--fraction', '1']
    env = dict(os.environ, GDN_SOURCE=str(source), PYTHONHASHSEED='0',
        GDN_INDUCTOR_CACHE_DIR=str(base/'cache/qwen-merge-b6-inductor'),
        GDN_TRITON_CACHE_DIR=str(base/'cache/qwen-merge-b6-triton'))
    before = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
    assert before == revision and not dirty
    manifest = dict(revision=before, command=command, started_unix=time.time(),
                    gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a',
                    source_path=str(source), triton_cache=env['GDN_TRITON_CACHE_DIR'],
                    inductor_cache=env['GDN_INDUCTOR_CACHE_DIR'])
    with (directory/'launcher.log').open('w') as log:
        result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode,finished_unix=time.time(),
        revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if result.returncode or manifest['revision_after'] != revision or manifest['dirty_after']:
        raise SystemExit(result.returncode or 1)
    output=json.loads((directory/'result.rank0.json').read_text())
    assert output['text_layers'] == 24 and len(output['states']) == 3
    assert bool(output['offload_status']['selected_transfer_bytes']) == (arm == 'offloaded')
    states.append(output['states'])
    print(arm, 'passed', flush=True)
comparison=dict(revision=revision, exact_states=states[0]==states[1]==states[2], steps=3)
(base/'qwen-merge-b6fa63-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
print(json.dumps(comparison),flush=True)
assert comparison['exact_states']
