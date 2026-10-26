"""Fresh-process Qwen checks with a repeated disabled control and identical tuning."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'feature-final'
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
artifacts = Path('/home/songbinbin/gdn-offload-artifacts')
pythonpath = ':'.join(map(str, [source, root/'scripts', artifacts/'flash-attn-python', artifacts/'causal-conv1d-python', artifacts/'bridge-python', artifacts/'split-pr-20261015/bridge/src']))
results = []
for arm in ('disabled', 'repeated-disabled', 'offloaded'):
    output = root / ('qwen-correctness-' + arm)
    output.mkdir()
    command = ['docker', 'exec', '--workdir', str(source), '--env', 'PYTHONPATH='+pythonpath,
        '--env', 'CUDA_VISIBLE_DEVICES=GPU-df9e07cc-8ebc-3204-fabd-601044f800f8',
        '--env', 'TRITON_CACHE_DIR='+str(root/'cache/qwen-correctness'), 'megatron-gdn-final-20261026',
        '/home/songbinbin/.venvs/megatron-gdn-offload/bin/python', '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=1', '--log-dir', str(output/'ranks'), '--redirects', '3',
        str(root/'scripts/rank_env.py'), str(source/'tools/ssm/qwen_gdn_offload_training.py'),
        '--weights', str(artifacts/'models/Qwen3.5-0.8B'), '--data', str(artifacts/'training-data/ks-item-sft.jsonl'),
        '--seq-length', '128', '--microbatches', '4', '--warmup', '1', '--iterations', '2',
        '--min-offloaded-tensor-size', '1024', '--recompute-norm', '--check-state', '--output', str(output/'result.json')]
    if arm == 'offloaded':
        command += ['--fraction', '1']
    manifest = dict(revision=revision, command=command, started_unix=time.time())
    with (output/'launcher.log').open('w') as handle:
        status = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    manifest.update(exit_code=status.returncode, finished_unix=time.time(),
        revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert status.returncode == 0 and manifest['revision_after'] == revision and not manifest['dirty_after'], arm
    results.append(json.loads((output/'result.rank0.json').read_text()))
    print('Finished Qwen '+arm, flush=True)
assert len(results[0]['states']) == 3
assert results[0]['states'] == results[1]['states'] == results[2]['states']
assert results[2]['offload_status']['selected_transfer_bytes'] > 0
(root/'qwen-correctness-comparison.json').write_text(json.dumps(dict(revision=revision, exact_states=True, steps=3, arms=['disabled','repeated-disabled','offloaded']),indent=2)+'\n')
