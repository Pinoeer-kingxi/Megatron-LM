"""Retry the unchanged docs snapshot on the data disk after the root disk filled."""

import json
import pathlib
import shlex
import subprocess
from datetime import datetime, timezone

OUT = pathlib.Path(__file__).resolve().parent
ART = pathlib.Path('/home/songbinbin/gdn-offload-artifacts')
VENV = pathlib.Path('/home/songbinbin/.venvs/megatron-gdn-offload')
NAME = 'megatron-gdn-doc-cache-retry-20261016'
old = json.loads((OUT/'validation.json').read_text())
assert old['checks'][0]['exit_code'] == 0
assert old['checks'][1]['exit_code'] == 1
assert 'No space left on device' in (OUT/'strict-docs.log').read_text()
(OUT/'tmp').mkdir()
command = ['docker','run','--detach','--rm','--name',NAME,'--network','none']
for source, target in [(ART,ART),(ART,pathlib.Path('/artifacts')),(VENV,VENV),
                       (pathlib.Path('/data/songbinbin/minicoda'),pathlib.Path('/data/songbinbin/minicoda'))]:
    command += ['--mount',f'type=bind,source={source},target={target},readonly']
command += ['--mount',f'type=bind,source={OUT},target={OUT}']
for key,value in {'TMPDIR':str(OUT/'tmp'),'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void',
                  'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'}.items():
    command += ['--env',f'{key}={value}']
command += ['sha256:061e8fda67a4dd1c1c635129d4ba0442b7a093cc6fd737423e9589eb6be22034','sleep','infinity']
record = {'source_commit':old['source_commit'],'gpu_used':False,
          'initial_documentation_failure':'Root disk full: OSError [Errno 28]. Source unchanged.',
          'initial_record':'validation.json','container_command':command,'checks':[old['checks'][0]]}
subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
try:
    cmd = ['docker','exec','--workdir',str(OUT/'source'),NAME,
           str(ART/'split-pr-20261015/docs-venv/bin/python'),'-m','sphinx',
           '-j','4','-b','html','-W','--keep-going','docs',str(OUT/'html-retry')]
    item = {'name':'strict-docs-retry','command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()}
    with (OUT/'strict-docs-retry.log').open('w') as log:
        result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    item.update(exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    record['checks'].append(item)
    assert result.returncode == 0
    args = shlex.split((OUT/'documented-command.sh').read_text().split('python -m torch.distributed.run',1)[1].replace('\\\n',' '))
    expected = args[args.index('tools/ssm/qwen_gdn_offload_training.py')+1:]
    reference = []
    for arm in ('baseline','repeat','offload'):
        caches = []
        for rank in range(4):
            row = json.loads((OUT/f'{arm}.rank{rank}.json').read_text())
            assert row['rank'] == rank and row['environment_set_before_target_imports']
            assert row['ownership'] == {'rank':rank,'created_by':'baseline'}
            actual = row['argv']
            assert actual[-2:] == ['--review-argument','two words']
            actual = actual[:-2]
            if arm == 'offload':
                assert actual[-2:] == ['--fraction','1']
                actual = actual[:-2]
            actual[actual.index('--output')+1] = '/tmp/qwen-disabled.json'
            assert actual == expected
            caches.append(row['cache'])
        assert len(set(caches)) == 4
        if arm == 'baseline':
            reference = caches
        else:
            assert caches == reference
    record['launcher_result'] = {'ranks':4,'arms':3,'worker_records':12,
        'cache_isolated_between_ranks':True,'cache_reused_between_arms':True,
        'all_documented_arguments_preserved':True,'environment_set_before_target_imports':True,
        'target':'CPU probe; no model or Triton kernel execution'}
    record['status'] = 'passed'
finally:
    stopped = subprocess.run(['docker','stop','--timeout','10',NAME],text=True,capture_output=True)
    record['cleanup_exit_code'] = stopped.returncode
    (OUT/'cleanup-retry.log').write_text(stopped.stdout+stopped.stderr)
    (OUT/'validation-final.json').write_text(json.dumps(record,indent=2)+'\n')
print('Documented four-worker launcher and strict Sphinx build passed.',flush=True)
