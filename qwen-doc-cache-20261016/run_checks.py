"""Check the documented Qwen launcher with four CPU workers and build the docs."""

import json
import pathlib
import re
import shlex
import subprocess
from datetime import datetime, timezone

ART = pathlib.Path('/home/songbinbin/gdn-offload-artifacts')
OUT = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path('/data/songbinbin/Proj/Proj_LWX/Megatron-LM')
VENV = pathlib.Path('/home/songbinbin/.venvs/megatron-gdn-offload')
DOCS_PYTHON = ART / 'split-pr-20261015/docs-venv/bin/python'
SOURCE = OUT / 'source'
NAME = 'megatron-gdn-doc-cache-check-20261016'
SHA = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO)
SOURCE.mkdir()
archive = OUT / 'source.tar'
subprocess.run(['git', 'archive', '--output', str(archive), SHA], cwd=REPO, check=True)
subprocess.run(['tar', '-xf', str(archive), '-C', str(SOURCE)], check=True)
archive.unlink()
doc = (SOURCE / 'docs/developer/gdn_activation_offloading.md').read_text()
block = next(b for b in re.findall(r'```bash\n(.*?)\n```', doc, re.S)
             if 'qwen-training tools/ssm/qwen_gdn_offload_training.py' in b)
subprocess.run(['bash', '-n'], input=block, text=True, check=True)
(OUT / 'documented-command.sh').write_text(block + '\n')
probe = OUT / 'probe.py'
probe.write_text('''import json, os, sys
from pathlib import Path
rank = int(os.environ['RANK'])
cache = Path(os.environ['TRITON_CACHE_DIR'])
assert cache.name == f'rank-{rank}'
assert 'torch' not in sys.modules and 'triton' not in sys.modules
cache.mkdir(parents=True, exist_ok=True)
marker = cache / 'ownership.json'
if not marker.exists():
    marker.write_text(json.dumps({'rank':rank, 'created_by':os.environ['DOC_TEST_ARM']}))
ownership = json.loads(marker.read_text())
assert ownership == {'rank':rank, 'created_by':'baseline'}
assert sys.argv[-2:] == ['--review-argument', 'two words']
output = Path(sys.argv[sys.argv.index('--output')+1])
output.with_name(f'{output.stem}.rank{rank}.json').write_text(json.dumps({
    'rank':rank, 'cache':str(cache), 'argv':sys.argv[1:], 'ownership':ownership,
    'environment_set_before_target_imports':True,
}, indent=2)+'\\n')
''')
setup, launch = block.split('python -m torch.distributed.run', 1)
launch = 'python -m torch.distributed.run' + launch
launch = launch.replace('tools/ssm/qwen_gdn_offload_training.py', shlex.quote(str(probe)))
script = 'set -eu\n' + setup + '\n'
for arm in ('baseline', 'repeat', 'offload'):
    command = launch.replace('/tmp/qwen-disabled-ranks', str(OUT / f'ranks-{arm}'))
    command = command.replace('/tmp/qwen-disabled.json', str(OUT / f'{arm}.json'))
    if arm == 'offload':
        command += ' --fraction 1'
    command += " --review-argument 'two words'"
    script += f'export DOC_TEST_ARM={arm}\n' + command + '\n'
(OUT / 'launcher-smoke.sh').write_text(script)
subprocess.run(['bash', '-n'], input=script, text=True, check=True)
command = ['docker', 'run', '--rm', '--detach', '--name', NAME, '--network', 'none',
           '--hostname', 'gdn-doc-cpu', '--add-host', 'gdn-doc-cpu:127.0.0.1']
for source, target in [(ART, ART), (ART, pathlib.Path('/artifacts')), (VENV, VENV),
                       (pathlib.Path('/data/songbinbin/minicoda'), pathlib.Path('/data/songbinbin/minicoda'))]:
    command += ['--mount', f'type=bind,source={source},target={target},readonly']
command += ['--mount', f'type=bind,source={OUT},target={OUT}']
for key, value in {'PATH':f'{VENV}/bin:/usr/local/bin:/usr/bin:/bin',
                   'CUDA_VISIBLE_DEVICES':'', 'NVIDIA_VISIBLE_DEVICES':'void',
                   'PYTHONDONTWRITEBYTECODE':'1', 'OMP_NUM_THREADS':'1'}.items():
    command += ['--env', f'{key}={value}']
command += ['sha256:061e8fda67a4dd1c1c635129d4ba0442b7a093cc6fd737423e9589eb6be22034',
            'sleep', 'infinity']
record = {'source_commit':SHA, 'gpu_used':False, 'container_command':command, 'checks':[]}
(OUT / 'validation.json').write_text(json.dumps(record, indent=2)+'\n')
subprocess.run(command, stdout=subprocess.DEVNULL, check=True)
try:
    jobs = [
        ('launcher-smoke', ['docker','exec',NAME,'bash',str(OUT/'launcher-smoke.sh')]),
        ('strict-docs', ['docker','exec','--workdir',str(SOURCE),NAME,str(DOCS_PYTHON),
         '-m','sphinx','-j','4','-b','html','-W','--keep-going','docs',str(OUT/'html')]),
    ]
    for name, cmd in jobs:
        item = {'name':name, 'command':cmd, 'started_utc':datetime.now(timezone.utc).isoformat()}
        with (OUT / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        item.update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        record['checks'].append(item)
        (OUT / 'validation.json').write_text(json.dumps(record, indent=2)+'\n')
        assert result.returncode == 0, name
        print(name + ' passed', flush=True)
    results = [[json.loads((OUT/f'{arm}.rank{rank}.json').read_text()) for rank in range(4)]
               for arm in ('baseline','repeat','offload')]
    first = results[0]
    assert len({row['cache'] for row in first}) == 4
    for rows in results[1:]:
        assert [row['cache'] for row in rows] == [row['cache'] for row in first]
        assert [row['ownership'] for row in rows] == [row['ownership'] for row in first]
    record['launcher_result'] = {'ranks':4, 'arms':3, 'independent_rank_directories':True,
        'cache_reused_between_arms':True, 'arguments_preserved':True,
        'environment_set_before_target_imports':True, 'target':'CPU probe; no model or Triton kernel execution'}
    record['production_diff_from_gpu_tested_commit'] = subprocess.check_output(
        ['git','diff','--name-only','8a2d19894288a8eb890dfdf20c9d54683035de80',SHA],cwd=REPO,text=True).splitlines()
    assert record['production_diff_from_gpu_tested_commit'] == ['docs/developer/gdn_activation_offloading.md']
finally:
    stopped = subprocess.run(['docker','stop','--timeout','10',NAME],text=True,capture_output=True)
    record['cleanup_exit_code'] = stopped.returncode
    (OUT/'cleanup.log').write_text(stopped.stdout+stopped.stderr)
    (OUT/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print('CPU container cleanup: '+str(stopped.returncode),flush=True)
