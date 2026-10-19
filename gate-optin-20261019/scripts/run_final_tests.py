import json, subprocess, sys, time, importlib.util
from pathlib import Path
from datetime import datetime, timezone
OUT=Path(__file__).resolve().parents[1]
HELPERS=Path('/data/songbinbin/Proj/Proj_LWX/.codex-artifacts/gdn-followup-20261016/scripts')
sys.path.insert(0,str(HELPERS))
spec=importlib.util.spec_from_file_location('gpu_helper',HELPERS/'run_gpu_acceptance.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
CONTAINER='megatron-gate-optin-dev-20261019'
SOURCE_ROOT=Path('/home/songbinbin/gdn-offload-artifacts/split-pr-20261015')
PYPATH=lambda source:helper.python_path(source,SOURCE_ROOT)
GATE='tests/unit_tests/determinism/kernels/test_gdn_gate_backward.py'
JOBS=[('final-gate-replay','source-final',GATE,0),('final-registry','source-final','tests/unit_tests/determinism/kernels/test_manifest.py',0),('final-fusion','source-final','tests/unit_tests/ssm/test_gated_delta_net.py::TestFusedPreGatedDeltaRule',0)]
manifest={'source':json.loads((OUT/'final-source-identity.json').read_text())['source'],'jobs':[],'status':'running'}
for name,src,path,expected in JOBS:
 helper.require_idle(helper.gpu_inventory(['1']))
 directory=OUT/name;directory.mkdir(exist_ok=True)
 source=OUT/src
 cmd=['docker','exec','--workdir',str(source),'--env','PYTHONPATH='+PYPATH(source),CONTAINER,str(helper.VENV/'bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3','--tee','3',str(OUT/'scripts/rank_pytest.py'),str(directory/'reports'),'-q','-x','-p','no:cacheprovider','--confcutdir='+str(source/'tests/unit_tests'),path]
 # Keep this host's optional dataset/conftest downloads outside isolated kernel/unit runs.
 cmd[cmd.index('--confcutdir='+str(source/'tests/unit_tests'))]='--confcutdir='+str(source/Path(path.split('::')[0]).parent)
 if name=='final-fusion':cmd[cmd.index(CONTAINER):cmd.index(CONTAINER)]=['--env','CAUSAL_CONV1D_DETERMINISTIC=1']
 row={'name':name,'command':cmd,'started':datetime.now(timezone.utc).isoformat(),'expected_exit':expected};manifest['jobs'].append(row)
 (OUT/'final-test-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('START',name,flush=True)
 with (directory/'launcher.log').open('w') as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while process.poll() is None:time.sleep(1)
 row['exit_code']=process.returncode;row['finished']=datetime.now(timezone.utc).isoformat()
 (OUT/'final-test-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('DONE',name,process.returncode,flush=True)
 if process.returncode!=expected:
  manifest['status']='failed';(OUT/'final-test-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');raise SystemExit(1)
manifest['status']='completed_pending_verification';(OUT/'final-test-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
