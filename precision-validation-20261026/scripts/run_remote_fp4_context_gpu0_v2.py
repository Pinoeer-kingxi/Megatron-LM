"""Reproduce absent NVFP4 GEMMs in the old tool, then validate the corrected tool."""
import json
import os
import subprocess
import time
from pathlib import Path

base=Path('/home/lwx/.cache/megatron-gdn-20261026')
outputs=[]
for arm in ('old-disabled', 'disabled', 'repeated-disabled', 'offloaded'):
    old=arm=='old-disabled'
    source=base/('precision' if old else 'precision-context')
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    directory=base/f'qwen-sm120-fp4-context-v2-{arm}'
    directory.mkdir()
    command=['/usr/bin/python3',str(base/'remote_container_gpu0.py'),'/home/songbinbin/.venvs/megatron-gdn-offload/bin/python',
       '-m','torch.distributed.run','--standalone','--nproc-per-node=1','--log-dir',str(directory/'ranks'),'--redirects','3',
       str(base/'rank_env.py'),str(base/'observe_qwen_gemm_v2.py'),str(directory/'gemm.json'),'NVFP4',str(source/'tools/ssm/qwen_gdn_offload_training.py'),
       '--weights',str(base/'models/Qwen3.5-0.8B'),'--data',str(base/'data/ks-item-sft.jsonl'),
       '--precision','fp4','--seq-length','128','--microbatches','4','--warmup','1','--iterations','2',
       '--min-offloaded-tensor-size','1024','--recompute-norm','--check-state','--output',str(directory/'result.json')]
    if arm=='offloaded': command+=['--fraction','1']
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
    manifest=dict(revision=revision,command=command,started_unix=time.time(),expected_exit=1 if old else 0,
                  gpu_uuid='GPU-69c3be35-a9f7-eae1-6746-0f1e08f57e9a',nvfp4_disable_rht=True,nvfp4_disable_stochastic_rounding=True)
    with (directory/'launcher.log').open('w') as log:
        result=subprocess.run(command,env=dict(os.environ,GDN_SOURCE=str(source)),stdout=log,stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode,finished_unix=time.time(),revision_after=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),dirty_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert manifest['revision_after']==revision and not manifest['dirty_after']
    observed=json.loads((directory/'gemm.json').read_text())
    actual=any('NVFP4' in key for key in observed['gemm_operand_types'])
    assert actual != old and result.returncode==manifest['expected_exit'],manifest
    output=json.loads((directory/'result.rank0.json').read_text())
    assert bool(output['offload_status']['selected_transfer_bytes'])==(arm=='offloaded')
    if not old: outputs.append(output['states'])
    print('Finished '+arm+'; actual_nvfp4='+str(actual),flush=True)
comparison=dict(revision=revision,exact_states=outputs[0]==outputs[1]==outputs[2],
    pairs={f'{a} vs {b}':outputs[i]==outputs[j] for i,a in enumerate(('disabled','repeated-disabled','offloaded')) for j,b in enumerate(('disabled','repeated-disabled','offloaded')) if i<j})
(base/'qwen-sm120-fp4-context-v2-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
print(json.dumps(comparison),flush=True)
assert comparison['exact_states']
