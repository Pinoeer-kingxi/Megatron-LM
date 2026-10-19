import argparse, importlib.util, json, os, sys
from pathlib import Path
import torch
from megatron.core.tuning import AutotunePolicy, choice_digest, choice_log, install, use_deterministic_mode
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--record-configs',type=Path);p.add_argument('--table-path',type=Path)
own,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
assert not use_deterministic_mode()
modules=('megatron.core','transformer_engine','fla')
if own.record_configs:
 install(AutotunePolicy(mode='record',modules=modules,record_path=str(own.record_configs)))
else:
 assert own.table_path is not None
 install(AutotunePolicy(mode='pinned',modules=modules,table_path=(str(own.table_path),),on_miss='error'))
path=Path(__file__).with_name('qwen_training.py')
spec=importlib.util.spec_from_file_location('qwen_training',path);training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
original_batches=training._batches;inputs=[];vectors={};original_fingerprint=training._fingerprint
capture_states='--check-state' in remaining

def capture(value):
 result=original_fingerprint(value)
 if capture_states and isinstance(value,torch.Tensor) and value.ndim<=1:
  vectors[(result['dtype'],tuple(result['shape']),result['sha256'])]=value.detach().cpu().clone()
 return result
training._fingerprint=capture

def batches(*args,**kwargs):
 result=original_batches(*args,**kwargs)
 inputs.append([{key:original_fingerprint(value) for key,value in batch.items()} for batch in result])
 return result
training._batches=batches
training.main()
assert not use_deterministic_mode()
output=Path(remaining[remaining.index('--output')+1]).parent
(output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
(output/'choices.json').write_text(json.dumps({'digest':choice_digest(),'choices':choice_log(),'deterministic_arithmetic':False},indent=2)+'\n')
if capture_states:torch.save(vectors,output/'vectors.rank0.pt')
