import subprocess
from pathlib import Path
base='/home/lwx/.cache/megatron-gdn-20261026'
root=Path('/home/songbinbin/gdn-offload-artifacts')
for source,target in [(root/'models/Qwen3.5-0.8B', 'models/'),(root/'training-data/ks-item-sft.jsonl','data/')]:
    subprocess.run(['ssh','A6000PRO','mkdir -p '+base+'/'+target],check=True)
    subprocess.run(['rsync','-a',str(source),'A6000PRO:'+base+'/'+target],check=True)
