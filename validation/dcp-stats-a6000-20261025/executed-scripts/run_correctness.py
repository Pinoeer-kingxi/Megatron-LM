import os, subprocess, sys
from pathlib import Path
base=Path(__file__).resolve().parent
steps=[
 ("2,4",["final-2-off","--audit","--load","initial-aligned","--save","--eval"]),
 ("2,4",["final-2-on","--audit","--load","initial-aligned","--save","--eval","--on"]),
 ("2,4,5,6",["initial-tp","--gpus","4","--tp","2","--iters","1","--save"]),
 ("2,4,5,6",["final-tp-off","--gpus","4","--tp","2","--audit","--load","initial-tp","--save","--eval"]),
 ("2,4,5,6",["final-tp-on","--gpus","4","--tp","2","--audit","--load","initial-tp","--save","--eval","--on"]),
 ("2,4,5,6",["initial-pp-v2","--gpus","4","--pp","2","--vpp","--iters","1","--save"]),
 ("2,4,5,6",["final-pp-off","--gpus","4","--pp","2","--vpp","--audit","--load","initial-pp-v2","--save","--eval"]),
 ("2,4,5,6",["final-pp-on","--gpus","4","--pp","2","--vpp","--audit","--load","initial-pp-v2","--save","--eval","--on"])]
for devices,args in steps:
 env=dict(os.environ,DCP_REPO_ROOT="/data/songbinbin/Proj/Proj_LWX/Megatron-LM-dcp-revalidation",DCP_GPU_INDICES=devices)
 subprocess.run([sys.executable,"-S",str(base/"launch_train.py"),*args,"--container","dcp-stats-revalidation"],env=env,check=True)
