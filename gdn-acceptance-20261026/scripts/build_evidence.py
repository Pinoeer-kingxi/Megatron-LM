"""Copy only reviewable logs, commands and source metadata, never model/data payloads."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
target=root/'evidence-final/gdn-acceptance-20261026'
target.mkdir(parents=True,exist_ok=True)
names=['lifecycle-fixed','lifecycle-control-eval-matched','lifecycle-controls-calibrated',
       'lifecycle-candidate','shared-native-lifecycle','feature-native-tests',
       'qwen-correctness-disabled','qwen-correctness-repeated-disabled','qwen-correctness-offloaded',
       'feature-final-lifecycle','shared-control-growth','shared-control-evaluation',
       'remote-prototype','remote-native-final','remote-environment-attempts','shared-graph-compatibility','shared-graph-fractions']
names += [p.name for p in sorted(root.glob('performance-*')) if p.is_dir()]
for name in names:
    source=root/name
    if source.exists(): shutil.copytree(source,target/name,dirs_exist_ok=True)
for path in root.glob('*.json'):
    if path.name.startswith('pr'): continue
    shutil.copyfile(path,target/path.name)
for pattern in ('shared-*.log','audit-output.log','precision-autoformat.log','remote-*.log','flash-attn-source.json'):
    for path in root.glob(pattern): shutil.copyfile(path,target/path.name)
shutil.copytree(root/'scripts',target/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
identities={}
for name in ('shared-final','feature-final','candidate','precision'):
    source=root/name
    def git(*args): return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    identities[name]=dict(commit=git('rev-parse','HEAD'),production_tree=git('rev-parse','HEAD:megatron'),
                          tests_tree=git('rev-parse','HEAD:tests'),dirty=git('status','--porcelain'))
(target/'source-identities.json').write_text(json.dumps(identities,indent=2)+'\n')
checksums=[]
for path in sorted(target.rglob('*')):
    if path.is_file() and path.name!='SHA256SUMS':
        checksums.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(target)))
(target/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
print(str(target),len(checksums),flush=True)
