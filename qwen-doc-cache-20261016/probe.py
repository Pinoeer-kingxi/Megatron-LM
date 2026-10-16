import json, os, sys
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
}, indent=2)+'\n')
