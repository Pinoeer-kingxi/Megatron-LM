"""Recheck the scope job without CUDA or third-party Python dependencies."""
import json
from pathlib import Path
from verify_results import verify_pytest

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'validation.json').read_text())
assert manifest['source_commit'] == '87c22d1abae13abf375e8ed4ddfe62301cde87f4'
assert manifest['status'] == 'passed' and manifest['cleanup']['exit_code'] == 0
assert len(manifest['jobs']) == 1 and manifest['jobs'][0]['exit_code'] == 0
result = dict(source_commit=manifest['source_commit'], scope=verify_pytest(ROOT / 'gdn-scope/reports', 1, 73))
(ROOT / 'validation-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS: 73 scope/configuration cases, zero skips; source and job exits verified.')
