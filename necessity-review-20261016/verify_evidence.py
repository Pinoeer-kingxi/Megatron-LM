"""Verify the documentation-only review evidence without external dependencies."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'manifest.json').read_text())
check = json.loads((ROOT / 'document-check.json').read_text())
command = json.loads((ROOT / 'document-check-command.json').read_text())
assert manifest['head'] == 'a8d3a787b7ee24bf8bc1b927f8c3208d63b378fb' == check['head']
assert manifest['changed_paths'] == ['docs/developer/gdn_activation_offloading.md']
assert set(manifest['unchanged_trees']) == {'megatron', 'tools', 'tests'}
assert not manifest['gpu_used_this_round'] and not command['gpu_used']
assert command['exit_code'] == 0 and check['myst_parse_passed']
assert check['parser_warning_count'] == 0 and check['gpu_devices_exposed'] == 0
assert not check['full_sphinx_build']
before = (ROOT / 'document-before.md').read_text()
after = (ROOT / 'document.md').read_text()
assert hashlib.sha256(after.encode()).hexdigest() == manifest['document_sha256']
assert re.findall(r'```bash\n(.*?)\n```', before, re.S) == re.findall(r'```bash\n(.*?)\n```', after, re.S)
assert check['unchanged_bash_blocks'] == 4 and check['bash_syntax_passed']
assert check['historical_performance_versions_preserved']
if (ROOT / 'SHA256SUMS').exists():
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
print('Documentation-only review evidence verified; no GPU result relabeled.')
