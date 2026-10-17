"""Parse the changed documentation and verify that reproduction commands are preserved."""

import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

from docutils import nodes
from docutils.core import publish_doctree
from myst_parser.parsers.docutils_ import Parser

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'manifest.json').read_text())
before = (ROOT / 'document-before.md').read_text()
current = (ROOT / 'document.md').read_text()
assert hashlib.sha256(current.encode()).hexdigest() == manifest['document_sha256']
assert not any(Path(f'/dev/nvidia{index}').exists() for index in range(8))
warnings = io.StringIO()
document = publish_doctree(current, parser=Parser(), source_path='gdn_activation_offloading.md',
                          settings_overrides={'warning_stream': warnings, 'halt_level': 2,
                                              'report_level': 2, 'file_insertion_enabled': False})
messages = list(document.findall(nodes.system_message))
assert not messages and not warnings.getvalue(), warnings.getvalue()
blocks = re.findall(r'```bash\n(.*?)\n```', current, re.S)
assert blocks == re.findall(r'```bash\n(.*?)\n```', before, re.S)
for block in blocks:
    subprocess.run(['bash', '-n'], input=block, text=True, check=True)
for value in ['4e18562c6bcf894795d23378a8f1699c79e4b797',
              '2f1ee4f2e3dd6d909d08981338f5879632131139',
              '+1.58%', '+5.80%', '952.00 MiB']:
    assert value in before and value in current, value
assert '| Recorded workload |' not in current
assert '6ca000df1d21df1fe042be13f008bd900de1f9b4' in current
result = {'head': manifest['head'], 'myst_parse_passed': True,
          'parser_warning_count': len(messages), 'unchanged_bash_blocks': len(blocks),
          'bash_syntax_passed': True, 'historical_performance_versions_preserved': True,
          'gpu_devices_exposed': 0, 'full_sphinx_build': False}
(ROOT / 'document-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
