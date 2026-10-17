"""Recheck this immutable evidence bundle on CPU without model or GPU dependencies."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def checksums():
    entries = (ROOT / 'SHA256SUMS').read_text().splitlines()
    for line in entries:
        expected, name = line.split(None, 1)
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    print(f'PASS: {len(entries)} exported file checksums', flush=True)


checksums()
versions = read('versions.json')
assert versions['feature'] == '87c22d1abae13abf375e8ed4ddfe62301cde87f4'
maps = []
for stage in ('four-gpu', 'scope', 'performance'):
    source = read(stage + '/source.json')
    assert source['source_commit'] == versions['feature']
    archive = read(stage + '/frozen-source-check.json')
    assert len(archive) == 1 and archive[0]['archive_files_unchanged']
    assert archive[0]['source_commit'] == versions['feature']
    maps.append(archive[0]['files'])
assert maps[0] == maps[1] == maps[2]
initial = read('four-gpu-initial/materialization-check.json')
old_archive = read('two-gpu/frozen-source-check.json')
assert initial['regular_file_content_match']
assert initial['source_commit'] == versions['previous_feature'] == old_archive[0]['source_commit']
assert initial['files'] == old_archive[0]['files']
assert len(initial['dereferenced_symlinks']) == 6
assert old_archive[1]['source_commit'] == versions['gate']
four_inputs = {row['file']: row['sha256'] for row in read('four-gpu/input-provenance.json')['files']}
assert four_inputs == read('performance/manifest.json')['input_sha256']
for name, expected in read('performance/manifest.json')['script_sha256'].items():
    assert hashlib.sha256((ROOT / 'performance/scripts' / name).read_bytes()).hexdigest() == expected

for stage in ('two-gpu', 'four-gpu', 'scope'):
    subprocess.run([sys.executable, '-B', '-S', str(ROOT / stage / 'scripts/verify_validation.py')],
                   cwd=ROOT / stage, check=True)
subprocess.run([sys.executable, '-B', '-S', str(ROOT / 'performance/scripts/summarize_performance.py')],
               cwd=ROOT / 'performance', check=True)

# Recompute the exact observation from raw states, not just the difference reports.
# This describes these recorded retests; native fusion's acceptance remains numerical.
for stage, ranks, prefix in [('four-gpu', 4, 'qwen'), ('two-gpu', 2, 'numeric')]:
    for norm in (0, 1):
        for rank in range(ranks):
            baseline = read(f'{stage}/{prefix}-f1-n{norm}-disabled/result.rank{rank}.json')['states']
            for arm in ('repeat', 'offload'):
                actual = read(f'{stage}/{prefix}-f1-n{norm}-{arm}/result.rank{rank}.json')['states']
                assert baseline == actual, (stage, norm, rank, arm)
print('PASS: raw fused retest fingerprints match the recorded zero-difference observation', flush=True)
checksums()
print('PASS: source identities, raw rank outcomes, complete states and all 120 timing samples verified; no GPU execution performed.')
