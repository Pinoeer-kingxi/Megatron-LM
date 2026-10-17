"""Verify that the executed source archives were not changed by the runs."""

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'source.json').read_text())
results = []
for directory, expected in [('source', source['source_commit'])]:
    hashes = {}
    with tarfile.open(ROOT / (directory + '.tar')) as archive:
        assert archive.pax_headers['comment'] == expected
        for member in archive.getmembers():
            path = ROOT / directory / member.name
            if member.isfile():
                expected_bytes = archive.extractfile(member).read()
                assert path.read_bytes() == expected_bytes, str(path)
                hashes[member.name] = hashlib.sha256(expected_bytes).hexdigest()
            elif member.issym():
                assert path.is_symlink() and str(path.readlink()) == member.linkname
    results.append(dict(directory=directory, source_commit=expected,
                        checked_files=len(hashes), files=hashes, archive_files_unchanged=True))
(ROOT / 'frozen-source-check.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps([{k:v for k,v in row.items() if k != 'files'} for row in results], indent=2))
