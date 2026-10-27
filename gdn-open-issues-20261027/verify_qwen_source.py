import os
import sys
from pathlib import Path

import megatron.core

expected = Path(os.environ['GDN_SOURCE']).resolve()
actual = Path(megatron.core.__file__).resolve()
assert actual.is_relative_to(expected), (actual, expected)
print(f'Imported Megatron Core from {actual}', flush=True)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
