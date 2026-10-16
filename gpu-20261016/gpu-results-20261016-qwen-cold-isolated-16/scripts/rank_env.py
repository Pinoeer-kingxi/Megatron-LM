"""Give each rank sole ownership of its persistent Triton autotuning results."""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    os.environ["TRITON_CACHE_DIR"] = str(
        Path(os.environ["TRITON_CACHE_DIR"]) / f"rank-{os.environ['RANK']}"
    )
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
