"""Restore session cleanup when isolated pytest excludes the root conftest."""

import sys

import pytest
import torch

status = pytest.main(sys.argv[1:])
if torch.distributed.is_initialized():
    torch.cuda.synchronize()
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    torch.distributed.destroy_process_group()
sys.exit(status)
