"""Observe real DMA event order without inserting waits or changing pool policy."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    OffloadTensorPool,
)


@pytest.fixture(autouse=True)
def trace_pool_reuse(request, monkeypatch):
    last_read = {}
    reuse = []
    scope = {}
    allocate = OffloadTensorPool.allocate
    reload = ChunkOffloadHandler.reload
    bulk_offload = ChunkOffloadHandler.bulk_offload_group

    def traced_allocate(pool, shape, dtype):
        buffer = allocate(pool, shape, dtype)
        address = buffer.data_ptr()
        if address in last_read:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            reuse.append((last_read[address], start, dict(scope), tuple(shape), str(dtype), address))
        return buffer

    def traced_reload(handler, state, *args, **kwargs):
        output = reload(handler, state, *args, **kwargs)
        if state[2]:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            last_read[state[1].data_ptr()] = end
        return output

    def traced_bulk(handler, group):
        scope.update(chunk=id(handler), group=id(group), name=group._name,
                     warmup=handler.is_warmup, group_index=handler._offloaded_group_index)
        return bulk_offload(handler, group)

    monkeypatch.setattr(OffloadTensorPool, "allocate", traced_allocate)
    monkeypatch.setattr(ChunkOffloadHandler, "reload", traced_reload)
    monkeypatch.setattr(ChunkOffloadHandler, "bulk_offload_group", traced_bulk)
    yield
    torch.cuda.synchronize()
    samples = [dict(metadata=meta, shape=shape, dtype=dtype, cpu_address=address,
                    h2d_end_to_next_d2h_start_ms=old.elapsed_time(new))
               for old, new, meta, shape, dtype, address in reuse]
    report = dict(nodeid=request.node.nodeid, rank=int(os.environ["RANK"]),
                  reuse_count=len(samples), samples=samples)
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"pool-rank-{os.environ['RANK']}.jsonl").open("a") as stream:
        stream.write(json.dumps(report) + "\n")
