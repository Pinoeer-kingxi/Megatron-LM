"""Exercise lifecycle boundaries through native decoder/DDP/BF16 Adam training."""

import json
import os
import copy
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, diff
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    FineGrainedActivationOffloadingInterface as off_interface,
    PipelineOffloadManager,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.ssm.test_gdn_offload_training import (
    _build_training_model,
    _forward_step,
    _snapshot_leaf,
    _training_config,
)
from tests.unit_tests.test_utilities import Utils


def differences(expected, actual, prefix=()):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        return [row for name in expected for row in differences(expected[name], actual[name], prefix + (name,))]
    if isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        return [row for index, (left, right) in enumerate(zip(expected, actual))
                for row in differences(left, right, prefix + (index,))]
    if isinstance(expected, torch.Tensor):
        if torch.equal(expected, actual):
            return []
        delta = (expected.float() - actual.float()).abs()
        row = dict(path=list(prefix), dtype=str(expected.dtype), shape=list(expected.shape),
                   changed=int((expected != actual).sum()), max_absolute=float(delta.max()))
        if expected.numel() <= 64:
            row.update(expected=expected.tolist(), actual=actual.tolist())
        return [row]
    return [] if expected == actual else [dict(path=list(prefix), expected=expected, actual=actual)]


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "case,lengths,threshold",
    [
        ("fixed", [128, 128, 128], 1024),
        ("evaluation_first", [128, 128, 128], 1024),
        ("resize_after_warmup", [8, 256, 64, 256], 131072),
    ],
)
def test_native_lifecycle(case, lengths, threshold, fraction, monkeypatch):
    """Compare actual training states and actual copies, without editing manager state."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    Utils.initialize_model_parallel()
    references = []
    initial_state = None
    transfer_rows = []
    phase = "build"
    original_offload = ChunkOffloadHandler.offload

    def record_transfer(handler, *args, **kwargs):
        result = original_offload(handler, *args, **kwargs)
        backup = result[1]
        transfer_rows.append(
            dict(phase=phase, warmup=handler.is_warmup,
                 bytes=backup.numel() * backup.element_size())
        )
        return result

    monkeypatch.setattr(ChunkOffloadHandler, "offload", record_transfer)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        for arm_index, enabled in enumerate((False, False, False, True)):
            off_interface.reset_instance()
            config = replace(
                _training_config(1, False, 1, 1, None, fraction, False),
                fine_grained_activation_offloading=enabled,
                offload_modules=["gdn_core_attn"] if enabled else [],
                min_offloaded_tensor_size=threshold,
            )
            models, optimizer = _build_training_model(config, groups)
            if initial_state is None:
                initial_state = copy.deepcopy(models[0].state_dict())
            else:
                models[0].load_state_dict(initial_state)
                optimizer.reload_model_params()
            manager = PipelineOffloadManager.get_instance()
            pool_checks = []

            def finalize(*args, **kwargs):
                finalize_model_grads(*args, **kwargs)
                in_use = manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"]
                pool_checks.append(in_use)
                assert in_use == 0

            if enabled:
                config.finalize_model_grads_func = partial(finalize, pg_collection=groups)
            schedule = get_forward_backward_func(pp_size=1, vp_size=None)
            generator = torch.Generator().manual_seed(19 + groups.dp.rank())

            def run_schedule(length, forward_only):
                batches = []
                for _ in range(4):
                    tokens = torch.randint(0, 512, (1, length + 1), generator=generator).cuda()
                    batches.append(dict(tokens=tokens[:, :-1].contiguous(),
                                        labels=tokens[:, 1:].contiguous(),
                                        position_ids=torch.arange(length, device="cuda").unsqueeze(0)))
                return schedule(
                    forward_step_func=_forward_step, data_iterator=[iter(batches)],
                    model=models, num_microbatches=4, seq_length=length,
                    micro_batch_size=1, forward_only=forward_only,
                    p2p_communicator=P2PCommunicator(groups.pp, config), pg_collection=groups,
                )

            evaluations = []
            if case == "evaluation_first":
                phase = "evaluation"
                for model in models:
                    model.eval()
                with torch.no_grad():
                    evaluations = run_schedule(128, True)
                for model in models:
                    model.train()

            states = []
            for step, length in enumerate(lengths):
                phase = f"train-{step}"
                optimizer.zero_grad()
                for model in models:
                    model.zero_grad_buffer()
                losses = run_schedule(length, False)
                gradients = {name: p.main_grad.detach().clone()
                             for name, p in models[0].named_parameters()}
                success, grad_norm, _ = optimizer.step()
                assert success and grad_norm > 0
                states.append(dict(losses=losses, grads=gradients, grad_norm=grad_norm,
                                   weights={name: p.detach().clone()
                                            for name, p in models[0].named_parameters()},
                                   optimizer=dict_list_map_outplace(_snapshot_leaf, optimizer.state_dict())))
            torch.cuda.synchronize()
            if enabled:
                state_differences = differences(references, [evaluations, states])
                assert len(pool_checks) == len(lengths)
                steady = [row for row in transfer_rows if not row["warmup"]]
                report = dict(case=case, lengths=lengths, threshold=threshold, fraction=fraction,
                              rank=torch.distributed.get_rank(), states_exact=not state_differences,
                              state_differences=state_differences,
                              pool_before_reset=pool_checks, transfers=transfer_rows,
                              pool=manager.cpu_tensor_pool.get_pool_status()["global_stats"],
                              selected_summary_bytes=manager.offload_summary_total_bytes)
                directory = Path(os.environ["LIFECYCLE_OUTPUT"])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{case}-fraction-{fraction}.rank{torch.distributed.get_rank()}.json").write_text(
                    json.dumps(report, indent=2) + "\n")
                assert not state_differences, state_differences
                assert bool(steady) == (fraction > 0), report
            else:
                if references:
                    directory = Path(os.environ["LIFECYCLE_OUTPUT"])
                    directory.mkdir(parents=True, exist_ok=True)
                    baseline_differences = differences(references, [evaluations, states])
                    (directory / f"{case}-fraction-{fraction}.baseline-{arm_index}.rank{torch.distributed.get_rank()}.json").write_text(
                        json.dumps(baseline_differences, indent=2) + "\n")
                    if arm_index == 2:
                        assert not baseline_differences, baseline_differences
                references = [evaluations, states]
            del models, optimizer, states
            torch.cuda.synchronize()
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()
