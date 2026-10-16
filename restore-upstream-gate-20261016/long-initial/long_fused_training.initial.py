"""Compare native fused training with a repeated disabled control at sequence 2048."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, diff
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
    OffloadTensorPool,
    PipelineOffloadManager,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.ssm.test_gdn_offload_training import (
    _build_training_model, _forward_step, _training_config,
)
from tests.unit_tests.test_utilities import Utils

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--recompute-norm', action='store_true')
args = parser.parse_args()
torch.cuda.set_device(0)
Utils.initialize_model_parallel()
off_interface.reset_instance()
groups = ProcessGroupCollection.use_mpu_process_groups()
config = _training_config(1, False, 1, 1, None, 1.0, args.recompute_norm, fused_pre_gdr=True)
disabled = replace(config, fine_grained_activation_offloading=False, offload_modules=[])
arms = [_build_training_model(c, groups) for c in (disabled, disabled, config)]
for models, optimizer in arms[1:]:
    for expected, actual in zip(arms[0][0], models, strict=True):
        actual.load_state_dict(expected.state_dict())
    optimizer.reload_model_params()
schedule = get_forward_backward_func(pp_size=1, vp_size=None)
generator = torch.Generator().manual_seed(19)
batches = []
for step in range(3):
    step_batches = []
    for microbatch in range(4):
        tokens = torch.randint(0, 512, (1, 2049), generator=generator).cuda()
        step_batches.append({'tokens': tokens[:, :-1].contiguous(),
                             'labels': tokens[:, 1:].contiguous(),
                             'position_ids': torch.arange(2048, device='cuda').unsqueeze(0)})
    batches.append(step_batches)

ownership = []
original_reset = OffloadTensorPool.reset

def reset_pool(pool):
    ownership.append(pool.get_pool_status()['global_stats']['current_in_use'])
    original_reset(pool)

OffloadTensorPool.reset = reset_pool
states = []
for arm, (models, optimizer) in enumerate(arms):
    arm_states = []
    for step in range(3):
        optimizer.zero_grad()
        for model in models:
            model.zero_grad_buffer()
        losses = schedule(forward_step_func=_forward_step,
                          data_iterator=[iter(batches[step]) for _ in models], model=models,
                          num_microbatches=4, seq_length=2048, micro_batch_size=1,
                          forward_only=False, p2p_communicator=P2PCommunicator(groups.pp, models[0].config),
                          pg_collection=groups)
        grads = {f'{stage}.{name}': p.main_grad.detach().cpu().clone()
                 for stage, model in enumerate(models) for name, p in model.named_parameters()}
        success, norm, _ = optimizer.step()
        assert success and norm > 0
        torch.cuda.synchronize()
        def cpu_leaf(value):
            return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        arm_states.append({'losses': dict_list_map_outplace(cpu_leaf, losses), 'grad_norm': cpu_leaf(norm),
                           'grads': grads,
                           'weights': {f'{stage}.{name}': p.detach().cpu().clone()
                                       for stage, model in enumerate(models)
                                       for name, p in model.named_parameters()},
                           'optimizer': dict_list_map_outplace(cpu_leaf, optimizer.state_dict())})
        print(f'Completed arm={arm} step={step}', flush=True)
    states.append(arm_states)
    torch.save(arm_states, args.output.with_name(args.output.stem + f'.arm{arm}.pt'))

# Fixed tolerances are declared before inspection. Both comparisons use the same
# bounds; the repeated disabled arm exposes the native reduction variation.
rtol, atol = 1e-5, 1e-8
comparisons = []

def tensor_differences(expected, actual, path=''):
    if isinstance(expected, dict):
        for key in expected:
            yield from tensor_differences(expected[key], actual[key], path + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            yield from tensor_differences(left, right, path + '/' + str(index))
    elif isinstance(expected, torch.Tensor):
        delta = (actual.double() - expected.double()).abs()
        yield {'path': path, 'dtype': str(expected.dtype), 'elements': expected.numel(),
               'exact': torch.equal(expected, actual),
               'max_absolute_difference': delta.max().item() if delta.numel() else 0,
               'max_tolerance_ratio': (delta / (atol + rtol * expected.double().abs())).max().item()
                   if delta.numel() else 0}

for name, expected, actual in [('disabled-repeat', states[0], states[1]),
                                ('offload', states[0], states[2])]:
    differences = list(tensor_differences(expected, actual))
    (args.output.parent / (name + '-tensor-differences.json')).write_text(json.dumps(differences, indent=2) + '\n')
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    comparisons.append({'name': name, 'numerically_close': True,
                        'exact': not any(diff(expected, actual)),
                        'tensor_leaves': len(differences),
                        'differing_tensor_leaves': sum(not row['exact'] for row in differences),
                        'max_tolerance_ratio': max(row['max_tolerance_ratio'] for row in differences)})
assert ownership and all(count == 0 for count in ownership)
manager = PipelineOffloadManager.get_instance()
assert not manager._is_warmup and manager.offload_summary_total_bytes > 0
result = {'source_commit': json.loads((args.output.parents[1] / 'source.json').read_text())['source_commit'],
          'sequence': 2048, 'steps_per_arm': 3, 'microbatches': 4, 'fused_pre_gdr': True,
          'recompute_norm': args.recompute_norm, 'rtol': rtol, 'atol': atol,
          'comparisons': comparisons, 'pool_ownership_before_reset': ownership,
          'offload_summary_bytes': manager.offload_summary_total_bytes,
          'scope': 'Reduced complete native decoder; this is not pretrained Qwen qualification.'}
args.output.write_text(json.dumps(result, indent=2) + '\n')
OffloadTensorPool.reset = original_reset
torch.cuda.synchronize()
off_interface.reset_instance()
Utils.destroy_model_parallel()
torch.distributed.destroy_process_group()
