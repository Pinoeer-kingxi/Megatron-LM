"""Change one graph API option per diagnostic process; leave numerical code intact."""
import os
import runpy
import sys
import traceback

from megatron.core.transformer import cuda_graphs

mode = os.environ['GDN_GRAPH_PROBE_MODE']
original = cuda_graphs.make_graphed_callables


def capture(*args, **kwargs):
    overrides = {
        'plain': {},
        'no-weight-cache': {'fp8_weight_caching': False},
        'no-io-reuse': {'_reuse_graph_input_output_buffers': False},
        'clone-param-grads': {'clone_param_grads_on_return': True},
        'trace': {},
        'disable-existing-cache': {},
        'disable-both-caches': {'fp8_weight_caching': False},
    }[mode]
    kwargs.update(overrides)
    print('Graph option diagnostic:', mode, overrides, flush=True)
    return original(*args, **kwargs)


cuda_graphs.make_graphed_callables = capture
original_delete = cuda_graphs.TECudaGraphHelper.delete_cuda_graphs


def delete(self):
    if sys.exc_info()[0] is not None:
        print('Original failure before graph cleanup; offload=' +
              str(self.config.fine_grained_activation_offloading), file=sys.stderr, flush=True)
        traceback.print_exc()
    return original_delete(self)


cuda_graphs.TECudaGraphHelper.delete_cuda_graphs = delete
if mode in ('disable-existing-cache', 'disable-both-caches'):
    from dataclasses import replace
    from tests.unit_tests.ssm import test_gdn_offload_cuda_graphs as graph_tests
    training_config = graph_tests._training_config

    def without_weight_cache(*args, **kwargs):
        return replace(training_config(*args, **kwargs), disable_parameter_transpose_cache=True)

    graph_tests._training_config = without_weight_cache
if mode == 'trace':
    import torch
    from megatron.core.transformer.module import GraphableMegatronModule
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    replay = GraphableMegatronModule._te_cuda_graph_replay

    def traced_replay(self, *args, **kwargs):
        value = args[0] if args else kwargs['hidden_states']
        flag = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        print('REPLAY INPUT', self.layer_number, self.current_microbatch,
              bool(torch.isfinite(value).all()), 'skip_weight_update', flag.item(), flush=True)
        result = replay(self, *args, **kwargs)
        for index, output in enumerate(result):
            if isinstance(output, torch.Tensor):
                print('REPLAY OUTPUT', self.layer_number, self.current_microbatch, index,
                      bool(torch.isfinite(output).all()), flush=True)
        return result

    GraphableMegatronModule._te_cuda_graph_replay = traced_replay
runpy.run_path('/home/lwx/.cache/megatron-gdn-20261026/rank_distributed_pytest.py',
              run_name='__main__')
