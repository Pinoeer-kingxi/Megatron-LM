"""Print an existing exception before graph cleanup can mask its first failure."""
import runpy
import sys
import traceback

from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

original = TECudaGraphHelper.delete_cuda_graphs


def delete(self):
    if sys.exc_info()[0] is not None:
        print('Original failure before graph cleanup; offload=' +
              str(self.config.fine_grained_activation_offloading), file=sys.stderr, flush=True)
        traceback.print_exc()
    return original(self)


TECudaGraphHelper.delete_cuda_graphs = delete
runpy.run_path('/home/lwx/.cache/megatron-gdn-20261026/rank_distributed_pytest.py',
              run_name='__main__')
