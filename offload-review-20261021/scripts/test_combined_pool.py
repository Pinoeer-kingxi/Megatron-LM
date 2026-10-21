"""Run the repository's real layer-overlapped MoE schedule with two EP ranks."""

import inspect

from tests.unit_tests.pipeline_parallel import test_fine_grained_activation_offloading as original


def test_combined_pool():
    name = "test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility"
    source = inspect.getsource(getattr(original, name))
    source = source[source.index("def " + name):]
    assert source.count("ep_size = 4") == 1
    source = source.replace("ep_size = 4", "ep_size = 2")
    namespace = vars(original).copy()
    exec(compile(source, inspect.getsourcefile(original), "exec"), namespace)
    namespace[name](
        "alltoall", False,
        ["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"],
    )
