import os
from pathlib import Path
os.environ["CUDA_HOME"] = "/home/lwx/.cache/megatron-gdn-20261026/apex-wgrad-build/cuda"
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parent
cuda = Path('/home/songbinbin/.venvs/megatron-gdn-offload/lib/python3.12/site-packages/nvidia/cu13')
setup(name='fused_weight_gradient_mlp_cuda', ext_modules=[CUDAExtension(
    name='fused_weight_gradient_mlp_cuda',
    sources=[str(root / name) for name in ('binding.cpp', 'fp32.cu', 'fp16.cu')],
    include_dirs=[str(root), str(root / 'cuda-include'), str(root / 'cuda-include/cccl')],
    library_dirs=[str(root / 'lib')],
    libraries=['cublas', 'cudart', 'c10_cuda', 'torch_cuda'],
    extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '--expt-relaxed-constexpr', '--expt-extended-lambda', '--use_fast_math']},
)], cmdclass={'build_ext': BuildExtension})
