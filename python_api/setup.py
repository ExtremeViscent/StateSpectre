"""Build the state_spectre C++ extension (`state_spectre._state_spectre`).

The extension links the CUDA/torch-free OffloadAgent + common transport TUs
directly into a single libtorch-linked shared object. We deliberately use
CppExtension (not CUDAExtension): agent.cpp is *host* code that calls the CUDA
runtime API (cudaMemcpyAsync, cudaHostRegister, events) — there are no .cu
kernels to compile — so plain g++ + libcudart suffices. bindings.cpp is the only
TU that includes torch/pybind; torch/extension.h pulls in pybind11.

Build:
    cd python_api && python setup.py build_ext --inplace
Verify:
    cd python_api && python -c "import torch; import state_spectre; print('import OK')"
"""

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

SRC = os.path.join(ROOT, "src")
ABI = os.path.join(ROOT, "abi")
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

# All TUs compiled into the extension. The agent + common + rpc_client objects
# must be present or `import state_spectre` fails with undefined symbols.
sources = [
    os.path.join(SRC, "python", "bindings.cpp"),
    os.path.join(SRC, "agent", "agent.cpp"),
    os.path.join(SRC, "agent", "rpc_client.cpp"),
    os.path.join(SRC, "common", "wire.cpp"),
    os.path.join(SRC, "common", "wire_v2.cpp"),
    os.path.join(SRC, "common", "uds.cpp"),
    os.path.join(SRC, "common", "metrics.cpp"),
    os.path.join(SRC, "common", "numa_util.cpp"),
]

include_dirs = [
    ABI,
    os.path.join(SRC, "common"),
    os.path.join(SRC, "agent"),
    os.path.join(CUDA_HOME, "include"),
]

TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
library_dirs = [os.path.join(CUDA_HOME, "lib64"), TORCH_LIB]

# cudart: CUDA runtime API used by agent.cpp. numa: used by numa_util.cpp.
# c10_cuda: c10::cuda::getCurrentCUDAStream() in bindings.cpp (CppExtension does
# not link the CUDA half of libtorch by default).
libraries = ["cudart", "numa", "c10_cuda", "torch_cuda"]

# libtorch on this machine is built with the CXX11 ABI (verified:
# torch._C._GLIBCXX_USE_CXX11_ABI == True). Force it on for the non-torch TUs
# (agent/common) too so std::string layout matches across the link.
abi_flag = "-D_GLIBCXX_USE_CXX11_ABI=1" if torch._C._GLIBCXX_USE_CXX11_ABI \
    else "-D_GLIBCXX_USE_CXX11_ABI=0"

extra_compile_args = ["-std=c++17", "-O2", "-fvisibility=hidden", abi_flag]

ext = CppExtension(
    name="state_spectre._state_spectre",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=[
        f"-Wl,-rpath,{os.path.join(CUDA_HOME, 'lib64')}",
        f"-Wl,-rpath,{TORCH_LIB}",
    ],
)

setup(
    name="state_spectre",
    version="0.1.0",
    description="Centralized GPU tensor-offload runtime (Python user API)",
    packages=["state_spectre"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
