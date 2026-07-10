cmake_minimum_required(VERSION 3.18)
project(fastoffload LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Centralized GPU tensor-offload runtime.
#   offloadd            : the CUDA-free daemon
#   libofld_agent       : rank-side agent (links CUDA runtime)
#   libofld_common      : shared transport / wire / metrics / numa
#   C++ tests           : unit + functional (functional need a GPU + memfd)
#
# The Python extension is built separately via python_api/setup.py (it must be
# compiled by torch's cpp_extension to match the libtorch ABI). See README.
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

find_package(Threads REQUIRED)

# CUDA runtime (headers + libcudart). We do not compile .cu kernels; the agent
# only calls the CUDA *runtime API*, so plain C++ + libcudart is sufficient.
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
          PATHS /usr/local/cuda/include /usr/local/cuda-12.9/include)
find_library(CUDART_LIB cudart
             PATHS /usr/local/cuda/lib64 /usr/local/cuda-12.9/lib64)
if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIB)
  message(FATAL_ERROR "CUDA runtime not found (cuda_runtime.h / libcudart)")
endif()

find_library(NUMA_LIB numa REQUIRED)
find_library(URING_LIB uring REQUIRED)

set(ABI_DIR      ${CMAKE_SOURCE_DIR}/abi)
set(COMMON_DIR   ${CMAKE_SOURCE_DIR}/src/common)
set(DAEMON_DIR   ${CMAKE_SOURCE_DIR}/src/daemon)
set(AGENT_DIR    ${CMAKE_SOURCE_DIR}/src/agent)

# ------------------------- common -------------------------
add_library(ofld_common STATIC
  ${COMMON_DIR}/wire.cpp
  ${COMMON_DIR}/wire_v2.cpp
  ${COMMON_DIR}/uds.cpp
  ${COMMON_DIR}/metrics.cpp
  ${COMMON_DIR}/numa_util.cpp)
target_include_directories(ofld_common PUBLIC ${ABI_DIR} ${COMMON_DIR})
target_link_libraries(ofld_common PUBLIC Threads::Threads ${NUMA_LIB})

# ------------------------- daemon lib -------------------------
add_library(ofld_daemon STATIC
  ${DAEMON_DIR}/daemon.cpp
  ${DAEMON_DIR}/config.cpp)
target_include_directories(ofld_daemon PUBLIC ${ABI_DIR} ${COMMON_DIR} ${DAEMON_DIR})
target_link_libraries(ofld_daemon PUBLIC ofld_common ${URING_LIB})

# ------------------------- daemon binary -------------------------
add_executable(offloadd ${DAEMON_DIR}/main.cpp)
target_link_libraries(offloadd PRIVATE ofld_daemon)

# ------------------------- tools -------------------------
add_executable(offload_stat ${CMAKE_SOURCE_DIR}/src/tools/offload_stat.cpp)
target_link_libraries(offload_stat PRIVATE ofld_common)
set_target_properties(offload_stat PROPERTIES OUTPUT_NAME offload-stat)

# ------------------------- agent lib -------------------------
add_library(ofld_agent STATIC
  ${AGENT_DIR}/agent.cpp
  ${AGENT_DIR}/rpc_client.cpp)
target_include_directories(ofld_agent PUBLIC
  ${ABI_DIR} ${COMMON_DIR} ${AGENT_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(ofld_agent PUBLIC ofld_common ${CUDART_LIB})

# ------------------------- tests -------------------------
enable_testing()

# Unit tests: no GPU, no memfd required (pure logic: wire codec, config, layout).
add_executable(test_wire      tests/cpp/test_wire.cpp)
target_link_libraries(test_wire PRIVATE ofld_common)

add_executable(test_wire_v2   tests/cpp/test_wire_v2.cpp)
target_link_libraries(test_wire_v2 PRIVATE ofld_common)
target_include_directories(test_wire_v2 PRIVATE ${ABI_DIR} ${COMMON_DIR})

add_executable(test_config    tests/cpp/test_config.cpp)
target_link_libraries(test_config PRIVATE ofld_daemon)

add_executable(test_layout    tests/cpp/test_layout.cpp)
target_link_libraries(test_layout PRIVATE ofld_common)
target_include_directories(test_layout PRIVATE ${ABI_DIR} ${COMMON_DIR})

# Integration tests: exercise a live daemon over a real socket + memfd control
# region. No GPU needed (they drive RPC + slot state, not CUDA copies).
add_executable(test_daemon_rpc tests/cpp/test_daemon_rpc.cpp)
target_link_libraries(test_daemon_rpc PRIVATE ofld_daemon)

add_executable(test_fault_injection tests/cpp/test_fault_injection.cpp)
target_link_libraries(test_fault_injection PRIVATE ofld_daemon)

# Functional GPU test: full evict/restore byte-correctness via the agent + a
# live daemon. Requires a GPU and memfd_create (run outside restrictive sandbox).
add_executable(test_evict_restore tests/cpp/test_evict_restore.cpp)
target_link_libraries(test_evict_restore PRIVATE ofld_agent ofld_daemon)

# Benchmark harness: in-process daemon + real agent, CUDA-timed. Not a ctest.
add_executable(offload_bench tests/cpp/bench.cpp)
target_link_libraries(offload_bench PRIVATE ofld_agent ofld_daemon)

add_test(NAME wire        COMMAND test_wire)
add_test(NAME wire_v2     COMMAND test_wire_v2)
add_test(NAME config      COMMAND test_config)
add_test(NAME layout      COMMAND test_layout)
add_test(NAME daemon_rpc  COMMAND test_daemon_rpc)
add_test(NAME fault_injection COMMAND test_fault_injection)
add_test(NAME evict_restore COMMAND test_evict_restore)
# Mark the GPU/memfd test so it can be excluded where unavailable:
set_tests_properties(evict_restore PROPERTIES LABELS "gpu")
set_tests_properties(daemon_rpc    PROPERTIES LABELS "memfd")
