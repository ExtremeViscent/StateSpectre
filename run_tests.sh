#!/usr/bin/env bash
# Build + run the full test suite for the centralized GPU offload runtime.
#
#   ./run_tests.sh            build everything + run all C++ tests + python e2e
#   ./run_tests.sh --cpp      C++ tests only
#   ./run_tests.sh --python   python e2e only (assumes daemon binary is built)
#
# Some tests require memfd_create and a GPU. In restrictive sandboxes memfd may
# be unavailable; those tests self-skip cleanly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

run_cpp() {
    echo "==> Configuring + building (CMake)"
    mkdir -p "$BUILD"
    (cd "$BUILD" && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j"$(nproc)")
    echo "==> Running C++ tests (ctest)"
    (cd "$BUILD" && CUDA_VISIBLE_DEVICES="$GPU" OFLD_LOG_LEVEL=0 ctest --output-on-failure)
}

run_python() {
    echo "==> Building python extension"
    (cd "$ROOT/python_api" && python setup.py build_ext --inplace >/dev/null 2>&1)
    echo "==> Running python end-to-end test against a live daemon"
    local sock="/tmp/fastoffload_runtests_$$.sock"
    rm -f "$sock"
    OFLD_LOG_LEVEL=1 "$BUILD/offloadd" --smoke-arena-mb 8192 --numa 0 --gpu "$GPU" \
        --socket "$sock" &
    local dpid=$!
    sleep 1
    local rc=0
    ( cd "$ROOT/python_api" && \
      PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES="$GPU" \
      python "$ROOT/tests/python/test_python_e2e.py" "$sock" ) || rc=$?
    if [ $rc -eq 0 ]; then
        echo "==> Running multi-process safety test (4 procs)"
        ( cd "$ROOT/python_api" && \
          PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES="$GPU" \
          python "$ROOT/tests/python/test_multiproc.py" "$sock" 4 20 ) || rc=$?
    fi
    if [ $rc -eq 0 ]; then
        echo "==> Running LLM-style workload test"
        ( cd "$ROOT/python_api" && \
          PYTHONPATH="$(pwd)" CUDA_VISIBLE_DEVICES="$GPU" \
          python "$ROOT/tests/python/test_llm_workload.py" "$sock" ) || rc=$?
    fi
    kill "$dpid" 2>/dev/null || true
    wait "$dpid" 2>/dev/null || true
    rm -f "$sock"
    return $rc
}

case "${1:-all}" in
    --cpp)    run_cpp ;;
    --python) run_python ;;
    all|"")   run_cpp; run_python ;;
    *) echo "usage: $0 [--cpp|--python]"; exit 2 ;;
esac
echo "==> ALL TESTS PASSED"
