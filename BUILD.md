# Build & Run

This file has moved into the consolidated documentation:

- **Build, run, configure, test:** [docs/USAGE.md](docs/USAGE.md)
- **Architecture & implementation:** [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
- **Benchmarks:** [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- **Index:** [docs/README.md](docs/README.md)

Quick start:

```bash
mkdir -p build && cd build && cmake .. && make -j"$(nproc)" && ctest
cd .. && ./run_tests.sh        # full GPU + Python suite
```
