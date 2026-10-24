# Native code

C++ kernels live here and build independently of Python.

- `CMakeLists.txt`: the native build; sets the C++ standard, warnings and a Release default.
- `tests/`: plain executables run through CTest. `toolchain_smoke` only checks the toolchain.

Build and test:

    cmake -S cpp -B cpp/build
    cmake --build cpp/build
    ctest --test-dir cpp/build --output-on-failure

Kernels take raw pointers and integers, never Python objects. Python bindings come later.
Add headers, a kernel library target and `benchmarks/` here when the first kernel exists.
