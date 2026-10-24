# Benchmarks

Performance measurements live here, outside `tests/`, so a slow benchmark never slows or fails a test run.

Guidelines:

- Check the result for correctness before timing it. A fast wrong answer is not a result.
- Warm up, repeat, and report the median. Record the machine and compiler.
- Compare against a NumPy baseline.
- Keep raw result files out of Git unless you decide a result is worth keeping.

Native benchmarks that run before any Python integration belong with the C++ code under `cpp/`.
