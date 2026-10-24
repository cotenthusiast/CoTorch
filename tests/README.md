# Tests

Three kinds of test live in separate places because they answer different questions.

| Directory | Question | May import | Expected results come from |
|---|---|---|---|
| `conformance/` | Does CoTorch do what its spec says? | `cotorch` public API, NumPy, Hypothesis | The spec in `docs/specs/`, hand-worked cases, NumPy, mathematical properties |
| `unit/` | Do internal pieces work? | `cotorch._*` | Whatever the internal contract is |
| `smoke/` | Does the package install and import? | `cotorch` | n/a |

Performance is not tested here; see `benchmarks/`. Native C++ tests live in `cpp/tests/`.

Rules for `conformance/`:

- Write a test only after the spec it checks exists in `docs/specs/`, and name that spec in the test.
- Never derive an expected value by running CoTorch, and never use one CoTorch operation to check another.
- Never import `cotorch._*`, so the tests survive refactors.

Run with `pytest` from the repository root.
