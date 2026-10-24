# CoTorch architecture

CoTorch is an experimental, early-stage project. This document records how it is
meant to be organised and which decisions are made. Nothing described under
"Components" exists yet except the empty package and the native build skeleton.

## Components

Each layer may depend only on the layers below it.

```
nn / optim / classical ML      built on the public Tensor API
autograd                       records and replays operations
Tensor                         shape, strides, offset, views, validation
numerical kernels              raw buffers + integers, in Python and C++
layout utilities               pure shape/stride arithmetic
Storage                        owns typed numerical memory
```

- **Storage** owns a flat, typed block of memory (`float32` or `float64`). It knows nothing about shape.
- **Layout utilities** are pure functions on tuples of integers: contiguous strides, broadcast shapes, shape checks. They touch no memory, which makes them the easiest thing to specify and test.
- **Numerical kernels** do the arithmetic on raw buffers plus integers (sizes, strides, offsets). They never see `Tensor`, autograd or model classes, and they do not validate user input.
- **Tensor** combines Storage with shape, strides and offset (all in *elements*, not bytes). It validates inputs, allocates outputs, creates views, and calls kernels. Autograd metadata is added here later.
- **Autograd** (dynamic, reverse-mode) wraps Tensor operations. Kernels never change for it.
- **Higher-level ML** (neural networks, optimisers, classical algorithms, model architectures) uses only the public Tensor API.

Because kernels only see buffers, the same contract can be met by a Python
reference implementation, a C++ implementation, and one day a GPU one.

## Python and C++

- Python reference kernels are kept for correctness comparison and as a possible fallback.
- C++ kernels live in `cpp/` and build with CMake without Python, so they can be tested (`cpp/tests/`) and benchmarked before any integration.
- Python/C++ bindings are not designed yet. They will be once working native kernels exist.
- Tensors start dense, row-major and contiguous, but the representation (strides, offset) already allows views.

## Public and private API

- Public API is whatever `src/cotorch/__init__.py` exports in `__all__`. Today that is `__version__` only.
- Every other module is private (leading underscore, `cotorch._*`) and can be reorganised freely.
- To make something public, re-export it from `__init__.py`. Do not move it.
- A private module can later become a package without changing its import path.

## Testing

Specification first: write the spec (`docs/specs/_template.md`), derive conformance
tests from it, implement, check against independent references, then benchmark.
See `tests/README.md` and `benchmarks/README.md`.

## Settled decisions

- Python 3.12+, `src/` layout, Hatchling build backend, package name `cotorch`, version `0.0.1`.
- No runtime dependencies. NumPy, pytest, Hypothesis and Ruff are development tools only.
- dtypes: `float32` (eventual default) and `float64`.
- Strides and offsets are in elements.
- C++ from the start, built with CMake, C++20.
- Tensor operations and kernels are separate layers; there is no `_array` abstraction.

## Deferred on purpose

- In-place operations and mutation.
- Autograd design.
- Backend selection mechanism, and any dispatch system.
- Python/C++ bindings (tool and design).
- GPU support (nothing should prevent it).
- Random-number generator design.
- CI, GitHub Actions and release process.

## Growing the project

Neural networks, optimisers and classical ML (regression, k-means, PCA and so on)
are layers on top of Tensor and autograd. Transformers are `nn` modules built
from the same tensor operations. Model-serving concerns such as KV caches belong
in CoInfer, not here. Add a subpackage (for example `cotorch.nn`) when it has
code, and export it from `__init__.py`.
