# Spec: <operation or component name>

Status: draft | agreed
Written before the implementation. Tests are derived from this document, not from code.

## Purpose
What this does and why it exists, in a few sentences. What it deliberately does not do.

## Supported inputs
Types, dtypes, shapes, memory layouts (contiguous, strided, views).

## Preconditions
What the caller must guarantee. Who checks it (the Tensor layer or the caller of the kernel).

## Expected outputs
Shape, dtype, layout, and whether the result shares memory with the inputs.
Define the result mathematically, with one or two small hand-worked examples.

## Invariants
Properties that must always hold (for example algebraic identities, inputs left unmodified).

## Error conditions
Each invalid input and the exact exception raised.

## Numerical accuracy requirements
Tolerances per dtype (relative and absolute) and how they scale with problem size.
Behaviour for NaN, infinity and overflow. Omit if not relevant.

## Edge cases
Empty and size-1 dimensions, non-contiguous views, extreme values, and so on.

## Conformance testing strategy
Which independent references are used (hand-worked cases, NumPy, properties via Hypothesis).
Which spec items each test covers. What is explicitly not tested.
