import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import cotorch


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Benchmark Tool",
        description="Benchmark harness to measure matmul performance",
    )
    parser.add_argument("-n", "--iterations", type=int, required=True)
    parser.add_argument("-m", "--matrix_size", type=int, required=True)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument(
        "-f", "--path", type=Path, help="Subfolder inside benchmark_results"
    )
    args = parser.parse_args(argv)

    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.matrix_size <= 0:
        parser.error("matrix size must be positive")

    return args


def make_inputs(m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((m, m), dtype=np.float32), rng.random((m, m), dtype=np.float32)


def resolve_results_dir(subpath: Path | None) -> Path:
    base_dir = (Path(__file__).resolve().parent.parent / "benchmark_results").resolve()
    results_dir = base_dir
    if subpath is not None:
        if subpath.is_absolute():
            raise ValueError("--path must be relative to benchmark_results")
        results_dir = (base_dir / subpath).resolve()
        if not results_dir.is_relative_to(base_dir):
            raise ValueError("--path must stay inside benchmark_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_benchmark(n: int, m1: np.ndarray, m2: np.ndarray) -> list[dict[str, float]]:
    pm1 = m1.tolist()
    pm2 = m2.tolist()
    reference = m1 @ m2
    results = []
    for _ in range(n):
        result = cotorch.benchmark_matmul(pm1, pm2)
        if not np.allclose(np.array(result[0]), reference, rtol=1e-4, atol=1e-6):
            raise ValueError("Matmul output is incorrect")
        results.append(result[1])
    return results


def write_results(
    path: Path, metadata: dict[str, int | str], timings: list[dict[str, float]]
) -> None:
    file = path / f"run_{metadata['timestamp']}.json"
    with open(file, "w") as f:
        f.write(json.dumps({"metadata": metadata}) + "\n")
        for timing in timings:
            f.write(json.dumps(timing) + "\n")


def main():
    args = parse_args()
    metadata = {
        "matrix_size": args.matrix_size,
        "iterations": args.iterations,
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    }
    m1, m2 = make_inputs(metadata["matrix_size"], metadata["seed"])
    path = resolve_results_dir(args.path)
    timings = run_benchmark(metadata["iterations"], m1, m2)
    write_results(path, metadata, timings)


if __name__ == "__main__":
    main()
