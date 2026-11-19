import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

import cotorch

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
args = parser.parse_args()

n, m, seed = args.iterations, args.matrix_size, args.seed
if n <= 0:
    raise ValueError("Invalid value for n")
if m <= 0:
    raise ValueError("Invalid value for m")

random.seed(seed)
m1 = [[random.random() for i in range(m)] for _ in range(m)]
m2 = [[random.random() for i in range(m)] for _ in range(m)]

npm1 = np.array(m1)
npm2 = np.array(m2)
npresult = npm1 @ npm2

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
base_dir = (Path(__file__).resolve().parent.parent / "benchmark_results").resolve()
results_dir = base_dir
if args.path is not None:
    if args.path.is_absolute():
        parser.error("--path must be relative to benchmark_results")
    results_dir = (base_dir / args.path).resolve()
    if not results_dir.is_relative_to(base_dir):
        parser.error("--path must stay inside benchmark_results")
results_dir.mkdir(parents=True, exist_ok=True)

filename = results_dir / f"run_{timestamp}.json"
results = []

for _ in range(n):
    result = cotorch.benchmark_matmul(m1, m2)
    if not np.allclose(np.array(result[0]), npresult, rtol=1e-5, atol=1e-6):
        raise ValueError("Matmul output is incorrect")
    results.append(result)

with open(filename, "a") as file:
    metadata = {
        "matrix_size": m,
        "iterations": n,
        "seed": seed,
        "timestamp": timestamp,
    }
    file.write(json.dumps({"metadata": metadata}) + "\n")
    for result in results:
        file.write(json.dumps(result[1]) + "\n")
