#!/usr/bin/env python3
"""CPU-only QiboTN QFT benchmark with correctness and controlled variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import qibo
from qibo.models import QFT
from qibotn.eval_qu import dense_vector_tn_qu
from threadpoolctl import threadpool_info, threadpool_limits


VARIANTS = {
    "baseline_tn": None,
    "mps_exact": {"method": "svd", "cutoff": 0.0, "cutoff_mode": "abs"},
    "mps_cutoff_1e12": {
        "method": "svd",
        "cutoff": 1e-12,
        "cutoff_mode": "abs",
    },
    "mps_cutoff_1e8": {
        "method": "svd",
        "cutoff": 1e-8,
        "cutoff_mode": "abs",
    },
}


def normalized_state(nqubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**nqubits) + 1j * rng.normal(size=2**nqubits)
    return state / np.linalg.norm(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["QUIMB_NUM_PROCS"] = str(args.threads)
    qibo.set_backend(backend="qibojit", platform="numpy")
    rows: list[dict] = []

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "threads": args.threads,
        "threadpools": threadpool_info(),
        "timing_scope": "QASM parse + TN/MPS construction + simplify + dense output",
    }
    (args.output / "environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    with threadpool_limits(limits=args.threads):
        for nqubits in args.qubits:
            state = normalized_state(nqubits, seed=240810010427 + nqubits)
            circuit = QFT(nqubits, with_swaps=True)
            qasm = circuit.to_qasm()

            ref_start = time.perf_counter()
            reference = circuit(state).state(numpy=True)
            reference_s = time.perf_counter() - ref_start

            for variant, mps_opts in VARIANTS.items():
                for repeat in range(args.repeats):
                    start_rss = psutil.Process().memory_info().rss
                    t0 = time.perf_counter()
                    result = dense_vector_tn_qu(
                        qasm, state.copy(), mps_opts, backend="numpy"
                    ).reshape(-1)
                    elapsed = time.perf_counter() - t0
                    max_abs_error = float(np.max(np.abs(result - reference)))
                    fidelity = float(
                        np.abs(np.vdot(reference, result)) ** 2
                        / (np.vdot(reference, reference).real * np.vdot(result, result).real)
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "qubits": nqubits,
                            "repeat": repeat + 1,
                            "threads": args.threads,
                            "runtime_s": elapsed,
                            "reference_runtime_s": reference_s,
                            "max_abs_error": max_abs_error,
                            "fidelity": fidelity,
                            "norm": float(np.linalg.norm(result)),
                            "rss_delta_mib": (psutil.Process().memory_info().rss - start_rss)
                            / 2**20,
                            "status": "success" if max_abs_error <= 1e-6 else "quality_warning",
                        }
                    )

    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "mean_runtime_s": float(np.mean([r["runtime_s"] for r in selected])),
            "max_abs_error": max(r["max_abs_error"] for r in selected),
            "min_fidelity": min(r["fidelity"] for r in selected),
            "all_status": sorted({r["status"] for r in selected}),
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
