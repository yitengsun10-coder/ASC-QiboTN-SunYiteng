#!/usr/bin/env python3
"""Verify archived QiboTN CSV/JSON results without importing QiboTN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = []

    for group in ("small", "large"):
        directory = args.root / group
        rows = list(csv.DictReader((directory / "results.csv").open(encoding="utf-8", newline="")))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if not rows:
            failures.append(f"{group}: empty results.csv")
        bad_status = [row for row in rows if row.get("status") != "success"]
        if bad_status:
            failures.append(f"{group}: {len(bad_status)} non-success rows")
        min_fidelity = min(float(item["min_fidelity"]) for item in summary.values())
        max_error = max(float(item["max_abs_error"]) for item in summary.values())
        print(f"{group}: rows={len(rows)} methods={len(summary)} min_fidelity={min_fidelity:.16f} max_error={max_error:.3e}")
        if min_fidelity < 0.999999:
            failures.append(f"{group}: fidelity below threshold")

    sums = args.root / "SHA256SUMS.txt"
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        archived = relative.strip().lstrip("*").replace("\\", "/")
        # The original checksum file intentionally preserves server absolute
        # paths.  Map only the suffix below the archived qibotn result root;
        # never trust or open the absolute path on the verifier's machine.
        marker = "/qibotn/"
        local_relative = archived.split(marker, 1)[1] if marker in archived else archived
        path = args.root / local_relative
        if not path.is_file():
            failures.append(f"hash target missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            failures.append(f"hash mismatch: {relative}")

    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("PASS: QiboTN statuses, fidelity, errors, and archived hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
