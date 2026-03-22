#!/usr/bin/env python3

import filecmp
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")

    _case_name = sys.argv[1]
    output_dir = Path(sys.argv[2])
    repo_root = Path.cwd()
    unit_dir = Path(__file__).resolve().parent
    loom = repo_root / "build/bin/loom"
    dfg = unit_dir / "dfg.mlir"
    fabric = unit_dir / "fabric.mlir"

    run_a = output_dir / "determinism-a"
    run_b = output_dir / "determinism-b"
    shutil.rmtree(run_a, ignore_errors=True)
    shutil.rmtree(run_b, ignore_errors=True)
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        str(loom),
        "--mapper-seed",
        "7",
        "--adg",
        str(fabric),
        "--dfg",
        str(dfg),
    ]

    subprocess.run(base_cmd + ["-o", str(run_a)], cwd=repo_root, check=True)
    subprocess.run(base_cmd + ["-o", str(run_b)], cwd=repo_root, check=True)

    base_name = "dfg.fabric"
    artifacts = [
        f"{base_name}.config.bin",
        f"{base_name}.config.json",
        f"{base_name}.map.json",
        f"{base_name}.map.txt",
    ]
    for name in artifacts:
        left = run_a / name
        right = run_b / name
        if not left.exists() or not right.exists():
            raise SystemExit(f"missing determinism artifact {name}")
        if not filecmp.cmp(left, right, shallow=False):
            raise SystemExit(f"determinism mismatch in {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
