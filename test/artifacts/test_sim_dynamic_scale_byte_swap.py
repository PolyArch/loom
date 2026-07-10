#!/usr/bin/env python3
"""DFG integration test for byte_swap dynamic extents."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import artifact_test_common


GRAPH = "g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
SCALE_CASES = {"byte_swap_small": 8, "byte_swap_large": 32}
UINT32_MASK = 0xFFFFFFFF


def read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object in {path.name}: {data}")
    return data


def signed_i32(value: int) -> int:
    value &= UINT32_MASK
    return value - (UINT32_MASK + 1) if value > 0x7FFFFFFF else value


def byte_swap_inputs(count: int) -> list[int]:
    seeds = (
        0,
        UINT32_MASK,
        0x12345678,
        0x11223344,
        0xFF000000,
        0x000000FF,
        0xABCDEF01,
        0x01020304,
    )
    return [
        signed_i32(seeds[index] if index < len(seeds) else index * 0x01020304)
        for index in range(count)
    ]


def byte_swap_i32(value: int) -> int:
    value &= UINT32_MASK
    swapped = (
        ((value & 0x000000FF) << 24)
        | ((value & 0x0000FF00) << 8)
        | ((value & 0x00FF0000) >> 8)
        | ((value & 0xFF000000) >> 24)
    )
    return signed_i32(swapped)


def generate_byte_swap_dfg(repo: Path, out_dir: Path) -> Path:
    build_root = out_dir / "app-ir"
    env = os.environ.copy()
    env.update(
        {
            "LOOM_CC": str(artifact_test_common.find_tool(repo, "loom-cc")),
            "LOOM_CXX": str(artifact_test_common.find_tool(repo, "loom-c++")),
            "LOOM_RAISE": str(artifact_test_common.find_tool(repo, "loom-raise")),
            "LOOM_LOWER": str(artifact_test_common.find_tool(repo, "loom-lower")),
            "LOOM_RAISE_OPT": str(
                artifact_test_common.find_tool(repo, "loom-raise-opt")
            ),
        }
    )
    artifact_test_common.require_success(
        repo,
        [
            sys.executable,
            "test/app/ir_runner.py",
            "--stage",
            "dfg",
            "--case",
            "byte_swap",
            "--build-root",
            str(build_root),
        ],
        "byte_swap DFG generation",
        env=env,
    )
    dfg_mlir = build_root / "byte_swap" / "main_func.dfg.mlir"
    if not dfg_mlir.is_file():
        raise AssertionError(f"byte_swap DFG generation missed {dfg_mlir}")
    return dfg_mlir


def run_dfg_sim(
    repo: Path,
    dfg_mlir: Path,
    out_dir: Path,
    workload: str,
    inputs: list[int],
) -> Path:
    report = out_dir / f"{workload}-dfg-sim-report.json"
    args = [str(artifact_test_common.find_tool(repo, "loom-dfg-sim")), str(dfg_mlir)]
    for _ in inputs:
        args.extend(["--arg", "0=none"])
    args.extend(["--memref", f"1={','.join(map(str, inputs))}"])
    args.extend(["--memref", f"2={','.join('0' for _ in inputs)}"])
    for index in range(len(inputs)):
        args.extend(["--arg", f"3={index}"])
    args.extend(["--graph", GRAPH, "--workload", workload, "--output", str(report)])
    artifact_test_common.require_success(repo, args, f"{workload} DFG simulation")
    return report


def assert_destination_memory(
    report: dict[str, object],
    workload: str,
    inputs: list[int],
) -> None:
    final_memory = report.get("final_memory_state")
    if not isinstance(final_memory, dict):
        raise AssertionError(f"{workload} lacks final memory state: {report}")
    expected = [f"i32:{byte_swap_i32(value)}" for value in inputs]
    if final_memory.get("arg2") != expected:
        raise AssertionError(
            f"unexpected {workload} destination memory: {final_memory}"
        )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-byte-swap-scale-") as tmp:
        out_dir = Path(tmp)
        dfg_mlir = generate_byte_swap_dfg(repo, out_dir)
        for workload, count in SCALE_CASES.items():
            inputs = byte_swap_inputs(count)
            report_path = run_dfg_sim(repo, dfg_mlir, out_dir, workload, inputs)
            report = read_json_object(report_path)
            assert_destination_memory(report, workload, inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
