#!/usr/bin/env python3
"""Run bounded CMSIS DFG-sim attempts for row-level status evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


@dataclass(frozen=True)
class Attempt:
    suite: str
    case: str
    stem: str
    graph: str
    dfg_dir_arg: str
    args: tuple[str, ...]
    memrefs: tuple[str, ...]
    hardware_mlir: str = ""
    hardware: str = ""


ATTEMPTS = (
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_abs_f32.c",
        stem="arm_abs_f32",
        graph="g_t_arm_abs_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=0", "2=4", "3=1"),
        memrefs=(
            "4=-1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_mult_f32.c",
        stem="arm_mult_f32",
        graph="g_t_arm_mult_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=0", "2=4", "3=1"),
        memrefs=(
            "4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
            "6=2.000000e+00,-1.000000e+00,3.000000e+00,5.000000e-01",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_add_q15.c",
        stem="arm_add_q15",
        graph="g_t_arm_add_q15_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=0", "2=4", "3=1"),
        memrefs=(
            "4=1000,20000,-30000,32760",
            "5=0,0,0,0",
            "6=2000,15000,-10000,1000",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_offset_f32.c",
        stem="arm_offset_f32",
        graph="g_t_arm_offset_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=0",
            "2=4",
            "3=1",
            "4=1.250000e+00",
        ),
        memrefs=(
            "5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "6=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="MatrixFunctions/arm_mat_add_f32.c",
        stem="arm_mat_add_f32",
        graph="g_t_arm_mat_add_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=0", "2=4", "3=1"),
        memrefs=(
            "4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=2.000000e+00,-1.000000e+00,3.000000e+00,5.000000e-01",
            "6=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="SupportFunctions/arm_copy_f32.c",
        stem="arm_copy_f32",
        graph="g_t_arm_copy_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=0", "2=4", "3=1"),
        memrefs=(
            "4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="SupportFunctions/arm_fill_f32.c",
        stem="arm_fill_f32",
        graph="g_t_arm_fill_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=0",
            "2=4",
            "3=1",
            "4=3.250000e+00",
        ),
        memrefs=("5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu_q15.c",
        stem="arm_relu_q15",
        graph="g_t_arm_relu_q15_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=2",
            "2=0",
            "3=-1",
            "4=15",
            "5=65537",
            "6=0",
            "7=-1",
        ),
        memrefs=("8=-2147516401,65538",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmsis-dsp-dfg-dir", required=True)
    parser.add_argument("--cmsis-nn-dfg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loom-dfg-sim")
    parser.add_argument("--loom-pnr-map")
    parser.add_argument("--loom-cgra-sim")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def resolve_tool(explicit: str | None, env_var: str, default: Path) -> Path:
    value = explicit or os.environ.get(env_var)
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return candidate
        resolved = shutil.which(value)
        if resolved:
            return Path(resolved)
    return default


def require_tool(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"[cmsis-dfg-sim] {label} not executable: {path}")


def run_command(
    command: list[str],
    timeout_seconds: int,
    label: str,
    env_overrides: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[cmsis-dfg-sim] {label} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def run_attempt(
    dfg_tool: Path,
    pnr_tool: Path,
    cgra_tool: Path,
    output_dir: Path,
    args: argparse.Namespace,
    attempt: Attempt,
) -> list[Path]:
    dfg_dir = Path(getattr(args, attempt.dfg_dir_arg))
    dfg_mlir = dfg_dir / f"{attempt.stem}.dfg.mlir"
    if not dfg_mlir.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] missing DFG MLIR for {attempt.case}: {dfg_mlir}")

    output = output_dir / f"{attempt.stem}.dfg.report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(dfg_tool),
        str(dfg_mlir),
        "--graph",
        attempt.graph,
        "--workload",
        attempt.case,
        "--output",
        str(output),
    ]
    for arg in attempt.args:
        command.extend(["--arg", arg])
    for memref in attempt.memrefs:
        command.extend(["--memref", memref])

    run_command(command, args.timeout_seconds, attempt.case)
    if not output.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} produced no report: {output}")
    data = json.loads(output.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} report is not a JSON object: {output}")
    data["input_artifact_fingerprints"] = intermediate_artifacts.input_artifact_fingerprints([dfg_mlir])
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    artifacts = [output]

    if not attempt.hardware_mlir:
        return artifacts
    if data.get("status") != "pass":
        return artifacts

    hardware_mlir = ROOT / attempt.hardware_mlir
    mapping_output = output_dir / f"{attempt.stem}.mapping.csv"
    mapping_artifact = output_dir / f"{attempt.stem}.mapping.json"
    run_command(
        [
            sys.executable,
            str(ROOT / "test" / "pnr" / "mapping_summary.py"),
            "--dfg-mlir",
            str(dfg_mlir),
            "--graph",
            attempt.graph,
            "--hardware-mlir",
            str(hardware_mlir),
            "--hardware",
            attempt.hardware,
            "--workload",
            attempt.case,
            "--output",
            str(mapping_output),
            "--artifact",
            str(mapping_artifact),
        ],
        args.timeout_seconds,
        f"{attempt.case} PnR",
        {"LOOM_PNR_MAP": str(pnr_tool)},
    )
    artifacts.extend([mapping_output, mapping_artifact])
    mapping_data = json.loads(mapping_artifact.read_text())
    if mapping_data.get("status") != "pass":
        return artifacts

    require_tool(cgra_tool, "loom-cgra-sim")
    cgra_report = output_dir / f"{attempt.stem}.cgra.report.json"
    run_command(
        [
            str(cgra_tool),
            "--dfg-report",
            str(output),
            "--mapping-artifact",
            str(mapping_artifact),
            "--hardware-mlir",
            str(hardware_mlir),
            "--output",
            str(cgra_report),
        ],
        args.timeout_seconds,
        f"{attempt.case} CGRA-sim",
    )
    artifacts.append(cgra_report)
    return artifacts


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dfg_tool = resolve_tool(
        args.loom_dfg_sim,
        "LOOM_DFG_SIM",
        ROOT / "build" / "tools" / "loom-dfg-sim" / "loom-dfg-sim",
    )
    pnr_tool = resolve_tool(
        args.loom_pnr_map,
        "LOOM_PNR_MAP",
        ROOT / "build" / "tools" / "loom-pnr-map" / "loom-pnr-map",
    )
    cgra_tool = resolve_tool(
        args.loom_cgra_sim,
        "LOOM_CGRA_SIM",
        ROOT / "build" / "tools" / "loom-cgra-sim" / "loom-cgra-sim",
    )
    require_tool(dfg_tool, "loom-dfg-sim")
    require_tool(pnr_tool, "loom-pnr-map")
    output_dir = Path(args.output_dir)
    artifacts: list[Path] = []
    for attempt in ATTEMPTS:
        artifacts.extend(run_attempt(dfg_tool, pnr_tool, cgra_tool, output_dir, args, attempt))
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
