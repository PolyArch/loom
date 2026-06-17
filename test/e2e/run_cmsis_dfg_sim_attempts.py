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
from urllib.parse import quote


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
    artifact_stem: str = ""
    aggregate_stem: str = ""


@dataclass(frozen=True)
class AttemptResult:
    attempt: Attempt
    dfg_report: Path
    mapping_summary: Path | None = None
    mapping_artifact: Path | None = None
    cgra_report: Path | None = None


ATTEMPTS = (
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_abs_f32.c",
        stem="arm_abs_f32",
        graph="g_t_arm_abs_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
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
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
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
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
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
            "1=4",
            "2=0",
            "3=-1",
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
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
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
        case="StatisticsFunctions/arm_mean_f32.c",
        stem="arm_mean_f32",
        graph="g_t_arm_mean_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=0.000000e+00",
        ),
        memrefs=("5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_var_f32.c",
        stem="arm_var_f32",
        graph="g_t_arm_var_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "5=0.000000e+00",
        ),
        memrefs=("4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_var_f32.red0",
        aggregate_stem="arm_var_f32",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_var_f32.c",
        stem="arm_var_f32",
        graph="g_t_arm_var_f32_red_1_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=9.375000e-01",
            "6=0.000000e+00",
        ),
        memrefs=("5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_var_f32.red1",
        aggregate_stem="arm_var_f32",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="SupportFunctions/arm_copy_f32.c",
        stem="arm_copy_f32",
        graph="g_t_arm_copy_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
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
            "1=4",
            "2=0",
            "3=-1",
            "4=3.250000e+00",
        ),
        memrefs=("5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="FilteringFunctions/arm_biquad_cascade_df1_f32.c",
        stem="arm_biquad_cascade_df1_f32",
        graph="g_t_arm_biquad_cascade_df1_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=5.000000e-01",
            "5=2.500000e-01",
            "6=1.250000e-01",
            "7=6.250000e-02",
            "8=3.125000e-02",
            "10=0.000000e+00",
            "11=0.000000e+00",
            "12=0.000000e+00",
            "13=0.000000e+00",
        ),
        memrefs=(
            "9=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "14=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_max_f32.c",
        stem="arm_max_f32",
        graph="g_t_arm_max_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "1=3",
            "2=0",
            "3=-1",
            "4=4",
            "5=0",
            "6=1.000000e+00",
        ),
        memrefs=("7=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
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
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu_q7.c",
        stem="arm_relu_q7",
        graph="g_t_arm_relu_q7_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=2",
            "2=0",
            "3=-1",
            "4=7",
            "5=16843009",
            "6=0",
            "7=-1",
        ),
        memrefs=("8=-2139062144,2130706433",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_relu_q7.red0",
        aggregate_stem="arm_relu_q7",
    ),
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu_q7.c",
        stem="arm_relu_q7",
        graph="g_t_arm_relu_q7_red_1_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=("0=none", "1=3", "2=0", "3=-1", "4=0"),
        memrefs=("5=-1,2,-3",),
        artifact_stem="arm_relu_q7.red1",
        aggregate_stem="arm_relu_q7",
    ),
    Attempt(
        suite="cmsis-nn",
        case="FullyConnectedFunctions/arm_vector_sum_s8.c",
        stem="arm_vector_sum_s8",
        graph="g_t_arm_vector_sum_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=2",
            "3=1",
            "4=2",
            "5=1",
            "6=0",
            "7=1",
        ),
        memrefs=(
            "8=0,0",
            "9=1,2,3,4",
        ),
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
) -> AttemptResult:
    dfg_dir = Path(getattr(args, attempt.dfg_dir_arg))
    dfg_mlir = dfg_dir / f"{attempt.stem}.dfg.mlir"
    if not dfg_mlir.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] missing DFG MLIR for {attempt.case}: {dfg_mlir}")

    output_stem = attempt.artifact_stem or attempt.stem
    output = output_dir / f"{output_stem}.dfg.report.json"
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

    if not attempt.hardware_mlir:
        return AttemptResult(attempt=attempt, dfg_report=output)
    if data.get("status") != "pass":
        return AttemptResult(attempt=attempt, dfg_report=output)

    hardware_mlir = ROOT / attempt.hardware_mlir
    mapping_output = output_dir / f"{output_stem}.mapping.csv"
    mapping_artifact = output_dir / f"{output_stem}.mapping.json"
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
    mapping_data = json.loads(mapping_artifact.read_text())
    if mapping_data.get("status") != "pass":
        return AttemptResult(
            attempt=attempt,
            dfg_report=output,
            mapping_summary=mapping_output,
            mapping_artifact=mapping_artifact,
        )

    require_tool(cgra_tool, "loom-cgra-sim")
    cgra_report = output_dir / f"{output_stem}.cgra.report.json"
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
    return AttemptResult(
        attempt=attempt,
        dfg_report=output,
        mapping_summary=mapping_output,
        mapping_artifact=mapping_artifact,
        cgra_report=cgra_report,
    )


def aggregate_mapping_id(case: str, hardware: str) -> str:
    return f"{quote(case, safe='')}__workload_graph_set__{hardware}"


def run_aggregates(
    output_dir: Path,
    results: list[AttemptResult],
    timeout_seconds: int,
) -> list[Path]:
    grouped: dict[tuple[str, str, str], list[AttemptResult]] = {}
    for result in results:
        stem = result.attempt.aggregate_stem
        if not stem:
            continue
        key = (result.attempt.case, result.attempt.hardware, stem)
        grouped.setdefault(key, []).append(result)

    artifacts: list[Path] = []
    for (case, hardware, stem), group in grouped.items():
        if len(group) < 2:
            continue
        if any(
            item.mapping_artifact is None or item.mapping_summary is None or item.cgra_report is None
            for item in group
        ):
            continue
        dfg_output = output_dir / f"{stem}.dfg.report.json"
        mapping_output = output_dir / f"{stem}.mapping.json"
        cgra_output = output_dir / f"{stem}.cgra.report.json"
        mapping_summary = output_dir / f"{stem}.mapping.csv"
        command = [
            sys.executable,
            str(ROOT / "test" / "e2e" / "aggregate_workload_graph_artifacts.py"),
            "--workload",
            case,
            "--hardware",
            hardware,
            "--mapping-id",
            aggregate_mapping_id(case, hardware),
            "--dfg-output",
            str(dfg_output),
            "--mapping-output",
            str(mapping_output),
            "--cgra-output",
            str(cgra_output),
            "--mapping-summary-output",
            str(mapping_summary),
        ]
        for result in group:
            command.extend(["--dfg-report", str(result.dfg_report)])
        for result in group:
            if result.mapping_artifact is not None:
                command.extend(["--mapping-artifact", str(result.mapping_artifact)])
        for result in group:
            if result.cgra_report is not None:
                command.extend(["--cgra-report", str(result.cgra_report)])
        run_command(command, timeout_seconds, f"{case} aggregate")
        artifacts.extend([dfg_output, mapping_output, cgra_output, mapping_summary])
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
    results: list[AttemptResult] = []
    for attempt in ATTEMPTS:
        results.append(run_attempt(dfg_tool, pnr_tool, cgra_tool, output_dir, args, attempt))
    artifacts: list[Path] = []
    for result in results:
        artifacts.append(result.dfg_report)
        for artifact in (result.mapping_summary, result.mapping_artifact, result.cgra_report):
            if artifact is not None:
                artifacts.append(artifact)
    artifacts.extend(run_aggregates(output_dir, results, args.timeout_seconds))
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
