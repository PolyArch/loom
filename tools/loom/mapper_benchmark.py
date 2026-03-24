#!/usr/bin/env python3

import argparse
import csv
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("build/tests/unit/unit_manifest.tsv")


@dataclass(frozen=True)
class UnitCase:
    name: str
    unit_dir: Path
    dfg_file: Path
    fabric_file: Path | None
    builder_exec: Path | None
    generated_fabric_file: Path | None
    loom_exec: Path


MODE_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no-local-repair": {"mapper": {"local_repair": {"enabled": False}}},
    "no-cpsat": {"mapper": {"enable_cpsat": False}},
    "route-checkpoint": {
        "mapper": {
            "refinement": {
                "route_aware_checkpoint_enabled": True,
                "route_aware_neighborhood_enabled": False,
            }
        }
    },
    "exact-neighborhood": {
        "mapper": {
            "refinement": {
                "route_aware_checkpoint_enabled": False,
                "route_aware_neighborhood_enabled": True,
            }
        }
    },
    "joint-pnr-buffer": {
        "mapper": {
            "bufferization": {
                "enabled": True,
                "outer_joint_iterations": 2,
            }
        }
    },
    "route-aware-sa": {
        "mapper": {
            "enable_route_aware_sa_main_loop": True,
        }
    },
    "no-route-aware-sa": {
        "mapper": {
            "enable_route_aware_sa_main_loop": False,
        }
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--loom", default="")
    parser.add_argument("--cases", required=True,
                        help="Comma-separated unit-case names from the unit manifest")
    parser.add_argument(
        "--modes",
        default="baseline,no-local-repair,no-cpsat,route-checkpoint,exact-neighborhood,joint-pnr-buffer",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def load_cases(manifest_path: Path) -> dict[str, UnitCase]:
    cases: dict[str, UnitCase] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            while len(row) < 7:
                row.append("")
            name, _output_dir, dfg_file, fabric_file, builder_exec, generated_fabric_file, loom_exec = row[:7]
            dfg_path = Path(dfg_file)
            cases[name] = UnitCase(
                name=name,
                unit_dir=dfg_path.parent,
                dfg_file=dfg_path,
                fabric_file=Path(fabric_file) if fabric_file else None,
                builder_exec=Path(builder_exec) if builder_exec else None,
                generated_fabric_file=Path(generated_fabric_file)
                if generated_fabric_file
                else None,
                loom_exec=Path(loom_exec),
            )
    return cases


def load_loom_args(unit_dir: Path) -> list[str]:
    args_file = unit_dir / "loom.args"
    if not args_file.exists():
        return []
    tokens: list[str] = []
    for line in args_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens.extend(shlex.split(stripped))
    return tokens


def strip_mapper_base_config_args(args: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--mapper-base-config":
            skip_next = True
            continue
        if token.startswith("--mapper-base-config="):
            continue
        filtered.append(token)
    return filtered


def normalize_adg_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".fabric"):
        return stem[:-7]
    return stem


def discover_map_json(run_dir: Path, dfg_file: Path, adg_path: Path) -> Path:
    dfg_stem = dfg_file.stem
    adg_stem = normalize_adg_stem(adg_path)
    preferred = run_dir / f"{dfg_stem}.{adg_stem}.map.json"
    legacy = run_dir / f"{dfg_stem}.map.json"
    if preferred.exists():
        return preferred
    return legacy


def merge_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_patch(merged[key], value)
        else:
            merged[key] = value
    return merged


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def dump_yaml(obj: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(obj, dict):
        lines: list[str] = []
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
        return "\n".join(lines)
    raise TypeError(f"unsupported YAML payload: {type(obj)!r}")


def build_override_config(mode: str) -> dict[str, Any]:
    if mode not in MODE_OVERRIDES:
        raise KeyError(f"unknown benchmark mode {mode!r}")
    return merge_patch({"version": 1}, MODE_OVERRIDES[mode])


def build_adg(case: UnitCase, run_dir: Path) -> Path:
    if case.builder_exec is not None:
        adg_path = run_dir / f"{case.unit_dir.name}.fabric.mlir"
        subprocess.run(
            [str(case.builder_exec), str(adg_path)],
            check=True,
            cwd=Path.cwd(),
        )
        return adg_path
    if case.fabric_file is None:
        raise RuntimeError(f"case {case.name} has no ADG source")
    return case.fabric_file


def collect_metrics(mapping: dict[str, Any]) -> dict[str, Any]:
    edge_routings = mapping.get("edge_routings", [])
    routed = [entry for entry in edge_routings if entry.get("kind") == "routed"]
    unrouted = [entry for entry in edge_routings if entry.get("kind") == "unrouted"]
    timing = mapping.get("timing", {})
    search = mapping.get("search", {})
    techmap = mapping.get("techmap", {})
    return {
        "routed_edge_count": len(routed),
        "unrouted_edge_count": len(unrouted),
        "total_routed_path_length": sum(max(0, len(entry.get("path", [])) - 1) for entry in routed),
        "estimated_clock_period": timing.get("estimated_clock_period"),
        "estimated_initiation_interval": timing.get("estimated_initiation_interval"),
        "estimated_throughput_cost": timing.get("estimated_throughput_cost"),
        "recurrence_pressure": timing.get("recurrence_pressure"),
        "mapper_selected_buffered_fifo_count": timing.get("mapper_selected_buffered_fifo_count"),
        "fifo_buffer_count": timing.get("fifo_buffer_count"),
        "critical_path_edge_count": len(timing.get("critical_path_edges", [])),
        "placement_seed_lane_count": search.get("placement_seed_lane_count"),
        "successful_placement_seed_count": search.get("successful_placement_seed_count"),
        "routed_lane_count": search.get("routed_lane_count"),
        "local_repair_attempts": search.get("local_repair_attempts"),
        "local_repair_successes": search.get("local_repair_successes"),
        "route_aware_checkpoint_rescore_passes": search.get("route_aware_checkpoint_rescore_passes"),
        "route_aware_neighborhood_attempts": search.get("route_aware_neighborhood_attempts"),
        "fifo_bufferization_accepted_toggles": search.get("fifo_bufferization_accepted_toggles"),
        "outer_joint_accepted_rounds": search.get("outer_joint_accepted_rounds"),
        "techmap_selected_candidate_count": techmap.get("selected_candidate_count"),
        "techmap_layer2_handoff_status": techmap.get("layer2_handoff_status"),
    }


def run_case_mode(case: UnitCase, mode: str, loom_path: Path,
                  output_root: Path, timeout_sec: int) -> dict[str, Any]:
    run_dir = output_root / mode / case.name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "run.stdout"
    stderr_path = run_dir / "run.stderr"

    adg_path = build_adg(case, run_dir)
    override_cfg = build_override_config(mode)
    override_path = run_dir / "mapper-benchmark-config.yaml"
    override_path.write_text(dump_yaml(override_cfg) + "\n", encoding="utf-8")

    extra_args = strip_mapper_base_config_args(load_loom_args(case.unit_dir))
    cmd = [
        str(loom_path),
        "--adg",
        str(adg_path),
        "--dfg",
        str(case.dfg_file),
        "--mapper-base-config",
        str(override_path),
        "-o",
        str(run_dir),
    ]
    cmd.extend(extra_args)

    start = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        stdout_text = completed.stdout
        stderr_text = completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        return_code = -1
    elapsed = time.perf_counter() - start
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")

    result: dict[str, Any] = {
        "case": case.name,
        "mode": mode,
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_sec": elapsed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "adg_path": str(adg_path),
        "dfg_path": str(case.dfg_file),
    }

    map_json_path = discover_map_json(run_dir, case.dfg_file, adg_path)
    result["map_json_path"] = str(map_json_path)
    if map_json_path.exists():
        mapping = json.loads(map_json_path.read_text(encoding="utf-8"))
        result["mapping_available"] = True
        result.update(collect_metrics(mapping))
    else:
        result["mapping_available"] = False
    return result


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    json_path = output_root / "summary.json"
    tsv_path = output_root / "summary.tsv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    all_cases = load_cases(manifest_path)
    requested_cases = [name.strip() for name in args.cases.split(",") if name.strip()]
    requested_modes = [name.strip() for name in args.modes.split(",") if name.strip()]

    missing_cases = [name for name in requested_cases if name not in all_cases]
    if missing_cases:
        raise SystemExit(f"unknown cases: {', '.join(missing_cases)}")
    unknown_modes = [name for name in requested_modes if name not in MODE_OVERRIDES]
    if unknown_modes:
        raise SystemExit(f"unknown modes: {', '.join(unknown_modes)}")

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_root = Path("temp") / f"mapper-benchmark-{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.loom:
        loom_path = Path(args.loom)
    else:
        loom_path = next(iter(all_cases.values())).loom_exec

    rows: list[dict[str, Any]] = []
    for mode in requested_modes:
        for case_name in requested_cases:
            row = run_case_mode(all_cases[case_name], mode, loom_path,
                                output_root, args.timeout)
            rows.append(row)
            print(
                f"[{mode}] {case_name}: rc={row['return_code']} "
                f"elapsed={row['elapsed_sec']:.3f}s "
                f"map={'yes' if row.get('mapping_available') else 'no'}"
            )

    write_summary(rows, output_root)
    print(f"wrote benchmark summary to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
