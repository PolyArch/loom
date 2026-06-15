#!/usr/bin/env python3
"""Emit row-complete CGRA status baseline evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))
sys.path.insert(0, str(ROOT / "test" / "loombench"))
sys.path.insert(0, str(ROOT / "test" / "simulator"))

import intermediate_artifacts  # noqa: E402
import loombench_manifest  # noqa: E402
import sim_comparison_report  # noqa: E402


STATUS_KEYS = ("pass", "fail", "blocked", "unsupported")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    parser.add_argument(
        "--legacy-loombench-root",
        default=str(ROOT / "temp" / "old_implementation_loom" / "loom" / "tests" / "app"),
    )
    parser.add_argument(
        "--loombench-manifest",
        help="Dedicated LoomBench manifest JSON. Without it, legacy rows remain blocked on manifest reconciliation.",
    )
    parser.add_argument(
        "--no-legacy-loombench",
        action="store_true",
        help="Do not include legacy LoomBench rows in this status rollup.",
    )
    parser.add_argument(
        "--sim-evidence-dir",
        help="Directory containing per-workload DFG, mapping, CGRA, and comparison evidence.",
    )
    parser.add_argument(
        "--comparison-output-dir",
        help="Directory for generated simulation comparison reports. Defaults next to the CSV output.",
    )
    parser.add_argument(
        "--cmsis-dsp-dfg-dir",
        help="Directory containing CMSIS-DSP lowered .dfg.mlir evidence.",
    )
    parser.add_argument(
        "--cmsis-nn-dfg-dir",
        help="Directory containing CMSIS-NN lowered .dfg.mlir evidence.",
    )
    return parser.parse_args(argv)


def empty_stage_fields() -> dict[str, str]:
    return {
        "dfg_mlir": "",
        "dfg_mlir_fingerprint": "",
        "hardware_system": "",
        "spatialcore_template": "",
        "mapping_id": "",
        "dfg_report": "",
        "dfg_report_fingerprint": "",
        "dfg_status": "not_run",
        "mapping_artifact": "",
        "mapping_artifact_fingerprint": "",
        "mapping_status": "not_run",
        "cgra_report": "",
        "cgra_report_fingerprint": "",
        "cgra_status": "not_run",
        "comparison_report": "",
        "comparison_report_fingerprint": "",
        "comparison_status": "not_run",
        "final_outputs_present": "false",
        "final_memory_state_present": "false",
        "status": "not_run",
        "diagnostic_class": "missing_status",
        "owner": "implementation",
    }


def row(
    *,
    suite: str,
    case: str,
    source_row: str,
    software_root: str,
    graph_ids: str = "",
    required_slice_count: str = "0",
    blocking_prerequisite: str,
    diagnostic: str,
) -> dict[str, str]:
    data = {
        "suite": suite,
        "case": case,
        "source_row": source_row,
        "software_root": software_root,
        "graph_ids": graph_ids,
        "required_slice_count": required_slice_count,
        "blocking_prerequisite": blocking_prerequisite,
        "diagnostic": diagnostic,
    }
    data.update(empty_stage_fields())
    return data


def load_app_manifest() -> dict[str, object]:
    path = ROOT / "test" / "app" / "manifest.json"
    return json.loads(path.read_text())


def app_rows() -> list[dict[str, str]]:
    manifest = load_app_manifest()
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("test/app/manifest.json cases must be a list")
    rows: list[dict[str, str]] = []
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case = str(entry.get("case", ""))
        if not case:
            continue
        tiers = entry.get("tiers", [])
        has_dfg = isinstance(tiers, list) and "dfg" in tiers
        prerequisite = "mapping_artifact" if has_dfg else "dataflow"
        diagnostic = (
            "CGRA status missing after app dataflow tier; mapping artifact and CGRA-sim report are absent"
            if has_dfg
            else "CGRA status missing because app row has no dataflow tier yet"
        )
        row_data = row(
            suite="app",
            case=case,
            source_row=case,
            software_root=f"test/app/{case}",
            required_slice_count="1" if has_dfg else "0",
            blocking_prerequisite=prerequisite,
            diagnostic=diagnostic,
        )
        if not has_dfg:
            row_data["status"] = "blocked"
            row_data["diagnostic_class"] = "app_dataflow_tier_missing"
            row_data["owner"] = "compiler_pipeline"
            row_data["diagnostic"] = (
                f"app manifest has no dfg tier for {case}; dataflow lowering, DFG-sim, "
                "mapping, and CGRA-sim evidence are absent"
            )
        rows.append(row_data)
    return rows


def iter_target_rows(path: Path) -> Iterable[list[str]]:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield line.split("|")


def positive_shape(columns: list[str]) -> bool:
    if len(columns) < 17:
        return False
    # expect_thread through expect_demux are columns 6..15.
    for cell in columns[6:16]:
        text = cell.strip()
        if text.startswith(">="):
            text = text[2:]
        try:
            if int(text) > 0:
                return True
        except ValueError:
            continue
    return False


def expected_symbols(columns: list[str]) -> list[str]:
    if len(columns) > 4:
        symbols: list[str] = []
        for symbol in columns[4].split(","):
            stripped = symbol.strip()
            if stripped:
                symbols.append(stripped)
        if symbols:
            return symbols
    return [Path(columns[0]).stem] if columns else []


def source_stem(columns: list[str]) -> str:
    return Path(columns[0]).stem if columns else ""


def ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def dfg_graph_ids_from_text(text: str) -> list[str]:
    definitions = re.findall(r"\bdataflow\.graph\.func\s+(?:private\s+)?@([A-Za-z_.$][\w.$-]*)", text)
    launches = re.findall(r"\bdataflow\.graph\.launch\s+@([A-Za-z_.$][\w.$-]*)", text)
    return ordered_unique([*definitions, *launches])


def mlir_mentions_symbol(text: str, symbol: str) -> bool:
    return re.search(rf"@{re.escape(symbol)}\b", text) is not None


def apply_cmsis_dfg_mlir_evidence(
    row_data: dict[str, str],
    columns: list[str],
    dfg_dir: Path | None,
) -> None:
    if dfg_dir is None:
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "cmsis_dfg_mlir_missing"
        row_data["owner"] = "compiler_pipeline"
        row_data["blocking_prerequisite"] = "dfg_mlir"
        row_data["diagnostic"] = (
            "CMSIS DFG MLIR evidence is absent; DFG-sim, mapping, and CGRA-sim "
            "reports cannot be produced for this row"
        )
        return
    stem = source_stem(columns)
    if not stem:
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "cmsis_dfg_mlir_missing"
        row_data["owner"] = "compiler_pipeline"
        row_data["blocking_prerequisite"] = "dfg_mlir"
        row_data["diagnostic"] = (
            "CMSIS target row lacks a source basename for DFG MLIR evidence lookup"
        )
        return
    dfg_mlir = dfg_dir / f"{stem}.dfg.mlir"
    if not dfg_mlir.is_file():
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "cmsis_dfg_mlir_missing"
        row_data["owner"] = "compiler_pipeline"
        row_data["blocking_prerequisite"] = "dfg_mlir"
        row_data["diagnostic"] = (
            f"CMSIS DFG MLIR evidence is absent at {relative_path_text(dfg_mlir)}; "
            "DFG-sim, mapping, and CGRA-sim reports cannot be produced for this row"
        )
        return
    text = dfg_mlir.read_text(errors="replace")
    graph_ids = dfg_graph_ids_from_text(text)
    evidence_path = relative_path_text(dfg_mlir)
    row_data["dfg_mlir"] = evidence_path
    row_data["dfg_mlir_fingerprint"] = artifact_fingerprint(dfg_mlir)
    row_data["owner"] = "compiler_pipeline"
    symbols = expected_symbols(columns)
    if symbols and not any(mlir_mentions_symbol(text, symbol) for symbol in symbols):
        row_data["graph_ids"] = ",".join(graph_ids)
        row_data["required_slice_count"] = str(len(graph_ids))
        row_data["status"] = "fail"
        row_data["diagnostic_class"] = "cmsis_dfg_mlir_identity_mismatch"
        row_data["blocking_prerequisite"] = "dataflow_graph_identity"
        row_data["diagnostic"] = (
            f"CMSIS DFG MLIR evidence exists at {evidence_path} but does not contain "
            f"any expected symbol: {','.join(symbols)}"
        )
        return
    if graph_ids:
        row_data["graph_ids"] = ",".join(graph_ids)
        row_data["required_slice_count"] = str(len(graph_ids))
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "cmsis_dfg_mlir_ready_for_dfg_sim"
        row_data["blocking_prerequisite"] = "dfg_sim_report"
        row_data["diagnostic"] = (
            f"CMSIS DFG MLIR evidence exists at {evidence_path} with "
            f"{len(graph_ids)} dataflow graph(s); DFG-sim, mapping, and CGRA-sim reports are absent"
        )
        return
    row_data["graph_ids"] = ""
    row_data["required_slice_count"] = "0"
    row_data["status"] = "unsupported"
    row_data["diagnostic_class"] = "cmsis_no_dataflow_graph"
    row_data["blocking_prerequisite"] = "dataflow_graph"
    row_data["diagnostic"] = (
        f"CMSIS DFG MLIR evidence exists at {evidence_path} but contains no "
        "dataflow.graph.func or dataflow.graph.launch operation"
    )


def cmsis_rows(
    suite: str,
    directory: str,
    targets_name: str,
    software_root: str,
    dfg_dir: Path | None = None,
) -> list[dict[str, str]]:
    targets = ROOT / "test" / directory / targets_name
    rows: list[dict[str, str]] = []
    for columns in iter_target_rows(targets):
        source = columns[0]
        has_shape = positive_shape(columns)
        prerequisite = "mapping_artifact" if has_shape else "dataflow_graph"
        diagnostic = (
            "CGRA status missing after CMSIS dataflow-shape row; mapping artifact and CGRA-sim report are absent"
            if has_shape
            else "CGRA status missing because CMSIS row emits no dataflow graph/thread shape"
        )
        row_data = row(
            suite=suite,
            case=source,
            source_row=source,
            software_root=software_root,
            required_slice_count="1" if has_shape else "0",
            blocking_prerequisite=prerequisite,
            diagnostic=diagnostic,
        )
        apply_cmsis_dfg_mlir_evidence(row_data, columns, dfg_dir)
        rows.append(row_data)
    return rows


def load_loombench_manifest(path: Path) -> list[dict[str, object]]:
    data = read_json(path)
    if data.get("schema_version") != 1 or data.get("kind") != "loombench_manifest":
        raise SystemExit(f"invalid LoomBench manifest: {path}")
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise SystemExit(f"LoomBench manifest cases must be a list: {path}")
    typed_cases: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, case_data in enumerate(cases):
        if not isinstance(case_data, dict):
            raise SystemExit(f"LoomBench manifest case {index} is not an object: {path}")
        case = case_data.get("case")
        if not isinstance(case, str) or not case:
            raise SystemExit(f"LoomBench manifest case {index} lacks case identity: {path}")
        if case in seen:
            raise SystemExit(f"LoomBench manifest has duplicate case: {case}")
        seen.add(case)
        typed_cases.append(case_data)
    return typed_cases


def loombench_row_from_manifest_case(
    manifest_case: dict[str, object],
    app_row_by_case: dict[str, dict[str, str]],
) -> dict[str, str]:
    case = str(manifest_case["case"])
    source_row = str(manifest_case.get("source_row", case))
    import_state = str(manifest_case.get("import_state", ""))
    app_case = str(manifest_case.get("manifest_case", ""))
    reason = str(manifest_case.get("reason", ""))
    row_data = row(
        suite="loombench",
        case=case,
        source_row=source_row,
        software_root=str(manifest_case.get("software_root", "")),
        blocking_prerequisite="loombench_manifest",
        diagnostic=reason or "LoomBench manifest row has no CGRA status evidence yet",
    )
    row_data["owner"] = "loombench_manifest"

    if import_state == "excluded":
        row_data["status"] = "unsupported"
        row_data["diagnostic_class"] = "loombench_import_excluded"
        row_data["blocking_prerequisite"] = "legacy_source"
        row_data["diagnostic"] = reason or "legacy LoomBench source row is excluded from migration"
        return row_data

    if import_state == "deferred":
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = f"loombench_import_{import_state}"
        row_data["blocking_prerequisite"] = "app_import"
        row_data["diagnostic"] = reason or f"legacy LoomBench source row import is {import_state}"
        return row_data

    if import_state not in loombench_manifest.VALID_IMPORT_STATES:
        row_data["status"] = "fail"
        row_data["diagnostic_class"] = "loombench_manifest_invalid_import_state"
        row_data["blocking_prerequisite"] = "loombench_manifest"
        row_data["diagnostic"] = f"invalid LoomBench import_state {import_state!r}"
        return row_data

    if not app_case:
        row_data["status"] = "fail"
        row_data["diagnostic_class"] = "loombench_manifest_missing_app_case"
        row_data["blocking_prerequisite"] = "app_import"
        row_data["diagnostic"] = "accepted LoomBench row lacks manifest_case"
        return row_data
    if app_case != case:
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "loombench_workload_identity_bridge_missing"
        row_data["blocking_prerequisite"] = "loombench_workload_identity_bridge"
        row_data["diagnostic"] = (
            f"accepted LoomBench row maps to app case {app_case!r}; row-level CGRA evidence "
            "cannot be reused until workload identity bridging is explicit"
        )
        return row_data
    app_row = app_row_by_case.get(app_case)
    if app_row is None:
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "loombench_app_row_missing"
        row_data["blocking_prerequisite"] = "app_manifest"
        row_data["diagnostic"] = f"accepted LoomBench row maps to missing app case {app_case!r}"
        return row_data

    row_data["status"] = "blocked"
    row_data["diagnostic_class"] = "loombench_workload_identity_fingerprint_missing"
    row_data["blocking_prerequisite"] = "loombench_workload_identity_fingerprint"
    row_data["diagnostic"] = (
        f"accepted LoomBench row maps to app case {app_case!r}, but source/oracle/input "
        "equivalence is not proven by an explicit fingerprint bridge"
    )
    return row_data


def loombench_rows(
    source_root: Path,
    manifest_path: Path | None = None,
    app_row_by_case: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    if manifest_path is not None:
        app_rows_by_case = app_row_by_case or {}
        return [
            loombench_row_from_manifest_case(case_data, app_rows_by_case)
            for case_data in load_loombench_manifest(manifest_path)
        ]
    if not source_root.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for case_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        case = case_dir.name
        row_data = row(
            suite="loombench",
            case=case,
            source_row=case,
            software_root=case_dir.relative_to(ROOT).as_posix()
            if case_dir.is_relative_to(ROOT)
            else case_dir.as_posix(),
            blocking_prerequisite="loombench_manifest",
            diagnostic="dedicated LoomBench manifest reconciliation is absent",
        )
        row_data["status"] = "blocked"
        row_data["diagnostic_class"] = "loombench_manifest_missing"
        row_data["owner"] = "loombench_manifest"
        rows.append(row_data)
    return rows


def suite_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row_data in rows:
        suite = row_data["suite"]
        suite_counts = counts.setdefault(
            suite,
            {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "blocked": 0,
                "unsupported": 0,
                "missing_status": 0,
            },
        )
        suite_counts["total"] += 1
        status = row_data["status"]
        if status in STATUS_KEYS:
            suite_counts[status] += 1
        if row_data.get("diagnostic_class") == "missing_status":
            suite_counts["missing_status"] += 1
    return counts


def relative_path_text(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def artifact_fingerprint(path: Path) -> str:
    return intermediate_artifacts.artifact_fingerprint(path)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"status": "fail", "_json_error": str(exc)}
    return data if isinstance(data, dict) else {}


def string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def valid_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def valid_memory_state(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) and valid_string_list(item) for key, item in value.items())


def stage_status(data: dict[str, object], path: Path) -> str:
    if not path.is_file():
        return "not_run"
    status = string_field(data, "status")
    return status if status in intermediate_artifacts.BASE_STATUSES else "fail"


def fill_artifact_fields(row_data: dict[str, str], prefix: str, path: Path) -> None:
    row_data[prefix] = relative_path_text(path)
    row_data[f"{prefix}_fingerprint"] = artifact_fingerprint(path)


def comparison_report_path(comparison_dir: Path, case: str) -> Path:
    return comparison_dir / f"{case}.sim-comparison-report.json"


def generate_comparison_report(
    comparison_dir: Path,
    case: str,
    dfg: Path,
    cgra: Path,
    mapping: Path,
) -> Path:
    output = comparison_report_path(comparison_dir, case)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = sim_comparison_report.build_report(dfg, cgra, mapping)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output


def has_component_evidence(evidence_dir: Path, stems: Iterable[str]) -> bool:
    for stem in stems:
        patterns = (
            f"{stem}.*.mapping.json",
            f"{stem}.*.cgra.report.json",
            f"{stem}.dfg.*.report.json",
        )
        if any(any(evidence_dir.glob(pattern)) for pattern in patterns):
            return True
    return False


def first_problem(
    *,
    dfg_path: Path,
    mapping_path: Path,
    cgra_path: Path,
    comparison_path: Path,
    dfg_status: str,
    mapping_status: str,
    cgra_status: str,
    comparison_status: str,
    component_evidence: bool,
) -> tuple[str, str]:
    if component_evidence and (
        not mapping_path.is_file()
        or not cgra_path.is_file()
        or not comparison_path.is_file()
    ):
        return "missing_aggregate_cgra_status_evidence", "aggregate_workload_graph_artifact"
    if not dfg_path.is_file():
        return "missing_dfg_report", "dfg_report"
    if dfg_status == "fail":
        return "dfg_report_failed", "dfg_report"
    if dfg_status in {"blocked", "unsupported", "skipped", "not_run"}:
        return f"dfg_report_{dfg_status}", "dfg_report"
    if not mapping_path.is_file():
        return "missing_mapping_artifact", "mapping_artifact"
    if mapping_status == "fail":
        return "mapping_artifact_failed", "mapping_artifact"
    if mapping_status in {"blocked", "unsupported", "skipped", "not_run"}:
        return f"mapping_artifact_{mapping_status}", "mapping_artifact"
    if not cgra_path.is_file():
        return "missing_cgra_report", "cgra_report"
    if cgra_status == "fail":
        return "cgra_report_failed", "cgra_report"
    if cgra_status in {"blocked", "unsupported", "skipped", "not_run"}:
        return f"cgra_report_{cgra_status}", "cgra_report"
    if not comparison_path.is_file():
        return "missing_sim_comparison_report", "sim_comparison_report"
    if comparison_status == "fail":
        return "sim_comparison_failed", "sim_comparison_report"
    return "sim_comparison_blocked", "sim_comparison_report"


def comparison_diagnostic(comparison: dict[str, object]) -> str:
    diagnostics = comparison.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        return "; ".join(str(item) for item in diagnostics)
    status = string_field(comparison, "status")
    return f"simulation comparison status is {status}" if status else "simulation comparison is unavailable"


def artifact_diagnostic(label: str, data: dict[str, object], path: Path, status: str) -> str:
    if not path.is_file():
        return f"{label} is absent"
    if "_json_error" in data:
        return f"{label} is not valid JSON: {data['_json_error']}"
    diagnostics = data.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        return "; ".join(str(item) for item in diagnostics)
    return f"{label} status is {status}"


def workload_identity_diagnostics(
    case: str,
    artifacts: tuple[tuple[str, dict[str, object], Path], ...],
) -> list[str]:
    diagnostics: list[str] = []
    for label, data, path in artifacts:
        if not path.is_file():
            continue
        if "_json_error" in data:
            continue
        workload = string_field(data, "workload")
        if not workload:
            diagnostics.append(f"{label} lacks workload identity for row case {case!r}")
        elif workload != case:
            diagnostics.append(f"{label} workload identity {workload!r} does not match row case {case!r}")
    return diagnostics


def sim_evidence_stems(row_data: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    for value in (row_data.get("case", ""), row_data.get("source_row", "")):
        if not value:
            continue
        candidates.append(value)
        stem = Path(value).stem
        if stem:
            candidates.append(stem)
    return ordered_unique(candidates)


def first_sim_evidence_path(evidence_dir: Path, stems: list[str], suffix: str) -> Path:
    for stem in stems:
        path = evidence_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    return evidence_dir / f"{stems[0]}{suffix}"


def apply_sim_evidence_to_row(row_data: dict[str, str], evidence_dir: Path, comparison_dir: Path) -> None:
    suite = row_data.get("suite")
    if suite == "app":
        if row_data.get("diagnostic_class") == "app_dataflow_tier_missing":
            return
    elif suite in {"cmsis-dsp", "cmsis-nn"}:
        if row_data.get("diagnostic_class") != "cmsis_dfg_mlir_ready_for_dfg_sim":
            return
    else:
        return
    case = row_data["case"]
    stems = sim_evidence_stems(row_data)
    dfg_path = first_sim_evidence_path(evidence_dir, stems, ".dfg.report.json")
    mapping_path = first_sim_evidence_path(evidence_dir, stems, ".mapping.json")
    cgra_path = first_sim_evidence_path(evidence_dir, stems, ".cgra.report.json")
    component_evidence = suite == "app" and has_component_evidence(evidence_dir, stems)
    original_graph_ids = [item for item in row_data.get("graph_ids", "").split(",") if item]

    if not any(path.is_file() for path in (dfg_path, mapping_path, cgra_path)):
        if component_evidence:
            row_data["status"] = "blocked"
            row_data["diagnostic_class"] = "missing_aggregate_cgra_status_evidence"
            row_data["owner"] = "sim_report"
            row_data["blocking_prerequisite"] = "aggregate_workload_graph_artifact"
            row_data["diagnostic"] = (
                "component simulator evidence exists, but row-level aggregate DFG, mapping, "
                "CGRA, and comparison artifacts are absent"
            )
        return

    dfg = read_json(dfg_path)
    mapping = read_json(mapping_path)
    cgra = read_json(cgra_path)
    if dfg_path.is_file():
        fill_artifact_fields(row_data, "dfg_report", dfg_path)
    if mapping_path.is_file():
        fill_artifact_fields(row_data, "mapping_artifact", mapping_path)
    if cgra_path.is_file():
        fill_artifact_fields(row_data, "cgra_report", cgra_path)

    dfg_status = stage_status(dfg, dfg_path)
    mapping_status = stage_status(mapping, mapping_path)
    cgra_status = stage_status(cgra, cgra_path)
    row_data["dfg_status"] = dfg_status
    row_data["mapping_status"] = mapping_status
    row_data["cgra_status"] = cgra_status

    if dfg_path.is_file() and mapping_path.is_file() and cgra_path.is_file():
        comparison_path = generate_comparison_report(comparison_dir, case, dfg_path, cgra_path, mapping_path)
    else:
        comparison_path = comparison_report_path(comparison_dir, case)
    comparison = read_json(comparison_path)
    comparison_status = stage_status(comparison, comparison_path)
    if comparison_path.is_file():
        fill_artifact_fields(row_data, "comparison_report", comparison_path)
        row_data["comparison_status"] = comparison_status

    graph = string_field(dfg, "graph") or string_field(mapping, "graph")
    graph_identity_diagnostics: list[str] = []
    if graph:
        if suite in {"cmsis-dsp", "cmsis-nn"} and original_graph_ids:
            if graph not in original_graph_ids:
                graph_identity_diagnostics.append(
                    f"dfg_report graph {graph!r} is not listed in row graph_ids {','.join(original_graph_ids)!r}"
                )
        else:
            row_data["graph_ids"] = graph
    mapping_id = string_field(mapping, "mapping_id") or string_field(cgra, "mapping_id")
    if mapping_id:
        row_data["mapping_id"] = mapping_id
    hardware = string_field(mapping, "hardware") or string_field(cgra, "hardware")
    if hardware:
        row_data["hardware_system"] = hardware
        row_data["spatialcore_template"] = hardware
    if row_data.get("required_slice_count", "0") == "0":
        row_data["required_slice_count"] = "1"

    final_outputs_present = (
        dfg_status == "pass"
        and valid_string_list(dfg.get("final_outputs"))
        and valid_string_list(cgra.get("final_outputs"))
        and dfg.get("final_outputs") == cgra.get("final_outputs")
    )
    final_memory_present = (
        dfg_status == "pass"
        and valid_memory_state(dfg.get("final_memory_state"))
        and valid_memory_state(cgra.get("final_memory_state"))
        and dfg.get("final_memory_state") == cgra.get("final_memory_state")
    )
    row_data["final_outputs_present"] = "true" if final_outputs_present else "false"
    row_data["final_memory_state_present"] = "true" if final_memory_present else "false"

    identity_diagnostics = workload_identity_diagnostics(
        case,
        (
            ("dfg_report", dfg, dfg_path),
            ("mapping_artifact", mapping, mapping_path),
            ("cgra_report", cgra, cgra_path),
            ("comparison_report", comparison, comparison_path),
        ),
    )
    identity_diagnostics.extend(graph_identity_diagnostics)
    if identity_diagnostics:
        row_data["status"] = "fail"
        row_data["diagnostic_class"] = "evidence_identity_mismatch"
        row_data["owner"] = "sim_report"
        row_data["blocking_prerequisite"] = "sim_evidence_identity"
        row_data["diagnostic"] = "; ".join(identity_diagnostics)
        return

    stage_values = (dfg_status, mapping_status, cgra_status, comparison_status)
    if all(value == "pass" for value in stage_values) and (final_outputs_present or final_memory_present):
        row_data["status"] = "pass"
        row_data["diagnostic_class"] = "cgra_sim_pass"
        row_data["owner"] = "sim_report"
        row_data["blocking_prerequisite"] = ""
        row_data["diagnostic"] = "DFG-sim, mapping, CGRA-sim, and simulation comparison evidence passed"
        return
    if any(value == "fail" for value in stage_values):
        row_data["status"] = "fail"
    else:
        row_data["status"] = "blocked"
    diagnostic_class, prerequisite = first_problem(
        dfg_path=dfg_path,
        mapping_path=mapping_path,
        cgra_path=cgra_path,
        comparison_path=comparison_path,
        dfg_status=dfg_status,
        mapping_status=mapping_status,
        cgra_status=cgra_status,
        comparison_status=comparison_status,
        component_evidence=component_evidence,
    )
    row_data["diagnostic_class"] = diagnostic_class
    row_data["owner"] = "sim_report"
    row_data["blocking_prerequisite"] = prerequisite
    if prerequisite == "aggregate_workload_graph_artifact":
        row_data["diagnostic"] = (
            "component simulator evidence exists, but row-level aggregate DFG, mapping, "
            "CGRA, and comparison artifacts are incomplete"
        )
    elif prerequisite == "dfg_report":
        row_data["diagnostic"] = artifact_diagnostic("DFG-sim report", dfg, dfg_path, dfg_status)
    elif prerequisite == "mapping_artifact":
        row_data["diagnostic"] = artifact_diagnostic("PnR mapping artifact", mapping, mapping_path, mapping_status)
    elif prerequisite == "cgra_report":
        row_data["diagnostic"] = artifact_diagnostic("CGRA-sim report", cgra, cgra_path, cgra_status)
    elif prerequisite == "sim_comparison_report":
        row_data["diagnostic"] = comparison_diagnostic(comparison)
    else:
        row_data["diagnostic"] = f"{case} has incomplete CGRA status evidence: {diagnostic_class}"


def apply_sim_evidence(rows: list[dict[str, str]], evidence_dir: Path, comparison_dir: Path) -> None:
    if not evidence_dir.is_dir():
        return
    for row_data in rows:
        apply_sim_evidence_to_row(row_data, evidence_dir, comparison_dir)


def json_path_for(csv_output: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return csv_output.with_suffix(".json")


def default_cmsis_dfg_dir(output: Path, suite: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    return None


def write_json(path: Path, csv_output: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "kind": "cgra_status_summary",
        "csv_projection": str(csv_output),
        "counts": suite_counts(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.no_legacy_loombench and args.loombench_manifest:
        raise SystemExit("--no-legacy-loombench cannot be combined with --loombench-manifest")
    legacy_root = Path(args.legacy_loombench_root)
    output = Path(args.output)
    cmsis_dsp_dfg_dir = default_cmsis_dfg_dir(output, "cmsis-dsp", args.cmsis_dsp_dfg_dir)
    cmsis_nn_dfg_dir = default_cmsis_dfg_dir(output, "cmsis-nn", args.cmsis_nn_dfg_dir)
    sim_evidence_dir = Path(args.sim_evidence_dir) if args.sim_evidence_dir else output.parent / "current-sim-cycle"
    comparison_dir = (
        Path(args.comparison_output_dir)
        if args.comparison_output_dir
        else output.parent / "cgra-status-comparisons"
    )
    app_status_rows = app_rows()
    rows = []
    rows.extend(app_status_rows)
    rows.extend(
        cmsis_rows(
            "cmsis-dsp",
            "cmsis-dsp",
            "cmsis_dsp_targets.txt",
            "externals/cmsis-dsp/Source",
            cmsis_dsp_dfg_dir,
        )
    )
    rows.extend(
        cmsis_rows(
            "cmsis-nn",
            "cmsis-nn",
            "cmsis_nn_targets.txt",
            "externals/cmsis-nn/Source",
            cmsis_nn_dfg_dir,
        )
    )
    if not args.no_legacy_loombench:
        rows.extend(
            loombench_rows(
                legacy_root,
                Path(args.loombench_manifest) if args.loombench_manifest else None,
                {row_data["case"]: row_data for row_data in app_status_rows},
            )
        )

    apply_sim_evidence(rows, sim_evidence_dir, comparison_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("cgra_status", output, rows)
    write_json(json_path_for(output, args.json_output), output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
