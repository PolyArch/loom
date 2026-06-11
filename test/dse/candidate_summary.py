#!/usr/bin/env python3
"""Emit DSE candidate summary rows from mapping, sim, and FPA artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import dse_objectives  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--objective", default="minimize_runtime")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def attach_path(row: dict[str, str], path: Path) -> dict[str, str]:
    copied = dict(row)
    copied["__path"] = str(path)
    return copied


def artifacts_by_kind(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def mapping_rows_from_artifacts(artifacts: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for artifact in artifacts:
        workload = artifact.get("workload")
        hardware = artifact.get("hardware")
        mapping_id = artifact.get("mapping_id")
        status = artifact.get("status")
        if (
            not isinstance(workload, str)
            or workload in {"", "scaffold"}
            or not isinstance(hardware, str)
            or hardware in {"", "scaffold"}
            or not isinstance(mapping_id, str)
        ):
            continue
        rows.append(
            {
                "workload": workload,
                "hardware": hardware,
                "mapping_id": mapping_id,
                "status": str(status) if status not in {"", None} else "",
                "__path": artifact_ref(artifact.get("__path")),
            }
        )
    return rows


def resolve_manifest_path(manifest_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or raw_path == "":
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def mapping_rows_from_manifests(manifests: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest in manifests:
        manifest_path_text = manifest.get("__path")
        if not isinstance(manifest_path_text, str) or manifest_path_text == "":
            continue
        manifest_path = Path(manifest_path_text)
        candidates = manifest.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            mapping_path = resolve_manifest_path(manifest_path, candidate.get("mapping_artifact"))
            if mapping_path is None or not mapping_path.is_file():
                continue
            try:
                mapping = json.loads(mapping_path.read_text())
            except json.JSONDecodeError:
                continue
            if not isinstance(mapping, dict) or mapping.get("kind") != "pnr_mapping":
                continue
            mapping["__path"] = str(mapping_path)
            for row in mapping_rows_from_artifacts([mapping]):
                row["__manifest_path"] = str(manifest_path)
                policy_id = manifest.get("policy_id")
                if isinstance(policy_id, str):
                    row["__policy_id"] = policy_id
                objective = manifest.get("objective")
                if isinstance(objective, str):
                    row["__objective"] = objective
                rows.append(row)
    return rows


def mapping_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(
            attach_path(row, path)
            for row in read_csv(path)
            if row.get("workload") not in {"", "scaffold", None}
            and row.get("hardware") not in {"", "scaffold", None}
        )
    return rows


def dedupe_mapping_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("workload", ""), row.get("hardware", ""), row.get("mapping_id", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def sim_for_workload(paths: list[Path], workload: str) -> dict[str, str]:
    for path in paths:
        for row in read_csv(path):
            if row.get("kernel") == workload:
                return attach_path(row, path)
    return {}


def fpa_for_candidate(paths: list[Path], workload: str, hardware: str) -> dict[str, str]:
    suffix_matches: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("workload") != workload:
                continue
            row_hardware = row.get("hardware", "")
            if row_hardware == hardware:
                return attach_path(row, path)
            if row_hardware.rsplit("::", 1)[-1] == hardware:
                suffix_matches.append(attach_path(row, path))
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return {}


def hardware_refs_match(candidate: str, evidence: object) -> bool:
    if not isinstance(evidence, str) or not evidence:
        return False
    return evidence == candidate or evidence.rsplit("::", 1)[-1] == candidate


def mapping_artifact_for_candidate(
    artifacts: list[dict[str, object]],
    workload: str,
    hardware: str,
    mapping_id: str,
) -> dict[str, object]:
    matches = [
        artifact
        for artifact in artifacts
        if artifact.get("workload") == workload
        and artifact.get("mapping_id") == mapping_id
        and hardware_refs_match(hardware, artifact.get("hardware"))
    ]
    if len(matches) == 1:
        return matches[0]
    return {}


def mapping_artifact_from_row(row: dict[str, str]) -> dict[str, object]:
    path_text = row.get("__path", "")
    if path_text == "":
        return {}
    path = Path(path_text)
    if path.suffix != ".json" or not path.is_file():
        return {}
    try:
        artifact = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(artifact, dict) or artifact.get("kind") != "pnr_mapping":
        return {}
    if (
        artifact.get("workload") == row.get("workload")
        and artifact.get("mapping_id") == row.get("mapping_id")
        and hardware_refs_match(row.get("hardware", ""), artifact.get("hardware"))
    ):
        artifact["__path"] = str(path)
        return artifact
    return {}


def mapping_artifact_evidence(
    artifacts: list[dict[str, object]],
    row: dict[str, str],
) -> dict[str, object]:
    explicit = mapping_artifact_for_candidate(
        artifacts,
        row["workload"],
        row["hardware"],
        row.get("mapping_id", ""),
    )
    if explicit:
        return explicit
    return mapping_artifact_from_row(row)


def cgra_report_for_candidate(
    reports: list[dict[str, object]],
    workload: str,
    hardware: str,
    mapping_id: str,
) -> dict[str, object]:
    matches = [
        report
        for report in reports
        if report.get("workload") == workload
        and report.get("mapping_id") == mapping_id
        and hardware_refs_match(hardware, report.get("hardware"))
    ]
    if len(matches) == 1:
        return matches[0]
    return {}


def parse_float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def parse_positive_float(row: dict[str, str], column: str) -> float | None:
    parsed = parse_float(row, column)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def artifact_ref(value: object) -> str:
    return str(value) if value not in {"", None} else ""


def artifact_identity(ref: str) -> str:
    return intermediate_artifacts.artifact_id_for_path(Path(ref)) if ref else ""


def artifact_identity_list(refs: list[str]) -> list[str]:
    identities: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        identity = artifact_identity(ref)
        if identity == "" or identity in seen:
            continue
        identities.append(identity)
        seen.add(identity)
    return identities


def input_artifact_fingerprints(refs: list[str]) -> str:
    entries: list[str] = []
    for ref in refs:
        path = Path(ref)
        identity = artifact_identity(ref)
        if path.is_file():
            entries.append(f"{identity}={intermediate_artifacts.artifact_fingerprint(path)}")
    return ";".join(entries)


def unique_refs(refs: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref == "" or ref in seen:
            continue
        unique.append(ref)
        seen.add(ref)
    return unique


def policy_id_for_objective(objective: str, mapping: dict[str, str] | None = None) -> str:
    if mapping is not None and mapping.get("__policy_id"):
        return mapping["__policy_id"]
    return dse_objectives.policy_id_for_objective(objective)


def ordering_rule_for_objective(objective: str) -> str:
    return dse_objectives.ordering_rule_for_objective(objective)


def complete_candidate_id(workload: str, hardware: str, mapping_id: str) -> str:
    return f"candidate::{workload}::{hardware}::{mapping_id}"


def path_aliases(raw_path: str, base_path: str = "") -> set[str]:
    if raw_path == "":
        return set()
    path = Path(raw_path)
    aliases = {raw_path, str(path)}
    if base_path and not path.is_absolute():
        path = Path(base_path).parent / path
        aliases.add(str(path))
    aliases.add(str(path.resolve()))
    return aliases


def blocking_input_matches(row: dict[str, str], input_refs: list[str]) -> bool:
    blocking_input = row.get("blocking_input", "")
    if blocking_input == "":
        return False
    input_aliases: set[str] = set()
    for ref in input_refs:
        input_aliases.update(path_aliases(ref))
    return bool(path_aliases(blocking_input, row.get("__path", "")) & input_aliases)


def matching_unsupported_scope_rows(
    ledger_rows: list[dict[str, str]],
    candidate_id: str,
    mapping_id: str,
    input_refs: list[str],
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for row in ledger_rows:
        case = row.get("case", "")
        if case == candidate_id or case.startswith(f"{candidate_id}:"):
            matches.append(row)
            continue
        if case == mapping_id or case.endswith(f":{mapping_id}"):
            matches.append(row)
            continue
        if blocking_input_matches(row, input_refs):
            matches.append(row)
    return matches


def unsupported_scope_count(
    ledger_rows: list[dict[str, str]],
    candidate_id: str,
    mapping_id: str,
    input_refs: list[str],
) -> int:
    return len(matching_unsupported_scope_rows(ledger_rows, candidate_id, mapping_id, input_refs))


def complete_evidence(
    mapping: dict[str, str],
    sim: dict[str, str],
    fpa: dict[str, str],
    mapping_artifact: dict[str, object],
    cgra_report: dict[str, object],
) -> tuple[float, float] | None:
    if mapping.get("status") != "pass" or not mapping.get("mapping_id"):
        return None
    if sim.get("status") != "pass":
        return None
    if fpa.get("status") != "pass":
        return None
    if mapping_artifact.get("status") != "pass":
        return None
    if cgra_report.get("status") != "pass":
        return None

    report_cycles = cgra_report.get("hardware_aware_cycles")
    if not isinstance(report_cycles, int) or report_cycles <= 0:
        return None
    cycles = float(report_cycles)
    frequency_mhz = parse_positive_float(fpa, "frequency_mhz")
    dynamic_power_mw = parse_float(fpa, "dynamic_power_mw")
    leakage_power_mw = parse_float(fpa, "leakage_power_mw")
    area_um2 = parse_positive_float(fpa, "area_um2")
    if (
        cycles is None
        or frequency_mhz is None
        or dynamic_power_mw is None
        or leakage_power_mw is None
        or area_um2 is None
    ):
        return None
    total_power_mw = dynamic_power_mw + leakage_power_mw
    energy_nj = total_power_mw * cycles / frequency_mhz
    return cycles, energy_nj


def metric_fidelity_record_values(fpa: dict[str, str]) -> list[str]:
    fidelity = fpa.get("fidelity_level", "")
    frequency_source = fpa.get("frequency_source", "")
    area_source = fpa.get("area_source", "")
    power_source = fpa.get("power_source", "")
    activity_source = fpa.get("activity_source", "")
    return [
        f"frequency_mhz={fidelity}:{frequency_source}",
        f"area_um2={fidelity}:{area_source}",
        f"dynamic_power_mw={fidelity}:{power_source}:{activity_source}",
        f"leakage_power_mw={fidelity}:{power_source}:{activity_source}",
        f"energy_nj={fidelity}:derived_from_fpa_and_cgra_sim",
    ]


def hardware_evidence_kind_for_fpa(fpa: dict[str, str]) -> str:
    fidelity = fpa.get("fidelity_level", "")
    if fidelity == "analytic":
        return "analytic_model_only"
    if fidelity == "mapped_activity":
        return "sim_activity_model"
    if fidelity in {"rtl_structural", "rtl_activity", "physical_estimate", "fpga_estimate"}:
        return "backend_evidence"
    if fidelity in {"custom", "custom_calibrated"}:
        return "custom_model"
    return "unknown"


def candidate_row(
    mapping: dict[str, str],
    sim: dict[str, str],
    fpa: dict[str, str],
    mapping_artifact: dict[str, object],
    cgra_report: dict[str, object],
    objective: str,
    output_artifact: Path,
    unsupported_scope_rows: list[dict[str, str]],
) -> dict[str, str]:
    workload = mapping["workload"]
    hardware = mapping["hardware"]
    effective_objective = mapping.get("__objective") or objective
    complete = complete_evidence(mapping, sim, fpa, mapping_artifact, cgra_report)
    if complete is not None:
        cycles, energy_nj = complete
        cycle_text = str(int(cycles))
        input_refs = unique_refs(
            [
                ref
                for ref in (
                    artifact_ref(mapping.get("__manifest_path")),
                    artifact_ref(mapping.get("__path")),
                    artifact_ref(mapping_artifact.get("__path")),
                    artifact_ref(sim.get("__path")),
                    artifact_ref(cgra_report.get("__path")),
                    artifact_ref(fpa.get("__path")),
                )
                if ref
            ]
        )
        candidate_id = complete_candidate_id(workload, hardware, mapping["mapping_id"])
        unsupported_count = unsupported_scope_count(
            unsupported_scope_rows,
            candidate_id,
            mapping["mapping_id"],
            input_refs,
        )
        ledger_refs = [
            ref
            for ref in (
                artifact_ref(row.get("__path"))
                for row in (
                    unsupported_scope_rows
                    if effective_objective == "minimize_unsupported_scope_diagnostics"
                    else matching_unsupported_scope_rows(
                        unsupported_scope_rows,
                        candidate_id,
                        mapping["mapping_id"],
                        input_refs,
                    )
                )
            )
            if ref
        ]
        input_refs = unique_refs(input_refs + ledger_refs)
        input_artifacts = ";".join(artifact_identity_list(input_refs))
        metric_record_values = [
            f"cgra_sim_cycles={cycle_text}",
            f"frequency_mhz={fpa['frequency_mhz']}",
            f"area_um2={fpa['area_um2']}",
            f"dynamic_power_mw={fpa['dynamic_power_mw']}",
            f"leakage_power_mw={fpa['leakage_power_mw']}",
            f"energy_nj={energy_nj:.3f}",
        ]
        if effective_objective == "minimize_unsupported_scope_diagnostics":
            metric_record_values.append(f"unsupported_scope_diagnostics_count={unsupported_count}")
        metric_records = ";".join(metric_record_values)
        feedback_fidelity_records = ";".join(metric_fidelity_record_values(fpa))
        row = {
            "candidate": candidate_id,
            "workload": workload,
            "hardware": hardware,
            "mapping_id": mapping["mapping_id"],
            "objective": effective_objective,
            "cgra_sim_cycles": cycle_text,
            "frequency_mhz": fpa["frequency_mhz"],
            "area_um2": fpa["area_um2"],
            "dynamic_power_mw": fpa["dynamic_power_mw"],
            "leakage_power_mw": fpa["leakage_power_mw"],
            "energy_nj": f"{energy_nj:.3f}",
            "selection_status": "selected",
            "candidate_kind": "combined_full_stack_candidate",
            "hardware_evidence_kind": hardware_evidence_kind_for_fpa(fpa),
            "input_artifacts": input_artifacts,
            "input_artifact_fingerprints": input_artifact_fingerprints(input_refs),
            "output_artifacts": intermediate_artifacts.artifact_id_for_path(output_artifact),
            "objective_record": f"objective::{effective_objective}",
            "metric_records": metric_records,
            "feedback_fidelity_records": feedback_fidelity_records,
            "policy_id": policy_id_for_objective(effective_objective, mapping),
            "ordering_rule": ordering_rule_for_objective(effective_objective),
            "diagnostic": (
                "cycle-frequency-power-area energy estimate; "
                "energy_nj=(dynamic_power_mw+leakage_power_mw)*"
                "cgra_sim_cycles/frequency_mhz"
            ),
        }
        if effective_objective == "minimize_unsupported_scope_diagnostics":
            row["unsupported_scope_diagnostics_count"] = str(unsupported_count)
        return row
    blocked_mapping_id = ""
    if mapping_artifact or cgra_report:
        blocked_mapping_id = mapping.get("mapping_id", "")
    blocked_input_refs = unique_refs(
        [
            ref
            for ref in (
                artifact_ref(mapping.get("__manifest_path")),
                artifact_ref(mapping.get("__path")),
                artifact_ref(mapping_artifact.get("__path")),
                artifact_ref(sim.get("__path")),
                artifact_ref(cgra_report.get("__path")),
                artifact_ref(fpa.get("__path")),
            )
            if ref
        ]
    )
    return {
        "candidate": complete_candidate_id(workload, hardware, blocked_mapping_id)
        if blocked_mapping_id
        else f"candidate::{workload}::{hardware}",
        "workload": workload,
        "hardware": hardware,
        "mapping_id": blocked_mapping_id,
        "objective": effective_objective,
        "cgra_sim_cycles": "",
        "frequency_mhz": "",
        "area_um2": "",
        "dynamic_power_mw": "",
        "leakage_power_mw": "",
        "energy_nj": "",
        "unsupported_scope_diagnostics_count": "",
        "selection_status": "blocked",
        "candidate_kind": "combined_full_stack_candidate",
        "input_artifacts": ";".join(artifact_identity_list(blocked_input_refs)),
        "input_artifact_fingerprints": input_artifact_fingerprints(blocked_input_refs),
        "output_artifacts": intermediate_artifacts.artifact_id_for_path(output_artifact),
        "objective_record": f"objective::{effective_objective}",
        "metric_records": "",
        "feedback_fidelity_records": "",
        "policy_id": policy_id_for_objective(effective_objective, mapping),
        "ordering_rule": ordering_rule_for_objective(effective_objective),
        "diagnostic": (
            "missing mapping, simulator, or FPA evidence for DSE selection; "
            "requires matching mapping artifact and CGRA simulator report"
        ),
    }


def runtime_score(row: dict[str, str]) -> float:
    cycles = parse_positive_float(row, "cgra_sim_cycles")
    frequency_mhz = parse_positive_float(row, "frequency_mhz")
    if cycles is None or frequency_mhz is None:
        return float("inf")
    return cycles / frequency_mhz


def energy_score(row: dict[str, str]) -> float:
    energy = parse_positive_float(row, "energy_nj")
    return energy if energy is not None else float("inf")


def area_score(row: dict[str, str]) -> float:
    area = parse_positive_float(row, "area_um2")
    return area if area is not None else float("inf")


def dynamic_power_score(row: dict[str, str]) -> float:
    dynamic_power = parse_positive_float(row, "dynamic_power_mw")
    return dynamic_power if dynamic_power is not None else float("inf")


def leakage_power_score(row: dict[str, str]) -> float:
    leakage_power = parse_positive_float(row, "leakage_power_mw")
    return leakage_power if leakage_power is not None else float("inf")


def unsupported_scope_diagnostics_score(row: dict[str, str]) -> float:
    count = parse_float(row, "unsupported_scope_diagnostics_count")
    return count if count is not None else float("inf")


def throughput_score(row: dict[str, str]) -> float:
    cycles = parse_positive_float(row, "cgra_sim_cycles")
    frequency_mhz = parse_positive_float(row, "frequency_mhz")
    if cycles is None or frequency_mhz is None:
        return float("-inf")
    return frequency_mhz / cycles


def performance_per_watt_score(row: dict[str, str]) -> float:
    cycles = parse_positive_float(row, "cgra_sim_cycles")
    frequency_mhz = parse_positive_float(row, "frequency_mhz")
    energy = parse_positive_float(row, "energy_nj")
    if cycles is None or frequency_mhz is None or energy is None:
        return float("-inf")
    runtime_us = cycles / frequency_mhz
    total_power_mw = energy / runtime_us
    if runtime_us <= 0 or total_power_mw <= 0:
        return float("-inf")
    return (1.0 / runtime_us) / (total_power_mw / 1000.0)


def performance_per_area_score(row: dict[str, str]) -> float:
    cycles = parse_positive_float(row, "cgra_sim_cycles")
    frequency_mhz = parse_positive_float(row, "frequency_mhz")
    area_um2 = parse_positive_float(row, "area_um2")
    if cycles is None or frequency_mhz is None or area_um2 is None:
        return float("-inf")
    runtime_us = cycles / frequency_mhz
    if runtime_us <= 0:
        return float("-inf")
    return (1.0 / runtime_us) / area_um2


def select_candidates(rows: list[dict[str, str]], objective: str) -> None:
    score: Callable[[dict[str, str]], float]
    objectives = sorted({row.get("objective") or objective for row in rows})
    for effective_objective in objectives:
        complete = [
            row
            for row in rows
            if row.get("selection_status") == "selected"
            and (row.get("objective") or objective) == effective_objective
        ]
        if len(complete) <= 1:
            continue
        if effective_objective == "maximize_throughput":
            score = throughput_score
            selected = max(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective == "maximize_performance_per_watt":
            score = performance_per_watt_score
            selected = max(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective == "maximize_performance_per_area":
            score = performance_per_area_score
            selected = max(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective == "minimize_area":
            score = area_score
            selected = min(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective == "minimize_dynamic_power":
            score = dynamic_power_score
            selected = min(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective == "minimize_leakage_power":
            score = leakage_power_score
            selected = min(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective == "minimize_unsupported_scope_diagnostics":
            score = unsupported_scope_diagnostics_score
            selected = min(complete, key=lambda row: (score(row), row["candidate"]))
        elif effective_objective in {"minimize_energy", "minimize_power"}:
            score = energy_score
            selected = min(complete, key=lambda row: (score(row), row["candidate"]))
        else:
            score = runtime_score
            selected = min(complete, key=lambda row: (score(row), row["candidate"]))
        for row in complete:
            if row is selected:
                continue
            row["selection_status"] = "rejected"
            row["diagnostic"] = (
                "complete cycle-frequency-power-area evidence; rejected by "
                f"{effective_objective} deterministic ordering"
            )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    paths = intermediate_artifacts.discover_artifact_paths(
        ROOT,
        args.artifact,
        include_unsupported_scope=False,
    )
    grouped = artifacts_by_kind(paths)
    json_grouped = intermediate_artifacts.json_objects_by_kind(paths)
    pnr_mapping_rows = mapping_rows(grouped.get("pnr_mapping", []))
    pnr_mapping_rows.extend(mapping_rows_from_artifacts(json_grouped.get("pnr_mapping_artifact", [])))
    pnr_mapping_rows.extend(mapping_rows_from_manifests(json_grouped.get("mapping_set_manifest", [])))
    pnr_mapping_rows = dedupe_mapping_rows(pnr_mapping_rows)
    unsupported_scope_rows = [
        attach_path(row, path)
        for path in grouped.get("unsupported_scope", [])
        for row in read_csv(path)
    ]
    rows = [
        candidate_row(
            row,
            sim_for_workload(grouped.get("sim_cycle", []), row["workload"]),
            fpa_for_candidate(grouped.get("rtl_fpa", []), row["workload"], row["hardware"]),
            mapping_artifact_evidence(
                json_grouped.get("pnr_mapping_artifact", []),
                row,
            ),
            cgra_report_for_candidate(
                json_grouped.get("cgra_sim_report", []),
                row["workload"],
                row["hardware"],
                row.get("mapping_id", ""),
            ),
            args.objective,
            output,
            unsupported_scope_rows,
        )
        for row in pnr_mapping_rows
    ]
    select_candidates(rows, args.objective)

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("dse_candidate", output, rows)
    else:
        intermediate_artifacts.write_csv("dse_candidate", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
