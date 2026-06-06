#!/usr/bin/env python3
"""Emit full-stack artifact manifest records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def component_for_kind(kind: str) -> str:
    return f"{kind}-producer" if kind else ""


def add_edge(edges: list[dict[str, str]], edge_keys: set[tuple[str, str]], left: str, right: str) -> None:
    if (left, right) in edge_keys:
        return
    edge_keys.add((left, right))
    edges.append(
        {
            "id": f"edge::{left}->{right}",
            "from": left,
            "to": right,
            "kind": "producer-consumer",
        }
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_id(path: Path) -> str:
    for suffix in (".csv", ".json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def discover_artifacts(explicit: list[str]) -> list[Path]:
    return intermediate_artifacts.discover_artifact_paths(
        ROOT,
        explicit,
        include_unsupported_scope=True,
    )


def build_manifest(paths: list[Path]) -> dict[str, object]:
    artifacts = []
    diagnostics = []
    seen_ids = set()
    ids_by_kind: dict[str, list[str]] = {}
    kind_by_id: dict[str, str] = {}
    fingerprint_by_id: dict[str, str] = {}
    for path in paths:
        kind = intermediate_artifacts.artifact_kind_for_path(path)
        identity = artifact_id(path)
        if not path.is_file():
            diagnostics.append({"status": "blocked", "message": f"missing artifact: {path}"})
            continue
        if kind == "unknown":
            diagnostics.append({"status": "blocked", "message": f"unknown artifact schema: {path}"})
        if identity in seen_ids:
            diagnostics.append({"status": "blocked", "message": f"duplicate artifact id: {identity}"})
        seen_ids.add(identity)
        ids_by_kind.setdefault(kind, []).append(identity)
        kind_by_id[identity] = kind
        artifact_fingerprint = fingerprint(path)
        fingerprint_by_id[identity] = artifact_fingerprint
        artifacts.append(
            {
                "kind": kind,
                "id": identity,
                "path": str(path),
                "producer": "artifact summary command",
                "status": "present",
                "fingerprint": artifact_fingerprint,
            }
        )

    edges = []
    edge_keys: set[tuple[str, str]] = set()
    for left, right in intermediate_artifacts.ARTIFACT_EDGE_PAIRS:
        if left in seen_ids and right in seen_ids:
            add_edge(edges, edge_keys, left, right)

    for mapping_id in ids_by_kind.get("pnr_mapping_artifact", []):
        for source_kind in ("dataflow_primitive_coverage", "adg_hardware", "pnr_mapping"):
            for source_id in ids_by_kind.get(source_kind, []):
                add_edge(edges, edge_keys, source_id, mapping_id)
        for cgra_id in ids_by_kind.get("cgra_sim_report", []):
            add_edge(edges, edge_keys, mapping_id, cgra_id)
        for dse_id in ids_by_kind.get("dse_candidate", []):
            add_edge(edges, edge_keys, mapping_id, dse_id)

    for sim_id in ids_by_kind.get("sim_cycle", []):
        for dfg_id in ids_by_kind.get("dfg_sim_report", []):
            add_edge(edges, edge_keys, dfg_id, sim_id)
        if sim_id == "sim-cycle-summary":
            for cgra_id in ids_by_kind.get("cgra_sim_report", []):
                add_edge(edges, edge_keys, cgra_id, sim_id)

    for dfg_id in ids_by_kind.get("dfg_sim_report", []):
        for source_id in ids_by_kind.get("dataflow_primitive_coverage", []):
            add_edge(edges, edge_keys, source_id, dfg_id)

    for cgra_id in ids_by_kind.get("cgra_sim_report", []):
        for dse_id in ids_by_kind.get("dse_candidate", []):
            add_edge(edges, edge_keys, cgra_id, dse_id)

    for comparison_id in ids_by_kind.get("sim_comparison_report", []):
        for source_kind in ("dfg_sim_report", "cgra_sim_report", "pnr_mapping_artifact"):
            for source_id in ids_by_kind.get(source_kind, []):
                add_edge(edges, edge_keys, source_id, comparison_id)

    for runtime_id in ids_by_kind.get("runtime_package", []):
        for source_kind in ("pnr_mapping_artifact", "cgra_sim_report", "sim_comparison_report"):
            for source_id in ids_by_kind.get(source_kind, []):
                add_edge(edges, edge_keys, source_id, runtime_id)

    for report_id in ids_by_kind.get("workload_report_bundle", []):
        for source_kind in (
            "source_compat",
            "compiler_pipeline",
            "dataflow_primitive_coverage",
            "adg_hardware",
            "pnr_mapping_artifact",
            "dfg_sim_report",
            "cgra_sim_report",
            "sim_comparison_report",
            "runtime_package",
            "sim_cycle",
            "rtl_fpa",
            "dse_candidate",
        ):
            for source_id in ids_by_kind.get(source_kind, []):
                add_edge(edges, edge_keys, source_id, report_id)
        for demonstrator_id in ids_by_kind.get("e2e_demonstrator", []):
            add_edge(edges, edge_keys, report_id, demonstrator_id)

    for hardware_report_id in ids_by_kind.get("hardware_report_bundle", []):
        for source_kind in ("adg_hardware", "rtl_fpa"):
            for source_id in ids_by_kind.get(source_kind, []):
                add_edge(edges, edge_keys, source_id, hardware_report_id)
        for demonstrator_id in ids_by_kind.get("e2e_demonstrator", []):
            add_edge(edges, edge_keys, hardware_report_id, demonstrator_id)

    for dse_report_id in ids_by_kind.get("dse_report_bundle", []):
        for source_kind in ("dse_candidate", "workload_report_bundle", "hardware_report_bundle"):
            for source_id in ids_by_kind.get(source_kind, []):
                add_edge(edges, edge_keys, source_id, dse_report_id)

    for edge in edges:
        edge["producer_artifact_kind"] = kind_by_id.get(edge["from"], "")
        edge["consumer_artifact_kind"] = kind_by_id.get(edge["to"], "")
        edge["producer_component"] = component_for_kind(edge["producer_artifact_kind"])
        edge["consumer_component"] = component_for_kind(edge["consumer_artifact_kind"])
        edge["required_input_fingerprints"] = {
            edge["from"]: fingerprint_by_id.get(edge["from"], "")
        }
        edge["produced_output_fingerprints"] = {
            edge["to"]: fingerprint_by_id.get(edge["to"], "")
        }

    return {
        "schema_version": 1,
        "run_id": "artifact-manifest",
        "artifacts": artifacts,
        "edges": edges,
        "diagnostics": diagnostics,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    paths = discover_artifacts(args.artifact)
    if not paths:
        intermediate_artifacts.write_json("artifact_manifest", intermediate_artifacts.output_path(args.output))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(paths)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 1 if manifest["diagnostics"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
