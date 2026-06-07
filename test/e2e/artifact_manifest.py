#!/usr/bin/env python3
"""Emit full-stack artifact manifest records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


artifact_id = intermediate_artifacts.artifact_id_for_path
fingerprint = intermediate_artifacts.artifact_fingerprint


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
    for left, right in intermediate_artifacts.iter_artifact_manifest_required_edges(seen_ids, ids_by_kind):
        add_edge(edges, edge_keys, left, right)

    for edge in edges:
        edge["producer_artifact_kind"] = kind_by_id.get(edge["from"], "")
        edge["consumer_artifact_kind"] = kind_by_id.get(edge["to"], "")
        edge["producer_component"] = intermediate_artifacts.manifest_component_for_kind(edge["producer_artifact_kind"])
        edge["consumer_component"] = intermediate_artifacts.manifest_component_for_kind(edge["consumer_artifact_kind"])
        edge["public_spec_owner"] = "docs/spec-full-stack-traceability.md"
        edge["schema_or_verifier"] = "intermediate_artifact_audit"
        edge["validation_command_role"] = "artifact content audit"
        edge["negative_diagnostic_classes"] = ["missing_edge", "stale_fingerprint"]
        edge["minimal_positive_demonstrator_requirement"] = "intermediate artifact chain"
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
