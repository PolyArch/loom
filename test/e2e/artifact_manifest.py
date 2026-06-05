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


ARTIFACT_EDGES = (
    ("old-app-corpus-inventory", "app-corpus-import-status"),
    ("app-corpus-import-status", "source-compat-summary"),
    ("source-compat-summary", "compiler-pipeline-summary"),
    ("compiler-pipeline-summary", "dataflow-primitive-coverage"),
    ("dataflow-primitive-coverage", "pnr-mapping-summary"),
    ("adg-hardware-summary", "pnr-mapping-summary"),
    ("dataflow-primitive-coverage", "sim-cycle-summary"),
    ("dataflow-primitive-coverage", "rtl-fpa-summary"),
    ("adg-hardware-summary", "rtl-fpa-summary"),
    ("pnr-mapping-summary", "e2e-demonstrator-summary"),
    ("sim-cycle-summary", "e2e-demonstrator-summary"),
    ("rtl-fpa-summary", "e2e-demonstrator-summary"),
    ("pnr-mapping-summary", "dse-candidate-summary"),
    ("sim-cycle-summary", "dse-candidate-summary"),
    ("rtl-fpa-summary", "dse-candidate-summary"),
    ("dataflow-primitive-coverage", "unsupported-scope-ledger"),
    ("pnr-mapping-summary", "unsupported-scope-ledger"),
    ("sim-cycle-summary", "unsupported-scope-ledger"),
    ("rtl-fpa-summary", "unsupported-scope-ledger"),
    ("e2e-demonstrator-summary", "unsupported-scope-ledger"),
    ("dse-candidate-summary", "unsupported-scope-ledger"),
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
        artifacts.append(
            {
                "kind": kind,
                "id": identity,
                "path": str(path),
                "producer": "artifact summary command",
                "status": "present",
                "fingerprint": fingerprint(path),
            }
        )

    edges = []
    for left, right in ARTIFACT_EDGES:
        if left in seen_ids and right in seen_ids:
            edges.append({"from": left, "to": right, "kind": "producer-consumer"})

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
