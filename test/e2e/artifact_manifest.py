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


DISCOVERY_PATHS = (
    "temp/source-compat-summary.csv",
    "temp/compiler-pipeline-summary.csv",
    "temp/dataflow-primitive-coverage.csv",
    "temp/adg-hardware-summary.csv",
    "temp/pnr-mapping-summary.csv",
    "temp/sim-cycle-summary.csv",
    "temp/rtl-fpa-summary.csv",
    "temp/e2e-demonstrator-summary.csv",
    "temp/dse-candidate-summary.csv",
    "temp/unsupported-scope-ledger.csv",
)

CHAIN = (
    "source_compat",
    "compiler_pipeline",
    "dataflow_primitive_coverage",
    "adg_hardware",
    "pnr_mapping",
    "sim_cycle",
    "rtl_fpa",
    "e2e_demonstrator",
    "dse_candidate",
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


def artifact_kind(path: Path) -> str:
    schema = intermediate_artifacts.schema_for_path(path)
    if schema is not None:
        return schema.kind
    kind = intermediate_artifacts.json_kind_for_path(path)
    if kind is not None:
        return kind
    return "unknown"


def discover_artifacts(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(value) for value in explicit]
    return [ROOT / value for value in DISCOVERY_PATHS if (ROOT / value).is_file()]


def build_manifest(paths: list[Path]) -> dict[str, object]:
    artifacts = []
    diagnostics = []
    seen_kinds = set()
    for path in paths:
        kind = artifact_kind(path)
        if not path.is_file():
            diagnostics.append({"status": "blocked", "message": f"missing artifact: {path}"})
            continue
        if kind == "unknown":
            diagnostics.append({"status": "blocked", "message": f"unknown artifact schema: {path}"})
        seen_kinds.add(kind)
        artifacts.append(
            {
                "kind": kind,
                "id": kind,
                "path": str(path),
                "producer": "artifact summary command",
                "status": "present",
                "fingerprint": fingerprint(path),
            }
        )

    edges = []
    for left, right in zip(CHAIN, CHAIN[1:]):
        if left in seen_kinds and right in seen_kinds:
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
    output.write_text(json.dumps(build_manifest(paths), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
