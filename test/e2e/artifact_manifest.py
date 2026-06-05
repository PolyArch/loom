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
    ("old_app_corpus_inventory", "app_import_status"),
    ("app_import_status", "source_compat"),
    ("source_compat", "compiler_pipeline"),
    ("compiler_pipeline", "dataflow_primitive_coverage"),
    ("dataflow_primitive_coverage", "pnr_mapping"),
    ("adg_hardware", "pnr_mapping"),
    ("dataflow_primitive_coverage", "sim_cycle"),
    ("dataflow_primitive_coverage", "rtl_fpa"),
    ("adg_hardware", "rtl_fpa"),
    ("pnr_mapping", "e2e_demonstrator"),
    ("sim_cycle", "e2e_demonstrator"),
    ("rtl_fpa", "e2e_demonstrator"),
    ("pnr_mapping", "dse_candidate"),
    ("sim_cycle", "dse_candidate"),
    ("rtl_fpa", "dse_candidate"),
    ("dataflow_primitive_coverage", "unsupported_scope"),
    ("pnr_mapping", "unsupported_scope"),
    ("sim_cycle", "unsupported_scope"),
    ("rtl_fpa", "unsupported_scope"),
    ("e2e_demonstrator", "unsupported_scope"),
    ("dse_candidate", "unsupported_scope"),
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


def discover_artifacts(explicit: list[str]) -> list[Path]:
    return intermediate_artifacts.discover_artifact_paths(
        ROOT,
        explicit,
        include_unsupported_scope=True,
    )


def build_manifest(paths: list[Path]) -> dict[str, object]:
    artifacts = []
    diagnostics = []
    seen_kinds = set()
    for path in paths:
        kind = intermediate_artifacts.artifact_kind_for_path(path)
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
    for left, right in ARTIFACT_EDGES:
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
