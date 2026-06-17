#!/usr/bin/env python3
"""Load and validate the default shared-ADG CGRA simulator batch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import app_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "test" / "app" / "default-cgra-sim-batch.json"
APP_MANIFEST = ROOT / "test" / "app" / "manifest.json"
INTERMEDIATE_CHAIN = ROOT / "test" / "e2e" / "run_intermediate_artifact_chain.sh"
ALLOWED_HARDWARE = frozenset({"shared_reduction_adg", "shared_vector_alu_adg"})
CASE_LABEL_RE = re.compile(r"^\s*([A-Za-z0-9_.+-]+)\)\s*$")
CASE_GRAPH_RE = re.compile(r'case_graph="([^"]+)"')


def default_manifest_path() -> Path:
    override = os.environ.get("LOOM_DEFAULT_CGRA_SIM_BATCH")
    return Path(override) if override else DEFAULT_MANIFEST


def load_app_dfg_cases(path: Path = APP_MANIFEST) -> set[str]:
    data, diagnostics = app_manifest.validate_manifest(path)
    if diagnostics:
        raise ValueError(f"{path} failed validation: {'; '.join(diagnostics[:3])}")
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path} cases must be a list")
    result: set[str] = set()
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        case = entry.get("case")
        tiers = entry.get("tiers")
        if isinstance(case, str) and isinstance(tiers, list) and "dfg" in tiers:
            result.add(case)
    return result


def load_chain_graph_cases(path: Path = INTERMEDIATE_CHAIN) -> dict[str, str]:
    graph_by_case: dict[str, str] = {}
    current_case: str | None = None
    in_case_dispatch = False
    for line in path.read_text().splitlines():
        if not in_case_dispatch:
            in_case_dispatch = line.strip() == 'case "${CASE}" in'
            continue
        if line.strip() == "esac":
            break
        label = CASE_LABEL_RE.match(line)
        if label is not None:
            current_case = label.group(1)
            continue
        graph = CASE_GRAPH_RE.search(line)
        if current_case is not None and graph is not None:
            graph_by_case[current_case] = graph.group(1)
            current_case = None
    if not graph_by_case:
        raise ValueError(f"{path} does not expose full-stack case wiring")
    return graph_by_case


def validate_case_wiring(
    case: str,
    manifest: Path,
    app_cases: set[str],
    graph_by_case: dict[str, str],
) -> None:
    if case not in app_cases:
        raise ValueError(f"{manifest} contains unknown default batch case {case}")
    graph = graph_by_case.get(case)
    if graph is None:
        raise ValueError(f"{manifest} case {case} is not wired for the full-stack artifact chain")
    if graph == "missing_primary_graph":
        raise ValueError(f"{manifest} case {case} has missing primary graph wiring")


def load_default_batch(path: Path | None = None) -> list[dict[str, str]]:
    manifest = path if path is not None else default_manifest_path()
    data = json.loads(manifest.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{manifest} schema_version must be 1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{manifest} cases must be a non-empty list")

    app_cases = load_app_dfg_cases()
    graph_by_case = load_chain_graph_cases()
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for entry in cases:
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest} cases entries must be objects")
        case = entry.get("case")
        hardware = entry.get("hardware")
        if not isinstance(case, str) or not case:
            raise ValueError(f"{manifest} contains an invalid case name")
        if not isinstance(hardware, str) or not hardware:
            raise ValueError(f"{manifest} case {case} is missing hardware")
        if hardware not in ALLOWED_HARDWARE:
            raise ValueError(f"{manifest} case {case} has unsupported hardware {hardware}")
        if case in seen:
            raise ValueError(f"{manifest} contains duplicate case {case}")
        validate_case_wiring(case, manifest, app_cases, graph_by_case)
        seen.add(case)
        result.append({"case": case, "hardware": hardware})
    return result


def load_default_cases(path: Path | None = None) -> tuple[str, ...]:
    return tuple(entry["case"] for entry in load_default_batch(path))


def load_default_hardware(path: Path | None = None) -> dict[str, str]:
    return {entry["case"]: entry["hardware"] for entry in load_default_batch(path)}


def _read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_evidence_dir(evidence_dir: Path, path: Path | None = None) -> None:
    for case, expected_hardware in load_default_hardware(path).items():
        dfg_path = evidence_dir / f"{case}.dfg.report.json"
        mapping_path = evidence_dir / f"{case}.mapping.json"
        cgra_path = evidence_dir / f"{case}.cgra.report.json"
        comparison_path = evidence_dir / f"{case}.sim-comparison-report.json"
        for label, artifact_path in (
            ("DFG-sim", dfg_path),
            ("mapping", mapping_path),
            ("CGRA-sim", cgra_path),
            ("comparison", comparison_path),
        ):
            if not artifact_path.is_file():
                raise ValueError(f"missing default batch {label} evidence: {artifact_path}")
            data = _read_json(artifact_path)
            if data.get("status") != "pass":
                raise ValueError(
                    f"default batch {label} evidence for {case} has status {data.get('status')!r}"
                )
        mapping = _read_json(mapping_path)
        cgra = _read_json(cgra_path)
        if mapping.get("hardware") != expected_hardware:
            raise ValueError(
                f"{mapping_path} hardware {mapping.get('hardware')!r} does not match manifest {expected_hardware!r}"
            )
        if cgra.get("hardware") != expected_hardware:
            raise ValueError(
                f"{cgra_path} hardware {cgra.get('hardware')!r} does not match manifest {expected_hardware!r}"
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--emit-cases", action="store_true")
    parser.add_argument("--validate-evidence-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = args.manifest
    try:
        if args.emit_cases:
            for case in load_default_cases(manifest):
                print(case)
        if args.validate_evidence_dir is not None:
            validate_evidence_dir(args.validate_evidence_dir, manifest)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
