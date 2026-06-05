#!/usr/bin/env python3
"""Regression test for the ordered intermediate artifact chain."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import artifact_test_common


EXPECTED_FILES = [
    "old-app-corpus-inventory.csv",
    "app-corpus-import-status.csv",
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "sim-cycle-summary.csv",
    "rtl-fpa-summary.csv",
    "full-stack-artifact-manifest.json",
    "e2e-demonstrator-summary.csv",
    "dse-candidate-summary.csv",
    "unsupported-scope-ledger.csv",
    "artifact-audit-summary.json",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_legacy_case(root: Path, name: str) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.cpp").write_text("int main() { return 0; }\n")
    (case_dir / f"{name}.cpp").write_text(f'#include "{name}.h"\n')
    (case_dir / f"{name}.h").write_text("#pragma once\n")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-artifact-chain-") as tmp:
        out_dir = Path(tmp)
        legacy_root = out_dir / "legacy-app"
        write_legacy_case(legacy_root, "legacy_missing")
        write_legacy_case(legacy_root, "vecadd")
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--legacy-app-root",
                str(legacy_root),
            ],
            "intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing chain artifacts: {missing}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        vecadd_rows = [row for row in sim_rows if row["kernel"] == "vecadd"]
        if len(vecadd_rows) != 1:
            raise AssertionError(f"expected one vecadd sim row, got {sim_rows}")
        if vecadd_rows[0]["dfg_sim_cycles"] != "" or vecadd_rows[0]["cgra_sim_cycles"] != "":
            raise AssertionError(f"blocked sim row must not fake cycles: {vecadd_rows[0]}")

        import_rows = read_csv_rows(out_dir / "app-corpus-import-status.csv")
        states = {row["case"]: row["import_state"] for row in import_rows}
        if states != {"legacy_missing": "deferred", "vecadd": "accepted"}:
            raise AssertionError(f"unexpected app import states: {import_rows}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected chain audit pass, got {audit}")
        reviewed = {Path(review["artifact"]).name for review in audit.get("artifact_reviews", [])}
        expected_reviewed = set(EXPECTED_FILES) - {"artifact-audit-summary.json"}
        if reviewed != expected_reviewed:
            raise AssertionError(f"audit reviewed {reviewed}, expected {expected_reviewed}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"chain should not have cross-artifact findings: {audit}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        manifest_artifacts = {Path(artifact["path"]).name for artifact in manifest.get("artifacts", [])}
        expected_manifest = set(EXPECTED_FILES) - {
            "artifact-audit-summary.json",
            "full-stack-artifact-manifest.json",
        }
        if manifest_artifacts != expected_manifest:
            raise AssertionError(f"manifest artifacts {manifest_artifacts} do not match {expected_manifest}")
        edges = {(edge["from"], edge["to"]) for edge in manifest.get("edges", [])}
        if ("e2e_demonstrator", "dse_candidate") in edges:
            raise AssertionError(f"manifest should not imply demonstrator feeds DSE: {edges}")
        required_edges = {
            ("old_app_corpus_inventory", "app_import_status"),
            ("app_import_status", "source_compat"),
            ("source_compat", "compiler_pipeline"),
            ("pnr_mapping", "e2e_demonstrator"),
            ("pnr_mapping", "dse_candidate"),
            ("dse_candidate", "unsupported_scope"),
        }
        if not required_edges.issubset(edges):
            raise AssertionError(f"manifest edges {edges} missing {required_edges - edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
