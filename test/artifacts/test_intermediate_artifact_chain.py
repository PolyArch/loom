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
    "cmsis-compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "pnr-mapping.json",
    "vecsum-dfg-sim-report.json",
    "vecsum-cgra-sim-report.json",
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

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        vecsum_mapping_rows = [row for row in mapping_rows if row["workload"] == "vecsum"]
        if len(vecsum_mapping_rows) != 1:
            raise AssertionError(f"expected one vecsum mapping row, got {mapping_rows}")
        if (
            vecsum_mapping_rows[0]["mapping_id"] != "vecsum__shared_reduction_adg"
            or vecsum_mapping_rows[0]["placed_records"] != "5"
            or vecsum_mapping_rows[0]["routed_edges"] != "6"
            or vecsum_mapping_rows[0].get("status") != "pass"
        ):
            raise AssertionError(f"expected real vecsum mapping evidence: {vecsum_mapping_rows[0]}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        vecsum_rows = [row for row in sim_rows if row["kernel"] == "vecsum"]
        if len(vecsum_rows) != 1:
            raise AssertionError(f"expected one vecsum sim row, got {sim_rows}")
        if (
            vecsum_rows[0]["dfg_sim_cycles"] == ""
            or vecsum_rows[0]["cgra_sim_cycles"] == ""
            or vecsum_rows[0].get("status") != "pass"
        ):
            raise AssertionError(f"expected real vecsum simulator cycle evidence: {vecsum_rows[0]}")
        if int(vecsum_rows[0]["cgra_sim_cycles"]) < int(vecsum_rows[0]["dfg_sim_cycles"]):
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {vecsum_rows[0]}")

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
        if ("e2e-demonstrator-summary", "dse-candidate-summary") in edges:
            raise AssertionError(f"manifest should not imply demonstrator feeds DSE: {edges}")
        required_edges = {
            ("old-app-corpus-inventory", "app-corpus-import-status"),
            ("app-corpus-import-status", "source-compat-summary"),
            ("source-compat-summary", "compiler-pipeline-summary"),
            ("pnr-mapping-summary", "e2e-demonstrator-summary"),
            ("pnr-mapping-summary", "dse-candidate-summary"),
            ("dse-candidate-summary", "unsupported-scope-ledger"),
        }
        if not required_edges.issubset(edges):
            raise AssertionError(f"manifest edges {edges} missing {required_edges - edges}")
        if ("cmsis-compiler-pipeline-summary", "dataflow-primitive-coverage") in edges:
            raise AssertionError(f"CMSIS pipeline summary must not feed app primitive coverage: {edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
