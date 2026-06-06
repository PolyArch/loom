#!/usr/bin/env python3
"""Regression test for full-stack artifact manifest identity records."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_KEYS = {"schema_version", "run_id", "artifacts", "edges", "diagnostics"}
REQUIRED_KINDS = {
    "source_compat",
    "compiler_pipeline",
    "dataflow_primitive_coverage",
}


def run(repo: Path, argv: list[str]) -> None:
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with {result.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def run_raw(repo: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    temp_root = repo / "temp" / "test-runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="loom-artifact-manifest-", dir=temp_root) as tmp:
        out_dir = Path(tmp)
        source = out_dir / "source-compat-summary.csv"
        pipeline = out_dir / "compiler-pipeline-summary.csv"
        cmsis_pipeline = out_dir / "cmsis-compiler-pipeline-summary.csv"
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        manifest = out_dir / "full-stack-artifact-manifest.json"

        run(repo, ["bash", "test/app/run_source_compat_summary.sh", "--case", "vecadd", "--output", str(source)])
        run(repo, ["bash", "test/app/run_compiler_pipeline_summary.sh", "--case", "vecadd", "--output", str(pipeline)])
        cmsis_pipeline.write_text(pipeline.read_text().replace("vecadd", "cmsis-dsp").replace(",app,", ",CMSIS-DSP,"))
        run(repo, ["bash", "test/dataflow/run_primitive_coverage.sh", "--case", "vecadd", "--output", str(primitive)])
        run(
            repo,
            [
                "bash",
                "test/e2e/run_artifact_manifest.sh",
                "--artifact",
                str(source),
                "--artifact",
                str(pipeline),
                "--artifact",
                str(cmsis_pipeline),
                "--artifact",
                str(primitive),
                "--output",
                str(manifest),
            ],
        )

        data = json.loads(manifest.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"manifest missing keys: {sorted(missing)}")
        if data["schema_version"] != 1:
            raise AssertionError(f"unexpected schema version: {data['schema_version']!r}")
        artifacts = data["artifacts"]
        fingerprints = {
            artifact["id"]: artifact["fingerprint"]
            for artifact in artifacts
        }
        kinds = {artifact["kind"] for artifact in artifacts}
        if kinds != REQUIRED_KINDS:
            raise AssertionError(f"unexpected artifact kinds: {kinds}")
        ids = [artifact["id"] for artifact in artifacts]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"artifact ids must be unique: {ids}")
        if ids != [
            "source-compat-summary",
            "compiler-pipeline-summary",
            "cmsis-compiler-pipeline-summary",
            "dataflow-primitive-coverage",
        ]:
            raise AssertionError(f"unexpected artifact ids: {ids}")
        for artifact in artifacts:
            if len(artifact.get("fingerprint", "")) != 64:
                raise AssertionError(f"missing sha256 fingerprint: {artifact}")
            if artifact.get("status") != "present":
                raise AssertionError(f"artifact should be present: {artifact}")
        edge_pairs = {(edge["from"], edge["to"]) for edge in data["edges"]}
        expected_edges = {
            ("source-compat-summary", "compiler-pipeline-summary"),
            ("compiler-pipeline-summary", "dataflow-primitive-coverage"),
        }
        if edge_pairs != expected_edges:
            raise AssertionError(f"unexpected edges: {edge_pairs}")
        expected_edge_ids = {
            f"edge::{left}->{right}"
            for left, right in expected_edges
        }
        edge_ids = {edge.get("id") for edge in data["edges"]}
        if edge_ids != expected_edge_ids:
            raise AssertionError(f"unexpected edge ids: {edge_ids}")
        expected_edge_kinds = {
            ("source-compat-summary", "compiler-pipeline-summary"): (
                "source_compat",
                "compiler_pipeline",
            ),
            ("compiler-pipeline-summary", "dataflow-primitive-coverage"): (
                "compiler_pipeline",
                "dataflow_primitive_coverage",
            ),
        }
        for edge in data["edges"]:
            key = (edge.get("from"), edge.get("to"))
            expected_kinds = expected_edge_kinds.get(key)
            if expected_kinds is None:
                raise AssertionError(f"unexpected edge for kind check: {edge}")
            if (
                edge.get("producer_artifact_kind"),
                edge.get("consumer_artifact_kind"),
            ) != expected_kinds:
                raise AssertionError(f"edge missed artifact kinds: {edge}")
            expected_components = tuple(f"{kind}-producer" for kind in expected_kinds)
            if (
                edge.get("producer_component"),
                edge.get("consumer_component"),
            ) != expected_components:
                raise AssertionError(f"edge missed producer/consumer components: {edge}")
            left, right = key
            if edge.get("required_input_fingerprints") != {left: fingerprints[left]}:
                raise AssertionError(f"edge missed input fingerprint: {edge}")
            if edge.get("produced_output_fingerprints") != {right: fingerprints[right]}:
                raise AssertionError(f"edge missed output fingerprint: {edge}")
        if data["diagnostics"]:
            raise AssertionError(f"unexpected diagnostics: {data['diagnostics']}")

        left_dir = out_dir / "left"
        right_dir = out_dir / "right"
        left_dir.mkdir()
        right_dir.mkdir()
        duplicate_left = left_dir / "compiler-pipeline-summary.csv"
        duplicate_right = right_dir / "compiler-pipeline-summary.csv"
        duplicate_left.write_text(pipeline.read_text())
        duplicate_right.write_text(pipeline.read_text())
        duplicate_manifest = out_dir / "duplicate-full-stack-artifact-manifest.json"
        result = run_raw(
            repo,
            [
                "bash",
                "test/e2e/run_artifact_manifest.sh",
                "--artifact",
                str(duplicate_left),
                "--artifact",
                str(duplicate_right),
                "--output",
                str(duplicate_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("duplicate artifact id unexpectedly produced a passing manifest")
        duplicate_data = json.loads(duplicate_manifest.read_text())
        if not duplicate_data.get("diagnostics"):
            raise AssertionError(f"duplicate manifest should record diagnostics: {duplicate_data}")
        audit = out_dir / "duplicate-artifact-audit-summary.json"
        result = run_raw(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(duplicate_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("manifest with blocked diagnostics unexpectedly passed audit")

        dangling_manifest = out_dir / "dangling-full-stack-artifact-manifest.json"
        dangling_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "dangling-edge",
                    "artifacts": [
                        {
                            "kind": "source_compat",
                            "id": "source-compat-summary",
                            "path": str(source),
                            "producer": "artifact summary command",
                            "status": "present",
                        }
                    ],
                    "edges": [
                        {
                            "from": "source-compat-summary",
                            "to": "missing-artifact",
                            "kind": "producer-consumer",
                        }
                    ],
                    "diagnostics": [],
                }
            )
        )
        dangling_audit = out_dir / "dangling-artifact-audit-summary.json"
        result = run_raw(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(dangling_audit),
                str(dangling_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("manifest with dangling edge unexpectedly passed audit")

        missing_edge_id_manifest = out_dir / "missing-edge-id-full-stack-artifact-manifest.json"
        missing_edge_id_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "missing-edge-id",
                    "artifacts": [
                        {
                            "kind": "source_compat",
                            "id": "source-compat-summary",
                            "path": str(source),
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "compiler_pipeline",
                            "id": "compiler-pipeline-summary",
                            "path": str(pipeline),
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                    ],
                    "edges": [
                        {
                            "from": "source-compat-summary",
                            "to": "compiler-pipeline-summary",
                            "kind": "producer-consumer",
                        }
                    ],
                    "diagnostics": [],
                }
            )
        )
        missing_edge_id_audit = out_dir / "missing-edge-id-artifact-audit-summary.json"
        result = run_raw(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_edge_id_audit),
                str(missing_edge_id_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("manifest edge without edge id unexpectedly passed audit")

        disconnected_manifest = out_dir / "disconnected-full-stack-artifact-manifest.json"
        disconnected_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "missing-trace-edges",
                    "artifacts": [
                        {
                            "kind": "dataflow_primitive_coverage",
                            "id": "dataflow-primitive-coverage",
                            "path": "dataflow-primitive-coverage.csv",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "adg_hardware",
                            "id": "adg-hardware-summary",
                            "path": "adg-hardware-summary.csv",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "pnr_mapping",
                            "id": "pnr-mapping-summary",
                            "path": "pnr-mapping-summary.csv",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "pnr_mapping_artifact",
                            "id": "pnr-mapping",
                            "path": "pnr-mapping.json",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "dfg_sim_report",
                            "id": "vecsum-dfg-sim-report",
                            "path": "vecsum-dfg-sim-report.json",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "cgra_sim_report",
                            "id": "vecsum-cgra-sim-report",
                            "path": "vecsum-cgra-sim-report.json",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "sim_cycle",
                            "id": "sim-cycle-summary",
                            "path": "sim-cycle-summary.csv",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                        {
                            "kind": "dse_candidate",
                            "id": "dse-candidate-summary",
                            "path": "dse-candidate-summary.csv",
                            "producer": "artifact summary command",
                            "status": "present",
                        },
                    ],
                    "edges": [],
                    "diagnostics": [],
                }
            )
        )
        disconnected_audit = out_dir / "disconnected-artifact-audit-summary.json"
        result = run_raw(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(disconnected_audit),
                str(disconnected_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("manifest with disconnected trace artifacts unexpectedly passed audit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
