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


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-artifact-manifest-") as tmp:
        out_dir = Path(tmp)
        source = out_dir / "source-compat-summary.csv"
        pipeline = out_dir / "compiler-pipeline-summary.csv"
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        manifest = out_dir / "full-stack-artifact-manifest.json"

        run(repo, ["bash", "test/app/run_source_compat_summary.sh", "--case", "vecadd", "--output", str(source)])
        run(repo, ["bash", "test/app/run_compiler_pipeline_summary.sh", "--case", "vecadd", "--output", str(pipeline)])
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
        kinds = {artifact["kind"] for artifact in artifacts}
        if kinds != REQUIRED_KINDS:
            raise AssertionError(f"unexpected artifact kinds: {kinds}")
        for artifact in artifacts:
            if len(artifact.get("fingerprint", "")) != 64:
                raise AssertionError(f"missing sha256 fingerprint: {artifact}")
            if artifact.get("status") != "present":
                raise AssertionError(f"artifact should be present: {artifact}")
        edge_pairs = {(edge["from"], edge["to"]) for edge in data["edges"]}
        expected_edges = {
            ("source_compat", "compiler_pipeline"),
            ("compiler_pipeline", "dataflow_primitive_coverage"),
        }
        if edge_pairs != expected_edges:
            raise AssertionError(f"unexpected edges: {edge_pairs}")
        if data["diagnostics"]:
            raise AssertionError(f"unexpected diagnostics: {data['diagnostics']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
