#!/usr/bin/env python3
"""Anchor tests for exact corpus TechMapping coverage."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))

import corpus_tech_mapping  # noqa: E402


def _write_source_gate(root: Path) -> Path:
    digest = "0123456789abcdef"
    dfg = root / "loombench" / "axpy" / digest / "program.dfg.mlir"
    dfg.parent.mkdir(parents=True)
    dfg.write_text("module {}\n")
    summary = {
        "case_count": 2,
        "cases": [
            {
                "actors": 2,
                "case": "axpy",
                "category": None,
                "graphs": 1,
                "identity": f"loombench:axpy:{digest}",
                "status": "pass",
                "suite": "loombench",
            },
            {
                "case": "neon-only",
                "category": "target-profile-unsupported",
                "detail": "exact target profile is unavailable",
                "identity": "cmsis-dsp:neon-only:fedcba9876543210",
                "status": "unsupported",
                "suite": "cmsis-dsp",
            },
        ],
        "failed": 0,
        "passed": 1,
        "stage": "dfg-sim",
        "unsupported": 1,
        "unsupported_categories": {"target-profile-unsupported": 1},
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary))
    return path


def _generated_report(
    root_identities: dict[str, str], actor_count: int = 2
) -> dict[str, object]:
    return {
        "actor_count": actor_count,
        "canonical_dataflow": "d" * 64,
        "fabrics": [
            {
                "candidate_count": 1,
                "classification": "pending-spatial-capacity",
                "covered_actor_count": actor_count,
                "fabric": value[::-1],
                "generation_cpu_seconds": 0.01,
                "generation_wall_seconds": 0.02,
                "input_fabric_root": value,
                "status": "generated",
            }
            for value in root_identities.values()
        ],
        "graph_count": 1,
        "kind": "tech_mapping_coverage",
    }


class SourceGateTest(unittest.TestCase):
    def test_exact_pass_and_unsupported_partition_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="loom-tech-coverage-") as raw:
            summary = _write_source_gate(Path(raw))

            gate = corpus_tech_mapping.load_source_gate(summary)

            self.assertEqual(
                [row.identity for row in gate.workloads],
                ["loombench:axpy:0123456789abcdef"],
            )
            self.assertEqual(len(gate.unsupported), 1)
            self.assertEqual(gate.workloads[0].actors, 2)
            self.assertTrue(gate.workloads[0].program.is_file())

    def test_missing_or_extra_dfg_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="loom-tech-coverage-") as raw:
            root = Path(raw)
            summary = _write_source_gate(root)
            expected = next(root.rglob("program.dfg.mlir"))
            expected.unlink()
            with self.assertRaisesRegex(
                corpus_tech_mapping.CoverageError, "missing Canonical Dataflow"
            ):
                corpus_tech_mapping.load_source_gate(summary)

            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_text("module {}\n")
            extra = root / "loombench" / "other" / "bad" / "program.dfg.mlir"
            extra.parent.mkdir(parents=True)
            extra.write_text("module {}\n")
            with self.assertRaisesRegex(
                corpus_tech_mapping.CoverageError, "unexpected Canonical Dataflow"
            ):
                corpus_tech_mapping.load_source_gate(summary)

    def test_non_target_unsupported_or_failed_source_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="loom-tech-coverage-") as raw:
            summary_path = _write_source_gate(Path(raw))
            summary = json.loads(summary_path.read_text())
            summary["cases"][1]["category"] = "dfg-sim"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(
                corpus_tech_mapping.CoverageError, "unsupported category"
            ):
                corpus_tech_mapping.load_source_gate(summary_path)


class CoverageAccountingTest(unittest.TestCase):
    def test_worker_limit_reserves_four_cpus_and_caps_at_120(self) -> None:
        self.assertEqual(corpus_tech_mapping.default_worker_count(32), 28)
        self.assertEqual(corpus_tech_mapping.default_worker_count(256), 120)
        self.assertEqual(corpus_tech_mapping.default_worker_count(4), 1)

    def test_process_watchdog_covers_every_atomic_builtin_invocation(self) -> None:
        required = (
            len(corpus_tech_mapping.BUILTIN_PRESETS)
            * corpus_tech_mapping.MAX_GENERATION_WALL_SECONDS
        )
        self.assertGreaterEqual(
            corpus_tech_mapping.DEFAULT_CASE_TIMEOUT_SECONDS, required
        )

    def test_workload_run_cannot_reuse_a_stale_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="loom-tech-coverage-") as raw:
            root = Path(raw)
            output_root = root / "out"
            identity = "loombench:axpy:0123456789abcdef"
            stale_report = (
                output_root
                / "cases"
                / "loombench"
                / "axpy"
                / "0123456789abcdef"
                / "tech-mapping.json"
            )
            stale_report.parent.mkdir(parents=True)
            roots = {"small": "1" * 64, "default": "2" * 64, "large": "3" * 64}
            stale_report.write_text(json.dumps(_generated_report(roots)))
            tool = root / "no-report"
            tool.write_text("#!/bin/sh\nexit 2\n")
            tool.chmod(0o755)
            program = root / "program.dfg.mlir"
            program.write_text("module {}\n")

            result = corpus_tech_mapping.run_workload(
                corpus_tech_mapping.WorkloadInput(
                    identity=identity,
                    suite="loombench",
                    case="axpy",
                    graphs=1,
                    actors=2,
                    program=program,
                ),
                loom_tech_map=tool,
                artifact_store=root / "store",
                output_root=output_root,
                reference_paths={
                    preset: root / f"{preset}.ref"
                    for preset in corpus_tech_mapping.BUILTIN_PRESETS
                },
                root_identities=roots,
                timeout_seconds=corpus_tech_mapping.DEFAULT_CASE_TIMEOUT_SECONDS,
            )

            self.assertEqual(result["status"], "failed")
            self.assertIn("cannot read JSON", result["detail"])

    def test_exact_builtin_roots_and_complete_actor_cover_are_required(self) -> None:
        roots = {"small": "1" * 64, "default": "2" * 64, "large": "3" * 64}
        report = _generated_report(roots)

        normalized = corpus_tech_mapping.validate_tool_report(
            "loombench:axpy:0123456789abcdef", 1, 2, report, roots
        )

        self.assertEqual(normalized["canonical_dataflow"], "d" * 64)
        self.assertEqual(set(normalized["fabrics"]), set(roots))

        report["fabrics"][0]["covered_actor_count"] = 1
        with self.assertRaisesRegex(corpus_tech_mapping.CoverageError, "actor cover"):
            corpus_tech_mapping.validate_tool_report(
                "loombench:axpy:0123456789abcdef", 1, 2, report, roots
            )

    def test_rejection_is_reported_but_cannot_pass_the_coverage_gate(self) -> None:
        roots = {"small": "1" * 64, "default": "2" * 64, "large": "3" * 64}
        report = _generated_report(roots)
        report["fabrics"][2] = {
            "classification": "capability-rejected",
            "fabric": "c" * 64,
            "generation_cpu_seconds": 0.01,
            "generation_wall_seconds": 0.02,
            "input_fabric_root": roots["large"],
            "reason": "no-complete-exact-cover",
            "status": "proven-infeasible",
        }
        normalized = corpus_tech_mapping.validate_tool_report(
            "loombench:axpy:0123456789abcdef", 1, 2, report, roots
        )

        summary = corpus_tech_mapping.summarize_results(
            expected_identities=["loombench:axpy:0123456789abcdef"],
            results={"loombench:axpy:0123456789abcdef": normalized},
            root_identities=roots,
        )

        self.assertEqual(summary["anti_join"], {"extra": [], "missing": []})
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["builtins"]["large"]["capability_rejected"], 1)

    def test_semantic_limit_is_preserved_as_incomplete(self) -> None:
        roots = {"small": "1" * 64, "default": "2" * 64, "large": "3" * 64}
        report = _generated_report(roots)
        report["fabrics"][0] = {
            "accounting": {
                "match_row_attempts": 65536,
                "partial_cover_expansions": 7,
                "publication_slots": 0,
            },
            "classification": "incomplete",
            "fabric": "a" * 64,
            "generation_cpu_seconds": 0.20,
            "generation_wall_seconds": 0.25,
            "input_fabric_root": roots["small"],
            "reason": "proof-not-established",
            "status": "incomplete",
        }

        normalized = corpus_tech_mapping.validate_tool_report(
            "loombench:axpy:0123456789abcdef", 1, 2, report, roots
        )
        summary = corpus_tech_mapping.summarize_results(
            expected_identities=["loombench:axpy:0123456789abcdef"],
            results={"loombench:axpy:0123456789abcdef": normalized},
            root_identities=roots,
        )

        self.assertEqual(summary["builtins"]["small"]["incomplete"], 1)
        self.assertFalse(summary["passed"])

    def test_cpu_target_and_wall_limit_are_independent(self) -> None:
        roots = {"small": "1" * 64, "default": "2" * 64, "large": "3" * 64}
        identity = "loombench:axpy:0123456789abcdef"
        report = _generated_report(roots)

        normalized = corpus_tech_mapping.validate_tool_report(
            identity, 1, 2, report, roots
        )
        summary = corpus_tech_mapping.summarize_results(
            expected_identities=[identity],
            results={identity: normalized},
            root_identities=roots,
        )
        self.assertTrue(summary["performance"]["passed"])

        report["fabrics"][0]["generation_wall_seconds"] = (
            corpus_tech_mapping.MAX_GENERATION_WALL_SECONDS + 0.01
        )
        normalized = corpus_tech_mapping.validate_tool_report(
            identity, 1, 2, report, roots
        )
        summary = corpus_tech_mapping.summarize_results(
            expected_identities=[identity],
            results={identity: normalized},
            root_identities=roots,
        )
        self.assertFalse(summary["performance"]["passed"])

        report["fabrics"][0]["generation_wall_seconds"] = 0.02
        report["fabrics"][0]["generation_cpu_seconds"] = (
            corpus_tech_mapping.P95_GENERATION_CPU_TARGET_SECONDS + 0.01
        )
        normalized = corpus_tech_mapping.validate_tool_report(
            identity, 1, 2, report, roots
        )
        summary = corpus_tech_mapping.summarize_results(
            expected_identities=[identity],
            results={identity: normalized},
            root_identities=roots,
        )
        self.assertFalse(summary["performance"]["passed"])

    def test_small_graph_cpu_target_does_not_replace_large_graph_wall_limit(
        self,
    ) -> None:
        roots = {"small": "1" * 64, "default": "2" * 64, "large": "3" * 64}
        small_identity = "loombench:small:0123456789abcdef"
        large_identity = "loombench:large:fedcba9876543210"
        small = corpus_tech_mapping.validate_tool_report(
            small_identity, 1, 2, _generated_report(roots), roots
        )
        large_actor_count = corpus_tech_mapping.SMALL_GRAPH_MAX_ACTORS + 1
        large_report = _generated_report(roots, large_actor_count)
        for fabric in large_report["fabrics"]:
            fabric["generation_cpu_seconds"] = (
                corpus_tech_mapping.P95_GENERATION_CPU_TARGET_SECONDS + 1.0
            )
        large = corpus_tech_mapping.validate_tool_report(
            large_identity, 1, large_actor_count, large_report, roots
        )
        summary = corpus_tech_mapping.summarize_results(
            expected_identities=[small_identity, large_identity],
            results={small_identity: small, large_identity: large},
            root_identities=roots,
        )
        self.assertTrue(summary["performance"]["passed"])
        self.assertEqual(
            summary["builtins"]["small"]["small_graph_generation_cpu_seconds"]["count"],
            1,
        )

        large_report["fabrics"][0]["generation_wall_seconds"] = (
            corpus_tech_mapping.MAX_GENERATION_WALL_SECONDS + 0.01
        )
        large = corpus_tech_mapping.validate_tool_report(
            large_identity, 1, large_actor_count, large_report, roots
        )
        summary = corpus_tech_mapping.summarize_results(
            expected_identities=[small_identity, large_identity],
            results={small_identity: small, large_identity: large},
            root_identities=roots,
        )
        self.assertFalse(summary["performance"]["passed"])


if __name__ == "__main__":
    unittest.main()
