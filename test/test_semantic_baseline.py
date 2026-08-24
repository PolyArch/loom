#!/usr/bin/env python3
"""Anchor tests for the repository semantic baseline driver."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_gate  # noqa: E402
import corpus_inventory  # noqa: E402
import corpus_target_profile  # noqa: E402
import semantic_baseline  # noqa: E402


def _dfg_projection(seed: int) -> dict[str, object]:
    digit = format(seed % 16, "x")
    return {
        "artifacts": {
            "canonical_dataflow": digit * 64,
            "simulation_runtime_input": "a" * 64,
            "simulation_workload": "b" * 64,
        },
        "dynamic_calls": 1,
        "dse_execution": {
            "search_complete": True,
            "generate_invocations": 5,
            "generate_lineage_edges": 24,
            "incomplete_generate_invocations": 0,
            "input_artifacts": 54,
            "input_bindings": 10,
            "output_artifacts": 69,
            "output_bindings": 6,
            "plan_executions": 2,
        },
        "event_count": 7,
        "execution_terminal": "retired",
        "floating_variance_bytes": 0,
        "floating_variance_kind": "none",
        "memory_bytes_compared": 64,
        "operation_firings": {"arith.addi": 1},
        "selected_source_files": ["source.c"],
        "simulation_seconds": 0.001,
        "value_lanes_compared": 1,
        "wavefront_steps": 3,
        "wavefront_steps_per_second": 3000.0,
    }


def _valid_summary() -> tuple[dict[str, object], tuple[object, ...]]:
    workloads = corpus_inventory.load_workload_inventory(ROOT)
    cases: list[dict[str, object]] = []
    passed = 0
    unsupported = 0
    for index, workload in enumerate(workloads):
        profile = corpus_target_profile.resolve_target_profile(
            workload.suite,
            workload.target_profile,
            corpus_gate.TARGET_TRIPLE,
        )
        common: dict[str, object] = {
            "case": workload.case,
            "cpu_seconds": 0.01,
            "duration_seconds": 0.02,
            "identity": workload.identity,
            "peak_resident_bytes": 4096,
            "sources": len(workload.sources),
            "suite": workload.suite,
        }
        if (
            profile.disposition
            is corpus_target_profile.TargetProfileDisposition.RUNNABLE
        ):
            common.update(
                {
                    "actors": 2,
                    "category": None,
                    "detail": None,
                    "dfg_simulation": _dfg_projection(index),
                    "graphs": 1,
                    "selected_sources": ["source.c"],
                    "status": "pass",
                }
            )
            passed += 1
        else:
            common.update(
                {
                    "category": corpus_gate.CATEGORY_TARGET_PROFILE_UNSUPPORTED,
                    "detail": profile.detail,
                    "status": "unsupported",
                }
            )
            unsupported += 1
        cases.append(common)
    suite_counts: dict[str, dict[str, int]] = {}
    for case in cases:
        counts = suite_counts.setdefault(
            str(case["suite"]), {"pass": 0, "unsupported": 0, "fail": 0}
        )
        counts[str(case["status"])] += 1
    return (
        {
            "candidate_jobs": 1,
            "case_count": len(cases),
            "case_timeout_seconds": semantic_baseline.CASE_WALL_LIMIT_SECONDS,
            "cases": cases,
            "cpu_seconds": 1.0,
            "dfg_simulation_timeout_seconds": semantic_baseline.DFG_WALL_LIMIT_SECONDS,
            "duration_seconds": 2.0,
            "failed": 0,
            "failure_categories": {},
            "jobs": 28,
            "passed": passed,
            "peak_resident_bytes": 4096,
            "stage": "dfg-sim",
            "suite_counts": suite_counts,
            "target": {"triple": corpus_gate.TARGET_TRIPLE},
            "unsupported": unsupported,
            "unsupported_categories": {
                corpus_gate.CATEGORY_TARGET_PROFILE_UNSUPPORTED: unsupported
            },
        },
        workloads,
    )


class CorpusSummaryValidationTest(unittest.TestCase):
    def test_exact_manifest_and_target_dispositions_are_accepted(self) -> None:
        summary, workloads = _valid_summary()

        result = semantic_baseline.validate_corpus_summary(summary, workloads)

        self.assertEqual(result["workloads"], 892)
        self.assertEqual(result["passed"], 845)
        self.assertEqual(result["unsupported"], 47)
        self.assertEqual(len(result["unsupported_rows"]), 47)

    def test_target_incompatibility_cannot_be_counted_as_pass(self) -> None:
        summary, workloads = _valid_summary()
        case = next(
            case for case in summary["cases"] if case["status"] == "unsupported"
        )
        case.update(
            {
                "actors": 1,
                "category": None,
                "detail": None,
                "dfg_simulation": _dfg_projection(15),
                "graphs": 1,
                "selected_sources": [case["identity"]],
                "status": "pass",
            }
        )

        with self.assertRaisesRegex(
            semantic_baseline.BaselineError, "exact target disposition"
        ):
            semantic_baseline.validate_corpus_summary(summary, workloads)

    def test_inventory_order_and_identity_are_exact(self) -> None:
        summary, workloads = _valid_summary()
        summary["cases"][0], summary["cases"][1] = (
            summary["cases"][1],
            summary["cases"][0],
        )

        with self.assertRaisesRegex(semantic_baseline.BaselineError, "identity order"):
            semantic_baseline.validate_corpus_summary(summary, workloads)


class ReplayProjectionTest(unittest.TestCase):
    def test_resource_and_rate_noise_do_not_change_semantic_replay(self) -> None:
        first, workloads = _valid_summary()
        selected_identities = tuple(workload.identity for workload in workloads[:4])
        first["cases"] = first["cases"][:4]
        second = copy.deepcopy(first)
        second["duration_seconds"] = 9.0
        second["cpu_seconds"] = 7.0
        for case in second["cases"]:
            case["duration_seconds"] = 4.0
            case["cpu_seconds"] = 3.0
            case["peak_resident_bytes"] = 8192
            if "dfg_simulation" in case:
                case["dfg_simulation"]["simulation_seconds"] = 0.5
                case["dfg_simulation"]["wavefront_steps_per_second"] = 6.0

        semantic_baseline.compare_replays(first, second, selected_identities)

        second["cases"][0]["dfg_simulation"]["artifacts"]["canonical_dataflow"] = (
            "f" * 64
        )
        with self.assertRaisesRegex(
            semantic_baseline.BaselineError, "semantic replay differs"
        ):
            semantic_baseline.compare_replays(first, second, selected_identities)


class BuiltinExportValidationTest(unittest.TestCase):
    def test_static_system_and_spatial_views_are_required(self) -> None:
        with tempfile.TemporaryDirectory(prefix="loom-baseline-export-") as root:
            base = Path(root) / "small"
            base.with_suffix(".mlir").write_text(
                "module { fabric.system @small { fabric.system.acc_core } }\n"
            )
            base.with_suffix(".html").write_text(
                '<html data-layout-engine="loom-layered-v1">'
                '<svg data-view-kind="system"></svg>'
                '<svg data-view-kind="spatial-core"></svg>'
                '<g data-entity-kind="fabric.acc_core_occurrence"></g>'
                "</html>\n"
            )

            result = semantic_baseline.validate_builtin_export("a" * 64, base)
            self.assertEqual(result["identity"], "a" * 64)

            base.with_suffix(".html").write_text(
                base.with_suffix(".html").read_text() + "dagre.layout()\n"
            )
            with self.assertRaisesRegex(
                semantic_baseline.BaselineError, "browser-side layout"
            ):
                semantic_baseline.validate_builtin_export("a" * 64, base)


class HardwareAnchorValidationTest(unittest.TestCase):
    def test_exact_hardware_anchor_report_is_required(self) -> None:
        report = {
            "anchors": [
                "regular-topology",
                "irregular-directed-topology",
                "heterogeneous-multi-acc-core",
                "temporal-resource-grant",
                "memory-service-forwarding",
            ]
        }

        result = semantic_baseline.validate_hardware_anchor_report(report)

        self.assertEqual(result["anchors"], report["anchors"])

        report["anchors"].pop()
        with self.assertRaisesRegex(
            semantic_baseline.BaselineError, "hardware anchor inventory"
        ):
            semantic_baseline.validate_hardware_anchor_report(report)

    def test_duplicate_or_unknown_hardware_anchors_are_rejected(self) -> None:
        anchors = list(semantic_baseline.HARDWARE_ANCHORS)
        for replacement in (anchors + [anchors[-1]], anchors[:-1] + ["unknown"]):
            with self.assertRaisesRegex(
                semantic_baseline.BaselineError, "hardware anchor inventory"
            ):
                semantic_baseline.validate_hardware_anchor_report(
                    {"anchors": replacement}
                )


if __name__ == "__main__":
    unittest.main()
