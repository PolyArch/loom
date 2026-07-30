#!/usr/bin/env python3
"""Anchor tests for the high-level source corpus inventory."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))
sys.path.insert(0, str(TEST_ROOT / "app"))

import app_manifest  # noqa: E402
import corpus_inventory  # noqa: E402


EXTERNALS_ROOT = corpus_inventory.resolve_externals_root(ROOT)
CMSIS_SUITES = {
    "cmsis-dsp": EXTERNALS_ROOT / "cmsis-dsp",
    "cmsis-nn": EXTERNALS_ROOT / "cmsis-nn",
}


class DualInventoryContractTest(unittest.TestCase):
    def test_operator_gate_uses_one_representative_execution_per_operator(self) -> None:
        workloads = corpus_inventory.load_workload_inventory(ROOT)

        self.assertEqual(
            Counter(workload.suite for workload in workloads),
            {"loombench": 132, "cmsis-dsp": 571, "cmsis-nn": 186},
        )
        self.assertEqual(len(workloads), 889)
        self.assertEqual(
            len({workload.operator_id for workload in workloads}),
            len(workloads),
        )
        self.assertTrue(all(workload.vector_identity for workload in workloads))

    def test_loombench_workloads_are_real_manifest_programs(self) -> None:
        sources = corpus_inventory.load_source_inventory(ROOT)
        workloads = corpus_inventory.load_workload_inventory(ROOT)

        axpy_source = next(row for row in sources if row.identity == "loombench:axpy")
        self.assertEqual(
            axpy_source.sources,
            (
                "test/app/axpy/main_func.cpp",
                "test/app/axpy/main_inline.cpp",
            ),
        )

        axpy_workloads = [
            row for row in workloads if row.suite == "loombench" and row.case == "axpy"
        ]
        self.assertEqual(len(axpy_workloads), 1)
        workload = axpy_workloads[0]
        self.assertEqual(workload.sources, ("test/app/axpy/main_func.cpp",))
        self.assertEqual(workload.entry_symbol, "main")
        self.assertEqual(workload.target_profile, "riscv64-portable-scalar")
        self.assertEqual(workload.oracle.kind, "expected-stdout")
        self.assertEqual(workload.oracle.path, "test/app/axpy/expected.txt")
        self.assertEqual(workload.vector_identity, "axpy:manifest-vector")

    def test_cmsis_workloads_are_upstream_test_owners(self) -> None:
        workloads = corpus_inventory.load_workload_inventory(ROOT)

        dsp = [row for row in workloads if row.suite == "cmsis-dsp"]
        self.assertEqual(len(dsp), 571)
        self.assertTrue(
            all(row.producer.kind == "cmsis-dsp-operator-harness" for row in dsp)
        )
        self.assertEqual(
            {row.oracle.kind for row in dsp},
            {"cmsis-dsp-patterns", "generated-native-reference"},
        )

        nn = [row for row in workloads if row.suite == "cmsis-nn"]
        self.assertEqual(len(nn), 186)
        avgpool = next(
            row
            for row in nn
            if row.operator_id == "cmsis-nn:arm-avgpool-s8:533aaa9d3f64c768"
        )
        self.assertEqual(avgpool.producer.kind, "cmsis-nn-unit-test")
        self.assertEqual(avgpool.producer.target, "test_arm_avgpool_s8")
        self.assertEqual(avgpool.entry_symbol, "main")
        self.assertEqual(avgpool.oracle.kind, "cmsis-nn-unity")
        self.assertEqual(
            avgpool.oracle.path,
            "externals/cmsis-nn/Tests/UnitTest/TestCases/test_arm_avgpool_s8",
        )
        batch_matmul = next(
            row
            for row in nn
            if row.operator_id == "cmsis-nn:arm-batch-matmul-s16:baf58c6a708ee1e0"
        )
        self.assertEqual(batch_matmul.producer.target, "batch_matmul_s16")
        self.assertEqual(
            batch_matmul.executable,
            "workload_baf58c6a708ee1e0",
        )


def git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.decode()


def tracked_translation_units(submodule: Path, revision: str = "HEAD") -> list[str]:
    output = git_output(
        submodule,
        "ls-tree",
        "-r",
        "-z",
        "--name-only",
        revision,
        "--",
        "Source",
    )
    return sorted(
        path
        for path in output.split("\0")
        if path.startswith("Source/")
        and path.endswith(".c")
        and not Path(path).name.startswith("_")
    )


class CorpusInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = corpus_inventory.load_source_inventory(ROOT)

    def test_external_sources_resolve_from_primary_worktree(self) -> None:
        worktrees = git_output(ROOT, "worktree", "list", "--porcelain")
        primary = next(
            Path(line.split(" ", 1)[1]).resolve()
            for line in worktrees.splitlines()
            if line.startswith("worktree ")
        )
        self.assertEqual(
            corpus_inventory.resolve_externals_root(ROOT),
            primary / "externals",
        )

    def test_inventory_matches_all_three_membership_owners(self) -> None:
        manifest, diagnostics = app_manifest.validate_manifest(
            TEST_ROOT / "app" / "manifest.json"
        )
        self.assertEqual(diagnostics, [])
        manifest_cases = manifest["cases"]
        self.assertIsInstance(manifest_cases, list)

        actual_by_suite = {
            suite: [case for case in self.cases if case.suite == suite]
            for suite in corpus_inventory.SUITE_ORDER
        }
        expected_loombench = {
            entry["case"]: tuple(
                f"test/app/{entry['case']}/{source}" for source in entry["sources"]
            )
            for entry in manifest_cases
        }
        self.assertEqual(
            {case.case: case.sources for case in actual_by_suite["loombench"]},
            expected_loombench,
        )

        for suite, submodule in CMSIS_SUITES.items():
            tracked = tracked_translation_units(submodule)
            self.assertEqual(
                [case.case for case in actual_by_suite[suite]],
                [path.removeprefix("Source/") for path in tracked],
            )
            self.assertEqual(
                [case.sources for case in actual_by_suite[suite]],
                [(f"externals/{submodule.name}/{path}",) for path in tracked],
            )

        expected_total = len(expected_loombench) + sum(
            len(tracked_translation_units(submodule))
            for submodule in CMSIS_SUITES.values()
        )
        self.assertEqual(len(self.cases), expected_total)

    def test_cmsis_membership_ignores_staged_index_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Inventory Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "inventory@example.com",
                ],
                check=True,
            )
            source_root = repository / "Source"
            source_root.mkdir()
            committed = source_root / "committed.c"
            committed.write_text("int committed(void) { return 0; }\n")
            private_fragment = source_root / "_fragment.c"
            private_fragment.write_text("FRAGMENT_BODY\n")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "add",
                    "Source/committed.c",
                    "Source/_fragment.c",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-qm", "Add source"],
                check=True,
            )
            pinned_revision = git_output(repository, "rev-parse", "HEAD").strip()

            staged = source_root / "staged.c"
            staged.write_text("int staged(void) { return 0; }\n")
            subprocess.run(
                ["git", "-C", str(repository), "add", "Source/staged.c"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "rm",
                    "--cached",
                    "Source/committed.c",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )

            self.assertEqual(
                corpus_inventory.tracked_c_translation_units_at_revision(
                    repository, pinned_revision
                ),
                ("Source/committed.c",),
            )

    def test_shared_submodule_rejects_tracked_modifications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            superproject = root / "superproject"
            shared_root = root / "shared-externals"
            submodule = shared_root / "cmsis-dsp"
            for repository in (superproject, submodule):
                repository.mkdir(parents=True)
                git_output(repository, "init", "-q")
                git_output(repository, "config", "user.name", "Inventory Test")
                git_output(
                    repository,
                    "config",
                    "user.email",
                    "inventory@example.com",
                )

            source = submodule / "Source" / "kernel.c"
            source.parent.mkdir()
            source.write_text("int kernel(void) { return 0; }\n")
            git_output(submodule, "add", "Source/kernel.c")
            git_output(submodule, "commit", "-qm", "Add source")
            pin = git_output(submodule, "rev-parse", "HEAD").strip()

            git_output(
                superproject,
                "update-index",
                "--add",
                "--cacheinfo",
                "160000",
                pin,
                "externals/cmsis-dsp",
            )
            git_output(superproject, "commit", "-qm", "Pin CMSIS-DSP")

            source.write_text("int kernel(void) { return 1; }\n")
            with self.assertRaisesRegex(
                corpus_inventory.InventoryError, "tracked modifications"
            ):
                corpus_inventory.require_pinned_submodule(
                    superproject,
                    shared_root,
                    Path("externals/cmsis-dsp"),
                )

    def test_loombench_manifest_rejects_duplicate_and_omitted_cases(self) -> None:
        entry = {
            "case": "alpha",
            "language": "c",
            "sources": ["main.c"],
            "expected_stdout": "expected.txt",
            "tiers": ["run"],
            "feature_tags": ["anchor"],
            "compiler_flags": [],
            "link_flags": [],
            "expected_executables": ["alpha"],
        }
        with tempfile.TemporaryDirectory() as directory:
            app_root = Path(directory)
            case_root = app_root / "alpha"
            case_root.mkdir()
            (case_root / "main.c").write_text("int main(void) { return 0; }\n")
            (case_root / "expected.txt").write_text("")
            manifest_path = app_root / "manifest.json"

            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": app_manifest.MANIFEST_SCHEMA_VERSION,
                        "cases": [entry, entry],
                    }
                )
            )
            _, diagnostics = app_manifest.validate_manifest(manifest_path)
            self.assertIn("duplicate case: alpha", diagnostics)

            duplicate_source_entry = {
                **entry,
                "sources": ["main.c", "main.c"],
                "expected_executables": ["alpha", "alpha-copy"],
            }
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": app_manifest.MANIFEST_SCHEMA_VERSION,
                        "cases": [duplicate_source_entry],
                    }
                )
            )
            _, diagnostics = app_manifest.validate_manifest(manifest_path)
            self.assertIn("alpha: duplicate source: main.c", diagnostics)

            unsupported_field_entry = {
                **entry,
                "obsolete_field": "unsupported",
            }
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": app_manifest.MANIFEST_SCHEMA_VERSION,
                        "cases": [unsupported_field_entry],
                    }
                )
            )
            _, diagnostics = app_manifest.validate_manifest(manifest_path)
            self.assertIn("alpha: unsupported fields ['obsolete_field']", diagnostics)

            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": app_manifest.MANIFEST_SCHEMA_VERSION,
                        "cases": [],
                    }
                )
            )
            _, diagnostics = app_manifest.validate_manifest(manifest_path)
            self.assertIn("alpha: existing app case omitted from manifest", diagnostics)

    def test_inventory_emission_rejects_duplicate_source_identity(self) -> None:
        cases = (
            corpus_inventory.SourceTranslationUnit(
                suite="loombench",
                case="alpha",
                sources=("test/app/shared/main.c",),
            ),
            corpus_inventory.SourceTranslationUnit(
                suite="loombench",
                case="beta",
                sources=("test/app/shared/main.c",),
            ),
        )
        with self.assertRaisesRegex(
            corpus_inventory.InventoryError,
            "duplicate corpus source identity: test/app/shared/main.c",
        ):
            corpus_inventory.render_json(
                cases, inventory_kind="source-translation-unit"
            )

    def test_inventory_order_and_json_are_deterministic(self) -> None:
        expected_order = sorted(
            self.cases,
            key=lambda case: (
                corpus_inventory.SUITE_ORDER.index(case.suite),
                case.case,
            ),
        )
        self.assertEqual(list(self.cases), expected_order)
        identities = [case.identity for case in self.cases]
        self.assertEqual(len(identities), len(set(identities)))

        first = corpus_inventory.render_json(
            self.cases, inventory_kind="source-translation-unit"
        )
        second = corpus_inventory.render_json(
            corpus_inventory.load_source_inventory(ROOT),
            inventory_kind="source-translation-unit",
        )
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["inventory_kind"], "source-translation-unit")
        self.assertEqual(payload["case_count"], len(self.cases))
        self.assertEqual(
            sum(payload["suite_counts"].values()),
            payload["case_count"],
        )

    def test_full_suite_and_explicit_case_selection(self) -> None:
        dsp_cases = corpus_inventory.select_rows(
            self.cases,
            suite_names=["cmsis-dsp"],
            case_ids=[],
        )
        self.assertTrue(dsp_cases)
        self.assertEqual({case.suite for case in dsp_cases}, {"cmsis-dsp"})

        requested = [self.cases[0].identity, dsp_cases[0].identity]
        selected = corpus_inventory.select_rows(
            self.cases,
            suite_names=[],
            case_ids=requested,
        )
        self.assertEqual([case.identity for case in selected], requested)

        with self.assertRaisesRegex(
            corpus_inventory.InventoryError, "duplicate case selector"
        ):
            corpus_inventory.select_rows(
                self.cases,
                suite_names=[],
                case_ids=[requested[0], requested[0]],
            )
        with self.assertRaisesRegex(
            corpus_inventory.InventoryError, "unknown case selector"
        ):
            corpus_inventory.select_rows(
                self.cases,
                suite_names=[],
                case_ids=["cmsis-dsp:not/a/source.c"],
            )


if __name__ == "__main__":
    unittest.main()
