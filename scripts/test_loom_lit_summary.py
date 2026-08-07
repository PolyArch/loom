#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    from loom_lit_summary import (
        run_lit_with_unsupported_summary,
        unsupported_summary,
    )
except ModuleNotFoundError:
    from scripts.loom_lit_summary import (
        run_lit_with_unsupported_summary,
        unsupported_summary,
    )


class LoomLitSummaryTest(unittest.TestCase):
    def test_groups_unsupported_tests_by_lit_reason(self) -> None:
        tests = [{"code": "PASS", "output": ""} for _ in range(382)]
        tests.extend(
            {
                "code": "UNSUPPORTED",
                "output": (
                    "Test requires the following unavailable features: "
                    + ("yosys, verilator" if index == 0 else "verilator, yosys")
                ),
            }
            for index in range(19)
        )
        tests.append(
            {
                "code": "UNSUPPORTED",
                "output": (
                    "Test requires the following unavailable features: yosys"
                ),
            }
        )

        self.assertEqual(
            unsupported_summary({"tests": tests}),
            (
                "Unsupported (verilator, yosys): 19 (4.73%)",
                "Unsupported (yosys): 1 (0.25%)",
            ),
        )

    def test_omits_summary_when_every_test_ran(self) -> None:
        self.assertEqual(
            unsupported_summary({"tests": [{"code": "PASS", "output": ""}]}),
            (),
        )

    def test_runs_lit_once_and_removes_internal_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_root = root / "test"
            test_root.mkdir()
            calls = []

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                report_path = Path(command[command.index("--output") + 1])
                report_path.write_text(
                    json.dumps(
                        {
                            "tests": [
                                {
                                    "code": "UNSUPPORTED",
                                    "output": (
                                        "Test requires the following unavailable "
                                        "features: yosys"
                                    ),
                                }
                            ]
                        }
                    )
                )

            output = io.StringIO()
            with redirect_stdout(output):
                run_lit_with_unsupported_summary(
                    Path("llvm-lit"), test_root, 7, (), {"PATH": "/bin"}, runner
                )

            self.assertEqual(len(calls), 1)
            command, kwargs = calls[0]
            report_path = Path(command[command.index("--output") + 1])
            self.assertFalse(report_path.exists())
            self.assertEqual(command[-1], str(test_root))
            self.assertEqual(kwargs, {"env": {"PATH": "/bin"}})
            self.assertEqual(output.getvalue(), "Unsupported (yosys): 1 (100.00%)\n")

    def test_preserves_explicit_lit_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_root = root / "test"
            test_root.mkdir()
            report_path = root / "requested.json"

            def runner(command, **kwargs):
                self.assertEqual(command.count("--output"), 1)
                self.assertEqual(command[command.index("--output") + 1], str(report_path))
                report_path.write_text(json.dumps({"tests": [{"code": "PASS"}]}))

            run_lit_with_unsupported_summary(
                Path("llvm-lit"),
                test_root,
                3,
                ("--output", str(report_path)),
                {},
                runner,
            )

            self.assertTrue(report_path.exists())

    def test_preserves_compact_explicit_lit_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_root = root / "test"
            test_root.mkdir()
            report_path = root / "requested.json"

            def runner(command, **kwargs):
                self.assertNotIn("--output", command)
                report_path.write_text(json.dumps({"tests": [{"code": "PASS"}]}))

            run_lit_with_unsupported_summary(
                Path("llvm-lit"),
                test_root,
                3,
                (f"-o{report_path}",),
                {},
                runner,
            )

            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
