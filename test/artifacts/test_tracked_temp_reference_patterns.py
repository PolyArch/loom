#!/usr/bin/env python3
"""Unit tests for tracked repository scratch-path detection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_tracked_temp_references as scanner


SCRATCH_COMPONENT = "te" + "mp"
ROOT = Path(__file__).resolve().parents[2]


class ScratchReferencePatternTest(unittest.TestCase):
    def assert_lines(
        self, source_name: str, text: str, expected_lines: list[int]
    ) -> None:
        self.assertEqual(
            scanner.find_scratch_reference_lines(text, Path(source_name)),
            expected_lines,
        )

    def test_rejects_representative_path_references(self) -> None:
        cases = (
            ("fixture.txt", f"{SCRATCH_COMPONENT}/artifact", [1]),
            (
                "fixture.py",
                f'root / "{SCRATCH_COMPONENT}"\nPath("{SCRATCH_COMPONENT}")',
                [1, 2],
            ),
            (
                "fixture.sh",
                "\n".join(
                    (
                        f"cd {SCRATCH_COMPONENT}",
                        f"rm -rf {SCRATCH_COMPONENT}",
                        f"mkdir {SCRATCH_COMPONENT}",
                    )
                ),
                [1, 2, 3],
            ),
            (
                "fixture.mlir",
                f"// RUN: mkdir {SCRATCH_COMPONENT}",
                [1],
            ),
        )

        for source_name, text, expected_lines in cases:
            with self.subTest(source_name=source_name, text=text):
                self.assert_lines(source_name, text, expected_lines)

    def test_rejects_python_path_api_components(self) -> None:
        cases = (
            f'path.joinpath("{SCRATCH_COMPONENT}")',
            f'os.path.join(root, "{SCRATCH_COMPONENT}", "artifact")',
            f'PosixPath("{SCRATCH_COMPONENT}")',
            f'PurePath("{SCRATCH_COMPONENT}")',
            f'WindowsPath("{SCRATCH_COMPONENT}")',
        )

        for text in cases:
            with self.subTest(text=text):
                self.assert_lines("fixture.py", text, [1])

    def test_rejects_path_commands_and_simple_wrappers(self) -> None:
        shell_text = "\n".join(
            (
                f"cp source {SCRATCH_COMPONENT}",
                f"mv source {SCRATCH_COMPONENT}",
                f"rmdir {SCRATCH_COMPONENT}",
                f"touch {SCRATCH_COMPONENT}",
                f"sudo rm -rf {SCRATCH_COMPONENT}",
                f"env MODE=test mkdir {SCRATCH_COMPONENT}",
            )
        )
        lit_text = f"// RUN: env MODE=test touch {SCRATCH_COMPONENT}"

        self.assert_lines("fixture.sh", shell_text, [1, 2, 3, 4, 5, 6])
        self.assert_lines("fixture.mlir", lit_text, [1])

    def test_allows_command_text_inside_quoted_data(self) -> None:
        text = f'printf "%s\\n" "keep; rm {SCRATCH_COMPONENT} later"'
        self.assert_lines("fixture.sh", text, [])

    def test_allows_non_path_uses(self) -> None:
        cases = (
            ("fixture.py", f"{SCRATCH_COMPONENT}_value = 1"),
            ("fixture.py", f'label = "{SCRATCH_COMPONENT}"'),
            ("fixture.txt", f"rm {SCRATCH_COMPONENT}"),
            ("fixture.mlir", f"// note: cd {SCRATCH_COMPONENT}"),
        )

        for source_name, text in cases:
            with self.subTest(source_name=source_name, text=text):
                self.assert_lines(source_name, text, [])

    def test_scan_skips_gitignore_and_binary_files(self) -> None:
        scratch_root = ROOT / "build" / "test-runs"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="loom-tracked-path-policy-",
            dir=scratch_root,
        ) as tmp:
            repo = Path(tmp)
            gitignore = repo / ".gitignore"
            binary = repo / "fixture.bin"
            source = repo / "fixture.txt"
            gitignore.write_text(f"{SCRATCH_COMPONENT}/ignored\n")
            binary.write_bytes(f"{SCRATCH_COMPONENT}/ignored".encode() + b"\0")
            source.write_text("ordinary text\n")

            with (
                patch.object(
                    scanner,
                    "tracked_paths",
                    return_value=[gitignore, binary, source],
                ),
                patch.object(sys, "argv", ["test", str(repo)]),
            ):
                self.assertEqual(scanner.main(), 0)
                source.write_text(f"{SCRATCH_COMPONENT}/artifact\n")
                with self.assertRaisesRegex(AssertionError, "scratch path"):
                    scanner.main()


if __name__ == "__main__":
    unittest.main()
