#!/usr/bin/env python3
"""Reject repository scratch-path references in tracked Loom-owned text files."""

from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


_SCRATCH_COMPONENT = "te" + "mp"
_ESCAPED_SCRATCH_COMPONENT = re.escape(_SCRATCH_COMPONENT)
_DIRECT_PATH_PATTERNS = (
    re.compile(rf"(?<![\w.-]){_ESCAPED_SCRATCH_COMPONENT}(?=[\\/])"),
    re.compile(
        rf'''[\\/]{_ESCAPED_SCRATCH_COMPONENT}(?=$|[\s"'`)\]>;,])'''
    ),
)
_PATHLIB_CONSTRUCTORS = {
    "Path",
    "PosixPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "WindowsPath",
}
_PYTHON_SUFFIXES = {".py", ".pyi"}
_SHELL_SUFFIXES = {".bash", ".sh", ".zsh"}
_LIT_SUFFIXES = {".mlir", ".test"}
_SHELL_PATH_COMMANDS = {"cd", "cp", "mkdir", "mv", "rm", "rmdir", "touch"}
_SHELL_WRAPPERS = {"env", "sudo"}
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_LIT_RUN_PATTERN = re.compile(r"^\s*(?://|#|;)\s*RUN:\s*(.*)$")


def tracked_paths(repo: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"],
    )
    return [repo / os.fsdecode(entry) for entry in output.split(b"\0") if entry]


def direct_path_reference(line: str) -> bool:
    return any(pattern.search(line) for pattern in _DIRECT_PATH_PATTERNS)


def dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def scratch_string_line(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and node.value == _SCRATCH_COMPONENT:
        return node.lineno
    return None


def python_path_reference_lines(text: str, source_path: Path) -> set[int]:
    try:
        tree = ast.parse(text, filename=str(source_path))
    except SyntaxError:
        return set()

    references: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for operand in (node.left, node.right):
                if line_number := scratch_string_line(operand):
                    references.add(line_number)

        if not isinstance(node, ast.Call):
            continue
        call_name = dotted_name(node.func)
        constructor_name = call_name.rsplit(".", 1)[-1] if call_name else None
        is_joinpath_method = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath"
        )
        if not (
            constructor_name in _PATHLIB_CONSTRUCTORS
            or is_joinpath_method
            or call_name == "os.path.join"
        ):
            continue
        for argument in (*node.args, *(item.value for item in node.keywords)):
            if line_number := scratch_string_line(argument):
                references.add(line_number)

    return references


def shell_command_segments(line: str) -> list[list[str]]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token and all(character in ";&|" for character in token):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def shell_command(segment: list[str]) -> tuple[str | None, list[str]]:
    index = 0
    while index < len(segment) and Path(segment[index]).name in _SHELL_WRAPPERS:
        wrapper = Path(segment[index]).name
        index += 1
        if index < len(segment) and segment[index] == "--":
            index += 1
        if wrapper == "env":
            while index < len(segment) and _ENV_ASSIGNMENT.fullmatch(
                segment[index]
            ):
                index += 1

    if index >= len(segment):
        return None, []
    return Path(segment[index]).name, segment[index + 1 :]


def shell_path_reference(line: str) -> bool:
    for segment in shell_command_segments(line):
        command, arguments = shell_command(segment)
        if command in _SHELL_PATH_COMMANDS and _SCRATCH_COMPONENT in arguments:
            return True
    return False


def lit_shell_reference_lines(text: str) -> set[int]:
    references: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _LIT_RUN_PATTERN.match(line)
        if match and shell_path_reference(match.group(1)):
            references.add(line_number)
    return references


def find_scratch_reference_lines(text: str, source_path: Path) -> list[int]:
    references = {
        line_number
        for line_number, line in enumerate(text.splitlines(), start=1)
        if direct_path_reference(line)
    }

    suffix = source_path.suffix.lower()
    if suffix in _PYTHON_SUFFIXES:
        references.update(python_path_reference_lines(text, source_path))
    elif suffix in _SHELL_SUFFIXES:
        references.update(
            line_number
            for line_number, line in enumerate(text.splitlines(), start=1)
            if shell_path_reference(line)
        )
    elif suffix in _LIT_SUFFIXES:
        references.update(lit_shell_reference_lines(text))

    return sorted(references)


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    offenders: list[str] = []

    for path in tracked_paths(repo):
        relative = path.relative_to(repo)
        if relative == Path(".gitignore") or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        for line_number in find_scratch_reference_lines(text, relative):
            offenders.append(
                f"{relative}:{line_number}: {lines[line_number - 1].strip()}"
            )

    if offenders:
        raise AssertionError(
            "tracked Loom-owned scratch path references are not allowed:\n"
            + "\n".join(offenders)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
