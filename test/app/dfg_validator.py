#!/usr/bin/env python3
"""Check that MLIR contains the app's expected launched dataflow graph."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence


DFG_DEFINITION = re.compile(
    r"^\s*dataflow\.(?:thread(?=\s)|graph\.func(?=\s))"
    r"\s+(?:private\s+)?@([^\s(]+)[^\n]*\{\s*$",
    re.MULTILINE,
)
GRAPH_DEFINITION = re.compile(
    r"^\s*dataflow\.graph\.func\s+(?:private\s+)?@([^\s(]+)[^\n]*\{\s*$",
    re.MULTILINE,
)
GRAPH_LAUNCH = re.compile(r"\bdataflow\.graph\.launch\s+@([^\s(]+)")


class DFGValidationError(ValueError):
    """Raised when an MLIR artifact lacks the expected app DFG anchors."""


def mask_line_comments(text: str) -> str:
    chars = list(text)
    in_string = False
    escaped = False
    index = 0
    while index < len(chars):
        char = chars[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char == "/" and index + 1 < len(chars) and chars[index + 1] == "/":
            while index < len(chars) and chars[index] != "\n":
                chars[index] = " "
                index += 1
            continue
        index += 1
    return "".join(chars)


def validate_text(text: str, symbol: str | None = None) -> None:
    scan_text = mask_line_comments(text)
    definitions = DFG_DEFINITION.findall(scan_text)
    if not definitions:
        raise DFGValidationError(
            "has no dataflow.thread or dataflow.graph.func definition with a body"
        )
    if symbol and not any(symbol in definition for definition in definitions):
        raise DFGValidationError(f"has no dataflow definition for {symbol}")

    launches = GRAPH_LAUNCH.findall(scan_text)
    if not launches:
        raise DFGValidationError("has no dataflow.graph.launch")
    if symbol and not any(symbol in target for target in launches):
        raise DFGValidationError(f"has no launched dataflow graph for {symbol}")

    definitions = set(GRAPH_DEFINITION.findall(scan_text))
    for target in launches:
        if target not in definitions:
            raise DFGValidationError(f"launches undefined graph @{target}")


def validate_file(path: Path, symbol: str | None = None) -> None:
    try:
        text = path.read_text()
    except OSError as exc:
        raise DFGValidationError(f"cannot read {path}: {exc}") from exc
    validate_text(text, symbol)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--symbol")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    try:
        validate_file(args.input, args.symbol)
    except DFGValidationError as exc:
        print(f"{args.input}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
