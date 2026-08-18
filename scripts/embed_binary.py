#!/usr/bin/env python3
"""Embed one binary file as a generated C++ byte array."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def render(symbol: str, data: bytes) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ValueError("symbol must be a C++ identifier")
    lines = [
        "#ifndef LOOM_GENERATED_FREESTANDING_RUNTIME_BITCODE_H",
        "#define LOOM_GENERATED_FREESTANDING_RUNTIME_BITCODE_H",
        "",
        "#include <cstdint>",
        "",
        "namespace loom::application::detail {",
        "",
        f"inline constexpr std::uint8_t {symbol}[] = {{",
    ]
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        lines.append("    " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    lines.extend(
        [
            "};",
            "",
            "} // namespace loom::application::detail",
            "",
            "#endif // LOOM_GENERATED_FREESTANDING_RUNTIME_BITCODE_H",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    arguments = parser.parse_args()

    rendered = render(arguments.symbol, arguments.input.read_bytes())
    if arguments.output.exists() and arguments.output.read_text() == rendered:
        return
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
