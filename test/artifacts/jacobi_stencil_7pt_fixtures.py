#!/usr/bin/env python3
"""Source-derived fixtures for jacobi_stencil_7pt CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class JacobiStencil7ptFixture:
    case: str
    graph: str
    input_arg: int
    interior_arg: int
    index_arg: int
    scalar_args: tuple[tuple[int, str], ...]
    input_values: tuple[float, ...]
    interior_values: tuple[float, ...]
    final_values: tuple[float, ...]
    expected_fire_counts: dict[str, int]

    @property
    def interior_count(self) -> int:
        return len(self.interior_values)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_const(text: str, name: str) -> int:
    match = re.search(
        rf"constexpr\s+uint32_t\s+{name}\s*=\s*(?P<value>[0-9]+)u?\s*;",
        text,
    )
    require(match is not None, f"missing constexpr {name}")
    return int(match.group("value"))


def parse_input(text: str) -> tuple[float, ...]:
    match = re.search(
        r"constexpr\s+std::array<float,\s*kSize>\s+kInput\s*=\s*\{(?P<body>.*?)\};",
        text,
        re.S,
    )
    require(match is not None, "missing kInput array")
    values = []
    for token in re.findall(
        r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?f?",
        match.group("body"),
    ):
        values.append(float(token.rstrip("f")))
    return tuple(values)


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "jacobi_stencil_7pt" / "main_func.cpp"


def fixture_from_source(source: Path | None = None) -> JacobiStencil7ptFixture:
    source = source or source_path()
    text = source.read_text()
    depth = parse_const(text, "kDepth")
    rows = parse_const(text, "kRows")
    cols = parse_const(text, "kCols")
    plane = rows * cols
    input_values = parse_input(text)
    require(
        len(input_values) == depth * plane,
        f"kInput length does not match {depth}x{rows}x{cols}",
    )

    interior_values: list[float] = []
    final_values = list(input_values)
    for z in range(1, depth - 1):
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                idx = z * plane + row * cols + col
                value = (
                    input_values[idx - plane]
                    + input_values[idx + plane]
                    + input_values[idx - cols]
                    + input_values[idx + cols]
                    + input_values[idx - 1]
                    + input_values[idx + 1]
                ) * (1.0 / 6.0)
                interior_values.append(value)
                final_values[idx] = value

    count = len(interior_values)
    return JacobiStencil7ptFixture(
        case="jacobi_stencil_7pt",
        graph="g_t_jacobi_stencil_7pt_kernel_0_0",
        input_arg=6,
        interior_arg=13,
        index_arg=14,
        scalar_args=(
            (0, "none"),
            (1, "1"),
            (2, "2"),
            (3, "-16"),
            (4, "4"),
            (5, "5"),
            (7, "37"),
            (8, "17"),
            (9, "25"),
            (10, "20"),
            (11, "22"),
            (12, "1.666666672e-01"),
        ),
        input_values=input_values,
        interior_values=tuple(interior_values),
        final_values=tuple(final_values),
        expected_fire_counts={
            "arith.addf": 5 * count,
            "arith.addi": 6 * count,
            "arith.andi": 3 * count,
            "arith.index_cast": 13 * count,
            "arith.mulf": count,
            "arith.ori": 2 * count,
            "arith.shli": 2 * count,
            "dataflow.load": 6 * count,
            "dataflow.store": count,
            "dataflow.sync": count,
            "llvm.trunc": 3 * count,
        },
    )


def emit_dfg_args(fixture: JacobiStencil7ptFixture) -> None:
    print(fixture.input_arg)
    print(fixture.interior_arg)
    print(fixture.index_arg)
    print(fixture.interior_count)
    print(csv_floats(fixture.input_values))
    print(",".join("0.000000000e+00" for _ in range(fixture.interior_count)))
    for index, value in fixture.scalar_args:
        print(f"{index}={value}")


def emit_json(fixture: JacobiStencil7ptFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "input_arg": fixture.input_arg,
                "interior_arg": fixture.interior_arg,
                "index_arg": fixture.index_arg,
                "interior_count": fixture.interior_count,
                "checksum": math.fsum(
                    (index + 1) * value
                    for index, value in enumerate(fixture.final_values)
                ),
                "interior_values": fixture.interior_values,
                "expected_fire_counts": fixture.expected_fire_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--emit", choices=("dfg-args", "json"), default="json")
    args = parser.parse_args(argv)
    fixture = fixture_from_source(args.source)
    if args.emit == "dfg-args":
        emit_dfg_args(fixture)
    else:
        emit_json(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
