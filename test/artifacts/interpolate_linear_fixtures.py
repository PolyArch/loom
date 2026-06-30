#!/usr/bin/env python3
"""Source-derived fixtures for interpolate_linear CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InterpolateLinearFixture:
    case: str
    graph: str
    input_xq_arg: int
    input_x_arg: int
    input_y_arg: int
    output_arg: int
    index_arg: int
    scalar_args: tuple[tuple[int, str], ...]
    input_xq: tuple[float, ...]
    input_x: tuple[float, ...]
    input_y: tuple[float, ...]
    outputs: tuple[float, ...]
    expected_fire_counts: dict[str, int]

    @property
    def query_count(self) -> int:
        return len(self.input_xq)

    @property
    def data_count(self) -> int:
        return len(self.input_x)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{name}\s*=\s*(?P<value>[0-9]+)u?\s*;", text)
    require(match is not None, f"missing constexpr {name}")
    return int(match.group("value"))


def require_source_shape(text: str, source: Path) -> None:
    require("input_x[i] = static_cast<float>(i);" in text, f"unsupported input_x initializer in {source}")
    require("input_y[i] = static_cast<float>(i * i);" in text, f"unsupported input_y initializer in {source}")
    require(
        "input_xq[i] = static_cast<float>(i) * 0.5f;" in text,
        f"unsupported input_xq initializer in {source}",
    )


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "interpolate_linear" / "main_func.cpp"


def interpolate(input_x: tuple[float, ...], input_y: tuple[float, ...], xq: float) -> float:
    interval = 0
    for index in range(len(input_x) - 1):
        if xq >= input_x[index] and xq <= input_x[index + 1]:
            interval = index
            break
    x0 = input_x[interval]
    x1 = input_x[interval + 1]
    y0 = input_y[interval]
    y1 = input_y[interval + 1]
    return y0 + ((xq - x0) / (x1 - x0)) * (y1 - y0)


def fixture_from_source(source: Path | None = None) -> InterpolateLinearFixture:
    source = source or source_path()
    text = source.read_text()
    require_source_shape(text, source)
    data_count = parse_const(text, "kDataCount")
    query_count = parse_const(text, "kQueryCount")
    input_x = tuple(float(index) for index in range(data_count))
    input_y = tuple(float(index * index) for index in range(data_count))
    input_xq = tuple(float(index) * 0.5 for index in range(query_count))
    outputs = tuple(interpolate(input_x, input_y, value) for value in input_xq)
    return InterpolateLinearFixture(
        case="interpolate_linear",
        graph="g_t_interpolate_linear_kernel_0_0",
        input_xq_arg=1,
        input_x_arg=3,
        input_y_arg=13,
        output_arg=14,
        index_arg=15,
        scalar_args=(
            (0, "none"),
            (2, "0"),
            (4, "true"),
            (5, "0"),
            (6, "0"),
            (7, "1"),
            (8, "1"),
            (9, str(data_count - 1)),
            (10, "2"),
            (11, "0"),
            (12, "false"),
        ),
        input_xq=input_xq,
        input_x=input_x,
        input_y=input_y,
        outputs=outputs,
        expected_fire_counts={
            "arith.addi": 993,
            "arith.cmpf": 1986,
            "arith.cmpi": 1860,
            "arith.divf": query_count,
            "arith.extui": 1923,
            "arith.index_cast": 1182,
            "arith.index_castui": 1056,
            "arith.select": 930,
            "arith.subf": 189,
            "arith.trunci": 993,
            "arith.xori": 993,
            "dataflow.constant": 993,
            "dataflow.load": 2301,
            "dataflow.store": query_count,
            "dataflow.sync": query_count,
            "llvm.getelementptr": 1986,
            "llvm.intr.fmuladd": query_count,
            "llvm.trunc": query_count,
            "scf.if": 1056,
            "scf.index_switch": 1056,
        },
    )


def emit_dfg_args(fixture: InterpolateLinearFixture) -> None:
    print(fixture.input_xq_arg)
    print(fixture.input_x_arg)
    print(fixture.input_y_arg)
    print(fixture.output_arg)
    print(fixture.index_arg)
    print(fixture.query_count)
    print(csv_floats(fixture.input_xq))
    print(csv_floats(fixture.input_x))
    print(csv_floats(fixture.input_y))
    print(",".join("0.000000000e+00" for _ in range(fixture.query_count)))
    for index, value in fixture.scalar_args:
        print(f"{index}={value}")


def emit_json(fixture: InterpolateLinearFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "input_xq_arg": fixture.input_xq_arg,
                "input_x_arg": fixture.input_x_arg,
                "input_y_arg": fixture.input_y_arg,
                "output_arg": fixture.output_arg,
                "index_arg": fixture.index_arg,
                "query_count": fixture.query_count,
                "data_count": fixture.data_count,
                "scalar_args": list(fixture.scalar_args),
                "checksum": math.fsum(fixture.outputs),
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
