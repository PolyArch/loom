#!/usr/bin/env python3
"""Source-derived fixtures for distance_point app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DistancePointFixture:
    case: str
    graph: str
    a_arg: int
    b_arg: int
    output_arg: int
    index_arg: int
    scalar_args: tuple[tuple[int, str], ...]
    a_values: tuple[float, ...]
    b_values: tuple[float, ...]
    outputs: tuple[float, ...]
    expected_fire_counts: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.outputs)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{name}\s*=\s*(?P<value>[0-9]+)u?\s*;", text)
    require(match is not None, f"missing constexpr {name}")
    return int(match.group("value"))


def parse_assignment_expr(text: str, array_name: str, lane: int) -> str:
    pattern = (
        rf"{array_name}\s*\[\s*i\s*\*\s*3u?\s*\+\s*{lane}u?\s*\]\s*=\s*"
        r"(?P<expr>[^;]+);"
    )
    match = re.search(pattern, text)
    require(match is not None, f"missing {array_name} lane {lane} assignment")
    return match.group("expr")


def eval_source_expr(expr: str, index: int) -> float:
    normalized = expr.replace("static_cast<float>(i)", "i")
    normalized = re.sub(r"([0-9.])f\b", r"\1", normalized)
    normalized = re.sub(r"([0-9])u\b", r"\1", normalized)
    require(
        re.fullmatch(r"[0-9i+\-*/ ().]+", normalized) is not None,
        f"unsupported initializer expression: {expr}",
    )
    return float(eval(normalized, {"__builtins__": {}}, {"i": float(index)}))


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "distance_point" / "main_func.cpp"


def fixture_from_source(source: Path | None = None) -> DistancePointFixture:
    source = source or source_path()
    text = source.read_text()
    count = parse_const(text, "kCount")
    a_exprs = tuple(parse_assignment_expr(text, "a", lane) for lane in range(3))
    b_exprs = tuple(parse_assignment_expr(text, "b", lane) for lane in range(3))
    a_values: list[float] = []
    b_values: list[float] = []
    outputs: list[float] = []
    for index in range(count):
        a_point = tuple(eval_source_expr(expr, index) for expr in a_exprs)
        b_point = tuple(eval_source_expr(expr, index) for expr in b_exprs)
        a_values.extend(a_point)
        b_values.extend(b_point)
        dx = a_point[0] - b_point[0]
        dy = a_point[1] - b_point[1]
        dz = a_point[2] - b_point[2]
        outputs.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return DistancePointFixture(
        case="distance_point",
        graph="g_t_distance_point_kernel_0_0",
        a_arg=2,
        b_arg=3,
        output_arg=6,
        index_arg=7,
        scalar_args=((0, "none"), (1, "3"), (4, "1"), (5, "2")),
        a_values=tuple(a_values),
        b_values=tuple(b_values),
        outputs=tuple(outputs),
        expected_fire_counts={
            "arith.addi": 2 * count,
            "arith.index_cast": 9 * count,
            "arith.mulf": count,
            "arith.muli": 3 * count,
            "arith.subf": 3 * count,
            "dataflow.load": 6 * count,
            "dataflow.store": count,
            "dataflow.sync": count,
            "llvm.intr.fmuladd": 2 * count,
            "llvm.trunc": count,
            "math.sqrt": count,
        },
    )


def emit_dfg_args(fixture: DistancePointFixture) -> None:
    print(fixture.a_arg)
    print(fixture.b_arg)
    print(fixture.output_arg)
    print(fixture.index_arg)
    print(fixture.size)
    print(csv_floats(fixture.a_values))
    print(csv_floats(fixture.b_values))
    print(",".join("0.000000000e+00" for _ in range(fixture.size)))
    for index, value in fixture.scalar_args:
        print(f"{index}={value}")


def emit_json(fixture: DistancePointFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "a_arg": fixture.a_arg,
                "b_arg": fixture.b_arg,
                "output_arg": fixture.output_arg,
                "index_arg": fixture.index_arg,
                "size": fixture.size,
                "a_values": fixture.a_values,
                "b_values": fixture.b_values,
                "outputs": fixture.outputs,
                "expected_fire_counts": fixture.expected_fire_counts,
            },
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
