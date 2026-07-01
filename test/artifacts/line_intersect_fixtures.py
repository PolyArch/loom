#!/usr/bin/env python3
"""Source-derived fixtures for line_intersect app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LineIntersectFixture:
    case: str
    graph: str
    line_a_arg: int
    line_b_arg: int
    output_arg: int
    index_arg: int
    scalar_args: tuple[tuple[int, str], ...]
    line_a_values: tuple[float, ...]
    line_b_values: tuple[float, ...]
    outputs: tuple[int, ...]
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


def eval_source_expr(expr: str, index: int, bindings: dict[str, str] | None = None) -> float:
    normalized = expr
    for name, value in (bindings or {}).items():
        normalized = re.sub(rf"\b{name}\b", f"({value})", normalized)
    normalized = normalized.replace("static_cast<float>(i)", "i")
    normalized = re.sub(r"([0-9.])f\b", r"\1", normalized)
    normalized = re.sub(r"([0-9])u\b", r"\1", normalized)
    require(
        re.fullmatch(r"[0-9i+\-*/ ().]+", normalized) is not None,
        f"unsupported initializer expression: {expr}",
    )
    return float(eval(normalized, {"__builtins__": {}}, {"i": float(index)}))


def parse_initial_literal(text: str, array_name: str, offset: int) -> float:
    pattern = rf"{array_name}\s*\[\s*{offset}\s*\]\s*=\s*(?P<value>[-+0-9.]+f?)\s*;"
    match = re.search(pattern, text)
    require(match is not None, f"missing {array_name}[{offset}] literal")
    return float(match.group("value").rstrip("f"))


def parse_loop_expr(text: str, array_name: str, lane: int) -> str:
    pattern = (
        rf"{array_name}\s*\[\s*i\s*\*\s*4u?\s*\+\s*{lane}u?\s*\]\s*=\s*"
        r"(?P<expr>[^;]+);"
    )
    match = re.search(pattern, text)
    require(match is not None, f"missing {array_name} lane {lane} loop assignment")
    return match.group("expr")


def parse_loop_binding(text: str, name: str) -> str:
    pattern = rf"const\s+float\s+{name}\s*=\s*(?P<expr>[^;]+);"
    match = re.search(pattern, text)
    require(match is not None, f"missing loop binding {name}")
    return match.group("expr")


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "line_intersect" / "main_func.cpp"


def intersect(line_a: tuple[float, float, float, float], line_b: tuple[float, float, float, float]) -> int:
    ax1, ay1, ax2, ay2 = line_a
    bx1, by1, bx2, by2 = line_b
    dax = ax2 - ax1
    day = ay2 - ay1
    dbx = bx2 - bx1
    dby = by2 - by1
    denom = dax * dby - day * dbx
    if math.fabs(denom) < 1.0e-8:
        return 0
    dx = bx1 - ax1
    dy = by1 - ay1
    t = (dx * dby - dy * dbx) / denom
    u = (dx * day - dy * dax) / denom
    return 1 if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else 0


def fixture_from_source(source: Path | None = None) -> LineIntersectFixture:
    source = source or source_path()
    text = source.read_text()
    count = parse_const(text, "kCount")
    line_a_values = [0.0 for _ in range(count * 4)]
    line_b_values = [0.0 for _ in range(count * 4)]

    for offset in range(12):
        line_a_values[offset] = parse_initial_literal(text, "line_a", offset)
        line_b_values[offset] = parse_initial_literal(text, "line_b", offset)

    line_a_exprs = tuple(parse_loop_expr(text, "line_a", lane) for lane in range(4))
    line_b_exprs = tuple(parse_loop_expr(text, "line_b", lane) for lane in range(4))
    loop_bindings = {"offset": parse_loop_binding(text, "offset")}
    for index in range(3, count):
        for lane, expr in enumerate(line_a_exprs):
            line_a_values[index * 4 + lane] = eval_source_expr(expr, index, loop_bindings)
        for lane, expr in enumerate(line_b_exprs):
            line_b_values[index * 4 + lane] = eval_source_expr(expr, index, loop_bindings)

    outputs = tuple(
        intersect(
            tuple(line_a_values[index * 4 + lane] for lane in range(4)),  # type: ignore[arg-type]
            tuple(line_b_values[index * 4 + lane] for lane in range(4)),  # type: ignore[arg-type]
        )
        for index in range(count)
    )
    return LineIntersectFixture(
        case="line_intersect",
        graph="g_t_line_intersect_kernel_0_0",
        line_a_arg=2,
        line_b_arg=5,
        output_arg=11,
        index_arg=12,
        scalar_args=(
            (0, "none"),
            (1, "2"),
            (3, "1"),
            (4, "3"),
            (6, "1.000000000e-08"),
            (7, "0"),
            (8, "0.000000000e+00"),
            (9, "1.000000000e+00"),
            (10, "false"),
        ),
        line_a_values=tuple(line_a_values),
        line_b_values=tuple(line_b_values),
        outputs=outputs,
        expected_fire_counts={
            "arith.andi": 2 * count,
            "arith.cmpf": 5 * count,
            "arith.divf": 2 * count,
            "arith.index_cast": 6 * count,
            "arith.mulf": 3 * count,
            "arith.ori": 3 * count,
            "arith.select": count,
            "arith.shli": 2 * count,
            "arith.subf": 6 * count,
            "dataflow.load": 8 * count,
            "dataflow.mux": count,
            "dataflow.store": count,
            "dataflow.sync": count,
            "llvm.fneg": 2 * count,
            "llvm.intr.fabs": count,
            "llvm.intr.fmuladd": 3 * count,
            "llvm.trunc": count,
            "llvm.zext": count,
        },
    )


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def csv_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def emit_dfg_args(fixture: LineIntersectFixture) -> None:
    print(fixture.line_a_arg)
    print(fixture.line_b_arg)
    print(fixture.output_arg)
    print(fixture.index_arg)
    print(fixture.size)
    print(csv_floats(fixture.line_a_values))
    print(csv_floats(fixture.line_b_values))
    print(",".join("0" for _ in range(fixture.size)))
    for index, value in fixture.scalar_args:
        print(f"{index}={value}")


def emit_json(fixture: LineIntersectFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "line_a_arg": fixture.line_a_arg,
                "line_b_arg": fixture.line_b_arg,
                "output_arg": fixture.output_arg,
                "index_arg": fixture.index_arg,
                "size": fixture.size,
                "checksum": sum((index + 1) * value for index, value in enumerate(fixture.outputs)),
                "line_a_values": fixture.line_a_values,
                "line_b_values": fixture.line_b_values,
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
