#!/usr/bin/env python3
"""Source-derived fixtures for normalize_vec3 app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NormalizeVec3Fixture:
    case: str
    graph: str
    input_arg: int
    output_arg: int
    size_arg: int
    scalar_args: tuple[tuple[int, str], ...]
    input_values: tuple[float, ...]
    outputs: tuple[float, ...]
    expected_fire_counts: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.outputs) // 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_uint_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{name}\s*=\s*(?P<value>[0-9]+)u?\s*;", text)
    require(match is not None, f"missing constexpr {name}")
    return int(match.group("value"))


def parse_float_const(text: str, name: str) -> float:
    match = re.search(
        rf"constexpr\s+float\s+{name}\s*=\s*(?P<value>[0-9.+\-eE]+)f?\s*;",
        text,
    )
    require(match is not None, f"missing constexpr {name}")
    return float(match.group("value"))


def parse_assignment_expr(text: str, array_name: str, lane: int) -> str:
    pattern = (
        rf"{array_name}\s*\[\s*i\s*\*\s*3u?\s*\+\s*{lane}u?\s*\]\s*=\s*"
        r"(?P<expr>[^;]+);"
    )
    match = re.search(pattern, text)
    require(match is not None, f"missing {array_name} lane {lane} assignment")
    return match.group("expr")


def parse_scalar_overrides(text: str, array_name: str) -> dict[int, float]:
    overrides: dict[int, float] = {}
    pattern = rf"{array_name}\s*\[\s*(?P<index>[0-9]+)u?\s*\]\s*=\s*(?P<value>[0-9.+\-eE]+)f?\s*;"
    for match in re.finditer(pattern, text):
        overrides[int(match.group("index"))] = float(match.group("value"))
    return overrides


def eval_source_expr(expr: str, index: int) -> float:
    normalized = expr.replace("static_cast<float>(i)", "i")
    normalized = re.sub(r"([0-9.])f\b", r"\1", normalized)
    normalized = re.sub(r"([0-9])u\b", r"\1", normalized)
    require(
        re.fullmatch(r"[0-9eEi+\-*/ ().]+", normalized) is not None,
        f"unsupported initializer expression: {expr}",
    )
    return float(eval(normalized, {"__builtins__": {}}, {"i": float(index)}))


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "normalize_vec3" / "main_func.cpp"


def fixture_from_source(source: Path | None = None) -> NormalizeVec3Fixture:
    source = source or source_path()
    text = source.read_text()
    size = parse_uint_const(text, "kSize")
    epsilon = parse_float_const(text, "kEpsilon")
    exprs = tuple(parse_assignment_expr(text, "input", lane) for lane in range(3))
    overrides = parse_scalar_overrides(text, "input")
    input_values: list[float] = []
    outputs: list[float] = []
    nonzero_vectors = 0
    for index in range(size):
        point = [eval_source_expr(expr, index) for expr in exprs]
        for lane in range(3):
            flat_index = index * 3 + lane
            if flat_index in overrides:
                point[lane] = overrides[flat_index]
        input_values.extend(point)
        length = math.sqrt(sum(component * component for component in point))
        if length > epsilon:
            outputs.extend(component / length for component in point)
            nonzero_vectors += 1
        else:
            outputs.extend((0.0, 0.0, 0.0))
    return NormalizeVec3Fixture(
        case="normalize_vec3",
        graph="g_normalize_vec3_kernel_0",
        input_arg=1,
        output_arg=2,
        size_arg=3,
        scalar_args=((0, "none"),),
        input_values=tuple(input_values),
        outputs=tuple(outputs),
        expected_fire_counts={
            "arith.addi": 4 * size,
            "arith.cmpf": size,
            "arith.cmpi": 1,
            "arith.divf": 3 * nonzero_vectors,
            "arith.index_cast": 16 * size,
            "arith.mulf": size,
            "arith.muli": 6 * size,
            "dataflow.constant": 8,
            "dataflow.load": 3 * size,
            "dataflow.store": 3 * size,
            "llvm.intr.fmuladd": 2 * size,
            "llvm.trunc": size,
            "llvm.zext": 1,
            "math.sqrt": size,
            "scf.if": size + 1,
        },
    )


def emit_dfg_args(fixture: NormalizeVec3Fixture) -> None:
    print(fixture.input_arg)
    print(fixture.output_arg)
    print(fixture.size_arg)
    print(fixture.size)
    print(csv_floats(fixture.input_values))
    print(",".join("0.000000000e+00" for _ in range(len(fixture.outputs))))
    for index, value in fixture.scalar_args:
        print(f"{index}={value}")


def emit_json(fixture: NormalizeVec3Fixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "input_arg": fixture.input_arg,
                "output_arg": fixture.output_arg,
                "size_arg": fixture.size_arg,
                "size": fixture.size,
                "input_values": fixture.input_values,
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
