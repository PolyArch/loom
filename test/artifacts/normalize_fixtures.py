#!/usr/bin/env python3
"""Source-derived fixtures for normalize app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NormalizeFixture:
    case: str
    sum_graph: str
    max_graph: str
    scale_graph: str
    size: int
    input_values: tuple[float, ...]
    output_values: tuple[float, ...]
    sum_value: float
    max_value: float
    scale_value: float
    zero_output_values: tuple[float, ...]
    expected_fire_counts: dict[str, dict[str, int]]

    @property
    def graphs(self) -> tuple[str, str, str]:
        return (self.sum_graph, self.max_graph, self.scale_graph)

    @property
    def expected_memory(self) -> dict[str, list[str]]:
        input_tokens = float_tokens(self.input_values)
        output_tokens = float_tokens(self.output_values)
        return {
            f"{self.sum_graph}:arg4": input_tokens,
            f"{self.max_graph}:arg4": input_tokens,
            f"{self.scale_graph}:arg1": input_tokens,
            f"{self.scale_graph}:arg3": output_tokens,
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_uint_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{name}\s*=\s*(?P<value>[0-9]+)u?\s*;", text)
    require(match is not None, f"missing constexpr {name}")
    return int(match.group("value"))


def parse_float_array(text: str, name: str, expected_size: int) -> tuple[float, ...]:
    match = re.search(
        rf"constexpr\s+std::array<float,\s*{re.escape('kSize')}\s*>\s+{name}\s*=\s*\{{(?P<body>.*?)\}};",
        text,
        re.S,
    )
    require(match is not None, f"missing constexpr array {name}")
    tokens = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f?", match.group("body"))
    require(len(tokens) == expected_size, f"{name} element count changed")
    return tuple(float(token.rstrip("f")) for token in tokens)


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "normalize" / "main_func.cpp"


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def float_token(value: float) -> str:
    return f"f32:{value:.6g}"


def float_tokens(values: tuple[float, ...]) -> list[str]:
    return [float_token(value) for value in values]


def fixture_from_source(source: Path | None = None) -> NormalizeFixture:
    source = source or source_path()
    text = source.read_text()
    size = parse_uint_const(text, "kSize")
    input_values = parse_float_array(text, "kInput", size)
    require(size == 8, "normalize fixture expects the legacy eight-element input")
    require("normalize_sum_kernel(input, &sum_result, n);" in text, "normalize must call the sum leaf")
    require("normalize_max_kernel(input, &max_result, n);" in text, "normalize must call the max leaf")
    require("normalize_scale_kernel(input, sum_result, output, n);" in text, "normalize must call the scale leaf")
    require("(void)max_result;" in text, "normalize must preserve the fork-join max component")

    sum_value = sum(input_values)
    max_value = max(input_values)
    scale_value = (1.0 / sum_value) if sum_value > 0.0 else 1.0
    output_values = tuple(value * scale_value for value in input_values)
    return NormalizeFixture(
        case="normalize",
        sum_graph="g_t_normalize_sum_kernel_red_0_0",
        max_graph="g_t_normalize_max_kernel_red_0_0",
        scale_graph="g_t_normalize_scale_kernel_0_0",
        size=size,
        input_values=input_values,
        output_values=output_values,
        sum_value=sum_value,
        max_value=max_value,
        scale_value=scale_value,
        zero_output_values=tuple(0.0 for _ in range(size)),
        expected_fire_counts={
            "sum": {
                "arith.addf": size,
                "arith.index_cast": size,
                "dataflow.load": size,
            },
            "max": {
                "arith.cmpf": size - 1,
                "arith.index_cast": size - 1,
                "arith.select": size - 1,
                "dataflow.load": size - 1,
            },
            "scale": {
                "arith.mulf": size,
                "dataflow.load": size,
                "dataflow.store": size,
            },
        },
    )


def emit_dfg_args(fixture: NormalizeFixture) -> None:
    print(fixture.size)
    print(csv_floats(fixture.input_values))
    print(csv_floats(fixture.zero_output_values))


def emit_json(fixture: NormalizeFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graphs": fixture.graphs,
                "size": fixture.size,
                "input_values": fixture.input_values,
                "output_values": fixture.output_values,
                "sum_value": fixture.sum_value,
                "max_value": fixture.max_value,
                "scale_value": fixture.scale_value,
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
