#!/usr/bin/env python3
"""Source-derived fixtures for batchnorm app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BatchnormFixture:
    case: str
    graph: str
    channels: int
    height: int
    width: int
    epsilon: float
    input_arg: int
    mean_arg: int
    variance_arg: int
    gamma_arg: int
    beta_arg: int
    output_arg: int
    inputs: tuple[float, ...]
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    outputs: tuple[float, ...]
    expected_fire_counts: dict[str, int]

    @property
    def element_count(self) -> int:
        return self.channels * self.height * self.width

    @property
    def dynamic_work_items(self) -> int:
        return self.height


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_numeric_literal(raw: str) -> float:
    return float(raw.strip().rstrip("fFuU"))


def parse_const(text: str, name: str) -> float:
    match = re.search(
        rf"constexpr\s+(?:float|uint32_t)\s+{name}\s*=\s*(?P<value>[^;]+);",
        text,
    )
    require(match is not None, f"missing constexpr {name}")
    return parse_numeric_literal(match.group("value"))


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "batchnorm" / "main_func.cpp"


def fixture_from_source(source: Path | None = None) -> BatchnormFixture:
    source = source or source_path()
    text = source.read_text()
    channels = int(parse_const(text, "kChannels"))
    height = int(parse_const(text, "kHeight"))
    width = int(parse_const(text, "kWidth"))
    epsilon = parse_const(text, "kEpsilon")
    element_count = channels * height * width

    inputs = tuple(float(index % 100) - 50.0 for index in range(element_count))
    mean = tuple(float(channel * 10) for channel in range(channels))
    variance = tuple(float(channel + 1) * 2.0 for channel in range(channels))
    gamma = tuple(1.0 for _ in range(channels))
    beta = tuple(0.0 for _ in range(channels))
    outputs: list[float] = []
    for channel in range(channels):
        inv_std = 1.0 / math.sqrt(variance[channel] + epsilon)
        for row in range(height):
            for column in range(width):
                index = channel * (height * width) + row * width + column
                normalized = (inputs[index] - mean[channel]) * inv_std
                outputs.append(gamma[channel] * normalized + beta[channel])

    return BatchnormFixture(
        case="batchnorm",
        graph="g_t_batchnorm_kernel_0_0",
        channels=channels,
        height=height,
        width=width,
        epsilon=epsilon,
        input_arg=10,
        mean_arg=5,
        variance_arg=1,
        gamma_arg=6,
        beta_arg=7,
        output_arg=11,
        inputs=inputs,
        mean=mean,
        variance=variance,
        gamma=gamma,
        beta=beta,
        outputs=tuple(outputs),
        expected_fire_counts={
            "arith.addf": channels,
            "arith.addi": 4 * element_count,
            "arith.divf": channels,
            "arith.index_cast": 9 * element_count + 10 * channels,
            "arith.mulf": element_count,
            "arith.muli": 4 * element_count,
            "arith.subf": element_count,
            "dataflow.load": 4 * element_count + channels,
            "dataflow.store": element_count,
            "dataflow.sync": channels,
            "llvm.intr.fmuladd": element_count,
            "llvm.trunc": element_count + channels,
            "math.sqrt": channels,
            "scf.forall": channels + channels * height,
            "scf.if": channels + channels * height,
        },
    )


def emit_cli_args(fixture: BatchnormFixture) -> None:
    tokens: list[str] = [
        "--graph",
        fixture.graph,
        "--workload",
        fixture.case,
        "--memref",
        f"{fixture.variance_arg}={csv_floats(fixture.variance)}",
        "--memref",
        f"{fixture.mean_arg}={csv_floats(fixture.mean)}",
        "--memref",
        f"{fixture.gamma_arg}={csv_floats(fixture.gamma)}",
        "--memref",
        f"{fixture.beta_arg}={csv_floats(fixture.beta)}",
        "--memref",
        f"{fixture.input_arg}={csv_floats(fixture.inputs)}",
        "--memref",
        f"{fixture.output_arg}={csv_floats(tuple(0.0 for _ in range(fixture.element_count)))}",
    ]
    for channel in range(fixture.channels):
        tokens.extend(
            [
                "--arg",
                "0=none",
                "--arg",
                f"2={float_arg(fixture.epsilon)}",
                "--arg",
                "3=1.000000000e+00",
                "--arg",
                f"4={fixture.height}",
                "--arg",
                f"8={fixture.width}",
                "--arg",
                f"9={fixture.width}",
                "--arg",
                "12=false",
                "--arg",
                "13=false",
                "--arg",
                f"14={channel}",
            ]
        )
    print("\n".join(tokens))


def emit_json(fixture: BatchnormFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "channels": fixture.channels,
                "height": fixture.height,
                "width": fixture.width,
                "epsilon": fixture.epsilon,
                "element_count": fixture.element_count,
                "input_arg": fixture.input_arg,
                "mean_arg": fixture.mean_arg,
                "variance_arg": fixture.variance_arg,
                "gamma_arg": fixture.gamma_arg,
                "beta_arg": fixture.beta_arg,
                "output_arg": fixture.output_arg,
                "dynamic_work_items": fixture.dynamic_work_items,
                "expected_fire_counts": fixture.expected_fire_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--emit", choices=("dfg-args", "json"), default="json")
    args = parser.parse_args()
    fixture = fixture_from_source(args.source)
    if args.emit == "dfg-args":
        emit_cli_args(fixture)
    else:
        emit_json(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
