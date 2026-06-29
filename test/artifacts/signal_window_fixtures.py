#!/usr/bin/env python3
"""Source-derived fixtures for signal-window app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WindowFixture:
    case: str
    graph: str
    input_arg: int
    output_arg: int
    index_arg: int
    scalar_args: tuple[tuple[int, str], ...]
    inputs: tuple[float, ...]
    outputs: tuple[float, ...]
    expected_fire_counts: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.inputs)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def parse_float_literal(raw: str) -> float:
    return float(raw.strip().rstrip("fFuU"))


def parse_const(text: str, name: str) -> float:
    match = re.search(rf"constexpr\s+(?:float|uint32_t)\s+{name}\s*=\s*(?P<value>[^;]+);", text)
    require(match is not None, f"missing constexpr {name}")
    return parse_float_literal(match.group("value"))


def parse_common_source(source: Path) -> tuple[int, float, float, float]:
    text = source.read_text()
    size = int(parse_const(text, "kSize"))
    input_pi = parse_const(text, "kInputPi")
    window_pi = parse_const(text, "kWindowPi")
    denominator_match = re.search(
        r"std::sin\(\s*2\.0f\s*\*\s*kInputPi\s*\*\s*static_cast<float>\(i\)\s*/\s*([0-9.]+)f?\s*\)",
        text,
        re.S,
    )
    require(denominator_match is not None, f"missing signal-window input denominator in {source}")
    return size, input_pi, window_pi, parse_float_literal(denominator_match.group(1))


def source_path(case: str) -> Path:
    return REPO_ROOT / "test" / "app" / case / "main_func.cpp"


def hamming_fixture(source: Path) -> WindowFixture:
    text = source.read_text()
    size, input_pi, window_pi, input_denominator = parse_common_source(source)
    window_match = re.search(
        r"const\s+float\s+window\s*=\s*([0-9.]+)f?\s*-\s*([0-9.]+)f?\s*\*\s*std::cos",
        text,
    )
    require(window_match is not None, f"missing Hamming window coefficients in {source}")
    base = parse_float_literal(window_match.group(1))
    amplitude = parse_float_literal(window_match.group(2))
    twopi = f32(2.0 * f32(window_pi))
    denominator = float(size - 1)
    inputs = tuple(math.sin(2.0 * input_pi * float(index) / input_denominator) for index in range(size))
    outputs = tuple(
        inputs[index] * (base - amplitude * math.cos(twopi * float(index) / denominator))
        for index in range(size)
    )
    return WindowFixture(
        case="window_hamming",
        graph="g_t_window_hamming_kernel_0_0",
        input_arg=5,
        output_arg=6,
        index_arg=7,
        scalar_args=(
            (0, "none"),
            (1, float_arg(twopi)),
            (2, float_arg(float(size - 1))),
            (3, float_arg(-amplitude)),
            (4, float_arg(base)),
        ),
        inputs=inputs,
        outputs=outputs,
        expected_fire_counts={
            "arith.divf": size,
            "arith.index_cast": size,
            "arith.mulf": 2 * size,
            "dataflow.load": size,
            "dataflow.store": size,
            "dataflow.sync": size,
            "llvm.intr.fmuladd": size,
            "llvm.trunc": size,
            "llvm.uitofp": size,
            "math.cos": size,
        },
    )


def hanning_fixture(source: Path) -> WindowFixture:
    text = source.read_text()
    size, input_pi, window_pi, input_denominator = parse_common_source(source)
    window_match = re.search(
        r"const\s+float\s+window\s*=\s*([0-9.]+)f?\s*\*\s*\(\s*([0-9.]+)f?\s*-\s*std::cos",
        text,
    )
    require(window_match is not None, f"missing Hanning window coefficients in {source}")
    scale = parse_float_literal(window_match.group(1))
    one = parse_float_literal(window_match.group(2))
    twopi = f32(2.0 * f32(window_pi))
    denominator = float(size - 1)
    inputs = tuple(math.sin(2.0 * input_pi * float(index) / input_denominator) for index in range(size))
    outputs = tuple(
        inputs[index] * (scale * (one - math.cos(twopi * float(index) / denominator)))
        for index in range(size)
    )
    return WindowFixture(
        case="window_hanning",
        graph="g_t_window_hanning_kernel_0_0",
        input_arg=5,
        output_arg=6,
        index_arg=7,
        scalar_args=(
            (0, "none"),
            (1, float_arg(twopi)),
            (2, float_arg(float(size - 1))),
            (3, float_arg(one)),
            (4, float_arg(scale)),
        ),
        inputs=inputs,
        outputs=outputs,
        expected_fire_counts={
            "arith.divf": size,
            "arith.index_cast": size,
            "arith.mulf": 3 * size,
            "arith.subf": size,
            "dataflow.load": size,
            "dataflow.store": size,
            "dataflow.sync": size,
            "llvm.trunc": size,
            "llvm.uitofp": size,
            "math.cos": size,
        },
    )


def blackman_fixture(source: Path) -> WindowFixture:
    text = source.read_text()
    size, input_pi, window_pi, input_denominator = parse_common_source(source)
    window_match = re.search(
        r"const\s+float\s+window\s*=\s*([0-9.]+)f?\s*-\s*([0-9.]+)f?\s*\*\s*std::cos\(t\)\s*\+\s*"
        r"([0-9.]+)f?\s*\*\s*std::cos\(\s*([0-9.]+)f?\s*\*\s*t\s*\)",
        text,
        re.S,
    )
    require(window_match is not None, f"missing Blackman window coefficients in {source}")
    base = parse_float_literal(window_match.group(1))
    first_amplitude = parse_float_literal(window_match.group(2))
    second_amplitude = parse_float_literal(window_match.group(3))
    second_multiplier = parse_float_literal(window_match.group(4))
    twopi = f32(2.0 * f32(window_pi))
    denominator = float(size - 1)
    inputs = tuple(math.sin(2.0 * input_pi * float(index) / input_denominator) for index in range(size))
    outputs = tuple(
        inputs[index]
        * (
            base
            - first_amplitude * math.cos(twopi * float(index) / denominator)
            + second_amplitude * math.cos(second_multiplier * twopi * float(index) / denominator)
        )
        for index in range(size)
    )
    return WindowFixture(
        case="window_blackman",
        graph="g_t_window_blackman_kernel_0_0",
        input_arg=7,
        output_arg=8,
        index_arg=9,
        scalar_args=(
            (0, "none"),
            (1, float_arg(twopi)),
            (2, float_arg(float(size - 1))),
            (3, float_arg(-first_amplitude)),
            (4, float_arg(base)),
            (5, float_arg(second_multiplier)),
            (6, float_arg(second_amplitude)),
        ),
        inputs=inputs,
        outputs=outputs,
        expected_fire_counts={
            "arith.divf": size,
            "arith.index_cast": size,
            "arith.mulf": 3 * size,
            "dataflow.load": size,
            "dataflow.store": size,
            "dataflow.sync": size,
            "llvm.intr.fmuladd": 2 * size,
            "llvm.trunc": size,
            "llvm.uitofp": size,
            "math.cos": 2 * size,
        },
    )


def fixture_for_case(case: str, source: Path | None = None) -> WindowFixture:
    source = source or source_path(case)
    if case == "window_hamming":
        return hamming_fixture(source)
    if case == "window_hanning":
        return hanning_fixture(source)
    if case == "window_blackman":
        return blackman_fixture(source)
    raise ValueError(f"unsupported signal-window case: {case}")


def emit_dfg_args(fixture: WindowFixture) -> None:
    print(fixture.input_arg)
    print(fixture.output_arg)
    print(fixture.index_arg)
    print(fixture.size)
    print(csv_floats(fixture.inputs))
    print(",".join("0.000000000e+00" for _ in range(fixture.size)))
    for index, value in fixture.scalar_args:
        print(f"{index}={value}")


def emit_json(fixture: WindowFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "input_arg": fixture.input_arg,
                "output_arg": fixture.output_arg,
                "index_arg": fixture.index_arg,
                "size": fixture.size,
                "scalar_args": list(fixture.scalar_args),
                "expected_fire_counts": fixture.expected_fire_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--emit", choices=("dfg-args", "json"), default="json")
    args = parser.parse_args()
    fixture = fixture_for_case(args.case, args.source)
    if args.emit == "dfg-args":
        emit_dfg_args(fixture)
    else:
        emit_json(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
