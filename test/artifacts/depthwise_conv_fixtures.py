#!/usr/bin/env python3
"""Source-derived fixtures for depthwise_conv app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DepthwiseConvComponent:
    index: int
    channel: int
    out_y: int
    out_x: int
    input_base: int
    kernel_values: tuple[float, ...]
    expected: float
    scalar_args: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class DepthwiseConvFixture:
    case: str
    graph: str
    kernel_arg: int
    input_arg: int
    input_values: tuple[float, ...]
    kernel_values: tuple[float, ...]
    outputs: tuple[float, ...]
    components: tuple[DepthwiseConvComponent, ...]
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


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "depthwise_conv" / "main_func.cpp"


def source_defines_expected_initializers(text: str) -> None:
    require(
        "input[i] = static_cast<float>(i % 10u);" in text,
        "depthwise_conv input initializer changed",
    )
    require(
        "kernel[i] = (static_cast<float>(i % 5u) - 2.0f) / 10.0f;" in text,
        "depthwise_conv kernel initializer changed",
    )


def float_arg(value: float) -> str:
    return f"{value:.9e}"


def csv_floats(values: tuple[float, ...]) -> str:
    return ",".join(float_arg(value) for value in values)


def fixture_from_source(source: Path | None = None) -> DepthwiseConvFixture:
    source = source or source_path()
    text = source.read_text()
    source_defines_expected_initializers(text)
    channels = parse_const(text, "kChannels")
    height = parse_const(text, "kHeight")
    width = parse_const(text, "kWidth")
    kernel_h = parse_const(text, "kKernelH")
    kernel_w = parse_const(text, "kKernelW")
    out_h = parse_const(text, "kOutH")
    out_w = parse_const(text, "kOutW")
    require(
        re.search(r"constexpr\s+uint32_t\s+kOutputCount\s*=\s*kChannels\s*\*\s*kOutH\s*\*\s*kOutW\s*;", text)
        is not None,
        "kOutputCount must remain kChannels * kOutH * kOutW",
    )
    output_count = channels * out_h * out_w
    require(output_count == channels * out_h * out_w, "kOutputCount must match output shape")
    require(width == 8, "depthwise_conv fixture currently expects width shift 3")
    require(kernel_h == 3 and kernel_w == 3, "depthwise_conv fixture expects a 3x3 kernel")

    input_values = tuple(float(index % 10) for index in range(channels * height * width))
    kernel_values = tuple((float(index % 5) - 2.0) / 10.0 for index in range(channels * kernel_h * kernel_w))
    outputs: list[float] = []
    components: list[DepthwiseConvComponent] = []
    for index in range(output_count):
        channel = index // (out_h * out_w)
        rem = index - channel * (out_h * out_w)
        out_y = rem // out_w
        out_x = rem - out_y * out_w
        total = 0.0
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_y = out_y + kh
                in_x = out_x + kw
                input_value = input_values[channel * (height * width) + in_y * width + in_x]
                kernel_value = kernel_values[channel * (kernel_h * kernel_w) + kh * kernel_w + kw]
                total += input_value * kernel_value
        outputs.append(total)
        channel_kernel = kernel_values[channel * kernel_h * kernel_w :]
        components.append(
            DepthwiseConvComponent(
                index=index,
                channel=channel,
                out_y=out_y,
                out_x=out_x,
                input_base=channel * height * width + out_x,
                kernel_values=channel_kernel,
                expected=total,
                scalar_args=(
                    (0, "none"),
                    (1, "0"),
                    (2, str(kernel_h)),
                    (3, "1"),
                    (4, str(out_y)),
                    (5, "3"),
                    (6, str(channel * height * width + out_x)),
                    (7, "12"),
                    (10, "0.000000000e+00"),
                ),
            )
        )

    count = output_count
    return DepthwiseConvFixture(
        case="depthwise_conv",
        graph="g_t_depthwise_conv_kernel_0_0",
        kernel_arg=8,
        input_arg=9,
        input_values=input_values,
        kernel_values=kernel_values,
        outputs=tuple(outputs),
        components=tuple(components),
        expected_fire_counts={
            "arith.addi": 27 * count,
            "arith.index_cast": 54 * count,
            "arith.muli": 3 * count,
            "arith.shli": 9 * count,
            "dataflow.load": 18 * count,
            "llvm.getelementptr": 3 * count,
            "llvm.intr.fmuladd": 9 * count,
            "llvm.trunc": 12 * count,
        },
    )


def emit_dfg_args(fixture: DepthwiseConvFixture) -> None:
    print(fixture.kernel_arg)
    print(fixture.input_arg)
    print(fixture.size)
    print(csv_floats(fixture.input_values))
    for component in fixture.components:
        fields = [csv_floats(component.kernel_values)]
        fields.extend(f"{index}={value}" for index, value in component.scalar_args)
        print(";".join(fields))


def emit_json(fixture: DepthwiseConvFixture) -> None:
    print(
        json.dumps(
            {
                "case": fixture.case,
                "graph": fixture.graph,
                "kernel_arg": fixture.kernel_arg,
                "input_arg": fixture.input_arg,
                "size": fixture.size,
                "checksum": sum((index + 1) * value for index, value in enumerate(fixture.outputs)),
                "input_values": fixture.input_values,
                "kernel_values": fixture.kernel_values,
                "outputs": fixture.outputs,
                "expected_fire_counts": fixture.expected_fire_counts,
                "components": [
                    {
                        "index": component.index,
                        "channel": component.channel,
                        "out_y": component.out_y,
                        "out_x": component.out_x,
                        "input_base": component.input_base,
                        "expected": component.expected,
                    }
                    for component in fixture.components
                ],
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
