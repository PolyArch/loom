#!/usr/bin/env python3
"""Source-derived fixtures for fft_butterfly app CGRA evidence."""

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
class FftButterflyFixture:
    size: int
    input_real: tuple[float, ...]
    input_imag: tuple[float, ...]
    expected_real: tuple[float, ...]
    expected_imag: tuple[float, ...]
    components: tuple[tuple[str, ...], ...]

    @property
    def expected_checksum(self) -> float:
        total = f32(0.0)
        for index, (real, imag) in enumerate(zip(self.expected_real, self.expected_imag)):
            weight = f32(float(index + 1))
            total = f32(total + f32(weight * real))
            total = f32(total + f32(f32(weight + f32(0.25)) * imag))
        return total


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def source_path() -> Path:
    return REPO_ROOT / "test" / "app" / "fft_butterfly" / "main_func.cpp"


def parse_uint_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{re.escape(name)}\s*=\s*(\d+)u?\s*;", text)
    require(match is not None, f"missing {name}")
    return int(match.group(1))


def parse_float_const(text: str, name: str) -> float:
    match = re.search(
        rf"constexpr\s+float\s+{re.escape(name)}\s*=\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)f?\s*;",
        text,
    )
    require(match is not None, f"missing {name}")
    return f32(float(match.group(1)))


def parse_float_array(text: str, name: str) -> tuple[float, ...]:
    match = re.search(
        rf"const\s+std::array<float,\s*kSize>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
        text,
        re.S,
    )
    require(match is not None, f"missing {name}")
    values = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f?",
        match.group("body"),
    )
    require(values, f"{name} must contain values")
    return tuple(f32(float(value.rstrip("f"))) for value in values)


def fft_stage_count(size: int) -> int:
    stages = 0
    current = size
    while current > 1:
        stages += 1
        current >>= 1
    return stages


def reference_fft(
    input_real: tuple[float, ...],
    input_imag: tuple[float, ...],
    *,
    pi_value: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    size = len(input_real)
    real = [f32(value) for value in input_real]
    imag = [f32(value) for value in input_imag]
    for stage in range(1, fft_stage_count(size) + 1):
        m = 1 << stage
        angle = f32(f32(-2.0) * f32(pi_value / f32(float(m))))
        wm_r = f32(math.cos(angle))
        wm_i = f32(math.sin(angle))
        for base in range(0, size, m):
            w_r = f32(1.0)
            w_i = f32(0.0)
            for offset in range(m // 2):
                lo = base + offset
                hi = lo + m // 2
                t_r = f32(f32(w_r * real[hi]) - f32(w_i * imag[hi]))
                t_i = f32(f32(w_r * imag[hi]) + f32(w_i * real[hi]))
                u_r = real[lo]
                u_i = imag[lo]
                real[lo] = f32(u_r + t_r)
                imag[lo] = f32(u_i + t_i)
                real[hi] = f32(u_r - t_r)
                imag[hi] = f32(u_i - t_i)
                next_w_r = f32(f32(w_r * wm_r) - f32(w_i * wm_i))
                next_w_i = f32(f32(w_r * wm_i) + f32(w_i * wm_r))
                w_r = next_w_r
                w_i = next_w_i
    return tuple(real), tuple(imag)


def component_schedule(size: int, *, pi_value: float) -> tuple[tuple[str, ...], ...]:
    components: list[tuple[str, ...]] = []
    for index in range(size):
        components.append(("copy", f"copy{index:02d}", str(index)))
    for stage in range(1, fft_stage_count(size) + 1):
        m = 1 << stage
        half = m // 2
        angle = f32(f32(-2.0) * f32(pi_value / f32(float(m))))
        wm_r = f32(math.cos(angle))
        wm_i = f32(math.sin(angle))
        for base in range(0, size, m):
            w_r = f32(1.0)
            w_i = f32(0.0)
            for offset in range(half):
                components.append(
                    (
                        "butterfly",
                        f"stage{stage:02d}-k{base:02d}-j{offset:02d}",
                        str(base),
                        str(half),
                        str(offset),
                        str(offset + 1),
                        float_arg(f32(-wm_i)),
                        float_arg(wm_r),
                        float_arg(wm_i),
                        float_arg(w_r),
                        float_arg(w_i),
                    )
                )
                next_w_r = f32(f32(w_r * wm_r) - f32(w_i * wm_i))
                next_w_i = f32(f32(w_r * wm_i) + f32(w_i * wm_r))
                w_r = next_w_r
                w_i = next_w_i
    return tuple(components)


def fixture_from_source(source: Path | None = None) -> FftButterflyFixture:
    source = source or source_path()
    text = source.read_text()
    size = parse_uint_const(text, "kSize")
    require(size == 16, "fft_butterfly fixture expects kSize 16")
    pi_value = parse_float_const(text, "kPi")
    input_real = parse_float_array(text, "kInputReal")
    input_imag = parse_float_array(text, "kInputImag")
    require(len(input_real) == size, "kInputReal length mismatch")
    require(len(input_imag) == size, "kInputImag length mismatch")
    expected_real, expected_imag = reference_fft(input_real, input_imag, pi_value=pi_value)
    return FftButterflyFixture(
        size=size,
        input_real=input_real,
        input_imag=input_imag,
        expected_real=expected_real,
        expected_imag=expected_imag,
        components=component_schedule(size, pi_value=pi_value),
    )


def float_arg(value: float) -> str:
    return f"{f32(value):.9e}"


def csv_floats(values: tuple[float, ...] | list[float]) -> str:
    return ",".join(float_arg(value) for value in values)


def parse_memory_token(token: str) -> float:
    if ":" not in token:
        raise ValueError(f"bad memory token {token!r}")
    kind, value = token.split(":", 1)
    require(kind == "f32", f"expected f32 token, saw {token!r}")
    return f32(float(value))


def emit_plan(fixture: FftButterflyFixture) -> None:
    print(f"input_real;{csv_floats(fixture.input_real)}")
    print(f"input_imag;{csv_floats(fixture.input_imag)}")
    print(f"output_real;{csv_floats([0.0] * fixture.size)}")
    print(f"output_imag;{csv_floats([0.0] * fixture.size)}")
    for component in fixture.components:
        print(";".join(component))


def emit_memory_csv(report: Path, key: str) -> None:
    data = json.loads(report.read_text())
    memory = data.get("final_memory_state")
    require(isinstance(memory, dict), f"{report} lacks final_memory_state")
    values = memory.get(key)
    require(isinstance(values, list), f"{report} lacks final_memory_state.{key}")
    parsed: list[float] = []
    for value in values:
        require(isinstance(value, str), f"bad memory token in {report}: {value!r}")
        parsed.append(parse_memory_token(value))
    print(csv_floats(parsed))


def emit_json(fixture: FftButterflyFixture) -> None:
    print(
        json.dumps(
            {
                "size": fixture.size,
                "input_real": fixture.input_real,
                "input_imag": fixture.input_imag,
                "expected_real": fixture.expected_real,
                "expected_imag": fixture.expected_imag,
                "expected_checksum": fixture.expected_checksum,
                "component_count": len(fixture.components),
                "components": fixture.components,
            },
            sort_keys=True,
        )
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--emit", choices=("json", "plan", "memory-csv"), default="json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--memory-key")
    args = parser.parse_args(argv)

    if args.emit == "memory-csv":
        require(args.report is not None, "--report is required for memory-csv")
        require(args.memory_key is not None, "--memory-key is required for memory-csv")
        emit_memory_csv(args.report, args.memory_key)
        return 0

    fixture = fixture_from_source(args.source)
    if args.emit == "plan":
        emit_plan(fixture)
    else:
        emit_json(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
