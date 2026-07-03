#!/usr/bin/env python3
"""Source-derived fixtures for ifft_butterfly app CGRA evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fft_butterfly_fixtures as fft


def source_path() -> Path:
    return fft.REPO_ROOT / "test" / "app" / "ifft_butterfly" / "main_func.cpp"


def fixture_from_source(source: Path | None = None) -> fft.FftButterflyFixture:
    source = source or source_path()
    text = source.read_text()
    size = fft.parse_uint_const(text, "kSize")
    fft.require(size == 16, "ifft_butterfly fixture expects kSize 16")
    pi_value = fft.parse_float_const(text, "kPi")
    input_real = fft.parse_float_array(text, "kInputReal")
    input_imag = fft.parse_float_array(text, "kInputImag")
    fft.require(len(input_real) == size, "kInputReal length mismatch")
    fft.require(len(input_imag) == size, "kInputImag length mismatch")

    expected_real, expected_imag = fft.reference_fft(input_real, input_imag, pi_value=pi_value)
    scale = fft.f32(1.0 / float(size))
    expected_real = tuple(fft.f32(value * scale) for value in expected_real)
    expected_imag = tuple(fft.f32(fft.f32(-value) * scale) for value in expected_imag)
    components = list(fft.component_schedule(size, pi_value=pi_value))
    for index in range(size):
        components.append(("scale", f"scale{index:02d}", str(index), fft.float_arg(scale)))

    return fft.FftButterflyFixture(
        size=size,
        input_real=input_real,
        input_imag=input_imag,
        expected_real=expected_real,
        expected_imag=expected_imag,
        components=tuple(components),
    )


def emit_plan(fixture: fft.FftButterflyFixture) -> None:
    print(f"input_real;{fft.csv_floats(fixture.input_real)}")
    print(f"input_imag;{fft.csv_floats(fixture.input_imag)}")
    print(f"output_real;{fft.csv_floats([0.0] * fixture.size)}")
    print(f"output_imag;{fft.csv_floats([0.0] * fixture.size)}")
    for component in fixture.components:
        print(";".join(component))


def emit_json(fixture: fft.FftButterflyFixture) -> None:
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
        fft.require(args.report is not None, "--report is required for memory-csv")
        fft.require(args.memory_key is not None, "--memory-key is required for memory-csv")
        fft.emit_memory_csv(args.report, args.memory_key)
        return 0

    fixture = fixture_from_source(args.source)
    if args.emit == "plan":
        emit_plan(fixture)
    else:
        emit_json(fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
