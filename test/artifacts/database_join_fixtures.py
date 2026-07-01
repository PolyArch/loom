#!/usr/bin/env python3
"""Source-derived fixtures for database_join app CGRA evidence."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def default_source() -> Path:
    return REPO_ROOT / "test" / "app" / "database_join" / "main_func.cpp"


def parse_uint_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{re.escape(name)}\s*=\s*(\d+)\s*;", text)
    require(match is not None, f"missing {name} in database_join source")
    return int(match.group(1))


def parse_i32_array(text: str, name: str) -> list[int]:
    match = re.search(
        rf"std::array<int32_t,\s*k[A-Za-z0-9_]+>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
        text,
        re.S,
    )
    require(match is not None, f"missing {name} initializer in database_join source")
    values = [int(value) for value in re.findall(r"[-+]?\d+", match.group("body"))]
    require(values, f"{name} initializer must contain values")
    return values


def token_i32(values: list[int]) -> list[str]:
    return [f"i32:{value}" for value in values]


def csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


@dataclass(frozen=True)
class DatabaseJoinFixture:
    size_a: int
    size_b: int
    max_output: int
    a_ids: tuple[int, ...]
    b_ids: tuple[int, ...]
    a_values: tuple[int, ...]
    b_values: tuple[int, ...]
    output_ids: tuple[int, ...]
    output_a_values: tuple[int, ...]
    output_b_values: tuple[int, ...]

    @property
    def match_count(self) -> int:
        return sum(1 for value in self.output_ids if value != 0)

    @property
    def expected_fire_counts(self) -> dict[str, int]:
        compares = self.size_a * self.size_b
        matches = self.match_count
        return {
            "arith.addi": matches,
            "arith.cmpi": compares,
            "arith.index_cast": 2 * compares + 5 * matches,
            "dataflow.load": 2 * compares + 2 * matches,
            "dataflow.store": 3 * matches,
            "scf.if": compares + self.size_a,
        }

    @property
    def final_outputs(self) -> list[str]:
        return ["none", f"i32:{self.match_count}"]

    @property
    def expected_memory(self) -> dict[str, list[str]]:
        return {
            "arg4": token_i32(list(self.a_ids)),
            "arg5": token_i32(list(self.a_values)),
            "arg6": token_i32(list(self.b_ids)),
            "arg7": token_i32(list(self.output_ids)),
            "arg8": token_i32(list(self.output_a_values)),
            "arg9": token_i32(list(self.b_values)),
            "arg10": token_i32(list(self.output_b_values)),
        }

    def dfg_argv(self) -> list[str]:
        zeros = [0] * self.max_output
        return [
            "--arg",
            "0=none",
            "--arg",
            "1=0",
            "--arg",
            f"2={self.size_a}",
            "--arg",
            "3=1",
            "--memref",
            f"4={csv(list(self.a_ids))}",
            "--memref",
            f"5={csv(list(self.a_values))}",
            "--memref",
            f"6={csv(list(self.b_ids))}",
            "--memref",
            f"7={csv(zeros)}",
            "--memref",
            f"8={csv(zeros)}",
            "--memref",
            f"9={csv(list(self.b_values))}",
            "--memref",
            f"10={csv(zeros)}",
            "--arg",
            "11=1",
            "--arg",
            f"12={self.size_b}",
            "--arg",
            "13=false",
            "--arg",
            "14=0",
        ]


def fixture_from_source(source: Path | None = None) -> DatabaseJoinFixture:
    source = source or default_source()
    text = source.read_text()
    size_a = parse_uint_const(text, "kSizeA")
    size_b = parse_uint_const(text, "kSizeB")
    max_output = parse_uint_const(text, "kMaxOutput")
    a_ids = parse_i32_array(text, "a_ids")
    b_ids = parse_i32_array(text, "b_ids")
    a_values = parse_i32_array(text, "a_values")
    b_values = parse_i32_array(text, "b_values")
    require(len(a_ids) == size_a, f"a_ids length {len(a_ids)} != kSizeA {size_a}")
    require(len(a_values) == size_a, f"a_values length {len(a_values)} != kSizeA {size_a}")
    require(len(b_ids) == size_b, f"b_ids length {len(b_ids)} != kSizeB {size_b}")
    require(len(b_values) == size_b, f"b_values length {len(b_values)} != kSizeB {size_b}")
    output_ids = [0] * max_output
    output_a_values = [0] * max_output
    output_b_values = [0] * max_output
    out_idx = 0
    for i, a_id in enumerate(a_ids):
        for j, b_id in enumerate(b_ids):
            if a_id == b_id:
                require(out_idx < max_output, "database_join fixture output overflow")
                output_ids[out_idx] = a_id
                output_a_values[out_idx] = a_values[i]
                output_b_values[out_idx] = b_values[j]
                out_idx += 1
    require(out_idx == 2, f"database_join fixture should keep the two-match legacy row, got {out_idx}")
    return DatabaseJoinFixture(
        size_a=size_a,
        size_b=size_b,
        max_output=max_output,
        a_ids=tuple(a_ids),
        b_ids=tuple(b_ids),
        a_values=tuple(a_values),
        b_values=tuple(b_values),
        output_ids=tuple(output_ids),
        output_a_values=tuple(output_a_values),
        output_b_values=tuple(output_b_values),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--emit", choices=("dfg-args",), required=True)
    args = parser.parse_args()

    fixture = fixture_from_source(args.source)
    if args.emit == "dfg-args":
        print("\n".join(fixture.dfg_argv()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
