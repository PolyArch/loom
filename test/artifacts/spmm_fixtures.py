#!/usr/bin/env python3
"""Source-derived fixtures for spmm app CGRA evidence."""

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
    return REPO_ROOT / "test" / "app" / "spmm" / "main_func.cpp"


def parse_uint_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{re.escape(name)}\s*=\s*(\d+)\s*;", text)
    require(match is not None, f"missing {name} in spmm source")
    return int(match.group(1))


def parse_u32_array(text: str, name: str) -> list[int]:
    match = re.search(
        rf"constexpr\s+std::array<uint32_t,\s*[^>]+>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
        text,
        re.S,
    )
    require(match is not None, f"missing {name} initializer in spmm source")
    values = [int(value) for value in re.findall(r"\d+", match.group("body"))]
    require(values, f"{name} initializer must contain values")
    return values


def token_i32(values: list[int]) -> list[str]:
    return [f"i32:{value}" for value in values]


def csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


@dataclass(frozen=True)
class SpmmFixture:
    rows: int
    cols: int
    out_cols: int
    nnz: int
    values: tuple[int, ...]
    col_indices: tuple[int, ...]
    row_ptr: tuple[int, ...]
    dense: tuple[int, ...]
    expected: tuple[int, ...]

    @property
    def final_outputs(self) -> list[str]:
        return ["none"]

    @property
    def expected_memory(self) -> dict[str, list[str]]:
        return {
            "arg1": token_i32(list(self.values)),
            "arg2": token_i32(list(self.col_indices)),
            "arg3": token_i32(list(self.row_ptr)),
            "arg4": token_i32(list(self.dense)),
            "arg5": token_i32(list(self.expected)),
        }

    @property
    def expected_fire_counts(self) -> dict[str, int]:
        return {
            "arith.addi": 36,
            "arith.cmpi": 7,
            "arith.index_cast": 88,
            "arith.muli": 34,
            "dataflow.constant": 9,
            "dataflow.load": 28,
            "dataflow.store": 12,
            "llvm.trunc": 18,
            "llvm.zext": 6,
            "scf.if": 8,
        }

    def dfg_argv(self) -> list[str]:
        zeros = [0] * (self.rows * self.out_cols)
        return [
            "--arg",
            "0=none",
            "--memref",
            f"1={csv(list(self.values))}",
            "--memref",
            f"2={csv(list(self.col_indices))}",
            "--memref",
            f"3={csv(list(self.row_ptr))}",
            "--memref",
            f"4={csv(list(self.dense))}",
            "--memref",
            f"5={csv(zeros)}",
            "--arg",
            f"6={self.rows}",
            "--arg",
            f"7={self.out_cols}",
        ]


def fixture_from_source(source: Path | None = None) -> SpmmFixture:
    source = source or default_source()
    text = source.read_text()
    rows = parse_uint_const(text, "kRows")
    cols = parse_uint_const(text, "kCols")
    out_cols = parse_uint_const(text, "kOutCols")
    nnz = parse_uint_const(text, "kNnz")
    values = parse_u32_array(text, "kValues")
    col_indices = parse_u32_array(text, "kColIndices")
    row_ptr = parse_u32_array(text, "kRowPtr")
    dense = parse_u32_array(text, "kDense")
    expected = parse_u32_array(text, "kExpected")

    require(len(values) == nnz, f"kValues length {len(values)} != kNnz {nnz}")
    require(len(col_indices) == nnz, f"kColIndices length {len(col_indices)} != kNnz {nnz}")
    require(len(row_ptr) == rows + 1, f"kRowPtr length {len(row_ptr)} != kRows+1 {rows + 1}")
    require(len(dense) == cols * out_cols, f"kDense length {len(dense)} != kCols*kOutCols {cols * out_cols}")
    require(
        len(expected) == rows * out_cols,
        f"kExpected length {len(expected)} != kRows*kOutCols {rows * out_cols}",
    )
    computed = [0] * (rows * out_cols)
    for row in range(rows):
        begin = row_ptr[row]
        end = row_ptr[row + 1]
        require(0 <= begin <= end <= nnz, f"bad CSR row range {row}: {begin}..{end}")
        for idx in range(begin, end):
            dense_row = col_indices[idx]
            require(dense_row < cols, f"kColIndices[{idx}]={dense_row} exceeds kCols {cols}")
            for col in range(out_cols):
                computed[row * out_cols + col] += values[idx] * dense[dense_row * out_cols + col]
    require(computed == expected, f"source kExpected does not match computed SpMM: {computed} != {expected}")
    require(computed == [11, 14, 29, 36], f"spmm fixture should keep the legacy row output, got {computed}")

    return SpmmFixture(
        rows=rows,
        cols=cols,
        out_cols=out_cols,
        nnz=nnz,
        values=tuple(values),
        col_indices=tuple(col_indices),
        row_ptr=tuple(row_ptr),
        dense=tuple(dense),
        expected=tuple(expected),
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
