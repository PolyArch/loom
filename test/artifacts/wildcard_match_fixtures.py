#!/usr/bin/env python3
"""Source-derived fixtures for wildcard_match app CGRA evidence."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_NAMES = ("match", "no_match", "all_wildcards")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def default_source() -> Path:
    return REPO_ROOT / "test" / "app" / "wildcard_match" / "main_func.cpp"


def parse_uint_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{re.escape(name)}\s*=\s*(\d+)\s*;", text)
    require(match is not None, f"missing {name} in wildcard_match source")
    return int(match.group(1))


def parse_char_const(text: str, name: str) -> int:
    match = re.search(rf"constexpr\s+uint32_t\s+{re.escape(name)}\s*=\s*'(.?)'\s*;", text)
    require(match is not None, f"missing {name} character constant in wildcard_match source")
    return ord(match.group(1))


def parse_function_body(text: str, name: str) -> str:
    match = re.search(rf"void\s+{re.escape(name)}\s*\([^)]*\)\s*{{", text)
    require(match is not None, f"missing {name} in wildcard_match source")
    start = match.end()
    depth = 1
    index = start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
        index += 1
    raise AssertionError(f"unterminated {name} in wildcard_match source")


def parse_scalar_expr(expr: str, *, wildcard: int) -> int:
    expr = expr.strip()
    char_match = re.fullmatch(r"'(.)'", expr)
    if char_match is not None:
        return ord(char_match.group(1))
    if expr == "kWildcard":
        return wildcard
    decimal_match = re.fullmatch(r"\d+", expr)
    if decimal_match is not None:
        return int(expr)
    raise AssertionError(f"unsupported wildcard_match fixture scalar expression: {expr}")


def parse_array_initializer(expr: str, *, wildcard: int) -> list[int]:
    return [parse_scalar_expr(token, wildcard=wildcard) for token in expr.split(",")]


def apply_loop_fill(
    target: list[int | None],
    expr: str,
    *,
    wildcard: int,
) -> None:
    expr = expr.strip()
    ramp_match = re.fullmatch(r"static_cast<uint32_t>\('(.?)'\)\s*\+\s*\(i\s*%\s*(\d+)\)", expr)
    if ramp_match is not None:
        base = ord(ramp_match.group(1))
        period = int(ramp_match.group(2))
        for index in range(len(target)):
            target[index] = base + (index % period)
        return

    value = parse_scalar_expr(expr, wildcard=wildcard)
    for index in range(len(target)):
        target[index] = value


def parse_case_arrays(
    text_source: str,
    case_name: str,
    *,
    text_size: int,
    pattern_size: int,
    wildcard: int,
) -> tuple[list[int], list[int]]:
    source_names = {
        "match": "fill_match_case",
        "no_match": "fill_no_match_case",
        "all_wildcards": "fill_all_wildcards_case",
    }
    body = parse_function_body(text_source, source_names[case_name])
    text_values: list[int | None] = [None] * text_size
    pattern_values: list[int | None] = [None] * pattern_size

    loop_pattern = re.compile(
        r"for\s*\(\s*uint32_t\s+i\s*=\s*0;\s*i\s*<\s*(kTextSize|kPatternSize)\s*;\s*\+\+i\s*\)\s*"
        r"{\s*(text|pattern)\[i\]\s*=\s*(.*?);\s*}",
        re.S,
    )
    for limit_name, target_name, expr in loop_pattern.findall(body):
        target = text_values if target_name == "text" else pattern_values
        expected_limit = "kTextSize" if target_name == "text" else "kPatternSize"
        require(limit_name == expected_limit, f"{target_name} loop uses unexpected limit {limit_name}")
        apply_loop_fill(target, expr, wildcard=wildcard)

    for initializer in re.findall(r"pattern\s*=\s*{([^}]*)};", body, re.S):
        values = parse_array_initializer(initializer, wildcard=wildcard)
        require(len(values) == pattern_size, f"pattern initializer length {len(values)} != {pattern_size}")
        pattern_values = list(values)

    assignment_pattern = re.compile(r"(text|pattern)\[(\d+)\]\s*=\s*(.*?);")
    for target_name, index_text, expr in assignment_pattern.findall(body):
        index = int(index_text)
        target = text_values if target_name == "text" else pattern_values
        require(0 <= index < len(target), f"{target_name}[{index}] is outside fixture bounds")
        target[index] = parse_scalar_expr(expr, wildcard=wildcard)

    require(all(value is not None for value in text_values), f"{case_name} text fixture is incomplete")
    require(all(value is not None for value in pattern_values), f"{case_name} pattern fixture is incomplete")
    return [int(value) for value in text_values], [int(value) for value in pattern_values]


def token_i32(values: list[int]) -> list[str]:
    return [f"i32:{value}" for value in values]


def csv(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


@dataclass(frozen=True)
class WildcardMatchFixture:
    case_name: str
    text_size: int
    pattern_size: int
    wildcard: int
    text: tuple[int, ...]
    pattern: tuple[int, ...]
    expected_match: int

    @property
    def final_outputs(self) -> list[str]:
        return ["none"]

    @property
    def expected_memory(self) -> dict[str, list[str]]:
        return {
            "arg1": token_i32(list(self.text)),
            "arg2": token_i32(list(self.pattern)),
            "arg3": [f"i32:{self.expected_match}"],
        }

    def dfg_argv(self) -> list[str]:
        return [
            "--arg",
            "0=none",
            "--memref",
            f"1={csv(list(self.text))}",
            "--memref",
            f"2={csv(list(self.pattern))}",
            "--memref",
            "3=0",
            "--arg",
            f"4={self.text_size}",
            "--arg",
            f"5={self.pattern_size}",
        ]


def wildcard_match(text: list[int], pattern: list[int], wildcard: int) -> int:
    if len(pattern) > len(text):
        return 0
    for start in range(len(text) - len(pattern) + 1):
        match = True
        for offset, pattern_value in enumerate(pattern):
            if pattern_value != wildcard and text[start + offset] != pattern_value:
                match = False
                break
        if match:
            return 1
    return 0


def fixture_from_source(case_name: str, source: Path | None = None) -> WildcardMatchFixture:
    require(case_name in CASE_NAMES, f"unknown wildcard_match fixture case: {case_name}")
    source = source or default_source()
    text_source = source.read_text()
    text_size = parse_uint_const(text_source, "kTextSize")
    pattern_size = parse_uint_const(text_source, "kPatternSize")
    wildcard = parse_char_const(text_source, "kWildcard")

    text, pattern = parse_case_arrays(
        text_source,
        case_name,
        text_size=text_size,
        pattern_size=pattern_size,
        wildcard=wildcard,
    )

    require(len(text) == text_size, f"text length {len(text)} != kTextSize {text_size}")
    require(len(pattern) == pattern_size, f"pattern length {len(pattern)} != kPatternSize {pattern_size}")
    expected = wildcard_match(text, pattern, wildcard)
    if case_name == "no_match":
        require(expected == 0, "no_match fixture must remain non-matching")
    else:
        require(expected == 1, f"{case_name} fixture must remain matching")
    return WildcardMatchFixture(
        case_name=case_name,
        text_size=text_size,
        pattern_size=pattern_size,
        wildcard=wildcard,
        text=tuple(text),
        pattern=tuple(pattern),
        expected_match=expected,
    )


def expected_json(fixture: WildcardMatchFixture) -> dict[str, object]:
    return {
        "case_name": fixture.case_name,
        "final_outputs": fixture.final_outputs,
        "expected_memory": fixture.expected_memory,
        "expected_match": fixture.expected_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--emit", choices=("case-names", "dfg-args", "expected-json"), required=True)
    args = parser.parse_args()

    if args.emit == "case-names":
        print("\n".join(CASE_NAMES))
        return 0

    require(args.case is not None, "--case is required for this emit mode")
    fixture = fixture_from_source(args.case, args.source)
    if args.emit == "dfg-args":
        print("\n".join(fixture.dfg_argv()))
    elif args.emit == "expected-json":
        print(json.dumps(expected_json(fixture), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
