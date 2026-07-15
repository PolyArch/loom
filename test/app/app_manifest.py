#!/usr/bin/env python3
"""Validate and query the app corpus manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "test" / "app"
DEFAULT_MANIFEST = APP_ROOT / "manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"
VALID_LANGUAGES = {"c", "cxx"}
VALID_TIERS = {"run", "raise", "dfg"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    list_cases = sub.add_parser("list")
    list_cases.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    list_cases.add_argument("--tier", choices=sorted(VALID_TIERS), required=True)
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def existing_app_cases(app_root: Path = APP_ROOT) -> set[str]:
    return {
        path.name
        for path in app_root.iterdir()
        if path.is_dir()
        and (path / "expected.txt").is_file()
        and any(
            child.is_file() and child.suffix in {".c", ".cc", ".cpp", ".cxx"}
            for child in path.iterdir()
        )
    }


def require(condition: bool, message: str, diagnostics: list[str]) -> None:
    if not condition:
        diagnostics.append(message)


def is_file_name(value: str) -> bool:
    return value not in {".", ".."} and Path(value).name == value


def non_empty_string_field(
    entry: dict[object, object], field: str, context: str, diagnostics: list[str]
) -> str | None:
    value = entry.get(field)
    if not isinstance(value, str) or value == "":
        diagnostics.append(f"{context}: {field} must be a non-empty string")
        return None
    return value


def non_empty_string_list_field(
    entry: dict[object, object],
    field: str,
    context: str,
    diagnostics: list[str],
) -> list[str]:
    value = entry.get(field)
    if not isinstance(value, list) or not value:
        diagnostics.append(f"{context}: {field} must be a non-empty list")
        return []
    strings: list[str] = []
    has_invalid = False
    for item in value:
        if isinstance(item, str) and item != "":
            strings.append(item)
        else:
            has_invalid = True
    if has_invalid:
        diagnostics.append(f"{context}: {field} must contain non-empty strings")
    return strings


def string_list_field(
    entry: dict[object, object],
    field: str,
    context: str,
    diagnostics: list[str],
) -> list[str]:
    value = entry.get(field)
    if not isinstance(value, list):
        diagnostics.append(f"{context}: {field} must be a list")
        return []
    strings: list[str] = []
    has_invalid = False
    for item in value:
        if isinstance(item, str):
            strings.append(item)
        else:
            has_invalid = True
    if has_invalid:
        diagnostics.append(f"{context}: {field} must contain strings")
    return strings


def validate_manifest(path: Path) -> tuple[dict[str, object], list[str]]:
    diagnostics: list[str] = []
    app_root = path.parent
    if not path.is_file():
        return {}, [f"missing manifest: {path}"]
    try:
        data = load_manifest(path)
    except json.JSONDecodeError as exc:
        return {}, [f"manifest is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return {}, ["manifest root must be an object"]

    require(
        data.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        f'schema_version must be "{MANIFEST_SCHEMA_VERSION}"',
        diagnostics,
    )
    cases = data.get("cases")
    require(isinstance(cases, list), "cases must be a list", diagnostics)
    if not isinstance(cases, list):
        return data, diagnostics
    require(bool(cases), "manifest must list at least one case", diagnostics)

    seen: set[str] = set()
    for index, entry in enumerate(cases):
        if not isinstance(entry, dict):
            diagnostics.append(f"case entry {index} must be an object")
            continue
        case = non_empty_string_field(entry, "case", f"case entry {index}", diagnostics)
        context = case if case else f"case entry {index}"
        if case:
            require(case not in seen, f"duplicate case: {case}", diagnostics)
            seen.add(case)
        case_dir = app_root / context
        if case:
            require(case_dir.is_dir(), f"{case}: missing case directory", diagnostics)

        language = entry.get("language")
        require(
            language in VALID_LANGUAGES,
            f"{context}: invalid language {language!r}",
            diagnostics,
        )
        if "dfg_symbol" in entry:
            non_empty_string_field(entry, "dfg_symbol", context, diagnostics)

        raw_sources = entry.get("sources")
        sources = non_empty_string_list_field(entry, "sources", context, diagnostics)
        duplicate_sources = sorted(
            source for source, count in Counter(sources).items() if count > 1
        )
        for source in duplicate_sources:
            diagnostics.append(f"{context}: duplicate source: {source}")
        for source in sources:
            if not is_file_name(source):
                continue
            source_path = case_dir / source
            require(
                source_path.is_file(),
                f"{context}: missing source {source}",
                diagnostics,
            )
        require(
            all(is_file_name(source) for source in sources),
            f"{context}: sources entries must be file names",
            diagnostics,
        )

        expected = non_empty_string_field(
            entry, "expected_stdout", context, diagnostics
        )
        if expected:
            if is_file_name(expected):
                require(
                    (case_dir / expected).is_file(),
                    f"{context}: missing expected stdout {expected}",
                    diagnostics,
                )
            else:
                diagnostics.append(f"{context}: expected_stdout must be a file name")

        tiers = non_empty_string_list_field(entry, "tiers", context, diagnostics)
        invalid = sorted(set(tiers) - VALID_TIERS)
        require(not invalid, f"{context}: invalid tiers {invalid}", diagnostics)
        string_list_field(entry, "compiler_flags", context, diagnostics)
        string_list_field(entry, "link_flags", context, diagnostics)
        raw_expected_executables = entry.get("expected_executables")
        non_empty_string_list_field(
            entry,
            "expected_executables",
            context,
            diagnostics,
        )
        if isinstance(raw_sources, list) and isinstance(raw_expected_executables, list):
            require(
                len(raw_sources) == len(raw_expected_executables),
                f"{context}: sources and expected_executables must have equal length",
                diagnostics,
            )
        non_empty_string_list_field(entry, "feature_tags", context, diagnostics)

    discovered = existing_app_cases(app_root)
    omitted = sorted(discovered - seen)
    for case in omitted:
        diagnostics.append(f"{case}: existing app case omitted from manifest")
    extra = sorted(seen - discovered)
    for case in extra:
        diagnostics.append(f"{case}: manifest references missing app case")

    return data, diagnostics


def command_validate(path: Path) -> int:
    _, diagnostics = validate_manifest(path)
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    return 0


def command_list(path: Path, tier: str) -> int:
    data, diagnostics = validate_manifest(path)
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    cases = data["cases"]
    assert isinstance(cases, list)
    for entry in cases:
        assert isinstance(entry, dict)
        tiers = entry.get("tiers", [])
        if tier in tiers:
            print(entry["case"])
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = Path(args.manifest)
    if args.command == "validate":
        return command_validate(manifest)
    if args.command == "list":
        return command_list(manifest, args.tier)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
