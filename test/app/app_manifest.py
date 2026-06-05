#!/usr/bin/env python3
"""Validate and query the app corpus manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "test" / "app"
DEFAULT_MANIFEST = APP_ROOT / "manifest.json"
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


def existing_app_cases() -> set[str]:
    return {
        path.name
        for path in APP_ROOT.iterdir()
        if path.is_dir() and (path / "run_check.sh").is_file()
    }


def require(condition: bool, message: str, diagnostics: list[str]) -> None:
    if not condition:
        diagnostics.append(message)


def validate_manifest(path: Path) -> tuple[dict[str, object], list[str]]:
    diagnostics: list[str] = []
    if not path.is_file():
        return {}, [f"missing manifest: {path}"]
    try:
        data = load_manifest(path)
    except json.JSONDecodeError as exc:
        return {}, [f"manifest is not valid JSON: {exc}"]

    require(data.get("schema_version") == 1, "schema_version must be 1", diagnostics)
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
        case = str(entry.get("case", ""))
        require(case != "", f"case entry {index} has blank case name", diagnostics)
        require(case not in seen, f"duplicate case: {case}", diagnostics)
        seen.add(case)
        case_dir = APP_ROOT / case
        require(case_dir.is_dir(), f"{case}: missing case directory", diagnostics)

        language = entry.get("language")
        require(language in VALID_LANGUAGES, f"{case}: invalid language {language!r}", diagnostics)

        sources = entry.get("sources")
        require(isinstance(sources, list) and bool(sources), f"{case}: sources must be a non-empty list", diagnostics)
        if isinstance(sources, list):
            for source in sources:
                source_path = case_dir / str(source)
                require(source_path.is_file(), f"{case}: missing source {source}", diagnostics)

        expected = entry.get("expected_stdout")
        require(isinstance(expected, str) and expected != "", f"{case}: expected_stdout is required", diagnostics)
        if isinstance(expected, str) and expected:
            require((case_dir / expected).is_file(), f"{case}: missing expected stdout {expected}", diagnostics)

        tiers = entry.get("tiers")
        require(isinstance(tiers, list) and bool(tiers), f"{case}: tiers must be a non-empty list", diagnostics)
        if isinstance(tiers, list):
            invalid = sorted(set(str(tier) for tier in tiers) - VALID_TIERS)
            require(not invalid, f"{case}: invalid tiers {invalid}", diagnostics)
            for tier in tiers:
                script = {"run": "run_check.sh", "raise": "raise_check.sh", "dfg": "dfg_check.sh"}.get(str(tier))
                if script:
                    require((case_dir / script).is_file(), f"{case}: missing {script}", diagnostics)

        tags = entry.get("feature_tags")
        require(isinstance(tags, list) and bool(tags), f"{case}: feature_tags must be a non-empty list", diagnostics)

    omitted = sorted(existing_app_cases() - seen)
    for case in omitted:
        diagnostics.append(f"{case}: existing app case omitted from manifest")
    extra = sorted(seen - existing_app_cases())
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
