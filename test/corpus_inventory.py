#!/usr/bin/env python3
"""Enumerate and select Loom's canonical high-level source corpus."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "test" / "app"
sys.path.insert(0, str(APP_ROOT))

import app_manifest  # noqa: E402


SUITE_ORDER = ("loombench", "cmsis-dsp", "cmsis-nn")
CMSIS_SUBMODULES = {
    "cmsis-dsp": Path("externals/cmsis-dsp"),
    "cmsis-nn": Path("externals/cmsis-nn"),
}


class InventoryError(ValueError):
    """Raised when canonical corpus membership cannot be enumerated."""


@dataclass(frozen=True)
class CorpusCase:
    suite: str
    case: str
    sources: tuple[str, ...]

    @property
    def identity(self) -> str:
        return f"{self.suite}:{self.case}"

    def as_dict(self) -> dict[str, object]:
        return {
            "case": self.case,
            "identity": self.identity,
            "sources": list(self.sources),
            "suite": self.suite,
        }


def run_git(arguments: Sequence[str], cwd: Path) -> bytes:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise InventoryError(f"cannot run git in {cwd}: {exc}") from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode(errors="replace").strip()
        raise InventoryError(f"git {' '.join(arguments)} failed in {cwd}: {diagnostic}")
    return completed.stdout


def git_text(arguments: Sequence[str], cwd: Path) -> str:
    return run_git(arguments, cwd).decode(errors="replace").strip()


def load_loombench(repo_root: Path) -> list[CorpusCase]:
    manifest_path = repo_root / "test" / "app" / "manifest.json"
    data, diagnostics = app_manifest.validate_manifest(manifest_path)
    if diagnostics:
        raise InventoryError("invalid LoomBench manifest:\n" + "\n".join(diagnostics))
    entries = data.get("cases")
    if not isinstance(entries, list):
        raise InventoryError("invalid LoomBench manifest: cases must be a list")

    cases: list[CorpusCase] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise InventoryError(
                "invalid LoomBench manifest: case entry must be an object"
            )
        case = entry.get("case")
        sources = entry.get("sources")
        if not isinstance(case, str) or not isinstance(sources, list):
            raise InventoryError("invalid LoomBench manifest case identity")
        cases.append(
            CorpusCase(
                suite="loombench",
                case=case,
                sources=tuple(
                    f"test/app/{case}/{source}"
                    for source in sources
                    if isinstance(source, str)
                ),
            )
        )
    return cases


def require_pinned_submodule(repo_root: Path, relative_path: Path) -> tuple[Path, str]:
    submodule = (repo_root / relative_path).resolve()
    if not submodule.is_dir():
        raise InventoryError(
            f"missing submodule checkout: {relative_path}; run git submodule update --init"
        )

    actual_root = Path(git_text(["rev-parse", "--show-toplevel"], submodule)).resolve()
    if actual_root != submodule:
        raise InventoryError(
            f"submodule is not initialized: {relative_path}; run git submodule update --init"
        )

    pinned_revision = git_text(
        ["rev-parse", f"HEAD:{relative_path.as_posix()}"], repo_root
    )
    checked_out_revision = git_text(["rev-parse", "HEAD"], submodule)
    if checked_out_revision != pinned_revision:
        raise InventoryError(
            f"submodule revision mismatch for {relative_path}: "
            f"expected {pinned_revision}, found {checked_out_revision}"
        )
    return submodule, pinned_revision


def tracked_c_sources_at_revision(submodule: Path, revision: str) -> tuple[str, ...]:
    tracked = run_git(
        ["ls-tree", "-r", "-z", "--name-only", revision, "--", "Source"],
        submodule,
    )
    source_paths = sorted(
        path
        for path in tracked.decode(errors="surrogateescape").split("\0")
        if path.startswith("Source/") and path.endswith(".c")
    )
    if len(source_paths) != len(set(source_paths)):
        raise InventoryError("commit tree returned duplicate source paths")
    return tuple(source_paths)


def load_cmsis_suite(repo_root: Path, suite: str) -> list[CorpusCase]:
    relative_path = CMSIS_SUBMODULES[suite]
    submodule, pinned_revision = require_pinned_submodule(repo_root, relative_path)
    source_paths = tracked_c_sources_at_revision(submodule, pinned_revision)
    if not source_paths:
        raise InventoryError(f"{suite} has no tracked C sources under Source")

    prefix = relative_path.as_posix()
    return [
        CorpusCase(
            suite=suite,
            case=source.removeprefix("Source/"),
            sources=(f"{prefix}/{source}",),
        )
        for source in source_paths
    ]


def load_inventory(repo_root: Path = ROOT) -> tuple[CorpusCase, ...]:
    root = repo_root.resolve()
    cases = load_loombench(root)
    for suite in SUITE_ORDER[1:]:
        cases.extend(load_cmsis_suite(root, suite))

    order = {suite: index for index, suite in enumerate(SUITE_ORDER)}
    cases.sort(key=lambda case: (order[case.suite], case.case))
    require_unique_inventory(cases)
    return tuple(cases)


def require_unique_inventory(cases: Sequence[CorpusCase]) -> None:
    identities = [case.identity for case in cases]
    duplicates = sorted(
        identity for identity, count in Counter(identities).items() if count > 1
    )
    if duplicates:
        raise InventoryError(f"duplicate corpus identity: {', '.join(duplicates)}")

    sources = [source for case in cases for source in case.sources]
    duplicate_sources = sorted(
        source for source, count in Counter(sources).items() if count > 1
    )
    if duplicate_sources:
        raise InventoryError(
            f"duplicate corpus source identity: {', '.join(duplicate_sources)}"
        )


def select_cases(
    cases: Sequence[CorpusCase],
    *,
    suite_names: Sequence[str],
    case_ids: Sequence[str],
) -> tuple[CorpusCase, ...]:
    duplicate_suites = sorted(
        suite for suite, count in Counter(suite_names).items() if count > 1
    )
    if duplicate_suites:
        raise InventoryError(f"duplicate suite selector: {', '.join(duplicate_suites)}")
    unknown_suites = sorted(set(suite_names) - set(SUITE_ORDER))
    if unknown_suites:
        raise InventoryError(f"unknown suite selector: {', '.join(unknown_suites)}")

    duplicate_cases = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )
    if duplicate_cases:
        raise InventoryError(f"duplicate case selector: {', '.join(duplicate_cases)}")

    by_identity = {case.identity: case for case in cases}
    unknown_cases = [case_id for case_id in case_ids if case_id not in by_identity]
    if unknown_cases:
        raise InventoryError(f"unknown case selector: {', '.join(unknown_cases)}")

    suites = set(suite_names)
    requested = set(case_ids)
    selected = tuple(
        case
        for case in cases
        if (not suites or case.suite in suites)
        and (not requested or case.identity in requested)
    )
    if case_ids and len(selected) != len(case_ids):
        excluded = [
            case_id for case_id in case_ids if by_identity[case_id] not in selected
        ]
        raise InventoryError(
            f"case selector excluded by suite selection: {', '.join(excluded)}"
        )
    if not selected:
        raise InventoryError("corpus selection is empty")
    return selected


def render_json(cases: Sequence[CorpusCase]) -> str:
    require_unique_inventory(cases)
    counts = Counter(case.suite for case in cases)
    payload = {
        "case_count": len(cases),
        "cases": [case.as_dict() for case in cases],
        "suite_counts": {
            suite: counts[suite] for suite in SUITE_ORDER if counts[suite]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_smoke_sources(
    path: Path, suite_cases: Sequence[CorpusCase]
) -> tuple[str, ...]:
    if not path.is_file():
        raise InventoryError(f"missing smoke target table: {path}")
    suites = {case.suite for case in suite_cases}
    if len(suites) != 1:
        raise InventoryError("smoke validation requires one non-empty suite inventory")
    suite = next(iter(suites))
    members = {case.case for case in suite_cases}

    sources: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.removesuffix("\r")
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 5 or not all(fields[:4]):
            raise InventoryError(f"{path}:{line_number}: malformed smoke target row")
        source = fields[0]
        if source in sources:
            raise InventoryError(f"duplicate smoke source: {source}")
        if source not in members:
            raise InventoryError(f"{source} is not in the {suite} inventory")
        sources.append(source)

    if not sources:
        raise InventoryError(f"smoke target table contains no sources: {path}")
    if set(sources) == members:
        raise InventoryError(f"{suite} smoke targets must be a strict subset")
    return tuple(sources)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    list_command = commands.add_parser("list", help="emit selected inventory as JSON")
    list_command.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=SUITE_ORDER,
        dest="suites",
    )
    list_command.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        metavar="SUITE:CASE",
    )

    validate_smoke = commands.add_parser(
        "validate-smoke", help="verify a CMSIS smoke table is an inventory subset"
    )
    validate_smoke.add_argument(
        "--suite", required=True, choices=tuple(CMSIS_SUBMODULES)
    )
    validate_smoke.add_argument("--targets", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    try:
        inventory = load_inventory(ROOT)
        if args.command == "list":
            selected = select_cases(
                inventory,
                suite_names=args.suites,
                case_ids=args.cases,
            )
            sys.stdout.write(render_json(selected))
            return 0
        if args.command == "validate-smoke":
            suite_cases = [case for case in inventory if case.suite == args.suite]
            load_smoke_sources(args.targets, suite_cases)
            return 0
    except InventoryError as exc:
        print(f"corpus inventory: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
