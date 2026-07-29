#!/usr/bin/env python3
"""Enumerate Loom's source and linked-program conformance inventories."""

from __future__ import annotations

import argparse
import json
import os
import re
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
CMSIS_SUPPORT_SUBMODULES = (Path("externals/cmsis-core"),)
CMSIS_WORKLOAD_SUPPORT_SUBMODULES = (
    *CMSIS_SUPPORT_SUBMODULES,
    Path("externals/unity"),
)


class InventoryError(ValueError):
    """Raised when canonical corpus membership cannot be enumerated."""


@dataclass(frozen=True)
class SourceTranslationUnit:
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


@dataclass(frozen=True)
class WorkloadOracle:
    kind: str
    path: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path}


@dataclass(frozen=True)
class WorkloadProducer:
    kind: str
    definition: str
    target: str

    def as_dict(self) -> dict[str, str]:
        return {
            "definition": self.definition,
            "kind": self.kind,
            "target": self.target,
        }


@dataclass(frozen=True)
class ProgramWorkload:
    suite: str
    case: str
    executable: str
    sources: tuple[str, ...]
    entry_symbol: str
    target_profile: str
    oracle: WorkloadOracle
    producer: WorkloadProducer
    compiler_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        return f"{self.suite}:{self.case}/{self.executable}"

    def as_dict(self) -> dict[str, object]:
        return {
            "case": self.case,
            "compiler_flags": list(self.compiler_flags),
            "entry_symbol": self.entry_symbol,
            "executable": self.executable,
            "identity": self.identity,
            "link_flags": list(self.link_flags),
            "oracle": self.oracle.as_dict(),
            "producer": self.producer.as_dict(),
            "sources": list(self.sources),
            "suite": self.suite,
            "target_profile": self.target_profile,
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


def resolve_externals_root(repo_root: Path) -> Path:
    dispatcher = repo_root / "scripts" / "make-worktree.py"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(dispatcher),
                "--root",
                str(repo_root),
                "externals-root",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise InventoryError(
            f"cannot resolve shared external sources: {exc}"
        ) from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()
        raise InventoryError(
            f"cannot resolve shared external sources: {diagnostic or 'unknown error'}"
        )
    resolved = completed.stdout.strip()
    if not resolved:
        raise InventoryError(
            "worktree dispatcher returned an empty external source path"
        )
    return Path(resolved).resolve()


def load_loombench_manifest(repo_root: Path) -> list[dict[str, object]]:
    manifest_path = repo_root / "test" / "app" / "manifest.json"
    data, diagnostics = app_manifest.validate_manifest(manifest_path)
    if diagnostics:
        raise InventoryError("invalid LoomBench manifest:\n" + "\n".join(diagnostics))
    entries = data.get("cases")
    if not isinstance(entries, list):
        raise InventoryError("invalid LoomBench manifest: cases must be a list")

    validated: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise InventoryError(
                "invalid LoomBench manifest: case entry must be an object"
            )
        case = entry.get("case")
        sources = entry.get("sources")
        if not isinstance(case, str) or not isinstance(sources, list):
            raise InventoryError("invalid LoomBench manifest case identity")
        validated.append(entry)
    return validated


def load_loombench_sources(repo_root: Path) -> list[SourceTranslationUnit]:
    cases: list[SourceTranslationUnit] = []
    for entry in load_loombench_manifest(repo_root):
        case = entry["case"]
        sources = entry["sources"]
        assert isinstance(case, str)
        assert isinstance(sources, list)
        cases.append(
            SourceTranslationUnit(
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


def load_loombench_workloads(repo_root: Path) -> list[ProgramWorkload]:
    workloads: list[ProgramWorkload] = []
    for entry in load_loombench_manifest(repo_root):
        case = entry["case"]
        sources = entry["sources"]
        executables = entry["expected_executables"]
        expected_stdout = entry["expected_stdout"]
        compiler_flags = entry["compiler_flags"]
        link_flags = entry["link_flags"]
        assert isinstance(case, str)
        assert isinstance(sources, list)
        assert isinstance(executables, list)
        assert isinstance(expected_stdout, str)
        assert isinstance(compiler_flags, list)
        assert isinstance(link_flags, list)
        for source, executable in zip(sources, executables, strict=True):
            assert isinstance(source, str)
            assert isinstance(executable, str)
            workloads.append(
                ProgramWorkload(
                    suite="loombench",
                    case=case,
                    executable=executable,
                    sources=(f"test/app/{case}/{source}",),
                    entry_symbol="main",
                    target_profile="riscv64-portable-scalar",
                    oracle=WorkloadOracle(
                        kind="expected-stdout",
                        path=f"test/app/{case}/{expected_stdout}",
                    ),
                    producer=WorkloadProducer(
                        kind="direct-source",
                        definition="test/app/manifest.json",
                        target=executable,
                    ),
                    compiler_flags=tuple(compiler_flags),
                    link_flags=tuple(link_flags),
                )
            )
    return workloads


def require_pinned_submodule(
    repo_root: Path, external_root: Path, relative_path: Path
) -> tuple[Path, str]:
    if not relative_path.parts or relative_path.parts[0] != "externals":
        raise InventoryError(f"invalid external submodule path: {relative_path}")
    submodule = external_root.joinpath(*relative_path.parts[1:]).resolve()
    if not submodule.is_dir():
        raise InventoryError(
            f"missing shared submodule checkout: {relative_path}; initialize it only "
            f"in the primary worktree at {external_root}"
        )

    actual_root = Path(git_text(["rev-parse", "--show-toplevel"], submodule)).resolve()
    if actual_root != submodule:
        raise InventoryError(
            f"shared submodule is not initialized: {relative_path}; initialize it only "
            f"in the primary worktree at {external_root}"
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
    dirty = git_text(
        ["status", "--porcelain", "--untracked-files=no"], submodule
    )
    if dirty:
        raise InventoryError(
            f"shared submodule has tracked modifications: {relative_path}"
        )
    return submodule, pinned_revision


def tracked_c_translation_units_at_revision(
    submodule: Path, revision: str
) -> tuple[str, ...]:
    tracked = run_git(
        ["ls-tree", "-r", "-z", "--name-only", revision, "--", "Source"],
        submodule,
    )
    source_paths = sorted(
        path
        for path in tracked.decode(errors="surrogateescape").split("\0")
        if path.startswith("Source/")
        and path.endswith(".c")
        and not Path(path).name.startswith("_")
    )
    if len(source_paths) != len(set(source_paths)):
        raise InventoryError("commit tree returned duplicate source paths")
    return tuple(source_paths)


def load_cmsis_suite(
    repo_root: Path, external_root: Path, suite: str
) -> list[SourceTranslationUnit]:
    relative_path = CMSIS_SUBMODULES[suite]
    submodule, pinned_revision = require_pinned_submodule(
        repo_root, external_root, relative_path
    )
    source_paths = tracked_c_translation_units_at_revision(submodule, pinned_revision)
    if not source_paths:
        raise InventoryError(f"{suite} has no tracked C translation units under Source")

    prefix = relative_path.as_posix()
    return [
        SourceTranslationUnit(
            suite=suite,
            case=source.removeprefix("Source/"),
            sources=(f"{prefix}/{source}",),
        )
        for source in source_paths
    ]


def load_source_inventory(
    repo_root: Path = ROOT,
) -> tuple[SourceTranslationUnit, ...]:
    root = repo_root.resolve()
    external_root = resolve_externals_root(root)
    for relative_path in CMSIS_SUPPORT_SUBMODULES:
        require_pinned_submodule(root, external_root, relative_path)
    cases = load_loombench_sources(root)
    for suite in SUITE_ORDER[1:]:
        cases.extend(load_cmsis_suite(root, external_root, suite))

    order = {suite: index for index, suite in enumerate(SUITE_ORDER)}
    cases.sort(key=lambda case: (order[case.suite], case.case))
    require_unique_inventory(cases)
    return tuple(cases)


def load_workload_inventory(repo_root: Path = ROOT) -> tuple[ProgramWorkload, ...]:
    root = repo_root.resolve()
    external_root = resolve_externals_root(root)
    for relative_path in (
        *CMSIS_WORKLOAD_SUPPORT_SUBMODULES,
        *CMSIS_SUBMODULES.values(),
    ):
        require_pinned_submodule(root, external_root, relative_path)

    workloads = load_loombench_workloads(root)
    workloads.extend(load_cmsis_dsp_workloads(external_root))
    workloads.extend(load_cmsis_nn_workloads(external_root))
    workloads.sort(
        key=lambda workload: (
            SUITE_ORDER.index(workload.suite),
            workload.case,
            workload.executable,
        )
    )
    require_unique_identities(workloads)
    return tuple(workloads)


def load_cmsis_dsp_workloads(external_root: Path) -> list[ProgramWorkload]:
    definitions = (
        ("float16", "desc_f16.txt", "riscv64-standard-float16"),
        ("scalar", "desc.txt", "riscv64-portable-scalar"),
    )
    testing = external_root / "cmsis-dsp" / "Testing"
    workloads: list[ProgramWorkload] = []
    for executable, descriptor, profile in definitions:
        descriptor_path = testing / descriptor
        if not descriptor_path.is_file():
            raise InventoryError(
                f"missing CMSIS-DSP workload descriptor: {descriptor_path}"
            )
        repo_path = f"externals/cmsis-dsp/Testing/{descriptor}"
        workloads.append(
            ProgramWorkload(
                suite="cmsis-dsp",
                case="official-tests",
                executable=executable,
                sources=(),
                entry_symbol="main",
                target_profile=profile,
                oracle=WorkloadOracle(
                    kind="cmsis-dsp-patterns",
                    path=repo_path,
                ),
                producer=WorkloadProducer(
                    kind="cmsis-dsp-test-framework",
                    definition=repo_path,
                    target="test",
                ),
            )
        )
    return workloads


def load_cmsis_nn_workloads(external_root: Path) -> list[ProgramWorkload]:
    unit_root = external_root / "cmsis-nn" / "Tests" / "UnitTest"
    root_cmake = unit_root / "CMakeLists.txt"
    try:
        root_text = root_cmake.read_text()
    except OSError as exc:
        raise InventoryError(f"cannot read {root_cmake}: {exc}") from exc

    case_names = re.findall(
        r"(?m)^\s*add_subdirectory\(TestCases/([A-Za-z0-9_]+)\)\s*$",
        root_text,
    )
    if not case_names or len(case_names) != len(set(case_names)):
        raise InventoryError(
            "CMSIS-NN unit-test owner has no unique TestCases subdirectory list"
        )

    workloads: list[ProgramWorkload] = []
    target_pattern = re.compile(
        r"(?m)^\s*add_cmsis_nn_unit_test_executable\(\s*"
        r"([A-Za-z0-9_]+)\s*\)\s*$"
    )
    for case_name in case_names:
        case_dir = unit_root / "TestCases" / case_name
        cmake_path = case_dir / "CMakeLists.txt"
        try:
            cmake_text = cmake_path.read_text()
        except OSError as exc:
            raise InventoryError(f"cannot read {cmake_path}: {exc}") from exc
        targets = target_pattern.findall(cmake_text)
        if len(targets) != 1:
            raise InventoryError(
                f"CMSIS-NN workload {case_name} must define exactly one unit-test "
                "executable"
            )
        target = targets[0]
        repo_case_dir = (
            f"externals/cmsis-nn/Tests/UnitTest/TestCases/{case_name}"
        )
        workloads.append(
            ProgramWorkload(
                suite="cmsis-nn",
                case=case_name,
                executable=target,
                sources=(),
                entry_symbol="main",
                target_profile="riscv64-portable-scalar",
                oracle=WorkloadOracle(
                    kind="cmsis-nn-unity",
                    path=repo_case_dir,
                ),
                producer=WorkloadProducer(
                    kind="cmsis-nn-unit-test",
                    definition=f"{repo_case_dir}/CMakeLists.txt",
                    target=target,
                ),
            )
        )
    return workloads


def require_unique_identities(
    rows: Sequence[SourceTranslationUnit | ProgramWorkload],
) -> None:
    identities = [row.identity for row in rows]
    duplicates = sorted(
        identity for identity, count in Counter(identities).items() if count > 1
    )
    if duplicates:
        raise InventoryError(f"duplicate corpus identity: {', '.join(duplicates)}")


def require_unique_inventory(cases: Sequence[SourceTranslationUnit]) -> None:
    require_unique_identities(cases)

    sources = [source for case in cases for source in case.sources]
    duplicate_sources = sorted(
        source for source, count in Counter(sources).items() if count > 1
    )
    if duplicate_sources:
        raise InventoryError(
            f"duplicate corpus source identity: {', '.join(duplicate_sources)}"
        )


def select_rows(
    cases: Sequence[SourceTranslationUnit | ProgramWorkload],
    *,
    suite_names: Sequence[str],
    case_ids: Sequence[str],
) -> tuple[SourceTranslationUnit | ProgramWorkload, ...]:
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


def render_json(
    cases: Sequence[SourceTranslationUnit | ProgramWorkload],
    *,
    inventory_kind: str,
) -> str:
    if inventory_kind == "source-translation-unit":
        if not all(isinstance(case, SourceTranslationUnit) for case in cases):
            raise InventoryError(
                "source inventory contains a non-source workload row"
            )
        require_unique_inventory(
            [case for case in cases if isinstance(case, SourceTranslationUnit)]
        )
    elif inventory_kind == "program-workload":
        if not all(isinstance(case, ProgramWorkload) for case in cases):
            raise InventoryError(
                "workload inventory contains a source translation-unit row"
            )
        require_unique_identities(cases)
    else:
        raise InventoryError(f"unknown inventory kind: {inventory_kind}")
    counts = Counter(case.suite for case in cases)
    payload = {
        "case_count": len(cases),
        "cases": [case.as_dict() for case in cases],
        "inventory_kind": inventory_kind,
        "suite_counts": {
            suite: counts[suite] for suite in SUITE_ORDER if counts[suite]
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("list-sources", "emit selected source inventory as JSON"),
        ("list-workloads", "emit selected workload inventory as JSON"),
    ):
        list_command = commands.add_parser(command, help=help_text)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    try:
        if args.command == "list-sources":
            inventory = load_source_inventory(ROOT)
            kind = "source-translation-unit"
        elif args.command == "list-workloads":
            inventory = load_workload_inventory(ROOT)
            kind = "program-workload"
        else:
            raise AssertionError(args.command)
        if args.command in {"list-sources", "list-workloads"}:
            selected = select_rows(
                inventory,
                suite_names=args.suites,
                case_ids=args.cases,
            )
            sys.stdout.write(render_json(selected, inventory_kind=kind))
            return 0
    except InventoryError as exc:
        print(f"corpus inventory: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
