#!/usr/bin/env python3
"""Enumerate Loom's source and linked-program conformance inventories."""

from __future__ import annotations

import argparse
import hashlib
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
OPERATOR_GATE_MANIFEST = Path("test/data/corpus-operator-gate-v1.jsonl")


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
class CmsisNnWorkloadProducer:
    definition: str
    target: str
    test_function: str

    @property
    def kind(self) -> str:
        return "cmsis-nn-unit-test"

    def as_dict(self) -> dict[str, str]:
        return {
            "definition": self.definition,
            "kind": self.kind,
            "target": self.target,
            "test_function": self.test_function,
        }


@dataclass(frozen=True)
class OperatorProtocolCall:
    symbol: str
    signature: str

    def as_dict(self) -> dict[str, str]:
        return {"signature": self.signature, "symbol": self.symbol}


@dataclass(frozen=True)
class ProgramWorkload:
    suite: str
    case: str
    executable: str
    sources: tuple[str, ...]
    entry_symbol: str
    target_profile: str
    oracle: WorkloadOracle
    producer: WorkloadProducer | CmsisNnWorkloadProducer
    operator_id: str
    vector_identity: str
    protocol: tuple[OperatorProtocolCall, ...]
    compiler_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        return self.operator_id

    def as_dict(self) -> dict[str, object]:
        return {
            "case": self.case,
            "compiler_flags": list(self.compiler_flags),
            "entry_symbol": self.entry_symbol,
            "executable": self.executable,
            "identity": self.identity,
            "link_flags": list(self.link_flags),
            "operator_id": self.operator_id,
            "oracle": self.oracle.as_dict(),
            "producer": self.producer.as_dict(),
            "protocol": [call.as_dict() for call in self.protocol],
            "sources": list(self.sources),
            "suite": self.suite,
            "target_profile": self.target_profile,
            "vector_identity": self.vector_identity,
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
        raise InventoryError(f"cannot resolve shared external sources: {exc}") from exc
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
    dirty = git_text(["status", "--porcelain", "--untracked-files=no"], submodule)
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
    document = _load_operator_gate_document(root)
    provenance = _require_mapping(document.get("provenance"), "provenance")
    expected_provenance = {
        "cmsis_core_revision",
        "cmsis_dsp_revision",
        "cmsis_nn_revision",
        "loombench_manifest_digest",
    }
    if set(provenance) != expected_provenance:
        raise InventoryError("operator gate provenance has unknown or missing fields")

    for suite, relative_path in CMSIS_SUBMODULES.items():
        _, revision = require_pinned_submodule(root, external_root, relative_path)
        key = suite.replace("-", "_") + "_revision"
        if provenance.get(key) != revision:
            raise InventoryError(
                f"operator gate {key} does not match the pinned submodule"
            )
    for relative_path in CMSIS_SUPPORT_SUBMODULES:
        _, revision = require_pinned_submodule(root, external_root, relative_path)
        key = relative_path.name.replace("-", "_") + "_revision"
        if provenance.get(key) != revision:
            raise InventoryError(
                f"operator gate {key} does not match the pinned submodule"
            )
    for relative_path in set(CMSIS_WORKLOAD_SUPPORT_SUBMODULES) - set(
        CMSIS_SUPPORT_SUBMODULES
    ):
        require_pinned_submodule(root, external_root, relative_path)

    manifest_digest = hashlib.sha256(
        (root / "test" / "app" / "manifest.json").read_bytes()
    ).hexdigest()
    if provenance.get("loombench_manifest_digest") != manifest_digest:
        raise InventoryError(
            "operator gate LoomBench manifest digest does not match the pinned manifest"
        )

    raw_workloads = document.get("workloads")
    if not isinstance(raw_workloads, list):
        raise InventoryError("operator gate workloads must be an array")
    workloads = [
        _parse_operator_gate_workload(row, external_root) for row in raw_workloads
    ]
    workloads.sort(key=lambda workload: workload.operator_id)
    require_unique_identities(workloads)
    operator_ids = [workload.operator_id for workload in workloads]
    if len(operator_ids) != len(set(operator_ids)):
        raise InventoryError("operator gate repeats a typed operator identity")

    counts = _require_mapping(document.get("counts"), "counts")
    suite_counts = Counter(workload.suite for workload in workloads)
    expected_suite_counts = counts.get("suite_counts")
    if counts.get("operator_execution_count") != len(workloads):
        raise InventoryError("operator gate execution count is stale")
    if expected_suite_counts != dict(sorted(suite_counts.items())):
        raise InventoryError("operator gate suite counts are stale")
    return tuple(workloads)


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InventoryError(f"operator gate {context} must be an object")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise InventoryError(f"operator gate {context} must be a nonempty string")
    return value


def _require_string_array(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise InventoryError(f"operator gate {context} must be a string array")
    return tuple(value)


def _load_operator_gate_document(repo_root: Path) -> dict[str, object]:
    path = repo_root / OPERATOR_GATE_MANIFEST
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or any(not line for line in lines):
            raise InventoryError(
                "operator gate manifest must contain metadata and rows"
            )
        document = json.loads(lines[0])
        workloads = [json.loads(line) for line in lines[1:]]
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(
            f"cannot read operator gate manifest {path}: {exc}"
        ) from exc
    document = _require_mapping(document, "root")
    if set(document) != {"counts", "provenance", "schema_version"}:
        raise InventoryError("operator gate root has unknown or missing fields")
    if document["schema_version"] != 1:
        raise InventoryError("operator gate schema version is unsupported")
    document["workloads"] = workloads
    return document


def _execution_target_profile(profile: str) -> str:
    if profile in {"portable-scalar", "scalar"}:
        return "riscv64-portable-scalar"
    if profile == "f16":
        return "riscv64-standard-float16"
    return profile


def _parse_operator_gate_workload(
    value: object, external_root: Path
) -> ProgramWorkload:
    row = _require_mapping(value, "workload")
    expected_fields = {
        "compiler_flags",
        "entry_symbol",
        "link_flags",
        "operator_id",
        "producer",
        "profile",
        "protocol",
        "suite",
        "vector",
    }
    if set(row) != expected_fields:
        raise InventoryError("operator gate workload has unknown or missing fields")

    suite = _require_string(row["suite"], "workload suite")
    if suite not in SUITE_ORDER:
        raise InventoryError(f"operator gate has unknown suite: {suite}")
    operator_id = _require_string(row["operator_id"], "operator identity")
    parts = operator_id.split(":", 2)
    if len(parts) != 3 or parts[0] != suite:
        raise InventoryError(f"operator gate has malformed identity: {operator_id}")

    raw_protocol = row["protocol"]
    if not isinstance(raw_protocol, list) or not raw_protocol:
        raise InventoryError(f"operator gate protocol is empty: {operator_id}")
    protocol: list[OperatorProtocolCall] = []
    for ordinal, value in enumerate(raw_protocol):
        call = _require_mapping(value, f"protocol call {ordinal}")
        if set(call) != {"signature", "symbol"}:
            raise InventoryError("operator gate protocol call has invalid fields")
        protocol.append(
            OperatorProtocolCall(
                symbol=_require_string(call["symbol"], "protocol symbol"),
                signature=_require_string(call["signature"], "protocol signature"),
            )
        )

    raw_producer = _require_mapping(row["producer"], "producer")
    if set(raw_producer) != {"definitions", "kind", "sources", "variant"}:
        raise InventoryError("operator gate producer has invalid fields")
    definitions = _require_string_array(
        raw_producer["definitions"], "producer definitions"
    )
    sources = _require_string_array(raw_producer["sources"], "producer sources")
    producer_kind = _require_string(raw_producer["kind"], "producer kind")
    producer_variant = _require_string(raw_producer["variant"], "producer variant")

    raw_vector = _require_mapping(row["vector"], "vector")
    if set(raw_vector) != {"identity", "oracle", "selector"}:
        raise InventoryError("operator gate vector has invalid fields")
    vector_identity = _require_string(raw_vector["identity"], "vector identity")
    raw_oracle = _require_mapping(raw_vector["oracle"], "oracle")
    if set(raw_oracle) != {"kind", "path"}:
        raise InventoryError("operator gate oracle has invalid fields")
    oracle = WorkloadOracle(
        kind=_require_string(raw_oracle["kind"], "oracle kind"),
        path=_require_string(raw_oracle["path"], "oracle path"),
    )
    selector = _require_mapping(raw_vector["selector"], "vector selector")
    selector_kind = _require_string(selector.get("kind"), "vector selector kind")

    if producer_kind == "cmsis-nn-operator-harness" and selector_kind == "upstream":
        case = _require_string(selector.get("case"), "CMSIS-NN case")
        test_function = _require_string(selector.get("test"), "CMSIS-NN test")
        definition = Path(definitions[0])
        if definition.parts[:2] != ("externals", "cmsis-nn"):
            raise InventoryError(
                f"CMSIS-NN producer definition escapes its owner: {definition}"
            )
        if definition.name != "CMakeLists.txt" or definition.parent.name != case:
            raise InventoryError(
                f"CMSIS-NN vector case does not own its build description: {case}"
            )
        target = load_cmsis_nn_case_target(
            external_root.joinpath(*definition.parts[1:])
        )
        producer: WorkloadProducer | CmsisNnWorkloadProducer = CmsisNnWorkloadProducer(
            definition=definitions[0],
            target=target,
            test_function=test_function,
        )
        executable = cmsis_nn_workload_target(target, test_function)
    else:
        producer = WorkloadProducer(
            kind=producer_kind,
            definition=definitions[0],
            target=producer_variant,
        )
        executable = producer_variant if producer_kind == "direct-source" else parts[2]

    return ProgramWorkload(
        suite=suite,
        case=parts[1],
        executable=executable,
        sources=sources,
        entry_symbol=_require_string(row["entry_symbol"], "entry symbol"),
        target_profile=_execution_target_profile(
            _require_string(row["profile"], "target profile")
        ),
        oracle=oracle,
        producer=producer,
        operator_id=operator_id,
        vector_identity=vector_identity,
        protocol=tuple(protocol),
        compiler_flags=_require_string_array(row["compiler_flags"], "compiler flags"),
        link_flags=_require_string_array(row["link_flags"], "link flags"),
    )


_CMSIS_NN_UNITY_TEST = re.compile(
    r"(?m)^\s*void\s+(test_[A-Za-z0-9_]+)\s*\(\s*void\s*\)\s*\{"
)
_CMSIS_NN_TARGET_DECLARATION = re.compile(
    r"(?m)^\s*add_cmsis_nn_unit_test_executable\(\s*"
    r"([A-Za-z0-9_]+)\s*\)\s*$"
)


def load_cmsis_nn_case_target(cmake_path: Path) -> str:
    try:
        text = cmake_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InventoryError(f"cannot read {cmake_path}: {exc}") from exc
    targets = _CMSIS_NN_TARGET_DECLARATION.findall(text)
    if len(targets) != 1:
        raise InventoryError(
            f"CMSIS-NN case must declare one unit-test target: {cmake_path}"
        )
    return targets[0]


def load_cmsis_nn_unity_test_functions(case_dir: Path) -> tuple[str, ...]:
    wrappers = tuple(sorted((case_dir / "Unity").glob("unity_test_arm*.c")))
    if len(wrappers) != 1:
        raise InventoryError(f"CMSIS-NN case must own one Unity wrapper: {case_dir}")
    try:
        text = wrappers[0].read_text(encoding="utf-8")
    except OSError as exc:
        raise InventoryError(f"cannot read {wrappers[0]}: {exc}") from exc
    tests = tuple(_CMSIS_NN_UNITY_TEST.findall(text))
    if not tests:
        raise InventoryError(f"Unity wrapper defines no tests: {wrappers[0]}")
    if len(tests) != len(set(tests)):
        raise InventoryError(f"Unity wrapper repeats a test function: {wrappers[0]}")
    return tests


def cmsis_nn_workload_target(target: str, test_function: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", target):
        raise InventoryError(f"invalid CMSIS-NN target: {target}")
    if not re.fullmatch(r"test_[A-Za-z0-9_]+", test_function):
        raise InventoryError(f"invalid CMSIS-NN test function: {test_function}")
    return f"{target}__{test_function}"


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
            raise InventoryError("source inventory contains a non-source workload row")
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
