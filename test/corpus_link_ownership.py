#!/usr/bin/env python3
"""Exact linker-selected source ownership projection for corpus gates."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class LinkOwnershipError(ValueError):
    """Raised when a build projection cannot identify one exact owner."""


@dataclass(frozen=True)
class LinkedWorkloadModules:
    target: Path
    resolution: Path
    link_root: Path
    object_sources: tuple[tuple[Path, Path], ...]


@dataclass(frozen=True)
class LinkSelection:
    prevailing_definitions: tuple[tuple[str, str], ...]
    selected_owners: tuple[str, ...]


def load_compilation_owners(
    path: Path,
) -> tuple[tuple[tuple[Path, Path], ...] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read exact compilation database {path}: {exc}"
    if not isinstance(payload, list):
        return None, f"compilation database is not an array: {path}"

    owners: dict[Path, Path] = {}
    for ordinal, record in enumerate(payload):
        if not isinstance(record, dict):
            return None, f"compilation record {ordinal} is not an object: {path}"
        directory = record.get("directory")
        output = record.get("output")
        source = record.get("file")
        if not all(isinstance(value, str) and value for value in (directory, output, source)):
            return None, (
                f"compilation record {ordinal} lacks directory/file/output: {path}"
            )
        base = Path(directory)
        if not base.is_absolute():
            return None, f"compilation record {ordinal} has a relative directory: {path}"
        output_path = Path(output)
        source_path = Path(source)
        if not output_path.is_absolute():
            output_path = base / output_path
        if not source_path.is_absolute():
            source_path = base / source_path
        output_path = output_path.resolve()
        source_path = source_path.resolve()
        previous = owners.get(output_path)
        if previous is not None and previous != source_path:
            return None, f"compilation output has multiple source owners: {output_path}"
        owners[output_path] = source_path
    return tuple(sorted(owners.items(), key=lambda item: str(item[0]))), None


def _parse_link_selection(
    path: Path,
) -> tuple[LinkSelection | None, str | None]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        return None, f"cannot read LLD resolution report {path}: {exc}"
    definitions: dict[str, str] = {}
    selected_owners: list[str] = []
    for line in lines:
        if not line.startswith("-r="):
            if line:
                selected_owners.append(line)
            continue
        fields = line[3:].rsplit(",", 2)
        if len(fields) != 3 or not fields[0] or not fields[1]:
            return None, f"malformed LLD resolution record in {path}: {line!r}"
        owner, symbol, flags = fields
        if "p" not in flags:
            continue
        previous = definitions.get(symbol)
        if previous is not None and previous != owner:
            return None, (
                f"symbol {symbol!r} has multiple prevailing owners in {path}"
            )
        definitions[symbol] = owner
    return (
        LinkSelection(
            prevailing_definitions=tuple(sorted(definitions.items())),
            selected_owners=tuple(dict.fromkeys(selected_owners)),
        ),
        None,
    )


_ARCHIVE_OWNER = re.compile(r"^(?P<archive>.+\.a)\((?P<member>.+) at (?P<offset>[0-9]+)\)$")
_AR_MEMBER_HEADER_BYTES = 60


def _normalize_link_path(path: str, link_root: Path) -> Path:
    result = Path(path)
    if not result.is_absolute():
        result = link_root / result
    return result.resolve()


def _run_quiet(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            env={**os.environ, "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise LinkOwnershipError(f"cannot run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or "unknown error"
        raise LinkOwnershipError(
            f"{' '.join(command)} failed: {diagnostic}"
        )
    return completed.stdout


def _parse_archive_members(output: str) -> tuple[tuple[str, int], ...]:
    members: list[tuple[str, int]] = []
    for line in output.splitlines():
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            raise LinkOwnershipError(
                f"malformed llvm-ar member-offset record: {line!r}"
            )
        try:
            offset = int(fields[1], 0)
        except ValueError as exc:
            raise LinkOwnershipError(
                f"malformed llvm-ar member offset: {line!r}"
            ) from exc
        members.append((fields[0], offset))
    if not members:
        raise LinkOwnershipError("llvm-ar returned an empty archive member catalog")
    return tuple(members)


def _parse_ninja_archive_inputs(output: str) -> tuple[str, ...]:
    inputs: list[str] = []
    in_inputs = False
    for line in output.splitlines():
        stripped = line.strip()
        if line.startswith("  input:"):
            in_inputs = True
            continue
        if line.startswith("  outputs:"):
            break
        if in_inputs and line.startswith("    ") and stripped:
            inputs.append(stripped)
    if not inputs:
        raise LinkOwnershipError("Ninja returned an empty archive input catalog")
    return tuple(inputs)


def _resolve_archive_member_object(
    owner: str,
    linked: LinkedWorkloadModules,
    llvm_ar: Path,
) -> tuple[Path | None, str | None]:
    match = _ARCHIVE_OWNER.fullmatch(owner)
    if match is None:
        return None, f"malformed archive owner in LLD resolution: {owner!r}"
    archive = _normalize_link_path(match.group("archive"), linked.link_root)
    try:
        archive_relative = archive.relative_to(linked.link_root.resolve())
    except ValueError:
        return None, f"archive owner escapes the linked build root: {archive}"
    try:
        requested_offset = int(match.group("offset"), 10)
        archive_members = _parse_archive_members(
            _run_quiet([str(llvm_ar), "tO", str(archive)])
        )
        archive_inputs = _parse_ninja_archive_inputs(
            _run_quiet(
                [
                    "ninja",
                    "-C",
                    str(linked.link_root),
                    "-t",
                    "query",
                    str(archive_relative),
                ]
            )
        )
    except LinkOwnershipError as exc:
        return None, str(exc)
    if len(archive_members) != len(archive_inputs):
        return None, (
            f"archive member/input cardinality differs for {archive}: "
            f"{len(archive_members)} != {len(archive_inputs)}"
        )
    for (member, offset), input_path in zip(
        archive_members, archive_inputs, strict=True
    ):
        if Path(input_path).name != member:
            return None, (
                f"archive member order disagrees with the Ninja build graph for {archive}"
            )
        if offset == requested_offset + _AR_MEMBER_HEADER_BYTES:
            if member != match.group("member"):
                return None, (
                    f"archive member name disagrees at offset {requested_offset}: {owner!r}"
                )
            return _normalize_link_path(input_path, linked.link_root), None
    return None, f"archive owner offset is absent from {archive}: {requested_offset}"


_TEXT_DEFINITION_KINDS = frozenset({"T", "t", "W"})


def _read_text_definitions(
    object_path: Path, llvm_nm: Path
) -> tuple[frozenset[str] | None, str | None]:
    try:
        output = _run_quiet([str(llvm_nm), "-a", "--format=posix", str(object_path)])
    except LinkOwnershipError as exc:
        return None, str(exc)
    definitions: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2:
            return None, f"malformed llvm-nm POSIX record for {object_path}: {line!r}"
        if fields[1] in _TEXT_DEFINITION_KINDS:
            definitions.add(fields[0])
    return frozenset(definitions), None


def _resolve_selected_internal_owners(
    callable_names: frozenset[str],
    selected_owners: Sequence[str],
    linked: LinkedWorkloadModules,
    llvm_ar: Path,
) -> tuple[dict[str, str] | None, str | None]:
    llvm_nm = llvm_ar.with_name("llvm-nm")
    definitions_by_object: dict[Path, frozenset[str]] = {}
    owners_by_object: dict[Path, str] = {}
    matches: dict[str, list[str]] = {name: [] for name in callable_names}
    for owner in selected_owners:
        if _ARCHIVE_OWNER.fullmatch(owner) is None:
            object_path = _normalize_link_path(owner, linked.link_root)
        else:
            object_path, defect = _resolve_archive_member_object(
                owner, linked, llvm_ar
            )
            if defect is not None:
                return None, defect
            assert object_path is not None
        previous_owner = owners_by_object.get(object_path)
        if previous_owner is not None and previous_owner != owner:
            return None, (
                f"selected object has multiple linker owner records: {object_path}"
            )
        owners_by_object[object_path] = owner
        definitions = definitions_by_object.get(object_path)
        if definitions is None:
            definitions, defect = _read_text_definitions(object_path, llvm_nm)
            if defect is not None:
                return None, defect
            assert definitions is not None
            definitions_by_object[object_path] = definitions
        for callable_name in definitions.intersection(callable_names):
            matches[callable_name].append(owner)
    resolved: dict[str, str] = {}
    for callable_name in sorted(callable_names):
        owners = matches[callable_name]
        if not owners:
            return None, (
                f"selected callable {callable_name!r} has no exact definition in the "
                "linker-selected object set"
            )
        if len(owners) != 1:
            return None, (
                f"selected callable {callable_name!r} has multiple definitions in the "
                "linker-selected object set"
            )
        resolved[callable_name] = owners[0]
    return resolved, None


def _canonical_corpus_source(
    source: Path, external_root: Path, repo_root: Path
) -> str | None:
    source = source.resolve()
    try:
        return str(Path("externals") / source.relative_to(external_root.resolve()))
    except ValueError:
        pass
    try:
        return str(source.relative_to(repo_root.resolve()))
    except ValueError:
        return None


def resolve_selected_corpus_sources(
    linked: LinkedWorkloadModules,
    selected_callables: Sequence[str],
    llvm_ar: Path,
    external_root: Path,
    repo_root: Path,
    allowed_sources: frozenset[str],
) -> tuple[tuple[str, ...] | None, str | None]:
    selection, defect = _parse_link_selection(linked.resolution)
    if defect is not None:
        return None, defect
    assert selection is not None
    definitions = dict(selection.prevailing_definitions)
    missing_callables = frozenset(
        callable_name
        for callable_name in selected_callables
        if callable_name not in definitions
    )
    if missing_callables:
        internal_definitions, defect = _resolve_selected_internal_owners(
            missing_callables,
            selection.selected_owners,
            linked,
            llvm_ar,
        )
        if defect is not None:
            return None, defect
        assert internal_definitions is not None
        definitions.update(internal_definitions)
    object_sources = {
        output.resolve(): source.resolve() for output, source in linked.object_sources
    }
    selected_sources: set[str] = set()
    for callable_name in selected_callables:
        owner = definitions[callable_name]
        if _ARCHIVE_OWNER.fullmatch(owner) is None:
            object_path = _normalize_link_path(owner, linked.link_root)
        else:
            object_path, defect = _resolve_archive_member_object(
                owner, linked, llvm_ar
            )
            if defect is not None:
                return None, defect
            assert object_path is not None
        source = object_sources.get(object_path)
        if source is None:
            return None, f"selected callable owner has no exact compilation record: {owner}"
        canonical = _canonical_corpus_source(source, external_root, repo_root)
        if canonical is None or canonical not in allowed_sources:
            return None, (
                f"selected callable {callable_name!r} is not owned by an exact "
                f"corpus source row: {source}"
            )
        selected_sources.add(canonical)
    if not selected_sources:
        return None, "selected graph has no exact corpus source owner"
    return tuple(sorted(selected_sources)), None
