#!/usr/bin/env python3
"""Exact source ownership projection for linked corpus workloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LinkedWorkloadModules:
    target: Path
    resolution: Path
    link_root: Path
    object_sources: tuple[tuple[Path, Path], ...]


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
        if not all(
            isinstance(value, str) and value for value in (directory, output, source)
        ):
            return None, (
                f"compilation record {ordinal} lacks directory/file/output: {path}"
            )
        base = Path(directory)
        if not base.is_absolute():
            return None, (
                f"compilation record {ordinal} has a relative directory: {path}"
            )
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
            return None, (
                f"compilation output has multiple source owners: {output_path}"
            )
        owners[output_path] = source_path
    return tuple(sorted(owners.items(), key=lambda item: str(item[0]))), None


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


def _resolve_provenance_path(
    raw_path: str,
    linked: LinkedWorkloadModules,
    external_root: Path,
    repo_root: Path,
    known_sources: frozenset[Path],
) -> Path | None:
    path = Path(raw_path)
    candidates = (
        (path,)
        if path.is_absolute()
        else (
            linked.link_root / path,
            repo_root / path,
            external_root / path,
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in known_sources:
            return resolved

    # A debug producer may omit its compilation directory. Accept a basename
    # only when the exact linked compilation database makes it unambiguous.
    if path.parent == Path("."):
        matches = [source for source in known_sources if source.name == path.name]
        if len(matches) == 1:
            return matches[0]
    return None


def resolve_selected_corpus_sources(
    linked: LinkedWorkloadModules,
    selected_source_files: Sequence[str],
    external_root: Path,
    repo_root: Path,
    allowed_sources: frozenset[str],
) -> tuple[tuple[str, ...] | None, str | None]:
    known_sources = frozenset(source.resolve() for _, source in linked.object_sources)
    selected_sources: set[str] = set()
    for source_file in selected_source_files:
        source = _resolve_provenance_path(
            source_file, linked, external_root, repo_root, known_sources
        )
        if source is None:
            continue
        canonical = _canonical_corpus_source(source, external_root, repo_root)
        if canonical is not None and canonical in allowed_sources:
            selected_sources.add(canonical)
    if not selected_sources:
        return None, (
            "selected graph source provenance does not cover an exact corpus source row"
        )
    return tuple(sorted(selected_sources)), None
