"""Shared helpers for report and runtime artifact readers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import intermediate_artifacts


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def read_json_or_empty(path: Path | None) -> dict[str, object]:
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return {}


def group_paths(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def first_path(grouped: dict[str, list[Path]], kind: str) -> Path | None:
    paths = grouped.get(kind, [])
    return paths[0] if paths else None


def select_by_artifact_id(items: list, path_for_item, limit: int = 1) -> list:
    return sorted(items, key=lambda item: intermediate_artifacts.artifact_id_for_path(path_for_item(item)))[:limit]


def hardware_matches(candidate: str, hardware: str) -> bool:
    return candidate == hardware or candidate.rsplit("::", 1)[-1] == hardware


def matching_rtl_manifest_path(paths: list[Path], hardware: str) -> Path | None:
    for path in paths:
        data = read_json_or_empty(path)
        if data.get("kind") != "rtl_manifest" or data.get("status") != "pass":
            continue
        source = data.get("source_fabric_adg_identity")
        if isinstance(source, str) and hardware_matches(source, hardware):
            return path
    return None


def matching_eda_report_paths(paths: list[Path], rtl_manifest_identity: str) -> list[Path]:
    matches: list[Path] = []
    for path in paths:
        data = read_json_or_empty(path)
        if data.get("kind") != "eda_report" or data.get("status") != "pass":
            continue
        if data.get("capability_class") != "rtl_lint":
            continue
        if data.get("rtl_manifest_identity") != rtl_manifest_identity:
            continue
        matches.append(path)
    return select_by_artifact_id(matches, lambda item: item, limit=1)
