"""Shared typed-value and source-owned ABI checks for evidence projections."""

from __future__ import annotations

import re
from typing import Any


def owned_projection_literal(name: str, owner: str) -> str:
    value = re.search(
        rf'\b{re.escape(name)}\s*=\s*"([^"]+)"',
        owner,
    )
    if value is None:
        raise RuntimeError("application projection ABI owner is malformed")
    return value.group(1)


def integer_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def artifact_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def artifact_root_reference(
    value: Any, schema: str | None = None, version: str | None = None
) -> dict[str, str] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "schema_version",
        "artifact",
    }:
        return None
    if not isinstance(value.get("schema"), str) or not isinstance(
        value.get("schema_version"), str
    ):
        return None
    artifact = value.get("artifact")
    if not artifact_digest(artifact):
        return None
    if schema is not None and value["schema"] != schema:
        return None
    if version is not None and value["schema_version"] != version:
        return None
    return dict(value)
