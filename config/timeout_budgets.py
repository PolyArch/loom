#!/usr/bin/env python3
"""Typed access to Loom's canonical wall-time budget tiers."""

from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Sequence


class Tier(str, Enum):
    ULTRAFAST = "ultrafast"
    FAST = "fast"
    MEDIUM = "medium"
    LONG = "long"
    XLONG = "xlong"
    NIGHTLY = "nightly"


_CONFIG_PATH = Path(__file__).with_name("timeout-budgets.json")
_DOCUMENT = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
if _DOCUMENT.get("schema") != "loom.timeout_budgets":
    raise ValueError(f"unexpected timeout budget schema in {_CONFIG_PATH}")

_raw_tiers = _DOCUMENT.get("tiers")
if not isinstance(_raw_tiers, dict):
    raise ValueError(f"timeout budget tiers are missing in {_CONFIG_PATH}")

_seconds: dict[str, int] = {}
for _tier in Tier:
    _entry = _raw_tiers.get(_tier.value)
    if not isinstance(_entry, dict) or not isinstance(_entry.get("seconds"), int):
        raise ValueError(f"timeout budget {_tier.value} is invalid")
    if _entry["seconds"] <= 0:
        raise ValueError(f"timeout budget {_tier.value} is not positive")
    _seconds[_tier.value] = _entry["seconds"]

BUDGET_SECONDS = MappingProxyType(_seconds)


def seconds(tier: Tier | str) -> int:
    """Return the canonical wall-time budget in seconds for one tier."""

    key = tier.value if isinstance(tier, Tier) else tier
    try:
        return BUDGET_SECONDS[key]
    except KeyError as error:
        raise ValueError(f"unknown timeout budget tier: {key}") from error


def milliseconds(tier: Tier | str) -> int:
    return seconds(tier) * 1000


def shell(tier: Tier | str) -> str:
    return f"{seconds(tier)}s"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=[tier.value for tier in Tier])
    args = parser.parse_args(argv)
    print(seconds(args.tier))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
