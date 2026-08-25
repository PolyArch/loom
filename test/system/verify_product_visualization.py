#!/usr/bin/env python3
"""Validate the public product visualization closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fabric-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    arguments = parser.parse_args()

    fabric = read_object(arguments.fabric_root)
    bundle = read_object(arguments.bundle)
    if bundle.get("schema") != "loom.visualization_bundle":
        raise ValueError("visualization bundle has the wrong schema")
    if bundle.get("version") != "1.1":
        raise ValueError("visualization bundle has the wrong version")
    if bundle.get("fabric") != fabric:
        raise ValueError("visualization bundle names a different Fabric root")
    for field in ("tech_mappings", "spatial_mappings", "system_mappings"):
        if not isinstance(bundle.get(field), list) or not bundle[field]:
            raise ValueError(f"visualization bundle has no {field}")
    deployment = bundle.get("deployment")
    if not isinstance(deployment, dict) or deployment.get("schema") != (
        "loom.deployment"
    ):
        raise ValueError("visualization bundle has no Deployment reference")
    pair = bundle.get("pair_decision")
    successful = {
        "verified_acceleration",
        "verified_feasible_but_not_beneficial",
        "hardware_dse_alternative",
    }
    if not isinstance(pair, dict) or pair.get("disposition") not in successful:
        raise ValueError("visualization bundle has no successful pair decision")
    for field in ("resource_time_endpoints", "resource_time_transitions"):
        if not isinstance(bundle.get(field), list):
            raise ValueError(f"visualization bundle has no {field} array")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
