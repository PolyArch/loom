#!/usr/bin/env python3
"""Verify the real TinyML host run and product refusal projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from loom_evidence_portfolio import (  # noqa: E402
    APPLICATION_OBJECTIVE_DIMENSIONS,
    TINYML_TYPED_FALLBACK_DISPOSITION,
    collect_portfolio_inventory,
    portfolio_host_key,
    portfolio_pair_key,
    validate_portfolio_host_run,
    validate_portfolio_pair,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def read_pair_evidence(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        payload = event.get("payload")
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "loom.application_pair_disposition"
        ):
            result.append(payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--host-report", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    arguments = parser.parse_args()

    inventory, errors = collect_portfolio_inventory(read_object(arguments.inventory))
    require(not errors, f"canonical inventory is invalid: {errors}")
    by_key = {
        (row["application_identity"], row["input_name"]): row for row in inventory
    }

    host_report = read_object(arguments.host_report)
    host = validate_portfolio_host_run(
        host_report, by_key.get(portfolio_host_key(host_report))
    )
    require(host["complete"], f"TinyML host conformance failed: {host}")

    evidence = read_pair_evidence(arguments.diagnostics)
    require(len(evidence) == 1, "expected one product refusal projection")
    pair = validate_portfolio_pair(
        evidence[0], by_key.get(portfolio_pair_key(evidence[0]))
    )
    require(pair is not None, "product refusal omitted its portfolio row")
    require(pair["typed_complete"], f"product refusal is untyped: {pair}")
    require(
        pair["disposition"] == TINYML_TYPED_FALLBACK_DISPOSITION
        and not pair["canonical_qor_complete"],
        "TinyML product refusal was misreported as canonical QoR",
    )
    require(
        set(pair["unsupported_objective_dimensions"])
        == set(APPLICATION_OBJECTIVE_DIMENSIONS),
        "TinyML refusal lost typed null objective dimensions",
    )


if __name__ == "__main__":
    main()
