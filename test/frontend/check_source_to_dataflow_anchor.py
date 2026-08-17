#!/usr/bin/env python3
"""Validate the stable source-to-Dataflow conformance projection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from corpus_simulation_report import parse_dse_execution_projection  # noqa: E402


ARTIFACT_FIELDS = {
    "canonical_dataflow",
    "canonical_dataflow_initial",
    "simulation_runtime_input",
    "simulation_workload",
    "structured_initial",
    "structured_selected",
}
LINEAGE_FIELDS = {
    "dataflow_rewrite",
    "execution_shape",
    "memory_communication",
    "ownership",
    "schedule",
    "special_math_accuracy",
}
IDENTITY = re.compile(r"[0-9a-f]{64}")
ENTITY = re.compile(r"0|[1-9][0-9]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--require-operation", action="append", default=[])
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text())

    require(report.get("kind") == "source_backed_dfg_comparison", "wrong kind")
    require(report.get("status") == "pass", "execution did not pass")
    require(report.get("execution_terminal") == "retired", "execution did not retire")

    artifacts = report.get("artifacts")
    require(isinstance(artifacts, dict), "artifacts are absent")
    require(set(artifacts) == ARTIFACT_FIELDS, "artifact inventory changed")
    require(
        all(
            isinstance(value, str) and IDENTITY.fullmatch(value)
            for value in artifacts.values()
        ),
        "artifact identity is not canonical hexadecimal",
    )

    actor_refs = report.get("actor_refs")
    require(isinstance(actor_refs, list) and actor_refs, "stable ActorRefs are absent")
    require(len(actor_refs) == report.get("actors"), "ActorRef inventory is incomplete")
    observed_entities: set[str] = set()
    for reference in actor_refs:
        require(isinstance(reference, dict), "ActorRef is not an object")
        require(set(reference) == {"artifact", "entity"}, "ActorRef shape changed")
        require(
            reference["artifact"] == artifacts["canonical_dataflow"],
            "ActorRef has a foreign artifact owner",
        )
        entity = reference["entity"]
        require(
            isinstance(entity, str) and ENTITY.fullmatch(entity),
            "ActorRef entity is invalid",
        )
        require(entity not in observed_entities, "ActorRef entity is duplicated")
        observed_entities.add(entity)

    oracle = report.get("source_oracle")
    require(
        oracle == {"comparison": "equivalent", "entry_result": 0},
        "source-backed independent oracle is incomplete",
    )

    lineage = report.get("transform_lineage")
    require(isinstance(lineage, dict), "transform lineage is absent")
    require(set(lineage) == LINEAGE_FIELDS, "transform lineage inventory changed")
    for field in ("ownership", "execution_shape", "special_math_accuracy", "schedule"):
        require(
            nonnegative_integer(lineage[field]), f"{field} lineage count is invalid"
        )
    for field in ("memory_communication", "dataflow_rewrite"):
        kinds = lineage[field]
        require(isinstance(kinds, list), f"{field} lineage is not a list")
        require(
            all(nonnegative_integer(kind) for kind in kinds),
            f"{field} lineage contains an invalid decision kind",
        )

    require(positive_integer(report.get("graphs")), "no graph was published")
    require(positive_integer(report.get("dynamic_calls")), "no graph was executed")
    require(positive_integer(report.get("event_count")), "no DFG event was executed")
    require(
        positive_integer(report.get("value_lanes_compared"))
        or positive_integer(report.get("memory_bytes_compared")),
        "no terminal observable was compared",
    )

    firings = report.get("operation_firings")
    require(isinstance(firings, dict) and firings, "operation firings are absent")
    for operation in args.require_operation:
        require(
            positive_integer(firings.get(operation)), f"{operation} did not execute"
        )

    source_files = report.get("selected_source_files")
    require(isinstance(source_files, list), "selected source files are absent")
    expected_source = args.source.resolve()
    require(
        expected_source in {Path(source).resolve() for source in source_files},
        "selected source provenance omitted the anchor source",
    )

    try:
        parse_dse_execution_projection(report.get("dse_execution"))
    except ValueError as exc:
        raise ValueError(f"invalid DSE execution summary: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
