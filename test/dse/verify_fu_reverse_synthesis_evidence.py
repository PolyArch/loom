#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load(path: str) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as source:
        value = json.load(source)
    require(isinstance(value, dict), f"{path}: evidence is not an object")
    return value


def main() -> None:
    require(
        len(sys.argv) == 7,
        "expected initial, cached replay, independent replay, and rejection paths",
    )
    initial = load(sys.argv[1])
    replay = load(sys.argv[2])
    independent = load(sys.argv[3])

    for label, evidence in (
        ("initial", initial),
        ("cached replay", replay),
        ("independent replay", independent),
    ):
        require(
            evidence.get("schema")
            == "loom.fu_reverse_synthesis.workflow_evidence",
            f"{label}: wrong evidence schema",
        )
        require(evidence.get("schema_version") == "1.0", f"{label}: version")
        require(
            evidence.get("status") == "completed_selection",
            f"{label}: no selected candidate",
        )
        require(evidence.get("search_complete") is True, f"{label}: search")
        require(evidence.get("graph_count") == 2, f"{label}: graph count")
        require(
            evidence.get("covered_graph_count") == 2,
            f"{label}: graph coverage",
        )
        require(
            evidence.get("generate_invocation_count") == 5,
            f"{label}: production owner count",
        )
        roots = evidence.get("roots")
        require(isinstance(roots, dict), f"{label}: missing roots")
        for key in (
            "module",
            "joint_tech_mapping",
            "system",
            "configuration_abi",
        ):
            require(isinstance(roots.get(key), dict), f"{label}: missing {key}")
        for key, count in (
            ("tech_mappings", evidence["graph_count"]),
            ("physical_timing_profiles", 1),
        ):
            value = roots.get(key)
            require(
                isinstance(value, list) and len(value) == count,
                f"{label}: wrong {key} cardinality",
            )
        spatial = roots.get("spatial_mappings")
        require(
            isinstance(spatial, list)
            and len(spatial) >= evidence["graph_count"]
            and len(spatial) % evidence["graph_count"] == 0,
            f"{label}: incomplete per-graph spatial mappings",
        )
        for key in (
            "joint_spatial_mappings",
            "system_mappings",
            "portable_rtl_implementations",
        ):
            value = roots.get(key)
            require(
                isinstance(value, list) and len(value) >= 1,
                f"{label}: missing {key}",
            )

    require(initial.get("dispatch_count") == 5, "initial run did not dispatch")
    require(
        replay.get("dispatch_count") == 0,
        "cached journal replay redispatched work",
    )
    require(
        independent.get("dispatch_count") == 5,
        "independent replay did not recompute provider work",
    )
    for label, evidence in (
        ("cached replay", replay),
        ("independent replay", independent),
    ):
        require(initial.get("dataflow") == evidence.get("dataflow"), f"{label}: dataflow changed")
        require(
            initial.get("resolved_config") == evidence.get("resolved_config"),
            f"{label}: config changed",
        )
        require(initial.get("run_key") == evidence.get("run_key"), f"{label}: run key changed")
        require(initial.get("roots") == evidence.get("roots"), f"{label}: artifact roots changed")
    require(
        replay.get("occurrence") == initial.get("occurrence") + 1,
        "replay occurrence did not advance",
    )
    require(
        independent.get("occurrence") == initial.get("occurrence"),
        "independent replay did not begin a separate journal",
    )

    expected_rejections = (
        (sys.argv[4], "unsupported_actor_schema"),
        (sys.argv[5], "unsupported_actor_schema"),
        (sys.argv[6], "unsupported_actor_projection"),
    )
    for path, expected_failure in expected_rejections:
        evidence = load(path)
        require(
            evidence.get("schema")
            == "loom.fu_reverse_synthesis.workflow_evidence",
            f"{path}: wrong rejection evidence schema",
        )
        require(evidence.get("schema_version") == "1.0", f"{path}: version")
        require(evidence.get("status") == "rejected", f"{path}: status")
        require(evidence.get("search_started") is False, f"{path}: search")
        require(evidence.get("failure") == expected_failure, f"{path}: failure")
        require(isinstance(evidence.get("dataflow"), dict), f"{path}: dataflow")
        require(evidence.get("graph_count") == 1, f"{path}: graph count")
        require(
            isinstance(evidence.get("diagnostic"), str)
            and bool(evidence["diagnostic"]),
            f"{path}: diagnostic",
        )


if __name__ == "__main__":
    main()
