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
    require(len(sys.argv) == 3, "expected initial and replay evidence paths")
    initial = load(sys.argv[1])
    replay = load(sys.argv[2])

    for label, evidence in (("initial", initial), ("replay", replay)):
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
    require(replay.get("dispatch_count") == 0, "journal replay redispatched work")
    require(initial.get("dataflow") == replay.get("dataflow"), "dataflow changed")
    require(initial.get("resolved_config") == replay.get("resolved_config"), "config changed")
    require(initial.get("run_key") == replay.get("run_key"), "run key changed")
    require(initial.get("roots") == replay.get("roots"), "artifact roots changed")
    require(
        replay.get("occurrence") == initial.get("occurrence") + 1,
        "replay occurrence did not advance",
    )


if __name__ == "__main__":
    main()
