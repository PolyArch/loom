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


def require_root(value: object, label: str) -> None:
    require(isinstance(value, dict), f"{label}: artifact root is not an object")
    assert isinstance(value, dict)
    require(isinstance(value.get("schema"), str), f"{label}: schema")
    require(isinstance(value.get("schema_version"), str), f"{label}: version")
    artifact = value.get("artifact")
    require(
        isinstance(artifact, str)
        and len(artifact) == 64
        and all(character in "0123456789abcdef" for character in artifact),
        f"{label}: artifact identity",
    )


def require_projection(value: dict[str, object], label: str) -> None:
    require(
        value.get("projection_kind") == "fu_reverse_synthesis_workflow",
        f"{label}: projection kind",
    )
    require(value.get("projection_format") == 1, f"{label}: projection format")
    require_root(value.get("dataflow"), f"{label}: dataflow")
    require_root(value.get("resolved_config"), f"{label}: resolved config")


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "--manifest":
        manifest = load(sys.argv[2])
        initial = load(sys.argv[3])
        rejection = load(sys.argv[4])
        facts = manifest.get("facts")
        require(isinstance(facts, dict), "manifest: missing facts")
        assert isinstance(facts, dict)
        require(
            facts.get("fu_reverse_synthesis_workflow_projections")
            == [initial, rejection],
            "manifest: reverse-FU projections were reinterpreted or omitted",
        )
        return
    require(
        len(sys.argv) == 9,
        "expected complete, incomplete, and rejection evidence paths",
    )
    initial = load(sys.argv[1])
    replay = load(sys.argv[2])
    independent = load(sys.argv[3])
    incomplete = load(sys.argv[4])

    for label, evidence in (
        ("initial", initial),
        ("cached replay", replay),
        ("independent replay", independent),
    ):
        require_projection(evidence, label)
        require(evidence.get("graph_count") == 2, f"{label}: graph count")
        require(
            evidence.get("covered_graph_count") == 2,
            f"{label}: graph coverage",
        )
        require(
            evidence.get("generate_invocation_count") == 5,
            f"{label}: production owner count",
        )
        invocation = evidence.get("invocation")
        require(isinstance(invocation, dict), f"{label}: invocation")
        assert isinstance(invocation, dict)
        outcome = invocation.get("outcome")
        require(isinstance(outcome, dict), f"{label}: outcome")
        assert isinstance(outcome, dict)
        require(outcome.get("kind") == "completed_selection", f"{label}: outcome kind")
        workflow = evidence.get("workflow")
        require(isinstance(workflow, dict), f"{label}: workflow")
        assert isinstance(workflow, dict)
        require(
            workflow.get("disposition") == "complete_candidate",
            f"{label}: workflow disposition",
        )
        require(workflow.get("failing_required_outputs") == [], f"{label}: failures")
        roots = workflow.get("required_outputs")
        require(isinstance(roots, dict), f"{label}: missing roots")
        assert isinstance(roots, dict)
        for key in (
            "module",
            "joint_tech_mapping",
            "system",
            "configuration_abi",
        ):
            value = roots.get(key)
            require(
                isinstance(value, list) and len(value) == 1,
                f"{label}: missing {key}",
            )
            require_root(value[0], f"{label}: {key}")
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
        require(
            initial.get("dataflow") == evidence.get("dataflow"),
            f"{label}: dataflow changed",
        )
        require(
            initial.get("resolved_config") == evidence.get("resolved_config"),
            f"{label}: config changed",
        )
        initial_invocation = initial["invocation"]
        evidence_invocation = evidence["invocation"]
        assert isinstance(initial_invocation, dict)
        assert isinstance(evidence_invocation, dict)
        require(
            initial_invocation.get("run_key") == evidence_invocation.get("run_key"),
            f"{label}: run key changed",
        )
        initial_workflow = initial["workflow"]
        evidence_workflow = evidence["workflow"]
        assert isinstance(initial_workflow, dict)
        assert isinstance(evidence_workflow, dict)
        require(
            initial_workflow.get("required_outputs")
            == evidence_workflow.get("required_outputs"),
            f"{label}: artifact roots changed",
        )
    initial_invocation = initial["invocation"]
    replay_invocation = replay["invocation"]
    independent_invocation = independent["invocation"]
    assert isinstance(initial_invocation, dict)
    assert isinstance(replay_invocation, dict)
    assert isinstance(independent_invocation, dict)
    require(
        replay_invocation.get("occurrence") == initial_invocation.get("occurrence") + 1,
        "replay occurrence did not advance",
    )
    require(
        independent_invocation.get("occurrence")
        == initial_invocation.get("occurrence"),
        "independent replay did not begin a separate journal",
    )

    require_projection(incomplete, "incomplete")
    require(incomplete.get("dispatch_count") == 1, "incomplete: dispatch count")
    incomplete_invocation = incomplete.get("invocation")
    require(isinstance(incomplete_invocation, dict), "incomplete: invocation")
    assert isinstance(incomplete_invocation, dict)
    incomplete_outcome = incomplete_invocation.get("outcome")
    require(isinstance(incomplete_outcome, dict), "incomplete: outcome")
    assert isinstance(incomplete_outcome, dict)
    require(incomplete_outcome.get("kind") == "incomplete", "incomplete: kind")
    require(isinstance(incomplete_outcome.get("reason"), str), "incomplete: reason")
    for key in ("unsatisfied_obligations", "retained_artifacts", "retained_evidence"):
        require(isinstance(incomplete_outcome.get(key), list), f"incomplete: {key}")
    require(
        bool(incomplete_outcome["retained_artifacts"]), "incomplete: retained roots"
    )
    incomplete_workflow = incomplete.get("workflow")
    require(isinstance(incomplete_workflow, dict), "incomplete: workflow")
    assert isinstance(incomplete_workflow, dict)
    require(
        incomplete_workflow.get("disposition") == "incomplete",
        "incomplete: disposition",
    )
    failing = incomplete_workflow.get("failing_required_outputs")
    require(isinstance(failing, list) and bool(failing), "incomplete: failing outputs")
    required = incomplete_workflow.get("required_outputs")
    require(isinstance(required, dict), "incomplete: required outputs")
    assert isinstance(required, dict)
    for key in ("module", "tech_mappings", "joint_tech_mapping", "system"):
        require(
            isinstance(required.get(key), list) and bool(required[key]),
            f"incomplete: {key}",
        )

    expected_rejections = (
        (sys.argv[5], "unsupported_actor_schema"),
        (sys.argv[6], "unsupported_actor_schema"),
        (sys.argv[7], "unsupported_actor_projection"),
        (sys.argv[8], "unsupported_actor_schema"),
    )
    for path, expected_failure in expected_rejections:
        evidence = load(path)
        require_projection(evidence, path)
        require("invocation" not in evidence, f"{path}: fabricated invocation")
        failure = evidence.get("typed_failure")
        require(isinstance(failure, dict), f"{path}: typed failure")
        assert isinstance(failure, dict)
        require(failure.get("stage") == "preflight", f"{path}: stage")
        require(failure.get("kind") == expected_failure, f"{path}: failure")
        require(evidence.get("graph_count") == 1, f"{path}: graph count")
        require(
            isinstance(failure.get("diagnostic"), str) and bool(failure["diagnostic"]),
            f"{path}: diagnostic",
        )


if __name__ == "__main__":
    main()
