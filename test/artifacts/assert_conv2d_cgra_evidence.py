#!/usr/bin/env python3
"""Assert row-complete conv2d CGRA-sim evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


EXPECTED_OUTPUT = [
    -2.5,
    -2.0,
    -1.5,
    -0.5,
    0.0,
    0.5,
    1.5,
    2.0,
    2.5,
    7.25,
    8.75,
    10.25,
    13.25,
    14.75,
    16.25,
    19.25,
    20.75,
    22.25,
]
EXPECTED_INPUT = [
    "f32:1",
    "f32:2",
    "f32:3",
    "f32:4",
    "f32:5",
    "f32:6",
    "f32:7",
    "f32:8",
    "f32:9",
    "f32:10",
    "f32:11",
    "f32:12",
    "f32:13",
    "f32:14",
    "f32:15",
    "f32:16",
]
EXPECTED_KERNEL = [
    "f32:1",
    "f32:0",
    "f32:0.500000",
    "f32:-1",
    "f32:-0.500000",
    "f32:1",
    "f32:0.250000",
    "f32:0.750000",
]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def final_f32_outputs(artifact: dict[str, object]) -> list[float]:
    raw = artifact.get("final_outputs")
    require(isinstance(raw, list), f"conv2d final_outputs must be a list: {artifact}")
    require(len(raw) == len(EXPECTED_OUTPUT) * 2, f"conv2d final_outputs should include token/value pairs: {raw}")
    values: list[float] = []
    for index, item in enumerate(raw):
        require(isinstance(item, str), f"conv2d final output should be serialized: {item!r}")
        if index % 2 == 0:
            require(item == "none", f"conv2d token output should be none: {raw}")
            continue
        require(item.startswith("f32:"), f"conv2d value output should be f32: {raw}")
        values.append(float(item.split(":", 1)[1]))
    return values


def require_expected_outputs(artifact: dict[str, object], label: str) -> None:
    values = final_f32_outputs(artifact)
    require(len(values) == len(EXPECTED_OUTPUT), f"{label} output count mismatch: {values}")
    for index, (actual, expected) in enumerate(zip(values, EXPECTED_OUTPUT)):
        require(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-5),
            f"{label} output {index} mismatch: got {actual}, expected {expected}",
        )
    require(len({round(value, 5) for value in values}) > 8, f"{label} outputs are not distinct: {values}")


def require_component_identities(evidence: Path, artifact: dict[str, object], field: str, count: int) -> None:
    component_ids = artifact.get(field)
    require(isinstance(component_ids, list), f"conv2d {field} must be a list: {artifact}")
    require(len(component_ids) == count, f"conv2d {field} must cite one component per output: {artifact}")
    require(len(set(component_ids)) == count, f"conv2d {field} must cite unique components: {artifact}")
    fingerprints = artifact.get("input_artifact_fingerprints")
    require(isinstance(fingerprints, dict), f"conv2d aggregate must fingerprint components: {artifact}")
    for identity in component_ids:
        require(isinstance(identity, str) and identity.startswith("conv2d."), f"bad component id: {identity!r}")
        path = evidence / f"{identity}.json"
        require(path.is_file(), f"missing component artifact {identity}")
        require(identity in fingerprints, f"missing component fingerprint for {identity}")


def require_component_memory_state(artifact: dict[str, object], label: str, count: int) -> None:
    memory = artifact.get("final_memory_state")
    require(isinstance(memory, dict), f"conv2d {label} must expose final memory state: {artifact}")
    arg11_keys = sorted(key for key in memory if isinstance(key, str) and key.endswith(":arg11"))
    arg12_keys = sorted(key for key in memory if isinstance(key, str) and key.endswith(":arg12"))
    require(len(arg11_keys) == count, f"conv2d {label} must retain one arg11 memory state per component: {memory}")
    require(len(arg12_keys) == count, f"conv2d {label} must retain one arg12 memory state per component: {memory}")
    require(len(set(arg11_keys)) == count, f"conv2d {label} arg11 memory labels must be unique: {memory}")
    require(len(set(arg12_keys)) == count, f"conv2d {label} arg12 memory labels must be unique: {memory}")
    for key in arg11_keys:
        require(memory.get(key) == EXPECTED_INPUT, f"conv2d {label} {key} should carry source input: {memory.get(key)}")
    for key in arg12_keys:
        require(memory.get(key) == EXPECTED_KERNEL, f"conv2d {label} {key} should carry source kernel: {memory.get(key)}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])

    dfg = read_json(evidence / "conv2d.dfg.report.json")
    mapping = read_json(evidence / "conv2d.mapping.json")
    cgra = read_json(evidence / "conv2d.cgra.report.json")
    comparison = read_json(evidence / "conv2d.sim-comparison-report.json")

    for label, artifact in (
        ("DFG", dfg),
        ("mapping", mapping),
        ("CGRA", cgra),
        ("comparison", comparison),
    ):
        require(artifact.get("status") == "pass", f"conv2d {label} evidence must pass: {artifact}")

    expected_graphs = ["g_t_conv2d_kernel_0_0"] * len(EXPECTED_OUTPUT)
    require(dfg.get("aggregation_kind") == "workload_graph_set", f"conv2d DFG must be aggregate: {dfg}")
    require(dfg.get("component_graphs") == expected_graphs, f"conv2d DFG must cover all output elements: {dfg}")
    require(mapping.get("component_graphs") == expected_graphs, f"conv2d mapping must cover all output elements: {mapping}")
    require(cgra.get("component_graphs") == expected_graphs, f"conv2d CGRA must cover all output elements: {cgra}")

    require_expected_outputs(dfg, "DFG")
    require_expected_outputs(cgra, "CGRA")
    require(cgra.get("final_outputs") == dfg.get("final_outputs"), f"conv2d CGRA must carry DFG outputs: {cgra}")
    require_component_memory_state(dfg, "DFG", len(EXPECTED_OUTPUT))
    final_memory = dfg.get("final_memory_state")
    require(
        cgra.get("final_memory_state") == final_memory,
        f"conv2d CGRA must carry the same final memory state: {cgra}",
    )
    require_component_memory_state(cgra, "CGRA", len(EXPECTED_OUTPUT))
    require(cgra.get("functional_state_source") == "component_cgra_sim_reports_carried_from_dfg_sim_reports",
            f"conv2d CGRA must disclose final-state provenance: {cgra}")
    require(mapping.get("hardware") == "shared_memory_reduction_adg", f"conv2d should use shared memory ADG: {mapping}")
    require(int(mapping.get("unplaced_records", -1)) == 0, f"conv2d mapping has unplaced records: {mapping}")
    require(int(mapping.get("unrouted_edges", -1)) == 0, f"conv2d mapping has unrouted edges: {mapping}")
    require(int(mapping.get("routed_edges", 0)) > 0, f"conv2d mapping must route real edges: {mapping}")
    require(int(mapping.get("route_segments", 0)) > 0, f"conv2d mapping must expose route segments: {mapping}")

    require_component_identities(evidence, dfg, "component_dfg_sim_report_identities", len(EXPECTED_OUTPUT))
    require_component_identities(evidence, mapping, "component_mapping_artifact_identities", len(EXPECTED_OUTPUT))
    require_component_identities(evidence, cgra, "component_dfg_sim_report_identities", len(EXPECTED_OUTPUT))
    require_component_identities(evidence, cgra, "component_cgra_sim_report_identities", len(EXPECTED_OUTPUT))

    require(comparison.get("functional_comparison_status") == "pass", f"conv2d functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"conv2d memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"conv2d performance comparison failed: {comparison}")
    require(
        int(cgra.get("hardware_aware_cycles", -1)) >= int(dfg.get("optimistic_cycles", 0)),
        f"conv2d CGRA cycles must not be more optimistic than DFG: {dfg} {cgra}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
