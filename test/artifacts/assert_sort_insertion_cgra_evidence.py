#!/usr/bin/env python3
"""Assert row-complete sort_insertion CGRA-sim evidence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


EXPECTED_GRAPHS = [
    "g_t_sort_insertion_kernel_0_0",
    "g_t_sort_insertion_kernel_effect_0",
]
HARDWARE = "shared_memory_reduction_adg"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def parse_float_array(source: Path, name: str) -> list[float]:
    text = source.read_text()
    match = re.search(
        rf"constexpr\s+std::array<float,\s*kSize>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
        text,
        re.S,
    )
    require(match is not None, f"missing {name} in {source}")
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f?", match.group("body"))
    require(values, f"{name} must contain values")
    return [float(value.rstrip("f")) for value in values]


def format_f32(value: float) -> str:
    if value == 0:
        value = 0.0
    if float(value).is_integer():
        return f"f32:{int(value)}"
    return f"f32:{value:.6f}"


def expected_memories() -> tuple[list[str], list[str]]:
    source = Path(__file__).resolve().parents[2] / "test/app/sort_insertion/main_func.cpp"
    input_values = parse_float_array(source, "kInput")
    expected_values = parse_float_array(source, "kExpected")
    require(len(input_values) == 12, f"sort_insertion graph fixture expects 12 inputs, saw {len(input_values)}")
    require(len(expected_values) == 12, f"sort_insertion graph fixture expects 12 outputs, saw {len(expected_values)}")
    require(sorted(input_values) == expected_values, f"kExpected must sort kInput: {input_values} -> {expected_values}")
    return [format_f32(value) for value in input_values], [format_f32(value) for value in expected_values]


def require_status(label: str, artifact: dict[str, object]) -> None:
    require(artifact.get("status") == "pass", f"{label} evidence must pass: {artifact}")


def require_memory(memory: object, key: str, expected: list[str], label: str) -> None:
    require(isinstance(memory, dict), f"{label} final memory state must be an object: {memory}")
    values = memory.get(key)
    require(values == expected, f"{label} {key} mismatch: {values}")


def require_component_identities(
    evidence: Path,
    artifact: dict[str, object],
    field: str,
    expected_count: int,
) -> None:
    identities = artifact.get(field)
    require(isinstance(identities, list), f"{field} must be a list: {artifact}")
    require(len(identities) == expected_count, f"{field} count mismatch: {identities}")
    require(len(set(identities)) == expected_count, f"{field} must contain unique entries: {identities}")
    fingerprints = artifact.get("input_artifact_fingerprints")
    require(isinstance(fingerprints, dict), f"aggregate must fingerprint inputs: {artifact}")
    for identity in identities:
        require(isinstance(identity, str), f"bad component identity: {identity!r}")
        require((evidence / f"{identity}.json").is_file(), f"missing copied component {identity}")
        require(identity in fingerprints, f"missing fingerprint for component {identity}")


def require_real_route_segments(mapping: dict[str, object], evidence: Path) -> None:
    route_sources: list[tuple[str, dict[str, object]]] = [("aggregate", mapping)]
    identities = mapping.get("component_mapping_artifact_identities")
    require(isinstance(identities, list), f"mapping must list component identities: {mapping}")
    for identity in identities:
        require(isinstance(identity, str), f"bad component mapping identity: {identity!r}")
        route_sources.append((identity, read_json(evidence / f"{identity}.json")))

    aggregate_segment_count = 0
    for label, artifact in route_sources:
        routes = artifact.get("routes")
        require(isinstance(routes, list) and routes, f"{label} mapping must expose routes: {artifact}")
        for route_index, route in enumerate(routes):
            require(isinstance(route, dict), f"{label} route {route_index} must be an object: {route}")
            require(route.get("status") == "routed", f"{label} route {route_index} must be routed: {route}")
            segments = route.get("segments")
            require(isinstance(segments, list) and segments, f"{label} route {route_index} has no segments: {route}")
            previous_sink = None
            for segment_index, segment in enumerate(segments):
                require(
                    isinstance(segment, dict),
                    f"{label} route {route_index} segment {segment_index} must be an object: {segment}",
                )
                kind = segment.get("segment_kind")
                require(
                    kind in {"resource_edge", "module_path", "buffer"},
                    f"{label} segment has bad kind: {segment}",
                )
                for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                    value = segment.get(field)
                    require(isinstance(value, str) and value, f"{label} segment lacks {field}: {segment}")
                    require("::" in value, f"{label} segment {field} is not a structured hardware ref: {segment}")
                    require(".out" not in value and ".in" not in value, f"{label} segment has placeholder endpoint: {segment}")
                if previous_sink is not None:
                    require(
                        segment["source_endpoint"] == previous_sink,
                        f"{label} route {route_index} is not contiguous at segment {segment_index}: {route}",
                    )
                previous_sink = segment["sink_endpoint"]
                if label == "aggregate":
                    aggregate_segment_count += 1
            require(
                any(segment.get("segment_kind") == "module_path" for segment in segments),
                f"{label} route {route_index} should include a module path segment: {route}",
            )
    require(
        int(mapping.get("route_segments", -1)) == aggregate_segment_count,
        f"aggregate route_segments does not match route records: {mapping}",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])

    dfg = read_json(evidence / "sort_insertion.dfg.report.json")
    mapping = read_json(evidence / "sort_insertion.mapping.json")
    cgra = read_json(evidence / "sort_insertion.cgra.report.json")
    comparison = read_json(evidence / "sort_insertion.sim-comparison-report.json")
    expected_input, expected_sorted = expected_memories()

    for label, artifact in (
        ("DFG", dfg),
        ("mapping", mapping),
        ("CGRA", cgra),
        ("comparison", comparison),
    ):
        require_status(label, artifact)

    require(dfg.get("aggregation_kind") == "workload_graph_set", f"DFG must aggregate components: {dfg}")
    require(dfg.get("component_graphs") == EXPECTED_GRAPHS, f"DFG graph coverage mismatch: {dfg}")
    require(mapping.get("component_graphs") == EXPECTED_GRAPHS, f"mapping graph coverage mismatch: {mapping}")
    require(cgra.get("component_graphs") == EXPECTED_GRAPHS, f"CGRA graph coverage mismatch: {cgra}")

    require(mapping.get("hardware") == HARDWARE, f"unexpected hardware: {mapping}")
    require(int(mapping.get("unplaced_records", -1)) == 0, f"mapping has unplaced records: {mapping}")
    require(int(mapping.get("unrouted_edges", -1)) == 0, f"mapping has unrouted edges: {mapping}")
    require(int(mapping.get("routed_edges", 0)) > 0, f"mapping must route real edges: {mapping}")
    require(int(mapping.get("route_segments", 0)) > 0, f"mapping must expose route segments: {mapping}")
    require_real_route_segments(mapping, evidence)

    require_memory(dfg.get("final_memory_state"), "g_t_sort_insertion_kernel_0_0:arg2", expected_input, "DFG")
    require_memory(cgra.get("final_memory_state"), "g_t_sort_insertion_kernel_0_0:arg2", expected_input, "CGRA")
    require_memory(
        dfg.get("final_memory_state"),
        "g_t_sort_insertion_kernel_effect_0:arg1",
        expected_sorted,
        "DFG",
    )
    require_memory(
        cgra.get("final_memory_state"),
        "g_t_sort_insertion_kernel_effect_0:arg1",
        expected_sorted,
        "CGRA",
    )
    require(cgra.get("final_memory_state") == dfg.get("final_memory_state"), f"CGRA must carry DFG memory: {cgra}")
    require(
        cgra.get("functional_state_source") == "component_cgra_sim_reports_carried_from_dfg_sim_reports",
        f"CGRA must disclose final-state provenance: {cgra}",
    )

    require_component_identities(evidence, dfg, "component_dfg_sim_report_identities", len(EXPECTED_GRAPHS))
    require_component_identities(evidence, mapping, "component_mapping_artifact_identities", len(EXPECTED_GRAPHS))
    require_component_identities(evidence, cgra, "component_dfg_sim_report_identities", len(EXPECTED_GRAPHS))
    require_component_identities(evidence, cgra, "component_cgra_sim_report_identities", len(EXPECTED_GRAPHS))

    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    require(
        int(cgra.get("hardware_aware_cycles", -1)) >= int(dfg.get("optimistic_cycles", 0)),
        f"CGRA cycles must not be more optimistic than DFG: {dfg} {cgra}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
