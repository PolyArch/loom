#!/usr/bin/env python3
"""Assert row-complete fft_butterfly CGRA-sim evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fft_butterfly_fixtures


GRAPH_COPY = "g_t_fft_butterfly_kernel_0_0"
GRAPH_BUTTERFLY = "g_t_fft_butterfly_kernel_red_0_0"
EXPECTED_COMPONENTS = 48
EXPECTED_WORK_ITEMS = 48
TOLERANCE = 1.0e-4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def status_pass(label: str, artifact: dict[str, object]) -> None:
    require(artifact.get("status") == "pass", f"{label} must pass: {artifact}")


def component_identities(
    evidence: Path,
    artifact: dict[str, object],
    field: str,
    expected_count: int,
) -> list[str]:
    identities = artifact.get(field)
    require(isinstance(identities, list), f"{field} must be a list: {artifact}")
    require(len(identities) == expected_count, f"{field} count mismatch: {identities}")
    require(len(set(identities)) == expected_count, f"{field} must contain unique entries: {identities}")
    fingerprints = artifact.get("input_artifact_fingerprints")
    require(isinstance(fingerprints, dict), f"aggregate must fingerprint inputs: {artifact}")
    for identity in identities:
        require(isinstance(identity, str) and identity, f"bad component identity: {identity!r}")
        require((evidence / f"{identity}.json").is_file(), f"missing copied component {identity}")
        require(identity in fingerprints, f"missing fingerprint for component {identity}")
    return identities


def token_value(token: str) -> float:
    require(token.startswith("f32:"), f"expected f32 token: {token!r}")
    return float(token.split(":", 1)[1])


def memory_by_suffix(memory: object, suffix: str) -> list[float]:
    require(isinstance(memory, dict), f"final memory state must be an object: {memory}")
    matches = [values for key, values in memory.items() if isinstance(key, str) and key.endswith(suffix)]
    require(len(matches) == 1, f"expected one memory key ending {suffix}, saw {list(memory)}")
    values = matches[0]
    require(isinstance(values, list), f"memory {suffix} must be a list: {values}")
    parsed: list[float] = []
    for value in values:
        require(isinstance(value, str), f"bad memory token for {suffix}: {value!r}")
        parsed.append(token_value(value))
    return parsed


def assert_close_vector(name: str, actual: list[float], expected: tuple[float, ...]) -> None:
    require(len(actual) == len(expected), f"{name} length mismatch: {actual}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        delta = abs(actual_value - expected_value)
        require(
            delta <= TOLERANCE,
            f"{name}[{index}] mismatch: got {actual_value:.9e}, expected {expected_value:.9e}",
        )


def checksum(real: list[float], imag: list[float]) -> float:
    total = 0.0
    for index, (real_value, imag_value) in enumerate(zip(real, imag)):
        weight = float(index + 1)
        total += weight * real_value + (weight + 0.25) * imag_value
    return total


def require_routes(mapping: dict[str, object]) -> None:
    require(mapping.get("hardware") == "shared_signal_window_adg", f"unexpected hardware: {mapping}")
    require(mapping.get("aggregation_kind") == "workload_graph_set", f"mapping must aggregate components: {mapping}")
    require(int(mapping.get("unplaced_records", -1)) == 0, f"mapping has unplaced records: {mapping}")
    require(int(mapping.get("unrouted_edges", -1)) == 0, f"mapping has unrouted edges: {mapping}")
    require(int(mapping.get("routed_edges", 0)) > 0, f"mapping must route real edges: {mapping}")
    require(int(mapping.get("route_segments", 0)) > 0, f"mapping must expose route segments: {mapping}")
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping must include routes: {mapping}")
    for route in routes:
        require(isinstance(route, dict), f"route must be an object: {route}")
        require(route.get("status") == "routed", f"route must be routed: {route}")
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route must expose segments: {route}")
        previous_sink = None
        for segment in segments:
            require(isinstance(segment, dict), f"route segment must be an object: {segment}")
            require(segment.get("segment_kind") in {"resource_edge", "module_path", "buffer"}, f"bad segment: {segment}")
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(".out" not in value and ".in" not in value, f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous: {route}")
            previous_sink = segment["sink_endpoint"]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])
    fixture = fft_butterfly_fixtures.fixture_from_source()

    dfg = read_json(evidence / "fft_butterfly.dfg.report.json")
    mapping = read_json(evidence / "fft_butterfly.mapping.json")
    cgra = read_json(evidence / "fft_butterfly.cgra.report.json")
    comparison = read_json(evidence / "fft_butterfly.sim-comparison-report.json")

    for label, artifact in (
        ("DFG", dfg),
        ("mapping", mapping),
        ("CGRA", cgra),
        ("comparison", comparison),
    ):
        status_pass(label, artifact)

    require(dfg.get("graph") == "workload_graph_set", f"DFG must aggregate components: {dfg}")
    require(dfg.get("aggregation_kind") == "workload_graph_set", f"DFG must aggregate components: {dfg}")
    graphs = dfg.get("component_graphs")
    require(isinstance(graphs, list), f"DFG must list component graphs: {dfg}")
    require(graphs.count(GRAPH_COPY) == 16, f"copy component count mismatch: {graphs}")
    require(graphs.count(GRAPH_BUTTERFLY) == 32, f"butterfly component count mismatch: {graphs}")
    require(len(graphs) == EXPECTED_COMPONENTS, f"component graph count mismatch: {graphs}")
    require(int(dfg.get("dynamic_work_items", -1)) == EXPECTED_WORK_ITEMS, f"work item count mismatch: {dfg}")

    require_routes(mapping)
    require(cgra.get("hardware") == "shared_signal_window_adg", f"unexpected CGRA hardware: {cgra}")
    require(cgra.get("aggregation_kind") == "workload_graph_set", f"CGRA must aggregate components: {cgra}")
    require(cgra.get("final_memory_state") == dfg.get("final_memory_state"), "CGRA memory must match DFG memory")
    require(
        cgra.get("functional_state_source") == "component_cgra_sim_reports_carried_from_dfg_sim_reports",
        f"CGRA must disclose final-state provenance: {cgra}",
    )
    require(int(cgra.get("hardware_aware_cycles", -1)) >= int(dfg.get("optimistic_cycles", 0)), "CGRA cycles too optimistic")

    component_identities(evidence, dfg, "component_dfg_sim_report_identities", EXPECTED_COMPONENTS)
    component_identities(evidence, mapping, "component_mapping_artifact_identities", EXPECTED_COMPONENTS)
    component_identities(evidence, cgra, "component_dfg_sim_report_identities", EXPECTED_COMPONENTS)
    component_identities(evidence, cgra, "component_cgra_sim_report_identities", EXPECTED_COMPONENTS)

    final_real = memory_by_suffix(dfg.get("final_memory_state"), "stage04-k00-j07.report:arg6")
    final_imag = memory_by_suffix(dfg.get("final_memory_state"), "stage04-k00-j07.report:arg7")
    assert_close_vector("real", final_real, fixture.expected_real)
    assert_close_vector("imag", final_imag, fixture.expected_imag)
    require(len(set(final_real)) > 8, f"real output lacks distinct values: {final_real}")
    require(len(set(final_imag)) > 8, f"imag output lacks distinct values: {final_imag}")
    require(
        abs(checksum(final_real, final_imag) - fixture.expected_checksum) <= 2.5e-4,
        f"checksum mismatch: {checksum(final_real, final_imag)} vs {fixture.expected_checksum}",
    )

    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
