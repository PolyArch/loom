#!/usr/bin/env python3
"""Assert row-complete bitrev_complex CGRA-sim evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


GRAPH = "g_bitrev_complex_kernel_0"
HARDWARE = "shared_memory_reduction_adg"
SIZE = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def bit_reverse(index: int) -> int:
    reversed_index = 0
    current = index
    mask = SIZE >> 1
    while mask > 0:
        reversed_index = (reversed_index << 1) | (current & 1)
        current >>= 1
        mask >>= 1
    return reversed_index


def token(value: float) -> str:
    return f"f32:{int(value)}" if value.is_integer() else f"f32:{value:.6f}"


def expected_tokens() -> tuple[list[str], list[str]]:
    real = [0.0] * SIZE
    imag = [0.0] * SIZE
    for index in range(SIZE):
        reversed_index = bit_reverse(index)
        real[reversed_index] = float(index)
        imag[reversed_index] = float(SIZE - index)
    return [token(value) for value in real], [token(value) for value in imag]


def require_memory(artifact: dict[str, object], key: str, expected: list[str], label: str) -> None:
    memory = artifact.get("final_memory_state")
    require(isinstance(memory, dict), f"{label} final memory state must be an object: {artifact}")
    actual = memory.get(key)
    require(actual == expected, f"{label} {key} mismatch: {actual}")


def require_route_segments(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping must expose routes: {mapping}")
    segment_count = 0
    for route_index, route in enumerate(routes):
        require(isinstance(route, dict), f"route {route_index} must be an object: {route}")
        require(route.get("status") == "routed", f"route {route_index} must be routed: {route}")
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route {route_index} has no segments: {route}")
        previous_sink = None
        for segment_index, segment in enumerate(segments):
            require(isinstance(segment, dict), f"route {route_index} segment {segment_index} is not an object")
            kind = segment.get("segment_kind")
            require(kind in {"resource_edge", "module_path", "buffer"}, f"bad route segment kind: {segment}")
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and value, f"segment lacks {field}: {segment}")
                require("::" in value, f"segment {field} is not a structured hardware ref: {segment}")
                require(".out" not in value and ".in" not in value, f"segment has placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(
                    segment["source_endpoint"] == previous_sink,
                    f"route {route_index} is not contiguous at segment {segment_index}: {route}",
                )
            previous_sink = segment["sink_endpoint"]
            segment_count += 1
    declared_segments = mapping.get("route_segments")
    if declared_segments is not None:
        require(declared_segments == segment_count, f"route segment count mismatch: {mapping}")
    require(segment_count > 0, f"mapping must expose route segments: {mapping}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])
    dfg = read_json(evidence / "bitrev_complex.dfg.report.json")
    mapping = read_json(evidence / "bitrev_complex.mapping.json")
    cgra = read_json(evidence / "bitrev_complex.cgra.report.json")
    comparison = read_json(evidence / "bitrev_complex.sim-comparison-report.json")
    expected_real, expected_imag = expected_tokens()

    for label, artifact in (("DFG", dfg), ("mapping", mapping), ("CGRA", cgra), ("comparison", comparison)):
        require(artifact.get("status") == "pass", f"{label} evidence must pass: {artifact}")
    require(dfg.get("graph") == GRAPH, f"DFG graph mismatch: {dfg}")
    require(mapping.get("graph") == GRAPH, f"mapping graph mismatch: {mapping}")
    require(GRAPH in str(cgra.get("mapping_id", "")), f"CGRA mapping id should name the graph: {cgra}")
    require(mapping.get("hardware") == HARDWARE, f"mapping hardware mismatch: {mapping}")
    require(cgra.get("hardware") == HARDWARE, f"CGRA hardware mismatch: {cgra}")
    require(mapping.get("unplaced_records") == 0, f"mapping has unplaced records: {mapping}")
    require(mapping.get("unrouted_edges") == 0, f"mapping has unrouted edges: {mapping}")
    require(int(mapping.get("routed_edges", 0)) > 0, f"mapping must route real edges: {mapping}")
    require_route_segments(mapping)

    require(dfg.get("dynamic_work_items") == SIZE, f"DFG should execute {SIZE} work items: {dfg}")
    fire_counts = dfg.get("operation_fire_counts")
    require(isinstance(fire_counts, dict), f"DFG operation fire counts missing: {dfg}")
    require(fire_counts.get("dataflow.load") == SIZE * 2, f"bitrev_complex should load both streams: {fire_counts}")
    require(fire_counts.get("dataflow.store") == SIZE * 2, f"bitrev_complex should store both streams: {fire_counts}")
    require_memory(dfg, "arg3", expected_real, "DFG")
    require_memory(dfg, "arg4", expected_imag, "DFG")
    require_memory(cgra, "arg3", expected_real, "CGRA")
    require_memory(cgra, "arg4", expected_imag, "CGRA")
    require(cgra.get("final_memory_state") == dfg.get("final_memory_state"), "CGRA final memory must match DFG")
    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    require(
        int(cgra.get("hardware_aware_cycles", -1)) >= int(dfg.get("optimistic_cycles", 0)),
        f"CGRA cycles must not be more optimistic than DFG: {dfg} {cgra}",
    )
    real_values = [float(token.split(":", 1)[1]) for token in expected_real]
    imag_values = [float(token.split(":", 1)[1]) for token in expected_imag]
    checksum = sum((index + 1.0) * real + (index + 1.5) * imag for index, (real, imag) in enumerate(zip(real_values, imag_values)))
    require(math.isclose(checksum, 1060896.0, rel_tol=0.0, abs_tol=1.0e-6), f"checksum mismatch: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
