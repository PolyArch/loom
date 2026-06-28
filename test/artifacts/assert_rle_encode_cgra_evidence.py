#!/usr/bin/env python3
"""Assert row-complete rle_encode SpatialCore loop evidence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


GRAPH = "g_t_rle_encode_kernel_red_0_0"
HARDWARE = "shared_reduction_adg"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def parse_input(source: Path) -> list[int]:
    text = source.read_text()
    match = re.search(r"const\s+std::array<uint32_t,\s*kSize>\s+input\s*=\s*\{(?P<body>.*?)\};", text, re.S)
    require(match is not None, f"missing input initializer in {source}")
    values = [int(value) for value in re.findall(r"\b\d+\b", match.group("body"))]
    require(values, f"input initializer is empty in {source}")
    return values


def rle(values: list[int]) -> tuple[list[int], list[int], int, int, int]:
    require(values, "rle input cannot be empty")
    encoded_values: list[int] = []
    encoded_counts: list[int] = []
    current = values[0]
    count = 1
    write = 0
    for value in values[1:]:
        if value == current:
            count += 1
            continue
        encoded_values.append(current)
        encoded_counts.append(count)
        write += 1
        current = value
        count = 1
    return encoded_values, encoded_counts, count, current, write


def i32(values: list[int], total: int) -> list[str]:
    padded = values + [0] * (total - len(values))
    return [f"i32:{value}" for value in padded]


def require_status(label: str, artifact: dict[str, object]) -> None:
    require(artifact.get("status") == "pass", f"{label} must pass: {artifact}")


def require_memory(artifact: dict[str, object], key: str, expected: list[str], label: str) -> None:
    memory = artifact.get("final_memory_state")
    require(isinstance(memory, dict), f"{label} final_memory_state must be present")
    require(memory.get(key) == expected, f"{label} {key} mismatch: {memory.get(key)} != {expected}")


def require_route_segments(mapping: dict[str, object]) -> None:
    require(int(mapping.get("unplaced_records", -1)) == 0, f"mapping has unplaced records: {mapping}")
    require(int(mapping.get("unrouted_edges", -1)) == 0, f"mapping has unrouted edges: {mapping}")
    require(int(mapping.get("routed_edges", 0)) > 0, f"mapping should route real edges: {mapping}")
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose concrete routes: {mapping}")
    saw_switch = False
    segment_count = 0
    for route in routes:
        require(isinstance(route, dict), f"route must be an object: {route}")
        require(route.get("status") == "routed", f"route must be routed: {route}")
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route must contain segments: {route}")
        previous_sink = None
        for segment in segments:
            segment_count += 1
            require(isinstance(segment, dict), f"segment must be an object: {segment}")
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(".out" not in value and ".in" not in value, f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous: {route}")
            previous_sink = segment["sink_endpoint"]
            saw_switch = saw_switch or "fabric.switch" in str(segment.get("hardware_ref", ""))
    require(segment_count > 0, f"mapping should expose route segments: {mapping}")
    require(saw_switch, f"rle_encode should use real shared-fabric switch routes: {mapping}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])
    source = Path(__file__).resolve().parents[2] / "test/app/rle_encode/main_func.cpp"
    input_values = parse_input(source)
    encoded_values, encoded_counts, final_count, final_current, final_write = rle(input_values)
    total = len(input_values)

    dfg = read_json(evidence / "rle_encode.dfg.report.json")
    mapping = read_json(evidence / "rle_encode.mapping.json")
    cgra = read_json(evidence / "rle_encode.cgra.report.json")
    comparison = read_json(evidence / "rle_encode.sim-comparison-report.json")

    for label, artifact in (("DFG", dfg), ("mapping", mapping), ("CGRA", cgra), ("comparison", comparison)):
        require_status(label, artifact)

    require(dfg.get("graph") == GRAPH, f"unexpected DFG graph: {dfg}")
    require(mapping.get("graph") == GRAPH, f"unexpected mapping graph: {mapping}")
    require(
        GRAPH in str(cgra.get("mapping_id", "")),
        f"CGRA report should identify the mapped graph through mapping_id: {cgra}",
    )
    require(mapping.get("hardware") == HARDWARE, f"unexpected hardware: {mapping}")
    require(cgra.get("hardware") == HARDWARE, f"unexpected CGRA hardware: {cgra}")
    require_route_segments(mapping)

    expected_input = i32(input_values, total)
    expected_values = i32(encoded_values, total)
    expected_counts = i32(encoded_counts, total)
    for artifact, label in ((dfg, "DFG"), (cgra, "CGRA")):
        require_memory(artifact, "arg4", expected_input, label)
        require_memory(artifact, "arg6", expected_values, label)
        require_memory(artifact, "arg7", expected_counts, label)
        require(
            artifact.get("final_outputs") == ["none", f"i32:{final_count}", f"i32:{final_current}", f"i32:{final_write}"],
            f"{label} final loop outputs mismatch: {artifact.get('final_outputs')}",
        )
    require(cgra.get("final_memory_state") == dfg.get("final_memory_state"), "CGRA must carry DFG memory state")
    require(cgra.get("final_outputs") == dfg.get("final_outputs"), "CGRA must carry DFG final outputs")

    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    require(int(cgra.get("hardware_aware_cycles", -1)) >= int(dfg.get("optimistic_cycles", 0)), "CGRA cycles too optimistic")
    require(int(dfg.get("dynamic_work_items", -1)) == total - 1, f"unexpected RLE loop trip count: {dfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
