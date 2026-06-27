#!/usr/bin/env python3
"""Regression tests for ADG inventory evidence."""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import artifact_test_common
import intermediate_artifacts


HARDWARE_HEADER = [
    "hardware",
    "topology_class",
    "node_count",
    "link_count",
    "verify_status",
    "diagnostic",
    "tile_kinds",
    "schedule_kinds",
    "adg_builder_recipe_identity",
    "node_kinds",
]

REQUIRED_TOPOLOGY_FAMILIES = {
    "regular": {
        "chain_1d",
        "folded_ring",
        "mesh_2d",
        "mesh_diagonal",
        "multi_lane_pipeline",
        "torus_edge",
        "systolic_array",
        "clustered_array",
    },
    "irregular": {
        "diamond_bypass",
        "reduction_tree",
        "cross_coupled_switch",
        "memory_fanout",
        "mixed_temporal_bridge",
        "sparse_long_link",
        "heterogeneous_islands",
    },
}

REQUIRED_SYSTEM_FAMILIES = {
    "heterogeneous_soc",
    "dual_spatial_shared_memory",
    "cached_dual_accel",
    "dma_scratchpad",
    "fixed_and_spatial",
}

EXPECTED_TOPOLOGY_SIGNATURES = {
    "vector_alu_network": {
        "layout_class": "irregular",
        "fabric_root": "shared_vector_alu_adg",
        "tile_counts": {"mem": 1, "pe": 7, "switch": 3},
        "schedule_kinds": {"spatial"},
    },
    "vector_math_network": {
        "layout_class": "irregular",
        "fabric_root": "shared_vector_math_adg",
        "tile_counts": {"mem": 1, "pe": 16, "switch": 2},
        "schedule_kinds": {"spatial"},
    },
    "vector_mesh": {
        "layout_class": "regular",
        "fabric_root": "shared_vector_mesh_adg",
        "tile_counts": {"mem": 1, "pe": 3, "switch": 5},
        "schedule_kinds": {"spatial"},
    },
    "chain_1d": {
        "layout_class": "regular",
        "fabric_root": "matrix_chain1d_adg",
        "tile_counts": {"mem": 1, "pe": 3, "switch": 1},
        "schedule_kinds": {"spatial"},
    },
    "mesh_2d": {
        "layout_class": "regular",
        "fabric_root": "matrix_mesh2d_adg",
        "tile_counts": {"mem": 1, "pe": 4, "switch": 1},
        "schedule_kinds": {"spatial"},
    },
    "torus_edge": {
        "layout_class": "regular",
        "fabric_root": "matrix_torus_edge_adg",
        "tile_counts": {"mem": 1, "pe": 4, "switch": 2},
        "schedule_kinds": {"spatial"},
    },
    "systolic_array": {
        "layout_class": "regular",
        "fabric_root": "matrix_systolic_array_adg",
        "tile_counts": {"mem": 1, "pe": 3, "switch": 1},
        "schedule_kinds": {"spatial"},
    },
    "clustered_array": {
        "layout_class": "regular",
        "fabric_root": "matrix_clustered_array_adg",
        "tile_counts": {"mem": 1, "pe": 5, "switch": 2},
        "schedule_kinds": {"spatial"},
    },
    "reduction_tree": {
        "layout_class": "irregular",
        "fabric_root": "matrix_reduction_tree_adg",
        "tile_counts": {"mem": 1, "pe": 3, "switch": 1},
        "schedule_kinds": {"spatial"},
    },
    "cross_coupled_switch": {
        "layout_class": "irregular",
        "fabric_root": "matrix_cross_coupled_switch_adg",
        "tile_counts": {"mem": 1, "pe": 3, "switch": 2},
        "schedule_kinds": {"spatial"},
    },
    "sparse_long_link": {
        "layout_class": "irregular",
        "fabric_root": "matrix_sparse_long_link_adg",
        "tile_counts": {"mem": 1, "pe": 4, "switch": 1},
        "schedule_kinds": {"spatial"},
    },
    "heterogeneous_islands": {
        "layout_class": "irregular",
        "fabric_root": "matrix_heterogeneous_islands_adg",
        "tile_counts": {"mem": 1, "pe": 4, "switch": 1},
        "schedule_kinds": {"spatial", "temporal"},
    },
}


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_candidate_field(candidate: dict[str, object], field: str) -> object:
    value = candidate.get(field)
    if value in (None, ""):
        raise AssertionError(f"candidate {candidate.get('candidate_id')} lacks {field}")
    return value


def resolve_inventory_reference(inventory_path: Path, reference: object) -> Path:
    if not isinstance(reference, str) or reference == "":
        raise AssertionError(f"invalid inventory path reference: {reference!r}")
    path = Path(reference)
    if path.is_absolute():
        return path
    return inventory_path.parent / path


def assert_inventory_shape(inventory_path: Path) -> dict[str, object]:
    data = json.loads(inventory_path.read_text())
    if data.get("kind") != "adg_inventory":
        raise AssertionError(f"unexpected inventory kind: {data}")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 24:
        raise AssertionError(f"inventory should contain a reusable ADG matrix: {data}")

    ids: set[str] = set()
    root_kinds: set[str] = set()
    layout_classes: set[str] = set()
    families_by_layout = {"regular": set(), "irregular": set()}
    system_families: set[str] = set()
    topology_fingerprints: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise AssertionError(f"candidate should be an object: {candidate!r}")
        candidate_id = require_candidate_field(candidate, "candidate_id")
        if not isinstance(candidate_id, str):
            raise AssertionError(f"candidate_id should be a string: {candidate}")
        if candidate_id in ids:
            raise AssertionError(f"duplicate candidate_id {candidate_id}")
        ids.add(candidate_id)
        root_kind = require_candidate_field(candidate, "root_kind")
        layout_class = require_candidate_field(candidate, "layout_class")
        topology_family = require_candidate_field(candidate, "topology_family")
        root_kinds.add(str(root_kind))
        layout_classes.add(str(layout_class))
        if str(layout_class) in families_by_layout:
            families_by_layout[str(layout_class)].add(str(topology_family))
        if root_kind == "fabric.system":
            system_families.add(str(topology_family))
        if candidate.get("coordinates_semantic") is not False:
            raise AssertionError(f"{candidate_id} must not make visual coordinates semantic")
        if candidate.get("visual_metadata_role") not in {"metadata_only", "absent"}:
            raise AssertionError(f"{candidate_id} has invalid visual metadata role")
        connectivity = candidate.get("semantic_connectivity_source")
        if root_kind == "fabric.module" and connectivity != "graph_region_ssa":
            raise AssertionError(f"{candidate_id} module connectivity should be graph_region_ssa")
        if root_kind == "fabric.system" and connectivity != "fabric.link":
            raise AssertionError(f"{candidate_id} system connectivity should be fabric.link")
        if candidate.get("verifier_status") != "pass":
            raise AssertionError(f"{candidate_id} should carry verifier pass evidence")
        source_path = resolve_inventory_reference(
            inventory_path,
            require_candidate_field(candidate, "source_mlir"),
        )
        if not source_path.is_file():
            raise AssertionError(f"{candidate_id} source MLIR does not exist: {source_path}")
        expected_fingerprint = artifact_test_common.fingerprint(source_path)
        if candidate.get("source_mlir_fingerprint") != expected_fingerprint:
            raise AssertionError(f"{candidate_id} source MLIR fingerprint mismatch")
        coverage = candidate.get("construct_coverage")
        if not isinstance(coverage, dict) or not coverage:
            raise AssertionError(f"{candidate_id} lacks construct coverage")
        if str(topology_family) in EXPECTED_TOPOLOGY_SIGNATURES:
            signature = EXPECTED_TOPOLOGY_SIGNATURES[str(topology_family)]
            if layout_class != signature["layout_class"]:
                raise AssertionError(f"{candidate_id} has wrong layout signature")
            if candidate.get("fabric_root") != signature["fabric_root"]:
                raise AssertionError(f"{candidate_id} has wrong topology root")
            if coverage.get("tile_counts") != signature["tile_counts"]:
                raise AssertionError(f"{candidate_id} has wrong tile-count signature")
            if set(coverage.get("schedule_kinds", [])) != signature["schedule_kinds"]:
                raise AssertionError(f"{candidate_id} has wrong schedule signature")
            topology_fingerprints[str(topology_family)] = str(
                candidate.get("source_mlir_fingerprint")
            )
        consumers = candidate.get("downstream_consumers")
        if not isinstance(consumers, list) or not consumers:
            raise AssertionError(f"{candidate_id} lacks downstream consumer records")
        if not any(
            isinstance(record, dict)
            and record.get("consumer") == "fabric_verifier"
            and record.get("status") == "pass"
            for record in consumers
        ):
            raise AssertionError(f"{candidate_id} lacks fabric verifier consumer evidence")

    if "fabric.module" not in root_kinds or "fabric.system" not in root_kinds:
        raise AssertionError(f"inventory should include module and system roots: {root_kinds}")
    if "regular" not in layout_classes or "irregular" not in layout_classes:
        raise AssertionError(f"inventory should classify regular and irregular candidates: {layout_classes}")
    for layout_class, required_families in REQUIRED_TOPOLOGY_FAMILIES.items():
        missing = required_families - families_by_layout[layout_class]
        if missing:
            raise AssertionError(
                f"inventory missed {layout_class} topology families {sorted(missing)}: "
                f"{families_by_layout[layout_class]}"
            )
    missing_system_families = REQUIRED_SYSTEM_FAMILIES - system_families
    if missing_system_families:
        raise AssertionError(
            f"inventory missed system topology families {sorted(missing_system_families)}: "
            f"{system_families}"
        )
    shared_vector_id = "adg-builder::shared-vector-alu::shared_vector_alu_adg"
    if shared_vector_id not in ids:
        raise AssertionError(f"inventory missed reusable shared-vector ALU ADG: {ids}")
    shared_vector_math_id = "adg-builder::shared-vector-math::shared_vector_math_adg"
    if shared_vector_math_id not in ids:
        raise AssertionError(f"inventory missed reusable shared-vector math ADG: {ids}")
    shared_memory_reduction_id = "adg-builder::shared-memory-reduction::shared_memory_reduction_adg"
    if shared_memory_reduction_id not in ids:
        raise AssertionError(f"inventory missed reusable shared-memory reduction ADG: {ids}")
    shared_vector_mesh_id = "adg-builder::shared-vector-mesh::shared_vector_mesh_adg"
    if shared_vector_mesh_id not in ids:
        raise AssertionError(f"inventory missed reusable shared-vector mesh ADG: {ids}")
    duplicate_fingerprints = len(set(topology_fingerprints.values())) != len(
        topology_fingerprints
    )
    if duplicate_fingerprints:
        raise AssertionError(
            f"topology families should not share generated sources: {topology_fingerprints}"
        )
    return data


def assert_summary_projection(
    inventory: dict[str, object],
    summary_rows: list[dict[str, str]],
) -> None:
    candidates = {
        str(candidate["hardware_identity"]): str(candidate["candidate_id"])
        for candidate in inventory.get("candidates", [])
        if isinstance(candidate, dict)
    }
    rows_by_hardware = {row["hardware"]: row for row in summary_rows}
    missing = sorted(set(candidates) - set(rows_by_hardware))
    if missing:
        raise AssertionError(f"summary projection missed inventory candidates: {missing}")
    for hardware, candidate_id in candidates.items():
        row = rows_by_hardware[hardware]
        if row["verify_status"] != "pass":
            raise AssertionError(f"inventory projection row is not pass: {row}")
        diagnostic = row["diagnostic"]
        if f"candidate_id={candidate_id}" not in diagnostic:
            raise AssertionError(f"summary row does not reference candidate_id: {row}")
        if "inventory_id=" not in diagnostic:
            raise AssertionError(f"summary row does not reference inventory id: {row}")


def assert_audit_passes(paths: list[Path]) -> None:
    audit = intermediate_artifacts.audit(paths)
    if audit["verdict"] != "pass":
        raise AssertionError(json.dumps(audit, indent=2, sort_keys=True))


def assert_audit_fails(path: Path, expected: str) -> None:
    audit = intermediate_artifacts.audit([path])
    if audit["verdict"] != "fail":
        raise AssertionError(f"expected audit failure for {path}, got {audit}")
    diagnostics = "\n".join(str(item) for item in audit["diagnostics"])
    if expected not in diagnostics:
        raise AssertionError(f"expected {expected!r} in diagnostics:\n{diagnostics}")


def assert_summary_fails(
    repo: Path,
    inventory_path: Path,
    output: Path,
    expected: str,
) -> None:
    result = artifact_test_common.run_command(
        repo,
        [
            "bash",
            "test/fabric/run_adg_hardware_summary.sh",
            "--inventory",
            str(inventory_path),
            "--output",
            str(output),
        ],
    )
    if result.returncode == 0:
        raise AssertionError("blocked inventory summary projection unexpectedly succeeded")
    combined = result.stdout + result.stderr
    if expected not in combined:
        raise AssertionError(f"expected {expected!r} in projection diagnostics:\n{combined}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-adg-inventory-") as tmp:
        out_dir = Path(tmp)
        inventory_path = out_dir / "adg-inventory.json"
        mlir_dir = out_dir / "adg-inventory-mlir"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/fabric/run_adg_inventory.sh",
                "--output",
                str(inventory_path),
                "--mlir-output-dir",
                str(mlir_dir),
            ],
            "ADG inventory producer",
        )

        inventory = assert_inventory_shape(inventory_path)
        summary_path = out_dir / "adg-hardware-summary.csv"
        summary_rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            summary_path,
            HARDWARE_HEADER,
            "--inventory",
            str(inventory_path),
            label="ADG hardware inventory projection",
        )
        assert_summary_projection(inventory, summary_rows)
        assert_audit_passes([inventory_path, summary_path])

        forged = copy.deepcopy(inventory)
        forged["candidates"][0]["coordinates_semantic"] = True
        forged_path = out_dir / "forged-coordinate-semantic-adg-inventory.json"
        forged_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
        assert_audit_fails(forged_path, "visual coordinates must not be semantic")

        missing_diagnostic = copy.deepcopy(inventory)
        del missing_diagnostic["candidates"][0]["diagnostic"]
        missing_diagnostic_path = out_dir / "missing-diagnostic-adg-inventory.json"
        missing_diagnostic_path.write_text(
            json.dumps(missing_diagnostic, indent=2, sort_keys=True) + "\n"
        )
        assert_audit_fails(missing_diagnostic_path, "missing diagnostic")

        bad_tile = copy.deepcopy(inventory)
        module_candidate = next(
            candidate
            for candidate in bad_tile["candidates"]
            if candidate["root_kind"] == "fabric.module"
        )
        module_candidate["construct_coverage"]["tile_kinds"] = ["fu"]
        bad_tile_path = out_dir / "bad-tile-adg-inventory.json"
        bad_tile_path.write_text(json.dumps(bad_tile, indent=2, sort_keys=True) + "\n")
        assert_audit_fails(bad_tile_path, "unknown module tile kind")

        bad_system = copy.deepcopy(inventory)
        system_candidate = next(
            candidate
            for candidate in bad_system["candidates"]
            if candidate["root_kind"] == "fabric.system"
        )
        system_candidate["construct_coverage"]["link_count"] = 0
        bad_system_path = out_dir / "bad-system-adg-inventory.json"
        bad_system_path.write_text(json.dumps(bad_system, indent=2, sort_keys=True) + "\n")
        assert_audit_fails(bad_system_path, "system verifier pass has no links")

        blocked_inventory = copy.deepcopy(inventory)
        blocked_inventory["status"] = "blocked"
        blocked_inventory["diagnostics"] = ["one ADG Builder recipe failed"]
        blocked_inventory["candidate_count"] = len(blocked_inventory["candidates"]) + 1
        blocked_path = out_dir / "blocked-adg-inventory.json"
        blocked_path.write_text(json.dumps(blocked_inventory, indent=2, sort_keys=True) + "\n")
        assert_summary_fails(
            repo,
            blocked_path,
            out_dir / "blocked-adg-hardware-summary.csv",
            "ADG inventory is not pass",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
