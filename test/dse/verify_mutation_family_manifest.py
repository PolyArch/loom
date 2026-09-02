"""Verify a mutation-family repair left a durable manifest observation.

Each sharded mutation-matrix invocation must publish one
``joint_hardware_mutation_repair`` migration observation whose typed
families cover the expected family set, whose durable record identity
(``loom.dse.hardware_mutation_repair_record``) is named, and whose repair
dispositions and cold-versus-incremental accounting survived manifest
collection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"mutation family manifest: {message}")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        fail("usage: verify_mutation_family_manifest.py MANIFEST FAMILY...")
    manifest = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    expected = set(argv[2:])
    observations = [
        entry
        for entry in manifest["facts"]["migration_observations"]
        if entry.get("operation") == "joint_hardware_mutation_repair"
    ]
    if not observations:
        fail("manifest has no joint_hardware_mutation_repair observation")
    matched = None
    for entry in observations:
        families = set(entry.get("families", []))
        if expected <= families:
            matched = entry
            break
    if matched is None:
        observed = sorted(
            family
            for entry in observations
            for family in entry.get("families", [])
        )
        fail(
            "no observation covers families "
            f"{sorted(expected)}; observed {observed}"
        )
    for key in (
        "record",
        "parent_mapping",
        "child_system",
        "mapping_reuse_disposition",
        "system_mapping_reuse_disposition",
        "incremental_mapping_count",
        "incremental_wall_time_ns",
        "incremental_verifier_work",
    ):
        if key not in matched or matched[key] in ("", None):
            fail(f"observation omits {key}")
    if matched.get("cold_comparison_baseline"):
        for key in ("cold_mapping_count", "cold_wall_time_ns",
                    "cold_verifier_work"):
            if key not in matched:
                fail(f"cold baseline observation omits {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
