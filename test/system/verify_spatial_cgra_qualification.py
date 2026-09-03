#!/usr/bin/env python3
"""Verify product-owned Spatial CGRA qualification observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


PROFILE_SCHEMA = "loom.spatial_cgra_qualification.1"
MANIFEST_SCHEMA = "loom.execution_matrix_workspace.2.0"
U64_MAX = (1 << 64) - 1

PROFILE_FIELDS = {
    "schema",
    "warmup_runs",
    "measurement_runs",
    "process_peak_resident_bytes",
    "runtime_manifest",
    "deployment",
    "invocations",
}
INVOCATION_FIELDS = {
    "invocation_ordinal",
    "dense_coordinates",
    "dataflow",
    "spatial_mapping",
    "workload",
    "runtime_input",
    "warmups",
    "measurements",
}
SAMPLE_FIELDS = {
    "attempt_ordinal",
    "reference_cycles",
    "attempt_setup_wall_nanoseconds",
    "attempt_setup_process_cpu_nanoseconds",
    "engine_active_wall_nanoseconds",
    "engine_active_process_cpu_nanoseconds",
    "observation_projection_wall_nanoseconds",
    "observation_projection_process_cpu_nanoseconds",
    "artifact_publication_wall_nanoseconds",
    "artifact_publication_process_cpu_nanoseconds",
    "event_frame_count",
    "physical_request_count",
    "physical_grant_count",
    "physical_retirement_count",
    "physical_grant_wait_cycle_sum",
    "physical_grant_wait_cycle_max",
    "physical_grant_delayed_count",
    "request",
    "evidence",
    "execution",
}
ROOT_FIELDS = {"schema", "schema_version", "artifact"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_without_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats field {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> object:
    try:
        text = path.read_bytes().decode("ascii")
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical ASCII JSON from {path}") from error


def package_artifact(package: Path, name: str) -> str:
    path = package / name
    try:
        artifact = path.read_bytes().decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read Deployment package {name}") from error
    require(
        len(artifact) == 64
        and artifact == artifact.lower()
        and all(character in "0123456789abcdef" for character in artifact),
        f"Deployment package {name} is not a canonical artifact identity",
    )
    object_path = package / "objects" / artifact
    require(
        object_path.is_file() and not object_path.is_symlink(),
        f"Deployment package {name} object is unavailable",
    )
    return artifact


def exact_object(value: object, fields: set[str], what: str) -> Mapping[str, object]:
    require(
        isinstance(value, dict) and set(value) == fields, f"{what} has the wrong shape"
    )
    return value


def unsigned(value: object, what: str, *, positive: bool = False) -> int:
    require(
        not isinstance(value, bool)
        and isinstance(value, int)
        and (value > 0 if positive else value >= 0)
        and value <= U64_MAX,
        f"{what} is not a canonical unsigned 64-bit value",
    )
    return value


def artifact_root(value: object, schema: str, what: str) -> Mapping[str, object]:
    root = exact_object(value, ROOT_FIELDS, what)
    version = root["schema_version"]
    artifact = root["artifact"]
    require(root["schema"] == schema, f"{what} has the wrong schema")
    require(
        isinstance(version, str)
        and len(version.split(".")) == 2
        and all(part.isdecimal() and part for part in version.split(".")),
        f"{what} has a noncanonical schema version",
    )
    require(
        isinstance(artifact, str)
        and len(artifact) == 64
        and artifact == artifact.lower()
        and all(character in "0123456789abcdef" for character in artifact),
        f"{what} has a noncanonical artifact identity",
    )
    return root


def validate_sample(
    value: object, expected_attempt: int, what: str
) -> tuple[tuple[int, ...], tuple[Mapping[str, object], ...]]:
    sample = exact_object(value, SAMPLE_FIELDS, what)
    require(
        unsigned(sample["attempt_ordinal"], f"{what} attempt ordinal")
        == expected_attempt,
        f"{what} is not bound to its attempt generation",
    )
    cycles = unsigned(
        sample["reference_cycles"], f"{what} reference cycles", positive=True
    )
    engine_wall = unsigned(
        sample["engine_active_wall_nanoseconds"],
        f"{what} engine-active wall time",
        positive=True,
    )
    setup_wall = unsigned(
        sample["attempt_setup_wall_nanoseconds"], f"{what} attempt-setup wall time"
    )
    projection_wall = unsigned(
        sample["observation_projection_wall_nanoseconds"],
        f"{what} observation-projection wall time",
    )
    publication_wall = unsigned(
        sample["artifact_publication_wall_nanoseconds"],
        f"{what} artifact-publication wall time",
    )
    require(
        setup_wall + engine_wall + projection_wall + publication_wall <= U64_MAX,
        f"{what} disjoint wall-time components overflow",
    )
    cpu_fields = (
        "attempt_setup_process_cpu_nanoseconds",
        "engine_active_process_cpu_nanoseconds",
        "observation_projection_process_cpu_nanoseconds",
        "artifact_publication_process_cpu_nanoseconds",
    )
    cpu_values = tuple(sample[field] for field in cpu_fields)
    require(
        all(value is None for value in cpu_values)
        or all(value is not None for value in cpu_values),
        f"{what} process CPU timing is fragmentary",
    )
    if cpu_values[0] is not None:
        cpu_sum = sum(
            unsigned(value, f"{what} process CPU timing") for value in cpu_values
        )
        require(cpu_sum <= U64_MAX, f"{what} process CPU timing overflows")

    event_frames = unsigned(
        sample["event_frame_count"], f"{what} event-frame count", positive=True
    )
    requests = unsigned(sample["physical_request_count"], f"{what} request count")
    grants = unsigned(sample["physical_grant_count"], f"{what} grant count")
    retirements = unsigned(
        sample["physical_retirement_count"], f"{what} retirement count"
    )
    wait_sum = unsigned(
        sample["physical_grant_wait_cycle_sum"], f"{what} grant-wait sum"
    )
    wait_max = unsigned(
        sample["physical_grant_wait_cycle_max"], f"{what} grant-wait maximum"
    )
    delayed = unsigned(
        sample["physical_grant_delayed_count"], f"{what} delayed-grant count"
    )
    require(
        requests == grants == retirements,
        f"{what} physical request lifecycle did not close",
    )
    require(
        delayed <= grants and wait_max <= wait_sum,
        f"{what} contention counters are inconsistent",
    )
    require(
        (delayed == 0 and wait_sum == 0 and wait_max == 0)
        or (delayed > 0 and wait_sum >= delayed and wait_max > 0),
        f"{what} delayed-grant accounting is inconsistent",
    )
    roots = (
        artifact_root(sample["request"], "evaluation.request", f"{what} request"),
        artifact_root(sample["evidence"], "evaluation.evidence", f"{what} Evidence"),
        artifact_root(
            sample["execution"], "loom.simulation_execution", f"{what} execution"
        ),
    )
    deterministic = (
        cycles,
        event_frames,
        requests,
        grants,
        retirements,
        wait_sum,
        wait_max,
        delayed,
    )
    return deterministic, roots


def validate(profile_value: object, manifest_value: object, package: Path) -> None:
    profile = exact_object(profile_value, PROFILE_FIELDS, "qualification profile")
    require(
        profile["schema"] == PROFILE_SCHEMA,
        "qualification profile has the wrong schema",
    )
    warmup_runs = unsigned(profile["warmup_runs"], "warmup count", positive=True)
    measurement_runs = unsigned(
        profile["measurement_runs"], "measurement count", positive=True
    )
    unsigned(
        profile["process_peak_resident_bytes"],
        "process peak resident memory",
        positive=True,
    )
    runtime_manifest = artifact_root(
        profile["runtime_manifest"],
        "loom.application.runtime_manifest",
        "qualification runtime_manifest",
    )
    deployment = artifact_root(
        profile["deployment"], "loom.deployment", "qualification deployment"
    )
    require(
        runtime_manifest["artifact"] == package_artifact(package, "application"),
        "profile names a foreign runtime Manifest",
    )
    require(
        deployment["artifact"] == package_artifact(package, "root"),
        "profile names a foreign Deployment package root",
    )

    invocations = profile["invocations"]
    require(
        isinstance(invocations, list) and invocations, "qualification has no invocation"
    )
    final_roots: dict[int, tuple[Mapping[str, object], ...]] = {}
    dense_coordinates: dict[int, list[object]] = {}
    invocation_inputs: dict[int, dict[str, Mapping[str, object]]] = {}
    for ordinal, value in enumerate(invocations):
        invocation = exact_object(value, INVOCATION_FIELDS, "qualification invocation")
        require(
            unsigned(invocation["invocation_ordinal"], "invocation ordinal") == ordinal,
            "qualification invocation order is not canonical",
        )
        coordinates = invocation["dense_coordinates"]
        require(isinstance(coordinates, list), "dense coordinates are not an array")
        for coordinate in coordinates:
            unsigned(coordinate, "dense coordinate")
        dense_coordinates[ordinal] = coordinates
        inputs: dict[str, Mapping[str, object]] = {}
        for field, schema in (
            ("dataflow", "loom.canonical_dataflow"),
            ("spatial_mapping", "loom.mapping"),
            ("workload", "loom.simulation_workload"),
            ("runtime_input", "loom.simulation_runtime_input"),
        ):
            inputs[field] = artifact_root(
                invocation[field], schema, f"invocation {field}"
            )
        invocation_inputs[ordinal] = inputs
        warmups = invocation["warmups"]
        measurements = invocation["measurements"]
        require(
            isinstance(warmups, list) and len(warmups) == warmup_runs,
            "invocation has the wrong warmup count",
        )
        require(
            isinstance(measurements, list) and len(measurements) == measurement_runs,
            "invocation has the wrong measurement count",
        )
        samples = warmups + measurements
        deterministic: tuple[int, ...] | None = None
        canonical_roots: tuple[Mapping[str, object], ...] | None = None
        for attempt, sample in enumerate(samples):
            observed, roots = validate_sample(
                sample, attempt, f"invocation {ordinal} attempt {attempt}"
            )
            if deterministic is None:
                deterministic = observed
                canonical_roots = roots
            else:
                require(
                    observed == deterministic, "CGRA counters changed across attempts"
                )
                require(
                    roots == canonical_roots,
                    "canonical CGRA roots changed across attempts",
                )
        final_roots[ordinal] = validate_sample(
            measurements[-1],
            warmup_runs + measurement_runs - 1,
            f"invocation {ordinal} final measurement",
        )[1]

    require(isinstance(manifest_value, dict), "execution manifest is not an object")
    manifest = manifest_value
    require(
        manifest.get("schema") == MANIFEST_SCHEMA,
        "execution manifest has the wrong schema",
    )
    require(
        manifest.get("deployment") == profile["deployment"],
        "profile names a foreign Deployment",
    )
    runs = manifest.get("runs")
    require(isinstance(runs, list), "execution manifest has no run inventory")
    product_roots: dict[int, tuple[Mapping[str, object], ...]] = {}
    for run in runs:
        if (
            not isinstance(run, dict)
            or run.get("scope") != "spatial"
            or run.get("engine") != "cgra"
        ):
            continue
        ordinal = unsigned(run.get("invocation_ordinal"), "manifest invocation ordinal")
        require(
            ordinal not in product_roots, "manifest repeats a Spatial CGRA invocation"
        )
        require(
            run.get("dense_coordinates") == dense_coordinates.get(ordinal),
            "manifest and profile dense coordinates differ",
        )
        expected_inputs = invocation_inputs.get(ordinal)
        require(expected_inputs is not None, "manifest names a foreign invocation")
        for field, schema in (
            ("dataflow", "loom.canonical_dataflow"),
            ("spatial_mapping", "loom.mapping"),
            ("workload", "loom.simulation_workload"),
            ("runtime_input", "loom.simulation_runtime_input"),
        ):
            require(
                artifact_root(run.get(field), schema, f"manifest {field}")
                == expected_inputs[field],
                f"manifest and profile {field} differ",
            )
        product_roots[ordinal] = (
            artifact_root(run.get("request"), "evaluation.request", "manifest request"),
            artifact_root(
                run.get("evidence"), "evaluation.evidence", "manifest Evidence"
            ),
            artifact_root(
                run.get("execution"), "loom.simulation_execution", "manifest execution"
            ),
        )
    require(
        product_roots == final_roots,
        "final measured CGRA roots differ from the product manifest",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--deployment-package", required=True, type=Path)
    arguments = parser.parse_args()
    validate(
        read_json(arguments.profile),
        read_json(arguments.manifest),
        arguments.deployment_package,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
