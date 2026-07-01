#!/usr/bin/env python3
"""Regression test for the app corpus manifest contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


EXPECTED_CASES = {
    "autocorrelation",
    "axpy",
    "batchnorm",
    "bit_reverse",
    "bitrev",
    "bitrev_complex",
    "bitonic_stage",
    "bitonic_stage-modified",
    "bitonic_stage-tweak",
    "byte_swap",
    "cdma",
    "bisection_step",
    "binary_search",
    "clz",
    "col2im",
    "compare_swap",
    "compact",
    "compact_predicate",
    "convolve_1d",
    "convolve_1d_same",
    "conv2d",
    "correlation",
    "conv1d",
    "covariance",
    "crc32",
    "cross_product",
    "ctz",
    "cumsum",
    "distance_point",
    "edit_distance_step",
    "delta_decode",
    "delta_encode",
    "dot_product_3d",
    "dotprod",
    "dotproduct",
    "downsample",
    "downsample_avg",
    "edge_update",
    "edge_update_batch",
    "find_first_set",
    "fir_filter",
    "fir_filter_stateful",
    "gemm",
    "gemv",
    "gf_mul",
    "gather",
    "hash_mix",
    "hist_bin",
    "histogram",
    "histogram_strided",
    "im2col",
    "integrate_trapz",
    "interpolate_linear",
    "lower_bound",
    "mean",
    "mat3x3_mult",
    "matmul",
    "mmtile",
    "matvec",
    "merge",
    "moving_avg",
    "modexp",
    "modmul",
    "normalize_vec3",
    "newton_iter",
    "outer",
    "pack_bits",
    "partition",
    "parity",
    "pool_avg",
    "pool_max",
    "prefix_sum",
    "prefix_sum_exclusive",
    "prefix_sum_inclusive",
    "quantile",
    "quat_mult",
    "popcount",
    "relu",
    "reduction",
    "rotate_bits",
    "rle_decode",
    "rle_encode",
    "runge_kutta_step",
    "scatter_add",
    "sbox_lookup",
    "sigmoid",
    "softmax",
    "spmv",
    "spmspm",
    "spmspv",
    "sort_bubble",
    "sort_insertion",
    "sort_merge",
    "sort_quick",
    "string_compare",
    "string_hash",
    "stream_update",
    "transpose",
    "transform_point",
    "unpack_bits",
    "upsample",
    "upsample_linear",
    "upper_bound",
    "variance",
    "vecadd",
    "vecmul",
    "vecnorm_l1",
    "vecnorm_l2",
    "vecscale",
    "vecsum",
    "vecsum-while",
    "window_blackman",
    "window_hamming",
    "window_hanning",
    "xor_block",
}


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    manifest = repo / "test" / "app" / "manifest.json"
    artifact_test_common.require_success(
        repo,
        ["bash", "test/app/run_manifest_check.sh"],
        "app manifest validation",
    )

    result = artifact_test_common.require_success(
        repo,
        ["python3", "test/app/app_manifest.py", "list", "--tier", "run"],
        "app manifest list",
    )
    cases = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if cases != EXPECTED_CASES:
        raise AssertionError(f"manifest run tier cases {cases} do not match {EXPECTED_CASES}")

    data = json.loads(manifest.read_text())
    if {entry["case"] for entry in data["cases"]} != EXPECTED_CASES:
        raise AssertionError(f"manifest cases do not match expected seed set: {data}")
    for entry in data["cases"]:
        case = entry["case"]
        for field in ("compiler_flags", "link_flags", "expected_executables"):
            if field not in entry:
                raise AssertionError(f"{case}: manifest entry lacks {field}")
        if not isinstance(entry["compiler_flags"], list) or any(
            not isinstance(flag, str) for flag in entry["compiler_flags"]
        ):
            raise AssertionError(f"{case}: compiler_flags must be a string list")
        if not isinstance(entry["link_flags"], list) or any(
            not isinstance(flag, str) for flag in entry["link_flags"]
        ):
            raise AssertionError(f"{case}: link_flags must be a string list")
        if not isinstance(entry["expected_executables"], list) or len(entry["expected_executables"]) != 2:
            raise AssertionError(f"{case}: expected_executables should name two variants")
        if not all(isinstance(name, str) and name for name in entry["expected_executables"]):
            raise AssertionError(f"{case}: expected_executables must contain non-empty strings")

    with artifact_test_common.repo_temp_dir(repo, "loom-bad-app-manifest-") as tmp:
        bad_manifest = Path(tmp) / "manifest.json"
        bad_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": [
                        {
                            "case": "vecadd",
                            "language": "c",
                            "sources": ["missing.c"],
                            "expected_stdout": "expected.txt",
                            "tiers": ["run"],
                            "compiler_flags": [],
                            "link_flags": [],
                            "expected_executables": ["main_func", "main_inline"],
                            "feature_tags": ["vector"],
                        }
                    ],
                }
            )
            + "\n"
        )
        result = artifact_test_common.run_command(
            repo,
            ["python3", "test/app/app_manifest.py", "validate", "--manifest", str(bad_manifest)],
        )
        if result.returncode == 0:
            raise AssertionError("bad manifest with missing source unexpectedly passed")
        if "missing source" not in result.stderr:
            raise AssertionError(f"bad manifest diagnostic should name missing source: {result.stderr}")

        bad_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": [
                        {
                            "case": "vecadd",
                            "language": "c",
                            "sources": ["main_func.c"],
                            "expected_stdout": "expected.txt",
                            "tiers": ["run"],
                            "compiler_flags": [],
                            "link_flags": [],
                            "expected_executables": ["main_func", "main_inline"],
                            "feature_tags": [1],
                        }
                    ],
                }
            )
            + "\n"
        )
        result = artifact_test_common.run_command(
            repo,
            ["python3", "test/app/app_manifest.py", "validate", "--manifest", str(bad_manifest)],
        )
        if result.returncode == 0:
            raise AssertionError("bad manifest with non-string tag unexpectedly passed")
        if "feature_tags must contain non-empty strings" not in result.stderr:
            raise AssertionError(f"bad manifest diagnostic should name non-string tag: {result.stderr}")

        bad_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": [
                        {
                            "case": "vecadd",
                            "language": "c",
                            "sources": ["main_func.c"],
                            "expected_stdout": "expected.txt",
                            "tiers": ["run"],
                            "compiler_flags": "-O2",
                            "link_flags": [],
                            "expected_executables": ["main_func", "main_inline"],
                            "feature_tags": ["vector"],
                        }
                    ],
                }
            )
            + "\n"
        )
        result = artifact_test_common.run_command(
            repo,
            ["python3", "test/app/app_manifest.py", "validate", "--manifest", str(bad_manifest)],
        )
        if result.returncode == 0:
            raise AssertionError("bad manifest with scalar compiler_flags unexpectedly passed")
        if "compiler_flags must be a list" not in result.stderr:
            raise AssertionError(f"bad manifest diagnostic should name compiler_flags: {result.stderr}")

        bad_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": [
                        {
                            "case": "vecadd",
                            "language": "c",
                            "sources": ["main_func.c"],
                            "expected_stdout": "expected.txt",
                            "tiers": ["run"],
                            "compiler_flags": [],
                            "link_flags": [],
                            "expected_executables": [],
                            "feature_tags": ["vector"],
                        }
                    ],
                }
            )
            + "\n"
        )
        result = artifact_test_common.run_command(
            repo,
            ["python3", "test/app/app_manifest.py", "validate", "--manifest", str(bad_manifest)],
        )
        if result.returncode == 0:
            raise AssertionError("bad manifest with empty expected_executables unexpectedly passed")
        if "expected_executables must be a non-empty list" not in result.stderr:
            raise AssertionError(f"bad manifest diagnostic should name expected_executables: {result.stderr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
