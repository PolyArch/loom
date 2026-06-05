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
    "byte_swap",
    "bisection_step",
    "binary_search",
    "clz",
    "compare_swap",
    "convolve_1d",
    "convolve_1d_same",
    "correlation",
    "conv1d",
    "covariance",
    "crc32",
    "cross_product",
    "ctz",
    "cumsum",
    "delta_decode",
    "delta_encode",
    "dot_product_3d",
    "dotproduct",
    "downsample",
    "downsample_avg",
    "find_first_set",
    "fir_filter",
    "gemm",
    "gemv",
    "gather",
    "hash_mix",
    "hist_bin",
    "histogram",
    "integrate_trapz",
    "interpolate_linear",
    "lower_bound",
    "mean",
    "matvec",
    "merge",
    "moving_avg",
    "normalize_vec3",
    "newton_iter",
    "outer",
    "pack_bits",
    "partition",
    "parity",
    "prefix_sum",
    "prefix_sum_exclusive",
    "prefix_sum_inclusive",
    "quantile",
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
    "string_compare",
    "string_hash",
    "transpose",
    "transform_point",
    "unpack_bits",
    "upsample",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
