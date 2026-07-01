#!/usr/bin/env python3
"""Shared assertions for the default app CGRA-sim batch."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED_DEFAULT_BATCH_CASES = {
    "autocorrelation",
    "axpy",
    "batchnorm",
    "binary_search",
    "bisection_step",
    "bitonic_stage",
    "bitonic_stage-modified",
    "bitonic_stage-tweak",
    "bit_reverse",
    "bitrev",
    "bitrev_complex",
    "byte_swap",
    "cdma",
    "clz",
    "compare_swap",
    "compact",
    "compact_predicate",
    "conv1d",
    "conv2d",
    "convolve_1d",
    "convolve_1d_same",
    "correlation",
    "covariance",
    "crc32",
    "cross_product",
    "quat_mult",
    "ctz",
    "cumsum",
    "delta_decode",
    "delta_encode",
    "distance_point",
    "edit_distance_step",
    "dot_product_3d",
    "dotprod",
    "dotproduct",
    "downsample",
    "downsample_avg",
    "find_first_set",
    "gemm",
    "gemv",
    "gf_mul",
    "fir_filter",
    "fir_filter_stateful",
    "gather",
    "hash_mix",
    "hist_bin",
    "histogram",
    "histogram_strided",
    "im2col",
    "integrate_trapz",
    "interpolate_linear",
    "jacobi_stencil_5pt",
    "lower_bound",
    "mat3x3_mult",
    "matmul",
    "matvec",
    "mean",
    "merge",
    "mmtile",
    "moving_avg",
    "modexp",
    "modmul",
    "newton_iter",
    "normalize_vec3",
    "outer",
    "pack_bits",
    "parity",
    "partition",
    "popcount",
    "pool_avg",
    "pool_max",
    "prefix_sum",
    "prefix_sum_exclusive",
    "prefix_sum_inclusive",
    "quantile",
    "reduction",
    "relu",
    "rotate_bits",
    "rle_decode",
    "rle_encode",
    "runge_kutta_step",
    "sbox_lookup",
    "sigmoid",
    "softmax",
    "window_blackman",
    "window_hamming",
    "window_hanning",
    "scatter_add",
    "spmv",
    "spmspv",
    "sort_bubble",
    "stream_update",
    "string_hash",
    "transform_point",
    "transpose",
    "unpack_bits",
    "upper_bound",
    "upsample",
    "upsample_linear",
    "variance",
    "vecadd",
    "vecmul",
    "vecnorm_l1",
    "vecnorm_l2",
    "vecscale",
    "vecsum",
    "vecsum-while",
    "xor_block",
}


def default_batch_hardware(repo: Path) -> dict[str, str]:
    manifest = json.loads((repo / "test/app/default-cgra-sim-batch.json").read_text())
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise AssertionError(f"default CGRA-sim batch manifest cases must be a list: {manifest}")
    hardware = {}
    for entry in cases:
        if not isinstance(entry, dict):
            raise AssertionError(f"default CGRA-sim batch entry must be an object: {entry}")
        case = entry.get("case")
        target = entry.get("hardware")
        if not isinstance(case, str) or not isinstance(target, str):
            raise AssertionError(f"default CGRA-sim batch entry has invalid fields: {entry}")
        hardware[case] = target
    if len(hardware) != len(cases):
        raise AssertionError(f"default CGRA-sim batch manifest contains duplicate cases: {manifest}")
    if set(hardware) != EXPECTED_DEFAULT_BATCH_CASES:
        raise AssertionError(
            "default CGRA-sim batch should expose the shared-ADG promoted app set: "
            f"{sorted(hardware)}"
        )
    return hardware
