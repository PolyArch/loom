#!/usr/bin/env python3
"""Shared assertions for the default app CGRA-sim batch."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED_DEFAULT_BATCH_CASES = {
    "axpy",
    "bit_reverse",
    "byte_swap",
    "compare_swap",
    "conv1d",
    "convolve_1d",
    "correlation",
    "cumsum",
    "delta_encode",
    "dot_product_3d",
    "dotproduct",
    "downsample",
    "downsample_avg",
    "gemm",
    "gemv",
    "hash_mix",
    "integrate_trapz",
    "matmul",
    "matvec",
    "mean",
    "prefix_sum",
    "prefix_sum_exclusive",
    "prefix_sum_inclusive",
    "reduction",
    "relu",
    "rotate_bits",
    "sbox_lookup",
    "spmv",
    "upsample",
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
    if set(hardware) != EXPECTED_DEFAULT_BATCH_CASES:
        raise AssertionError(
            "default CGRA-sim batch should expose the shared-ADG promoted app set: "
            f"{sorted(hardware)}"
        )
    return hardware
