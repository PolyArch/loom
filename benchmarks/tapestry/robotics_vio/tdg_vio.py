#!/usr/bin/env python3
"""TDG description for Visual-Inertial Odometry pipeline."""

kernels = [
    {"name": "imu_integration", "type": "sequential_accum", "source": "imu_integration.c"},
    {"name": "fast_detect", "type": "stencil_2d", "source": "fast_detect.c"},
    {"name": "orb_descriptor", "type": "patch_compute", "source": "orb_descriptor.c"},
    {"name": "feature_match", "type": "brute_force_search", "source": "feature_match.c"},
    {"name": "pose_estimate", "type": "linear_algebra", "source": "pose_estimate.c"},
]

contracts = [
    {
        "producer": "imu_integration",
        "consumer": "pose_estimate",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [200, 3],
        "production_rate": 600,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "fast_detect",
        "consumer": "orb_descriptor",
        "ordering": "FIFO",
        "data_type": "int32",
        "tile_shape": [500, 2],
        "production_rate": 1000,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "orb_descriptor",
        "consumer": "feature_match",
        "ordering": "FIFO",
        "data_type": "uint32",
        "tile_shape": [500, 8],
        "production_rate": 4000,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "feature_match",
        "consumer": "pose_estimate",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [100, 4],
        "production_rate": 400,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
]

if __name__ == "__main__":
    print(f"TDG: {len(kernels)} kernels, {len(contracts)} contracts")
    for k in kernels:
        print(f"  Kernel: {k['name']} ({k['type']})")
    for c in contracts:
        print(f"  Contract: {c['producer']} -> {c['consumer']} [{c['ordering']}]")
