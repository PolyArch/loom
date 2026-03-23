#!/usr/bin/env python3
"""TDG description for Stereo Vision pipeline."""

kernels = [
    {"name": "harris_corner", "type": "feature_detect", "source": "harris_corner.c"},
    {"name": "sad_matching", "type": "matching", "source": "sad_matching.c"},
    {"name": "stereo_disparity", "type": "optimization", "source": "stereo_disparity.c"},
    {"name": "image_warp", "type": "interpolation", "source": "image_warp.c"},
    {"name": "post_filter", "type": "filter", "source": "post_filter.c"},
]

contracts = [
    {
        "producer": "harris_corner",
        "consumer": "sad_matching",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [64, 64],
        "production_rate": 4096,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "sad_matching",
        "consumer": "stereo_disparity",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [64, 64, 64],
        "production_rate": 262144,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "stereo_disparity",
        "consumer": "image_warp",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [64, 64],
        "production_rate": 4096,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "image_warp",
        "consumer": "post_filter",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [64, 64],
        "production_rate": 4096,
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
