#!/usr/bin/env python3
"""TDG description for Graph Analytics pipeline."""

kernels = [
    {"name": "bfs_traversal", "type": "frontier_based", "source": "bfs_traversal.c"},
    {"name": "pagerank_spmv", "type": "spmv_iterative", "source": "pagerank_spmv.c"},
    {"name": "triangle_count", "type": "set_intersection", "source": "triangle_count.c"},
    {"name": "label_prop", "type": "neighbor_vote", "source": "label_prop.c"},
]

contracts = [
    {
        "producer": "bfs_traversal",
        "consumer": "pagerank_spmv",
        "ordering": "FIFO",
        "data_type": "int32",
        "tile_shape": [1024],
        "production_rate": 1024,
        "double_buffering": False,
        "visibility": "GLOBAL_MEM",
        "note": "BFS levels used as initial vertex ordering for PageRank",
    },
    {
        "producer": "pagerank_spmv",
        "consumer": "label_prop",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [1024],
        "production_rate": 1024,
        "double_buffering": False,
        "visibility": "GLOBAL_MEM",
        "note": "PageRank scores can seed initial label weights",
    },
    {
        "producer": "bfs_traversal",
        "consumer": "triangle_count",
        "ordering": "FIFO",
        "data_type": "int32",
        "tile_shape": [1024],
        "production_rate": 1024,
        "double_buffering": False,
        "visibility": "GLOBAL_MEM",
        "note": "BFS ordering for triangle count vertex partitioning",
    },
]

if __name__ == "__main__":
    print(f"TDG: {len(kernels)} kernels, {len(contracts)} contracts")
    for k in kernels:
        print(f"  Kernel: {k['name']} ({k['type']})")
    for c in contracts:
        print(f"  Contract: {c['producer']} -> {c['consumer']} [{c['ordering']}]")
