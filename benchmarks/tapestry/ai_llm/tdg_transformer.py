#!/usr/bin/env python3
"""TDG description for Transformer Layer pipeline."""

kernels = [
    {"name": "qkv_proj", "type": "matmul", "source": "qkv_proj.c"},
    {"name": "attn_score", "type": "batched_matmul", "source": "attn_score.c"},
    {"name": "softmax", "type": "elementwise", "source": "softmax.c"},
    {"name": "attn_output", "type": "batched_matmul", "source": "attn_output.c"},
    {"name": "ffn1", "type": "matmul", "source": "ffn1.c"},
    {"name": "gelu", "type": "elementwise", "source": "gelu.c"},
    {"name": "ffn2", "type": "matmul", "source": "ffn2.c"},
    {"name": "layernorm", "type": "reduction", "source": "layernorm.c"},
]

contracts = [
    {
        "producer": "qkv_proj",
        "consumer": "attn_score",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 64],
        "production_rate": 2048,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "attn_score",
        "consumer": "softmax",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 128],
        "production_rate": 4096,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "softmax",
        "consumer": "attn_output",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 128],
        "production_rate": 4096,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "attn_output",
        "consumer": "ffn1",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 512],
        "production_rate": 16384,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "ffn1",
        "consumer": "gelu",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 2048],
        "production_rate": 65536,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "gelu",
        "consumer": "ffn2",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 2048],
        "production_rate": 65536,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "ffn2",
        "consumer": "layernorm",
        "ordering": "FIFO",
        "data_type": "float32",
        "tile_shape": [32, 512],
        "production_rate": 16384,
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
