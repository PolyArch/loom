#!/usr/bin/env python3
"""TDG description for STARK Proof pipeline."""

kernels = [
    {"name": "ntt", "type": "butterfly_transform", "source": "ntt.c"},
    {"name": "msm", "type": "bucket_accumulate", "source": "msm.c"},
    {"name": "poseidon_hash", "type": "permutation_sponge", "source": "poseidon_hash.c"},
    {"name": "poly_eval", "type": "horner_batch", "source": "poly_eval.c"},
    {"name": "proof_compose", "type": "linear_combination", "source": "proof_compose.c"},
]

contracts = [
    {
        "producer": "ntt",
        "consumer": "poly_eval",
        "ordering": "FIFO",
        "data_type": "uint32",
        "tile_shape": [1024],
        "production_rate": 1024,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
        "note": "NTT output coefficients evaluated at query points",
    },
    {
        "producer": "poly_eval",
        "consumer": "proof_compose",
        "ordering": "FIFO",
        "data_type": "uint32",
        "tile_shape": [256],
        "production_rate": 256,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
        "note": "Evaluated polynomials fed into linear combination",
    },
    {
        "producer": "poseidon_hash",
        "consumer": "proof_compose",
        "ordering": "FIFO",
        "data_type": "uint32",
        "tile_shape": [4],
        "production_rate": 4,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
        "note": "Hash output used as Fiat-Shamir challenge seed",
    },
    {
        "producer": "msm",
        "consumer": "proof_compose",
        "ordering": "FIFO",
        "data_type": "uint32",
        "tile_shape": [3],
        "production_rate": 3,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
        "note": "MSM result (commitment point) included in proof",
    },
    {
        "producer": "ntt",
        "consumer": "poseidon_hash",
        "ordering": "FIFO",
        "data_type": "uint32",
        "tile_shape": [8],
        "production_rate": 8,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
        "note": "NTT results committed via Poseidon hash",
    },
]

if __name__ == "__main__":
    print(f"TDG: {len(kernels)} kernels, {len(contracts)} contracts")
    for k in kernels:
        print(f"  Kernel: {k['name']} ({k['type']})")
    for c in contracts:
        print(f"  Contract: {c['producer']} -> {c['consumer']} [{c['ordering']}]")
