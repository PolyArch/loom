#!/usr/bin/env python3
"""TDG description for OFDM Receiver Chain pipeline."""

kernels = [
    {"name": "fft_butterfly", "type": "fft", "source": "fft_butterfly.c"},
    {"name": "channel_est", "type": "interpolation", "source": "channel_est.c"},
    {"name": "equalizer", "type": "elementwise", "source": "equalizer.c"},
    {"name": "qam_demod", "type": "demapping", "source": "qam_demod.c"},
    {"name": "viterbi", "type": "decoder", "source": "viterbi.c"},
    {"name": "crc_check", "type": "check", "source": "crc_check.c"},
]

contracts = [
    {
        "producer": "fft_butterfly",
        "consumer": "channel_est",
        "ordering": "FIFO",
        "data_type": "complex64",
        "tile_shape": [4096],
        "production_rate": 4096,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "channel_est",
        "consumer": "equalizer",
        "ordering": "FIFO",
        "data_type": "complex64",
        "tile_shape": [1200],
        "production_rate": 1200,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "equalizer",
        "consumer": "qam_demod",
        "ordering": "FIFO",
        "data_type": "complex64",
        "tile_shape": [1200],
        "production_rate": 1200,
        "double_buffering": False,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "qam_demod",
        "consumer": "viterbi",
        "ordering": "FIFO",
        "data_type": "int32",
        "tile_shape": [7200],
        "production_rate": 7200,
        "double_buffering": True,
        "visibility": "LOCAL_SPM",
    },
    {
        "producer": "viterbi",
        "consumer": "crc_check",
        "ordering": "FIFO",
        "data_type": "int32",
        "tile_shape": [1800],
        "production_rate": 1800,
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
