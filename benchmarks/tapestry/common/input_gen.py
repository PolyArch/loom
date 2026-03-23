#!/usr/bin/env python3
"""Input data generators for all Tapestry benchmark kernels."""

import struct
import math
import os


def write_float_bin(filename, data):
    """Write a list of floats to a binary file."""
    with open(filename, "wb") as f:
        for val in data:
            f.write(struct.pack("f", val))


def write_int_bin(filename, data):
    """Write a list of ints to a binary file."""
    with open(filename, "wb") as f:
        for val in data:
            f.write(struct.pack("i", val))


def gen_random_float(n, seed=42, scale=1.0):
    """Generate deterministic pseudo-random floats."""
    result = []
    state = seed
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        val = ((state % 1000) - 500) / 500.0 * scale
        result.append(val)
    return result


# --- AI/LLM generators ---

def gen_qkv_proj(output_dir, seq_len=128, d_model=512):
    """Generate inputs for QKV projection."""
    n_out = 3 * d_model
    A = gen_random_float(seq_len * d_model, seed=1)
    B = gen_random_float(d_model * n_out, seed=2)
    write_float_bin(os.path.join(output_dir, "qkv_proj_A.bin"), A)
    write_float_bin(os.path.join(output_dir, "qkv_proj_B.bin"), B)
    print(f"  qkv_proj: A[{seq_len}x{d_model}], B[{d_model}x{n_out}]")


def gen_attn_score(output_dir, num_heads=8, seq_len=128, head_dim=64):
    """Generate inputs for attention score."""
    size = num_heads * seq_len * head_dim
    Q = gen_random_float(size, seed=10)
    K = gen_random_float(size, seed=11)
    write_float_bin(os.path.join(output_dir, "attn_score_Q.bin"), Q)
    write_float_bin(os.path.join(output_dir, "attn_score_K.bin"), K)
    print(f"  attn_score: Q/K[{num_heads}x{seq_len}x{head_dim}]")


def gen_softmax(output_dir, num_heads=8, seq_len=128):
    """Generate inputs for softmax."""
    size = num_heads * seq_len * seq_len
    data = gen_random_float(size, seed=20, scale=5.0)
    write_float_bin(os.path.join(output_dir, "softmax_data.bin"), data)
    print(f"  softmax: data[{num_heads}x{seq_len}x{seq_len}]")


def gen_ffn(output_dir, seq_len=128, d_model=512, d_ff=2048):
    """Generate inputs for FFN layers."""
    inp = gen_random_float(seq_len * d_model, seed=30)
    w1 = gen_random_float(d_model * d_ff, seed=31, scale=0.1)
    b1 = gen_random_float(d_ff, seed=32)
    w2 = gen_random_float(d_ff * d_model, seed=33, scale=0.1)
    b2 = gen_random_float(d_model, seed=34)
    write_float_bin(os.path.join(output_dir, "ffn_input.bin"), inp)
    write_float_bin(os.path.join(output_dir, "ffn_w1.bin"), w1)
    write_float_bin(os.path.join(output_dir, "ffn_b1.bin"), b1)
    write_float_bin(os.path.join(output_dir, "ffn_w2.bin"), w2)
    write_float_bin(os.path.join(output_dir, "ffn_b2.bin"), b2)
    print(f"  ffn: input[{seq_len}x{d_model}], W1[{d_model}x{d_ff}]")


def gen_layernorm(output_dir, seq_len=128, d_model=512):
    """Generate inputs for layer normalization."""
    inp = gen_random_float(seq_len * d_model, seed=40)
    gamma = [0.8 + 0.4 * (i % 17) / 17.0 for i in range(d_model)]
    beta = [(i % 23 - 11.0) / 110.0 for i in range(d_model)]
    write_float_bin(os.path.join(output_dir, "layernorm_input.bin"), inp)
    write_float_bin(os.path.join(output_dir, "layernorm_gamma.bin"), gamma)
    write_float_bin(os.path.join(output_dir, "layernorm_beta.bin"), beta)
    print(f"  layernorm: input[{seq_len}x{d_model}]")


# --- DSP/OFDM generators ---

def gen_fft(output_dir, n=4096):
    """Generate inputs for FFT."""
    data_re = []
    data_im = []
    for i in range(n):
        t = i / n
        val = math.sin(2 * math.pi * 3 * t) + 0.5 * math.cos(2 * math.pi * 7 * t)
        data_re.append(val)
        data_im.append(0.0)
    write_float_bin(os.path.join(output_dir, "fft_re.bin"), data_re)
    write_float_bin(os.path.join(output_dir, "fft_im.bin"), data_im)
    print(f"  fft: {n} complex points")


def gen_channel_est(output_dir, num_pilots=200):
    """Generate inputs for channel estimation."""
    rx_re, rx_im = [], []
    tx_re, tx_im = [], []
    for i in range(num_pilots):
        phase = i * 0.1
        atten = 0.5 + 0.5 * math.cos(phase)
        tx_re.append(1.0)
        tx_im.append(0.0)
        rx_re.append(atten * math.cos(phase))
        rx_im.append(atten * math.sin(phase))
    write_float_bin(os.path.join(output_dir, "chanest_rx_re.bin"), rx_re)
    write_float_bin(os.path.join(output_dir, "chanest_rx_im.bin"), rx_im)
    write_float_bin(os.path.join(output_dir, "chanest_tx_re.bin"), tx_re)
    write_float_bin(os.path.join(output_dir, "chanest_tx_im.bin"), tx_im)
    print(f"  channel_est: {num_pilots} pilots")


def gen_viterbi(output_dir, num_bits=1800):
    """Generate inputs for Viterbi decoder."""
    bits = [(i * 37 + 13) % 2 for i in range(num_bits)]
    write_int_bin(os.path.join(output_dir, "viterbi_input.bin"), bits)
    print(f"  viterbi: {num_bits} data bits")


# --- AR/VR generators ---

def gen_stereo_images(output_dir, w=160, h=120):
    """Generate stereo image pair."""
    left = []
    right = []
    for y in range(h):
        for x in range(w):
            val = float((x * 3 + y * 7 + 42) % 256)
            left.append(val)
            sx = min(x + 10, w - 1)
            right.append(float((sx * 3 + y * 7 + 42) % 256))
    write_float_bin(os.path.join(output_dir, "stereo_left.bin"), left)
    write_float_bin(os.path.join(output_dir, "stereo_right.bin"), right)
    print(f"  stereo: {w}x{h} image pair")


def generate_all(output_dir="generated_inputs"):
    """Generate all benchmark input data."""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating benchmark inputs:")

    gen_qkv_proj(output_dir)
    gen_attn_score(output_dir)
    gen_softmax(output_dir)
    gen_ffn(output_dir)
    gen_layernorm(output_dir)
    gen_fft(output_dir)
    gen_channel_est(output_dir)
    gen_viterbi(output_dir)
    gen_stereo_images(output_dir)

    print("Done.")


if __name__ == "__main__":
    generate_all()
