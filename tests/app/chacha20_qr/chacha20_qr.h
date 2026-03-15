// Loom kernel: chacha20_qr
#ifndef CHACHA20_QR_H
#define CHACHA20_QR_H

#include <cstdint>
#include <cstddef>

void chacha20_qr_cpu(const uint32_t* __restrict__ input_state, uint32_t* __restrict__ output_state, const uint32_t num_rounds);

void chacha20_qr_dsa(const uint32_t* __restrict__ input_state, uint32_t* __restrict__ output_state, const uint32_t num_rounds);

#endif // CHACHA20_QR_H
