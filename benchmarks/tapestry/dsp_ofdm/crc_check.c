/*
 * CRC-16 Check -- cyclic redundancy check for OFDM receiver.
 * CRC-16-CCITT polynomial: x^16 + x^12 + x^5 + 1 (0x1021).
 * Processes 3600 bits and computes/verifies the CRC.
 * Tiled: process TILE_BITS bits at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define NUM_DATA_BITS  3600
#define CRC_BITS       16
#define TOTAL_BITS     (NUM_DATA_BITS + CRC_BITS)
#define TILE_BITS      256
#define CRC_POLY       0x1021
#define CRC_INIT       0xFFFF

/* Compute CRC-16 over an array of bits (0/1 values) */
unsigned int crc16_compute(const int *bits, int num_bits) {
    unsigned int crc = CRC_INIT;

    TILE_FOR(tb, 0, num_bits, TILE_BITS) {
        int tb_end = TILE_END(tb, num_bits, TILE_BITS);
        int i;
        for (i = tb; i < tb_end; i++) {
            int bit = bits[i] & 1;
            int msb = (crc >> 15) & 1;
            crc = (crc << 1) & 0xFFFF;
            if (msb ^ bit) {
                crc ^= CRC_POLY;
            }
        }
    }

    return crc;
}

unsigned int crc16_compute_ref(const int *bits, int num_bits) {
    unsigned int crc = CRC_INIT;
    int i;
    for (i = 0; i < num_bits; i++) {
        int bit = bits[i] & 1;
        int msb = (crc >> 15) & 1;
        crc = (crc << 1) & 0xFFFF;
        if (msb ^ bit) {
            crc ^= CRC_POLY;
        }
    }
    return crc;
}

/* Append CRC bits to data: compute CRC over data, then store it */
void crc16_append(int *bits, int num_data_bits) {
    unsigned int crc = crc16_compute_ref(bits, num_data_bits);

    /* Store CRC bits (MSB first) */
    int i;
    for (i = 0; i < CRC_BITS; i++) {
        bits[num_data_bits + i] = (crc >> (CRC_BITS - 1 - i)) & 1;
    }
}

/* Extract stored CRC from the bit stream */
static unsigned int crc16_extract(const int *bits, int num_data_bits) {
    unsigned int crc = 0;
    int i;
    for (i = 0; i < CRC_BITS; i++) {
        crc = (crc << 1) | (bits[num_data_bits + i] & 1);
    }
    return crc;
}

/* Check CRC: recompute over data and compare with stored CRC */
int crc16_check(const int *bits, int num_data_bits) {
    unsigned int computed = crc16_compute(bits, num_data_bits);
    unsigned int stored = crc16_extract(bits, num_data_bits);
    return (computed == stored) ? 1 : 0;
}

int main(void) {
    int *bits = (int *)malloc((size_t)TOTAL_BITS * sizeof(int));
    if (!bits) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test data */
    for (int i = 0; i < NUM_DATA_BITS; i++) {
        bits[i] = ((i * 53 + 17) % 100) > 49 ? 1 : 0;
    }

    /* Append CRC */
    crc16_append(bits, NUM_DATA_BITS);

    /* Verify CRC passes */
    unsigned int crc_tiled = crc16_compute(bits, NUM_DATA_BITS);
    unsigned int crc_ref   = crc16_compute_ref(bits, NUM_DATA_BITS);

    printf("crc_check: crc_tiled=0x%04X, crc_ref=0x%04X\n", crc_tiled, crc_ref);

    int pass1 = (crc_tiled == crc_ref);
    int pass2 = crc16_check(bits, NUM_DATA_BITS);

    /* Test with corrupted bit */
    bits[42] ^= 1;
    int pass3 = !crc16_check(bits, NUM_DATA_BITS); /* Should fail */

    int pass = pass1 && pass2 && pass3;
    printf("crc_check: match=%d, valid=%d, detect_error=%d\n", pass1, pass2, pass3);
    printf("crc_check: %s\n", pass ? "PASS" : "FAIL");

    free(bits);
    return pass ? 0 : 1;
}
