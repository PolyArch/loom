
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kPolynomial = 0xedb88320u;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = i * 0x12345678u;
    }
}

uint32_t crc32_ref(const uint32_t *input_data, uint32_t size) {
    uint32_t crc = 0xffffffffu;

    for (uint32_t i = 0; i < size; ++i) {
        const uint32_t data = input_data[i];
        for (uint32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint32_t byte = (data >> (byte_idx * 8u)) & 0xffu;
            crc ^= byte;

            for (uint32_t bit = 0; bit < 8; ++bit) {
                if ((crc & 1u) != 0u) {
                    crc = (crc >> 1) ^ kPolynomial;
                } else {
                    crc >>= 1;
                }
            }
        }
    }

    return ~crc;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    initialize_input(input);

    const uint32_t reference = crc32_ref(input.data(), kSize);
    uint32_t crc = 0xffffffffu;

    for (uint32_t i = 0; i < kSize; ++i) {
        const uint32_t data = input[i];
        for (uint32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
            const uint32_t byte = (data >> (byte_idx * 8u)) & 0xffu;
            crc ^= byte;

            for (uint32_t bit = 0; bit < 8; ++bit) {
                if ((crc & 1u) != 0u) {
                    crc = (crc >> 1) ^ kPolynomial;
                } else {
                    crc >>= 1;
                }
            }
        }
    }

    const uint32_t candidate = ~crc;
    if (reference != candidate) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("crc32 checksum: %u\n", candidate);
    std::puts("PASSED");
    return 0;
}
