#include <stdint.h>

void hash_mix(const uint32_t *restrict input_state,
              const uint32_t *restrict input_data,
              uint32_t *restrict output_state, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t state = input_state[i];
    uint32_t data = input_data[i];

    state = state + data;
    state = (state << 7) | (state >> 25);
    state = state ^ data;
    state = state * 0x5BD1E995u;
    state = (state << 13) | (state >> 19);

    output_state[i] = state;
  }
}

int main(void) {
  uint32_t input_state[4] = {0u, 123456789u, 987654321u, 42u};
  uint32_t input_data[4] = {1234567u, 7654321u, 1111111u, 99u};
  uint32_t output_state[4] = {0u, 0u, 0u, 0u};
  hash_mix(input_state, input_data, output_state, 4u);
  return (output_state[0] == 3280104067u &&
          output_state[1] == 3325253434u &&
          output_state[2] == 684477807u)
             ? 0
             : 1;
}
