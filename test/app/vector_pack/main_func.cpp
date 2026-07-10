// vector_pack: integer reduction with source-level unroll hints.
// Function variant: kernel implemented as a separate function.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kSize = 8;

__attribute__((noinline))
std::uint16_t vector_pack_kernel(const std::uint16_t *input, int size) {
  std::uint16_t acc = 0;
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
  for (int i = 0; i < size; ++i) {
    acc = static_cast<std::uint16_t>(acc + input[i]);
  }
  return acc;
}

} // namespace

int main() {
  std::array<std::uint16_t, kSize> input{};
  for (int i = 0; i < kSize; ++i) {
    input[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(i + 1);
  }

  std::uint16_t result = vector_pack_kernel(input.data(), kSize);
  std::printf("%u\n", static_cast<unsigned>(result));
  return 0;
}
