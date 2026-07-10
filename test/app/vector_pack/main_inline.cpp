// vector_pack: integer reduction with source-level unroll hints.
// Inline variant: kernel loop written directly in main.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr int kSize = 8;
  std::array<std::uint16_t, kSize> input{};
  for (int i = 0; i < kSize; ++i) {
    input[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(i + 1);
  }

  std::uint16_t acc = 0;
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
  for (int i = 0; i < kSize; ++i) {
    acc = static_cast<std::uint16_t>(acc + input[static_cast<std::size_t>(i)]);
  }

  std::printf("%u\n", static_cast<unsigned>(acc));
  return 0;
}
