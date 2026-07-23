#ifndef LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
#define LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H

#include <cstdint>
#include <optional>

namespace loom {
namespace sim {
namespace detail {

template <std::uint64_t ExhaustedOrdinal>
inline std::optional<std::uint64_t>
takeChildOrdinalBefore(std::uint64_t &nextOrdinal) {
  if (nextOrdinal >= ExhaustedOrdinal)
    return std::nullopt;
  return nextOrdinal++;
}

std::optional<std::uint64_t> takeChildOrdinal(std::uint64_t &nextOrdinal);

} // namespace detail
} // namespace sim
} // namespace loom

#endif // LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
