#ifndef LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
#define LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H

#include <cstdint>
#include <limits>
#include <optional>

namespace loom {
namespace sim {
namespace detail {

inline std::optional<std::uint64_t>
takeChildOrdinal(std::uint64_t &nextOrdinal) {
  if (nextOrdinal == std::numeric_limits<std::uint64_t>::max())
    return std::nullopt;
  return nextOrdinal++;
}

} // namespace detail
} // namespace sim
} // namespace loom

#endif // LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
