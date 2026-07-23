#include "DynamicWorkOrdinal.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace loom {
namespace sim {
namespace detail {

std::optional<std::uint64_t> takeChildOrdinal(std::uint64_t &nextOrdinal) {
  return takeChildOrdinalBefore<std::numeric_limits<std::uint64_t>::max()>(
      nextOrdinal);
}

} // namespace detail
} // namespace sim
} // namespace loom
