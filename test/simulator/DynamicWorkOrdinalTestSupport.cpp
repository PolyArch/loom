#include "DynamicWorkOrdinal.h"

#include <cstdint>
#include <optional>

namespace loom {
namespace sim {
namespace detail {

// The anchor target substitutes only this private ordinal transition. Domain
// state and responsibility authority remain reachable solely through the
// production public API.
std::optional<std::uint64_t> takeChildOrdinal(std::uint64_t &nextOrdinal) {
  return takeChildOrdinalBefore<3>(nextOrdinal);
}

} // namespace detail
} // namespace sim
} // namespace loom
