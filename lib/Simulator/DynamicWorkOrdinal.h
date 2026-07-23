#ifndef LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
#define LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H

#include <cstdint>
#include <limits>
#include <optional>

namespace loom {
namespace sim {
namespace detail {

class ChildOrdinalCursor {
public:
  ChildOrdinalCursor() = default;
  explicit ChildOrdinalCursor(std::uint64_t nextOrdinal)
      : nextOrdinal_(nextOrdinal) {}

  std::optional<std::uint64_t> take() {
    if (!nextOrdinal_)
      return std::nullopt;

    std::uint64_t ordinal = *nextOrdinal_;
    if (ordinal == std::numeric_limits<std::uint64_t>::max())
      nextOrdinal_.reset();
    else
      ++*nextOrdinal_;
    return ordinal;
  }

private:
  std::optional<std::uint64_t> nextOrdinal_ = 0;
};

} // namespace detail
} // namespace sim
} // namespace loom

#endif // LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
