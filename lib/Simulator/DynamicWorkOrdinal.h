#ifndef LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
#define LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

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

  std::optional<std::vector<std::uint64_t>> take(std::size_t count) {
    std::vector<std::uint64_t> ordinals;
    if (count == 0)
      return ordinals;
    if (!nextOrdinal_ ||
        count - 1 >
            std::numeric_limits<std::uint64_t>::max() - *nextOrdinal_)
      return std::nullopt;
    ordinals.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
      ordinals.push_back(*take());
    return ordinals;
  }

private:
  std::optional<std::uint64_t> nextOrdinal_ = 0;
};

} // namespace detail
} // namespace sim
} // namespace loom

#endif // LOOM_LIB_SIMULATOR_DYNAMICWORKORDINAL_H
