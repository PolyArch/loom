#include "Common/HwShareGroup.h"

#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace common {

static ::llvm::SmallVector<::llvm::DenseSet<::llvm::StringRef>, 32>
buildHwShareGroups() {
  ::llvm::SmallVector<::llvm::DenseSet<::llvm::StringRef>, 32> groups;
#define LOOM_HW_SHARE_GROUP(ID) groups.emplace_back();
#define LOOM_HW_SHARE_MEMBER(NAME) groups.back().insert(NAME);
#define LOOM_HW_SHARE_GROUP_END()
#include "Common/HwShareGroups.def"
#undef LOOM_HW_SHARE_GROUP
#undef LOOM_HW_SHARE_MEMBER
#undef LOOM_HW_SHARE_GROUP_END
  return groups;
}

::llvm::ArrayRef<::llvm::DenseSet<::llvm::StringRef>> hwShareGroups() {
  // Single-member groups are implicit: an unlisted op is its own singleton.
  static const auto groups = buildHwShareGroups();
  return groups;
}

std::optional<size_t> findShareGroup(::llvm::StringRef name) {
  const auto groups = hwShareGroups();
  for (size_t i = 0; i < groups.size(); ++i)
    if (groups[i].contains(name))
      return i;
  return std::nullopt;
}

bool sameShareGroup(::llvm::StringRef a, ::llvm::StringRef b) {
  auto ga = findShareGroup(a);
  auto gb = findShareGroup(b);
  if (ga && gb)
    return *ga == *gb;
  // At least one is a singleton: they share only if they are the literal
  // same op name and both are singletons.
  if (!ga && !gb)
    return a == b;
  return false;
}

} // namespace common
} // namespace loom
