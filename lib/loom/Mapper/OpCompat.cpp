#include "loom/Mapper/OpCompat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <utility>

namespace loom {
namespace opcompat {

llvm::ArrayRef<AliasPair> getAliasPairs() {
  // Keep this table intentionally small and explicit. LOOM does not infer broad
  // semantic equivalence between ops such as `arith.select` and
  // `handshake.mux`, or between `arith.constant` and `handshake.constant`.
  static constexpr std::array<AliasPair, 0> kAliases{};
  return llvm::ArrayRef(kAliases);
}

llvm::StringRef getCompatibleOp(llvm::StringRef opName) {
  // When a new alias is added here, it becomes part of both Layer-2 tech-map
  // matching and mapper-side FU compatibility checks.
  for (const auto &entry : getAliasPairs()) {
    if (entry.lhs == opName)
      return entry.rhs;
    if (entry.rhs == opName)
      return entry.lhs;
  }
  return "";
}

} // namespace opcompat
} // namespace loom
