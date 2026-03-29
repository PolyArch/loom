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
  //
  // Each alias allows the tech-mapper to match a DFG node with opName==lhs
  // against an FU body that uses opName==rhs (or vice-versa).
  //
  //   arith.maxsi  <=>  arith.cmpi + arith.select (compare-select FU body)
  //   arith.minsi  <=>  arith.cmpi + arith.select (compare-select FU body)
  //   math.absf    <=>  arith.negf + arith.select (negate-select FU body)
  //
  // The RHS strings must match the TemplateOp::opName that the variant builder
  // produces for the corresponding FU body pattern.
  static constexpr std::array<AliasPair, 3> kAliases{{
      {{"arith.maxsi"}, {"arith.select"}},
      {{"arith.minsi"}, {"arith.select"}},
      {{"math.absf"}, {"arith.select"}},
  }};
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
