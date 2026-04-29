#include "Common/HwShareGroup.h"

#include "llvm/ADT/SmallVector.h"

namespace loom {
namespace common {

::llvm::ArrayRef<::llvm::DenseSet<::llvm::StringRef>> hwShareGroups() {
  // Multi-member hardware-share groups. Single-member groups are implicit:
  // any op not in any group below is its own singleton (cannot share with
  // any other op).
  static const ::llvm::SmallVector<::llvm::DenseSet<::llvm::StringRef>, 32>
      groups = {
          {"arith.addi", "arith.subi"},
          {"arith.divsi", "arith.remsi"},
          {"arith.divui", "arith.remui"},
          {"arith.shli", "arith.shrsi", "arith.shrui"},
          {"arith.andi", "arith.ori", "arith.xori"},
          {"arith.minsi", "arith.maxsi"},
          {"arith.minui", "arith.maxui"},
          {"arith.sitofp", "arith.uitofp"},
          {"arith.fptosi", "arith.fptoui"},
          {"arith.addf", "arith.subf"},
          {"arith.divf", "arith.remf"},
          {"arith.minimumf", "arith.maximumf"},
          {"math.sin", "math.cos"},
          {"math.sinh", "math.cosh"},
          {"math.exp", "math.exp2", "math.expm1"},
          {"math.log", "math.log2", "math.log10", "math.log1p"},
          {"math.floor", "math.ceil", "math.round", "math.trunc",
           "math.roundeven"},
          {"math.sqrt", "math.rsqrt"},
          {"math.tanh", "math.erf"},
      };
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
