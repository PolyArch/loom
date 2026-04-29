#ifndef LOOM_COMMON_HWSHAREGROUP_H
#define LOOM_COMMON_HWSHAREGROUP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <optional>

namespace loom {
namespace common {

// Returns the canonical table of multi-member hardware-share groups.
//
// Each entry is a set of MLIR op names (e.g. "arith.addi") whose operations
// are allowed to share a single hardware functional unit. An op name that
// does not appear in any group is treated as its own implicit singleton
// group: it cannot share hardware with any other op name.
//
// The returned reference is to a static immutable table; the data outlives
// any caller and is safe to cache by index.
::llvm::ArrayRef<::llvm::DenseSet<::llvm::StringRef>> hwShareGroups();

// Returns the index into `hwShareGroups()` for the multi-member group that
// contains `name`, or `std::nullopt` if `name` is not in any multi-member
// group (i.e. it is an implicit singleton).
std::optional<size_t> findShareGroup(::llvm::StringRef name);

// Returns true iff `a` and `b` may share the same hardware functional unit.
//
// Two op names share hardware when:
//   * Both belong to the same multi-member group in `hwShareGroups()`, or
//   * They are the same singleton (`a == b` and neither is in any
//     multi-member group), i.e. an op trivially shares with itself.
//
// All other combinations return false; in particular two distinct singletons
// never share, and a singleton never shares with a multi-member group entry.
bool sameShareGroup(::llvm::StringRef a, ::llvm::StringRef b);

} // namespace common
} // namespace loom

#endif // LOOM_COMMON_HWSHAREGROUP_H
