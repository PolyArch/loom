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
// `include/Common/HwShareGroups.def` is the single source of truth for the
// returned registry. Entries are software operation family keys whose modes
// may share one hardware functional unit. Most keys are registered MLIR op
// names. Intrinsic semantic keys require an explicit registered-operation
// representation before they can appear in a normalized Fabric mode.
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
