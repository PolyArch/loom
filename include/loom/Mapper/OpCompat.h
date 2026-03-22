#ifndef LOOM_MAPPER_OPCOMPAT_H
#define LOOM_MAPPER_OPCOMPAT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace loom {
namespace opcompat {

struct AliasPair {
  llvm::StringRef lhs;
  llvm::StringRef rhs;
};

// Returns one conservative matcher-visible alias when LOOM intentionally
// treats two operation names as equivalent for candidate generation.
//
// The table is intentionally narrow. If no explicit alias is declared,
// matching remains exact-name only.
llvm::StringRef getCompatibleOp(llvm::StringRef opName);
llvm::ArrayRef<AliasPair> getAliasPairs();

} // namespace opcompat
} // namespace loom

#endif // LOOM_MAPPER_OPCOMPAT_H
