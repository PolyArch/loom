#ifndef LOOM_FRONTEND_IR_LOOMDIALECT_H
#define LOOM_FRONTEND_IR_LOOMDIALECT_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

#include "Frontend/IR/LoomDialect.h.inc"

namespace loom {

/// Structured-candidate marker for an LLVM memory access whose selected
/// projection is RootRelative. It participates in Structured identity and is
/// consumed before Canonical Dataflow publication.
inline constexpr llvm::StringLiteral rootRelativeAddressAttrName =
    "loom.root_relative_address";

} // namespace loom

#endif // LOOM_FRONTEND_IR_LOOMDIALECT_H
