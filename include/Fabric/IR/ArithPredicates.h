#ifndef FABRIC_IR_ARITHPREDICATES_H
#define FABRIC_IR_ARITHPREDICATES_H

// Loom takes upstream MLIR's `arith::CmpIPredicate` / `arith::CmpFPredicate`
// as the canonical enum representation of the cmpi / cmpf predicate sets.
// The parallel `kCmpIPredicates` / `kCmpFPredicates` arrays in the .cpp are
// statically asserted to match upstream's cardinality so an MLIR upgrade
// that adds or removes a predicate fires a compile-time alarm rather than
// silently dropping cases.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace loom {

using CmpIPredicate = ::mlir::arith::CmpIPredicate;
using CmpFPredicate = ::mlir::arith::CmpFPredicate;

::llvm::ArrayRef<CmpIPredicate> getKnownCmpIPredicates();
::llvm::ArrayRef<CmpFPredicate> getKnownCmpFPredicates();

inline std::optional<CmpIPredicate>
symbolizeCmpIPredicate(::llvm::StringRef symbol) {
  return ::mlir::arith::symbolizeCmpIPredicate(symbol);
}

inline std::optional<CmpFPredicate>
symbolizeCmpFPredicate(::llvm::StringRef symbol) {
  return ::mlir::arith::symbolizeCmpFPredicate(symbol);
}

inline ::llvm::StringRef stringifyCmpIPredicate(CmpIPredicate p) {
  return ::mlir::arith::stringifyCmpIPredicate(p);
}

inline ::llvm::StringRef stringifyCmpFPredicate(CmpFPredicate p) {
  return ::mlir::arith::stringifyCmpFPredicate(p);
}

} // namespace loom

#endif // FABRIC_IR_ARITHPREDICATES_H
