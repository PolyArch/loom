#include "Fabric/IR/ArithPredicates.h"

namespace loom {

// All cmpi predicates loom currently understands. Order corresponds to the
// I64EnumAttrCase ordering in mlir/Dialect/Arith/IR/ArithBase.td. The
// static_assert below cross-checks against upstream's auto-generated
// getMaxEnumValForCmpIPredicate() so any divergence after rebasing the
// externals/llvm submodule fires at compile time.
static constexpr CmpIPredicate kCmpIPredicates[] = {
    CmpIPredicate::eq,  CmpIPredicate::ne,  CmpIPredicate::slt,
    CmpIPredicate::sle, CmpIPredicate::sgt, CmpIPredicate::sge,
    CmpIPredicate::ult, CmpIPredicate::ule, CmpIPredicate::ugt,
    CmpIPredicate::uge,
};

static_assert(
    (sizeof(kCmpIPredicates) / sizeof(kCmpIPredicates[0])) ==
        ::mlir::arith::getMaxEnumValForCmpIPredicate() + 1u,
    "loom's CmpIPredicate enumeration is out of sync with upstream MLIR; "
    "audit kCmpIPredicates after rebasing externals/llvm.");

static constexpr CmpFPredicate kCmpFPredicates[] = {
    CmpFPredicate::AlwaysFalse,
    CmpFPredicate::OEQ, CmpFPredicate::OGT, CmpFPredicate::OGE,
    CmpFPredicate::OLT, CmpFPredicate::OLE, CmpFPredicate::ONE,
    CmpFPredicate::ORD,
    CmpFPredicate::UEQ, CmpFPredicate::UGT, CmpFPredicate::UGE,
    CmpFPredicate::ULT, CmpFPredicate::ULE, CmpFPredicate::UNE,
    CmpFPredicate::UNO,
    CmpFPredicate::AlwaysTrue,
};

static_assert(
    (sizeof(kCmpFPredicates) / sizeof(kCmpFPredicates[0])) ==
        ::mlir::arith::getMaxEnumValForCmpFPredicate() + 1u,
    "loom's CmpFPredicate enumeration is out of sync with upstream MLIR; "
    "audit kCmpFPredicates after rebasing externals/llvm.");

::llvm::ArrayRef<CmpIPredicate> getKnownCmpIPredicates() {
  return kCmpIPredicates;
}

::llvm::ArrayRef<CmpFPredicate> getKnownCmpFPredicates() {
  return kCmpFPredicates;
}

} // namespace loom
