#ifndef LOOM_DATAFLOW_IR_GEPADDRESSPLAN_H
#define LOOM_DATAFLOW_IR_GEPADDRESSPLAN_H

#include "Common/PointerLayout.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace dataflow::semantics {

/// One typed GEP path component. A dynamic component names its actor operand;
/// otherwise constantIndex carries the exact source integer. scale is already
/// projected to A(AS) bits from the exact LLVM DataLayout.
struct GepOffsetTerm {
  std::optional<unsigned> dynamicOperandOrdinal;
  llvm::APInt constantIndex = llvm::APInt(1, 0);
  llvm::APInt scale = llvm::APInt(1, 0);
};

/// Immutable address projection of one scalar LLVM GEP. Type walking and
/// DataLayout queries happen once during graph preparation, never per firing.
struct GepAddressPlan {
  ::loom::PointerLayout pointerLayout;
  mlir::LLVM::GEPNoWrapFlags noWrapFlags = mlir::LLVM::GEPNoWrapFlags::none;
  llvm::SmallVector<GepOffsetTerm, 4> terms;
};

/// Derive byte-offset terms from the operation and its exact enclosing layout.
llvm::Expected<GepAddressPlan> projectGepAddressPlan(mlir::LLVM::GEPOp op,
                                                    mlir::Operation *scope);

} // namespace dataflow::semantics

#endif
