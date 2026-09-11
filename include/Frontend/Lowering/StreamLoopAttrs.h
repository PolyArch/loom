#ifndef LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H
#define LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H

#include "Dataflow/IR/DataflowEnums.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace loom {
namespace lowering {

inline ::llvm::StringRef streamStepKindAttrName() {
  return "loom.stream_step_kind";
}

inline ::llvm::StringRef streamPredicateAttrName() {
  return "loom.stream_predicate";
}

inline ::mlir::FailureOr<::dataflow::StreamStepKind>
inferStreamStepKind(::mlir::scf::ForOp loop) {
  auto attr = loop->getAttr(streamStepKindAttrName());
  if (!attr)
    return ::dataflow::StreamStepKind::Add;
  auto kind = ::llvm::dyn_cast<::dataflow::StreamStepKindAttr>(attr);
  if (!kind)
    return ::mlir::failure();
  return kind.getValue();
}

/// Without an explicit configuration the loop's own comparison domain decides
/// the stream predicate: an unsigned loop compares unsigned, and a signed loop
/// with a negative constant step counts down.
inline ::mlir::arith::CmpIPredicate
inferStreamPredicateFromDomain(::mlir::scf::ForOp loop) {
  if (loop.getUnsignedCmp())
    return ::mlir::arith::CmpIPredicate::ult;
  if (auto constOp =
          loop.getStep().getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (auto intAttr =
            ::llvm::dyn_cast<::mlir::IntegerAttr>(constOp.getValue())) {
      if (intAttr.getValue().isNegative())
        return ::mlir::arith::CmpIPredicate::sgt;
    }
  }
  return ::mlir::arith::CmpIPredicate::slt;
}

inline ::mlir::FailureOr<::mlir::arith::CmpIPredicate>
inferStreamPredicate(::mlir::scf::ForOp loop) {
  auto attr = loop->getAttr(streamPredicateAttrName());
  if (!attr)
    return inferStreamPredicateFromDomain(loop);
  auto predicate = ::llvm::dyn_cast<::mlir::arith::CmpIPredicateAttr>(attr);
  if (!predicate)
    return ::mlir::failure();
  return predicate.getValue();
}

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H
