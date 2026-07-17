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

inline ::mlir::arith::CmpIPredicate
inferStreamPredicateFromStep(::mlir::Value step) {
  if (auto constOp = step.getDefiningOp<::mlir::arith::ConstantOp>()) {
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
    return inferStreamPredicateFromStep(loop.getStep());
  auto predicate = ::llvm::dyn_cast<::mlir::arith::CmpIPredicateAttr>(attr);
  if (!predicate)
    return ::mlir::failure();
  return predicate.getValue();
}

inline void setStreamLoopConfiguration(::mlir::OpBuilder &builder,
                                       ::mlir::scf::ForOp loop,
                                       ::dataflow::StreamStepKind stepKind,
                                       ::mlir::arith::CmpIPredicate predicate) {
  loop->setAttr(streamStepKindAttrName(), ::dataflow::StreamStepKindAttr::get(
                                              builder.getContext(), stepKind));
  loop->setAttr(
      streamPredicateAttrName(),
      ::mlir::arith::CmpIPredicateAttr::get(builder.getContext(), predicate));
}

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H
