#ifndef LOOM_FRONTEND_LOWERING_STREAM_ORDINAL_H
#define LOOM_FRONTEND_LOWERING_STREAM_ORDINAL_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace loom {
namespace lowering {

inline ::mlir::IntegerAttr getIntegerConstantAttr(::mlir::Value value) {
  ::mlir::Attribute attr;
  if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>())
    attr = constant.getValue();
  else if (auto constant = value.getDefiningOp<::dataflow::ConstantOp>())
    attr = constant.getConstValue();
  return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(attr);
}

inline bool isZeroBasedUnitOrdinalStream(::dataflow::StreamOp stream,
                                         ::mlir::Operation *scope) {
  if (!stream || stream.getStepKind() != ::dataflow::StreamStepKind::Add)
    return false;
  auto integer =
      ::llvm::dyn_cast<::mlir::IntegerType>(stream.getIv().getType());
  if (!integer || !integer.isSignless())
    return false;
  ::mlir::DataLayout dataLayout = ::mlir::DataLayout::closest(scope);
  ::llvm::TypeSize indexBits = dataLayout.getTypeSizeInBits(
      ::mlir::IndexType::get(integer.getContext()));
  if (indexBits.isScalable() || integer.getWidth() != indexBits.getFixedValue())
    return false;
  ::mlir::IntegerAttr init = getIntegerConstantAttr(stream.getInit());
  ::mlir::IntegerAttr step = getIntegerConstantAttr(stream.getStep());
  return init && init.getValue().isZero() && step && step.getValue().isOne();
}

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_STREAM_ORDINAL_H
