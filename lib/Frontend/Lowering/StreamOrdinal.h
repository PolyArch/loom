#ifndef LOOM_FRONTEND_LOWERING_STREAM_ORDINAL_H
#define LOOM_FRONTEND_LOWERING_STREAM_ORDINAL_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

// `indexBits` is the canonical index width the caller's pass boundary already
// resolved; this recognition never resolves it again.
inline bool isZeroBasedUnitOrdinalStream(::dataflow::StreamOp stream,
                                         unsigned indexBits) {
  if (!stream || stream.getStepKind() != ::dataflow::StreamStepKind::Add)
    return false;
  auto integer =
      ::llvm::dyn_cast<::mlir::IntegerType>(stream.getIv().getType());
  if (!integer || !integer.isSignless())
    return false;
  if (integer.getWidth() != indexBits)
    return false;
  ::mlir::IntegerAttr init = getIntegerConstantAttr(stream.getInit());
  ::mlir::IntegerAttr step = getIntegerConstantAttr(stream.getStep());
  return init && init.getValue().isZero() && step && step.getValue().isOne();
}

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_STREAM_ORDINAL_H
