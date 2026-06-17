#ifndef LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H
#define LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace loom {
namespace lowering {

inline ::llvm::StringRef streamContCondAttrName() {
  return "loom.stream_cont_cond";
}

inline ::mlir::StringAttr
inferStreamContCondFromStep(::mlir::OpBuilder &builder, ::mlir::Value step) {
  if (auto constOp = step.getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (auto intAttr =
            ::llvm::dyn_cast<::mlir::IntegerAttr>(constOp.getValue())) {
      if (intAttr.getValue().isNegative())
        return builder.getStringAttr(">");
    }
  }
  return builder.getStringAttr("<");
}

inline ::mlir::StringAttr inferStreamContCond(::mlir::OpBuilder &builder,
                                              ::mlir::scf::ForOp loop) {
  if (auto attr =
          loop->getAttrOfType<::mlir::StringAttr>(streamContCondAttrName()))
    return attr;
  return inferStreamContCondFromStep(builder, loop.getStep());
}

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_STREAM_LOOP_ATTRS_H
