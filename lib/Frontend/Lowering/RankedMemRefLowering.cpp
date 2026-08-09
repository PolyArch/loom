#include "RankedMemRefLowering.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Lowering/ExactMemRefLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {
namespace lowering {
namespace detail {

::mlir::LogicalResult checkRankedMemRefAccess(::mlir::Operation *access,
                                              ::mlir::MemRefType type,
                                              ::mlir::ValueRange indices,
                                              unsigned indexBits) {
  if (indices.size() != static_cast<std::size_t>(type.getRank()))
    return access->emitError()
           << "loom-lower-graph-memory: " << access->getName().getStringRef()
           << " requires one index per memref dimension";
  auto layout = resolveExactMemRefLayout(type, indexBits);
  if (!layout)
    return access->emitError()
           << "loom-lower-graph-memory: " << access->getName().getStringRef()
           << " has no exactly addressable ranked layout: "
           << ::llvm::toString(layout.takeError());
  return ::mlir::success();
}

::mlir::LogicalResult checkRankedMemRefCopy(::mlir::memref::CopyOp copy,
                                            unsigned indexBits) {
  auto source =
      ::llvm::dyn_cast<::mlir::MemRefType>(copy.getSource().getType());
  auto target =
      ::llvm::dyn_cast<::mlir::MemRefType>(copy.getTarget().getType());
  if (!source || !target || !source.hasStaticShape() ||
      !target.hasStaticShape() || source.getShape() != target.getShape() ||
      source.getElementType() != target.getElementType())
    return copy.emitOpError(
        "loom-expand-graph-memref-copy: cannot expand memref.copy into a "
        "structured load/store loop; source and target must be ranked, "
        "statically shaped memrefs with the same shape and element type");

  for (std::int64_t extent : source.getShape()) {
    ::llvm::APInt value(64, static_cast<std::uint64_t>(extent));
    if (value.getActiveBits() >= indexBits)
      return copy.emitOpError(
                 "loom-expand-graph-memref-copy: cannot expand memref.copy "
                 "into a structured load/store loop; bound ")
             << extent << " is not representable in the graph's resolved "
             << "signed index domain 'i" << indexBits << "'";
  }
  auto sourceLayout = resolveExactMemRefLayout(source, indexBits);
  if (!sourceLayout)
    return copy.emitOpError("source layout is not exactly addressable: ")
           << ::llvm::toString(sourceLayout.takeError());
  auto targetLayout = resolveExactMemRefLayout(target, indexBits);
  if (!targetLayout)
    return copy.emitOpError("target layout is not exactly addressable: ")
           << ::llvm::toString(targetLayout.takeError());
  return ::mlir::success();
}

::mlir::Value buildExactLinearIndex(::mlir::OpBuilder &builder,
                                    ::mlir::Location loc,
                                    ::mlir::MemRefType type,
                                    ::mlir::ValueRange indices,
                                    ::mlir::Value execution) {
  ::llvm::SmallVector<std::int64_t, 4> strides;
  std::int64_t offset = 0;
  if (::mlir::failed(type.getStridesAndOffset(strides, offset)))
    llvm_unreachable("validated ranked layout has no exact strides");
  ::mlir::Value linear;
  if (offset != 0)
    linear =
        ::dataflow::ConstantOp::create(builder, loc, builder.getIndexType(),
                                       execution, builder.getIndexAttr(offset))
            .getValue();
  for (auto [index, strideValue] : ::llvm::zip(indices, strides)) {
    ::mlir::Value term = index;
    if (strideValue != 1) {
      ::mlir::Value stride = ::dataflow::ConstantOp::create(
                                 builder, loc, builder.getIndexType(),
                                 execution, builder.getIndexAttr(strideValue))
                                 .getValue();
      term = ::mlir::arith::MulIOp::create(builder, loc, index, stride);
    }
    linear = linear ? ::mlir::arith::AddIOp::create(builder, loc, linear, term)
                    : term;
  }
  if (!linear)
    linear =
        ::dataflow::ConstantOp::create(builder, loc, builder.getIndexType(),
                                       execution, builder.getIndexAttr(0))
            .getValue();
  return linear;
}

} // namespace detail
} // namespace lowering
} // namespace loom
