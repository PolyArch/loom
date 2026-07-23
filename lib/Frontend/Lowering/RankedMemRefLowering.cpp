#include "RankedMemRefLowering.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"

#include <cstdint>
#include <string>

namespace {

bool fitsSignedIndex(const ::llvm::APInt &value, unsigned indexBits) {
  return value.getActiveBits() < indexBits;
}

::llvm::APInt getElementCount(::llvm::ArrayRef<std::int64_t> shape) {
  ::llvm::APInt count(1, 1);
  for (std::int64_t extent : shape) {
    if (extent == 0)
      return ::llvm::APInt(1, 0);
    ::llvm::APInt factor(64, static_cast<std::uint64_t>(extent));
    unsigned width = count.getActiveBits() + factor.getActiveBits();
    count = count.zext(width);
    count *= factor.zextOrTrunc(width);
    count = count.trunc(count.getActiveBits());
  }
  return count;
}

::llvm::APInt getMaximumLinearAddress(::mlir::MemRefType type) {
  ::llvm::APInt count = getElementCount(type.getShape());
  if (count.isZero())
    return count;
  --count;
  return count;
}

std::string formatUnsigned(const ::llvm::APInt &value) {
  ::llvm::SmallString<32> text;
  value.toString(text, 10, false);
  return text.str().str();
}

bool hasStaticIdentityLayout(::mlir::MemRefType type) {
  return type.hasStaticShape() && type.getLayout().isIdentity();
}

} // namespace

namespace loom {
namespace lowering {
namespace detail {

::mlir::LogicalResult checkRankedMemRefAccess(::mlir::Operation *access,
                                              ::mlir::MemRefType type,
                                              ::mlir::ValueRange indices,
                                              unsigned indexBits) {
  if (!type.getLayout().isIdentity() ||
      (type.getRank() > 1 && !type.hasStaticShape()))
    return access->emitError()
           << "loom-lower-graph-memory: " << access->getName().getStringRef()
           << " requires an identity-layout memref whose shape is static when "
              "rank exceeds one";
  if (indices.size() != static_cast<std::size_t>(type.getRank()))
    return access->emitError()
           << "loom-lower-graph-memory: " << access->getName().getStringRef()
           << " requires one index per memref dimension";

  if (!type.hasStaticShape())
    return ::mlir::success();

  for (std::int64_t dimension = 1; dimension < type.getRank(); ++dimension) {
    std::int64_t extent = type.getDimSize(dimension);
    ::llvm::APInt value(64, static_cast<std::uint64_t>(extent));
    if (!fitsSignedIndex(value, indexBits))
      return access->emitError()
             << "loom-lower-graph-memory: memref dimension extent " << extent
             << " is not representable in the graph's resolved signed index "
                "domain 'i"
             << indexBits << "'";
  }

  ::llvm::APInt maximum = getMaximumLinearAddress(type);
  if (!fitsSignedIndex(maximum, indexBits)) {
    auto diagnostic =
        access->emitError("loom-lower-graph-memory: maximum linear address ");
    diagnostic << formatUnsigned(maximum);
    diagnostic << " is not representable in the graph's resolved signed "
                  "index domain 'i"
               << indexBits << "'";
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult checkRankedMemRefCopy(::mlir::memref::CopyOp copy,
                                            unsigned indexBits) {
  auto source =
      ::llvm::dyn_cast<::mlir::MemRefType>(copy.getSource().getType());
  auto target =
      ::llvm::dyn_cast<::mlir::MemRefType>(copy.getTarget().getType());
  if (!source || !target || !hasStaticIdentityLayout(source) ||
      !hasStaticIdentityLayout(target) ||
      source.getShape() != target.getShape() ||
      source.getElementType() != target.getElementType())
    return copy.emitOpError(
        "loom-expand-graph-memref-copy: cannot expand memref.copy into a "
        "structured load/store loop; source and target must be ranked, "
        "statically shaped, identity-layout memrefs with the same shape and "
        "element type");

  for (std::int64_t extent : source.getShape()) {
    ::llvm::APInt value(64, static_cast<std::uint64_t>(extent));
    if (!fitsSignedIndex(value, indexBits))
      return copy.emitOpError(
                 "loom-expand-graph-memref-copy: cannot expand memref.copy "
                 "into a structured load/store loop; bound ")
             << extent << " is not representable in the graph's resolved "
             << "signed index domain 'i" << indexBits << "'";
  }

  ::llvm::APInt maximum = getMaximumLinearAddress(source);
  if (!fitsSignedIndex(maximum, indexBits)) {
    auto diagnostic = copy.emitOpError(
        "loom-expand-graph-memref-copy: cannot expand memref.copy into a "
        "structured load/store loop; maximum linear address ");
    diagnostic << formatUnsigned(maximum);
    diagnostic << " is not representable in the graph's resolved signed "
                  "index domain 'i"
               << indexBits << "'";
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Value buildRowMajorLinearIndex(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc,
                                       ::mlir::MemRefType type,
                                       ::mlir::ValueRange indices,
                                       ::mlir::Value execution) {
  if (indices.empty())
    return ::dataflow::ConstantOp::create(builder, loc, builder.getIndexType(),
                                          execution, builder.getIndexAttr(0))
        .getValue();

  ::mlir::Value linear = indices.front();
  for (unsigned dimension = 1; dimension < indices.size(); ++dimension) {
    ::mlir::Value extent = ::dataflow::ConstantOp::create(
                               builder, loc, builder.getIndexType(), execution,
                               builder.getIndexAttr(type.getDimSize(dimension)))
                               .getValue();
    linear = ::mlir::arith::MulIOp::create(builder, loc, linear, extent);
    linear =
        ::mlir::arith::AddIOp::create(builder, loc, linear, indices[dimension]);
  }
  return linear;
}

} // namespace detail
} // namespace lowering
} // namespace loom
