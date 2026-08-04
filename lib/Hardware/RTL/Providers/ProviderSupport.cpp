#include "ProviderSupport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#include <cstddef>

namespace loom::hardware::rtl::detail {

llvm::APInt decodePhysicalCode(llvm::ArrayRef<std::uint8_t> bytes,
                               std::uint64_t bitCount) {
  llvm::APInt result(static_cast<unsigned>(bitCount), 0);
  for (std::uint64_t bit = 0; bit < bitCount; ++bit)
    if (((bytes[static_cast<std::size_t>(bit / 8)] >> (bit % 8)) & 1U) != 0)
      result.setBit(static_cast<unsigned>(bit));
  return result;
}

const FiniteCodebookEntry *
findFiniteCodebookEntry(const FiniteCodebookEncoding &codebook,
                        llvm::ArrayRef<std::uint8_t> semanticValue) {
  const auto found =
      llvm::find_if(codebook.entries, [&](const FiniteCodebookEntry &entry) {
        return llvm::ArrayRef<std::uint8_t>(entry.semanticValue)
            .equals(semanticValue);
      });
  return found == codebook.entries.end() ? nullptr : &*found;
}

mlir::Value resizeUnsigned(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value value, unsigned width) {
  const unsigned current =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  if (current == width)
    return value;
  if (current > width)
    return circt::comb::ExtractOp::create(builder, location, value, 0, width);
  mlir::Value highZeros = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt(width - current, 0));
  return circt::comb::ConcatOp::create(builder, location,
                                       mlir::ValueRange{highZeros, value});
}

mlir::Value addOrSubtract(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value lhs, mlir::Value rhs,
                          mlir::Value subtract) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(lhs.getType()).getWidth();
  mlir::Value subtractMask =
      width == 1 ? subtract
                 : circt::comb::ReplicateOp::create(builder, location, subtract,
                                                    width);
  mlir::Value adjustedRhs =
      circt::comb::XorOp::create(builder, location, rhs, subtractMask);
  mlir::Value carryIn = resizeUnsigned(builder, location, subtract, width);
  return circt::comb::AddOp::create(
      builder, location, mlir::ValueRange{lhs, adjustedRhs, carryIn}, true);
}

} // namespace loom::hardware::rtl::detail
