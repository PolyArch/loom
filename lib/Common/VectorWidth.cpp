#include "Common/VectorWidth.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <cstdint>
#include <optional>
#include <system_error>

llvm::Expected<std::uint64_t>
loom::getFixedVectorBitWidth(mlir::VectorType vector,
                             std::uint64_t elementBitWidth) {
  if (vector.isScalable())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "scalable vector has no fixed bit width");
  if (vector.getRank() < 1)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "rank-zero vector has no fixed bit width");
  if (elementBitWidth == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "zero-width vector element has no fixed bit width");

  // Each dimension is folded under its own check, so a product that leaves the
  // exact range is reported at the axis that leaves it rather than wrapping
  // into a smaller legal width.
  std::uint64_t width = elementBitWidth;
  for (std::int64_t extent : vector.getShape()) {
    std::optional<std::uint64_t> product =
        llvm::checkedMulUnsigned<std::uint64_t>(
            width, static_cast<std::uint64_t>(extent));
    if (!product)
      return llvm::createStringError(std::errc::value_too_large,
                                     "vector bit width exceeds 64-bit range");
    width = *product;
  }
  return width;
}
