#include "Common/VectorWidth.h"

#include <cstdint>
#include <limits>
#include <system_error>

llvm::Expected<unsigned>
loom::getFixedVectorBitWidth(mlir::VectorType vector,
                             unsigned elementBitWidth) {
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

  std::uint64_t width = elementBitWidth;
  for (std::int64_t extent : vector.getShape()) {
    const auto lanes = static_cast<std::uint64_t>(extent);
    if (lanes > std::numeric_limits<unsigned>::max() / width)
      return llvm::createStringError(std::errc::value_too_large,
                                     "vector bit width exceeds unsigned range");
    width *= lanes;
  }
  return static_cast<unsigned>(width);
}
