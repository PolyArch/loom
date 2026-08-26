#include "Frontend/Lowering/ExactMemRefLayout.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>

namespace loom::lowering {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "exact_memref_layout_invalid: " + message);
}

bool fitsSignedIndex(std::uint64_t value, unsigned indexBits) {
  llvm::APInt encoded(64, value);
  return encoded.getActiveBits() < indexBits;
}

llvm::APInt multiply(std::uint64_t lhs, std::uint64_t rhs) {
  llvm::APInt left(64, lhs);
  llvm::APInt right(64, rhs);
  const unsigned width =
      std::max(1U, left.getActiveBits() + right.getActiveBits());
  left = left.zextOrTrunc(width);
  left *= right.zextOrTrunc(width);
  return left;
}

llvm::APInt add(llvm::APInt lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getActiveBits(), rhs.getActiveBits()) + 1;
  lhs = lhs.zextOrTrunc(width);
  lhs += rhs.zextOrTrunc(width);
  return lhs;
}

bool fitsSignedIndex(const llvm::APInt &value, unsigned indexBits) {
  return value.getActiveBits() < indexBits;
}

std::string formatUnsigned(const llvm::APInt &value) {
  llvm::SmallString<32> text;
  value.toString(text, 10, false);
  return text.str().str();
}

} // namespace

bool isProvablyInjectiveMemRefLayout(mlir::MemRefType type) {
  if (type.getLayout().isIdentity())
    return true;

  llvm::SmallVector<std::int64_t, 4> strides;
  std::int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != static_cast<std::size_t>(type.getRank()))
    return false;

  struct ActiveDimension final {
    std::uint64_t stride = 0;
    std::int64_t extent = 0;
  };
  llvm::SmallVector<ActiveDimension, 4> active;
  for (auto [extent, stride] : llvm::zip(type.getShape(), strides)) {
    if (extent == 0 || extent == 1)
      continue;
    if (stride == mlir::ShapedType::kDynamic || stride <= 0)
      return false;
    active.push_back({static_cast<std::uint64_t>(stride), extent});
  }
  llvm::sort(active,
             [](const ActiveDimension &lhs, const ActiveDimension &rhs) {
               return lhs.stride < rhs.stride;
             });

  llvm::APInt coveredSpan(1, 0);
  for (auto [ordinal, dimension] : llvm::enumerate(active)) {
    llvm::APInt stride(64, dimension.stride);
    const unsigned comparisonWidth =
        std::max(stride.getBitWidth(), coveredSpan.getBitWidth());
    if (stride.zextOrTrunc(comparisonWidth)
            .ule(coveredSpan.zextOrTrunc(comparisonWidth)))
      return false;
    if (dimension.extent == mlir::ShapedType::kDynamic)
      return ordinal + 1 == active.size();
    if (dimension.extent < 0)
      return false;
    coveredSpan = add(std::move(coveredSpan),
                      multiply(static_cast<std::uint64_t>(dimension.extent - 1),
                               dimension.stride));
  }
  return true;
}

llvm::Expected<ExactMemRefLayout>
resolveExactMemRefLayout(mlir::MemRefType type, unsigned indexBits) {
  if (indexBits == 0)
    return invalid("signed index width is zero");
  if (!type.hasStaticShape() &&
      (type.getRank() > 1 || !type.getLayout().isIdentity()))
    return invalid("dynamic shape is not exact for this ranked layout");

  ExactMemRefLayout result;
  if (mlir::failed(type.getStridesAndOffset(result.strides, result.offset)))
    return invalid("layout has no strided interpretation");
  if (result.offset == mlir::ShapedType::kDynamic || result.offset < 0)
    return invalid("layout offset is not a static nonnegative value");
  if (!fitsSignedIndex(static_cast<std::uint64_t>(result.offset), indexBits))
    return invalid("layout offset exceeds the signed index domain");
  for (std::int64_t stride : result.strides) {
    if (stride == mlir::ShapedType::kDynamic || stride <= 0)
      return invalid("layout stride is not a static positive value");
    if (!fitsSignedIndex(static_cast<std::uint64_t>(stride), indexBits))
      return invalid("layout stride exceeds the signed index domain");
  }

  if (!type.hasStaticShape())
    return result;
  if (llvm::is_contained(type.getShape(), 0)) {
    result.staticElementSpan = 0;
    return result;
  }
  llvm::APInt maximum(64, static_cast<std::uint64_t>(result.offset));
  for (auto [extent, stride] : llvm::zip(type.getShape(), result.strides)) {
    if (extent < 0)
      return invalid("static shape has a negative extent");
    maximum = add(maximum, multiply(static_cast<std::uint64_t>(extent - 1),
                                    static_cast<std::uint64_t>(stride)));
  }
  if (!fitsSignedIndex(maximum, indexBits))
    return invalid("maximum linear address " + formatUnsigned(maximum) +
                   " exceeds the signed index domain");
  maximum = add(maximum, llvm::APInt(1, 1));
  if (maximum.getActiveBits() > 64)
    return invalid("static element span exceeds u64");
  result.staticElementSpan = maximum.getZExtValue();
  return result;
}

llvm::Expected<llvm::SmallVector<unsigned, 4>>
resolveDenseMemRefStorageOrder(mlir::MemRefType type, unsigned indexBits) {
  if (type.getRank() < 2 || !type.hasStaticShape())
    return invalid("dense permutation requires static rank of at least two");
  if (llvm::any_of(type.getShape(),
                   [](std::int64_t extent) { return extent <= 0; }))
    return invalid("dense permutation requires a positive static shape");
  if (type.getMemorySpace())
    return invalid("dense permutation has a non-default memory space");
  auto layout = resolveExactMemRefLayout(type, indexBits);
  if (!layout)
    return layout.takeError();

  llvm::SmallVector<unsigned, 4> order(type.getRank());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    if (layout->strides[lhs] != layout->strides[rhs])
      return layout->strides[lhs] > layout->strides[rhs];
    return lhs < rhs;
  });

  llvm::APInt expected(64, 1);
  for (unsigned dimension : llvm::reverse(order)) {
    if (expected.getActiveBits() > 63 ||
        layout->strides[dimension] !=
            static_cast<std::int64_t>(expected.getZExtValue()))
      return invalid("layout is not a dense permutation");
    expected = multiply(expected.getZExtValue(),
                        static_cast<std::uint64_t>(type.getDimSize(dimension)));
  }
  return order;
}

} // namespace loom::lowering
