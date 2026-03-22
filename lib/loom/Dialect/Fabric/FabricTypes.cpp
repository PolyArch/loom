// Type verify methods are defined here.
// GET_TYPEDEF_CLASSES is included in FabricDialect.cpp to avoid duplicate symbols.

#include "loom/Dialect/Fabric/FabricTypes.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <optional>

using namespace mlir;
using namespace loom::fabric;

namespace {

std::optional<unsigned> parseIndexWidthOverrideFromEnv() {
  const char *env = std::getenv("LOOM_INDEX_WIDTH");
  if (!env || *env == '\0')
    return std::nullopt;

  char *end = nullptr;
  errno = 0;
  unsigned long value = std::strtoul(env, &end, 10);
  if (errno != 0 || end == env || *end != '\0' ||
      value > std::numeric_limits<unsigned>::max()) {
    return std::nullopt;
  }

  unsigned width = static_cast<unsigned>(value);
  if (!isSupportedIndexBitWidth(width))
    return std::nullopt;
  return width;
}

} // namespace

bool loom::fabric::isSupportedIndexBitWidth(unsigned width) {
  return width >= MIN_INDEX_BIT_WIDTH && width <= MAX_INDEX_BIT_WIDTH;
}

unsigned loom::fabric::getConfiguredIndexBitWidth() {
  static const unsigned configuredWidth = []() -> unsigned {
    if (auto overrideWidth = parseIndexWidthOverrideFromEnv())
      return *overrideWidth;
    return getDefaultIndexBitWidth();
  }();
  return configuredWidth;
}

mlir::IntegerType loom::fabric::getIndexIntegerType(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, getConfiguredIndexBitWidth());
}

LogicalResult BitsType::verify(function_ref<InFlightDiagnostic()> emitError,
                               unsigned width) {
  if (width < 1 || width > 4096)
    return emitError() << "bits width must be in [1, 4096], got " << width;
  return success();
}

LogicalResult
TaggedType::verify(function_ref<InFlightDiagnostic()> emitError, Type valueType,
                   IntegerType tagType) {
  unsigned tagWidth = tagType.getWidth();
  if (tagWidth < 1 || tagWidth > 16)
    return emitError() << "tag width must be in [1, 16], got " << tagWidth;
  return success();
}
