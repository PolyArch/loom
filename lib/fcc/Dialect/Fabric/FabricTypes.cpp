// Type verify methods are defined here.
// GET_TYPEDEF_CLASSES is included in FabricDialect.cpp to avoid duplicate symbols.

#include "fcc/Dialect/Fabric/FabricTypes.h"

using namespace mlir;
using namespace fcc::fabric;

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
