#include "Fabric/IR/FabricDialect.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace fabric;

#include "Fabric/IR/FabricDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Fabric/IR/FabricTypes.cpp.inc"

LogicalResult BitsTagType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    unsigned /*width*/, unsigned tagWidth) {
  if (tagWidth == 0)
    return emitError() << "fabric.bits_tag requires tagWidth > 0";
  return success();
}

void FabricDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Fabric/IR/FabricTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Fabric/IR/FabricOps.cpp.inc"
      >();
}
