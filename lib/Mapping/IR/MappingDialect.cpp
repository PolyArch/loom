#include "Mapping/IR/MappingDialect.h"

#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "Mapping/IR/MappingDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Mapping/IR/MappingAttrs.cpp.inc"

void mapping::MappingDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Mapping/IR/MappingAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Mapping/IR/MappingOps.cpp.inc"
      >();
}
