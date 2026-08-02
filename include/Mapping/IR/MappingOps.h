#ifndef LOOM_MAPPING_IR_MAPPINGOPS_H
#define LOOM_MAPPING_IR_MAPPINGOPS_H

#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingEnums.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Mapping/IR/MappingOps.h.inc"

#endif // LOOM_MAPPING_IR_MAPPINGOPS_H
