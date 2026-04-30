#ifndef FABRIC_IR_FABRICOPS_H
#define FABRIC_IR_FABRICOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Fabric/IR/FabricTypes.h"

#include "Fabric/IR/FabricEnums.h.inc"

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.h.inc"

namespace fabric {
// Returns true if the software op named by `name` is one of the operations
// supported as a member of fabric.op's `op_list`. This is the canonical
// allowlist of "ops a fabric tile can implement" and is also consumed by the
// dataflow.subgraph body verifier.
bool isFabricOpSupported(::llvm::StringRef name);
} // namespace fabric

#endif // FABRIC_IR_FABRICOPS_H
