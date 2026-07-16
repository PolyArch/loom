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
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

#include "Fabric/IR/FabricTypes.h"

#include "Fabric/IR/FabricEnums.h.inc"

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.h.inc"

namespace fabric {
inline constexpr ::llvm::StringLiteral kInnerInputTypesPropertyName =
    "inner_input_types";

enum class FabricOpModeKind { Legacy, Normalized, Malformed };

struct FabricOpModeClassification {
  FabricOpModeKind kind = FabricOpModeKind::Legacy;
  std::string diagnostic;
};

bool isFabricModulePortType(::mlir::Type type);
bool haveSameFabricModulePortKind(::mlir::Type source,
                                  ::mlir::Type destination);
std::optional<unsigned> getFabricBitsWidth(::mlir::Type type);
FabricOpModeClassification classifyFabricOpModes(OpOp op);
::mlir::FailureOr<unsigned> getSemanticPayloadWidth(::mlir::Type type,
                                                    std::string &error);
::mlir::LogicalResult
verifyInnerInputTypesProperty(::mlir::Operation *op, ::mlir::ValueRange inputs,
                              ::llvm::ArrayRef<::mlir::Type> innerInputTypes);

// Returns true if the software op named by `name` is one of the operations
// supported as a member of fabric.op's `op_list`. This is the canonical
// allowlist of "ops a fabric tile can implement" and is also consumed by the
// dataflow.subgraph body verifier.
bool isFabricOpSupported(::llvm::StringRef name);

// Resolve the Loom address bit width for `op`. Walks up to the enclosing
// fabric.module; if that module sets a `loom_addr_bits` override returns
// it, otherwise returns ::loom::getDefaultLoomAddrBits().
unsigned resolveLoomAddrBits(::mlir::Operation *op);

// Resolve the Loom memory bus width (in bits) for `op`. Walks up to the
// enclosing fabric.module; if that module sets a `loom_mem_bus_width`
// override returns it, otherwise returns
// ::loom::getDefaultLoomMemBusWidth().
unsigned resolveLoomMemBusWidth(::mlir::Operation *op);
} // namespace fabric

#endif // FABRIC_IR_FABRICOPS_H
