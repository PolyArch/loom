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

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricTypes.h"

#define GET_OP_CLASSES
#include "Fabric/IR/FabricOps.h.inc"

namespace mlir {
template <>
void RegisteredOperationName::Model<::fabric::BoundaryOp>::setInherentAttr(
    Operation *op, StringAttr name, Attribute value);

template <>
LogicalResult
RegisteredOperationName::Model<::fabric::BoundaryOp>::setPropertiesFromAttr(
    OperationName opName, PropertyRef properties, Attribute attr,
    function_ref<InFlightDiagnostic()> emitError);

template <>
LogicalResult
RegisteredOperationName::Model<::fabric::MemOp>::setPropertiesFromAttr(
    OperationName opName, PropertyRef properties, Attribute attr,
    function_ref<InFlightDiagnostic()> emitError);
} // namespace mlir

namespace fabric {
inline constexpr ::llvm::StringLiteral kInnerInputTypesPropertyName =
    "inner_input_types";

bool isFabricModulePortType(::mlir::Type type);
bool haveSameFabricModulePortKind(::mlir::Type source,
                                  ::mlir::Type destination);
std::optional<unsigned> getFabricBitsWidth(::mlir::Type type);
::mlir::FailureOr<unsigned> getSemanticPayloadWidth(::mlir::Type type,
                                                    std::string &error);
::mlir::LogicalResult
verifyInnerInputTypesProperty(::mlir::Operation *op, ::mlir::ValueRange inputs,
                              ::llvm::ArrayRef<::mlir::Type> innerInputTypes);

// Resolve an instantiate target with the dialect's canonical outward symbol
// table lookup semantics.
::mlir::Operation *
resolveInstantiateTarget(InstantiateOp instantiate,
                         ::mlir::SymbolTableCollection &symbolTables);

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
