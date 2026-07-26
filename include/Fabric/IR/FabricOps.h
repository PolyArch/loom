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
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

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
std::optional<unsigned> getFabricTransportPayloadWidth(::mlir::Type type);
::llvm::Expected<std::vector<std::uint8_t>>
encodeFabricTransportType(::mlir::Type type);
::llvm::Expected<std::vector<std::uint8_t>>
encodeFabricTransportFunctionType(::mlir::FunctionType type);
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
} // namespace fabric

#endif // FABRIC_IR_FABRICOPS_H
