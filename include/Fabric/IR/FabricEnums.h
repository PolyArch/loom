#ifndef FABRIC_IR_FABRICENUMS_H
#define FABRIC_IR_FABRICENUMS_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#include <cstdint>
#include <optional>

#include "Fabric/IR/FabricEnums.h.inc"

namespace fabric {

/// One memory actor firing outstanding at a time: the serialized Operation
/// Engine. It is the smallest admitted `operation_issue_depth` of
/// `Fabric_MemoryEngineAttr`, the depth its canonical text elides, and the
/// depth every authoring recipe selects unless it declares a deeper engine.
inline constexpr std::uint64_t serializedMemoryOperationIssueDepth = 1;

} // namespace fabric

#endif // FABRIC_IR_FABRICENUMS_H
