#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_TEMPORALPEDIAGNOSTICS_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_TEMPORALPEDIAGNOSTICS_H

#include "Support.h"

namespace loom::hardware::rtl::hierarchy {

// Borrowed signals from the TemporalPE owner; diagnostics add no state equations.
struct TemporalPeOperandDiagnostics final {
  fabric::FabricOrdinal context;
  fabric::FabricEntityId fu;
  fabric::FabricOrdinal input;
  mlir::Value active, route, target, tag;
  mlir::Value occupied, enqueueReady, enqueueGrant, enqueue, selected, ready;
  std::optional<mlir::Value> data;
};

struct TemporalPeFuInputDiagnostics final {
  fabric::FabricEntityId fu;
  fabric::FabricOrdinal input;
  mlir::Value context, enabled, valid, ready;
  std::optional<mlir::Value> data;
};

struct TemporalPeResultDiagnostics final {
  fabric::FabricEntityId fu;
  fabric::FabricOrdinal output;
  mlir::Value context, requester, offer, valid, ready, presented;
  mlir::Value active, route, discard, target, tag;
  std::optional<mlir::Value> data;
};

struct TemporalPeFifoDiagnostics final {
  mlir::Value valid, ready, tag;
  std::optional<mlir::Value> data;
};

void emitTemporalPeDiagnostics(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor, fabric::FabricEntityId pe,
    llvm::ArrayRef<EndpointPlan> endpoints,
    llvm::ArrayRef<TemporalPeOperandDiagnostics> operands,
    llvm::ArrayRef<TemporalPeFuInputDiagnostics> inputs,
    llvm::ArrayRef<TemporalPeResultDiagnostics> results,
    llvm::ArrayRef<TemporalPeFifoDiagnostics> fifos);

} // namespace loom::hardware::rtl::hierarchy

#endif
