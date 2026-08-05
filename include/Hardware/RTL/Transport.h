#ifndef LOOM_HARDWARE_RTL_TRANSPORT_H
#define LOOM_HARDWARE_RTL_TRANSPORT_H

#include "Fabric/IR/BoundaryDataPath.h"
#include "Fabric/Identity/FabricRefs.h"

#include "circt/Dialect/HW/PortImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace mlir {
class OpBuilder;
}

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::hardware::rtl {

/// The forward half of a Fabric ready/valid connection. Ready travels in the
/// opposite direction and is connected unchanged by the structural owner.
struct ForwardTransportSignals final {
  mlir::Value valid;
  std::optional<mlir::Value> payload;
  std::optional<mlir::Value> tag;
};

/// The transient CIRCT port projection of one token-plane Module boundary.
/// The Fabric reference remains the authority for endpoint identity and type;
/// port names are emission details only.
struct ModuleBoundaryTransportPortProjection final {
  loom::fabric::FabricModuleBoundaryEndpointRef boundary;
  std::optional<circt::hw::PortInfo> data;
  std::optional<circt::hw::PortInfo> tag;
  circt::hw::PortInfo valid;
  circt::hw::PortInfo ready;
};

/// Projects every token-plane endpoint of one finalized Module root in input
/// signature order followed by output signature order. Memory-plane endpoints
/// are not RTL token ports. The complete boundary is validated before any
/// result is produced.
llvm::Expected<std::vector<ModuleBoundaryTransportPortProjection>>
deriveModuleBoundaryTransportPorts(
    mlir::OpBuilder &builder, const loom::fabric::FabricArtifactView &artifact);

llvm::Expected<ForwardTransportSignals>
adaptForwardTransportSignals(mlir::OpBuilder &builder, mlir::Location location,
                             ::fabric::DataPathType sourceType,
                             ::fabric::DataPathType destinationType,
                             ForwardTransportSignals sourceSignals);

/// Adapts the forward signals of one exact finalized Fabric point connection.
/// Endpoint types come only from the Fabric owner. Ready remains the reverse
/// signal of that connection and is materialized unchanged by the surrounding
/// structural lowering.
llvm::Expected<ForwardTransportSignals>
adaptFabricPointConnectionForwardSignals(
    mlir::OpBuilder &builder, mlir::Location location,
    const loom::fabric::FabricArtifactView &artifact,
    const loom::fabric::FabricPointConnectionPayload &connection,
    ForwardTransportSignals sourceSignals);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_TRANSPORT_H
