#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICOPERATIONTRANSPORT_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICOPERATIONTRANSPORT_H

#include "Fabric/Identity/FabricRefs.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::fabric::detail {

struct FabricOperationTransportTypes {
  llvm::SmallVector<::mlir::Type, 8> inputs;
  llvm::SmallVector<::mlir::Type, 8> outputs;
};

/// Resolves the one canonical token-plane signature used by Fabric artifact
/// materialization. Memory-service memrefs are deliberately excluded.
llvm::Expected<FabricOperationTransportTypes>
resolveFabricOperationTransportTypes(::mlir::Operation *operation);

/// Projects an entity kind that owns occurrence-local token endpoints.
std::optional<FabricTransportEndpointOwnerRef>
projectFabricTransportOwner(FabricEntityKind kind, FabricEntityId id);

/// Maps an operation signature ordinal into its token-only transport
/// inventory. Memory-plane positions have no token ordinal.
llvm::Expected<std::optional<FabricOrdinal>>
resolveFabricTokenInputOrdinal(::mlir::Operation *operation,
                               std::uint64_t signatureOrdinal);
llvm::Expected<std::optional<FabricOrdinal>>
resolveFabricTokenOutputOrdinal(::mlir::Operation *operation,
                                std::uint64_t signatureOrdinal);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICOPERATIONTRANSPORT_H
