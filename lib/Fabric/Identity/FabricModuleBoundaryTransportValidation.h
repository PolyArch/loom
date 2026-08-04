#ifndef LOOM_LIB_FABRIC_IDENTITY_FABRICMODULEBOUNDARYTRANSPORTVALIDATION_H
#define LOOM_LIB_FABRIC_IDENTITY_FABRICMODULEBOUNDARYTRANSPORTVALIDATION_H

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::fabric::detail {

struct FabricArtifactViewData;

/// Canonicalizes and validates the sealed Module-boundary transport rows
/// before constructing the immutable view.
llvm::Error canonicalizeFabricModuleBoundaryTransportRelations(
    FabricArtifactViewData &data);

/// Validates every relation against the resolved endpoint types and
/// occurrence-local endpoint inventories in an immutable view.
llvm::Error
validateFabricModuleBoundaryTransportRelations(const FabricArtifactView &view);

/// Encoded Fabric transport types share a kind exactly when their canonical
/// kind tags agree. Payload and tag widths remain connection-local.
bool haveSameFabricTransportKind(llvm::ArrayRef<std::uint8_t> left,
                                 llvm::ArrayRef<std::uint8_t> right);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_IDENTITY_FABRICMODULEBOUNDARYTRANSPORTVALIDATION_H
