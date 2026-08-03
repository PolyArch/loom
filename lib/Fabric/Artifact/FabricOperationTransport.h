#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICOPERATIONTRANSPORT_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICOPERATIONTRANSPORT_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace loom::fabric::detail {

struct FabricOperationTransportTypes {
  llvm::SmallVector<::mlir::Type, 8> inputs;
  llvm::SmallVector<::mlir::Type, 8> outputs;
};

/// Resolves the one canonical token-plane signature used by Fabric artifact
/// materialization. Memory-service memrefs are deliberately excluded.
llvm::Expected<FabricOperationTransportTypes>
resolveFabricOperationTransportTypes(::mlir::Operation *operation);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICOPERATIONTRANSPORT_H
