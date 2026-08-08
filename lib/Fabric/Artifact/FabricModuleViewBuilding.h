#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEVIEWBUILDING_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEVIEWBUILDING_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir {
class Operation;
class Type;
} // namespace mlir

namespace fabric {
class MemOp;
}

namespace loom::fabric::detail {

struct FabricArtifactViewData;
struct FabricCanonicalLabeling;
struct FabricEntityViewData;
struct FabricNestedOwnerViewData;

std::vector<std::uint64_t> emptyFabricInventories();
FabricFuNodeKind classifyFabricFuNode(mlir::Operation *operation);
void setFabricPortInventories(FabricNestedOwnerViewData &owner,
                              std::uint64_t inputs, std::uint64_t outputs);
llvm::Error setFabricTransportEndpoints(FabricNestedOwnerViewData &owner,
                                        llvm::ArrayRef<mlir::Type> inputs,
                                        llvm::ArrayRef<mlir::Type> outputs);
llvm::Error
setFabricOperationTransportEndpoints(mlir::Operation *operation,
                                     FabricNestedOwnerViewData &owner);
llvm::Error populateFabricMemoryView(::fabric::MemOp memory,
                                     FabricEntityViewData &entity);
llvm::Error
appendFabricModuleMemoryConnections(const FabricCanonicalLabeling &labeling,
                                    FabricArtifactViewData &data);
llvm::Error appendFabricPeSelectorTraversals(FabricArtifactViewData &data);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEVIEWBUILDING_H
