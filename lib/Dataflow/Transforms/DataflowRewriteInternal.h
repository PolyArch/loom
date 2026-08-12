#ifndef LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_INTERNAL_H
#define LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_INTERNAL_H

#include "Dataflow/Transforms/DataflowRewrite.h"

#include "mlir/IR/IRMapping.h"

namespace dataflow::detail {

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateSyncRendezvousDecisions(const CanonicalDataflowArtifact &parent);

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeSyncRendezvousRewrite(const CanonicalDataflowArtifact &parent,
                                 const SyncRendezvousRewrite &decision);

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeSyncRendezvousRewriteProjection(
    const CanonicalDataflowArtifact &parent,
    const SyncRendezvousRewrite &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches);

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateCardinalityCommuteDecisions(const CanonicalDataflowArtifact &parent);

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeCardinalityCommuteRewrite(
    const CanonicalDataflowArtifact &parent,
    const ElementwiseCardinalityCommuteRewrite &decision);

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeCardinalityCommuteRewriteProjection(
    const CanonicalDataflowArtifact &parent,
    const ElementwiseCardinalityCommuteRewrite &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches);

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumeratePureComputeFanoutDecisions(const CanonicalDataflowArtifact &parent);

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializePureComputeFanoutRewrite(const CanonicalDataflowArtifact &parent,
                                    const DataflowRewriteDecision &decision);

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializePureComputeFanoutRewriteProjection(
    const CanonicalDataflowArtifact &parent,
    const DataflowRewriteDecision &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches);

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateGraphDefinitionRefactorDecisions(
    const CanonicalDataflowArtifact &parent);

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeGraphDefinitionRefactor(const CanonicalDataflowArtifact &parent,
                                   const DataflowRewriteDecision &decision);

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeGraphDefinitionRefactorProjection(
    const CanonicalDataflowArtifact &parent,
    const DataflowRewriteDecision &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches);

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeFixedDataflowRewrite(const CanonicalDataflowArtifact &parent,
                                const DataflowRewriteDecision &decision);

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeFixedDataflowRewriteProjection(
    const CanonicalDataflowArtifact &parent,
    const DataflowRewriteDecision &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches);

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
finalizeDataflowRewriteCandidate(
    const CanonicalDataflowArtifact &parent, mlir::ModuleOp candidate,
    const mlir::IRMapping &mapping,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches);

} // namespace dataflow::detail

#endif // LOOM_DATAFLOW_TRANSFORMS_DATAFLOW_REWRITE_INTERNAL_H
