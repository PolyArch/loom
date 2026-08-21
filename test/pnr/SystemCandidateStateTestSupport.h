#pragma once

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingOps.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;
struct ResolvedConfig;

namespace adg {
class FinalizedFabricDesign;
}

namespace fabric {
class FinalizedFabricRoot;
}

namespace pnr::test {

mlir::DenseI8ArrayAttr bytesAttr(mlir::MLIRContext *context,
                                 llvm::ArrayRef<std::uint8_t> bytes);

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute);

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes);

CanonicalSemanticBytes rawSystemBytes(::mapping::SystemOp root);

std::size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle);

::mapping::SystemPresburgerCellAttr
withFirstCoordinateLowerBound(::mapping::SystemPresburgerCellAttr cell,
                              std::int64_t lowerBound);

adg::FinalizedFabricDesign buildHeterogeneousSystem(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context, bool extraSupportsRead = true,
    bool routeExtraMemoryThroughTransform = false);

adg::FinalizedFabricDesign
buildSystemCandidateSpatialModule(ArtifactStore &store, bool addBoundaryBuffer);

ResolvedConfig buildSystemCandidateResolvedConfig();

adg::FinalizedFabricDesign
buildNegotiatedRoutingSystem(ArtifactStore &store,
                             const fabric::FinalizedFabricRoot &baselineSystem,
                             const fabric::FinalizedFabricRoot &primaryModule,
                             mlir::MLIRContext &context);

void verifyFinalizedSystemMappingWorkflow(
    const SystemCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FabricSystemRootView &fabric,
    const mapping::SystemMappingConstraintSetView &emptyConstraints,
    ArtifactStore &store, mlir::MLIRContext &context,
    std::size_t expectedServiceCount);

void verifySystemServiceTargetRejections(
    ::mapping::SystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FabricSystemRootView &fabric, ArtifactStore &store,
    mlir::MLIRContext &context,
    llvm::ArrayRef<fabric::SystemServiceTransformRef> foreignTransformPath,
    fabric::FabricMemoryServiceRegionRef foreignRegion);

void verifySystemResourceAction(const SystemCandidateStateHandle &candidate);

void verifySystemFixedTerminalCutAndAnnealing(
    FrozenSystemPnrProblemHandle problem,
    const SystemCandidateStateHandle &baseline);

void verifySystemResourceActionWorkflow(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ArtifactRootReference &spatialMapping, const ResolvedConfig &resolved,
    const ResolvedPnrConfigView &config, mlir::MLIRContext &context);

void verifySystemNegotiatedRoutingWorkflow(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ArtifactRootReference &spatialMapping, const ResolvedConfig &resolved,
    mlir::MLIRContext &context);

void verifySystemImportedCapacityWorkflow(
    const SystemCandidateState &candidate);

mapping::SystemAccCoreCapacityPressure verifySystemCapacityPressureRoundTrip(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &parentSystemRoot,
    const fabric::FabricSystemRootView &parentSystem,
    const ArtifactRootReference &targetModule,
    const mapping::FinalizedSystemExecutionBindingCheckpoint &checkpoint,
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    std::uint64_t assignmentAttempts);

SystemExecutionBindingCorrespondence verifySystemAccCoreCorrespondence(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &parentSystemRoot,
    const fabric::FabricSystemRootView &parentSystem,
    const fabric::FinalizedFabricRoot &childSystemRoot,
    std::vector<SystemAccCoreCorrespondence> correspondence);

} // namespace pnr::test
} // namespace loom
