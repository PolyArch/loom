#pragma once

#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "PnR/PnrConfig.h"

#include <cstdint>

namespace mlir {
class MLIRContext;
}

namespace loom {

class ArtifactStore;

namespace test {

mlir::MLIRContext makeContext();

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context);

dataflow::CanonicalDataflowArtifact
buildAlternateDataflow(mlir::MLIRContext &context);

dataflow::CanonicalDataflowArtifact
buildVectorDataflow(mlir::MLIRContext &context);

fabric::FinalizedFabricRoot
buildAlternativeTechSpatialCore(ArtifactStore &store);

ResolvedConfig buildSpatialResolvedConfig();

pnr::ResolvedPnrConfigView buildSpatialConfig();

ResolvedConfig buildSingleCandidateSpatialResolvedConfig();

pnr::ResolvedPnrConfigView buildSingleCandidateSpatialConfig();

pnr::ResolvedPnrConfigView buildFeedbackSpatialConfig();

dataflow::CanonicalDataflowArtifact
buildRootCompleteSpatialDataflow(mlir::MLIRContext &context);

dataflow::CanonicalDataflowArtifact
buildAlternateRootCompleteSpatialDataflow(mlir::MLIRContext &context);

dataflow::CanonicalDataflowArtifact
buildVectorRootCompleteSpatialDataflow(mlir::MLIRContext &context);

fabric::FinalizedFabricRoot buildSpatialCore(ArtifactStore &store,
                                             std::uint32_t payloadWidth = 128);

fabric::FinalizedFabricRoot
buildAlternativeTechSpatialCore(ArtifactStore &store);

fabric::FinalizedFabricRoot
buildLineageSpatialCore(ArtifactStore &store, std::uint32_t payloadWidth = 128);

fabric::FinalizedFabricRoot
buildFeedbackPruningSpatialCore(ArtifactStore &store);

ResolvedConfig buildSpatialResolvedConfig();

pnr::ResolvedPnrConfigView buildSpatialConfig();

ResolvedConfig buildSingleCandidateSpatialResolvedConfig();

pnr::ResolvedPnrConfigView buildSingleCandidateSpatialConfig();

pnr::ResolvedPnrConfigView buildFeedbackSpatialConfig();

} // namespace test
} // namespace loom
