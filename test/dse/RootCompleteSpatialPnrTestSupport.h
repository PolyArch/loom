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

fabric::FinalizedFabricRoot buildSpatialCore(ArtifactStore &store,
                                             std::uint32_t payloadWidth = 128);

fabric::FinalizedFabricRoot
buildLineageSpatialCore(ArtifactStore &store, std::uint32_t payloadWidth = 128);

fabric::FinalizedFabricRoot
buildFeedbackPruningSpatialCore(ArtifactStore &store);

} // namespace test
} // namespace loom
