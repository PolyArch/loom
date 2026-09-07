#pragma once

#include "Common/Artifact.h"
#include "DSE/CandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;
class BlobStore;

namespace test {

ArtifactRootReference generateTechMapping(
    const ArtifactRootReference &dataflow, const ArtifactRootReference &fabric,
    ArtifactStore &store, const BlobStore &blobs);

std::vector<ArtifactRootReference> generateTechMappingSet(
    const ArtifactRootReference &dataflow, const ArtifactRootReference &fabric,
    ArtifactStore &store, const BlobStore &blobs);

struct RootCompleteSpatialPnrFixture final {
  dataflow::CanonicalDataflowArtifact dataflow;
  loom::ArtifactRootReference dataflowReference;
  loom::fabric::FinalizedFabricRoot fabric;
  loom::ArtifactRootReference physicalTimingProfile;
  loom::ArtifactRootReference techMappingReference;
};

RootCompleteSpatialPnrFixture buildRootCompleteSpatialPnrFixture(
    mlir::MLIRContext &context, ArtifactStore &store, const BlobStore &blobs);

void requireSpatialWorkSummary(
    llvm::ArrayRef<dse::CandidateGeneratorWorkUnitSummary> summary,
    bool expectConsumedWork);

ArtifactRootReference normalizedTimingProfile(
    const ArtifactRootReference &fabricReference, ArtifactStore &store);

ArtifactRootReference generateSpatialMapping(
    const ArtifactRootReference &techMapping,
    const ArtifactRootReference &fabric, ArtifactStore &store,
    const BlobStore &blobs);

std::vector<ArtifactRootReference> generateSpatialMappingSet(
    llvm::ArrayRef<ArtifactRootReference> techMappings,
    const ArtifactRootReference &fabric, ArtifactStore &store,
    const BlobStore &blobs);

struct PublishedSpatialInputs final {
  loom::ArtifactRootReference workload;
  loom::ArtifactRootReference runtimeInput;
};

struct GeneratedSpatialFeedbackFixture final {
  loom::ArtifactRootReference mapping;
  loom::mapping::FinalizedSpatialMappingConstraintSet constraints;
};

GeneratedSpatialFeedbackFixture generateSpatialFeedbackFixture(
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &techMappingReference,
    const fabric::FinalizedFabricRoot &fabric, ArtifactStore &store);

PublishedSpatialInputs publishSpatialInputs(
    const dataflow::CanonicalDataflowArtifact &dataflow, ArtifactStore &store);

PublishedSpatialInputs publishVectorSpatialInputs(
    const dataflow::CanonicalDataflowArtifact &dataflow, ArtifactStore &store,
    unsigned laneWidth = 32);

struct PublishedStructuredSimulationInputs final {
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
  ArtifactRootReference workloadReference;
  ArtifactRootReference runtimeInputReference;
};

frontend::StructuredProgramCandidate
buildWideVectorStructuredSource(mlir::MLIRContext &context);

frontend::StructuredEntityRef
findStructuredCallable(const frontend::StructuredProgramCandidate &candidate,
                       llvm::StringRef name);

PublishedStructuredSimulationInputs publishWideVectorStructuredInputs(
    const frontend::StructuredProgramCandidate &source, ArtifactStore &store);

} // namespace test
} // namespace loom
