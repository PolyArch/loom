#ifndef LOOM_LIB_SIMULATOR_SIMULATIONEXECUTIONINTERNAL_H
#define LOOM_LIB_SIMULATOR_SIMULATIONEXECUTIONINTERNAL_H

#include "SimulationWireInternal.h"

#include "Simulator/SimulationExecution.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Request.h"

namespace loom::sim::detail {

struct SpatialExecutionContext {
  evaluation::EvaluationRequest request;
  ImportedSpatialSimulationInputs inputs;
  dataflow::CanonicalDataflowProgramView dataflowView;
  ResolvedLaunchContext launch;
  evaluation::ArtifactCollectionCardinality stoppedExecutionCardinality;
  const evaluation::CaseArtifactResolution *resolution = nullptr;
  const ArtifactStore *artifactStore = nullptr;
  const BlobStore *blobStore = nullptr;
};

struct SystemExecutionContext {
  evaluation::EvaluationRequest request;
  ImportedSystemSimulationInputs inputs;
  ResolvedSystemContext system;
  evaluation::ArtifactCollectionCardinality stoppedExecutionCardinality;
  const evaluation::CaseArtifactResolution *resolution = nullptr;
  const ArtifactStore *artifactStore = nullptr;
  const BlobStore *blobStore = nullptr;
};

llvm::Expected<evaluation::ArtifactCollectionCardinality>
resolveSimulationOutputCardinality(
    const evaluation::EvaluationRequest &request);

llvm::Error validateExecutionTerminal(
    const ExecutionTerminal &terminal,
    const evaluation::FindingTerminalWitnessContext &context);
llvm::Error encodeExecutionTerminal(
    WireWriter &writer, const ExecutionTerminal &terminal,
    const evaluation::FindingTerminalWitnessContext &context);
llvm::Expected<ExecutionTerminal> decodeExecutionTerminal(
    WireReader &reader,
    const evaluation::FindingTerminalWitnessContext &context);

llvm::Expected<SpatialExecutionContext> resolveSpatialExecutionContext(
    const ArtifactRootReference &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<SpatialSimulationExecution> decodeSpatialSimulationExecution(
    llvm::ArrayRef<std::uint8_t> bytes,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<SystemExecutionContext> resolveSystemExecutionContext(
    const ArtifactRootReference &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Error validateSpatialFunctionalObservations(
    const SpatialFunctionalObservations &observations,
    const ExecutionTerminal &terminal, const SpatialExecutionContext &context);

void encodeSpatialFunctionalObservations(
    WireWriter &writer, const SpatialFunctionalObservations &observations,
    const SpatialExecutionContext &context);

llvm::Expected<SpatialFunctionalObservations>
decodeSpatialFunctionalObservations(WireReader &reader,
                                    const SpatialExecutionContext &context);

llvm::Error validateActorActivitySummaries(
    llvm::ArrayRef<ActorTransitionsActivitySummary> summaries,
    const ExecutionTerminal &terminal,
    const SpatialProgressObservations &progress,
    const SpatialExecutionContext &context);

void encodeActorActivitySummaries(
    WireWriter &writer,
    llvm::ArrayRef<ActorTransitionsActivitySummary> summaries);

llvm::Expected<std::vector<ActorTransitionsActivitySummary>>
decodeActorActivitySummaries(WireReader &reader);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_SIMULATIONEXECUTIONINTERNAL_H
