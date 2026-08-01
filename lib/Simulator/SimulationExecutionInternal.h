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
};

llvm::Expected<SpatialExecutionContext> resolveSpatialExecutionContext(
    const ArtifactRootReference &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store);

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
