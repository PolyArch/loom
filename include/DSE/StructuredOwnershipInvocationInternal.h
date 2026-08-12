#ifndef LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H
#define LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H

#include "DSE/StructuredOwnershipInvocation.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Simulator/NativeSimulationOracle.h"

namespace loom::dse::detail {

/// Invocation-local, removable index for exact D0-rooted rewrite lineage.
/// Canonical Artifact references and typed decisions remain the authorities;
/// this cache only avoids rescanning or rewalking unrelated derivations.
class StructuredOwnershipDataflowLineageIndex final {
public:
  StructuredOwnershipDataflowLineageIndex();
  ~StructuredOwnershipDataflowLineageIndex();

  StructuredOwnershipDataflowLineageIndex(
      const StructuredOwnershipDataflowLineageIndex &) = delete;
  StructuredOwnershipDataflowLineageIndex &
  operator=(const StructuredOwnershipDataflowLineageIndex &) = delete;

  bool empty() const;
  llvm::Error recordRoot(const ArtifactRootReference &structuredParent,
                         const ArtifactRootReference &dataflowRoot);
  llvm::Expected<ArtifactRootReference>
  root(const ArtifactRootReference &structuredParent) const;
  llvm::Error recordDecision(
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const dataflow::DataflowRewriteDecision &decision,
      llvm::ArrayRef<dataflow::StaticGraphLaunchRef> parentLaunches = {},
      llvm::ArrayRef<dataflow::StaticGraphLaunchRef> childLaunches = {});
  llvm::Expected<std::optional<std::vector<DataflowRewriteDerivation>>>
  tryResolve(const ArtifactRootReference &structuredParent,
             const ArtifactRootReference &candidate) const;
  llvm::Expected<dataflow::StaticGraphLaunchRef>
  projectStaticGraphLaunch(const ArtifactRootReference &structuredParent,
                           const ArtifactRootReference &candidate,
                           dataflow::StaticGraphLaunchRef rootLaunch) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct StructuredOwnershipPreparedSource final {
  const ArtifactRootReference &generationParentReference;
  const ArtifactRootReference &workloadReference;
  const ArtifactRootReference &runtimeInputReference;
  const sim::NativeStructuredProgramObservations &generationParentObservations;
};

struct StructuredOwnershipCandidateState final {
  ArtifactRootReference reference;
  frontend::MaterializedStructuredOwnershipCandidate candidate;
};

class StructuredOwnershipInvocationAccess final {
public:
  static StructuredOwnershipInvocation *current();
  static StructuredOwnershipInvocation *
  bind(StructuredOwnershipInvocation *invocation);

  static llvm::Error prepareGeneration(
      StructuredOwnershipInvocation &invocation,
      const frontend::StructuredProgramCandidate &generationParent,
      const frontend::StructuredProgramCandidate &sourceProgram,
      const sim::CanonicalSimulationWorkload &workload,
      const sim::CanonicalSimulationRuntimeInput &runtimeInput,
      const fabric::FinalizedFabricRoot &fabric, const ArtifactStore &store,
      StructuredOwnershipGenerationOptions &options);

  static llvm::Expected<StructuredOwnershipPreparedSource>
  preparedSource(const StructuredOwnershipInvocation &invocation);

  static const ResolvedConfig &
  config(const StructuredOwnershipInvocation &invocation);
  static const lowering::CanonicalDataflowLoweringOptions &
  loweringOptions(const StructuredOwnershipInvocation &invocation);

  static const fabric::FinalizedFabricRoot &
  fabric(const StructuredOwnershipInvocation &invocation);
  static evaluation::models::StructuredEvaluationInvocationCache &
  evaluationCache(StructuredOwnershipInvocation &invocation);
  static llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
  sourceProvenance(const StructuredOwnershipInvocation &invocation);

  static llvm::Error recordGeneration(
      StructuredOwnershipInvocation &invocation,
      ArtifactRootReference generationParentReference,
      ArtifactRootReference workloadReference,
      ArtifactRootReference runtimeInputReference,
      llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions,
      std::vector<StructuredOwnershipCandidateState> candidates,
      const ArtifactStore &store);

  static llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
  clonePreClosureCandidate(StructuredOwnershipInvocation &invocation,
                           const ArtifactRootReference &reference);

  static llvm::Expected<std::optional<frontend::StructuredEntityRef>>
  ownedSpatialRegion(const StructuredOwnershipInvocation &invocation,
                     const ArtifactRootReference &reference);

  static llvm::Expected<
      llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>>
  sourceProvenance(const StructuredOwnershipInvocation &invocation,
                   const ArtifactRootReference &reference);

  static llvm::Error recordScheduleCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const frontend::StructuredScheduleDecision &decision,
      frontend::MaterializedStructuredScheduleCandidate candidate,
      const ArtifactStore &store);

  static llvm::Error recordExecutionShapeCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      std::optional<frontend::StructuredExecutionShapeDecision> decision,
      frontend::MaterializedStructuredOwnershipCandidate candidate,
      const ArtifactStore &store);

  static llvm::Error recordSpecialMathAccuracyDerivation(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const frontend::StructuredSpecialMathAccuracyDecision &decision,
      const ArtifactStore &store);

  static llvm::Error recordSpecialMathAccuracyMechanicalCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const ArtifactStore &store);

  static llvm::Error recordSpecialMathAccuracyFinalCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &child,
      frontend::MaterializedOwnershipCandidate candidate,
      const ArtifactStore &store);

  static llvm::Error recordMemoryCommunicationCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const frontend::StructuredMemoryCommunicationDecision &decision,
      frontend::MaterializedStructuredMemoryCommunicationCandidate candidate,
      const ArtifactStore &store);

  static llvm::Error recordDataflowRewriteCandidate(
      StructuredOwnershipInvocation &invocation,
      const ArtifactRootReference &parent, const ArtifactRootReference &child,
      const dataflow::DataflowRewriteDecision &decision,
      llvm::ArrayRef<dataflow::StaticGraphLaunchRef> parentLaunches,
      llvm::ArrayRef<dataflow::StaticGraphLaunchRef> childLaunches,
      const ArtifactStore &store);

  static llvm::Error
  primeAnalyticCandidate(StructuredOwnershipInvocation &invocation,
                         const ArtifactRootReference &candidate,
                         const ArtifactStore &store);

  static llvm::Error
  primeFunctionalReplay(StructuredOwnershipInvocation &invocation,
                        const ArtifactRootReference &candidate,
                        const ArtifactStore &store);

  static llvm::Error
  primeDataflowFunctionalReplay(StructuredOwnershipInvocation &invocation,
                                const ArtifactRootReference &structuredParent,
                                const ArtifactRootReference &dataflowCandidate,
                                const ArtifactStore &store);
};

} // namespace loom::dse::detail

#endif // LOOM_DSE_STRUCTUREDOWNERSHIPINVOCATIONINTERNAL_H
