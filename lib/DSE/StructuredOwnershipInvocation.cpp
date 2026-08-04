#include "DSE/StructuredOwnershipInvocation.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/NativeSimulationOracle.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

thread_local StructuredOwnershipInvocation *currentInvocation = nullptr;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_ownership_invocation_invalid: " +
                                     message);
}

bool sameRoot(const ArtifactRootReference &reference,
              const ArtifactSchemaDescriptor &schema,
              const ArtifactIdentity &identity) {
  return reference.schemaIdentity == schema.identity &&
         reference.schemaVersion == schema.version &&
         reference.artifact == identity;
}

} // namespace

class StructuredOwnershipInvocation::Impl final {
public:
  struct FinalCandidateState final {
    frontend::StructuredProgramCandidate structuredProgram;
    std::vector<frontend::StructuredBlockActivityLineage> blockActivityLineage;
    std::vector<frontend::StructuredOperationSourceProvenance> sourceProvenance;
    lowering::ProjectedCanonicalDataflow projected;
  };

  struct ResolvedLineage final {
    std::vector<StructuredOwnershipDerivation> ownership;
    std::vector<StructuredExecutionShapeDerivation> executionShape;
    std::vector<StructuredScheduleDerivation> schedule;
  };

  Impl(const frontend::StructuredProgramCandidate &sourceProgram,
       const sim::CanonicalSimulationWorkload &workload,
       const sim::CanonicalSimulationRuntimeInput &runtimeInput,
       const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
       const lowering::CanonicalDataflowLoweringOptions &lowering,
       std::uint32_t candidateWorkerCount,
       sim::SourceBackedDfgValidationLimits functionalReplayLimits,
       llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
           sourceProvenance)
      : sourceProgram(sourceProgram), workload(workload),
        runtimeInput(runtimeInput), fabric(fabric), config(config),
        lowering(lowering), candidateWorkerCount(candidateWorkerCount),
        functionalReplayLimits(functionalReplayLimits),
        sourceProvenance(sourceProvenance.begin(), sourceProvenance.end()),
        ownershipCandidates(&artifactRootReferenceLess),
        materialized(&artifactRootReferenceLess),
        executionShapeLineage(&artifactRootReferenceLess),
        scheduleLineage(&artifactRootReferenceLess),
        finalCandidates(&artifactRootReferenceLess) {}

  const frontend::StructuredProgramCandidate &sourceProgram;
  const sim::CanonicalSimulationWorkload &workload;
  const sim::CanonicalSimulationRuntimeInput &runtimeInput;
  const fabric::FinalizedFabricRoot &fabric;
  const ResolvedConfig &config;
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::uint32_t candidateWorkerCount;
  sim::SourceBackedDfgValidationLimits functionalReplayLimits;
  std::vector<frontend::StructuredOperationSourceProvenance> sourceProvenance;
  evaluation::models::StructuredEvaluationInvocationCache evaluationCache;

  std::optional<ArtifactRootReference> sourceReference;
  std::optional<ArtifactRootReference> workloadReference;
  std::optional<ArtifactRootReference> runtimeInputReference;
  std::optional<sim::NativeStructuredProgramObservations> sourceObservations;
  std::uint64_t sourceNativeExecutionCount = 0;
  bool generationRecorded = false;
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::map<ArtifactRootReference,
           frontend::MaterializedStructuredOwnershipCandidate,
           decltype(&artifactRootReferenceLess)>
      ownershipCandidates;
  std::map<ArtifactRootReference, frontend::MaterializedOwnershipCandidate,
           decltype(&artifactRootReferenceLess)>
      materialized;
  std::map<ArtifactRootReference,
           std::vector<StructuredExecutionShapeDerivation>,
           decltype(&artifactRootReferenceLess)>
      executionShapeLineage;
  std::map<ArtifactRootReference, std::vector<StructuredScheduleDerivation>,
           decltype(&artifactRootReferenceLess)>
      scheduleLineage;
  std::map<ArtifactRootReference, FinalCandidateState,
           decltype(&artifactRootReferenceLess)>
      finalCandidates;
  std::vector<ArtifactRootReference> primedCandidates;

  llvm::Expected<ResolvedLineage>
  resolveLineage(const ArtifactRootReference &candidate) const {
    ResolvedLineage result;
    std::vector<ArtifactRootReference> visiting;
    std::vector<ArtifactRootReference> visited;
    std::function<llvm::Error(const ArtifactRootReference &)> visit =
        [&](const ArtifactRootReference &reference) -> llvm::Error {
      if (llvm::is_contained(visited, reference))
        return llvm::Error::success();
      if (llvm::is_contained(visiting, reference))
        return invalid("Structured candidate lineage contains a cycle");
      visiting.push_back(reference);
      for (const StructuredOwnershipCandidateDisposition &disposition :
           dispositions) {
        const auto *derived =
            std::get_if<ArtifactRootReference>(&disposition.result);
        if (!derived || *derived != reference ||
            !disposition.coordinate.decision)
          continue;
        StructuredOwnershipDerivation derivation{
            disposition.coordinate.scope, *disposition.coordinate.decision};
        if (!llvm::is_contained(result.ownership, derivation))
          result.ownership.push_back(std::move(derivation));
      }
      auto edges = scheduleLineage.find(reference);
      if (edges != scheduleLineage.end()) {
        for (const StructuredScheduleDerivation &edge : edges->second) {
          if (llvm::Error error = visit(edge.parent))
            return error;
          if (!llvm::is_contained(result.schedule, edge))
            result.schedule.push_back(edge);
        }
      }
      auto executionEdges = executionShapeLineage.find(reference);
      if (executionEdges != executionShapeLineage.end()) {
        for (const StructuredExecutionShapeDerivation &edge :
             executionEdges->second) {
          if (llvm::Error error = visit(edge.parent))
            return error;
          if (!llvm::is_contained(result.executionShape, edge))
            result.executionShape.push_back(edge);
        }
      }
      visiting.pop_back();
      visited.push_back(reference);
      return llvm::Error::success();
    };
    if (llvm::Error error = visit(candidate))
      return error;
    if (result.ownership.empty())
      return invalid("Structured candidate has no Ownership lineage");
    return result;
  }
};

class StructuredOwnershipInvocationScope::Impl final {
public:
  explicit Impl(StructuredOwnershipInvocation &invocation)
      : previous(
            detail::StructuredOwnershipInvocationAccess::bind(&invocation)),
        evaluationScope(
            detail::StructuredOwnershipInvocationAccess::evaluationCache(
                invocation)) {}

  ~Impl() { detail::StructuredOwnershipInvocationAccess::bind(previous); }

private:
  StructuredOwnershipInvocation *previous;
  evaluation::models::StructuredEvaluationInvocationCacheScope evaluationScope;
};

StructuredOwnershipInvocation::StructuredOwnershipInvocation(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const lowering::CanonicalDataflowLoweringOptions &lowering,
    std::uint32_t candidateWorkerCount,
    sim::SourceBackedDfgValidationLimits functionalReplayLimits,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance)
    : impl_(std::make_unique<Impl>(
          sourceProgram, workload, runtimeInput, fabric, config, lowering,
          candidateWorkerCount, functionalReplayLimits, sourceProvenance)) {}

StructuredOwnershipInvocation::~StructuredOwnershipInvocation() = default;

llvm::Error StructuredOwnershipInvocation::prepareSource(
    const ArtifactRootReference &source, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ArtifactStore &store) {
  Impl &impl = *impl_;
  if (impl.sourceReference) {
    if (!impl.workloadReference || !impl.runtimeInputReference ||
        !impl.sourceObservations || *impl.sourceReference != source ||
        *impl.workloadReference != workload ||
        *impl.runtimeInputReference != runtimeInput)
      return invalid("prepared source roots differ from the invocation");
    return llvm::Error::success();
  }
  if (impl.workloadReference || impl.runtimeInputReference ||
      impl.sourceObservations)
    return invalid("source preparation is partially initialized");
  if (!sameRoot(source, frontend::structuredProgramArtifactSchema,
                impl.sourceProgram.identity()) ||
      !sameRoot(workload, sim::simulationWorkloadSchema,
                impl.workload.identity()) ||
      !sameRoot(runtimeInput, sim::simulationRuntimeInputSchema,
                impl.runtimeInput.identity()))
    return invalid("prepared roots differ from the bound invocation");

  auto observations = sim::executeNativeStructuredProgram(
      impl.sourceProgram, impl.workload, impl.runtimeInput);
  if (!observations)
    return observations.takeError();
  evaluation::models::StructuredEvaluationInvocationCacheScope cacheScope(
      impl.evaluationCache);
  if (llvm::Error error =
          evaluation::models::primeStructuredProgramSourceObservations(
              source, workload, runtimeInput, *observations))
    return error;
  const evaluation::models::StructuredFabricAnalyticInvocation invocation{
      workload,          runtimeInput,       impl.workload,
      impl.runtimeInput, impl.sourceProgram, *observations};
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              source, {impl.sourceProgram, nullptr, {}, {}, &*observations},
              invocation, impl.fabric, impl.config, store))
    return error;

  impl.sourceReference.emplace(source);
  impl.workloadReference.emplace(workload);
  impl.runtimeInputReference.emplace(runtimeInput);
  impl.sourceObservations.emplace(std::move(*observations));
  ++impl.sourceNativeExecutionCount;
  return llvm::Error::success();
}

std::uint64_t
StructuredOwnershipInvocation::sourceNativeExecutionCount() const {
  return impl_->sourceNativeExecutionCount;
}

evaluation::models::StructuredEvaluationInvocationCacheStatistics
StructuredOwnershipInvocation::evaluationCacheStatistics() const {
  return impl_->evaluationCache.statistics();
}

llvm::ArrayRef<StructuredOwnershipCandidateDisposition>
StructuredOwnershipInvocation::dispositions() const {
  return impl_->dispositions;
}

llvm::Expected<SelectedStructuredOwnershipCandidate>
StructuredOwnershipInvocation::materializeSelectedCandidate(
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  if (!impl_->sourceReference || !impl_->workloadReference ||
      !impl_->runtimeInputReference)
    return invalid("candidate generation has not completed");

  std::optional<frontend::MaterializedOwnershipCandidate> materialized;
  if (candidate == *impl_->sourceReference) {
    auto structured = frontend::importStructuredProgram(candidate, store);
    if (!structured)
      return structured.takeError();
    auto dataflow = lowering::lowerStructuredProgramToCanonicalDataflow(
        *structured, impl_->lowering);
    if (!dataflow)
      return dataflow.takeError();
    materialized.emplace(frontend::MaterializedOwnershipCandidate{
        std::move(*structured), std::move(*dataflow), {}, {}, {}});
  } else {
    auto found = impl_->materialized.find(candidate);
    if (found == impl_->materialized.end())
      return invalid("selected candidate has no functional replay projection");
    auto dataflow = dataflow::importCanonicalDataflow(
        found->second.canonicalDataflow.identity(),
        found->second.canonicalDataflow.canonicalBytes());
    if (!dataflow)
      return dataflow.takeError();
    materialized.emplace(frontend::MaterializedOwnershipCandidate{
        std::move(found->second.structuredProgram), std::move(*dataflow),
        std::move(found->second.spatialGraphs),
        std::move(found->second.blockActivityLineage),
        std::move(found->second.sourceProvenance)});
    impl_->materialized.erase(found);
  }

  std::vector<StructuredOwnershipDerivation> derivations;
  std::vector<StructuredExecutionShapeDerivation> executionShapeDerivations;
  std::vector<StructuredScheduleDerivation> scheduleDerivations;
  if (candidate != *impl_->sourceReference) {
    auto lineage = impl_->resolveLineage(candidate);
    if (!lineage)
      return lineage.takeError();
    derivations = std::move(lineage->ownership);
    executionShapeDerivations = std::move(lineage->executionShape);
    scheduleDerivations = std::move(lineage->schedule);
  }

  std::optional<sim::SourceBackedDfgValidationResult> functionalReplay;
  if (candidate != *impl_->sourceReference) {
    auto replay =
        evaluation::models::getPrimedStructuredProgramFunctionalReplay(
            candidate, *impl_->workloadReference,
            *impl_->runtimeInputReference);
    if (!replay)
      return replay.takeError();
    if (replay->status != sim::SourceBackedDfgValidationStatus::Equivalent)
      return invalid("selected accelerator candidate lacks equivalent replay");
    functionalReplay.emplace(std::move(*replay));
  }
  return SelectedStructuredOwnershipCandidate{
      std::move(*materialized), std::move(derivations),
      std::move(executionShapeDerivations), std::move(scheduleDerivations),
      std::move(functionalReplay)};
}

StructuredOwnershipInvocationScope::StructuredOwnershipInvocationScope(
    StructuredOwnershipInvocation &invocation)
    : impl_(std::make_unique<Impl>(invocation)) {}

StructuredOwnershipInvocationScope::~StructuredOwnershipInvocationScope() =
    default;

StructuredOwnershipInvocation *
detail::StructuredOwnershipInvocationAccess::current() {
  return currentInvocation;
}

StructuredOwnershipInvocation *
detail::StructuredOwnershipInvocationAccess::bind(
    StructuredOwnershipInvocation *invocation) {
  StructuredOwnershipInvocation *previous = currentInvocation;
  currentInvocation = invocation;
  return previous;
}

llvm::Error detail::StructuredOwnershipInvocationAccess::prepareGeneration(
    StructuredOwnershipInvocation &invocation,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ArtifactStore &store,
    StructuredOwnershipGenerationOptions &options) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (impl.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (sourceProgram.identity() != impl.sourceProgram.identity() ||
      workload.identity() != impl.workload.identity() ||
      runtimeInput.identity() != impl.runtimeInput.identity() ||
      fabric.reference() != impl.fabric.reference())
    return invalid("Generate inputs differ from the bound invocation");
  options.lowering = impl.lowering;
  options.candidateWorkerCount = impl.candidateWorkerCount;
  if (!impl.sourceReference) {
    auto sourceReference =
        frontend::publishStructuredProgram(impl.sourceProgram, store);
    if (!sourceReference)
      return sourceReference.takeError();
    auto workloadReference =
        sim::publishSimulationWorkload(impl.workload, store);
    if (!workloadReference)
      return workloadReference.takeError();
    auto runtimeInputReference =
        sim::publishSimulationRuntimeInput(impl.runtimeInput, store);
    if (!runtimeInputReference)
      return runtimeInputReference.takeError();
    if (llvm::Error error =
            invocation.prepareSource(*sourceReference, *workloadReference,
                                     *runtimeInputReference, store))
      return error;
  }
  return llvm::Error::success();
}

llvm::Expected<detail::StructuredOwnershipPreparedSource>
detail::StructuredOwnershipInvocationAccess::preparedSource(
    const StructuredOwnershipInvocation &invocation) {
  const StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.sourceReference || !impl.workloadReference ||
      !impl.runtimeInputReference || !impl.sourceObservations)
    return invalid("source preparation has not completed");
  return StructuredOwnershipPreparedSource{
      *impl.sourceReference, *impl.workloadReference,
      *impl.runtimeInputReference, *impl.sourceObservations};
}

const ResolvedConfig &detail::StructuredOwnershipInvocationAccess::config(
    const StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->config;
}

const lowering::CanonicalDataflowLoweringOptions &
detail::StructuredOwnershipInvocationAccess::loweringOptions(
    const StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->lowering;
}

const fabric::FinalizedFabricRoot &
detail::StructuredOwnershipInvocationAccess::fabric(
    const StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->fabric;
}

evaluation::models::StructuredEvaluationInvocationCache &
detail::StructuredOwnershipInvocationAccess::evaluationCache(
    StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->evaluationCache;
}

llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
detail::StructuredOwnershipInvocationAccess::sourceProvenance(
    const StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->sourceProvenance;
}

llvm::Error detail::StructuredOwnershipInvocationAccess::recordGeneration(
    StructuredOwnershipInvocation &invocation,
    ArtifactRootReference sourceReference,
    ArtifactRootReference workloadReference,
    ArtifactRootReference runtimeInputReference,
    llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions,
    std::vector<StructuredOwnershipCandidateState> candidates,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (impl.generationRecorded)
    return invalid("invocation contains more than one Ownership generation");
  if (!impl.sourceReference || !impl.workloadReference ||
      !impl.runtimeInputReference || !impl.sourceObservations ||
      *impl.sourceReference != sourceReference ||
      *impl.workloadReference != workloadReference ||
      *impl.runtimeInputReference != runtimeInputReference)
    return invalid("generated roots differ from the bound invocation");
  if (!impl.ownershipCandidates.empty())
    return invalid("Ownership cache is initialized before generation");
  std::map<ArtifactRootReference, std::size_t,
           decltype(&artifactRootReferenceLess)>
      uniqueCandidates(&artifactRootReferenceLess);
  for (auto item : llvm::enumerate(candidates)) {
    StructuredOwnershipCandidateState &state = item.value();
    if (state.reference.schemaIdentity !=
            frontend::structuredProgramArtifactSchema.identity ||
        state.reference.schemaVersion !=
            frontend::structuredProgramArtifactSchema.version ||
        state.reference.artifact !=
            state.candidate.structuredProgram.identity())
      return invalid("Ownership cache contains a foreign Structured reference");
    auto stored = store.get(state.reference);
    if (!stored)
      return stored.takeError();
    if (!stored->bytes().equals(
            state.candidate.structuredProgram.canonicalBytes().bytes()))
      return invalid("Ownership cache differs from its published bytes");
    auto [found, inserted] =
        uniqueCandidates.try_emplace(state.reference, item.index());
    if (!inserted &&
        candidates[found->second]
                .candidate.structuredProgram.canonicalBytes()
                .bytes() !=
            state.candidate.structuredProgram.canonicalBytes().bytes())
      return invalid(
          "deduplicated Ownership candidate changed canonical bytes");
  }
  impl.dispositions.assign(dispositions.begin(), dispositions.end());
  for (auto [reference, index] : uniqueCandidates)
    impl.ownershipCandidates.try_emplace(
        reference, std::move(candidates[index].candidate));
  impl.generationRecorded = true;
  return llvm::Error::success();
}

llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
detail::StructuredOwnershipInvocationAccess::cloneOwnershipCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &reference) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded)
    return invalid("ExecutionShape generation precedes Ownership generation");
  auto found = impl.ownershipCandidates.find(reference);
  if (found == impl.ownershipCandidates.end())
    return invalid("ExecutionShape input has no Ownership candidate state");
  auto clone = frontend::importStructuredProgram(
      found->second.structuredProgram.identity(),
      found->second.structuredProgram.canonicalBytes());
  if (!clone)
    return clone.takeError();
  return frontend::MaterializedStructuredOwnershipCandidate{
      std::move(*clone), found->second.blockActivityLineage,
      found->second.sourceProvenance};
}

llvm::Error
detail::StructuredOwnershipInvocationAccess::recordScheduleCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    const frontend::StructuredScheduleDecision &decision,
    frontend::MaterializedStructuredScheduleCandidate candidate,
    lowering::ProjectedCanonicalDataflow projected,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded || !impl.sourceReference)
    return invalid("Schedule generation precedes Ownership generation");
  if (parent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      child.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.artifact != candidate.structuredProgram.identity())
    return invalid("Schedule lineage contains a foreign Structured reference");
  if (parent == child)
    return llvm::Error::success();
  if (auto lineage = impl.resolveLineage(parent); !lineage)
    return lineage.takeError();
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          candidate.structuredProgram.canonicalBytes().bytes()))
    return invalid("Schedule child differs from its published bytes");

  StructuredScheduleDerivation derivation{parent, decision};
  std::vector<StructuredScheduleDerivation> &edges =
      impl.scheduleLineage[child];
  if (!llvm::is_contained(edges, derivation))
    edges.push_back(derivation);

  auto found = impl.finalCandidates.find(child);
  if (found == impl.finalCandidates.end()) {
    impl.finalCandidates.try_emplace(
        child, StructuredOwnershipInvocation::Impl::FinalCandidateState{
                   std::move(candidate.structuredProgram),
                   {},
                   std::move(candidate.sourceProvenance),
                   std::move(projected)});
  } else if (found->second.structuredProgram.canonicalBytes().bytes() !=
             candidate.structuredProgram.canonicalBytes().bytes()) {
    return invalid("deduplicated Schedule child changed canonical bytes");
  }
  return llvm::Error::success();
}

llvm::Error
detail::StructuredOwnershipInvocationAccess::recordExecutionShapeCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    std::optional<frontend::StructuredExecutionShapeDecision> decision,
    frontend::MaterializedStructuredOwnershipCandidate candidate,
    lowering::ProjectedCanonicalDataflow projected,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded || !impl.sourceReference)
    return invalid("ExecutionShape generation precedes Ownership generation");
  if (parent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      child.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.artifact != candidate.structuredProgram.identity())
    return invalid(
        "ExecutionShape lineage contains a foreign Structured reference");
  if (auto lineage = impl.resolveLineage(parent); !lineage)
    return lineage.takeError();
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          candidate.structuredProgram.canonicalBytes().bytes()))
    return invalid("ExecutionShape child differs from its published bytes");

  if (decision && parent != child) {
    StructuredExecutionShapeDerivation derivation{parent, *decision};
    std::vector<StructuredExecutionShapeDerivation> &edges =
        impl.executionShapeLineage[child];
    if (!llvm::is_contained(edges, derivation))
      edges.push_back(std::move(derivation));
  } else if (decision || parent != child) {
    return invalid("ExecutionShape pass-through has inconsistent lineage");
  }

  auto found = impl.finalCandidates.find(child);
  if (found == impl.finalCandidates.end()) {
    impl.finalCandidates.try_emplace(
        child,
        StructuredOwnershipInvocation::Impl::FinalCandidateState{
            std::move(candidate.structuredProgram),
            std::move(candidate.blockActivityLineage),
            std::move(candidate.sourceProvenance), std::move(projected)});
  } else if (found->second.structuredProgram.canonicalBytes().bytes() !=
             candidate.structuredProgram.canonicalBytes().bytes()) {
    return invalid("deduplicated ExecutionShape child changed canonical bytes");
  }
  return llvm::Error::success();
}

llvm::Error detail::StructuredOwnershipInvocationAccess::primeAnalyticCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  auto found = impl.finalCandidates.find(candidate);
  if (found == impl.finalCandidates.end())
    return llvm::Error::success();
  auto &state = found->second;
  auto observations = sim::executeProfiledSelectedStructuredProgram(
      state.structuredProgram, impl.sourceProgram, impl.workload,
      impl.runtimeInput);
  if (!observations) {
    llvm::consumeError(observations.takeError());
    return llvm::Error::success();
  }
  const evaluation::models::StructuredFabricAnalyticInvocation analytic{
      *impl.workloadReference, *impl.runtimeInputReference,
      impl.workload,           impl.runtimeInput,
      impl.sourceProgram,      *impl.sourceObservations};
  return evaluation::models::primeStructuredFabricAnalyticResult(
      candidate,
      {state.structuredProgram,
       &state.projected.artifact,
       state.projected.spatialGraphs,
       {},
       &*observations},
      analytic, impl.fabric, impl.config, store);
}

llvm::Error detail::StructuredOwnershipInvocationAccess::primeFunctionalReplay(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.sourceReference || !impl.workloadReference ||
      !impl.runtimeInputReference || !impl.sourceObservations)
    return invalid("functional acquisition precedes Ownership generation");
  if (candidate == *impl.sourceReference)
    return llvm::Error::success();
  if (llvm::is_contained(impl.primedCandidates, candidate))
    return llvm::Error::success();

  auto lineage = impl.resolveLineage(candidate);
  if (!lineage)
    return lineage.takeError();

  std::optional<frontend::MaterializedOwnershipCandidate> materialized;
  auto scheduled = impl.finalCandidates.find(candidate);
  if (scheduled != impl.finalCandidates.end()) {
    auto state = std::move(scheduled->second);
    impl.finalCandidates.erase(scheduled);
    materialized.emplace(frontend::MaterializedOwnershipCandidate{
        std::move(state.structuredProgram), std::move(state.projected.artifact),
        std::move(state.projected.spatialGraphs),
        std::move(state.blockActivityLineage),
        std::move(state.sourceProvenance)});
  } else {
    const StructuredOwnershipDerivation &derivation =
        lineage->ownership.front();
    auto direct = frontend::materializeSpatialOwnershipDecision(
        impl.sourceProgram, derivation.scope, derivation.decision, impl.fabric,
        impl.lowering, impl.sourceProvenance);
    if (!direct)
      return direct.takeError();
    materialized.emplace(std::move(*direct));
  }
  if (materialized->structuredProgram.identity() != candidate.artifact)
    return invalid("rematerialized candidate changed identity");
  auto stored = store.get(candidate);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          materialized->structuredProgram.canonicalBytes().bytes()))
    return invalid("rematerialized candidate changed canonical bytes");

  auto [found, inserted] =
      impl.materialized.try_emplace(candidate, std::move(*materialized));
  if (!inserted)
    return invalid("functional candidate was materialized twice");
  std::vector<frontend::StructuredExecutionShapeDecision>
      executionShapeDecisions;
  executionShapeDecisions.reserve(lineage->executionShape.size());
  for (const StructuredExecutionShapeDerivation &derivation :
       lineage->executionShape)
    executionShapeDecisions.push_back(derivation.decision);
  for (const StructuredOwnershipDerivation &derivation : lineage->ownership) {
    if (llvm::Error error =
            evaluation::models::primeStructuredProgramFunctionalReplay(
                candidate,
                {*impl.workloadReference, *impl.runtimeInputReference,
                 impl.sourceProgram, derivation.scope, derivation.decision,
                 executionShapeDecisions, found->second, impl.workload,
                 impl.runtimeInput, *impl.sourceObservations,
                 impl.functionalReplayLimits},
                store))
      return error;
  }
  impl.primedCandidates.push_back(candidate);
  llvm::sort(impl.primedCandidates, artifactRootReferenceLess);
  return llvm::Error::success();
}

} // namespace loom::dse
