#include "DSE/StructuredOwnershipInvocation.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "StructuredOwnershipInvocationInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
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
        materialized(&artifactRootReferenceLess) {}

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
  std::vector<StructuredOwnershipCandidateDisposition> dispositions;
  std::map<ArtifactRootReference, frontend::MaterializedOwnershipCandidate,
           decltype(&artifactRootReferenceLess)>
      materialized;
  std::vector<ArtifactRootReference> primedCandidates;
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
  for (const StructuredOwnershipCandidateDisposition &disposition :
       impl_->dispositions) {
    const auto *reference =
        std::get_if<ArtifactRootReference>(&disposition.result);
    if (!reference || *reference != candidate ||
        !disposition.coordinate.decision)
      continue;
    derivations.push_back(
        {disposition.coordinate.scope, *disposition.coordinate.decision});
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
  return SelectedStructuredOwnershipCandidate{std::move(*materialized),
                                              std::move(derivations),
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
    const fabric::FinalizedFabricRoot &fabric,
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
  return llvm::Error::success();
}

const ResolvedConfig &detail::StructuredOwnershipInvocationAccess::config(
    const StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->config;
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
    sim::NativeStructuredProgramObservations sourceObservations,
    llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (impl.sourceReference)
    return invalid("invocation contains more than one Ownership generation");
  if (!sameRoot(sourceReference, frontend::structuredProgramArtifactSchema,
                impl.sourceProgram.identity()) ||
      !sameRoot(workloadReference, sim::simulationWorkloadSchema,
                impl.workload.identity()) ||
      !sameRoot(runtimeInputReference, sim::simulationRuntimeInputSchema,
                impl.runtimeInput.identity()))
    return invalid("generated roots differ from the bound invocation");
  impl.sourceReference.emplace(std::move(sourceReference));
  impl.workloadReference.emplace(std::move(workloadReference));
  impl.runtimeInputReference.emplace(std::move(runtimeInputReference));
  impl.sourceObservations.emplace(std::move(sourceObservations));
  impl.dispositions.assign(dispositions.begin(), dispositions.end());
  return llvm::Error::success();
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

  std::vector<const StructuredOwnershipCandidateDisposition *> derivations;
  for (const StructuredOwnershipCandidateDisposition &disposition :
       impl.dispositions) {
    const auto *reference =
        std::get_if<ArtifactRootReference>(&disposition.result);
    if (reference && *reference == candidate && disposition.coordinate.decision)
      derivations.push_back(&disposition);
  }
  if (derivations.empty())
    return invalid("functional candidate has no Ownership derivation");

  auto materialized = frontend::materializeSpatialOwnershipDecision(
      impl.sourceProgram, derivations.front()->coordinate.scope,
      *derivations.front()->coordinate.decision, impl.fabric, impl.lowering,
      impl.sourceProvenance);
  if (!materialized)
    return materialized.takeError();
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
  for (const StructuredOwnershipCandidateDisposition *derivation :
       derivations) {
    if (llvm::Error error =
            evaluation::models::primeStructuredProgramFunctionalReplay(
                candidate,
                {*impl.workloadReference, *impl.runtimeInputReference,
                 impl.sourceProgram, derivation->coordinate.scope,
                 *derivation->coordinate.decision, found->second, impl.workload,
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
