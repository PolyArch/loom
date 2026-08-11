#include "DSE/StructuredOwnershipInvocation.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Evaluation/Models/CanonicalDataflowFunctional.h"
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
#include <set>
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

struct ArtifactReferenceLess final {
  bool operator()(const ArtifactRootReference &lhs,
                  const ArtifactRootReference &rhs) const {
    return artifactRootReferenceLess(lhs, rhs);
  }
};

using ArtifactReferenceSet =
    std::set<ArtifactRootReference, ArtifactReferenceLess>;

struct CanonicalDerivationKey final {
  ArtifactRootReference parent;
  std::vector<std::uint8_t> ownerPayload;
};

struct CanonicalDerivationKeyLess final {
  bool operator()(const CanonicalDerivationKey &lhs,
                  const CanonicalDerivationKey &rhs) const {
    if (artifactRootReferenceLess(lhs.parent, rhs.parent))
      return true;
    if (artifactRootReferenceLess(rhs.parent, lhs.parent))
      return false;
    return lhs.ownerPayload < rhs.ownerPayload;
  }
};

template <typename Derivation>
using CanonicalDerivationSet =
    std::map<CanonicalDerivationKey, Derivation, CanonicalDerivationKeyLess>;

template <typename Derivation>
using CanonicalDerivationIndex =
    std::map<ArtifactRootReference, CanonicalDerivationSet<Derivation>,
             ArtifactReferenceLess>;

struct DataflowReplayKeyLess final {
  bool
  operator()(const std::pair<ArtifactRootReference, ArtifactRootReference> &lhs,
             const std::pair<ArtifactRootReference, ArtifactRootReference> &rhs)
      const {
    if (artifactRootReferenceLess(lhs.first, rhs.first))
      return true;
    if (artifactRootReferenceLess(rhs.first, lhs.first))
      return false;
    return artifactRootReferenceLess(lhs.second, rhs.second);
  }
};

} // namespace

class detail::StructuredOwnershipDataflowLineageIndex::Impl final {
public:
  std::map<ArtifactRootReference, ArtifactRootReference,
           decltype(&artifactRootReferenceLess)>
      roots{&artifactRootReferenceLess};
  CanonicalDerivationIndex<DataflowRewriteDerivation> lineage;
  ArtifactReferenceSet nodes;
};

detail::StructuredOwnershipDataflowLineageIndex::
    StructuredOwnershipDataflowLineageIndex()
    : impl_(std::make_unique<Impl>()) {}

detail::StructuredOwnershipDataflowLineageIndex::
    ~StructuredOwnershipDataflowLineageIndex() = default;

bool detail::StructuredOwnershipDataflowLineageIndex::empty() const {
  return impl_->roots.empty();
}

llvm::Error detail::StructuredOwnershipDataflowLineageIndex::recordRoot(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &dataflowRoot) {
  if (structuredParent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      structuredParent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      dataflowRoot.schemaIdentity !=
          dataflow::canonicalDataflowSchema.identity ||
      dataflowRoot.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("Dataflow root has a foreign Artifact schema");
  auto [root, inserted] =
      impl_->roots.try_emplace(structuredParent, dataflowRoot);
  if (!inserted && root->second != dataflowRoot)
    return invalid("Structured parent changed its Canonical Dataflow root");
  impl_->nodes.insert(dataflowRoot);
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference>
detail::StructuredOwnershipDataflowLineageIndex::root(
    const ArtifactRootReference &structuredParent) const {
  auto root = impl_->roots.find(structuredParent);
  if (root == impl_->roots.end())
    return invalid("Structured parent has no Canonical Dataflow root");
  return root->second;
}

llvm::Error detail::StructuredOwnershipDataflowLineageIndex::recordDecision(
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    const dataflow::DataflowRewriteDecision &decision) {
  if (parent.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      parent.schemaVersion != dataflow::canonicalDataflowSchema.version ||
      child.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      child.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("Dataflow rewrite lineage contains a foreign reference");
  if (impl_->nodes.find(parent) == impl_->nodes.end())
    return invalid("Dataflow rewrite parent is outside the prepared lineage");
  if (parent == child)
    return invalid("Dataflow rewrite cannot derive a candidate from itself");
  auto payload = dataflow::encodeDataflowRewriteDecision(decision);
  if (!payload)
    return payload.takeError();
  DataflowRewriteDerivation derivation{parent, child, decision};
  auto [lineage, inserted] = impl_->lineage[child].try_emplace(
      CanonicalDerivationKey{parent, std::move(*payload)}, derivation);
  if (!inserted && !(lineage->second == derivation))
    return invalid("Dataflow lineage key has conflicting decisions");
  impl_->nodes.insert(child);
  return llvm::Error::success();
}

llvm::Expected<std::optional<std::vector<DataflowRewriteDerivation>>>
detail::StructuredOwnershipDataflowLineageIndex::tryResolve(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &candidate) const {
  auto root = impl_->roots.find(structuredParent);
  if (root == impl_->roots.end())
    return invalid("Structured parent has no Canonical Dataflow root");
  if (impl_->nodes.find(candidate) == impl_->nodes.end())
    return std::optional<std::vector<DataflowRewriteDerivation>>{};

  std::vector<DataflowRewriteDerivation> result;
  enum class VisitState { Visiting, DeadEnd, ReachesRoot };
  std::map<ArtifactRootReference, VisitState,
           decltype(&artifactRootReferenceLess)>
      states(&artifactRootReferenceLess);
  std::function<llvm::Expected<bool>(const ArtifactRootReference &)> visit =
      [&](const ArtifactRootReference &reference) -> llvm::Expected<bool> {
    if (reference == root->second)
      return true;
    auto state = states.find(reference);
    if (state != states.end()) {
      if (state->second == VisitState::Visiting)
        return invalid("Dataflow candidate lineage contains a cycle");
      return state->second == VisitState::ReachesRoot;
    }
    auto edges = impl_->lineage.find(reference);
    if (edges == impl_->lineage.end()) {
      states.try_emplace(reference, VisitState::DeadEnd);
      return false;
    }
    states.try_emplace(reference, VisitState::Visiting);
    bool reachesRoot = false;
    for (const auto &entry : edges->second) {
      const DataflowRewriteDerivation &edge = entry.second;
      auto reaches = visit(edge.parent);
      if (!reaches)
        return reaches.takeError();
      if (!*reaches)
        continue;
      reachesRoot = true;
      result.push_back(edge);
    }
    states.find(reference)->second =
        reachesRoot ? VisitState::ReachesRoot : VisitState::DeadEnd;
    return reachesRoot;
  };
  auto reaches = visit(candidate);
  if (!reaches)
    return reaches.takeError();
  if (!*reaches)
    return std::optional<std::vector<DataflowRewriteDerivation>>{};
  llvm::sort(result, [](const DataflowRewriteDerivation &lhs,
                        const DataflowRewriteDerivation &rhs) {
    if (artifactRootReferenceLess(lhs.parent, rhs.parent))
      return true;
    if (artifactRootReferenceLess(rhs.parent, lhs.parent))
      return false;
    if (artifactRootReferenceLess(lhs.child, rhs.child))
      return true;
    if (artifactRootReferenceLess(rhs.child, lhs.child))
      return false;
    return dataflow::dataflowRewriteDecisionLess(lhs.decision, rhs.decision);
  });
  return std::optional<std::vector<DataflowRewriteDerivation>>(
      std::move(result));
}

class StructuredOwnershipInvocation::Impl final {
public:
  struct FinalCandidateState final {
    frontend::StructuredProgramCandidate structuredProgram;
    std::optional<frontend::StructuredEntityRef> ownedSpatialRegion;
    std::vector<frontend::StructuredBlockActivityLineage> blockActivityLineage;
    std::vector<frontend::StructuredOperationSourceProvenance> sourceProvenance;
    lowering::ProjectedCanonicalDataflow projected;
  };

  struct ResolvedLineage final {
    std::vector<StructuredOwnershipDerivation> ownership;
    std::vector<StructuredExecutionShapeDerivation> executionShape;
    std::vector<StructuredSpecialMathAccuracyDerivation> specialMathAccuracy;
    std::vector<StructuredScheduleDerivation> schedule;
    std::vector<StructuredMemoryCommunicationDerivation> memoryCommunication;
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
        structuredCandidates(&artifactRootReferenceLess),
        materialized(&artifactRootReferenceLess),
        specialMathMechanicalParents(&artifactRootReferenceLess),
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
      structuredCandidates;
  std::map<ArtifactRootReference, frontend::MaterializedOwnershipCandidate,
           decltype(&artifactRootReferenceLess)>
      materialized;
  CanonicalDerivationIndex<StructuredOwnershipDerivation> ownershipLineage;
  CanonicalDerivationIndex<StructuredExecutionShapeDerivation>
      executionShapeLineage;
  CanonicalDerivationIndex<StructuredSpecialMathAccuracyDerivation>
      specialMathAccuracyLineage;
  std::map<ArtifactRootReference, ArtifactRootReference,
           decltype(&artifactRootReferenceLess)>
      specialMathMechanicalParents;
  CanonicalDerivationIndex<StructuredScheduleDerivation> scheduleLineage;
  CanonicalDerivationIndex<StructuredMemoryCommunicationDerivation>
      memoryCommunicationLineage;
  detail::StructuredOwnershipDataflowLineageIndex dataflowLineage;
  std::map<ArtifactRootReference, FinalCandidateState,
           decltype(&artifactRootReferenceLess)>
      finalCandidates;
  ArtifactReferenceSet structuredReachable;
  ArtifactReferenceSet primedCandidates;
  std::set<std::pair<ArtifactRootReference, ArtifactRootReference>,
           DataflowReplayKeyLess>
      primedDataflowCandidates;

  llvm::Expected<ResolvedLineage>
  resolveLineage(const ArtifactRootReference &candidate) const {
    ResolvedLineage result;
    ArtifactReferenceSet visiting;
    ArtifactReferenceSet visited;
    std::function<llvm::Error(const ArtifactRootReference &)> visit =
        [&](const ArtifactRootReference &reference) -> llvm::Error {
      if (visited.find(reference) != visited.end())
        return llvm::Error::success();
      if (!visiting.insert(reference).second)
        return invalid("Structured candidate lineage contains a cycle");
      auto ownershipEdges = ownershipLineage.find(reference);
      if (ownershipEdges != ownershipLineage.end())
        for (const auto &entry : ownershipEdges->second)
          result.ownership.push_back(entry.second);
      auto memoryEdges = memoryCommunicationLineage.find(reference);
      if (memoryEdges != memoryCommunicationLineage.end()) {
        for (const auto &entry : memoryEdges->second) {
          const StructuredMemoryCommunicationDerivation &edge = entry.second;
          if (llvm::Error error = visit(edge.parent))
            return error;
          result.memoryCommunication.push_back(edge);
        }
      }
      auto edges = scheduleLineage.find(reference);
      if (edges != scheduleLineage.end()) {
        for (const auto &entry : edges->second) {
          const StructuredScheduleDerivation &edge = entry.second;
          if (llvm::Error error = visit(edge.parent))
            return error;
          result.schedule.push_back(edge);
        }
      }
      auto executionEdges = executionShapeLineage.find(reference);
      auto accuracyEdges = specialMathAccuracyLineage.find(reference);
      auto mechanicalParent = specialMathMechanicalParents.find(reference);
      if (mechanicalParent != specialMathMechanicalParents.end())
        if (llvm::Error error = visit(mechanicalParent->second))
          return error;
      if (accuracyEdges != specialMathAccuracyLineage.end()) {
        for (const auto &entry : accuracyEdges->second) {
          const StructuredSpecialMathAccuracyDerivation &edge = entry.second;
          if (llvm::Error error = visit(edge.parent))
            return error;
          result.specialMathAccuracy.push_back(edge);
        }
      }
      if (executionEdges != executionShapeLineage.end()) {
        for (const auto &entry : executionEdges->second) {
          const StructuredExecutionShapeDerivation &edge = entry.second;
          if (llvm::Error error = visit(edge.parent))
            return error;
          result.executionShape.push_back(edge);
        }
      }
      visiting.erase(reference);
      visited.insert(reference);
      return llvm::Error::success();
    };
    if (llvm::Error error = visit(candidate))
      return error;
    if (result.ownership.empty())
      return invalid("Structured candidate has no Ownership lineage");
    return result;
  }

  llvm::Expected<std::optional<std::vector<DataflowRewriteDerivation>>>
  tryResolveDataflowLineage(const ArtifactRootReference &structuredParent,
                            const ArtifactRootReference &candidate) const {
    return dataflowLineage.tryResolve(structuredParent, candidate);
  }

  llvm::Expected<std::vector<DataflowRewriteDerivation>>
  resolveDataflowLineage(const ArtifactRootReference &structuredParent,
                         const ArtifactRootReference &candidate) const {
    auto result = tryResolveDataflowLineage(structuredParent, candidate);
    if (!result)
      return result.takeError();
    if (!*result)
      return invalid("Dataflow candidate does not descend from its exact D0");
    return std::move(**result);
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
        std::move(*structured), std::move(*dataflow), {}, std::nullopt, {}, {}});
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
        std::move(found->second.ownedSpatialRegion),
        std::move(found->second.blockActivityLineage),
        std::move(found->second.sourceProvenance)});
    impl_->materialized.erase(found);
  }

  std::vector<StructuredOwnershipDerivation> derivations;
  std::vector<StructuredExecutionShapeDerivation> executionShapeDerivations;
  std::vector<StructuredSpecialMathAccuracyDerivation>
      specialMathAccuracyDerivations;
  std::vector<StructuredScheduleDerivation> scheduleDerivations;
  std::vector<StructuredMemoryCommunicationDerivation>
      memoryCommunicationDerivations;
  if (candidate != *impl_->sourceReference) {
    auto lineage = impl_->resolveLineage(candidate);
    if (!lineage)
      return lineage.takeError();
    derivations = std::move(lineage->ownership);
    executionShapeDerivations = std::move(lineage->executionShape);
    specialMathAccuracyDerivations = std::move(lineage->specialMathAccuracy);
    scheduleDerivations = std::move(lineage->schedule);
    memoryCommunicationDerivations = std::move(lineage->memoryCommunication);
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
      std::move(*materialized),
      std::move(derivations),
      std::move(executionShapeDerivations),
      std::move(specialMathAccuracyDerivations),
      std::move(scheduleDerivations),
      std::move(memoryCommunicationDerivations),
      {},
      std::move(functionalReplay)};
}

llvm::Expected<ArtifactRootReference>
StructuredOwnershipInvocation::prepareDataflowGeneration(
    const ArtifactRootReference &structuredParent, const ArtifactStore &store) {
  auto found = impl_->materialized.find(structuredParent);
  if (found == impl_->materialized.end())
    return invalid("Dataflow generation requires a promoted Structured parent");
  auto published = dataflow::publishCanonicalDataflow(
      found->second.canonicalDataflow, store);
  if (!published)
    return published.takeError();
  if (llvm::Error error =
          impl_->dataflowLineage.recordRoot(structuredParent, *published))
    return std::move(error);
  return *published;
}

llvm::Expected<SelectedStructuredOwnershipCandidate>
StructuredOwnershipInvocation::materializeSelectedDataflowCandidate(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &dataflowCandidate,
    const ArtifactStore &store) {
  if (!impl_->workloadReference || !impl_->runtimeInputReference)
    return invalid("Dataflow selection precedes source preparation");
  auto dataflowDerivations =
      impl_->resolveDataflowLineage(structuredParent, dataflowCandidate);
  if (!dataflowDerivations)
    return dataflowDerivations.takeError();
  auto replay = evaluation::models::getPrimedCanonicalDataflowFunctionalReplay(
      dataflowCandidate, structuredParent, *impl_->workloadReference,
      *impl_->runtimeInputReference);
  if (!replay)
    return replay.takeError();
  if (replay->status != sim::SourceBackedDfgValidationStatus::Equivalent)
    return invalid("selected Dataflow candidate lacks equivalent replay");

  auto found = impl_->materialized.find(structuredParent);
  if (found == impl_->materialized.end())
    return invalid("selected Dataflow candidate lost its Structured parent");
  auto selectedStructured =
      frontend::importStructuredProgram(structuredParent, store);
  if (!selectedStructured)
    return selectedStructured.takeError();
  auto selectedDataflow =
      dataflow::importCanonicalDataflow(dataflowCandidate, store);
  if (!selectedDataflow)
    return selectedDataflow.takeError();
  frontend::MaterializedOwnershipCandidate materialized{
      std::move(*selectedStructured), std::move(*selectedDataflow),
      found->second.spatialGraphs, found->second.ownedSpatialRegion,
      found->second.blockActivityLineage, found->second.sourceProvenance};

  auto lineage = impl_->resolveLineage(structuredParent);
  if (!lineage)
    return lineage.takeError();
  return SelectedStructuredOwnershipCandidate{
      std::move(materialized),
      std::move(lineage->ownership),
      std::move(lineage->executionShape),
      std::move(lineage->specialMathAccuracy),
      std::move(lineage->schedule),
      std::move(lineage->memoryCommunication),
      std::move(*dataflowDerivations),
      std::move(*replay)};
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
  if (!impl.structuredCandidates.empty())
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
  for (const StructuredOwnershipCandidateDisposition &disposition :
       dispositions) {
    const auto *child = std::get_if<ArtifactRootReference>(&disposition.result);
    if (!child || *child == sourceReference)
      continue;
    if (!disposition.coordinate.decision)
      return invalid("Ownership child has no typed decision");
    if (uniqueCandidates.find(*child) == uniqueCandidates.end())
      return invalid("Ownership lineage target has no candidate state");
    StructuredOwnershipDerivation derivation{disposition.coordinate.scope,
                                             *disposition.coordinate.decision};
    auto payload = frontend::encodeSpatialOwnershipDecision(
        frontend::SpatialOwnershipDecision{derivation.scope,
                                           derivation.decision});
    if (!payload)
      return payload.takeError();
    auto [found, inserted] = impl.ownershipLineage[*child].try_emplace(
        CanonicalDerivationKey{sourceReference, std::move(*payload)},
        derivation);
    if (!inserted && !(found->second == derivation))
      return invalid("Ownership lineage key has conflicting decisions");
  }
  for (auto [reference, index] : uniqueCandidates)
    if (impl.structuredCandidates
            .try_emplace(reference, std::move(candidates[index].candidate))
            .second)
      impl.structuredReachable.insert(reference);
  impl.generationRecorded = true;
  return llvm::Error::success();
}

llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
detail::StructuredOwnershipInvocationAccess::clonePreClosureCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &reference) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded)
    return invalid("ExecutionShape generation precedes Ownership generation");
  auto found = impl.structuredCandidates.find(reference);
  if (found == impl.structuredCandidates.end())
    return invalid("pre-closure input has no Structured candidate state");
  auto clone = frontend::importStructuredProgram(
      found->second.structuredProgram.identity(),
      found->second.structuredProgram.canonicalBytes());
  if (!clone)
    return clone.takeError();
  return frontend::MaterializedStructuredOwnershipCandidate{
      std::move(*clone), found->second.ownedSpatialRegion,
      found->second.blockActivityLineage,
      found->second.sourceProvenance};
}

llvm::Expected<std::optional<frontend::StructuredEntityRef>>
detail::StructuredOwnershipInvocationAccess::ownedSpatialRegion(
    const StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &reference) {
  const StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded)
    return invalid("Spatial projection lookup precedes Ownership generation");
  auto final = impl.finalCandidates.find(reference);
  if (final != impl.finalCandidates.end())
    return final->second.ownedSpatialRegion;
  auto structured = impl.structuredCandidates.find(reference);
  if (structured != impl.structuredCandidates.end())
    return structured->second.ownedSpatialRegion;
  auto materialized = impl.materialized.find(reference);
  if (materialized != impl.materialized.end())
    return materialized->second.ownedSpatialRegion;
  return invalid("Structured candidate has no bound Spatial projection");
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
  if (impl.structuredReachable.find(parent) == impl.structuredReachable.end())
    return invalid("Schedule parent has no Ownership lineage");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          candidate.structuredProgram.canonicalBytes().bytes()))
    return invalid("Schedule child differs from its published bytes");

  auto payload = frontend::encodeStructuredScheduleDecision(decision);
  if (!payload)
    return payload.takeError();
  StructuredScheduleDerivation derivation{parent, decision};
  auto [lineage, inserted] = impl.scheduleLineage[child].try_emplace(
      CanonicalDerivationKey{parent, std::move(*payload)}, derivation);
  if (!inserted && !(lineage->second == derivation))
    return invalid("Schedule lineage key has conflicting decisions");
  impl.structuredReachable.insert(child);

  auto found = impl.finalCandidates.find(child);
  if (found == impl.finalCandidates.end()) {
    impl.finalCandidates.try_emplace(
        child, StructuredOwnershipInvocation::Impl::FinalCandidateState{
                   std::move(candidate.structuredProgram),
                   std::move(candidate.trackedSpatialRegion),
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
detail::StructuredOwnershipInvocationAccess::recordMemoryCommunicationCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    const frontend::StructuredMemoryCommunicationDecision &decision,
    frontend::MaterializedStructuredMemoryCommunicationCandidate candidate,
    lowering::ProjectedCanonicalDataflow projected,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded || !impl.sourceReference)
    return invalid(
        "MemoryCommunication generation precedes Ownership generation");
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
        "MemoryCommunication lineage contains a foreign Structured reference");
  if (parent == child)
    return llvm::Error::success();
  if (impl.structuredReachable.find(parent) == impl.structuredReachable.end())
    return invalid("MemoryCommunication parent has no Ownership lineage");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          candidate.structuredProgram.canonicalBytes().bytes()))
    return invalid(
        "MemoryCommunication child differs from its published bytes");

  auto payload =
      frontend::encodeStructuredMemoryCommunicationDecision(decision);
  if (!payload)
    return payload.takeError();
  StructuredMemoryCommunicationDerivation derivation{parent, decision};
  auto [lineage, inserted] = impl.memoryCommunicationLineage[child].try_emplace(
      CanonicalDerivationKey{parent, std::move(*payload)}, derivation);
  if (!inserted && !(lineage->second == derivation))
    return invalid("MemoryCommunication lineage key has conflicting decisions");
  impl.structuredReachable.insert(child);

  auto found = impl.finalCandidates.find(child);
  if (found == impl.finalCandidates.end()) {
    impl.finalCandidates.try_emplace(
        child, StructuredOwnershipInvocation::Impl::FinalCandidateState{
                   std::move(candidate.structuredProgram),
                   std::move(candidate.trackedSpatialRegion),
                   {},
                   std::move(candidate.sourceProvenance),
                   std::move(projected)});
  } else if (found->second.structuredProgram.canonicalBytes().bytes() !=
             candidate.structuredProgram.canonicalBytes().bytes()) {
    return invalid(
        "deduplicated MemoryCommunication child changed canonical bytes");
  }
  return llvm::Error::success();
}

llvm::Error
detail::StructuredOwnershipInvocationAccess::recordExecutionShapeCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    std::optional<frontend::StructuredExecutionShapeDecision> decision,
    frontend::MaterializedStructuredOwnershipCandidate candidate,
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
  if (impl.structuredReachable.find(parent) == impl.structuredReachable.end())
    return invalid("ExecutionShape parent has no Ownership lineage");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          candidate.structuredProgram.canonicalBytes().bytes()))
    return invalid("ExecutionShape child differs from its published bytes");

  if (decision && parent != child) {
    auto payload = frontend::encodeStructuredExecutionShapeDecision(*decision);
    if (!payload)
      return payload.takeError();
    StructuredExecutionShapeDerivation derivation{parent, *decision};
    auto [lineage, inserted] = impl.executionShapeLineage[child].try_emplace(
        CanonicalDerivationKey{parent, std::move(*payload)}, derivation);
    if (!inserted && !(lineage->second == derivation))
      return invalid("ExecutionShape lineage key has conflicting decisions");
  } else if (decision || parent != child) {
    return invalid("ExecutionShape pass-through has inconsistent lineage");
  }
  impl.structuredReachable.insert(child);

  auto found = impl.structuredCandidates.find(child);
  if (found == impl.structuredCandidates.end()) {
    impl.structuredCandidates.try_emplace(child, std::move(candidate));
  } else if (found->second.structuredProgram.canonicalBytes().bytes() !=
             candidate.structuredProgram.canonicalBytes().bytes()) {
    return invalid("deduplicated ExecutionShape child changed canonical bytes");
  }
  return llvm::Error::success();
}

llvm::Error detail::StructuredOwnershipInvocationAccess::
    recordSpecialMathAccuracyDerivation(
        StructuredOwnershipInvocation &invocation,
        const ArtifactRootReference &parent, const ArtifactRootReference &child,
        const frontend::StructuredSpecialMathAccuracyDecision &decision,
        const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded || !impl.sourceReference)
    return invalid(
        "SpecialMathAccuracy generation precedes Ownership generation");
  if (parent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      child.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      parent == child)
    return invalid(
        "SpecialMathAccuracy lineage contains invalid Structured references");
  if (impl.structuredReachable.find(parent) == impl.structuredReachable.end())
    return invalid("SpecialMathAccuracy parent has no Structured lineage");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();

  auto payload =
      frontend::encodeStructuredSpecialMathAccuracyDecision(decision);
  if (!payload)
    return payload.takeError();
  StructuredSpecialMathAccuracyDerivation derivation{parent, decision};
  auto [lineage, inserted] = impl.specialMathAccuracyLineage[child].try_emplace(
      CanonicalDerivationKey{parent, std::move(*payload)}, derivation);
  if (!inserted && !(lineage->second == derivation))
    return invalid("SpecialMathAccuracy lineage key has conflicting decisions");
  impl.structuredReachable.insert(child);
  return llvm::Error::success();
}

llvm::Error detail::StructuredOwnershipInvocationAccess::
    recordSpecialMathAccuracyMechanicalCandidate(
        StructuredOwnershipInvocation &invocation,
        const ArtifactRootReference &parent, const ArtifactRootReference &child,
        const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded || !impl.sourceReference)
    return invalid(
        "SpecialMathAccuracy generation precedes Ownership generation");
  if (parent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      child.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      parent == child)
    return invalid(
        "SpecialMathAccuracy mechanical lineage contains invalid Structured "
        "references");
  if (impl.structuredReachable.find(parent) == impl.structuredReachable.end())
    return invalid(
        "SpecialMathAccuracy mechanical parent has no Structured lineage");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  auto [position, inserted] =
      impl.specialMathMechanicalParents.try_emplace(child, parent);
  if (!inserted && position->second != parent)
    return invalid(
        "SpecialMathAccuracy mechanical child has conflicting parents");
  impl.structuredReachable.insert(child);
  return llvm::Error::success();
}

llvm::Error detail::StructuredOwnershipInvocationAccess::
    recordSpecialMathAccuracyFinalCandidate(
        StructuredOwnershipInvocation &invocation,
        const ArtifactRootReference &child,
        frontend::MaterializedOwnershipCandidate candidate,
        const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded || !impl.sourceReference)
    return invalid(
        "SpecialMathAccuracy finalization precedes Ownership generation");
  if (child.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      child.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      child.artifact != candidate.structuredProgram.identity())
    return invalid(
        "SpecialMathAccuracy final candidate has a foreign Structured root");
  if (impl.structuredReachable.find(child) == impl.structuredReachable.end())
    return invalid(
        "SpecialMathAccuracy final candidate has no Structured lineage");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          candidate.structuredProgram.canonicalBytes().bytes()))
    return invalid(
        "SpecialMathAccuracy final candidate differs from published bytes");

  lowering::ProjectedCanonicalDataflow projected{
      std::move(candidate.canonicalDataflow),
      std::move(candidate.spatialGraphs)};
  auto found = impl.finalCandidates.find(child);
  if (found == impl.finalCandidates.end()) {
    impl.finalCandidates.try_emplace(
        child,
        StructuredOwnershipInvocation::Impl::FinalCandidateState{
            std::move(candidate.structuredProgram),
            std::move(candidate.ownedSpatialRegion),
            std::move(candidate.blockActivityLineage),
            std::move(candidate.sourceProvenance), std::move(projected)});
  } else if (found->second.structuredProgram.canonicalBytes().bytes() !=
             candidate.structuredProgram.canonicalBytes().bytes()) {
    return invalid(
        "deduplicated SpecialMathAccuracy candidate changed canonical bytes");
  }
  return llvm::Error::success();
}

llvm::Error
detail::StructuredOwnershipInvocationAccess::recordDataflowRewriteCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    const dataflow::DataflowRewriteDecision &decision,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (impl.dataflowLineage.empty())
    return invalid("Dataflow rewrite precedes D0 preparation");
  if (parent.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      parent.schemaVersion != dataflow::canonicalDataflowSchema.version ||
      child.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      child.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("Dataflow rewrite lineage contains a foreign reference");
  auto stored = store.get(child);
  if (!stored)
    return stored.takeError();
  return impl.dataflowLineage.recordDecision(parent, child, decision);
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
  if (impl.primedCandidates.find(candidate) != impl.primedCandidates.end())
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
        std::move(state.ownedSpatialRegion),
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
  impl.primedCandidates.insert(candidate);
  return llvm::Error::success();
}

llvm::Error
detail::StructuredOwnershipInvocationAccess::primeDataflowFunctionalReplay(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &dataflowCandidate,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.sourceReference || !impl.workloadReference ||
      !impl.runtimeInputReference || !impl.sourceObservations)
    return invalid(
        "Dataflow functional acquisition precedes source preparation");
  const auto key = std::make_pair(structuredParent, dataflowCandidate);
  if (impl.primedDataflowCandidates.find(key) !=
      impl.primedDataflowCandidates.end())
    return llvm::Error::success();

  auto dataflowLineage =
      impl.resolveDataflowLineage(structuredParent, dataflowCandidate);
  if (!dataflowLineage)
    return dataflowLineage.takeError();

  auto d0 = impl.dataflowLineage.root(structuredParent);
  if (!d0)
    return d0.takeError();
  if (dataflowCandidate == *d0) {
    auto structuredReplay =
        evaluation::models::getPrimedStructuredProgramFunctionalReplay(
            structuredParent, *impl.workloadReference,
            *impl.runtimeInputReference);
    if (!structuredReplay)
      return structuredReplay.takeError();
    if (llvm::Error error =
            evaluation::models::primeCanonicalDataflowFunctionalReplayResult(
                dataflowCandidate, structuredParent, *impl.workloadReference,
                *impl.runtimeInputReference, *structuredReplay))
      return error;
    impl.primedDataflowCandidates.insert(key);
    return llvm::Error::success();
  }

  auto parentState = impl.materialized.find(structuredParent);
  if (parentState == impl.materialized.end())
    return invalid("Dataflow candidate has no selected Structured parent");
  auto lineage = impl.resolveLineage(structuredParent);
  if (!lineage)
    return lineage.takeError();
  auto structured = frontend::importStructuredProgram(structuredParent, store);
  if (!structured)
    return structured.takeError();
  auto dataflow = dataflow::importCanonicalDataflow(dataflowCandidate, store);
  if (!dataflow)
    return dataflow.takeError();

  frontend::MaterializedOwnershipCandidate candidate{
      std::move(*structured), std::move(*dataflow),
      parentState->second.spatialGraphs,
      parentState->second.ownedSpatialRegion,
      parentState->second.blockActivityLineage,
      parentState->second.sourceProvenance};
  std::vector<frontend::StructuredExecutionShapeDecision>
      executionShapeDecisions;
  executionShapeDecisions.reserve(lineage->executionShape.size());
  for (const StructuredExecutionShapeDerivation &derivation :
       lineage->executionShape)
    executionShapeDecisions.push_back(derivation.decision);
  for (const StructuredOwnershipDerivation &derivation : lineage->ownership) {
    if (llvm::Error error =
            evaluation::models::primeCanonicalDataflowFunctionalReplay(
                dataflowCandidate, structuredParent,
                {*impl.workloadReference, *impl.runtimeInputReference,
                 impl.sourceProgram, derivation.scope, derivation.decision,
                 executionShapeDecisions, candidate, impl.workload,
                 impl.runtimeInput, *impl.sourceObservations,
                 impl.functionalReplayLimits},
                store))
      return error;
  }
  impl.primedDataflowCandidates.insert(key);
  return llvm::Error::success();
}

} // namespace loom::dse
