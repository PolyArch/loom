#include "DSE/StructuredOwnershipInvocation.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Evaluation/Models/CanonicalDataflowFunctional.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/NativeSimulationOracle.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
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

class EvaluationTimer final {
public:
  explicit EvaluationTimer(std::uint64_t &elapsed)
      : elapsed_(elapsed), start_(std::chrono::steady_clock::now()) {}

  ~EvaluationTimer() {
    const auto duration = std::chrono::steady_clock::now() - start_;
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           duration)
                           .count();
    if (nanos <= 0)
      return;
    const auto delta = static_cast<std::uint64_t>(nanos);
    if (elapsed_ > std::numeric_limits<std::uint64_t>::max() - delta)
      elapsed_ = std::numeric_limits<std::uint64_t>::max();
    else
      elapsed_ += delta;
  }

private:
  std::uint64_t &elapsed_;
  std::chrono::steady_clock::time_point start_;
};

bool sameRoot(const ArtifactRootReference &reference,
              const ArtifactSchemaDescriptor &schema,
              const ArtifactIdentity &identity) {
  return reference.schemaIdentity == schema.identity &&
         reference.schemaVersion == schema.version &&
         reference.artifact == identity;
}

/// Returns the logical-domain fact already materialized in a Structured
/// candidate, or nullopt when the candidate has not crossed the Dataflow
/// boundary yet.  Compiler providers may query this fact between semantic
/// generators; at that point special-math (and other Dataflow-facing choices)
/// can still be intentionally unresolved, so canonical lowering is not a
/// valid way to answer a structural thread-domain question.
std::optional<bool> inspectMaterializedLogicalThreadDomain(
    const frontend::StructuredProgramCandidate &candidate) {
  bool sawThread = false;
  bool validShape = true;
  bool logical = false;
  candidate.module().walk([&](dataflow::ThreadOp thread) {
    sawThread = true;
    if (thread.isExternal())
      return;
    mlir::Block &entry = thread.getBody().front();
    const std::size_t inputCount = thread.getFunctionType().getNumInputs();
    if (entry.getNumArguments() < inputCount + 1) {
      validShape = false;
      return;
    }
    if (thread.getDomain().getKind() == dataflow::ThreadDomainKind::DynamicWork) {
      if (entry.getNumArguments() != inputCount + 1)
        validShape = false;
      return;
    }
    for (std::size_t ordinal = inputCount + 1;
         ordinal < entry.getNumArguments(); ++ordinal)
      if (!llvm::isa<mlir::IndexType>(entry.getArgument(ordinal).getType()))
        validShape = false;
    logical |= entry.getNumArguments() > inputCount + 1;
  });
  if (!sawThread || !validShape)
    return std::nullopt;
  return logical;
}

sim::SourceBackedDfgValidationLimits boundedReplayLimits(
    sim::SourceBackedDfgValidationLimits limits,
    ExecutionControlView executionControl) {
  if (executionControl.stopRequested()) {
    limits.maxSimulationWallTime =
        std::chrono::steady_clock::duration::zero();
    return limits;
  }
  if (auto remaining = executionControl.remainingTime())
    limits.maxSimulationWallTime =
        std::min(limits.maxSimulationWallTime,
                 std::max(*remaining,
                          std::chrono::steady_clock::duration::zero()));
  return limits;
}

llvm::Error requireProjectedDataflowOwner(
    const lowering::ProjectedCanonicalDataflow &projected,
    llvm::StringRef producer) {
  for (const lowering::StructuredSpatialGraphProjection &projection :
       projected.spatialGraphs)
    if (projection.staticGraphLaunch.artifact != projected.artifact.identity())
      return invalid(producer +
                     " produced a foreign static graph launch projection");
  return llvm::Error::success();
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

struct StaticGraphLaunchLineage final {
  dataflow::StaticGraphLaunchRef parent;
  dataflow::StaticGraphLaunchRef child;

  bool operator==(const StaticGraphLaunchLineage &other) const {
    return parent == other.parent && child == other.child;
  }
};

struct DataflowRewriteLineageEdge final {
  DataflowRewriteDerivation derivation;
  std::vector<StaticGraphLaunchLineage> staticGraphLaunches;

  bool operator==(const DataflowRewriteLineageEdge &other) const {
    return derivation == other.derivation &&
           staticGraphLaunches == other.staticGraphLaunches;
  }
};

} // namespace

llvm::Expected<std::shared_ptr<const sim::NativeStructuredProgramObservations>>
StructuredOwnershipSharedEvaluation::profiledObservations(
    const ArtifactRootReference &candidate, const ArtifactRootReference &source,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const frontend::StructuredProgramCandidate &candidateProgram,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &simulationWorkload,
    const sim::CanonicalSimulationRuntimeInput &simulationRuntimeInput) const {
  if (!sameRoot(candidate, frontend::structuredProgramArtifactSchema,
                candidateProgram.identity()) ||
      !sameRoot(source, frontend::structuredProgramArtifactSchema,
                sourceProgram.identity()) ||
      !sameRoot(workload, sim::simulationWorkloadSchema,
                simulationWorkload.identity()) ||
      !sameRoot(runtimeInput, sim::simulationRuntimeInputSchema,
                simulationRuntimeInput.identity()))
    return invalid("profile cache key differs from its exact inputs");
  const ProfileKey key{candidate, source, workload, runtimeInput};
  {
    std::unique_lock<std::mutex> lock(profileMutex_);
    while (true) {
      auto found = profiles_.find(key);
      if (found == profiles_.end()) {
        profiles_.emplace(key, ProfileEntry{});
        ++statistics_.profileCacheMisses;
        break;
      }
      if (!found->second.inFlight) {
        ++statistics_.profileCacheHits;
        return found->second.observations;
      }
      ++statistics_.profileSingleFlightWaits;
      profileChanged_.wait(lock, [&] {
        auto current = profiles_.find(key);
        return current == profiles_.end() || !current->second.inFlight;
      });
    }
  }

  auto executed = sim::executeProfiledSelectedStructuredProgram(
      candidateProgram, sourceProgram, simulationWorkload,
      simulationRuntimeInput);
  if (!executed) {
    llvm::Error error = executed.takeError();
    {
      std::lock_guard<std::mutex> lock(profileMutex_);
      profiles_.erase(key);
    }
    profileChanged_.notify_all();
    return std::move(error);
  }
  auto observations =
      std::make_shared<const sim::NativeStructuredProgramObservations>(
          std::move(*executed));
  bool invalidState = false;
  {
    std::lock_guard<std::mutex> lock(profileMutex_);
    auto found = profiles_.find(key);
    if (found == profiles_.end() || !found->second.inFlight) {
      profiles_.erase(key);
      invalidState = true;
    } else {
      found->second.observations = observations;
      found->second.inFlight = false;
    }
  }
  profileChanged_.notify_all();
  if (invalidState)
    return invalid("profile single-flight state changed during execution");
  return observations;
}

StructuredOwnershipSharedEvaluationStatistics
StructuredOwnershipSharedEvaluation::statistics() const {
  std::lock_guard<std::mutex> lock(profileMutex_);
  return statistics_;
}

class detail::StructuredOwnershipDataflowLineageIndex::Impl final {
public:
  std::map<ArtifactRootReference, ArtifactRootReference,
           decltype(&artifactRootReferenceLess)>
      roots{&artifactRootReferenceLess};
  CanonicalDerivationIndex<DataflowRewriteLineageEdge> lineage;
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
    const dataflow::DataflowRewriteDecision &decision,
    llvm::ArrayRef<dataflow::StaticGraphLaunchRef> parentLaunches,
    llvm::ArrayRef<dataflow::StaticGraphLaunchRef> childLaunches) {
  if (parent.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      parent.schemaVersion != dataflow::canonicalDataflowSchema.version ||
      child.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      child.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("Dataflow rewrite lineage contains a foreign reference");
  if (impl_->nodes.find(parent) == impl_->nodes.end())
    return invalid("Dataflow rewrite parent is outside the prepared lineage");
  if (parent == child)
    return invalid("Dataflow rewrite cannot derive a candidate from itself");
  if (parentLaunches.size() != childLaunches.size())
    return invalid("Dataflow rewrite changed tracked launch cardinality");
  std::vector<StaticGraphLaunchLineage> launches;
  launches.reserve(parentLaunches.size());
  for (auto [parentLaunch, childLaunch] :
       llvm::zip_equal(parentLaunches, childLaunches)) {
    if (parentLaunch.artifact != parent.artifact ||
        childLaunch.artifact != child.artifact)
      return invalid("Dataflow rewrite launch lineage has a foreign owner");
    launches.push_back({parentLaunch, childLaunch});
  }
  auto payload = dataflow::encodeDataflowRewriteDecision(decision);
  if (!payload)
    return payload.takeError();
  DataflowRewriteDerivation derivation{parent, child, decision};
  DataflowRewriteLineageEdge edge{derivation, std::move(launches)};
  auto [lineage, inserted] = impl_->lineage[child].try_emplace(
      CanonicalDerivationKey{parent, std::move(*payload)}, edge);
  if (!inserted && !(lineage->second == edge))
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
      const DataflowRewriteDerivation &edge = entry.second.derivation;
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

llvm::Expected<dataflow::StaticGraphLaunchRef>
detail::StructuredOwnershipDataflowLineageIndex::projectStaticGraphLaunch(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &candidate,
    dataflow::StaticGraphLaunchRef rootLaunch) const {
  auto root = impl_->roots.find(structuredParent);
  if (root == impl_->roots.end())
    return invalid("Structured parent has no Canonical Dataflow root");
  if (rootLaunch.artifact != root->second.artifact)
    return invalid("Dataflow root launch has a foreign artifact owner");

  ArtifactReferenceSet visiting;
  std::function<llvm::Expected<std::optional<dataflow::StaticGraphLaunchRef>>(
      const ArtifactRootReference &)>
      visit = [&](const ArtifactRootReference &reference)
      -> llvm::Expected<std::optional<dataflow::StaticGraphLaunchRef>> {
    if (reference == root->second)
      return std::optional<dataflow::StaticGraphLaunchRef>(rootLaunch);
    if (!visiting.insert(reference).second)
      return invalid("Dataflow launch lineage contains a cycle");
    auto edges = impl_->lineage.find(reference);
    if (edges == impl_->lineage.end()) {
      visiting.erase(reference);
      return std::optional<dataflow::StaticGraphLaunchRef>{};
    }

    std::optional<dataflow::StaticGraphLaunchRef> projected;
    for (const auto &entry : edges->second) {
      const DataflowRewriteLineageEdge &edge = entry.second;
      auto parentLaunch = visit(edge.derivation.parent);
      if (!parentLaunch)
        return parentLaunch.takeError();
      if (!*parentLaunch)
        continue;
      auto mapped = llvm::find_if(edge.staticGraphLaunches,
                                  [&](const StaticGraphLaunchLineage &lineage) {
                                    return lineage.parent == **parentLaunch;
                                  });
      if (mapped == edge.staticGraphLaunches.end())
        return invalid("Dataflow rewrite omitted a tracked graph launch");
      if (projected && *projected != mapped->child)
        return invalid("Dataflow rewrite paths disagree on graph launch");
      projected = mapped->child;
    }
    visiting.erase(reference);
    return projected;
  };

  auto projected = visit(candidate);
  if (!projected)
    return projected.takeError();
  if (!*projected)
    return invalid("Dataflow candidate does not descend from its exact D0");
  if ((*projected)->artifact != candidate.artifact)
    return invalid("Dataflow projected launch has the wrong child owner");
  return **projected;
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

  Impl(const frontend::StructuredProgramCandidate &generationParent,
       const frontend::StructuredProgramCandidate &sourceProgram,
       const sim::CanonicalSimulationWorkload &workload,
       const sim::CanonicalSimulationRuntimeInput &runtimeInput,
       const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
       const lowering::CanonicalDataflowLoweringOptions &lowering,
       std::uint32_t candidateWorkerCount,
       sim::SourceBackedDfgValidationLimits functionalReplayLimits,
       llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
           sourceProvenance,
       const StructuredOwnershipSharedEvaluation *sharedEvaluation,
       ExecutionControlView executionControl,
       bool generationParentFunctionallyVerified)
      : generationParent(generationParent), sourceProgram(sourceProgram),
        workload(workload), runtimeInput(runtimeInput), fabric(fabric),
        config(config), lowering(lowering),
        candidateWorkerCount(candidateWorkerCount),
        functionalReplayLimits(functionalReplayLimits),
        sourceProvenance(sourceProvenance.begin(), sourceProvenance.end()),
        sharedEvaluation(sharedEvaluation), executionControl(executionControl),
        generationParentFunctionallyVerified(
            generationParentFunctionallyVerified),
        structuredCandidates(&artifactRootReferenceLess),
        materialized(&artifactRootReferenceLess),
        specialMathMechanicalParents(&artifactRootReferenceLess),
        finalCandidates(&artifactRootReferenceLess),
        selectedFunctionalReplays(&artifactRootReferenceLess),
        logicalThreadDomainFacts(&artifactRootReferenceLess) {}

  const frontend::StructuredProgramCandidate &generationParent;
  const frontend::StructuredProgramCandidate &sourceProgram;
  const sim::CanonicalSimulationWorkload &workload;
  const sim::CanonicalSimulationRuntimeInput &runtimeInput;
  const fabric::FinalizedFabricRoot &fabric;
  const ResolvedConfig &config;
  lowering::CanonicalDataflowLoweringOptions lowering;
  std::uint32_t candidateWorkerCount;
  sim::SourceBackedDfgValidationLimits functionalReplayLimits;
  std::vector<frontend::StructuredOperationSourceProvenance> sourceProvenance;
  const StructuredOwnershipSharedEvaluation *sharedEvaluation = nullptr;
  ExecutionControlView executionControl;
  bool generationParentFunctionallyVerified = true;
  evaluation::models::StructuredEvaluationInvocationCache localEvaluationCache;

  evaluation::models::StructuredEvaluationInvocationCache &evaluationCache() {
    return sharedEvaluation ? sharedEvaluation->cache() : localEvaluationCache;
  }

  std::optional<ArtifactRootReference> generationParentReference;
  std::optional<ArtifactRootReference> sourceReference;
  std::optional<ArtifactRootReference> workloadReference;
  std::optional<ArtifactRootReference> runtimeInputReference;
  std::optional<sim::NativeStructuredProgramObservations>
      ownedSourceObservations;
  const sim::NativeStructuredProgramObservations *sourceObservations = nullptr;
  std::optional<sim::NativeStructuredProgramObservations>
      generationParentObservations;
  std::uint64_t sourceNativeExecutionCount = 0;
  StructuredOwnershipEvaluationTiming evaluationTiming;
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
  std::map<ArtifactRootReference, sim::SourceBackedDfgValidationResult,
           decltype(&artifactRootReferenceLess)>
      selectedFunctionalReplays;
  std::map<ArtifactRootReference, bool, decltype(&artifactRootReferenceLess)>
      logicalThreadDomainFacts;
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
    const frontend::StructuredProgramCandidate &generationParent,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ResolvedConfig &config,
    const lowering::CanonicalDataflowLoweringOptions &lowering,
    std::uint32_t candidateWorkerCount,
    sim::SourceBackedDfgValidationLimits functionalReplayLimits,
    llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
        sourceProvenance,
    const StructuredOwnershipSharedEvaluation *sharedEvaluation,
    ExecutionControlView executionControl,
    bool generationParentFunctionallyVerified)
    : impl_(std::make_unique<Impl>(generationParent, sourceProgram, workload,
                                   runtimeInput, fabric, config, lowering,
                                   candidateWorkerCount, functionalReplayLimits,
                                   sourceProvenance, sharedEvaluation,
                                   executionControl,
                                   generationParentFunctionallyVerified)) {}

StructuredOwnershipInvocation::~StructuredOwnershipInvocation() = default;

llvm::Error StructuredOwnershipInvocation::prepareInputs(
    const ArtifactRootReference &generationParent,
    const ArtifactRootReference &source, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ArtifactStore &store) {
  Impl &impl = *impl_;
  if (impl.generationParentReference) {
    if (!impl.sourceReference ||
        *impl.generationParentReference != generationParent ||
        *impl.sourceReference != source || !impl.generationParentObservations ||
        !impl.workloadReference || !impl.runtimeInputReference ||
        !impl.sourceObservations || *impl.workloadReference != workload ||
        *impl.runtimeInputReference != runtimeInput)
      return invalid("prepared input roots differ from the invocation");
    return llvm::Error::success();
  }
  if (impl.sourceReference || impl.generationParentObservations ||
      impl.workloadReference || impl.runtimeInputReference ||
      impl.sourceObservations)
    return invalid("input preparation is partially initialized");
  if (!sameRoot(generationParent, frontend::structuredProgramArtifactSchema,
                impl.generationParent.identity()) ||
      !sameRoot(source, frontend::structuredProgramArtifactSchema,
                impl.sourceProgram.identity()) ||
      !sameRoot(workload, sim::simulationWorkloadSchema,
                impl.workload.identity()) ||
      !sameRoot(runtimeInput, sim::simulationRuntimeInputSchema,
                impl.runtimeInput.identity()))
    return invalid("prepared roots differ from the bound invocation");

  std::optional<sim::NativeStructuredProgramObservations> observations;
  const sim::NativeStructuredProgramObservations *sourceObservations = nullptr;
  if (impl.sharedEvaluation) {
    sourceObservations = &impl.sharedEvaluation->sourceObservations();
  } else {
    auto executed = sim::executeNativeStructuredProgram(
        impl.sourceProgram, impl.workload, impl.runtimeInput);
    if (!executed)
      return executed.takeError();
    observations.emplace(std::move(*executed));
    sourceObservations = &*observations;
  }
  std::optional<sim::NativeStructuredProgramObservations> parentObservations;
  if (impl.generationParent.identity() == impl.sourceProgram.identity()) {
    parentObservations = *sourceObservations;
  } else if (impl.sharedEvaluation) {
    auto profiled = impl.sharedEvaluation->profiledObservations(
        generationParent, source, workload, runtimeInput, impl.generationParent,
        impl.sourceProgram, impl.workload, impl.runtimeInput);
    if (!profiled)
      return profiled.takeError();
    parentObservations = **profiled;
  } else {
    auto profiled = sim::executeProfiledSelectedStructuredProgram(
        impl.generationParent, impl.sourceProgram, impl.workload,
        impl.runtimeInput);
    if (!profiled)
      return profiled.takeError();
    parentObservations.emplace(std::move(*profiled));
  }
  evaluation::models::StructuredEvaluationInvocationCacheScope cacheScope(
      impl.evaluationCache());
  if (llvm::Error error =
          evaluation::models::primeStructuredProgramSourceObservations(
              source, workload, runtimeInput, *sourceObservations))
    return error;
  const evaluation::models::StructuredFabricAnalyticInvocation invocation{
      workload,          runtimeInput,       impl.workload,
      impl.runtimeInput, impl.sourceProgram, *sourceObservations};
  if (llvm::Error error =
          evaluation::models::primeStructuredFabricAnalyticResult(
              source, {impl.sourceProgram, nullptr, {}, {}, sourceObservations},
              invocation, impl.fabric, impl.config, store))
    return error;

  impl.generationParentReference.emplace(generationParent);
  impl.sourceReference.emplace(source);
  impl.workloadReference.emplace(workload);
  impl.runtimeInputReference.emplace(runtimeInput);
  if (observations) {
    impl.ownedSourceObservations.emplace(std::move(*observations));
    impl.sourceObservations = &*impl.ownedSourceObservations;
    ++impl.sourceNativeExecutionCount;
  } else {
    impl.sourceObservations = sourceObservations;
  }
  impl.generationParentObservations.emplace(std::move(*parentObservations));
  return llvm::Error::success();
}

std::uint64_t
StructuredOwnershipInvocation::sourceNativeExecutionCount() const {
  return impl_->sourceNativeExecutionCount;
}

evaluation::models::StructuredEvaluationInvocationCacheStatistics
StructuredOwnershipInvocation::evaluationCacheStatistics() const {
  return impl_->evaluationCache().statistics();
}

StructuredOwnershipEvaluationTiming
StructuredOwnershipInvocation::evaluationTiming() const {
  return impl_->evaluationTiming;
}

llvm::Error
StructuredOwnershipInvocation::ensureSelectedCandidateFunctionalReplay(
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  if (!impl_->generationParentReference || !impl_->workloadReference ||
      !impl_->runtimeInputReference)
    return invalid("functional replay precedes candidate generation");
  if (candidate == *impl_->generationParentReference &&
      impl_->generationParentFunctionallyVerified)
    return llvm::Error::success();
  if (impl_->selectedFunctionalReplays.find(candidate) !=
      impl_->selectedFunctionalReplays.end())
    return llvm::Error::success();
  auto replay = evaluation::models::getPrimedStructuredProgramFunctionalReplay(
      candidate, *impl_->workloadReference, *impl_->runtimeInputReference);
  if (!replay) {
    llvm::consumeError(replay.takeError());
    if (llvm::Error error =
            detail::StructuredOwnershipInvocationAccess::primeFunctionalReplay(
                *this, candidate, store))
      return error;
    replay =
        evaluation::models::getPrimedStructuredProgramFunctionalReplay(
            candidate, *impl_->workloadReference,
            *impl_->runtimeInputReference);
    if (!replay)
      return replay.takeError();
  }
  impl_->selectedFunctionalReplays.try_emplace(candidate,
                                               std::move(*replay));
  return llvm::Error::success();
}

llvm::Expected<bool>
StructuredOwnershipInvocation::selectedCandidateHasLogicalThreadDomain(
    const ArtifactRootReference &candidate) const {
  if (!impl_->generationParentReference)
    return invalid("candidate generation has not completed");
  if (auto found = impl_->logicalThreadDomainFacts.find(candidate);
      found != impl_->logicalThreadDomainFacts.end())
    return found->second;
  const auto remember = [&](bool value) {
    impl_->logicalThreadDomainFacts.try_emplace(candidate, value);
    return value;
  };
  if (candidate == *impl_->generationParentReference) {
    if (std::optional<bool> structural =
            inspectMaterializedLogicalThreadDomain(impl_->generationParent))
      return remember(*structural);
    auto projected = lowering::lowerStructuredProgramToCanonicalDataflow(
        impl_->generationParent, impl_->lowering);
    if (!projected)
      return projected.takeError();
    auto view = projected->view();
    if (!view)
      return view.takeError();
    for (const dataflow::CanonicalRootThreadLaunchView &launch :
         view->rootThreadLaunches()) {
      auto domain = view->projectRootThreadLogicalDomain(launch.ref);
      if (!domain)
        return domain.takeError();
      if (domain->coordinateRank != 0)
        return remember(true);
    }
    return remember(false);
  }

  std::optional<dataflow::CanonicalDataflowArtifact *> dataflow;
  auto materialized = impl_->materialized.find(candidate);
  if (materialized != impl_->materialized.end())
    dataflow = &materialized->second.canonicalDataflow;
  auto final = impl_->finalCandidates.find(candidate);
  if (!dataflow && final != impl_->finalCandidates.end())
    dataflow = &final->second.projected.artifact;
  if (!dataflow) {
    auto structured = impl_->structuredCandidates.find(candidate);
    if (structured != impl_->structuredCandidates.end()) {
      if (std::optional<bool> structural =
              inspectMaterializedLogicalThreadDomain(
                  structured->second.structuredProgram))
        return remember(*structural);
      auto lowered = lowering::lowerStructuredProgramToCanonicalDataflow(
          structured->second.structuredProgram, impl_->lowering);
      if (!lowered)
        return lowered.takeError();
      // Keep the temporary artifact alive through the projection below. This
      // path is only used to classify an existing candidate; replay ownership
      // remains with the promotion provider.
      auto view = lowered->view();
      if (!view)
        return view.takeError();
      for (const dataflow::CanonicalRootThreadLaunchView &launch :
           view->rootThreadLaunches()) {
        auto domain = view->projectRootThreadLogicalDomain(launch.ref);
        if (!domain)
          return domain.takeError();
        if (domain->coordinateRank != 0)
          return remember(true);
      }
      return remember(false);
    }
    return invalid(
        "selected candidate has no retained Structured/Dataflow state: candidate=" +
        llvm::toHex(encodeArtifactRootReference(candidate)) +
        ", generation_parent=" +
        llvm::toHex(encodeArtifactRootReference(*impl_->generationParentReference)));
  }
  auto view = (*dataflow)->view();
  if (!view)
    return view.takeError();
  for (const dataflow::CanonicalRootThreadLaunchView &launch :
       view->rootThreadLaunches()) {
    auto domain = view->projectRootThreadLogicalDomain(launch.ref);
    if (!domain)
      return domain.takeError();
    if (domain->coordinateRank != 0)
      return remember(true);
  }
  return remember(false);
}

llvm::ArrayRef<StructuredOwnershipCandidateDisposition>
StructuredOwnershipInvocation::dispositions() const {
  return impl_->dispositions;
}

llvm::Expected<SelectedStructuredOwnershipCandidate>
StructuredOwnershipInvocation::materializeSelectedCandidate(
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  return materializeCandidate(candidate, store, true);
}

llvm::Expected<SelectedStructuredOwnershipCandidate>
StructuredOwnershipInvocation::materializeAnalyticContinuationCandidate(
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  return materializeCandidate(candidate, store, false);
}

llvm::Expected<SelectedStructuredOwnershipCandidate>
StructuredOwnershipInvocation::materializeCandidate(
    const ArtifactRootReference &candidate, const ArtifactStore &store,
    bool requireFunctionalReplay) {
  if (!impl_->generationParentReference || !impl_->sourceReference ||
      !impl_->workloadReference || !impl_->runtimeInputReference)
    return invalid("candidate generation has not completed");
  if (requireFunctionalReplay &&
      (candidate != *impl_->generationParentReference ||
       !impl_->generationParentFunctionallyVerified))
    if (llvm::Error error =
            ensureSelectedCandidateFunctionalReplay(candidate, store))
      return std::move(error);

  std::optional<frontend::MaterializedOwnershipCandidate> materialized;
  if (candidate == *impl_->generationParentReference) {
    auto structured = frontend::importStructuredProgram(candidate, store);
    if (!structured)
      return structured.takeError();
    auto projected =
        lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
            *structured, impl_->lowering);
    if (!projected)
      return projected.takeError();
    materialized.emplace(
        frontend::MaterializedOwnershipCandidate{std::move(*structured),
                                                 std::move(projected->artifact),
                                                 std::move(projected->spatialGraphs),
                                                 std::nullopt,
                                                 {},
                                                 impl_->sourceProvenance});
  } else {
    if (requireFunctionalReplay) {
      auto found = impl_->materialized.find(candidate);
      if (found != impl_->materialized.end()) {
        auto structured = frontend::importStructuredProgram(candidate, store);
        if (!structured)
          return structured.takeError();
        auto dataflow = dataflow::importCanonicalDataflow(
            found->second.canonicalDataflow.identity(),
            found->second.canonicalDataflow.canonicalBytes());
        if (!dataflow)
          return dataflow.takeError();
        materialized.emplace(frontend::MaterializedOwnershipCandidate{
            std::move(*structured), std::move(*dataflow),
            found->second.spatialGraphs, found->second.ownedSpatialRegion,
            found->second.blockActivityLineage,
            found->second.sourceProvenance});
      } else {
        // An exact functional Evidence cache hit may satisfy Promote without
        // calling this invocation's priming callback. Reconstruct the local
        // selected view from the same finalized candidate state, then require
        // the cached replay below before returning it as terminal.
        auto finalized = impl_->finalCandidates.find(candidate);
        if (finalized == impl_->finalCandidates.end())
          return invalid(
              "selected candidate has neither local nor finalized functional "
              "projection");
        auto state = std::move(finalized->second);
        impl_->finalCandidates.erase(finalized);
        materialized.emplace(frontend::MaterializedOwnershipCandidate{
            std::move(state.structuredProgram),
            std::move(state.projected.artifact),
            std::move(state.projected.spatialGraphs),
            std::move(state.ownedSpatialRegion),
            std::move(state.blockActivityLineage),
            std::move(state.sourceProvenance)});
      }
    } else {
      auto found = impl_->finalCandidates.find(candidate);
      if (found == impl_->finalCandidates.end())
        return invalid(
            "analytic continuation candidate has no finalized projection");
      auto state = std::move(found->second);
      impl_->finalCandidates.erase(found);
      materialized.emplace(frontend::MaterializedOwnershipCandidate{
          std::move(state.structuredProgram),
          std::move(state.projected.artifact),
          std::move(state.projected.spatialGraphs),
          std::move(state.ownedSpatialRegion),
          std::move(state.blockActivityLineage),
          std::move(state.sourceProvenance)});
    }
  }

  std::vector<StructuredOwnershipDerivation> derivations;
  std::vector<StructuredExecutionShapeDerivation> executionShapeDerivations;
  std::vector<StructuredSpecialMathAccuracyDerivation>
      specialMathAccuracyDerivations;
  std::vector<StructuredScheduleDerivation> scheduleDerivations;
  std::vector<StructuredMemoryCommunicationDerivation>
      memoryCommunicationDerivations;
  if (candidate != *impl_->generationParentReference) {
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
  if (requireFunctionalReplay &&
      (candidate != *impl_->generationParentReference ||
       !impl_->generationParentFunctionallyVerified)) {
    auto replay = impl_->selectedFunctionalReplays.find(candidate);
    if (replay == impl_->selectedFunctionalReplays.end())
      return invalid("terminal candidate lost its retained functional replay");
    if (replay->second.status !=
        sim::SourceBackedDfgValidationStatus::Equivalent)
      return invalid("selected accelerator candidate lacks equivalent replay");
    functionalReplay.emplace(replay->second);
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
      std::move(*selectedStructured),     std::move(*selectedDataflow),
      found->second.spatialGraphs,        found->second.ownedSpatialRegion,
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
    const frontend::StructuredProgramCandidate &generationParent,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const fabric::FinalizedFabricRoot &fabric, const ArtifactStore &store,
    StructuredOwnershipGenerationOptions &options) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (impl.candidateWorkerCount == 0)
    return invalid("candidate worker count must be positive");
  if (generationParent.identity() != impl.generationParent.identity() ||
      sourceProgram.identity() != impl.sourceProgram.identity() ||
      workload.identity() != impl.workload.identity() ||
      runtimeInput.identity() != impl.runtimeInput.identity() ||
      fabric.reference() != impl.fabric.reference())
    return invalid("Generate inputs differ from the bound invocation");
  options.lowering = impl.lowering;
  options.candidateWorkerCount = impl.candidateWorkerCount;
  if (!impl.generationParentReference) {
    auto generationParentReference =
        frontend::publishStructuredProgram(impl.generationParent, store);
    if (!generationParentReference)
      return generationParentReference.takeError();
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
    if (llvm::Error error = invocation.prepareInputs(
            *generationParentReference, *sourceReference, *workloadReference,
            *runtimeInputReference, store))
      return error;
  }
  return llvm::Error::success();
}

llvm::Expected<detail::StructuredOwnershipPreparedSource>
detail::StructuredOwnershipInvocationAccess::preparedSource(
    const StructuredOwnershipInvocation &invocation) {
  const StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationParentReference || !impl.workloadReference ||
      !impl.runtimeInputReference || !impl.generationParentObservations)
    return invalid("input preparation has not completed");
  return StructuredOwnershipPreparedSource{
      *impl.generationParentReference, *impl.workloadReference,
      *impl.runtimeInputReference, *impl.generationParentObservations};
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
  return invocation.impl_->evaluationCache();
}

llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
detail::StructuredOwnershipInvocationAccess::sourceProvenance(
    const StructuredOwnershipInvocation &invocation) {
  return invocation.impl_->sourceProvenance;
}

llvm::Error detail::StructuredOwnershipInvocationAccess::recordGeneration(
    StructuredOwnershipInvocation &invocation,
    ArtifactRootReference generationParentReference,
    ArtifactRootReference workloadReference,
    ArtifactRootReference runtimeInputReference,
    llvm::ArrayRef<StructuredOwnershipCandidateDisposition> dispositions,
    std::vector<StructuredOwnershipCandidateState> candidates,
    const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (impl.generationRecorded)
    return invalid("invocation contains more than one Ownership generation");
  if (!impl.generationParentReference || !impl.sourceReference ||
      !impl.workloadReference || !impl.runtimeInputReference ||
      !impl.sourceObservations || !impl.generationParentObservations ||
      *impl.generationParentReference != generationParentReference ||
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
    if (!child || *child == generationParentReference)
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
        CanonicalDerivationKey{generationParentReference, std::move(*payload)},
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
      found->second.blockActivityLineage, found->second.sourceProvenance};
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

llvm::Expected<llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>>
detail::StructuredOwnershipInvocationAccess::sourceProvenance(
    const StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &reference) {
  const StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  if (!impl.generationRecorded)
    return invalid("source provenance lookup precedes Ownership generation");
  auto final = impl.finalCandidates.find(reference);
  if (final != impl.finalCandidates.end())
    return llvm::ArrayRef(final->second.sourceProvenance);
  auto structured = impl.structuredCandidates.find(reference);
  if (structured != impl.structuredCandidates.end())
    return llvm::ArrayRef(structured->second.sourceProvenance);
  auto materialized = impl.materialized.find(reference);
  if (materialized != impl.materialized.end())
    return llvm::ArrayRef(materialized->second.sourceProvenance);
  return invalid("Structured candidate has no source provenance state");
}

llvm::Error
detail::StructuredOwnershipInvocationAccess::recordScheduleCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    const frontend::StructuredScheduleDecision &decision,
    frontend::MaterializedStructuredScheduleCandidate candidate,
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

  auto found = impl.structuredCandidates.find(child);
  if (found == impl.structuredCandidates.end()) {
    impl.structuredCandidates.try_emplace(
        child, frontend::MaterializedStructuredOwnershipCandidate{
                   std::move(candidate.structuredProgram),
                   std::move(candidate.trackedSpatialRegion),
                   {},
                   std::move(candidate.sourceProvenance)});
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

  auto found = impl.structuredCandidates.find(child);
  if (found == impl.structuredCandidates.end()) {
    impl.structuredCandidates.try_emplace(
        child, frontend::MaterializedStructuredOwnershipCandidate{
                   std::move(candidate.structuredProgram),
                   std::move(candidate.trackedSpatialRegion),
                   {},
                   std::move(candidate.sourceProvenance)});
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
  if (llvm::Error error =
          requireProjectedDataflowOwner(projected, "SpecialMathAccuracy"))
    return error;
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
    llvm::ArrayRef<dataflow::StaticGraphLaunchRef> parentLaunches,
    llvm::ArrayRef<dataflow::StaticGraphLaunchRef> childLaunches,
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
  return impl.dataflowLineage.recordDecision(parent, child, decision,
                                             parentLaunches, childLaunches);
}

llvm::Error detail::StructuredOwnershipInvocationAccess::primeAnalyticCandidate(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  ++impl.evaluationTiming.analyticCalls;
  EvaluationTimer timer(impl.evaluationTiming.analyticElapsedNanoseconds);
  auto found = impl.finalCandidates.find(candidate);
  if (found == impl.finalCandidates.end())
    return llvm::Error::success();
  auto cached = evaluation::models::hasStructuredFabricAnalyticResult(
      candidate, impl.fabric.reference(), *impl.workloadReference,
      *impl.runtimeInputReference, impl.config, impl.evaluationCache());
  if (!cached)
    return cached.takeError();
  if (*cached)
    return llvm::Error::success();
  auto &state = found->second;
  std::shared_ptr<const sim::NativeStructuredProgramObservations>
      sharedObservations;
  std::optional<sim::NativeStructuredProgramObservations> ownedObservations;
  const sim::NativeStructuredProgramObservations *observations = nullptr;
  if (impl.sharedEvaluation) {
    auto profiled = impl.sharedEvaluation->profiledObservations(
        candidate, *impl.sourceReference, *impl.workloadReference,
        *impl.runtimeInputReference, state.structuredProgram,
        impl.sourceProgram, impl.workload, impl.runtimeInput);
    if (!profiled) {
      llvm::consumeError(profiled.takeError());
      return llvm::Error::success();
    }
    sharedObservations = std::move(*profiled);
    observations = sharedObservations.get();
  } else {
    auto profiled = sim::executeProfiledSelectedStructuredProgram(
        state.structuredProgram, impl.sourceProgram, impl.workload,
        impl.runtimeInput);
    if (!profiled) {
      llvm::consumeError(profiled.takeError());
      return llvm::Error::success();
    }
    ownedObservations.emplace(std::move(*profiled));
    observations = &*ownedObservations;
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
       observations},
      analytic, impl.fabric, impl.config, store);
}

llvm::Error detail::StructuredOwnershipInvocationAccess::primeFunctionalReplay(
    StructuredOwnershipInvocation &invocation,
    const ArtifactRootReference &candidate, const ArtifactStore &store) {
  StructuredOwnershipInvocation::Impl &impl = *invocation.impl_;
  ++impl.evaluationTiming.functionalReplayCalls;
  EvaluationTimer timer(
      impl.evaluationTiming.functionalReplayElapsedNanoseconds);
  if (!impl.generationParentReference || !impl.sourceReference ||
      !impl.workloadReference || !impl.runtimeInputReference ||
      !impl.sourceObservations)
    return invalid("functional acquisition precedes Ownership generation");
  const bool isGenerationParent =
      candidate == *impl.generationParentReference;
  if (isGenerationParent && impl.generationParentFunctionallyVerified)
    return llvm::Error::success();
  if (impl.primedCandidates.find(candidate) != impl.primedCandidates.end()) {
    auto replay =
        evaluation::models::getPrimedStructuredProgramFunctionalReplay(
            candidate, *impl.workloadReference, *impl.runtimeInputReference);
    if (replay)
      return llvm::Error::success();
    llvm::consumeError(replay.takeError());
    impl.primedCandidates.erase(candidate);
  }

  std::optional<StructuredOwnershipInvocation::Impl::ResolvedLineage> lineage;
  if (!isGenerationParent) {
    auto resolved = impl.resolveLineage(candidate);
    if (!resolved)
      return resolved.takeError();
    lineage.emplace(std::move(*resolved));
  }

  auto materialized = impl.materialized.find(candidate);
  if (materialized == impl.materialized.end()) {
    std::optional<frontend::MaterializedOwnershipCandidate> reconstructed;
    auto scheduled = impl.finalCandidates.find(candidate);
    if (scheduled != impl.finalCandidates.end()) {
      auto state = std::move(scheduled->second);
      impl.finalCandidates.erase(scheduled);
      reconstructed.emplace(frontend::MaterializedOwnershipCandidate{
          std::move(state.structuredProgram),
          std::move(state.projected.artifact),
          std::move(state.projected.spatialGraphs),
          std::move(state.ownedSpatialRegion),
          std::move(state.blockActivityLineage),
          std::move(state.sourceProvenance)});
    } else if (isGenerationParent) {
      auto structured = frontend::importStructuredProgram(candidate, store);
      if (!structured)
        return structured.takeError();
      auto projected =
          lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
              *structured, impl.lowering);
      if (!projected)
        return projected.takeError();
      reconstructed.emplace(frontend::MaterializedOwnershipCandidate{
          std::move(*structured), std::move(projected->artifact),
          std::move(projected->spatialGraphs), std::nullopt, {},
          impl.sourceProvenance});
    } else {
      const StructuredOwnershipDerivation &derivation =
          lineage->ownership.front();
      auto direct = frontend::materializeSpatialOwnershipDecision(
          impl.generationParent, derivation.scope, derivation.decision,
          impl.fabric, impl.lowering, impl.sourceProvenance);
      if (!direct)
        return direct.takeError();
      reconstructed.emplace(std::move(*direct));
    }
    auto inserted =
        impl.materialized.try_emplace(candidate, std::move(*reconstructed));
    if (!inserted.second)
      return invalid("functional candidate materialization raced its owner");
    materialized = inserted.first;
  }
  for (const lowering::StructuredSpatialGraphProjection &projection :
       materialized->second.spatialGraphs)
    if (projection.staticGraphLaunch.artifact !=
        materialized->second.canonicalDataflow.identity())
      return invalid("functional candidate has a foreign graph projection");
  if (materialized->second.structuredProgram.identity() != candidate.artifact)
    return invalid("rematerialized candidate changed identity");
  auto stored = store.get(candidate);
  if (!stored)
    return stored.takeError();
  if (!stored->bytes().equals(
          materialized->second.structuredProgram.canonicalBytes().bytes()))
    return invalid("rematerialized candidate changed canonical bytes");
  if (llvm::Error error =
          evaluation::models::primeStructuredProgramFunctionalReplay(
              candidate,
              {*impl.workloadReference, *impl.runtimeInputReference,
               impl.sourceProgram, materialized->second, impl.workload,
               impl.runtimeInput, *impl.sourceObservations,
               boundedReplayLimits(impl.functionalReplayLimits,
                                   impl.executionControl)},
              store))
    return error;
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
  ++impl.evaluationTiming.functionalReplayCalls;
  EvaluationTimer timer(
      impl.evaluationTiming.functionalReplayElapsedNanoseconds);
  if (!impl.generationParentReference || !impl.sourceReference ||
      !impl.workloadReference || !impl.runtimeInputReference ||
      !impl.sourceObservations)
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
  auto structured = frontend::importStructuredProgram(structuredParent, store);
  if (!structured)
    return structured.takeError();
  auto dataflow = dataflow::importCanonicalDataflow(dataflowCandidate, store);
  if (!dataflow)
    return dataflow.takeError();

  std::vector<lowering::StructuredSpatialGraphProjection> spatialGraphs =
      parentState->second.spatialGraphs;
  for (lowering::StructuredSpatialGraphProjection &projection : spatialGraphs) {
    auto projected = impl.dataflowLineage.projectStaticGraphLaunch(
        structuredParent, dataflowCandidate, projection.staticGraphLaunch);
    if (!projected)
      return projected.takeError();
    projection.staticGraphLaunch = *projected;
  }

  frontend::MaterializedOwnershipCandidate candidate{
      std::move(*structured),
      std::move(*dataflow),
      std::move(spatialGraphs),
      parentState->second.ownedSpatialRegion,
      parentState->second.blockActivityLineage,
      parentState->second.sourceProvenance};
  if (llvm::Error error =
          evaluation::models::primeCanonicalDataflowFunctionalReplay(
              dataflowCandidate, structuredParent,
              {*impl.workloadReference, *impl.runtimeInputReference,
               impl.sourceProgram, candidate, impl.workload, impl.runtimeInput,
               *impl.sourceObservations,
               boundedReplayLimits(impl.functionalReplayLimits,
                                   impl.executionControl)},
              store))
    return error;
  impl.primedDataflowCandidates.insert(key);
  return llvm::Error::success();
}

} // namespace loom::dse
