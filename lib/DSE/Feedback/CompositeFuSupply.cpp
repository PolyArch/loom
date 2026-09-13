//===- CompositeFuSupply.cpp - mined supply for an actor demand ----------===//
//
// Two observations ask for fewer realizations over the same actors. A
// compute-context Hall deficit asks from the cover side, and a verified
// parent whose mapped replay trails its own dataflow oracle asks from the
// measured side. This file owns the one answer both spend: composing a new
// capability from the common subgraph of the software they name. It selects a
// mined template and sizes its occurrences; it authors no hardware and
// describes no FU structure.
//
//===----------------------------------------------------------------------===//

#include "DSE/CompositeFuSupply.h"

#include "DSE/CompositeFuMining.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "composite_fu_supply_invalid: " + message);
}

/// Spatial PE sites one demand needs when each realization of the proposal
/// answers `actorsPerRealization` of it. Sites are bounded by the target's
/// Spatial PE count, because the recipe places at most one occurrence per site.
std::uint32_t sizeOccurrences(std::uint64_t demand,
                              std::uint64_t actorsPerRealization,
                              std::uint32_t spatialPeCount) {
  if (spatialPeCount == 0 || actorsPerRealization == 0)
    return 0;
  const std::uint64_t needed =
      (demand + actorsPerRealization - 1) / actorsPerRealization;
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      std::max<std::uint64_t>(needed, 1), spatialPeCount));
}

} // namespace

llvm::Expected<CanonicalDataflowActorCensus>
censusCanonicalDataflowActors(const ArtifactRootReference &dataflow,
                              const ArtifactStore &store) {
  auto program = ::dataflow::importCanonicalDataflow(dataflow, store);
  if (!program)
    return program.takeError();
  CanonicalDataflowActorCensus census;
  for (const ::dataflow::CanonicalActorView &actor : program->view().actors()) {
    switch (actor.kind) {
    case ::dataflow::CanonicalDataflowActorKind::Compute:
      ++census.computeActors;
      break;
    case ::dataflow::CanonicalDataflowActorKind::Control:
      ++census.controlActors;
      break;
    case ::dataflow::CanonicalDataflowActorKind::Memory:
      ++census.memoryActors;
      break;
    }
  }
  return census;
}

llvm::Expected<MinedCompositeFuSupplyOutcome>
proposeMinedCompositeFuSupply(const ArtifactRootReference &dataflow,
                              std::uint64_t actorDemand,
                              const loom::adg::BuiltinTargetScale &scale,
                              const ArtifactStore &store) {
  if (actorDemand == 0)
    return invalid("composed supply requires a positive actor demand");
  auto program = ::dataflow::importCanonicalDataflow(dataflow, store);
  if (!program)
    return program.takeError();
  std::vector<::dataflow::GraphRef> graphs;
  for (const ::dataflow::CanonicalGraphView &graph : program->view().graphs())
    graphs.push_back(graph.ref);
  MinedCompositeFuSupplyOutcome outcome;
  if (graphs.empty())
    return outcome;

  const auto started = std::chrono::steady_clock::now();
  auto mined = mineCompositeFuCandidates(program->view(), graphs,
                                         productionCompositeFuMiningLimits);
  if (!mined)
    return mined.takeError();
  outcome.searchMilliseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
  outcome.minedCandidateCount = mined->candidates.size();
  outcome.bounded = mined->bounded;
  outcome.exploredActorCount = mined->exploredActorCount;

  for (const CompositeFuCandidate &candidate : mined->candidates) {
    if (candidate.inputs.size() > loom::adg::builtinPeInputPortCount ||
        candidate.outputs.size() > loom::adg::builtinPeOutputPortCount) {
      ++outcome.boundaryRefusedCount;
      continue;
    }
    // The canonical capability derivation is the admission owner. A candidate
    // it rejects is one the generator could not rebuild either, so the
    // selection skips it here instead of naming a shape that would fail. Its
    // reason names the operation family whose inverse policy is missing, which
    // is a Fabric owner's gap rather than a missing opportunity.
    auto request = deriveCompositeFuTemplate(program->view(), candidate);
    if (!request) {
      ++outcome.capabilityRefusedCount;
      const std::string reason = llvm::toString(request.takeError());
      if (outcome.firstCapabilityRefusal.empty())
        outcome.firstCapabilityRefusal = reason;
      continue;
    }
    const std::uint32_t occurrences = sizeOccurrences(
        actorDemand, candidate.nodes.size(), scale.spatialPeCount);
    if (occurrences == 0)
      continue;
    MinedCompositeFuProposal proposal{
        MinedCompositeFuSelection{dataflow.artifact, {}}, occurrences};
    proposal.selection.templates.push_back(
        {candidate.canonicalKey, occurrences});
    proposal.actorsPerRealization = candidate.nodes.size();
    proposal.support = candidate.support;
    outcome.proposal = std::move(proposal);
    return outcome;
  }
  return outcome;
}

void describeMinedCompositeFuSupply(
    llvm::json::Object &fields, const MinedCompositeFuSupplyOutcome &supply) {
  fields["mined_candidate_count"] = supply.minedCandidateCount;
  fields["mined_boundary_refused_count"] = supply.boundaryRefusedCount;
  fields["mined_capability_refused_count"] = supply.capabilityRefusedCount;
  fields["mined_first_capability_refusal"] =
      supply.firstCapabilityRefusal.empty()
          ? llvm::json::Value(nullptr)
          : llvm::json::Value(supply.firstCapabilityRefusal);
  fields["mined_search_bounded"] = supply.bounded;
  fields["mined_search_explored_actor_count"] = supply.exploredActorCount;
  fields["mined_search_milliseconds"] = supply.searchMilliseconds;
  if (!supply.proposal) {
    fields["mined_actors_per_realization"] = nullptr;
    fields["mined_support"] = nullptr;
    fields["mined_occurrences"] = nullptr;
    return;
  }
  fields["mined_actors_per_realization"] = supply.proposal->actorsPerRealization;
  fields["mined_support"] = supply.proposal->support;
  fields["mined_occurrences"] = supply.proposal->occurrences;
}

} // namespace loom::dse
