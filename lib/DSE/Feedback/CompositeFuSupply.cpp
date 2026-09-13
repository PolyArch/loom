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

llvm::Expected<std::optional<MinedCompositeFuProposal>>
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
  if (graphs.empty())
    return std::optional<MinedCompositeFuProposal>();

  const auto started = std::chrono::steady_clock::now();
  auto mined = mineCompositeFuCandidates(program->view(), graphs,
                                         productionCompositeFuMiningLimits);
  if (!mined)
    return mined.takeError();
  const auto searchMilliseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started)
          .count());

  for (const CompositeFuCandidate &candidate : mined->candidates) {
    if (candidate.nodes.size() < 2)
      continue;
    if (candidate.inputs.size() > loom::adg::builtinPeInputPortCount ||
        candidate.outputs.size() > loom::adg::builtinPeOutputPortCount)
      continue;
    // The canonical capability derivation is the admission owner. A candidate
    // it rejects is one the generator could not rebuild either, so the
    // selection skips it here instead of naming a shape that would fail.
    auto request = deriveCompositeFuTemplate(program->view(), candidate);
    if (!request) {
      llvm::consumeError(request.takeError());
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
    proposal.bounded = mined->bounded;
    proposal.exploredActorCount = mined->exploredActorCount;
    proposal.searchMilliseconds = searchMilliseconds;
    return std::optional<MinedCompositeFuProposal>(std::move(proposal));
  }
  return std::optional<MinedCompositeFuProposal>();
}

} // namespace loom::dse
