#include "HandshakeProjectionTestSupport.h"
#include "../TestAllocationProbe.h"

#include "PnR/HandshakeCandidateState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "handshake projection test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

std::pair<std::size_t, std::size_t> expectedProjectionExtent(
    const loom::pnr::FrozenSpatialHandshakeIndex &handshake,
    llvm::ArrayRef<loom::pnr::PnrIndex> selectedFragments) {
  std::vector<std::uint8_t> activeArcs(handshake.projectionArcs().size(), 0);
  const auto activateFragment = [&](loom::pnr::PnrIndex fragment) {
    const auto offsets = handshake.projectionFragmentArcOffsets();
    if (fragment >= handshake.fragments().size())
      fail("projection extent fragment is out of range");
    for (loom::pnr::PnrIndex arc : handshake.projectionFragmentArcs().slice(
             offsets[fragment], offsets[fragment + 1] - offsets[fragment]))
      activeArcs[arc] = 1;
  };
  for (loom::pnr::PnrIndex arc : handshake.projectionFixedArcs())
    activeArcs[arc] = 1;
  for (loom::pnr::PnrIndex fragment : handshake.fixedFragments())
    activateFragment(fragment);
  for (loom::pnr::PnrIndex fragment : selectedFragments)
    activateFragment(fragment);

  std::vector<std::uint8_t> activeNodes(handshake.projectionNodeCount(), 0);
  std::size_t arcCount = 0;
  for (auto [ordinal, arc] : llvm::enumerate(handshake.projectionArcs())) {
    if (!activeArcs[ordinal])
      continue;
    ++arcCount;
    activeNodes[arc.source] = 1;
    activeNodes[arc.destination] = 1;
  }
  return {static_cast<std::size_t>(std::count(
              activeNodes.begin(), activeNodes.end(), std::uint8_t{1})),
          arcCount};
}

void verifyFixedArcProjection(
    const loom::pnr::FrozenSpatialHandshakeIndex &handshake) {
  if (handshake.projectionFixedArcs().empty())
    fail("handshake fixture has no Fabric-unconditional projection arc");

  const std::vector<loom::pnr::PnrIndex> emptyFragments;
  const std::vector<loom::pnr::PnrIndex> traversalUses(
      handshake.traversalFragmentOffsets().size() - 1, 0);
  loom::pnr::HandshakeProjectionScratch scratch;
  requireSuccess(scratch.prepare(handshake));
  const bool cold =
      take(loom::pnr::independentlyVerifyHandshakeProjectionAcyclic(
          handshake, emptyFragments, traversalUses));
  const bool hot =
      take(scratch.projectAcyclic(handshake, emptyFragments, traversalUses));
  const auto extent = expectedProjectionExtent(handshake, emptyFragments);
  const auto statistics = scratch.statistics();
  if (!hot || hot != cold || extent.second == 0 ||
      statistics.peakActiveNodeCount != extent.first ||
      statistics.peakActiveArcCount != extent.second)
    fail("dense always-active handshake projection disagrees with its cold "
         "oracle");
}

} // namespace

void loom::test::exerciseDenseHandshakeProjection(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  const pnr::FrozenSpatialHandshakeIndex &handshake = problem->handshake();
  const auto placementOffsets = handshake.computePlacementFragmentOffsets();
  if (placementOffsets.size() < 2)
    fail("handshake fixture has no compute placement fragment domain");
  const auto fragments = handshake.computePlacementFragments().slice(
      placementOffsets.front(), placementOffsets[1] - placementOffsets.front());
  const std::vector<pnr::PnrIndex> traversalUses(
      handshake.traversalFragmentOffsets().size() - 1, 0);

  pnr::HandshakeProjectionScratch projectionScratch;
  requireSuccess(projectionScratch.prepare(handshake));
  const bool coldProjection =
      take(pnr::independentlyVerifyHandshakeProjectionAcyclic(
          handshake, fragments, traversalUses));
  const bool activeProjection = take(
      projectionScratch.projectAcyclic(handshake, fragments, traversalUses));
  if (activeProjection != coldProjection)
    fail("dense handshake projection disagrees with its cold oracle");
  const auto firstStatistics = projectionScratch.statistics();
  const std::size_t warmedBytes = projectionScratch.retainedStorageBytes();
  const bool repeatedProjection = take(
      projectionScratch.projectAcyclic(handshake, fragments, traversalUses));
  const auto repeatedStatistics = projectionScratch.statistics();
  if (repeatedProjection != coldProjection ||
      repeatedStatistics.projectionCount != 2 ||
      repeatedStatistics.deterministicWork -
              firstStatistics.deterministicWork !=
          firstStatistics.deterministicWork ||
      repeatedStatistics.peakActiveNodeCount == 0 ||
      repeatedStatistics.peakActiveArcCount == 0 ||
      projectionScratch.retainedStorageBytes() != warmedBytes)
    fail("reused dense handshake projection changed its exact result, work, "
         "or storage");

  const auto beforeProbe = projectionScratch.statistics();
  startAllocationProbe();
  const bool probedProjection = take(
      projectionScratch.projectAcyclic(handshake, fragments, traversalUses));
  const std::size_t allocations = stopAllocationProbe();
  const auto afterProbe = projectionScratch.statistics();
  if (probedProjection != coldProjection ||
      afterProbe.deterministicWork - beforeProbe.deterministicWork !=
          firstStatistics.deterministicWork ||
      projectionScratch.retainedStorageBytes() != warmedBytes)
    fail("warmed dense handshake projection changed its deterministic work or "
         "storage");
  if (afterProbe.coldVerificationCount == beforeProbe.coldVerificationCount &&
      allocations != 0)
    fail("warmed dense handshake projection allocated heap storage");

  if (handshake.projectionFixedArcs().empty())
    return;

  std::vector<pnr::PnrIndex> firstContributor(handshake.projectionArcs().size(),
                                              pnr::getInvalidPnrIndex());
  std::optional<std::pair<pnr::PnrIndex, pnr::PnrIndex>> duplicateFragments;
  const auto fragmentOffsets = handshake.projectionFragmentArcOffsets();
  for (std::size_t fragment = 0; fragment < handshake.fragments().size();
       ++fragment) {
    for (pnr::PnrIndex arc : handshake.projectionFragmentArcs().slice(
             fragmentOffsets[fragment],
             fragmentOffsets[fragment + 1] - fragmentOffsets[fragment])) {
      if (firstContributor[arc] != pnr::getInvalidPnrIndex() &&
          firstContributor[arc] != fragment) {
        duplicateFragments = {firstContributor[arc],
                              static_cast<pnr::PnrIndex>(fragment)};
        break;
      }
      firstContributor[arc] = static_cast<pnr::PnrIndex>(fragment);
    }
    if (duplicateFragments)
      break;
  }
  if (!duplicateFragments)
    fail("handshake fixture has no duplicate projection arc contributor");
  const std::array<pnr::PnrIndex, 2> duplicateContributors{
      duplicateFragments->first, duplicateFragments->second};
  pnr::HandshakeProjectionScratch duplicateScratch;
  requireSuccess(duplicateScratch.prepare(handshake));
  const bool coldDuplicate =
      take(pnr::independentlyVerifyHandshakeProjectionAcyclic(
          handshake, duplicateContributors, traversalUses));
  const bool hotDuplicate = take(duplicateScratch.projectAcyclic(
      handshake, duplicateContributors, traversalUses));
  const auto duplicateExtent =
      expectedProjectionExtent(handshake, duplicateContributors);
  const auto duplicateStatistics = duplicateScratch.statistics();
  if (hotDuplicate != coldDuplicate ||
      duplicateStatistics.peakActiveNodeCount != duplicateExtent.first ||
      duplicateStatistics.peakActiveArcCount != duplicateExtent.second)
    fail("duplicate handshake arc contributors changed dense projection");
}

void loom::test::exerciseDenseHandshakeFixedArcProjection(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  verifyFixedArcProjection(problem->handshake());
  exerciseDenseHandshakeProjection(problem);
}

void loom::test::exerciseDenseHandshakeCycleProjection(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  const pnr::FrozenSpatialHandshakeIndex &handshake = problem->handshake();
  std::optional<pnr::PnrIndex> bypassTraversal;
  std::optional<pnr::PnrIndex> feedbackTraversal;
  for (auto [ordinal, traversal] :
       llvm::enumerate(problem->routing().traversals())) {
    if (const auto *fifo =
            std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
                &traversal.reference.payload)) {
      if (fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Bypass)
        bypassTraversal = static_cast<pnr::PnrIndex>(ordinal);
      continue;
    }
    const auto *crosspoint =
        std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
            &traversal.reference.payload);
    if (crosspoint && crosspoint->input == 1 && crosspoint->output == 0)
      feedbackTraversal = static_cast<pnr::PnrIndex>(ordinal);
  }
  if (!bypassTraversal || !feedbackTraversal)
    fail("feedback fixture has no FIFO bypass or switch backedge traversal");

  std::vector<pnr::PnrIndex> selectedFragments;
  bool selectedActivation = false;
  for (const pnr::FrozenSpatialSwitchHandshakeActivation &activation :
       handshake.switchActivations()) {
    if (activation.input != 1)
      continue;
    for (const pnr::FrozenSpatialSwitchHandshakeTraversalSelection &selection :
         handshake.switchTraversalSelections().slice(
             activation.traversalSelectionOffset,
             activation.traversalSelectionCount)) {
      if (selection.traversal != *feedbackTraversal)
        continue;
      llvm::append_range(
          selectedFragments,
          handshake.switchActivationBaseFragments().slice(
              activation.baseFragmentOffset, activation.baseFragmentCount));
      llvm::append_range(
          selectedFragments,
          handshake.switchTraversalFragments().slice(selection.fragmentOffset,
                                                     selection.fragmentCount));
      selectedActivation = true;
      break;
    }
    if (selectedActivation)
      break;
  }
  if (!selectedActivation)
    fail("feedback traversal has no exact Temporal switch activation");

  std::vector<pnr::PnrIndex> traversalUses(
      handshake.traversalFragmentOffsets().size() - 1, 0);
  traversalUses[*bypassTraversal] = 1;
  traversalUses[*feedbackTraversal] = 1;
  pnr::HandshakeProjectionScratch scratch;
  requireSuccess(scratch.prepare(handshake));
  const bool cold = take(pnr::independentlyVerifyHandshakeProjectionAcyclic(
      handshake, selectedFragments, traversalUses));
  const bool hot =
      take(scratch.projectAcyclic(handshake, selectedFragments, traversalUses));
  if (hot != cold || hot)
    fail("dense handshake projection did not preserve a cold cycle result");
}
