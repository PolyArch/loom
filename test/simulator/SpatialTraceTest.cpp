#include "Simulator/SpatialTrace.h"

#include "CGRAExecutionPlan.h"
#include "CgraPhysicalTraceProjection.h"
#include "Common/Artifact.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial trace test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::ArtifactIdentity identity() {
  return take(loom::ArtifactIdentity::fromBytes(
      std::vector<std::uint8_t>(loom::ArtifactIdentity::byteSize, 0)));
}

loom::sim::SpatialEventCoordinate coordinate(std::uint64_t cycle) {
  return {take(loom::evaluation::ExactRatio::get(cycle, 1)), 0};
}

loom::sim::ActorTransitionOccurrenceRef transition(std::uint64_t ordinal) {
  const loom::ArtifactIdentity owner = identity();
  return {{0}, {owner, dataflow::ActorId(4)}, ordinal};
}

void levelsAndCanonicalKeysAreClosed() {
  using namespace loom::sim;
  SpatialDiagnosticTrace trace{TraceCaptureLevel::Microarchitecture, {}};
  ActorTransitionOccurrenceRef actor = transition(3);
  ActorResultTokenOccurrenceRef token{actor, 0, 5};
  PhysicalActionOccurrenceRef physical{TransitionPhysicalActionParent{actor},
                                       2};
  const loom::fabric::SystemTransportResourceRef resource(1);
  const loom::fabric::FabricUsePatternOwnerRef useOwner(
      loom::fabric::FabricInventoryOwnerRef::of(resource));

  SpatialTraceFrame frame{coordinate(7), {}};
  frame.events.push_back(PhysicalRequestedTraceEvent{
      physical,
      PhysicalUseTarget{loom::fabric::FabricUsePatternRef{useOwner, 0}}});
  frame.events.push_back(TokenPublishedTraceEvent{
      TokenOccurrenceRef{token},
      CanonicalValueSequence{1, {SemanticLane::defined(llvm::APInt(32, 9))}}});
  frame.events.push_back(ActorCommittedTraceEvent{actor});
  if (llvm::Error error = appendSpatialTraceFrame(trace, std::move(frame)))
    fail(llvm::toString(std::move(error)));
  if (trace.frames.size() != 1 || trace.frames.front().events.size() != 3 ||
      trace.frames.front().events[0].index() != 0 ||
      trace.frames.front().events[1].index() != 2 ||
      trace.frames.front().events[2].index() != 4)
    fail("events are not in canonical discriminant/key order");

  SpatialTraceFrame duplicate{
      coordinate(8),
      {ActorRetiredTraceEvent{actor}, ActorRetiredTraceEvent{actor}}};
  llvm::Error duplicateError = canonicalizeSpatialTraceFrame(
      duplicate, TraceCaptureLevel::Microarchitecture);
  if (!duplicateError)
    fail("duplicate event key was accepted");
  llvm::consumeError(std::move(duplicateError));

  SpatialTraceFrame tooDetailed{
      coordinate(8),
      {PhysicalGrantedTraceEvent{PhysicalActionOccurrenceRef{
          TransitionPhysicalActionParent{actor}, 0}}}};
  llvm::Error levelError =
      canonicalizeSpatialTraceFrame(tooDetailed, TraceCaptureLevel::Semantic);
  if (!levelError)
    fail("physical event was accepted at Semantic level");
  llvm::consumeError(std::move(levelError));

  const loom::fabric::FabricPhysicalTraversalRef traversal =
      loom::fabric::FabricPhysicalTraversalRef::switchTraversal(
          loom::fabric::FabricSwitchOccurrenceRef(7), 0, 1);
  const PhysicalTransferTarget transfer{{traversal, traversal}, {}};
  SpatialTraceFrame duplicateTransfer{
      coordinate(8), {PhysicalRequestedTraceEvent{physical, transfer}}};
  llvm::Error targetError = canonicalizeSpatialTraceFrame(
      duplicateTransfer, TraceCaptureLevel::Microarchitecture);
  if (!targetError)
    fail("noncanonical physical transfer target was accepted");
  llvm::consumeError(std::move(targetError));
}

void transferProjectionPreservesAtomicPatternSet() {
  using namespace loom::sim;
  using namespace loom::sim::detail;
  using namespace loom::fabric;

  const FabricSwitchOccurrenceRef owner(7);
  const FabricInventoryOwnerRef inventory = FabricInventoryOwnerRef::of(owner);
  CgraFrozenExecutionPlan plan;
  plan.physicalUses.push_back({0, 2, 0});
  plan.physicalUsePatterns = {
      FabricUsePatternRef{FabricUsePatternOwnerRef(inventory), 0},
      FabricUsePatternRef{FabricUsePatternOwnerRef(inventory), 1}};

  const FabricPhysicalTraversalRef first =
      FabricPhysicalTraversalRef::switchTraversal(owner, 0, 0);
  const FabricPhysicalTraversalRef second =
      FabricPhysicalTraversalRef::switchTraversal(owner, 0, 1);
  const std::array<FabricPhysicalTraversalRef, 2> noncanonical = {second,
                                                                  first};
  PhysicalActionTarget target =
      take(projectPhysicalTransferTarget(plan, 0, noncanonical));
  const auto *transfer = std::get_if<PhysicalTransferTarget>(&target);
  if (!transfer || transfer->traversals.size() != 2 ||
      transfer->traversals.front() != first ||
      transfer->traversals.back() != second ||
      transfer->usePatterns != plan.physicalUsePatterns)
    fail("transfer projection lost its canonical atomic pattern set");

  SpatialTraceFrame frame{
      coordinate(9),
      {PhysicalRequestedTraceEvent{
          PhysicalActionOccurrenceRef{
              TransitionPhysicalActionParent{transition(4)}, 0},
          std::move(target)}}};
  if (llvm::Error error = canonicalizeSpatialTraceFrame(
          frame, TraceCaptureLevel::Microarchitecture))
    fail(llvm::toString(std::move(error)));
}

} // namespace

int main() {
  levelsAndCanonicalKeysAreClosed();
  transferProjectionPreservesAtomicPatternSet();
  return EXIT_SUCCESS;
}
