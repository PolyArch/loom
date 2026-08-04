#include "Simulator/SpatialObservationComparison.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "SpatialObservationComparisonTest: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

loom::sim::SpatialFunctionalObservations observations(std::uint32_t value) {
  loom::sim::CanonicalValueSequence scalar{
      1,
      {loom::sim::SemanticLane::definedPointer(llvm::APInt(64, value), 3,
                                               llvm::APInt(64, 8))}};
  loom::sim::CanonicalStreamSequence stream{
      {1, {loom::sim::SemanticLane::defined(llvm::APInt(32, value))}},
      loom::sim::StreamTermination::ClosedAfterLast};
  loom::sim::FullMemoryObservation memory{
      {{loom::sim::SemanticState::Defined, static_cast<std::uint8_t>(value)}}};
  return {{{loom::sim::PublishedValueResult{std::move(scalar)}}},
          {std::move(stream)},
          {std::move(memory)}};
}

void exactRelationCoversAllSemanticPayloads() {
  auto reference = observations(7);
  auto candidate = observations(7);
  if (!loom::sim::haveExactlyEqualSpatialFunctionalObservations(reference,
                                                                candidate))
    fail("equal positional observations were rejected");

  auto differentPointer = observations(7);
  auto &pointer = std::get<loom::sim::PublishedValueResult>(
                      differentPointer.valueResults.front())
                      .value.lanes.front()
                      .pointerTarget;
  pointer->byteOffset = llvm::APInt(64, 9);
  if (loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          reference, differentPointer))
    fail("pointer provenance was ignored");

  auto differentTermination = observations(7);
  differentTermination.streamOutputs.front().termination =
      loom::sim::StreamTermination::OpenAfterLast;
  if (loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          reference, differentTermination))
    fail("stream termination was ignored");

  auto differentMemory = observations(7);
  std::get<loom::sim::FullMemoryObservation>(differentMemory.memories.front())
      .bytes.front()
      .value = 8;
  if (loom::sim::haveExactlyEqualSpatialFunctionalObservations(reference,
                                                               differentMemory))
    fail("memory payload was ignored");
}

} // namespace

int main() {
  exactRelationCoversAllSemanticPayloads();
  return EXIT_SUCCESS;
}
