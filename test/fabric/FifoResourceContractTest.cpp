#include "Fabric/IR/FifoResourceContract.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "FIFO resource contract: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

fabric::ResourceContract contract(std::uint32_t depth, bool bypassable) {
  llvm::Expected<fabric::ResourceContract> result =
      fabric::createFifoResourceContract(depth, bypassable);
  if (!result)
    fail(llvm::toString(result.takeError()));
  return std::move(*result);
}

fabric::ResourceContract contract(std::uint32_t depth,
                                  fabric::FifoQueueDiscipline discipline,
                                  std::uint32_t tagWidthBits) {
  llvm::Expected<fabric::ResourceContract> result =
      fabric::createFifoResourceContract(depth, false, discipline,
                                         tagWidthBits);
  if (!result)
    fail(llvm::toString(result.takeError()));
  return std::move(*result);
}

void bufferedContractOwnsQueueUses() {
  fabric::ResourceContract value = contract(7, false);
  require(value.stateCount() == 1, "buffered FIFO state inventory differs");
  require(value.resourceTransitionCount() == 3,
          "buffered FIFO transition inventory differs");
  require(value.usePatternCount() == 3,
          "buffered FIFO use-pattern inventory differs");
  require(value.capacityDimensions(fabric::StateKey(0))[0].capacity ==
              fabric::CapacityUnits(7),
          "queue capacity did not preserve max_depth");
  require(!value.grantPolicy(), "single-requester FIFO has an arbiter");

  const fabric::UsePattern simultaneous =
      value.usePattern(fabric::fifoUsePattern(
          fabric::FifoUsePattern::SimultaneousDequeueEnqueue));
  require(simultaneous.claims.size() == 3,
          "simultaneous use does not reserve both services and one slot");
  require(simultaneous.commit.has_value(),
          "simultaneous use has no durable queue transition");
}

void bypassCapabilityAddsOnlyItsExactAlternative() {
  fabric::ResourceContract value = contract(4, true);
  require(value.stateCount() == 2, "bypass service state is absent");
  require(value.usePatternCount() == 4,
          "bypass transfer use pattern is absent");
  const fabric::UsePattern bypass = value.usePattern(
      fabric::fifoUsePattern(fabric::FifoUsePattern::BypassTransfer));
  require(bypass.acquire == bypass.release,
          "bypass transfer is not one combinational atomic event");
  require(!bypass.commit, "bypass transfer mutates queue state");
}

void zeroDepthIsRejected() {
  llvm::Expected<fabric::ResourceContract> invalid =
      fabric::createFifoResourceContract(0, true);
  require(!invalid, "zero-depth FIFO contract was accepted");
  llvm::consumeError(invalid.takeError());
}

void strictDisciplinePreservesTheSharedPoolContract() {
  require(contract(7, false) ==
              contract(7, fabric::FifoQueueDiscipline::StrictFifo, 0),
          "explicit StrictFifo changed the pre-discipline contract");
  llvm::Expected<fabric::ResourceContract> explicitBypass =
      fabric::createFifoResourceContract(
          7, true, fabric::FifoQueueDiscipline::StrictFifo, 0);
  require(explicitBypass && *explicitBypass == contract(7, true),
          "explicit StrictFifo changed the bypassable contract");
}

void virtualChannelContractOwnsTagQualifiedOffer() {
  fabric::ResourceContract value =
      contract(4, fabric::FifoQueueDiscipline::PerTagVirtualChannel, 4);
  require(value.stateCount() == 1, "virtual channel FIFO gained a state");
  require(value.resourceTransitionCount() == 4,
          "virtual channel FIFO does not own the offer-advance transition");
  require(value.usePatternCount() == 4,
          "virtual channel FIFO use-pattern inventory differs");
  require(value.capacityDimensions(fabric::StateKey(0))[0].capacity ==
              fabric::CapacityUnits(4),
          "virtual channel queue capacity is not one shared pool");

  const fabric::UsePattern dequeue = value.usePattern(
      fabric::fifoUsePattern(fabric::FifoUsePattern::Dequeue));
  require(dequeue.parameters.size() == 1 &&
              dequeue.parameters.front() ==
                  fabric::UsePatternValueSchema::physicalTag(4),
          "virtual channel dequeue is not qualified by the Physical Tag");
  const fabric::UsePattern simultaneous =
      value.usePattern(fabric::fifoUsePattern(
          fabric::FifoUsePattern::SimultaneousDequeueEnqueue));
  require(simultaneous.parameters.size() == 1 &&
              simultaneous.parameters.front() ==
                  fabric::UsePatternValueSchema::physicalTag(4),
          "virtual channel simultaneous use is not qualified by the tag");

  const fabric::UsePattern advance = value.usePattern(
      fabric::fifoVirtualChannelOfferAdvancePattern());
  require(advance.claims.empty(),
          "a refused offer must not reserve queue capacity");
  require(advance.commit &&
              advance.commit->transition == fabric::ResourceTransitionKey(3) &&
              advance.commit->event == fabric::EventKey(2),
          "offer advance does not commit at the cycle boundary");
  require(advance.parameters.size() == 1 &&
              advance.parameters.front() ==
                  fabric::UsePatternValueSchema::physicalTag(4),
          "offer advance does not name the refused channel");
}

void virtualChannelRejectsBypassAndMissingTagWidth() {
  llvm::Expected<fabric::ResourceContract> bypass =
      fabric::createFifoResourceContract(
          4, true, fabric::FifoQueueDiscipline::PerTagVirtualChannel, 4);
  require(!bypass, "a virtual channel FIFO accepted a bypass alternative");
  llvm::consumeError(bypass.takeError());
  llvm::Expected<fabric::ResourceContract> untagged =
      fabric::createFifoResourceContract(
          4, false, fabric::FifoQueueDiscipline::PerTagVirtualChannel, 0);
  require(!untagged, "a virtual channel FIFO accepted a missing tag width");
  llvm::consumeError(untagged.takeError());
}

} // namespace

int main() {
  bufferedContractOwnsQueueUses();
  bypassCapabilityAddsOnlyItsExactAlternative();
  zeroDepthIsRejected();
  strictDisciplinePreservesTheSharedPoolContract();
  virtualChannelContractOwnsTagQualifiedOffer();
  virtualChannelRejectsBypassAndMissingTagWidth();
  return EXIT_SUCCESS;
}
