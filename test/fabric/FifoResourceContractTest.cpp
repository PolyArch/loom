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

} // namespace

int main() {
  bufferedContractOwnsQueueUses();
  bypassCapabilityAddsOnlyItsExactAlternative();
  zeroDepthIsRejected();
  return EXIT_SUCCESS;
}
