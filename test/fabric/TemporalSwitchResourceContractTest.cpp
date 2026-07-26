#include "Fabric/IR/TemporalSwitchResourceContract.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(const char *test, bool condition, const char *message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void physicalTraversalsOwnOneLinearResourceProjection() {
  const char *test = __func__;
  fabric::TemporalSwitchResourceContract contract =
      take(test, fabric::TemporalSwitchResourceContract::create(
                     {2,
                      2,
                      {{0, 1}, {0}},
                      fabric::TemporalSwitchGrantPolicy(
                          fabric::TemporalSwitchRoundRobin{{0, 1}, 0})}));

  require(test, contract.resourceContract().stateCount() == 4,
          "switch did not expose one state per physical port");
  require(test, contract.resourceContract().usePatternCount() == 3,
          "switch did not expose one pattern per physical traversal");
  require(test, contract.resourceContract().requesterCount() == 2,
          "switch requester domain did not match its input domain");

  const fabric::UsePatternKey first =
      take(test, contract.traversalPattern(0, 0));
  const fabric::UsePatternKey broadcast =
      take(test, contract.traversalPattern(0, 1));
  const fabric::UsePatternKey competing =
      take(test, contract.traversalPattern(1, 0));
  require(test,
          contract.resourceContract().usePattern(first).requester ==
                  contract.inputRequester(0) &&
              contract.resourceContract().usePattern(broadcast).requester ==
                  contract.inputRequester(0) &&
              contract.resourceContract().usePattern(competing).requester ==
                  contract.inputRequester(1),
          "broadcast or fan-in changed the input-owned requester relation");
  require(test, contract.resourceContract().grantPolicy().has_value(),
          "fan-in did not retain its exact arbitration policy");

  llvm::Expected<fabric::UsePatternKey> disconnected =
      contract.traversalPattern(1, 1);
  require(test, !disconnected,
          "disconnected input/output pair acquired a use pattern");
  llvm::consumeError(disconnected.takeError());
}

void fanInWithoutPolicyIsRejected() {
  const char *test = __func__;
  auto contract = fabric::TemporalSwitchResourceContract::create(
      {2, 1, {{0, 1}}, std::nullopt});
  require(test, !contract, "fan-in without an exact policy was accepted");
  llvm::consumeError(contract.takeError());
}

} // namespace

int main() {
  physicalTraversalsOwnOneLinearResourceProjection();
  fanInWithoutPolicyIsRejected();
  llvm::outs() << "temporal switch resource contract ok\n";
  return EXIT_SUCCESS;
}
