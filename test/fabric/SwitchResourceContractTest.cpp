#include "Fabric/IR/SwitchResourceContract.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

std::vector<std::vector<std::uint32_t>>
completeConnectivity(std::uint32_t inputCount, std::uint32_t outputCount) {
  std::vector<std::uint32_t> sources;
  sources.reserve(inputCount);
  for (std::uint32_t input = 0; input != inputCount; ++input)
    sources.push_back(input);
  return std::vector<std::vector<std::uint32_t>>(outputCount, sources);
}

void expectRejected(const char *test,
                    llvm::Expected<fabric::SwitchResourceContract> contract,
                    llvm::StringRef diagnostic) {
  if (contract)
    fail(test, "accepted an oversized switch crossbar");
  std::string message = llvm::toString(contract.takeError());
  if (!llvm::StringRef(message).contains(diagnostic))
    fail(test, "unexpected diagnostic: " + message);
}

void physicalTraversalsOwnOneLinearResourceProjection() {
  const char *test = __func__;
  fabric::SwitchResourceContract contract =
      take(test, fabric::SwitchResourceContract::create(
                     {fabric::Schedule::Temporal,
                      2,
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
  auto contract = fabric::SwitchResourceContract::create(
      {fabric::Schedule::Temporal, 2, 1, {{0, 1}}, std::nullopt});
  require(test, !contract, "fan-in without an exact policy was accepted");
  llvm::consumeError(contract.takeError());
}

void spatialAlternativesShareOneConfigurationRequester() {
  const char *test = __func__;
  fabric::SwitchResourceContract contract =
      take(test,
           fabric::SwitchResourceContract::create(
               {fabric::Schedule::Spatial, 2, 2, {{0, 1}, {1}}, std::nullopt}));

  require(test, contract.resourceContract().stateCount() == 4,
          "spatial switch did not expose one state per physical port");
  require(test, contract.resourceContract().usePatternCount() == 3,
          "spatial switch did not expose one pattern per traversal");
  require(test, contract.resourceContract().requesterCount() == 1,
          "spatial switch did not use one configuration requester");
  require(test, !contract.resourceContract().grantPolicy(),
          "spatial switch manufactured a runtime grant policy");
  for (std::uint32_t ordinal = 0;
       ordinal != contract.resourceContract().usePatternCount(); ++ordinal)
    require(test,
            contract.resourceContract()
                    .usePattern(fabric::UsePatternKey(ordinal))
                    .requester == fabric::RequesterKey(0),
            "spatial traversal escaped the configuration requester");
}

void crosspointProductOwnsTheShapeLimit() {
  const char *test = __func__;

  take(test, fabric::SwitchResourceContract::create(
                 {fabric::Schedule::Spatial, 16, 16,
                  completeConnectivity(16, 16), std::nullopt}));
  take(test, fabric::SwitchResourceContract::create(
                 {fabric::Schedule::Spatial, 1, 256,
                  completeConnectivity(1, 256), std::nullopt}));

  expectRejected(test,
                 fabric::SwitchResourceContract::create(
                     {fabric::Schedule::Spatial, 1, 257,
                      completeConnectivity(1, 257), std::nullopt}),
                 "crosspoint");
  expectRejected(test,
                 fabric::SwitchResourceContract::create(
                     {fabric::Schedule::Spatial, 17, 16,
                      completeConnectivity(17, 16), std::nullopt}),
                 "crosspoint");
}

} // namespace

int main() {
  physicalTraversalsOwnOneLinearResourceProjection();
  fanInWithoutPolicyIsRejected();
  spatialAlternativesShareOneConfigurationRequester();
  crosspointProductOwnsTheShapeLimit();
  llvm::outs() << "switch resource contract ok\n";
  return EXIT_SUCCESS;
}
