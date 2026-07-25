#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

namespace {

constexpr FabricEntityId kClockA = 21;
constexpr FabricEntityId kClockB = 22;
constexpr FabricEntityId kCarrierA = 11;

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    const llvm::Twine &message) {
  if (value)
    fail(test, message);
  llvm::consumeError(value.takeError());
}

ClockDomainRef clockA() { return ClockDomainRef(HardwareDomainRef(kClockA)); }
ClockDomainRef clockB() { return ClockDomainRef(HardwareDomainRef(kClockB)); }
FabricTransferPatternRef patternA() {
  return {SystemTransportResourceRef(kCarrierA), 0};
}

void checkClockContract() {
  constexpr llvm::StringLiteral test = "clock contract";
  const ClockDomainContractRecord clock =
      take(test, ClockDomainContractRecord::create(1'000, 125));
  require(test, clock.periodFs() == 1'000 && clock.phaseFs() == 125,
          "lost clock fields");

  expectRejected(test, ClockDomainContractRecord::create(0, 0),
                 "accepted zero period");
  expectRejected(test, ClockDomainContractRecord::create(10, 10),
                 "accepted phase equal to period");

  const std::vector<std::uint8_t> encoded =
      take(test, encodeClockDomainContractRecord(clock));
  require(test, take(test, decodeClockDomainContractRecord(encoded)) == clock,
          "clock roundtrip changed the record");

  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  expectRejected(test, decodeClockDomainContractRecord(trailing),
                 "accepted trailing clock bytes");
}

void checkResetContract() {
  constexpr llvm::StringLiteral test = "reset contract";
  const ResetDomainContractRecord asynchronous =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveLow, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Asserted,
                     std::nullopt, 0));
  const ResetDomainContractRecord synchronousRelease =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Synchronous, ResetInitialState::Deasserted,
                     clockA(), 0));
  const ResetDomainContractRecord synchronousAssertion =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Synchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     clockA(), 2));

  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     clockA(), 0),
                 "accepted a clock on a fully asynchronous reset");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Synchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 0),
                 "accepted synchronous assertion without a clock");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 1),
                 "accepted clock-measured latency without a clock");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     static_cast<ResetPolarity>(2), ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 0),
                 "accepted an unknown reset polarity");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, static_cast<ResetTiming>(2),
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 0),
                 "accepted an unknown reset timing");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous,
                     static_cast<ResetInitialState>(2), std::nullopt, 0),
                 "accepted an unknown reset initial state");

  const std::vector<std::uint8_t> encoded =
      take(test, encodeResetDomainContractRecord(synchronousRelease));
  require(test,
          take(test, decodeResetDomainContractRecord(encoded)) ==
              synchronousRelease,
          "reset roundtrip changed the record");

  const std::vector<std::uint8_t> asyncBytes =
      take(test, encodeResetDomainContractRecord(asynchronous));
  require(test,
          take(test, decodeResetDomainContractRecord(asyncBytes)) ==
              asynchronous,
          "asynchronous reset roundtrip changed the record");

  const std::vector<std::uint8_t> assertionBytes =
      take(test, encodeResetDomainContractRecord(synchronousAssertion));
  require(test,
          take(test, decodeResetDomainContractRecord(assertionBytes)) ==
              synchronousAssertion,
          "clocked release latency changed during roundtrip");
}

void checkClockCrossingContract() {
  constexpr llvm::StringLiteral test = "clock crossing contract";
  const ClockCrossingContractRecord crossing =
      take(test, ClockCrossingContractRecord::createAsyncFifo(
                     patternA(), clockA(), clockB(), 8, 2));

  expectRejected(test,
                 ClockCrossingContractRecord::createAsyncFifo(
                     patternA(), clockA(), clockB(), 0, 2),
                 "accepted zero FIFO depth");
  expectRejected(test,
                 ClockCrossingContractRecord::createAsyncFifo(
                     patternA(), clockA(), clockB(), 8, 0),
                 "accepted zero synchronizer stages");

  const std::vector<std::uint8_t> encoded =
      take(test, encodeClockCrossingContractRecord(crossing));
  require(test,
          take(test, decodeClockCrossingContractRecord(encoded)) == crossing,
          "crossing roundtrip changed the record");

  FabricByteWriter unknownVariant;
  unknownVariant.tag(1);
  expectRejected(test, decodeClockCrossingContractRecord(unknownVariant.take()),
                 "accepted an unknown crossing variant");

  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  expectRejected(test, decodeClockCrossingContractRecord(trailing),
                 "accepted trailing crossing bytes");
}

} // namespace

int main() {
  checkClockContract();
  checkResetContract();
  checkClockCrossingContract();
  return EXIT_SUCCESS;
}
