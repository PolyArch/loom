#include "Fabric/Artifact/FabricSystemContracts.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

namespace {

constexpr std::uint32_t kActiveHighWireTag = 0;
constexpr std::uint32_t kActiveLowWireTag = 1;
constexpr std::uint32_t kSynchronousWireTag = 0;
constexpr std::uint32_t kAsynchronousWireTag = 1;
constexpr std::uint32_t kAssertedWireTag = 0;
constexpr std::uint32_t kDeassertedWireTag = 1;
constexpr std::uint32_t kOptionalClockAbsentWireTag = 0;
constexpr std::uint32_t kOptionalClockPresentWireTag = 1;
constexpr std::uint32_t kAsyncFifoWireTag = 0;

bool isKnown(ResetPolarity value) {
  switch (value) {
  case ResetPolarity::ActiveHigh:
  case ResetPolarity::ActiveLow:
    return true;
  }
  return false;
}

bool isKnown(ResetTiming value) {
  switch (value) {
  case ResetTiming::Synchronous:
  case ResetTiming::Asynchronous:
    return true;
  }
  return false;
}

bool isKnown(ResetInitialState value) {
  switch (value) {
  case ResetInitialState::Asserted:
  case ResetInitialState::Deasserted:
    return true;
  }
  return false;
}

llvm::Error invalidContract(llvm::StringRef contract,
                            const llvm::Twine &message) {
  return llvm::createStringError("invalid %s contract: %s",
                                 contract.str().c_str(), message.str().c_str());
}

llvm::Error malformedRecord(llvm::StringRef record,
                            const llvm::Twine &message) {
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("invalid ") + record +
                                " record: " + message);
}

std::uint32_t resetPolarityWireTag(ResetPolarity value) {
  switch (value) {
  case ResetPolarity::ActiveHigh:
    return kActiveHighWireTag;
  case ResetPolarity::ActiveLow:
    return kActiveLowWireTag;
  }
  llvm_unreachable("validated reset polarity has no wire tag");
}

llvm::Expected<ResetPolarity> readResetPolarity(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  switch (*raw) {
  case kActiveHighWireTag:
    return ResetPolarity::ActiveHigh;
  case kActiveLowWireTag:
    return ResetPolarity::ActiveLow;
  default:
    return malformedRecord("reset polarity",
                           llvm::Twine("unknown discriminant ") +
                               llvm::Twine(*raw));
  }
}

std::uint32_t resetTimingWireTag(ResetTiming value) {
  switch (value) {
  case ResetTiming::Synchronous:
    return kSynchronousWireTag;
  case ResetTiming::Asynchronous:
    return kAsynchronousWireTag;
  }
  llvm_unreachable("validated reset timing has no wire tag");
}

llvm::Expected<ResetTiming> readResetTiming(FabricByteReader &reader,
                                            llvm::StringRef field) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  switch (*raw) {
  case kSynchronousWireTag:
    return ResetTiming::Synchronous;
  case kAsynchronousWireTag:
    return ResetTiming::Asynchronous;
  default:
    return malformedRecord(field, llvm::Twine("unknown discriminant ") +
                                      llvm::Twine(*raw));
  }
}

std::uint32_t resetInitialStateWireTag(ResetInitialState value) {
  switch (value) {
  case ResetInitialState::Asserted:
    return kAssertedWireTag;
  case ResetInitialState::Deasserted:
    return kDeassertedWireTag;
  }
  llvm_unreachable("validated reset initial state has no wire tag");
}

llvm::Expected<ResetInitialState>
readResetInitialState(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  switch (*raw) {
  case kAssertedWireTag:
    return ResetInitialState::Asserted;
  case kDeassertedWireTag:
    return ResetInitialState::Deasserted;
  default:
    return malformedRecord("reset initial state",
                           llvm::Twine("unknown discriminant ") +
                               llvm::Twine(*raw));
  }
}

llvm::Error readAsyncFifoVariant(FabricByteReader &reader) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  if (*raw == kAsyncFifoWireTag)
    return llvm::Error::success();
  return malformedRecord("clock crossing variant",
                         llvm::Twine("unknown discriminant ") +
                             llvm::Twine(*raw));
}

llvm::Error requireFinished(FabricByteReader &reader, llvm::StringRef record) {
  if (!reader.empty())
    return malformedRecord(record, "trailing bytes");
  return llvm::Error::success();
}

template <typename Record, typename Encoder>
llvm::Error requireCanonical(llvm::ArrayRef<std::uint8_t> bytes,
                             const Record &record, Encoder encode,
                             llvm::StringRef name) {
  llvm::Expected<std::vector<std::uint8_t>> canonical = encode(record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return malformedRecord(name, "noncanonical encoding");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ClockDomainContractRecord>
ClockDomainContractRecord::create(std::uint64_t periodFs,
                                  std::uint64_t phaseFs) {
  if (periodFs == 0)
    return invalidContract("clock", "period_fs must be positive");
  if (phaseFs >= periodFs)
    return invalidContract("clock", "phase_fs must be less than period_fs");
  return ClockDomainContractRecord(periodFs, phaseFs);
}

llvm::Expected<ResetDomainContractRecord> ResetDomainContractRecord::create(
    ResetPolarity polarity, ResetTiming assertion, ResetTiming deassertion,
    ResetInitialState initialState, std::optional<ClockDomainRef> synchronousTo,
    std::uint32_t releaseLatencyCycles) {
  if (!isKnown(polarity) || !isKnown(assertion) || !isKnown(deassertion) ||
      !isKnown(initialState))
    return invalidContract("reset", "unknown closed-enum value");

  const bool namesClock = assertion == ResetTiming::Synchronous ||
                          deassertion == ResetTiming::Synchronous;
  if (namesClock != synchronousTo.has_value())
    return invalidContract(
        "reset", namesClock
                     ? "synchronous timing requires synchronous_to"
                     : "fully asynchronous reset forbids synchronous_to");

  if (!synchronousTo && releaseLatencyCycles != 0)
    return invalidContract(
        "reset", "clock-measured release latency requires synchronous_to");

  return ResetDomainContractRecord(polarity, assertion, deassertion,
                                   initialState, std::move(synchronousTo),
                                   releaseLatencyCycles);
}

llvm::Expected<ClockCrossingContractRecord>
ClockCrossingContractRecord::createAsyncFifo(
    FabricTransferPatternRef transferPattern, ClockDomainRef sourceClock,
    ClockDomainRef destinationClock, std::uint32_t depth,
    std::uint32_t synchronizerStages) {
  if (sourceClock == destinationClock)
    return invalidContract("AsyncFifo clock crossing",
                           "source and destination clocks must differ");
  if (depth == 0)
    return invalidContract("AsyncFifo clock crossing",
                           "depth must be positive");
  if (synchronizerStages == 0)
    return invalidContract("AsyncFifo clock crossing",
                           "synchronizer_stages must be positive");
  return ClockCrossingContractRecord(
      std::move(transferPattern), std::move(sourceClock),
      std::move(destinationClock), depth, synchronizerStages);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeClockDomainContractRecord(
    const ClockDomainContractRecord &record) {
  FabricByteWriter writer;
  writer.field(record.periodFs());
  writer.field(record.phaseFs());
  return writer.take();
}

llvm::Expected<ClockDomainContractRecord>
loom::fabric::decodeClockDomainContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<std::uint64_t> period = reader.field();
  if (!period)
    return period.takeError();
  llvm::Expected<std::uint64_t> phase = reader.field();
  if (!phase)
    return phase.takeError();
  if (llvm::Error error = requireFinished(reader, "clock domain"))
    return std::move(error);

  llvm::Expected<ClockDomainContractRecord> record =
      ClockDomainContractRecord::create(*period, *phase);
  if (!record)
    return record.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *record, encodeClockDomainContractRecord, "clock domain"))
    return std::move(error);
  return std::move(*record);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeResetDomainContractRecord(
    const ResetDomainContractRecord &record) {
  FabricByteWriter writer;
  writer.tag(resetPolarityWireTag(record.polarity()));
  writer.tag(resetTimingWireTag(record.assertion()));
  writer.tag(resetTimingWireTag(record.deassertion()));
  writer.tag(resetInitialStateWireTag(record.initialState()));
  if (record.synchronousTo()) {
    writer.tag(kOptionalClockPresentWireTag);
    encodeFabricRef(writer, *record.synchronousTo());
  } else {
    writer.tag(kOptionalClockAbsentWireTag);
  }
  writer.tag(record.releaseLatencyCycles());
  return writer.take();
}

llvm::Expected<ResetDomainContractRecord>
loom::fabric::decodeResetDomainContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  llvm::Expected<ResetPolarity> polarity = readResetPolarity(reader);
  if (!polarity)
    return polarity.takeError();
  llvm::Expected<ResetTiming> assertion =
      readResetTiming(reader, "reset assertion timing");
  if (!assertion)
    return assertion.takeError();
  llvm::Expected<ResetTiming> deassertion =
      readResetTiming(reader, "reset deassertion timing");
  if (!deassertion)
    return deassertion.takeError();
  llvm::Expected<ResetInitialState> initialState =
      readResetInitialState(reader);
  if (!initialState)
    return initialState.takeError();
  llvm::Expected<std::uint32_t> clockTag = reader.tag();
  if (!clockTag)
    return clockTag.takeError();
  if (*clockTag != kOptionalClockAbsentWireTag &&
      *clockTag != kOptionalClockPresentWireTag)
    return malformedRecord("reset synchronous_to",
                           llvm::Twine("unknown discriminant ") +
                               llvm::Twine(*clockTag));

  std::optional<ClockDomainRef> synchronousTo;
  if (*clockTag == kOptionalClockPresentWireTag) {
    ClockDomainRef clock;
    if (llvm::Error error = decodeFabricRefInto(reader, clock))
      return std::move(error);
    synchronousTo = std::move(clock);
  }

  llvm::Expected<std::uint32_t> releaseLatency = reader.tag();
  if (!releaseLatency)
    return releaseLatency.takeError();
  if (llvm::Error error = requireFinished(reader, "reset domain"))
    return std::move(error);

  llvm::Expected<ResetDomainContractRecord> record =
      ResetDomainContractRecord::create(*polarity, *assertion, *deassertion,
                                        *initialState, std::move(synchronousTo),
                                        *releaseLatency);
  if (!record)
    return record.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *record, encodeResetDomainContractRecord, "reset domain"))
    return std::move(error);
  return std::move(*record);
}

llvm::Expected<std::vector<std::uint8_t>>
loom::fabric::encodeClockCrossingContractRecord(
    const ClockCrossingContractRecord &record) {
  FabricByteWriter writer;
  writer.tag(kAsyncFifoWireTag);
  encodeFabricRef(writer, record.transferPattern());
  encodeFabricRef(writer, record.sourceClock());
  encodeFabricRef(writer, record.destinationClock());
  writer.tag(record.depth());
  writer.tag(record.synchronizerStages());
  return writer.take();
}

llvm::Expected<ClockCrossingContractRecord>
loom::fabric::decodeClockCrossingContractRecord(
    llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  if (llvm::Error error = readAsyncFifoVariant(reader))
    return std::move(error);

  FabricTransferPatternRef transferPattern;
  ClockDomainRef sourceClock;
  ClockDomainRef destinationClock;
  if (llvm::Error error = decodeFabricRefInto(reader, transferPattern))
    return std::move(error);
  if (llvm::Error error = decodeFabricRefInto(reader, sourceClock))
    return std::move(error);
  if (llvm::Error error = decodeFabricRefInto(reader, destinationClock))
    return std::move(error);
  llvm::Expected<std::uint32_t> depth = reader.tag();
  if (!depth)
    return depth.takeError();
  llvm::Expected<std::uint32_t> synchronizerStages = reader.tag();
  if (!synchronizerStages)
    return synchronizerStages.takeError();
  if (llvm::Error error = requireFinished(reader, "clock crossing"))
    return std::move(error);

  llvm::Expected<ClockCrossingContractRecord> record =
      ClockCrossingContractRecord::createAsyncFifo(
          std::move(transferPattern), std::move(sourceClock),
          std::move(destinationClock), *depth, *synchronizerStages);
  if (!record)
    return record.takeError();
  if (llvm::Error error = requireCanonical(
          bytes, *record, encodeClockCrossingContractRecord, "clock crossing"))
    return std::move(error);
  return std::move(*record);
}
