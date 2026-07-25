#ifndef LOOM_FABRIC_ARTIFACT_FABRICSYSTEMCONTRACTS_H
#define LOOM_FABRIC_ARTIFACT_FABRICSYSTEMCONTRACTS_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom {
namespace fabric {

enum class ResetPolarity : std::uint32_t {
  ActiveHigh,
  ActiveLow,
};

enum class ResetTiming : std::uint32_t {
  Synchronous,
  Asynchronous,
};

enum class ResetInitialState : std::uint32_t {
  Asserted,
  Deasserted,
};

class ClockDomainContractRecord {
public:
  static llvm::Expected<ClockDomainContractRecord>
  create(std::uint64_t periodFs, std::uint64_t phaseFs);

  std::uint64_t periodFs() const { return periodFs_; }
  std::uint64_t phaseFs() const { return phaseFs_; }

  friend bool operator==(const ClockDomainContractRecord &lhs,
                         const ClockDomainContractRecord &rhs) {
    return lhs.periodFs_ == rhs.periodFs_ && lhs.phaseFs_ == rhs.phaseFs_;
  }
  friend bool operator!=(const ClockDomainContractRecord &lhs,
                         const ClockDomainContractRecord &rhs) {
    return !(lhs == rhs);
  }

private:
  ClockDomainContractRecord(std::uint64_t periodFs, std::uint64_t phaseFs)
      : periodFs_(periodFs), phaseFs_(phaseFs) {}

  std::uint64_t periodFs_;
  std::uint64_t phaseFs_;
};

class ResetDomainContractRecord {
public:
  static llvm::Expected<ResetDomainContractRecord>
  create(ResetPolarity polarity, ResetTiming assertion, ResetTiming deassertion,
         ResetInitialState initialState,
         std::optional<ClockDomainRef> synchronousTo,
         std::uint32_t releaseLatencyCycles);

  ResetPolarity polarity() const { return polarity_; }
  ResetTiming assertion() const { return assertion_; }
  ResetTiming deassertion() const { return deassertion_; }
  ResetInitialState initialState() const { return initialState_; }
  const std::optional<ClockDomainRef> &synchronousTo() const {
    return synchronousTo_;
  }
  std::uint32_t releaseLatencyCycles() const { return releaseLatencyCycles_; }

  friend bool operator==(const ResetDomainContractRecord &lhs,
                         const ResetDomainContractRecord &rhs) {
    return lhs.polarity_ == rhs.polarity_ && lhs.assertion_ == rhs.assertion_ &&
           lhs.deassertion_ == rhs.deassertion_ &&
           lhs.initialState_ == rhs.initialState_ &&
           lhs.synchronousTo_ == rhs.synchronousTo_ &&
           lhs.releaseLatencyCycles_ == rhs.releaseLatencyCycles_;
  }
  friend bool operator!=(const ResetDomainContractRecord &lhs,
                         const ResetDomainContractRecord &rhs) {
    return !(lhs == rhs);
  }

private:
  ResetDomainContractRecord(ResetPolarity polarity, ResetTiming assertion,
                            ResetTiming deassertion,
                            ResetInitialState initialState,
                            std::optional<ClockDomainRef> synchronousTo,
                            std::uint32_t releaseLatencyCycles)
      : polarity_(polarity), assertion_(assertion), deassertion_(deassertion),
        initialState_(initialState), synchronousTo_(std::move(synchronousTo)),
        releaseLatencyCycles_(releaseLatencyCycles) {}

  ResetPolarity polarity_;
  ResetTiming assertion_;
  ResetTiming deassertion_;
  ResetInitialState initialState_;
  std::optional<ClockDomainRef> synchronousTo_;
  std::uint32_t releaseLatencyCycles_;
};

/// Version 1 has one crossing variant. Depth and synchronizer stages are
/// canonical unsigned 32-bit fields; the carrier remains structural context.
class ClockCrossingContractRecord {
public:
  static llvm::Expected<ClockCrossingContractRecord>
  createAsyncFifo(FabricTransferPatternRef transferPattern,
                  ClockDomainRef sourceClock, ClockDomainRef destinationClock,
                  std::uint32_t depth, std::uint32_t synchronizerStages);

  const FabricTransferPatternRef &transferPattern() const {
    return transferPattern_;
  }
  const ClockDomainRef &sourceClock() const { return sourceClock_; }
  const ClockDomainRef &destinationClock() const { return destinationClock_; }
  std::uint32_t depth() const { return depth_; }
  std::uint32_t synchronizerStages() const { return synchronizerStages_; }

  friend bool operator==(const ClockCrossingContractRecord &lhs,
                         const ClockCrossingContractRecord &rhs) {
    return lhs.transferPattern_ == rhs.transferPattern_ &&
           lhs.sourceClock_ == rhs.sourceClock_ &&
           lhs.destinationClock_ == rhs.destinationClock_ &&
           lhs.depth_ == rhs.depth_ &&
           lhs.synchronizerStages_ == rhs.synchronizerStages_;
  }
  friend bool operator!=(const ClockCrossingContractRecord &lhs,
                         const ClockCrossingContractRecord &rhs) {
    return !(lhs == rhs);
  }

private:
  ClockCrossingContractRecord(FabricTransferPatternRef transferPattern,
                              ClockDomainRef sourceClock,
                              ClockDomainRef destinationClock,
                              std::uint32_t depth,
                              std::uint32_t synchronizerStages)
      : transferPattern_(std::move(transferPattern)),
        sourceClock_(std::move(sourceClock)),
        destinationClock_(std::move(destinationClock)), depth_(depth),
        synchronizerStages_(synchronizerStages) {}

  FabricTransferPatternRef transferPattern_;
  ClockDomainRef sourceClock_;
  ClockDomainRef destinationClock_;
  std::uint32_t depth_;
  std::uint32_t synchronizerStages_;
};

/// These codecs own only the canonical record bytes and intrinsic field
/// invariants. Exact artifact scope, refined domain kinds, transfer-pattern
/// ownership, and whole-root relations are validated by the sealed System root
/// importer that embeds these records.
llvm::Expected<std::vector<std::uint8_t>>
encodeClockDomainContractRecord(const ClockDomainContractRecord &record);
llvm::Expected<ClockDomainContractRecord>
decodeClockDomainContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
encodeResetDomainContractRecord(const ResetDomainContractRecord &record);
llvm::Expected<ResetDomainContractRecord>
decodeResetDomainContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
encodeClockCrossingContractRecord(const ClockCrossingContractRecord &record);
llvm::Expected<ClockCrossingContractRecord>
decodeClockCrossingContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_ARTIFACT_FABRICSYSTEMCONTRACTS_H
