#ifndef FABRIC_IR_MEMORYCONSISTENCYCONTRACT_H
#define FABRIC_IR_MEMORYCONSISTENCYCONTRACT_H

#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {
class FabricArtifactView;
struct FabricImportBinding;
} // namespace loom::fabric

namespace fabric {

/// The two exact participant roles admitted by a MemoryConsistency hardware
/// domain. This field sum adds no persistent identity: each alternative keeps
/// the complete existing Fabric service or provider reference.
enum class MemoryConsistencyParticipantKind : std::uint32_t {
  Service,
  Provider,
};

struct MemoryConsistencyParticipant {
  using Payload = std::variant<loom::fabric::FabricMemoryServiceRef,
                               loom::fabric::SubordinateEndpointRef>;

  Payload payload;

  MemoryConsistencyParticipantKind kind() const {
    return static_cast<MemoryConsistencyParticipantKind>(payload.index());
  }

  static MemoryConsistencyParticipant
  service(loom::fabric::FabricMemoryServiceRef service) {
    return MemoryConsistencyParticipant{
        Payload(std::in_place_type<loom::fabric::FabricMemoryServiceRef>,
                std::move(service))};
  }

  static MemoryConsistencyParticipant
  provider(loom::fabric::SubordinateEndpointRef provider) {
    return MemoryConsistencyParticipant{
        Payload(std::in_place_type<loom::fabric::SubordinateEndpointRef>,
                std::move(provider))};
  }
};

inline bool operator==(const MemoryConsistencyParticipant &lhs,
                       const MemoryConsistencyParticipant &rhs) {
  return lhs.payload == rhs.payload;
}
inline bool operator!=(const MemoryConsistencyParticipant &lhs,
                       const MemoryConsistencyParticipant &rhs) {
  return !(lhs == rhs);
}

/// The only configurable release-summary visibility choice in schema 1.0.
enum class ReleaseVisibilityPoint : std::uint32_t {
  AtLinearization,
  ByRetirement,
};

struct BoundedCompletion {
  loom::fabric::ClockDomainRef progressClock;
  std::uint64_t maxIssueToRetireTicks = 0;
};

inline bool operator==(const BoundedCompletion &lhs,
                       const BoundedCompletion &rhs) {
  return lhs.progressClock == rhs.progressClock &&
         lhs.maxIssueToRetireTicks == rhs.maxIssueToRetireTicks;
}
inline bool operator!=(const BoundedCompletion &lhs,
                       const BoundedCompletion &rhs) {
  return !(lhs == rhs);
}

struct FairEventual {};

inline bool operator==(FairEventual, FairEventual) { return true; }
inline bool operator!=(FairEventual, FairEventual) { return false; }

using MemoryConsistencyProgress = std::variant<BoundedCompletion, FairEventual>;

struct MemoryConsistencyContractDeclaration {
  std::vector<MemoryConsistencyParticipant> participants;
  ReleaseVisibilityPoint releaseVisibilityPoint;
  MemoryConsistencyProgress progress;
  ResourceContract resourceContract;
};

enum class MemoryConsistencyContractViolation : std::uint32_t {
  EmptyParticipantDomain,
  DuplicateParticipant,
  InvalidReleaseVisibilityPoint,
  NonPositiveCompletionBound,
};

llvm::StringRef getMemoryConsistencyContractViolationName(
    MemoryConsistencyContractViolation violation);

class MemoryConsistencyContractError final
    : public llvm::ErrorInfo<MemoryConsistencyContractError> {
public:
  static char ID;

  MemoryConsistencyContractError(MemoryConsistencyContractViolation violation,
                                 std::string message)
      : violation_(violation), message_(std::move(message)) {}

  MemoryConsistencyContractViolation violation() const { return violation_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  MemoryConsistencyContractViolation violation_;
  std::string message_;
};

/// One complete owner-local MemoryConsistency hardware-domain contract.
///
/// The fixed linearization, modification-order, indivisible-RMW,
/// sequential-consistency, acquire-before-retirement, and addressed/fence
/// retirement guarantees are semantics of this type. They are deliberately
/// absent from the declaration and wire, so no field can weaken them. Dynamic
/// consistency state belongs to an execution provider and is not represented
/// here.
class MemoryConsistencyContract {
public:
  static llvm::Expected<MemoryConsistencyContract>
  create(MemoryConsistencyContractDeclaration declaration);

  llvm::ArrayRef<MemoryConsistencyParticipant> participants() const {
    return participants_;
  }
  ReleaseVisibilityPoint releaseVisibilityPoint() const {
    return releaseVisibilityPoint_;
  }
  const MemoryConsistencyProgress &progress() const { return progress_; }
  const ResourceContract &resourceContract() const { return resourceContract_; }

private:
  MemoryConsistencyContract(
      std::vector<MemoryConsistencyParticipant> participants,
      ReleaseVisibilityPoint releaseVisibilityPoint,
      MemoryConsistencyProgress progress, ResourceContract resourceContract)
      : participants_(std::move(participants)),
        releaseVisibilityPoint_(releaseVisibilityPoint),
        progress_(std::move(progress)),
        resourceContract_(std::move(resourceContract)) {}

  std::vector<MemoryConsistencyParticipant> participants_;
  ReleaseVisibilityPoint releaseVisibilityPoint_;
  MemoryConsistencyProgress progress_;
  ResourceContract resourceContract_;
};

/// Encodes one validated contract as its complete owner-local persistent
/// record. The surrounding HardwareDomain supplies root identity and schema
/// versioning.
llvm::Expected<std::vector<std::uint8_t>> encodeMemoryConsistencyContractRecord(
    const MemoryConsistencyContract &contract);

/// Strictly imports one complete canonical record. Unknown tags, malformed
/// references, noncanonical participant order, duplicate participants,
/// nonpositive bounds, malformed embedded ResourceContract records, and
/// trailing fields are rejected.
llvm::Expected<MemoryConsistencyContract>
decodeMemoryConsistencyContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

/// Resolves every participant and the optional progress clock against the
/// exact Fabric artifact binding. Root-level membership and clock-coverage
/// relations remain the future FabricSystemRootView finalizer's authority.
llvm::Error validateMemoryConsistencyContractReferences(
    const MemoryConsistencyContract &contract,
    const loom::fabric::FabricArtifactView &view,
    const loom::fabric::FabricImportBinding &binding);

} // namespace fabric

#endif // FABRIC_IR_MEMORYCONSISTENCYCONTRACT_H
