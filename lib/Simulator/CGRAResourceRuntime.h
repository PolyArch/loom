#ifndef LOOM_LIB_SIMULATOR_CGRARESOURCERUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRARESOURCERUNTIME_H

#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace loom::sim::detail {

inline constexpr std::uint64_t noCgraResourceDomain =
    std::numeric_limits<std::uint64_t>::max();

enum class CgraGrantPolicyKind : std::uint8_t {
  None,
  FixedPriority,
  RoundRobin,
};

struct CgraResourcePatternSelection final {
  std::uint64_t ownerOrdinal = 0;
  ::fabric::UsePatternKey pattern = ::fabric::UsePatternKey(0);
};

/// One derived atomic activation over exact owner-local UsePatterns. Pattern
/// rows remain the exact Fabric selections; this transient slice only groups
/// claims that must acquire and release as one envelope.
struct CgraResourceActivationSelection final {
  std::uint64_t patternOffset = 0;
  std::uint32_t patternCount = 0;
};

struct CgraResourceDimensionPlan final {
  std::uint32_t capacity = 0;
  std::uint32_t initialOccupancy = 0;
};

struct CgraResourceClaimPlan final {
  std::uint64_t dimensionOrdinal = 0;
  std::uint32_t amount = 0;
};

struct CgraResourceDomainPlan final {
  std::uint64_t ownerOrdinal = 0;
  CgraGrantPolicyKind policy = CgraGrantPolicyKind::None;
  std::uint64_t requesterOffset = 0;
  std::uint32_t requesterCount = 0;
  std::uint32_t resetPosition = 0;
};

struct CgraResourceUsePlan final {
  std::uint64_t ownerOrdinal = 0;
  std::uint64_t domainOrdinal = noCgraResourceDomain;
  std::uint32_t requesterOrdinal = 0;
  std::uint32_t requesterPosition = 0;
  std::uint64_t claimOffset = 0;
  std::uint32_t claimCount = 0;
};

/// Removable dense projection of exact Fabric ResourceContracts and selected
/// Mapping UsePatterns. The source contracts remain the only semantic owner.
struct CgraResourceRuntimePlan final {
  std::vector<CgraResourceDimensionPlan> dimensions;
  std::vector<CgraResourceDomainPlan> domains;
  std::vector<std::uint32_t> domainRequesters;
  std::vector<CgraResourceUsePlan> selectedUses;
  std::vector<CgraResourceClaimPlan> claims;
};

llvm::Expected<CgraResourceRuntimePlan> freezeCgraResourceRuntimePlan(
    llvm::ArrayRef<const ::fabric::ResourceContract *> ownerContracts,
    llvm::ArrayRef<CgraResourcePatternSelection> selectedPatterns);

llvm::Expected<CgraResourceRuntimePlan> freezeCgraResourceRuntimePlan(
    llvm::ArrayRef<const ::fabric::ResourceContract *> ownerContracts,
    llvm::ArrayRef<CgraResourcePatternSelection> selectedPatterns,
    llvm::ArrayRef<CgraResourceActivationSelection> activations);

struct CgraResourceRequest final {
  std::uint64_t selectedUseOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
};

struct CgraClaimEnvelope final {
  std::uint32_t slot = 0;
  std::uint64_t generation = 0;
};

struct CgraResourceGrant final {
  std::uint64_t selectedUseOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  CgraClaimEnvelope claimEnvelope;
};

/// Execution-local temporary-capacity and arbitration state. Durable resource
/// transitions remain owned by typed concrete-resource providers.
class CgraResourceRuntime final {
public:
  static llvm::Expected<CgraResourceRuntime>
  create(const CgraResourceRuntimePlan &plan);

  llvm::Expected<std::vector<CgraResourceGrant>>
  grant(llvm::ArrayRef<CgraResourceRequest> requests);

  llvm::Error release(CgraClaimEnvelope envelope);

  std::uint32_t occupancy(std::uint64_t dimensionOrdinal) const;

private:
  struct EnvelopeSlot final {
    std::uint64_t generation = 0;
    std::uint64_t selectedUseOrdinal = 0;
    bool active = false;
  };

  explicit CgraResourceRuntime(const CgraResourceRuntimePlan &plan)
      : plan_(&plan) {}

  const CgraResourceRuntimePlan *plan_ = nullptr;
  std::vector<std::uint32_t> occupancy_;
  std::vector<std::uint32_t> domainCursors_;
  std::vector<EnvelopeSlot> envelopes_;
  std::vector<std::uint32_t> freeEnvelopes_;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRARESOURCERUNTIME_H
