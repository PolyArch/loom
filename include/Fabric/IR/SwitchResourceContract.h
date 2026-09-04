#ifndef FABRIC_IR_SWITCHRESOURCECONTRACT_H
#define FABRIC_IR_SWITCHRESOURCECONTRACT_H

#include "Fabric/IR/FabricEnums.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace fabric {

class SwitchOp;

inline constexpr llvm::StringLiteral
    kSwitchGrantPolicyParameterName("grant_policy");

inline constexpr std::uint64_t kSwitchCrosspointWarningThreshold = 64;
inline constexpr std::uint64_t kSwitchCrosspointLimit = 256;

/// Returns the exact crosspoint product for a nonempty switch shape within the
/// Fabric limit. Multiplication uses the shared overflow-safe arithmetic.
llvm::Expected<std::uint64_t>
validatedSwitchCrosspointCount(std::uint64_t inputCount,
                               std::uint64_t outputCount);

struct TemporalSwitchFixedPriority final {
  std::vector<std::uint32_t> requesterOrder;
};

struct TemporalSwitchRoundRobin final {
  std::vector<std::uint32_t> requesterCycle;
  std::uint32_t resetRequester = 0;
};

using TemporalSwitchGrantPolicy =
    std::variant<TemporalSwitchFixedPriority, TemporalSwitchRoundRobin>;

/// One canonical connected component of the switch's physical bipartite
/// input/output graph. Inputs and outputs are ascending physical ordinals;
/// requesterOrder is the exact component-local projection of GrantPolicy, or
/// ascending input order when no runtime policy is admitted.
struct SwitchArbitrationComponent final {
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::vector<std::uint32_t> requesterOrder;
  std::optional<std::uint32_t> roundRobinResetPosition;
};

struct SwitchResourceDeclaration final {
  Schedule schedule = Schedule::Spatial;
  std::uint32_t inputCount = 0;
  std::uint32_t outputCount = 0;
  std::vector<std::vector<std::uint32_t>> sourcesByOutput;
  std::optional<TemporalSwitchGrantPolicy> grantPolicy;
};

/// Derives the one canonical physical component and policy projection shared
/// by Fabric handshake compilation and RTL lowering. The generic contract is
/// the sole grant-policy source; admitted connectivity supplies only the
/// switch-owned bipartite topology.
llvm::Expected<std::vector<SwitchArbitrationComponent>>
deriveSwitchArbitrationComponents(
    Schedule schedule, std::uint32_t inputCount, std::uint32_t outputCount,
    llvm::ArrayRef<std::vector<std::uint32_t>> sourcesByOutput,
    const ResourceContract &contract);

/// The complete Mapping-visible resource projection of one switch.
/// It remains linear in the physical connectivity: every admitted input/output
/// traversal is one use pattern and each pattern claims exactly that ingress
/// and egress service. Spatial patterns share one configuration requester;
/// temporal patterns use their input requester. Mapping separately derives the
/// exact selected broadcast activation, so this contract neither enumerates
/// output subsets nor permits a requester key to merge resident rows.
class SwitchResourceContract final {
public:
  static llvm::Expected<SwitchResourceContract>
  create(SwitchResourceDeclaration declaration);

  std::uint32_t inputCount() const { return inputCount_; }
  std::uint32_t outputCount() const { return outputCount_; }

  const ResourceContract &resourceContract() const { return contract_; }

  StateKey inputState(std::uint32_t input) const;
  StateKey outputState(std::uint32_t output) const;
  RequesterKey inputRequester(std::uint32_t input) const;

  /// Resolves the unique canonical use-pattern key for one admitted physical
  /// traversal. A disconnected or out-of-range pair is rejected.
  llvm::Expected<UsePatternKey> traversalPattern(std::uint32_t input,
                                                 std::uint32_t output) const;

private:
  SwitchResourceContract(Schedule schedule, std::uint32_t inputCount,
                         std::uint32_t outputCount,
                         std::vector<std::uint32_t> inputOffsets,
                         std::vector<std::uint32_t> outputsByInput,
                         ResourceContract contract)
      : schedule_(schedule), inputCount_(inputCount), outputCount_(outputCount),
        inputOffsets_(std::move(inputOffsets)),
        outputsByInput_(std::move(outputsByInput)),
        contract_(std::move(contract)) {}

  Schedule schedule_ = Schedule::Spatial;
  std::uint32_t inputCount_ = 0;
  std::uint32_t outputCount_ = 0;
  std::vector<std::uint32_t> inputOffsets_;
  std::vector<std::uint32_t> outputsByInput_;
  ResourceContract contract_;
};

/// Projects one verified fabric.switch occurrence into its complete
/// Mapping-visible resource contract. This is the only IR-to-contract
/// projection used by the op verifier and Fabric finalizer.
llvm::Expected<SwitchResourceContract>
deriveSwitchResourceContract(SwitchOp operation);

/// Resolves the unique use pattern of one traversal from an imported canonical
/// switch ResourceContract. This is the cold owner projection used by sealed
/// Fabric views; consumers must not reconstruct the pattern ordinal from
/// connectivity order.
llvm::Expected<UsePatternKey>
resolveSwitchTraversalPattern(const ResourceContract &contract,
                              std::uint32_t inputCount, std::uint32_t input,
                              std::uint32_t output);

} // namespace fabric

#endif // FABRIC_IR_SWITCHRESOURCECONTRACT_H
