#ifndef LOOM_FABRIC_IDENTITY_FABRICHANDSHAKEINTERNAL_H
#define LOOM_FABRIC_IDENTITY_FABRICHANDSHAKEINTERNAL_H

#include "Fabric/Identity/FabricHandshake.h"

#include "llvm/ADT/Hashing.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace loom::fabric::detail {

struct CanonicalFabricByteKeyHash final {
  std::size_t operator()(const std::vector<std::uint8_t> &key) const {
    return static_cast<std::size_t>(
        llvm::hash_combine_range(key.begin(), key.end()));
  }
};

struct HandshakeStructuralFragment final {
  std::uint32_t contributionOffset = 0;
  std::uint32_t contributionCount = 0;
  HandshakeActivationKind activationKind =
      HandshakeActivationKind::ExactOwnerSelection;
  std::uint32_t witnessOffset = 0;
  std::uint32_t witnessCount = 0;
};

struct HandshakeStructuralTemplate final {
  std::vector<HandshakeOwnerNodeKind> nodeKinds;
  std::vector<HandshakeOwnerArc> arcs;
  std::vector<HandshakeStructuralFragment> fragments;
  std::vector<std::uint32_t> fragmentContributionOrdinals;
};

struct HandshakeOwnerModelInstance final {
  std::shared_ptr<const HandshakeStructuralTemplate> structure;
  std::shared_ptr<const std::vector<HandshakeOwnerNode>> nodeBindings;
  std::shared_ptr<const std::vector<FabricPhysicalTraversalRef>>
      traversalWitnessBindings;
  std::shared_ptr<const std::vector<HandshakeFragmentSelector>> selectors;
  std::optional<FabricSwitchHandshakeActivationKey> switchActivationOverride;
  std::uint32_t projectionShapeOrdinal = 0;
};

struct HandshakeOwnerModelInstanceLayout final {
  std::uint32_t nodeOffset = 0;
  std::uint32_t arcOffset = 0;
  std::uint32_t fragmentOffset = 0;
  std::uint32_t contributionOffset = 0;
  std::uint32_t witnessOffset = 0;
};

struct HandshakeOwnerModelStorage final {
  explicit HandshakeOwnerModelStorage(FabricHandshakeOwner owner)
      : owner(std::move(owner)) {}

  FabricHandshakeOwner owner;
  std::vector<HandshakeOwnerModelInstance> instances;
  std::vector<HandshakeOwnerModelInstanceLayout> layouts;
  std::uint32_t nodeCount = 0;
  std::uint32_t arcCount = 0;
  std::uint32_t fragmentCount = 0;
  std::uint32_t contributionCount = 0;
  std::uint32_t witnessCount = 0;
};

class HandshakeOwnerModelFactory final {
public:
  static llvm::Expected<HandshakeOwnerModel>
  rebindFuOccurrence(const FabricArtifactView &view,
                     const HandshakeOwnerModel &definitionModel,
                     FabricFuOccurrenceRef occurrence);

  static llvm::Expected<HandshakeOwnerModel>
  rebindMemoryOccurrence(const FabricArtifactView &view,
                         const HandshakeOwnerModel &definitionModel,
                         FabricMemoryOccurrenceRef occurrence);

  static llvm::Expected<HandshakeOwnerModel>
  instantiateSwitchRows(FabricSwitchOccurrenceRef occurrence,
                        llvm::ArrayRef<HandshakeOwnerModel> rowShapes,
                        std::uint64_t residentRows, bool temporal);

  static llvm::Expected<std::vector<HandshakeDependencyArc>>
  deriveUnconditionalDependencyArcs(llvm::ArrayRef<HandshakeOwnerModel> models);

  static void
  accumulateStatistics(llvm::ArrayRef<HandshakeOwnerModel> models,
                       FabricHandshakeContextStatistics &statistics);
};

class HandshakeOwnerModelBuilder final {
public:
  explicit HandshakeOwnerModelBuilder(FabricHandshakeOwner owner);

  std::uint32_t boundarySignal(HandshakeSignalRef signal);
  std::uint32_t junction(std::vector<std::uint8_t> ownerLocalKey);
  void addFragment(HandshakeFragmentSelector selector,
                   std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs);
  llvm::Expected<HandshakeOwnerModel> finish();

private:
  struct PendingFragment final {
    HandshakeFragmentSelector selector;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs;
  };

  FabricHandshakeOwner owner_;
  std::vector<HandshakeOwnerNode> nodes_;
  std::unordered_map<std::vector<std::uint8_t>, std::uint32_t,
                     CanonicalFabricByteKeyHash>
      boundaryNodes_;
  std::unordered_map<std::vector<std::uint8_t>, std::uint32_t,
                     CanonicalFabricByteKeyHash>
      junctionNodes_;
  std::vector<PendingFragment> pending_;
};

std::vector<std::uint8_t> handshakeSignalKey(const HandshakeSignalRef &signal);
std::vector<std::uint8_t> handshakeOwnerKey(const FabricHandshakeOwner &owner);
std::optional<FabricHandshakeOwner>
handshakeTraversalOwner(const FabricPhysicalTraversalRef &traversal);
void sortHandshakeDependencyArcs(std::vector<HandshakeDependencyArc> &arcs,
                                 bool deduplicate);
llvm::Error verifyMemoryInternalHandshakeClosure(
    llvm::ArrayRef<FabricMemoryHandshakeSelection> selections);

llvm::Expected<HandshakeOwnerModel>
compileFuHandshakeModel(const FabricArtifactView &view,
                        FabricFuOccurrenceRef occurrence);

} // namespace loom::fabric::detail

#endif // LOOM_FABRIC_IDENTITY_FABRICHANDSHAKEINTERNAL_H
