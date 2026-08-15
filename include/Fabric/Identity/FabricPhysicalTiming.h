#ifndef LOOM_FABRIC_IDENTITY_FABRICPHYSICALTIMING_H
#define LOOM_FABRIC_IDENTITY_FABRICPHYSICALTIMING_H

#include "Common/ComponentViewDigest.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::fabric {

class FabricSystemRootView;

inline constexpr ArtifactSchemaDescriptor
    fabricPhysicalTimingProfileArtifactSchema{
        "loom.fabric.physical_timing_profile", SchemaVersion{1, 0}};

/// Whether one provider-owned traversal timing record continues the current
/// combinational path or starts a new registered segment at its destination.
enum class FabricPhysicalTimingBoundaryKind : std::uint8_t {
  Combinational,
  RegisteredDestination,
};

/// Evidence strength of one provider-owned physical timing profile. A
/// normalized heuristic can guide Mapping but cannot support target frequency
/// or routed timing claims. A target characterization names the exact
/// technology and characterization dataset from which its delay primitives
/// were derived.
enum class FabricPhysicalTimingProfileKind : std::uint8_t {
  NormalizedHeuristic,
  TargetCharacterization,
};

/// Physical delay supplied by one exact timing provider for one exact Fabric
/// traversal. Delay is expressed in that provider profile's integer quanta;
/// it is deliberately independent of ResourceContract cycle timing.
struct FabricTraversalPhysicalTiming final {
  FabricPhysicalTraversalRef traversal;
  std::uint64_t delayQuanta = 0;
  FabricPhysicalTimingBoundaryKind boundary =
      FabricPhysicalTimingBoundaryKind::Combinational;
};

/// Invocation-frozen physical timing input for Spatial Mapping. The profile is
/// a provider-owned component view, not a Fabric Artifact field and not EDA
/// Evidence. The exact Fabric identity and digest make every finite search
/// result replayable when the provider model changes.
class FabricPhysicalTimingProfileView final {
public:
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  FabricPhysicalTimingProfileKind kind() const { return kind_; }
  bool isNormalizedHeuristic() const {
    return kind_ == FabricPhysicalTimingProfileKind::NormalizedHeuristic;
  }
  llvm::StringRef providerIdentity() const { return providerIdentity_; }
  llvm::StringRef technologyIdentity() const { return technologyIdentity_; }
  llvm::StringRef characterizationIdentity() const {
    return characterizationIdentity_;
  }
  std::uint64_t requiredCombinationalDelayQuanta() const {
    return requiredCombinationalDelayQuanta_;
  }
  llvm::ArrayRef<FabricTraversalPhysicalTiming> traversals() const {
    return traversals_;
  }
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes() const;
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  FabricPhysicalTimingProfileView(
      ArtifactIdentity fabricIdentity, FabricPhysicalTimingProfileKind kind,
      std::string providerIdentity, std::string technologyIdentity,
      std::string characterizationIdentity,
      std::uint64_t requiredCombinationalDelayQuanta,
      std::vector<FabricTraversalPhysicalTiming> traversals,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : fabricIdentity_(std::move(fabricIdentity)), kind_(kind),
        providerIdentity_(std::move(providerIdentity)),
        technologyIdentity_(std::move(technologyIdentity)),
        characterizationIdentity_(std::move(characterizationIdentity)),
        requiredCombinationalDelayQuanta_(requiredCombinationalDelayQuanta),
        traversals_(std::move(traversals)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  ArtifactIdentity fabricIdentity_;
  FabricPhysicalTimingProfileKind kind_ =
      FabricPhysicalTimingProfileKind::NormalizedHeuristic;
  std::string providerIdentity_;
  std::string technologyIdentity_;
  std::string characterizationIdentity_;
  std::uint64_t requiredCombinationalDelayQuanta_ = 0;
  std::vector<FabricTraversalPhysicalTiming> traversals_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<FabricPhysicalTimingProfileView>
  projectNormalizedFabricPhysicalTimingProfile(
      const FabricArtifactView &fabric);
  friend llvm::Expected<FabricPhysicalTimingProfileView>
  createFabricPhysicalTimingProfile(
      const FabricArtifactView &fabric, FabricPhysicalTimingProfileKind kind,
      llvm::StringRef providerIdentity, llvm::StringRef technologyIdentity,
      llvm::StringRef characterizationIdentity,
      std::uint64_t requiredCombinationalDelayQuanta,
      llvm::ArrayRef<FabricTraversalPhysicalTiming> traversals);
};

llvm::ArrayRef<std::uint8_t> fabricPhysicalTimingProfileSchemaDescriptorBytes();

/// Finalizes one exact provider result. The Fabric owns the traversal domain;
/// the provider owns delay, boundary, technology, and characterization facts.
/// The result is canonicalized and its digest is mechanically derived.
llvm::Expected<FabricPhysicalTimingProfileView>
createFabricPhysicalTimingProfile(
    const FabricArtifactView &fabric, FabricPhysicalTimingProfileKind kind,
    llvm::StringRef providerIdentity, llvm::StringRef technologyIdentity,
    llvm::StringRef characterizationIdentity,
    std::uint64_t requiredCombinationalDelayQuanta,
    llvm::ArrayRef<FabricTraversalPhysicalTiming> traversals);

/// Projects the versioned target-neutral timing provider used when an exact
/// target timing provider is not bound. One quantum is one eighth of its
/// normalized clock budget. This is deterministic routing guidance and never
/// claims target frequency or post-route slack evidence.
llvm::Expected<FabricPhysicalTimingProfileView>
projectNormalizedFabricPhysicalTimingProfile(const FabricArtifactView &fabric);

/// Explicitly projects one normalized profile for every distinct SpatialCore
/// Module attached to a System. Callers choose this heuristic deliberately;
/// no Mapping or replay entry point invokes it as a fallback.
llvm::Expected<std::vector<FabricPhysicalTimingProfileView>>
projectNormalizedSystemPhysicalTimingProfiles(
    const FabricSystemRootView &system);

/// Publishes one exact provider-owned timing profile as an ordinary immutable
/// Artifact. This is the invocation input used by Mapping and System replay;
/// it is neither a Fabric field nor EDA evidence.
llvm::Expected<ArtifactRootReference> publishFabricPhysicalTimingProfile(
    const FabricPhysicalTimingProfileView &profile,
    const ArtifactStore &store);

/// Imports and strictly reconstructs one timing profile against its exact
/// Fabric owner. Canonical bytes, embedded owner identity, traversal domain,
/// and component digest are all re-derived before the view is returned.
llvm::Expected<FabricPhysicalTimingProfileView>
importFabricPhysicalTimingProfile(const ArtifactRootReference &reference,
                                  const FabricArtifactView &fabric,
                                  const ArtifactStore &store);

/// Reads the exact Fabric owner carried by a stored profile. Full import still
/// requires that owner Fabric and remains the strict validation boundary.
llvm::Expected<ArtifactIdentity> resolveFabricPhysicalTimingProfileOwner(
    const ArtifactRootReference &reference, const ArtifactStore &store);

llvm::Error validateFabricPhysicalTimingProfile(
    const FabricArtifactView &fabric,
    const FabricPhysicalTimingProfileView &profile);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICPHYSICALTIMING_H
