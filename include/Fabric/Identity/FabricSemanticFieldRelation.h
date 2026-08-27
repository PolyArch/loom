#ifndef LOOM_FABRIC_IDENTITY_FABRICSEMANTICFIELDRELATION_H
#define LOOM_FABRIC_IDENTITY_FABRICSEMANTICFIELDRELATION_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace loom::fabric {

class FabricArtifactView;

/// Canonical source identity of one semantic configuration relation within a
/// finalized Fabric. Operation occurrences that resolve to the same concrete
/// capability template share an identity; shape-dependent resources retain
/// their exact local field identity. Callers pair this value with the Fabric
/// artifact identity and their own schema/algorithm version.
llvm::Expected<CanonicalSemanticBytes>
semanticFieldRelationSourceIdentity(const FabricArtifactView &fabric,
                                    const FabricSemanticConfigFieldRef &field);

struct FabricBoundaryTagRewrite final {
  llvm::APInt inputTag;
  llvm::APInt outputTag;
};

/// The active payload of one boundary configuration. Absence at the codec
/// boundary denotes Disabled. Empty active payload is legal only for a
/// token-written or tag-removing boundary.
struct FabricBoundaryConfiguration final {
  std::optional<llvm::APInt> configuredTag;
  std::vector<FabricBoundaryTagRewrite> tagRewrites;
};

/// One active Temporal switch table row. Traversals sharing the tag form the
/// row's spatial crosspoint selection.
struct FabricTemporalSwitchRouteEntry final {
  llvm::APInt tag;
  std::vector<FabricPhysicalTraversalRef> selectedTraversals;
};

enum class FabricSemanticFieldRelationKind : std::uint8_t {
  None,
  Finite,
  Direct,
};

/// The one rebuildable semantic domain owned by a Fabric configuration field.
/// Physical codes and placement remain ConfigurationABI-owned.
class FabricSemanticFieldRelation final {
public:
  FabricSemanticFieldRelationKind kind() const { return kind_; }
  bool hasConfigurationField() const {
    return kind_ != FabricSemanticFieldRelationKind::None;
  }
  llvm::ArrayRef<CanonicalSemanticBytes> finiteDomain() const {
    return finiteDomain_;
  }
  std::optional<std::uint64_t> directEncodedBitCount() const {
    return kind_ == FabricSemanticFieldRelationKind::Direct
               ? std::optional<std::uint64_t>(directEncodedBitCount_)
               : std::nullopt;
  }
  const CanonicalSemanticBytes *canonicalInactiveValue() const {
    return canonicalInactiveValue_ ? &*canonicalInactiveValue_ : nullptr;
  }
  llvm::Error validateSemanticValue(llvm::ArrayRef<std::uint8_t> value) const;

private:
  using Validator = std::function<llvm::Error(llvm::ArrayRef<std::uint8_t>)>;

  FabricSemanticFieldRelation(FabricSemanticFieldRelationKind kind,
                              std::vector<CanonicalSemanticBytes> finiteDomain,
                              std::uint64_t directEncodedBitCount,
                              Validator validator = {},
                              std::optional<CanonicalSemanticBytes>
                                  canonicalInactiveValue = std::nullopt)
      : kind_(kind), finiteDomain_(std::move(finiteDomain)),
        directEncodedBitCount_(directEncodedBitCount),
        validator_(std::move(validator)),
        canonicalInactiveValue_(std::move(canonicalInactiveValue)) {
    if (!canonicalInactiveValue_ &&
        kind_ == FabricSemanticFieldRelationKind::Finite &&
        !finiteDomain_.empty())
      canonicalInactiveValue_ = finiteDomain_.front();
  }

  FabricSemanticFieldRelationKind kind_ = FabricSemanticFieldRelationKind::None;
  std::vector<CanonicalSemanticBytes> finiteDomain_;
  std::uint64_t directEncodedBitCount_ = 0;
  Validator validator_;
  std::optional<CanonicalSemanticBytes> canonicalInactiveValue_;

  friend class FabricArtifactView;
};

llvm::Expected<CanonicalSemanticBytes> encodeFabricFuConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricFuCapabilityTemplateRef> activeTemplate);

llvm::Expected<CanonicalSemanticBytes> encodeFabricFifoConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricFifoTraversalMode> activeMode);

/// Encodes the selected transfer-pattern controls of one programmable System
/// transport resource. An empty selection is the canonical disabled value.
llvm::Expected<CanonicalSemanticBytes>
encodeSystemTransportResourceConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    llvm::ArrayRef<FabricTransferPatternRef> selectedPatterns);

llvm::Expected<CanonicalSemanticBytes> encodeSpatialSwitchConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    llvm::ArrayRef<FabricPhysicalTraversalRef> selectedTraversals);

llvm::Expected<CanonicalSemanticBytes> encodeTemporalSwitchConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    llvm::ArrayRef<FabricTemporalSwitchRouteEntry> entries);

llvm::Expected<CanonicalSemanticBytes> encodeFabricBoundaryConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricBoundaryConfiguration> activeConfiguration);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICSEMANTICFIELDRELATION_H
