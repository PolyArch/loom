#ifndef LOOM_FABRIC_IDENTITY_FABRICSEMANTICFIELDRELATION_H
#define LOOM_FABRIC_IDENTITY_FABRICSEMANTICFIELDRELATION_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace loom::fabric {

class FabricArtifactView;

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
  llvm::Error validateSemanticValue(llvm::ArrayRef<std::uint8_t> value) const;

private:
  using Validator = std::function<llvm::Error(llvm::ArrayRef<std::uint8_t>)>;

  FabricSemanticFieldRelation(FabricSemanticFieldRelationKind kind,
                              std::vector<CanonicalSemanticBytes> finiteDomain,
                              std::uint64_t directEncodedBitCount,
                              Validator validator = {})
      : kind_(kind), finiteDomain_(std::move(finiteDomain)),
        directEncodedBitCount_(directEncodedBitCount),
        validator_(std::move(validator)) {}

  FabricSemanticFieldRelationKind kind_ = FabricSemanticFieldRelationKind::None;
  std::vector<CanonicalSemanticBytes> finiteDomain_;
  std::uint64_t directEncodedBitCount_ = 0;
  Validator validator_;

  friend class FabricArtifactView;
};

llvm::Expected<CanonicalSemanticBytes> encodeFabricFuConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricFuCapabilityTemplateRef> activeTemplate);

llvm::Expected<CanonicalSemanticBytes> encodeFabricFifoConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricFifoTraversalMode> activeMode);

llvm::Expected<CanonicalSemanticBytes> encodeSpatialSwitchConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    llvm::ArrayRef<FabricPhysicalTraversalRef> selectedTraversals);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICSEMANTICFIELDRELATION_H
