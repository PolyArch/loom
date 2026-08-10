#ifndef LOOM_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTION_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::fabric {
class FabricSystemRootView;
}

namespace loom::mapping {
class FinalizedSpatialMapping;
class FinalizedSystemMapping;
namespace detail {
struct ConfiguredHardwareProjectionViewAccess;
struct PhysicalConfiguredHardwareProjectionViewAccess;
} // namespace detail

struct ConfiguredHardwareFieldValueView final {
  ::loom::fabric::FabricConfigurationSlotRef slot;
  CanonicalSemanticBytes value;
};

/// Sealed, removable projection of the exact semantic configuration selected
/// by one complete Mapping. Fabric owns field meaning and value codecs;
/// Mapping owns only the selected physical slot and the unique derived value.
class ConfiguredHardwareProjectionView final {
public:
  llvm::ArrayRef<ConfiguredHardwareFieldValueView> fields() const {
    return fields_;
  }

private:
  explicit ConfiguredHardwareProjectionView(
      std::vector<ConfiguredHardwareFieldValueView> fields)
      : fields_(std::move(fields)) {}

  std::vector<ConfiguredHardwareFieldValueView> fields_;

  friend struct detail::ConfiguredHardwareProjectionViewAccess;
};

struct PhysicalConfiguredHardwareFieldValueView final {
  ::loom::fabric::FabricPhysicalConfigurationSlotRef slot;
  CanonicalSemanticBytes value;
};

/// Occurrence-qualified, removable projection of configuration values already
/// validated by the imported SpatialMapping set. This view only qualifies and
/// joins Mapping-owned values; it never reprojects actor or route semantics.
class PhysicalConfiguredHardwareProjectionView final {
public:
  llvm::ArrayRef<PhysicalConfiguredHardwareFieldValueView> fields() const {
    return fields_;
  }

private:
  explicit PhysicalConfiguredHardwareProjectionView(
      std::vector<PhysicalConfiguredHardwareFieldValueView> fields)
      : fields_(std::move(fields)) {}

  std::vector<PhysicalConfiguredHardwareFieldValueView> fields_;

  friend struct detail::PhysicalConfiguredHardwareProjectionViewAccess;
  friend llvm::Expected<PhysicalConfiguredHardwareProjectionView>
  qualifyConfiguredHardwareProjection(
      const FinalizedSpatialMapping &,
      const ::loom::fabric::FabricSystemRootView &,
      ::loom::fabric::SpatialCoreOccurrenceRef);
  friend llvm::Expected<PhysicalConfiguredHardwareProjectionView>
  deriveConfiguredHardwareProjection(const FinalizedSystemMapping &,
                                     const ArtifactStore &);
};

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
qualifyConfiguredHardwareProjection(
    const FinalizedSpatialMapping &mapping,
    const ::loom::fabric::FabricSystemRootView &system,
    ::loom::fabric::SpatialCoreOccurrenceRef occurrence);

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
deriveConfiguredHardwareProjection(const FinalizedSystemMapping &mapping,
                                   const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTION_H
