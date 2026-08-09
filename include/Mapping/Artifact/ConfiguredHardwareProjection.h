#ifndef LOOM_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTION_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace detail {
struct ConfiguredHardwareProjectionViewAccess;
}

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

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTION_H
