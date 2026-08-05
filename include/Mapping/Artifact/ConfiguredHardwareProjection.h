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

/// One independently configurable physical semantic field. The field owner is
/// occurrence-scoped; the instruction context distinguishes resident
/// configurations of a temporal resource. Both references are existing
/// Fabric identities, so this slot introduces no Mapping-local identity.
struct ConfiguredHardwareFieldSlotRef final {
  ::loom::fabric::InstructionContextRef context;
  ::loom::fabric::FabricSemanticConfigFieldRef field;

  friend bool operator==(const ConfiguredHardwareFieldSlotRef &lhs,
                         const ConfiguredHardwareFieldSlotRef &rhs) {
    return lhs.context == rhs.context && lhs.field == rhs.field;
  }
};

struct ConfiguredHardwareFieldValueView final {
  ConfiguredHardwareFieldSlotRef slot;
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
