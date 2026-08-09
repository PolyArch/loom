#ifndef LOOM_HARDWARE_IMPLEMENTATION_DEFPHYSICAL_H
#define LOOM_HARDWARE_IMPLEMENTATION_DEFPHYSICAL_H

#include "Hardware/Implementation/RepresentationFormat.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::hardware {

enum class DefSupplyUse : std::uint8_t {
  Power,
  Ground,
};

struct DefSpecialNet final {
  std::string name;
  std::optional<DefSupplyUse> use;
  bool routed = false;
};

struct DefTopLevelPin final {
  std::string name;
  std::string net;
  std::optional<DefSupplyUse> use;
  bool placedOrFixed = false;
  bool hasLayerGeometry = false;
};

struct DefPhysicalDesign final {
  std::string designName;
  std::vector<DefSpecialNet> specialNets;
  std::vector<DefTopLevelPin> topLevelPins;
};

struct DefSingleSupplyNetwork final {
  std::string powerNet;
  std::string groundNet;
};

/// Parses the provider-neutral DEF facts owned by indexed_def_physical. This
/// is intentionally not a general layout database: it validates the exact
/// design and section framing, then projects only stage and supply facts used
/// by owner validation and rail capability admission.
llvm::Expected<DefPhysicalDesign>
parseDefPhysicalDesign(llvm::StringRef contents, llvm::StringRef expectedTop,
                       RepresentationPhysicalStage stage);

/// Returns the sole complete connected supply network, or nullopt when the
/// valid DEF requires a multi-domain, partial-network, or external pad model.
std::optional<DefSingleSupplyNetwork>
deriveDefSingleSupplyNetwork(const DefPhysicalDesign &design);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_DEFPHYSICAL_H
