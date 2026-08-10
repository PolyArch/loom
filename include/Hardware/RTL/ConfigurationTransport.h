#ifndef LOOM_HARDWARE_RTL_CONFIGURATIONTRANSPORT_H
#define LOOM_HARDWARE_RTL_CONFIGURATIONTRANSPORT_H

#include "Hardware/Configuration/ConfigurationABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::hardware::rtl {

inline constexpr std::uint32_t portableConfigurationAddressWidth = 32;
inline constexpr std::uint32_t portableConfigurationDataWidth = 32;
inline constexpr std::uint32_t portableConfigurationByteCount = 4;
inline constexpr llvm::StringLiteral portableConfigurationRuntimeAbiIdentity =
    "loom.runtime.portable_axi_lite.v1";

struct ConfigurationTransportUnitLayout final {
  ProgrammingUnitRef programmingUnit;
  std::uint64_t payloadBitCount = 0;
  std::uint64_t payloadByteCount = 0;
  std::uint64_t payloadWordCount = 0;
  std::uint32_t baseAddress = 0;
  std::uint32_t commitAddress = 0;
  std::uint32_t statusAddress = 0;
  std::vector<std::uint8_t> inactiveImage;
};

struct ConfigurationTransportLayout final {
  fabric::SpatialCoreOccurrenceRef spatialCore;
  std::vector<ConfigurationTransportUnitLayout> units;
  std::uint64_t byteSpan = 0;

  const ConfigurationTransportUnitLayout *find(ProgrammingUnitId unitId) const;
};

/// Derives the common 32-bit AXI4-Lite window from one exact ConfigurationABI.
/// The result is transient implementation metadata and is not an Artifact.
llvm::Expected<ConfigurationTransportLayout>
derivePortableConfigurationTransportLayout(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef spatialCore);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_CONFIGURATIONTRANSPORT_H
