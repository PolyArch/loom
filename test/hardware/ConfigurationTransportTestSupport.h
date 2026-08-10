#ifndef LOOM_TEST_HARDWARE_CONFIGURATIONTRANSPORTTESTSUPPORT_H
#define LOOM_TEST_HARDWARE_CONFIGURATIONTRANSPORTTESTSUPPORT_H

#include "Hardware/RTL/ConfigurationTransport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom::hardware::test {

struct PortableConfigurationTarget final {
  ProgrammingUnitId unitId = 0;
  std::uint64_t payloadBitCount = 0;
  std::uint64_t payloadByteCount = 0;
  std::uint64_t payloadWordCount = 0;
  std::uint32_t baseAddress = 0;
  std::uint32_t commitAddress = 0;
  std::uint32_t statusAddress = 0;
};

struct PortableConfigurationValue final {
  ProgrammingUnitId unitId = 0;
  SemanticConfigurationValue value;
};

llvm::Expected<PortableConfigurationTarget> derivePortableConfigurationTarget(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef spatialCore, ProgrammingUnitId unitId);

llvm::Expected<PortableConfigurationValue>
deriveSpatialSingleTemplateFuActivation(
    const fabric::FabricArtifactView &fabric,
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    fabric::FabricFuOccurrenceRef fu);

std::string portableAxiLiteSignalDeclarations();
std::string portableAxiLiteDriverTasks();
std::string portableAxiLiteInitialization();
std::string portableCycleWatchdog(std::uint64_t cycleLimit = 4096);

llvm::Expected<std::string>
portableAxiLiteProgramAndVerify(const PortableConfigurationTarget &target,
                                llvm::ArrayRef<std::uint8_t> image,
                                llvm::StringRef indentation = "    ");

} // namespace loom::hardware::test

#endif // LOOM_TEST_HARDWARE_CONFIGURATIONTRANSPORTTESTSUPPORT_H
