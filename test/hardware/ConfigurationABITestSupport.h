#ifndef LOOM_TEST_HARDWARE_CONFIGURATIONABITESTSUPPORT_H
#define LOOM_TEST_HARDWARE_CONFIGURATIONABITESTSUPPORT_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::hardware::test {

using ConfigurationFieldEncodingOverride =
    PackedConfigurationFieldEncodingOverride;

llvm::Expected<fabric::FinalizedFabricRoot>
makeSingleSpatialCoreSystem(const fabric::FinalizedFabricRoot &module,
                            const ArtifactStore &store);

llvm::Expected<fabric::SpatialCoreOccurrenceRef>
requireSingleSpatialCoreOccurrence(
    const fabric::FinalizedFabricRoot &system);

llvm::Expected<fabric::FinalizedFabricRoot>
makeSpatialCoreSystem(const fabric::FinalizedFabricRoot &module,
                      const ArtifactStore &store,
                      std::uint64_t spatialCoreCount);

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyPhysicalConfigurationField(
    const fabric::FabricPhysicalOccurrenceOwnerRef &owner,
    fabric::FabricOrdinal fieldOrdinal);

llvm::Expected<ConfigurationABIDraft> makeCompleteConfigurationABIDraft(
    const fabric::FinalizedFabricRoot &system,
    llvm::ArrayRef<ConfigurationFieldEncodingOverride> overrides = {});

} // namespace loom::hardware::test

#endif // LOOM_TEST_HARDWARE_CONFIGURATIONABITESTSUPPORT_H
