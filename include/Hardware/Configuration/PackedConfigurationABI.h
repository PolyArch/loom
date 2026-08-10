#ifndef LOOM_HARDWARE_CONFIGURATION_PACKEDCONFIGURATIONABI_H
#define LOOM_HARDWARE_CONFIGURATION_PACKEDCONFIGURATIONABI_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace loom::hardware {

/// One explicit physical encoding choice for a Fabric field whose semantic
/// relation cannot be assigned by the finite packed profile.
struct PackedConfigurationFieldEncodingOverride final {
  fabric::FabricPhysicalConfigurationFieldRef field;
  SemanticFieldEncoding semanticEncoding;
  std::vector<std::uint8_t> inactiveValue;
};

/// Derives one occurrence-qualified ProgrammingUnit per configured
/// SpatialCore. Finite semantic domains receive a dense canonical codebook;
/// direct relations require an explicit override. Fields are packed in exact
/// Fabric inventory order and acquire no backend-local identity.
llvm::Expected<ConfigurationABIDraft> derivePackedConfigurationABIDraft(
    const fabric::FinalizedFabricRoot &system, mlir::MLIRContext &context,
    llvm::ArrayRef<PackedConfigurationFieldEncodingOverride> overrides = {});

} // namespace loom::hardware

#endif // LOOM_HARDWARE_CONFIGURATION_PACKEDCONFIGURATIONABI_H
