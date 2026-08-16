#ifndef LOOM_DEPLOYMENT_DEPLOYMENTDIAGNOSTICS_H
#define LOOM_DEPLOYMENT_DEPLOYMENTDIAGNOSTICS_H

#include <cstdint>

namespace loom::deployment {

enum class DeploymentConstructionMode : std::uint8_t {
  Build,
  Import,
};

enum class DeploymentConstructionOperation : std::uint8_t {
  StaticMemoryDerivation,
  InputCanonicalization,
  MappingOwnerImport,
  HardwareClosureValidation,
  ConfigurationImageDerivation,
  RuntimeImageDerivation,
  ExecutableClosureValidation,
  ArtifactFinalization,
};

struct DeploymentConstructionOperationStatistics final {
  DeploymentConstructionMode mode;
  DeploymentConstructionOperation operation;
  std::uint64_t durationNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
};

void emitDeploymentConstructionOperationStatistics(
    const DeploymentConstructionOperationStatistics &statistics);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENTDIAGNOSTICS_H
