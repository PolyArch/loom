#ifndef LOOM_DEPLOYMENT_DEPLOYMENTDIAGNOSTICS_H
#define LOOM_DEPLOYMENT_DEPLOYMENTDIAGNOSTICS_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

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
  std::optional<std::uint64_t> selfCpuNanoseconds;
  std::optional<std::uint64_t> childCpuNanoseconds;
};

llvm::StringRef deploymentConstructionModeName(DeploymentConstructionMode mode);
llvm::StringRef
deploymentConstructionOperationName(DeploymentConstructionOperation operation);

void emitDeploymentConstructionOperationStatistics(
    const DeploymentConstructionOperationStatistics &statistics);

/// Captures exact operation observations emitted on the current thread. It is
/// diagnostic state only and never changes Deployment construction behavior.
class DeploymentConstructionStatisticsSession final {
public:
  class Impl;

  DeploymentConstructionStatisticsSession();
  ~DeploymentConstructionStatisticsSession();

  DeploymentConstructionStatisticsSession(
      const DeploymentConstructionStatisticsSession &) = delete;
  DeploymentConstructionStatisticsSession &
  operator=(const DeploymentConstructionStatisticsSession &) = delete;

  std::vector<DeploymentConstructionOperationStatistics> statistics() const;

private:
  std::unique_ptr<Impl> impl_;
  Impl *previous_ = nullptr;
};

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENTDIAGNOSTICS_H
