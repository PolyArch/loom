#ifndef LOOM_APPLICATION_BUILDDIAGNOSTICS_H
#define LOOM_APPLICATION_BUILDDIAGNOSTICS_H

#include <cstdint>

namespace loom::application {

enum class ApplicationBuildOperation : std::uint8_t {
  ProductTargetPreparation,
  FinalLinkImport,
  ApplicationPreparation,
  MappingExecution,
  MappingImport,
  ConfigurationAbiDerivation,
  HardwareBindingDerivation,
  CompilerTargetResolution,
  HostProgramFinalization,
  InstructionBinaryFinalization,
  DeclarativeDeploymentFinalization,
  DeploymentConstruction,
  PackagePublication,
};

struct ApplicationBuildOperationStatistics final {
  ApplicationBuildOperation operation;
  std::uint64_t durationNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
};

void emitApplicationBuildOperationStatistics(
    const ApplicationBuildOperationStatistics &statistics);

} // namespace loom::application

#endif // LOOM_APPLICATION_BUILDDIAGNOSTICS_H
