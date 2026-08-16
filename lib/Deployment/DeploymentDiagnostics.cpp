#include "Deployment/DeploymentDiagnostics.h"

#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

namespace loom::deployment {
namespace {

llvm::StringRef spelling(DeploymentConstructionMode mode) {
  switch (mode) {
  case DeploymentConstructionMode::Build:
    return "build";
  case DeploymentConstructionMode::Import:
    return "import";
  }
  llvm_unreachable("unknown Deployment construction mode");
}

llvm::StringRef spelling(DeploymentConstructionOperation operation) {
  switch (operation) {
  case DeploymentConstructionOperation::StaticMemoryDerivation:
    return "static_memory_derivation";
  case DeploymentConstructionOperation::InputCanonicalization:
    return "input_canonicalization";
  case DeploymentConstructionOperation::MappingOwnerImport:
    return "mapping_owner_import";
  case DeploymentConstructionOperation::HardwareClosureValidation:
    return "hardware_closure_validation";
  case DeploymentConstructionOperation::ConfigurationImageDerivation:
    return "configuration_image_derivation";
  case DeploymentConstructionOperation::RuntimeImageDerivation:
    return "runtime_image_derivation";
  case DeploymentConstructionOperation::ExecutableClosureValidation:
    return "executable_closure_validation";
  case DeploymentConstructionOperation::ArtifactFinalization:
    return "artifact_finalization";
  }
  llvm_unreachable("unknown Deployment construction operation");
}

} // namespace

void emitDeploymentConstructionOperationStatistics(
    const DeploymentConstructionOperationStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::DeploymentConstructionStatistics, [&] {
        llvm::json::Object payload;
        payload["mode"] = spelling(statistics.mode);
        payload["operation"] = spelling(statistics.operation);
        payload["duration_ns"] = statistics.durationNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
        return llvm::json::Value(std::move(payload));
      });
}

} // namespace loom::deployment
