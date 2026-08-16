#include "Application/BuildDiagnostics.h"

#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

namespace loom::application {
namespace {

llvm::StringRef spelling(ApplicationBuildOperation operation) {
  switch (operation) {
  case ApplicationBuildOperation::ProductTargetPreparation:
    return "product_target_preparation";
  case ApplicationBuildOperation::FinalLinkImport:
    return "final_link_import";
  case ApplicationBuildOperation::ApplicationPreparation:
    return "application_preparation";
  case ApplicationBuildOperation::MappingExecution:
    return "mapping_execution";
  case ApplicationBuildOperation::MappingImport:
    return "mapping_import";
  case ApplicationBuildOperation::ConfigurationAbiDerivation:
    return "configuration_abi_derivation";
  case ApplicationBuildOperation::HardwareBindingDerivation:
    return "hardware_binding_derivation";
  case ApplicationBuildOperation::CompilerTargetResolution:
    return "compiler_target_resolution";
  case ApplicationBuildOperation::HostProgramFinalization:
    return "host_program_finalization";
  case ApplicationBuildOperation::InstructionBinaryFinalization:
    return "instruction_binary_finalization";
  case ApplicationBuildOperation::DeclarativeDeploymentFinalization:
    return "declarative_deployment_finalization";
  case ApplicationBuildOperation::DeploymentConstruction:
    return "deployment_construction";
  case ApplicationBuildOperation::PackagePublication:
    return "package_publication";
  }
  llvm_unreachable("unknown application build operation");
}

} // namespace

void emitApplicationBuildOperationStatistics(
    const ApplicationBuildOperationStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::ApplicationBuildStatistics, [&] {
        llvm::json::Object payload;
        payload["operation"] = spelling(statistics.operation);
        payload["duration_ns"] = statistics.durationNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
        return llvm::json::Value(std::move(payload));
      });
}

} // namespace loom::application
