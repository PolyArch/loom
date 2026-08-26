#include "Deployment/DeploymentDiagnostics.h"

#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

namespace loom::deployment {
namespace {

thread_local DeploymentConstructionStatisticsSession::Impl *currentSession;

} // namespace

class DeploymentConstructionStatisticsSession::Impl final {
public:
  void record(const DeploymentConstructionOperationStatistics &statistics) {
    records_.push_back(statistics);
  }

  std::vector<DeploymentConstructionOperationStatistics> statistics() const {
    return records_;
  }

private:
  std::vector<DeploymentConstructionOperationStatistics> records_;
};

llvm::StringRef
deploymentConstructionModeName(DeploymentConstructionMode mode) {
  switch (mode) {
  case DeploymentConstructionMode::Build:
    return "build";
  case DeploymentConstructionMode::Import:
    return "import";
  }
  llvm_unreachable("unknown Deployment construction mode");
}

llvm::StringRef
deploymentConstructionOperationName(DeploymentConstructionOperation operation) {
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

void emitDeploymentConstructionOperationStatistics(
    const DeploymentConstructionOperationStatistics &statistics) {
  if (currentSession)
    currentSession->record(statistics);
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::DeploymentConstructionStatistics, [&] {
        llvm::json::Object payload;
        payload["mode"] = deploymentConstructionModeName(statistics.mode);
        payload["operation"] =
            deploymentConstructionOperationName(statistics.operation);
        payload["duration_ns"] = statistics.durationNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
        if (statistics.selfCpuNanoseconds)
          payload["self_cpu_ns"] = *statistics.selfCpuNanoseconds;
        if (statistics.childCpuNanoseconds)
          payload["child_cpu_ns"] = *statistics.childCpuNanoseconds;
        return llvm::json::Value(std::move(payload));
      });
}

DeploymentConstructionStatisticsSession::
    DeploymentConstructionStatisticsSession()
    : impl_(std::make_unique<Impl>()), previous_(currentSession) {
  currentSession = impl_.get();
}

DeploymentConstructionStatisticsSession::
    ~DeploymentConstructionStatisticsSession() {
  currentSession = previous_;
}

std::vector<DeploymentConstructionOperationStatistics>
DeploymentConstructionStatisticsSession::statistics() const {
  return impl_ ? impl_->statistics()
               : std::vector<DeploymentConstructionOperationStatistics>{};
}

} // namespace loom::deployment
