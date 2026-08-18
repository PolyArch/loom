#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <string>

namespace loom {
namespace {

llvm::StringRef spelling(InvocationDiagnosticStage stage) {
  switch (stage) {
  case InvocationDiagnosticStage::DataflowLowering:
    return "dataflow_lowering";
  case InvocationDiagnosticStage::TechMapping:
    return "tech_mapping";
  case InvocationDiagnosticStage::SpatialPnr:
    return "spatial_pnr";
  case InvocationDiagnosticStage::SystemPnr:
    return "system_pnr";
  case InvocationDiagnosticStage::HardwareConfiguration:
    return "hardware_configuration";
  case InvocationDiagnosticStage::Deployment:
    return "deployment";
  }
  llvm_unreachable("unknown invocation diagnostic stage");
}

llvm::StringRef spelling(InvocationDiagnosticEvent event) {
  switch (event) {
  case InvocationDiagnosticEvent::InvocationBegin:
    return "invocation_begin";
  case InvocationDiagnosticEvent::InvocationEnd:
    return "invocation_end";
  case InvocationDiagnosticEvent::Statistics:
    return "statistics";
  case InvocationDiagnosticEvent::Candidate:
    return "candidate";
  case InvocationDiagnosticEvent::Seed:
    return "seed";
  case InvocationDiagnosticEvent::NegotiationIteration:
    return "negotiation_iteration";
  case InvocationDiagnosticEvent::CapacityConflict:
    return "capacity_conflict";
  case InvocationDiagnosticEvent::ActionProposal:
    return "action_proposal";
  case InvocationDiagnosticEvent::ActionOutcome:
    return "action_outcome";
  case InvocationDiagnosticEvent::ContextChoice:
    return "context_choice";
  case InvocationDiagnosticEvent::NetRoute:
    return "net_route";
  case InvocationDiagnosticEvent::CutAnalysis:
    return "cut_analysis";
  case InvocationDiagnosticEvent::DerivedContext:
    return "derived_context";
  case InvocationDiagnosticEvent::TopologyQuality:
    return "topology_quality";
  case InvocationDiagnosticEvent::TagDomainPressure:
    return "tag_domain_pressure";
  case InvocationDiagnosticEvent::ArithmeticFailure:
    return "arithmetic_failure";
  case InvocationDiagnosticEvent::MappingFailure:
    return "mapping_failure";
  case InvocationDiagnosticEvent::ConfigurationAbiDerivation:
    return "configuration_abi_derivation";
  case InvocationDiagnosticEvent::ConfigurationAbiConstruction:
    return "configuration_abi_construction";
  case InvocationDiagnosticEvent::ConfigurationAbiImportSession:
    return "configuration_abi_import_session";
  case InvocationDiagnosticEvent::ConfigurationImageProjectionSession:
    return "configuration_image_projection_session";
  case InvocationDiagnosticEvent::SystemMappingImportSession:
    return "system_mapping_import_session";
  case InvocationDiagnosticEvent::ArtifactImportSession:
    return "artifact_import_session";
  case InvocationDiagnosticEvent::ApplicationBuildStatistics:
    return "application_build_statistics";
  case InvocationDiagnosticEvent::DeploymentConstructionStatistics:
    return "deployment_construction_statistics";
  case InvocationDiagnosticEvent::DeploymentPackageStatistics:
    return "deployment_package_statistics";
  }
  llvm_unreachable("unknown invocation diagnostic event");
}

struct OutputState final {
  std::mutex mutex;
  std::uint64_t nextSequence = 0;
};

OutputState &outputState() {
  static OutputState state;
  return state;
}

} // namespace

bool invocationDiagnosticEnabled(DiagnosticVerbosity minimum) {
  return diagnosticVerbosityEnabled(minimum);
}

void emitInvocationDiagnostic(
    DiagnosticVerbosity minimum, InvocationDiagnosticStage stage,
    InvocationDiagnosticEvent event,
    llvm::function_ref<llvm::json::Value()> buildPayload) {
  if (!invocationDiagnosticEnabled(minimum))
    return;

  llvm::json::Value payload = buildPayload();
  OutputState &state = outputState();
  std::lock_guard<std::mutex> lock(state.mutex);
  llvm::json::Object envelope;
  envelope["schema"] = "loom.invocation.diagnostic.1";
  envelope["level"] = static_cast<std::int64_t>(minimum);
  envelope["event"] = spelling(event);
  envelope["stage"] = spelling(stage);
  envelope["sequence"] = static_cast<std::int64_t>(state.nextSequence++);
  envelope["payload"] = std::move(payload);

  std::string line;
  llvm::raw_string_ostream stream(line);
  stream << llvm::json::Value(std::move(envelope));
  stream.flush();
  llvm::errs() << line << '\n';
}

} // namespace loom
