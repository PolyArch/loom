#include "Evaluation/Models/MappedRtlSimulation.h"

#include "Config/ResolvedConfig.h"
#include "Evaluation/ProductionRegistry.h"

#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr CaseSubjectRoleRef kHardwareRole(0);
constexpr CaseSubjectRoleRef kDeploymentRole(1);

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_simulation_invalid: " + detail);
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.evaluation.mapped_rtl_simulator.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &config) {
  if (!config.evaluation.mappedRtlSimulator)
    return invalid("HDL simulator provider binding is unavailable");
  if (llvm::Error error = validateMappedRtlSimulatorBinding(
          *config.evaluation.mappedRtlSimulator))
    return std::move(error);
  return OwnerValue::get(*config.evaluation.mappedRtlSimulator);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  const auto *binding = value.getIf<MappedRtlSimulatorBinding>();
  if (!binding)
    return invalid("config has the wrong owner type");
  if (llvm::Error error = validateMappedRtlSimulatorBinding(*binding))
    return std::move(error);
  return std::vector<std::uint8_t>(
      binding->stableHdlSimulatorBuildIdentity.begin(),
      binding->stableHdlSimulatorBuildIdentity.end());
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  MappedRtlSimulatorBinding binding{std::string(bytes.begin(), bytes.end())};
  if (llvm::Error error = validateMappedRtlSimulatorBinding(binding))
    return std::move(error);
  return OwnerValue::get(std::move(binding));
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

} // namespace

llvm::Error registerMappedRtlSimulationModel() {
  return registerProductionEvaluationRegistry();
}

EvaluationModelDescriptorRef mappedRtlSimulatorModelDescriptorRef() {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::MappedRtlSimulator));
}

CaseSubjectRoleRef mappedRtlHardwareImplementationSubjectRole() {
  return kHardwareRole;
}

CaseSubjectRoleRef mappedRtlDeploymentSubjectRole() { return kDeploymentRole; }

const ResolvedModelConfigViewContract &mappedRtlSimulationConfigViewContract() {
  return kConfigView;
}

llvm::Expected<MappedRtlSimulationConfiguration>
projectVerifiedMappedRtlSimulationConfiguration(
    const EvaluationRequest &request) {
  if (request.modelBinding().descriptorRef() !=
      mappedRtlSimulatorModelDescriptorRef())
    return invalid("request selects a foreign model descriptor");
  const auto *binding = request.modelBinding()
                            .resolvedModelConfig()
                            .getIf<MappedRtlSimulatorBinding>();
  if (!binding)
    return invalid("request does not carry the HDL simulator binding");
  if (llvm::Error error = validateMappedRtlSimulatorBinding(*binding))
    return std::move(error);
  return MappedRtlSimulationConfiguration{*binding};
}

llvm::Expected<MappedRtlSimulationConfiguration>
projectMappedRtlSimulationConfiguration(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  RequestVerifier verifier(resolution, artifacts, blobs);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  return projectVerifiedMappedRtlSimulationConfiguration(request);
}

} // namespace loom::evaluation::models
