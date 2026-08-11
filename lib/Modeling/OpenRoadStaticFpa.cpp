#include "Evaluation/Models/OpenRoadStaticFpa.h"

#include "Config/ResolvedConfig.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr ScopeFormRef kWholeCaseScope(0);

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "openroad_static_fpa_invalid: " + detail);
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.evaluation.openroad_routed_static_fpa.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &config) {
  if (!config.evaluation.openRoadStaticFpa)
    return invalid("OpenROAD provider binding is unavailable");
  if (llvm::Error error = validateOpenRoadStaticFpaProviderBinding(
          *config.evaluation.openRoadStaticFpa))
    return std::move(error);
  return OwnerValue::get(*config.evaluation.openRoadStaticFpa);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  const auto *binding = value.getIf<OpenRoadStaticFpaProviderBinding>();
  if (!binding)
    return invalid("config has the wrong provider-binding owner type");
  if (llvm::Error error = validateOpenRoadStaticFpaProviderBinding(*binding))
    return std::move(error);
  return std::vector<std::uint8_t>(binding->stableProviderBuildIdentity.begin(),
                                   binding->stableProviderBuildIdentity.end());
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  OpenRoadStaticFpaProviderBinding binding{
      std::string(bytes.begin(), bytes.end())};
  if (llvm::Error error = validateOpenRoadStaticFpaProviderBinding(binding))
    return std::move(error);
  return OwnerValue::get(std::move(binding));
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

bool isExactHardwareRootTarget(const SubjectTargetRef &target,
                               const ArtifactRootReference &hardware) {
  const auto *root = std::get_if<ArtifactRootReference>(&target.target);
  return target.caseSubjectRole ==
             hardwareImplementationPhysicalSubjectRole() &&
         target.anchorSubjectArtifact == hardware && root && *root == hardware;
}

llvm::Expected<std::vector<MetricKind>>
validateMetricRequests(const EvaluationRequest &request) {
  constexpr std::array<MetricKind, 4> supported{
      MetricKind::LimitingClockFrequency, MetricKind::TotalArea,
      MetricKind::DynamicPower, MetricKind::LeakagePower};
  if (request.metricRequests().empty() || !request.findingRequests().empty())
    return invalid("request must contain a nonempty FPA metric subset");
  std::array<bool, supported.size()> seen{};
  std::vector<MetricKind> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().scope.form != kWholeCaseScope ||
        !metric.query().scope.targets.empty() || !metric.conditions().empty())
      return invalid("FPA metrics must use an unconditioned whole-case scope");
    const auto found = llvm::find(supported, metric.query().metric);
    if (found == supported.end())
      return invalid("request contains a metric outside the FPA contract");
    const std::size_t ordinal =
        static_cast<std::size_t>(found - supported.begin());
    if (seen[ordinal])
      return invalid("request contains a duplicate FPA metric");
    seen[ordinal] = true;
    metrics.push_back(metric.query().metric);
  }
  return metrics;
}

} // namespace

llvm::Error registerOpenRoadStaticFpaModel() {
  return registerProductionEvaluationRegistry();
}

EvaluationModelDescriptorRef openRoadStaticFpaModelDescriptorRef() {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::OpenRoadRoutedStaticFpa));
}

const ResolvedModelConfigViewContract &openRoadStaticFpaConfigViewContract() {
  return kConfigView;
}

llvm::Expected<PreparedOpenRoadStaticFpaEvaluation>
prepareOpenRoadStaticFpaEvaluation(
    const ArtifactRootReference &hardwareImplementation,
    llvm::ArrayRef<EvaluationCondition> conditions,
    llvm::ArrayRef<MetricKind> metrics, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerOpenRoadStaticFpaModel())
    return std::move(error);
  auto externalContracts = eda::makeKnownAsicStandardCellContractCatalog();
  if (!externalContracts)
    return externalContracts.takeError();
  auto resolution = resolveHardwareImplementationPhysicalCase(
      hardwareImplementation, *externalContracts, artifactStore, blobStore);
  if (!resolution)
    return resolution.takeError();
  auto subjects = EvaluationSubjectBindings::get(
      {{hardwareImplementationPhysicalSubjectRole(),
        {hardwareImplementation}}});
  if (!subjects)
    return subjects.takeError();
  auto evaluationCase =
      EvaluationCase::get(hardwareImplementationPhysicalCaseSignatureRef(),
                          std::move(*subjects), std::nullopt, std::nullopt,
                          conditions, *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  std::vector<MetricRequest> requests;
  requests.reserve(metrics.size());
  for (MetricKind metric : metrics) {
    auto request =
        MetricRequest::get({metric, EvaluationScope{kWholeCaseScope, {}}}, {},
                           *evaluationCase, *resolution, artifactStore);
    if (!request)
      return request.takeError();
    requests.push_back(std::move(*request));
  }
  auto binding = ResolvedModelBinding::project(
      openRoadStaticFpaModelDescriptorRef(), {}, config);
  if (!binding)
    return binding.takeError();
  auto request =
      EvaluationRequest::get(*evaluationCase, requests, {}, std::move(*binding),
                             0, *resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  return PreparedOpenRoadStaticFpaEvaluation{
      std::move(*request), std::move(*resolution),
      hardwareImplementationPhysicalSubjectRole()};
}

llvm::Expected<CompleteOpenRoadStaticFpaConfiguration>
projectCompleteOpenRoadStaticFpaConfiguration(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  if (request.modelBinding().descriptorRef() !=
      openRoadStaticFpaModelDescriptorRef())
    return invalid("request selects a foreign model descriptor");
  const auto *providerBinding = request.modelBinding()
                                    .resolvedModelConfig()
                                    .getIf<OpenRoadStaticFpaProviderBinding>();
  if (!providerBinding)
    return invalid("request does not carry the OpenROAD provider binding");
  if (llvm::Error error =
          validateOpenRoadStaticFpaProviderBinding(*providerBinding))
    return std::move(error);
  auto metrics = validateMetricRequests(request);
  if (!metrics)
    return metrics.takeError();

  const auto hardwareSubjects = request.subjectBindings().subjects(
      hardwareImplementationPhysicalSubjectRole());
  if (hardwareSubjects.size() != 1)
    return invalid("request requires one HardwareImplementation subject");
  const ArtifactRootReference &hardware = hardwareSubjects.front();

  const ProcessCornerCondition *processCorner = nullptr;
  const SupplyVoltageCondition *supplyVoltage = nullptr;
  const TemperatureCondition *temperature = nullptr;
  const RequiredClockPeriodCondition *clockPeriod = nullptr;
  const ActivityBindingCondition *activity = nullptr;
  for (const EvaluationCondition &condition : request.baseConditions()) {
    if (const auto *value =
            std::get_if<ProcessCornerCondition>(&condition.payload))
      processCorner = value;
    else if (const auto *value =
                 std::get_if<SupplyVoltageCondition>(&condition.payload))
      supplyVoltage = value;
    else if (const auto *value =
                 std::get_if<TemperatureCondition>(&condition.payload))
      temperature = value;
    else if (const auto *value =
                 std::get_if<RequiredClockPeriodCondition>(&condition.payload))
      clockPeriod = value;
    else if (const auto *value =
                 std::get_if<ActivityBindingCondition>(&condition.payload))
      activity = value;
  }
  if (!processCorner || !supplyVoltage || !temperature || !clockPeriod ||
      request.baseConditions().size() != (activity ? 5 : 4))
    return invalid("request conditions do not form one complete FPA input");
  if (!isExactHardwareRootTarget(processCorner->target, hardware) ||
      !isExactHardwareRootTarget(supplyVoltage->powerDomain, hardware) ||
      !isExactHardwareRootTarget(temperature->thermalDomainOrRoot, hardware) ||
      !isExactHardwareRootTarget(clockPeriod->clockDomain, hardware))
    return invalid("FPA conditions must target the exact global subject");

  std::optional<ExplicitAssumptionSource> assumption;
  if (activity) {
    if (!isExactHardwareRootTarget(activity->target, hardware))
      return invalid("FPA activity must target the exact global subject");
    if (const auto *explicitSource =
            std::get_if<ExplicitAssumptionSource>(&activity->source)) {
      if (!isExactHardwareRootTarget(explicitSource->clockDomain, hardware) ||
          clockPeriod->clockDomain != explicitSource->clockDomain)
        return invalid("FPA activity must use the requested global clock");
      assumption = *explicitSource;
    }
  }

  return CompleteOpenRoadStaticFpaConfiguration{
      *providerBinding,
      *processCorner,
      *supplyVoltage,
      *temperature,
      *clockPeriod,
      activity ? std::optional<ActivityBindingCondition>(*activity)
               : std::nullopt,
      std::move(assumption),
      std::move(*metrics)};
}

} // namespace loom::evaluation::models
