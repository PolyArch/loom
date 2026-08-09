#include "Evaluation/Models/PhysicalRailAnalysis.h"

#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/Support/Error.h"

#include <array>
#include <system_error>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(5);
constexpr EvaluationModelKind kModelKind(12);
constexpr CaseSubjectRoleRef kHardwareRole(0);
constexpr ScopeFormRef kWholeCaseScope(0);

llvm::Error railError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "physical_rail_analysis_invalid: " + detail);
}

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), kCaseKind));
}

SubjectReferenceType hardwareRootType() {
  return SubjectReferenceType{
      ArtifactRootType{hardware::hardwareImplementationSchema}};
}

SubjectTargetPattern hardwareRootPattern() {
  return SubjectTargetPattern{kHardwareRole, hardwareRootType()};
}

const ArtifactSchemaDescriptor *const kHardwareSchemas[] = {
    &hardware::hardwareImplementationSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {{
    kHardwareRole,
    "hardware_implementation",
    SubjectRoleCardinality::ExactlyOne,
    kHardwareSchemas,
    nullptr,
}};

const std::vector<ConditionApplicabilityPattern> kBaseConditionPatterns = {
    {EvaluationConditionKind::ProcessCorner,
     {caseSignatureRef(), {hardwareRootPattern()}}},
    {EvaluationConditionKind::SupplyVoltage,
     {caseSignatureRef(), {hardwareRootPattern()}}},
    {EvaluationConditionKind::Temperature,
     {caseSignatureRef(), {hardwareRootPattern()}}},
    {EvaluationConditionKind::RequiredClockPeriod,
     {caseSignatureRef(), {hardwareRootPattern()}}},
    {EvaluationConditionKind::ActivityBinding,
     {caseSignatureRef(), {hardwareRootPattern()}}},
    {EvaluationConditionKind::ActivityBinding,
     {caseSignatureRef(), {hardwareRootPattern(), hardwareRootPattern()}}},
};

const EvaluationCaseSignatureDescriptor kCaseSignature{
    kCaseKind,
    "hardware_implementation_physical",
    "One exact HardwareImplementation analyzed under exact physical operating "
    "conditions.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    kBaseConditionPatterns,
};

const std::array<ModelConditionCapability, 4> kConditionCapabilities = {{
    {kBaseConditionPatterns[0], ConditionDisposition::Required},
    {kBaseConditionPatterns[1], ConditionDisposition::Required},
    {kBaseConditionPatterns[3], ConditionDisposition::Required},
    {kBaseConditionPatterns[5], ConditionDisposition::Required},
}};

const ScopeFormRef kWholeCaseScopes[] = {kWholeCaseScope};
const MetricCapability kMetricCapabilities[] = {{
    MetricKind::MaximumVoltageDrop,
    kWholeCaseScopes,
    observationFormMask(ObservationForm::Point),
}};

const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::PhysicalImplementation};

constexpr RailAnalysisModelConfig kModelConfig{
    RailAnalysisMethod::Static,
    RailActivityBasis::ExplicitAssumption,
    RailNetworkCoverage::CompleteAnalyzedNetwork,
    UncertaintyKind::ExactWithinModel,
};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.evaluation.static_explicit_rail.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(kModelConfig);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  const auto *config = value.getIf<RailAnalysisModelConfig>();
  if (!config || *config != kModelConfig)
    return railError("config has the wrong owner type or fixed value");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue>
adoptConfig(llvm::ArrayRef<std::uint8_t> canonicalBytes,
            const ComponentViewDigest &) {
  if (!canonicalBytes.empty())
    return railError("fixed config view must be empty");
  return OwnerValue::get(kModelConfig);
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    kModelKind,
    "cadence_voltus_static_rail",
    cadenceVoltusRailImplementationSemanticIdentity,
    caseSignatureRef(),
    kConditionCapabilities,
    kMetricCapabilities,
    {},
    {},
    {},
    kConfigView,
    kModeledPhenomena,
    EvaluationExecutionMethod::ToolMeasurement,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::ExternalPrepareImport,
};

bool isExactHardwareRootTarget(const SubjectTargetRef &target,
                               const ArtifactRootReference &hardware) {
  const auto *root = std::get_if<ArtifactRootReference>(&target.target);
  return target.caseSubjectRole == kHardwareRole &&
         target.anchorSubjectArtifact == hardware && root && *root == hardware;
}

} // namespace

llvm::Error registerCadenceVoltusStaticRailModel() {
  if (llvm::Error error =
          platform::registerImplementationPlatformLocalReferenceKinds())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  return registerEvaluationModelDescriptor(kModelDescriptor);
}

EvaluationCaseSignatureRef hardwareImplementationPhysicalCaseSignatureRef() {
  return caseSignatureRef();
}

EvaluationModelDescriptorRef cadenceVoltusStaticRailModelDescriptorRef() {
  return kModelDescriptor.reference();
}

CaseSubjectRoleRef hardwareImplementationPhysicalSubjectRole() {
  return kHardwareRole;
}

const RailAnalysisModelConfig &staticExplicitRailAnalysisModelConfig() {
  return kModelConfig;
}

llvm::Expected<CompleteRailAnalysisConfiguration>
projectCompleteRailAnalysisConfiguration(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore) {
  RequestVerifier verifier(resolution, artifactStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  if (request.modelBinding().descriptorRef() != kModelDescriptor.reference())
    return railError("request selects a foreign model descriptor");

  const auto *modelConfig = request.modelBinding()
                                .resolvedModelConfig()
                                .getIf<RailAnalysisModelConfig>();
  if (!modelConfig || *modelConfig != kModelConfig)
    return railError("request does not carry the fixed rail model config");

  const auto hardwareSubjects =
      request.subjectBindings().subjects(kHardwareRole);
  if (hardwareSubjects.size() != 1)
    return railError("request requires one HardwareImplementation subject");
  const ArtifactRootReference &hardware = hardwareSubjects.front();

  if (request.metricRequests().size() != 1 ||
      !request.findingRequests().empty())
    return railError("request requires only one rail metric");
  const MetricRequest &metric = request.metricRequests().front();
  if (metric.query().metric != MetricKind::MaximumVoltageDrop ||
      metric.query().scope.form != kWholeCaseScope ||
      !metric.query().scope.targets.empty() || !metric.conditions().empty())
    return railError("request does not select whole-case maximum voltage drop");

  const ProcessCornerCondition *processCorner = nullptr;
  const SupplyVoltageCondition *supplyVoltage = nullptr;
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
                 std::get_if<RequiredClockPeriodCondition>(&condition.payload))
      clockPeriod = value;
    else if (const auto *value =
                 std::get_if<ActivityBindingCondition>(&condition.payload))
      activity = value;
  }
  if (!processCorner || !supplyVoltage || !clockPeriod || !activity ||
      request.baseConditions().size() != 4)
    return railError("request conditions do not form one complete rail input");

  const auto *assumption =
      std::get_if<ExplicitAssumptionSource>(&activity->source);
  if (!assumption)
    return railError("request activity is not an explicit assumption");
  if (!isExactHardwareRootTarget(processCorner->target, hardware) ||
      !isExactHardwareRootTarget(supplyVoltage->powerDomain, hardware) ||
      !isExactHardwareRootTarget(clockPeriod->clockDomain, hardware) ||
      !isExactHardwareRootTarget(activity->target, hardware) ||
      !isExactHardwareRootTarget(assumption->clockDomain, hardware) ||
      clockPeriod->clockDomain != assumption->clockDomain)
    return railError("rail conditions must target the exact global subject");

  return CompleteRailAnalysisConfiguration{
      kModelConfig, *processCorner, *supplyVoltage, *clockPeriod,
      ExplicitRailActivityBinding{activity->target, *assumption},
  };
}

} // namespace loom::evaluation::models
