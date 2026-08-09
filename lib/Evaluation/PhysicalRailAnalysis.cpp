#include "Evaluation/Models/PhysicalRailAnalysis.h"

#include "CanonicalSupport.h"

#include "Config/ResolvedConfig.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/Support/Error.h"

#include <array>
#include <set>
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

const std::array<ModelConditionCapability, 5> kConditionCapabilities = {{
    {kBaseConditionPatterns[0], ConditionDisposition::Required},
    {kBaseConditionPatterns[1], ConditionDisposition::Required},
    {kBaseConditionPatterns[2], ConditionDisposition::Required},
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

class ConfigReader final {
public:
  explicit ConfigReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint64_t> u64(const llvm::Twine &field) {
    if (bytes_.size() - offset_ < 8)
      return railError(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::string> text(llvm::StringRef field) {
    auto sizeOrErr = u64(field + " length");
    if (!sizeOrErr)
      return sizeOrErr.takeError();
    if (*sizeOrErr > bytes_.size() - offset_)
      return railError(field + " is truncated");
    const std::size_t size = static_cast<std::size_t>(*sizeOrErr);
    std::string value(reinterpret_cast<const char *>(bytes_.data() + offset_),
                      size);
    offset_ += size;
    return value;
  }

  llvm::Expected<ExternalFileFingerprint> fingerprint(llvm::StringRef field) {
    if (bytes_.size() - offset_ < ExternalFileFingerprint::byteSize)
      return railError(field + " is truncated");
    auto value = ExternalFileFingerprint::fromBytes(
        bytes_.slice(offset_, ExternalFileFingerprint::byteSize));
    offset_ += ExternalFileFingerprint::byteSize;
    return value;
  }

  bool empty() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

std::vector<std::uint8_t>
encodeProviderBinding(const CadenceVoltusStaticRailProviderBinding &binding) {
  std::vector<std::uint8_t> bytes;
  detail::appendFramedString(bytes, binding.stableProviderBuildIdentity);
  detail::appendU64Be(bytes, binding.powerGridLibraryMembers.size());
  for (const external_tool::ExternalFileTreeMember &member :
       binding.powerGridLibraryMembers) {
    detail::appendFramedString(bytes, member.relativePath);
    bytes.insert(bytes.end(), member.fingerprint.bytes().begin(),
                 member.fingerprint.bytes().end());
  }
  detail::appendU64Be(bytes, binding.powerGridLibraryEntrypoints.size());
  for (const std::string &entrypoint : binding.powerGridLibraryEntrypoints)
    detail::appendFramedString(bytes, entrypoint);
  return bytes;
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.evaluation.static_explicit_rail.config.3.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &config) {
  if (!config.evaluation.cadenceVoltusStaticRail)
    return railError("Voltus provider binding is unavailable");
  if (llvm::Error error = validateCadenceVoltusStaticRailProviderBinding(
          *config.evaluation.cadenceVoltusStaticRail))
    return std::move(error);
  return OwnerValue::get(*config.evaluation.cadenceVoltusStaticRail);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  const auto *binding = value.getIf<CadenceVoltusStaticRailProviderBinding>();
  if (!binding)
    return railError("config has the wrong provider-binding owner type");
  if (llvm::Error error =
          validateCadenceVoltusStaticRailProviderBinding(*binding))
    return std::move(error);
  return encodeProviderBinding(*binding);
}

llvm::Expected<OwnerValue>
adoptConfig(llvm::ArrayRef<std::uint8_t> canonicalBytes,
            const ComponentViewDigest &) {
  ConfigReader reader(canonicalBytes);
  auto buildOrErr = reader.text("stable provider build identity");
  if (!buildOrErr)
    return buildOrErr.takeError();
  auto countOrErr = reader.u64("power-grid library member count");
  if (!countOrErr)
    return countOrErr.takeError();
  if (*countOrErr > canonicalBytes.size())
    return railError("power-grid library member count is invalid");

  std::vector<external_tool::ExternalFileTreeMember> members;
  members.reserve(static_cast<std::size_t>(*countOrErr));
  for (std::uint64_t index = 0; index < *countOrErr; ++index) {
    auto pathOrErr = reader.text("power-grid library member path");
    if (!pathOrErr)
      return pathOrErr.takeError();
    auto fingerprintOrErr =
        reader.fingerprint("power-grid library member fingerprint");
    if (!fingerprintOrErr)
      return fingerprintOrErr.takeError();
    members.push_back({std::move(*pathOrErr), std::move(*fingerprintOrErr)});
  }
  auto entrypointCountOrErr = reader.u64("power-grid library entrypoint count");
  if (!entrypointCountOrErr)
    return entrypointCountOrErr.takeError();
  if (*entrypointCountOrErr > canonicalBytes.size())
    return railError("power-grid library entrypoint count is invalid");
  std::vector<std::string> entrypoints;
  entrypoints.reserve(static_cast<std::size_t>(*entrypointCountOrErr));
  for (std::uint64_t index = 0; index < *entrypointCountOrErr; ++index) {
    auto entrypointOrErr = reader.text("power-grid library entrypoint path");
    if (!entrypointOrErr)
      return entrypointOrErr.takeError();
    entrypoints.push_back(std::move(*entrypointOrErr));
  }
  if (!reader.empty())
    return railError("resolved config view has trailing bytes");

  CadenceVoltusStaticRailProviderBinding binding{
      std::move(*buildOrErr), std::move(members), std::move(entrypoints)};
  if (llvm::Error error =
          validateCadenceVoltusStaticRailProviderBinding(binding))
    return std::move(error);
  const std::vector<std::uint8_t> reencoded = encodeProviderBinding(binding);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalBytes)
    return railError("resolved config view is not canonical");
  return OwnerValue::get(std::move(binding));
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

llvm::Error validateCadenceVoltusStaticRailProviderBinding(
    const CadenceVoltusStaticRailProviderBinding &binding) {
  const llvm::StringRef build(binding.stableProviderBuildIdentity);
  if (build.empty() || build.trim() != build ||
      !llvm::all_of(build, [](unsigned char character) {
        return character >= 0x20 && character <= 0x7e;
      }))
    return railError("provider build identity is not one normalized line");
  if (llvm::Error error = external_tool::validateExternalFileTreeRequirement(
          {cadenceVoltusPowerGridLibraryInputSlot.str(),
           binding.powerGridLibraryMembers}))
    return error;
  if (binding.powerGridLibraryEntrypoints.empty())
    return railError("power-grid library entrypoint catalog is empty");
  std::set<std::string> seen;
  for (const std::string &entrypoint : binding.powerGridLibraryEntrypoints) {
    if (!seen.insert(entrypoint).second)
      return railError(
          "power-grid library entrypoint catalog contains a duplicate");
    const bool member = llvm::any_of(
        binding.powerGridLibraryMembers,
        [&](const external_tool::ExternalFileTreeMember &candidate) {
          return candidate.relativePath == entrypoint;
        });
    if (!member)
      return railError(
          "power-grid library entrypoint is absent from the member table");
  }
  return llvm::Error::success();
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

  const auto *providerBinding =
      request.modelBinding()
          .resolvedModelConfig()
          .getIf<CadenceVoltusStaticRailProviderBinding>();
  if (!providerBinding)
    return railError("request does not carry the Voltus provider binding");
  if (llvm::Error error =
          validateCadenceVoltusStaticRailProviderBinding(*providerBinding))
    return std::move(error);

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
      !activity || request.baseConditions().size() != 5)
    return railError("request conditions do not form one complete rail input");

  const auto *assumption =
      std::get_if<ExplicitAssumptionSource>(&activity->source);
  if (!assumption)
    return railError("request activity is not an explicit assumption");
  if (!isExactHardwareRootTarget(processCorner->target, hardware) ||
      !isExactHardwareRootTarget(supplyVoltage->powerDomain, hardware) ||
      !isExactHardwareRootTarget(temperature->thermalDomainOrRoot, hardware) ||
      !isExactHardwareRootTarget(clockPeriod->clockDomain, hardware) ||
      !isExactHardwareRootTarget(activity->target, hardware) ||
      !isExactHardwareRootTarget(assumption->clockDomain, hardware) ||
      clockPeriod->clockDomain != assumption->clockDomain)
    return railError("rail conditions must target the exact global subject");

  return CompleteRailAnalysisConfiguration{
      kModelConfig,
      *providerBinding,
      *processCorner,
      *supplyVoltage,
      *temperature,
      *clockPeriod,
      ExplicitRailActivityBinding{activity->target, *assumption},
  };
}

} // namespace loom::evaluation::models
