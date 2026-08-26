#include "EDA/Adapters/OpenSource/OpenRoadStaticFpa.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "ExternalTool/ExternalFile.h"
#include "Hardware/Implementation/DefPhysical.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::eda::open_source {
namespace {

constexpr llvm::StringLiteral kResultOutput =
    "outputs/openroad-static-fpa-result.json";
constexpr llvm::StringLiteral kRawReport = "work/openroad-static-fpa-raw.txt";
constexpr llvm::StringLiteral kDriver = "drivers/openroad-static-fpa.tcl";
constexpr llvm::StringLiteral kPublisher =
    "drivers/openroad-static-fpa-publish.tcl";
constexpr llvm::StringLiteral kResultSchema = "loom.openroad_static_fpa_result";
constexpr llvm::StringLiteral kResultVersion = "1.0";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "openroad_static_fpa_invalid: " + detail);
}

bool isPortableIdentifier(llvm::StringRef value) {
  const auto first = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto rest = [&](char character) {
    return first(character) || (character >= '0' && character <= '9');
  };
  return !value.empty() && first(value.front()) &&
         llvm::all_of(value.drop_front(), rest);
}

llvm::Expected<std::string> tclString(llvm::StringRef value,
                                      llvm::StringRef description) {
  if (value.empty() || value.contains('\0') || value.contains('\n') ||
      value.contains('\r'))
    return invalid(description + " is empty or contains a line separator");
  std::string result = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"' || character == '$' ||
        character == '[' || character == ']')
      result.push_back('\\');
    result.push_back(character);
  }
  result.push_back('"');
  return result;
}

llvm::Expected<std::string> tclList(llvm::ArrayRef<std::string> values,
                                    llvm::StringRef description) {
  if (values.empty())
    return invalid(description + " closure is empty");
  std::string result = "[list";
  for (const std::string &value : values) {
    auto encoded = tclString(value, description);
    if (!encoded)
      return encoded.takeError();
    result += " " + *encoded;
  }
  result += "]";
  return result;
}

std::string decimalSpelling(evaluation::DecimalValue value) {
  return std::to_string(value.coefficient()) + "e" +
         std::to_string(value.base10Exponent());
}

std::string ratioExpression(evaluation::ExactRatio value) {
  return "[expr {double(" + std::to_string(value.numerator()) + ") / " +
         std::to_string(value.denominator()) + "}]";
}

llvm::Expected<evaluation::DecimalValue> parseDecimal(llvm::StringRef spelling,
                                                      llvm::StringRef field,
                                                      bool requirePositive) {
  if (spelling.empty() || spelling != spelling.trim())
    return invalid(field + " is not one normalized decimal");
  std::size_t index = 0;
  if (spelling[index] == '+')
    ++index;
  if (index == spelling.size())
    return invalid(field + " has no decimal digits");

  std::string digits;
  bool sawDigit = false;
  std::size_t fractionalDigits = 0;
  while (index < spelling.size() && spelling[index] >= '0' &&
         spelling[index] <= '9') {
    digits.push_back(spelling[index++]);
    sawDigit = true;
  }
  if (index < spelling.size() && spelling[index] == '.') {
    ++index;
    while (index < spelling.size() && spelling[index] >= '0' &&
           spelling[index] <= '9') {
      digits.push_back(spelling[index++]);
      ++fractionalDigits;
      sawDigit = true;
    }
  }
  if (!sawDigit || index == spelling.size() ||
      (spelling[index] != 'e' && spelling[index] != 'E'))
    return invalid(field + " is not a finite scientific decimal");
  ++index;
  bool negativeExponent = false;
  if (index < spelling.size() &&
      (spelling[index] == '+' || spelling[index] == '-')) {
    negativeExponent = spelling[index] == '-';
    ++index;
  }
  if (index == spelling.size())
    return invalid(field + " exponent has no digits");
  std::uint64_t exponentMagnitude = 0;
  for (; index < spelling.size(); ++index) {
    const char character = spelling[index];
    if (character < '0' || character > '9' || exponentMagnitude > 1000000)
      return invalid(field + " exponent is invalid");
    exponentMagnitude =
        exponentMagnitude * 10 + static_cast<std::uint64_t>(character - '0');
  }
  while (digits.size() > 1 && digits.front() == '0')
    digits.erase(digits.begin());
  if (digits.size() > 18)
    return invalid(field + " has more than 18 significant digits");
  std::uint64_t magnitude = 0;
  for (char digit : digits)
    magnitude = magnitude * 10 + static_cast<std::uint64_t>(digit - '0');
  if (magnitude >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid(field + " coefficient exceeds int64");
  const __int128 signedExponent =
      (negativeExponent ? -static_cast<__int128>(exponentMagnitude)
                        : static_cast<__int128>(exponentMagnitude)) -
      static_cast<__int128>(fractionalDigits);
  if (signedExponent < std::numeric_limits<std::int64_t>::min() ||
      signedExponent > std::numeric_limits<std::int64_t>::max())
    return invalid(field + " exponent exceeds int64");
  auto value =
      evaluation::DecimalValue::get(static_cast<std::int64_t>(magnitude),
                                    static_cast<std::int64_t>(signedExponent));
  if (!value)
    return value.takeError();
  const auto zero = llvm::cantFail(evaluation::DecimalValue::get(0, 0));
  if ((requirePositive && evaluation::compareDecimalValue(*value, zero) <= 0) ||
      (!requirePositive && evaluation::compareDecimalValue(*value, zero) < 0))
    return invalid(field + " is outside its metric domain");
  return *value;
}

llvm::Expected<std::pair<llvm::StringRef, llvm::StringRef>>
metricField(const llvm::json::Object &root, llvm::StringRef field,
            llvm::StringRef expectedUnit) {
  const llvm::json::Object *metric = root.getObject(field);
  if (!metric || metric->size() != 2)
    return invalid(field + " has an invalid object shape");
  const auto value = metric->getString("value");
  const auto unit = metric->getString("unit");
  if (!value || !unit || *unit != expectedUnit)
    return invalid(field + " value or canonical unit is invalid");
  return std::pair<llvm::StringRef, llvm::StringRef>{*value, *unit};
}

struct FpaMetricField final {
  evaluation::MetricKind kind;
  const char *rawKey;
  const char *resultKey;
  const char *unit;
  const char *driverVariable;
  bool requirePositive;
};

constexpr std::array<FpaMetricField, 4> kFpaMetricFields{{
    {evaluation::MetricKind::LimitingClockFrequency,
     "limiting_clock_frequency_hz", "limiting_clock_frequency", "hertz",
     "$loom_frequency", true},
    {evaluation::MetricKind::TotalArea, "total_area_square_meters",
     "total_area", "square_meter", "$loom_area", false},
    {evaluation::MetricKind::DynamicPower, "dynamic_power_watts",
     "dynamic_power", "watt", "$loom_dynamic", false},
    {evaluation::MetricKind::LeakagePower, "leakage_power_watts",
     "leakage_power", "watt", "$loom_leakage", false},
}};

llvm::Expected<std::vector<const FpaMetricField *>>
resolveMetricFields(llvm::ArrayRef<evaluation::MetricKind> metrics) {
  if (metrics.empty())
    return invalid("metric selection is empty");
  std::array<bool, kFpaMetricFields.size()> seen{};
  std::vector<const FpaMetricField *> fields;
  fields.reserve(metrics.size());
  for (evaluation::MetricKind metric : metrics) {
    const auto found = llvm::find_if(kFpaMetricFields, [&](const auto &field) {
      return field.kind == metric;
    });
    if (found == kFpaMetricFields.end())
      return invalid("metric selection contains an unsupported kind");
    const std::size_t ordinal =
        static_cast<std::size_t>(found - kFpaMetricFields.begin());
    if (seen[ordinal])
      return invalid("metric selection contains a duplicate kind");
    seen[ordinal] = true;
    fields.push_back(&*found);
  }
  return fields;
}

bool hasMetric(llvm::ArrayRef<evaluation::MetricKind> metrics,
               evaluation::MetricKind expected) {
  return llvm::is_contained(metrics, expected);
}

llvm::Expected<evaluation::DecimalValue>
observationValue(const OpenRoadStaticFpaObservation &observation,
                 evaluation::MetricKind metric) {
  const std::optional<evaluation::DecimalValue> *value = nullptr;
  switch (metric) {
  case evaluation::MetricKind::LimitingClockFrequency:
    value = &observation.limitingClockFrequencyHertz;
    break;
  case evaluation::MetricKind::TotalArea:
    value = &observation.totalAreaSquareMeters;
    break;
  case evaluation::MetricKind::DynamicPower:
    value = &observation.dynamicPowerWatts;
    break;
  case evaluation::MetricKind::LeakagePower:
    value = &observation.leakagePowerWatts;
    break;
  default:
    return invalid("observation was queried with an unsupported metric");
  }
  if (!*value)
    return invalid("observation omitted a requested metric");
  return **value;
}

struct FpaInvocationFacts final {
  evaluation::models::CompleteOpenRoadStaticFpaConfiguration analysis;
  hardware::FinalizedHardwareImplementation implementation;
  platform::FinalizedImplementationPlatform platform;
  external_tool::ExternalToolSemanticContract semanticContract;
  std::vector<external_tool::MaterializedBundleFile> semanticFiles;
  std::vector<external_tool::ExternalFileRequirement> externalRequirements;
  OpenRoadStaticFpaDriverFiles driverFiles;
  std::string top;
};

using FactsOrUnsupported =
    std::variant<FpaInvocationFacts, evaluation::UnsupportedEvidence>;

evaluation::UnsupportedEvidence unsupported() {
  return {evaluation::OutcomeReason::RuntimeCapabilityUnavailable};
}

llvm::Expected<std::string> blobText(const BlobStore &blobs,
                                     const BlobDigest &digest) {
  auto bytes = blobs.get(digest);
  if (!bytes)
    return bytes.takeError();
  return std::string(reinterpret_cast<const char *>(bytes->data()),
                     bytes->size());
}

llvm::Expected<FactsOrUnsupported>
invocationFacts(const evaluation::EvaluationRequest &request,
                const evaluation::CaseArtifactResolution &resolution,
                const ArtifactStore &artifacts, const BlobStore &blobs) {
  using namespace evaluation;
  using namespace evaluation::models;
  using namespace hardware;
  auto analysis = projectCompleteOpenRoadStaticFpaConfiguration(
      request, resolution, artifacts, blobs);
  if (!analysis)
    return analysis.takeError();
  if ((hasMetric(analysis->metrics, MetricKind::DynamicPower) &&
       !analysis->activity) ||
      (analysis->activity && !analysis->activityAssumption))
    return FactsOrUnsupported{unsupported()};
  const auto subjects = request.subjectBindings().subjects(
      hardwareImplementationPhysicalSubjectRole());
  if (subjects.size() != 1)
    return invalid("request does not bind one HardwareImplementation");
  auto contracts = makeKnownAsicStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto implementation = importHardwareImplementation(
      subjects.front(), *contracts, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  const HardwareImplementation &hardware = implementation->implementation();
  const ImplementationRepresentationRoot &root = hardware.representationRoot();
  if (root.variant != RepresentationRootVariant::AsicPhysical ||
      root.stage != RepresentationPhysicalStage::Routed ||
      root.formatRef.kind() != RepresentationFormatKind::IndexedDefPhysical ||
      root.top.kind != RepresentationObjectKind::PhysicalObject ||
      !isPortableIdentifier(root.top.canonicalName) ||
      !hardware.implementationPlatform())
    return FactsOrUnsupported{unsupported()};
  const auto bindings = hardware.externalImplementationBindings();
  if (bindings.size() != 1 ||
      bindings.front().providerContractRef !=
          openRoadRoutedStandardCellContractRef ||
      bindings.front().externalInputs.size() != 3 ||
      !bindings.front().blackBoxContractPayloadRef ||
      !hardware.memoryMacroBindings().empty())
    return FactsOrUnsupported{unsupported()};

  const ArtifactRootReference platformReference{
      platform::implementationPlatformSchema.identity.str(),
      platform::implementationPlatformSchema.version,
      analysis->processCorner.corner.artifact};
  if (*hardware.implementationPlatform() != platformReference)
    return invalid("request corner belongs to a foreign platform");
  auto target =
      platform::importImplementationPlatform(platformReference, artifacts);
  if (!target)
    return target.takeError();
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()) ||
      !target->platform().findTechnologyCorner(
          analysis->processCorner.corner.entity))
    return invalid("request does not select an ASIC platform corner");
  const std::string top = root.top.canonicalName;

  FpaInvocationFacts facts{std::move(*analysis),
                           std::move(*implementation),
                           std::move(*target),
                           {},
                           {},
                           {},
                           {},
                           top};
  const ImplementationRepresentationRoot &materializedRoot =
      facts.implementation.implementation().representationRoot();
  auto semanticContract = deriveExternalToolSemanticContract(request);
  if (!semanticContract)
    return semanticContract.takeError();
  facts.semanticContract = std::move(*semanticContract);
  facts.semanticFiles.push_back(
      {"inputs/hardware-implementation.json",
       std::string(facts.implementation.canonicalBytes().bytes().begin(),
                   facts.implementation.canonicalBytes().bytes().end()),
       subjects.front(), false});
  facts.semanticFiles.push_back(
      {"inputs/implementation-platform.json",
       std::string(facts.platform.canonicalBytes().bytes().begin(),
                   facts.platform.canonicalBytes().bytes().end()),
       platformReference, false});

  std::size_t netlistOrdinal = 0;
  std::size_t constraintOrdinal = 0;
  std::size_t databaseOrdinal = 0;
  for (const ImplementationPayload &payload : materializedRoot.payloads) {
    if (payload.role == PayloadRole::RepresentationIndex)
      continue;
    auto contents = blobText(blobs, payload.blobDigest);
    if (!contents)
      return contents.takeError();
    std::string path;
    switch (payload.role) {
    case PayloadRole::Netlist:
      path = "inputs/netlist/" + std::to_string(netlistOrdinal++) + ".v";
      facts.driverFiles.netlists.push_back(path);
      break;
    case PayloadRole::GenerationConstraint:
      path =
          "inputs/constraints/" + std::to_string(constraintOrdinal++) + ".sdc";
      facts.driverFiles.constraints.push_back(path);
      break;
    case PayloadRole::PhysicalDatabase:
      if (databaseOrdinal++ != 0)
        return invalid("routed implementation contains multiple DEF payloads");
      path = "inputs/database/routed.def";
      facts.driverFiles.physicalDatabase = path;
      break;
    case PayloadRole::BlackBoxContract:
      path = "inputs/contracts/standard-cells.txt";
      break;
    default:
      return invalid("routed implementation contains an unsupported payload");
    }
    facts.semanticFiles.push_back(
        {std::move(path), std::move(*contents), subjects.front(), false});
  }
  if (facts.driverFiles.netlists.empty() ||
      facts.driverFiles.constraints.empty() ||
      facts.driverFiles.physicalDatabase.empty())
    return invalid("routed implementation payload closure is incomplete");

  for (const ExternalInputBinding &input : bindings.front().externalInputs) {
    const auto *file =
        std::get_if<ExplicitFileDependency>(&input.dependencyIdentity);
    if (!file)
      return invalid("OpenROAD FPA requires explicit external files");
    facts.externalRequirements.push_back(
        {input.providerInputSlotRef, file->contentSha256});
  }
  return FactsOrUnsupported{std::move(facts)};
}

external_tool::ExternalToolInvocationImportExpectation
importExpectation(const FpaInvocationFacts &facts) {
  external_tool::ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = facts.semanticContract;
  for (const external_tool::MaterializedBundleFile &file : facts.semanticFiles)
    expectation.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact,
         computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
             reinterpret_cast<const std::uint8_t *>(file.contents.data()),
             file.contents.size()))});
  for (const external_tool::ExternalFileRequirement &file :
       facts.externalRequirements)
    expectation.externalInputs.push_back(
        {file.providerInputSlot, file.fingerprint});
  expectation.declaredOutputs = {kResultOutput.str()};
  return expectation;
}

llvm::Expected<evaluation::EvaluationModelProviderPreparation>
prepareProvider(const evaluation::EvaluationRequest &request,
                const evaluation::CaseArtifactResolution &resolution,
                const ArtifactStore &artifacts, const BlobStore &blobs,
                const external_tool::ExternalToolPreparationContext &context) {
  using namespace evaluation;
  auto factsOrUnsupported =
      invocationFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *value =
          std::get_if<UnsupportedEvidence>(&*factsOrUnsupported))
    return EvaluationModelProviderPreparation{*value};
  FpaInvocationFacts facts =
      std::get<FpaInvocationFacts>(std::move(*factsOrUnsupported));

  auto execution = resolveOpenRoadExecution(
      facts.analysis.providerBinding.stableProviderBuildIdentity, context);
  if (!execution) {
    llvm::consumeError(execution.takeError());
    return EvaluationModelProviderPreparation{unsupported()};
  }
  auto externalFiles = external_tool::resolveExternalFiles(
      facts.externalRequirements, context.localConfig);
  if (!externalFiles) {
    llvm::consumeError(externalFiles.takeError());
    return EvaluationModelProviderPreparation{unsupported()};
  }
  for (const external_tool::ResolvedExternalFile &file : *externalFiles) {
    if (file.providerInputSlot == openRoadTechnologyLefInputSlot)
      facts.driverFiles.technologyLef = file.absolutePath;
    else if (file.providerInputSlot == openRoadCellLefInputSlot)
      facts.driverFiles.cellLef = file.absolutePath;
    else if (file.providerInputSlot == openRoadLibertyInputSlot)
      facts.driverFiles.liberty = file.absolutePath;
    else
      return invalid("resolved external file has an unknown input slot");
  }
  if (facts.driverFiles.technologyLef.empty() ||
      facts.driverFiles.cellLef.empty() || facts.driverFiles.liberty.empty())
    return invalid("resolved external file closure is incomplete");
  OpenRoadStaticFpaDriverConfiguration configuration{
      facts.top, facts.driverFiles, facts.analysis};
  auto driver = renderOpenRoadStaticFpaDriver(configuration);
  if (!driver)
    return driver.takeError();

  external_tool::ExternalToolInvocationBundleSpec specification;
  specification.semanticContract = std::move(facts.semanticContract);
  specification.tool = execution->tool;
  specification.toolVersionProbe = execution->provider.versionProbe;
  specification.runtime = execution->runtime;
  specification.containerVersionProbe = execution->containerVersionProbe;
  specification.commands = {{execution->tool.executable, "-no_init",
                             "-no_splash", "-no_settings", "-threads", "1",
                             "-exit", kDriver.str()}};
  specification.declaredOutputs = {kResultOutput.str()};
  specification.files.push_back(
      {kDriver.str(), std::move(*driver), std::nullopt, false});
  auto publisher = renderOpenRoadStaticFpaPublisher(facts.analysis.metrics);
  if (!publisher)
    return publisher.takeError();
  specification.files.push_back(
      {kPublisher.str(), std::move(*publisher), std::nullopt, false});
  specification.files.insert(
      specification.files.end(),
      std::make_move_iterator(facts.semanticFiles.begin()),
      std::make_move_iterator(facts.semanticFiles.end()));
  specification.externalFiles = std::move(*externalFiles);
  auto prepared = external_tool::finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
  if (!prepared)
    return prepared.takeError();
  return EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<evaluation::EvaluationModelResult> importProviderImpl(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolInvocationExecutionObservation
        *executionObservation = nullptr) {
  using namespace evaluation;
  auto factsOrUnsupported =
      invocationFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  const auto *facts = std::get_if<FpaInvocationFacts>(&*factsOrUnsupported);
  if (!facts)
    return invalid("prepared invocation is no longer in stable capability");
  auto attempt =
      executionObservation
          ? external_tool::importExternalToolInvocationAttempt(
                prepared, importExpectation(*facts), *executionObservation)
          : external_tool::importExternalToolInvocationAttempt(
                prepared, importExpectation(*facts));
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<
          external_tool::IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<
        external_tool::IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<external_tool::FailedExternalToolInvocationAttempt>(
              &*attempt)) {
    using Status = external_tool::InvocationCompletionStatus;
    switch (failed->status) {
    case Status::Success:
      return invalid("failed invocation outcome carries success status");
    case Status::MissingEnvironment:
    case Status::ModuleActivationFailed:
    case Status::VersionMismatch:
      return EvaluationModelResult{{}, unsupported()};
    case Status::BundleContentMismatch:
      return invalid("invocation bundle content changed before execution");
    case Status::ToolExit:
    case Status::MissingOutput:
      return EvaluationModelResult{
          {}, ExecutionFailedEvidence{OutcomeReason::ToolFailure}};
    }
  }
  auto imported = std::get<external_tool::ImportedExternalToolInvocationBundle>(
      std::move(*attempt));
  auto contents = external_tool::readExternalToolInvocationDeclaredOutput(
      imported, kResultOutput);
  if (!contents)
    return contents.takeError();
  auto observation = parseOpenRoadStaticFpaResult(*contents, facts->top,
                                                  facts->analysis.metrics);
  if (!observation) {
    llvm::consumeError(observation.takeError());
    return EvaluationModelResult{
        {}, ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};
  }
  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto value = observationValue(*observation, metric.query().metric);
    if (!value)
      return value.takeError();
    results.push_back(MetricResult{UncertaintyKind::ExactWithinModel,
                                   PointObservation{MetricValue{*value}},
                                   {}});
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

llvm::Expected<evaluation::EvaluationModelResult>
importProvider(const evaluation::EvaluationRequest &request,
               const evaluation::CaseArtifactResolution &resolution,
               const external_tool::PreparedExternalToolInvocation &prepared,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderImpl(request, resolution, prepared, artifacts, blobs);
}

llvm::Expected<evaluation::EvaluationModelResult> importProviderWithExecution(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderImpl(request, resolution, prepared, artifacts, blobs,
                            &execution);
}

} // namespace

llvm::Expected<std::string> renderOpenRoadStaticFpaDriver(
    const OpenRoadStaticFpaDriverConfiguration &configuration) {
  const auto &files = configuration.files;
  const auto &analysis = configuration.analysis;
  if (!isPortableIdentifier(configuration.top))
    return invalid("top is not a portable identifier");
  if (llvm::Error error =
          evaluation::models::validateOpenRoadStaticFpaProviderBinding(
              analysis.providerBinding))
    return std::move(error);
  auto metricFields = resolveMetricFields(analysis.metrics);
  if (!metricFields)
    return metricFields.takeError();
  const bool needsFrequency = hasMetric(
      analysis.metrics, evaluation::MetricKind::LimitingClockFrequency);
  const bool needsArea =
      hasMetric(analysis.metrics, evaluation::MetricKind::TotalArea);
  const bool needsDynamic =
      hasMetric(analysis.metrics, evaluation::MetricKind::DynamicPower);
  const bool needsLeakage =
      hasMetric(analysis.metrics, evaluation::MetricKind::LeakagePower);
  if (needsDynamic && !analysis.activityAssumption)
    return invalid("dynamic power has no supported activity projection");
  if (analysis.activity && !analysis.activityAssumption)
    return invalid("execution activity projection is unavailable");
  auto netlists = tclList(files.netlists, "netlist path");
  auto constraints = tclList(files.constraints, "constraint path");
  auto database = tclString(files.physicalDatabase, "DEF path");
  auto technology = tclString(files.technologyLef, "technology LEF path");
  auto cells = tclString(files.cellLef, "cell LEF path");
  auto liberty = tclString(files.liberty, "Liberty path");
  if (!netlists)
    return netlists.takeError();
  if (!constraints)
    return constraints.takeError();
  if (!database)
    return database.takeError();
  if (!technology)
    return technology.takeError();
  if (!cells)
    return cells.takeError();
  if (!liberty)
    return liberty.takeError();

  const std::string voltage = decimalSpelling(analysis.supplyVoltage.volts);
  const std::string temperatureKelvin =
      decimalSpelling(analysis.temperature.kelvin);
  const std::string period = decimalSpelling(analysis.clockPeriod.seconds);
  std::string driver;
  driver += "set loom_netlists " + *netlists + "\n";
  driver += "set loom_constraints " + *constraints + "\n";
  driver += "read_lef " + *technology + "\n";
  driver += "read_lef " + *cells + "\n";
  driver += "read_liberty " + *liberty + "\n";
  driver += "foreach loom_path $loom_netlists { if {![file isfile "
            "$loom_path]} { error {routed implementation netlist is "
            "missing} } }\n";
  driver += "read_def " + *database + "\n";
  driver += "foreach loom_path $loom_constraints { read_sdc $loom_path }\n";
  driver += "set loom_block [ord::get_db_block]\n";
  driver += "if {[$loom_block getName] ne {" + configuration.top +
            "}} { error {DEF top differs from expected top} }\n";
  driver += "set loom_clocks [all_clocks]\n";
  driver += "if {[llength $loom_clocks] != 1} { error {FPA requires exactly "
            "one clock} }\n";
  driver += "set loom_clock [lindex $loom_clocks 0]\n";
  driver += "set loom_expected_period " + period + "\n";
  driver += "set loom_actual_period [sta::Clock_period $loom_clock]\n";
  driver += "set loom_period_tolerance [expr {max(abs($loom_expected_period) * "
            "1.0e-6, 1.0e-18)}]\n";
  driver += "if {abs($loom_actual_period - $loom_expected_period) > "
            "$loom_period_tolerance} { error {SDC clock period differs from "
            "Request} }\n";
  driver += "set loom_cells [get_cells -hierarchical *]\n";
  driver +=
      "if {[llength $loom_cells] == 0} { error {FPA design has no cells} }\n";
  driver += "set loom_voltage " + voltage + "\n";
  driver += "set loom_temperature_celsius [expr {double(" + temperatureKelvin +
            ") - 273.15}]\n";
  driver += "set_pvt $loom_cells -voltage $loom_voltage -temperature "
            "$loom_temperature_celsius\n";
  driver += "set_voltage $loom_voltage\n";
  if (analysis.activityAssumption) {
    const std::string activity =
        ratioExpression(analysis.activityAssumption->transitionsPerClock);
    const std::string duty =
        ratioExpression(analysis.activityAssumption->staticProbability);
    driver += "set_power_activity -global -activity " + activity + " -duty " +
              duty + " -clock $loom_clock\n";
  }
  driver += "extract_parasitics -version 2.0 -lef_rc\n";
  driver += "set loom_rseg_count [llength [$loom_block getRSegs]]\n";
  driver += "if {$loom_rseg_count == 0} { error {OpenRCX produced no "
            "parasitic segments} }\n";
  if (needsFrequency) {
    driver += "set loom_min_period [sta::find_clk_min_period $loom_clock 0]\n";
    driver +=
        "if {$loom_min_period <= 0.0} { error {timing report has no finite "
        "limiting period} }\n";
    driver += "set loom_frequency [expr {1.0 / $loom_min_period}]\n";
  }
  if (needsArea)
    driver += "set loom_area [rsz::design_area]\n";
  if (needsDynamic || needsLeakage) {
    driver += "set loom_power [sta::design_power [sta::cmd_scene]]\n";
    driver += "if {[llength $loom_power] != 24} { error {power report shape is "
              "incompatible} }\n";
    if (needsDynamic)
      driver += "set loom_dynamic [expr {[lindex $loom_power 0] + [lindex "
                "$loom_power 1]}]\n";
    if (needsLeakage)
      driver += "set loom_leakage [lindex $loom_power 2]\n";
  }
  driver += "file mkdir {work}\n";
  driver += "set loom_raw [open {" + kRawReport.str() + "} w]\n";
  driver += "puts $loom_raw {schema=loom.openroad_static_fpa_raw_report}\n";
  driver += "puts $loom_raw {version=1.0}\n";
  driver += "puts $loom_raw {top=" + configuration.top + "}\n";
  for (const FpaMetricField *field : *metricFields)
    driver += "puts $loom_raw [format {" + std::string(field->rawKey) +
              "=%.12e} " + field->driverVariable + "]\n";
  driver += "close $loom_raw\n";
  driver += "source {" + kPublisher.str() + "}\n";
  return driver;
}

llvm::Expected<std::string> renderOpenRoadStaticFpaPublisher(
    llvm::ArrayRef<evaluation::MetricKind> metrics) {
  auto fields = resolveMetricFields(metrics);
  if (!fields)
    return fields.takeError();
  std::string publisher;
  publisher += "set loom_allowed_keys [list schema version top";
  for (const FpaMetricField *field : *fields)
    publisher += " " + std::string(field->rawKey);
  publisher += "]\n";
  publisher += R"tcl(array set loom_values {}
set loom_report [open {work/openroad-static-fpa-raw.txt} r]
while {[gets $loom_report loom_line] >= 0} {
  set loom_equals [string first {=} $loom_line]
  if {$loom_equals <= 0} { error {OpenROAD FPA raw report has a malformed line} }
  set loom_key [string range $loom_line 0 [expr {$loom_equals - 1}]]
  set loom_value [string range $loom_line [expr {$loom_equals + 1}] end]
  if {[lsearch -exact $loom_allowed_keys $loom_key] < 0} { error {OpenROAD FPA raw report has an unknown key} }
  if {[info exists loom_values($loom_key)]} { error {OpenROAD FPA raw report has a duplicate key} }
  set loom_values($loom_key) $loom_value
}
close $loom_report
if {[array size loom_values] != [llength $loom_allowed_keys]} { error {OpenROAD FPA raw report is incomplete} }
if {$loom_values(schema) ne {loom.openroad_static_fpa_raw_report} || $loom_values(version) ne {1.0}} { error {OpenROAD FPA raw report schema is invalid} }
if {![regexp {^[A-Za-z_][A-Za-z0-9_]*$} $loom_values(top)]} { error {OpenROAD FPA raw report top is invalid} }
set loom_decimal {^[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-][0-9]+)$}
)tcl";
  publisher += "foreach loom_key [list";
  for (const FpaMetricField *field : *fields)
    publisher += " " + std::string(field->rawKey);
  publisher += R"tcl(] {
  if {![regexp $loom_decimal $loom_values($loom_key)]} { error {OpenROAD FPA raw report has a non-finite metric} }
  if {$loom_values($loom_key) < 0.0} { error {OpenROAD FPA raw report has a negative metric} }
}
)tcl";
  for (const FpaMetricField *field : *fields)
    if (field->requirePositive)
      publisher += "if {$loom_values(" + std::string(field->rawKey) +
                   ") <= 0.0} { error {OpenROAD FPA raw report has a "
                   "non-positive metric} }\n";
  publisher +=
      R"tcl(set loom_output [open {outputs/openroad-static-fpa-result.json} w]
puts -nonewline $loom_output "{\"schema\":\"loom.openroad_static_fpa_result\",\"version\":\"1.0\",\"top\":\"$loom_values(top)\""
)tcl";
  for (const FpaMetricField *field : *fields)
    publisher += "puts -nonewline $loom_output \",\\\"" +
                 std::string(field->resultKey) +
                 "\\\":{\\\"value\\\":\\\"$loom_values(" + field->rawKey +
                 ")\\\",\\\"unit\\\":\\\"" + field->unit + "\\\"}\"\n";
  publisher += "puts $loom_output \"}\"\nclose $loom_output\n";
  return publisher;
}

llvm::Expected<OpenRoadStaticFpaObservation>
parseOpenRoadStaticFpaResult(llvm::StringRef contents,
                             llvm::StringRef expectedTop,
                             llvm::ArrayRef<evaluation::MetricKind> metrics) {
  if (!isPortableIdentifier(expectedTop))
    return invalid("expected top is not a portable identifier");
  auto fields = resolveMetricFields(metrics);
  if (!fields)
    return fields.takeError();
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("result is malformed JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->size() != 3 + fields->size())
    return invalid("result has an invalid object shape");
  const auto schema = root->getString("schema");
  const auto version = root->getString("version");
  const auto top = root->getString("top");
  if (!schema || *schema != kResultSchema || !version ||
      *version != kResultVersion || !top || *top != expectedTop)
    return invalid("result schema, version, or top is inconsistent");
  OpenRoadStaticFpaObservation observation;
  for (const FpaMetricField *field : *fields) {
    auto encoded = metricField(*root, field->resultKey, field->unit);
    if (!encoded)
      return encoded.takeError();
    auto value =
        parseDecimal(encoded->first, field->resultKey, field->requirePositive);
    if (!value)
      return value.takeError();
    switch (field->kind) {
    case evaluation::MetricKind::LimitingClockFrequency:
      observation.limitingClockFrequencyHertz = *value;
      break;
    case evaluation::MetricKind::TotalArea:
      observation.totalAreaSquareMeters = *value;
      break;
    case evaluation::MetricKind::DynamicPower:
      observation.dynamicPowerWatts = *value;
      break;
    case evaluation::MetricKind::LeakagePower:
      observation.leakagePowerWatts = *value;
      break;
    default:
      llvm_unreachable("resolved FPA metric field is not supported");
    }
  }
  return observation;
}

llvm::Error registerOpenRoadStaticFpaEvaluationProvider() {
  if (llvm::Error error = evaluation::models::registerOpenRoadStaticFpaModel())
    return error;
  static const evaluation::EvaluationModelProvider provider{
      evaluation::models::openRoadStaticFpaModelDescriptorRef(),
      evaluation::EvaluationModelExternalPrepareImportProvider{
          &prepareProvider, &importProvider, nullptr,
          &importProviderWithExecution}};
  return evaluation::registerEvaluationModelProvider(provider);
}

} // namespace loom::eda::open_source
