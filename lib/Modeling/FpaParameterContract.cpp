#include "Evaluation/Models/FpaParameterContract.h"

#include "FixedTabularGbdt.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr std::uint32_t kIntegralFeatureCount = 24;
constexpr std::uint32_t kDecimalFeatureCount = 14;
constexpr std::uint32_t kCategoricalFeatureCount = 3;
constexpr std::uint32_t kPresenceFeatureCount = 5;
constexpr std::uint32_t kTargetCount = 4;

constexpr llvm::StringLiteral kParameterSchema =
    "loom.fpa.gbdt_parameter_payload.3.0";
constexpr llvm::StringLiteral kTargetFidelity =
    "routed_static_fpa.whole_exact_case.point.1";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fpa_parameter_contract_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> parameterSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(kParameterSchema.data()),
          kParameterSchema.size()};
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendFramed(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendFramed(bytes, {reinterpret_cast<const std::uint8_t *>(value.data()),
                       value.size()});
}

llvm::Error add(std::uint64_t &target, std::uint64_t amount,
                llvm::StringRef field) {
  const std::optional<std::uint64_t> sum =
      llvm::checkedAddUnsigned(target, amount);
  if (!sum)
    return invalid(field + " overflows uint64");
  target = *sum;
  return llvm::Error::success();
}

llvm::Error addScaled(std::uint64_t &target, std::uint64_t amount,
                      std::uint64_t multiplicity, llvm::StringRef field) {
  const std::optional<std::uint64_t> scaled =
      llvm::checkedMulUnsigned(amount, multiplicity);
  if (!scaled)
    return invalid(field + " multiplicity overflows uint64");
  return add(target, *scaled, field);
}

llvm::Expected<std::int64_t> checkedFeature(std::uint64_t value,
                                            llvm::StringRef field) {
  constexpr std::uint64_t limit = std::uint64_t{1} << 40;
  if (value > limit)
    return invalid(field + " exceeds the admitted feature magnitude");
  return static_cast<std::int64_t>(value);
}

llvm::Expected<DecimalValue> exactRatioDecimal(ExactRatio value) {
  constexpr std::uint64_t scale = 1000000000000000000ULL;
  const unsigned __int128 numerator =
      static_cast<unsigned __int128>(value.numerator()) * scale;
  unsigned __int128 quotient = numerator / value.denominator();
  const unsigned __int128 remainder = numerator % value.denominator();
  if (remainder * 2 > value.denominator() ||
      (remainder * 2 == value.denominator() && (quotient & 1) != 0))
    ++quotient;
  if (quotient >
      static_cast<unsigned __int128>(std::numeric_limits<std::int64_t>::max()))
    return invalid("exact ratio exceeds the 18-digit feature domain");
  return DecimalValue::get(static_cast<std::int64_t>(quotient), -18);
}

int compareRatio(ExactRatio lhs, ExactRatio rhs) {
  const unsigned __int128 left =
      static_cast<unsigned __int128>(lhs.numerator()) * rhs.denominator();
  const unsigned __int128 right =
      static_cast<unsigned __int128>(rhs.numerator()) * lhs.denominator();
  return left == right ? 0 : (left < right ? -1 : 1);
}

template <typename T, typename Compare>
std::pair<T, T> extrema(llvm::ArrayRef<T> values, Compare compare) {
  T minimum = values.front();
  T maximum = values.front();
  for (T value : values) {
    if (compare(value, minimum) < 0)
      minimum = value;
    if (compare(value, maximum) > 0)
      maximum = value;
  }
  return {minimum, maximum};
}

llvm::Error summarizeView(const fabric::FabricArtifactView &view,
                          std::uint64_t multiplicity,
                          FpaFabricStructureFeatureView &summary) {
  for (fabric::FabricEntityId id = 0;; ++id) {
    const std::optional<fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (llvm::Error error =
            addScaled(summary.entityCount, 1, multiplicity, "entity count"))
      return error;
    if (*kind == fabric::FabricEntityKind::SystemTransportResource) {
      if (llvm::Error error =
              addScaled(summary.systemTransportResourceCount, 1, multiplicity,
                        "System transport resource count"))
        return error;
    } else if (*kind == fabric::FabricEntityKind::HardwareDomain) {
      if (llvm::Error error = addScaled(summary.hardwareDomainCount, 1,
                                        multiplicity, "hardware domain count"))
        return error;
    }
  }
  if (llvm::Error error =
          addScaled(summary.peOccurrenceCount, view.peOccurrences().size(),
                    multiplicity, "PE occurrence count"))
    return error;
  if (llvm::Error error =
          addScaled(summary.fuOccurrenceCount, view.fuOccurrences().size(),
                    multiplicity, "FU occurrence count"))
    return error;
  for (fabric::FabricFuOccurrenceRef occurrence : view.fuOccurrences()) {
    const auto definition = view.fuTemplateOf(occurrence);
    if (!definition)
      return invalid("FU occurrence has no template");
    if (llvm::Error error =
            addScaled(summary.operationCapabilityCount,
                      view.resolvedFabricOpCapabilities(*definition).size(),
                      multiplicity, "operation capability count"))
      return error;
  }
  if (llvm::Error error = addScaled(summary.memoryOccurrenceCount,
                                    view.memoryOccurrences().size(),
                                    multiplicity, "memory occurrence count"))
    return error;
  for (fabric::FabricMemoryOccurrenceRef memory : view.memoryOccurrences())
    if (llvm::Error error =
            addScaled(summary.memoryOperationPortCount,
                      view.memoryOperationPorts(memory).size(), multiplicity,
                      "memory operation port count"))
      return error;
  if (llvm::Error error = addScaled(summary.switchOccurrenceCount,
                                    view.switchOccurrences().size(),
                                    multiplicity, "switch occurrence count"))
    return error;
  if (llvm::Error error =
          addScaled(summary.fifoOccurrenceCount, view.fifoOccurrences().size(),
                    multiplicity, "FIFO occurrence count"))
    return error;
  if (llvm::Error error = addScaled(summary.boundaryOccurrenceCount,
                                    view.boundaryOccurrences().size(),
                                    multiplicity, "boundary occurrence count"))
    return error;
  if (llvm::Error error = addScaled(summary.hostCoreOccurrenceCount,
                                    view.hostCoreOccurrences().size(),
                                    multiplicity, "host-core occurrence count"))
    return error;
  if (llvm::Error error = addScaled(
          summary.accCoreOccurrenceCount, view.accCoreOccurrences().size(),
          multiplicity, "accelerator-core occurrence count"))
    return error;
  if (llvm::Error error = addScaled(
          summary.systemMemoryServiceCount, view.systemMemoryServices().size(),
          multiplicity, "System memory-service count"))
    return error;
  if (llvm::Error error = addScaled(summary.transportEndpointCount,
                                    view.transportEndpoints().size(),
                                    multiplicity, "transport endpoint count"))
    return error;
  if (llvm::Error error = addScaled(summary.pointConnectionCount,
                                    view.pointConnections().size(),
                                    multiplicity, "point connection count"))
    return error;
  return addScaled(summary.admittedTraversalCount,
                   view.admittedTraversals().size(), multiplicity,
                   "admitted traversal count");
}

llvm::Expected<FpaFabricStructureFeatureView>
summarizeFabric(const fabric::FinalizedFabricRoot &root) {
  FpaFabricStructureFeatureView summary;
  if (llvm::Error error = summarizeView(root.view(), 1, summary))
    return std::move(error);
  summary.importedModuleCount = root.view().importedModules().size();
  if (root.view().rootKind() != fabric::FabricRootKind::System) {
    for (const fabric::FabricArtifactView &module :
         root.view().importedModules())
      if (llvm::Error error = summarizeView(module, 1, summary))
        return std::move(error);
    return summary;
  }

  auto system = fabric::requireSystemRoot(root.view());
  if (!system)
    return system.takeError();
  std::vector<std::uint64_t> multiplicities(
      root.view().importedModules().size(), 0);
  std::set<std::pair<std::uint64_t, fabric::FabricEntityId>> seen;
  for (const fabric::FabricSpatialAttachmentRecordView &attachment :
       system->spatialAttachments()) {
    const fabric::SpatialCoreOccurrenceRef *spatial = nullptr;
    if (const auto *endpoint = attachment.spatialEndpoint.transport())
      spatial = std::get_if<fabric::SpatialCoreOccurrenceRef>(
          &endpoint->owner.payload);
    else if (const auto *endpoint = attachment.spatialEndpoint.memory())
      spatial = std::get_if<fabric::SpatialCoreOccurrenceRef>(
          &endpoint->owner.payload);
    if (!spatial ||
        attachment.moduleEndpoint.dependencyOrdinal >= multiplicities.size())
      return invalid("System has a malformed Spatial attachment");
    if (seen.emplace(attachment.moduleEndpoint.dependencyOrdinal,
                     spatial->core.id())
            .second)
      ++multiplicities[attachment.moduleEndpoint.dependencyOrdinal];
  }
  for (std::size_t ordinal = 0; ordinal != root.view().importedModules().size();
       ++ordinal)
    if (llvm::Error error =
            summarizeView(root.view().importedModules()[ordinal],
                          multiplicities[ordinal], summary))
      return std::move(error);
  return summary;
}

llvm::Expected<FpaOperatingConditionFeatureView>
projectConditions(const EvaluationCase &evaluationCase) {
  FpaOperatingConditionFeatureView result;
  std::vector<DecimalValue> voltages;
  std::vector<DecimalValue> temperatures;
  std::vector<DecimalValue> periods;
  std::vector<ExactRatio> probabilities;
  std::vector<ExactRatio> transitions;
  std::vector<ExactRatio> relativePeriods;
  std::vector<ExactRatio> relativePhases;
  std::vector<std::vector<std::uint8_t>> cornerKeys;
  std::map<EvaluationConditionKind, std::uint64_t> conditionShape;

  for (const EvaluationCondition &condition : evaluationCase.baseConditions()) {
    ++conditionShape[condition.kind()];
    switch (condition.kind()) {
    case EvaluationConditionKind::ProcessCorner: {
      ++result.processCornerCount;
      const auto &value = std::get<ProcessCornerCondition>(condition.payload);
      cornerKeys.push_back(encodeArtifactLocalReference(
          platform::encodeTechnologyCornerRef(value.corner)));
      break;
    }
    case EvaluationConditionKind::SupplyVoltage:
      ++result.supplyVoltageCount;
      voltages.push_back(
          std::get<SupplyVoltageCondition>(condition.payload).volts);
      break;
    case EvaluationConditionKind::Temperature:
      ++result.temperatureCount;
      temperatures.push_back(
          std::get<TemperatureCondition>(condition.payload).kelvin);
      break;
    case EvaluationConditionKind::RequiredClockPeriod:
      ++result.requiredClockCount;
      periods.push_back(
          std::get<RequiredClockPeriodCondition>(condition.payload).seconds);
      break;
    case EvaluationConditionKind::RelativeClockSchedule: {
      ++result.relativeClockCount;
      const auto &value =
          std::get<RelativeClockScheduleCondition>(condition.payload);
      relativePeriods.push_back(value.dependentPeriodPerReferencePeriod);
      relativePhases.push_back(value.dependentPhaseInReferenceCycles);
      break;
    }
    case EvaluationConditionKind::ActivityBinding: {
      ++result.activityBindingCount;
      const auto &value = std::get<ActivityBindingCondition>(condition.payload);
      const auto *assumption =
          std::get_if<ExplicitAssumptionSource>(&value.source);
      if (!assumption)
        return invalid(
            "SimulationExecution activity projection is unavailable");
      probabilities.push_back(assumption->staticProbability);
      transitions.push_back(assumption->transitionsPerClock);
      break;
    }
    case EvaluationConditionKind::Quantile:
      return invalid("Quantile is not a Base condition");
    }
  }

  const auto setDecimalExtrema = [](llvm::ArrayRef<DecimalValue> values,
                                    std::optional<DecimalValue> &minimum,
                                    std::optional<DecimalValue> &maximum) {
    if (values.empty())
      return;
    auto selected = extrema(values, compareDecimalValue);
    minimum = selected.first;
    maximum = selected.second;
  };
  setDecimalExtrema(voltages, result.minimumSupplyVoltage,
                    result.maximumSupplyVoltage);
  setDecimalExtrema(temperatures, result.minimumTemperature,
                    result.maximumTemperature);
  setDecimalExtrema(periods, result.minimumRequiredClockPeriod,
                    result.maximumRequiredClockPeriod);

  const auto setRatioExtrema =
      [](llvm::ArrayRef<ExactRatio> values,
         std::optional<DecimalValue> &minimum,
         std::optional<DecimalValue> &maximum) -> llvm::Error {
    if (values.empty())
      return llvm::Error::success();
    auto selected = extrema(values, compareRatio);
    auto low = exactRatioDecimal(selected.first);
    if (!low)
      return low.takeError();
    auto high = exactRatioDecimal(selected.second);
    if (!high)
      return high.takeError();
    minimum = *low;
    maximum = *high;
    return llvm::Error::success();
  };
  if (llvm::Error error =
          setRatioExtrema(probabilities, result.minimumStaticProbability,
                          result.maximumStaticProbability))
    return std::move(error);
  if (llvm::Error error =
          setRatioExtrema(transitions, result.minimumTransitionsPerClock,
                          result.maximumTransitionsPerClock))
    return std::move(error);
  if (llvm::Error error =
          setRatioExtrema(relativePeriods, result.minimumRelativeClockPeriod,
                          result.maximumRelativeClockPeriod))
    return std::move(error);
  if (llvm::Error error =
          setRatioExtrema(relativePhases, result.minimumRelativeClockPhase,
                          result.maximumRelativeClockPhase))
    return std::move(error);

  llvm::sort(cornerKeys);
  appendU64(result.processCornerCohortKey, cornerKeys.size());
  for (const auto &key : cornerKeys)
    appendFramed(result.processCornerCohortKey, key);
  appendU64(result.conditionTargetShapeKey, conditionShape.size());
  for (const auto &[kind, count] : conditionShape) {
    appendU32(result.conditionTargetShapeKey, static_cast<std::uint32_t>(kind));
    appendU64(result.conditionTargetShapeKey, count);
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
implementationFamily(const hardware::HardwareImplementation &implementation) {
  std::vector<std::uint8_t> key;
  const hardware::ImplementationRepresentationRoot &root =
      implementation.representationRoot();
  appendU32(key, static_cast<std::uint32_t>(root.variant));
  appendU32(key, root.stage ? 1 : 0);
  if (root.stage)
    appendU32(key, static_cast<std::uint32_t>(*root.stage));
  std::vector<std::string> contracts;
  contracts.reserve(implementation.externalImplementationBindings().size());
  for (const hardware::ExternalImplementationBinding &binding :
       implementation.externalImplementationBindings())
    contracts.push_back(binding.providerContractRef);
  llvm::sort(contracts);
  appendU64(key, contracts.size());
  for (const std::string &contract : contracts)
    appendFramed(key, contract);
  return key;
}

llvm::Expected<FpaFeatureView>
projectFeatureView(const EvaluationCase &evaluationCase,
                   const CaseArtifactResolution &resolution,
                   const ArtifactStore &artifactStore,
                   const BlobStore &blobStore) {
  std::optional<ArtifactRootReference> fabricReference;
  const EvaluationCaseSignatureRef signature = evaluationCase.signature();
  if (signature == hardwareImplementationPhysicalCaseSignatureRef()) {
    const auto subjects = evaluationCase.subjectBindings().subjects(
        hardwareImplementationPhysicalSubjectRole());
    if (subjects.size() != 1)
      return invalid("physical case does not bind exactly one implementation");
    auto externalContracts = eda::makeKnownAsicStandardCellContractCatalog();
    if (!externalContracts)
      return externalContracts.takeError();
    auto implementation = hardware::importHardwareImplementation(
        subjects.front(), *externalContracts, artifactStore, blobStore);
    if (!implementation)
      return implementation.takeError();
    fabricReference = implementation->implementation().fabric();
  } else {
    CaseSubjectRoleRef fabricRole(0);
    if (signature == builtinEvaluationCaseSignatureRef(
                         BuiltinEvaluationCase::StructuredProgramWithFabric))
      fabricRole = structuredFabricAnalyticFabricRole();
    else if (signature ==
             builtinEvaluationCaseSignatureRef(
                 BuiltinEvaluationCase::CanonicalDataflowWithFabric))
      fabricRole = canonicalDataflowFabricAnalyticFabricRole();
    else if (signature != fabricHardwareAnalysisCaseSignatureRef())
      return invalid("feature projector received a foreign Evaluation case");
    const auto subjects = evaluationCase.subjectBindings().subjects(fabricRole);
    if (subjects.size() != 1)
      return invalid("architecture case does not bind exactly one Fabric");
    fabricReference = subjects.front();
  }

  if (!fabricReference || !resolution.find(*fabricReference))
    return invalid("feature Fabric is absent from CaseArtifactResolution");
  auto fabric = fabric::importEntireFabricRoot(*fabricReference, artifactStore);
  if (!fabric)
    return fabric.takeError();
  auto structure = summarizeFabric(*fabric);
  if (!structure)
    return structure.takeError();
  auto conditions = projectConditions(evaluationCase);
  if (!conditions)
    return conditions.takeError();
  std::vector<std::uint8_t> family;
  appendU32(family, static_cast<std::uint32_t>(fabric->view().rootKind()));
  return FpaFeatureView{std::move(*structure), std::move(*conditions),
                        std::move(family)};
}

llvm::Expected<detail::FixedTabularFeatureView>
fixedFeatures(const FpaFeatureView &features) {
  detail::FixedTabularFeatureView fixed;
  const std::array<std::pair<std::uint64_t, llvm::StringRef>,
                   kIntegralFeatureCount>
      integral = {
          {{features.fabric.entityCount, "entity count"},
           {features.fabric.peOccurrenceCount, "PE occurrence count"},
           {features.fabric.fuOccurrenceCount, "FU occurrence count"},
           {features.fabric.operationCapabilityCount,
            "operation capability count"},
           {features.fabric.memoryOccurrenceCount, "memory occurrence count"},
           {features.fabric.memoryOperationPortCount,
            "memory operation port count"},
           {features.fabric.switchOccurrenceCount, "switch occurrence count"},
           {features.fabric.fifoOccurrenceCount, "FIFO occurrence count"},
           {features.fabric.boundaryOccurrenceCount,
            "boundary occurrence count"},
           {features.fabric.hostCoreOccurrenceCount,
            "host-core occurrence count"},
           {features.fabric.accCoreOccurrenceCount,
            "accelerator-core occurrence count"},
           {features.fabric.systemMemoryServiceCount,
            "System memory-service count"},
           {features.fabric.systemTransportResourceCount,
            "System transport resource count"},
           {features.fabric.hardwareDomainCount, "hardware domain count"},
           {features.fabric.transportEndpointCount, "transport endpoint count"},
           {features.fabric.pointConnectionCount, "point connection count"},
           {features.fabric.admittedTraversalCount, "admitted traversal count"},
           {features.fabric.importedModuleCount, "imported Module count"},
           {features.conditions.processCornerCount, "process-corner count"},
           {features.conditions.supplyVoltageCount, "supply-voltage count"},
           {features.conditions.temperatureCount, "temperature count"},
           {features.conditions.requiredClockCount, "required-clock count"},
           {features.conditions.relativeClockCount, "relative-clock count"},
           {features.conditions.activityBindingCount,
            "activity-binding count"}}};
  fixed.integral.reserve(integral.size());
  for (const auto &[value, field] : integral) {
    auto admitted = checkedFeature(value, field);
    if (!admitted)
      return admitted.takeError();
    fixed.integral.push_back(*admitted);
  }

  auto zero = DecimalValue::get(0, 0);
  if (!zero)
    return zero.takeError();
  const auto &conditions = features.conditions;
  const std::array<std::optional<DecimalValue>, kDecimalFeatureCount> decimal =
      {conditions.minimumSupplyVoltage,
       conditions.maximumSupplyVoltage,
       conditions.minimumTemperature,
       conditions.maximumTemperature,
       conditions.minimumRequiredClockPeriod,
       conditions.maximumRequiredClockPeriod,
       conditions.minimumStaticProbability,
       conditions.maximumStaticProbability,
       conditions.minimumTransitionsPerClock,
       conditions.maximumTransitionsPerClock,
       conditions.minimumRelativeClockPeriod,
       conditions.maximumRelativeClockPeriod,
       conditions.minimumRelativeClockPhase,
       conditions.maximumRelativeClockPhase};
  fixed.decimal.reserve(decimal.size());
  for (const std::optional<DecimalValue> &value : decimal)
    fixed.decimal.push_back(value.value_or(*zero));
  fixed.categorical = {conditions.processCornerCohortKey,
                       conditions.conditionTargetShapeKey,
                       features.implementationFamilyKey};
  fixed.presence = {conditions.minimumSupplyVoltage.has_value(),
                    conditions.minimumTemperature.has_value(),
                    conditions.minimumRequiredClockPeriod.has_value(),
                    conditions.minimumStaticProbability.has_value(),
                    conditions.minimumRelativeClockPeriod.has_value()};
  return fixed;
}

std::vector<DecimalValue> fixedTargets(const FpaMetricPredictionView &view) {
  return {view.limitingClockFrequency, view.totalArea, view.dynamicPower,
          view.leakagePower};
}

llvm::Expected<CaseArtifactResolution>
resolveGroundTruthRequest(const ArtifactRootReference &requestReference,
                          const ArtifactStore &artifactStore,
                          const BlobStore &blobStore) {
  auto direct = importEvaluationRequestArtifactReferences(requestReference,
                                                          artifactStore);
  if (!direct)
    return direct.takeError();
  std::vector<ArtifactRootReference> implementations;
  for (const ArtifactRootReference &reference : *direct)
    if (reference.schemaIdentity ==
            hardware::hardwareImplementationSchema.identity &&
        reference.schemaVersion ==
            hardware::hardwareImplementationSchema.version)
      implementations.push_back(reference);
  if (implementations.size() != 1)
    return invalid("ground-truth Request must name one HardwareImplementation");
  auto externalContracts = eda::makeKnownAsicStandardCellContractCatalog();
  if (!externalContracts)
    return externalContracts.takeError();
  auto implementation = hardware::importHardwareImplementation(
      implementations.front(), *externalContracts, artifactStore, blobStore);
  if (!implementation)
    return implementation.takeError();

  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      completed(&artifactRootReferenceLess);
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      visiting(&artifactRootReferenceLess);
  const auto merge = [&](const ArtifactRootReference &owner,
                         llvm::ArrayRef<ArtifactRootReference> dependencies) {
    std::vector<ArtifactRootReference> &closure = entries[owner];
    closure.insert(closure.end(), dependencies.begin(), dependencies.end());
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  };
  for (const ArtifactRootReference &reference : *direct) {
    auto stored = artifactStore.get(reference);
    if (!stored)
      return stored.takeError();
    entries.emplace(reference, std::vector<ArtifactRootReference>{});
  }

  std::function<llvm::Expected<std::vector<ArtifactRootReference>>(
      const ArtifactRootReference &)>
      fabricClosure = [&](const ArtifactRootReference &reference)
      -> llvm::Expected<std::vector<ArtifactRootReference>> {
    if (completed.count(reference) != 0)
      return entries[reference];
    if (!visiting.insert(reference).second)
      return invalid("Fabric dependency closure is cyclic");
    auto root = fabric::importEntireFabricRoot(reference, artifactStore);
    if (!root) {
      visiting.erase(reference);
      return root.takeError();
    }
    std::vector<ArtifactRootReference> closure;
    for (const fabric::FabricDirectDependency &dependency :
         root->directDependencies()) {
      closure.push_back(dependency.root);
      auto nested = fabricClosure(dependency.root);
      if (!nested) {
        visiting.erase(reference);
        return nested.takeError();
      }
      closure.insert(closure.end(), nested->begin(), nested->end());
    }
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    entries[reference] = closure;
    visiting.erase(reference);
    completed.insert(reference);
    return closure;
  };

  const hardware::HardwareImplementation &implementationView =
      implementation->implementation();
  std::vector<ArtifactRootReference> hardwareClosure = {
      implementationView.fabric(), implementationView.configurationAbi()};
  auto fabricDependencies = fabricClosure(implementationView.fabric());
  if (!fabricDependencies)
    return fabricDependencies.takeError();
  hardwareClosure.insert(hardwareClosure.end(), fabricDependencies->begin(),
                         fabricDependencies->end());
  const std::array<ArtifactRootReference, 1> configurationDependencies = {
      implementationView.fabric()};
  merge(implementationView.configurationAbi(), configurationDependencies);
  if (implementationView.implementationPlatform())
    hardwareClosure.push_back(*implementationView.implementationPlatform());
  merge(implementations.front(), hardwareClosure);

  std::vector<CaseArtifactResolution::Entry> resolved;
  resolved.reserve(entries.size());
  for (auto &[artifact, closure] : entries)
    resolved.push_back({artifact, std::move(closure)});
  return CaseArtifactResolution::get(std::move(resolved));
}

llvm::Expected<EvaluationCase>
requestCase(const EvaluationRequest &request,
            const CaseArtifactResolution &resolution,
            const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const EvaluationModelDescriptor *model =
      request.modelBinding().descriptorRef().descriptor();
  if (!model)
    return invalid("Request model descriptor is unavailable");
  return EvaluationCase::get(model->caseSignature, request.subjectBindings(),
                             request.workload(), request.runtimeInput(),
                             request.baseConditions(), resolution,
                             artifactStore, blobStore);
}

llvm::Expected<std::vector<std::uint8_t>>
targetKey(const EvaluationRequest &request,
          const CaseArtifactResolution &resolution,
          const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  auto expected = builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::OpenRoadRoutedStaticFpa);
  if (!expected)
    return expected.takeError();
  if (request.modelBinding().descriptorRef() != *expected)
    return invalid("Request selects a foreign FPA ground-truth model");
  const EvaluationModelDescriptor *descriptor = expected->descriptor();
  std::vector<std::uint8_t> key;
  appendU32(key, expected->schemaVersion().major);
  appendU32(key, expected->schemaVersion().minor);
  appendU32(key, expected->modelKind().ordinal());
  appendFramed(key, descriptor->implementationSemanticIdentity);
  appendFramed(key, descriptor->resolvedConfigView.schemaDescriptorBytes);
  appendFramed(
      key, request.modelBinding().resolvedModelConfig().canonicalViewBytes());
  appendFramed(key, fpaMetricPredictionViewSchemaDescriptorBytes());
  appendFramed(key, kTargetFidelity);
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
deriveSampleGroupKey(const ArtifactRootReference &implementationReference,
                     const ArtifactStore &artifactStore,
                     const BlobStore &blobStore) {
  auto externalContracts = eda::makeKnownAsicStandardCellContractCatalog();
  if (!externalContracts)
    return externalContracts.takeError();
  auto implementation = hardware::importHardwareImplementation(
      implementationReference, *externalContracts, artifactStore, blobStore);
  if (!implementation)
    return implementation.takeError();
  auto family = implementationFamily(implementation->implementation());
  if (!family)
    return family.takeError();
  std::vector<std::uint8_t> key;
  const std::vector<std::uint8_t> fabric =
      encodeArtifactRootReference(implementation->implementation().fabric());
  appendFramed(key, fabric);
  appendFramed(key, *family);
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
sampleGroup(const EvaluationEvidence &, const EvaluationRequest &request,
            const CaseArtifactResolution &, const ArtifactStore &artifactStore,
            const BlobStore &blobStore) {
  const auto subjects = request.subjectBindings().subjects(
      hardwareImplementationPhysicalSubjectRole());
  if (subjects.size() != 1)
    return invalid("ground-truth Request does not bind one implementation");
  return deriveSampleGroupKey(subjects.front(), artifactStore, blobStore);
}

llvm::Expected<FpaMetricPredictionView>
requiredObservations(const EvaluationEvidence &evidence,
                     const EvaluationRequest &request) {
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  if (!completed)
    return invalid("training Evidence is not Completed");
  if (completed->metricResults.size() != request.metricRequests().size())
    return invalid("training Evidence metric shape does not match its Request");
  std::array<std::optional<DecimalValue>, kTargetCount> values;
  for (std::size_t index = 0; index != completed->metricResults.size();
       ++index) {
    const MetricKind kind = request.metricRequests()[index].query().metric;
    std::size_t ordinal = kTargetCount;
    switch (kind) {
    case MetricKind::LimitingClockFrequency:
      ordinal = 0;
      break;
    case MetricKind::TotalArea:
      ordinal = 1;
      break;
    case MetricKind::DynamicPower:
      ordinal = 2;
      break;
    case MetricKind::LeakagePower:
      ordinal = 3;
      break;
    default:
      continue;
    }
    const MetricQuery &query = request.metricRequests()[index].query();
    if (query.scope.form != ScopeFormRef(0) || !query.scope.targets.empty())
      return invalid("required FPA observation is not WholeExactCase");
    const auto *point = std::get_if<PointObservation>(
        &completed->metricResults[index].observation);
    const auto *decimal =
        point ? std::get_if<DecimalValue>(&point->value) : nullptr;
    if (!decimal || values[ordinal])
      return invalid("required FPA observation is missing, duplicate, or not "
                     "a Decimal Point");
    values[ordinal] = *decimal;
  }
  if (llvm::any_of(values,
                   [](const auto &value) { return !value.has_value(); }))
    return invalid("training Evidence omits a required FPA metric");
  return FpaMetricPredictionView{*values[0], *values[1], *values[2],
                                 *values[3]};
}

llvm::Expected<OwnerValue> adoptParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = adoptFpaGbdtParameters(bytes);
  if (!parameters)
    return parameters.takeError();
  return OwnerValue::get(std::move(*parameters));
}

llvm::Expected<std::vector<std::uint8_t>>
encodeParameters(const OwnerValue &value) {
  const auto *parameters = value.getIf<FpaGbdtParameters>();
  if (!parameters)
    return invalid("parameter value has a foreign owner type");
  return encodeFpaGbdtParameters(*parameters);
}

llvm::Expected<std::vector<std::uint8_t>>
parameterTargetKey(const OwnerValue &value) {
  const auto *parameters = value.getIf<FpaGbdtParameters>();
  if (!parameters)
    return invalid("parameter value has a foreign owner type");
  return parameters->groundTruthTargetKey().vec();
}

llvm::Expected<OwnerValue>
projectFeatures(const EvaluationCase &evaluationCase,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifactStore,
                const BlobStore &blobStore) {
  auto features =
      projectFeatureView(evaluationCase, resolution, artifactStore, blobStore);
  if (!features)
    return features.takeError();
  return OwnerValue::get(std::move(*features));
}

llvm::Expected<ModelParameterInferenceOutcome>
inferParameters(const OwnerValue &parameters, const OwnerValue &features) {
  const auto *typedParameters = parameters.getIf<FpaGbdtParameters>();
  const auto *typedFeatures = features.getIf<FpaFeatureView>();
  if (!typedParameters || !typedFeatures)
    return invalid("inference received a foreign owner value");
  return inferFpaGbdtParameters(*typedParameters, *typedFeatures);
}

const std::vector<EvaluationCaseSignatureRef> &predictionCases() {
  static const std::vector<EvaluationCaseSignatureRef> values = {
      builtinEvaluationCaseSignatureRef(
          BuiltinEvaluationCase::StructuredProgramWithFabric),
      builtinEvaluationCaseSignatureRef(
          BuiltinEvaluationCase::CanonicalDataflowWithFabric),
      fabricHardwareAnalysisCaseSignatureRef()};
  return values;
}

const std::vector<EvaluationModelDescriptorRef> &groundTruthModels() {
  static const std::vector<EvaluationModelDescriptorRef> values = {
      llvm::cantFail(builtinEvaluationModelDescriptorRef(
          BuiltinEvaluationModel::OpenRoadRoutedStaticFpa))};
  return values;
}

const std::vector<ModelParameterConditionPatternSet> &conditionTable() {
  static const std::vector<ModelParameterConditionPatternSet> values = [] {
    std::vector<EvaluationCaseSignatureRef> cases = predictionCases();
    cases.push_back(hardwareImplementationPhysicalCaseSignatureRef());
    llvm::sort(cases, evaluationCaseSignatureRefLess);
    cases.erase(std::unique(cases.begin(), cases.end()), cases.end());
    std::vector<ModelParameterConditionPatternSet> table;
    table.reserve(cases.size());
    for (EvaluationCaseSignatureRef reference : cases)
      table.push_back(
          {reference, reference.descriptor()->permittedBaseConditions});
    return table;
  }();
  return values;
}

const ModelParameterContractDescriptor &descriptor() {
  static const ModelParameterContractDescriptor value{
      fpaModelParameterContractRef(),
      "Deterministic four-head routed FPA prediction over exact Fabric and "
      "operating-condition features.",
      predictionCases(),
      groundTruthModels(),
      conditionTable(),
      fpaMetricPredictionViewSchemaDescriptorBytes(),
      {18, ModelParameterDecimalRounding::RoundToNearestTiesToEven},
      maximumFpaModelParameterPayloadBytes,
      &adoptParameters,
      &encodeParameters,
      &parameterTargetKey,
      &projectFeatures,
      &inferParameters,
      &targetKey,
      &sampleGroup};
  return value;
}

} // namespace

struct FpaGbdtParameters::Storage final {
  detail::FixedTabularGbdtParameters parameters;
};

llvm::Expected<std::vector<std::uint8_t>>
deriveFpaSampleGroupKey(const ArtifactRootReference &hardwareImplementation,
                        const ArtifactStore &artifactStore,
                        const BlobStore &blobStore) {
  return deriveSampleGroupKey(hardwareImplementation, artifactStore,
                              blobStore);
}

llvm::ArrayRef<std::uint8_t> FpaGbdtParameters::groundTruthTargetKey() const {
  return storage_ ? llvm::ArrayRef<std::uint8_t>(
                        storage_->parameters.groundTruthTargetKey)
                  : llvm::ArrayRef<std::uint8_t>();
}

const ModelParameterContractRef &fpaModelParameterContractRef() {
  static const ModelParameterContractRef reference =
      llvm::cantFail(ModelParameterContractRef::get("loom.fpa", {4, 0}, 0));
  return reference;
}

llvm::ArrayRef<std::uint8_t> fpaMetricPredictionViewSchemaDescriptorBytes() {
  static const std::vector<std::uint8_t> bytes = [] {
    std::vector<std::uint8_t> result;
    constexpr llvm::StringLiteral owner = "loom.fpa.metric_prediction_view";
    appendU64(result, owner.size());
    result.insert(result.end(), owner.bytes_begin(), owner.bytes_end());
    appendU32(result, 1);
    appendU32(result, 0);
    appendU64(result, kTargetCount);
    for (MetricKind metric :
         {MetricKind::LimitingClockFrequency, MetricKind::TotalArea,
          MetricKind::DynamicPower, MetricKind::LeakagePower})
      appendU32(result, static_cast<std::uint32_t>(metric));
    return result;
  }();
  return bytes;
}

const ModelParameterContractDescriptor &fpaModelParameterContractDescriptor() {
  return descriptor();
}

llvm::Error registerFpaModelParameterContract() {
  return registerModelParameterContract(descriptor());
}

llvm::Expected<CaseArtifactResolution>
resolveFpaCalibrationCaseArtifactResolution(
    const ArtifactRootReference &parameterBundle,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto bundle =
      importModelParameterBundle(parameterBundle, artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();
  if (!bundle->parametersIf<FpaGbdtParameters>())
    return invalid("calibration parameter bundle has a foreign payload");
  if (evidence.empty())
    return invalid("calibration Evidence collection is empty");

  std::vector<ArtifactRootReference> canonicalEvidence(evidence.begin(),
                                                       evidence.end());
  llvm::sort(canonicalEvidence, artifactRootReferenceLess);
  if (std::adjacent_find(canonicalEvidence.begin(), canonicalEvidence.end()) !=
      canonicalEvidence.end())
    return invalid("calibration Evidence collection contains a duplicate");

  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  const auto merge = [&](const ArtifactRootReference &owner,
                         llvm::ArrayRef<ArtifactRootReference> dependencies) {
    std::vector<ArtifactRootReference> &closure = entries[owner];
    closure.insert(closure.end(), dependencies.begin(), dependencies.end());
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  };
  entries.emplace(parameterBundle, std::vector<ArtifactRootReference>{});
  for (const ArtifactRootReference &evidenceReference : canonicalEvidence) {
    auto requestReference = importEvaluationEvidenceRequestReference(
        evidenceReference, artifactStore);
    if (!requestReference)
      return requestReference.takeError();
    auto source =
        resolveGroundTruthRequest(*requestReference, artifactStore, blobStore);
    if (!source)
      return source.takeError();
    std::vector<ArtifactRootReference> requestClosure;
    for (const CaseArtifactResolution::Entry &entry : source->entries()) {
      requestClosure.push_back(entry.artifact);
      requestClosure.insert(requestClosure.end(),
                            entry.dependencyClosure.begin(),
                            entry.dependencyClosure.end());
      merge(entry.artifact, entry.dependencyClosure);
    }
    merge(*requestReference, requestClosure);
    requestClosure.push_back(*requestReference);
    merge(evidenceReference, requestClosure);
  }

  std::vector<CaseArtifactResolution::Entry> resolved;
  resolved.reserve(entries.size());
  for (auto &[artifact, closure] : entries)
    resolved.push_back({artifact, std::move(closure)});
  return CaseArtifactResolution::get(std::move(resolved));
}

llvm::Expected<FpaTrainingEvidenceSample>
importFpaTrainingEvidenceSample(const ArtifactRootReference &evidenceReference,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore) {
  auto requestReference = importEvaluationEvidenceRequestReference(
      evidenceReference, artifactStore);
  if (!requestReference)
    return requestReference.takeError();
  auto resolution =
      resolveGroundTruthRequest(*requestReference, artifactStore, blobStore);
  if (!resolution)
    return resolution.takeError();
  return importFpaTrainingEvidenceSample(evidenceReference, *resolution,
                                         artifactStore, blobStore);
}

llvm::Expected<FpaTrainingEvidenceSample>
importFpaTrainingEvidenceSample(const ArtifactRootReference &evidenceReference,
                                const CaseArtifactResolution &resolution,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore) {
  auto requestReference = importEvaluationEvidenceRequestReference(
      evidenceReference, artifactStore);
  if (!requestReference)
    return requestReference.takeError();
  auto request = importEvaluationRequest(*requestReference, resolution,
                                         artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto evidence = importEvaluationEvidence(evidenceReference, resolution,
                                           artifactStore, blobStore);
  if (!evidence)
    return evidence.takeError();
  auto observation = requiredObservations(*evidence, *request);
  if (!observation)
    return observation.takeError();
  auto evaluationCase =
      requestCase(*request, resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto features =
      projectFeatureView(*evaluationCase, resolution, artifactStore, blobStore);
  if (!features)
    return features.takeError();
  auto key = targetKey(*request, resolution, artifactStore, blobStore);
  if (!key)
    return key.takeError();
  auto group =
      sampleGroup(*evidence, *request, resolution, artifactStore, blobStore);
  if (!group)
    return group.takeError();
  return FpaTrainingEvidenceSample{std::move(*features), *observation,
                                   std::move(*key), std::move(*group)};
}

llvm::Expected<FpaGbdtParameters>
trainFpaGbdtParameters(llvm::ArrayRef<FpaTrainingEvidenceSample> training,
                       const FpaGbdtTrainingConfig &config,
                       const FpaGbdtParameters *prior) {
  if (training.empty())
    return invalid("Training partition is empty");
  const std::vector<std::uint8_t> &targetKey =
      training.front().groundTruthTargetKey;
  if (targetKey.empty())
    return invalid("Training target key is empty");
  std::vector<detail::FixedTabularTrainingRow> rows;
  rows.reserve(training.size());
  for (const FpaTrainingEvidenceSample &sample : training) {
    if (sample.groundTruthTargetKey != targetKey)
      return invalid("Training partition mixes ground-truth target keys");
    if (sample.sampleGroupKey.empty())
      return invalid("Training sample-group key is empty");
    auto features = fixedFeatures(sample.features);
    if (!features)
      return features.takeError();
    rows.push_back({std::move(*features), fixedTargets(sample.observation)});
  }
  detail::DeterministicGbdtConfig trainingConfig{
      config.seed,
      config.treeCount,
      config.maximumDepth,
      config.minimumRowsPerLeaf,
      config.learningRateNumerator,
      config.learningRateDenominator};
  auto parameters = detail::trainFixedTabularGbdt(
      rows, targetKey, trainingConfig,
      prior && prior->storage_ ? &prior->storage_->parameters : nullptr);
  if (!parameters)
    return parameters.takeError();
  auto storage = std::make_shared<FpaGbdtParameters::Storage>();
  storage->parameters = std::move(*parameters);
  return FpaGbdtParameters(std::move(storage));
}

llvm::Expected<FpaGbdtParameters>
adoptFpaGbdtParameters(llvm::ArrayRef<std::uint8_t> canonicalPayloadBytes) {
  if (llvm::Error error =
          validateFpaModelParameterPayloadSize(canonicalPayloadBytes.size()))
    return std::move(error);
  auto parameters = detail::decodeFixedTabularGbdt(
      canonicalPayloadBytes, parameterSchemaBytes(), kIntegralFeatureCount,
      kDecimalFeatureCount, kCategoricalFeatureCount, kPresenceFeatureCount,
      kTargetCount);
  if (!parameters)
    return parameters.takeError();
  auto storage = std::make_shared<FpaGbdtParameters::Storage>();
  storage->parameters = std::move(*parameters);
  return FpaGbdtParameters(std::move(storage));
}

llvm::Error validateFpaModelParameterPayloadSize(std::uint64_t byteCount) {
  if (byteCount > maximumFpaModelParameterPayloadBytes)
    return invalid("parameter payload exceeds the 10 GB artifact bound");
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
encodeFpaGbdtParameters(const FpaGbdtParameters &parameters) {
  if (!parameters.storage_)
    return invalid("parameter storage is empty");
  auto encoded = detail::encodeFixedTabularGbdt(parameters.storage_->parameters,
                                                parameterSchemaBytes());
  if (!encoded)
    return encoded.takeError();
  if (llvm::Error error = validateFpaModelParameterPayloadSize(encoded->size()))
    return std::move(error);
  return encoded;
}

llvm::Expected<ModelParameterInferenceOutcome>
inferFpaGbdtParameters(const FpaGbdtParameters &parameters,
                       const FpaFeatureView &features) {
  if (!parameters.storage_)
    return invalid("parameter storage is empty");
  auto fixed = fixedFeatures(features);
  if (!fixed)
    return fixed.takeError();
  auto prediction =
      detail::inferFixedTabularGbdt(parameters.storage_->parameters, *fixed);
  if (!prediction)
    return prediction.takeError();
  if (!*prediction)
    return ModelParameterInferenceOutcome{OutOfDomainModelParameterInference{}};
  if ((**prediction).size() != kTargetCount)
    return invalid("inference returned the wrong target count");
  FpaMetricPredictionView view{(**prediction)[0], (**prediction)[1],
                               (**prediction)[2], (**prediction)[3]};
  return ModelParameterInferenceOutcome{
      ModelParameterPrediction{OwnerValue::get(std::move(view))}};
}

} // namespace loom::evaluation::models
