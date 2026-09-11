#include "AnalyticModelSupport.h"
#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/IndexWidth.h"
#include "Common/MappingDebugLog.h"

#include "llvm/Support/JSON.h"
#include "Common/VectorWidth.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowStaticScheduleAnalysis.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Evaluation/NumericValue.h"
#include "Evaluation/OwnerValue.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation::models::detail {
namespace {

constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;

struct EmptyConfigView {};

struct ActorDemand final {
  mlir::Operation *representative = nullptr;
  std::uint64_t count = 0;
};

struct PhysicalEstimate final {
  std::uint64_t areaSquareMicrometers = 0;
  std::uint64_t leakageMicrowatts = 0;
  std::uint64_t criticalDelayPicoseconds = 1;
};

struct EntityCost final {
  std::uint64_t areaSquareMicrometers;
  std::uint64_t leakageMicrowatts;
  std::uint64_t delayPicoseconds;
};

llvm::Error accumulateScaled(std::uint64_t &total, std::uint64_t value,
                             std::uint64_t count, llvm::StringRef context) {
  const std::optional<std::uint64_t> product =
      llvm::checkedMulUnsigned(value, count);
  if (!product)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "low_confidence_model_overflow: %s",
                                   context.str().c_str());
  const std::optional<std::uint64_t> sum =
      llvm::checkedAddUnsigned(total, *product);
  if (!sum)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "low_confidence_model_overflow: %s",
                                   context.str().c_str());
  total = *sum;
  return llvm::Error::success();
}

EntityCost entityCost(fabric::FabricEntityKind kind) {
  using Kind = fabric::FabricEntityKind;
  switch (kind) {
  case Kind::FabricModuleTemplate:
  case Kind::FabricFuTemplate:
  case Kind::FabricMemoryEngineTemplate:
    return {0, 0, 1};
  case Kind::FabricPeOccurrence:
    return {120, 6, 180};
  case Kind::FabricFuOccurrence:
    return {20, 1, 80};
  case Kind::FabricMemoryOccurrence:
    return {1500, 75, 700};
  case Kind::FabricSwitchOccurrence:
    return {80, 4, 220};
  case Kind::FabricFifoOccurrence:
    return {60, 3, 160};
  case Kind::FabricBoundaryOccurrence:
    return {12, 1, 100};
  case Kind::HostCoreOccurrence:
    return {600, 30, 700};
  case Kind::AccCoreOccurrence:
    return {250, 12, 250};
  case Kind::SystemMemoryService:
    return {2500, 125, 900};
  case Kind::SystemServiceEndpoint:
    return {20, 1, 80};
  case Kind::SystemServiceTransform:
    return {60, 3, 200};
  case Kind::SystemTransportResource:
    return {120, 6, 250};
  case Kind::HardwareDomain:
    return {5, 1, 50};
  case Kind::ExternalBoundary:
    return {15, 1, 100};
  }
  llvm_unreachable("unknown Fabric entity kind");
}

std::uint64_t familyComplexity(::fabric::ImplementationFamilyId family,
                               ::fabric::CapabilityParamsSchemaId schema) {
  using Family = ::fabric::ImplementationFamilyId;
  switch (family) {
  case Family::ScalarIntegerMultiply:
  case Family::FixedVectorIntegerMultiply:
    return 5;
  case Family::ScalarFloatMultiply:
  case Family::FixedVectorFloatMultiply:
    return 12;
  case Family::ScalarFloatFma:
  case Family::FixedVectorFloatFma:
    return 16;
  case Family::ScalarSignedIntegerDivRem:
  case Family::ScalarUnsignedIntegerDivRem:
    return 18;
  case Family::ScalarFloatDivide:
  case Family::ScalarFloatRemainder:
    return 24;
  case Family::ScalarMathPow:
    return 36;
  case Family::FixedVectorSliceAlignMerge:
    return 6;
  case Family::FixedVectorShuffle:
    return 10;
  default:
    break;
  }
  const auto ordinal = static_cast<std::uint32_t>(family);
  if (ordinal >= static_cast<std::uint32_t>(Family::ScalarMathSin) &&
      ordinal <= static_cast<std::uint32_t>(Family::ScalarMathErf))
    return 28;

  using Schema = ::fabric::CapabilityParamsSchemaId;
  switch (schema) {
  case Schema::ScalarSpecialMathParams:
    return 28;
  case Schema::ScalarFloatParams:
  case Schema::ScalarFloatCompareMinMaxParams:
  case Schema::ScalarFloatWidthCastParams:
  case Schema::ScalarIntegerFloatConversionParams:
    return 7;
  case Schema::FixedVectorFloatParams:
  case Schema::FixedVectorFloatCompareMinMaxParams:
    return 9;
  case Schema::FixedVectorIntegerParams:
  case Schema::FixedVectorIntegerCompareMinMaxParams:
  case Schema::FixedVectorValueSelectParams:
  case Schema::FixedVectorAdapterParams:
    return 4;
  case Schema::FixedVectorSliceAlignMergeParams:
    return 6;
  case Schema::FixedVectorShuffleParams:
    return 10;
  case Schema::ScalarIntegerParams:
  case Schema::ScalarIntegerCompareMinMaxParams:
  case Schema::ScalarValueSelectParams:
  case Schema::ScalarIntegerCastParams:
  case Schema::ScalarBitReinterpretParams:
  case Schema::LoopStreamParams:
    return 2;
  case Schema::TokenPlaneParams:
  case Schema::PayloadCapacityParams:
  case Schema::RoutedTokenParams:
    return 1;
  }
  llvm_unreachable("unknown capability parameter schema");
}

llvm::Expected<EntityCost>
operationCost(const fabric::ResolvedFabricOpCapabilityView &capability) {
  std::uint64_t totalPortBits = 0;
  std::uint64_t maximumPortBits = 1;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    if (llvm::Error error = accumulateScaled(
            totalPortBits, port.payloadWidthBits, 1, "operation port width"))
      return std::move(error);
    maximumPortBits =
        std::max<std::uint64_t>(maximumPortBits, port.payloadWidthBits);
  }
  totalPortBits = std::max<std::uint64_t>(totalPortBits, 1);
  const std::uint64_t complexity = familyComplexity(
      capability.implementationFamily,
      ::fabric::capabilityParamsSchema(capability.parameterizedCapability));
  const std::uint64_t portBytes =
      totalPortBits / 8 + (totalPortBits % 8 != 0 ? 1 : 0);
  const std::uint64_t widthWords =
      maximumPortBits / 32 + (maximumPortBits % 32 != 0 ? 1 : 0);

  std::uint64_t area = 8;
  if (llvm::Error error =
          accumulateScaled(area, complexity, portBytes, "operation area"))
    return std::move(error);
  const ::fabric::ResourceContract &resource =
      capability.resourceStateAndTimingContract;
  if (llvm::Error error = accumulateScaled(area, resource.stateCount(), 5,
                                           "operation state area"))
    return std::move(error);
  if (llvm::Error error = accumulateScaled(area, resource.usePatternCount(), 3,
                                           "operation use-pattern area"))
    return std::move(error);

  std::uint64_t delay = 80;
  if (llvm::Error error =
          accumulateScaled(delay, complexity, 30, "operation critical delay"))
    return std::move(error);
  if (llvm::Error error =
          accumulateScaled(delay, widthWords, 15, "operation width delay"))
    return std::move(error);
  return EntityCost{area, std::max<std::uint64_t>(1, area / 20), delay};
}

llvm::Error addEntityCost(PhysicalEstimate &estimate, EntityCost cost,
                          std::uint64_t count) {
  if (count == 0)
    return llvm::Error::success();
  if (llvm::Error error =
          accumulateScaled(estimate.areaSquareMicrometers,
                           cost.areaSquareMicrometers, count, "Fabric area"))
    return error;
  if (llvm::Error error =
          accumulateScaled(estimate.leakageMicrowatts, cost.leakageMicrowatts,
                           count, "Fabric leakage power"))
    return error;
  estimate.criticalDelayPicoseconds =
      std::max(estimate.criticalDelayPicoseconds, cost.delayPicoseconds);
  return llvm::Error::success();
}

/// Silicon of one byte of a memory operation issue-queue entry. One entry
/// holds one pending request record, so an Operation Engine that may hold
/// `operation_issue_depth` firings outstanding buys that many records; the
/// serialized engine's single record is already inside the occurrence cost.
constexpr std::uint64_t kMemoryIssueQueueRecordByteArea = 4;
constexpr std::uint64_t kLeakageAreaDivisor = 20;

/// Cost of the extra issue-queue entries one memory occurrence declares. The
/// request record is the engine's input token endpoints: address, data, mask,
/// and control as the exact engine template declares them.
llvm::Expected<std::pair<EntityCost, std::uint64_t>>
memoryIssueQueueCost(const fabric::FabricArtifactView &view,
                     fabric::FabricMemoryOccurrenceRef memory) {
  const auto definition = view.memoryEngineTemplateOf(memory);
  if (!definition)
    return std::pair<EntityCost, std::uint64_t>{EntityCost{0, 0, 0}, 0};
  const fabric::FabricMemoryEngineTemplateRecord *engine =
      view.memoryEngineTemplate(*definition);
  if (!engine)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_invalid: memory occurrence has no engine");
  const std::uint64_t depth = view.memoryOperationIssueDepth(memory);
  if (depth <= ::fabric::serializedMemoryOperationIssueDepth)
    return std::pair<EntityCost, std::uint64_t>{EntityCost{0, 0, 0}, 0};
  std::uint64_t recordBits = 0;
  for (const ::fabric::MemoryTransportEndpointDescriptor &endpoint :
       engine->tokenEndpoints)
    if (endpoint.direction == fabric::FabricPortDirection::Input)
      recordBits += endpoint.payloadWidth + endpoint.tagWidth.value_or(0);
  const std::uint64_t recordBytes = (recordBits + 7) / 8;
  const std::uint64_t area = recordBytes * kMemoryIssueQueueRecordByteArea;
  return std::pair<EntityCost, std::uint64_t>{
      EntityCost{area, std::max<std::uint64_t>(1, area / kLeakageAreaDivisor),
                 0},
      depth - ::fabric::serializedMemoryOperationIssueDepth};
}

llvm::Error summarizeFabricView(const fabric::FabricArtifactView &view,
                                std::uint64_t occurrenceCount,
                                PhysicalEstimate &estimate) {
  std::map<fabric::FabricEntityId, std::uint64_t> fuOccurrences;
  for (fabric::FabricEntityId id = 0;; ++id) {
    const std::optional<fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (llvm::Error error =
            addEntityCost(estimate, entityCost(*kind), occurrenceCount))
      return error;
    if (*kind == fabric::FabricEntityKind::FabricMemoryOccurrence) {
      auto issueQueue =
          memoryIssueQueueCost(view, fabric::FabricMemoryOccurrenceRef(id));
      if (!issueQueue)
        return issueQueue.takeError();
      const std::optional<std::uint64_t> entries =
          llvm::checkedMulUnsigned(issueQueue->second, occurrenceCount);
      if (!entries)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "low_confidence_model_overflow: memory issue queue entries");
      if (llvm::Error error =
              addEntityCost(estimate, issueQueue->first, *entries))
        return error;
    }
    if (*kind != fabric::FabricEntityKind::FabricFuOccurrence)
      continue;
    const auto definition =
        view.fuTemplateOf(fabric::FabricFuOccurrenceRef(id));
    if (!definition)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "low_confidence_model_invalid: FU occurrence has no template");
    ++fuOccurrences[definition->id()];
  }

  for (const auto &[definitionId, localCount] : fuOccurrences) {
    const fabric::FabricFuTemplateRef definition(definitionId);
    const std::optional<std::uint64_t> concreteCount =
        llvm::checkedMulUnsigned(localCount, occurrenceCount);
    if (!concreteCount)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "low_confidence_model_overflow: FU occurrence count");
    for (const fabric::ResolvedFabricOpCapabilityView &capability :
         view.resolvedFabricOpCapabilities(definition)) {
      auto cost = operationCost(capability);
      if (!cost)
        return cost.takeError();
      if (llvm::Error error = addEntityCost(estimate, *cost, *concreteCount))
        return error;
    }
  }

  const std::optional<std::uint64_t> connectionCount = llvm::checkedMulUnsigned(
      occurrenceCount,
      static_cast<std::uint64_t>(view.pointConnections().size()));
  if (!connectionCount)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_overflow: point connection count");
  if (llvm::Error error =
          addEntityCost(estimate, EntityCost{2, 1, 100}, *connectionCount))
    return error;
  return llvm::Error::success();
}

llvm::Expected<PhysicalEstimate>
summarizeFabric(const fabric::FinalizedFabricRoot &root) {
  PhysicalEstimate estimate;
  const fabric::FabricArtifactView &view = root.view();
  if (llvm::Error error = summarizeFabricView(view, 1, estimate))
    return std::move(error);

  if (view.rootKind() != fabric::FabricRootKind::System) {
    for (const fabric::FabricArtifactView &module : view.importedModules()) {
      if (llvm::Error error = summarizeFabricView(module, 1, estimate))
        return std::move(error);
    }
    return estimate;
  }

  auto system = fabric::requireSystemRoot(view);
  if (!system)
    return system.takeError();
  std::vector<std::uint64_t> moduleOccurrences(view.importedModules().size(),
                                               0);
  std::set<std::pair<std::uint64_t, fabric::FabricEntityId>> seen;
  for (const fabric::FabricSpatialAttachmentRecordView &attachment :
       system->spatialAttachments()) {
    const fabric::SpatialCoreOccurrenceRef *spatial = nullptr;
    if (const auto *endpoint = attachment.spatialEndpoint.transport()) {
      if (endpoint->owner.kind() !=
          fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "low_confidence_model_invalid: transport attachment owner");
      spatial = std::get_if<fabric::SpatialCoreOccurrenceRef>(
          &endpoint->owner.payload);
    } else if (const auto *endpoint = attachment.spatialEndpoint.memory()) {
      if (endpoint->owner.kind() !=
          fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "low_confidence_model_invalid: memory attachment owner");
      spatial = std::get_if<fabric::SpatialCoreOccurrenceRef>(
          &endpoint->owner.payload);
    }
    if (!spatial ||
        attachment.moduleEndpoint.dependencyOrdinal >= moduleOccurrences.size())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "low_confidence_model_invalid: malformed Spatial attachment");
    if (seen.emplace(attachment.moduleEndpoint.dependencyOrdinal,
                     spatial->core.id())
            .second)
      ++moduleOccurrences[attachment.moduleEndpoint.dependencyOrdinal];
  }
  for (std::size_t ordinal = 0; ordinal < view.importedModules().size();
       ++ordinal) {
    if (llvm::Error error =
            summarizeFabricView(view.importedModules()[ordinal],
                                moduleOccurrences[ordinal], estimate))
      return std::move(error);
  }
  return estimate;
}

/// Bytes one memory actor firing moves across its service: every integer,
/// floating-point, or vector operand and result, so a load counts its data
/// result, a store its data operand, and a read-modify-write both.
llvm::Expected<std::uint64_t> memoryActorTransferBytes(mlir::Operation *actor);

llvm::Expected<std::uint64_t> typeBitWidth(mlir::Type type,
                                           mlir::Operation *owner) {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  if (type.isIndex()) {
    auto width = loom::getIndexBitWidth(owner);
    if (!width)
      return width.takeError();
    return *width;
  }
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    auto element = typeBitWidth(vector.getElementType(), owner);
    if (!element)
      return element.takeError();
    return loom::getFixedVectorBitWidth(vector, *element);
  }
  return 1;
}

llvm::Expected<std::uint64_t> memoryActorTransferBytes(mlir::Operation *actor) {
  std::uint64_t bytes = 0;
  const auto accumulate = [&](mlir::Type type) -> llvm::Error {
    if (!(type.isIntOrFloat() || mlir::isa<mlir::VectorType>(type)))
      return llvm::Error::success();
    auto width = typeBitWidth(type, actor);
    if (!width)
      return width.takeError();
    return accumulateScaled(bytes, *width / 8 + (*width % 8 != 0 ? 1 : 0), 1,
                            "memory actor transfer bytes");
  };
  for (mlir::Value operand : actor->getOperands())
    if (llvm::Error error = accumulate(operand.getType()))
      return std::move(error);
  for (mlir::Value result : actor->getResults())
    if (llvm::Error error = accumulate(result.getType()))
      return std::move(error);
  return bytes;
}

bool containsFloat(mlir::Type type) {
  if (mlir::isa<mlir::FloatType>(type))
    return true;
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    return containsFloat(vector.getElementType());
  return false;
}

llvm::Expected<std::uint64_t> actorActivityUnits(mlir::Operation *actor) {
  std::uint64_t bits = 0;
  bool floating = false;
  for (mlir::Type type : actor->getOperandTypes()) {
    auto width = typeBitWidth(type, actor);
    if (!width)
      return width.takeError();
    if (llvm::Error error =
            accumulateScaled(bits, *width, 1, "actor operand activity"))
      return std::move(error);
    floating |= containsFloat(type);
  }
  for (mlir::Type type : actor->getResultTypes()) {
    auto width = typeBitWidth(type, actor);
    if (!width)
      return width.takeError();
    if (llvm::Error error =
            accumulateScaled(bits, *width, 1, "actor result activity"))
      return std::move(error);
    floating |= containsFloat(type);
  }
  const std::uint64_t payloadBytes =
      std::max<std::uint64_t>(1, bits / 8 + (bits % 8 != 0 ? 1 : 0));
  std::uint64_t factor = floating ? 5 : 2;
  switch (dataflow::actorKind(dataflow::requireOperationSchema(actor))) {
  case dataflow::CanonicalDataflowActorKind::Compute:
    break;
  case dataflow::CanonicalDataflowActorKind::Control:
    factor = 1;
    break;
  case dataflow::CanonicalDataflowActorKind::Memory:
    factor = 4;
    break;
  }
  const std::optional<std::uint64_t> result =
      llvm::checkedMulUnsigned(payloadBytes, factor);
  if (!result)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_overflow: actor activity");
  return *result;
}

llvm::Expected<MetricResult> decimalMetric(std::uint64_t coefficient,
                                           std::int64_t exponent,
                                           llvm::StringRef name) {
  if (coefficient >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "low_confidence_model_overflow: %s",
                                   name.str().c_str());
  auto decimal =
      DecimalValue::get(static_cast<std::int64_t>(coefficient), exponent);
  if (!decimal)
    return decimal.takeError();
  return MetricResult{UncertaintyKind::Unquantified,
                      PointObservation{MetricValue{*decimal}},
                      {}};
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.low_confidence_analytic.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyConfigView{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyConfigView>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "low-confidence model config has the "
                                   "wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "low-confidence model config must be empty");
  return OwnerValue::get(EmptyConfigView{});
}

} // namespace

const ResolvedModelConfigViewContract &emptyLowConfidenceConfigView() {
  static const ResolvedModelConfigViewContract contract{
      configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};
  return contract;
}

llvm::Expected<CaseArtifactResolution> resolveSingleSubjectFabricCase(
    const ArtifactRootReference &subject,
    const ArtifactRootReference &fabricReference,
    const ArtifactStore &artifactStore,
    llvm::ArrayRef<CaseArtifactResolution::Entry> additionalEntries) {
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      resolved(&artifactRootReferenceLess);

  std::function<llvm::Error(const ArtifactRootReference &)> visit =
      [&](const ArtifactRootReference &current) -> llvm::Error {
    if (resolved.count(current) != 0)
      return llvm::Error::success();
    auto root = importCachedFabricRoot(current, artifactStore);
    if (!root)
      return root.takeError();
    std::vector<ArtifactRootReference> closure;
    for (const fabric::FabricDirectDependency &dependency :
         (*root)->directDependencies()) {
      if (llvm::Error error = visit(dependency.root))
        return error;
      closure.push_back(dependency.root);
      const auto &nested = resolved.find(dependency.root)->second;
      closure.insert(closure.end(), nested.begin(), nested.end());
    }
    std::sort(closure.begin(), closure.end(), artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    resolved.emplace(current, std::move(closure));
    return llvm::Error::success();
  };

  if (llvm::Error error = visit(fabricReference))
    return std::move(error);
  auto mergeEntry = [&](const CaseArtifactResolution::Entry &entry) {
    std::vector<ArtifactRootReference> &closure = resolved[entry.artifact];
    closure.insert(closure.end(), entry.dependencyClosure.begin(),
                   entry.dependencyClosure.end());
    std::sort(closure.begin(), closure.end(), artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  };
  mergeEntry({subject, {}});
  for (const CaseArtifactResolution::Entry &entry : additionalEntries) {
    if (auto bytes = artifactStore.get(entry.artifact); !bytes)
      return bytes.takeError();
    mergeEntry(entry);
  }

  std::vector<CaseArtifactResolution::Entry> entries;
  entries.reserve(resolved.size());
  for (auto &[artifact, closure] : resolved)
    entries.push_back({artifact, std::move(closure)});
  return CaseArtifactResolution::get(std::move(entries));
}

llvm::Expected<MetricResult>
LowConfidenceMetricSet::result(MetricKind metric) const {
  auto exponent = lowConfidenceMetricQuantumBase10Exponent(metric);
  if (!exponent)
    return exponent.takeError();
  switch (metric) {
  case MetricKind::Runtime:
    return decimalMetric(runtimePicoseconds, *exponent, "Runtime estimate");
  case MetricKind::LimitingClockFrequency:
    return decimalMetric(limitingClockFrequencyHertz, *exponent,
                         "frequency estimate");
  case MetricKind::TotalArea:
    return decimalMetric(totalAreaSquareMicrometers, *exponent,
                         "area estimate");
  case MetricKind::DynamicPower:
    return decimalMetric(dynamicPowerMicrowatts, *exponent,
                         "dynamic-power estimate");
  case MetricKind::LeakagePower:
    return decimalMetric(leakagePowerMicrowatts, *exponent,
                         "leakage-power estimate");
  case MetricKind::CycleCount:
  case MetricKind::ClockPeriod:
  case MetricKind::MaximumVoltageDrop:
  case MetricKind::LimitingClockFrequencyPredictionError:
  case MetricKind::TotalAreaPredictionError:
  case MetricKind::DynamicPowerPredictionError:
  case MetricKind::LeakagePowerPredictionError:
  case MetricKind::RuntimePredictionError:
    llvm_unreachable("unsupported metric passed quantum validation");
  }
  llvm_unreachable("unknown MetricKind");
}

llvm::Expected<std::int64_t>
lowConfidenceMetricQuantumBase10Exponent(MetricKind metric) {
  switch (metric) {
  case MetricKind::Runtime:
  case MetricKind::TotalArea:
    return -12;
  case MetricKind::LimitingClockFrequency:
    return 0;
  case MetricKind::DynamicPower:
  case MetricKind::LeakagePower:
    return -6;
  case MetricKind::CycleCount:
  case MetricKind::ClockPeriod:
  case MetricKind::MaximumVoltageDrop:
  case MetricKind::LimitingClockFrequencyPredictionError:
  case MetricKind::TotalAreaPredictionError:
  case MetricKind::DynamicPowerPredictionError:
  case MetricKind::LeakagePowerPredictionError:
  case MetricKind::RuntimePredictionError:
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_invalid: unsupported metric '%s'",
        toString(metric).str().c_str());
  }
  llvm_unreachable("unknown MetricKind");
}

namespace {

llvm::Expected<std::uint64_t>
lowConfidenceClockFrequencyHertz(const PhysicalEstimate &physical) {
  if (physical.criticalDelayPicoseconds == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_invalid: zero critical delay");
  const std::uint64_t frequency =
      kPicosecondsPerSecond / physical.criticalDelayPicoseconds;
  if (frequency == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_invalid: critical delay exceeds one second");
  return frequency;
}

} // namespace

llvm::Expected<LowConfidenceMetricSet>
estimateLowConfidenceMetrics(std::uint64_t instructionLeaves,
                             AnalyticWorkloadEstimate workload,
                             llvm::ArrayRef<AnalyticLaunchEstimate> launches,
                             const SystemPlatformModel &platform,
                             const fabric::FinalizedFabricRoot &fabricRoot) {
  auto runtime = estimateHostResidualPicoseconds(platform, instructionLeaves);
  if (!runtime)
    return runtime.takeError();
  const std::uint64_t hostResidualPicoseconds = *runtime;
  std::uint64_t configuredCores = 0;
  std::uint64_t invocationPicoseconds = 0;
  llvm::json::Array launchRecords;
  for (const AnalyticLaunchEstimate &launch : launches) {
    auto duration =
        estimateLaunchDuration(platform, launch, platform.accCoreCount);
    if (!duration)
      return duration.takeError();
    if (llvm::Error error = accumulateScaled(*runtime, duration->picoseconds, 1,
                                             "launch Runtime"))
      return std::move(error);
    if (llvm::Error error = accumulateScaled(
            invocationPicoseconds, duration->picoseconds, 1, "invocation phase"))
      return std::move(error);
    configuredCores = std::max(
        configuredCores, std::min(platform.accCoreCount, launch.activations));
    if (mapping_debug::enabled(mapping_debug::Level::Detail)) {
      llvm::json::Object record;
      record["static_graph_launch"] = launch.launch.entity.value();
      record["activations"] = launch.activations;
      record["compute_cycles_per_activation"] =
          launch.computeCyclesPerActivation;
      record["external_memory_bytes_per_activation"] =
          launch.externalMemoryBytesPerActivation;
      record["memory_transactions_per_activation"] =
          launch.memoryTransactionsPerActivation;
      record["memory_actors"] = launch.memoryActors;
      record["boundary_payload_bytes_per_activation"] =
          launch.boundaryPayloadBytesPerActivation;
      record["duration_ps"] = duration->picoseconds;
      record["compute_ps"] = duration->computePicoseconds;
      record["bandwidth_ps"] = duration->bandwidthPicoseconds;
      record["latency_chain_ps"] = duration->latencyChainPicoseconds;
      record["in_flight_requests"] = duration->inFlightRequests;
      record["fixed_ps"] = duration->fixedPicoseconds;
      record["dispatch_ps"] = duration->dispatchPicoseconds;
      record["bottleneck"] = [&]() -> llvm::StringRef {
        switch (duration->bottleneck) {
        case AnalyticLaunchBottleneck::Launch:
          return "launch";
        case AnalyticLaunchBottleneck::Compute:
          return "compute";
        case AnalyticLaunchBottleneck::MemoryBandwidth:
          return "memory_bandwidth";
        case AnalyticLaunchBottleneck::MemoryLatency:
          return "memory_latency";
        }
        return "unknown";
      }();
      launchRecords.push_back(std::move(record));
    }
  }
  auto configuration =
      estimateConfigurationResidencyPicoseconds(platform, configuredCores);
  if (!configuration)
    return configuration.takeError();
  if (llvm::Error error = accumulateScaled(*runtime, *configuration, 1,
                                           "configuration residency phase"))
    return std::move(error);
  // The estimate carries the same two accelerated-window phases the measured
  // System QoR projection reports: configuration residency streams the
  // Fabric-derived binary configuration image into every configured AccCore,
  // and the invocation phase runs the launch sites over that residency.
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "analytic_runtime_estimate";
        fields["instruction_leaves"] = instructionLeaves;
        fields["host_residual_ps"] = hostResidualPicoseconds;
        fields["configuration_residency_ps"] = *configuration;
        fields["invocation_ps"] = invocationPicoseconds;
        fields["configuration_image_bytes_per_core"] =
            platform.configurationBytesPerCore;
        fields["configured_acc_cores"] = configuredCores;
        fields["runtime_ps"] = *runtime;
        fields["acc_core_count"] = platform.accCoreCount;
        fields["memory_latency_ps"] = platform.memoryLatencyPicoseconds;
        fields["memory_service_ps_per_operation"] =
            platform.memoryServicePicosecondsPerOperation;
        fields["outstanding_requests"] = platform.accCoreOutstandingRequests;
        fields["memory_operation_issue_depth"] =
            platform.memoryOperationIssueDepth;
        // The service's bandwidth-delay product over one request: the
        // in-flight depth a latency-bound window would need to saturate it.
        fields["required_in_flight_requests"] =
            platform.requiredInFlightRequests;
        fields["launches"] = std::move(launchRecords);
      });
  if (*runtime >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_overflow: Runtime estimate");
  auto physical = summarizeFabric(fabricRoot);
  if (!physical)
    return physical.takeError();
  auto frequency = lowConfidenceClockFrequencyHertz(*physical);
  if (!frequency)
    return frequency.takeError();
  std::uint64_t dynamicActivity = workload.activityUnits;
  if (llvm::Error error =
          accumulateScaled(dynamicActivity, workload.boundaryPayloadBytes, 1,
                           "boundary dynamic activity"))
    return std::move(error);
  if (llvm::Error error =
          accumulateScaled(dynamicActivity, workload.memoryTransactions, 4,
                           "memory dynamic activity"))
    return std::move(error);
  const std::optional<std::uint64_t> dynamicPower =
      llvm::checkedMulUnsigned(dynamicActivity, std::uint64_t{2});
  if (!dynamicPower)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_overflow: dynamic power estimate");
  return LowConfidenceMetricSet{
      *runtime, *frequency,
      std::max<std::uint64_t>(physical->areaSquareMicrometers, 1),
      *dynamicPower, std::max<std::uint64_t>(physical->leakageMicrowatts, 1)};
}

llvm::Expected<std::optional<SystemPlatformModel>>
projectFabricPlatformModel(const fabric::FinalizedFabricRoot &fabricRoot) {
  auto system = fabric::requireSystemRoot(fabricRoot.view());
  if (!system) {
    llvm::consumeError(system.takeError());
    return std::optional<SystemPlatformModel>{};
  }
  auto platform = projectSystemPlatformModel(fabricRoot);
  if (!platform)
    return platform.takeError();
  return std::optional<SystemPlatformModel>(*platform);
}

llvm::Expected<LowConfidenceMetricSet> estimateLowConfidenceFabricMetrics(
    const fabric::FinalizedFabricRoot &fabricRoot,
    std::uint64_t activityPartsPer1024) {
  auto physical = summarizeFabric(fabricRoot);
  if (!physical)
    return physical.takeError();
  auto frequency = lowConfidenceClockFrequencyHertz(*physical);
  if (!frequency)
    return frequency.takeError();

  const std::uint64_t area =
      std::max<std::uint64_t>(physical->areaSquareMicrometers, 1);
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(area) * activityPartsPer1024;
  const unsigned __int128 rounded =
      (scaled + 1023) / static_cast<unsigned __int128>(1024);
  if (rounded > std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "low_confidence_model_overflow: hardware dynamic power estimate");
  return LowConfidenceMetricSet{
      0, *frequency, area, static_cast<std::uint64_t>(rounded),
      std::max<std::uint64_t>(physical->leakageMicrowatts, 1)};
}

llvm::Expected<std::uint64_t> lowConfidenceClockPeriodPicoseconds(
    const fabric::FinalizedFabricRoot &fabricRoot) {
  auto physical = summarizeFabric(fabricRoot);
  if (!physical)
    return physical.takeError();
  auto frequency = lowConfidenceClockFrequencyHertz(*physical);
  if (!frequency)
    return frequency.takeError();
  return physical->criticalDelayPicoseconds;
}

llvm::Expected<std::optional<AnalyticWorkloadEstimate>>
projectCanonicalDataflowWorkloadImpl(
    const ::dataflow::CanonicalDataflowProgramView &program,
    const fabric::FinalizedFabricRoot &fabricRoot,
    std::optional<::dataflow::GraphRef> selectedGraph) {
  if (selectedGraph) {
    auto resolved = program.resolve(*selectedGraph);
    if (!resolved)
      return resolved.takeError();
  }
  std::map<std::vector<std::uint8_t>, ActorDemand> actorDemands;
  for (const dataflow::CanonicalActorView &actor : program.actors()) {
    if (selectedGraph && actor.graph != *selectedGraph)
      continue;
    auto projection =
        dataflow::projectRegisteredActorSchemaProjectionBytes(actor.op);
    if (!projection)
      return projection.takeError();
    std::vector<std::uint8_t> key(projection->bytes().begin(),
                                  projection->bytes().end());
    ActorDemand &demand = actorDemands[key];
    if (!demand.representative)
      demand.representative = actor.op;
    ++demand.count;
  }

  frontend::FabricCapabilityIndex capabilities(fabricRoot.view());
  AnalyticWorkloadEstimate workload;
  for (const auto &entry : actorDemands) {
    const ActorDemand &demand = entry.second;
    llvm::Expected<std::uint64_t> capacity =
        dataflow::isCanonicalDataflowActor(
            demand.representative, dataflow::CanonicalDataflowActorKind::Memory)
            ? capabilities.admittingMemoryResourceCount(demand.representative)
            : capabilities.admittingOperationResourceCount(
                  demand.representative);
    if (!capacity)
      return capacity.takeError();
    if (*capacity == 0) {
      mapping_debug::emit(
          mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "structured_fabric_capability";
            fields["closure_status"] = "proven_infeasible";
            fields["operation"] =
                demand.representative->getName().getStringRef();
            fields["demand_count"] = demand.count;
          });
      return std::optional<AnalyticWorkloadEstimate>{};
    }
    const std::uint64_t pressure =
        demand.count / *capacity + (demand.count % *capacity != 0 ? 1 : 0);
    workload.schedulingPressure =
        std::max(workload.schedulingPressure, pressure);
    auto activity = actorActivityUnits(demand.representative);
    if (!activity)
      return activity.takeError();
    if (llvm::Error error =
            accumulateScaled(workload.activityUnits, *activity, demand.count,
                             "Canonical Dataflow activity"))
      return std::move(error);
    if (dataflow::isCanonicalDataflowActor(
            demand.representative,
            dataflow::CanonicalDataflowActorKind::Memory)) {
      if (llvm::Error error =
              accumulateScaled(workload.memoryTransactions, 1, demand.count,
                               "Canonical Dataflow memory transactions"))
        return std::move(error);
      auto bytes = memoryActorTransferBytes(demand.representative);
      if (!bytes)
        return bytes.takeError();
      if (llvm::Error error =
              accumulateScaled(workload.externalMemoryBytes, *bytes,
                               demand.count,
                               "Canonical Dataflow memory bytes"))
        return std::move(error);
    }
  }

  llvm::SmallVector<dataflow::GraphRef> coveredGraphs;
  if (selectedGraph) {
    coveredGraphs.push_back(*selectedGraph);
  } else {
    coveredGraphs.reserve(program.graphs().size());
    for (const dataflow::CanonicalGraphView &graph : program.graphs())
      coveredGraphs.push_back(graph.ref);
  }
  auto schedule =
      dataflow::deriveStaticScheduleAnalysis(program, coveredGraphs);
  if (!schedule)
    return schedule.takeError();
  std::uint64_t criticalPathPressure = 0;
  if (selectedGraph) {
    criticalPathPressure = schedule->graphCriticalLength(*selectedGraph);
  } else if (!program.staticGraphLaunches().empty()) {
    for (const dataflow::CanonicalStaticGraphLaunchView &launch :
         program.staticGraphLaunches()) {
      const std::optional<std::uint64_t> updated = llvm::checkedAddUnsigned(
          criticalPathPressure, schedule->graphCriticalLength(launch.callee));
      if (!updated)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "low_confidence_model_overflow: graph critical path pressure");
      criticalPathPressure = *updated;
    }
  } else {
    for (dataflow::GraphRef graph : coveredGraphs)
      criticalPathPressure =
          std::max(criticalPathPressure, schedule->graphCriticalLength(graph));
  }
  workload.criticalPathLength = criticalPathPressure;
  for (const dataflow::StaticActorCriticality &actor : schedule->actors())
    if (llvm::is_contained(coveredGraphs, actor.graph))
      workload.recurrenceLength =
          std::max(workload.recurrenceLength, actor.recurrenceCriticalLength);

  workload.graphActivations =
      selectedGraph ? 1 : program.staticGraphLaunches().size();
  for (const dataflow::CanonicalGraphView &graphView : program.graphs()) {
    if (selectedGraph && graphView.ref != *selectedGraph)
      continue;
    auto graph = llvm::dyn_cast_or_null<dataflow::GraphOp>(graphView.op);
    if (!graph)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "low_confidence_model_invalid: graph view has the wrong owner");
    mlir::FunctionType type = graph.getFunctionType();
    for (auto [ordinal, portType] : llvm::enumerate(type.getInputs())) {
      if (graph.getInputPortKind(ordinal) == dataflow::GraphPortKind::Memory) {
        if (llvm::Error error =
                accumulateScaled(workload.memoryBoundaryBindings, 1, 1,
                                 "graph memory input binding"))
          return std::move(error);
        continue;
      }
      auto width = typeBitWidth(portType, graph);
      if (!width)
        return width.takeError();
      const std::uint64_t bytes = *width / 8 + (*width % 8 != 0 ? 1 : 0);
      if (llvm::Error error =
              accumulateScaled(workload.boundaryPayloadBytes, bytes, 1,
                               "graph input boundary bytes"))
        return std::move(error);
    }
    for (auto [ordinal, portType] : llvm::enumerate(type.getResults())) {
      if (graph.getResultPortKind(ordinal) == dataflow::GraphPortKind::Memory) {
        if (llvm::Error error =
                accumulateScaled(workload.memoryBoundaryBindings, 1, 1,
                                 "graph memory result binding"))
          return std::move(error);
        continue;
      }
      auto width = typeBitWidth(portType, graph);
      if (!width)
        return width.takeError();
      const std::uint64_t bytes = *width / 8 + (*width % 8 != 0 ? 1 : 0);
      if (llvm::Error error =
              accumulateScaled(workload.boundaryPayloadBytes, bytes, 1,
                               "graph result boundary bytes"))
        return std::move(error);
    }
  }
  return std::optional<AnalyticWorkloadEstimate>(workload);
}

llvm::Expected<std::optional<AnalyticWorkloadEstimate>>
projectCanonicalDataflowWorkload(
    const ::dataflow::CanonicalDataflowProgramView &program,
    const fabric::FinalizedFabricRoot &fabricRoot) {
  return projectCanonicalDataflowWorkloadImpl(program, fabricRoot,
                                              std::nullopt);
}

llvm::Expected<std::optional<AnalyticWorkloadEstimate>>
projectCanonicalDataflowGraphWorkload(
    const ::dataflow::CanonicalDataflowProgramView &program,
    ::dataflow::GraphRef graph, const fabric::FinalizedFabricRoot &fabricRoot) {
  return projectCanonicalDataflowWorkloadImpl(program, fabricRoot, graph);
}

} // namespace loom::evaluation::models::detail
