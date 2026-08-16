#include "SimulationExecutionInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelDescriptor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {

int compareSpatialEventCoordinates(const SpatialEventCoordinate &lhs,
                                   const SpatialEventCoordinate &rhs) {
  if (lhs.referenceCycle.denominator() == rhs.referenceCycle.denominator()) {
    if (lhs.referenceCycle.numerator() != rhs.referenceCycle.numerator())
      return lhs.referenceCycle.numerator() < rhs.referenceCycle.numerator()
                 ? -1
                 : 1;
    if (lhs.delta == rhs.delta)
      return 0;
    return lhs.delta < rhs.delta ? -1 : 1;
  }
  using u128 = unsigned __int128;
  const u128 lhsScaled = static_cast<u128>(lhs.referenceCycle.numerator()) *
                         rhs.referenceCycle.denominator();
  const u128 rhsScaled = static_cast<u128>(rhs.referenceCycle.numerator()) *
                         lhs.referenceCycle.denominator();
  if (lhsScaled != rhsScaled)
    return lhsScaled < rhsScaled ? -1 : 1;
  if (lhs.delta == rhs.delta)
    return 0;
  return lhs.delta < rhs.delta ? -1 : 1;
}

namespace {

using detail::WireReader;
using detail::WireWriter;

llvm::Expected<std::shared_ptr<const evaluation::EvaluationRequest>>
importCachedRequest(const ArtifactRootReference &reference,
                    const evaluation::CaseArtifactResolution &resolution,
                    const ArtifactStore &store, const BlobStore &blobs) {
  const std::array<ArtifactRootReference, 1> references{reference};
  return evaluation::importCachedArtifact<evaluation::EvaluationRequest>(
      store, &blobs, references, [&]() {
        return evaluation::importEvaluationRequest(reference, resolution, store,
                                                   blobs);
      });
}

llvm::Expected<std::shared_ptr<const ImportedSpatialSimulationInputs>>
importCachedSpatialInputs(const ArtifactRootReference &workload,
                          const ArtifactRootReference &runtimeInput,
                          const ArtifactStore &store) {
  const std::array<ArtifactRootReference, 2> references{workload, runtimeInput};
  return evaluation::importCachedArtifact<ImportedSpatialSimulationInputs>(
      store, nullptr, references, [&]() {
        return importSpatialSimulationInputs(workload, runtimeInput, store);
      });
}

llvm::Error validateProgress(const SpatialProgressObservations &progress,
                             const ExecutionTerminal &terminal) {
  if (compareSpatialEventCoordinates(progress.launchAccepted,
                                     progress.terminalObserved) > 0)
    return detail::invalid("simulation execution: launch coordinate follows "
                           "the terminal coordinate");
  if (progress.graphRetirementVisible) {
    if (compareSpatialEventCoordinates(progress.launchAccepted,
                                       *progress.graphRetirementVisible) > 0 ||
        compareSpatialEventCoordinates(*progress.graphRetirementVisible,
                                       progress.terminalObserved) > 0)
      return detail::invalid("simulation execution: graph-retirement "
                             "coordinate is out of order");
  }
  if (std::holds_alternative<RetiredExecution>(terminal) &&
      !progress.graphRetirementVisible)
    return detail::invalid(
        "simulation execution: Retired execution requires graph retirement");
  return llvm::Error::success();
}

void encodeCoordinate(WireWriter &writer,
                      const SpatialEventCoordinate &coordinate) {
  writer.u64(coordinate.referenceCycle.numerator());
  writer.u64(coordinate.referenceCycle.denominator());
  writer.u64(coordinate.delta);
}

llvm::Expected<SpatialEventCoordinate> decodeCoordinate(WireReader &reader) {
  llvm::Expected<std::uint64_t> numerator = reader.u64();
  if (!numerator)
    return numerator.takeError();
  llvm::Expected<std::uint64_t> denominator = reader.u64();
  if (!denominator)
    return denominator.takeError();
  llvm::Expected<std::uint64_t> delta = reader.u64();
  if (!delta)
    return delta.takeError();
  llvm::Expected<evaluation::ExactRatio> ratio =
      evaluation::ExactRatio::get(*numerator, *denominator);
  if (!ratio)
    return ratio.takeError();
  if (ratio->numerator() != *numerator || ratio->denominator() != *denominator)
    return detail::invalid(
        "simulation execution: noncanonical exact cycle ratio");
  return SpatialEventCoordinate{std::move(*ratio), *delta};
}

void encodeProgress(WireWriter &writer,
                    const SpatialProgressObservations &progress) {
  encodeCoordinate(writer, progress.launchAccepted);
  if (progress.graphRetirementVisible) {
    writer.u32(1);
    encodeCoordinate(writer, *progress.graphRetirementVisible);
  } else {
    writer.u32(0);
  }
  encodeCoordinate(writer, progress.terminalObserved);
}

llvm::Expected<SpatialProgressObservations> decodeProgress(WireReader &reader) {
  auto launch = decodeCoordinate(reader);
  if (!launch)
    return launch.takeError();
  llvm::Expected<std::uint32_t> retirementTag = reader.u32();
  if (!retirementTag)
    return retirementTag.takeError();
  std::optional<SpatialEventCoordinate> retirement;
  if (*retirementTag == 1) {
    auto coordinate = decodeCoordinate(reader);
    if (!coordinate)
      return coordinate.takeError();
    retirement = std::move(*coordinate);
  } else if (*retirementTag != 0) {
    return detail::invalid(
        "simulation execution: unknown retirement-anchor state");
  }
  auto terminal = decodeCoordinate(reader);
  if (!terminal)
    return terminal.takeError();
  return SpatialProgressObservations{std::move(*launch), std::move(retirement),
                                     std::move(*terminal)};
}

llvm::Expected<const evaluation::ModelOutputSlotDescriptor *>
resolveSimulationOutputSlot(const evaluation::EvaluationRequest &request) {
  const evaluation::EvaluationModelDescriptor *descriptor =
      evaluation::resolveEvaluationModelDescriptor(request);
  if (!descriptor)
    return detail::invalid(
        "simulation execution: Request model descriptor is unavailable");
  const evaluation::ModelOutputSlotDescriptor *executionSlot = nullptr;
  for (const evaluation::ModelOutputSlotDescriptor &slot :
       descriptor->outputSlots) {
    if (*slot.schema == simulationExecutionSchema) {
      if (executionSlot)
        return detail::invalid("simulation execution: model descriptor has "
                               "multiple execution output slots");
      executionSlot = &slot;
    }
  }
  if (!executionSlot)
    return detail::invalid("simulation execution: model descriptor has no "
                           "SimulationExecution output slot");
  if (executionSlot->cardinality(evaluation::EvidenceOutcomeKind::Completed) !=
          evaluation::ArtifactCollectionCardinality::ExactlyOne ||
      executionSlot->cardinality(
          evaluation::EvidenceOutcomeKind::Unsupported) !=
          evaluation::ArtifactCollectionCardinality::Forbidden ||
      executionSlot->cardinality(
          evaluation::EvidenceOutcomeKind::ExecutionFailed) !=
          evaluation::ArtifactCollectionCardinality::Forbidden)
    return detail::invalid("simulation execution: model output cardinality "
                           "does not match the execution contract");
  if (executionSlot->cardinality(
          evaluation::EvidenceOutcomeKind::CancelledOrTimeout) ==
      evaluation::ArtifactCollectionCardinality::OneOrMore)
    return detail::invalid("simulation execution: model output cardinality "
                           "permits multiple stopped executions");
  return executionSlot;
}

llvm::Expected<std::vector<std::uint8_t>> canonicalTerminalWitness(
    const HaltedExecution &halted,
    const evaluation::FindingTerminalWitnessContext &context) {
  const evaluation::FindingDescriptor *descriptor =
      evaluation::findFindingDescriptor(halted.findingKind);
  if (!descriptor || !descriptor->terminalWitnessCodec)
    return detail::invalid("simulation execution: Halted terminal has no "
                           "registered terminal-witness owner");
  if (!halted.witness)
    return detail::invalid(
        "simulation execution: Halted terminal has no witness");
  const evaluation::FindingTerminalWitnessCodec &codec =
      *descriptor->terminalWitnessCodec;
  if (llvm::Error error = codec.validate(halted.witness, context))
    return std::move(error);
  auto encoded = codec.encode(halted.witness);
  if (!encoded)
    return encoded.takeError();
  auto adopted = codec.decode(*encoded);
  if (!adopted)
    return adopted.takeError();
  if (!*adopted)
    return detail::invalid(
        "simulation execution: terminal witness decoder returned no value");
  if (llvm::Error error = codec.validate(*adopted, context))
    return std::move(error);
  auto reencoded = codec.encode(*adopted);
  if (!reencoded)
    return reencoded.takeError();
  if (*reencoded != *encoded)
    return detail::invalid(
        "simulation execution: terminal witness is not canonical");
  return encoded;
}

llvm::Error
validateTerminal(const ExecutionTerminal &terminal,
                 const evaluation::FindingTerminalWitnessContext &context) {
  if (const auto *halted = std::get_if<HaltedExecution>(&terminal)) {
    auto canonical = canonicalTerminalWitness(*halted, context);
    if (!canonical)
      return canonical.takeError();
  }
  return llvm::Error::success();
}

llvm::Error validateExecution(const SpatialSimulationExecution &execution,
                              const detail::SpatialExecutionContext &context) {
  if (std::holds_alternative<StoppedByLimitExecution>(execution.terminal) &&
      context.stoppedExecutionCardinality ==
          evaluation::ArtifactCollectionCardinality::Forbidden)
    return detail::invalid("simulation execution: model output slot does not "
                           "retain StoppedByLimit execution");
  if (!execution.activitySummaries.empty()) {
    if (llvm::Error error = detail::validateActorActivitySummaries(
            execution.activitySummaries, execution.terminal,
            execution.progressObservations, context))
      return error;
  }
  const evaluation::FindingTerminalWitnessContext terminalContext(
      *context.request, *context.resolution, *context.artifactStore,
      *context.blobStore);
  if (llvm::Error error = validateTerminal(execution.terminal, terminalContext))
    return error;
  if (llvm::Error error =
          validateProgress(execution.progressObservations, execution.terminal))
    return error;
  return detail::validateSpatialFunctionalObservations(
      execution.functionalObservations, execution.terminal, context);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeExecution(const SpatialSimulationExecution &execution,
                const detail::SpatialExecutionContext &context) {
  std::vector<std::uint8_t> bytes =
      encodeArtifactRootReference(execution.request);
  WireWriter writer;
  const evaluation::FindingTerminalWitnessContext terminalContext(
      *context.request, *context.resolution, *context.artifactStore,
      *context.blobStore);
  if (llvm::Error error = detail::encodeExecutionTerminal(
          writer, execution.terminal, terminalContext))
    return std::move(error);
  detail::encodeSpatialFunctionalObservations(
      writer, execution.functionalObservations, context);
  encodeProgress(writer, execution.progressObservations);
  detail::encodeActorActivitySummaries(writer, execution.activitySummaries);
  std::vector<std::uint8_t> tail = writer.take();
  bytes.insert(bytes.end(), tail.begin(), tail.end());
  return bytes;
}

llvm::Expected<ExecutionTerminal>
decodeTerminal(WireReader &reader,
               const evaluation::FindingTerminalWitnessContext &context) {
  llvm::Expected<std::uint32_t> tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0)
    return ExecutionTerminal{RetiredExecution{}};
  if (*tag == 2)
    return ExecutionTerminal{StoppedByLimitExecution{}};
  if (*tag != 1)
    return detail::invalid(
        "simulation execution: unknown terminal discriminant");
  llvm::Expected<std::uint32_t> kind = reader.u32();
  if (!kind)
    return kind.takeError();
  llvm::Expected<std::uint64_t> byteCount = reader.u64();
  if (!byteCount)
    return byteCount.takeError();
  if (*byteCount > std::numeric_limits<std::size_t>::max())
    return detail::invalid(
        "simulation execution: terminal witness is too large");
  auto payload = reader.bytes(static_cast<std::size_t>(*byteCount));
  if (!payload)
    return payload.takeError();
  const evaluation::FindingKind findingKind(*kind);
  const evaluation::FindingDescriptor *descriptor =
      evaluation::findFindingDescriptor(findingKind);
  if (!descriptor || !descriptor->terminalWitnessCodec)
    return detail::invalid("simulation execution: unknown terminal finding");
  const evaluation::FindingTerminalWitnessCodec &codec =
      *descriptor->terminalWitnessCodec;
  auto witness = codec.decode(*payload);
  if (!witness)
    return witness.takeError();
  if (!*witness)
    return detail::invalid(
        "simulation execution: terminal witness decoder returned no value");
  if (llvm::Error error = codec.validate(*witness, context))
    return std::move(error);
  auto canonical = codec.encode(*witness);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != *payload)
    return detail::invalid(
        "simulation execution: terminal witness is not canonical");
  return ExecutionTerminal{HaltedExecution{findingKind, std::move(*witness)}};
}

llvm::Expected<SpatialSimulationExecution>
decodeExecution(llvm::ArrayRef<std::uint8_t> bytes,
                const evaluation::CaseArtifactResolution &resolution,
                const ArtifactStore &store, const BlobStore &blobs) {
  auto requestPrefix = decodeArtifactRootReferencePrefix(bytes);
  if (!requestPrefix)
    return requestPrefix.takeError();
  auto context = detail::resolveSpatialExecutionContext(
      requestPrefix->reference, resolution, store, blobs);
  if (!context)
    return context.takeError();
  WireReader reader(bytes.drop_front(requestPrefix->byteCount));
  const evaluation::FindingTerminalWitnessContext terminalContext(
      *context->request, *context->resolution, *context->artifactStore,
      *context->blobStore);
  auto terminal = decodeTerminal(reader, terminalContext);
  if (!terminal)
    return terminal.takeError();
  auto functional =
      detail::decodeSpatialFunctionalObservations(reader, *context);
  if (!functional)
    return functional.takeError();
  auto progress = decodeProgress(reader);
  if (!progress)
    return progress.takeError();
  auto activities = detail::decodeActorActivitySummaries(reader);
  if (!activities)
    return activities.takeError();
  if (!reader.atEnd())
    return detail::invalid("simulation execution: trailing bytes");
  SpatialSimulationExecution execution{
      requestPrefix->reference, std::move(*terminal), std::move(*functional),
      std::move(*progress), std::move(*activities)};
  if (llvm::Error error = validateExecution(execution, *context))
    return std::move(error);
  auto canonical = encodeExecution(execution, *context);
  if (!canonical)
    return canonical.takeError();
  if (!std::equal(canonical->begin(), canonical->end(), bytes.begin(),
                  bytes.end()))
    return detail::invalid("simulation execution: bytes are not canonical");
  return execution;
}

} // namespace

namespace detail {

llvm::Expected<evaluation::ArtifactCollectionCardinality>
resolveSimulationOutputCardinality(
    const evaluation::EvaluationRequest &request) {
  auto outputSlot = resolveSimulationOutputSlot(request);
  if (!outputSlot)
    return outputSlot.takeError();
  return (*outputSlot)
      ->cardinality(evaluation::EvidenceOutcomeKind::CancelledOrTimeout);
}

llvm::Error validateExecutionTerminal(
    const ExecutionTerminal &terminal,
    const evaluation::FindingTerminalWitnessContext &context) {
  return validateTerminal(terminal, context);
}

llvm::Error encodeExecutionTerminal(
    WireWriter &writer, const ExecutionTerminal &terminal,
    const evaluation::FindingTerminalWitnessContext &context) {
  if (std::holds_alternative<RetiredExecution>(terminal)) {
    writer.u32(0);
    return llvm::Error::success();
  }
  if (const auto *halted = std::get_if<HaltedExecution>(&terminal)) {
    auto witness = canonicalTerminalWitness(*halted, context);
    if (!witness)
      return witness.takeError();
    writer.u32(1);
    writer.u32(halted->findingKind.ordinal());
    writer.u64(witness->size());
    writer.bytes(*witness);
    return llvm::Error::success();
  }
  writer.u32(2);
  return llvm::Error::success();
}

llvm::Expected<ExecutionTerminal> decodeExecutionTerminal(
    WireReader &reader,
    const evaluation::FindingTerminalWitnessContext &context) {
  return decodeTerminal(reader, context);
}

llvm::Expected<SpatialExecutionContext> resolveSpatialExecutionContext(
    const ArtifactRootReference &requestReference,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto request =
      importCachedRequest(requestReference, resolution, store, blobs);
  if (!request)
    return request.takeError();
  auto stoppedCardinality = resolveSimulationOutputCardinality(**request);
  if (!stoppedCardinality)
    return stoppedCardinality.takeError();
  if (!(*request)->workload() || !(*request)->runtimeInput())
    return invalid("simulation execution: Request workload inputs are not "
                   "total");
  auto inputs = importCachedSpatialInputs(*(*request)->workload(),
                                          *(*request)->runtimeInput(), store);
  if (!inputs)
    return inputs.takeError();
  auto view = (*inputs)->dataflow.view();
  if (!view)
    return view.takeError();
  auto launch =
      resolveLaunchContext(*view, (*inputs)->workload.spatial()->launchRef);
  if (!launch)
    return launch.takeError();
  const CanonicalSimulationWorkload *workload = &(*inputs)->workload;
  const CanonicalSimulationRuntimeInput *runtimeInput =
      &(*inputs)->runtimeInput;
  return SpatialExecutionContext{std::move(*request),
                                 std::move(*inputs),
                                 workload,
                                 runtimeInput,
                                 std::move(*view),
                                 std::move(*launch),
                                 *stoppedCardinality,
                                 &resolution,
                                 &store,
                                 &blobs};
}

llvm::Expected<SpatialExecutionContext> resolveSpatialEngineResultContext(
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const ArtifactStore &store) {
  auto inputs = importCachedSpatialInputs(workloadReference,
                                          runtimeInputReference, store);
  if (!inputs)
    return inputs.takeError();
  auto view = (*inputs)->dataflow.view();
  if (!view)
    return view.takeError();
  auto launch =
      resolveLaunchContext(*view, (*inputs)->workload.spatial()->launchRef);
  if (!launch)
    return launch.takeError();
  const CanonicalSimulationWorkload *workload = &(*inputs)->workload;
  const CanonicalSimulationRuntimeInput *runtimeInput =
      &(*inputs)->runtimeInput;
  return SpatialExecutionContext{
      {},
      std::move(*inputs),
      workload,
      runtimeInput,
      std::move(*view),
      std::move(*launch),
      evaluation::ArtifactCollectionCardinality::OneOrMore,
      nullptr,
      &store,
      nullptr};
}

llvm::Expected<SpatialExecutionContext> resolveSpatialEngineResultContext(
    const ImportedSpatialSimulationInputs &inputs) {
  auto view = inputs.dataflow.view();
  if (!view)
    return view.takeError();
  auto launch =
      resolveLaunchContext(*view, inputs.workload.spatial()->launchRef);
  if (!launch)
    return launch.takeError();
  return SpatialExecutionContext{
      {},
      {},
      &inputs.workload,
      &inputs.runtimeInput,
      std::move(*view),
      std::move(*launch),
      evaluation::ArtifactCollectionCardinality::OneOrMore,
      nullptr,
      nullptr,
      nullptr};
}

llvm::Expected<SpatialExecutionContext> resolveSpatialEngineResultContext(
    const ImportedSpatialSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  const auto *spatialWorkload = workload.workload.spatial();
  const auto *spatialRuntime = runtimeInput.spatial();
  if (!spatialWorkload || !spatialRuntime ||
      spatialRuntime->workloadIdentity != workload.workload.identity())
    return invalid("simulation execution: Spatial engine context owners are "
                   "inconsistent");
  auto view = workload.dataflow.view();
  if (!view)
    return view.takeError();
  auto launch = resolveLaunchContext(*view, spatialWorkload->launchRef);
  if (!launch)
    return launch.takeError();
  return SpatialExecutionContext{
      {},
      {},
      &workload.workload,
      &runtimeInput,
      std::move(*view),
      std::move(*launch),
      evaluation::ArtifactCollectionCardinality::OneOrMore,
      nullptr,
      nullptr,
      nullptr};
}

llvm::Error
validateSpatialProgressObservations(const SpatialProgressObservations &progress,
                                    const ExecutionTerminal &terminal) {
  return validateProgress(progress, terminal);
}

void encodeSpatialProgressObservations(
    WireWriter &writer, const SpatialProgressObservations &progress) {
  encodeProgress(writer, progress);
}

llvm::Expected<SpatialProgressObservations>
decodeSpatialProgressObservations(WireReader &reader) {
  return decodeProgress(reader);
}

llvm::Expected<SpatialSimulationExecution> decodeSpatialSimulationExecution(
    llvm::ArrayRef<std::uint8_t> bytes,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs) {
  return decodeExecution(bytes, resolution, store, blobs);
}

} // namespace detail

const evaluation::FindingOccurrenceCodec &terminalWitnessRefOccurrenceCodec() {
  static const evaluation::FindingOccurrenceCodec codec{
      {"loom.terminal_witness_ref", {1, 0}},
      [](const evaluation::OwnerValue &occurrence)
          -> llvm::Expected<std::vector<std::uint8_t>> {
        const auto *reference = occurrence.getIf<TerminalWitnessRef>();
        if (!reference)
          return detail::invalid(
              "terminal witness reference has the wrong owner type");
        WireWriter writer;
        writer.u32(reference->executionOutputSlot.ordinal());
        writer.u64(reference->executionOutputOrdinal);
        return writer.take();
      },
      [](llvm::ArrayRef<std::uint8_t> payload)
          -> llvm::Expected<evaluation::OwnerValue> {
        WireReader reader(payload);
        auto slot = reader.u32();
        if (!slot)
          return slot.takeError();
        auto ordinal = reader.u64();
        if (!ordinal)
          return ordinal.takeError();
        if (!reader.atEnd())
          return detail::invalid(
              "terminal witness reference has trailing bytes");
        return evaluation::OwnerValue::get(TerminalWitnessRef{
            evaluation::ModelOutputSlotRef(*slot), *ordinal});
      },
      [](const evaluation::OwnerValue &occurrence,
         const evaluation::FindingOccurrenceContext &context) -> llvm::Error {
        const auto *reference = occurrence.getIf<TerminalWitnessRef>();
        if (!reference)
          return detail::invalid(
              "terminal witness reference has the wrong owner type");
        const ArtifactRootReference *executionReference = context.resolveOutput(
            reference->executionOutputSlot, reference->executionOutputOrdinal);
        if (!executionReference ||
            executionReference->schemaIdentity !=
                simulationExecutionSchema.identity ||
            executionReference->schemaVersion !=
                simulationExecutionSchema.version)
          return detail::invalid(
              "terminal witness reference does not resolve an execution");
        auto execution = importSimulationExecution(
            *executionReference, context.resolution(), context.artifactStore(),
            context.blobStore());
        if (!execution)
          return execution.takeError();
        if (execution->request() !=
            evaluation::evaluationRequestReference(context.request()))
          return detail::invalid(
              "terminal witness execution has a foreign Request");
        const auto *halted =
            std::get_if<HaltedExecution>(&execution->terminal());
        const evaluation::FindingRequest *request =
            context.request().resolve(context.findingRequestOrdinal());
        if (!halted || !request || halted->findingKind != request->query().kind)
          return detail::invalid(
              "terminal witness execution has a foreign finding kind");
        return llvm::Error::success();
      }};
  return codec;
}

llvm::Expected<CanonicalSimulationExecution> finalizeSimulationExecution(
    const SpatialSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto context = detail::resolveSpatialExecutionContext(
      execution.request, resolution, store, blobs);
  if (!context)
    return context.takeError();
  if (llvm::Error error = validateExecution(execution, *context))
    return std::move(error);
  auto bytes = encodeExecution(execution, *context);
  if (!bytes)
    return bytes.takeError();
  CanonicalSemanticBytes canonical(std::move(*bytes));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationExecutionSchema, canonical);
  return CanonicalSimulationExecution(identity, execution,
                                      std::move(canonical));
}

llvm::Expected<ArtifactRootReference>
publishSimulationExecution(const CanonicalSimulationExecution &execution,
                           const ArtifactStore &store) {
  auto identity =
      store.put(simulationExecutionSchema, execution.canonicalBytes());
  if (!identity)
    return identity.takeError();
  if (*identity != execution.identity())
    return detail::invalid(
        "ArtifactStore returned a foreign SimulationExecution identity");
  return ArtifactRootReference{simulationExecutionSchema.identity.str(),
                               simulationExecutionSchema.version,
                               std::move(*identity)};
}

llvm::Expected<ArtifactRootReference>
simulationExecutionRequestReference(const ArtifactRootReference &reference,
                                    const ArtifactStore &store) {
  if (reference.schemaIdentity != simulationExecutionSchema.identity ||
      reference.schemaVersion != simulationExecutionSchema.version)
    return detail::invalid("foreign SimulationExecution reference schema");
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto prefix = decodeArtifactRootReferencePrefix(bytes->bytes());
  if (!prefix)
    return prefix.takeError();
  return std::move(prefix->reference);
}

} // namespace loom::sim
