#include "SimulationExecutionInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/OwnerError.h"

#include <algorithm>
#include <cstdint>
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

llvm::Error validateTerminal(const ExecutionTerminal &terminal) {
  if (const auto *halted = std::get_if<HaltedExecution>(&terminal)) {
    const evaluation::FindingDescriptor *descriptor =
        evaluation::findFindingDescriptor(halted->findingKind);
    if (!descriptor || !descriptor->terminalWitnessSchema)
      return detail::invalid("simulation execution: Halted terminal has no "
                             "registered terminal-witness owner");
    if (!halted->witness)
      return detail::invalid(
          "simulation execution: Halted terminal has no witness");
    return evaluation::requireFindingOccurrenceOwner(*descriptor);
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
  if (llvm::Error error = validateTerminal(execution.terminal))
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
  if (std::holds_alternative<RetiredExecution>(execution.terminal)) {
    writer.u32(0);
  } else if (const auto *halted =
                 std::get_if<HaltedExecution>(&execution.terminal)) {
    writer.u32(1);
    writer.u32(halted->findingKind.ordinal());
    return detail::invalid(
        "simulation execution: Halted witness encoder is unavailable");
  } else {
    writer.u32(2);
  }
  detail::encodeSpatialFunctionalObservations(
      writer, execution.functionalObservations, context);
  encodeProgress(writer, execution.progressObservations);
  detail::encodeActorActivitySummaries(writer, execution.activitySummaries);
  std::vector<std::uint8_t> tail = writer.take();
  bytes.insert(bytes.end(), tail.begin(), tail.end());
  return bytes;
}

llvm::Expected<ExecutionTerminal> decodeTerminal(WireReader &reader) {
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
  const evaluation::FindingDescriptor *descriptor =
      evaluation::findFindingDescriptor(evaluation::FindingKind(*kind));
  if (!descriptor || !descriptor->terminalWitnessSchema)
    return detail::invalid("simulation execution: unknown terminal finding");
  return evaluation::requireFindingOccurrenceOwner(*descriptor);
}

llvm::Expected<SpatialSimulationExecution>
decodeExecution(llvm::ArrayRef<std::uint8_t> bytes,
                const evaluation::CaseArtifactResolution &resolution,
                const ArtifactStore &store) {
  auto requestPrefix = decodeArtifactRootReferencePrefix(bytes);
  if (!requestPrefix)
    return requestPrefix.takeError();
  auto context = detail::resolveSpatialExecutionContext(
      requestPrefix->reference, resolution, store);
  if (!context)
    return context.takeError();
  WireReader reader(bytes.drop_front(requestPrefix->byteCount));
  auto terminal = decodeTerminal(reader);
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

llvm::Expected<SpatialExecutionContext> resolveSpatialExecutionContext(
    const ArtifactRootReference &requestReference,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store) {
  auto request =
      evaluation::importEvaluationRequest(requestReference, resolution, store);
  if (!request)
    return request.takeError();
  auto outputSlot = resolveSimulationOutputSlot(*request);
  if (!outputSlot)
    return outputSlot.takeError();
  if (!request->workload() || !request->runtimeInput())
    return invalid("simulation execution: Request workload inputs are not "
                   "total");
  auto inputs = importSpatialSimulationInputs(*request->workload(),
                                              *request->runtimeInput(), store);
  if (!inputs)
    return inputs.takeError();
  auto view = inputs->dataflow.view();
  if (!view)
    return view.takeError();
  auto launch =
      resolveLaunchContext(*view, inputs->workload.spatial()->launchRef);
  if (!launch)
    return launch.takeError();
  return SpatialExecutionContext{
      std::move(*request), std::move(*inputs), std::move(*view),
      std::move(*launch),
      (*outputSlot)
          ->cardinality(evaluation::EvidenceOutcomeKind::CancelledOrTimeout)};
}

} // namespace detail

llvm::Expected<CanonicalSimulationExecution> finalizeSimulationExecution(
    const SpatialSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store) {
  auto context = detail::resolveSpatialExecutionContext(execution.request,
                                                        resolution, store);
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

llvm::Expected<CanonicalSimulationExecution>
importSimulationExecution(const ArtifactRootReference &reference,
                          const evaluation::CaseArtifactResolution &resolution,
                          const ArtifactStore &store) {
  if (reference.schemaIdentity != simulationExecutionSchema.identity ||
      reference.schemaVersion != simulationExecutionSchema.version)
    return detail::invalid("foreign SimulationExecution reference schema");
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto execution = decodeExecution(bytes->bytes(), resolution, store);
  if (!execution)
    return execution.takeError();
  auto context = detail::resolveSpatialExecutionContext(execution->request,
                                                        resolution, store);
  if (!context)
    return context.takeError();
  CanonicalSemanticBytes canonical(
      std::vector<std::uint8_t>(bytes->bytes().begin(), bytes->bytes().end()));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationExecutionSchema, canonical);
  if (identity != reference.artifact)
    return detail::invalid("stale SimulationExecution reference identity");
  return CanonicalSimulationExecution(identity, std::move(*execution),
                                      std::move(canonical));
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
