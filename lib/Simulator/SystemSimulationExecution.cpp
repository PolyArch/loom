//===- SystemSimulationExecution.cpp - Deployment execution artifact -----===//

#include "SimulationExecutionInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"

#include <algorithm>
#include <utility>

namespace loom::sim {

int compareSystemEventCoordinates(const SystemEventCoordinate &lhs,
                                  const SystemEventCoordinate &rhs) {
  if (lhs.gem5Tick != rhs.gem5Tick)
    return lhs.gem5Tick < rhs.gem5Tick ? -1 : 1;
  if (lhs.delta == rhs.delta)
    return 0;
  return lhs.delta < rhs.delta ? -1 : 1;
}

namespace {

bool isRetired(const ExecutionTerminal &terminal) {
  return std::holds_alternative<RetiredExecution>(terminal);
}

bool sameByte(const SemanticMemoryByte &lhs, const SemanticMemoryByte &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.value == rhs.value);
}

const detail::LaneShape &
interfaceShape(const detail::ResolvedSystemContext &context,
               const deployment::DeploymentExternalInterfaceRef &reference) {
  auto position =
      std::lower_bound(context.interfaces.begin(), context.interfaces.end(),
                       reference.externalInterfaceOrdinal,
                       [](const deployment::HostExternalInterface &interface,
                          std::uint64_t ordinal) {
                         return interface.interfaceOrdinal < ordinal;
                       });
  const std::size_t index =
      static_cast<std::size_t>(position - context.interfaces.begin());
  assert(position != context.interfaces.end() &&
         position->interfaceOrdinal == reference.externalInterfaceOrdinal &&
         context.externalInterfaceShapes[index] &&
         "validated System value interface has no lane shape");
  return *context.externalInterfaceShapes[index];
}

const SystemMemoryInterfaceBindingEntry *
findBinding(const SystemSimulationRuntimeInput &input,
            const deployment::DeploymentExternalInterfaceRef &reference) {
  auto position = std::lower_bound(
      input.memoryInterfaceBindings.begin(),
      input.memoryInterfaceBindings.end(), reference,
      [](const SystemMemoryInterfaceBindingEntry &entry,
         const deployment::DeploymentExternalInterfaceRef &target) {
        return deployment::deploymentExternalInterfaceRefLess(
            entry.interfaceRef, target);
      });
  if (position == input.memoryInterfaceBindings.end() ||
      !(position->interfaceRef == reference))
    return nullptr;
  return &*position;
}

struct ImportedMemoryProjection {
  std::uint64_t objectOrdinal = 0;
  std::uint64_t byteOffset = 0;
  std::vector<SemanticMemoryByte> finalBytes;
};

llvm::Expected<ImportedMemoryProjection>
validateMemoryObservation(const SystemMemoryObservable &observable,
                          const MemoryObservationPayload &payload,
                          const detail::SystemExecutionContext &context) {
  const auto &runtime = *context.inputs.runtimeInput.system();
  const SystemMemoryInterfaceBindingEntry *binding =
      findBinding(runtime, observable.interfaceRef);
  if (!binding ||
      binding->binding.objectOrdinal >= runtime.memoryObjects.size())
    return detail::invalid("simulation execution: System memory observable "
                           "has no exact runtime baseline");
  llvm::ArrayRef<SemanticMemoryByte> baseline =
      runtime.memoryObjects[binding->binding.objectOrdinal].initialBytes;
  if (binding->binding.byteOffset > baseline.size())
    return detail::invalid("simulation execution: System memory baseline "
                           "offset is out of range");
  baseline = baseline.drop_front(binding->binding.byteOffset);

  std::vector<SemanticMemoryByte> finalBytes;
  if (observable.form == MemoryObservationForm::FullState) {
    const auto *full = std::get_if<FullMemoryObservation>(&payload);
    if (!full)
      return detail::invalid("simulation execution: System memory payload "
                             "does not match FullState");
    if (llvm::Error error = detail::validateSemanticMemoryBytes(
            full->bytes, "simulation execution: full memory observation"))
      return std::move(error);
    if (full->bytes.size() != baseline.size())
      return detail::invalid("simulation execution: System FullState byte "
                             "count does not match its runtime projection");
    finalBytes = full->bytes;
  } else {
    const auto *diff = std::get_if<DiffMemoryObservation>(&payload);
    if (!diff)
      return detail::invalid("simulation execution: System memory payload "
                             "does not match DiffFromRuntimeInput");
    if (diff->byteCount != baseline.size())
      return detail::invalid("simulation execution: System diff byte count "
                             "does not match its runtime baseline");
    finalBytes.assign(baseline.begin(), baseline.end());
    std::uint64_t previousEnd = 0;
    for (std::size_t index = 0; index < diff->runs.size(); ++index) {
      const MemoryDiffRun &run = diff->runs[index];
      if (run.changedBytes.empty())
        return detail::invalid(
            "simulation execution: System memory diff run is empty");
      if (llvm::Error error = detail::validateSemanticMemoryBytes(
              run.changedBytes, "simulation execution: memory diff run"))
        return std::move(error);
      if (run.byteOffset > diff->byteCount ||
          run.changedBytes.size() > diff->byteCount - run.byteOffset)
        return detail::invalid(
            "simulation execution: System memory diff run is out of range");
      if (index != 0 && run.byteOffset <= previousEnd)
        return detail::invalid("simulation execution: System memory diff runs "
                               "overlap or are adjacent");
      for (std::size_t byte = 0; byte < run.changedBytes.size(); ++byte) {
        const std::size_t offset = run.byteOffset + byte;
        if (sameByte(run.changedBytes[byte], baseline[offset]))
          return detail::invalid("simulation execution: System memory diff "
                                 "contains an unchanged byte");
        finalBytes[offset] = run.changedBytes[byte];
      }
      previousEnd = run.byteOffset + run.changedBytes.size();
    }
  }
  return ImportedMemoryProjection{binding->binding.objectOrdinal,
                                  binding->binding.byteOffset,
                                  std::move(finalBytes)};
}

llvm::Error
validateOverlap(llvm::ArrayRef<ImportedMemoryProjection> projections) {
  for (std::size_t lhsIndex = 0; lhsIndex < projections.size(); ++lhsIndex) {
    const ImportedMemoryProjection &lhs = projections[lhsIndex];
    for (std::size_t rhsIndex = lhsIndex + 1; rhsIndex < projections.size();
         ++rhsIndex) {
      const ImportedMemoryProjection &rhs = projections[rhsIndex];
      if (lhs.objectOrdinal != rhs.objectOrdinal)
        continue;
      const std::uint64_t begin = std::max(lhs.byteOffset, rhs.byteOffset);
      const std::uint64_t end =
          std::min(lhs.byteOffset + lhs.finalBytes.size(),
                   rhs.byteOffset + rhs.finalBytes.size());
      for (std::uint64_t offset = begin; offset < end; ++offset)
        if (!sameByte(lhs.finalBytes[offset - lhs.byteOffset],
                      rhs.finalBytes[offset - rhs.byteOffset]))
          return detail::invalid("simulation execution: overlapping System "
                                 "memory observations disagree");
    }
  }
  return llvm::Error::success();
}

llvm::Error validatePublishedValue(const ValueResultObservation &observation,
                                   const detail::LaneShape &shape,
                                   const ExecutionTerminal &terminal,
                                   llvm::ArrayRef<RuntimeMemoryObject> objects,
                                   mlir::Operation *layoutScope,
                                   const llvm::Twine &what) {
  const auto *published = std::get_if<PublishedValueResult>(&observation);
  if (!published)
    return isRetired(terminal)
               ? detail::invalid(what +
                                 ": Retired value result is not published")
               : llvm::Error::success();
  if (published->value.tokenCount != 1)
    return detail::invalid(what +
                           ": published value does not contain one token");
  if (llvm::Error error = detail::validateValueSequence(published->value, shape,
                                                        what, objects.size()))
    return error;
  return detail::validateCanonicalPointerValueSequence(
      published->value, shape, objects, layoutScope, what);
}

llvm::Error
validateFunctionalObservations(const SystemFunctionalObservations &observations,
                               const ExecutionTerminal &terminal,
                               const detail::SystemExecutionContext &context) {
  const SystemObservableContract &contract =
      context.inputs.workload.system()->observableContract;
  const auto &runtime = *context.inputs.runtimeInput.system();
  if (observations.valueResults.size() != contract.valueResults.size() ||
      observations.externalValueOutputs.size() !=
          contract.externalValueOutputs.size() ||
      observations.externalStreamOutputs.size() !=
          contract.externalStreamOutputs.size() ||
      observations.memories.size() != contract.memories.size())
    return detail::invalid("simulation execution: System functional arrays "
                           "are not total over the workload contract");

  for (std::size_t index = 0; index < observations.valueResults.size(); ++index)
    if (llvm::Error error = validatePublishedValue(
            observations.valueResults[index],
            context.system.valueResultShapes[contract.valueResults[index]],
            terminal, runtime.memoryObjects, context.system.layoutOperation(),
            "simulation execution: System program value result"))
      return error;
  for (std::size_t index = 0; index < observations.externalValueOutputs.size();
       ++index)
    if (llvm::Error error = validatePublishedValue(
            observations.externalValueOutputs[index],
            interfaceShape(context.system,
                           contract.externalValueOutputs[index]),
            terminal, runtime.memoryObjects, context.system.layoutOperation(),
            "simulation execution: System external value output"))
      return error;
  for (std::size_t index = 0; index < observations.externalStreamOutputs.size();
       ++index) {
    const CanonicalStreamSequence &stream =
        observations.externalStreamOutputs[index];
    const detail::LaneShape &shape =
        interfaceShape(context.system, contract.externalStreamOutputs[index]);
    if (llvm::Error error = detail::validateValueSequence(
            stream.values, shape,
            "simulation execution: System external stream output",
            runtime.memoryObjects.size()))
      return error;
    if (llvm::Error error = detail::validateCanonicalPointerValueSequence(
            stream.values, shape, runtime.memoryObjects,
            context.system.layoutOperation(),
            "simulation execution: System external stream output"))
      return error;
    if (static_cast<std::uint32_t>(stream.termination) >
        static_cast<std::uint32_t>(StreamTermination::OpenAfterLast))
      return detail::invalid("simulation execution: System external stream "
                             "termination is out of domain");
    if (isRetired(terminal) &&
        stream.termination != StreamTermination::ClosedAfterLast)
      return detail::invalid(
          "simulation execution: Retired System stream output is open");
  }

  std::vector<ImportedMemoryProjection> projections;
  projections.reserve(observations.memories.size());
  for (std::size_t index = 0; index < observations.memories.size(); ++index) {
    auto projection = validateMemoryObservation(
        contract.memories[index], observations.memories[index], context);
    if (!projection)
      return projection.takeError();
    projections.push_back(std::move(*projection));
  }
  return validateOverlap(projections);
}

llvm::Error validateProgress(const SystemProgressObservations &progress,
                             const ExecutionTerminal &terminal) {
  if (compareSystemEventCoordinates(progress.programEntryAccepted,
                                    progress.terminalObserved) > 0)
    return detail::invalid("simulation execution: program-entry coordinate "
                           "follows the terminal coordinate");
  if (progress.programExitVisible &&
      (compareSystemEventCoordinates(progress.programEntryAccepted,
                                     *progress.programExitVisible) > 0 ||
       compareSystemEventCoordinates(*progress.programExitVisible,
                                     progress.terminalObserved) > 0))
    return detail::invalid("simulation execution: program-exit coordinate is "
                           "out of order");
  if (isRetired(terminal) && !progress.programExitVisible)
    return detail::invalid(
        "simulation execution: Retired execution requires program exit");
  return llvm::Error::success();
}

llvm::Error validateExecution(const SystemSimulationExecution &execution,
                              const detail::SystemExecutionContext &context) {
  if (std::holds_alternative<StoppedByLimitExecution>(execution.terminal) &&
      context.stoppedExecutionCardinality ==
          evaluation::ArtifactCollectionCardinality::Forbidden)
    return detail::invalid("simulation execution: model output slot does not "
                           "retain StoppedByLimit execution");
  if (!execution.activitySummaries.empty())
    return detail::invalid("simulation execution: System activity summary "
                           "has no unique rooted-launch reference-cycle "
                           "source");
  const evaluation::FindingTerminalWitnessContext terminalContext(
      context.request, *context.resolution, *context.artifactStore,
      *context.blobStore);
  if (llvm::Error error = detail::validateExecutionTerminal(execution.terminal,
                                                            terminalContext))
    return error;
  if (llvm::Error error =
          validateProgress(execution.progressObservations, execution.terminal))
    return error;
  return validateFunctionalObservations(execution.functionalObservations,
                                        execution.terminal, context);
}

void encodeValueObservation(detail::WireWriter &writer,
                            const ValueResultObservation &observation,
                            const detail::LaneShape &shape) {
  if (const auto *published = std::get_if<PublishedValueResult>(&observation)) {
    writer.u32(0);
    detail::encodeValueSequence(writer, published->value, shape);
  } else {
    writer.u32(1);
  }
}

llvm::Expected<ValueResultObservation>
decodeValueObservation(detail::WireReader &reader,
                       const detail::LaneShape &shape) {
  auto tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    auto value = detail::decodeValueSequence(reader, shape);
    if (!value)
      return value.takeError();
    return ValueResultObservation{PublishedValueResult{std::move(*value)}};
  }
  if (*tag == 1)
    return ValueResultObservation{NotPublishedValueResult{}};
  return detail::invalid(
      "simulation execution: unknown System value-result state");
}

void encodeMemoryObservation(detail::WireWriter &writer,
                             const MemoryObservationPayload &payload) {
  if (const auto *full = std::get_if<FullMemoryObservation>(&payload)) {
    writer.u64(full->bytes.size());
    detail::encodeSemanticMemoryByteArray(writer, full->bytes);
    return;
  }
  const auto &diff = std::get<DiffMemoryObservation>(payload);
  writer.u64(diff.byteCount);
  writer.u64(diff.runs.size());
  for (const MemoryDiffRun &run : diff.runs) {
    writer.u64(run.byteOffset);
    detail::encodeSemanticMemoryByteArray(writer, run.changedBytes);
  }
}

llvm::Expected<MemoryObservationPayload>
decodeMemoryObservation(detail::WireReader &reader,
                        MemoryObservationForm form) {
  auto byteCount = reader.u64();
  if (!byteCount)
    return byteCount.takeError();
  if (form == MemoryObservationForm::FullState) {
    auto bytes = detail::decodeSemanticMemoryByteArray(reader);
    if (!bytes)
      return bytes.takeError();
    if (bytes->size() != *byteCount)
      return detail::invalid("simulation execution: System FullState byte "
                             "count does not match its byte array");
    return MemoryObservationPayload{FullMemoryObservation{std::move(*bytes)}};
  }
  auto runCount = reader.u64();
  if (!runCount)
    return runCount.takeError();
  if (llvm::Error error = reader.guardCount(*runCount, 16))
    return std::move(error);
  DiffMemoryObservation diff;
  diff.byteCount = *byteCount;
  diff.runs.reserve(*runCount);
  for (std::uint64_t index = 0; index < *runCount; ++index) {
    auto offset = reader.u64();
    if (!offset)
      return offset.takeError();
    auto changed = detail::decodeSemanticMemoryByteArray(reader);
    if (!changed)
      return changed.takeError();
    diff.runs.push_back({*offset, std::move(*changed)});
  }
  return MemoryObservationPayload{std::move(diff)};
}

void encodeFunctional(detail::WireWriter &writer,
                      const SystemFunctionalObservations &observations,
                      const detail::SystemExecutionContext &context) {
  const SystemObservableContract &contract =
      context.inputs.workload.system()->observableContract;
  writer.u64(observations.valueResults.size());
  for (std::size_t index = 0; index < observations.valueResults.size(); ++index)
    encodeValueObservation(
        writer, observations.valueResults[index],
        context.system.valueResultShapes[contract.valueResults[index]]);
  writer.u64(observations.externalValueOutputs.size());
  for (std::size_t index = 0; index < observations.externalValueOutputs.size();
       ++index)
    encodeValueObservation(
        writer, observations.externalValueOutputs[index],
        interfaceShape(context.system, contract.externalValueOutputs[index]));
  writer.u64(observations.externalStreamOutputs.size());
  for (std::size_t index = 0; index < observations.externalStreamOutputs.size();
       ++index)
    detail::encodeStreamSequence(
        writer, observations.externalStreamOutputs[index],
        interfaceShape(context.system, contract.externalStreamOutputs[index]));
  writer.u64(observations.memories.size());
  for (const MemoryObservationPayload &memory : observations.memories)
    encodeMemoryObservation(writer, memory);
}

llvm::Expected<SystemFunctionalObservations>
decodeFunctional(detail::WireReader &reader,
                 const detail::SystemExecutionContext &context) {
  const SystemObservableContract &contract =
      context.inputs.workload.system()->observableContract;
  SystemFunctionalObservations observations;
  auto valueCount = reader.u64();
  if (!valueCount)
    return valueCount.takeError();
  if (*valueCount != contract.valueResults.size())
    return detail::invalid(
        "simulation execution: System value-result array is not total");
  observations.valueResults.reserve(*valueCount);
  for (std::size_t index = 0; index < *valueCount; ++index) {
    auto value = decodeValueObservation(
        reader, context.system.valueResultShapes[contract.valueResults[index]]);
    if (!value)
      return value.takeError();
    observations.valueResults.push_back(std::move(*value));
  }

  auto externalValueCount = reader.u64();
  if (!externalValueCount)
    return externalValueCount.takeError();
  if (*externalValueCount != contract.externalValueOutputs.size())
    return detail::invalid("simulation execution: external value-output "
                           "array is not total");
  observations.externalValueOutputs.reserve(*externalValueCount);
  for (std::size_t index = 0; index < *externalValueCount; ++index) {
    auto value = decodeValueObservation(
        reader,
        interfaceShape(context.system, contract.externalValueOutputs[index]));
    if (!value)
      return value.takeError();
    observations.externalValueOutputs.push_back(std::move(*value));
  }

  auto streamCount = reader.u64();
  if (!streamCount)
    return streamCount.takeError();
  if (*streamCount != contract.externalStreamOutputs.size())
    return detail::invalid("simulation execution: external stream-output "
                           "array is not total");
  observations.externalStreamOutputs.reserve(*streamCount);
  for (std::size_t index = 0; index < *streamCount; ++index) {
    auto stream = detail::decodeStreamSequence(
        reader,
        interfaceShape(context.system, contract.externalStreamOutputs[index]));
    if (!stream)
      return stream.takeError();
    observations.externalStreamOutputs.push_back(std::move(*stream));
  }

  auto memoryCount = reader.u64();
  if (!memoryCount)
    return memoryCount.takeError();
  if (*memoryCount != contract.memories.size())
    return detail::invalid(
        "simulation execution: System memory array is not total");
  observations.memories.reserve(*memoryCount);
  for (std::size_t index = 0; index < *memoryCount; ++index) {
    auto memory =
        decodeMemoryObservation(reader, contract.memories[index].form);
    if (!memory)
      return memory.takeError();
    observations.memories.push_back(std::move(*memory));
  }
  return observations;
}

void encodeCoordinate(detail::WireWriter &writer,
                      const SystemEventCoordinate &coordinate) {
  writer.u64(coordinate.gem5Tick);
  writer.u64(coordinate.delta);
}

llvm::Expected<SystemEventCoordinate>
decodeCoordinate(detail::WireReader &reader) {
  auto tick = reader.u64();
  if (!tick)
    return tick.takeError();
  auto delta = reader.u64();
  if (!delta)
    return delta.takeError();
  return SystemEventCoordinate{*tick, *delta};
}

void encodeProgress(detail::WireWriter &writer,
                    const SystemProgressObservations &progress) {
  encodeCoordinate(writer, progress.programEntryAccepted);
  if (progress.programExitVisible) {
    writer.u32(1);
    encodeCoordinate(writer, *progress.programExitVisible);
  } else {
    writer.u32(0);
  }
  encodeCoordinate(writer, progress.terminalObserved);
}

llvm::Expected<SystemProgressObservations>
decodeProgress(detail::WireReader &reader) {
  auto entry = decodeCoordinate(reader);
  if (!entry)
    return entry.takeError();
  auto exitTag = reader.u32();
  if (!exitTag)
    return exitTag.takeError();
  std::optional<SystemEventCoordinate> exit;
  if (*exitTag == 1) {
    auto coordinate = decodeCoordinate(reader);
    if (!coordinate)
      return coordinate.takeError();
    exit = std::move(*coordinate);
  } else if (*exitTag != 0) {
    return detail::invalid(
        "simulation execution: unknown program-exit anchor state");
  }
  auto terminal = decodeCoordinate(reader);
  if (!terminal)
    return terminal.takeError();
  return SystemProgressObservations{std::move(*entry), std::move(exit),
                                    std::move(*terminal)};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeExecution(const SystemSimulationExecution &execution,
                const detail::SystemExecutionContext &context) {
  std::vector<std::uint8_t> bytes =
      encodeArtifactRootReference(execution.request);
  detail::WireWriter writer;
  const evaluation::FindingTerminalWitnessContext terminalContext(
      context.request, *context.resolution, *context.artifactStore,
      *context.blobStore);
  if (llvm::Error error = detail::encodeExecutionTerminal(
          writer, execution.terminal, terminalContext))
    return std::move(error);
  encodeFunctional(writer, execution.functionalObservations, context);
  encodeProgress(writer, execution.progressObservations);
  writer.u64(0);
  std::vector<std::uint8_t> tail = writer.take();
  bytes.insert(bytes.end(), tail.begin(), tail.end());
  return bytes;
}

llvm::Expected<SystemSimulationExecution>
decodeExecution(llvm::ArrayRef<std::uint8_t> bytes,
                const evaluation::CaseArtifactResolution &resolution,
                const ArtifactStore &store, const BlobStore &blobs) {
  auto requestPrefix = decodeArtifactRootReferencePrefix(bytes);
  if (!requestPrefix)
    return requestPrefix.takeError();
  auto context = detail::resolveSystemExecutionContext(
      requestPrefix->reference, resolution, store, blobs);
  if (!context)
    return context.takeError();
  detail::WireReader reader(bytes.drop_front(requestPrefix->byteCount));
  const evaluation::FindingTerminalWitnessContext terminalContext(
      context->request, *context->resolution, *context->artifactStore,
      *context->blobStore);
  auto terminal = detail::decodeExecutionTerminal(reader, terminalContext);
  if (!terminal)
    return terminal.takeError();
  auto functional = decodeFunctional(reader, *context);
  if (!functional)
    return functional.takeError();
  auto progress = decodeProgress(reader);
  if (!progress)
    return progress.takeError();
  auto activityCount = reader.u64();
  if (!activityCount)
    return activityCount.takeError();
  if (*activityCount != 0)
    return detail::invalid("simulation execution: System activity summary "
                           "has no exact source attachment");
  if (!reader.atEnd())
    return detail::invalid("simulation execution: trailing bytes");
  SystemSimulationExecution execution{requestPrefix->reference,
                                      std::move(*terminal),
                                      std::move(*functional),
                                      std::move(*progress),
                                      {}};
  if (llvm::Error error = validateExecution(execution, *context))
    return std::move(error);
  auto canonical = encodeExecution(execution, *context);
  if (!canonical)
    return canonical.takeError();
  if (!llvm::ArrayRef<std::uint8_t>(*canonical).equals(bytes))
    return detail::invalid(
        "simulation execution: System bytes are not canonical");
  return execution;
}

llvm::Expected<SimulationWorkloadKind>
executionWorkloadKind(const ArtifactRootReference &requestReference,
                      const evaluation::CaseArtifactResolution &resolution,
                      const ArtifactStore &store, const BlobStore &blobs) {
  auto request = evaluation::importEvaluationRequest(requestReference,
                                                     resolution, store, blobs);
  if (!request)
    return request.takeError();
  if (!request->workload())
    return detail::invalid(
        "simulation execution: Request has no workload reference");
  auto bytes = store.get(*request->workload());
  if (!bytes)
    return bytes.takeError();
  detail::WireReader reader(bytes->bytes());
  auto root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root >
      static_cast<std::uint32_t>(SimulationWorkloadKind::StructuredProgram))
    return detail::invalid(
        "simulation execution: unknown workload root discriminant");
  return static_cast<SimulationWorkloadKind>(*root);
}

} // namespace

namespace detail {

llvm::Expected<SystemExecutionContext> resolveSystemExecutionContext(
    const ArtifactRootReference &requestReference,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto request = evaluation::importEvaluationRequest(requestReference,
                                                     resolution, store, blobs);
  if (!request)
    return request.takeError();
  auto stoppedCardinality = resolveSimulationOutputCardinality(*request);
  if (!stoppedCardinality)
    return stoppedCardinality.takeError();
  if (!request->workload() || !request->runtimeInput())
    return invalid("simulation execution: Request workload inputs are not "
                   "total");
  auto inputs = importSystemSimulationInputs(
      *request->workload(), *request->runtimeInput(), store, blobs);
  if (!inputs)
    return inputs.takeError();
  auto system = resolveSystemContext(
      inputs->deployment, inputs->workload.system()->programEntryRef, store);
  if (!system)
    return system.takeError();
  return SystemExecutionContext{std::move(*request),
                                std::move(*inputs),
                                std::move(*system),
                                *stoppedCardinality,
                                &resolution,
                                &store,
                                &blobs};
}

} // namespace detail

llvm::Expected<CanonicalSimulationExecution> finalizeSimulationExecution(
    const SystemSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto context = detail::resolveSystemExecutionContext(
      execution.request, resolution, store, blobs);
  if (!context)
    return context.takeError();
  if (llvm::Error error = validateExecution(execution, *context))
    return std::move(error);
  auto encoded = encodeExecution(execution, *context);
  if (!encoded)
    return encoded.takeError();
  CanonicalSemanticBytes bytes(std::move(*encoded));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationExecutionSchema, bytes);
  return CanonicalSimulationExecution(identity, execution, std::move(bytes));
}

llvm::Expected<CanonicalSimulationExecution>
importSimulationExecution(const ArtifactRootReference &reference,
                          const evaluation::CaseArtifactResolution &resolution,
                          const ArtifactStore &store, const BlobStore &blobs) {
  if (reference.schemaIdentity != simulationExecutionSchema.identity ||
      reference.schemaVersion != simulationExecutionSchema.version)
    return detail::invalid("foreign SimulationExecution reference schema");
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto requestPrefix = decodeArtifactRootReferencePrefix(bytes->bytes());
  if (!requestPrefix)
    return requestPrefix.takeError();
  auto kind =
      executionWorkloadKind(requestPrefix->reference, resolution, store, blobs);
  if (!kind)
    return kind.takeError();
  if (*kind == SimulationWorkloadKind::Spatial) {
    auto execution = detail::decodeSpatialSimulationExecution(
        bytes->bytes(), resolution, store, blobs);
    if (!execution)
      return execution.takeError();
    CanonicalSemanticBytes canonical(std::vector<std::uint8_t>(
        bytes->bytes().begin(), bytes->bytes().end()));
    ArtifactIdentity identity =
        finalizeArtifactIdentity(simulationExecutionSchema, canonical);
    if (identity != reference.artifact)
      return detail::invalid("stale SimulationExecution reference identity");
    return CanonicalSimulationExecution(identity, std::move(*execution),
                                        std::move(canonical));
  }
  if (*kind != SimulationWorkloadKind::System)
    return detail::invalid("simulation execution: workload root does not have "
                           "an execution observation form");
  auto execution = decodeExecution(bytes->bytes(), resolution, store, blobs);
  if (!execution)
    return execution.takeError();
  CanonicalSemanticBytes canonical(
      std::vector<std::uint8_t>(bytes->bytes().begin(), bytes->bytes().end()));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationExecutionSchema, canonical);
  if (identity != reference.artifact)
    return detail::invalid("stale SimulationExecution reference identity");
  return CanonicalSimulationExecution(identity, std::move(*execution),
                                      std::move(canonical));
}

} // namespace loom::sim
