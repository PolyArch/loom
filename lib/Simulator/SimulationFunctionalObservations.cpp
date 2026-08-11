#include "SimulationExecutionInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim::detail {
namespace {

bool isRetired(const ExecutionTerminal &terminal) {
  return std::holds_alternative<RetiredExecution>(terminal);
}

bool sameByte(const SemanticMemoryByte &lhs, const SemanticMemoryByte &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.value == rhs.value);
}

llvm::Expected<dataflow::LogicalMemoryRootOrViewRef>
resolveObservableRole(const SpatialMemoryObservableTarget &target,
                      const SpatialSimulationWorkload &workload,
                      const dataflow::CanonicalDataflowProgramView &program) {
  if (const auto *direct =
          std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&target))
    return *direct;
  return program.resolveExposure(dataflow::MemoryExposureRef{
      workload.launchRef,
      std::get<MemoryExposureTarget>(target).memoryResultOrdinal});
}

dataflow::LogicalMemoryRootRef
rootOf(const dataflow::LogicalMemoryRootOrViewRef &role) {
  if (const auto *root = std::get_if<dataflow::LogicalMemoryRootRef>(&role))
    return *root;
  return std::get<dataflow::LogicalMemoryViewRef>(role).root;
}

const MemoryRootBindingEntry *
findBinding(const SpatialSimulationRuntimeInput &input,
            dataflow::LogicalMemoryRootRef root) {
  for (const MemoryRootBindingEntry &entry : input.memoryRootBindings)
    if (compareRootKeys(entry.root, root) == 0)
      return &entry;
  return nullptr;
}

struct ImportedMemoryProjection {
  std::uint64_t objectOrdinal;
  std::uint64_t byteOffset;
  std::vector<SemanticMemoryByte> finalBytes;
};

llvm::Expected<std::optional<ImportedMemoryProjection>>
validateMemoryObservation(const SpatialMemoryObservable &observable,
                          const MemoryObservationPayload &payload,
                          const SpatialExecutionContext &context) {
  const auto &workload = *context.inputs->workload.spatial();
  const auto &runtime = *context.inputs->runtimeInput.spatial();
  llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> role =
      resolveObservableRole(observable.target, workload, context.dataflowView);
  if (!role)
    return role.takeError();
  const MemoryRootBindingEntry *binding = findBinding(runtime, rootOf(*role));
  llvm::ArrayRef<SemanticMemoryByte> baseline;
  if (binding) {
    if (binding->binding.objectOrdinal >= runtime.memoryObjects.size())
      return invalid("simulation execution: memory binding object is out of "
                     "range");
    baseline =
        runtime.memoryObjects[binding->binding.objectOrdinal].initialBytes;
    if (binding->binding.byteOffset > baseline.size())
      return invalid("simulation execution: memory binding offset is out of "
                     "range");
    baseline = baseline.drop_front(binding->binding.byteOffset);
  }

  std::vector<SemanticMemoryByte> finalBytes;
  if (observable.form == MemoryObservationForm::FullState) {
    const auto *full = std::get_if<FullMemoryObservation>(&payload);
    if (!full)
      return invalid("simulation execution: memory observation payload does "
                     "not match FullState");
    if (llvm::Error error = validateSemanticMemoryBytes(
            full->bytes, "simulation execution: full memory observation"))
      return std::move(error);
    if (!binding)
      return invalid("simulation execution: fresh-memory FullState extent "
                     "owner is unavailable");
    if (full->bytes.size() != baseline.size())
      return invalid("simulation execution: FullState byte count does not "
                     "match the exact runtime projection");
    finalBytes = full->bytes;
  } else {
    const auto *diff = std::get_if<DiffMemoryObservation>(&payload);
    if (!diff)
      return invalid("simulation execution: memory observation payload does "
                     "not match DiffFromRuntimeInput");
    if (!binding)
      return invalid("simulation execution: diff observation has no exact "
                     "runtime baseline");
    if (diff->byteCount != baseline.size())
      return invalid("simulation execution: diff byte count does not match "
                     "the exact runtime baseline");
    finalBytes.assign(baseline.begin(), baseline.end());
    std::uint64_t previousEnd = 0;
    for (std::size_t index = 0; index < diff->runs.size(); ++index) {
      const MemoryDiffRun &run = diff->runs[index];
      if (run.changedBytes.empty())
        return invalid("simulation execution: memory diff run is empty");
      if (llvm::Error error = validateSemanticMemoryBytes(
              run.changedBytes, "simulation execution: memory diff run"))
        return std::move(error);
      if (run.byteOffset > diff->byteCount ||
          run.changedBytes.size() > diff->byteCount - run.byteOffset)
        return invalid("simulation execution: memory diff run is out of "
                       "range");
      if (index != 0 && run.byteOffset <= previousEnd)
        return invalid("simulation execution: memory diff runs overlap or "
                       "are adjacent");
      for (std::size_t byte = 0; byte < run.changedBytes.size(); ++byte) {
        const std::size_t offset = run.byteOffset + byte;
        if (sameByte(run.changedBytes[byte], baseline[offset]))
          return invalid("simulation execution: memory diff contains an "
                         "unchanged byte");
        finalBytes[offset] = run.changedBytes[byte];
      }
      previousEnd = run.byteOffset + run.changedBytes.size();
    }
  }

  return std::optional<ImportedMemoryProjection>(ImportedMemoryProjection{
      binding->binding.objectOrdinal, binding->binding.byteOffset,
      std::move(finalBytes)});
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
      const std::uint64_t lhsEnd = lhs.byteOffset + lhs.finalBytes.size();
      const std::uint64_t rhsEnd = rhs.byteOffset + rhs.finalBytes.size();
      const std::uint64_t end = std::min(lhsEnd, rhsEnd);
      for (std::uint64_t offset = begin; offset < end; ++offset)
        if (!sameByte(lhs.finalBytes[offset - lhs.byteOffset],
                      rhs.finalBytes[offset - rhs.byteOffset]))
          return invalid("simulation execution: overlapping memory "
                         "observations disagree");
    }
  }
  return llvm::Error::success();
}

void encodeMemoryObservation(WireWriter &writer,
                             const MemoryObservationPayload &payload) {
  if (const auto *full = std::get_if<FullMemoryObservation>(&payload)) {
    writer.u64(full->bytes.size());
    encodeSemanticMemoryByteArray(writer, full->bytes);
    return;
  }
  const auto &diff = std::get<DiffMemoryObservation>(payload);
  writer.u64(diff.byteCount);
  writer.u64(diff.runs.size());
  for (const MemoryDiffRun &run : diff.runs) {
    writer.u64(run.byteOffset);
    encodeSemanticMemoryByteArray(writer, run.changedBytes);
  }
}

llvm::Expected<MemoryObservationPayload>
decodeMemoryObservation(WireReader &reader, MemoryObservationForm form) {
  llvm::Expected<std::uint64_t> byteCount = reader.u64();
  if (!byteCount)
    return byteCount.takeError();
  if (form == MemoryObservationForm::FullState) {
    llvm::Expected<std::vector<SemanticMemoryByte>> bytes =
        decodeSemanticMemoryByteArray(reader);
    if (!bytes)
      return bytes.takeError();
    if (bytes->size() != *byteCount)
      return invalid("simulation execution: FullState byte count does not "
                     "match its byte array");
    return MemoryObservationPayload{FullMemoryObservation{std::move(*bytes)}};
  }

  llvm::Expected<std::uint64_t> runCount = reader.u64();
  if (!runCount)
    return runCount.takeError();
  if (llvm::Error error = reader.guardCount(*runCount, 16))
    return std::move(error);
  DiffMemoryObservation diff;
  diff.byteCount = *byteCount;
  diff.runs.reserve(*runCount);
  for (std::uint64_t index = 0; index < *runCount; ++index) {
    llvm::Expected<std::uint64_t> offset = reader.u64();
    if (!offset)
      return offset.takeError();
    llvm::Expected<std::vector<SemanticMemoryByte>> changed =
        decodeSemanticMemoryByteArray(reader);
    if (!changed)
      return changed.takeError();
    diff.runs.push_back(MemoryDiffRun{*offset, std::move(*changed)});
  }
  return MemoryObservationPayload{std::move(diff)};
}

int compareActorRefs(const dataflow::ActorRef &lhs,
                     const dataflow::ActorRef &rhs) {
  const int identity = compareIdentities(lhs.artifact, rhs.artifact);
  if (identity != 0)
    return identity;
  if (lhs.entity.value() < rhs.entity.value())
    return -1;
  if (lhs.entity.value() > rhs.entity.value())
    return 1;
  return 0;
}

std::vector<dataflow::ActorRef>
graphActors(const SpatialExecutionContext &context) {
  std::vector<dataflow::ActorRef> actors;
  for (const dataflow::CanonicalActorView &actor :
       context.dataflowView.actors())
    if (actor.graph == context.launch.graph)
      actors.push_back(actor.ref);
  return actors;
}

} // namespace

llvm::Error validateSpatialFunctionalObservations(
    const SpatialFunctionalObservations &observations,
    const ExecutionTerminal &terminal, const SpatialExecutionContext &context) {
  const SpatialSimulationWorkload &workload =
      *context.inputs->workload.spatial();
  const SpatialSimulationRuntimeInput &runtime =
      *context.inputs->runtimeInput.spatial();
  const SpatialObservableContract &contract = workload.observableContract;
  if (observations.valueResults.size() != contract.valueResults.size() ||
      observations.streamOutputs.size() != contract.streamOutputs.size() ||
      observations.memories.size() != contract.memories.size())
    return invalid("simulation execution: functional observations are not "
                   "total over the workload contract");

  for (std::size_t index = 0; index < observations.valueResults.size();
       ++index) {
    const ValueResultObservation &observation =
        observations.valueResults[index];
    if (const auto *published =
            std::get_if<PublishedValueResult>(&observation)) {
      if (published->value.tokenCount != 1)
        return invalid("simulation execution: published value result does not "
                       "contain exactly one token");
      if (llvm::Error error = validateValueSequence(
              published->value,
              context.launch.valueResultShapes[contract.valueResults[index]],
              "simulation execution: value result",
              runtime.memoryObjects.size()))
        return error;
    } else if (isRetired(terminal)) {
      return invalid(
          "simulation execution: Retired value result is not published");
    }
  }

  for (std::size_t index = 0; index < observations.streamOutputs.size();
       ++index) {
    const CanonicalStreamSequence &stream = observations.streamOutputs[index];
    if (llvm::Error error = validateValueSequence(
            stream.values,
            context.launch.streamOutputShapes[contract.streamOutputs[index]],
            "simulation execution: stream output",
            runtime.memoryObjects.size()))
      return error;
    if (isRetired(terminal) &&
        stream.termination != StreamTermination::ClosedAfterLast)
      return invalid("simulation execution: Retired stream output is open");
  }

  std::vector<ImportedMemoryProjection> projections;
  projections.reserve(observations.memories.size());
  for (std::size_t index = 0; index < observations.memories.size(); ++index) {
    auto projection = validateMemoryObservation(
        contract.memories[index], observations.memories[index], context);
    if (!projection)
      return projection.takeError();
    if (*projection)
      projections.push_back(std::move(**projection));
  }
  return validateOverlap(projections);
}

void encodeSpatialFunctionalObservations(
    WireWriter &writer, const SpatialFunctionalObservations &observations,
    const SpatialExecutionContext &context) {
  const SpatialObservableContract &contract =
      context.inputs->workload.spatial()->observableContract;
  writer.u64(observations.valueResults.size());
  for (std::size_t index = 0; index < observations.valueResults.size();
       ++index) {
    const ValueResultObservation &observation =
        observations.valueResults[index];
    if (const auto *published =
            std::get_if<PublishedValueResult>(&observation)) {
      writer.u32(0);
      encodeValueSequence(
          writer, published->value,
          context.launch.valueResultShapes[contract.valueResults[index]]);
    } else {
      writer.u32(1);
    }
  }
  writer.u64(observations.streamOutputs.size());
  for (std::size_t index = 0; index < observations.streamOutputs.size();
       ++index)
    encodeStreamSequence(
        writer, observations.streamOutputs[index],
        context.launch.streamOutputShapes[contract.streamOutputs[index]]);
  writer.u64(observations.memories.size());
  for (const MemoryObservationPayload &memory : observations.memories)
    encodeMemoryObservation(writer, memory);
}

llvm::Expected<SpatialFunctionalObservations>
decodeSpatialFunctionalObservations(WireReader &reader,
                                    const SpatialExecutionContext &context) {
  const SpatialObservableContract &contract =
      context.inputs->workload.spatial()->observableContract;
  SpatialFunctionalObservations observations;
  llvm::Expected<std::uint64_t> valueCount = reader.u64();
  if (!valueCount)
    return valueCount.takeError();
  if (*valueCount != contract.valueResults.size())
    return invalid("simulation execution: value-result array is not total");
  observations.valueResults.reserve(*valueCount);
  for (std::uint64_t index = 0; index < *valueCount; ++index) {
    llvm::Expected<std::uint32_t> tag = reader.u32();
    if (!tag)
      return tag.takeError();
    if (*tag == 0) {
      auto sequence = decodeValueSequence(
          reader,
          context.launch.valueResultShapes[contract.valueResults[index]]);
      if (!sequence)
        return sequence.takeError();
      observations.valueResults.emplace_back(
          PublishedValueResult{std::move(*sequence)});
    } else if (*tag == 1) {
      observations.valueResults.emplace_back(NotPublishedValueResult{});
    } else {
      return invalid("simulation execution: unknown value-result state");
    }
  }

  llvm::Expected<std::uint64_t> streamCount = reader.u64();
  if (!streamCount)
    return streamCount.takeError();
  if (*streamCount != contract.streamOutputs.size())
    return invalid("simulation execution: stream-output array is not total");
  observations.streamOutputs.reserve(*streamCount);
  for (std::uint64_t index = 0; index < *streamCount; ++index) {
    auto stream = decodeStreamSequence(
        reader,
        context.launch.streamOutputShapes[contract.streamOutputs[index]]);
    if (!stream)
      return stream.takeError();
    observations.streamOutputs.push_back(std::move(*stream));
  }

  llvm::Expected<std::uint64_t> memoryCount = reader.u64();
  if (!memoryCount)
    return memoryCount.takeError();
  if (*memoryCount != contract.memories.size())
    return invalid("simulation execution: memory array is not total");
  observations.memories.reserve(*memoryCount);
  for (std::uint64_t index = 0; index < *memoryCount; ++index) {
    auto memory =
        decodeMemoryObservation(reader, contract.memories[index].form);
    if (!memory)
      return memory.takeError();
    observations.memories.push_back(std::move(*memory));
  }
  return observations;
}

llvm::Error validateActorActivitySummaries(
    llvm::ArrayRef<ActorTransitionsActivitySummary> summaries,
    const ExecutionTerminal &terminal,
    const SpatialProgressObservations &progress,
    const SpatialExecutionContext &context) {
  const std::vector<dataflow::ActorRef> completeActors = graphActors(context);
  for (std::size_t summaryIndex = 0; summaryIndex < summaries.size();
       ++summaryIndex) {
    const ActorTransitionsActivitySummary &summary = summaries[summaryIndex];
    if (static_cast<std::uint32_t>(summary.window) >
            static_cast<std::uint32_t>(ActivityWindow::LaunchToTerminal) ||
        static_cast<std::uint32_t>(summary.coverage) >
            static_cast<std::uint32_t>(ActivityCoverage::Partial))
      return invalid("simulation execution: activity summary enum is out of "
                     "domain");
    if (summaryIndex != 0 &&
        static_cast<std::uint32_t>(summary.window) <=
            static_cast<std::uint32_t>(summaries[summaryIndex - 1].window))
      return invalid("simulation execution: activity summaries are not "
                     "canonical or contain a duplicate");
    if (summary.window == ActivityWindow::LaunchToGraphRetirement &&
        !progress.graphRetirementVisible)
      return invalid("simulation execution: graph-retirement activity window "
                     "has no retirement anchor");
    if (summary.transitions.empty())
      return invalid("simulation execution: empty activity summary must be "
                     "omitted");
    for (std::size_t index = 0; index < summary.transitions.size(); ++index) {
      const ActorTransitionEntry &entry = summary.transitions[index];
      if (index != 0 &&
          compareActorRefs(entry.actor, summary.transitions[index - 1].actor) <=
              0)
        return invalid("simulation execution: actor activity table is not "
                       "canonical or contains a duplicate");
      llvm::Expected<dataflow::CanonicalActorView> actor =
          context.dataflowView.resolve(entry.actor);
      if (!actor)
        return actor.takeError();
      if (actor->graph != context.launch.graph)
        return invalid("simulation execution: activity actor is outside the "
                       "rooted launch graph");
      if (entry.counts.committedFirings < entry.counts.retiredFirings)
        return invalid("simulation execution: actor retired count exceeds "
                       "committed count");
      if (isRetired(terminal) &&
          summary.window == ActivityWindow::LaunchToTerminal &&
          entry.counts.committedFirings != entry.counts.retiredFirings)
        return invalid("simulation execution: Retired terminal activity has "
                       "unretired actor firings");
    }
    if (summary.coverage == ActivityCoverage::Complete) {
      if (summary.transitions.size() != completeActors.size())
        return invalid("simulation execution: complete actor activity table "
                       "is not total");
      for (std::size_t index = 0; index < completeActors.size(); ++index)
        if (compareActorRefs(summary.transitions[index].actor,
                             completeActors[index]) != 0)
          return invalid("simulation execution: complete actor activity table "
                         "does not match the rooted graph inventory");
    }
  }
  return llvm::Error::success();
}

void encodeActorActivitySummaries(
    WireWriter &writer,
    llvm::ArrayRef<ActorTransitionsActivitySummary> summaries) {
  writer.u64(summaries.size());
  for (const ActorTransitionsActivitySummary &summary : summaries) {
    writer.u32(static_cast<std::uint32_t>(summary.window));
    writer.u32(static_cast<std::uint32_t>(summary.coverage));
    writer.u32(0); // ActorTransitions
    writer.u64(summary.transitions.size());
    for (const ActorTransitionEntry &entry : summary.transitions) {
      writer.identity(entry.actor.artifact);
      writer.u64(entry.actor.entity.value());
      writer.u64(entry.counts.committedFirings);
      writer.u64(entry.counts.retiredFirings);
    }
  }
}

llvm::Expected<std::vector<ActorTransitionsActivitySummary>>
decodeActorActivitySummaries(WireReader &reader) {
  llvm::Expected<std::uint64_t> count = reader.u64();
  if (!count)
    return count.takeError();
  if (llvm::Error error = reader.guardCount(*count, 20))
    return std::move(error);
  std::vector<ActorTransitionsActivitySummary> summaries;
  summaries.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    llvm::Expected<std::uint32_t> window = reader.u32();
    if (!window)
      return window.takeError();
    llvm::Expected<std::uint32_t> coverage = reader.u32();
    if (!coverage)
      return coverage.takeError();
    llvm::Expected<std::uint32_t> payload = reader.u32();
    if (!payload)
      return payload.takeError();
    if (*payload != 0)
      return llvm::createStringError(
          std::make_error_code(std::errc::not_supported),
          "simulation execution activity owner is unavailable");
    llvm::Expected<std::uint64_t> transitionCount = reader.u64();
    if (!transitionCount)
      return transitionCount.takeError();
    if (llvm::Error error = reader.guardCount(*transitionCount, 56))
      return std::move(error);
    ActorTransitionsActivitySummary summary{
        static_cast<ActivityWindow>(*window),
        static_cast<ActivityCoverage>(*coverage),
        {}};
    summary.transitions.reserve(*transitionCount);
    for (std::uint64_t entry = 0; entry < *transitionCount; ++entry) {
      llvm::Expected<ArtifactIdentity> artifact = reader.identity();
      if (!artifact)
        return artifact.takeError();
      llvm::Expected<std::uint64_t> entity = reader.u64();
      if (!entity)
        return entity.takeError();
      llvm::Expected<std::uint64_t> committed = reader.u64();
      if (!committed)
        return committed.takeError();
      llvm::Expected<std::uint64_t> retired = reader.u64();
      if (!retired)
        return retired.takeError();
      summary.transitions.push_back(ActorTransitionEntry{
          dataflow::ActorRef{*artifact, dataflow::ActorId(*entity)},
          {*committed, *retired}});
    }
    summaries.push_back(std::move(summary));
  }
  return summaries;
}

} // namespace loom::sim::detail
