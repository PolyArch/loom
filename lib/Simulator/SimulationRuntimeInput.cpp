//===- SimulationRuntimeInput.cpp - spatial runtime input artifact -------===//
//
// Schema-1.0 Spatial SimulationRuntimeInput: total runtime value/stream
// tables, neutral memory objects, total logical-root bindings with canonical
// object ordinals derived from sorted binding keys, the one strict canonical
// encoder/decoder, and failure-atomic finalization/import.
//
//===----------------------------------------------------------------------===//

#include "SimulationWireInternal.h"

#include "Common/ArtifactFinalizer.h"

#include <algorithm>
#include <utility>

namespace loom::sim {
namespace {

using detail::ResolvedLaunchContext;

//===----------------------------------------------------------------------===//
// Canonical object ordinals
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>>
deriveCanonicalOrdinals(llvm::ArrayRef<MemoryRootBindingEntry> bindings) {
  std::vector<detail::RuntimeObjectBindingKey> keys;
  keys.reserve(bindings.size());
  for (const MemoryRootBindingEntry &entry : bindings) {
    detail::WireWriter writer;
    writer.identity(entry.root.artifact);
    writer.u64(entry.root.entity.value());
    writer.u64(entry.binding.byteOffset);
    keys.push_back(detail::RuntimeObjectBindingKey{entry.binding.objectOrdinal,
                                                   writer.take()});
  }
  return detail::deriveCanonicalObjectOrdinals(keys);
}

//===----------------------------------------------------------------------===//
// Shared table validation
//===----------------------------------------------------------------------===//

llvm::Error
validateRuntimeValues(llvm::ArrayRef<RuntimeValueEntry> runtimeValues,
                      const SpatialSimulationWorkload &workload,
                      const ResolvedLaunchContext &context,
                      std::uint64_t objectCount) {
  std::uint64_t runtimeClassified = 0;
  for (const SpatialValueInputSource &source : workload.valueInputPlan)
    if (std::holds_alternative<RuntimeValueInput>(source))
      ++runtimeClassified;
  if (runtimeValues.size() != runtimeClassified)
    return detail::invalid("simulation runtime input: runtime values do not "
                           "exactly complement the Runtime classifications");
  for (std::size_t index = 0; index < runtimeValues.size(); ++index) {
    const RuntimeValueEntry &entry = runtimeValues[index];
    if (index > 0 &&
        entry.valueInputOrdinal <= runtimeValues[index - 1].valueInputOrdinal)
      return detail::invalid("simulation runtime input: runtime values are "
                             "not sorted or contain a duplicate");
    if (entry.valueInputOrdinal >= context.numValueInputs)
      return detail::invalid(
          "simulation runtime input: value-input ordinal out of range");
    if (!std::holds_alternative<RuntimeValueInput>(
            workload.valueInputPlan[entry.valueInputOrdinal]))
      return detail::invalid("simulation runtime input: a runtime value names "
                             "a value input the workload fixes");
    if (entry.value.tokenCount != 1)
      return detail::invalid(
          "simulation runtime input: a runtime value holds exactly one token");
    if (llvm::Error error = detail::validateValueSequence(
            entry.value, context.valueInputShapes[entry.valueInputOrdinal],
            "simulation runtime input: runtime value", objectCount))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error
validateRuntimeStreams(llvm::ArrayRef<CanonicalStreamSequence> runtimeStreams,
                       const ResolvedLaunchContext &context,
                       std::uint64_t objectCount) {
  if (runtimeStreams.size() != context.numStreamInputs)
    return detail::invalid("simulation runtime input: stream table is not "
                           "total over the graph stream inputs");
  for (std::uint64_t ordinal = 0; ordinal < context.numStreamInputs;
       ++ordinal) {
    const CanonicalStreamSequence &stream = runtimeStreams[ordinal];
    if (static_cast<std::uint32_t>(stream.termination) >
        static_cast<std::uint32_t>(StreamTermination::OpenAfterLast))
      return detail::invalid(
          "simulation runtime input: stream termination is out of domain");
    if (llvm::Error error = detail::validateValueSequence(
            stream.values, context.streamInputShapes[ordinal],
            "simulation runtime input: stream", objectCount))
      return error;
  }
  return llvm::Error::success();
}

void remapPointerTargets(
    CanonicalValueSequence &sequence,
    const llvm::DenseMap<std::uint64_t, std::uint64_t> &canonicalOrdinals) {
  for (SemanticLane &lane : sequence.lanes)
    if (lane.pointerTarget)
      lane.pointerTarget->objectOrdinal =
          canonicalOrdinals.at(lane.pointerTarget->objectOrdinal);
}

llvm::Error canonicalizePointerValues(SpatialSimulationRuntimeInput &input,
                                      const ResolvedLaunchContext &context) {
  for (RuntimeValueEntry &entry : input.runtimeValues)
    if (llvm::Error error = detail::canonicalizePointerValueSequence(
            entry.value, context.valueInputShapes[entry.valueInputOrdinal],
            input.memoryObjects, context.graphOp))
      return error;
  for (std::uint64_t ordinal = 0; ordinal < input.runtimeStreams.size();
       ++ordinal)
    if (llvm::Error error = detail::canonicalizePointerValueSequence(
            input.runtimeStreams[ordinal].values,
            context.streamInputShapes[ordinal], input.memoryObjects,
            context.graphOp))
      return error;
  return llvm::Error::success();
}

llvm::Error
validateCanonicalPointerValues(const SpatialSimulationRuntimeInput &input,
                               const ResolvedLaunchContext &context) {
  for (const RuntimeValueEntry &entry : input.runtimeValues)
    if (llvm::Error error = detail::validateCanonicalPointerValueSequence(
            entry.value, context.valueInputShapes[entry.valueInputOrdinal],
            input.memoryObjects, context.graphOp,
            "simulation runtime input: runtime value"))
      return error;
  for (std::uint64_t ordinal = 0; ordinal < input.runtimeStreams.size();
       ++ordinal)
    if (llvm::Error error = detail::validateCanonicalPointerValueSequence(
            input.runtimeStreams[ordinal].values,
            context.streamInputShapes[ordinal], input.memoryObjects,
            context.graphOp, "simulation runtime input: runtime stream"))
      return error;
  return llvm::Error::success();
}

// Validate the binding table against the reachable imported roots and the
// object array. `bindings` must already be sorted by the typed root key.
llvm::Error
validateRootBindings(llvm::ArrayRef<MemoryRootBindingEntry> bindings,
                     llvm::ArrayRef<RuntimeMemoryObject> objects,
                     const ResolvedLaunchContext &context,
                     const dataflow::CanonicalDataflowProgramView &view) {
  if (bindings.size() != context.importedRoots.size())
    return detail::invalid("simulation runtime input: memory root bindings "
                           "are not total over the imported roots reachable "
                           "from the launch");
  std::vector<bool> referenced(objects.size(), false);
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const MemoryRootBindingEntry &entry = bindings[index];
    if (index > 0 &&
        detail::compareRootKeys(entry.root, bindings[index - 1].root) <= 0)
      return detail::invalid("simulation runtime input: memory root bindings "
                             "are not sorted or contain a duplicate");
    if (detail::compareRootKeys(entry.root, context.importedRoots[index]) != 0)
      return detail::invalid("simulation runtime input: a memory root binding "
                             "is missing or names an unrelated root");
    llvm::Expected<dataflow::CanonicalLogicalMemoryRootView> resolved =
        view.resolve(entry.root);
    if (!resolved)
      return resolved.takeError();
    if (entry.binding.objectOrdinal >= objects.size())
      return detail::invalid(
          "simulation runtime input: object ordinal out of range");
    if (entry.binding.byteOffset >=
        objects[entry.binding.objectOrdinal].initialBytes.size())
      return detail::invalid(
          "simulation runtime input: binding byte offset out of range");
    referenced[entry.binding.objectOrdinal] = true;
  }
  for (std::size_t ordinal = 0; ordinal < objects.size(); ++ordinal)
    if (!referenced[ordinal])
      return detail::invalid(
          "simulation runtime input: unreferenced memory object");
  return llvm::Error::success();
}

llvm::Error validateDiffBaselineEligibility(
    const SpatialSimulationWorkload &workload,
    llvm::ArrayRef<MemoryRootBindingEntry> bindings,
    const dataflow::CanonicalDataflowProgramView &view) {
  for (const SpatialMemoryObservable &observable :
       workload.observableContract.memories) {
    if (observable.form != MemoryObservationForm::DiffFromRuntimeInput)
      continue;
    auto baselineRoot =
        [&]() -> llvm::Expected<dataflow::LogicalMemoryRootRef> {
      if (const auto *direct =
              std::get_if<dataflow::LogicalMemoryRootOrViewRef>(
                  &observable.target)) {
        if (const auto *root =
                std::get_if<dataflow::LogicalMemoryRootRef>(direct))
          return *root;
        return std::get<dataflow::LogicalMemoryViewRef>(*direct).root;
      }
      llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> exposure =
          view.resolveExposure(dataflow::MemoryExposureRef{
              workload.launchRef,
              std::get<MemoryExposureTarget>(observable.target)
                  .memoryResultOrdinal});
      if (!exposure)
        return exposure.takeError();
      if (const auto *root =
              std::get_if<dataflow::LogicalMemoryRootRef>(&*exposure))
        return *root;
      return std::get<dataflow::LogicalMemoryViewRef>(*exposure).root;
    };
    llvm::Expected<dataflow::LogicalMemoryRootRef> root = baselineRoot();
    if (!root)
      return root.takeError();
    bool hasBaseline =
        std::any_of(bindings.begin(), bindings.end(), [&](const auto &entry) {
          return detail::compareRootKeys(entry.root, *root) == 0;
        });
    if (!hasBaseline)
      return detail::invalid("simulation runtime input: a diff observable has "
                             "no runtime memory baseline");
  }
  return llvm::Error::success();
}

} // namespace

namespace detail {

llvm::Error validateSpatialRuntimeInput(
    const SpatialSimulationRuntimeInput &input,
    const SpatialSimulationWorkload &workload,
    const ::loom::ArtifactIdentity &workloadIdentity,
    const ResolvedLaunchContext &context,
    const dataflow::CanonicalDataflowProgramView &view) {
  if (input.workloadIdentity != workloadIdentity)
    return invalid("simulation runtime input: does not name the exact "
                   "workload");
  if (llvm::Error error =
          validateRuntimeMemoryObjects(input.memoryObjects, context.graphOp))
    return error;
  if (llvm::Error error = validateRuntimeValues(
          input.runtimeValues, workload, context, input.memoryObjects.size()))
    return error;
  if (llvm::Error error = validateRuntimeStreams(input.runtimeStreams, context,
                                                 input.memoryObjects.size()))
    return error;
  if (llvm::Error error = validateRootBindings(
          input.memoryRootBindings, input.memoryObjects, context, view))
    return error;
  // The serialized object ordinals must be the canonical order derived from
  // the sorted binding keys.
  llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>> canonical =
      deriveCanonicalOrdinals(input.memoryRootBindings);
  if (!canonical)
    return canonical.takeError();
  for (const MemoryRootBindingEntry &entry : input.memoryRootBindings)
    if (canonical->at(entry.binding.objectOrdinal) !=
        entry.binding.objectOrdinal)
      return invalid("simulation runtime input: object ordinals are not the "
                     "canonical sorted-binding-key order");
  if (llvm::Error error = validateCanonicalPointerValues(input, context))
    return error;
  return validateDiffBaselineEligibility(workload, input.memoryRootBindings,
                                         view);
}

llvm::Expected<SpatialSimulationRuntimeInput> canonicalizeSpatialRuntimeInput(
    const SpatialSimulationRuntimeInputDraft &draft,
    const SpatialSimulationWorkload &workload,
    const ::loom::ArtifactIdentity &workloadIdentity,
    const ResolvedLaunchContext &context,
    const dataflow::CanonicalDataflowProgramView &view) {
  if (draft.workloadIdentity != workloadIdentity)
    return invalid("simulation runtime input: does not name the exact "
                   "workload");

  SpatialSimulationRuntimeInput input{draft.workloadIdentity};
  input.runtimeValues = draft.runtimeValues;
  std::sort(input.runtimeValues.begin(), input.runtimeValues.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.valueInputOrdinal < rhs.valueInputOrdinal;
            });
  input.runtimeStreams = draft.runtimeStreams;
  if (llvm::Error error = validateRuntimeMemoryObjectStructure(
          draft.memoryObjects, context.graphOp))
    return std::move(error);
  if (llvm::Error error = validateRuntimeValues(
          input.runtimeValues, workload, context, draft.memoryObjects.size()))
    return std::move(error);
  if (llvm::Error error = validateRuntimeStreams(input.runtimeStreams, context,
                                                 draft.memoryObjects.size()))
    return std::move(error);

  // Bindings: sort by the typed root key, validate totality and ranges
  // against the draft object slots, then derive the canonical ordinals.
  std::vector<MemoryRootBindingEntry> bindings;
  bindings.reserve(draft.memoryRootBindings.size());
  for (const RuntimeMemoryBindingDraft &binding : draft.memoryRootBindings)
    bindings.push_back(MemoryRootBindingEntry{
        binding.root,
        RuntimeMemoryRootBinding{binding.authorObject, binding.byteOffset}});
  std::sort(bindings.begin(), bindings.end(),
            [](const auto &lhs, const auto &rhs) {
              return compareRootKeys(lhs.root, rhs.root) < 0;
            });
  if (llvm::Error error =
          validateRootBindings(bindings, draft.memoryObjects, context, view))
    return std::move(error);
  llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>> canonical =
      deriveCanonicalOrdinals(bindings);
  if (!canonical)
    return canonical.takeError();

  input.memoryObjects.resize(draft.memoryObjects.size());
  for (std::size_t author = 0; author < draft.memoryObjects.size(); ++author)
    input.memoryObjects[canonical->at(author)] = draft.memoryObjects[author];
  for (RuntimeMemoryObject &object : input.memoryObjects)
    for (RuntimeMemoryPointer &pointer : object.pointerValues)
      pointer.target.objectOrdinal =
          canonical->at(pointer.target.objectOrdinal);
  for (MemoryRootBindingEntry &entry : bindings)
    entry.binding.objectOrdinal = canonical->at(entry.binding.objectOrdinal);
  for (RuntimeValueEntry &entry : input.runtimeValues)
    remapPointerTargets(entry.value, *canonical);
  for (CanonicalStreamSequence &stream : input.runtimeStreams)
    remapPointerTargets(stream.values, *canonical);
  input.memoryRootBindings = std::move(bindings);

  if (llvm::Error error = canonicalizeRuntimeMemoryPointers(input.memoryObjects,
                                                            context.graphOp))
    return std::move(error);
  if (llvm::Error error = canonicalizePointerValues(input, context))
    return std::move(error);
  if (llvm::Error error = validateDiffBaselineEligibility(
          workload, input.memoryRootBindings, view))
    return std::move(error);
  return input;
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Canonical encoding
//===----------------------------------------------------------------------===//

namespace {

template <typename EntityId>
void encodeEntityRef(detail::WireWriter &writer,
                     const ::loom::ArtifactReference<EntityId> &reference) {
  writer.identity(reference.artifact);
  writer.u64(reference.entity.value());
}

std::vector<std::uint8_t>
encodeSpatialRuntimeInput(const SpatialSimulationRuntimeInput &input,
                          const detail::ResolvedLaunchContext &context) {
  detail::WireWriter writer;
  writer.u32(static_cast<std::uint32_t>(SimulationWorkloadKind::Spatial));
  writer.identity(input.workloadIdentity);
  writer.u64(input.runtimeValues.size());
  for (const RuntimeValueEntry &entry : input.runtimeValues) {
    writer.u64(entry.valueInputOrdinal);
    detail::encodeValueSequence(
        writer, entry.value, context.valueInputShapes[entry.valueInputOrdinal]);
  }
  writer.u64(input.runtimeStreams.size());
  for (std::uint64_t ordinal = 0; ordinal < input.runtimeStreams.size();
       ++ordinal) {
    writer.u64(ordinal);
    detail::encodeStreamSequence(writer, input.runtimeStreams[ordinal],
                                 context.streamInputShapes[ordinal]);
  }
  writer.u64(input.memoryObjects.size());
  for (const RuntimeMemoryObject &object : input.memoryObjects)
    detail::encodeMemoryObject(writer, object);
  writer.u64(input.memoryRootBindings.size());
  for (const MemoryRootBindingEntry &entry : input.memoryRootBindings) {
    encodeEntityRef(writer, entry.root);
    writer.u64(entry.binding.objectOrdinal);
    writer.u64(entry.binding.byteOffset);
  }
  return writer.take();
}

// The decoded model plus the one launch context resolved while parsing; the
// caller reuses the context for semantic validation instead of resolving the
// same launch again.
struct DecodedSpatialRuntimeInput {
  SpatialSimulationRuntimeInput input;
  detail::ResolvedLaunchContext context;
};

llvm::Expected<DecodedSpatialRuntimeInput>
decodeSpatialRuntimeInput(llvm::ArrayRef<std::uint8_t> bytes,
                          const SpatialSimulationWorkload &workload,
                          const dataflow::CanonicalDataflowProgramView &view) {
  detail::WireReader reader(bytes);
  llvm::Expected<std::uint32_t> root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root == 1)
    return detail::invalid(
        "simulation runtime input: the System root is fail-closed");
  if (*root != static_cast<std::uint32_t>(SimulationWorkloadKind::Spatial))
    return detail::invalid(
        "simulation runtime input: unknown root discriminant");

  llvm::Expected<::loom::ArtifactIdentity> workloadIdentity = reader.identity();
  if (!workloadIdentity)
    return workloadIdentity.takeError();
  SpatialSimulationRuntimeInput input{*workloadIdentity};

  llvm::Expected<ResolvedLaunchContext> context =
      detail::resolveLaunchContext(view, workload.launchRef);
  if (!context)
    return context.takeError();

  llvm::Expected<std::uint64_t> valueCount = reader.u64();
  if (!valueCount)
    return valueCount.takeError();
  if (llvm::Error error = reader.guardCount(*valueCount, 28))
    return std::move(error);
  input.runtimeValues.reserve(*valueCount);
  for (std::uint64_t index = 0; index < *valueCount; ++index) {
    llvm::Expected<std::uint64_t> ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (index > 0 && *ordinal <= input.runtimeValues.back().valueInputOrdinal)
      return detail::invalid("simulation runtime input: runtime values are "
                             "not sorted or contain a duplicate");
    if (*ordinal >= context->numValueInputs)
      return detail::invalid(
          "simulation runtime input: value-input ordinal out of range");
    llvm::Expected<CanonicalValueSequence> value = detail::decodeValueSequence(
        reader, context->valueInputShapes[*ordinal]);
    if (!value)
      return value.takeError();
    input.runtimeValues.push_back(
        RuntimeValueEntry{*ordinal, std::move(*value)});
  }

  llvm::Expected<std::uint64_t> streamCount = reader.u64();
  if (!streamCount)
    return streamCount.takeError();
  if (llvm::Error error = reader.guardCount(*streamCount, 28))
    return std::move(error);
  input.runtimeStreams.reserve(*streamCount);
  for (std::uint64_t index = 0; index < *streamCount; ++index) {
    llvm::Expected<std::uint64_t> ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal != index)
      return detail::invalid("simulation runtime input: stream table keys are "
                             "not the dense sorted ordinals");
    if (*ordinal >= context->numStreamInputs)
      return detail::invalid(
          "simulation runtime input: stream-input ordinal out of range");
    llvm::Expected<CanonicalStreamSequence> stream =
        detail::decodeStreamSequence(reader,
                                     context->streamInputShapes[*ordinal]);
    if (!stream)
      return stream.takeError();
    input.runtimeStreams.push_back(std::move(*stream));
  }

  llvm::Expected<std::uint64_t> objectCount = reader.u64();
  if (!objectCount)
    return objectCount.takeError();
  if (llvm::Error error = reader.guardCount(*objectCount, 16))
    return std::move(error);
  input.memoryObjects.reserve(*objectCount);
  for (std::uint64_t index = 0; index < *objectCount; ++index) {
    llvm::Expected<RuntimeMemoryObject> object =
        detail::decodeMemoryObject(reader, context->graphOp);
    if (!object)
      return object.takeError();
    input.memoryObjects.push_back(std::move(*object));
  }

  llvm::Expected<std::uint64_t> bindingCount = reader.u64();
  if (!bindingCount)
    return bindingCount.takeError();
  if (llvm::Error error = reader.guardCount(*bindingCount, 56))
    return std::move(error);
  input.memoryRootBindings.reserve(*bindingCount);
  for (std::uint64_t index = 0; index < *bindingCount; ++index) {
    llvm::Expected<::loom::ArtifactIdentity> artifact = reader.identity();
    if (!artifact)
      return artifact.takeError();
    llvm::Expected<std::uint64_t> entity = reader.u64();
    if (!entity)
      return entity.takeError();
    const dataflow::LogicalMemoryRootRef root{
        *artifact, dataflow::LogicalMemoryRootId(*entity)};
    if (index > 0 && detail::compareRootKeys(
                         root, input.memoryRootBindings.back().root) <= 0)
      return detail::invalid("simulation runtime input: memory root bindings "
                             "are not sorted or contain a duplicate");
    llvm::Expected<std::uint64_t> objectOrdinal = reader.u64();
    if (!objectOrdinal)
      return objectOrdinal.takeError();
    llvm::Expected<std::uint64_t> byteOffset = reader.u64();
    if (!byteOffset)
      return byteOffset.takeError();
    input.memoryRootBindings.push_back(MemoryRootBindingEntry{
        root, RuntimeMemoryRootBinding{*objectOrdinal, *byteOffset}});
  }

  if (!reader.atEnd())
    return detail::invalid("simulation runtime input: trailing bytes");
  return DecodedSpatialRuntimeInput{std::move(input), std::move(*context)};
}

} // namespace

//===----------------------------------------------------------------------===//
// Finalization and import
//===----------------------------------------------------------------------===//

llvm::Expected<CanonicalSimulationRuntimeInput> finalizeSimulationRuntimeInput(
    const SpatialSimulationRuntimeInputDraft &draft,
    const CanonicalSimulationWorkload &workload,
    const dataflow::CanonicalDataflowProgramView &view) {
  const SpatialSimulationWorkload *spatialWorkload = workload.spatial();
  if (!spatialWorkload)
    return detail::invalid("simulation runtime input: Spatial finalization "
                           "requires a Spatial workload root");
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(view, spatialWorkload->launchRef);
  if (!context)
    return context.takeError();
  llvm::Expected<SpatialSimulationRuntimeInput> input =
      detail::canonicalizeSpatialRuntimeInput(
          draft, *spatialWorkload, workload.identity(), *context, view);
  if (!input)
    return input.takeError();
  if (llvm::Error error = detail::validateSpatialRuntimeInput(
          *input, *spatialWorkload, workload.identity(), *context, view))
    return std::move(error);
  ::loom::CanonicalSemanticBytes bytes(
      encodeSpatialRuntimeInput(*input, *context));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationRuntimeInputSchema, bytes);
  return CanonicalSimulationRuntimeInput(identity, std::move(*input),
                                         std::move(bytes));
}

llvm::Expected<CanonicalSimulationRuntimeInput>
importSimulationRuntimeInput(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                             const CanonicalSimulationWorkload &workload,
                             const dataflow::CanonicalDataflowProgramView &view,
                             const ::loom::ArtifactIdentity &expectedIdentity) {
  const SpatialSimulationWorkload *spatialWorkload = workload.spatial();
  if (!spatialWorkload)
    return detail::invalid("simulation runtime input: Spatial import requires "
                           "a Spatial workload root");
  llvm::Expected<DecodedSpatialRuntimeInput> decoded =
      decodeSpatialRuntimeInput(canonicalBytes, *spatialWorkload, view);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = detail::validateSpatialRuntimeInput(
          decoded->input, *spatialWorkload, workload.identity(),
          decoded->context, view))
    return std::move(error);
  const std::vector<std::uint8_t> reencoded =
      encodeSpatialRuntimeInput(decoded->input, decoded->context);
  if (!llvm::ArrayRef<std::uint8_t>(reencoded).equals(canonicalBytes))
    return detail::invalid(
        "simulation runtime input: noncanonical bytes do not re-encode "
        "exactly");
  ::loom::CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(canonicalBytes.begin(), canonicalBytes.end()));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationRuntimeInputSchema, bytes);
  if (identity != expectedIdentity)
    return detail::invalid("simulation runtime input: identity does not "
                           "match the expected artifact");
  return CanonicalSimulationRuntimeInput(identity, std::move(decoded->input),
                                         std::move(bytes));
}

} // namespace loom::sim
