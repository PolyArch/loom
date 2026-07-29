//===- StructuredProgramSimulationRuntimeInput.cpp -----------------------===//

#include "SimulationWireInternal.h"

#include "Common/ArtifactFinalizer.h"

#include <algorithm>
#include <utility>

namespace loom::sim {
namespace {

llvm::Error
validateRuntimeValues(llvm::ArrayRef<StructuredRuntimeValueEntry> values,
                      const StructuredProgramSimulationWorkload &workload,
                      const detail::ResolvedStructuredProgramContext &context) {
  std::uint64_t expectedCount = 0;
  for (const StructuredProgramArgumentSource &source : workload.argumentPlan)
    if (std::holds_alternative<StructuredRuntimeValueInput>(source))
      ++expectedCount;
  if (values.size() != expectedCount)
    return detail::invalid("simulation runtime input: Structured runtime "
                           "values do not exactly complement the workload");
  for (std::size_t index = 0; index < values.size(); ++index) {
    const StructuredRuntimeValueEntry &entry = values[index];
    if (index > 0 && entry.argumentOrdinal <= values[index - 1].argumentOrdinal)
      return detail::invalid("simulation runtime input: Structured runtime "
                             "values are not sorted or contain a duplicate");
    if (entry.argumentOrdinal >= workload.argumentPlan.size() ||
        !std::holds_alternative<StructuredRuntimeValueInput>(
            workload.argumentPlan[entry.argumentOrdinal]))
      return detail::invalid("simulation runtime input: Structured runtime "
                             "value names a non-runtime ABI argument");
    if (entry.value.tokenCount != 1)
      return detail::invalid("simulation runtime input: Structured runtime "
                             "values hold exactly one token");
    if (llvm::Error error = detail::validateValueSequence(
            entry.value, *context.argumentShapes[entry.argumentOrdinal],
            "simulation runtime input: Structured runtime value"))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error
validatePointerBindings(llvm::ArrayRef<StructuredPointerBindingEntry> bindings,
                        llvm::ArrayRef<RuntimeMemoryObject> objects,
                        const StructuredProgramSimulationWorkload &workload) {
  std::uint64_t expectedCount = 0;
  for (const StructuredProgramArgumentSource &source : workload.argumentPlan)
    if (std::holds_alternative<StructuredRuntimeMemoryInput>(source))
      ++expectedCount;
  if (bindings.size() != expectedCount)
    return detail::invalid("simulation runtime input: pointer bindings are not "
                           "total over RuntimeMemory arguments");
  std::vector<bool> referenced(objects.size(), false);
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const StructuredPointerBindingEntry &entry = bindings[index];
    if (index > 0 &&
        entry.argumentOrdinal <= bindings[index - 1].argumentOrdinal)
      return detail::invalid("simulation runtime input: pointer bindings are "
                             "not sorted or contain a duplicate");
    if (entry.argumentOrdinal >= workload.argumentPlan.size() ||
        !std::holds_alternative<StructuredRuntimeMemoryInput>(
            workload.argumentPlan[entry.argumentOrdinal]))
      return detail::invalid("simulation runtime input: pointer binding names "
                             "a non-memory ABI argument");
    if (entry.binding.objectOrdinal >= objects.size())
      return detail::invalid(
          "simulation runtime input: object ordinal out of range");
    if (entry.binding.byteOffset >=
        objects[entry.binding.objectOrdinal].initialBytes.size())
      return detail::invalid(
          "simulation runtime input: pointer byte offset out of range");
    referenced[entry.binding.objectOrdinal] = true;
  }
  for (bool used : referenced)
    if (!used)
      return detail::invalid(
          "simulation runtime input: unreferenced memory object");
  return llvm::Error::success();
}

std::vector<detail::RuntimeObjectBindingKey>
bindingKeys(llvm::ArrayRef<StructuredPointerBindingEntry> bindings) {
  std::vector<detail::RuntimeObjectBindingKey> keys;
  keys.reserve(bindings.size());
  for (const StructuredPointerBindingEntry &entry : bindings) {
    detail::WireWriter writer;
    writer.u64(entry.argumentOrdinal);
    writer.u64(entry.binding.byteOffset);
    keys.push_back({entry.binding.objectOrdinal, std::move(writer).take()});
  }
  return keys;
}

llvm::Error validateStructuredRuntimeInput(
    const StructuredProgramSimulationRuntimeInput &input,
    const StructuredProgramSimulationWorkload &workload,
    const ::loom::ArtifactIdentity &workloadIdentity,
    const detail::ResolvedStructuredProgramContext &context) {
  if (input.workloadIdentity != workloadIdentity)
    return detail::invalid("simulation runtime input: does not name the exact "
                           "workload");
  if (llvm::Error error =
          validateRuntimeValues(input.runtimeValues, workload, context))
    return error;
  if (llvm::Error error =
          detail::validateRuntimeMemoryObjects(input.memoryObjects))
    return error;
  if (llvm::Error error = validatePointerBindings(
          input.pointerBindings, input.memoryObjects, workload))
    return error;
  llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>> canonical =
      detail::deriveCanonicalObjectOrdinals(bindingKeys(input.pointerBindings));
  if (!canonical)
    return canonical.takeError();
  for (const StructuredPointerBindingEntry &entry : input.pointerBindings)
    if (canonical->at(entry.binding.objectOrdinal) !=
        entry.binding.objectOrdinal)
      return detail::invalid("simulation runtime input: object ordinals are "
                             "not in canonical pointer-binding-key order");
  return llvm::Error::success();
}

llvm::Expected<StructuredProgramSimulationRuntimeInput>
canonicalizeStructuredRuntimeInput(
    const StructuredProgramSimulationRuntimeInputDraft &draft,
    const StructuredProgramSimulationWorkload &workload,
    const ::loom::ArtifactIdentity &workloadIdentity,
    const detail::ResolvedStructuredProgramContext &context) {
  if (draft.workloadIdentity != workloadIdentity)
    return detail::invalid("simulation runtime input: does not name the exact "
                           "workload");
  StructuredProgramSimulationRuntimeInput input{draft.workloadIdentity};
  input.runtimeValues = draft.runtimeValues;
  std::sort(input.runtimeValues.begin(), input.runtimeValues.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.argumentOrdinal < rhs.argumentOrdinal;
            });
  if (llvm::Error error =
          validateRuntimeValues(input.runtimeValues, workload, context))
    return std::move(error);
  if (llvm::Error error =
          detail::validateRuntimeMemoryObjects(draft.memoryObjects))
    return std::move(error);

  input.pointerBindings.reserve(draft.pointerBindings.size());
  for (const StructuredPointerBindingDraft &binding : draft.pointerBindings)
    input.pointerBindings.push_back(
        {binding.argumentOrdinal,
         RuntimeMemoryRootBinding{binding.authorObject, binding.byteOffset}});
  std::sort(input.pointerBindings.begin(), input.pointerBindings.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.argumentOrdinal < rhs.argumentOrdinal;
            });
  if (llvm::Error error = validatePointerBindings(
          input.pointerBindings, draft.memoryObjects, workload))
    return std::move(error);

  llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>> canonical =
      detail::deriveCanonicalObjectOrdinals(bindingKeys(input.pointerBindings));
  if (!canonical)
    return canonical.takeError();
  input.memoryObjects.resize(draft.memoryObjects.size());
  for (std::size_t author = 0; author < draft.memoryObjects.size(); ++author)
    input.memoryObjects[canonical->at(author)] = draft.memoryObjects[author];
  for (StructuredPointerBindingEntry &entry : input.pointerBindings)
    entry.binding.objectOrdinal = canonical->at(entry.binding.objectOrdinal);
  return input;
}

std::vector<std::uint8_t> encodeStructuredRuntimeInput(
    const StructuredProgramSimulationRuntimeInput &input) {
  detail::WireWriter writer;
  writer.u32(
      static_cast<std::uint32_t>(SimulationWorkloadKind::StructuredProgram));
  writer.identity(input.workloadIdentity);
  writer.u64(input.runtimeValues.size());
  for (const StructuredRuntimeValueEntry &entry : input.runtimeValues) {
    writer.u64(entry.argumentOrdinal);
    detail::encodeValueSequence(writer, entry.value);
  }
  writer.u64(input.memoryObjects.size());
  for (const RuntimeMemoryObject &object : input.memoryObjects)
    detail::encodeMemoryObject(writer, object);
  writer.u64(input.pointerBindings.size());
  for (const StructuredPointerBindingEntry &entry : input.pointerBindings) {
    writer.u64(entry.argumentOrdinal);
    writer.u64(entry.binding.objectOrdinal);
    writer.u64(entry.binding.byteOffset);
  }
  return writer.take();
}

struct DecodedStructuredRuntimeInput {
  StructuredProgramSimulationRuntimeInput input;
  detail::ResolvedStructuredProgramContext context;
};

llvm::Expected<DecodedStructuredRuntimeInput> decodeStructuredRuntimeInput(
    llvm::ArrayRef<std::uint8_t> bytes,
    const StructuredProgramSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &view) {
  detail::WireReader reader(bytes);
  llvm::Expected<std::uint32_t> root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root == static_cast<std::uint32_t>(SimulationWorkloadKind::Spatial))
    return detail::invalid("simulation runtime input: Structured import "
                           "received a Spatial root");
  if (*root == static_cast<std::uint32_t>(SimulationWorkloadKind::System))
    return detail::invalid(
        "simulation runtime input: the System root is fail-closed");
  if (*root !=
      static_cast<std::uint32_t>(SimulationWorkloadKind::StructuredProgram))
    return detail::invalid(
        "simulation runtime input: unknown root discriminant");

  llvm::Expected<::loom::ArtifactIdentity> workloadIdentity = reader.identity();
  if (!workloadIdentity)
    return workloadIdentity.takeError();
  StructuredProgramSimulationRuntimeInput input{*workloadIdentity};
  llvm::Expected<detail::ResolvedStructuredProgramContext> context =
      detail::resolveStructuredProgramContext(view, workload.entryRef);
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
    if (*ordinal >= context->argumentShapes.size() ||
        !context->argumentShapes[*ordinal])
      return detail::invalid("simulation runtime input: runtime-value ABI "
                             "argument is out of range or not a value");
    llvm::Expected<CanonicalValueSequence> value =
        detail::decodeValueSequence(reader, *context->argumentShapes[*ordinal]);
    if (!value)
      return value.takeError();
    input.runtimeValues.push_back({*ordinal, std::move(*value)});
  }

  llvm::Expected<std::uint64_t> objectCount = reader.u64();
  if (!objectCount)
    return objectCount.takeError();
  if (llvm::Error error = reader.guardCount(*objectCount, 16))
    return std::move(error);
  input.memoryObjects.reserve(*objectCount);
  for (std::uint64_t index = 0; index < *objectCount; ++index) {
    llvm::Expected<RuntimeMemoryObject> object =
        detail::decodeMemoryObject(reader);
    if (!object)
      return object.takeError();
    input.memoryObjects.push_back(std::move(*object));
  }

  llvm::Expected<std::uint64_t> bindingCount = reader.u64();
  if (!bindingCount)
    return bindingCount.takeError();
  if (llvm::Error error = reader.guardCount(*bindingCount, 24))
    return std::move(error);
  input.pointerBindings.reserve(*bindingCount);
  for (std::uint64_t index = 0; index < *bindingCount; ++index) {
    llvm::Expected<std::uint64_t> argumentOrdinal = reader.u64();
    if (!argumentOrdinal)
      return argumentOrdinal.takeError();
    llvm::Expected<std::uint64_t> objectOrdinal = reader.u64();
    if (!objectOrdinal)
      return objectOrdinal.takeError();
    llvm::Expected<std::uint64_t> byteOffset = reader.u64();
    if (!byteOffset)
      return byteOffset.takeError();
    input.pointerBindings.push_back(
        {*argumentOrdinal,
         RuntimeMemoryRootBinding{*objectOrdinal, *byteOffset}});
  }
  if (!reader.atEnd())
    return detail::invalid("simulation runtime input: trailing bytes");
  return DecodedStructuredRuntimeInput{std::move(input), std::move(*context)};
}

} // namespace

llvm::Expected<CanonicalSimulationRuntimeInput> finalizeSimulationRuntimeInput(
    const StructuredProgramSimulationRuntimeInputDraft &draft,
    const CanonicalSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &view) {
  const StructuredProgramSimulationWorkload *structured =
      workload.structuredProgram();
  if (!structured)
    return detail::invalid("simulation runtime input: Structured finalization "
                           "requires a Structured workload root");
  llvm::Expected<detail::ResolvedStructuredProgramContext> context =
      detail::resolveStructuredProgramContext(view, structured->entryRef);
  if (!context)
    return context.takeError();
  llvm::Expected<StructuredProgramSimulationRuntimeInput> input =
      canonicalizeStructuredRuntimeInput(draft, *structured,
                                         workload.identity(), *context);
  if (!input)
    return input.takeError();
  if (llvm::Error error = validateStructuredRuntimeInput(
          *input, *structured, workload.identity(), *context))
    return std::move(error);
  ::loom::CanonicalSemanticBytes bytes(encodeStructuredRuntimeInput(*input));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationRuntimeInputSchema, bytes);
  return CanonicalSimulationRuntimeInput(
      identity, SimulationRuntimeInputModel{std::move(*input)},
      std::move(bytes));
}

llvm::Expected<CanonicalSimulationRuntimeInput> importSimulationRuntimeInput(
    llvm::ArrayRef<std::uint8_t> canonicalBytes,
    const CanonicalSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &view,
    const ::loom::ArtifactIdentity &expectedIdentity) {
  const StructuredProgramSimulationWorkload *structured =
      workload.structuredProgram();
  if (!structured)
    return detail::invalid("simulation runtime input: Structured import "
                           "requires a Structured workload root");
  llvm::Expected<DecodedStructuredRuntimeInput> decoded =
      decodeStructuredRuntimeInput(canonicalBytes, *structured, view);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = validateStructuredRuntimeInput(
          decoded->input, *structured, workload.identity(), decoded->context))
    return std::move(error);
  const std::vector<std::uint8_t> reencoded =
      encodeStructuredRuntimeInput(decoded->input);
  if (!llvm::ArrayRef<std::uint8_t>(reencoded).equals(canonicalBytes))
    return detail::invalid("simulation runtime input: noncanonical bytes do "
                           "not re-encode exactly");
  ::loom::CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(canonicalBytes.begin(), canonicalBytes.end()));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationRuntimeInputSchema, bytes);
  if (identity != expectedIdentity)
    return detail::invalid("simulation runtime input: identity does not match "
                           "the expected artifact");
  return CanonicalSimulationRuntimeInput(
      identity, SimulationRuntimeInputModel{std::move(decoded->input)},
      std::move(bytes));
}

} // namespace loom::sim
