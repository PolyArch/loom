//===- SystemSimulationRuntimeInput.cpp - Deployment runtime input -------===//

#include "SimulationWireInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Deployment/DeploymentReference.h"

#include <algorithm>
#include <utility>

namespace loom::sim {
namespace {

bool acceptsInput(deployment::HostExternalInterfaceDirection direction) {
  return direction == deployment::HostExternalInterfaceDirection::Input ||
         direction == deployment::HostExternalInterfaceDirection::InOut;
}

void encodeInterfaceRef(
    detail::WireWriter &writer,
    const deployment::DeploymentExternalInterfaceRef &reference) {
  writer.bytes(deployment::encodeDeploymentExternalInterfaceRef(reference));
}

llvm::Expected<deployment::DeploymentExternalInterfaceRef>
decodeInterfaceRef(detail::WireReader &reader) {
  auto bytes = reader.bytes(deployment::deploymentCatalogReferenceWireSize);
  if (!bytes)
    return bytes.takeError();
  return deployment::decodeDeploymentExternalInterfaceRef(*bytes);
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

std::vector<std::uint64_t>
expectedRuntimeEntryValues(const SystemSimulationWorkload &workload) {
  std::vector<std::uint64_t> result;
  for (std::uint64_t ordinal = 0; ordinal < workload.valueInputPlan.size();
       ++ordinal)
    if (std::holds_alternative<RuntimeValueInput>(
            workload.valueInputPlan[ordinal]))
      result.push_back(ordinal);
  return result;
}

std::vector<deployment::DeploymentExternalInterfaceRef>
expectedRuntimeExternalValues(const SystemSimulationWorkload &workload) {
  std::vector<deployment::DeploymentExternalInterfaceRef> result;
  for (const SystemExternalValueInputPlanEntry &entry :
       workload.externalValueInputPlan)
    if (std::holds_alternative<RuntimeValueInput>(entry.source))
      result.push_back(entry.interfaceRef);
  return result;
}

std::vector<deployment::DeploymentExternalInterfaceRef>
expectedStreamInputs(const detail::ResolvedSystemContext &context) {
  std::vector<deployment::DeploymentExternalInterfaceRef> result;
  for (const deployment::HostExternalInterface &interface : context.interfaces)
    if (interface.kind == deployment::HostExternalInterfaceKind::Stream &&
        acceptsInput(interface.direction))
      result.push_back(
          {context.deploymentIdentity, interface.interfaceOrdinal});
  return result;
}

std::vector<deployment::DeploymentExternalInterfaceRef>
expectedMemoryBindings(const SystemSimulationWorkload &workload,
                       const detail::ResolvedSystemContext &context) {
  std::vector<deployment::DeploymentExternalInterfaceRef> result;
  for (const deployment::HostExternalInterface &interface : context.interfaces)
    if (interface.kind == deployment::HostExternalInterfaceKind::Memory &&
        acceptsInput(interface.direction))
      result.push_back(
          {context.deploymentIdentity, interface.interfaceOrdinal});
  for (const SystemMemoryObservable &observable :
       workload.observableContract.memories)
    result.push_back(observable.interfaceRef);
  std::sort(result.begin(), result.end(),
            deployment::deploymentExternalInterfaceRefLess);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

llvm::Error
validateRuntimeEntryValues(llvm::ArrayRef<SystemRuntimeEntryValue> values,
                           const SystemSimulationWorkload &workload,
                           const detail::ResolvedSystemContext &context,
                           std::uint64_t objectCount) {
  const std::vector<std::uint64_t> expected =
      expectedRuntimeEntryValues(workload);
  if (values.size() != expected.size())
    return detail::invalid("simulation runtime input: program runtime values "
                           "do not exactly complement the workload");
  for (std::size_t index = 0; index < values.size(); ++index) {
    const SystemRuntimeEntryValue &entry = values[index];
    if (entry.valueArgumentOrdinal != expected[index])
      return detail::invalid("simulation runtime input: program runtime "
                             "values are not the canonical complement");
    if (entry.value.tokenCount != 1)
      return detail::invalid("simulation runtime input: program runtime value "
                             "does not contain exactly one token");
    const detail::LaneShape &shape =
        context.valueArgumentShapes[entry.valueArgumentOrdinal];
    if (llvm::Error error = detail::validateValueSequence(
            entry.value, shape, "simulation runtime input: program value",
            objectCount))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error
validateRuntimeExternalValues(llvm::ArrayRef<SystemRuntimeExternalValue> values,
                              const SystemSimulationWorkload &workload,
                              const detail::ResolvedSystemContext &context,
                              llvm::ArrayRef<RuntimeMemoryObject> objects,
                              bool requireCanonicalPointers) {
  const auto expected = expectedRuntimeExternalValues(workload);
  if (values.size() != expected.size())
    return detail::invalid("simulation runtime input: external runtime values "
                           "do not exactly complement the workload");
  for (std::size_t index = 0; index < values.size(); ++index) {
    const SystemRuntimeExternalValue &entry = values[index];
    if (!(entry.interfaceRef == expected[index]))
      return detail::invalid("simulation runtime input: external runtime "
                             "values are not the canonical complement");
    if (entry.value.tokenCount != 1)
      return detail::invalid("simulation runtime input: external runtime "
                             "value does not contain exactly one token");
    const detail::LaneShape &shape =
        interfaceShape(context, entry.interfaceRef);
    if (llvm::Error error = detail::validateValueSequence(
            entry.value, shape, "simulation runtime input: external value",
            objects.size()))
      return error;
    if (requireCanonicalPointers)
      if (llvm::Error error = detail::validateCanonicalPointerValueSequence(
              entry.value, shape, objects, context.layoutOperation(),
              "simulation runtime input: external value"))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error
validateStreamInputs(llvm::ArrayRef<SystemExternalStreamInput> streams,
                     const detail::ResolvedSystemContext &context,
                     llvm::ArrayRef<RuntimeMemoryObject> objects,
                     bool requireCanonicalPointers) {
  const auto expected = expectedStreamInputs(context);
  if (streams.size() != expected.size())
    return detail::invalid("simulation runtime input: external stream table "
                           "is not total over input and inout streams");
  for (std::size_t index = 0; index < streams.size(); ++index) {
    const SystemExternalStreamInput &entry = streams[index];
    if (!(entry.interfaceRef == expected[index]))
      return detail::invalid("simulation runtime input: external stream table "
                             "is not the canonical total table");
    const detail::LaneShape &shape =
        interfaceShape(context, entry.interfaceRef);
    if (llvm::Error error = detail::validateValueSequence(
            entry.stream.values, shape,
            "simulation runtime input: external stream", objects.size()))
      return error;
    if (static_cast<std::uint32_t>(entry.stream.termination) >
        static_cast<std::uint32_t>(StreamTermination::OpenAfterLast))
      return detail::invalid("simulation runtime input: external stream "
                             "termination is out of domain");
    if (requireCanonicalPointers)
      if (llvm::Error error = detail::validateCanonicalPointerValueSequence(
              entry.stream.values, shape, objects, context.layoutOperation(),
              "simulation runtime input: external stream"))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error validateMemoryBindings(
    llvm::ArrayRef<SystemMemoryInterfaceBindingEntry> bindings,
    llvm::ArrayRef<RuntimeMemoryObject> objects,
    const SystemSimulationWorkload &workload,
    const detail::ResolvedSystemContext &context) {
  const auto expected = expectedMemoryBindings(workload, context);
  if (bindings.size() != expected.size())
    return detail::invalid("simulation runtime input: memory interface "
                           "bindings are not total");
  std::vector<bool> referenced(objects.size(), false);
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const SystemMemoryInterfaceBindingEntry &entry = bindings[index];
    if (!(entry.interfaceRef == expected[index]))
      return detail::invalid("simulation runtime input: memory interface "
                             "bindings are not the canonical total table");
    if (entry.binding.objectOrdinal >= objects.size())
      return detail::invalid(
          "simulation runtime input: memory object ordinal is out of range");
    if (entry.binding.byteOffset >=
        objects[entry.binding.objectOrdinal].initialBytes.size())
      return detail::invalid("simulation runtime input: memory interface "
                             "binding offset is out of range");
    referenced[entry.binding.objectOrdinal] = true;
  }
  for (bool used : referenced)
    if (!used)
      return detail::invalid(
          "simulation runtime input: unreferenced memory object");
  return llvm::Error::success();
}

std::vector<detail::RuntimeObjectBindingKey>
bindingKeys(llvm::ArrayRef<SystemMemoryInterfaceBindingEntry> bindings) {
  std::vector<detail::RuntimeObjectBindingKey> keys;
  keys.reserve(bindings.size());
  for (const SystemMemoryInterfaceBindingEntry &entry : bindings) {
    detail::WireWriter writer;
    encodeInterfaceRef(writer, entry.interfaceRef);
    writer.u64(entry.binding.byteOffset);
    keys.push_back({entry.binding.objectOrdinal, std::move(writer).take()});
  }
  return keys;
}

void remapPointerTargets(
    CanonicalValueSequence &sequence,
    const llvm::DenseMap<std::uint64_t, std::uint64_t> &ordinals) {
  for (SemanticLane &lane : sequence.lanes)
    if (lane.pointerTarget)
      lane.pointerTarget->objectOrdinal =
          ordinals.at(lane.pointerTarget->objectOrdinal);
}

llvm::Error
canonicalizePointerValues(SystemSimulationRuntimeInput &input,
                          const detail::ResolvedSystemContext &context) {
  for (SystemRuntimeEntryValue &entry : input.runtimeEntryValues) {
    const detail::LaneShape &shape =
        context.valueArgumentShapes[entry.valueArgumentOrdinal];
    if (llvm::Error error = detail::canonicalizePointerValueSequence(
            entry.value, shape, input.memoryObjects, context.layoutOperation()))
      return error;
  }
  for (SystemRuntimeExternalValue &entry : input.runtimeExternalValues)
    if (llvm::Error error = detail::canonicalizePointerValueSequence(
            entry.value, interfaceShape(context, entry.interfaceRef),
            input.memoryObjects, context.layoutOperation()))
      return error;
  for (SystemExternalStreamInput &entry : input.externalStreamInputs)
    if (llvm::Error error = detail::canonicalizePointerValueSequence(
            entry.stream.values, interfaceShape(context, entry.interfaceRef),
            input.memoryObjects, context.layoutOperation()))
      return error;
  return llvm::Error::success();
}

std::vector<std::uint8_t>
encodeSystemRuntimeInput(const SystemSimulationRuntimeInput &input,
                         const detail::ResolvedSystemContext &context) {
  detail::WireWriter writer;
  writer.u32(static_cast<std::uint32_t>(SimulationWorkloadKind::System));
  writer.identity(input.workloadIdentity);
  writer.u64(input.runtimeEntryValues.size());
  for (const SystemRuntimeEntryValue &entry : input.runtimeEntryValues) {
    writer.u64(entry.valueArgumentOrdinal);
    detail::encodeValueSequence(
        writer, entry.value,
        context.valueArgumentShapes[entry.valueArgumentOrdinal]);
  }
  writer.u64(input.runtimeExternalValues.size());
  for (const SystemRuntimeExternalValue &entry : input.runtimeExternalValues) {
    encodeInterfaceRef(writer, entry.interfaceRef);
    detail::encodeValueSequence(writer, entry.value,
                                interfaceShape(context, entry.interfaceRef));
  }
  writer.u64(input.externalStreamInputs.size());
  for (const SystemExternalStreamInput &entry : input.externalStreamInputs) {
    encodeInterfaceRef(writer, entry.interfaceRef);
    detail::encodeStreamSequence(writer, entry.stream,
                                 interfaceShape(context, entry.interfaceRef));
  }
  writer.u64(input.memoryObjects.size());
  for (const RuntimeMemoryObject &object : input.memoryObjects)
    detail::encodeMemoryObject(writer, object);
  writer.u64(input.memoryInterfaceBindings.size());
  for (const SystemMemoryInterfaceBindingEntry &entry :
       input.memoryInterfaceBindings) {
    encodeInterfaceRef(writer, entry.interfaceRef);
    writer.u64(entry.binding.objectOrdinal);
    writer.u64(entry.binding.byteOffset);
  }
  return writer.take();
}

struct DecodedSystemRuntimeInput {
  SystemSimulationRuntimeInput input;
  detail::ResolvedSystemContext context;
};

llvm::Expected<DecodedSystemRuntimeInput>
decodeSystemRuntimeInput(llvm::ArrayRef<std::uint8_t> bytes,
                         const SystemSimulationWorkload &workload,
                         const deployment::FinalizedDeployment &deployment,
                         const ArtifactStore &store) {
  detail::WireReader reader(bytes);
  auto root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root != static_cast<std::uint32_t>(SimulationWorkloadKind::System))
    return detail::invalid("simulation runtime input: System import received "
                           "a non-System root");
  auto workloadIdentity = reader.identity();
  if (!workloadIdentity)
    return workloadIdentity.takeError();
  SystemSimulationRuntimeInput input{*workloadIdentity};
  auto context =
      detail::resolveSystemContext(deployment, workload.programEntryRef, store);
  if (!context)
    return context.takeError();

  auto entryValueCount = reader.u64();
  if (!entryValueCount)
    return entryValueCount.takeError();
  if (llvm::Error error = reader.guardCount(*entryValueCount, 28))
    return std::move(error);
  input.runtimeEntryValues.reserve(*entryValueCount);
  for (std::uint64_t index = 0; index < *entryValueCount; ++index) {
    auto ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal >= context->valueArgumentShapes.size())
      return detail::invalid("simulation runtime input: program value "
                             "argument ordinal is out of range");
    auto value = detail::decodeValueSequence(
        reader, context->valueArgumentShapes[*ordinal]);
    if (!value)
      return value.takeError();
    input.runtimeEntryValues.push_back({*ordinal, std::move(*value)});
  }

  auto externalValueCount = reader.u64();
  if (!externalValueCount)
    return externalValueCount.takeError();
  if (llvm::Error error = reader.guardCount(
          *externalValueCount,
          deployment::deploymentCatalogReferenceWireSize + 20))
    return std::move(error);
  input.runtimeExternalValues.reserve(*externalValueCount);
  for (std::uint64_t index = 0; index < *externalValueCount; ++index) {
    auto reference = decodeInterfaceRef(reader);
    if (!reference)
      return reference.takeError();
    auto interfaceIndex =
        detail::resolveSystemInterfaceIndex(*context, *reference);
    if (!interfaceIndex)
      return interfaceIndex.takeError();
    if (!context->externalInterfaceShapes[*interfaceIndex])
      return detail::invalid("simulation runtime input: external value "
                             "interface has no lane shape");
    auto value = detail::decodeValueSequence(
        reader, *context->externalInterfaceShapes[*interfaceIndex]);
    if (!value)
      return value.takeError();
    input.runtimeExternalValues.push_back({*reference, std::move(*value)});
  }

  auto streamCount = reader.u64();
  if (!streamCount)
    return streamCount.takeError();
  if (llvm::Error error = reader.guardCount(
          *streamCount, deployment::deploymentCatalogReferenceWireSize + 24))
    return std::move(error);
  input.externalStreamInputs.reserve(*streamCount);
  for (std::uint64_t index = 0; index < *streamCount; ++index) {
    auto reference = decodeInterfaceRef(reader);
    if (!reference)
      return reference.takeError();
    auto interfaceIndex =
        detail::resolveSystemInterfaceIndex(*context, *reference);
    if (!interfaceIndex)
      return interfaceIndex.takeError();
    if (!context->externalInterfaceShapes[*interfaceIndex])
      return detail::invalid("simulation runtime input: external stream "
                             "interface has no lane shape");
    auto stream = detail::decodeStreamSequence(
        reader, *context->externalInterfaceShapes[*interfaceIndex]);
    if (!stream)
      return stream.takeError();
    input.externalStreamInputs.push_back({*reference, std::move(*stream)});
  }

  auto objectCount = reader.u64();
  if (!objectCount)
    return objectCount.takeError();
  if (llvm::Error error = reader.guardCount(*objectCount, 16))
    return std::move(error);
  input.memoryObjects.reserve(*objectCount);
  for (std::uint64_t index = 0; index < *objectCount; ++index) {
    auto object =
        detail::decodeMemoryObject(reader, context->layoutOperation());
    if (!object)
      return object.takeError();
    input.memoryObjects.push_back(std::move(*object));
  }

  auto bindingCount = reader.u64();
  if (!bindingCount)
    return bindingCount.takeError();
  if (llvm::Error error = reader.guardCount(
          *bindingCount, deployment::deploymentCatalogReferenceWireSize + 16))
    return std::move(error);
  input.memoryInterfaceBindings.reserve(*bindingCount);
  for (std::uint64_t index = 0; index < *bindingCount; ++index) {
    auto reference = decodeInterfaceRef(reader);
    if (!reference)
      return reference.takeError();
    auto objectOrdinal = reader.u64();
    if (!objectOrdinal)
      return objectOrdinal.takeError();
    auto byteOffset = reader.u64();
    if (!byteOffset)
      return byteOffset.takeError();
    input.memoryInterfaceBindings.push_back(
        {*reference, {*objectOrdinal, *byteOffset}});
  }
  if (!reader.atEnd())
    return detail::invalid("simulation runtime input: trailing bytes");
  return DecodedSystemRuntimeInput{std::move(input), std::move(*context)};
}

} // namespace

namespace detail {

llvm::Error
validateSystemRuntimeInput(const SystemSimulationRuntimeInput &input,
                           const SystemSimulationWorkload &workload,
                           const ArtifactIdentity &workloadIdentity,
                           const ResolvedSystemContext &context) {
  if (input.workloadIdentity != workloadIdentity)
    return invalid("simulation runtime input: does not name the exact "
                   "workload");
  if (llvm::Error error = validateRuntimeMemoryObjects(
          input.memoryObjects, context.layoutOperation()))
    return error;
  if (llvm::Error error =
          validateRuntimeEntryValues(input.runtimeEntryValues, workload,
                                     context, input.memoryObjects.size()))
    return error;
  for (const SystemRuntimeEntryValue &entry : input.runtimeEntryValues)
    if (llvm::Error error = validateCanonicalPointerValueSequence(
            entry.value,
            context.valueArgumentShapes[entry.valueArgumentOrdinal],
            input.memoryObjects, context.layoutOperation(),
            "simulation runtime input: program value"))
      return error;
  if (llvm::Error error =
          validateRuntimeExternalValues(input.runtimeExternalValues, workload,
                                        context, input.memoryObjects, true))
    return error;
  if (llvm::Error error = validateStreamInputs(
          input.externalStreamInputs, context, input.memoryObjects, true))
    return error;
  if (llvm::Error error =
          validateMemoryBindings(input.memoryInterfaceBindings,
                                 input.memoryObjects, workload, context))
    return error;
  auto canonical =
      deriveCanonicalObjectOrdinals(bindingKeys(input.memoryInterfaceBindings));
  if (!canonical)
    return canonical.takeError();
  for (const SystemMemoryInterfaceBindingEntry &entry :
       input.memoryInterfaceBindings)
    if (canonical->at(entry.binding.objectOrdinal) !=
        entry.binding.objectOrdinal)
      return invalid("simulation runtime input: object ordinals are not in "
                     "canonical interface-binding-key order");
  return llvm::Error::success();
}

llvm::Expected<SystemSimulationRuntimeInput>
canonicalizeSystemRuntimeInput(const SystemSimulationRuntimeInputDraft &draft,
                               const SystemSimulationWorkload &workload,
                               const ArtifactIdentity &workloadIdentity,
                               const ResolvedSystemContext &context) {
  if (draft.workloadIdentity != workloadIdentity)
    return invalid("simulation runtime input: does not name the exact "
                   "workload");
  SystemSimulationRuntimeInput input{draft.workloadIdentity};
  input.runtimeEntryValues = draft.runtimeEntryValues;
  std::sort(input.runtimeEntryValues.begin(), input.runtimeEntryValues.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.valueArgumentOrdinal < rhs.valueArgumentOrdinal;
            });
  input.runtimeExternalValues = draft.runtimeExternalValues;
  std::sort(input.runtimeExternalValues.begin(),
            input.runtimeExternalValues.end(),
            [](const auto &lhs, const auto &rhs) {
              return deployment::deploymentExternalInterfaceRefLess(
                  lhs.interfaceRef, rhs.interfaceRef);
            });
  input.externalStreamInputs = draft.externalStreamInputs;
  std::sort(input.externalStreamInputs.begin(),
            input.externalStreamInputs.end(),
            [](const auto &lhs, const auto &rhs) {
              return deployment::deploymentExternalInterfaceRefLess(
                  lhs.interfaceRef, rhs.interfaceRef);
            });
  if (llvm::Error error = validateRuntimeMemoryObjectStructure(
          draft.memoryObjects, context.layoutOperation()))
    return std::move(error);
  if (llvm::Error error =
          validateRuntimeEntryValues(input.runtimeEntryValues, workload,
                                     context, draft.memoryObjects.size()))
    return std::move(error);
  if (llvm::Error error =
          validateRuntimeExternalValues(input.runtimeExternalValues, workload,
                                        context, draft.memoryObjects, false))
    return std::move(error);
  if (llvm::Error error = validateStreamInputs(
          input.externalStreamInputs, context, draft.memoryObjects, false))
    return std::move(error);

  input.memoryInterfaceBindings.reserve(draft.memoryInterfaceBindings.size());
  for (const SystemMemoryInterfaceBindingDraft &binding :
       draft.memoryInterfaceBindings)
    input.memoryInterfaceBindings.push_back(
        {binding.interfaceRef, {binding.authorObject, binding.byteOffset}});
  std::sort(input.memoryInterfaceBindings.begin(),
            input.memoryInterfaceBindings.end(),
            [](const auto &lhs, const auto &rhs) {
              return deployment::deploymentExternalInterfaceRefLess(
                  lhs.interfaceRef, rhs.interfaceRef);
            });
  if (llvm::Error error =
          validateMemoryBindings(input.memoryInterfaceBindings,
                                 draft.memoryObjects, workload, context))
    return std::move(error);
  auto canonical =
      deriveCanonicalObjectOrdinals(bindingKeys(input.memoryInterfaceBindings));
  if (!canonical)
    return canonical.takeError();

  input.memoryObjects.resize(draft.memoryObjects.size());
  for (std::size_t author = 0; author < draft.memoryObjects.size(); ++author)
    input.memoryObjects[canonical->at(author)] = draft.memoryObjects[author];
  for (RuntimeMemoryObject &object : input.memoryObjects)
    for (RuntimeMemoryPointer &pointer : object.pointerValues)
      pointer.target.objectOrdinal =
          canonical->at(pointer.target.objectOrdinal);
  for (SystemMemoryInterfaceBindingEntry &entry : input.memoryInterfaceBindings)
    entry.binding.objectOrdinal = canonical->at(entry.binding.objectOrdinal);
  for (SystemRuntimeEntryValue &entry : input.runtimeEntryValues)
    remapPointerTargets(entry.value, *canonical);
  for (SystemRuntimeExternalValue &entry : input.runtimeExternalValues)
    remapPointerTargets(entry.value, *canonical);
  for (SystemExternalStreamInput &entry : input.externalStreamInputs)
    remapPointerTargets(entry.stream.values, *canonical);

  if (llvm::Error error = canonicalizeRuntimeMemoryPointers(
          input.memoryObjects, context.layoutOperation()))
    return std::move(error);
  if (llvm::Error error = canonicalizePointerValues(input, context))
    return std::move(error);
  return input;
}

} // namespace detail

llvm::Expected<CanonicalSimulationRuntimeInput> finalizeSimulationRuntimeInput(
    const SystemSimulationRuntimeInputDraft &draft,
    const CanonicalSimulationWorkload &workload,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &store) {
  const SystemSimulationWorkload *system = workload.system();
  if (!system)
    return detail::invalid("simulation runtime input: System finalization "
                           "requires a System workload root");
  auto context =
      detail::resolveSystemContext(deployment, system->programEntryRef, store);
  if (!context)
    return context.takeError();
  auto input = detail::canonicalizeSystemRuntimeInput(
      draft, *system, workload.identity(), *context);
  if (!input)
    return input.takeError();
  if (llvm::Error error = detail::validateSystemRuntimeInput(
          *input, *system, workload.identity(), *context))
    return std::move(error);
  CanonicalSemanticBytes bytes(encodeSystemRuntimeInput(*input, *context));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationRuntimeInputSchema, bytes);
  return CanonicalSimulationRuntimeInput(
      identity, SimulationRuntimeInputModel{std::move(*input)},
      std::move(bytes));
}

llvm::Expected<CanonicalSimulationRuntimeInput>
importSimulationRuntimeInput(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                             const CanonicalSimulationWorkload &workload,
                             const deployment::FinalizedDeployment &deployment,
                             const ArtifactStore &store,
                             const ArtifactIdentity &expectedIdentity) {
  const SystemSimulationWorkload *system = workload.system();
  if (!system)
    return detail::invalid("simulation runtime input: System import requires "
                           "a System workload root");
  auto decoded =
      decodeSystemRuntimeInput(canonicalBytes, *system, deployment, store);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = detail::validateSystemRuntimeInput(
          decoded->input, *system, workload.identity(), decoded->context))
    return std::move(error);
  const std::vector<std::uint8_t> reencoded =
      encodeSystemRuntimeInput(decoded->input, decoded->context);
  if (!llvm::ArrayRef<std::uint8_t>(reencoded).equals(canonicalBytes))
    return detail::invalid("simulation runtime input: noncanonical System "
                           "bytes do not re-encode exactly");
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(canonicalBytes.begin(), canonicalBytes.end()));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationRuntimeInputSchema, bytes);
  if (identity != expectedIdentity)
    return detail::invalid("simulation runtime input: identity does not match "
                           "the expected artifact");
  return CanonicalSimulationRuntimeInput(
      identity, SimulationRuntimeInputModel{std::move(decoded->input)},
      std::move(bytes));
}

} // namespace loom::sim
