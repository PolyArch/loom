//===- SystemSimulationWorkload.cpp - Deployment workload artifact -------===//

#include "SimulationWireInternal.h"

#include "Common/ArtifactFinalizer.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/DeploymentReference.h"
#include "Frontend/Executable/CompilerTargetBinding.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/IR/DataLayout.h"

#include <algorithm>
#include <utility>

namespace loom::sim {
namespace {

bool acceptsInput(deployment::HostExternalInterfaceDirection direction) {
  return direction == deployment::HostExternalInterfaceDirection::Input ||
         direction == deployment::HostExternalInterfaceDirection::InOut;
}

bool producesOutput(deployment::HostExternalInterfaceDirection direction) {
  return direction == deployment::HostExternalInterfaceDirection::Output ||
         direction == deployment::HostExternalInterfaceDirection::InOut;
}

mlir::MLIRContext &typeContext() {
  static thread_local mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<mlir::LLVM::LLVMDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *context;
}

llvm::Expected<mlir::Type> decodeType(llvm::ArrayRef<std::uint8_t> bytes,
                                      const llvm::Twine &what) {
  auto type = dataflow::decodeCanonicalType(bytes, &typeContext());
  if (!type)
    return detail::invalid(what + ": semantic type is not canonical: " +
                           llvm::toString(type.takeError()));
  return *type;
}

void encodeProgramEntryRef(
    detail::WireWriter &writer,
    const deployment::DeploymentProgramEntryRef &reference) {
  writer.bytes(deployment::encodeDeploymentProgramEntryRef(reference));
}

void encodeInterfaceRef(
    detail::WireWriter &writer,
    const deployment::DeploymentExternalInterfaceRef &reference) {
  writer.bytes(deployment::encodeDeploymentExternalInterfaceRef(reference));
}

llvm::Expected<deployment::DeploymentProgramEntryRef>
decodeProgramEntryRef(detail::WireReader &reader) {
  auto bytes = reader.bytes(deployment::deploymentCatalogReferenceWireSize);
  if (!bytes)
    return bytes.takeError();
  return deployment::decodeDeploymentProgramEntryRef(*bytes);
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

llvm::Error validateFixedSource(const SystemValueInputSource &source,
                                const detail::LaneShape &shape,
                                const llvm::Twine &what) {
  const auto *fixed = std::get_if<CanonicalValueSequence>(&source);
  if (!fixed)
    return llvm::Error::success();
  if (fixed->tokenCount != 1)
    return detail::invalid(what + ": a fixed value holds exactly one token");
  return detail::validateValueSequence(*fixed, shape, what);
}

llvm::Error validateInterfaceSet(
    llvm::ArrayRef<deployment::DeploymentExternalInterfaceRef> references,
    const detail::ResolvedSystemContext &context,
    deployment::HostExternalInterfaceKind kind, const llvm::Twine &what) {
  for (std::size_t index = 0; index < references.size(); ++index) {
    const auto &reference = references[index];
    if (index != 0 && detail::compareSystemInterfaceRefs(
                          reference, references[index - 1]) <= 0)
      return detail::invalid(what + ": references are not sorted or contain a "
                                    "duplicate");
    auto interface = detail::resolveSystemInterface(context, reference);
    if (!interface)
      return interface.takeError();
    if ((*interface)->kind != kind || !producesOutput((*interface)->direction))
      return detail::invalid(what +
                             ": interface kind or direction is not observable");
  }
  return llvm::Error::success();
}

std::vector<deployment::DeploymentExternalInterfaceRef>
expectedInputValueInterfaces(const detail::ResolvedSystemContext &context) {
  std::vector<deployment::DeploymentExternalInterfaceRef> result;
  for (const deployment::HostExternalInterface &interface : context.interfaces)
    if (interface.kind == deployment::HostExternalInterfaceKind::Value &&
        acceptsInput(interface.direction))
      result.push_back(
          {context.deploymentIdentity, interface.interfaceOrdinal});
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeSystemWorkload(const SystemSimulationWorkload &workload,
                     const detail::ResolvedSystemContext &context) {
  detail::WireWriter writer;
  writer.u32(static_cast<std::uint32_t>(SimulationWorkloadKind::System));
  encodeProgramEntryRef(writer, workload.programEntryRef);

  writer.u64(workload.valueInputPlan.size());
  for (std::uint64_t ordinal = 0; ordinal < workload.valueInputPlan.size();
       ++ordinal) {
    writer.u64(ordinal);
    const SystemValueInputSource &source = workload.valueInputPlan[ordinal];
    if (const auto *fixed = std::get_if<CanonicalValueSequence>(&source)) {
      writer.u32(0);
      detail::encodeValueSequence(writer, *fixed,
                                  context.valueArgumentShapes[ordinal]);
    } else {
      writer.u32(1);
    }
  }

  writer.u64(workload.externalValueInputPlan.size());
  for (const SystemExternalValueInputPlanEntry &entry :
       workload.externalValueInputPlan) {
    encodeInterfaceRef(writer, entry.interfaceRef);
    if (const auto *fixed =
            std::get_if<CanonicalValueSequence>(&entry.source)) {
      writer.u32(0);
      detail::encodeValueSequence(writer, *fixed,
                                  interfaceShape(context, entry.interfaceRef));
    } else {
      writer.u32(1);
    }
  }

  const SystemObservableContract &contract = workload.observableContract;
  writer.u64(contract.valueResults.size());
  for (std::uint64_t ordinal : contract.valueResults)
    writer.u64(ordinal);
  writer.u64(contract.externalValueOutputs.size());
  for (const auto &reference : contract.externalValueOutputs)
    encodeInterfaceRef(writer, reference);
  writer.u64(contract.externalStreamOutputs.size());
  for (const auto &reference : contract.externalStreamOutputs)
    encodeInterfaceRef(writer, reference);
  writer.u64(contract.memories.size());
  for (const SystemMemoryObservable &observable : contract.memories) {
    encodeInterfaceRef(writer, observable.interfaceRef);
    writer.u32(static_cast<std::uint32_t>(observable.form));
  }
  return writer.take();
}

llvm::Expected<std::vector<std::uint64_t>>
decodeOrdinalSet(detail::WireReader &reader, const llvm::Twine &what) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (llvm::Error error = reader.guardCount(*count, 8))
    return std::move(error);
  std::vector<std::uint64_t> ordinals;
  ordinals.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (index != 0 && *ordinal <= ordinals.back())
      return detail::invalid(what + ": ordinals are not sorted or contain a "
                                    "duplicate");
    ordinals.push_back(*ordinal);
  }
  return ordinals;
}

llvm::Expected<std::vector<deployment::DeploymentExternalInterfaceRef>>
decodeInterfaceSet(detail::WireReader &reader, const llvm::Twine &what) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (llvm::Error error = reader.guardCount(
          *count, deployment::deploymentCatalogReferenceWireSize))
    return std::move(error);
  std::vector<deployment::DeploymentExternalInterfaceRef> references;
  references.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto reference = decodeInterfaceRef(reader);
    if (!reference)
      return reference.takeError();
    if (index != 0 &&
        detail::compareSystemInterfaceRefs(*reference, references.back()) <= 0)
      return detail::invalid(what + ": references are not sorted or contain a "
                                    "duplicate");
    references.push_back(std::move(*reference));
  }
  return references;
}

struct DecodedSystemWorkload {
  SystemSimulationWorkload workload;
  detail::ResolvedSystemContext context;
};

llvm::Expected<DecodedSystemWorkload>
decodeSystemWorkload(llvm::ArrayRef<std::uint8_t> bytes,
                     const deployment::FinalizedDeployment &deployment,
                     const ArtifactStore &store) {
  detail::WireReader reader(bytes);
  auto root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root != static_cast<std::uint32_t>(SimulationWorkloadKind::System))
    return detail::invalid(
        "simulation workload: System import received a non-System root");
  auto entry = decodeProgramEntryRef(reader);
  if (!entry)
    return entry.takeError();
  SystemSimulationWorkload workload{*entry};
  auto context = detail::resolveSystemContext(deployment, *entry, store);
  if (!context)
    return context.takeError();

  auto planCount = reader.u64();
  if (!planCount)
    return planCount.takeError();
  if (llvm::Error error = reader.guardCount(*planCount, 12))
    return std::move(error);
  workload.valueInputPlan.reserve(*planCount);
  for (std::uint64_t index = 0; index < *planCount; ++index) {
    auto ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal != index || *ordinal >= context->valueArgumentShapes.size())
      return detail::invalid("simulation workload: System value-input plan "
                             "keys are not dense entry ordinals");
    auto tag = reader.u32();
    if (!tag)
      return tag.takeError();
    if (*tag == 0) {
      auto fixed = detail::decodeValueSequence(
          reader, context->valueArgumentShapes[*ordinal]);
      if (!fixed)
        return fixed.takeError();
      workload.valueInputPlan.emplace_back(std::move(*fixed));
    } else if (*tag == 1) {
      workload.valueInputPlan.emplace_back(RuntimeValueInput{});
    } else {
      return detail::invalid(
          "simulation workload: unknown System value-input source");
    }
  }

  auto externalCount = reader.u64();
  if (!externalCount)
    return externalCount.takeError();
  if (llvm::Error error = reader.guardCount(
          *externalCount, deployment::deploymentCatalogReferenceWireSize + 4))
    return std::move(error);
  workload.externalValueInputPlan.reserve(*externalCount);
  for (std::uint64_t index = 0; index < *externalCount; ++index) {
    auto reference = decodeInterfaceRef(reader);
    if (!reference)
      return reference.takeError();
    auto interfaceIndex =
        detail::resolveSystemInterfaceIndex(*context, *reference);
    if (!interfaceIndex)
      return interfaceIndex.takeError();
    if (!context->externalInterfaceShapes[*interfaceIndex])
      return detail::invalid("simulation workload: external value-input "
                             "interface has no value shape");
    auto tag = reader.u32();
    if (!tag)
      return tag.takeError();
    SystemValueInputSource source;
    if (*tag == 0) {
      auto fixed = detail::decodeValueSequence(
          reader, *context->externalInterfaceShapes[*interfaceIndex]);
      if (!fixed)
        return fixed.takeError();
      source = std::move(*fixed);
    } else if (*tag == 1) {
      source = RuntimeValueInput{};
    } else {
      return detail::invalid(
          "simulation workload: unknown external value-input source");
    }
    workload.externalValueInputPlan.push_back({*reference, std::move(source)});
  }

  auto valueResults =
      decodeOrdinalSet(reader, "simulation workload: System value results");
  if (!valueResults)
    return valueResults.takeError();
  workload.observableContract.valueResults = std::move(*valueResults);
  auto externalValues =
      decodeInterfaceSet(reader, "simulation workload: external value outputs");
  if (!externalValues)
    return externalValues.takeError();
  workload.observableContract.externalValueOutputs = std::move(*externalValues);
  auto externalStreams = decodeInterfaceSet(
      reader, "simulation workload: external stream outputs");
  if (!externalStreams)
    return externalStreams.takeError();
  workload.observableContract.externalStreamOutputs =
      std::move(*externalStreams);

  auto memoryCount = reader.u64();
  if (!memoryCount)
    return memoryCount.takeError();
  if (llvm::Error error = reader.guardCount(
          *memoryCount, deployment::deploymentCatalogReferenceWireSize + 4))
    return std::move(error);
  workload.observableContract.memories.reserve(*memoryCount);
  for (std::uint64_t index = 0; index < *memoryCount; ++index) {
    auto reference = decodeInterfaceRef(reader);
    if (!reference)
      return reference.takeError();
    auto form = reader.u32();
    if (!form)
      return form.takeError();
    if (*form >
        static_cast<std::uint32_t>(MemoryObservationForm::DiffFromRuntimeInput))
      return detail::invalid(
          "simulation workload: unknown System memory observation form");
    workload.observableContract.memories.push_back(
        {*reference, static_cast<MemoryObservationForm>(*form)});
  }
  if (!reader.atEnd())
    return detail::invalid("simulation workload: trailing bytes");
  return DecodedSystemWorkload{std::move(workload), std::move(*context)};
}

} // namespace

namespace detail {

int compareSystemInterfaceRefs(
    const deployment::DeploymentExternalInterfaceRef &lhs,
    const deployment::DeploymentExternalInterfaceRef &rhs) {
  if (deployment::deploymentExternalInterfaceRefLess(lhs, rhs))
    return -1;
  if (deployment::deploymentExternalInterfaceRefLess(rhs, lhs))
    return 1;
  return 0;
}

llvm::Expected<ResolvedSystemContext> resolveSystemContext(
    const deployment::FinalizedDeployment &deployment,
    const deployment::DeploymentProgramEntryRef &entryReference,
    const ArtifactStore &store) {
  auto entry =
      deployment::resolveDeploymentProgramEntry(deployment, entryReference);
  if (!entry)
    return entry.takeError();
  auto target = importCompilerTargetBinding(
      deployment.deployment().hostProgram().compilerTargetBinding(), store);
  if (!target)
    return target.takeError();
  auto dataLayout = llvm::DataLayout::parse(target->binding().dataLayout());
  if (!dataLayout)
    return dataLayout.takeError();

  mlir::MLIRContext &mlirContext = typeContext();
  mlir::OwningOpRef<mlir::ModuleOp> layoutScope(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext)));
  layoutScope.get()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&mlirContext, target->binding().dataLayout()));

  ResolvedSystemContext context{entryReference.deployment,
                                **entry,
                                {},
                                {},
                                {},
                                {},
                                dataLayout->isLittleEndian(),
                                std::move(layoutScope)};
  context.valueArgumentShapes.reserve((*entry)->valueArgumentTypes.size());
  for (const deployment::CanonicalTypeBytes &bytes :
       (*entry)->valueArgumentTypes) {
    auto type = decodeType(bytes, "simulation workload: program argument");
    if (!type)
      return type.takeError();
    auto shape = laneShapeOf(*type, context.layoutOperation());
    if (!shape)
      return shape.takeError();
    context.valueArgumentShapes.push_back(*shape);
  }
  context.valueResultShapes.reserve((*entry)->valueResultTypes.size());
  for (const deployment::CanonicalTypeBytes &bytes :
       (*entry)->valueResultTypes) {
    auto type = decodeType(bytes, "simulation workload: program result");
    if (!type)
      return type.takeError();
    auto shape = laneShapeOf(*type, context.layoutOperation());
    if (!shape)
      return shape.takeError();
    context.valueResultShapes.push_back(*shape);
  }

  llvm::ArrayRef<deployment::HostExternalInterface> catalog =
      deployment.deployment().hostProgram().externalInterfaces();
  context.interfaces.reserve((*entry)->externalInterfaceOrdinals.size());
  context.externalInterfaceShapes.reserve(
      (*entry)->externalInterfaceOrdinals.size());
  for (std::uint64_t ordinal : (*entry)->externalInterfaceOrdinals) {
    if (ordinal >= catalog.size() ||
        catalog[ordinal].interfaceOrdinal != ordinal)
      return invalid("simulation workload: Deployment external interface "
                     "catalog is not dense and canonical");
    context.interfaces.push_back(catalog[ordinal]);
    auto type = decodeType(catalog[ordinal].semanticType,
                           "simulation workload: external interface");
    if (!type)
      return type.takeError();
    if (catalog[ordinal].kind ==
        deployment::HostExternalInterfaceKind::Memory) {
      context.externalInterfaceShapes.push_back(std::nullopt);
      continue;
    }
    auto shape = laneShapeOf(*type, context.layoutOperation());
    if (!shape)
      return shape.takeError();
    context.externalInterfaceShapes.push_back(*shape);
  }
  return context;
}

llvm::Expected<std::size_t> resolveSystemInterfaceIndex(
    const ResolvedSystemContext &context,
    const deployment::DeploymentExternalInterfaceRef &reference) {
  if (reference.deployment != context.deploymentIdentity)
    return invalid("simulation workload: external interface names a foreign "
                   "Deployment");
  auto position =
      std::lower_bound(context.interfaces.begin(), context.interfaces.end(),
                       reference.externalInterfaceOrdinal,
                       [](const deployment::HostExternalInterface &interface,
                          std::uint64_t ordinal) {
                         return interface.interfaceOrdinal < ordinal;
                       });
  if (position == context.interfaces.end() ||
      position->interfaceOrdinal != reference.externalInterfaceOrdinal)
    return invalid("simulation workload: external interface is not selected "
                   "by the program entry");
  return static_cast<std::size_t>(position - context.interfaces.begin());
}

llvm::Expected<const deployment::HostExternalInterface *>
resolveSystemInterface(
    const ResolvedSystemContext &context,
    const deployment::DeploymentExternalInterfaceRef &reference) {
  auto index = resolveSystemInterfaceIndex(context, reference);
  if (!index)
    return index.takeError();
  return &context.interfaces[*index];
}

llvm::Error validateSystemWorkload(const SystemSimulationWorkload &workload,
                                   const ResolvedSystemContext &context) {
  if (workload.programEntryRef.deployment != context.deploymentIdentity ||
      workload.programEntryRef.programEntryOrdinal !=
          context.entry.entryOrdinal)
    return invalid("simulation workload: program entry does not match its "
                   "resolved Deployment context");
  if (workload.valueInputPlan.size() != context.valueArgumentShapes.size())
    return invalid("simulation workload: System value-input plan is not "
                   "total over program arguments");
  for (std::size_t ordinal = 0; ordinal < workload.valueInputPlan.size();
       ++ordinal)
    if (llvm::Error error =
            validateFixedSource(workload.valueInputPlan[ordinal],
                                context.valueArgumentShapes[ordinal],
                                "simulation workload: fixed program argument"))
      return error;

  const auto expectedInputs = expectedInputValueInterfaces(context);
  if (workload.externalValueInputPlan.size() != expectedInputs.size())
    return invalid("simulation workload: external value-input plan is not "
                   "total over input and inout value interfaces");
  for (std::size_t index = 0; index < expectedInputs.size(); ++index) {
    const SystemExternalValueInputPlanEntry &entry =
        workload.externalValueInputPlan[index];
    if (!(entry.interfaceRef == expectedInputs[index]))
      return invalid("simulation workload: external value-input plan is not "
                     "the canonical total interface table");
    if (llvm::Error error = validateFixedSource(
            entry.source, interfaceShape(context, entry.interfaceRef),
            "simulation workload: fixed external value"))
      return error;
  }

  const SystemObservableContract &contract = workload.observableContract;
  for (std::size_t index = 0; index < contract.valueResults.size(); ++index) {
    if (index != 0 &&
        contract.valueResults[index] <= contract.valueResults[index - 1])
      return invalid("simulation workload: System value results are not "
                     "sorted or contain a duplicate");
    if (contract.valueResults[index] >= context.valueResultShapes.size())
      return invalid(
          "simulation workload: System value-result ordinal is out of range");
  }
  if (llvm::Error error =
          validateInterfaceSet(contract.externalValueOutputs, context,
                               deployment::HostExternalInterfaceKind::Value,
                               "simulation workload: external value outputs"))
    return error;
  if (llvm::Error error =
          validateInterfaceSet(contract.externalStreamOutputs, context,
                               deployment::HostExternalInterfaceKind::Stream,
                               "simulation workload: external stream outputs"))
    return error;
  for (std::size_t index = 0; index < contract.memories.size(); ++index) {
    const SystemMemoryObservable &observable = contract.memories[index];
    if (index != 0 && compareSystemInterfaceRefs(
                          observable.interfaceRef,
                          contract.memories[index - 1].interfaceRef) <= 0)
      return invalid("simulation workload: System memory observables are not "
                     "sorted or contain a duplicate");
    if (static_cast<std::uint32_t>(observable.form) >
        static_cast<std::uint32_t>(MemoryObservationForm::DiffFromRuntimeInput))
      return invalid("simulation workload: System memory observation form is "
                     "out of domain");
    auto interface = resolveSystemInterface(context, observable.interfaceRef);
    if (!interface)
      return interface.takeError();
    if ((*interface)->kind != deployment::HostExternalInterfaceKind::Memory ||
        !producesOutput((*interface)->direction))
      return invalid("simulation workload: memory observable interface kind "
                     "or direction is not observable");
  }
  return llvm::Error::success();
}

llvm::Expected<ArtifactIdentity>
systemWorkloadOwnerIdentity(llvm::ArrayRef<std::uint8_t> bytes) {
  WireReader reader(bytes);
  auto root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root != static_cast<std::uint32_t>(SimulationWorkloadKind::System))
    return invalid("simulation workload: stored import requires a System root");
  auto entry = decodeProgramEntryRef(reader);
  if (!entry)
    return entry.takeError();
  return entry->deployment;
}

} // namespace detail

llvm::Expected<SystemSimulationBoundaryShapes>
projectSystemSimulationBoundaryShapes(
    const deployment::FinalizedDeployment &deployment,
    const deployment::DeploymentProgramEntryRef &entry,
    const ArtifactStore &store) {
  auto context = detail::resolveSystemContext(deployment, entry, store);
  if (!context)
    return context.takeError();

  const auto project = [](llvm::ArrayRef<detail::LaneShape> shapes) {
    std::vector<SystemSimulationValueShape> result;
    result.reserve(shapes.size());
    for (const detail::LaneShape &shape : shapes)
      result.push_back({shape.lanesPerToken, shape.laneBitWidth,
                        shape.pointerLayout.has_value()});
    return result;
  };
  return SystemSimulationBoundaryShapes{context->littleEndian,
                                        project(context->valueArgumentShapes),
                                        project(context->valueResultShapes)};
}

llvm::Expected<CanonicalSimulationWorkload>
finalizeSimulationWorkload(const SystemSimulationWorkload &workload,
                           const deployment::FinalizedDeployment &deployment,
                           const ArtifactStore &store) {
  auto context =
      detail::resolveSystemContext(deployment, workload.programEntryRef, store);
  if (!context)
    return context.takeError();
  if (llvm::Error error = detail::validateSystemWorkload(workload, *context))
    return std::move(error);
  auto encoded = encodeSystemWorkload(workload, *context);
  if (!encoded)
    return encoded.takeError();
  CanonicalSemanticBytes bytes(std::move(*encoded));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationWorkloadSchema, bytes);
  return CanonicalSimulationWorkload(
      identity, SimulationWorkloadModel{workload}, std::move(bytes));
}

llvm::Expected<CanonicalSimulationWorkload>
importSimulationWorkload(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                         const deployment::FinalizedDeployment &deployment,
                         const ArtifactStore &store,
                         const ArtifactIdentity &expectedIdentity) {
  auto decoded = decodeSystemWorkload(canonicalBytes, deployment, store);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error =
          detail::validateSystemWorkload(decoded->workload, decoded->context))
    return std::move(error);
  auto reencoded = encodeSystemWorkload(decoded->workload, decoded->context);
  if (!reencoded)
    return reencoded.takeError();
  if (!llvm::ArrayRef<std::uint8_t>(*reencoded).equals(canonicalBytes))
    return detail::invalid("simulation workload: noncanonical System bytes "
                           "do not re-encode exactly");
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(canonicalBytes.begin(), canonicalBytes.end()));
  ArtifactIdentity identity =
      finalizeArtifactIdentity(simulationWorkloadSchema, bytes);
  if (identity != expectedIdentity)
    return detail::invalid("simulation workload: identity does not match the "
                           "expected artifact");
  return CanonicalSimulationWorkload(
      identity, SimulationWorkloadModel{std::move(decoded->workload)},
      std::move(bytes));
}

} // namespace loom::sim
