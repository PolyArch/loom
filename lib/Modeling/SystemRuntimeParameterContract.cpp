#include "Evaluation/Models/SystemRuntimeParameterContract.h"

#include "FixedTabularGbdt.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/Deployment.h"
#include "Deployment/Package.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ProductionRegistry.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/RuntimePlatformBinding.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr std::uint32_t kIntegralFeatureCount = 35;
constexpr std::uint32_t kDecimalFeatureCount = 0;
constexpr std::uint32_t kCategoricalFeatureCount = 4;
constexpr std::uint32_t kPresenceFeatureCount = 1;
constexpr std::uint32_t kTargetCount = 1;

constexpr llvm::StringLiteral kParameterSchema =
    "loom.system_runtime.gbdt_parameter_payload.1.0";
constexpr llvm::StringLiteral kRuntimeTimingContract =
    "gem5_event_queue.tick_delta_seconds.1e-12.v1";
constexpr llvm::StringLiteral kRuntimeFidelity =
    "gem5_system_cgra.mapped_cgra.v1";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_runtime_parameter_contract_invalid: " +
                                     message);
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

void appendI64(std::vector<std::uint8_t> &bytes, std::int64_t value) {
  appendU64(bytes, static_cast<std::uint64_t>(value));
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

void appendApInt(std::vector<std::uint8_t> &bytes, const llvm::APInt &value) {
  appendU32(bytes, value.getBitWidth());
  const std::size_t byteCount = (value.getBitWidth() + 7U) / 8U;
  appendU64(bytes, byteCount);
  std::vector<std::uint8_t> encoded(byteCount);
  for (std::size_t index = 0; index != byteCount; ++index) {
    const unsigned bitOffset = static_cast<unsigned>(index * 8U);
    const unsigned bitCount =
        std::min<unsigned>(8U, value.getBitWidth() - bitOffset);
    encoded[byteCount - index - 1] = static_cast<std::uint8_t>(
        value.extractBitsAsZExtValue(bitCount, bitOffset));
  }
  bytes.insert(bytes.end(), encoded.begin(), encoded.end());
}

void appendLane(std::vector<std::uint8_t> &bytes,
                const sim::SemanticLane &lane) {
  appendU32(bytes, static_cast<std::uint32_t>(lane.state));
  if (lane.state != sim::SemanticState::Defined)
    return;
  appendApInt(bytes, lane.bits);
  appendU32(bytes, lane.pointerTarget ? 1U : 0U);
  if (!lane.pointerTarget)
    return;
  appendU64(bytes, lane.pointerTarget->objectOrdinal);
  appendApInt(bytes, lane.pointerTarget->byteOffset);
}

void appendValueSequence(std::vector<std::uint8_t> &bytes,
                         const sim::CanonicalValueSequence &sequence) {
  appendU64(bytes, sequence.tokenCount);
  appendU64(bytes, sequence.lanes.size());
  for (const sim::SemanticLane &lane : sequence.lanes)
    appendLane(bytes, lane);
}

void appendStreamSequence(std::vector<std::uint8_t> &bytes,
                          const sim::CanonicalStreamSequence &stream) {
  appendValueSequence(bytes, stream.values);
  appendU32(bytes, static_cast<std::uint32_t>(stream.termination));
}

void appendMemoryObject(std::vector<std::uint8_t> &bytes,
                        const sim::RuntimeMemoryObject &object) {
  appendU64(bytes, object.initialBytes.size());
  for (const sim::SemanticMemoryByte &byte : object.initialBytes) {
    appendU32(bytes, static_cast<std::uint32_t>(byte.state));
    if (byte.state == sim::SemanticState::Defined)
      bytes.push_back(byte.value);
  }
  appendU64(bytes, object.pointerValues.size());
  for (const sim::RuntimeMemoryPointer &pointer : object.pointerValues) {
    appendU64(bytes, pointer.storageByteOffset);
    appendU32(bytes, pointer.addressSpace);
    appendU64(bytes, pointer.target.objectOrdinal);
    appendApInt(bytes, pointer.target.byteOffset);
  }
}

void appendPresburgerCell(std::vector<std::uint8_t> &bytes,
                          const mapping::SystemPresburgerCell &cell) {
  appendU32(bytes, cell.dimensionCount);
  appendU32(bytes, cell.symbolCount);
  appendU32(bytes, cell.localCount);
  const auto appendRows = [&](llvm::ArrayRef<std::vector<std::int64_t>> rows) {
    appendU64(bytes, rows.size());
    for (const std::vector<std::int64_t> &row : rows) {
      appendU64(bytes, row.size());
      for (std::int64_t coefficient : row)
        appendI64(bytes, coefficient);
    }
  };
  appendRows(cell.equalities);
  appendRows(cell.inequalities);
}

llvm::Expected<SystemRuntimeFabricFeatureView>
summarizeFabric(const fabric::FinalizedFabricRoot &root) {
  SystemRuntimeFabricFeatureView summary;
  const auto summarize = [&](const fabric::FabricArtifactView &view,
                             bool systemRoot) -> llvm::Error {
    for (fabric::FabricEntityId id = 0;; ++id) {
      const std::optional<fabric::FabricEntityKind> kind = view.entityKind(id);
      if (!kind)
        break;
      if (llvm::Error error = add(summary.entityCount, 1, "entity count"))
        return error;
      if (*kind == fabric::FabricEntityKind::SystemTransportResource)
        if (llvm::Error error = add(summary.systemTransportResourceCount, 1,
                                    "System transport-resource count"))
          return error;
      if (*kind == fabric::FabricEntityKind::HardwareDomain)
        if (llvm::Error error =
                add(summary.hardwareDomainCount, 1, "hardware-domain count"))
          return error;
    }
    if (llvm::Error error =
            add(summary.pointConnectionCount, view.pointConnections().size(),
                "point-connection count"))
      return error;
    if (llvm::Error error =
            add(summary.admittedTraversalCount,
                view.admittedTraversals().size(), "admitted-traversal count"))
      return error;
    if (!systemRoot)
      return llvm::Error::success();
    if (llvm::Error error = add(summary.hostCoreOccurrenceCount,
                                view.hostCoreOccurrences().size(),
                                "host-core occurrence count"))
      return error;
    if (llvm::Error error = add(summary.acceleratorCoreOccurrenceCount,
                                view.accCoreOccurrences().size(),
                                "accelerator-core occurrence count"))
      return error;
    if (llvm::Error error = add(summary.systemMemoryServiceCount,
                                view.systemMemoryServices().size(),
                                "System memory-service count"))
      return error;
    if (llvm::Error error =
            add(summary.transportEndpointCount,
                view.transportEndpoints().size(), "transport-endpoint count"))
      return error;
    return llvm::Error::success();
  };
  if (llvm::Error error = summarize(root.view(), true))
    return std::move(error);
  for (const fabric::FabricArtifactView &module : root.view().importedModules())
    if (llvm::Error error = summarize(module, false))
      return std::move(error);
  return summary;
}

std::vector<std::uint8_t> softwarePartitioningKey(
    const mapping::SystemExecutionContextProjection &projection) {
  std::vector<std::uint8_t> key;
  appendU64(key, projection.instructionDomains.size());
  for (const mapping::SystemInstructionContextDomain &domain :
       projection.instructionDomains) {
    appendU64(key, domain.root.entity.value());
    appendU64(key, domain.context.accCore.id());
    appendU64(key, domain.cells.size());
    for (const mapping::SystemPresburgerCell &cell : domain.cells)
      appendPresburgerCell(key, cell);
  }
  appendU64(key, projection.spatialDomains.size());
  for (const mapping::SystemSpatialContextDomain &domain :
       projection.spatialDomains) {
    appendU64(key, domain.graph.rootThreadLaunch.entity.value());
    appendU64(key, domain.graph.staticGraphLaunch.entity.value());
    appendU64(key, domain.context.accCore.id());
    appendU64(key, domain.cells.size());
    for (const mapping::SystemPresburgerCell &cell : domain.cells)
      appendPresburgerCell(key, cell);
  }
  return key;
}

void appendGem5Object(std::vector<std::uint8_t> &key,
                      const runtime::Gem5SimObjectRef &object) {
  appendFramed(key, object.contract.identity);
  appendU32(key, object.contract.version.major);
  appendU32(key, object.contract.version.minor);
  appendFramed(key, object.payload);
}

void appendGem5Port(std::vector<std::uint8_t> &key,
                    const runtime::Gem5SimPortRef &port) {
  appendGem5Object(key, port.object);
  appendU32(key, port.kind);
  appendFramed(key, port.payload);
}

template <typename Ref>
void appendFabricRef(std::vector<std::uint8_t> &key, const Ref &reference) {
  appendFramed(key, fabric::canonicalFabricBytes(reference));
}

std::vector<std::uint8_t>
gem5FeatureKey(const runtime::Gem5SimulationBinding &binding) {
  std::vector<std::uint8_t> key;
  appendU64(key, binding.correspondences().size());
  for (const runtime::Gem5Correspondence &correspondence :
       binding.correspondences()) {
    appendU32(key, correspondence.index());
    if (const auto *processor =
            std::get_if<runtime::Gem5ProcessorCorrespondence>(
                &correspondence)) {
      appendU32(key, processor->processor.index());
      std::visit([&](const auto &ref) { appendFabricRef(key, ref); },
                 processor->processor);
      appendGem5Object(key, processor->simObject);
    } else if (const auto *bridge =
                   std::get_if<runtime::Gem5SpatialBridgeCorrespondence>(
                       &correspondence)) {
      appendFabricRef(key, bridge->spatialCore);
      appendFramed(key, fabric::encodeFabricSpatialAttachmentEndpointRef(
                            bridge->spatialBoundary));
      appendGem5Port(key, bridge->bridgeEndpoint);
    } else if (const auto *memory =
                   std::get_if<runtime::Gem5MemoryOrServiceCorrespondence>(
                       &correspondence)) {
      appendU32(key, memory->fabricRef.index());
      std::visit([&](const auto &ref) { appendFabricRef(key, ref); },
                 memory->fabricRef);
      appendGem5Object(key, memory->simObject);
      appendGem5Port(key, memory->simPort);
    } else if (const auto *transport =
                   std::get_if<runtime::Gem5TransportCorrespondence>(
                       &correspondence)) {
      appendU32(key, transport->fabricRef.index());
      std::visit([&](const auto &ref) { appendFabricRef(key, ref); },
                 transport->fabricRef);
      appendGem5Object(key, transport->simObject);
      appendGem5Port(key, transport->simPort);
    } else {
      const auto &external =
          std::get<runtime::Gem5ExternalEndpointCorrespondence>(correspondence);
      appendFabricRef(key, external.fabricRef);
      appendGem5Object(key, external.simObject);
      appendGem5Port(key, external.simPort);
    }
  }
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
modeledPlatformKey(const deployment::Deployment &deployment,
                   const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::vector<std::uint8_t> key;
  appendU64(key, deployment.hardwareBindings().size());
  for (const deployment::DeploymentHardwareBinding &binding :
       deployment.hardwareBindings()) {
    auto implementation = hardware::importHardwareImplementation(
        binding.hardwareImplementation, artifacts, blobs);
    if (!implementation)
      return implementation.takeError();
    auto runtimeBinding = runtime::importRuntimePlatformBinding(
        binding.runtimePlatformBinding, artifacts, blobs);
    if (!runtimeBinding)
      return runtimeBinding.takeError();
    const auto &representation =
        implementation->implementation().representationRoot();
    appendU32(key, static_cast<std::uint32_t>(representation.variant));
    appendU32(key, representation.stage ? 1U : 0U);
    if (representation.stage)
      appendU32(key, static_cast<std::uint32_t>(*representation.stage));
    const auto &platform =
        implementation->implementation().implementationPlatform();
    appendU32(key, platform ? 1U : 0U);
    if (platform)
      appendFramed(key, encodeArtifactRootReference(*platform));
    appendU64(key, implementation->implementation()
                       .externalImplementationBindings()
                       .size());
    for (const hardware::ExternalImplementationBinding &external :
         implementation->implementation().externalImplementationBindings())
      appendFramed(key, external.providerContractRef);
    const runtime::RuntimeProviderBinding &provider =
        runtimeBinding->binding().providerBinding();
    appendFramed(key, provider.descriptor.identity);
    appendU32(key, provider.descriptor.version.major);
    appendU32(key, provider.descriptor.version.minor);
    appendFramed(key, provider.implementationSemanticIdentity);
    appendFramed(key, provider.runtimeAbiIdentity);
    appendU64(key, runtimeBinding->binding().programmingBindings().size());
    appendU64(key, runtimeBinding->binding().memoryInterfaceBindings().size());
    appendU64(key,
              runtimeBinding->binding().completionInterfaceBindings().size());
  }
  return key;
}

struct ImportedSystemProjection final {
  sim::ImportedSystemSimulationInputs inputs;
  runtime::FinalizedGem5SimulationBinding gem5;
  mapping::FinalizedSystemMapping mapping;
  dataflow::CanonicalDataflowArtifact dataflow;
  fabric::FinalizedFabricRoot fabric;
  mapping::SystemMappingClosureProjection closure;
};

llvm::Expected<ImportedSystemProjection>
importSystemProjection(const EvaluationCase &evaluationCase,
                       const CaseArtifactResolution &resolution,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (evaluationCase.signature() != systemSimulationCaseSignatureRef())
    return invalid("feature projector received a foreign Evaluation case");
  const auto deployments =
      evaluationCase.subjectBindings().subjects(CaseSubjectRoleRef(0));
  const auto bindings =
      evaluationCase.subjectBindings().subjects(CaseSubjectRoleRef(1));
  if (deployments.size() != 1 || bindings.size() != 1 ||
      !evaluationCase.workload() || !evaluationCase.runtimeInput())
    return invalid("System Runtime case inputs are not total");
  for (const ArtifactRootReference &reference :
       {deployments.front(), bindings.front(), *evaluationCase.workload(),
        *evaluationCase.runtimeInput()})
    if (!resolution.find(reference))
      return invalid("System Runtime case input is unresolved");

  auto inputs = sim::importSystemSimulationInputs(
      *evaluationCase.workload(), *evaluationCase.runtimeInput(), artifacts,
      blobs);
  if (!inputs)
    return inputs.takeError();
  if (inputs->deployment.reference() != deployments.front())
    return invalid("System workload names a foreign Deployment");
  auto gem5 = runtime::importGem5SimulationBinding(bindings.front(), artifacts);
  if (!gem5)
    return gem5.takeError();
  const ArtifactRootReference &mappingReference =
      inputs->deployment.deployment().systemMapping();
  if (!resolution.find(mappingReference))
    return invalid("SystemMapping is absent from CaseArtifactResolution");
  auto mapping = mapping::importSystemMapping(mappingReference, artifacts);
  if (!mapping)
    return mapping.takeError();
  ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mapping->view().dataflowIdentity()};
  ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version, mapping->view().fabricIdentity()};
  if (!resolution.find(dataflowReference) || !resolution.find(fabricReference))
    return invalid("SystemMapping owner closure is unresolved");
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto fabric = fabric::importEntireFabricRoot(fabricReference, artifacts);
  if (!fabric)
    return fabric.takeError();
  auto system = fabric::requireSystemRoot(fabric->view());
  if (!system)
    return system.takeError();
  if (gem5->binding().fabric() != fabricReference)
    return invalid("Gem5SimulationBinding names a foreign Fabric System");
  auto closure = mapping::projectSystemMappingClosure(
      *dataflowView, *system, mapping->view(), artifacts);
  if (!closure)
    return closure.takeError();
  return ImportedSystemProjection{std::move(*inputs),  std::move(*gem5),
                                  std::move(*mapping), std::move(*dataflow),
                                  std::move(*fabric),  std::move(*closure)};
}

llvm::Expected<SystemRuntimeFeatureView>
projectFeatureView(const EvaluationCase &evaluationCase,
                   const CaseArtifactResolution &resolution,
                   const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto projection =
      importSystemProjection(evaluationCase, resolution, artifacts, blobs);
  if (!projection)
    return projection.takeError();
  const deployment::Deployment &deployment =
      projection->inputs.deployment.deployment();
  const sim::SystemSimulationWorkload &workload =
      *projection->inputs.workload.system();
  const sim::SystemSimulationRuntimeInput &runtimeInput =
      *projection->inputs.runtimeInput.system();

  SystemRuntimeFeatureView result;
  result.deployment = {deployment.instructionCoreBinaries().size(),
                       deployment.hardwareBindings().size(),
                       deployment.configurationImages().size(),
                       deployment.staticMemoryImages().size(),
                       deployment.spatialLaunchImage().has_value()};
  result.workload.entryValueInputCount = workload.valueInputPlan.size();
  result.workload.runtimeEntryValueInputCount = llvm::count_if(
      workload.valueInputPlan, [](const sim::SystemValueInputSource &source) {
        return std::holds_alternative<sim::RuntimeValueInput>(source);
      });
  result.workload.externalValueInputCount =
      workload.externalValueInputPlan.size();
  result.workload.runtimeExternalValueInputCount = llvm::count_if(
      workload.externalValueInputPlan,
      [](const sim::SystemExternalValueInputPlanEntry &entry) {
        return std::holds_alternative<sim::RuntimeValueInput>(entry.source);
      });
  result.workload.valueResultCount =
      workload.observableContract.valueResults.size();
  result.workload.externalValueOutputCount =
      workload.observableContract.externalValueOutputs.size();
  result.workload.externalStreamOutputCount =
      workload.observableContract.externalStreamOutputs.size();
  result.workload.memoryObservableCount =
      workload.observableContract.memories.size();
  result.runtimeInput.runtimeEntryValueCount =
      runtimeInput.runtimeEntryValues.size();
  result.runtimeInput.runtimeExternalValueCount =
      runtimeInput.runtimeExternalValues.size();
  result.runtimeInput.externalStreamInputCount =
      runtimeInput.externalStreamInputs.size();
  result.runtimeInput.memoryObjectCount = runtimeInput.memoryObjects.size();
  result.runtimeInput.memoryInterfaceBindingCount =
      runtimeInput.memoryInterfaceBindings.size();
  for (const sim::RuntimeMemoryObject &object : runtimeInput.memoryObjects)
    if (llvm::Error error =
            add(result.runtimeInput.memoryByteCount, object.initialBytes.size(),
                "runtime memory-byte count"))
      return std::move(error);
  for (const sim::SystemExternalStreamInput &stream :
       runtimeInput.externalStreamInputs)
    if (llvm::Error error =
            add(result.runtimeInput.streamTokenCount,
                stream.stream.values.tokenCount, "runtime stream-token count"))
      return std::move(error);

  const auto &closure = projection->closure;
  result.mapping.instructionContextDomainCount =
      closure.executionContexts.instructionDomains.size();
  result.mapping.spatialContextDomainCount =
      closure.executionContexts.spatialDomains.size();
  result.mapping.serviceRealizationCount = closure.serviceRealizations.size();
  result.mapping.capacityCellCount = closure.capacityCells.size();
  result.mapping.resourceActivationCount = closure.resourceActivations.size();
  for (const mapping::SystemResourceActivationProjection &activation :
       closure.resourceActivations) {
    if (llvm::Error error =
            add(result.mapping.capacityClaimCount,
                activation.capacityClaims.size(), "capacity-claim count"))
      return std::move(error);
    if (llvm::Error error =
            add(result.mapping.causalReleaseCount,
                activation.causalRelease.size(), "causal-release count"))
      return std::move(error);
  }
  auto fabricSummary = summarizeFabric(projection->fabric);
  if (!fabricSummary)
    return fabricSummary.takeError();
  result.fabric = std::move(*fabricSummary);
  result.softwarePartitioningKey =
      softwarePartitioningKey(closure.executionContexts);
  auto platform = modeledPlatformKey(deployment, artifacts, blobs);
  if (!platform)
    return platform.takeError();
  result.modeledPlatformKey = std::move(*platform);
  result.gem5BindingFeatureKey = gem5FeatureKey(projection->gem5.binding());
  const std::string conditions =
      serializeEvaluationConditions(evaluationCase.baseConditions());
  result.admittedRuntimeConditionKey.assign(conditions.begin(),
                                            conditions.end());
  return result;
}

llvm::Expected<CaseArtifactResolution>
resolveGroundTruthRequest(const ArtifactRootReference &requestReference,
                          const ArtifactStore &artifactStore,
                          const BlobStore &blobStore) {
  auto direct = importEvaluationRequestArtifactReferences(requestReference,
                                                          artifactStore);
  if (!direct)
    return direct.takeError();

  std::optional<ArtifactRootReference> deploymentReference;
  std::optional<ArtifactRootReference> gem5Reference;
  std::optional<ArtifactRootReference> workloadReference;
  std::optional<ArtifactRootReference> runtimeInputReference;
  for (const ArtifactRootReference &reference : *direct) {
    if (reference.schemaIdentity == deployment::deploymentSchema.identity &&
        reference.schemaVersion == deployment::deploymentSchema.version) {
      if (deploymentReference)
        return invalid("ground-truth Request names multiple Deployments");
      deploymentReference = reference;
    } else if (reference.schemaIdentity ==
                   runtime::gem5SimulationBindingSchema.identity &&
               reference.schemaVersion ==
                   runtime::gem5SimulationBindingSchema.version) {
      if (gem5Reference)
        return invalid(
            "ground-truth Request names multiple Gem5SimulationBindings");
      gem5Reference = reference;
    } else if (reference.schemaIdentity ==
                   sim::simulationWorkloadSchema.identity &&
               reference.schemaVersion ==
                   sim::simulationWorkloadSchema.version) {
      if (workloadReference)
        return invalid("ground-truth Request names multiple workloads");
      workloadReference = reference;
    } else if (reference.schemaIdentity ==
                   sim::simulationRuntimeInputSchema.identity &&
               reference.schemaVersion ==
                   sim::simulationRuntimeInputSchema.version) {
      if (runtimeInputReference)
        return invalid("ground-truth Request names multiple runtime inputs");
      runtimeInputReference = reference;
    }
  }
  if (!deploymentReference || !gem5Reference || !workloadReference ||
      !runtimeInputReference)
    return invalid("ground-truth Request does not name the complete System "
                   "simulation input set");

  auto inputs = sim::importSystemSimulationInputs(
      *workloadReference, *runtimeInputReference, artifactStore, blobStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->deployment.reference() != *deploymentReference)
    return invalid("ground-truth workload names a foreign Deployment");
  auto gem5 =
      runtime::importGem5SimulationBinding(*gem5Reference, artifactStore);
  if (!gem5)
    return gem5.takeError();
  auto package = deployment::deriveDeploymentPackageClosure(
      inputs->deployment, artifactStore, blobStore);
  if (!package)
    return package.takeError();

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
  for (const ArtifactRootReference &reference : package->artifacts())
    entries.emplace(reference, std::vector<ArtifactRootReference>{});
  for (const ArtifactRootReference &reference : *direct)
    entries.emplace(reference, std::vector<ArtifactRootReference>{});

  std::vector<ArtifactRootReference> deploymentClosure;
  deploymentClosure.reserve(package->artifacts().size());
  for (const ArtifactRootReference &reference : package->artifacts())
    if (reference != *deploymentReference)
      deploymentClosure.push_back(reference);
  merge(*deploymentReference, deploymentClosure);
  merge(*workloadReference, package->artifacts());
  std::vector<ArtifactRootReference> runtimeClosure =
      package->artifacts().vec();
  runtimeClosure.push_back(*workloadReference);
  merge(*runtimeInputReference, runtimeClosure);

  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      fabricClosure(&artifactRootReferenceLess);
  std::function<llvm::Error(const ArtifactRootReference &)> addFabric =
      [&](const ArtifactRootReference &reference) -> llvm::Error {
    if (!fabricClosure.insert(reference).second)
      return llvm::Error::success();
    auto root = fabric::importEntireFabricRoot(reference, artifactStore);
    if (!root)
      return root.takeError();
    entries.emplace(reference, std::vector<ArtifactRootReference>{});
    for (const fabric::FabricDirectDependency &dependency :
         root->directDependencies())
      if (llvm::Error error = addFabric(dependency.root))
        return error;
    return llvm::Error::success();
  };
  if (llvm::Error error = addFabric(gem5->binding().fabric()))
    return std::move(error);
  if (llvm::Error error =
          addFabric(gem5->binding().interconnectImplementation()))
    return std::move(error);
  std::vector<ArtifactRootReference> gem5Closure(fabricClosure.begin(),
                                                 fabricClosure.end());
  merge(*gem5Reference, gem5Closure);

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

void appendDigest(std::vector<std::uint8_t> &bytes, const BlobDigest &digest) {
  bytes.insert(bytes.end(), digest.bytes().begin(), digest.bytes().end());
}

void appendValueSource(std::vector<std::uint8_t> &bytes,
                       const sim::SystemValueInputSource &source) {
  appendU32(bytes, source.index());
  if (const auto *fixed = std::get_if<sim::CanonicalValueSequence>(&source))
    appendValueSequence(bytes, *fixed);
}

void appendInterfaceOrdinal(
    std::vector<std::uint8_t> &bytes,
    const deployment::DeploymentExternalInterfaceRef &reference) {
  appendU64(bytes, reference.externalInterfaceOrdinal);
}

llvm::Expected<std::vector<std::uint8_t>>
sourceInputKey(const EvaluationCase &evaluationCase,
               const CaseArtifactResolution &resolution,
               const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto projection = importSystemProjection(evaluationCase, resolution,
                                           artifactStore, blobStore);
  if (!projection)
    return projection.takeError();
  const deployment::Deployment &deployment =
      projection->inputs.deployment.deployment();
  const deployment::HostProgramLeaf &program = deployment.hostProgram();
  const sim::SystemSimulationWorkload &workload =
      *projection->inputs.workload.system();
  const sim::SystemSimulationRuntimeInput &runtimeInput =
      *projection->inputs.runtimeInput.system();

  std::vector<std::uint8_t> key;
  appendDigest(key, program.programBlob());
  appendDigest(key, program.registrationTableDigest());
  appendU64(key, workload.programEntryRef.programEntryOrdinal);

  appendU64(key, workload.valueInputPlan.size());
  for (const sim::SystemValueInputSource &source : workload.valueInputPlan)
    appendValueSource(key, source);
  appendU64(key, workload.externalValueInputPlan.size());
  for (const sim::SystemExternalValueInputPlanEntry &entry :
       workload.externalValueInputPlan) {
    appendInterfaceOrdinal(key, entry.interfaceRef);
    appendValueSource(key, entry.source);
  }
  appendU64(key, workload.observableContract.valueResults.size());
  for (std::uint64_t ordinal : workload.observableContract.valueResults)
    appendU64(key, ordinal);
  appendU64(key, workload.observableContract.externalValueOutputs.size());
  for (const deployment::DeploymentExternalInterfaceRef &reference :
       workload.observableContract.externalValueOutputs)
    appendInterfaceOrdinal(key, reference);
  appendU64(key, workload.observableContract.externalStreamOutputs.size());
  for (const deployment::DeploymentExternalInterfaceRef &reference :
       workload.observableContract.externalStreamOutputs)
    appendInterfaceOrdinal(key, reference);
  appendU64(key, workload.observableContract.memories.size());
  for (const sim::SystemMemoryObservable &memory :
       workload.observableContract.memories) {
    appendInterfaceOrdinal(key, memory.interfaceRef);
    appendU32(key, static_cast<std::uint32_t>(memory.form));
  }

  appendU64(key, runtimeInput.runtimeEntryValues.size());
  for (const sim::SystemRuntimeEntryValue &entry :
       runtimeInput.runtimeEntryValues) {
    appendU64(key, entry.valueArgumentOrdinal);
    appendValueSequence(key, entry.value);
  }
  appendU64(key, runtimeInput.runtimeExternalValues.size());
  for (const sim::SystemRuntimeExternalValue &entry :
       runtimeInput.runtimeExternalValues) {
    appendInterfaceOrdinal(key, entry.interfaceRef);
    appendValueSequence(key, entry.value);
  }
  appendU64(key, runtimeInput.externalStreamInputs.size());
  for (const sim::SystemExternalStreamInput &entry :
       runtimeInput.externalStreamInputs) {
    appendInterfaceOrdinal(key, entry.interfaceRef);
    appendStreamSequence(key, entry.stream);
  }
  appendU64(key, runtimeInput.memoryObjects.size());
  for (const sim::RuntimeMemoryObject &object : runtimeInput.memoryObjects)
    appendMemoryObject(key, object);
  appendU64(key, runtimeInput.memoryInterfaceBindings.size());
  for (const sim::SystemMemoryInterfaceBindingEntry &entry :
       runtimeInput.memoryInterfaceBindings) {
    appendInterfaceOrdinal(key, entry.interfaceRef);
    appendU64(key, entry.binding.objectOrdinal);
    appendU64(key, entry.binding.byteOffset);
  }
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
targetKey(const EvaluationRequest &request,
          const CaseArtifactResolution &resolution,
          const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  auto expected = builtinEvaluationModelDescriptorRef(
      BuiltinEvaluationModel::Gem5SystemCgra);
  if (!expected)
    return expected.takeError();
  if (request.modelBinding().descriptorRef() != *expected)
    return invalid("Request selects a foreign System Runtime ground-truth "
                   "model");
  const auto bindings =
      request.subjectBindings().subjects(CaseSubjectRoleRef(1));
  if (bindings.size() != 1)
    return invalid("ground-truth Request does not bind one gem5 owner");
  auto gem5 =
      runtime::importGem5SimulationBinding(bindings.front(), artifactStore);
  if (!gem5)
    return gem5.takeError();
  const EvaluationModelDescriptor *descriptor = expected->descriptor();
  if (!descriptor)
    return invalid("System Runtime ground-truth descriptor is unavailable");

  std::vector<std::uint8_t> key;
  appendU32(key, expected->schemaVersion().major);
  appendU32(key, expected->schemaVersion().minor);
  appendU32(key, expected->modelKind().ordinal());
  appendFramed(key, descriptor->implementationSemanticIdentity);
  appendFramed(key, descriptor->resolvedConfigView.schemaDescriptorBytes);
  appendFramed(
      key, request.modelBinding().resolvedModelConfig().canonicalViewBytes());
  const runtime::Gem5BuildIdentity &build = gem5->binding().gem5BuildIdentity();
  appendFramed(key, build.repositoryIdentity);
  appendFramed(key, build.fullCommitIdentity);
  appendFramed(key, build.buildConfigurationDigest);
  appendFramed(key, build.binaryFingerprint);
  appendFramed(key, gem5->binding().bridgeAbiIdentity());
  appendFramed(key, systemRuntimePredictionViewSchemaDescriptorBytes());
  appendFramed(key, kRuntimeTimingContract);
  appendFramed(key, kRuntimeFidelity);
  return key;
}

llvm::Expected<DecimalValue>
requiredRuntimeObservation(const EvaluationEvidence &evidence,
                           const EvaluationRequest &request) {
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  if (!completed)
    return invalid("training Evidence is not Completed");
  if (completed->metricResults.size() != request.metricRequests().size())
    return invalid("training Evidence metric shape does not match its Request");
  std::optional<DecimalValue> runtime;
  for (std::size_t index = 0; index != completed->metricResults.size();
       ++index) {
    const MetricRequest &metric = request.metricRequests()[index];
    if (metric.query().metric != MetricKind::Runtime)
      continue;
    if (metric.query().scope.form != ScopeFormRef(0) ||
        !metric.query().scope.targets.empty())
      return invalid("required Runtime observation is not WholeExactCase");
    const auto *point = std::get_if<PointObservation>(
        &completed->metricResults[index].observation);
    const auto *decimal =
        point ? std::get_if<DecimalValue>(&point->value) : nullptr;
    if (!decimal || runtime)
      return invalid("required Runtime observation is missing, duplicate, or "
                     "not a Decimal Point");
    runtime = *decimal;
  }
  if (!runtime)
    return invalid("training Evidence omits the required Runtime metric");
  return *runtime;
}

llvm::Expected<std::int64_t> checkedFeature(std::uint64_t value,
                                            llvm::StringRef field) {
  constexpr std::uint64_t limit = std::uint64_t{1} << 40;
  if (value > limit)
    return invalid(field + " exceeds the admitted feature magnitude");
  return static_cast<std::int64_t>(value);
}

llvm::Expected<detail::FixedTabularFeatureView>
fixedFeatures(const SystemRuntimeFeatureView &features) {
  detail::FixedTabularFeatureView fixed;
  const std::array<std::pair<std::uint64_t, llvm::StringRef>,
                   kIntegralFeatureCount>
      integral = {{
          {features.deployment.instructionCoreBinaryCount,
           "instruction-core binary count"},
          {features.deployment.hardwareBindingCount, "hardware-binding count"},
          {features.deployment.configurationImageCount,
           "configuration-image count"},
          {features.deployment.staticMemoryImageCount,
           "static-memory image count"},
          {features.workload.entryValueInputCount, "entry value-input count"},
          {features.workload.runtimeEntryValueInputCount,
           "runtime entry value-input count"},
          {features.workload.externalValueInputCount,
           "external value-input count"},
          {features.workload.runtimeExternalValueInputCount,
           "runtime external value-input count"},
          {features.workload.valueResultCount, "value-result count"},
          {features.workload.externalValueOutputCount,
           "external value-output count"},
          {features.workload.externalStreamOutputCount,
           "external stream-output count"},
          {features.workload.memoryObservableCount, "memory-observable count"},
          {features.runtimeInput.runtimeEntryValueCount,
           "runtime entry-value count"},
          {features.runtimeInput.runtimeExternalValueCount,
           "runtime external-value count"},
          {features.runtimeInput.externalStreamInputCount,
           "external stream-input count"},
          {features.runtimeInput.memoryObjectCount, "memory-object count"},
          {features.runtimeInput.memoryInterfaceBindingCount,
           "memory-interface binding count"},
          {features.runtimeInput.memoryByteCount, "memory-byte count"},
          {features.runtimeInput.streamTokenCount, "stream-token count"},
          {features.mapping.instructionContextDomainCount,
           "instruction-context domain count"},
          {features.mapping.spatialContextDomainCount,
           "spatial-context domain count"},
          {features.mapping.serviceRealizationCount,
           "service-realization count"},
          {features.mapping.capacityCellCount, "capacity-cell count"},
          {features.mapping.resourceActivationCount,
           "resource-activation count"},
          {features.mapping.capacityClaimCount, "capacity-claim count"},
          {features.mapping.causalReleaseCount, "causal-release count"},
          {features.fabric.entityCount, "Fabric entity count"},
          {features.fabric.hostCoreOccurrenceCount,
           "host-core occurrence count"},
          {features.fabric.acceleratorCoreOccurrenceCount,
           "accelerator-core occurrence count"},
          {features.fabric.systemMemoryServiceCount,
           "System memory-service count"},
          {features.fabric.systemTransportResourceCount,
           "System transport-resource count"},
          {features.fabric.hardwareDomainCount, "hardware-domain count"},
          {features.fabric.transportEndpointCount, "transport-endpoint count"},
          {features.fabric.pointConnectionCount, "point-connection count"},
          {features.fabric.admittedTraversalCount, "admitted-traversal count"},
      }};
  fixed.integral.reserve(integral.size());
  for (const auto &[value, field] : integral) {
    auto admitted = checkedFeature(value, field);
    if (!admitted)
      return admitted.takeError();
    fixed.integral.push_back(*admitted);
  }
  fixed.categorical = {
      features.softwarePartitioningKey, features.modeledPlatformKey,
      features.gem5BindingFeatureKey, features.admittedRuntimeConditionKey};
  fixed.presence = {features.deployment.hasSpatialLaunchImage};
  return fixed;
}

llvm::Expected<OwnerValue> adoptParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = adoptSystemRuntimeGbdtParameters(bytes);
  if (!parameters)
    return parameters.takeError();
  return OwnerValue::get(std::move(*parameters));
}

llvm::Expected<std::vector<std::uint8_t>>
encodeParameters(const OwnerValue &value) {
  const auto *parameters = value.getIf<SystemRuntimeGbdtParameters>();
  if (!parameters)
    return invalid("parameter value has a foreign owner type");
  return encodeSystemRuntimeGbdtParameters(*parameters);
}

llvm::Expected<std::vector<std::uint8_t>>
parameterTargetKey(const OwnerValue &value) {
  const auto *parameters = value.getIf<SystemRuntimeGbdtParameters>();
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
  const auto *typedParameters = parameters.getIf<SystemRuntimeGbdtParameters>();
  const auto *typedFeatures = features.getIf<SystemRuntimeFeatureView>();
  if (!typedParameters || !typedFeatures)
    return invalid("inference received a foreign owner value");
  return inferSystemRuntimeGbdtParameters(*typedParameters, *typedFeatures);
}

llvm::Expected<std::vector<std::uint8_t>> groundTruthTargetKey(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  return targetKey(request, resolution, artifactStore, blobStore);
}

llvm::Expected<std::vector<std::uint8_t>> calibrationSampleGroupKey(
    const EvaluationEvidence &, const EvaluationRequest &request,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto evaluationCase =
      requestCase(request, resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  return sourceInputKey(*evaluationCase, resolution, artifactStore, blobStore);
}

const std::vector<EvaluationCaseSignatureRef> &predictionCases() {
  static const std::vector<EvaluationCaseSignatureRef> values = {
      systemSimulationCaseSignatureRef()};
  return values;
}

const std::vector<EvaluationModelDescriptorRef> &groundTruthModels() {
  static const std::vector<EvaluationModelDescriptorRef> values = {
      llvm::cantFail(builtinEvaluationModelDescriptorRef(
          BuiltinEvaluationModel::Gem5SystemCgra))};
  return values;
}

const std::vector<ModelParameterConditionPatternSet> &conditionTable() {
  static const std::vector<ModelParameterConditionPatternSet> values = {
      {systemSimulationCaseSignatureRef(), systemSimulationCaseSignatureRef()
                                               .descriptor()
                                               ->permittedBaseConditions}};
  return values;
}

const ModelParameterContractDescriptor &descriptor() {
  static const ModelParameterContractDescriptor value{
      systemRuntimeModelParameterContractRef(),
      "Deterministic single-head System Runtime prediction over exact "
      "Deployment, workload, mapping, Fabric, gem5-binding, and runtime "
      "condition features.",
      predictionCases(),
      groundTruthModels(),
      conditionTable(),
      systemRuntimePredictionViewSchemaDescriptorBytes(),
      {18, ModelParameterDecimalRounding::RoundToNearestTiesToEven},
      &adoptParameters,
      &encodeParameters,
      &parameterTargetKey,
      &projectFeatures,
      &inferParameters,
      &groundTruthTargetKey,
      &calibrationSampleGroupKey};
  return value;
}

} // namespace

struct SystemRuntimeGbdtParameters::Storage final {
  detail::FixedTabularGbdtParameters parameters;
};

llvm::ArrayRef<std::uint8_t>
SystemRuntimeGbdtParameters::groundTruthTargetKey() const {
  return storage_ ? llvm::ArrayRef<std::uint8_t>(
                        storage_->parameters.groundTruthTargetKey)
                  : llvm::ArrayRef<std::uint8_t>();
}

const ModelParameterContractRef &systemRuntimeModelParameterContractRef() {
  static const ModelParameterContractRef reference = llvm::cantFail(
      ModelParameterContractRef::get("loom.system_runtime", {1, 0}, 0));
  return reference;
}

llvm::ArrayRef<std::uint8_t>
systemRuntimePredictionViewSchemaDescriptorBytes() {
  static const std::vector<std::uint8_t> bytes = [] {
    std::vector<std::uint8_t> result;
    constexpr llvm::StringLiteral owner = "loom.system_runtime.prediction_view";
    appendU64(result, owner.size());
    result.insert(result.end(), owner.bytes_begin(), owner.bytes_end());
    appendU32(result, 1);
    appendU32(result, 0);
    appendU64(result, 1);
    appendU32(result, static_cast<std::uint32_t>(MetricKind::Runtime));
    return result;
  }();
  return bytes;
}

const ModelParameterContractDescriptor &
systemRuntimeModelParameterContractDescriptor() {
  return descriptor();
}

llvm::Error registerSystemRuntimeModelParameterContract() {
  return registerModelParameterContract(descriptor());
}

llvm::Expected<SystemRuntimeTrainingSample>
importSystemRuntimeTrainingEvidenceSample(
    const ArtifactRootReference &evidenceReference,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto requestReference = importEvaluationEvidenceRequestReference(
      evidenceReference, artifactStore);
  if (!requestReference)
    return requestReference.takeError();
  auto resolution =
      resolveGroundTruthRequest(*requestReference, artifactStore, blobStore);
  if (!resolution)
    return resolution.takeError();
  auto request = importEvaluationRequest(*requestReference, *resolution,
                                         artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto evidence = importEvaluationEvidence(evidenceReference, *resolution,
                                           artifactStore, blobStore);
  if (!evidence)
    return evidence.takeError();
  auto runtime = requiredRuntimeObservation(*evidence, *request);
  if (!runtime)
    return runtime.takeError();
  auto evaluationCase =
      requestCase(*request, *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto features = projectFeatureView(*evaluationCase, *resolution,
                                     artifactStore, blobStore);
  if (!features)
    return features.takeError();
  auto key = targetKey(*request, *resolution, artifactStore, blobStore);
  if (!key)
    return key.takeError();
  auto group =
      sourceInputKey(*evaluationCase, *resolution, artifactStore, blobStore);
  if (!group)
    return group.takeError();
  return SystemRuntimeTrainingSample{std::move(*features), *runtime,
                                     std::move(*key), std::move(*group)};
}

llvm::Expected<SystemRuntimeGbdtParameters> trainSystemRuntimeGbdtParameters(
    llvm::ArrayRef<SystemRuntimeTrainingSample> training,
    const SystemRuntimeGbdtTrainingConfig &config,
    const SystemRuntimeGbdtParameters *prior) {
  if (training.empty())
    return invalid("Training partition is empty");
  const std::vector<std::uint8_t> &targetKey =
      training.front().groundTruthTargetKey;
  if (targetKey.empty())
    return invalid("Training target key is empty");
  std::vector<detail::FixedTabularTrainingRow> rows;
  rows.reserve(training.size());
  for (const SystemRuntimeTrainingSample &sample : training) {
    if (sample.groundTruthTargetKey != targetKey)
      return invalid("Training partition mixes ground-truth target keys");
    if (sample.sampleGroupKey.empty())
      return invalid("Training sample-group key is empty");
    auto features = fixedFeatures(sample.features);
    if (!features)
      return features.takeError();
    rows.push_back({std::move(*features), {sample.runtime}});
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
  auto storage = std::make_shared<SystemRuntimeGbdtParameters::Storage>();
  storage->parameters = std::move(*parameters);
  return SystemRuntimeGbdtParameters(std::move(storage));
}

llvm::Expected<SystemRuntimeGbdtParameters> adoptSystemRuntimeGbdtParameters(
    llvm::ArrayRef<std::uint8_t> canonicalPayloadBytes) {
  auto parameters = detail::decodeFixedTabularGbdt(
      canonicalPayloadBytes, parameterSchemaBytes(), kIntegralFeatureCount,
      kDecimalFeatureCount, kCategoricalFeatureCount, kPresenceFeatureCount,
      kTargetCount);
  if (!parameters)
    return parameters.takeError();
  auto storage = std::make_shared<SystemRuntimeGbdtParameters::Storage>();
  storage->parameters = std::move(*parameters);
  return SystemRuntimeGbdtParameters(std::move(storage));
}

llvm::Expected<std::vector<std::uint8_t>> encodeSystemRuntimeGbdtParameters(
    const SystemRuntimeGbdtParameters &parameters) {
  if (!parameters.storage_)
    return invalid("parameter storage is empty");
  return detail::encodeFixedTabularGbdt(parameters.storage_->parameters,
                                        parameterSchemaBytes());
}

llvm::Expected<ModelParameterInferenceOutcome>
inferSystemRuntimeGbdtParameters(const SystemRuntimeGbdtParameters &parameters,
                                 const SystemRuntimeFeatureView &features) {
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
    return ModelParameterInferenceOutcome{UnsupportedModelParameterInference{}};
  if ((**prediction).size() != kTargetCount)
    return invalid("inference returned the wrong target count");
  return ModelParameterInferenceOutcome{ModelParameterPrediction{
      OwnerValue::get(SystemRuntimePredictionView{(**prediction)[0]})}};
}

} // namespace loom::evaluation::models
