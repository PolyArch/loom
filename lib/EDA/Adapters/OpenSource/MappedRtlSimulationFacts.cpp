#include "MappedRtlSimulationInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Fabric/Artifact/FabricModuleRootView.h"
#include "Hardware/RTL/MemoryServiceTransport.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace loom::eda::open_source::detail {
namespace {

constexpr unsigned kBitsPerByte = 8;
using namespace evaluation;
using namespace external_tool;
using namespace hardware;

template <typename Value, typename Loader>
llvm::Expected<std::shared_ptr<const Value>>
importCachedOne(const ArtifactRootReference &reference,
                const ArtifactStore &artifacts, const BlobStore *blobs,
                Loader &&loader) {
  const std::array<ArtifactRootReference, 1> references{reference};
  return evaluation::importCachedArtifact<Value>(artifacts, blobs, references,
                                                 std::forward<Loader>(loader));
}

llvm::Expected<std::shared_ptr<const sim::ImportedSpatialSimulationInputs>>
importCachedSpatialInputs(const ArtifactRootReference &workload,
                          const ArtifactRootReference &runtimeInput,
                          const ArtifactStore &artifacts) {
  const std::array<ArtifactRootReference, 2> references{workload, runtimeInput};
  return evaluation::importCachedArtifact<sim::ImportedSpatialSimulationInputs>(
      artifacts, nullptr, references, [&]() {
        return sim::importSpatialSimulationInputs(workload, runtimeInput,
                                                  artifacts);
      });
}

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_simulation_invalid: " + detail);
}

MappedRtlFactsOrUnsupported unsupported() {
  return UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable};
}

std::string text(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<std::string> blobText(const BlobStore &blobs,
                                     const BlobDigest &digest) {
  auto bytes = blobs.get(digest);
  if (!bytes)
    return bytes.takeError();
  return text(*bytes);
}

llvm::Error addSemanticInput(std::vector<MaterializedBundleFile> &inputs,
                             llvm::StringRef path,
                             const ArtifactRootReference &source,
                             llvm::ArrayRef<std::uint8_t> bytes) {
  const std::filesystem::path candidate(path.str());
  if (candidate.is_absolute() || candidate.lexically_normal() != candidate ||
      !path.starts_with("inputs/semantic/") || path.contains('\0'))
    return invalid("semantic input path is not canonical");
  inputs.push_back({path.str(), text(bytes), source, false});
  return llvm::Error::success();
}

llvm::Expected<std::string>
rootPortName(const ImplementationRepresentationRoot &representation,
             const RepresentationLocator &locator) {
  const std::string prefix = representation.top.canonicalName + ".";
  const llvm::StringRef name(locator.canonicalName);
  if (locator.kind != RepresentationObjectKind::Port ||
      !name.starts_with(prefix))
    return invalid("interface locator is not a root Port");
  llvm::StringRef local = name.drop_front(prefix.size());
  if (local.empty() || local.contains('.'))
    return invalid("interface locator is not an immediate root Port");
  return local.str();
}

llvm::Expected<const ImplementationInterface *>
findInterface(const HardwareImplementation &implementation,
              const ImplementationInterfaceSemanticRef &semantic) {
  const ImplementationInterface *result = nullptr;
  for (const ImplementationInterface &candidate : implementation.interfaces())
    if (candidate.semanticRef == semantic) {
      if (result)
        return invalid("semantic interface has multiple root locators");
      result = &candidate;
    }
  if (!result)
    return invalid("semantic interface has no root locator");
  return result;
}

llvm::Expected<const ImplementationInterface *>
findDataInterface(const HardwareImplementation &implementation,
                  const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  return findInterface(implementation,
                       ImplementationDataInterfaceRef{endpoint});
}

llvm::Expected<const ImplementationInterface *> findMemoryInterface(
    const HardwareImplementation &implementation,
    const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  return findInterface(implementation,
                       ImplementationMemoryInterfaceRef{endpoint});
}

llvm::Expected<std::string> stripSuffix(llvm::StringRef value,
                                        llvm::StringRef suffix,
                                        llvm::StringRef context) {
  if (!value.ends_with(suffix) || value.size() == suffix.size())
    return invalid(context + " locator does not carry its ABI suffix");
  return value.drop_back(suffix.size()).str();
}

llvm::Expected<RepresentationSignalGeometry>
requirePort(const RepresentationIndex &index,
            const ImplementationRepresentationRoot &representation,
            llvm::StringRef localName) {
  RepresentationLocator locator{RepresentationObjectKind::Port,
                                representation.top.canonicalName + "." +
                                    localName.str()};
  auto found = index.lookup(locator);
  if (!found)
    return found.takeError();
  if (!*found || !(*found)->signalGeometry)
    return invalid("required RTL Port is absent from the exact representation");
  return *(*found)->signalGeometry;
}

llvm::Expected<TransportPort> deriveTransportPort(
    const HardwareImplementation &implementation,
    const RepresentationIndex &index,
    const fabric::FabricSpatialAttachmentRecordView &systemAttachment,
    const fabric::FabricArtifactView &module,
    std::uint32_t semanticPayloadWidth,
    std::optional<llvm::APInt> physicalTag) {
  const auto dataPath = module.moduleBoundaryEndpointDataPath(
      systemAttachment.moduleEndpoint.target);
  if (!dataPath || !dataPath->isWellFormed() ||
      dataPath->payloadWidthBits < semanticPayloadWidth)
    return invalid("mapped graph boundary has an unsupported data path");
  if ((dataPath->kind == ::fabric::DataPathKind::Bits) != !physicalTag ||
      (physicalTag && physicalTag->getBitWidth() != dataPath->tagWidthBits))
    return invalid("mapped graph boundary Physical Tag is inconsistent");
  auto interface =
      findDataInterface(implementation, systemAttachment.spatialEndpoint);
  if (!interface)
    return interface.takeError();
  auto validName = rootPortName(implementation.representationRoot(),
                                (*interface)->representationLocator);
  if (!validName)
    return validName.takeError();
  auto prefix = stripSuffix(*validName, "_valid", "Data interface");
  if (!prefix)
    return prefix.takeError();
  auto valid = requirePort(index, implementation.representationRoot(),
                           *prefix + "_valid");
  auto ready = requirePort(index, implementation.representationRoot(),
                           *prefix + "_ready");
  if (!valid || !ready)
    return llvm::joinErrors(valid ? llvm::Error::success() : valid.takeError(),
                            ready ? llvm::Error::success() : ready.takeError());
  const bool input = systemAttachment.moduleEndpoint.target.direction ==
                     fabric::FabricPortDirection::Input;
  if (valid->bitWidth != 1 || ready->bitWidth != 1 ||
      valid->direction != (input ? RepresentationSignalDirection::Input
                                 : RepresentationSignalDirection::Output) ||
      ready->direction != (input ? RepresentationSignalDirection::Output
                                 : RepresentationSignalDirection::Input))
    return invalid("Data interface handshake geometry is inconsistent");
  if (dataPath->payloadWidthBits != 0) {
    auto data = requirePort(index, implementation.representationRoot(),
                            *prefix + "_data");
    if (!data)
      return data.takeError();
    if (data->bitWidth != dataPath->payloadWidthBits ||
        data->direction != valid->direction)
      return invalid("Data interface payload geometry is inconsistent");
  }
  if (physicalTag) {
    auto tag = requirePort(index, implementation.representationRoot(),
                           *prefix + "_tag");
    if (!tag)
      return tag.takeError();
    if (tag->bitWidth != dataPath->tagWidthBits ||
        tag->direction != valid->direction)
      return invalid("Data interface Physical Tag geometry is inconsistent");
  }
  return TransportPort{std::move(*prefix), dataPath->payloadWidthBits,
                       std::move(physicalTag)};
}

llvm::Expected<const fabric::FabricSpatialAttachmentRecordView *>
findSystemAttachment(
    const fabric::FabricSystemRootView &system,
    const fabric::FabricImportedModuleBoundaryEndpointRef &moduleEndpoint,
    fabric::SpatialCoreOccurrenceRef spatialCore) {
  const fabric::FabricSpatialAttachmentRecordView *result = nullptr;
  for (const auto &attachment : system.spatialAttachments()) {
    if (!(attachment.moduleEndpoint == moduleEndpoint))
      continue;
    const auto *transport = attachment.spatialEndpoint.transport();
    if (!transport ||
        transport->owner.kind() !=
            fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence ||
        std::get<fabric::SpatialCoreOccurrenceRef>(transport->owner.payload) !=
            spatialCore)
      continue;
    if (result)
      return invalid(
          "Module boundary has multiple selected System attachments");
    result = &attachment;
  }
  if (!result)
    return invalid("Module boundary has no selected System attachment");
  return result;
}

llvm::Expected<const fabric::FabricSpatialAttachmentRecordView *>
findMemorySystemAttachment(
    const fabric::FabricSystemRootView &system,
    const fabric::FabricImportedModuleBoundaryEndpointRef &moduleEndpoint,
    fabric::SpatialCoreOccurrenceRef spatialCore) {
  const fabric::FabricSpatialAttachmentRecordView *result = nullptr;
  for (const auto &attachment : system.spatialAttachments()) {
    if (!(attachment.moduleEndpoint == moduleEndpoint))
      continue;
    const auto *memory = attachment.spatialEndpoint.memory();
    if (!memory ||
        memory->owner.kind() !=
            fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence ||
        std::get<fabric::SpatialCoreOccurrenceRef>(memory->owner.payload) !=
            spatialCore)
      continue;
    if (!attachment.serviceEndpoint)
      return invalid("System memory attachment has no service endpoint");
    if (result)
      return invalid("Module memory boundary has multiple selected System "
                     "attachments");
    result = &attachment;
  }
  if (!result)
    return invalid("Module memory boundary has no selected System attachment");
  return result;
}

llvm::Expected<MemoryBoundaryPort> deriveMemoryBoundaryPort(
    const HardwareImplementation &implementation,
    const RepresentationIndex &index,
    const fabric::FabricSystemRootView &system,
    const fabric::FabricImportedModuleTargetRef &moduleTarget,
    const fabric::FabricArtifactView &module,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const fabric::ManagerEndpointRef &manager,
    const hardware::rtl::PortableMemoryServiceLayout &expectedLayout) {
  const fabric::FabricModuleBoundaryMemoryAttachmentView *local = nullptr;
  for (const auto &attachment : module.moduleBoundaryMemoryAttachments()) {
    if (attachment.endpoint != manager.underlying())
      continue;
    if (attachment.boundary.direction != fabric::FabricPortDirection::Input)
      return invalid("manager endpoint is exported through a non-input Module "
                     "boundary");
    if (local)
      return invalid("manager endpoint has multiple Module memory boundaries");
    local = &attachment;
  }
  if (!local)
    return invalid("manager endpoint has no Module memory boundary");

  const fabric::FabricImportedModuleBoundaryEndpointRef imported{
      moduleTarget.dependencyOrdinal, local->boundary};
  auto attachment = findMemorySystemAttachment(system, imported, spatialCore);
  if (!attachment)
    return attachment.takeError();
  auto interface =
      findMemoryInterface(implementation, (*attachment)->spatialEndpoint);
  if (!interface)
    return interface.takeError();
  auto requestValid = rootPortName(implementation.representationRoot(),
                                   (*interface)->representationLocator);
  if (!requestValid)
    return requestValid.takeError();
  auto prefix =
      stripSuffix(*requestValid, "_request_valid", "Memory interface");
  if (!prefix)
    return prefix.takeError();

  const auto require = [&](llvm::StringRef suffix)
      -> llvm::Expected<RepresentationSignalGeometry> {
    return requirePort(index, implementation.representationRoot(),
                       *prefix + suffix.str());
  };
  auto requestKind = require("_request_kind");
  auto requestAddress = require("_request_address");
  auto requestData = require("_request_data");
  auto requestMask = require("_request_mask");
  auto activeLanes = require("_request_active_lanes_kind");
  auto accessForm = require("_request_access_form");
  auto addressForm = require("_request_address_form");
  auto elementWidth = require("_request_element_width");
  auto laneCount = require("_request_lane_count");
  auto addressLaneWidth = require("_request_address_lane_width");
  auto baseAddress = require("_request_base_address");
  auto context = require("_request_context");
  auto valid = require("_request_valid");
  auto ready = require("_request_ready");
  auto responseData = require("_response_data");
  auto responseValid = require("_response_valid");
  auto responseReady = require("_response_ready");
  if (!requestKind || !requestAddress || !requestData || !requestMask ||
      !activeLanes || !accessForm || !addressForm || !elementWidth ||
      !laneCount || !addressLaneWidth || !baseAddress || !context || !valid ||
      !ready || !responseData || !responseValid || !responseReady) {
    llvm::Error errors = llvm::Error::success();
    const auto append = [&](auto &value) {
      if (!value)
        errors = llvm::joinErrors(std::move(errors), value.takeError());
    };
    append(requestKind);
    append(requestAddress);
    append(requestData);
    append(requestMask);
    append(activeLanes);
    append(accessForm);
    append(addressForm);
    append(elementWidth);
    append(laneCount);
    append(addressLaneWidth);
    append(baseAddress);
    append(context);
    append(valid);
    append(ready);
    append(responseData);
    append(responseValid);
    append(responseReady);
    return std::move(errors);
  }
  const auto output = RepresentationSignalDirection::Output;
  const auto input = RepresentationSignalDirection::Input;
  if (requestKind->direction != output ||
      requestKind->bitWidth != hardware::rtl::portableMemoryRequestKindWidth ||
      requestAddress->direction != output ||
      requestAddress->bitWidth != expectedLayout.addressWidthBits ||
      requestData->direction != output ||
      requestData->bitWidth != expectedLayout.dataWidthBits ||
      requestData->bitWidth % kBitsPerByte != 0 ||
      requestMask->direction != output ||
      requestMask->bitWidth != expectedLayout.maskWidthBits ||
      activeLanes->direction != output ||
      activeLanes->bitWidth !=
          hardware::rtl::portableMemoryActiveLanesKindWidth ||
      accessForm->direction != output ||
      accessForm->bitWidth != hardware::rtl::portableMemoryAccessFormWidth ||
      addressForm->direction != output ||
      addressForm->bitWidth != hardware::rtl::portableMemoryAddressFormWidth ||
      elementWidth->direction != output ||
      elementWidth->bitWidth !=
          hardware::rtl::portableMemoryElementWidthFieldWidth ||
      laneCount->direction != output ||
      laneCount->bitWidth != hardware::rtl::portableMemoryLaneCountFieldWidth ||
      addressLaneWidth->direction != output ||
      addressLaneWidth->bitWidth !=
          hardware::rtl::portableMemoryAddressLaneWidthFieldWidth ||
      baseAddress->direction != output ||
      baseAddress->bitWidth !=
          hardware::rtl::portableMemoryBaseAddressFieldWidth ||
      context->direction != output ||
      context->bitWidth != hardware::rtl::portableMemoryContextFieldWidth ||
      valid->direction != output ||
      valid->bitWidth != hardware::rtl::portableMemoryHandshakeWidth ||
      ready->direction != input ||
      ready->bitWidth != hardware::rtl::portableMemoryHandshakeWidth ||
      responseData->direction != input ||
      responseData->bitWidth != requestData->bitWidth ||
      responseValid->direction != input ||
      responseValid->bitWidth != hardware::rtl::portableMemoryHandshakeWidth ||
      responseReady->direction != output ||
      responseReady->bitWidth != hardware::rtl::portableMemoryHandshakeWidth)
    return invalid("Memory interface has unsupported RTL geometry");

  return MemoryBoundaryPort{
      std::move(*prefix),
      static_cast<std::uint32_t>(requestAddress->bitWidth),
      static_cast<std::uint32_t>(requestData->bitWidth),
      static_cast<std::uint32_t>(requestMask->bitWidth),
      {}};
}

llvm::Expected<TransportPort> projectInternalEndpoint(
    const HardwareImplementation &implementation,
    const RepresentationIndex &index,
    const fabric::FabricSystemRootView &system,
    const fabric::FabricImportedModuleTargetRef &moduleTarget,
    const fabric::FabricArtifactView &module,
    const mapping::SpatialMappingView &mapping,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const fabric::FabricTransportEndpointRef &endpoint,
    fabric::FabricPortDirection direction, std::uint32_t semanticWidth,
    std::uint64_t routeTreeOrdinal, std::uint64_t nodeOrdinal) {
  const fabric::FabricModuleBoundaryTransportAttachmentView *local = nullptr;
  for (const auto &attachment : module.moduleBoundaryTransportAttachments())
    if (attachment.endpoint == endpoint &&
        attachment.boundary.direction == direction) {
      if (local)
        return invalid("physical endpoint has multiple Module boundaries");
      local = &attachment;
    }
  if (!local)
    return invalid(
        "mapped graph terminal is not attached to a Module boundary");
  const fabric::FabricImportedModuleBoundaryEndpointRef imported{
      moduleTarget.dependencyOrdinal, local->boundary};
  auto attachment = findSystemAttachment(system, imported, spatialCore);
  if (!attachment)
    return attachment.takeError();
  std::optional<llvm::APInt> physicalTag;
  const auto dataPath = module.transportEndpointDataPath(endpoint);
  if (!dataPath || !dataPath->isWellFormed())
    return invalid("mapped graph terminal has no valid data path");
  if (dataPath->kind == ::fabric::DataPathKind::BitsTag) {
    auto resolved = mapping::resolveSpatialPhysicalTag(
        mapping, module, routeTreeOrdinal, nodeOrdinal);
    if (!resolved)
      return resolved.takeError();
    physicalTag = std::move(*resolved);
  }
  return deriveTransportPort(implementation, index, **attachment, module,
                             semanticWidth, std::move(physicalTag));
}

enum class IngressKind { Start, Value, Stream };

struct IngressOrdinal final {
  IngressKind kind;
  std::uint64_t ordinal;
};

std::optional<IngressOrdinal>
classifyIngress(const dataflow::GraphIngressTokenRef &ingress,
                dataflow::GraphRef selectedGraph) {
  return std::visit(
      [&](const auto &token) -> std::optional<IngressOrdinal> {
        if (token.graph != selectedGraph)
          return std::nullopt;
        using T = std::decay_t<decltype(token)>;
        if constexpr (std::is_same_v<T, dataflow::GraphStartTokenRef>)
          return IngressOrdinal{IngressKind::Start, 0};
        else if constexpr (std::is_same_v<T, dataflow::GraphValueInputTokenRef>)
          return IngressOrdinal{IngressKind::Value, token.ordinal};
        else
          return IngressOrdinal{IngressKind::Stream, token.ordinal};
      },
      ingress);
}

enum class EgressKind { Value, Stream, Completion };

struct EgressOrdinal final {
  EgressKind kind;
  std::uint64_t ordinal;
};

std::optional<EgressOrdinal>
classifyEgress(const dataflow::GraphEgressTokenRef &egress,
               dataflow::GraphRef selectedGraph) {
  return std::visit(
      [&](const auto &token) -> std::optional<EgressOrdinal> {
        if (token.graph != selectedGraph)
          return std::nullopt;
        using T = std::decay_t<decltype(token)>;
        if constexpr (std::is_same_v<T, dataflow::GraphValueOutputTokenRef>)
          return EgressOrdinal{EgressKind::Value, token.ordinal};
        else if constexpr (std::is_same_v<T,
                                          dataflow::GraphStreamOutputTokenRef>)
          return EgressOrdinal{EgressKind::Stream, token.ordinal};
        else
          return EgressOrdinal{EgressKind::Completion, token.ordinal};
      },
      egress);
}

llvm::Expected<std::uint32_t>
semanticWidth(sim::SpatialSimulationValueShape shape) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint32_t>::max() / shape.laneBitWidth)
    return invalid("semantic token width exceeds RTL transport capacity");
  return static_cast<std::uint32_t>(shape.lanesPerToken * shape.laneBitWidth);
}

llvm::Expected<std::vector<llvm::APInt>>
packSequence(const sim::CanonicalValueSequence &sequence,
             sim::SpatialSimulationValueShape shape) {
  std::vector<llvm::APInt> tokens;
  tokens.reserve(sequence.tokenCount);
  for (std::uint64_t ordinal = 0; ordinal != sequence.tokenCount; ++ordinal) {
    auto token =
        sim::packDefinedSpatialSimulationToken(sequence, shape, ordinal);
    if (!token)
      return token.takeError();
    tokens.push_back(std::move(*token));
  }
  return tokens;
}

llvm::Expected<const sim::CanonicalValueSequence *>
valueInputSequence(const sim::SpatialSimulationWorkload &workload,
                   const sim::SpatialSimulationRuntimeInput &runtime,
                   std::uint64_t ordinal) {
  if (ordinal >= workload.valueInputPlan.size())
    return invalid("value-input ordinal exceeds the exact workload");
  if (const auto *fixed = std::get_if<sim::CanonicalValueSequence>(
          &workload.valueInputPlan[ordinal]))
    return fixed;
  const sim::CanonicalValueSequence *result = nullptr;
  for (const sim::RuntimeValueEntry &entry : runtime.runtimeValues)
    if (entry.valueInputOrdinal == ordinal) {
      if (result)
        return invalid("runtime value input is duplicated");
      result = &entry.value;
    }
  if (!result)
    return invalid("runtime value input is absent");
  return result;
}

dataflow::LogicalMemoryRootRef
memoryRoot(const dataflow::LogicalMemoryRootOrViewRef &memory) {
  if (const auto *root = std::get_if<dataflow::LogicalMemoryRootRef>(&memory))
    return *root;
  return std::get<dataflow::LogicalMemoryViewRef>(memory).root;
}

const sim::MemoryRootBindingEntry *
findRuntimeMemoryBinding(const sim::SpatialSimulationRuntimeInput &runtime,
                         dataflow::LogicalMemoryRootRef root) {
  const sim::MemoryRootBindingEntry *result = nullptr;
  for (const sim::MemoryRootBindingEntry &entry : runtime.memoryRootBindings) {
    if (entry.root != root)
      continue;
    if (result)
      return nullptr;
    result = &entry;
  }
  return result;
}

const mapping::SpatialMemoryBindingView *
findSpatialMemoryBinding(const mapping::SpatialMappingView &mapping,
                         std::uint64_t entityId) {
  const mapping::SpatialMemoryBindingView *result = nullptr;
  for (const mapping::SpatialMemoryBindingView &binding :
       mapping.memoryBindings()) {
    if (binding.entityId != entityId)
      continue;
    if (result)
      return nullptr;
    result = &binding;
  }
  return result;
}

llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> resolveObservableMemory(
    const sim::SpatialMemoryObservableTarget &target,
    const sim::SpatialSimulationWorkload &workload,
    const dataflow::CanonicalDataflowProgramView &dataflow) {
  if (const auto *direct =
          std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&target))
    return *direct;
  return dataflow.resolveExposure(dataflow::MemoryExposureRef{
      workload.launchRef,
      std::get<sim::MemoryExposureTarget>(target).memoryResultOrdinal});
}

struct MemoryProjection final {
  std::vector<RuntimeMemoryImage> images;
  std::vector<MemoryBoundaryPort> ports;
  std::vector<MemoryObservationPlan> observations;
  hardware::rtl::PortableMemoryAddressArithmetic addressArithmetic;
};

llvm::Expected<std::optional<std::vector<RuntimeMemoryImage>>>
projectRuntimeMemoryImages(const sim::SpatialSimulationRuntimeInput &runtime) {
  std::vector<RuntimeMemoryImage> images;
  images.reserve(runtime.memoryObjects.size());
  std::uint64_t nextBase = 1;
  for (const sim::RuntimeMemoryObject &object : runtime.memoryObjects) {
    if (object.initialBytes.empty())
      return invalid("runtime memory object is empty");
    if (llvm::any_of(object.initialBytes, [](const auto &byte) {
          return byte.state != sim::SemanticState::Defined;
        }))
      return std::optional<std::vector<RuntimeMemoryImage>>{};
    if (object.initialBytes.size() >
        std::numeric_limits<std::uint64_t>::max() - nextBase - 1)
      return invalid("runtime memory address space overflows 64 bits");
    images.push_back(RuntimeMemoryImage{nextBase, object.initialBytes});
    nextBase += object.initialBytes.size() + 1;
  }
  return std::optional<std::vector<RuntimeMemoryImage>>{std::move(images)};
}

llvm::Expected<std::optional<std::set<std::uint64_t>>>
projectExactLaunchMemoryBindings(
    const mapping::SpatialMappingView &mapping,
    const sim::SpatialSimulationWorkload &workload) {
  std::set<std::uint64_t> bindingIds;
  for (const mapping::SpatialMemoryEngineBindingView &engine :
       mapping.memoryEngineBindings()) {
    for (const mapping::SpatialMemoryOperationView &operation :
         engine.operations) {
      if (const auto *fence =
              std::get_if<mapping::SpatialFenceMemoryOperationView>(
                  &operation)) {
        for (const mapping::SpatialFenceMemoryUseView &use : fence->uses)
          if (use.launch == workload.launchRef &&
              std::holds_alternative<fabric::ManagerEndpointRef>(
                  use.consistency))
            return std::optional<std::set<std::uint64_t>>{};
        continue;
      }
      const auto &addressed =
          std::get<mapping::SpatialAddressedMemoryOperationView>(operation);
      for (const mapping::SpatialAddressedMemoryUseView &use : addressed.uses) {
        if (use.launch != workload.launchRef)
          continue;
        const mapping::SpatialMemoryBindingView *binding =
            findSpatialMemoryBinding(mapping, use.binding);
        if (!binding)
          return invalid("selected memory use has no unique binding");
        bindingIds.insert(binding->entityId);
      }
    }
  }
  return std::optional<std::set<std::uint64_t>>{std::move(bindingIds)};
}

llvm::Expected<std::optional<std::vector<MemoryObservationPlan>>>
projectRuntimeMemoryObservationPlans(
    const mapping::SpatialMappingView &mapping,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const sim::SpatialSimulationWorkload &workload,
    const sim::SpatialSimulationRuntimeInput &runtime,
    const std::set<std::uint64_t> &exactLaunchBindingIds) {
  std::vector<MemoryObservationPlan> observations;
  observations.reserve(workload.observableContract.memories.size());
  for (const sim::SpatialMemoryObservable &observable :
       workload.observableContract.memories) {
    auto memory =
        resolveObservableMemory(observable.target, workload, dataflow);
    if (!memory)
      return memory.takeError();
    const dataflow::LogicalMemoryRootRef root = memoryRoot(*memory);
    const sim::MemoryRootBindingEntry *runtimeBinding =
        findRuntimeMemoryBinding(runtime, root);
    if (!runtimeBinding ||
        runtimeBinding->binding.objectOrdinal >= runtime.memoryObjects.size())
      return invalid("observable memory has no runtime object binding");
    const sim::RuntimeMemoryObject &object =
        runtime.memoryObjects[runtimeBinding->binding.objectOrdinal];
    if (runtimeBinding->binding.byteOffset >= object.initialBytes.size())
      return invalid("observable memory binding offset is out of range");

    bool externallyServed = false;
    for (const mapping::SpatialMemoryBindingView &binding :
         mapping.memoryBindings()) {
      if (memoryRoot(binding.logicalMemory) != root ||
          !exactLaunchBindingIds.count(binding.entityId))
        continue;
      if (std::holds_alternative<mapping::SpatialMemoryBoundaryProxyView>(
              binding.target))
        externallyServed = true;
    }
    if (!externallyServed)
      return std::optional<std::vector<MemoryObservationPlan>>{};
    observations.push_back(MemoryObservationPlan{
        runtimeBinding->binding.objectOrdinal,
        runtimeBinding->binding.byteOffset, observable.form});
  }
  return std::optional<std::vector<MemoryObservationPlan>>{
      std::move(observations)};
}

llvm::Expected<std::optional<MemoryProjection>>
projectRuntimeMemory(const HardwareImplementation &implementation,
                     const RepresentationIndex &index,
                     const fabric::FabricSystemRootView &system,
                     const fabric::FabricImportedModuleTargetRef &moduleTarget,
                     const fabric::FabricArtifactView &module,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const mapping::SpatialMappingView &mapping,
                     const dataflow::CanonicalDataflowProgramView &dataflow,
                     const sim::SpatialSimulationWorkload &workload,
                     const sim::SpatialSimulationRuntimeInput &runtime) {
  MemoryProjection projection;
  auto images = projectRuntimeMemoryImages(runtime);
  if (!images)
    return images.takeError();
  if (!*images)
    return std::optional<MemoryProjection>{};
  projection.images = std::move(**images);
  auto exactLaunchBindingIds =
      projectExactLaunchMemoryBindings(mapping, workload);
  if (!exactLaunchBindingIds)
    return exactLaunchBindingIds.takeError();
  if (!*exactLaunchBindingIds)
    return std::optional<MemoryProjection>{};
  auto requestContexts =
      hardware::rtl::PortableMemoryRequestContextIndex::get(module);
  if (!requestContexts)
    return requestContexts.takeError();
  auto expectedMemoryLayout =
      hardware::rtl::derivePortableMemoryServiceLayout(module);
  if (!expectedMemoryLayout)
    return expectedMemoryLayout.takeError();
  // The portable profile has no exact address arithmetic for a lane wider
  // than its byte-address domain; the mapped-RTL engine is then Unsupported.
  const auto addressArithmetic =
      hardware::rtl::derivePortableMemoryAddressArithmetic(
          *expectedMemoryLayout);
  if (!addressArithmetic)
    return std::optional<MemoryProjection>{};
  projection.addressArithmetic = *addressArithmetic;

  std::map<std::string, MemoryBoundaryPort> ports;
  for (const mapping::SpatialMemoryEngineBindingView &engine :
       mapping.memoryEngineBindings()) {
    for (const mapping::SpatialMemoryOperationView &operation :
         engine.operations) {
      if (const auto *fence =
              std::get_if<mapping::SpatialFenceMemoryOperationView>(
                  &operation)) {
        (void)fence;
        continue;
      }

      const auto &addressed =
          std::get<mapping::SpatialAddressedMemoryOperationView>(operation);
      const auto *operationContext =
          std::get_if<fabric::FabricMemoryOperationContextRef>(
              &addressed.placement);
      const fabric::FabricMemoryOperationPortRef operationPort =
          operationContext ? operationContext->port
                           : std::get<fabric::FabricMemoryOperationPortRef>(
                                 addressed.placement);
      const std::uint64_t operationRowOrdinal =
          operationContext ? operationContext->ordinal : operationPort.ordinal;
      if (operationPort.memory != engine.occurrence)
        return invalid("memory operation placement has a foreign occurrence");
      auto requestContext =
          requestContexts->code(engine.occurrence, operationRowOrdinal);
      if (!requestContext)
        return requestContext.takeError();
      for (const mapping::SpatialAddressedMemoryUseView &use : addressed.uses) {
        if (use.launch != workload.launchRef)
          continue;
        const mapping::SpatialMemoryBindingView *binding =
            findSpatialMemoryBinding(mapping, use.binding);
        if (!binding)
          return invalid("selected memory use has no unique binding");
        if (std::holds_alternative<mapping::SpatialMemoryLocalRegionView>(
                binding->target))
          return std::optional<MemoryProjection>{};
        const auto *manager =
            std::get_if<fabric::ManagerEndpointRef>(&use.dispatch);
        if (!manager)
          return invalid("BoundaryProxy use has no manager endpoint");
        const dataflow::LogicalMemoryRootRef root =
            memoryRoot(binding->logicalMemory);
        const sim::MemoryRootBindingEntry *runtimeBinding =
            findRuntimeMemoryBinding(runtime, root);
        if (!runtimeBinding || runtimeBinding->binding.objectOrdinal >=
                                   runtime.memoryObjects.size())
          return invalid("mapped memory root has no runtime object binding");
        auto port = deriveMemoryBoundaryPort(implementation, index, system,
                                             moduleTarget, module, spatialCore,
                                             *manager, *expectedMemoryLayout);
        if (!port)
          return port.takeError();
        auto [entry, inserted] = ports.emplace(port->prefix, *port);
        if (!inserted &&
            (entry->second.addressBitWidth != port->addressBitWidth ||
             entry->second.dataBitWidth != port->dataBitWidth ||
             entry->second.maskBitWidth != port->maskBitWidth))
          return invalid("one memory boundary prefix has conflicting RTL "
                         "geometry");
        const MemoryBoundaryBinding projected{
            *requestContext, runtimeBinding->binding.objectOrdinal,
            runtimeBinding->binding.byteOffset};
        auto selected = llvm::find_if(
            entry->second.bindings, [&](const MemoryBoundaryBinding &binding) {
              return binding.requestContext == projected.requestContext;
            });
        if (selected != entry->second.bindings.end()) {
          if (selected->rootObjectOrdinal != projected.rootObjectOrdinal ||
              selected->rootByteOffset != projected.rootByteOffset)
            return invalid("one memory request context selects multiple "
                           "runtime roots");
        } else {
          entry->second.bindings.push_back(projected);
        }
      }
    }
  }
  for (auto &[prefix, port] : ports) {
    (void)prefix;
    llvm::sort(port.bindings, [](const MemoryBoundaryBinding &lhs,
                                 const MemoryBoundaryBinding &rhs) {
      return lhs.requestContext < rhs.requestContext;
    });
    projection.ports.push_back(std::move(port));
  }

  auto observations = projectRuntimeMemoryObservationPlans(
      mapping, dataflow, workload, runtime, **exactLaunchBindingIds);
  if (!observations)
    return observations.takeError();
  if (!*observations)
    return std::optional<MemoryProjection>{};
  projection.observations = std::move(**observations);
  return std::optional<MemoryProjection>{std::move(projection)};
}

llvm::Expected<std::vector<RtlPort>>
projectRootPorts(const RepresentationIndex &index) {
  std::vector<RtlPort> result;
  const std::string prefix = index.exactRoot().canonicalName + ".";
  for (const RepresentationBoundaryPort &port : index.rootBoundaryPorts()) {
    const llvm::StringRef local =
        llvm::StringRef(port.locator.canonicalName).drop_front(prefix.size());
    result.push_back(
        {local.str(), port.geometry.direction, port.geometry.bitWidth});
  }
  llvm::sort(result, [](const RtlPort &lhs, const RtlPort &rhs) {
    return lhs.name < rhs.name;
  });
  if (result.empty())
    return invalid("RTL representation has no indexed root Ports");
  return result;
}

llvm::Expected<std::pair<std::vector<ClockPort>, std::vector<ResetPort>>>
projectClockResetPorts(const HardwareImplementation &implementation,
                       const fabric::FabricSystemRootView &system) {
  std::vector<ClockPort> clocks;
  std::vector<ResetPort> resets;
  for (const ImplementationInterface &interface : implementation.interfaces()) {
    if (const auto *clock = std::get_if<ImplementationClockInterfaceRef>(
            &interface.semanticRef)) {
      const auto *record = system.hardwareDomainContract(clock->domain);
      const auto *contract =
          record ? std::get_if<fabric::ClockDomainContractRecord>(
                       &record->contract())
                 : nullptr;
      if (!contract)
        return invalid("Clock interface does not resolve to a Clock contract");
      auto name = rootPortName(implementation.representationRoot(),
                               interface.representationLocator);
      if (!name)
        return name.takeError();
      clocks.push_back(
          {std::move(*name), contract->periodFs(), contract->phaseFs()});
    } else if (const auto *reset = std::get_if<ImplementationResetInterfaceRef>(
                   &interface.semanticRef)) {
      const auto *record = system.hardwareDomainContract(reset->domain);
      const auto *contract =
          record ? std::get_if<fabric::ResetDomainContractRecord>(
                       &record->contract())
                 : nullptr;
      if (!contract)
        return invalid("Reset interface does not resolve to a Reset contract");
      auto name = rootPortName(implementation.representationRoot(),
                               interface.representationLocator);
      if (!name)
        return name.takeError();
      resets.push_back(
          {std::move(*name),
           contract->polarity() == fabric::ResetPolarity::ActiveHigh});
    }
  }
  if (clocks.empty() || resets.empty())
    return invalid("RTL representation omits Clock or Reset interfaces");
  return std::make_pair(std::move(clocks), std::move(resets));
}

llvm::Expected<std::string>
selectedClockPort(const HardwareImplementation &implementation,
                  const fabric::FabricSystemRootView &system,
                  fabric::SpatialCoreOccurrenceRef spatialCore,
                  std::uint64_t &periodFs) {
  auto effectiveClock = system.effectiveHardwareDomain(
      spatialCore, fabric::FabricClockResetKind::Clock);
  if (!effectiveClock)
    return effectiveClock.takeError();
  const ImplementationInterface *selected = nullptr;
  const fabric::ClockDomainContractRecord *selectedContract = nullptr;
  for (const ImplementationInterface &interface : implementation.interfaces()) {
    const auto *clock =
        std::get_if<ImplementationClockInterfaceRef>(&interface.semanticRef);
    if (!clock)
      continue;
    if (clock->domain != *effectiveClock)
      continue;
    const auto *record = system.hardwareDomainContract(clock->domain);
    if (!record)
      return invalid("selected Clock interface has no domain contract");
    const auto *contract =
        std::get_if<fabric::ClockDomainContractRecord>(&record->contract());
    if (!contract)
      return invalid("selected Clock domain has a non-Clock contract");
    if (selected)
      return invalid("selected SpatialCore belongs to multiple Clock domains");
    selected = &interface;
    selectedContract = contract;
  }
  if (!selected || !selectedContract)
    return invalid("selected SpatialCore has no exact Clock interface");
  periodFs = selectedContract->periodFs();
  return rootPortName(implementation.representationRoot(),
                      selected->representationLocator);
}

llvm::Expected<std::vector<ConfigurationProgram>> projectConfigurationPrograms(
    const deployment::DeploymentSpatialLaunchSelection &selection,
    const deployment::FinalizedDeployment &deployment,
    const HardwareImplementation &implementation,
    const FinalizedConfigurationABI &abi,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const ArtifactStore &artifacts) {
  auto layout =
      rtl::derivePortableConfigurationTransportLayout(abi, spatialCore);
  if (!layout)
    return layout.takeError();
  std::vector<deployment::FinalizedHardwareConfigurationImage> images;
  images.reserve(selection.configurationImages.size());
  for (const ArtifactRootReference &reference : selection.configurationImages) {
    auto image =
        deployment::importHardwareConfigurationImage(reference, artifacts);
    if (!image)
      return image.takeError();
    images.push_back(std::move(*image));
  }
  std::vector<ConfigurationProgram> result;
  result.reserve(layout->units.size());
  for (const rtl::ConfigurationTransportUnitLayout &unit : layout->units) {
    const deployment::FinalizedHardwareConfigurationImage *selectedImage =
        nullptr;
    for (const auto &image : images)
      if (image.image().programmingUnitId() == unit.programmingUnit.unitId) {
        if (selectedImage)
          return invalid("Programming Unit has multiple configuration images");
        selectedImage = &image;
      }
    if (!selectedImage)
      return invalid("Programming Unit has no configuration image");
    const auto &image = selectedImage->image();
    if (image.configurationAbi() != implementation.configurationAbi() ||
        image.payloadBitCount() != unit.payloadBitCount)
      return invalid("configuration image disagrees with the selected ABI");
    const bool exactSpatial =
        image.sourceMapping().kind ==
            deployment::ConfigurationImageSourceKind::SpatialMapping &&
        image.sourceMapping().mapping == selection.spatialMapping;
    const bool exactSystem =
        image.sourceMapping().kind ==
            deployment::ConfigurationImageSourceKind::SystemMapping &&
        image.sourceMapping().mapping ==
            deployment.deployment().systemMapping();
    if (!exactSpatial && !exactSystem)
      return invalid("configuration image names a foreign Mapping source");
    auto interface = findInterface(
        implementation,
        ImplementationConfigurationInterfaceRef{unit.programmingUnit});
    if (!interface)
      return interface.takeError();
    auto awaddr = rootPortName(implementation.representationRoot(),
                               (*interface)->representationLocator);
    if (!awaddr)
      return awaddr.takeError();
    auto prefix = stripSuffix(*awaddr, "_awaddr", "Configuration interface");
    if (!prefix)
      return prefix.takeError();
    result.push_back({unit, std::move(*prefix),
                      std::vector<std::uint8_t>(image.payload().begin(),
                                                image.payload().end())});
  }
  return result;
}

std::string configurationProgramPath(std::size_t ordinal) {
  return "inputs/configuration/program-" + std::to_string(ordinal) + ".hex";
}

/// The semantic inputs of one mapped-RTL invocation in their canonical bundle
/// order: the rendered configuration program images (derived from the
/// Deployment's configuration images through the ABI transport layout), the
/// canonical artifact bytes, and the RTL source payloads. Bundle preparation
/// and the import expectation both derive this list, so an executed bundle is
/// validated against exactly the files it was prepared from.
llvm::Expected<std::vector<MaterializedBundleFile>>
materializeMappedRtlSemanticInputs(
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeReference,
    const FinalizedHardwareImplementation &implementation,
    const deployment::FinalizedDeployment &deployment,
    const sim::ImportedSpatialSimulationInputs &inputs,
    const deployment::DeploymentSpatialLaunchSelection &selection,
    const mapping::FinalizedSpatialMapping &mapping,
    const FinalizedConfigurationABI &abi,
    const fabric::FinalizedFabricRoot &fabricSystem,
    llvm::ArrayRef<ConfigurationProgram> configurationPrograms,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::vector<MaterializedBundleFile> semanticInputs;
  for (const auto indexedProgram : llvm::enumerate(configurationPrograms)) {
    auto contents =
        renderMappedRtlConfigurationProgramFile(indexedProgram.value());
    if (!contents)
      return contents.takeError();
    semanticInputs.push_back({configurationProgramPath(indexedProgram.index()),
                              std::move(*contents), deployment.reference(),
                              false});
  }
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/hardware-implementation.json",
          implementation.reference(), implementation.canonicalBytes().bytes()))
    return std::move(error);
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/deployment.json",
          deployment.reference(), deployment.canonicalBytes().bytes()))
    return std::move(error);
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/workload.bin", workloadReference,
          inputs.workload.canonicalBytes().bytes()))
    return std::move(error);
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/runtime-input.bin", runtimeReference,
          inputs.runtimeInput.canonicalBytes().bytes()))
    return std::move(error);
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/spatial-mapping.mlir",
          mapping.reference(), mapping.canonicalBytes().bytes()))
    return std::move(error);
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/configuration-abi.json",
          abi.reference(), abi.canonicalBytes().bytes()))
    return std::move(error);
  if (llvm::Error error = addSemanticInput(
          semanticInputs, "inputs/semantic/fabric-system.mlir",
          fabricSystem.reference(), fabricSystem.canonicalBytes().bytes()))
    return std::move(error);
  for (const ArtifactRootReference &reference : selection.configurationImages) {
    auto image =
        deployment::importHardwareConfigurationImage(reference, artifacts);
    if (!image)
      return image.takeError();
    const std::string path = "inputs/semantic/configuration-images/" +
                             formatArtifactIdentityHex(reference.artifact) +
                             ".json";
    if (llvm::Error error = addSemanticInput(semanticInputs, path, reference,
                                             image->canonicalBytes().bytes()))
      return std::move(error);
  }

  const ImplementationRepresentationRoot &representation =
      implementation.implementation().representationRoot();
  for (const ImplementationPayload &payload : representation.payloads) {
    if (payload.role != PayloadRole::RtlSource)
      continue;
    auto contents = blobText(blobs, payload.blobDigest);
    if (!contents)
      return contents.takeError();
    const std::filesystem::path logical(payload.canonicalLogicalName);
    if (logical.is_absolute() || logical.lexically_normal() != logical ||
        llvm::is_contained(logical, std::filesystem::path("..")))
      return invalid("RTL payload logical name is not canonical");
    semanticInputs.push_back(
        {"inputs/implementation/" + payload.canonicalLogicalName,
         std::move(*contents), implementation.reference(), false});
  }
  if (llvm::none_of(semanticInputs, [](const MaterializedBundleFile &file) {
        return llvm::StringRef(file.relativePath)
            .starts_with("inputs/implementation/");
      }))
    return invalid("RTL representation has no source payload");
  return semanticInputs;
}

ExternalToolInvocationImportExpectation makeMappedRtlImportExpectation(
    const ExternalToolSemanticContract &semanticContract,
    llvm::ArrayRef<MaterializedBundleFile> semanticInputs) {
  ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = semanticContract;
  for (const MaterializedBundleFile &file : semanticInputs) {
    if (!file.sourceArtifact)
      continue;
    expectation.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact,
         computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
             reinterpret_cast<const std::uint8_t *>(file.contents.data()),
             file.contents.size()))});
  }
  expectation.declaredOutputs.push_back(mappedRtlResultPath.str());
  return expectation;
}

} // namespace

llvm::Expected<MappedRtlFactsOrUnsupported>
deriveMappedRtlInvocationFacts(const MappedRtlExecutionClosure &closure,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  if (llvm::Error error = evaluation::models::validateMappedRtlSimulatorBinding(
          closure.simulatorBinding))
    return std::move(error);

  auto implementation = importCachedOne<FinalizedHardwareImplementation>(
      closure.hardwareImplementation, artifacts, &blobs, [&]() {
        return importHardwareImplementation(closure.hardwareImplementation,
                                            artifacts, blobs);
      });
  if (!implementation)
    return implementation.takeError();
  const HardwareImplementation &hardware = (*implementation)->implementation();
  const ImplementationRepresentationRoot &representation =
      hardware.representationRoot();
  auto rtlFormat = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::SystemVerilogRtl);
  if (!rtlFormat)
    return rtlFormat.takeError();
  if (representation.variant != RepresentationRootVariant::Rtl ||
      representation.stage || representation.formatRef != *rtlFormat ||
      representation.top.kind != RepresentationObjectKind::Module)
    return unsupported();
  if (!hardware.memoryMacroBindings().empty() ||
      !hardware.externalImplementationBindings().empty())
    return unsupported();
  auto representationIndex = importCachedOne<RepresentationIndex>(
      closure.hardwareImplementation, artifacts, &blobs,
      [&]() { return indexRepresentationRoot(representation, blobs); });
  if (!representationIndex)
    return representationIndex.takeError();
  if (!(*representationIndex)->unresolvedExternalDefinitions().empty())
    return unsupported();

  auto deployment = importCachedOne<deployment::FinalizedDeployment>(
      closure.deployment, artifacts, &blobs, [&]() {
        return deployment::importDeployment(closure.deployment, artifacts,
                                            blobs);
      });
  if (!deployment)
    return deployment.takeError();
  auto inputs = importCachedSpatialInputs(closure.workload,
                                          closure.runtimeInput, artifacts);
  if (!inputs)
    return inputs.takeError();
  const auto *workload = (*inputs)->workload.spatial();
  const auto *runtime = (*inputs)->runtimeInput.spatial();
  if (!workload || !runtime)
    return invalid("mapped RTL provider requires Spatial inputs");
  auto selection = deployment::resolveDeploymentSpatialLaunchSelection(
      **deployment, workload->launchRef, workload->denseCoordinates, artifacts,
      blobs);
  if (!selection)
    return selection.takeError();
  if (selection->hardwareImplementation != (*implementation)->reference() ||
      selection->dataflow.artifact != (*inputs)->dataflow.identity())
    return invalid("Deployment selection disagrees with the exact Request");
  auto mapping = importCachedOne<mapping::FinalizedSpatialMapping>(
      selection->spatialMapping, artifacts, nullptr, [&]() {
        return mapping::importSpatialMapping(selection->spatialMapping,
                                             artifacts);
      });
  if (!mapping)
    return mapping.takeError();
  if ((*mapping)->view().identity() != selection->context.spatialMapping ||
      (*mapping)->view().dataflowIdentity() != (*inputs)->dataflow.identity())
    return invalid("Deployment selection names a foreign SpatialMapping");
  auto fabricSystem = importCachedOne<fabric::FinalizedFabricRoot>(
      hardware.fabric(), artifacts, nullptr, [&]() {
        return fabric::importEntireFabricRoot(hardware.fabric(), artifacts);
      });
  if (!fabricSystem)
    return fabricSystem.takeError();
  auto system = fabric::requireSystemRoot((*fabricSystem)->view());
  if (!system)
    return system.takeError();
  if ((*mapping)->view().fabricIdentity() == system->artifact().identity())
    return invalid("SpatialMapping unexpectedly targets the System root");
  const fabric::SpatialCoreOccurrenceRef spatialCore{
      selection->context.accCore};
  const auto moduleTarget =
      system->spatialCoreTarget(selection->context.accCore);
  if (!moduleTarget || moduleTarget->dependencyOrdinal >=
                           system->artifact().importedModules().size())
    return invalid("selected AccCore has no imported SpatialCore Module");
  const fabric::FabricArtifactView &module =
      system->artifact().importedModules()[moduleTarget->dependencyOrdinal];
  if (module.identity() != (*mapping)->view().fabricIdentity() ||
      module.moduleRootTemplate() !=
          std::optional<fabric::FabricModuleTemplateRef>{moduleTarget->target})
    return invalid("SpatialMapping targets a foreign imported Module");
  auto abi = importCachedOne<FinalizedConfigurationABI>(
      hardware.configurationAbi(), artifacts, nullptr, [&]() {
        return importConfigurationABI(hardware.configurationAbi(), artifacts);
      });
  if (!abi)
    return abi.takeError();
  auto rtlModuleGraph = hardware::rtl::projectPortableSpatialCoreRtlModuleGraph(
      **abi, **implementation);
  if (!rtlModuleGraph)
    return rtlModuleGraph.takeError();
  if (!*rtlModuleGraph)
    return unsupported();

  auto dataflow = (*inputs)->dataflow.view();
  if (!dataflow)
    return dataflow.takeError();
  auto selectedGraph = dataflow->resolve(workload->launchRef);
  if (!selectedGraph)
    return selectedGraph.takeError();
  auto shapes = sim::projectSpatialSimulationBoundaryShapes(
      *dataflow, workload->launchRef);
  if (!shapes)
    return shapes.takeError();

  std::optional<InputTokenStream> start;
  std::vector<std::optional<InputTokenStream>> values(
      shapes->valueInputs.size());
  std::vector<std::optional<InputTokenStream>> streams(
      shapes->streamInputs.size());
  std::vector<std::optional<OutputTokenStream>> valuePorts(
      shapes->valueResults.size());
  std::vector<std::optional<OutputTokenStream>> streamPorts(
      shapes->streamOutputs.size());
  std::map<std::uint64_t, TransportPort> completionPorts;

  std::vector<bool> requiredValues(values.size(), false);
  for (std::uint64_t ordinal = 0; ordinal != values.size(); ++ordinal) {
    auto consumers =
        dataflow->graphConsumers(dataflow::CanonicalGraphProducerEndpointRef{
            dataflow::GraphIngressTokenRef{
                dataflow::GraphValueInputTokenRef{*selectedGraph, ordinal}}});
    if (!consumers)
      return consumers.takeError();
    requiredValues[ordinal] = !consumers->empty();
  }
  std::vector<bool> requiredStreams(streams.size(), false);
  for (std::uint64_t ordinal = 0; ordinal != streams.size(); ++ordinal) {
    auto consumers =
        dataflow->graphConsumers(dataflow::CanonicalGraphProducerEndpointRef{
            dataflow::GraphIngressTokenRef{
                dataflow::GraphStreamInputTokenRef{*selectedGraph, ordinal}}});
    if (!consumers)
      return consumers.takeError();
    requiredStreams[ordinal] = !consumers->empty();
  }

  for (const auto indexedRoute :
       llvm::enumerate((*mapping)->view().routeTrees())) {
    const mapping::SpatialRouteTreeView &route = indexedRoute.value();
    if (const auto *ingress =
            std::get_if<dataflow::GraphIngressTokenRef>(&route.logicalNet)) {
      auto classified = classifyIngress(*ingress, *selectedGraph);
      if (classified) {
        std::uint32_t width = 0;
        if (classified->kind == IngressKind::Value) {
          if (classified->ordinal >= shapes->valueInputs.size())
            return invalid("mapped value input ordinal is out of range");
          auto projected =
              semanticWidth(shapes->valueInputs[classified->ordinal]);
          if (!projected)
            return projected.takeError();
          width = *projected;
        } else if (classified->kind == IngressKind::Stream) {
          if (classified->ordinal >= shapes->streamInputs.size())
            return invalid("mapped stream input ordinal is out of range");
          auto projected =
              semanticWidth(shapes->streamInputs[classified->ordinal]);
          if (!projected)
            return projected.takeError();
          width = *projected;
        }
        auto port = projectInternalEndpoint(
            hardware, **representationIndex, *system, *moduleTarget, module,
            (*mapping)->view(), spatialCore, route.rootEndpoint,
            fabric::FabricPortDirection::Input, width, indexedRoute.index(), 0);
        if (!port)
          return port.takeError();
        if (classified->kind == IngressKind::Start) {
          if (start)
            return invalid("graph start has multiple mapped input ports");
          start = InputTokenStream{std::move(*port), 1, {}, std::nullopt};
        } else if (classified->kind == IngressKind::Value) {
          if (values[classified->ordinal])
            return invalid("graph value input has multiple mapped ports");
          auto sequence =
              valueInputSequence(*workload, *runtime, classified->ordinal);
          if (!sequence)
            return sequence.takeError();
          auto tokens = packSequence(**sequence,
                                     shapes->valueInputs[classified->ordinal]);
          if (!tokens)
            return tokens.takeError();
          values[classified->ordinal] =
              InputTokenStream{std::move(*port), (*sequence)->tokenCount,
                               std::move(*tokens), std::nullopt};
        } else {
          if (streams[classified->ordinal])
            return invalid("graph stream input has multiple mapped ports");
          const auto &sequence = runtime->runtimeStreams[classified->ordinal];
          auto tokens = packSequence(sequence.values,
                                     shapes->streamInputs[classified->ordinal]);
          if (!tokens)
            return tokens.takeError();
          streams[classified->ordinal] =
              InputTokenStream{std::move(*port), sequence.values.tokenCount,
                               std::move(*tokens), classified->ordinal};
        }
      }
    }
    for (const mapping::SpatialRouteSinkView &sink : route.sinks) {
      const auto *egress =
          std::get_if<dataflow::GraphEgressTokenRef>(&sink.sink);
      if (!egress)
        continue;
      auto classified = classifyEgress(*egress, *selectedGraph);
      if (!classified)
        continue;
      if (sink.nodeOrdinal >= route.nodes.size())
        return invalid("mapped graph output names an absent route node");
      std::uint32_t width = 0;
      if (classified->kind == EgressKind::Value) {
        if (classified->ordinal >= shapes->valueResults.size())
          return invalid("mapped value result ordinal is out of range");
        auto projected =
            semanticWidth(shapes->valueResults[classified->ordinal]);
        if (!projected)
          return projected.takeError();
        width = *projected;
      } else if (classified->kind == EgressKind::Stream) {
        if (classified->ordinal >= shapes->streamOutputs.size())
          return invalid("mapped stream output ordinal is out of range");
        auto projected =
            semanticWidth(shapes->streamOutputs[classified->ordinal]);
        if (!projected)
          return projected.takeError();
        width = *projected;
      }
      auto port = projectInternalEndpoint(
          hardware, **representationIndex, *system, *moduleTarget, module,
          (*mapping)->view(), spatialCore,
          route.nodes[sink.nodeOrdinal].endpoint,
          fabric::FabricPortDirection::Output, width, indexedRoute.index(),
          sink.nodeOrdinal);
      if (!port)
        return port.takeError();
      if (classified->kind == EgressKind::Value) {
        if (valuePorts[classified->ordinal])
          return invalid("graph value result has multiple mapped ports");
        valuePorts[classified->ordinal] =
            OutputTokenStream{std::move(*port), width};
      } else if (classified->kind == EgressKind::Stream) {
        if (streamPorts[classified->ordinal])
          return invalid("graph stream output has multiple mapped ports");
        streamPorts[classified->ordinal] =
            OutputTokenStream{std::move(*port), width};
      } else if (!completionPorts.emplace(classified->ordinal, std::move(*port))
                      .second) {
        return invalid("graph completion has multiple mapped ports");
      }
    }
  }
  std::size_t missingValues = 0;
  for (std::size_t ordinal = 0; ordinal != values.size(); ++ordinal)
    missingValues += requiredValues[ordinal] && !values[ordinal];
  std::size_t missingStreams = 0;
  for (std::size_t ordinal = 0; ordinal != streams.size(); ++ordinal)
    missingStreams += requiredStreams[ordinal] && !streams[ordinal];
  if (!start || missingValues != 0 || missingStreams != 0 ||
      completionPorts.empty())
    return invalid("SpatialMapping omits required graph boundaries "
                   "(start=" +
                   llvm::Twine(start.has_value() ? 1 : 0) +
                   ", missing_value_inputs=" + llvm::Twine(missingValues) +
                   "/" + llvm::Twine(values.size()) +
                   ", missing_stream_inputs=" + llvm::Twine(missingStreams) +
                   "/" + llvm::Twine(streams.size()) + ", completion_ports=" +
                   llvm::Twine(completionPorts.size()) + ")");
  for (const auto &[ordinal, port] : completionPorts)
    if (ordinal >= completionPorts.size())
      return invalid("graph completion ordinals are not dense");

  std::vector<OutputTokenStream> selectedValuePorts;
  for (std::uint64_t ordinal : workload->observableContract.valueResults) {
    if (ordinal >= valuePorts.size() || !valuePorts[ordinal])
      return invalid("observable value result has no mapped port");
    selectedValuePorts.push_back(*valuePorts[ordinal]);
  }
  std::vector<OutputTokenStream> selectedStreamPorts;
  for (std::uint64_t ordinal : workload->observableContract.streamOutputs) {
    if (ordinal >= streamPorts.size() || !streamPorts[ordinal])
      return invalid("observable stream output has no mapped port");
    selectedStreamPorts.push_back(*streamPorts[ordinal]);
  }

  auto rootPorts = projectRootPorts(**representationIndex);
  if (!rootPorts)
    return rootPorts.takeError();
  auto clockReset = projectClockResetPorts(hardware, *system);
  if (!clockReset)
    return clockReset.takeError();
  std::uint64_t selectedPeriod = 0;
  auto selectedClock =
      selectedClockPort(hardware, *system, spatialCore, selectedPeriod);
  if (!selectedClock)
    return selectedClock.takeError();
  auto configurationPrograms = projectConfigurationPrograms(
      *selection, **deployment, hardware, **abi, spatialCore, artifacts);
  if (!configurationPrograms)
    return configurationPrograms.takeError();
  auto memoryProjection = projectRuntimeMemory(
      hardware, **representationIndex, *system, *moduleTarget, module,
      spatialCore, (*mapping)->view(), *dataflow, *workload, *runtime);
  if (!memoryProjection)
    return memoryProjection.takeError();
  if (!*memoryProjection)
    return unsupported();

  auto semanticInputs = materializeMappedRtlSemanticInputs(
      closure.workload, closure.runtimeInput, **implementation, **deployment,
      **inputs, *selection, **mapping, **abi, **fabricSystem,
      *configurationPrograms, artifacts, blobs);
  if (!semanticInputs)
    return semanticInputs.takeError();
  std::vector<std::string> configurationProgramPaths;
  for (std::size_t ordinal = 0; ordinal != configurationPrograms->size();
       ++ordinal)
    configurationProgramPaths.push_back(configurationProgramPath(ordinal));
  std::vector<std::string> rtlPaths;
  for (const MaterializedBundleFile &file : *semanticInputs)
    if (llvm::StringRef(file.relativePath)
            .starts_with("inputs/implementation/"))
      rtlPaths.push_back(file.relativePath);
  std::string top = representation.top.canonicalName;

  std::vector<InputTokenStream> denseValues;
  for (auto &entry : values)
    if (entry)
      denseValues.push_back(std::move(*entry));
  std::vector<InputTokenStream> denseStreams;
  for (auto &entry : streams)
    if (entry)
      denseStreams.push_back(std::move(*entry));
  std::vector<TransportPort> denseCompletions;
  for (auto &[ordinal, port] : completionPorts) {
    (void)ordinal;
    denseCompletions.push_back(std::move(port));
  }

  return MappedRtlFactsOrUnsupported{
      MappedRtlInvocationFacts{closure.simulatorBinding,
                               closure.semanticContract,
                               std::move(*semanticInputs),
                               std::move(rtlPaths),
                               {},
                               std::move(top),
                               std::move(**rtlModuleGraph),
                               std::move(*rootPorts),
                               std::move(clockReset->first),
                               std::move(clockReset->second),
                               std::move(*selectedClock),
                               selectedPeriod,
                               std::move(*configurationPrograms),
                               std::move(configurationProgramPaths),
                               std::move(start),
                               std::move(denseValues),
                               std::move(denseStreams),
                               std::move(selectedValuePorts),
                               std::move(selectedStreamPorts),
                               std::move(denseCompletions),
                               std::move((*memoryProjection)->images),
                               std::move((*memoryProjection)->ports),
                               std::move((*memoryProjection)->observations),
                               (*memoryProjection)->addressArithmetic,
                               0}};
}

namespace {

struct MappedRtlLaunchClosure final {
  std::shared_ptr<const deployment::FinalizedDeployment> deployment;
  std::shared_ptr<const sim::ImportedSpatialSimulationInputs> inputs;
  deployment::DeploymentSpatialLaunchSelection selection;
  std::shared_ptr<const mapping::FinalizedSpatialMapping> mapping;
};

llvm::Expected<MappedRtlLaunchClosure>
importMappedRtlLaunchClosure(const MappedRtlExecutionClosure &closure,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  auto deployment = importCachedOne<deployment::FinalizedDeployment>(
      closure.deployment, artifacts, &blobs, [&]() {
        return deployment::importDeployment(closure.deployment, artifacts,
                                            blobs);
      });
  if (!deployment)
    return deployment.takeError();
  auto inputs = importCachedSpatialInputs(closure.workload,
                                          closure.runtimeInput, artifacts);
  if (!inputs)
    return inputs.takeError();
  const auto *workload = (*inputs)->workload.spatial();
  if (!workload || !(*inputs)->runtimeInput.spatial())
    return invalid("mapped RTL provider requires Spatial inputs");
  auto selection = deployment::resolveDeploymentSpatialLaunchSelection(
      **deployment, workload->launchRef, workload->denseCoordinates, artifacts,
      blobs);
  if (!selection)
    return selection.takeError();
  if (selection->dataflow.artifact != (*inputs)->dataflow.identity())
    return invalid("Deployment selection disagrees with the exact Request");
  auto mapping = importCachedOne<mapping::FinalizedSpatialMapping>(
      selection->spatialMapping, artifacts, nullptr, [&]() {
        return mapping::importSpatialMapping(selection->spatialMapping,
                                             artifacts);
      });
  if (!mapping)
    return mapping.takeError();
  if ((*mapping)->view().identity() != selection->context.spatialMapping ||
      (*mapping)->view().dataflowIdentity() != (*inputs)->dataflow.identity())
    return invalid("Deployment selection names a foreign SpatialMapping");
  return MappedRtlLaunchClosure{std::move(*deployment), std::move(*inputs),
                                std::move(*selection), std::move(*mapping)};
}

} // namespace

llvm::Expected<ExternalToolInvocationImportExpectation>
deriveMappedRtlImportExpectation(
    const MappedRtlExecutionClosure &requestClosure,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto implementation = importCachedOne<FinalizedHardwareImplementation>(
      requestClosure.hardwareImplementation, artifacts, &blobs, [&]() {
        return importHardwareImplementation(
            requestClosure.hardwareImplementation, artifacts, blobs);
      });
  if (!implementation)
    return implementation.takeError();
  auto closure = importMappedRtlLaunchClosure(requestClosure, artifacts, blobs);
  if (!closure)
    return closure.takeError();
  if (closure->selection.hardwareImplementation !=
      (*implementation)->reference())
    return invalid("Deployment selection disagrees with the exact Request");
  const HardwareImplementation &hardware = (*implementation)->implementation();
  auto abi = importCachedOne<FinalizedConfigurationABI>(
      hardware.configurationAbi(), artifacts, nullptr, [&]() {
        return importConfigurationABI(hardware.configurationAbi(), artifacts);
      });
  if (!abi)
    return abi.takeError();
  auto fabricSystem = importCachedOne<fabric::FinalizedFabricRoot>(
      hardware.fabric(), artifacts, nullptr, [&]() {
        return fabric::importEntireFabricRoot(hardware.fabric(), artifacts);
      });
  if (!fabricSystem)
    return fabricSystem.takeError();
  auto configurationPrograms = projectConfigurationPrograms(
      closure->selection, *closure->deployment, hardware, **abi,
      fabric::SpatialCoreOccurrenceRef{closure->selection.context.accCore},
      artifacts);
  if (!configurationPrograms)
    return configurationPrograms.takeError();
  auto semanticInputs = materializeMappedRtlSemanticInputs(
      requestClosure.workload, requestClosure.runtimeInput, **implementation,
      *closure->deployment, *closure->inputs, closure->selection,
      *closure->mapping, **abi, **fabricSystem, *configurationPrograms,
      artifacts, blobs);
  if (!semanticInputs)
    return semanticInputs.takeError();
  return makeMappedRtlImportExpectation(requestClosure.semanticContract,
                                        *semanticInputs);
}

llvm::Expected<MappedRtlObservationFacts>
deriveMappedRtlObservationFacts(const MappedRtlExecutionClosure &requestClosure,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs) {
  auto closure = importMappedRtlLaunchClosure(requestClosure, artifacts, blobs);
  if (!closure)
    return closure.takeError();
  const auto *workload = closure->inputs->workload.spatial();
  const auto *runtime = closure->inputs->runtimeInput.spatial();
  if (!workload || !runtime)
    return invalid("mapped RTL provider requires Spatial inputs");
  auto dataflow = closure->inputs->dataflow.view();
  if (!dataflow)
    return dataflow.takeError();
  auto images = projectRuntimeMemoryImages(*runtime);
  if (!images)
    return images.takeError();
  auto bindingIds =
      projectExactLaunchMemoryBindings(closure->mapping->view(), *workload);
  if (!bindingIds)
    return bindingIds.takeError();
  if (!*images || !*bindingIds)
    return invalid("prepared invocation is no longer supported");
  auto observations = projectRuntimeMemoryObservationPlans(
      closure->mapping->view(), *dataflow, *workload, *runtime, **bindingIds);
  if (!observations)
    return observations.takeError();
  if (!*observations)
    return invalid("prepared invocation is no longer supported");
  return MappedRtlObservationFacts{std::move(closure->inputs),
                                   std::move(**images),
                                   std::move(**observations)};
}

} // namespace loom::eda::open_source::detail
