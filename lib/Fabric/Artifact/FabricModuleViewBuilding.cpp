#include "FabricModuleViewBuilding.h"

#include "../Identity/FabricArtifactViewInternal.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "FabricCanonicalLabeling.h"
#include "FabricCapabilityProjection.h"
#include "FabricMemoryEngineTemplate.h"
#include "FabricOperationTransport.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <system_error>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(llvm::ArrayRef<std::int8_t> bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (std::int8_t byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

} // namespace

llvm::Error setFabricTransportEndpoints(FabricNestedOwnerViewData &owner,
                                        ArrayRef<Type> inputs,
                                        ArrayRef<Type> outputs) {
  setFabricPortInventories(owner, inputs.size(), outputs.size());
  owner.transportEndpoints.clear();
  owner.transportEndpoints.reserve(inputs.size() + outputs.size());
  auto append = [&](Type type, FabricPortDirection direction) -> llvm::Error {
    auto encoded = ::fabric::encodeFabricTransportType(type);
    if (!encoded)
      return encoded.takeError();
    owner.transportEndpoints.push_back({direction, std::move(*encoded)});
    return llvm::Error::success();
  };
  for (Type type : inputs)
    if (llvm::Error error = append(type, FabricPortDirection::Input))
      return error;
  for (Type type : outputs)
    if (llvm::Error error = append(type, FabricPortDirection::Output))
      return error;
  return llvm::Error::success();
}

std::vector<std::uint64_t> emptyFabricInventories() {
  return std::vector<std::uint64_t>(fabricClosedBound(FabricInventoryKind{}),
                                    0);
}

FabricFuNodeKind classifyFabricFuNode(Operation *operation) {
  if (isa<::fabric::MuxOp>(operation))
    return FabricFuNodeKind::Mux;
  if (isa<::fabric::DemuxOp>(operation))
    return FabricFuNodeKind::Demux;
  return FabricFuNodeKind::Op;
}

void setFabricPortInventories(FabricNestedOwnerViewData &owner,
                              std::uint64_t inputs, std::uint64_t outputs) {
  owner.inventoryCounts = emptyFabricInventories();
  owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::InputPort)] = inputs;
  owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::OutputPort)] = outputs;
}

llvm::Error
setFabricOperationTransportEndpoints(Operation *operation,
                                     FabricNestedOwnerViewData &owner) {
  auto types = resolveFabricOperationTransportTypes(operation);
  if (!types)
    return types.takeError();
  return setFabricTransportEndpoints(owner, types->inputs, types->outputs);
}

llvm::Error populateFabricMemoryView(::fabric::MemOp memory,
                                     FabricEntityViewData &entity) {
  auto type = resolveFabricMemoryFunctionType(memory);
  if (!type)
    return type.takeError();

  llvm::SmallVector<Type> tokenInputTypes;
  llvm::SmallVector<Type> tokenOutputTypes;
  for (Type input : type->getInputs())
    if (!isa<MemRefType>(input))
      tokenInputTypes.push_back(input);
  for (Type output : type->getResults())
    if (!isa<MemRefType>(output))
      tokenOutputTypes.push_back(output);
  if (llvm::Error error = setFabricTransportEndpoints(
          entity.owner, tokenInputTypes, tokenOutputTypes))
    return error;

  ::fabric::MemoryContractAttr contract = memory.getMemoryContract();
  entity.owner.memoryEndpoints.clear();
  for (Type input : type->getInputs()) {
    if (!isa<MemRefType>(input))
      continue;
    auto encoded = projectMemoryEndpointType(input);
    if (!encoded)
      return encoded.takeError();
    entity.owner.memoryEndpoints.push_back(
        {FabricMemoryEndpointRole::Manager, std::move(*encoded)});
  }
  for (Type output : type->getResults()) {
    if (!isa<MemRefType>(output))
      continue;
    auto encoded = projectMemoryEndpointType(output);
    if (!encoded)
      return encoded.takeError();
    entity.owner.memoryEndpoints.push_back(
        {FabricMemoryEndpointRole::Subordinate, std::move(*encoded)});
  }

  auto connectivity = ::fabric::decodeMemoryConnectivityContractRecord(
      unsignedBytes(contract.getConnectivity().getRecord().asArrayRef()));
  if (!connectivity)
    return connectivity.takeError();
  entity.memoryConnectivity = std::move(*connectivity);

  if (::fabric::LocalMemoryServiceAttr local = contract.getLocalService()) {
    auto service = ::fabric::decodeMemoryServiceContractRecord(
        unsignedBytes(local.getServiceContract().getRecord().asArrayRef()),
        memory.getContext(), ::fabric::MemoryServiceOwnerKind::Local);
    if (!service)
      return service.takeError();
    FabricNestedOwnerViewData owner;
    owner.inventoryCounts = emptyFabricInventories();
    owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::MemoryServiceRegion)] = service->regions().size();
    owner.resourceContract = service->resourceContract();
    entity.localMemoryService =
        FabricLocalMemoryServiceViewData{std::move(owner), std::move(*service)};
  }

  auto derived = deriveFabricMemoryEngineTemplate(memory);
  if (!derived)
    return derived.takeError();
  if (!*derived)
    return llvm::Error::success();
  entity.memoryEngineTemplateProjection = (**derived).canonicalBytes;
  FabricMemoryEngineTemplateRecord &engine = (**derived).record;
  entity.memorySchedule = engine.schedule;
  entity.memoryResidentContextCount = engine.residentContextCount;
  entity.owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::MemoryOperationPort)] = engine.operationPorts.size();
  entity.memoryOperationPorts.reserve(engine.operationPorts.size());
  for (::fabric::MemoryOperationPortRecord &record : engine.operationPorts) {
    FabricNestedOwnerViewData owner;
    owner.inventoryCounts = emptyFabricInventories();
    owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::MemoryCapabilityAlternative)] =
        record.capabilityAlternatives().size();
    if (entity.memoryResidentContextCount)
      owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::MemoryOperationContext)] =
          *entity.memoryResidentContextCount;
    owner.resourceContract = record.resourceContract();
    entity.memoryOperationPorts.push_back(
        {std::move(owner), std::move(record)});
  }
  return llvm::Error::success();
}

llvm::Error
appendFabricModuleMemoryConnections(const FabricCanonicalLabeling &labeling,
                                    FabricArtifactViewData &data) {
  llvm::DenseMap<Operation *, const FabricEntityCarrier *> carrierByOperation;
  for (const FabricEntityCarrier &carrier : labeling.carriers)
    if (carrier.op)
      carrierByOperation[carrier.op] = &carrier;

  for (const auto &entry : carrierByOperation) {
    auto requester = dyn_cast<::fabric::MemOp>(entry.first);
    if (!requester ||
        entry.second->kind != FabricEntityKind::FabricMemoryOccurrence)
      continue;
    FabricOrdinal managerOrdinal = 0;
    for (OpOperand &operand : requester->getOpOperands()) {
      if (!isa<MemRefType>(operand.get().getType()))
        continue;
      auto result = dyn_cast<OpResult>(operand.get());
      auto provider = result ? dyn_cast<::fabric::MemOp>(result.getOwner())
                             : ::fabric::MemOp();
      auto providerCarrier =
          provider ? carrierByOperation.find(provider.getOperation())
                   : carrierByOperation.end();
      if (provider && providerCarrier != carrierByOperation.end() &&
          providerCarrier->second->kind ==
              FabricEntityKind::FabricMemoryOccurrence) {
        FabricOrdinal subordinateOrdinal = 0;
        for (Type type : provider.getOperandTypes())
          subordinateOrdinal += isa<MemRefType>(type);
        for (unsigned index = 0; index < result.getResultNumber(); ++index)
          subordinateOrdinal +=
              isa<MemRefType>(provider.getResult(index).getType());
        data.memoryServiceConnections.push_back(
            {{FabricMemoryEndpointOwnerRef::of(
                  FabricMemoryOccurrenceRef(entry.second->id)),
              managerOrdinal},
             {FabricMemoryEndpointOwnerRef::of(
                  FabricMemoryOccurrenceRef(providerCarrier->second->id)),
              subordinateOrdinal}});
      }
      ++managerOrdinal;
    }
  }
  return llvm::Error::success();
}

llvm::Error appendFabricPeSelectorTraversals(FabricArtifactViewData &data) {
  for (FabricEntityId id = 0; id < data.entities.size(); ++id) {
    FabricEntityViewData &fu = data.entities[id];
    if (fu.kind != FabricEntityKind::FabricFuOccurrence)
      continue;
    if (!fu.parentPe || fu.parentPe->id() >= data.entities.size())
      return invalid("an FU occurrence has no valid parent PE");
    const FabricEntityViewData &pe = data.entities[fu.parentPe->id()];
    if (pe.kind != FabricEntityKind::FabricPeOccurrence)
      return invalid("an FU occurrence parent is not a PE");

    const auto peOwner = FabricTransportEndpointOwnerRef::of(*fu.parentPe);
    const auto fuOwner =
        FabricTransportEndpointOwnerRef::of(FabricFuOccurrenceRef(id));
    for (auto [peOrdinal, peEndpoint] :
         llvm::enumerate(pe.owner.transportEndpoints)) {
      if (peEndpoint.direction != FabricPortDirection::Input)
        continue;
      for (auto [fuOrdinal, fuEndpoint] :
           llvm::enumerate(fu.owner.transportEndpoints)) {
        if (fuEndpoint.direction != FabricPortDirection::Input)
          continue;
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::peSelector(
                *fu.parentPe, {peOwner, peOrdinal}, {fuOwner, fuOrdinal}));
      }
    }
    for (auto [fuOrdinal, fuEndpoint] :
         llvm::enumerate(fu.owner.transportEndpoints)) {
      if (fuEndpoint.direction != FabricPortDirection::Output)
        continue;
      for (auto [peOrdinal, peEndpoint] :
           llvm::enumerate(pe.owner.transportEndpoints)) {
        if (peEndpoint.direction != FabricPortDirection::Output)
          continue;
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::peSelector(
                *fu.parentPe, {fuOwner, fuOrdinal}, {peOwner, peOrdinal}));
      }
    }
  }
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
