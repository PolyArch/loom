#include "Fabric/Artifact/FabricArtifact.h"

#include "../Identity/FabricArtifactViewInternal.h"
#include "Common/ArtifactFinalizer.h"
#include "Fabric/Artifact/FabricClockResetValidation.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/Elaboration.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "FabricArtifactBytecodeInternal.h"
#include "FabricArtifactDependencyClosureInternal.h"
#include "FabricCanonicalLabeling.h"
#include "FabricCapabilityProjection.h"
#include "FabricFuCapabilityDerivation.h"
#include "FabricMemoryEngineTemplate.h"
#include "FabricModuleBoundaryTransport.h"
#include "FabricOperationTransport.h"
#include "FabricResourceContractFinalization.h"
#include "FabricSystemCanonicalLabeling.h"
#include "FabricSystemServiceLegCarrier.h"
#include "FabricSystemValidation.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric {
namespace {

constexpr llvm::StringLiteral canonicalRootName("__loom_fabric_root");
constexpr llvm::StringLiteral systemEntityIdAttrName("entity_id");

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Error ownerUnavailable(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "fabric_artifact_owner_contract_unavailable: " + message);
}

std::vector<std::uint8_t> handshakeSignalKey(const HandshakeSignalRef &signal) {
  std::vector<std::uint8_t> key = canonicalFabricBytes(signal.endpoint);
  key.push_back(static_cast<std::uint8_t>(signal.signal));
  return key;
}

llvm::Error
validateUnconditionalHandshakeClosure(const FabricArtifactView &view) {
  auto arcs = deriveUnconditionalHandshakeDependencyArcs(view);
  if (!arcs)
    return arcs.takeError();

  std::set<std::vector<std::uint8_t>> orderedSignals;
  for (const HandshakeDependencyArc &arc : *arcs) {
    orderedSignals.insert(handshakeSignalKey(arc.source));
    orderedSignals.insert(handshakeSignalKey(arc.destination));
  }
  std::map<std::vector<std::uint8_t>, std::uint32_t> signalOrdinals;
  for (const std::vector<std::uint8_t> &signal : orderedSignals)
    signalOrdinals.emplace(signal,
                           static_cast<std::uint32_t>(signalOrdinals.size()));

  std::vector<std::vector<std::uint32_t>> adjacency(signalOrdinals.size());
  std::vector<std::uint32_t> indegree(signalOrdinals.size(), 0);
  std::set<std::pair<std::uint32_t, std::uint32_t>> uniqueArcs;
  for (const HandshakeDependencyArc &arc : *arcs) {
    const std::uint32_t source =
        signalOrdinals.at(handshakeSignalKey(arc.source));
    const std::uint32_t destination =
        signalOrdinals.at(handshakeSignalKey(arc.destination));
    if (!uniqueArcs.emplace(source, destination).second)
      continue;
    adjacency[source].push_back(destination);
    ++indegree[destination];
  }

  std::vector<std::uint32_t> ready;
  ready.reserve(signalOrdinals.size());
  for (std::uint32_t signal = 0; signal < indegree.size(); ++signal)
    if (indegree[signal] == 0)
      ready.push_back(signal);
  std::size_t retired = 0;
  while (!ready.empty()) {
    const std::uint32_t signal = ready.back();
    ready.pop_back();
    ++retired;
    for (std::uint32_t dependent : adjacency[signal])
      if (--indegree[dependent] == 0)
        ready.push_back(dependent);
  }
  if (retired != signalOrdinals.size())
    return invalid("UnconditionalCombinationalHandshakeCycle");
  return llvm::Error::success();
}

std::vector<std::uint8_t> unsignedBytes(llvm::ArrayRef<std::int8_t> bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (std::int8_t byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Error stripAuthoringState(::fabric::ModuleOp root) {
  static constexpr llvm::StringLiteral softwareConfigurationAttrs[] = {
      "sel",        "discard",   "disconnect",      "bypassed",
      "sw_configs", "pe_enable", "instruction_mem", "per_fu_sw_configs"};

  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    operation->removeAttr(::fabric::kEntityIdAttrName);
    operation->removeAttr(::fabric::kFuTemplateIdAttrName);
    operation->removeAttr(::fabric::kMemoryEngineTemplateIdAttrName);
    if (!isa<::fabric::OpOp>(operation))
      operation->removeAttr(::fabric::kResourceContractRecordAttrName);
    for (llvm::StringLiteral name : softwareConfigurationAttrs)
      operation->removeAttr(name);

    if (auto semantic =
            operation->getAttrOfType<BoolAttr>("coordinates_semantic");
        semantic && semantic.getValue()) {
      result = invalid("authoring coordinates claim semantic authority");
      return WalkResult::interrupt();
    }
    operation->removeAttr("coordinates_semantic");
    operation->removeAttr("visual_layout");
    return WalkResult::advance();
  });
  return result;
}

llvm::Error eraseElaboratedDeclarations(::fabric::ModuleOp root) {
  llvm::SmallVector<Operation *> declarations;
  root->walk<WalkOrder::PostOrder>([&](Operation *operation) {
    if (operation == root.getOperation())
      return;
    auto symbol = dyn_cast<SymbolOpInterface>(operation);
    if (symbol && symbol.getNameAttr())
      declarations.push_back(operation);
  });
  for (Operation *declaration : declarations) {
    for (Value result : declaration->getResults())
      if (!result.use_empty())
        return invalid("an elaborated declaration still has an SSA use");
    declaration->erase();
  }
  bool residualInstance = false;
  root->walk([&](::fabric::InstantiateOp) { residualInstance = true; });
  if (residualInstance)
    return invalid("a fully elaborated Fabric contains fabric.instantiate");
  return llvm::Error::success();
}

llvm::Error reorderCanonicalGraphRegions(
    ::fabric::ModuleOp root,
    llvm::ArrayRef<Operation *> canonicalOperationOrder) {
  llvm::DenseMap<Operation *, std::uint64_t> rank;
  for (auto [index, operation] : llvm::enumerate(canonicalOperationOrder))
    rank[operation] = index;

  llvm::SmallVector<Block *> blocks;
  root->walk([&](Operation *operation) {
    if (!isa<::fabric::ModuleOp, ::fabric::PeOp, ::fabric::FuOp>(operation))
      return;
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        blocks.push_back(&block);
  });

  for (Block *block : blocks) {
    llvm::SmallVector<Operation *> ordered;
    Operation *terminator = nullptr;
    for (Operation &operation : *block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        terminator = &operation;
        continue;
      }
      if (!rank.count(&operation))
        return invalid("canonical operation order omits a graph operation");
      ordered.push_back(&operation);
    }
    llvm::sort(ordered, [&](Operation *left, Operation *right) {
      return rank.lookup(left) < rank.lookup(right);
    });
    for (Operation *operation : ordered) {
      if (terminator)
        operation->moveBefore(terminator);
      else
        operation->moveBefore(block, block->end());
    }
  }
  return llvm::Error::success();
}

std::vector<std::uint64_t> emptyInventories() {
  return std::vector<std::uint64_t>(fabricClosedBound(FabricInventoryKind{}),
                                    0);
}

FabricFuNodeKind fuNodeKind(Operation *operation) {
  if (isa<::fabric::MuxOp>(operation))
    return FabricFuNodeKind::Mux;
  if (isa<::fabric::DemuxOp>(operation))
    return FabricFuNodeKind::Demux;
  return FabricFuNodeKind::Op;
}

void setPortInventories(detail::FabricNestedOwnerViewData &owner,
                        std::uint64_t inputs, std::uint64_t outputs) {
  owner.inventoryCounts = emptyInventories();
  owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::InputPort)] = inputs;
  owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::OutputPort)] = outputs;
}

llvm::Error setTransportEndpoints(detail::FabricNestedOwnerViewData &owner,
                                  ArrayRef<Type> inputs,
                                  ArrayRef<Type> outputs) {
  setPortInventories(owner, inputs.size(), outputs.size());
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

llvm::Error
setOperationTransportEndpoints(Operation *operation,
                               detail::FabricNestedOwnerViewData &owner) {
  auto types = detail::resolveFabricOperationTransportTypes(operation);
  if (!types)
    return types.takeError();
  return setTransportEndpoints(owner, types->inputs, types->outputs);
}

llvm::Error populateMemoryView(::fabric::MemOp memory,
                               detail::FabricEntityViewData &entity) {
  auto type = detail::resolveFabricMemoryFunctionType(memory);
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
  if (llvm::Error error = setTransportEndpoints(entity.owner, tokenInputTypes,
                                                tokenOutputTypes))
    return error;

  ::fabric::MemoryContractAttr contract = memory.getMemoryContract();
  entity.owner.memoryEndpoints.clear();
  for (Type input : type->getInputs()) {
    if (!isa<MemRefType>(input))
      continue;
    auto encoded = detail::projectMemoryEndpointType(input);
    if (!encoded)
      return encoded.takeError();
    entity.owner.memoryEndpoints.push_back(
        {FabricMemoryEndpointRole::Manager, std::move(*encoded)});
  }
  for (Type output : type->getResults()) {
    if (!isa<MemRefType>(output))
      continue;
    auto encoded = detail::projectMemoryEndpointType(output);
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
    detail::FabricNestedOwnerViewData owner;
    owner.inventoryCounts = emptyInventories();
    owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::MemoryServiceRegion)] = service->regions().size();
    owner.resourceContract = service->resourceContract();
    entity.localMemoryService = detail::FabricLocalMemoryServiceViewData{
        std::move(owner), std::move(*service)};
  }

  auto derived = detail::deriveFabricMemoryEngineTemplate(memory);
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
    detail::FabricNestedOwnerViewData owner;
    owner.inventoryCounts = emptyInventories();
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

llvm::Error appendPeSelectorTraversals(detail::FabricArtifactViewData &data) {
  for (FabricEntityId id = 0; id < data.entities.size(); ++id) {
    detail::FabricEntityViewData &fu = data.entities[id];
    if (fu.kind != FabricEntityKind::FabricFuOccurrence)
      continue;
    if (!fu.parentPe || fu.parentPe->id() >= data.entities.size())
      return invalid("an FU occurrence has no valid parent PE");
    const detail::FabricEntityViewData &pe = data.entities[fu.parentPe->id()];
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

llvm::Expected<FabricArtifactView>
buildModuleView(::fabric::ModuleOp root,
                const detail::FabricCanonicalLabeling &labeling,
                const ArtifactIdentity &identity) {
  detail::FabricArtifactViewData data(identity, FabricRootKind::Module);
  data.entities.resize(labeling.carriers.size());

  llvm::DenseMap<Operation *, const detail::FabricEntityCarrier *> carrierByOp;
  for (const detail::FabricEntityCarrier &carrier : labeling.carriers)
    if (carrier.op)
      carrierByOp[carrier.op] = &carrier;

  for (const detail::FabricEntityCarrier &carrier : labeling.carriers) {
    if (carrier.id >= data.entities.size())
      return invalid("canonical Fabric entity IDs are not dense");
    detail::FabricEntityViewData &entity = data.entities[carrier.id];
    entity.kind = carrier.kind;
    entity.owner.inventoryCounts = emptyInventories();
    if (!carrier.op)
      continue;
    const std::uint64_t inputs = carrier.op->getNumOperands();
    const std::uint64_t outputs = carrier.op->getNumResults();
    setPortInventories(entity.owner, inputs, outputs);
    if (detail::projectFabricTransportOwner(carrier.kind, carrier.id))
      if (llvm::Error error =
              setOperationTransportEndpoints(carrier.op, entity.owner))
        return std::move(error);
    if (auto memory = dyn_cast<::fabric::MemOp>(carrier.op))
      if (llvm::Error error = populateMemoryView(memory, entity))
        return std::move(error);
    if (carrier.kind == FabricEntityKind::FabricModuleTemplate) {
      if (llvm::Error error = detail::setModuleBoundaryInventory(root, entity))
        return std::move(error);
    }
    if (carrier.kind == FabricEntityKind::FabricFuOccurrence) {
      auto parent = carrierByOp.find(carrier.op->getParentOp());
      if (parent == carrierByOp.end() ||
          parent->second->kind != FabricEntityKind::FabricPeOccurrence)
        return invalid("an FU occurrence has no owning PE occurrence");
      entity.parentPe = FabricPeOccurrenceRef(parent->second->id);
      auto found = labeling.fuTemplateIdByOccurrence.find(carrier.op);
      if (found == labeling.fuTemplateIdByOccurrence.end())
        return invalid("an FU occurrence has no template relation");
      entity.fuTemplate = FabricFuTemplateRef(found->second);
    }
    if (auto pe = dyn_cast<::fabric::PeOp>(carrier.op)) {
      entity.peSchedule = pe.getSchedule();
      std::uint64_t contextCount = 1;
      if (pe.getSchedule() == ::fabric::Schedule::Temporal) {
        auto count = pe.getNumInstruction();
        if (!count || *count <= 0)
          return invalid("a temporal PE occurrence has no resident contexts");
        contextCount = static_cast<std::uint64_t>(*count);
      } else if (pe.getSchedule() != ::fabric::Schedule::Spatial)
        return invalid("a PE occurrence has an unknown schedule");
      entity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::InstructionContext)] = contextCount;
      entity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::RegisterFifo)] =
          pe.getSchedule() == ::fabric::Schedule::Temporal
              ? pe.getNumRegFifo().value_or(0)
              : 0;
      entity.instructionContexts.resize(contextCount);
      for (detail::FabricNestedOwnerViewData &context :
           entity.instructionContexts)
        context.inventoryCounts = emptyInventories();
    }
    if (carrier.kind == FabricEntityKind::FabricMemoryOccurrence) {
      auto found = labeling.memoryEngineTemplateIdByOccurrence.find(carrier.op);
      if (found != labeling.memoryEngineTemplateIdByOccurrence.end())
        entity.memoryEngineTemplate =
            FabricMemoryEngineTemplateRef(found->second);
    }

    auto contract =
        detail::validateFabricResourceContract(carrier.op, labeling);
    if (!contract)
      return contract.takeError();
    entity.owner.resourceContract = std::move(*contract);

    if (auto boundary = dyn_cast<::fabric::BoundaryOp>(carrier.op))
      entity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::BoundaryOutput)] = boundary.getNumResults();
    if (auto sw = dyn_cast<::fabric::SwitchOp>(carrier.op)) {
      entity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::SwitchInput)] = sw.getNumOperands();
      entity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::SwitchOutput)] = sw.getNumResults();
    }
  }

  for (const detail::FabricFuTemplateCarrier &carrier : labeling.fuTemplates) {
    if (carrier.id >= data.entities.size())
      return invalid("an FU template ID is outside the entity inventory");
    detail::FabricEntityViewData &entity = data.entities[carrier.id];
    auto fu = dyn_cast_or_null<::fabric::FuOp>(carrier.representative);
    if (!fu)
      return invalid("an FU template has no representative definition");
    setPortInventories(entity.owner, fu.getInputs().size(),
                       fu.getOutputs().size());
    entity.owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::FuNode)] = carrier.canonicalNodeOrder.size();
    for (auto [ordinal, operation] :
         llvm::enumerate(carrier.canonicalNodeOrder)) {
      detail::FabricFuNodeViewData node;
      node.kind = fuNodeKind(operation);
      node.owner.inventoryCounts = emptyInventories();
      if (llvm::Error error =
              setOperationTransportEndpoints(operation, node.owner))
        return std::move(error);
      auto contract =
          detail::validateFabricResourceContract(operation, labeling);
      if (!contract)
        return contract.takeError();
      node.owner.resourceContract = std::move(*contract);
      if (auto concrete = dyn_cast<::fabric::OpOp>(operation)) {
        FabricFuTemplateNodeRef reference{
            FabricFuNodeKind::Op, FabricFuTemplateRef(carrier.id), ordinal};
        auto capability =
            detail::resolveFabricOpCapability(concrete, reference, node);
        if (!capability)
          return capability.takeError();
        node.operationCapabilityIndex = entity.operationCapabilities.size();
        entity.operationCapabilities.push_back(std::move(*capability));
      }
      entity.fuNodes.push_back(std::move(node));
    }
    auto templates = detail::deriveFabricFuCapabilityTemplates(
        fu, FabricFuTemplateRef(carrier.id), carrier.canonicalNodeOrder);
    if (!templates)
      return templates.takeError();
    entity.fuCapabilityTemplates = std::move(*templates);
  }

  for (const detail::FabricMemoryEngineTemplateCarrier &carrier :
       labeling.memoryEngineTemplates) {
    if (carrier.id >= data.entities.size())
      return invalid(
          "a memory engine template ID is outside the entity inventory");
    auto memory = dyn_cast_or_null<::fabric::MemOp>(carrier.representative);
    if (!memory)
      return invalid("a memory engine template has no representative engine");
    auto derived = detail::deriveFabricMemoryEngineTemplate(memory);
    if (!derived)
      return derived.takeError();
    if (!*derived)
      return invalid("a memory engine template represents storage-only memory");
    detail::FabricEntityViewData &entity = data.entities[carrier.id];
    entity.memoryEngineTemplateProjection = (**derived).canonicalBytes;
    entity.memoryEngineTemplateRecord = std::move((**derived).record);
  }

  for (const auto &entry : carrierByOp) {
    Operation *destination = entry.first;
    const detail::FabricEntityCarrier &destinationCarrier = *entry.second;
    auto destinationOwner = detail::projectFabricTransportOwner(
        destinationCarrier.kind, destinationCarrier.id);
    if (!destinationOwner)
      continue;
    for (OpOperand &operand : destination->getOpOperands()) {
      Operation *source = operand.get().getDefiningOp();
      auto sourceEntry = carrierByOp.find(source);
      if (!source || sourceEntry == carrierByOp.end())
        continue;
      const detail::FabricEntityCarrier &sourceCarrier = *sourceEntry->second;
      auto sourceOwner = detail::projectFabricTransportOwner(sourceCarrier.kind,
                                                             sourceCarrier.id);
      if (!sourceOwner)
        continue;
      auto result = dyn_cast<OpResult>(operand.get());
      if (!result)
        return invalid("a physical point connection has no source result");
      auto sourceOrdinal = detail::resolveFabricTokenOutputOrdinal(
          source, result.getResultNumber());
      if (!sourceOrdinal)
        return sourceOrdinal.takeError();
      auto destinationOrdinal = detail::resolveFabricTokenInputOrdinal(
          destination, operand.getOperandNumber());
      if (!destinationOrdinal)
        return destinationOrdinal.takeError();
      if (!*sourceOrdinal || !*destinationOrdinal)
        continue;
      data.pointConnections.push_back(FabricPointConnectionPayload{
          FabricTransportEndpointRef{*sourceOwner, **sourceOrdinal},
          FabricTransportEndpointRef{*destinationOwner, **destinationOrdinal}});
    }
  }

  for (const FabricPointConnectionPayload &connection : data.pointConnections)
    data.admittedTraversals.push_back(
        FabricPhysicalTraversalRef::pointConnection(connection.source,
                                                    connection.destination));
  auto moduleCarrier = carrierByOp.find(root.getOperation());
  if (moduleCarrier == carrierByOp.end() ||
      moduleCarrier->second->kind != FabricEntityKind::FabricModuleTemplate)
    return invalid("a finalized Module has no canonical template owner");
  if (llvm::Error error = detail::appendFabricModuleBoundaryTransportRelations(
          root, FabricModuleTemplateRef(moduleCarrier->second->id), carrierByOp,
          data))
    return std::move(error);
  if (llvm::Error error = appendPeSelectorTraversals(data))
    return std::move(error);
  for (const detail::FabricEntityCarrier &carrier : labeling.carriers) {
    if (carrier.kind == FabricEntityKind::FabricPeOccurrence) {
      auto pe = cast<::fabric::PeOp>(carrier.op);
      if (pe.getSchedule() != ::fabric::Schedule::Temporal)
        continue;
      const std::uint32_t registerFifoCount = pe.getNumRegFifo().value_or(0);
      for (std::uint32_t fifo = 0; fifo != registerFifoCount; ++fifo) {
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::peRegisterFifo(
                FabricPeOccurrenceRef(carrier.id), fifo,
                FabricRegisterFifoPathRole::Write));
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::peRegisterFifo(
                FabricPeOccurrenceRef(carrier.id), fifo,
                FabricRegisterFifoPathRole::Read));
      }
    } else if (carrier.kind == FabricEntityKind::FabricFifoOccurrence) {
      auto fifo = cast<::fabric::FifoOp>(carrier.op);
      data.admittedTraversals.push_back(
          FabricPhysicalTraversalRef::fifoTraversal(
              FabricFifoOccurrenceRef(carrier.id),
              FabricFifoTraversalMode::Buffered));
      if (fifo.getBypassable())
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::fifoTraversal(
                FabricFifoOccurrenceRef(carrier.id),
                FabricFifoTraversalMode::Bypass));
    } else if (carrier.kind == FabricEntityKind::FabricBoundaryOccurrence) {
      for (std::uint64_t output = 0; output < carrier.op->getNumResults();
           ++output)
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::boundaryTraversal(
                FabricBoundaryOccurrenceRef(carrier.id), output));
    } else if (carrier.kind == FabricEntityKind::FabricSwitchOccurrence) {
      auto sw = cast<::fabric::SwitchOp>(carrier.op);
      ArrayAttr hardwareParameters = sw.getHwParamsAttr();
      if (!hardwareParameters || hardwareParameters.size() != 1)
        return invalid("a finalized switch has malformed hardware parameters");
      auto hardware = dyn_cast<DictionaryAttr>(hardwareParameters[0]);
      auto connectivity =
          hardware
              ? dyn_cast_or_null<ArrayAttr>(hardware.get("connectivity_table"))
              : ArrayAttr();
      if (!connectivity || connectivity.size() != sw.getNumResults())
        return invalid("a finalized switch has malformed connectivity");
      for (auto [output, rowAttribute] : llvm::enumerate(connectivity)) {
        auto row = dyn_cast<StringAttr>(rowAttribute);
        if (!row || row.getValue().size() != sw.getNumOperands())
          return invalid("a finalized switch has a malformed connectivity row");
        for (auto [position, enabled] : llvm::enumerate(row.getValue())) {
          if (enabled != '1')
            continue;
          const std::uint64_t input = sw.getNumOperands() - 1 - position;
          data.admittedTraversals.push_back(
              FabricPhysicalTraversalRef::switchTraversal(
                  FabricSwitchOccurrenceRef(carrier.id), input, output));
        }
      }
    }
  }
  return detail::buildFabricArtifactView(std::move(data));
}

struct StrictImportResult {
  DecodedFabricArtifact decoded;
  FabricArtifactView view;
  std::vector<detail::FabricModuleBoundaryEndpointViewData>
      moduleBoundaryInputs;
  std::vector<detail::FabricModuleBoundaryEndpointViewData>
      moduleBoundaryOutputs;
};

llvm::Expected<StrictImportResult>
strictImportModule(const ArtifactRootReference &reference,
                   const CanonicalSemanticBytes &canonicalBytes,
                   DecodedFabricArtifact decoded) {
  if (decoded.rootKind != FabricRootKind::Module)
    return invalid("Module importer received the wrong root kind");
  if (!decoded.dependencies.empty())
    return invalid("a fully elaborated Module root has a direct dependency");

  auto parsed =
      detail::parseFabricBytecodeModule(decoded.canonicalMlirBytecode);
  if (!parsed)
    return parsed.takeError();
  ModuleOp module = parsed->module.get();

  if (!llvm::hasSingleElement(module.getBody()->getOperations()))
    return invalid("canonical payload does not contain exactly one root");
  auto root = dyn_cast<::fabric::ModuleOp>(&module.getBody()->front());
  if (!root || root.getSymName() != canonicalRootName)
    return invalid("canonical payload has no canonical Module root");
  bool residualInstance = false;
  root->walk([&](::fabric::InstantiateOp) { residualInstance = true; });
  if (residualInstance)
    return invalid("canonical payload contains fabric.instantiate");
  auto labeling = detail::computeFabricModuleCanonicalLabeling(root);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error =
          detail::validateFabricResourceContracts(root, *labeling))
    return std::move(error);
  for (const detail::FabricEntityCarrier &carrier : labeling->carriers) {
    if (!carrier.op)
      continue;
    auto stored = carrier.op->getAttrOfType<::fabric::EntityIdAttr>(
        ::fabric::kEntityIdAttrName);
    if (!stored || stored.getId() != carrier.id)
      return invalid("canonical payload has a stale entity ID");
    if (carrier.kind == FabricEntityKind::FabricFuOccurrence) {
      auto expected = labeling->fuTemplateIdByOccurrence.find(carrier.op);
      auto templateId = carrier.op->getAttrOfType<::fabric::EntityIdAttr>(
          ::fabric::kFuTemplateIdAttrName);
      if (expected == labeling->fuTemplateIdByOccurrence.end() || !templateId ||
          templateId.getId() != expected->second)
        return invalid("canonical payload has a stale FU template ID");
    }
    if (carrier.kind == FabricEntityKind::FabricMemoryOccurrence) {
      auto expected =
          labeling->memoryEngineTemplateIdByOccurrence.find(carrier.op);
      auto templateId = carrier.op->getAttrOfType<::fabric::EntityIdAttr>(
          ::fabric::kMemoryEngineTemplateIdAttrName);
      if (expected == labeling->memoryEngineTemplateIdByOccurrence.end()) {
        if (templateId)
          return invalid(
              "canonical storage-only memory has an engine template ID");
      } else if (!templateId || templateId.getId() != expected->second) {
        return invalid("canonical payload has a stale memory engine template "
                       "ID");
      }
    }
  }

  auto rewritten = detail::writeCanonicalFabricBytecode(module);
  if (!rewritten)
    return rewritten.takeError();
  if (*rewritten != decoded.canonicalMlirBytecode)
    return invalid("canonical MLIR bytecode is not byte stable");
  detail::FabricEntityViewData boundaryProjection;
  boundaryProjection.owner.inventoryCounts = emptyInventories();
  if (llvm::Error error =
          detail::setModuleBoundaryInventory(root, boundaryProjection))
    return std::move(error);
  auto view = buildModuleView(root, *labeling, reference.artifact);
  if (!view)
    return view.takeError();
  if (llvm::Error error = validateUnconditionalHandshakeClosure(*view))
    return std::move(error);
  return StrictImportResult{
      std::move(decoded), std::move(*view),
      std::move(boundaryProjection.moduleBoundaryInputs),
      std::move(boundaryProjection.moduleBoundaryOutputs)};
}

llvm::Expected<FabricEntityId> canonicalEntityId(Operation *operation) {
  auto attribute =
      operation->getAttrOfType<::fabric::EntityIdAttr>(systemEntityIdAttrName);
  if (!attribute)
    return invalid("canonical System entity has no EntityId");
  return attribute.getId();
}

llvm::Expected<const StrictImportResult *>
resolveImportedModule(llvm::ArrayRef<StrictImportResult> importedModules,
                      const FabricImportedModuleTargetRef &target) {
  if (target.dependencyOrdinal >= importedModules.size())
    return invalid("System field references a dependency outside its table");
  const StrictImportResult &module = importedModules[target.dependencyOrdinal];
  if (llvm::Error error = validateFabricRef(module.view, target.target))
    return std::move(error);
  return &module;
}

llvm::Expected<const detail::FabricModuleBoundaryEndpointViewData *>
resolveImportedModuleBoundary(
    llvm::ArrayRef<StrictImportResult> importedModules,
    const FabricImportedModuleBoundaryEndpointRef &reference) {
  auto module = resolveImportedModule(
      importedModules,
      FabricImportedModuleTargetRef{reference.dependencyOrdinal,
                                    reference.target.module});
  if (!module)
    return module.takeError();
  const auto &endpoints =
      reference.target.direction == FabricPortDirection::Input
          ? (*module)->moduleBoundaryInputs
          : (*module)->moduleBoundaryOutputs;
  if (reference.target.ordinal >= endpoints.size())
    return invalid("ImportedModule boundary endpoint ordinal is out of range");
  return &endpoints[reference.target.ordinal];
}

detail::FabricNestedOwnerViewData instructionCoreView(
    const InstructionCoreMicroarchitecturalRealization &realization) {
  detail::FabricNestedOwnerViewData owner;
  owner.inventoryCounts = emptyInventories();
  owner.resourceContract = realization.resourceContract();
  return owner;
}

llvm::Expected<detail::FabricNestedOwnerViewData>
spatialCoreView(const StrictImportResult &module) {
  detail::FabricNestedOwnerViewData owner;
  owner.inventoryCounts = emptyInventories();
  for (const detail::FabricModuleBoundaryEndpointViewData &endpoint :
       module.moduleBoundaryInputs) {
    if (endpoint.plane == FabricSpatialAttachmentEndpointRef::Plane::Transport)
      owner.transportEndpoints.push_back(
          {FabricPortDirection::Input, endpoint.canonicalType});
    else
      owner.memoryEndpoints.push_back(
          {FabricMemoryEndpointRole::Manager, endpoint.canonicalType});
  }
  for (const detail::FabricModuleBoundaryEndpointViewData &endpoint :
       module.moduleBoundaryOutputs) {
    if (endpoint.plane == FabricSpatialAttachmentEndpointRef::Plane::Transport)
      owner.transportEndpoints.push_back(
          {FabricPortDirection::Output, endpoint.canonicalType});
    else
      owner.memoryEndpoints.push_back(
          {FabricMemoryEndpointRole::Subordinate, endpoint.canonicalType});
  }
  std::uint64_t inputs = 0;
  std::uint64_t outputs = 0;
  for (const detail::FabricTransportEndpointViewData &endpoint :
       owner.transportEndpoints) {
    inputs += endpoint.direction == FabricPortDirection::Input;
    outputs += endpoint.direction == FabricPortDirection::Output;
  }
  setPortInventories(owner, inputs, outputs);
  return owner;
}

llvm::Error validateServiceCapabilityReferences(
    const CanonicalServiceCapabilitySet &capabilities,
    const FabricArtifactView &view) {
  for (const CanonicalServiceCapabilityRecord &capability :
       capabilities.capabilities()) {
    if (llvm::Error error =
            validateFabricRef(view, capability.rate().rateClock()))
      return error;
    if (const auto *bounded = std::get_if<::fabric::BoundedCompletion>(
            &capability.rate().progress()))
      if (llvm::Error error = validateFabricRef(view, bounded->progressClock))
        return error;
    if (const auto *addressed = std::get_if<AddressedMemoryCapabilityDomain>(
            &capability.domain())) {
      if (addressed->consistencyDomain())
        if (llvm::Error error =
                validateFabricRef(view, *addressed->consistencyDomain()))
          return error;
      continue;
    }
    if (const auto *fence =
            std::get_if<FenceCapabilityDomain>(&capability.domain()))
      if (llvm::Error error =
              validateFabricRef(view, fence->consistencyDomain()))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error
validateSystemRelations(::fabric::SystemOp root,
                        const FabricSystemRootView &systemView,
                        llvm::ArrayRef<StrictImportResult> importedModules) {
  if (llvm::Error error = detail::validateInstructionCoreCohort(root))
    return error;
  const FabricArtifactView &view = systemView.artifact();
  const FabricImportBinding binding{view.identity(), FabricRootKind::System};
  std::set<FabricEntityId> externalBoundariesWithEndpoints;
  std::map<FabricEntityId,
           std::set<std::pair<FabricPortDirection, FabricOrdinal>>>
      attachmentCoverage;

  for (Operation &operation : root.getBody().front()) {
    if (auto endpoint =
            dyn_cast<::fabric::SystemServiceEndpointOp>(&operation)) {
      auto owner = decodeSystemServiceEndpointOwnerRef(
          unsignedBytes(endpoint.getOwnerAttr()));
      if (!owner)
        return owner.takeError();
      if (llvm::Error error = validateFabricRef(view, owner->owner()))
        return error;
      if (const auto *boundary =
              std::get_if<ExternalBoundaryRef>(&owner->owner().payload))
        externalBoundariesWithEndpoints.insert(boundary->id());
      auto capabilities = decodeCanonicalServiceCapabilitySet(
          unsignedBytes(endpoint.getCapabilitiesAttr()), root.getContext());
      if (!capabilities)
        return capabilities.takeError();
      if (llvm::Error error =
              validateServiceCapabilityReferences(*capabilities, view))
        return error;
      continue;
    }

    if (auto memory = dyn_cast<::fabric::SystemMemoryServiceOp>(&operation)) {
      auto record = ::fabric::decodeMemoryServiceContractRecord(
          unsignedBytes(memory.getServiceContractAttr().getRecord()),
          root.getContext(), ::fabric::MemoryServiceOwnerKind::System);
      if (!record)
        return record.takeError();
      for (const ::fabric::MemoryServiceCapabilityDeclaration &capability :
           record->capabilities())
        if (const auto *domain = std::get_if<MemoryConsistencyDomainRef>(
                &capability.consistencyBinding))
          if (llvm::Error error = validateFabricRef(view, *domain))
            return error;
      continue;
    }

    if (auto transform =
            dyn_cast<::fabric::SystemServiceTransformOp>(&operation)) {
      auto record = decodeSystemServiceTransformRecord(
          unsignedBytes(transform.getContractAttr()));
      if (!record)
        return record.takeError();
      for (const FabricMemoryEndpointRef &endpoint : record->inputs())
        if (llvm::Error error = validateFabricRef(view, endpoint))
          return error;
      for (const FabricMemoryEndpointRef &endpoint : record->outputs())
        if (llvm::Error error = validateFabricRef(view, endpoint))
          return error;
      if (const auto *coherent =
              std::get_if<CoherentMemoryTransform>(&record->contract())) {
        if (llvm::Error error =
                validateFabricRef(view, coherent->consistencyDomain))
          return error;
        for (const CoherentMemoryRegionCorrespondence &region :
             coherent->regions) {
          if (llvm::Error error = validateFabricRef(view, region.input))
            return error;
          if (llvm::Error error = validateFabricRef(view, region.output))
            return error;
        }
      }
      continue;
    }

    if (auto domain = dyn_cast<::fabric::SystemHardwareDomainOp>(&operation)) {
      auto record = decodeHardwareDomainContractRecord(
          unsignedBytes(domain.getContractAttr()));
      if (!record)
        return record.takeError();
      auto id = canonicalEntityId(domain);
      if (!id)
        return id.takeError();
      for (const FabricInventoryOwnerRef &member : record->members()) {
        if (llvm::Error error = validateFabricRef(view, member))
          return error;
      }
      if (const auto *reset =
              std::get_if<ResetDomainContractRecord>(&record->contract())) {
        if (reset->synchronousTo())
          if (llvm::Error error =
                  validateFabricRef(view, *reset->synchronousTo()))
            return error;
      } else if (const auto *consistency =
                     std::get_if<::fabric::MemoryConsistencyContract>(
                         &record->contract())) {
        if (llvm::Error error =
                ::fabric::validateMemoryConsistencyContractReferences(
                    *consistency, view, binding))
          return error;
      }
      continue;
    }

    if (auto resource =
            dyn_cast<::fabric::SystemTransportResourceOp>(&operation)) {
      if (DenseI8ArrayAttr crossing = resource.getClockCrossingAttr()) {
        auto record =
            decodeClockCrossingContractRecord(unsignedBytes(crossing));
        if (!record)
          return record.takeError();
        if (llvm::Error error =
                validateFabricRef(view, record->transferPattern()))
          return error;
        auto id = canonicalEntityId(resource);
        if (!id)
          return id.takeError();
        if (record->transferPattern().resource !=
            SystemTransportResourceRef(*id))
          return invalid("clock crossing selects a foreign transfer pattern");
        if (llvm::Error error = validateFabricRef(view, record->sourceClock()))
          return error;
        if (llvm::Error error =
                validateFabricRef(view, record->destinationClock()))
          return error;
      }
      continue;
    }

    if (auto pattern =
            dyn_cast<::fabric::SystemTransferPatternOp>(&operation)) {
      auto record = decodeSystemTransferPatternRecord(
          unsignedBytes(pattern.getContractAttr()));
      if (!record)
        return record.takeError();
      if (llvm::Error error = validateFabricRef(view, record->pattern()))
        return error;
      if (llvm::Error error = validateFabricRef(view, record->ingress()))
        return error;
      if (view.transportEndpointDirection(record->ingress()) !=
          FabricPortDirection::Input)
        return invalid("transfer-pattern ingress is not an input endpoint");
      for (const FabricTransportEndpointRef &egress : record->egresses()) {
        if (llvm::Error error = validateFabricRef(view, egress))
          return error;
        if (view.transportEndpointDirection(egress) !=
            FabricPortDirection::Output)
          return invalid("transfer-pattern egress is not an output endpoint");
      }
      if (llvm::Error error = validateFabricRef(view, record->usePattern()))
        return error;
      if (record->usePattern().owner.catalog() !=
          FabricInventoryOwnerRef::of(record->pattern().resource))
        return invalid("transfer pattern selects a foreign UsePattern");
      continue;
    }

    if (auto attachment =
            dyn_cast<::fabric::SystemSpatialAttachmentOp>(&operation)) {
      auto moduleEndpoint = decodeFabricImportedModuleBoundaryEndpointRef(
          unsignedBytes(attachment.getModuleEndpointAttr()));
      if (!moduleEndpoint)
        return moduleEndpoint.takeError();
      auto moduleRecord =
          resolveImportedModuleBoundary(importedModules, *moduleEndpoint);
      if (!moduleRecord)
        return moduleRecord.takeError();
      auto spatialEndpoint = decodeFabricSpatialAttachmentEndpointRef(
          unsignedBytes(attachment.getSpatialEndpointAttr()));
      if (!spatialEndpoint)
        return spatialEndpoint.takeError();

      AccCoreOccurrenceRef core;
      FabricOrdinal occurrenceOrdinal = 0;
      llvm::ArrayRef<std::uint8_t> occurrenceType;
      if (const FabricTransportEndpointRef *transport =
              spatialEndpoint->transport()) {
        if (transport->owner.kind() !=
            FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
          return invalid("attachment token endpoint is not SpatialCore-owned");
        core =
            std::get<SpatialCoreOccurrenceRef>(transport->owner.payload).core;
        occurrenceOrdinal = transport->ordinal;
        if (llvm::Error error = validateFabricRef(view, *transport))
          return error;
        occurrenceType = view.transportEndpointType(*transport);
      } else {
        const FabricMemoryEndpointRef &memory = *spatialEndpoint->memory();
        if (memory.owner.kind() !=
            FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
          return invalid("attachment memory endpoint is not SpatialCore-owned");
        core = std::get<SpatialCoreOccurrenceRef>(memory.owner.payload).core;
        occurrenceOrdinal = memory.ordinal;
        if (llvm::Error error = validateFabricRef(view, memory))
          return error;
        occurrenceType = view.memoryEndpointType(memory);
      }
      std::optional<FabricImportedModuleTargetRef> target =
          systemView.spatialCoreTarget(core);
      if (!target)
        return invalid("attachment names an unknown AccCore SpatialCore");
      if (target->dependencyOrdinal != moduleEndpoint->dependencyOrdinal ||
          target->target != moduleEndpoint->target.module)
        return invalid("attachment module endpoint disagrees with its AccCore");
      if ((*moduleRecord)->plane != spatialEndpoint->plane() ||
          (*moduleRecord)->occurrenceOrdinal != occurrenceOrdinal ||
          llvm::ArrayRef<std::uint8_t>((*moduleRecord)->canonicalType) !=
              occurrenceType)
        return invalid("attachment does not preserve endpoint plane, ordinal, "
                       "and type");
      auto key = std::make_pair(moduleEndpoint->target.direction,
                                moduleEndpoint->target.ordinal);
      if (!attachmentCoverage[core.id()].insert(key).second)
        return invalid("SpatialCore boundary endpoint is attached twice");
    }
  }

  for (FabricEntityId id = 0;; ++id) {
    const std::optional<FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != FabricEntityKind::AccCoreOccurrence)
      continue;
    std::optional<FabricImportedModuleTargetRef> target =
        systemView.spatialCoreTarget(AccCoreOccurrenceRef(id));
    if (!target || target->dependencyOrdinal >= importedModules.size())
      return invalid("AccCore has no valid imported SpatialCore target");
    const StrictImportResult &module =
        importedModules[target->dependencyOrdinal];
    const std::size_t expected = module.moduleBoundaryInputs.size() +
                                 module.moduleBoundaryOutputs.size();
    if (attachmentCoverage[id].size() != expected)
      return invalid("an AccCore does not attach every module boundary "
                     "endpoint exactly once");
  }
  for (FabricEntityId id = 0;; ++id) {
    const std::optional<FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind == FabricEntityKind::ExternalBoundary &&
        !externalBoundariesWithEndpoints.count(id))
      return invalid("external boundary has no owned service endpoint");
  }
  return llvm::Error::success();
}

llvm::Expected<FabricArtifactView>
buildSystemView(::fabric::SystemOp root,
                const detail::FabricSystemCanonicalLabeling &labeling,
                const ArtifactIdentity &identity,
                llvm::ArrayRef<StrictImportResult> importedModules) {
  detail::FabricArtifactViewData data(identity, FabricRootKind::System);
  data.entities.resize(labeling.carriers.size());
  data.importedModules.reserve(importedModules.size());
  for (const StrictImportResult &module : importedModules)
    data.importedModules.push_back(module.view);

  std::vector<bool> dependencyUsed(importedModules.size(), false);

  for (const detail::FabricSystemEntityCarrier &carrier : labeling.carriers) {
    if (!carrier.op || carrier.id >= data.entities.size())
      return invalid("canonical System entity inventory is malformed");
    detail::FabricEntityViewData &entity = data.entities[carrier.id];
    entity.kind = carrier.kind;
    entity.owner.inventoryCounts = emptyInventories();

    if (auto host = dyn_cast<::fabric::SystemHostCoreOp>(carrier.op)) {
      auto architecture = decodeInstructionCoreArchitecturalContract(
          unsignedBytes(host.getArchitectureAttr()));
      if (!architecture)
        return architecture.takeError();
      auto microarchitecture =
          decodeInstructionCoreMicroarchitecturalRealization(
              unsignedBytes(host.getMicroarchitectureAttr()));
      if (!microarchitecture)
        return microarchitecture.takeError();
      entity.instructionCoreArchitecture = std::move(*architecture);
      entity.instructionCoreMicroarchitecture = std::move(*microarchitecture);
      entity.owner.resourceContract =
          entity.instructionCoreMicroarchitecture->resourceContract();
      continue;
    }

    if (auto core = dyn_cast<::fabric::SystemAccCoreOp>(carrier.op)) {
      auto target = decodeFabricImportedModuleTargetRef(
          unsignedBytes(core.getSpatialCoreAttr()));
      if (!target)
        return target.takeError();
      auto module = resolveImportedModule(importedModules, *target);
      if (!module)
        return module.takeError();
      dependencyUsed[target->dependencyOrdinal] = true;
      entity.spatialCoreTarget = *target;
      auto architecture = decodeInstructionCoreArchitecturalContract(
          unsignedBytes(core.getArchitectureAttr()));
      if (!architecture)
        return architecture.takeError();
      auto microarchitecture =
          decodeInstructionCoreMicroarchitecturalRealization(
              unsignedBytes(core.getMicroarchitectureAttr()));
      if (!microarchitecture)
        return microarchitecture.takeError();
      entity.instructionCoreArchitecture = std::move(*architecture);
      entity.instructionCoreMicroarchitecture = std::move(*microarchitecture);
      entity.instructionCore =
          instructionCoreView(*entity.instructionCoreMicroarchitecture);
      auto spatial = spatialCoreView(**module);
      if (!spatial)
        return spatial.takeError();
      entity.spatialCore = std::move(*spatial);
      continue;
    }

    if (auto memory = dyn_cast<::fabric::SystemMemoryServiceOp>(carrier.op)) {
      auto record = ::fabric::decodeMemoryServiceContractRecord(
          unsignedBytes(memory.getServiceContractAttr().getRecord()),
          root.getContext(), ::fabric::MemoryServiceOwnerKind::System);
      if (!record)
        return record.takeError();
      entity.owner.resourceContract = record->resourceContract();
      entity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::MemoryServiceRegion)] = record->regions().size();
      entity.systemMemoryService = std::move(*record);
      continue;
    }

    if (auto endpoint =
            dyn_cast<::fabric::SystemServiceEndpointOp>(carrier.op)) {
      auto owner = decodeSystemServiceEndpointOwnerRef(
          unsignedBytes(endpoint.getOwnerAttr()));
      if (!owner)
        return owner.takeError();
      auto capabilities = decodeCanonicalServiceCapabilitySet(
          unsignedBytes(endpoint.getCapabilitiesAttr()), root.getContext());
      if (!capabilities)
        return capabilities.takeError();
      if (capabilities->plane() == CanonicalServiceEndpointPlane::Transport) {
        TypeAttr carrierType = endpoint.getCarrierTypeAttr();
        if (!carrierType)
          return invalid("message service endpoint has no carrier type");
        auto encoded =
            ::fabric::encodeFabricTransportType(carrierType.getValue());
        if (!encoded)
          return encoded.takeError();
        const FabricPortDirection direction =
            capabilities->role() == CanonicalServiceEndpointRole::Initiate
                ? FabricPortDirection::Output
                : FabricPortDirection::Input;
        entity.owner.transportEndpoints.push_back(
            {direction, std::move(*encoded)});
        setPortInventories(entity.owner,
                           direction == FabricPortDirection::Input,
                           direction == FabricPortDirection::Output);
      } else {
        entity.owner.memoryEndpoints.push_back(
            {capabilities->role() == CanonicalServiceEndpointRole::Initiate
                 ? FabricMemoryEndpointRole::Manager
                 : FabricMemoryEndpointRole::Subordinate,
             {}});
      }
      entity.systemServiceEndpointOwner = std::move(*owner);
      entity.systemServiceCapabilities = std::move(*capabilities);
      continue;
    }

    if (auto transform =
            dyn_cast<::fabric::SystemServiceTransformOp>(carrier.op)) {
      auto record = decodeSystemServiceTransformRecord(
          unsignedBytes(transform.getContractAttr()));
      if (!record)
        return record.takeError();
      entity.systemServiceTransform = std::move(*record);
      continue;
    }

    if (auto domain = dyn_cast<::fabric::SystemHardwareDomainOp>(carrier.op)) {
      auto record = decodeHardwareDomainContractRecord(
          unsignedBytes(domain.getContractAttr()));
      if (!record)
        return record.takeError();
      entity.hardwareDomainKind = record->kind();
      entity.hardwareDomainContract = std::move(*record);
      data.hardwareDomains.push_back(HardwareDomainRef(carrier.id));
      continue;
    }

    if (auto resource =
            dyn_cast<::fabric::SystemTransportResourceOp>(carrier.op)) {
      auto functionType =
          dyn_cast<FunctionType>(resource.getFunctionTypeAttr().getValue());
      if (!functionType)
        return invalid("System transport resource has no function type");
      if (llvm::Error error =
              setTransportEndpoints(entity.owner, functionType.getInputs(),
                                    functionType.getResults()))
        return std::move(error);
      auto contract = ::fabric::decodeResourceContractRecord(
          unsignedBytes(resource.getResourceContractAttr()));
      if (!contract)
        return contract.takeError();
      entity.owner.resourceContract = std::move(*contract);
      if (DenseI8ArrayAttr crossing = resource.getClockCrossingAttr()) {
        auto decoded =
            decodeClockCrossingContractRecord(unsignedBytes(crossing));
        if (!decoded)
          return decoded.takeError();
        entity.clockCrossing = std::move(*decoded);
      }
      data.transportResources.push_back(SystemTransportResourceRef(carrier.id));
      continue;
    }
  }

  for (Operation &operation : root.getBody().front()) {
    if (auto attachment =
            dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(
                &operation)) {
      auto record = decodeServiceLegCarrierAttachmentRecord(
          unsignedBytes(attachment.getRecordAttr()));
      if (!record)
        return record.takeError();
      data.serviceLegCarrierAttachments.push_back(std::move(*record));
      continue;
    }

    if (auto pattern =
            dyn_cast<::fabric::SystemTransferPatternOp>(&operation)) {
      auto record = decodeSystemTransferPatternRecord(
          unsignedBytes(pattern.getContractAttr()));
      if (!record)
        return record.takeError();
      const SystemTransportResourceRef resource = record->pattern().resource;
      if (resource.id() >= data.entities.size() ||
          data.entities[resource.id()].kind !=
              FabricEntityKind::SystemTransportResource)
        return invalid("transfer pattern has an unknown resource owner");
      auto ordinal = labeling.transferPatternOrdinalByOperation.find(
          pattern.getOperation());
      if (ordinal == labeling.transferPatternOrdinalByOperation.end() ||
          ordinal->second !=
              data.entities[resource.id()].transferPatternRecords.size() ||
          record->pattern().ordinal != ordinal->second)
        return invalid("transfer-pattern ordinal is not canonical");
      detail::FabricNestedOwnerViewData owner;
      owner.inventoryCounts = emptyInventories();
      owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::TransferPatternEgress)] =
          record->egresses().size();
      detail::FabricEntityViewData &resourceEntity =
          data.entities[resource.id()];
      resourceEntity.transferPatterns.push_back(std::move(owner));
      resourceEntity.transferPatternRefs.push_back(record->pattern());
      resourceEntity.transferPatternRecords.push_back(std::move(*record));
      resourceEntity.owner.inventoryCounts[static_cast<std::size_t>(
          FabricInventoryKind::TransferPattern)] =
          resourceEntity.transferPatternRecords.size();
      for (std::uint64_t egress = 0;
           egress <
           resourceEntity.transferPatternRecords.back().egresses().size();
           ++egress)
        data.admittedTraversals.push_back(
            FabricPhysicalTraversalRef::transferPatternLeg(
                resourceEntity.transferPatternRecords.back().pattern(),
                egress));
      continue;
    }

    if (auto connection = dyn_cast<::fabric::SystemConnectionOp>(&operation)) {
      auto source = decodeFabricRef<FabricTransportEndpointRef>(
          unsignedBytes(connection.getSourceAttr()));
      if (!source)
        return source.takeError();
      auto destination = decodeFabricRef<FabricTransportEndpointRef>(
          unsignedBytes(connection.getDestinationAttr()));
      if (!destination)
        return destination.takeError();
      data.pointConnections.push_back({*source, *destination});
      data.admittedTraversals.push_back(
          FabricPhysicalTraversalRef::pointConnection(*source, *destination));
      continue;
    }

    if (auto attachment =
            dyn_cast<::fabric::SystemSpatialAttachmentOp>(&operation)) {
      auto moduleEndpoint = decodeFabricImportedModuleBoundaryEndpointRef(
          unsignedBytes(attachment.getModuleEndpointAttr()));
      if (!moduleEndpoint)
        return moduleEndpoint.takeError();
      auto spatialEndpoint = decodeFabricSpatialAttachmentEndpointRef(
          unsignedBytes(attachment.getSpatialEndpointAttr()));
      if (!spatialEndpoint)
        return spatialEndpoint.takeError();
      data.spatialAttachments.push_back(
          {*moduleEndpoint, std::move(*spatialEndpoint)});
    }
  }

  for (std::size_t index = 0; index < dependencyUsed.size(); ++index)
    if (!dependencyUsed[index])
      return invalid("unused ImportedModule dependency");

  auto view = detail::buildFabricArtifactView(std::move(data));
  if (!view)
    return view.takeError();
  auto systemView = requireSystemRoot(*view);
  if (!systemView)
    return systemView.takeError();
  if (llvm::Error error =
          detail::validateSystemServiceLegCarrierAttachments(*systemView))
    return std::move(error);
  if (llvm::Error error =
          validateSystemRelations(root, *systemView, importedModules))
    return std::move(error);
  auto clockReset = validateClockReset(*systemView);
  if (!clockReset)
    return clockReset.takeError();
  if (llvm::Error error = validateUnconditionalHandshakeClosure(*view))
    return std::move(error);
  return std::move(*view);
}

llvm::Expected<StrictImportResult>
strictImportSystem(const ArtifactRootReference &reference,
                   const CanonicalSemanticBytes &canonicalBytes,
                   DecodedFabricArtifact decoded, const ArtifactStore &store);

llvm::Expected<StrictImportResult>
strictImport(const ArtifactRootReference &reference,
             const CanonicalSemanticBytes &canonicalBytes,
             const ArtifactStore &store) {
  if (reference.schemaIdentity != fabricArtifactSchema.identity ||
      reference.schemaVersion != fabricArtifactSchema.version)
    return invalid("root reference has the wrong Fabric schema");
  if (finalizeArtifactIdentity(fabricArtifactSchema, canonicalBytes) !=
      reference.artifact)
    return invalid("root reference identity does not match canonical bytes");
  auto decoded = decodeFabricArtifactEnvelope(canonicalBytes.bytes());
  if (!decoded)
    return decoded.takeError();
  switch (decoded->rootKind) {
  case FabricRootKind::Module:
    return strictImportModule(reference, canonicalBytes, std::move(*decoded));
  case FabricRootKind::System:
    return strictImportSystem(reference, canonicalBytes, std::move(*decoded),
                              store);
  case FabricRootKind::InterconnectImplementation:
    return ownerUnavailable(
        "InterconnectImplementation strict import provider is unavailable");
  }
  llvm_unreachable("closed Fabric root kind");
}

llvm::Expected<StrictImportResult>
strictImportSystem(const ArtifactRootReference &reference,
                   const CanonicalSemanticBytes &canonicalBytes,
                   DecodedFabricArtifact decoded, const ArtifactStore &store) {
  if (decoded.rootKind != FabricRootKind::System)
    return invalid("System importer received the wrong root kind");

  std::vector<StrictImportResult> importedModules;
  importedModules.reserve(decoded.dependencies.size());
  for (const FabricDirectDependency &dependency : decoded.dependencies) {
    if (dependency.role != FabricDependencyRole::ImportedModule)
      return invalid("System root has a non-ImportedModule dependency");
    auto bytes = store.get(dependency.root);
    if (!bytes)
      return bytes.takeError();
    auto imported = strictImport(dependency.root, *bytes, store);
    if (!imported)
      return imported.takeError();
    if (imported->view.rootKind() != FabricRootKind::Module)
      return invalid("ImportedModule dependency has the wrong root kind");
    importedModules.push_back(std::move(*imported));
  }

  auto parsed =
      detail::parseFabricBytecodeModule(decoded.canonicalMlirBytecode);
  if (!parsed)
    return parsed.takeError();
  ModuleOp module = parsed->module.get();
  if (!llvm::hasSingleElement(module.getBody()->getOperations()))
    return invalid("canonical payload does not contain exactly one root");
  auto root = dyn_cast<::fabric::SystemOp>(&module.getBody()->front());
  if (!root || root.getSymName() != canonicalRootName)
    return invalid("canonical payload has no canonical System root");

  auto labeling =
      detail::computeFabricSystemCanonicalLabeling(root, decoded.dependencies);
  if (!labeling)
    return labeling.takeError();
  for (const detail::FabricSystemEntityCarrier &carrier : labeling->carriers) {
    auto stored = carrier.op->getAttrOfType<::fabric::EntityIdAttr>(
        systemEntityIdAttrName);
    if (!stored || stored.getId() != carrier.id)
      return invalid(llvm::Twine("canonical System payload has stale entity ") +
                     llvm::Twine(stored ? stored.getId() : ~0ULL) +
                     "; expected " + llvm::Twine(carrier.id) + " on " +
                     carrier.op->getName().getStringRef());
  }

  auto expectedOperation = labeling->canonicalOperationOrder.begin();
  for (Operation &operation : root.getBody().front()) {
    if (expectedOperation == labeling->canonicalOperationOrder.end() ||
        *expectedOperation != &operation)
      return invalid("canonical System child operation order is not canonical");
    ++expectedOperation;
  }
  if (expectedOperation != labeling->canonicalOperationOrder.end())
    return invalid("canonical System child operation order is not canonical");

  auto rewritten = detail::writeCanonicalFabricBytecode(module);
  if (!rewritten)
    return rewritten.takeError();
  if (*rewritten != decoded.canonicalMlirBytecode)
    return invalid("canonical System MLIR bytecode is not byte stable");
  auto view =
      buildSystemView(root, *labeling, reference.artifact, importedModules);
  if (!view)
    return view.takeError();
  return StrictImportResult{std::move(decoded), std::move(*view), {}, {}};
}

struct CanonicalSystemCandidate {
  OwningOpRef<ModuleOp> module;
  std::vector<FabricDirectDependency> dependencies;
};

llvm::Expected<CanonicalSystemCandidate> buildCanonicalSystemCandidate(
    ::fabric::SystemOp source,
    llvm::ArrayRef<ArtifactRootReference> importedModules) {
  auto sourceModule = source->getParentOfType<ModuleOp>();
  if (!sourceModule || source->getParentOp() != sourceModule.getOperation())
    return invalid("the selected Fabric System must be top-level");
  if (failed(verify(sourceModule)))
    return invalid("the System authoring module does not verify");

  std::vector<FabricDirectDependency> sourceDependencies;
  sourceDependencies.reserve(importedModules.size());
  for (const ArtifactRootReference &module : importedModules)
    sourceDependencies.push_back(
        {FabricDependencyRole::ImportedModule, module});

  OwningOpRef<ModuleOp> scratch(cast<ModuleOp>(sourceModule->clone()));
  Operation *clonedOperation =
      SymbolTable::lookupSymbolIn(*scratch, source.getSymNameAttr());
  auto clonedRoot = dyn_cast_or_null<::fabric::SystemOp>(clonedOperation);
  if (!clonedRoot)
    return invalid("the selected Fabric System was not cloned");

  for (Operation &operation :
       llvm::make_early_inc_range(scratch->getBody()->getOperations()))
    if (&operation != clonedRoot.getOperation())
      operation.erase();
  clonedRoot.setSymName(canonicalRootName);

  if (llvm::Error error =
          detail::normalizeSystemServiceLegCarrierAttachments(clonedRoot))
    return std::move(error);

  auto labeling = detail::computeFabricSystemCanonicalLabeling(
      clonedRoot, sourceDependencies);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error =
          detail::materializeFabricSystemCanonicalForm(clonedRoot, *labeling))
    return std::move(error);

  std::vector<FabricDirectDependency> canonicalDependencies =
      sourceDependencies;
  for (auto [sourceOrdinal, canonicalOrdinal] :
       llvm::enumerate(labeling->sourceDependencyToCanonical)) {
    if (canonicalOrdinal >= canonicalDependencies.size())
      return invalid("canonical dependency permutation is out of range");
    canonicalDependencies[canonicalOrdinal] = sourceDependencies[sourceOrdinal];
  }

  auto canonicalLabeling = detail::computeFabricSystemCanonicalLabeling(
      clonedRoot, canonicalDependencies);
  if (!canonicalLabeling)
    return canonicalLabeling.takeError();
  if (canonicalLabeling->relationBytes.bytes() !=
      labeling->relationBytes.bytes())
    return invalid("System canonicalization changed the semantic relation");
  if (llvm::Error error = detail::materializeFabricSystemCanonicalForm(
          clonedRoot, *canonicalLabeling))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("canonical Fabric System produced invalid IR");
  return CanonicalSystemCandidate{std::move(scratch),
                                  std::move(canonicalDependencies)};
}

llvm::Expected<OwningOpRef<ModuleOp>>
buildCanonicalCandidate(::fabric::ModuleOp source) {
  auto sourceModule = source->getParentOfType<ModuleOp>();
  if (!sourceModule || source->getParentOp() != sourceModule.getOperation())
    return invalid("the selected Fabric Module must be top-level");
  if (failed(verify(sourceModule)))
    return invalid("the authoring module does not verify");

  OwningOpRef<ModuleOp> scratch(cast<ModuleOp>(sourceModule->clone()));
  Operation *clonedOperation =
      SymbolTable::lookupSymbolIn(*scratch, source.getSymNameAttr());
  auto clonedRoot = dyn_cast_or_null<::fabric::ModuleOp>(clonedOperation);
  if (!clonedRoot)
    return invalid("the selected Fabric root was not cloned");

  if (llvm::Error error = stripAuthoringState(clonedRoot))
    return std::move(error);
  if (failed(::fabric::elaborateInstances(clonedRoot)))
    return invalid("fabric.instantiate elaboration failed");
  if (llvm::Error error = eraseElaboratedDeclarations(clonedRoot))
    return std::move(error);

  for (Operation &operation :
       llvm::make_early_inc_range(scratch->getBody()->getOperations()))
    if (&operation != clonedRoot.getOperation())
      operation.erase();
  clonedRoot.setSymName(canonicalRootName);
  if (failed(verify(*scratch)))
    return invalid("the root-complete Fabric candidate does not verify");

  auto preliminary = detail::computeFabricModuleCanonicalLabeling(clonedRoot);
  if (!preliminary)
    return preliminary.takeError();
  if (llvm::Error error =
          detail::materializeFabricResourceContracts(clonedRoot, *preliminary))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("the complete Fabric resource contracts do not verify");

  auto labeling = detail::computeFabricModuleCanonicalLabeling(clonedRoot);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error = reorderCanonicalGraphRegions(
          clonedRoot, labeling->canonicalOperationOrder))
    return std::move(error);
  if (llvm::Error error =
          detail::materializeFabricCanonicalFuCapabilityDomains(*labeling))
    return std::move(error);
  auto reordered = detail::computeFabricModuleCanonicalLabeling(clonedRoot);
  if (!reordered)
    return reordered.takeError();
  if (llvm::Error error = detail::materializeFabricCanonicalIds(*reordered))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("canonical Fabric IDs produced invalid IR");
  return std::move(scratch);
}

} // namespace

llvm::Expected<FinalizedFabricRoot>
finalizeFabricRoot(::fabric::ModuleOp source, const ArtifactStore &store) {
  auto candidate = buildCanonicalCandidate(source);
  if (!candidate)
    return candidate.takeError();
  auto bytecode = detail::writeCanonicalFabricBytecode(candidate->get());
  if (!bytecode)
    return bytecode.takeError();
  auto canonical =
      encodeFabricArtifactEnvelope(FabricRootKind::Module, {}, *bytecode);
  if (!canonical)
    return canonical.takeError();
  ArtifactRootReference reference{
      fabricArtifactSchema.identity.str(), fabricArtifactSchema.version,
      finalizeArtifactIdentity(fabricArtifactSchema, *canonical)};
  auto imported = strictImport(reference, *canonical, store);
  if (!imported)
    return imported.takeError();
  if (llvm::Error error =
          detail::validateFabricArtifactDependencyFramingClosure(store,
                                                                 *canonical))
    return std::move(error);
  auto stored = store.put(fabricArtifactSchema, *canonical);
  if (!stored)
    return stored.takeError();
  if (*stored != reference.artifact)
    return invalid("ArtifactStore returned a different Fabric identity");
  return FinalizedFabricRoot(reference, std::move(*canonical),
                             std::move(imported->decoded.dependencies),
                             std::move(imported->view));
}

llvm::Expected<FinalizedFabricRoot>
finalizeFabricRoot(::fabric::SystemOp source,
                   llvm::ArrayRef<ArtifactRootReference> importedModules,
                   const ArtifactStore &store) {
  auto candidate = buildCanonicalSystemCandidate(source, importedModules);
  if (!candidate)
    return candidate.takeError();
  auto bytecode = detail::writeCanonicalFabricBytecode(candidate->module.get());
  if (!bytecode)
    return bytecode.takeError();
  auto canonical = encodeFabricArtifactEnvelope(
      FabricRootKind::System, candidate->dependencies, *bytecode);
  if (!canonical)
    return canonical.takeError();
  ArtifactRootReference reference{
      fabricArtifactSchema.identity.str(), fabricArtifactSchema.version,
      finalizeArtifactIdentity(fabricArtifactSchema, *canonical)};
  if (llvm::Error error =
          detail::validateFabricArtifactDependencyFramingClosure(store,
                                                                 *canonical))
    return std::move(error);
  auto imported = strictImport(reference, *canonical, store);
  if (!imported)
    return imported.takeError();
  auto stored = store.put(fabricArtifactSchema, *canonical);
  if (!stored)
    return stored.takeError();
  if (*stored != reference.artifact)
    return invalid("ArtifactStore returned a different Fabric identity");
  return FinalizedFabricRoot(reference, std::move(*canonical),
                             std::move(imported->decoded.dependencies),
                             std::move(imported->view));
}

llvm::Expected<FinalizedFabricRoot>
importEntireFabricRoot(const ArtifactRootReference &reference,
                       const ArtifactStore &store) {
  auto canonical = store.get(reference);
  if (!canonical)
    return canonical.takeError();
  if (llvm::Error error =
          detail::validateFabricArtifactDependencyFramingClosure(store,
                                                                 *canonical))
    return std::move(error);
  auto imported = strictImport(reference, *canonical, store);
  if (!imported)
    return imported.takeError();
  return FinalizedFabricRoot(reference, std::move(*canonical),
                             std::move(imported->decoded.dependencies),
                             std::move(imported->view));
}

} // namespace loom::fabric
