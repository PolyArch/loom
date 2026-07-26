#include "Fabric/Artifact/FabricArtifact.h"

#include "../Identity/FabricArtifactViewInternal.h"
#include "Common/ArtifactFinalizer.h"
#include "Fabric/IR/BoundaryTransfer.h"
#include "Fabric/IR/Elaboration.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/MemoryCapabilityFinalization.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "FabricArtifactDependencyClosureInternal.h"
#include "FabricCanonicalLabeling.h"
#include "FabricFuCapabilityDerivation.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric {
namespace {

constexpr llvm::StringLiteral canonicalRootName("__loom_fabric_root");

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Error ownerUnavailable(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "fabric_artifact_owner_contract_unavailable: " + message);
}

llvm::Expected<std::optional<::fabric::ResourceContract>>
deriveResourceContract(
    Operation *operation,
    const loom::fabric::detail::FabricCanonicalLabeling &labeling) {
  if (auto fifo = dyn_cast<::fabric::FifoOp>(operation)) {
    auto contract = ::fabric::createFifoResourceContract(
        static_cast<std::uint32_t>(fifo.getMaxDepth()), fifo.getBypassable());
    if (!contract)
      return contract.takeError();
    return std::optional<::fabric::ResourceContract>(std::move(*contract));
  }
  if (isa<::fabric::BoundaryOp>(operation)) {
    auto contract = ::fabric::ResourceContract::create(
        ::fabric::declareBoundaryTransferContract());
    if (!contract)
      return contract.takeError();
    return std::optional<::fabric::ResourceContract>(std::move(*contract));
  }
  if (auto memory = dyn_cast<::fabric::MemOp>(operation)) {
    if (llvm::Error error = ::fabric::validateMemoryCapabilityFinalization(
            memory.getMemoryContract(), memory.getMemoryOperationPortsAttr()))
      return std::move(error);
    ::fabric::MemoryContractAttr contract = memory.getMemoryContract();
    if (contract.getEngine().getSchedule() == ::fabric::Schedule::Temporal)
      return ownerUnavailable(
          "temporal fabric.mem finalization requires its exact resident "
          "operation-context contract");
    if (contract.getManagerEndpoints().size() != 1 ||
        !contract.getSubordinateEndpoints().empty())
      return ownerUnavailable(
          "fabric.mem finalization requires its exact nontrivial service "
          "dispatch contract");
    return std::optional<::fabric::ResourceContract>();
  }
  if (auto pe = dyn_cast<::fabric::PeOp>(operation);
      pe && pe.getSchedule() == ::fabric::Schedule::Temporal) {
    std::optional<std::uint64_t> peId;
    for (const detail::FabricEntityCarrier &carrier : labeling.carriers)
      if (carrier.op == operation) {
        if (carrier.kind != FabricEntityKind::FabricPeOccurrence)
          return invalid(
              "temporal fabric.pe has the wrong canonical entity kind");
        peId = carrier.id;
        break;
      }
    if (!peId)
      return invalid("temporal fabric.pe has no canonical occurrence");

    llvm::SmallVector<std::uint32_t, 8> fuInputCounts;
    for (Operation *candidate : labeling.canonicalOperationOrder) {
      auto fu = dyn_cast_or_null<::fabric::FuOp>(candidate);
      if (!fu || fu->getParentOp() != operation)
        continue;
      if (fu.getInputs().size() > std::numeric_limits<std::uint32_t>::max())
        return invalid("temporal fabric.pe FU input domain exceeds u32");
      fuInputCounts.push_back(
          static_cast<std::uint32_t>(fu.getInputs().size()));
    }

    auto contextCount = pe.getNumInstruction();
    auto mode = pe.getOperandBufferMode();
    auto entries = pe.getOperandBufferSize();
    if (!contextCount || !mode || !entries)
      return invalid("temporal fabric.pe lacks its verified buffer parameters");
    auto derived = ::fabric::TemporalOperandBufferContract::create(
        ::fabric::TemporalOperandBufferDeclaration{FabricPeOccurrenceRef(*peId),
                                                   *contextCount, fuInputCounts,
                                                   *mode, *entries});
    if (!derived)
      return derived.takeError();
    return std::optional<::fabric::ResourceContract>(
        derived->resourceContract());
  }
  if (auto sw = dyn_cast<::fabric::SwitchOp>(operation);
      sw && sw.getSchedule() == ::fabric::Schedule::Temporal)
    return ownerUnavailable(
        "temporal fabric.switch finalization requires its complete switch "
        "resource projection");
  return std::optional<::fabric::ResourceContract>();
}

std::vector<std::int8_t> signedBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> result;
  result.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    result.push_back(static_cast<std::int8_t>(byte));
  return result;
}

std::vector<std::uint8_t> unsignedBytes(llvm::ArrayRef<std::int8_t> bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (std::int8_t byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Expected<std::optional<::fabric::ResourceContract>>
validateResourceContractRecord(
    Operation *operation,
    const loom::fabric::detail::FabricCanonicalLabeling &labeling) {
  auto expected = deriveResourceContract(operation, labeling);
  if (!expected)
    return expected.takeError();

  auto record = operation->getAttrOfType<DenseI8ArrayAttr>(
      ::fabric::kResourceContractRecordAttrName);
  if (!*expected) {
    if (record)
      return invalid("an owner without a resource contract carries a record");
    return std::optional<::fabric::ResourceContract>();
  }
  if (!record)
    return invalid("a resource owner is missing its complete contract record");

  std::vector<std::uint8_t> bytes = unsignedBytes(record.asArrayRef());
  auto decoded = ::fabric::decodeResourceContractRecord(bytes);
  if (!decoded)
    return decoded.takeError();
  auto canonical = ::fabric::encodeResourceContractRecord(*decoded);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != bytes)
    return invalid("a resource contract record is not canonical");
  auto expectedBytes = ::fabric::encodeResourceContractRecord(**expected);
  if (!expectedBytes)
    return expectedBytes.takeError();
  if (*expectedBytes != bytes)
    return invalid("a resource contract record disagrees with its owner");
  return std::optional<::fabric::ResourceContract>(std::move(*decoded));
}

llvm::Error materializeResourceContractRecords(
    ::fabric::ModuleOp root,
    const loom::fabric::detail::FabricCanonicalLabeling &labeling) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    operation->removeAttr(::fabric::kResourceContractRecordAttrName);
    auto contract = deriveResourceContract(operation, labeling);
    if (!contract) {
      result = contract.takeError();
      return WalkResult::interrupt();
    }
    if (!*contract)
      return WalkResult::advance();
    auto bytes = ::fabric::encodeResourceContractRecord(**contract);
    if (!bytes) {
      result = bytes.takeError();
      return WalkResult::interrupt();
    }
    std::vector<std::int8_t> values = signedBytes(*bytes);
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       DenseI8ArrayAttr::get(root.getContext(), values));
    return WalkResult::advance();
  });
  return result;
}

llvm::Error validateResourceContractRecords(
    ::fabric::ModuleOp root,
    const loom::fabric::detail::FabricCanonicalLabeling &labeling) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    auto contract = validateResourceContractRecord(operation, labeling);
    if (!contract) {
      result = contract.takeError();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
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
    llvm::SmallVector<Operation *> remaining;
    Operation *terminator = nullptr;
    for (Operation &operation : *block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        terminator = &operation;
        continue;
      }
      remaining.push_back(&operation);
    }
    llvm::SmallVector<Operation *> ordered;
    llvm::DenseSet<Operation *> placed;
    while (!remaining.empty()) {
      auto selected = remaining.end();
      std::uint64_t selectedRank = std::numeric_limits<std::uint64_t>::max();
      for (auto candidate = remaining.begin(); candidate != remaining.end();
           ++candidate) {
        bool ready = true;
        for (Value operand : (*candidate)->getOperands()) {
          Operation *definition = operand.getDefiningOp();
          if (definition && definition->getBlock() == block &&
              !placed.contains(definition)) {
            ready = false;
            break;
          }
        }
        auto found = rank.find(*candidate);
        if (ready && found != rank.end() && found->second < selectedRank) {
          selected = candidate;
          selectedRank = found->second;
        }
      }
      if (selected == remaining.end())
        return invalid("canonical operation order is not SSA-topological");
      Operation *operation = *selected;
      ordered.push_back(operation);
      placed.insert(operation);
      remaining.erase(selected);
    }
    for (Operation *operation : ordered) {
      if (terminator)
        operation->moveBefore(terminator);
      else
        operation->moveBefore(block, block->end());
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
writeBytecodeOnce(Operation *operation) {
  llvm::SmallVector<char> storage;
  llvm::raw_svector_ostream stream(storage);
  BytecodeWriterConfig config("loom.fabric.1.0");
  config.setElideLocations();
  if (failed(writeBytecodeToFile(operation, stream, config)))
    return invalid("MLIR bytecode writer rejected the canonical root");
  return std::vector<std::uint8_t>(storage.begin(), storage.end());
}

struct ParsedBytecodeModule {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
};

llvm::Expected<ParsedBytecodeModule>
parseBytecodeModule(llvm::ArrayRef<std::uint8_t> bytes) {
  DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();

  llvm::StringRef byteString(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
  llvm::MemoryBufferRef buffer(byteString, "<canonical-fabric>");
  ParserConfig parserConfig(context.get());
  Block topLevel;
  if (failed(readBytecodeFile(buffer, &topLevel, parserConfig)))
    return invalid("canonical MLIR bytecode cannot be parsed");
  if (!llvm::hasSingleElement(topLevel))
    return invalid("canonical MLIR bytecode has multiple top-level roots");
  auto module = dyn_cast<ModuleOp>(&topLevel.front());
  if (!module || failed(verify(module)))
    return invalid("canonical MLIR bytecode is not a valid builtin module");
  module->remove();
  return ParsedBytecodeModule{std::move(context),
                              OwningOpRef<ModuleOp>(module)};
}

llvm::Expected<std::vector<std::uint8_t>>
writeCanonicalBytecode(Operation *operation) {
  auto initial = writeBytecodeOnce(operation);
  if (!initial)
    return initial.takeError();
  auto normalizedModule = parseBytecodeModule(*initial);
  if (!normalizedModule)
    return normalizedModule.takeError();
  auto canonical = writeBytecodeOnce(normalizedModule->module.get());
  if (!canonical)
    return canonical.takeError();

  auto verificationModule = parseBytecodeModule(*canonical);
  if (!verificationModule)
    return verificationModule.takeError();
  auto verified = writeBytecodeOnce(verificationModule->module.get());
  if (!verified)
    return verified.takeError();
  if (*verified != *canonical)
    return invalid("the Fabric schema writer did not reach a byte-stable "
                   "canonical form");
  return canonical;
}

std::vector<std::uint64_t> emptyInventories() {
  return std::vector<std::uint64_t>(fabricClosedBound(FabricInventoryKind{}),
                                    0);
}

std::optional<FabricTransportEndpointOwnerRef>
transportOwner(FabricEntityKind kind, FabricEntityId id) {
  switch (kind) {
  case FabricEntityKind::FabricPeOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricPeOccurrenceRef(id));
  case FabricEntityKind::FabricFuOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricFuOccurrenceRef(id));
  case FabricEntityKind::FabricMemoryOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricMemoryOccurrenceRef(id));
  case FabricEntityKind::FabricSwitchOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricSwitchOccurrenceRef(id));
  case FabricEntityKind::FabricFifoOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricFifoOccurrenceRef(id));
  case FabricEntityKind::FabricBoundaryOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricBoundaryOccurrenceRef(id));
  default:
    return std::nullopt;
  }
}

FabricFuNodeKind fuNodeKind(Operation *operation) {
  if (isa<::fabric::MuxOp>(operation))
    return FabricFuNodeKind::Mux;
  if (isa<::fabric::DemuxOp>(operation))
    return FabricFuNodeKind::Demux;
  return FabricFuNodeKind::Op;
}

void setPortInventories(detail::FabricNestedOwnerViewData &owner,
                        std::uint64_t inputs, std::uint64_t outputs,
                        bool transport) {
  owner.inventoryCounts = emptyInventories();
  owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::InputPort)] = inputs;
  owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::OutputPort)] = outputs;
  owner.transportEndpointCount = transport ? inputs + outputs : 0;
}

llvm::Expected<FunctionType> memoryFunctionType(::fabric::MemOp memory) {
  if (auto typeAttribute = memory.getFunctionTypeAttr()) {
    auto type = dyn_cast<FunctionType>(typeAttribute.getValue());
    if (!type)
      return invalid("fabric.mem function_type is not a FunctionType");
    return type;
  }

  llvm::SmallVector<Type> inputs;
  ArrayRef<Type> innerTypes = memory.getInnerInputTypes();
  if (!innerTypes.empty())
    inputs.append(innerTypes.begin(), innerTypes.end());
  else
    for (Value input : memory.getInputs())
      inputs.push_back(input.getType());
  return FunctionType::get(memory.getContext(), inputs,
                           memory.getResultTypes());
}

llvm::Expected<std::optional<std::uint64_t>>
tokenInputOrdinal(Operation *operation, std::uint64_t signatureOrdinal) {
  auto memory = dyn_cast<::fabric::MemOp>(operation);
  if (!memory)
    return std::optional<std::uint64_t>(signatureOrdinal);
  auto type = memoryFunctionType(memory);
  if (!type)
    return type.takeError();
  if (signatureOrdinal >= type->getNumInputs())
    return invalid("fabric.mem input ordinal is outside its signature");
  if (isa<MemRefType>(type->getInput(signatureOrdinal)))
    return std::optional<std::uint64_t>();
  std::uint64_t tokenOrdinal = 0;
  for (std::uint64_t index = 0; index < signatureOrdinal; ++index)
    tokenOrdinal += !isa<MemRefType>(type->getInput(index));
  return std::optional<std::uint64_t>(tokenOrdinal);
}

llvm::Expected<std::optional<std::uint64_t>>
tokenOutputOrdinal(Operation *operation, std::uint64_t signatureOrdinal) {
  auto memory = dyn_cast<::fabric::MemOp>(operation);
  if (!memory)
    return std::optional<std::uint64_t>(operation->getNumOperands() +
                                        signatureOrdinal);
  auto type = memoryFunctionType(memory);
  if (!type)
    return type.takeError();
  if (signatureOrdinal >= type->getNumResults())
    return invalid("fabric.mem result ordinal is outside its signature");
  if (isa<MemRefType>(type->getResult(signatureOrdinal)))
    return std::optional<std::uint64_t>();
  std::uint64_t tokenOrdinal = 0;
  for (Type input : type->getInputs())
    tokenOrdinal += !isa<MemRefType>(input);
  for (std::uint64_t index = 0; index < signatureOrdinal; ++index)
    tokenOrdinal += !isa<MemRefType>(type->getResult(index));
  return std::optional<std::uint64_t>(tokenOrdinal);
}

llvm::Error populateMemoryView(::fabric::MemOp memory,
                               detail::FabricEntityViewData &entity) {
  auto type = memoryFunctionType(memory);
  if (!type)
    return type.takeError();
  auto endpoints = ::fabric::deriveMemoryTransportEndpointInventory(*type);
  if (!endpoints)
    return endpoints.takeError();

  std::uint64_t tokenInputs = 0;
  std::uint64_t tokenOutputs = 0;
  for (const ::fabric::MemoryTransportEndpointDescriptor &endpoint :
       *endpoints) {
    if (endpoint.direction == FabricPortDirection::Input)
      ++tokenInputs;
    else
      ++tokenOutputs;
  }
  setPortInventories(entity.owner, tokenInputs, tokenOutputs, true);

  ::fabric::MemoryContractAttr contract = memory.getMemoryContract();
  entity.owner.memoryEndpointRoles.assign(contract.getManagerEndpoints().size(),
                                          FabricMemoryEndpointRole::Manager);
  entity.owner.memoryEndpointRoles.insert(
      entity.owner.memoryEndpointRoles.end(),
      contract.getSubordinateEndpoints().size(),
      FabricMemoryEndpointRole::Subordinate);

  auto records = ::fabric::decodeMemoryOperationPortInventory(
      memory.getMemoryOperationPortsAttr(), memory.getContext(),
      contract.getEngine().getSchedule(), *endpoints);
  if (!records)
    return records.takeError();
  entity.owner.inventoryCounts[static_cast<std::size_t>(
      FabricInventoryKind::MemoryOperationPort)] = records->size();
  entity.memoryOperationPorts.reserve(records->size());
  for (::fabric::MemoryOperationPortRecord &record : *records) {
    detail::FabricNestedOwnerViewData owner;
    owner.inventoryCounts = emptyInventories();
    owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::MemoryCapabilityAlternative)] =
        record.capabilityAlternatives().size();
    owner.resourceContract = record.resourceContract();
    entity.memoryOperationPorts.push_back(
        {std::move(owner), std::move(record)});
  }
  return llvm::Error::success();
}

llvm::Expected<FabricArtifactView>
buildModuleView(::fabric::ModuleOp root,
                const detail::FabricCanonicalLabeling &labeling,
                const ArtifactIdentity &identity) {
  detail::FabricArtifactViewData data{
      identity, FabricRootKind::Module, {}, {}, {}};
  data.entities.resize(labeling.carriers.size());

  llvm::DenseMap<Operation *, const detail::FabricEntityCarrier *> carrierByOp;
  for (const detail::FabricEntityCarrier &carrier : labeling.carriers) {
    if (carrier.id >= data.entities.size())
      return invalid("canonical Fabric entity IDs are not dense");
    detail::FabricEntityViewData &entity = data.entities[carrier.id];
    entity.kind = carrier.kind;
    entity.owner.inventoryCounts = emptyInventories();
    if (carrier.op)
      carrierByOp[carrier.op] = &carrier;

    if (!carrier.op)
      continue;
    const std::uint64_t inputs = carrier.op->getNumOperands();
    const std::uint64_t outputs = carrier.op->getNumResults();
    setPortInventories(entity.owner, inputs, outputs,
                       transportOwner(carrier.kind, carrier.id).has_value());
    if (auto memory = dyn_cast<::fabric::MemOp>(carrier.op))
      if (llvm::Error error = populateMemoryView(memory, entity))
        return std::move(error);
    if (carrier.kind == FabricEntityKind::FabricModuleTemplate) {
      FunctionType type = root.getFunctionType();
      setPortInventories(entity.owner, type.getNumInputs(),
                         type.getNumResults(), false);
    }
    if (carrier.kind == FabricEntityKind::FabricFuOccurrence) {
      auto found = labeling.fuTemplateIdByOccurrence.find(carrier.op);
      if (found == labeling.fuTemplateIdByOccurrence.end())
        return invalid("an FU occurrence has no template relation");
      entity.fuTemplate = FabricFuTemplateRef(found->second);
    }

    auto contract = validateResourceContractRecord(carrier.op, labeling);
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
                       fu.getOutputs().size(), false);
    entity.owner.inventoryCounts[static_cast<std::size_t>(
        FabricInventoryKind::FuNode)] = carrier.canonicalNodeOrder.size();
    for (Operation *operation : carrier.canonicalNodeOrder) {
      detail::FabricFuNodeViewData node;
      node.kind = fuNodeKind(operation);
      setPortInventories(node.owner, operation->getNumOperands(),
                         operation->getNumResults(), false);
      entity.fuNodes.push_back(std::move(node));
    }
    auto templates = detail::deriveFabricFuCapabilityTemplates(
        fu, FabricFuTemplateRef(carrier.id), carrier.canonicalNodeOrder);
    if (!templates)
      return templates.takeError();
    entity.fuCapabilityTemplates = std::move(*templates);
  }

  for (const auto &entry : carrierByOp) {
    Operation *destination = entry.first;
    const detail::FabricEntityCarrier &destinationCarrier = *entry.second;
    auto destinationOwner =
        transportOwner(destinationCarrier.kind, destinationCarrier.id);
    if (!destinationOwner)
      continue;
    for (OpOperand &operand : destination->getOpOperands()) {
      Operation *source = operand.get().getDefiningOp();
      auto sourceEntry = carrierByOp.find(source);
      if (!source || sourceEntry == carrierByOp.end())
        continue;
      const detail::FabricEntityCarrier &sourceCarrier = *sourceEntry->second;
      auto sourceOwner = transportOwner(sourceCarrier.kind, sourceCarrier.id);
      if (!sourceOwner)
        continue;
      auto result = dyn_cast<OpResult>(operand.get());
      if (!result)
        return invalid("a physical point connection has no source result");
      auto sourceOrdinal = tokenOutputOrdinal(source, result.getResultNumber());
      if (!sourceOrdinal)
        return sourceOrdinal.takeError();
      auto destinationOrdinal =
          tokenInputOrdinal(destination, operand.getOperandNumber());
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
  for (const detail::FabricEntityCarrier &carrier : labeling.carriers) {
    if (carrier.kind == FabricEntityKind::FabricFifoOccurrence) {
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
};

llvm::Expected<StrictImportResult>
strictImport(const ArtifactRootReference &reference,
             const CanonicalSemanticBytes &canonicalBytes) {
  if (reference.schemaIdentity != fabricArtifactSchema.identity ||
      reference.schemaVersion != fabricArtifactSchema.version)
    return invalid("root reference has the wrong Fabric schema");
  if (finalizeArtifactIdentity(fabricArtifactSchema, canonicalBytes) !=
      reference.artifact)
    return invalid("root reference identity does not match canonical bytes");

  auto decoded = decodeFabricArtifactEnvelope(canonicalBytes.bytes());
  if (!decoded)
    return decoded.takeError();
  if (decoded->rootKind != FabricRootKind::Module)
    return ownerUnavailable(
        "only the Module root finalizer is available in the current owner");
  if (!decoded->dependencies.empty())
    return invalid("a fully elaborated Module root has a direct dependency");

  auto parsed = parseBytecodeModule(decoded->canonicalMlirBytecode);
  if (!parsed)
    return parsed.takeError();
  ModuleOp module = parsed->module.get();

  ::fabric::ModuleOp root;
  for (::fabric::ModuleOp candidate : module.getOps<::fabric::ModuleOp>()) {
    if (root)
      return invalid("canonical payload has multiple Fabric roots");
    root = candidate;
  }
  if (!root || root.getSymName() != canonicalRootName)
    return invalid("canonical payload has no canonical Module root");
  bool residualInstance = false;
  root->walk([&](::fabric::InstantiateOp) { residualInstance = true; });
  if (residualInstance)
    return invalid("canonical payload contains fabric.instantiate");
  auto labeling = detail::computeFabricModuleCanonicalLabeling(root);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error = validateResourceContractRecords(root, *labeling))
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
  }

  auto rewritten = writeCanonicalBytecode(module);
  if (!rewritten)
    return rewritten.takeError();
  if (*rewritten != decoded->canonicalMlirBytecode)
    return invalid("canonical MLIR bytecode is not byte stable");
  auto view = buildModuleView(root, *labeling, reference.artifact);
  if (!view)
    return view.takeError();
  return StrictImportResult{std::move(*decoded), std::move(*view)};
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
          materializeResourceContractRecords(clonedRoot, *preliminary))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("the complete Fabric resource contracts do not verify");

  auto labeling = detail::computeFabricModuleCanonicalLabeling(clonedRoot);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error = reorderCanonicalGraphRegions(
          clonedRoot, labeling->canonicalOperationOrder))
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
  auto bytecode = writeCanonicalBytecode(candidate->get());
  if (!bytecode)
    return bytecode.takeError();
  auto canonical =
      encodeFabricArtifactEnvelope(FabricRootKind::Module, {}, *bytecode);
  if (!canonical)
    return canonical.takeError();
  ArtifactRootReference reference{
      fabricArtifactSchema.identity.str(), fabricArtifactSchema.version,
      finalizeArtifactIdentity(fabricArtifactSchema, *canonical)};
  auto imported = strictImport(reference, *canonical);
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
importEntireFabricRoot(const ArtifactRootReference &reference,
                       const ArtifactStore &store) {
  auto canonical = store.get(reference);
  if (!canonical)
    return canonical.takeError();
  if (llvm::Error error =
          detail::validateFabricArtifactDependencyFramingClosure(store,
                                                                 *canonical))
    return std::move(error);
  auto imported = strictImport(reference, *canonical);
  if (!imported)
    return imported.takeError();
  return FinalizedFabricRoot(reference, std::move(*canonical),
                             std::move(imported->decoded.dependencies),
                             std::move(imported->view));
}

} // namespace loom::fabric
