#include "FabricCanonicalLabeling.h"

#include "Common/CanonicalRelation.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "FabricFuCapabilityDerivation.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

constexpr std::uint32_t kNoOrdinal = 0xffffffffu;

enum class EdgeKind : std::uint8_t {
  OwnsBlock,
  BlockHoldsOp,
  DefResult,
  DefArg,
  Operand,
  Successor,
  SymbolUse,
  FuDefinition,
};

struct Edge {
  std::uint32_t source;
  std::uint32_t target;
  EdgeKind kind;
  std::uint32_t firstOrdinal;
  std::uint32_t secondOrdinal;
};

void appendU8(std::string &bytes, std::uint8_t value) {
  bytes.push_back(static_cast<char>(value));
}

void appendU32(std::string &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>(value >> shift));
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::string relationLabel(const Edge &edge) {
  std::string label;
  appendU8(label, static_cast<std::uint8_t>(edge.kind));
  appendU32(label, edge.firstOrdinal);
  appendU32(label, edge.secondOrdinal);
  return label;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

void serializeAttribute(llvm::raw_ostream &stream, Attribute attribute,
                        llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (auto symbol = dyn_cast<SymbolRefAttr>(attribute)) {
    stream << "#symbol";
    symbols.push_back(symbol);
    return;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    stream << '{';
    for (NamedAttribute field : dictionary) {
      stream << field.getName().getValue() << '=';
      serializeAttribute(stream, field.getValue(), symbols);
      stream << ';';
    }
    stream << '}';
    return;
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    stream << '[';
    for (Attribute element : array) {
      serializeAttribute(stream, element, symbols);
      stream << ',';
    }
    stream << ']';
    return;
  }
  stream << "#leaf<";
  attribute.print(stream);
  stream << '>';
}

llvm::Expected<std::string>
fabricOpIntrinsic(::fabric::OpOp op,
                  llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  (void)symbols;
  std::optional<::fabric::ImplementationFamilyId> family =
      op.getImplementationFamily();
  if (!family)
    return invalid("fabric.op has no implementation family");

  std::vector<std::uint32_t> schemas;
  schemas.reserve(op.getOpList().size());
  for (Attribute attribute : op.getOpList()) {
    auto symbol = dyn_cast<FlatSymbolRefAttr>(attribute);
    if (!symbol)
      return invalid("fabric.op has a non-symbol operation member");
    std::optional<::dataflow::OperationSchemaId> schema =
        ::dataflow::findOperationSchema(symbol.getValue());
    if (!schema)
      return invalid("fabric.op names an unregistered operation schema");
    schemas.push_back(static_cast<std::uint32_t>(*schema));
  }
  llvm::sort(schemas);
  if (std::adjacent_find(schemas.begin(), schemas.end()) != schemas.end())
    return invalid("fabric.op has duplicate operation schemas");

  std::string intrinsic = "FABRIC_OP\x1f";
  appendU32(intrinsic, static_cast<std::uint32_t>(*family));
  appendU64(intrinsic, schemas.size());
  for (std::uint32_t schema : schemas)
    appendU32(intrinsic, schema);

  std::string parameterBytes;
  llvm::raw_string_ostream parameterStream(parameterBytes);
  llvm::SmallVector<SymbolRefAttr> ignored;
  serializeAttribute(parameterStream, op.getHwParams(), ignored);
  parameterStream.flush();
  if (!ignored.empty())
    return invalid("fabric.op hw_params contains a symbol reference");
  appendU64(intrinsic, parameterBytes.size());
  intrinsic.append(parameterBytes);
  return intrinsic;
}

llvm::Expected<std::string>
operationIntrinsic(Operation *op,
                   llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (auto fabricOp = dyn_cast<::fabric::OpOp>(op))
    return fabricOpIntrinsic(fabricOp, symbols);

  std::string intrinsic;
  llvm::raw_string_ostream stream(intrinsic);
  stream << "OP\x1f" << op->getName().getStringRef() << '\x1e';
  stream << op->getNumOperands() << ':' << op->getNumResults() << ':'
         << op->getNumRegions() << ':' << op->getNumSuccessors() << '\x1e';

  llvm::SmallDenseSet<llvm::StringRef, 16> registered;
  for (StringAttr name : op->getName().getAttributeNames())
    registered.insert(name.getValue());
  auto process = [&](DictionaryAttr dictionary, bool registeredOnly) {
    if (!dictionary)
      return;
    for (NamedAttribute field : dictionary) {
      llvm::StringRef name = field.getName().getValue();
      if (name == ::fabric::kEntityIdAttrName ||
          name == ::fabric::kFuTemplateIdAttrName || name == "sym_name" ||
          name == "capability_templates" ||
          (registeredOnly && !registered.contains(name)))
        continue;
      stream << '\x1f' << name << '=';
      serializeAttribute(stream, field.getValue(), symbols);
      stream << '\x1e';
    }
  };
  if (auto properties =
          dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()))
    process(properties, false);
  process(op->getDiscardableAttrDictionary(), true);
  return stream.str();
}

std::string valueIntrinsic(Value value) {
  std::string intrinsic;
  llvm::raw_string_ostream stream(intrinsic);
  stream << (isa<BlockArgument>(value) ? "ARG\x1f" : "RESULT\x1f");
  value.getType().print(stream);
  return stream.str();
}

std::string blockIntrinsic(Block &block) {
  std::string intrinsic = "BLOCK\x1f";
  appendU64(intrinsic, block.getNumArguments());
  appendU8(intrinsic, block.isEntryBlock() ? 1 : 0);
  return intrinsic;
}

bool isUnorderedRegion(Region &region) {
  Operation *owner = region.getParentOp();
  return isa<::fabric::ModuleOp, ::fabric::PeOp, ::fabric::FuOp>(owner);
}

std::optional<FabricEntityKind> occurrenceKind(Operation *op, Operation *root) {
  if (op == root)
    return FabricEntityKind::FabricModuleTemplate;
  if (auto symbol = dyn_cast<SymbolOpInterface>(op))
    if (symbol.getNameAttr())
      return std::nullopt;
  if (isa<::fabric::PeOp>(op))
    return FabricEntityKind::FabricPeOccurrence;
  if (isa<::fabric::FuOp>(op))
    return FabricEntityKind::FabricFuOccurrence;
  if (isa<::fabric::MemOp>(op))
    return FabricEntityKind::FabricMemoryOccurrence;
  if (isa<::fabric::SwitchOp>(op))
    return FabricEntityKind::FabricSwitchOccurrence;
  if (isa<::fabric::FifoOp>(op))
    return FabricEntityKind::FabricFifoOccurrence;
  if (isa<::fabric::BoundaryOp>(op))
    return FabricEntityKind::FabricBoundaryOccurrence;
  return std::nullopt;
}

class SemanticGraph {
public:
  static llvm::Expected<SemanticGraph> build(Operation *root,
                                             bool fuDefinition = false) {
    SemanticGraph graph(root, fuDefinition);
    if (llvm::Error error = graph.collect(root))
      return std::move(error);
    if (llvm::Error error = graph.buildRelations(root))
      return std::move(error);
    return graph;
  }

  llvm::Expected<::loom::CanonicalRelationResult> canonicalize() const {
    std::vector<::loom::CanonicalRelationEdge> relations;
    relations.reserve(edges_.size());
    for (const Edge &edge : edges_)
      relations.push_back({edge.source, edge.target, relationLabel(edge)});
    return ::loom::canonicalizeRelationGraph(intrinsics_, relations);
  }

  llvm::Expected<FabricCanonicalLabeling> canonicalizeModule() {
    if (fuDefinition_)
      return invalid("an FU definition graph is not a Fabric Module root");

    struct FuTemplateDraft {
      std::uint32_t vertex = 0;
      Operation *representative = nullptr;
      std::vector<Operation *> canonicalNodeOrder;
    };
    std::map<std::vector<std::uint8_t>, FuTemplateDraft> templates;
    llvm::DenseMap<Operation *, std::uint32_t> templateVertexByOccurrence;
    llvm::DenseMap<Operation *, std::vector<std::uint8_t>>
        capabilityDomainByOccurrence;
    for (const auto &entry : operationVertices_) {
      auto fu = dyn_cast<::fabric::FuOp>(entry.first);
      if (!fu || fu.getSymNameAttr())
        continue;
      llvm::Expected<SemanticGraph> definition =
          SemanticGraph::build(fu.getOperation(), true);
      if (!definition)
        return definition.takeError();
      llvm::Expected<::loom::CanonicalRelationResult> canonical =
          definition->canonicalize();
      if (!canonical)
        return canonical.takeError();
      llvm::ArrayRef<std::uint8_t> definitionBytes = canonical->bytes.bytes();
      llvm::DenseMap<std::uint32_t, Operation *> definitionOperationByVertex;
      for (const auto &definitionEntry : definition->operationVertices_)
        definitionOperationByVertex[definitionEntry.second] =
            definitionEntry.first;
      std::vector<Operation *> canonicalNodeOrder;
      for (std::uint32_t vertex : canonical->canonicalOrder) {
        Operation *node = definitionOperationByVertex.lookup(vertex);
        if (isa_and_nonnull<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
                node))
          canonicalNodeOrder.push_back(node);
      }

      auto domain =
          canonicalizeFabricFuCapabilityDomain(fu, canonicalNodeOrder);
      if (!domain)
        return domain.takeError();
      auto domainBytes = ::fabric::encodeFuCapabilityDomainRecord(*domain);
      if (!domainBytes)
        return domainBytes.takeError();
      capabilityDomainByOccurrence[fu.getOperation()] = *domainBytes;

      std::vector<std::uint8_t> key;
      key.reserve(16 + definitionBytes.size() + domainBytes->size());
      appendU64(key, definitionBytes.size());
      key.insert(key.end(), definitionBytes.begin(), definitionBytes.end());
      appendU64(key, domainBytes->size());
      key.insert(key.end(), domainBytes->begin(), domainBytes->end());
      auto [position, inserted] = templates.emplace(key, FuTemplateDraft{});
      if (inserted) {
        std::string intrinsic = "FU_TEMPLATE\x1f";
        intrinsic.append(reinterpret_cast<const char *>(key.data()),
                         key.size());
        position->second.vertex = addVertex(std::move(intrinsic));
        position->second.representative = fu.getOperation();
        position->second.canonicalNodeOrder = canonicalNodeOrder;
        carriers_[position->second.vertex] = {
            FabricEntityKind::FabricFuTemplate, 0, nullptr};
      }
      templateVertexByOccurrence[fu.getOperation()] = position->second.vertex;
      addEdge(entry.second, position->second.vertex, EdgeKind::FuDefinition, 0);
    }

    llvm::Expected<::loom::CanonicalRelationResult> canonical = canonicalize();
    if (!canonical)
      return canonical.takeError();

    std::vector<std::uint64_t> ids(intrinsics_.size(),
                                   std::numeric_limits<std::uint64_t>::max());
    std::uint64_t nextId = 0;
    for (std::uint32_t vertex : canonical->canonicalOrder)
      if (carriers_.count(vertex))
        ids[vertex] = nextId++;

    std::vector<FabricEntityCarrier> carriers;
    for (std::uint32_t vertex : canonical->canonicalOrder) {
      auto carrier = carriers_.find(vertex);
      if (carrier == carriers_.end())
        continue;
      FabricEntityCarrier value = carrier->second;
      value.id = ids[vertex];
      carriers.push_back(value);
    }

    llvm::DenseMap<std::uint32_t, Operation *> operationByVertex;
    for (const auto &entry : operationVertices_)
      operationByVertex[entry.second] = entry.first;
    std::vector<Operation *> operationOrder;
    operationOrder.reserve(operationVertices_.size());
    for (std::uint32_t vertex : canonical->canonicalOrder)
      if (Operation *op = operationByVertex.lookup(vertex))
        operationOrder.push_back(op);

    llvm::DenseMap<Operation *, std::uint64_t> fuTemplateIds;
    for (const auto &entry : templateVertexByOccurrence)
      fuTemplateIds[entry.first] = ids[entry.second];

    std::vector<FabricFuTemplateCarrier> fuTemplates;
    fuTemplates.reserve(templates.size());
    for (auto &entry : templates) {
      FuTemplateDraft &draft = entry.second;
      fuTemplates.push_back({ids[draft.vertex], draft.representative,
                             std::move(draft.canonicalNodeOrder)});
    }
    llvm::sort(fuTemplates, [](const FabricFuTemplateCarrier &lhs,
                               const FabricFuTemplateCarrier &rhs) {
      return lhs.id < rhs.id;
    });

    return FabricCanonicalLabeling{
        std::move(canonical->bytes), std::move(carriers),
        std::move(fuTemplates),      std::move(operationOrder),
        std::move(fuTemplateIds),    std::move(capabilityDomainByOccurrence)};
  }

private:
  SemanticGraph(Operation *root, bool fuDefinition)
      : root_(root), fuDefinition_(fuDefinition) {}

  std::uint32_t addVertex(std::string intrinsic) {
    std::uint32_t vertex = intrinsics_.size();
    intrinsics_.push_back(std::move(intrinsic));
    return vertex;
  }

  void addEdge(std::uint32_t source, std::uint32_t target, EdgeKind kind,
               std::uint32_t firstOrdinal,
               std::uint32_t secondOrdinal = kNoOrdinal) {
    edges_.push_back({source, target, kind, firstOrdinal, secondOrdinal});
  }

  llvm::Error collect(Operation *op) {
    llvm::SmallVector<SymbolRefAttr> symbols;
    llvm::Expected<std::string> intrinsic = operationIntrinsic(op, symbols);
    if (!intrinsic)
      return intrinsic.takeError();
    std::uint32_t opVertex = addVertex(std::move(*intrinsic));
    operationVertices_[op] = opVertex;
    operationSymbols_[op] = std::move(symbols);

    if (!fuDefinition_)
      if (std::optional<FabricEntityKind> kind = occurrenceKind(op, root_)) {
        carriers_[opVertex] = {*kind, 0, op};
        intrinsics_[opVertex].append("\x1f"
                                     "ENTITY"
                                     "\x1f",
                                     8);
        appendU32(intrinsics_[opVertex], static_cast<std::uint32_t>(*kind));
      }

    for (OpResult result : op->getResults())
      valueVertices_[result] = addVertex(valueIntrinsic(result));
    for (Region &region : op->getRegions())
      for (Block &block : region) {
        blockVertices_[&block] = addVertex(blockIntrinsic(block));
        for (BlockArgument argument : block.getArguments())
          valueVertices_[argument] = addVertex(valueIntrinsic(argument));
        for (Operation &child : block)
          if (llvm::Error error = collect(&child))
            return error;
      }
    return llvm::Error::success();
  }

  llvm::Error buildRelations(Operation *op) {
    std::uint32_t opVertex = operationVertices_.lookup(op);
    std::uint32_t regionOrdinal = 0;
    for (Region &region : op->getRegions()) {
      const bool ordered = !isUnorderedRegion(region);
      std::uint32_t blockOrdinal = 0;
      for (Block &block : region) {
        std::uint32_t blockVertex = blockVertices_.lookup(&block);
        addEdge(opVertex, blockVertex, EdgeKind::OwnsBlock, regionOrdinal,
                ordered ? blockOrdinal : kNoOrdinal);
        std::uint32_t operationOrdinal = 0;
        for (Operation &child : block) {
          addEdge(blockVertex, operationVertices_.lookup(&child),
                  EdgeKind::BlockHoldsOp,
                  ordered ? operationOrdinal : kNoOrdinal);
          ++operationOrdinal;
        }
        for (BlockArgument argument : block.getArguments())
          addEdge(blockVertex, valueVertices_.lookup(argument),
                  EdgeKind::DefArg, argument.getArgNumber());
        ++blockOrdinal;
      }
      ++regionOrdinal;
    }
    for (OpResult result : op->getResults())
      addEdge(opVertex, valueVertices_.lookup(result), EdgeKind::DefResult,
              result.getResultNumber());
    for (OpOperand &operand : op->getOpOperands()) {
      auto source = valueVertices_.find(operand.get());
      if (source != valueVertices_.end()) {
        addEdge(source->second, opVertex, EdgeKind::Operand,
                operand.getOperandNumber());
        continue;
      }
      if (!fuDefinition_ || op != root_)
        return invalid("Fabric semantic graph contains an external SSA use");
      std::string intrinsic = "FU_INPUT\x1f";
      llvm::raw_string_ostream stream(intrinsic);
      operand.get().getType().print(stream);
      std::uint32_t port = addVertex(stream.str());
      addEdge(port, opVertex, EdgeKind::Operand, operand.getOperandNumber());
    }
    for (auto successor : llvm::enumerate(op->getSuccessors()))
      addEdge(opVertex, blockVertices_.lookup(successor.value()),
              EdgeKind::Successor, successor.index());

    const llvm::SmallVector<SymbolRefAttr> &symbols =
        operationSymbols_.lookup(op);
    for (std::uint32_t ordinal = 0; ordinal < symbols.size(); ++ordinal) {
      Operation *definition =
          SymbolTable::lookupNearestSymbolFrom(op, symbols[ordinal]);
      auto target = operationVertices_.find(definition);
      if (!definition || target == operationVertices_.end())
        return invalid("Fabric semantic graph has an unresolved symbol use");
      addEdge(opVertex, target->second, EdgeKind::SymbolUse, ordinal);
    }

    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &child : block)
          if (llvm::Error error = buildRelations(&child))
            return error;
    return llvm::Error::success();
  }

  Operation *root_;
  bool fuDefinition_;
  std::vector<std::string> intrinsics_;
  std::vector<Edge> edges_;
  llvm::DenseMap<Operation *, std::uint32_t> operationVertices_;
  llvm::DenseMap<Block *, std::uint32_t> blockVertices_;
  llvm::DenseMap<Value, std::uint32_t> valueVertices_;
  llvm::DenseMap<Operation *, llvm::SmallVector<SymbolRefAttr>>
      operationSymbols_;
  llvm::DenseMap<std::uint32_t, FabricEntityCarrier> carriers_;
};

} // namespace

llvm::Expected<FabricCanonicalLabeling>
computeFabricModuleCanonicalLabeling(::fabric::ModuleOp root) {
  llvm::Expected<SemanticGraph> graph =
      SemanticGraph::build(root.getOperation());
  if (!graph)
    return graph.takeError();
  return graph->canonicalizeModule();
}

llvm::Error
materializeFabricCanonicalIds(const FabricCanonicalLabeling &labeling) {
  for (const FabricEntityCarrier &carrier : labeling.carriers) {
    if (!carrier.op) {
      if (carrier.kind != FabricEntityKind::FabricFuTemplate)
        return invalid("a non-template entity has no operation carrier");
      continue;
    }

    MLIRContext *context = carrier.op->getContext();
    carrier.op->setAttr(::fabric::kEntityIdAttrName,
                        ::fabric::EntityIdAttr::get(context, carrier.id));
    if (carrier.kind != FabricEntityKind::FabricFuOccurrence)
      continue;

    auto found = labeling.fuTemplateIdByOccurrence.find(carrier.op);
    if (found == labeling.fuTemplateIdByOccurrence.end())
      return invalid("an FU occurrence has no canonical template relation");
    carrier.op->setAttr(::fabric::kFuTemplateIdAttrName,
                        ::fabric::EntityIdAttr::get(context, found->second));
  }
  return llvm::Error::success();
}

llvm::Error materializeFabricCanonicalFuCapabilityDomains(
    const FabricCanonicalLabeling &labeling) {
  for (const auto &entry : labeling.canonicalFuCapabilityDomainByOccurrence) {
    auto fu = dyn_cast_or_null<::fabric::FuOp>(entry.first);
    if (!fu)
      return invalid("a canonical FU capability domain has no FU carrier");
    std::vector<std::int8_t> signedBytes;
    signedBytes.reserve(entry.second.size());
    for (std::uint8_t byte : entry.second)
      signedBytes.push_back(static_cast<std::int8_t>(byte));
    fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
        fu.getContext(), DenseI8ArrayAttr::get(fu.getContext(), signedBytes)));
  }
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
