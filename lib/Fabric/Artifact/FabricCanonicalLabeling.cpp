#include "FabricCanonicalLabeling.h"

#include "Common/CanonicalRelation.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "FabricFuCapabilityDerivation.h"
#include "FabricMemoryEngineTemplate.h"
#include "FabricModuleDomainNormalization.h"

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
  MemoryEngineDefinition,
  ModuleDomainSlot,
  ModuleDomainMember,
  ModuleDomainAssignment,
  FuCapabilityTemplate,
  FuCapabilityActiveNode,
  FuCapabilityRoute,
  FuOrbitRoot,
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
operationIntrinsic(Operation *op,
                   llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (auto fabricOp = dyn_cast<::fabric::OpOp>(op))
    return encodeFabricOpCanonicalIntrinsic(fabricOp);

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
          name == ::fabric::kFuTemplateIdAttrName ||
          name == ::fabric::kMemoryEngineTemplateIdAttrName ||
          name == "domain_slots" || name == "domain_assignments" ||
          name == "sym_name" || name == "capability_templates" ||
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
  static llvm::Expected<SemanticGraph>
  build(Operation *root, bool fuDefinition = false,
        const NormalizedModuleDomainRelation *domainRelation = nullptr,
        FabricFuCapabilityOrdinalSpace capabilityOrdinalSpace =
            FabricFuCapabilityOrdinalSpace::AuthoringPhysical) {
    SemanticGraph graph(root, fuDefinition, domainRelation,
                        capabilityOrdinalSpace);
    if (llvm::Error error = graph.collect(root))
      return std::move(error);
    if (llvm::Error error = graph.buildDomainRelations())
      return std::move(error);
    if (llvm::Error error = graph.buildFuCapabilityRelations())
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

  llvm::Expected<FabricCanonicalFuDefinition> canonicalizeFuDefinition() const {
    if (!fuDefinition_)
      return invalid("a Module graph is not an FU definition");
    auto canonical = canonicalize();
    if (!canonical)
      return canonical.takeError();
    llvm::DenseMap<std::uint32_t, Operation *> operationByVertex;
    for (const auto &entry : operationVertices_)
      operationByVertex[entry.second] = entry.first;
    std::vector<Operation *> canonicalNodeOrder;
    std::vector<CanonicalSemanticBytes> canonicalNodeOrbitCertificates;
    for (std::uint32_t vertex : canonical->canonicalOrder) {
      Operation *node = operationByVertex.lookup(vertex);
      if (isa_and_nonnull<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
              node)) {
        canonicalNodeOrder.push_back(node);
        auto certificate = canonicalizeWithFuOrbitRoot(vertex);
        if (!certificate)
          return certificate.takeError();
        canonicalNodeOrbitCertificates.push_back(std::move(certificate->bytes));
      }
    }
    return FabricCanonicalFuDefinition{
        std::move(canonical->bytes), std::move(canonicalNodeOrder),
        std::move(canonicalNodeOrbitCertificates)};
  }

  llvm::Expected<FabricCanonicalLabeling> canonicalizeModule() {
    if (fuDefinition_)
      return invalid("an FU definition graph is not a Fabric Module root");

    auto contextual = canonicalize();
    if (!contextual)
      return contextual.takeError();
    llvm::DenseMap<std::uint32_t, std::uint32_t> contextualPosition;
    for (auto [position, vertex] : llvm::enumerate(contextual->canonicalOrder))
      contextualPosition[vertex] = position;

    struct FuTemplateDraft {
      std::uint32_t vertex = 0;
      Operation *representative = nullptr;
      std::vector<Operation *> canonicalNodeOrder;
      std::uint32_t representativePosition =
          std::numeric_limits<std::uint32_t>::max();
    };
    std::map<std::vector<std::uint8_t>, FuTemplateDraft> templates;
    llvm::DenseMap<Operation *, std::uint32_t> templateVertexByOccurrence;
    llvm::DenseMap<Operation *, FabricOrdinal>
        definitionFuNodeOrdinalByOperation;
    llvm::DenseMap<Operation *, std::vector<Operation *>>
        definitionNodeOrderByOccurrence;
    llvm::DenseMap<Operation *, std::vector<std::uint8_t>>
        capabilityDomainByOccurrence;
    for (const auto &entry : operationVertices_) {
      auto fu = dyn_cast<::fabric::FuOp>(entry.first);
      if (!fu || fu.getSymNameAttr())
        continue;
      auto definition =
          computeCanonicalFabricFuDefinition(fu, capabilityOrdinalSpace_);
      if (!definition)
        return definition.takeError();
      const llvm::ArrayRef<std::uint8_t> definitionBytes =
          definition->relationBytes.bytes();

      std::vector<Operation *> occurrenceNodeOrder =
          definition->canonicalNodeOrder;
      if (occurrenceNodeOrder.size() !=
          definition->canonicalNodeOrbitCertificates.size())
        return invalid("an FU definition has an incomplete orbit inventory");
      std::map<std::vector<std::uint8_t>, std::vector<std::size_t>>
          orbitPositions;
      for (auto [position, certificate] :
           llvm::enumerate(definition->canonicalNodeOrbitCertificates))
        orbitPositions[certificate.bytes().vec()].push_back(position);
      for (const auto &entry : orbitPositions) {
        std::vector<Operation *> orbitNodes;
        orbitNodes.reserve(entry.second.size());
        for (std::size_t position : entry.second)
          orbitNodes.push_back(occurrenceNodeOrder[position]);
        llvm::sort(orbitNodes, [&](Operation *lhs, Operation *rhs) {
          return contextualPosition.lookup(operationVertices_.lookup(lhs)) <
                 contextualPosition.lookup(operationVertices_.lookup(rhs));
        });
        for (auto [position, node] : llvm::zip_equal(entry.second, orbitNodes))
          occurrenceNodeOrder[position] = node;
      }
      auto domain = canonicalizeFabricFuCapabilityDomain(
          fu, occurrenceNodeOrder, capabilityOrdinalSpace_);
      if (!domain)
        return domain.takeError();
      auto domainBytes = ::fabric::encodeFuCapabilityDomainRecord(*domain);
      if (!domainBytes)
        return domainBytes.takeError();
      capabilityDomainByOccurrence[fu.getOperation()] = *domainBytes;
      for (auto [ordinal, node] : llvm::enumerate(occurrenceNodeOrder))
        definitionFuNodeOrdinalByOperation[node] = ordinal;
      definitionNodeOrderByOccurrence[fu.getOperation()] =
          std::move(occurrenceNodeOrder);

      std::vector<std::uint8_t> key(definitionBytes.begin(),
                                    definitionBytes.end());
      auto [position, inserted] = templates.emplace(key, FuTemplateDraft{});
      if (inserted) {
        std::string intrinsic = "FU_TEMPLATE\x1f";
        intrinsic.append(reinterpret_cast<const char *>(key.data()),
                         key.size());
        position->second.vertex = addVertex(std::move(intrinsic));
        carriers_[position->second.vertex] = {
            FabricEntityKind::FabricFuTemplate, 0, nullptr};
      }
      templateVertexByOccurrence[fu.getOperation()] = position->second.vertex;
      addEdge(entry.second, position->second.vertex, EdgeKind::FuDefinition, 0);
    }

    struct MemoryTemplateDraft {
      std::uint32_t vertex = 0;
      Operation *representative = nullptr;
    };
    std::map<std::vector<std::uint8_t>, MemoryTemplateDraft> memoryTemplates;
    llvm::DenseMap<Operation *, std::uint32_t> memoryTemplateVertexByOccurrence;
    for (const auto &entry : operationVertices_) {
      auto memory = dyn_cast<::fabric::MemOp>(entry.first);
      if (!memory)
        continue;
      auto derived = deriveFabricMemoryEngineTemplate(memory);
      if (!derived)
        return derived.takeError();
      if (!*derived)
        continue;
      const std::vector<std::uint8_t> &key = (**derived).canonicalBytes;
      auto [position, inserted] =
          memoryTemplates.emplace(key, MemoryTemplateDraft{});
      if (inserted) {
        std::string intrinsic = "MEMORY_ENGINE_TEMPLATE\x1f";
        intrinsic.append(reinterpret_cast<const char *>(key.data()),
                         key.size());
        position->second.vertex = addVertex(std::move(intrinsic));
        position->second.representative = memory.getOperation();
        carriers_[position->second.vertex] = {
            FabricEntityKind::FabricMemoryEngineTemplate, 0, nullptr};
      }
      memoryTemplateVertexByOccurrence[memory.getOperation()] =
          position->second.vertex;
      addEdge(entry.second, position->second.vertex,
              EdgeKind::MemoryEngineDefinition, 0);
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
    llvm::DenseMap<Operation *, std::uint32_t> operationPosition;
    for (auto [position, vertex] : llvm::enumerate(canonical->canonicalOrder))
      if (Operation *op = operationByVertex.lookup(vertex)) {
        operationOrder.push_back(op);
        operationPosition[op] = position;
      }

    for (const auto &entry : templateVertexByOccurrence) {
      auto fu = dyn_cast<::fabric::FuOp>(entry.first);
      if (!fu)
        return invalid("an FU template relation has no occurrence owner");
      auto definitionOrder = definitionNodeOrderByOccurrence.find(entry.first);
      if (definitionOrder == definitionNodeOrderByOccurrence.end())
        return invalid("an FU occurrence has no canonical definition order");

      auto draft = llvm::find_if(templates, [&](const auto &candidate) {
        return candidate.second.vertex == entry.second;
      });
      if (draft == templates.end())
        return invalid("an FU occurrence names an unknown template draft");
      auto occurrencePosition = operationPosition.find(fu.getOperation());
      if (occurrencePosition == operationPosition.end())
        return invalid("an FU occurrence has no Module-canonical position");
      if (occurrencePosition->second < draft->second.representativePosition) {
        draft->second.representativePosition = occurrencePosition->second;
        draft->second.representative = fu.getOperation();
        draft->second.canonicalNodeOrder = definitionOrder->second;
      }
    }
    for (const auto &entry : templates) {
      const FuTemplateDraft &draft = entry.second;
      if (!isa_and_nonnull<::fabric::FuOp>(draft.representative))
        return invalid("an FU template has no canonical representative");
    }

    llvm::DenseMap<Operation *, std::uint64_t> fuTemplateIds;
    for (const auto &entry : templateVertexByOccurrence)
      fuTemplateIds[entry.first] = ids[entry.second];

    llvm::DenseMap<Operation *, std::uint64_t> memoryTemplateIds;
    for (const auto &entry : memoryTemplateVertexByOccurrence)
      memoryTemplateIds[entry.first] = ids[entry.second];

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

    std::vector<FabricMemoryEngineTemplateCarrier> memoryEngineTemplates;
    memoryEngineTemplates.reserve(memoryTemplates.size());
    for (const auto &entry : memoryTemplates)
      memoryEngineTemplates.push_back(
          {ids[entry.second.vertex], entry.second.representative});
    llvm::sort(memoryEngineTemplates,
               [](const FabricMemoryEngineTemplateCarrier &lhs,
                  const FabricMemoryEngineTemplateCarrier &rhs) {
                 return lhs.id < rhs.id;
               });

    std::vector<FabricModuleDomainSlotCarrier> moduleDomainSlots;
    FabricOrdinal nextClock = 0;
    FabricOrdinal nextReset = 0;
    for (std::uint32_t vertex : canonical->canonicalOrder) {
      auto found = domainSlotByVertex_.find(vertex);
      if (found == domainSlotByVertex_.end())
        continue;
      const NormalizedModuleDomainSlot &slot =
          domainRelation_->slots[found->second];
      FabricOrdinal &next =
          slot.kind == FabricClockResetKind::Clock ? nextClock : nextReset;
      moduleDomainSlots.push_back({slot.kind, slot.provisionalOrdinal, next++});
    }

    return FabricCanonicalLabeling{
        std::move(canonical->bytes),
        std::move(carriers),
        std::move(fuTemplates),
        std::move(memoryEngineTemplates),
        std::move(operationOrder),
        std::move(fuTemplateIds),
        std::move(memoryTemplateIds),
        std::move(definitionFuNodeOrdinalByOperation),
        std::move(capabilityDomainByOccurrence),
        std::move(moduleDomainSlots)};
  }

private:
  SemanticGraph(Operation *root, bool fuDefinition,
                const NormalizedModuleDomainRelation *domainRelation,
                FabricFuCapabilityOrdinalSpace capabilityOrdinalSpace)
      : root_(root), fuDefinition_(fuDefinition),
        domainRelation_(domainRelation),
        capabilityOrdinalSpace_(capabilityOrdinalSpace) {}

  llvm::Error buildFuCapabilityRelations() {
    llvm::SmallVector<::fabric::FuOp, 8> functionalUnits;
    if (fuDefinition_) {
      auto fu = dyn_cast<::fabric::FuOp>(root_);
      if (!fu)
        return invalid("an FU definition graph has no FU root");
      functionalUnits.push_back(fu);
    } else {
      for (const auto &entry : operationVertices_)
        if (auto fu = dyn_cast<::fabric::FuOp>(entry.first);
            fu && !fu.getSymNameAttr())
          functionalUnits.push_back(fu);
    }

    for (::fabric::FuOp fu : functionalUnits) {
      llvm::SmallVector<Operation *, 16> physicalNodes;
      for (Operation &operation : fu.getBody().front().without_terminator())
        if (isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(operation))
          physicalNodes.push_back(&operation);
      auto domain = canonicalizeFabricFuCapabilityDomain(
          fu, physicalNodes, capabilityOrdinalSpace_);
      if (!domain)
        return domain.takeError();

      const std::uint32_t fuVertex =
          operationVertices_.lookup(fu.getOperation());
      for (const ::fabric::FuCapabilityTemplateSelection &selection :
           domain->templates()) {
        const std::uint32_t templateVertex =
            addVertex("FU_CAPABILITY_TEMPLATE\x1f");
        addEdge(fuVertex, templateVertex, EdgeKind::FuCapabilityTemplate, 0);
        for (std::uint64_t ordinal : selection.activeOperationNodeOrdinals) {
          if (ordinal >= physicalNodes.size())
            return invalid("FU capability names an unknown operation node");
          addEdge(templateVertex,
                  operationVertices_.lookup(physicalNodes[ordinal]),
                  EdgeKind::FuCapabilityActiveNode, 0);
        }
        for (const ::fabric::FuCapabilityRouteSelection &route :
             selection.routes) {
          if (route.selectorNodeOrdinal >= physicalNodes.size())
            return invalid("FU capability names an unknown selector node");
          addEdge(templateVertex,
                  operationVertices_.lookup(
                      physicalNodes[route.selectorNodeOrdinal]),
                  EdgeKind::FuCapabilityRoute, route.selectedPort);
        }
      }
    }
    return llvm::Error::success();
  }

  llvm::Expected<::loom::CanonicalRelationResult>
  canonicalizeWithFuOrbitRoot(std::uint32_t vertex) const {
    std::vector<std::string> intrinsics = intrinsics_;
    const std::uint32_t root = intrinsics.size();
    intrinsics.push_back("FU_NODE_ORBIT_ROOT\x1f");

    std::vector<::loom::CanonicalRelationEdge> relations;
    relations.reserve(edges_.size() + 1);
    for (const Edge &edge : edges_)
      relations.push_back({edge.source, edge.target, relationLabel(edge)});
    relations.push_back(
        {root, vertex,
         relationLabel({root, vertex, EdgeKind::FuOrbitRoot, 0, kNoOrdinal})});
    return ::loom::canonicalizeRelationGraph(intrinsics, relations);
  }

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

  bool includesDomainMember(const NormalizedModuleDomainMember &member) const {
    if (!fuDefinition_)
      return true;
    if (member.boundary ||
        member.role != ::fabric::ModuleDomainAuthoringRelation::
                           InternalMemberRole::FuNode ||
        !member.owner)
      return false;
    auto fu = member.owner->getParentOfType<::fabric::FuOp>();
    return fu && fu.getOperation() == root_;
  }

  llvm::Error buildDomainRelations() {
    if (!domainRelation_)
      return llvm::Error::success();
    std::vector<bool> selectedSlots(domainRelation_->slots.size(),
                                    !fuDefinition_);
    if (fuDefinition_)
      for (const NormalizedModuleDomainAssignment &assignment :
           domainRelation_->assignments)
        if (assignment.member < domainRelation_->members.size() &&
            assignment.slot < selectedSlots.size() &&
            includesDomainMember(domainRelation_->members[assignment.member]))
          selectedSlots[assignment.slot] = true;

    llvm::DenseMap<std::size_t, std::uint32_t> slotVertices;
    for (auto [index, slot] : llvm::enumerate(domainRelation_->slots)) {
      if (!selectedSlots[index])
        continue;
      std::string intrinsic = "MODULE_DOMAIN_SLOT";
      appendU32(intrinsic, static_cast<std::uint32_t>(slot.kind));
      const std::uint32_t vertex = addVertex(std::move(intrinsic));
      slotVertices[index] = vertex;
      domainSlotByVertex_[vertex] = index;
      addEdge(operationVertices_.lookup(root_), vertex,
              EdgeKind::ModuleDomainSlot, 0);
    }

    std::vector<std::optional<std::uint32_t>> memberVertices(
        domainRelation_->members.size());
    for (auto [index, member] : llvm::enumerate(domainRelation_->members)) {
      if (!includesDomainMember(member))
        continue;
      Operation *owner = member.boundary ? root_ : member.owner;
      auto ownerVertex = operationVertices_.find(owner);
      if (!owner || ownerVertex == operationVertices_.end())
        return invalid("Module domain member has no semantic graph owner");
      std::string intrinsic = "MODULE_DOMAIN_MEMBER";
      appendU8(intrinsic, member.boundary ? 1 : 0);
      if (member.boundary)
        appendU32(intrinsic, static_cast<std::uint32_t>(member.direction));
      else
        appendU32(intrinsic, static_cast<std::uint32_t>(member.role));
      appendU64(intrinsic, member.ordinal);
      const std::uint32_t vertex = addVertex(std::move(intrinsic));
      memberVertices[index] = vertex;
      addEdge(ownerVertex->second, vertex, EdgeKind::ModuleDomainMember, 0);
    }
    for (const NormalizedModuleDomainAssignment &assignment :
         domainRelation_->assignments) {
      if (assignment.member >= memberVertices.size() ||
          assignment.slot >= domainRelation_->slots.size())
        return invalid("Module domain relation index is out of range");
      if (!memberVertices[assignment.member])
        continue;
      auto slot = slotVertices.find(assignment.slot);
      if (slot == slotVertices.end())
        return invalid("Module domain assignment has no slot vertex");
      addEdge(*memberVertices[assignment.member], slot->second,
              EdgeKind::ModuleDomainAssignment, 0);
    }
    return llvm::Error::success();
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
  const NormalizedModuleDomainRelation *domainRelation_;
  FabricFuCapabilityOrdinalSpace capabilityOrdinalSpace_;
  std::vector<std::string> intrinsics_;
  std::vector<Edge> edges_;
  llvm::DenseMap<Operation *, std::uint32_t> operationVertices_;
  llvm::DenseMap<Block *, std::uint32_t> blockVertices_;
  llvm::DenseMap<Value, std::uint32_t> valueVertices_;
  llvm::DenseMap<Operation *, llvm::SmallVector<SymbolRefAttr>>
      operationSymbols_;
  llvm::DenseMap<std::uint32_t, FabricEntityCarrier> carriers_;
  llvm::DenseMap<std::uint32_t, std::size_t> domainSlotByVertex_;
};

} // namespace

llvm::Expected<FabricCanonicalFuDefinition>
computeCanonicalFabricFuDefinition(::fabric::FuOp fu) {
  return computeCanonicalFabricFuDefinition(
      fu, FabricFuCapabilityOrdinalSpace::AuthoringPhysical);
}

llvm::Expected<FabricCanonicalFuDefinition> computeCanonicalFabricFuDefinition(
    ::fabric::FuOp fu, FabricFuCapabilityOrdinalSpace sourceOrdinalSpace) {
  auto graph = SemanticGraph::build(fu.getOperation(), true, nullptr,
                                    sourceOrdinalSpace);
  if (!graph)
    return graph.takeError();
  return graph->canonicalizeFuDefinition();
}

llvm::Expected<std::string>
encodeFabricOpCanonicalIntrinsic(::fabric::OpOp op) {
  std::optional<::fabric::ImplementationFamilyId> family =
      op.getImplementationFamily();
  if (!family)
    return invalid("fabric.op has no implementation family");

  std::vector<std::vector<std::uint8_t>> schemaIdentities;
  schemaIdentities.reserve(op.getOpList().size());
  for (Attribute attribute : op.getOpList()) {
    auto symbol = dyn_cast<FlatSymbolRefAttr>(attribute);
    if (!symbol)
      return invalid("fabric.op has a non-symbol operation member");
    std::optional<::dataflow::OperationSchemaId> schema =
        ::dataflow::findOperationSchema(symbol.getValue());
    if (!schema)
      return invalid("fabric.op names an unregistered operation schema");
    auto identity = ::dataflow::encodeOperationSchemaId(*schema);
    if (!identity)
      return identity.takeError();
    schemaIdentities.push_back(identity->bytes().vec());
  }
  llvm::sort(schemaIdentities);
  if (std::adjacent_find(schemaIdentities.begin(), schemaIdentities.end()) !=
      schemaIdentities.end())
    return invalid("fabric.op has duplicate operation schemas");

  std::string intrinsic = "FABRIC_OP\x1f";
  appendU32(intrinsic, static_cast<std::uint32_t>(*family));
  appendU64(intrinsic, schemaIdentities.size());
  for (const std::vector<std::uint8_t> &identity : schemaIdentities) {
    appendU64(intrinsic, identity.size());
    intrinsic.append(reinterpret_cast<const char *>(identity.data()),
                     identity.size());
  }

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

llvm::Expected<FabricCanonicalLabeling>
computeFabricModuleCanonicalLabeling(::fabric::ModuleOp root) {
  llvm::Expected<SemanticGraph> graph =
      SemanticGraph::build(root.getOperation());
  if (!graph)
    return graph.takeError();
  return graph->canonicalizeModule();
}

llvm::Expected<FabricCanonicalLabeling> computeFabricModuleCanonicalLabeling(
    ::fabric::ModuleOp root,
    const NormalizedModuleDomainRelation &domainRelation) {
  auto graph =
      SemanticGraph::build(root.getOperation(), false, &domainRelation);
  if (!graph)
    return graph.takeError();
  return graph->canonicalizeModule();
}

llvm::Expected<FabricCanonicalLabeling>
computeCanonicalFabricModulePayloadLabeling(
    ::fabric::ModuleOp root,
    const NormalizedModuleDomainRelation &domainRelation) {
  auto graph =
      SemanticGraph::build(root.getOperation(), false, &domainRelation,
                           FabricFuCapabilityOrdinalSpace::CanonicalDefinition);
  if (!graph)
    return graph.takeError();
  return graph->canonicalizeModule();
}

llvm::Error
materializeFabricCanonicalIds(const FabricCanonicalLabeling &labeling) {
  for (const FabricEntityCarrier &carrier : labeling.carriers) {
    if (!carrier.op) {
      if (carrier.kind != FabricEntityKind::FabricFuTemplate &&
          carrier.kind != FabricEntityKind::FabricMemoryEngineTemplate)
        return invalid("a non-template entity has no operation carrier");
      continue;
    }

    MLIRContext *context = carrier.op->getContext();
    carrier.op->setAttr(::fabric::kEntityIdAttrName,
                        ::fabric::EntityIdAttr::get(context, carrier.id));
    if (carrier.kind == FabricEntityKind::FabricFuOccurrence) {
      auto found = labeling.fuTemplateIdByOccurrence.find(carrier.op);
      if (found == labeling.fuTemplateIdByOccurrence.end())
        return invalid("an FU occurrence has no canonical template relation");
      carrier.op->setAttr(::fabric::kFuTemplateIdAttrName,
                          ::fabric::EntityIdAttr::get(context, found->second));
    }
    if (carrier.kind == FabricEntityKind::FabricMemoryOccurrence) {
      auto found = labeling.memoryEngineTemplateIdByOccurrence.find(carrier.op);
      if (found != labeling.memoryEngineTemplateIdByOccurrence.end())
        carrier.op->setAttr(
            ::fabric::kMemoryEngineTemplateIdAttrName,
            ::fabric::EntityIdAttr::get(context, found->second));
    }
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

llvm::Error validateFabricCanonicalFuCapabilityDomains(
    const FabricCanonicalLabeling &labeling) {
  std::size_t occurrenceCount = 0;
  for (const FabricEntityCarrier &carrier : labeling.carriers) {
    if (carrier.kind != FabricEntityKind::FabricFuOccurrence)
      continue;
    ++occurrenceCount;
    auto fu = dyn_cast_or_null<::fabric::FuOp>(carrier.op);
    auto expected =
        labeling.canonicalFuCapabilityDomainByOccurrence.find(carrier.op);
    if (!fu ||
        expected == labeling.canonicalFuCapabilityDomainByOccurrence.end())
      return invalid("a canonical FU capability domain has no FU carrier");
    ::fabric::FuCapabilityDomainAttr stored = fu.getCapabilityTemplatesAttr();
    if (!stored)
      return invalid("canonical FU capability domain carrier is stale");
    llvm::ArrayRef<std::int8_t> storedBytes = stored.getRecord().asArrayRef();
    if (storedBytes.size() != expected->second.size())
      return invalid("canonical FU capability domain carrier is stale");
    for (auto [storedByte, expectedByte] :
         llvm::zip_equal(storedBytes, expected->second))
      if (static_cast<std::uint8_t>(storedByte) != expectedByte)
        return invalid("canonical FU capability domain carrier is stale");
  }
  if (occurrenceCount !=
      labeling.canonicalFuCapabilityDomainByOccurrence.size())
    return invalid("a canonical FU capability domain has no FU carrier");
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
