//===- DataflowCanonicalLabeling.cpp - canonical relation graph ----------===//
//
// Builds the exact typed semantic relation graph of a Canonical Dataflow
// Program and computes a canonical labeling of it. The labeling is an exact
// deterministic individualization-refinement over a colored, edge-labeled
// directed graph: color refinement to an equitable partition, then recursive
// individualization of the first ambiguous cell selecting the lexicographically
// minimal serialization. This makes canonical bytes and dense entity slots
// invariant under private-symbol/SSA/block renaming, locations, and unordered
// reordering while detecting every required semantic difference, with no
// pointer, container, or printer-order tie-break.
//
//===----------------------------------------------------------------------===//

#include "DataflowCanonicalLabeling.h"

#include "Common/CanonicalRelation.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowInterfaces.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace mlir;

namespace dataflow::detail {
namespace {

constexpr std::uint32_t kNoOrdinal = 0xFFFFFFFFu;

// Directed, ordinal-labeled relations of the semantic graph. Direction is part
// of every refinement signature, so one directed edge is seen by both ends.
enum class EdgeKind : std::uint8_t {
  OwnsBlock,    // op -> block: (region ordinal, block position or none)
  BlockHoldsOp, // block -> op: (op position or none)
  DefResult,    // op -> result value: (result ordinal)
  DefArg,       // block -> argument value: (argument ordinal)
  Operand,      // value -> op: (operand ordinal)
  Successor,    // op -> successor block: (successor ordinal)
  SymbolUse,    // user op -> defining op: (symbol-use ordinal)
};

struct Edge {
  unsigned from;
  unsigned to;
  EdgeKind kind;
  std::uint32_t ord0;
  std::uint32_t ord1;
};

void putU8(std::string &s, std::uint8_t v) {
  s.push_back(static_cast<char>(v));
}
void putU32(std::string &s, std::uint32_t v) {
  for (int i = 3; i >= 0; --i)
    s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

std::string edgeLabel(EdgeKind kind, std::uint32_t ord0, std::uint32_t ord1) {
  std::string label;
  putU8(label, static_cast<std::uint8_t>(kind));
  putU32(label, ord0);
  putU32(label, ord1);
  return label;
}

llvm::Error relationError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                 message.str().c_str());
}

bool isMemoryCapabilityType(Type type) {
  return DataflowDialect::isMemoryCapabilityType(type) ||
         DataflowDialect::containsMemoryCapability(type);
}

/// A region carries no operation-order relation only where the spec makes order
/// nonsemantic: a module symbol table and a dataflow.graph body. Every other
/// region is a stored-program region whose block and operation sequence is
/// semantic. Defaulting to ordered prevents a generic "ignore order" rule from
/// silently rewriting stored-program behavior.
bool isUnorderedRegion(Region &region) {
  Operation *parent = region.getParentOp();
  return isa<ModuleOp>(parent) || isa<GraphOp>(parent);
}

// Only an externally visible (Public) symbol name is semantic. A Private or a
// Nested symbol is not externally visible, so its spelling is a canonical label
// that must be redacted from identity.
bool isRedactedSymbolName(Operation *op) {
  if (auto symbol = dyn_cast<SymbolOpInterface>(op))
    return symbol.getVisibility() != SymbolTable::Visibility::Public;
  return false;
}

// Structurally serialize an attribute, redacting every symbol reference at any
// nested path so its private spelling never enters identity, and collecting the
// references in pre-order so a symbol-use relation can be keyed by that
// canonical traversal position rather than by spelling. Dictionaries emit in
// their canonical name-sorted order and arrays in order; both are handled
// structurally rather than through a generic printer so no container or
// source order leaks. A non-symbol leaf uses its deterministic printed form,
// which for a uniqued registered attribute contains no SSA name, symbol
// spelling, location, or pointer, and is therefore a stable semantic encoding.
void serializeAttr(llvm::raw_ostream &os, Attribute attr,
                   llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (auto ref = dyn_cast<SymbolRefAttr>(attr)) {
    os << "#sym";
    symbols.push_back(ref);
    return;
  }
  if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
    os << '{';
    for (NamedAttribute named : dict) {
      os << named.getName().getValue() << '=';
      serializeAttr(os, named.getValue(), symbols);
      os << ';';
    }
    os << '}';
    return;
  }
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    os << '[';
    for (Attribute element : array) {
      serializeAttr(os, element, symbols);
      os << ',';
    }
    os << ']';
    return;
  }
  os << "#leaf<";
  attr.print(os);
  os << '>';
}

// Serialize a function-like arg/res attribute array with the derived entity-id
// carrier removed. Materializing a memory-root ID makes this array appear where
// it was absent, so an array whose only remaining content is that carrier must
// serialize identically to no array at all. Position placeholders are preserved
// when any genuine attribute remains.
void appendFunctionArgResAttrs(llvm::raw_ostream &os, StringRef name,
                               Attribute value,
                               llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  auto array = dyn_cast<ArrayAttr>(value);
  if (!array)
    return;
  bool anySemantic = false;
  for (Attribute entry : array)
    if (auto dict = dyn_cast<DictionaryAttr>(entry))
      for (NamedAttribute nested : dict)
        anySemantic |= nested.getName().getValue() != kEntityIdAttrName;
  if (!anySemantic)
    return;
  os << '\x1f' << name << "=[";
  for (Attribute entry : array) {
    os << '{';
    if (auto dict = dyn_cast<DictionaryAttr>(entry))
      for (NamedAttribute nested : dict) {
        if (nested.getName().getValue() == kEntityIdAttrName)
          continue;
        os << nested.getName().getValue() << '=';
        serializeAttr(os, nested.getValue(), symbols);
        os << ';';
      }
    os << '}';
  }
  os << "]\x1e";
}

// The op color: registered operation name, arity, and the semantic content of
// the op's inherent/registered schema attributes and properties only. Excludes
// the derived entity-id carrier, private symbol spelling (a canonical label,
// not identity), symbol-reference spelling (a typed relation), and every
// non-schema discardable attribute (debug_label, provenance, or visual metadata
// the Canonical Semantic Relation Graph excludes). Inherent attributes reach
// identity through the properties dictionary; a legacy registered attribute not
// stored as a property is admitted only when its name appears in the op's
// registered schema, so identity never depends on an open ignore list. An
// externally visible linkage name is retained through its public sym_name
// attribute. The collected symbol references become the op's symbol-use
// relations, keyed by their pre-order position.
llvm::Expected<std::string>
opIntrinsic(Operation *op, llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (op->getParentOfType<GraphOp>() &&
      findOperationSchema(op->getName().getStringRef())) {
    llvm::Expected<::loom::CanonicalSemanticBytes> projection =
        projectRegisteredActorSchemaProjectionBytes(op);
    if (!projection) {
      std::string operationText;
      llvm::raw_string_ostream stream(operationText);
      op->print(stream);
      return relationError("canonical dataflow: actor projection failed for " +
                           operationText + ": " +
                           llvm::toString(projection.takeError()));
    }
    llvm::ArrayRef<std::uint8_t> bytes = projection->bytes();
    std::string result("ACTOR\x1f", 6);
    result.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return result;
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  os << "OP\x1f" << op->getName().getStringRef() << '\x1e';
  os << 'n' << op->getNumOperands() << ':' << op->getNumResults() << ':'
     << op->getNumRegions() << ':' << op->getNumSuccessors() << '\x1e';
  const bool redactedSymbolName = isRedactedSymbolName(op);
  const StringRef symName = SymbolTable::getSymbolAttrName();
  llvm::SmallDenseSet<StringRef, 8> registered;
  for (StringAttr name : op->getName().getAttributeNames())
    registered.insert(name.getValue());
  auto processDict = [&](DictionaryAttr dict, bool gateToRegisteredSchema) {
    if (!dict)
      return;
    for (NamedAttribute named : dict) {
      StringRef name = named.getName().getValue();
      if (name == kEntityIdAttrName)
        continue;
      if (name == symName && redactedSymbolName)
        continue;
      if (gateToRegisteredSchema && !registered.contains(name))
        continue;
      if (name == "arg_attrs" || name == "res_attrs") {
        appendFunctionArgResAttrs(os, name, named.getValue(), symbols);
        continue;
      }
      os << '\x1f' << name << '=';
      serializeAttr(os, named.getValue(), symbols);
      os << '\x1e';
    }
  };
  if (auto props =
          dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()))
    processDict(props, /*gateToRegisteredSchema=*/false);
  processDict(op->getDiscardableAttrDictionary(),
              /*gateToRegisteredSchema=*/true);
  return os.str();
}

std::string valueIntrinsic(Value value) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << (isa<BlockArgument>(value) ? "VA" : "VR") << '\x1f';
  // A uniqued registered type prints deterministically and, on the admitted
  // program surface, carries no SSA name, symbol spelling, or location, so its
  // printed form is a stable semantic leaf encoding.
  value.getType().print(os);
  os << '\x1e';
  return os.str();
}

std::string blockIntrinsic(Block &block) {
  std::string result;
  llvm::raw_string_ostream os(result);
  bool entry = block.isEntryBlock();
  os << "BL\x1f" << block.getNumArguments() << ':' << (entry ? 1 : 0) << '\x1e';
  return os.str();
}

//===----------------------------------------------------------------------===//
// SemanticGraph
//===----------------------------------------------------------------------===//

class SemanticGraph {
public:
  static llvm::Expected<SemanticGraph> build(ModuleOp module);

  llvm::Expected<CanonicalLabeling> canonicalize();

private:
  unsigned addVertex(std::string intrinsic) {
    unsigned id = numVertices_++;
    intrinsic_.push_back(std::move(intrinsic));
    return id;
  }
  void addEdge(unsigned from, unsigned to, EdgeKind kind, std::uint32_t ord0,
               std::uint32_t ord1 = kNoOrdinal) {
    edges_.push_back({from, to, kind, ord0, ord1});
  }

  llvm::Error collectVertices(Operation *op);
  llvm::Error detectCarriers(ModuleOp module);
  llvm::Error buildEdges(ModuleOp module);

  unsigned numVertices_ = 0;
  std::vector<std::string> intrinsic_;
  std::vector<Edge> edges_;

  llvm::DenseMap<Operation *, unsigned> opVertex_;
  llvm::DenseMap<Block *, unsigned> blockVertex_;
  llvm::DenseMap<Value, unsigned> valueVertex_;
  llvm::DenseMap<unsigned, EntityCarrier> carrier_;
  // Symbol references an op carries, in canonical pre-order; the index is the
  // symbol-use relation's semantic path key.
  llvm::DenseMap<Operation *, llvm::SmallVector<SymbolRefAttr>> opSymbols_;
};

llvm::Error SemanticGraph::collectVertices(Operation *op) {
  llvm::SmallVector<SymbolRefAttr> symbols;
  llvm::Expected<std::string> intrinsic = opIntrinsic(op, symbols);
  if (!intrinsic)
    return intrinsic.takeError();
  opVertex_[op] = addVertex(std::move(*intrinsic));
  opSymbols_[op] = std::move(symbols);
  for (Region &region : op->getRegions())
    for (Block &block : region) {
      blockVertex_[&block] = addVertex(blockIntrinsic(block));
      for (BlockArgument arg : block.getArguments())
        valueVertex_[arg] = addVertex(valueIntrinsic(arg));
      for (Operation &child : block) {
        for (OpResult res : child.getResults())
          valueVertex_[res] = addVertex(valueIntrinsic(res));
        if (llvm::Error error = collectVertices(&child))
          return error;
      }
    }
  return llvm::Error::success();
}

llvm::Error SemanticGraph::detectCarriers(ModuleOp module) {
  auto record = [&](Value vertexValue, EntityCarrier carrier) {
    unsigned vertex = valueVertex_.lookup(vertexValue);
    carrier_[vertex] = carrier;
    intrinsic_[vertex].append("\x1f"
                              "ENTITY"
                              "\x1f",
                              8);
    putU8(intrinsic_[vertex], static_cast<std::uint8_t>(carrier.kind));
  };
  auto recordOp = [&](Operation *op, EntityCarrier carrier) {
    unsigned vertex = opVertex_.lookup(op);
    carrier_[vertex] = carrier;
    intrinsic_[vertex].append("\x1f"
                              "ENTITY"
                              "\x1f",
                              8);
    putU8(intrinsic_[vertex], static_cast<std::uint8_t>(carrier.kind));
  };

  llvm::Error error = llvm::Error::success();
  module.walk([&](Operation *op) {
    if (error)
      return;
    if (isa<GraphOp>(op)) {
      // A graph memory formal is not a root: its identity comes from the exact
      // graph.launch binding, resolved through root-preserving views to the
      // upstream static role. Only the graph definition is an entity here.
      recordOp(op, {CanonicalDataflowEntityKind::Graph, 0, op});
      return;
    }
    if (auto thread = dyn_cast<ThreadOp>(op)) {
      // The static imported-memory role is the ordinary dataflow.thread memory
      // formal; it carries its entity ID in its own argument dictionary and
      // needs no owning-graph entity. A thread's function inputs are its
      // leading entry-block arguments, so the input ordinal is the
      // block-argument index.
      FunctionType type = thread.getFunctionType();
      Block &entry = thread.getBody().front();
      for (unsigned input = 0; input < type.getNumInputs(); ++input)
        if (isMemoryCapabilityType(type.getInput(input)))
          record(entry.getArgument(input),
                 {CanonicalDataflowEntityKind::LogicalMemoryRoot, 0, op,
                  nullptr, input});
      return;
    }
    if (isa<ThreadLaunchOp>(op) || isa<GraphLaunchOp>(op)) {
      SymbolRefAttr symbol;
      if (auto thread = dyn_cast<ThreadLaunchOp>(op))
        symbol = thread.getCalleeAttr();
      else
        symbol = cast<GraphLaunchOp>(op).getCalleeAttr();
      Operation *callee =
          symbol ? SymbolTable::lookupNearestSymbolFrom(op, symbol) : nullptr;
      if (!callee) {
        error = relationError("canonical dataflow: unresolved launch callee");
        return;
      }
      EntityCarrier carrier{
          isa<ThreadLaunchOp>(op)
              ? CanonicalDataflowEntityKind::RootThreadLaunch
              : CanonicalDataflowEntityKind::StaticGraphLaunch,
          0, op};
      carrier.calleeOp = callee;
      recordOp(op, carrier);
      return;
    }
    if (auto graph = op->getParentOfType<GraphOp>()) {
      if (isCanonicalDataflowActor(op)) {
        recordOp(op, {CanonicalDataflowEntityKind::Actor, 0, op,
                      graph.getOperation()});
        return;
      }
      // A fresh canonical allocation is a root-defining value in its own right;
      // its result carries the entity ID and it needs no owning-graph entity.
      if (isa<memref::AllocOp>(op))
        recordOp(op, {CanonicalDataflowEntityKind::LogicalMemoryRoot, 0, op});
    }
  });
  return error;
}

llvm::Error SemanticGraph::buildEdges(ModuleOp module) {
  llvm::Error error = llvm::Error::success();
  module.walk([&](Operation *op) {
    if (error)
      return;
    unsigned vOp = opVertex_.lookup(op);
    unsigned regionIndex = 0;
    for (Region &region : op->getRegions()) {
      bool ordered = !isUnorderedRegion(region);
      unsigned blockPos = 0;
      for (Block &block : region) {
        unsigned vBlock = blockVertex_.lookup(&block);
        addEdge(vOp, vBlock, EdgeKind::OwnsBlock, regionIndex,
                ordered ? blockPos : kNoOrdinal);
        unsigned opPos = 0;
        for (Operation &child : block) {
          addEdge(vBlock, opVertex_.lookup(&child), EdgeKind::BlockHoldsOp,
                  ordered ? opPos : kNoOrdinal);
          ++opPos;
        }
        for (BlockArgument arg : block.getArguments())
          addEdge(vBlock, valueVertex_.lookup(arg), EdgeKind::DefArg,
                  arg.getArgNumber());
        ++blockPos;
      }
      ++regionIndex;
    }
    for (OpResult result : op->getResults())
      addEdge(vOp, valueVertex_.lookup(result), EdgeKind::DefResult,
              result.getResultNumber());
    for (OpOperand &operand : op->getOpOperands())
      addEdge(valueVertex_.lookup(operand.get()), vOp, EdgeKind::Operand,
              operand.getOperandNumber());
    for (auto successor : llvm::enumerate(op->getSuccessors()))
      addEdge(vOp, blockVertex_.lookup(successor.value()), EdgeKind::Successor,
              static_cast<std::uint32_t>(successor.index()));

    const llvm::SmallVector<SymbolRefAttr> &symbols = opSymbols_.lookup(op);
    for (unsigned index = 0; index < symbols.size(); ++index) {
      Operation *def = SymbolTable::lookupNearestSymbolFrom(op, symbols[index]);
      if (!def) {
        error =
            relationError("canonical dataflow: unresolved symbol reference");
        return;
      }
      addEdge(vOp, opVertex_.lookup(def), EdgeKind::SymbolUse, index);
    }
  });
  return error;
}

llvm::Expected<SemanticGraph> SemanticGraph::build(ModuleOp module) {
  SemanticGraph graph;
  if (llvm::Error error = graph.collectVertices(module.getOperation()))
    return std::move(error);
  if (llvm::Error error = graph.detectCarriers(module))
    return std::move(error);
  if (llvm::Error error = graph.buildEdges(module))
    return std::move(error);
  return graph;
}

llvm::Expected<CanonicalLabeling> SemanticGraph::canonicalize() {
  std::vector<::loom::CanonicalRelationEdge> relations;
  relations.reserve(edges_.size());
  for (const Edge &edge : edges_)
    relations.push_back(
        {edge.from, edge.to, edgeLabel(edge.kind, edge.ord0, edge.ord1)});

  llvm::Expected<::loom::CanonicalRelationResult> result =
      ::loom::canonicalizeRelationGraph(intrinsic_, relations);
  if (!result)
    return result.takeError();

  std::vector<std::uint64_t> id(numVertices_,
                                std::numeric_limits<std::uint64_t>::max());
  std::uint64_t next = 0;
  for (std::uint32_t vertex : result->canonicalOrder)
    if (carrier_.count(vertex))
      id[vertex] = next++;
  std::vector<EntityCarrier> carriers;
  for (std::uint32_t vertex : result->canonicalOrder) {
    auto found = carrier_.find(vertex);
    if (found == carrier_.end())
      continue;
    EntityCarrier carrier = found->second;
    carrier.id = id[vertex];
    carriers.push_back(carrier);
  }
  std::sort(carriers.begin(), carriers.end(),
            [](const EntityCarrier &a, const EntityCarrier &b) {
              return a.id < b.id;
            });

  llvm::DenseMap<unsigned, Operation *> operationOfVertex;
  for (const auto &entry : opVertex_)
    operationOfVertex[entry.second] = entry.first;
  std::vector<Operation *> canonicalOperationOrder;
  canonicalOperationOrder.reserve(opVertex_.size());
  for (std::uint32_t vertex : result->canonicalOrder)
    if (Operation *op = operationOfVertex.lookup(vertex))
      canonicalOperationOrder.push_back(op);

  return CanonicalLabeling{std::move(result->bytes), std::move(carriers),
                           std::move(canonicalOperationOrder)};
}

} // namespace

llvm::Expected<CanonicalLabeling> computeCanonicalLabeling(ModuleOp module) {
  llvm::Expected<SemanticGraph> graph = SemanticGraph::build(module);
  if (!graph)
    return graph.takeError();
  return graph->canonicalize();
}

} // namespace dataflow::detail
