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

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowInterfaces.h"
#include "Dataflow/IR/DataflowOps.h"

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
#include <functional>
#include <limits>
#include <map>
#include <set>
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
void putU64(std::string &s, std::uint64_t v) {
  for (int i = 7; i >= 0; --i)
    s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void putStr(std::string &s, StringRef x) {
  putU32(s, static_cast<std::uint32_t>(x.size()));
  s.append(x.data(), x.size());
}

llvm::Error relationError(const char *message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
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
std::string opIntrinsic(Operation *op,
                        llvm::SmallVectorImpl<SymbolRefAttr> &symbols) {
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

  CanonicalLabeling canonicalize();

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

  void collectVertices(Operation *op);
  llvm::Error detectCarriers(ModuleOp module);
  llvm::Error buildEdges(ModuleOp module);
  void buildAdjacency();

  std::vector<std::uint64_t> refine(std::vector<std::uint64_t> color) const;
  std::vector<std::uint64_t> assignIds(llvm::ArrayRef<unsigned> order) const;
  std::vector<std::uint8_t> serialize(llvm::ArrayRef<unsigned> order) const;

  struct Leaf {
    std::vector<std::uint8_t> bytes;
    std::vector<unsigned> order;
  };
  Leaf search(std::vector<std::uint64_t> color,
              std::vector<unsigned> &path) const;

  // Automorphisms discovered while searching, each a vertex permutation that
  // preserves the colored relation graph. Used only to prune orbit-equivalent
  // candidates; it never affects the canonical bytes.
  mutable std::vector<std::vector<unsigned>> automorphisms_;

  unsigned numVertices_ = 0;
  std::vector<std::string> intrinsic_;
  std::vector<Edge> edges_;
  std::vector<llvm::SmallVector<std::uint32_t>> outAdj_;
  std::vector<llvm::SmallVector<std::uint32_t>> inAdj_;

  llvm::DenseMap<Operation *, unsigned> opVertex_;
  llvm::DenseMap<Block *, unsigned> blockVertex_;
  llvm::DenseMap<Value, unsigned> valueVertex_;
  llvm::DenseMap<unsigned, EntityCarrier> carrier_;
  // Symbol references an op carries, in canonical pre-order; the index is the
  // symbol-use relation's semantic path key.
  llvm::DenseMap<Operation *, llvm::SmallVector<SymbolRefAttr>> opSymbols_;
};

void SemanticGraph::collectVertices(Operation *op) {
  llvm::SmallVector<SymbolRefAttr> symbols;
  opVertex_[op] = addVertex(opIntrinsic(op, symbols));
  opSymbols_[op] = std::move(symbols);
  for (Region &region : op->getRegions())
    for (Block &block : region) {
      blockVertex_[&block] = addVertex(blockIntrinsic(block));
      for (BlockArgument arg : block.getArguments())
        valueVertex_[arg] = addVertex(valueIntrinsic(arg));
      for (Operation &child : block) {
        for (OpResult res : child.getResults())
          valueVertex_[res] = addVertex(valueIntrinsic(res));
        collectVertices(&child);
      }
    }
}

llvm::Error SemanticGraph::detectCarriers(ModuleOp module) {
  auto record = [&](Value vertexValue, EntityCarrier carrier) {
    carrier_[valueVertex_.lookup(vertexValue)] = carrier;
  };
  auto recordOp = [&](Operation *op, EntityCarrier carrier) {
    carrier_[opVertex_.lookup(op)] = carrier;
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

void SemanticGraph::buildAdjacency() {
  outAdj_.assign(numVertices_, {});
  inAdj_.assign(numVertices_, {});
  for (std::uint32_t index = 0; index < edges_.size(); ++index) {
    outAdj_[edges_[index].from].push_back(index);
    inAdj_[edges_[index].to].push_back(index);
  }
}

llvm::Expected<SemanticGraph> SemanticGraph::build(ModuleOp module) {
  SemanticGraph graph;
  graph.collectVertices(module.getOperation());
  if (llvm::Error error = graph.detectCarriers(module))
    return std::move(error);
  if (llvm::Error error = graph.buildEdges(module))
    return std::move(error);
  graph.buildAdjacency();
  return graph;
}

//===----------------------------------------------------------------------===//
// Individualization-refinement
//===----------------------------------------------------------------------===//

std::vector<std::uint64_t>
SemanticGraph::refine(std::vector<std::uint64_t> color) const {
  while (true) {
    std::vector<std::string> signature(numVertices_);
    for (unsigned v = 0; v < numVertices_; ++v) {
      std::string sig;
      putU64(sig, color[v]);
      llvm::SmallVector<std::string> neighbors;
      for (std::uint32_t index : outAdj_[v]) {
        const Edge &edge = edges_[index];
        std::string n;
        putU8(n, 0);
        putU8(n, static_cast<std::uint8_t>(edge.kind));
        putU32(n, edge.ord0);
        putU32(n, edge.ord1);
        putU64(n, color[edge.to]);
        neighbors.push_back(std::move(n));
      }
      for (std::uint32_t index : inAdj_[v]) {
        const Edge &edge = edges_[index];
        std::string n;
        putU8(n, 1);
        putU8(n, static_cast<std::uint8_t>(edge.kind));
        putU32(n, edge.ord0);
        putU32(n, edge.ord1);
        putU64(n, color[edge.from]);
        neighbors.push_back(std::move(n));
      }
      std::sort(neighbors.begin(), neighbors.end());
      putU32(sig, static_cast<std::uint32_t>(neighbors.size()));
      for (const std::string &n : neighbors)
        sig.append(n);
      signature[v] = std::move(sig);
    }

    std::map<std::string, std::uint64_t> rank;
    for (const std::string &sig : signature)
      rank.emplace(sig, 0);
    std::uint64_t next = 0;
    for (auto &entry : rank)
      entry.second = next++;

    std::set<std::uint64_t> previous(color.begin(), color.end());
    for (unsigned v = 0; v < numVertices_; ++v)
      color[v] = rank[signature[v]];
    if (next == previous.size())
      return color;
  }
}

std::vector<std::uint64_t>
SemanticGraph::assignIds(llvm::ArrayRef<unsigned> order) const {
  std::vector<std::uint64_t> id(numVertices_,
                                std::numeric_limits<std::uint64_t>::max());
  std::uint64_t next = 0;
  for (unsigned vertex : order)
    if (carrier_.count(vertex))
      id[vertex] = next++;
  return id;
}

std::vector<std::uint8_t>
SemanticGraph::serialize(llvm::ArrayRef<unsigned> order) const {
  std::vector<unsigned> rank(numVertices_);
  for (unsigned position = 0; position < order.size(); ++position)
    rank[order[position]] = position;
  std::vector<std::uint64_t> id = assignIds(order);

  // The schema name and version are not re-embedded here: the Common finalizer
  // already frames canonicalDataflowSchema around these family-owned bytes, so
  // duplicating the tag would create a second, divergable authority.
  std::string out;
  putU32(out, numVertices_);
  for (unsigned vertex : order) {
    putStr(out, intrinsic_[vertex]);
    auto found = carrier_.find(vertex);
    if (found != carrier_.end()) {
      putU8(out, 1);
      putU8(out, static_cast<std::uint8_t>(found->second.kind));
      putU64(out, id[vertex]);
    } else {
      putU8(out, 0);
    }
    llvm::SmallVector<std::array<std::uint64_t, 4>> outgoing;
    for (std::uint32_t index : outAdj_[vertex]) {
      const Edge &edge = edges_[index];
      outgoing.push_back({static_cast<std::uint64_t>(edge.kind), edge.ord0,
                          edge.ord1, rank[edge.to]});
    }
    std::sort(outgoing.begin(), outgoing.end());
    putU32(out, static_cast<std::uint32_t>(outgoing.size()));
    for (const std::array<std::uint64_t, 4> &edge : outgoing) {
      putU8(out, static_cast<std::uint8_t>(edge[0]));
      putU32(out, static_cast<std::uint32_t>(edge[1]));
      putU32(out, static_cast<std::uint32_t>(edge[2]));
      putU32(out, static_cast<std::uint32_t>(edge[3]));
    }
  }
  return std::vector<std::uint8_t>(out.begin(), out.end());
}

SemanticGraph::Leaf SemanticGraph::search(std::vector<std::uint64_t> color,
                                          std::vector<unsigned> &path) const {
  color = refine(std::move(color));

  std::map<std::uint64_t, llvm::SmallVector<unsigned>> cells;
  for (unsigned v = 0; v < numVertices_; ++v)
    cells[color[v]].push_back(v);

  const llvm::SmallVector<unsigned> *target = nullptr;
  for (const auto &cell : cells)
    if (cell.second.size() > 1) {
      target = &cell.second;
      break;
    }

  if (!target) {
    Leaf leaf;
    leaf.order.resize(numVertices_);
    for (unsigned v = 0; v < numVertices_; ++v)
      leaf.order[v] = v;
    std::sort(leaf.order.begin(), leaf.order.end(),
              [&](unsigned a, unsigned b) { return color[a] < color[b]; });
    leaf.bytes = serialize(leaf.order);
    return leaf;
  }

  std::uint64_t fresh = 0;
  for (std::uint64_t c : color)
    fresh = std::max(fresh, c);
  ++fresh;

  // Orbit pruning under the discovered automorphism group. Two target-cell
  // candidates in the same orbit of an automorphism that fixes every ancestor
  // individualization on `path` open isomorphic subtrees and therefore the same
  // canonical leaf, so only one orbit representative is explored. This is the
  // exact mechanism that collapses the factorial candidate product for a
  // symmetric input without any noncanonical source-order tie-break; the
  // canonical bytes are exactly those the unpruned search would minimize to.
  std::vector<unsigned> parent(numVertices_);
  for (unsigned v = 0; v < numVertices_; ++v)
    parent[v] = v;
  std::function<unsigned(unsigned)> find = [&](unsigned x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](unsigned a, unsigned b) { parent[find(a)] = find(b); };
  auto fixesPath = [&](const std::vector<unsigned> &perm) {
    for (unsigned u : path)
      if (perm[u] != u)
        return false;
    return true;
  };
  auto applyAutomorphism = [&](const std::vector<unsigned> &perm) {
    if (!fixesPath(perm))
      return;
    for (unsigned v = 0; v < numVertices_; ++v)
      unite(v, perm[v]);
  };
  for (const std::vector<unsigned> &perm : automorphisms_)
    applyAutomorphism(perm);

  Leaf best;
  bool haveBest = false;
  std::set<unsigned> exploredReps;
  for (unsigned candidate : *target) {
    unsigned rep = find(candidate);
    if (!exploredReps.insert(rep).second)
      continue;
    std::vector<std::uint64_t> individualized = color;
    individualized[candidate] = fresh;
    path.push_back(candidate);
    Leaf leaf = search(std::move(individualized), path);
    path.pop_back();
    if (!haveBest || leaf.bytes < best.bytes) {
      best = std::move(leaf);
      haveBest = true;
    } else if (leaf.bytes == best.bytes) {
      // Equal canonical bytes witness an automorphism: the vertex at each
      // canonical position in `leaf` plays the same role as in `best`. Record
      // it and, if it fixes this node's ancestors, extend the orbit partition
      // so later candidates in its orbit are skipped.
      std::vector<unsigned> perm(numVertices_);
      for (unsigned i = 0; i < numVertices_; ++i)
        perm[leaf.order[i]] = best.order[i];
      applyAutomorphism(perm);
      automorphisms_.push_back(std::move(perm));
    }
  }
  return best;
}

CanonicalLabeling SemanticGraph::canonicalize() {
  std::map<std::string, std::uint64_t> rank;
  for (const std::string &intrinsic : intrinsic_)
    rank.emplace(intrinsic, 0);
  std::uint64_t next = 0;
  for (auto &entry : rank)
    entry.second = next++;
  std::vector<std::uint64_t> initial(numVertices_);
  for (unsigned v = 0; v < numVertices_; ++v)
    initial[v] = rank[intrinsic_[v]];

  std::vector<unsigned> path;
  Leaf leaf = search(std::move(initial), path);

  std::vector<std::uint64_t> id = assignIds(leaf.order);
  std::vector<EntityCarrier> carriers;
  for (unsigned vertex : leaf.order) {
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
  for (unsigned vertex : leaf.order)
    if (Operation *op = operationOfVertex.lookup(vertex))
      canonicalOperationOrder.push_back(op);

  return CanonicalLabeling{
      ::loom::CanonicalSemanticBytes(std::move(leaf.bytes)),
      std::move(carriers), std::move(canonicalOperationOrder)};
}

} // namespace

llvm::Expected<CanonicalLabeling> computeCanonicalLabeling(ModuleOp module) {
  llvm::Expected<SemanticGraph> graph = SemanticGraph::build(module);
  if (!graph)
    return graph.takeError();
  return graph->canonicalize();
}

} // namespace dataflow::detail
