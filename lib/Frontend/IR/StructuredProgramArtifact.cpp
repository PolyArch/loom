#include "Frontend/IR/StructuredProgramArtifact.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/CanonicalRelation.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/IR/LoomDialect.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::frontend {
namespace {

constexpr char semanticDomain[] = "loom.structured_program.semantic.v1\0";
constexpr llvm::StringRef semanticDomainRef(semanticDomain,
                                            sizeof(semanticDomain) - 1);
constexpr std::uint32_t noOrdinal = 0xFFFFFFFFu;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_program_invalid: " + message);
}

std::size_t kindIndex(StructuredEntityKind kind) {
  const auto index = static_cast<std::size_t>(kind);
  assert(index < 4 && "unknown StructuredEntityKind");
  return index;
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint32_t> readU32(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 4)
    return invalid("truncated u32");
  std::uint32_t result = 0;
  for (unsigned index = 0; index != 4; ++index)
    result = (result << 8) | bytes[offset++];
  return result;
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("truncated u64");
  std::uint64_t result = 0;
  for (unsigned index = 0; index != 8; ++index)
    result = (result << 8) | bytes[offset++];
  return result;
}

bool isTransientAttribute(StringRef name) {
  return name == "loom.source_hint" || name == "loom.candidate_hint" ||
         name == "loom.visual_metadata" || name == "graph_name";
}

llvm::Error removeTransients(ModuleOp module) {
  llvm::Error result = llvm::Error::success();
  module.walk([&](Operation *op) {
    if (result)
      return WalkResult::interrupt();
    SmallVector<StringAttr> erase;
    for (NamedAttribute attribute : op->getAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (isTransientAttribute(name)) {
        erase.push_back(attribute.getName());
        continue;
      }
      if (name.starts_with("loom.")) {
        result = invalid(llvm::Twine("unresolved Loom-specific attribute '") +
                         name + "' on " + op->getName().getStringRef());
        return WalkResult::interrupt();
      }
    }
    for (StringAttr name : erase)
      op->removeAttr(name);
    return WalkResult::advance();
  });
  return result;
}

enum class RelationKind : std::uint8_t {
  OwnsRegion,
  RegionHoldsBlock,
  BlockHoldsOperation,
  DefinesResult,
  DefinesArgument,
  Operand,
  Successor,
  SymbolUse,
};

struct Relation {
  std::uint32_t source;
  std::uint32_t target;
  RelationKind kind;
  std::uint32_t first;
  std::uint32_t second = noOrdinal;
};

void appendU8(std::string &bytes, std::uint8_t value) {
  bytes.push_back(static_cast<char>(value));
}

void appendRelationU32(std::string &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>(value >> shift));
}

std::string relationLabel(RelationKind kind, std::uint32_t first,
                          std::uint32_t second) {
  std::string result;
  appendU8(result, static_cast<std::uint8_t>(kind));
  appendRelationU32(result, first);
  appendRelationU32(result, second);
  return result;
}

bool isUnorderedRegion(Region &region) {
  return region.getParentOp()->hasTrait<OpTrait::SymbolTable>();
}

bool redactSymbolName(Operation *op) {
  auto symbol = dyn_cast<SymbolOpInterface>(op);
  return symbol && symbol.getVisibility() != SymbolTable::Visibility::Public;
}

void serializeAttribute(llvm::raw_ostream &stream, Attribute attribute,
                        SmallVectorImpl<SymbolRefAttr> &symbols) {
  if (isa<LocationAttr>(attribute)) {
    stream << "#location";
    return;
  }
  if (auto symbol = dyn_cast<SymbolRefAttr>(attribute)) {
    stream << "#symbol";
    symbols.push_back(symbol);
    return;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    stream << '{';
    for (NamedAttribute entry : dictionary) {
      stream << entry.getName().getValue() << '=';
      serializeAttribute(stream, entry.getValue(), symbols);
      stream << ';';
    }
    stream << '}';
    return;
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    stream << '[';
    for (Attribute entry : array) {
      serializeAttribute(stream, entry, symbols);
      stream << ';';
    }
    stream << ']';
    return;
  }
  stream << "#leaf<";
  attribute.print(stream);
  stream << '>';
}

llvm::Expected<std::string>
operationIntrinsic(Operation *op, SmallVectorImpl<SymbolRefAttr> &symbols) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << "OP\x1f" << op->getName().getStringRef() << '\x1e';
  stream << op->getNumOperands() << ':' << op->getNumResults() << ':'
         << op->getNumRegions() << ':' << op->getNumSuccessors() << '\x1e';

  llvm::SmallDenseSet<StringRef, 8> registered;
  for (StringAttr name : op->getName().getAttributeNames())
    registered.insert(name.getValue());
  const StringRef symbolName = SymbolTable::getSymbolAttrName();
  const bool redacted = redactSymbolName(op);
  auto appendDictionary = [&](DictionaryAttr dictionary, bool registeredOnly) {
    if (!dictionary)
      return;
    for (NamedAttribute entry : dictionary) {
      StringRef name = entry.getName().getValue();
      if (name == symbolName && redacted)
        continue;
      if (registeredOnly && !registered.contains(name))
        continue;
      stream << name << '=';
      serializeAttribute(stream, entry.getValue(), symbols);
      stream << '\x1e';
    }
  };
  appendDictionary(
      dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()), false);
  appendDictionary(op->getDiscardableAttrDictionary(), true);
  return result;
}

std::string valueIntrinsic(Value value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << (isa<BlockArgument>(value) ? "argument" : "result") << ':';
  value.getType().print(stream);
  return result;
}

std::string regionIntrinsic(Region &region) {
  return region.empty() ? "region:empty" : "region:nonempty";
}

std::string blockIntrinsic(Block &block) {
  return "block:" + std::to_string(block.getNumArguments()) + ':' +
         (block.isEntryBlock() ? "entry" : "ordinary");
}

struct CanonicalLabeling {
  CanonicalSemanticBytes relationBytes;
  std::vector<Operation *> operations;
  struct EntityCarrier {
    StructuredEntityKind kind;
    Operation *operation = nullptr;
    Region *region = nullptr;
    Block *block = nullptr;
    Value value;
  };
  std::array<std::vector<EntityCarrier>, 4> entities;
};

class SemanticGraph {
public:
  static llvm::Expected<SemanticGraph> build(ModuleOp module) {
    SemanticGraph graph;
    if (llvm::Error error = graph.collect(module.getOperation()))
      return std::move(error);
    if (llvm::Error error = graph.connect(module.getOperation()))
      return std::move(error);
    return graph;
  }

  llvm::Expected<CanonicalLabeling> canonicalize() const {
    std::vector<CanonicalRelationEdge> relations;
    relations.reserve(relations_.size());
    for (const Relation &relation : relations_)
      relations.push_back(
          {relation.source, relation.target,
           relationLabel(relation.kind, relation.first, relation.second)});
    auto result = canonicalizeRelationGraph(intrinsics_, relations);
    if (!result)
      return result.takeError();

    CanonicalLabeling labeling{std::move(result->bytes), {}, {}};
    for (std::uint32_t vertex : result->canonicalOrder) {
      if (Operation *op = operationOfVertex_.lookup(vertex))
        labeling.operations.push_back(op);
      if (auto found = entityOfVertex_.find(vertex);
          found != entityOfVertex_.end())
        labeling.entities[kindIndex(found->second.kind)].push_back(
            found->second);
    }
    return labeling;
  }

private:
  std::uint32_t add(std::string intrinsic) {
    const std::uint32_t vertex = intrinsics_.size();
    intrinsics_.push_back(std::move(intrinsic));
    return vertex;
  }

  void edge(std::uint32_t source, std::uint32_t target, RelationKind kind,
            std::uint32_t first, std::uint32_t second = noOrdinal) {
    relations_.push_back({source, target, kind, first, second});
  }

  void recordEntity(std::uint32_t vertex, StructuredEntityKind kind,
                    Operation *operation = nullptr, Region *region = nullptr,
                    Block *block = nullptr, Value value = {}) {
    entityOfVertex_.try_emplace(
        vertex, CanonicalLabeling::EntityCarrier{kind, operation, region, block,
                                                 value});
  }

  llvm::Error collect(Operation *op) {
    SmallVector<SymbolRefAttr> symbols;
    auto intrinsic = operationIntrinsic(op, symbols);
    if (!intrinsic)
      return intrinsic.takeError();
    const std::uint32_t opVertex = add(std::move(*intrinsic));
    operationVertex_[op] = opVertex;
    operationOfVertex_[opVertex] = op;
    symbols_[op] = std::move(symbols);
    recordEntity(opVertex, StructuredEntityKind::Operation, op);

    for (Region &region : op->getRegions()) {
      const std::uint32_t regionVertex = add(regionIntrinsic(region));
      regionVertex_[&region] = regionVertex;
      recordEntity(regionVertex, StructuredEntityKind::Region, nullptr,
                   &region);
      for (Block &block : region) {
        const std::uint32_t blockVertex = add(blockIntrinsic(block));
        blockVertex_[&block] = blockVertex;
        recordEntity(blockVertex, StructuredEntityKind::Block, nullptr, nullptr,
                     &block);
        for (BlockArgument argument : block.getArguments()) {
          const std::uint32_t valueVertex = add(valueIntrinsic(argument));
          valueVertex_[argument] = valueVertex;
          recordEntity(valueVertex, StructuredEntityKind::Value, nullptr,
                       nullptr, nullptr, argument);
        }
        for (Operation &child : block) {
          for (OpResult result : child.getResults()) {
            const std::uint32_t valueVertex = add(valueIntrinsic(result));
            valueVertex_[result] = valueVertex;
            recordEntity(valueVertex, StructuredEntityKind::Value, nullptr,
                         nullptr, nullptr, result);
          }
          if (llvm::Error error = collect(&child))
            return error;
        }
      }
    }
    return llvm::Error::success();
  }

  llvm::Error connect(Operation *root) {
    llvm::Error result = llvm::Error::success();
    root->walk([&](Operation *op) {
      if (result)
        return WalkResult::interrupt();
      const std::uint32_t opVertex = operationVertex_.lookup(op);
      for (auto regionIt : llvm::enumerate(op->getRegions())) {
        Region &region = regionIt.value();
        const std::uint32_t regionVertex = regionVertex_.lookup(&region);
        edge(opVertex, regionVertex, RelationKind::OwnsRegion,
             static_cast<std::uint32_t>(regionIt.index()));
        const bool ordered = !isUnorderedRegion(region);
        for (auto blockIt : llvm::enumerate(region)) {
          Block &block = blockIt.value();
          const std::uint32_t blockVertex = blockVertex_.lookup(&block);
          edge(regionVertex, blockVertex, RelationKind::RegionHoldsBlock,
               ordered ? static_cast<std::uint32_t>(blockIt.index())
                       : noOrdinal);
          for (auto childIt : llvm::enumerate(block))
            edge(blockVertex, operationVertex_.lookup(&childIt.value()),
                 RelationKind::BlockHoldsOperation,
                 ordered ? static_cast<std::uint32_t>(childIt.index())
                         : noOrdinal);
          for (BlockArgument argument : block.getArguments())
            edge(blockVertex, valueVertex_.lookup(argument),
                 RelationKind::DefinesArgument, argument.getArgNumber());
        }
      }
      for (OpResult value : op->getResults())
        edge(opVertex, valueVertex_.lookup(value), RelationKind::DefinesResult,
             value.getResultNumber());
      for (OpOperand &operand : op->getOpOperands())
        edge(valueVertex_.lookup(operand.get()), opVertex,
             RelationKind::Operand, operand.getOperandNumber());
      for (auto successor : llvm::enumerate(op->getSuccessors()))
        edge(opVertex, blockVertex_.lookup(successor.value()),
             RelationKind::Successor,
             static_cast<std::uint32_t>(successor.index()));
      for (auto [index, symbol] : llvm::enumerate(symbols_.lookup(op))) {
        Operation *definition =
            SymbolTable::lookupNearestSymbolFrom(op, symbol);
        if (!definition) {
          result = invalid(llvm::Twine("unresolved symbol reference in ") +
                           op->getName().getStringRef());
          return WalkResult::interrupt();
        }
        edge(opVertex, operationVertex_.lookup(definition),
             RelationKind::SymbolUse, static_cast<std::uint32_t>(index));
      }
      return WalkResult::advance();
    });
    return result;
  }

  std::vector<std::string> intrinsics_;
  std::vector<Relation> relations_;
  DenseMap<Operation *, std::uint32_t> operationVertex_;
  DenseMap<std::uint32_t, Operation *> operationOfVertex_;
  DenseMap<Region *, std::uint32_t> regionVertex_;
  DenseMap<Block *, std::uint32_t> blockVertex_;
  DenseMap<Value, std::uint32_t> valueVertex_;
  DenseMap<Operation *, SmallVector<SymbolRefAttr>> symbols_;
  DenseMap<std::uint32_t, CanonicalLabeling::EntityCarrier> entityOfVertex_;
};

llvm::Expected<CanonicalLabeling> label(ModuleOp module) {
  auto graph = SemanticGraph::build(module);
  if (!graph)
    return graph.takeError();
  return graph->canonicalize();
}

llvm::Error renameSymbol(Operation *symbol, StringAttr replacement,
                         Operation *symbolTable) {
  StringAttr old = SymbolTable::getSymbolName(symbol);
  if (old == replacement)
    return llvm::Error::success();
  if (failed(SymbolTable::replaceAllSymbolUses(old, replacement, symbolTable)))
    return invalid(
        llvm::Twine("cannot update every symbol use while canonicalizing '") +
        old.getValue() + "'");
  symbol->setAttr(SymbolTable::getSymbolAttrName(), replacement);
  return llvm::Error::success();
}

llvm::Error canonicalizeSymbolTable(Operation *owner,
                                    const CanonicalLabeling &labeling) {
  if (!owner->hasTrait<OpTrait::SymbolTable>())
    return llvm::Error::success();
  if (owner->getNumRegions() != 1 || !owner->getRegion(0).hasOneBlock())
    return invalid("a Structured Program symbol table lacks one body block");
  Block &block = owner->getRegion(0).front();
  DenseMap<Operation *, std::size_t> rank;
  for (auto item : llvm::enumerate(labeling.operations))
    rank[item.value()] = item.index();

  SmallVector<Operation *> symbols;
  SmallVector<Operation *> privateSymbols;
  std::set<std::string> publicNames;
  for (Operation &child : block) {
    auto symbol = dyn_cast<SymbolOpInterface>(&child);
    if (!symbol)
      continue;
    symbols.push_back(&child);
    if (symbol.getVisibility() == SymbolTable::Visibility::Public)
      publicNames.insert(SymbolTable::getSymbolName(&child).str());
    else
      privateSymbols.push_back(&child);
  }
  llvm::sort(symbols, [&](Operation *lhs, Operation *rhs) {
    return rank.lookup(lhs) < rank.lookup(rhs);
  });
  llvm::sort(privateSymbols, [&](Operation *lhs, Operation *rhs) {
    return rank.lookup(lhs) < rank.lookup(rhs);
  });

  std::set<std::string> occupied;
  for (Operation *symbol : symbols)
    occupied.insert(SymbolTable::getSymbolName(symbol).str());
  auto fresh = [&](llvm::StringRef prefix) {
    std::string result = prefix.str();
    unsigned suffix = 0;
    while (occupied.count(result))
      result = (llvm::Twine(prefix) + "_" + std::to_string(++suffix)).str();
    occupied.insert(result);
    return result;
  };
  for (Operation *symbol : privateSymbols) {
    std::string staging = fresh("__loom_private_staging");
    if (llvm::Error error = renameSymbol(
            symbol, StringAttr::get(symbol->getContext(), staging), owner))
      return error;
  }
  for (auto item : llvm::enumerate(privateSymbols)) {
    std::string base = "__loom_private_" + std::to_string(item.index());
    std::string final = base;
    unsigned suffix = 0;
    while (publicNames.count(final))
      final = base + "_" + std::to_string(++suffix);
    if (llvm::Error error = renameSymbol(
            item.value(), StringAttr::get(owner->getContext(), final), owner))
      return error;
  }
  for (Operation *symbol : symbols)
    symbol->moveBefore(&block, block.end());
  return llvm::Error::success();
}

llvm::Error canonicalizeSymbols(ModuleOp module,
                                const CanonicalLabeling &labeling) {
  llvm::Error result = llvm::Error::success();
  module.walk([&](Operation *op) {
    if (result)
      return WalkResult::interrupt();
    if (llvm::Error error = canonicalizeSymbolTable(op, labeling)) {
      result = std::move(error);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

llvm::Expected<std::vector<std::uint8_t>> writeBytecodeOnce(Operation *root) {
  SmallVector<char> storage;
  llvm::raw_svector_ostream stream(storage);
  BytecodeWriterConfig config("loom.structured_program.1.0");
  config.setElideLocations();
  if (failed(writeBytecodeToFile(root, stream, config)))
    return invalid("MLIR bytecode writer rejected the candidate");
  return std::vector<std::uint8_t>(storage.begin(), storage.end());
}

struct ParsedModule {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
};

llvm::Expected<ParsedModule> parseBytecode(ArrayRef<std::uint8_t> bytes) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  registry.insert<::dataflow::DataflowDialect, ::loom::LoomDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  StringRef text(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  llvm::MemoryBufferRef buffer(text, "<canonical-structured-program>");
  ParserConfig config(context.get());
  Block topLevel;
  if (failed(readBytecodeFile(buffer, &topLevel, config)))
    return invalid("canonical MLIR bytecode cannot be parsed");
  if (!llvm::hasSingleElement(topLevel))
    return invalid("canonical MLIR bytecode has multiple roots");
  auto module = dyn_cast<ModuleOp>(&topLevel.front());
  if (!module || failed(verify(module)))
    return invalid("canonical MLIR bytecode is not a verified builtin.module");
  module->remove();
  return ParsedModule{std::move(context), OwningOpRef<ModuleOp>(module)};
}

struct CanonicalModule {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  std::vector<std::uint8_t> bytecode;
};

llvm::Expected<CanonicalModule> canonicalModule(ModuleOp source) {
  auto initial = writeBytecodeOnce(source.getOperation());
  if (!initial)
    return initial.takeError();
  auto normalized = parseBytecode(*initial);
  if (!normalized)
    return normalized.takeError();
  auto canonical = writeBytecodeOnce(normalized->module.get());
  if (!canonical)
    return canonical.takeError();

  // Finalization validates the exact bytes it will publish in a second fresh
  // context. The retained module and view therefore come from the same strict
  // import that proves the family writer is byte stable.
  auto verified = parseBytecode(*canonical);
  if (!verified)
    return verified.takeError();
  auto rewritten = writeBytecodeOnce(verified->module.get());
  if (!rewritten)
    return rewritten.takeError();
  if (*canonical != *rewritten)
    return invalid("the Structured Program bytecode writer is not byte stable");
  return CanonicalModule{std::move(verified->context),
                         std::move(verified->module), std::move(*canonical)};
}

CanonicalSemanticBytes frameSemanticBytes(ArrayRef<std::uint8_t> bytecode) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(semanticDomainRef.size() + 8 + bytecode.size());
  bytes.insert(bytes.end(), semanticDomainRef.begin(), semanticDomainRef.end());
  appendU64(bytes, bytecode.size());
  bytes.insert(bytes.end(), bytecode.begin(), bytecode.end());
  return CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<ArrayRef<std::uint8_t>>
extractBytecode(const CanonicalSemanticBytes &semanticBytes) {
  ArrayRef<std::uint8_t> bytes = semanticBytes.bytes();
  if (bytes.size() < semanticDomainRef.size() + 8 ||
      !bytes.take_front(semanticDomainRef.size())
           .equals(ArrayRef<std::uint8_t>(
               reinterpret_cast<const std::uint8_t *>(semanticDomainRef.data()),
               semanticDomainRef.size())))
    return invalid("wrong Structured Program semantic-byte domain");
  std::size_t offset = semanticDomainRef.size();
  auto length = readU64(bytes, offset);
  if (!length)
    return length.takeError();
  if (*length != bytes.size() - offset)
    return invalid("Structured Program bytecode length is noncanonical");
  return bytes.drop_front(offset);
}

} // namespace

llvm::Expected<StructuredProgramCandidateView>
buildStructuredProgramCandidateView(ModuleOp module,
                                    const ArtifactIdentity &identity) {
  auto labeling = label(module);
  if (!labeling)
    return labeling.takeError();
  StructuredProgramCandidateView view(identity);
  for (std::size_t index = 0; index < labeling->entities.size(); ++index) {
    auto &source = labeling->entities[index];
    auto &target = view.entities_[index];
    target.reserve(source.size());
    for (auto item : llvm::enumerate(source)) {
      const auto &carrier = item.value();
      target.push_back(
          {StructuredEntityRef{identity, carrier.kind,
                               static_cast<std::uint64_t>(item.index())},
           carrier.operation, carrier.region, carrier.block, carrier.value});
    }
  }
  return view;
}

namespace {

llvm::Expected<OwningOpRef<ModuleOp>> canonicalizeClone(ModuleOp source) {
  auto clone = OwningOpRef<ModuleOp>(cast<ModuleOp>(source->clone()));
  if (failed(verify(*clone)))
    return invalid("candidate does not verify before canonicalization");
  if (llvm::Error error = removeTransients(*clone))
    return std::move(error);
  auto first = label(*clone);
  if (!first)
    return first.takeError();
  if (llvm::Error error = canonicalizeSymbols(*clone, *first))
    return std::move(error);
  auto second = label(*clone);
  if (!second)
    return second.takeError();
  if (llvm::Error error = canonicalizeSymbols(*clone, *second))
    return std::move(error);
  if (failed(verify(*clone)))
    return invalid("candidate does not verify after canonicalization");
  return clone;
}

} // namespace

std::vector<std::uint8_t>
encodeStructuredEntityRef(const StructuredEntityRef &reference) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(ArtifactIdentity::byteSize + 12);
  bytes.insert(bytes.end(), reference.parent.bytes().begin(),
               reference.parent.bytes().end());
  appendU32(bytes, static_cast<std::uint32_t>(reference.kind));
  appendU64(bytes, reference.ordinal);
  return bytes;
}

llvm::Expected<StructuredEntityRef>
decodeStructuredEntityRef(ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != ArtifactIdentity::byteSize + 12)
    return invalid("StructuredEntityRef has the wrong wire size");
  auto parent =
      ArtifactIdentity::fromBytes(bytes.take_front(ArtifactIdentity::byteSize));
  if (!parent)
    return parent.takeError();
  std::size_t offset = ArtifactIdentity::byteSize;
  auto kind = readU32(bytes, offset);
  if (!kind)
    return kind.takeError();
  if (*kind > static_cast<std::uint32_t>(StructuredEntityKind::Value))
    return invalid("StructuredEntityRef has an unknown entity kind");
  auto ordinal = readU64(bytes, offset);
  if (!ordinal)
    return ordinal.takeError();
  return StructuredEntityRef{*parent, static_cast<StructuredEntityKind>(*kind),
                             *ordinal};
}

ArrayRef<StructuredEntity>
StructuredProgramCandidateView::entities(StructuredEntityKind kind) const {
  return entities_[kindIndex(kind)];
}

llvm::Expected<StructuredEntity> StructuredProgramCandidateView::resolve(
    const StructuredEntityRef &reference) const {
  if (reference.parent != identity_)
    return invalid("StructuredEntityRef belongs to a different candidate");
  ArrayRef<StructuredEntity> candidates = entities(reference.kind);
  if (reference.ordinal >= candidates.size())
    return invalid("StructuredEntityRef ordinal is out of range");
  return candidates[reference.ordinal];
}

llvm::Expected<StructuredProgramCandidateView>
StructuredProgramCandidate::view() const {
  return view_;
}

llvm::Expected<StructuredProgramCandidate>
finalizeStructuredProgram(ModuleOp source) {
  auto clone = canonicalizeClone(source);
  if (!clone)
    return clone.takeError();
  auto canonical = canonicalModule(clone->get());
  if (!canonical)
    return canonical.takeError();
  CanonicalSemanticBytes semantic = frameSemanticBytes(canonical->bytecode);
  ArtifactIdentity identity =
      finalizeArtifactIdentity(structuredProgramArtifactSchema, semantic);
  auto view =
      buildStructuredProgramCandidateView(canonical->module.get(), identity);
  if (!view)
    return view.takeError();
  return StructuredProgramCandidate(
      identity, std::move(semantic), std::move(canonical->context),
      std::move(canonical->module), std::move(*view));
}

llvm::Expected<StructuredProgramCandidate>
importStructuredProgram(const ArtifactIdentity &identity,
                        const CanonicalSemanticBytes &canonicalBytes) {
  if (finalizeArtifactIdentity(structuredProgramArtifactSchema,
                               canonicalBytes) != identity)
    return invalid(
        "Structured Program identity does not match canonical bytes");
  auto bytecode = extractBytecode(canonicalBytes);
  if (!bytecode)
    return bytecode.takeError();
  auto parsed = parseBytecode(*bytecode);
  if (!parsed)
    return parsed.takeError();
  auto rewritten = writeBytecodeOnce(parsed->module.get().getOperation());
  if (!rewritten)
    return rewritten.takeError();
  CanonicalSemanticBytes reencoded = frameSemanticBytes(*rewritten);
  if (!reencoded.bytes().equals(canonicalBytes.bytes()))
    return invalid("Structured Program bytes are not canonical");
  auto view =
      buildStructuredProgramCandidateView(parsed->module.get(), identity);
  if (!view)
    return view.takeError();
  return StructuredProgramCandidate(
      identity, canonicalBytes, std::move(parsed->context),
      std::move(parsed->module), std::move(*view));
}

llvm::Expected<ArtifactRootReference>
publishStructuredProgram(const StructuredProgramCandidate &candidate,
                         const ArtifactStore &store) {
  auto stored =
      store.put(structuredProgramArtifactSchema, candidate.canonicalBytes());
  if (!stored)
    return stored.takeError();
  if (*stored != candidate.identity())
    return invalid(
        "ArtifactStore returned a different Structured Program identity");
  return ArtifactRootReference{structuredProgramArtifactSchema.identity.str(),
                               structuredProgramArtifactSchema.version,
                               *stored};
}

llvm::Expected<StructuredProgramCandidate>
importStructuredProgram(const ArtifactRootReference &reference,
                        const ArtifactStore &store) {
  if (reference.schemaIdentity != structuredProgramArtifactSchema.identity ||
      reference.schemaVersion != structuredProgramArtifactSchema.version)
    return invalid("foreign Structured Program reference schema");
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  return importStructuredProgram(reference.artifact, *bytes);
}

} // namespace loom::frontend
