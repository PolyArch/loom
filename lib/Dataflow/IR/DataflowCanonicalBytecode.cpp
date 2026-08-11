#include "DataflowCanonicalBytecodeInternal.h"

#include "DataflowCanonicalLabeling.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace dataflow::detail {
namespace {

constexpr char semanticDomain[] = "loom.canonical_dataflow.semantic.v1\0";
constexpr llvm::StringRef semanticDomainRef(semanticDomain,
                                            sizeof(semanticDomain) - 1);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "canonical_dataflow_invalid: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("truncated bytecode length");
  std::uint64_t result = 0;
  for (unsigned index = 0; index != 8; ++index)
    result = (result << 8) | bytes[offset++];
  return result;
}

llvm::Error renameSymbol(Operation *symbol, StringAttr replacement,
                         Operation *symbolTable) {
  StringAttr old = SymbolTable::getSymbolName(symbol);
  if (old == replacement)
    return llvm::Error::success();
  if (failed(SymbolTable::replaceAllSymbolUses(old, replacement, symbolTable)))
    return invalid("cannot update symbol uses during canonicalization");
  symbol->setAttr(SymbolTable::getSymbolAttrName(), replacement);
  return llvm::Error::success();
}

llvm::Error canonicalizeSymbolTable(
    Operation *owner, const llvm::DenseMap<Operation *, std::uint64_t> &rank) {
  if (!owner->hasTrait<OpTrait::SymbolTable>())
    return llvm::Error::success();
  if (owner->getNumRegions() != 1 || !owner->getRegion(0).hasOneBlock())
    return invalid("symbol table lacks one body block");
  Block &block = owner->getRegion(0).front();
  llvm::SmallVector<Operation *> symbols;
  llvm::SmallVector<Operation *> privateSymbols;
  std::set<std::string> publicNames;
  std::set<std::string> occupied;
  for (Operation &child : block) {
    auto symbol = dyn_cast<SymbolOpInterface>(&child);
    if (!symbol)
      continue;
    if (!rank.count(&child))
      return invalid("canonical labeling omits a symbol");
    symbols.push_back(&child);
    occupied.insert(SymbolTable::getSymbolName(&child).str());
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
  auto fresh = [&](llvm::StringRef prefix) {
    std::string result = prefix.str();
    unsigned suffix = 0;
    while (occupied.count(result))
      result = (llvm::Twine(prefix) + "_" + std::to_string(++suffix)).str();
    occupied.insert(result);
    return result;
  };
  for (Operation *symbol : privateSymbols) {
    if (llvm::Error error = renameSymbol(
            symbol, StringAttr::get(owner->getContext(),
                                    fresh("__loom_private_staging")),
            owner))
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

llvm::Error canonicalizeUnorderedGraphBodies(
    ModuleOp module, const llvm::DenseMap<Operation *, std::uint64_t> &rank) {
  llvm::SmallVector<Block *> blocks;
  module.walk([&](GraphOp graph) {
    for (Region &region : graph->getRegions())
      for (Block &block : region)
        blocks.push_back(&block);
  });
  for (Block *block : blocks) {
    llvm::SmallVector<Operation *> operations;
    Operation *terminator = nullptr;
    for (Operation &operation : *block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        terminator = &operation;
        continue;
      }
      if (!rank.count(&operation))
        return invalid("canonical labeling omits a graph operation");
      operations.push_back(&operation);
    }
    llvm::sort(operations, [&](Operation *lhs, Operation *rhs) {
      return rank.lookup(lhs) < rank.lookup(rhs);
    });
    for (Operation *operation : operations) {
      if (terminator)
        operation->moveBefore(terminator);
      else
        operation->moveBefore(block, block->end());
    }
  }
  return llvm::Error::success();
}

llvm::Expected<CanonicalLabeling> canonicalizePresentation(ModuleOp module) {
  auto labeling = computeCanonicalLabeling(module);
  if (!labeling)
    return labeling.takeError();
  llvm::DenseMap<Operation *, std::uint64_t> rank;
  for (auto item : llvm::enumerate(labeling->canonicalOperationOrder))
    rank[item.value()] = item.index();
  llvm::Error result = llvm::Error::success();
  module.walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    if (llvm::Error error = canonicalizeSymbolTable(operation, rank)) {
      result = std::move(error);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result)
    return result;
  if (llvm::Error error = canonicalizeUnorderedGraphBodies(module, rank))
    return error;
  module.walk([&](Operation *operation) {
    operation->setLoc(UnknownLoc::get(module.getContext()));
  });
  if (failed(verify(module)))
    return invalid("presentation canonicalization produced invalid MLIR");
  return std::move(*labeling);
}

llvm::Expected<std::vector<std::uint8_t>> writeBytecodeOnce(Operation *root) {
  llvm::SmallVector<char> storage;
  llvm::raw_svector_ostream stream(storage);
  root->print(stream);
  return std::vector<std::uint8_t>(storage.begin(), storage.end());
}

} // namespace

llvm::Expected<CanonicalLabeling>
canonicalizeDataflowPresentation(ModuleOp module) {
  return canonicalizePresentation(module);
}

llvm::Expected<ParsedCanonicalDataflowModule>
parseCanonicalDataflowBytecode(llvm::ArrayRef<std::uint8_t> bytes) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<DataflowDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  llvm::StringRef byteString(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
  auto module = parseSourceString<ModuleOp>(byteString, context.get());
  if (!module || failed(verify(*module)))
    return invalid("canonical MLIR payload is not a valid builtin module");
  return ParsedCanonicalDataflowModule{std::move(context),
                                       std::move(module)};
}

llvm::Expected<std::vector<std::uint8_t>>
writeCanonicalizedDataflowBytecode(ModuleOp module) {
  if (failed(verify(module)))
    return invalid("canonical Dataflow presentation is not valid MLIR");
  return writeBytecodeOnce(module.getOperation());
}

::loom::CanonicalSemanticBytes
frameCanonicalDataflowBytes(llvm::ArrayRef<std::uint8_t> bytecode) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(semanticDomainRef.size() + 8 + bytecode.size());
  bytes.insert(bytes.end(), semanticDomainRef.begin(), semanticDomainRef.end());
  appendU64(bytes, bytecode.size());
  bytes.insert(bytes.end(), bytecode.begin(), bytecode.end());
  return ::loom::CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<llvm::ArrayRef<std::uint8_t>>
extractCanonicalDataflowBytecode(
    const ::loom::CanonicalSemanticBytes &canonicalBytes) {
  llvm::ArrayRef<std::uint8_t> bytes = canonicalBytes.bytes();
  llvm::ArrayRef<std::uint8_t> domain(
      reinterpret_cast<const std::uint8_t *>(semanticDomainRef.data()),
      semanticDomainRef.size());
  if (bytes.size() < domain.size() + 8 || !bytes.take_front(domain.size()).equals(domain))
    return invalid("wrong canonical Dataflow semantic-byte domain");
  std::size_t offset = domain.size();
  auto length = readU64(bytes, offset);
  if (!length)
    return length.takeError();
  if (*length != bytes.size() - offset)
    return invalid("canonical Dataflow bytecode length is noncanonical");
  return bytes.drop_front(offset);
}

} // namespace dataflow::detail
