#include "Fabric/Tech/TemplateLibrary.h"

#include "Fabric/Tech/SubgraphEnumerator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace fabric {

TemplateLibrary::~TemplateLibrary() = default;

namespace {

// The "root" of a candidate subgraph is the op that feeds the yield. For a
// 1-op body this is just that op; for multi-op bodies it's the producer of
// the (single) yielded value. The partitioner indexes templates by root
// op name to skip non-matching candidates fast.
::llvm::StringRef computeRootOpName(::dataflow::SubgraphOp sg) {
  ::mlir::Block &body = sg.getBody().front();
  // YieldOp is the terminator; its first operand's defining op is the root.
  ::mlir::Operation *term = body.getTerminator();
  if (!term || term->getNumOperands() == 0)
    return {};
  ::mlir::Value yielded = term->getOperand(0);
  ::mlir::Operation *def = yielded.getDefiningOp();
  if (!def)
    return {};
  return def->getName().getStringRef();
}

unsigned countBodyOps(::dataflow::SubgraphOp sg) {
  unsigned n = 0;
  for (auto &op : sg.getBody().front().without_terminator())
    (void)op, ++n;
  return n;
}

} // namespace

::llvm::ArrayRef<unsigned>
TemplateLibrary::templatesByRootOp(::llvm::StringRef rootName) const {
  auto it = rootIndex.find(rootName);
  if (it == rootIndex.end())
    return {};
  return it->second;
}

std::unique_ptr<TemplateLibrary>
TemplateLibrary::build(::mlir::MLIRContext *ctx,
                       ::llvm::ArrayRef<FuOp> fus) {
  std::unique_ptr<TemplateLibrary> lib(new TemplateLibrary());
  lib->moduleRef = ::mlir::OwningOpRef<::mlir::ModuleOp>(
      ::mlir::ModuleOp::create(::mlir::UnknownLoc::get(ctx), "loom_templates"));

  unsigned nextId = 0;
  for (FuOp fu : fus) {
    ::llvm::StringRef unsupported;
    auto cands = enumerateFuSubgraphs(fu, lib->moduleRef.get(),
                                       "tpl_" + std::to_string(nextId),
                                       &unsupported);
    if (!unsupported.empty())
      continue;
    for (auto &c : cands) {
      FuTemplate t;
      t.id = nextId++;
      t.fu = fu;
      t.subgraph = c.subgraph;
      t.bodyOpCount = countBodyOps(c.subgraph);
      t.rootOpName = computeRootOpName(c.subgraph).str();
      t.configDescription = std::move(c.configDescription);
      t.swConfigsByOp = std::move(c.swConfigsByOp);
      lib->entries.push_back(std::move(t));
    }
  }

  // Build root index after all templates are stable.
  for (const FuTemplate &t : lib->entries)
    if (!t.rootOpName.empty())
      lib->rootIndex[t.rootOpName].push_back(t.id);

  return lib;
}

} // namespace fabric
