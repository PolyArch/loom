// CLI helper for lit tests: parse an MLIR file, build a TemplateLibrary
// from any fabric.fu ops, then run the parallel CandidateCache against
// each dataflow.graph in the module and print the results in a stable
// FileCheck-friendly form.
//
// Usage: loom-candidate-dump <input.mlir> [--cache-threads=N]
//
// Output (one block per graph, header + one line per op):
//   graph #<id> @<symbol-or-loc>
//   op#<program-pos> name=<op-name> templates=<id_list>
//
// `id_list` is comma-separated and ascending. `<empty>` is printed when
// no template matches.

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/TemplateLibrary.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("<input>"),
              ::llvm::cl::Required);

static ::llvm::cl::opt<unsigned>
    threads("cache-threads",
            ::llvm::cl::desc("Worker threads for the candidate cache. "
                             "0 => hardware_concurrency()."),
            ::llvm::cl::init(0));

namespace {

// Best-effort label for a graph: the enclosing func.func symbol when
// available, otherwise the location.
std::string graphLabel(::dataflow::GraphOp graph) {
  ::mlir::Operation *parent = graph->getParentOp();
  while (parent) {
    if (auto attr = parent->getAttrOfType<::mlir::StringAttr>("sym_name"))
      return attr.getValue().str();
    parent = parent->getParentOp();
  }
  std::string buf;
  ::llvm::raw_string_ostream os(buf);
  graph.getLoc().print(os);
  os.flush();
  return buf;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(argc, argv,
                                       "loom-candidate-dump: build a "
                                       "CandidateCache and print it\n");

  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  registry.insert<::fabric::FabricDialect, ::dataflow::DataflowDialect>();
  ::mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto bufOrErr = ::llvm::MemoryBuffer::getFileOrSTDIN(inputPath.getValue());
  if (auto ec = bufOrErr.getError()) {
    ::llvm::errs() << "error: " << ec.message() << "\n";
    return 1;
  }
  ::llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*bufOrErr), ::llvm::SMLoc());
  ::mlir::OwningOpRef<::mlir::ModuleOp> mod =
      ::mlir::parseSourceFile<::mlir::ModuleOp>(sm, &ctx);
  if (!mod) {
    ::llvm::errs() << "error: parse failed\n";
    return 1;
  }

  ::llvm::SmallVector<::fabric::FuOp> fus;
  mod->walk([&](::fabric::FuOp fu) { fus.push_back(fu); });
  auto lib = ::fabric::TemplateLibrary::build(&ctx, fus);

  ::llvm::SmallVector<::dataflow::GraphOp> graphs;
  mod->walk([&](::dataflow::GraphOp g) { graphs.push_back(g); });

  unsigned graphId = 0;
  for (::dataflow::GraphOp g : graphs) {
    auto cache = ::fabric::CandidateCache::build(g, *lib, threads.getValue());
    ::llvm::outs() << "graph #" << graphId++ << " @" << graphLabel(g) << "\n";
    for (const auto &cs : cache.all()) {
      unsigned pos = static_cast<unsigned>(&cs - cache.all().data());
      ::llvm::outs() << "op#" << pos
                      << " name=" << cs.root->getName().getStringRef()
                      << " templates=";
      if (cs.templateIds.empty()) {
        ::llvm::outs() << "<empty>";
      } else {
        for (size_t i = 0; i < cs.templateIds.size(); ++i) {
          if (i)
            ::llvm::outs() << ",";
          ::llvm::outs() << cs.templateIds[i];
        }
      }
      ::llvm::outs() << "\n";
    }
  }
  return 0;
}
