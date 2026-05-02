// CLI helper for lit tests: parse an MLIR module containing one or more
// wrappers (each `fabric.module` or `func.func` containing a single
// `fabric.fu`), then run `loom::fabric::tech::CostModel::evaluate(fu)`
// on each and print the score one per line in `cost <name>=<double>`
// form. The reported `<name>` is the wrapper symbol name (with no
// leading `@`).
//
// Usage:
//   loom-cost-test <input.mlir> [--config <cfg.yaml|toml>]
//
// Default weights are taken from a default-constructed `loom::SynthConfig`.
// Pass `--config` to override mux/demux/carry penalties; this is the same
// loader used by the pass and the synth-config-test CLI.
//
// Output format is deterministic in func-symbol-walk order so FileCheck
// assertions can pin both names and numeric values.

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("<input>"),
              ::llvm::cl::Required);

static ::llvm::cl::opt<std::string>
    configPath("config",
               ::llvm::cl::desc("Path to a SynthConfig YAML/TOML file. When "
                                "absent, default weights are used."),
               ::llvm::cl::init(""));

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-cost-test: evaluate CostModel on each fabric.fu in a module\n");

  ::loom::SynthConfig config;
  if (!configPath.getValue().empty()) {
    auto loaded = ::loom::loadSynthConfig(configPath.getValue());
    if (!loaded) {
      ::llvm::errs() << "error: " << ::llvm::toString(loaded.takeError())
                     << "\n";
      return 1;
    }
    config = std::move(*loaded);
  }

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

  ::loom::fabric::tech::CostModel cost(config);

  // Walk each fabric.fu in the module (module-walk order, which is
  // deterministic). The spec helper-spec contract is exactly-one-FU
  // per wrapper, so we report the cost keyed by the immediate
  // top-level wrapper symbol name. Both fabric.module and func.func
  // wrappers are accepted so the helper works during the migration
  // window in either direction.
  for (auto fu : ([&] {
         ::llvm::SmallVector<::fabric::FuOp, 4> fus;
         mod->walk([&](::fabric::FuOp f) { fus.push_back(f); });
         return fus;
       }())) {
    ::mlir::Operation *wrapper = fu.getOperation();
    while (wrapper && wrapper->getParentOp() &&
           !::mlir::isa<::mlir::ModuleOp>(wrapper->getParentOp()))
      wrapper = wrapper->getParentOp();
    ::llvm::StringRef wrapperName;
    if (auto sym = ::mlir::dyn_cast_or_null<::mlir::SymbolOpInterface>(
            wrapper))
      wrapperName = sym.getName();
    const double score = cost.evaluate(fu);
    ::llvm::outs() << "cost " << wrapperName << "=" << score << "\n";
  }
  return 0;
}
