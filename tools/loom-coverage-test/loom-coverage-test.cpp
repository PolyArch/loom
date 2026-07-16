// CLI helper for lit tests: drives `loom::fabric::tech::CoverageVerifier`
// over a single fabric.fu and a list of input dataflow.subgraphs in the
// input MLIR module.
//
// Usage:
//   loom-coverage-test <input.mlir> [--config <path>] [--check-isolation]
//   loom-coverage-test <input.mlir> --project-first-encoding
//
// Input contract:
//   * The module must contain exactly one `fabric.fu`. The verifier
//     materializes that FU and matches every input against its
//     enumerated candidates.
//   * Each input subgraph lives in its own `func.func` body (one
//     `dataflow.subgraph` per func.func) annotated with attribute
//     `loom.coverage_input = true`. The annotation lets the helper
//     distinguish coverage inputs from the FU's wrapper func.func.
//   * `--config` (optional): YAML/TOML SynthConfig path. Defaults
//     to a default-constructed SynthConfig (parallel_match=true).
//   * `--check-isolation` (optional): after `verify` returns, walks
//     the user's module and prints `user_funcs_after=<N>` plus a
//     `candidate_in_user=<bool>` line for whether any func.func name
//     starts with `candidate_` (the prefix the verifier passes to
//     `enumerateFuSubgraphs`). Used by the isolation lit test.
//   * `--project-first-encoding` parses without running operation verifiers and
//     invokes `projectConfiguredFunction` directly on the first encoding. This
//     is reserved for malformed-input projector regressions.
//
// Output (one line per input subgraph, in module order):
//   coverage[i] funcname=<name> matched=<true|false> index=<n_or_none>
//
// Final line:
//   all_covered=<true|false>
//
// Exit: 0 on success, 1 on parse or input-shape failure.

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/ConfiguredFunctionAdapters.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("<input>"),
              ::llvm::cl::Required);

static ::llvm::cl::opt<std::string>
    configPath("config",
               ::llvm::cl::desc("Path to SynthConfig YAML/TOML. "
                                "Defaults to a default-constructed config."),
               ::llvm::cl::init(""));

static ::llvm::cl::opt<bool>
    checkIsolation("check-isolation",
                   ::llvm::cl::desc("After verify, print isolation diagnostics: "
                                    "user_funcs_after=<N> and "
                                    "candidate_in_user=<bool>."),
                   ::llvm::cl::init(false));

static ::llvm::cl::opt<bool> projectFirstEncoding(
    "project-first-encoding",
    ::llvm::cl::desc("Parse without verification and directly project the "
                     "first fabric.fu semantic encoding"),
    ::llvm::cl::init(false));

namespace {

bool hasCoverageInputAttr(::mlir::func::FuncOp f) {
  ::mlir::Attribute a = f->getAttr("loom.coverage_input");
  if (!a)
    return false;
  if (auto b = ::mlir::dyn_cast<::mlir::BoolAttr>(a))
    return b.getValue();
  // Treat any non-null attribute as "set to true" for ergonomic test
  // input (e.g. UnitAttr written as `loom.coverage_input`).
  return true;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-coverage-test: drive CoverageVerifier and dump match indices\n");

  // Resolve the SynthConfig.
  ::loom::SynthConfig cfg;
  if (!configPath.getValue().empty()) {
    auto parsed = ::loom::loadSynthConfig(configPath.getValue());
    if (!parsed) {
      ::llvm::errs() << "error: " << ::llvm::toString(parsed.takeError())
                     << "\n";
      return 1;
    }
    cfg = std::move(*parsed);
  }

  // Set up the MLIR context. Register every standard dialect plus the
  // loom dialects so any test fixture parses without per-test
  // `--register-dialect` plumbing.
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
  ::mlir::ParserConfig parserConfig(
      &ctx, /*verifyAfterParse=*/!projectFirstEncoding.getValue());
  ::mlir::OwningOpRef<::mlir::ModuleOp> mod =
      ::mlir::parseSourceFile<::mlir::ModuleOp>(sm, parserConfig);
  if (!mod) {
    ::llvm::errs() << "error: parse failed\n";
    return 1;
  }

  // Locate the single fabric.fu. Bail if zero or more than one.
  ::llvm::SmallVector<::fabric::FuOp, 1> fus;
  mod->walk([&](::fabric::FuOp fu) { fus.push_back(fu); });
  if (fus.size() != 1) {
    ::llvm::errs() << "error: expected exactly one fabric.fu, got "
                   << fus.size() << "\n";
    return 1;
  }
  ::fabric::FuOp fu = fus.front();

  if (projectFirstEncoding.getValue()) {
    auto encodings = fu.getValidEncodingsAttr();
    if (!encodings || encodings.empty()) {
      ::llvm::errs() << "error: fabric.fu has no semantic encoding\n";
      return 1;
    }
    auto encoding = ::mlir::dyn_cast<::mlir::DictionaryAttr>(encodings[0]);
    if (!encoding) {
      ::llvm::errs() << "error: first semantic encoding is not a dictionary\n";
      return 1;
    }
    ::fabric::ConfiguredFunction function;
    std::string error;
    if (::mlir::failed(::fabric::projectConfiguredFunction(fu, encoding,
                                                           function, error))) {
      ::llvm::outs() << "projection=failed\nerror=" << error << "\n";
      return 0;
    }
    ::llvm::outs() << "projection=success\n";
    return 0;
  }

  // Gather coverage-input subgraphs in module order. We iterate
  // top-level func.funcs and pick the unique dataflow.subgraph from
  // every one tagged with `loom.coverage_input`.
  ::llvm::SmallVector<::dataflow::SubgraphOp, 4> inputs;
  ::llvm::SmallVector<::fabric::ConfiguredFunction, 4> functions;
  ::llvm::SmallVector<std::string, 4> inputNames;
  for (::mlir::func::FuncOp f : mod->getOps<::mlir::func::FuncOp>()) {
    if (!hasCoverageInputAttr(f))
      continue;
    ::dataflow::SubgraphOp found;
    f.walk([&](::dataflow::SubgraphOp sg) {
      if (!found)
        found = sg;
    });
    if (!found) {
      ::llvm::errs() << "error: func @" << f.getSymName()
                     << " is loom.coverage_input but contains no "
                        "dataflow.subgraph\n";
      return 1;
    }
    inputs.push_back(found);
    ::fabric::ConfiguredFunction function;
    std::string adapterError;
    if (::mlir::failed(::fabric::configuredFunctionFromSubgraph(
            found, function, adapterError))) {
      ::llvm::errs() << "error: func @" << f.getSymName() << ": "
                     << adapterError << "\n";
      return 1;
    }
    functions.push_back(std::move(function));
    inputNames.push_back(f.getSymName().str());
  }

  // Run the verifier.
  ::loom::fabric::tech::CoverageVerifier verifier(cfg);
  ::loom::fabric::tech::CoverageReport report = verifier.verify(fu, functions);

  // Emit results. The report contains one optional witness per input.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &witness = report.witnesses[i];
    ::llvm::outs() << "coverage[" << i << "] funcname=" << inputNames[i]
                   << " matched=" << (witness.has_value() ? "true" : "false")
                   << " index=";
    if (witness.has_value())
      ::llvm::outs() << witness->encodingIndex;
    else
      ::llvm::outs() << "none";
    ::llvm::outs() << "\n";
  }
  ::llvm::outs() << "all_covered=" << (report.allCovered() ? "true" : "false")
                 << "\n";

  if (checkIsolation.getValue()) {
    unsigned funcCount = 0;
    bool foundCandidate = false;
    for (::mlir::func::FuncOp f : mod->getOps<::mlir::func::FuncOp>()) {
      ++funcCount;
      if (f.getSymName().starts_with("candidate"))
        foundCandidate = true;
    }
    ::llvm::outs() << "user_funcs_after=" << funcCount << "\n";
    ::llvm::outs() << "candidate_in_user="
                   << (foundCandidate ? "true" : "false") << "\n";
  }

  return 0;
}
