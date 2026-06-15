#include "ADG/Builder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

static llvm::cl::opt<bool>
    sharedReduction("shared-reduction",
                    llvm::cl::desc("emit a shared reduction SpatialCore ADG"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool>
    sharedVectorAlu("shared-vector-alu",
                    llvm::cl::desc("emit a shared vector ALU SpatialCore ADG"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> fullSpatialCore(
    "full-spatialcore",
    llvm::cl::desc("emit a full-construct SpatialCore ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    minimalSpatial("minimal-spatial",
                   llvm::cl::desc("emit a minimal SpatialCore ADG"),
                   llvm::cl::init(false));

static llvm::cl::opt<bool>
    minimalTemporal("minimal-temporal",
                    llvm::cl::desc("emit a minimal temporal SpatialCore ADG"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> heterogeneousSoc(
    "heterogeneous-soc",
    llvm::cl::desc("emit a heterogeneous SoC system-level ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> topologyMatrixCase(
    "topology-matrix-case",
    llvm::cl::desc("emit a named topology-matrix SpatialCore ADG"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> invalidYieldTypes(
    "invalid-yield-types",
    llvm::cl::desc("emit an invalid ADG with mismatched FU yield types"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> invalidYieldCount(
    "invalid-yield-count",
    llvm::cl::desc("emit an invalid ADG with mismatched FU yield values"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("output Fabric MLIR path"),
               llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-adg-builder-test: emit deterministic Fabric MLIR from the ADG "
      "Builder C++ API\n");

  unsigned selectedRecipes =
      (sharedReduction ? 1 : 0) + (sharedVectorAlu ? 1 : 0) +
      (fullSpatialCore ? 1 : 0) + (minimalSpatial ? 1 : 0) +
      (minimalTemporal ? 1 : 0) + (heterogeneousSoc ? 1 : 0) +
      (!topologyMatrixCase.empty() ? 1 : 0);
  selectedRecipes += (invalidYieldTypes ? 1 : 0) + (invalidYieldCount ? 1 : 0);
  if (selectedRecipes == 0) {
    llvm::errs() << "error: no ADG recipe selected\n";
    return 1;
  }
  if (selectedRecipes > 1) {
    llvm::errs() << "error: select exactly one ADG recipe\n";
    return 1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: could not open " << outputPath << ": "
                 << ec.message() << "\n";
    return 1;
  }

  auto writeSelectedRecipe = [&]() -> llvm::Error {
    if (minimalSpatial)
      return loom::adg::writeMinimalSpatialAdg(out);
    if (minimalTemporal)
      return loom::adg::writeMinimalTemporalAdg(out);
    if (sharedVectorAlu)
      return loom::adg::writeSharedVectorAluAdg(out);
    if (fullSpatialCore)
      return loom::adg::writeFullSpatialCoreAdg(out);
    if (heterogeneousSoc)
      return loom::adg::writeHeterogeneousSocAdg(out);
    if (!topologyMatrixCase.empty())
      return loom::adg::writeSpatialTopologyMatrixAdg(out,
                                                      topologyMatrixCase);
    if (invalidYieldTypes) {
      loom::adg::ModuleBuilder module("invalid_yield_types_adg");
      module.addInput("lhs", "!fabric.bits<32>");
      loom::adg::PeSpec pe;
      pe.inputs = {{"pa", "lhs", "!fabric.bits<32>", ""}};
      pe.resultTypes = {"!fabric.bits<32>"};
      loom::adg::FuSpec fu;
      fu.inputs = {{"value", "pa", "!fabric.bits<32>", ""}};
      fu.resultTypes = {"!fabric.bits<32>"};
      fu.yieldValues = {"value"};
      fu.yieldTypes = {"!fabric.bits<32>", "!fabric.bits<1>"};
      pe.fus.push_back(std::move(fu));
      module.addPe(std::move(pe));
      return module.print(out);
    }
    if (invalidYieldCount) {
      loom::adg::ModuleBuilder module("invalid_yield_count_adg");
      module.addInput("lhs", "!fabric.bits<32>");
      loom::adg::PeSpec pe;
      pe.inputs = {{"pa", "lhs", "!fabric.bits<32>", ""}};
      pe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
      loom::adg::FuSpec fu;
      fu.inputs = {{"value", "pa", "!fabric.bits<32>", ""}};
      fu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
      fu.yieldValues = {"value"};
      pe.fus.push_back(std::move(fu));
      module.addPe(std::move(pe));
      return module.print(out);
    }
    return loom::adg::writeSharedReductionAdg(out);
  };

  if (llvm::Error err = writeSelectedRecipe()) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
