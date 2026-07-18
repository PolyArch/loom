#include "ADG/Builder.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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

static llvm::cl::opt<bool> sharedMemoryReduction(
    "shared-memory-reduction",
    llvm::cl::desc("emit a shared memory-pressure reduction SpatialCore ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> sharedQuantizedWindow(
    "shared-quantized-window",
    llvm::cl::desc("emit a shared quantized-window SpatialCore ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> sharedSignalWindow(
    "shared-signal-window",
    llvm::cl::desc("emit a shared signal-window SpatialCore ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    sharedVectorAlu("shared-vector-alu",
                    llvm::cl::desc("emit a shared vector ALU SpatialCore ADG"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> sharedVectorMath(
    "shared-vector-math",
    llvm::cl::desc("emit a shared vector math SpatialCore ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> sharedVectorMesh(
    "shared-vector-mesh",
    llvm::cl::desc("emit a shared vector mesh SpatialCore ADG"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    fullSpatialCore("full-spatialcore",
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

static llvm::cl::opt<std::string> systemMatrixCase(
    "system-matrix-case",
    llvm::cl::desc("emit a named topology-matrix system-level ADG"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> invalidYieldTypes(
    "invalid-yield-types",
    llvm::cl::desc("emit an invalid ADG with mismatched FU yield types"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> invalidYieldCount(
    "invalid-yield-count",
    llvm::cl::desc("emit an invalid ADG with mismatched FU yield values"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> invalidStreamConfig(
    "invalid-stream-config",
    llvm::cl::desc("emit an invalid ADG stream configuration case"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("output Fabric MLIR path"),
               llvm::cl::Required);

static llvm::Error writeTemporalMemCapacityAnchors(llvm::raw_ostream &out) {
  loom::adg::ModuleBuilder module("temporal_mem_capacity_anchors_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("addr0", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl0", "!fabric.bits_tag<0, 4>")
      .addInput("addr1", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl1", "!fabric.bits_tag<0, 4>")
      .addInput("wide_addr", "!fabric.bits_tag<32, 64>")
      .addInput("wide_ctrl", "!fabric.bits_tag<0, 64>");

  loom::adg::MemSpec capacity(
      loom::adg::Schedule::Temporal, {"mgr"}, {},
      loom::adg::MemDispatchEligibility{{{0}, {0}}, {}});
  capacity.loads = {{"addr0", "ctrl0"}, {"addr1", "ctrl1"}};
  capacity.dataWidth = 32;
  capacity.temporalTagWidth = 4;
  capacity.temporalOperationTableSize = 17;
  module.addMem(std::move(capacity));

  loom::adg::MemSpec wideTag(loom::adg::Schedule::Temporal, {"mgr"}, {},
                             loom::adg::MemDispatchEligibility{{{0}}, {}});
  wideTag.loads = {{"wide_addr", "wide_ctrl"}};
  wideTag.dataWidth = 32;
  wideTag.temporalTagWidth = 64;
  wideTag.temporalOperationTableSize = 1;
  module.addMem(std::move(wideTag));
  return module.print(out);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-adg-builder-test: emit deterministic Fabric MLIR from the ADG "
      "Builder C++ API\n");

  unsigned selectedRecipes =
      (sharedReduction ? 1 : 0) + (sharedVectorAlu ? 1 : 0) +
      (sharedMemoryReduction ? 1 : 0) + (sharedVectorMath ? 1 : 0) +
      (sharedQuantizedWindow ? 1 : 0) + (sharedSignalWindow ? 1 : 0) +
      (sharedVectorMesh ? 1 : 0) + (fullSpatialCore ? 1 : 0) +
      (minimalSpatial ? 1 : 0) + (minimalTemporal ? 1 : 0) +
      (heterogeneousSoc ? 1 : 0) + (!topologyMatrixCase.empty() ? 1 : 0) +
      (!systemMatrixCase.empty() ? 1 : 0);
  selectedRecipes += (invalidYieldTypes ? 1 : 0) + (invalidYieldCount ? 1 : 0);
  selectedRecipes += !invalidStreamConfig.empty() ? 1 : 0;
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
    if (sharedMemoryReduction)
      return loom::adg::writeSharedMemoryReductionAdg(out);
    if (sharedQuantizedWindow)
      return loom::adg::writeSharedQuantizedWindowAdg(out);
    if (sharedSignalWindow)
      return loom::adg::writeSharedSignalWindowAdg(out);
    if (sharedVectorAlu)
      return loom::adg::writeSharedVectorAluAdg(out);
    if (sharedVectorMath)
      return loom::adg::writeSharedVectorMathAdg(out);
    if (sharedVectorMesh)
      return loom::adg::writeSharedVectorMeshAdg(out);
    if (fullSpatialCore) {
      if (llvm::Error err = loom::adg::writeFullSpatialCoreAdg(out))
        return err;
      out << '\n';
      return writeTemporalMemCapacityAnchors(out);
    }
    if (heterogeneousSoc)
      return loom::adg::writeHeterogeneousSocAdg(out);
    if (!topologyMatrixCase.empty())
      return loom::adg::writeSpatialTopologyMatrixAdg(out, topologyMatrixCase);
    if (!systemMatrixCase.empty())
      return loom::adg::writeSystemTopologyMatrixAdg(out, systemMatrixCase);
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
    if (!invalidStreamConfig.empty()) {
      loom::adg::ModuleBuilder module("invalid_stream_config_adg");
      module.addInput("init", "!fabric.bits<32>");
      module.addInput("limit", "!fabric.bits<32>");
      module.addInput("step", "!fabric.bits<32>");

      loom::adg::PeSpec pe;
      pe.inputs = {{"pa", "init", "!fabric.bits<32>", ""},
                   {"pb", "limit", "!fabric.bits<32>", ""},
                   {"pc", "step", "!fabric.bits<32>", ""}};
      loom::adg::FuSpec fu;
      fu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                   {"fb", "pb", "!fabric.bits<32>", ""},
                   {"fc", "pc", "!fabric.bits<32>", ""}};
      loom::adg::FabricOpSpec stream;
      stream.results = {"iv", "phase"};
      stream.opList = {"dataflow.stream"};
      stream.operands = {"fa", "fb", "fc"};
      stream.operandTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                             "!fabric.bits<32>"};
      stream.resultTypes = {"!fabric.bits<32>", "!fabric.bits<1>"};

      if (invalidStreamConfig == "generic") {
        stream.hwParams["step_kind"] = {"0"};
        stream.hwParams["predicate"] = {"2"};
      } else if (invalidStreamConfig == "step") {
        stream.streamConfig =
            loom::adg::StreamConfig{static_cast<dataflow::StreamStepKind>(99),
                                    {mlir::arith::CmpIPredicate::slt}};
      } else if (invalidStreamConfig == "predicate") {
        stream.streamConfig = loom::adg::StreamConfig{
            dataflow::StreamStepKind::Add,
            {static_cast<mlir::arith::CmpIPredicate>(99)}};
      } else if (invalidStreamConfig != "missing") {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "unknown invalid stream configuration case %s",
            invalidStreamConfig.c_str());
      }

      fu.operations.push_back(std::move(stream));
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
