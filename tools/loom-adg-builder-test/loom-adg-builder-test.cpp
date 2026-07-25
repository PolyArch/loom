#include "ADG/Builder.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FabricEnums.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

static_assert(
    std::is_same_v<decltype(loom::adg::TemporalPeConfig::operandBufferMode),
                   ::fabric::OperandBufferMode>);
static_assert(std::is_same_v<decltype(loom::adg::BoundarySpec::direction),
                             ::fabric::BoundaryDirection>);

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

static llvm::cl::opt<bool> publicFifoBoundary(
    "public-fifo-boundary",
    llvm::cl::desc("emit FIFO and boundary ops through the public ADG API"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> invalidFifoSpec(
    "invalid-fifo-spec",
    llvm::cl::desc("emit an invalid public ADG FIFO specification"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> invalidBoundarySpec(
    "invalid-boundary-spec",
    llvm::cl::desc("emit an invalid public ADG boundary specification"),
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
  wideTag.temporalOperationTableSize = 2147483647u;
  module.addMem(std::move(wideTag));
  return module.print(out);
}

static llvm::Error writePublicFifoBoundary(llvm::raw_ostream &out) {
  loom::adg::ModuleBuilder module("public_fifo_boundary_adg");
  module.addInput("data", "!fabric.bits<32>")
      .addInput("tag", "!fabric.bits<4>")
      .addBoundary(
          loom::adg::BoundarySpec{::fabric::BoundaryDirection::S2t,
                                  {{"data", "!fabric.bits<16>"}, {"tag"}},
                                  {"tagged"},
                                  {"!fabric.bits_tag<16, 4>"}})
      .addFifo(loom::adg::FifoSpec{"queued", "tagged", "!fabric.bits_tag<8, 4>",
                                   4, true, false})
      .addBoundary(
          loom::adg::BoundarySpec{::fabric::BoundaryDirection::T2s,
                                  {{"queued"}},
                                  {"untagged", "split_tag"},
                                  {"!fabric.bits<8>", "!fabric.bits<4>"}})
      .addOutput("untagged")
      .addOutput("split_tag");
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
  selectedRecipes += publicFifoBoundary ? 1 : 0;
  selectedRecipes += !invalidFifoSpec.empty() ? 1 : 0;
  selectedRecipes += !invalidBoundarySpec.empty() ? 1 : 0;
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
    if (publicFifoBoundary)
      return writePublicFifoBoundary(out);
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
      auto makeStreamCapability = [](dataflow::StreamStepKind stepKind,
                                     mlir::arith::CmpIPredicate predicate) {
        return loom::adg::FabricOpCapability{
            ::fabric::ImplementationFamilyId::LoopStream,
            ::fabric::LoopStreamParams{
                ::fabric::IntegerWidthSet::get(
                    {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
                     ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64}),
                stepKind, ::fabric::IntegerPredicateSet::get({predicate})}};
      };
      loom::adg::FabricOpCapability capability = makeStreamCapability(
          dataflow::StreamStepKind::Add, mlir::arith::CmpIPredicate::slt);
      if (invalidStreamConfig == "step") {
        capability =
            makeStreamCapability(static_cast<dataflow::StreamStepKind>(99),
                                 mlir::arith::CmpIPredicate::slt);
      } else if (invalidStreamConfig == "predicate") {
        capability =
            makeStreamCapability(dataflow::StreamStepKind::Add,
                                 static_cast<mlir::arith::CmpIPredicate>(99));
      } else {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "unknown invalid stream configuration case %s",
            invalidStreamConfig.c_str());
      }

      fu.operations.push_back(loom::adg::FabricOpSpec{
          {"iv", "phase"},
          std::move(capability),
          {::dataflow::OperationSchemaId::DataflowStream},
          {"fa", "fb", "fc"},
          {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
          {"!fabric.bits<32>", "!fabric.bits<1>"}});
      pe.fus.push_back(std::move(fu));
      module.addPe(std::move(pe));
      return module.print(out);
    }
    if (!invalidFifoSpec.empty()) {
      loom::adg::ModuleBuilder module("invalid_fifo_spec_adg");
      module.addInput("input", "!fabric.bits<8>");
      if (invalidFifoSpec == "depth") {
        module.addFifo(loom::adg::FifoSpec{"output", "input", "!fabric.bits<8>",
                                           0, false});
      } else if (invalidFifoSpec == "overflow") {
        module.addFifo(loom::adg::FifoSpec{
            "output", "input", "!fabric.bits<8>",
            static_cast<unsigned>(std::numeric_limits<std::int32_t>::max()) +
                1U,
            false});
      } else if (invalidFifoSpec == "kind") {
        module.addFifo(loom::adg::FifoSpec{"output", "input",
                                           "!fabric.bits_tag<8, 4>", 1, false});
      } else if (invalidFifoSpec == "bypass") {
        module.addFifo(loom::adg::FifoSpec{"output", "input", "!fabric.bits<8>",
                                           1, false, true});
      } else {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "unknown invalid FIFO case %s",
                                       invalidFifoSpec.c_str());
      }
      return module.print(out);
    }
    if (!invalidBoundarySpec.empty()) {
      loom::adg::ModuleBuilder module("invalid_boundary_spec_adg");
      module.addInput("data", "!fabric.bits<32>")
          .addInput("tag", "!fabric.bits<4>")
          .addInput("tagged", "!fabric.bits_tag<32, 4>");
      if (invalidBoundarySpec == "shape") {
        module.addBoundary(
            loom::adg::BoundarySpec{::fabric::BoundaryDirection::S2t,
                                    {{"data"}},
                                    {"tagged"},
                                    {"!fabric.bits_tag<32, 4>"}});
      } else if (invalidBoundarySpec == "direction") {
        module.addBoundary(loom::adg::BoundarySpec{
            static_cast<::fabric::BoundaryDirection>(99),
            {{"data"}, {"tag"}},
            {"tagged"},
            {"!fabric.bits_tag<32, 4>"}});
      } else if (invalidBoundarySpec == "t2t") {
        module.addBoundary(
            loom::adg::BoundarySpec{::fabric::BoundaryDirection::T2t,
                                    {{"data"}},
                                    {"tagged"},
                                    {"!fabric.bits_tag<32, 4>"}});
      } else if (invalidBoundarySpec == "t2s-spatial") {
        module.addBoundary(
            loom::adg::BoundarySpec{::fabric::BoundaryDirection::T2s,
                                    {{"data"}},
                                    {"untagged"},
                                    {"!fabric.bits<32>"}});
      } else if (invalidBoundarySpec == "t2s-tag-width") {
        module.addBoundary(
            loom::adg::BoundarySpec{::fabric::BoundaryDirection::T2s,
                                    {{"tagged"}},
                                    {"untagged", "split_tag"},
                                    {"!fabric.bits<32>", "!fabric.bits<8>"}});
      } else {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "unknown invalid boundary case %s",
                                       invalidBoundarySpec.c_str());
      }
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
