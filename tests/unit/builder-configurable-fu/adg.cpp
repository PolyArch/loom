//===-- adg.cpp - Builder configurable FU test ------------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static void buildConfigurableFUADG(const std::string &outputPath) {
  ADGBuilder builder("builder_configurable_fu_test");

  auto fuConst = builder.defineConstantFU("fu_constant", "i32", "0 : i32");
  FunctionUnitSpec joinSpec;
  joinSpec.name = "fu_join";
  joinSpec.inputTypes = {"none", "i32", "none", "i1"};
  joinSpec.outputTypes = {"none"};
  joinSpec.ops = {"handshake.join"};
  auto fuJoin = builder.defineFU(joinSpec);
  auto fuCmpi = builder.defineCmpiFU("fu_cmpi", "i32", "eq");
  auto fuCmpf = builder.defineCmpfFU("fu_cmpf", "f32", "oeq");
  auto fuStream = builder.defineStreamFU("fu_stream", "index", "+=", "<");

  auto constPE = builder.defineSpatialPE(
      SpatialPESpec{.name = "const_pe",
                    .inputTypes = {"none"},
                    .outputTypes = {"i32"},
                    .functionUnits = {fuConst}});
  auto cmpiPE = builder.defineSpatialPE(
      SpatialPESpec{.name = "cmpi_pe",
                    .inputTypes = {"i32", "i32"},
                    .outputTypes = {"i1"},
                    .functionUnits = {fuCmpi}});
  auto joinPE = builder.defineSpatialPE(
      SpatialPESpec{.name = "join_pe",
                    .inputTypes = {"none", "i32", "none", "i1"},
                    .outputTypes = {"none"},
                    .functionUnits = {fuJoin}});
  auto cmpfPE = builder.defineSpatialPE(
      SpatialPESpec{.name = "cmpf_pe",
                    .inputTypes = {"f32", "f32"},
                    .outputTypes = {"i1"},
                    .functionUnits = {fuCmpf}});
  auto streamPE = builder.defineSpatialPE(
      SpatialPESpec{.name = "stream_pe",
                    .inputTypes = {"index", "index", "index"},
                    .outputTypes = {"index", "i1"},
                    .functionUnits = {fuStream}});

  auto ctrl = builder.addInput("ctrl", "none");
  auto ctrl2 = builder.addInput("ctrl2", "none");
  auto a = builder.addInput("a", "i32");
  auto b = builder.addInput("b", "i32");
  auto joinGuard = builder.addInput("join_guard", "i1");
  auto af = builder.addInput("af", "f32");
  auto bf = builder.addInput("bf", "f32");
  auto start = builder.addInput("start", "index");
  auto step = builder.addInput("step", "index");
  auto bound = builder.addInput("bound", "index");

  auto constOut = builder.addOutput("const", "i32");
  auto joinOut = builder.addOutput("join_done", "none");
  auto cmpiOut = builder.addOutput("cmpi", "i1");
  auto cmpfOut = builder.addOutput("cmpf", "i1");
  auto streamIdxOut = builder.addOutput("stream_idx", "index");
  auto streamCondOut = builder.addOutput("stream_cond", "i1");

  auto constInst = builder.instantiatePE(constPE, "pe_const");
  auto joinInst = builder.instantiatePE(joinPE, "pe_join");
  auto cmpiInst = builder.instantiatePE(cmpiPE, "pe_cmpi");
  auto cmpfInst = builder.instantiatePE(cmpfPE, "pe_cmpf");
  auto streamInst = builder.instantiatePE(streamPE, "pe_stream");

  builder.connectInputToInstance(ctrl, constInst, 0);
  builder.connectInputToInstance(ctrl, joinInst, 0);
  builder.connectInputToInstance(a, joinInst, 1);
  builder.connectInputToInstance(ctrl2, joinInst, 2);
  builder.connectInputToInstance(joinGuard, joinInst, 3);
  builder.connectInputToInstance(a, cmpiInst, 0);
  builder.connectInputToInstance(b, cmpiInst, 1);
  builder.connectInputToInstance(af, cmpfInst, 0);
  builder.connectInputToInstance(bf, cmpfInst, 1);
  builder.connectInputToInstance(start, streamInst, 0);
  builder.connectInputToInstance(step, streamInst, 1);
  builder.connectInputToInstance(bound, streamInst, 2);

  builder.connectInstanceToOutput(constInst, 0, constOut);
  builder.connectInstanceToOutput(joinInst, 0, joinOut);
  builder.connectInstanceToOutput(cmpiInst, 0, cmpiOut);
  builder.connectInstanceToOutput(cmpfInst, 0, cmpfOut);
  builder.connectInstanceToOutput(streamInst, 0, streamIdxOut);
  builder.connectInstanceToOutput(streamInst, 1, streamCondOut);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-configurable-fu.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Builder configurable FU ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildConfigurableFUADG(outputFile);
  return 0;
}
