//===-- adg.cpp - Vecadd ADG generation + standalone tool ----------*- C++ -*-//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// Constructs a compact domain ADG for the vecadd kernel using the
// ADGBuilder API, and provides a main() entry point for standalone generation.
//
// Each PE has FUs for: arith.addi, arith.cmpi, arith.index_cast,
// dataflow.stream, dataflow.gate, dataflow.carry, handshake.load,
// handshake.store, handshake.constant, handshake.cond_br, handshake.join.
// 3 extmemory instances (for a, b, c arrays).
// A single central spatial switch connects a bank of homogeneous PEs.
// This keeps the test small while still exercising the full mapper/viz flow.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static void buildVecaddADG(const std::string &outputPath) {
  ADGBuilder builder("vecadd_adg");

  // Use a uniform 64-bit routing plane so i32 / i1 / none / index can all
  // share the same switch and PE boundary network.
  const unsigned dataWidth = 64;

  //=== Define Function Units ===

  // FU: arith.addi (2 inputs, 1 output)
  auto fuAddi = builder.defineFU(
      "fu_addi", {"i32", "i32"}, {"i32"}, {"arith.addi"});

  // FU: arith.cmpi (2 inputs, 1 output: i1)
  auto fuCmpi = builder.defineFU(
      "fu_cmpi", {"i32", "i32"}, {"i1"}, {"arith.cmpi"});

  // FU: arith.index_cast (1 input, 1 output)
  auto fuIndexCast = builder.defineFU(
      "fu_index_cast", {"index", "i32"}, {"i32"}, {"arith.index_cast"});

  // FU: dataflow.stream (3 inputs: start, step, bound; 2 outputs: index, i1)
  auto fuStream = builder.defineFU(
      "fu_stream", {"index", "index", "index"}, {"index", "i1"},
      {"dataflow.stream"});

  // FU: dataflow.gate (2 inputs: value, cond; 2 outputs: value, cond)
  auto fuGate = builder.defineFU(
      "fu_gate", {"i32", "i1"}, {"i32", "i1"}, {"dataflow.gate"});

  // FU: dataflow.carry (3 inputs: d, a, b; 1 output)
  auto fuCarry = builder.defineFU(
      "fu_carry", {"i1", "i32", "i32"}, {"i32"}, {"dataflow.carry"});

  // FU: handshake.load (3 inputs: addr, data_in, ctrl; 2 outputs: data, addr)
  auto fuLoad = builder.defineFU(
      "fu_load", {"index", "i32", "none"}, {"i32", "index"},
      {"handshake.load"});

  // FU: handshake.store (3 inputs: addr, data, ctrl; 2 outputs: data, addr)
  auto fuStore = builder.defineFU(
      "fu_store", {"index", "i32", "none"}, {"i32", "index"},
      {"handshake.store"});

  // FU: handshake.constant (1 input: ctrl; 1 output: value)
  auto fuConstant = builder.defineFU(
      "fu_constant", {"none"}, {"i32"}, {"handshake.constant"});

  // FU: handshake.cond_br (2 inputs: cond, data; 2 outputs: true, false)
  auto fuCondBr = builder.defineFU(
      "fu_cond_br", {"i1", "i32"}, {"i32", "i32"}, {"handshake.cond_br"});

  // FU: handshake.join
  auto fuJoin = builder.defineFU(
      "fu_join", {"none", "none", "none"}, {"none"}, {"handshake.join"});

  //=== Define Spatial PE (containing all FUs) ===

  std::vector<FUHandle> allFUs = {
      fuAddi, fuCmpi, fuIndexCast,
      fuStream, fuGate, fuCarry,
      fuLoad, fuStore, fuConstant,
      fuCondBr, fuJoin
  };
  constexpr unsigned kPEInputs = 4;
  constexpr unsigned kPEOutputs = 4;
  auto pe = builder.defineSpatialPE("vecadd_pe", kPEInputs, kPEOutputs,
                                    dataWidth, allFUs);

  //=== Define Spatial Switch ===

  constexpr unsigned kNumPEs = 24;
  constexpr unsigned kScalarInputs = 2;
  constexpr unsigned kScalarOutputs = 1;
  constexpr unsigned kExtMemOutputPorts = 2 + 2 + 1; // A(ld), B(ld), C(st)
  constexpr unsigned kExtMemInputPorts = 1 + 1 + 2;  // A(ld), B(ld), C(st)

  const unsigned numSwInputs =
      kNumPEs * kPEOutputs + kExtMemOutputPorts + kScalarInputs;
  const unsigned numSwOutputs =
      kNumPEs * kPEInputs + kExtMemInputPorts + kScalarOutputs;

  std::vector<unsigned> swInputWidths(numSwInputs, dataWidth);
  std::vector<unsigned> swOutputWidths(numSwOutputs, dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      numSwOutputs, std::vector<bool>(numSwInputs, true));
  auto sw = builder.defineSpatialSW("vecadd_sw", swInputWidths,
                                    swOutputWidths, fullCrossbar);

  //=== Define External Memories ===

  auto ldMemDef = builder.defineExtMemory("vecadd_ldmem", 1, 0);
  auto stMemDef = builder.defineExtMemory("vecadd_stmem", 0, 1);

  //=== Instantiate Compute Fabric ===

  auto swInst = builder.instantiateSW(sw, "sw_0");
  std::vector<InstanceHandle> peInsts;
  peInsts.reserve(kNumPEs);
  for (unsigned i = 0; i < kNumPEs; ++i)
    peInsts.push_back(
        builder.instantiatePE(pe, "pe_" + std::to_string(i)));

  unsigned swInputCursor = 0;
  unsigned swOutputCursor = 0;
  for (InstanceHandle peInst : peInsts) {
    for (unsigned p = 0; p < kPEOutputs; ++p)
      builder.connect(peInst, p, swInst, swInputCursor++);
    for (unsigned p = 0; p < kPEInputs; ++p)
      builder.connect(swInst, swOutputCursor++, peInst, p);
  }

  //=== Instantiate External Memories (for arrays a, b, c) ===

  auto extMemA = builder.instantiateExtMem(ldMemDef, "extmem_a");
  auto extMemB = builder.instantiateExtMem(ldMemDef, "extmem_b");
  auto extMemC = builder.instantiateExtMem(stMemDef, "extmem_c");

  //=== Add module-level memref inputs ===

  auto memA = builder.addMemrefInput("mem_a", "memref<?xi32>");
  auto memB = builder.addMemrefInput("mem_b", "memref<?xi32>");
  auto memC = builder.addMemrefInput("mem_c", "memref<?xi32>");

  builder.connectMemrefToExtMem(memA, extMemA);
  builder.connectMemrefToExtMem(memB, extMemB);
  builder.connectMemrefToExtMem(memC, extMemC);

  //=== Add scalar boundary inputs and outputs ===

  auto scalarN = builder.addScalarInput("scalar_n", dataWidth);
  auto scalarCtrl = builder.addScalarInput("scalar_ctrl", dataWidth);

  auto scalarDone = builder.addScalarOutput("scalar_done", dataWidth);

  //=== ExtMemory associations ===

  builder.associateExtMemWithSW(extMemA, swInst, swInputCursor, swOutputCursor);
  swInputCursor += 2;
  swOutputCursor += 1;
  builder.associateExtMemWithSW(extMemB, swInst, swInputCursor, swOutputCursor);
  swInputCursor += 2;
  swOutputCursor += 1;
  builder.associateExtMemWithSW(extMemC, swInst, swInputCursor, swOutputCursor);
  swInputCursor += 1;
  swOutputCursor += 2;

  // Inject scalar values and expose the completion token.
  builder.connectScalarInputToInstance(scalarN, swInst, swInputCursor++);
  builder.connectScalarInputToInstance(scalarCtrl, swInst, swInputCursor++);
  builder.connectInstanceToScalarOutput(swInst, swOutputCursor++, scalarDone);

  //=== Export ===

  builder.exportMLIR(outputPath);
}

//===----------------------------------------------------------------------===//
// Standalone tool entry point
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("vecadd.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "vecadd ADG generator\n");

  // Ensure output directory exists
  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: "
                   << parentPath << "\n";
      return 1;
    }
  }

  llvm::outs() << "Generating vecadd ADG -> " << outputFile << "\n";
  buildVecaddADG(outputFile);
  llvm::outs() << "Done.\n";

  return 0;
}
