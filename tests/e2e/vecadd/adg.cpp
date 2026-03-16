//===-- adg.cpp - Vecadd ADG generation + standalone tool ----------*- C++ -*-//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// Constructs a 6x6 mesh ADG for the vecadd kernel using the ADGBuilder API,
// and provides a main() entry point for standalone generation.
//
// Each PE has FUs for: arith.addi, arith.cmpi, arith.extui, arith.index_cast,
// dataflow.stream, dataflow.gate, dataflow.carry, handshake.load,
// handshake.store, handshake.constant, handshake.cond_br, handshake.mux,
// handshake.join.
// 3 extmemory instances (for a, b, c arrays).
// Boundary scalar and extmem connections to switches.
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

  // Data width: 32 bits for integer data
  // This is max(all FU port widths) -- i32 is the widest native type used.
  // i1, index, none are all narrower and get zero-extended to bits<32>.
  const unsigned dataWidth = 32;

  //=== Define Function Units ===

  // FU: arith.addi (2 inputs, 1 output)
  auto fuAddi = builder.defineFU(
      "fu_addi", {"i32", "i32"}, {"i32"}, {"arith.addi"});

  // FU: arith.cmpi (2 inputs, 1 output: i1)
  auto fuCmpi = builder.defineFU(
      "fu_cmpi", {"i32", "i32"}, {"i1"}, {"arith.cmpi"});

  // FU: arith.extui (1 input, 1 output)
  auto fuExtui = builder.defineFU(
      "fu_extui", {"i1", "i32"}, {"i32"}, {"arith.extui"});

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

  // FU: handshake.mux (3 inputs: sel, d0, d1; 1 output)
  auto fuMux = builder.defineFU(
      "fu_mux", {"index", "i32", "i32"}, {"i32"}, {"handshake.mux"});

  // FU: handshake.join (3 inputs: a, b, c; 1 output)
  // The vecadd DFG has a 3-input join that synchronizes three branches.
  auto fuJoin = builder.defineFU(
      "fu_join", {"none", "none", "none"}, {"none"}, {"handshake.join"});

  // FU: fu_mac (multiply-add with configurable output via static_mux)
  // Demonstrates multi-op FU with internal DAG:
  //   %d = arith.muli %a, %b : i32
  //   %e = arith.addi %d, %c : i32
  //   %g = fabric.static_mux {sel=0} %d, %e : i32
  // sel=0 -> multiply-only output; sel=1 -> multiply-add output.
  auto fuMac = builder.defineFU(
      "fu_mac", {"i32", "i32", "i32"}, {"i32"},
      {"arith.muli", "arith.addi"}, /*latency=*/1, /*interval=*/1);

  //=== Define Spatial PE (containing all FUs) ===

  // PE has 4 inputs, 4 outputs at bits<32> width.
  // This gives enough routing flexibility for the 4x4 mesh.
  std::vector<FUHandle> allFUs = {
      fuAddi, fuCmpi, fuExtui, fuIndexCast,
      fuStream, fuGate, fuCarry,
      fuLoad, fuStore, fuConstant,
      fuCondBr, fuMux, fuJoin, fuMac
  };
  auto pe = builder.defineSpatialPE("vecadd_pe", 4, 4, dataWidth, allFUs);

  //=== Define Spatial Switch ===

  // Switch: 11 inputs, 11 outputs at bits<32>
  // Ports [0..3] connect to/from local PE
  // Ports [4..7] connect to NSEW neighbors
  // Ports [8..10] connect to/from external memory (if adjacent)
  //   Port 8: extmem ldData / stData
  //   Port 9: extmem stDone / stAddr
  //   Port 10: extmem ldDone / ldAddr
  // Full crossbar connectivity
  unsigned numSwPorts = 11;
  std::vector<unsigned> swWidths(numSwPorts, dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      numSwPorts, std::vector<bool>(numSwPorts, true));
  auto sw = builder.defineSpatialSW("vecadd_sw", swWidths, swWidths,
                                    fullCrossbar);

  //=== Define External Memories ===

  auto extMemDef = builder.defineExtMemory("vecadd_extmem", 1, 1);

  //=== Build 6x6 Mesh ===
  // spatial_pe uses only 1 FU at a time (opcode selects which FU fires).
  // vecadd DFG has ~28 ops, so 6x6 = 36 PEs gives enough room.

  auto mesh = builder.buildMesh(6, 6, pe, sw);

  //=== Instantiate External Memories (for arrays a, b, c) ===

  auto extMemA = builder.instantiateExtMem(extMemDef, "extmem_a");
  auto extMemB = builder.instantiateExtMem(extMemDef, "extmem_b");
  auto extMemC = builder.instantiateExtMem(extMemDef, "extmem_c");

  //=== Add module-level memref inputs ===

  auto memA = builder.addMemrefInput("mem_a", "memref<?xi32>");
  auto memB = builder.addMemrefInput("mem_b", "memref<?xi32>");
  auto memC = builder.addMemrefInput("mem_c", "memref<?xi32>");

  builder.connectMemrefToExtMem(memA, extMemA);
  builder.connectMemrefToExtMem(memB, extMemB);
  builder.connectMemrefToExtMem(memC, extMemC);

  //=== Add scalar boundary inputs (for non-memref handshake.func args) ===

  // arg3: i32 (N, the array length) - 32 bits
  builder.addScalarInput("scalar_n", dataWidth);
  // arg4: none (start control token) - use 32-bit boundary
  builder.addScalarInput("scalar_ctrl", dataWidth);

  //=== Add scalar boundary output (done token) ===

  // return value: none (done control token) - use 32-bit boundary
  builder.addScalarOutput("scalar_done", dataWidth);

  //=== Connect ExtMemory to boundary switches ===

  // Associate each extmem with a nearby boundary switch and create real
  // SSA connections. ExtMem outputs feed SW input ports 8-10.
  // SW output ports 8-10 feed ExtMem data inputs (addr/data).
  // extmem_a -> sw_0_0, extmem_b -> sw_0_1, extmem_c -> sw_0_2
  builder.associateExtMemWithSW(extMemA, mesh.swGrid[0][0], 8, 8);
  builder.associateExtMemWithSW(extMemB, mesh.swGrid[0][1], 8, 8);
  builder.associateExtMemWithSW(extMemC, mesh.swGrid[0][2], 8, 8);

  //=== Connect scalar inputs to boundary switches ===

  // Scalar inputs feed into unused boundary switch input ports.
  // sw_0_5 unused inputs: port 4 (north, no row -1) and port 6 (east, no col 6)
  builder.connectScalarInputToInstance(0, mesh.swGrid[0][5], 4);
  builder.connectScalarInputToInstance(1, mesh.swGrid[0][5], 6);

  //=== Connect scalar output from boundary switch ===

  // Scalar output comes from a boundary switch's unused output port.
  // sw_5_5 unused outputs: port 5 (south, no row 6)
  builder.connectInstanceToScalarOutput(mesh.swGrid[5][5], 5, 0);

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
