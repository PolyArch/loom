//===-- adg.cpp - GUI test ADG generation (2x2 mesh) -----------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// Constructs a small 2x2 mesh ADG for GUI visualization testing.
//
// Each PE has 4 FUs:
//   - fu_add:   simple arith.addi
//   - fu_cmp:   simple arith.cmpi
//   - fu_mac:   compound multiply-add (arith.muli + arith.addi) with static_mux
//   - fu_logic: compound bitwise (arith.andi + arith.ori) with static_mux
//
// 1 extmemory instance (extmem_a), 3 scalar inputs, 1 scalar output.
// Torus topology on a 2x2 mesh.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static void buildGuiTestADG(const std::string &outputPath) {
  ADGBuilder builder("gui_test_adg");

  // Data width: 32 bits
  const unsigned dataWidth = 32;

  //=== Define Function Units ===

  // FU: fu_add - simple adder (2 inputs, 1 output)
  auto fuAdd = builder.defineFU(
      "fu_add", {"i32", "i32"}, {"i32"}, {"arith.addi"});

  // FU: fu_cmp - simple comparator (2 inputs, 1 output: i1)
  auto fuCmp = builder.defineFU(
      "fu_cmp", {"i32", "i32"}, {"i1"}, {"arith.cmpi"});

  // FU: fu_mac - compound multiply-add with static_mux
  //   %d = arith.muli %a, %b : i32
  //   %e = arith.addi %d, %c : i32
  //   %g = fabric.static_mux {sel=0} %d, %e : i32
  auto fuMac = builder.defineFU(
      "fu_mac", {"i32", "i32", "i32"}, {"i32"},
      {"arith.muli", "arith.addi"}, /*latency=*/1, /*interval=*/1);

  // FU: fu_logic - compound bitwise with static_mux
  //   %d = arith.andi %a, %b : i32
  //   %e = arith.ori %a, %b : i32
  //   %g = fabric.static_mux {sel=0} %d, %e : i32
  auto fuLogic = builder.defineFU(
      "fu_logic", {"i32", "i32"}, {"i32"},
      {"arith.andi", "arith.ori"}, /*latency=*/1, /*interval=*/1);

  // FU: fu_constant - handshake.constant (1 ctrl input, 1 output)
  auto fuConstant = builder.defineFU(
      "fu_constant", {"none"}, {"i32"}, {"handshake.constant"});

  // FU: fu_join - handshake.join (synchronize control tokens)
  auto fuJoin = builder.defineFU(
      "fu_join", {"none", "none"}, {"none"}, {"handshake.join"});

  //=== Define Spatial PE ===

  // PE: 2 inputs, 2 outputs at bits<32> width
  std::vector<FUHandle> allFUs = {fuAdd, fuCmp, fuMac, fuLogic,
                                   fuConstant, fuJoin};
  auto pe = builder.defineSpatialPE("gui_pe", 2, 2, dataWidth, allFUs);

  //=== Define Spatial Switch ===

  // Switch: 8 inputs, 8 outputs at bits<32>
  // Ports [0..1] connect to/from local PE
  // Ports [2..5] connect to NSEW neighbors
  // Ports [6..7] reserved for scalar I/O on boundary switches
  unsigned numSwPorts = 8;
  std::vector<unsigned> swWidths(numSwPorts, dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      numSwPorts, std::vector<bool>(numSwPorts, true));
  auto sw = builder.defineSpatialSW("gui_sw", swWidths, swWidths,
                                    fullCrossbar);

  //=== Define External Memory ===

  auto extMemDef = builder.defineExtMemory("gui_extmem", 1, 1);

  //=== Build 2x2 Mesh ===

  auto mesh = builder.buildMesh(2, 2, pe, sw);

  //=== Instantiate External Memory ===

  auto extMemA = builder.instantiateExtMem(extMemDef, "extmem_a");

  //=== Add module-level memref input ===

  auto memA = builder.addMemrefInput("mem_a", "memref<?xi32>");
  builder.connectMemrefToExtMem(memA, extMemA);

  //=== Add scalar boundary inputs and wire to switches ===

  auto sIn0 = builder.addScalarInput("scalar_in0", dataWidth);
  auto sIn1 = builder.addScalarInput("scalar_in1", dataWidth);
  auto sCtrl = builder.addScalarInput("scalar_ctrl", dataWidth);

  // Wire scalar inputs to boundary switch ports [6]
  builder.connectScalarInputToInstance(sIn0, mesh.swGrid[0][0], 6);
  builder.connectScalarInputToInstance(sIn1, mesh.swGrid[0][1], 6);
  builder.connectScalarInputToInstance(sCtrl, mesh.swGrid[1][0], 6);

  //=== Add scalar boundary output and wire from switch ===

  auto sOut0 = builder.addScalarOutput("scalar_done", dataWidth);
  builder.connectInstanceToScalarOutput(mesh.swGrid[0][0], 6, sOut0);

  //=== ExtMemory association ===

  builder.associateExtMemWithSW(extMemA, mesh.swGrid[0][0], 2, 2);

  //=== Export ===

  builder.exportMLIR(outputPath);
}

//===----------------------------------------------------------------------===//
// Standalone tool entry point
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("gui_test.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "GUI test ADG generator\n");

  // Ensure output directory exists
  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: "
                   << parentPath << "\n";
      return 1;
    }
  }

  llvm::outs() << "Generating GUI test ADG -> " << outputFile << "\n";
  buildGuiTestADG(outputFile);
  llvm::outs() << "Done.\n";

  return 0;
}
