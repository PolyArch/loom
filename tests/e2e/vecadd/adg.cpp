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
  auto fuAddi =
      builder.defineBinaryFU("fu_addi", "arith.addi", "i32", "i32");

  // FU: arith.cmpi (2 inputs, 1 output: i1)
  auto fuCmpi = builder.defineCmpiFU("fu_cmpi", "i32", "eq");

  // FU: arith.index_cast (1 input, 1 output)
  auto fuIndexCast = builder.defineIndexCastFU("fu_index_cast", "index", "i32");

  // FU: dataflow.stream (3 inputs: start, step, bound; 2 outputs: index, i1)
  auto fuStream = builder.defineStreamFU("fu_stream");

  // FU: dataflow.gate (2 inputs: value, cond; 2 outputs: value, cond)
  auto fuGate = builder.defineGateFU("fu_gate", "i32");

  // FU: dataflow.carry (3 inputs: d, a, b; 1 output)
  auto fuCarry = builder.defineCarryFU("fu_carry", "i32");

  // FU: handshake.load (3 inputs: addr, data_in, ctrl; 2 outputs: data, addr)
  auto fuLoad = builder.defineLoadFU("fu_load", "index", "i32");

  // FU: handshake.store (3 inputs: addr, data, ctrl; 2 outputs: data, addr)
  auto fuStore = builder.defineStoreFU("fu_store", "index", "i32");

  // FU: handshake.constant (1 input: ctrl; 1 output: value)
  auto fuConstant =
      builder.defineConstantFU("fu_constant", "i32", "0 : i32");

  // FU: handshake.cond_br (2 inputs: cond, data; 2 outputs: true, false)
  auto fuCondBr = builder.defineCondBrFU("fu_cond_br", "i32");

  // FU: handshake.join
  auto fuJoin = builder.defineJoinFU("fu_join", 3);

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

  //=== Instantiate Compute Fabric, Memory Banks, and Boundary Ports ===

  SwitchBankDomainSpec loadDomain;
  loadDomain.sw = sw;
  loadDomain.pe = pe;
  loadDomain.numPEs = kNumPEs;
  loadDomain.peInputCount = kPEInputs;
  loadDomain.peOutputCount = kPEOutputs;
  loadDomain.extMem = ldMemDef;
  loadDomain.numExtMems = 2;
  loadDomain.swInputPortsPerExtMem = 2;
  loadDomain.swOutputPortsPerExtMem = 1;
  loadDomain.extMemPrefix = "extmem_ld";
  loadDomain.extMemrefType = "memref<?xi32>";
  loadDomain.scalarInputTypes = {"!fabric.bits<64>", "!fabric.bits<64>"};
  loadDomain.scalarOutputTypes = {};
  auto domain = builder.buildSwitchBankDomain(loadDomain);

  auto stMems = builder.instantiateExtMemArray(1, stMemDef, "extmem_st");
  auto memC = builder.addMemrefInput("mem_2", "memref<?xi32>");
  builder.connectMemrefToExtMem(memC, stMems[0]);
  domain.cursor = builder.associateExtMemBankWithSW(stMems, domain.sw, 1, 2,
                                                    domain.cursor);

  auto scalarDone = builder.addScalarOutput("scalar_out_0", dataWidth);
  builder.connectInstanceToOutput(domain.sw, domain.cursor.nextOutputPort,
                                  scalarDone);

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
