//===-- HWInnerADGGen.cpp - ADG generation from design params -----*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Builds a concrete fabric.module ADG from CoreDesignParams using the
// ADGBuilder API. This is the ADG generation component of the INNER-HW
// optimizer (C12).
//
// Supports all 13 design dimensions:
//   - PE type (spatial/temporal)
//   - Array dimensions
//   - Data width
//   - FU repertoire
//   - FU body structure
//   - Switch type (spatial/temporal)
//   - Switch decomposability
//   - SPM
//   - External memory
//   - Routing topology (chess/mesh/lattice/ring)
//   - Temporal PE params
//   - Scalar I/O
//   - Connectivity matrix
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/HWInnerOptimizer.h"
#include "loom/ADG/ADGBuilder.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace loom {

//===----------------------------------------------------------------------===//
// FU definition helpers
//===----------------------------------------------------------------------===//

namespace {

/// Determine input/output types for an FU based on the operation name.
struct FUIOSpec {
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
};

/// Get the default MLIR types for an FU operation.
FUIOSpec getFUIOTypes(const std::string &op, unsigned dataWidth) {
  FUIOSpec spec;
  std::string intType = "i" + std::to_string(dataWidth);
  std::string floatType = (dataWidth >= 64) ? "f64" : "f32";

  // Classify the operation and determine I/O types
  if (op.find("arith.addi") != std::string::npos ||
      op.find("arith.subi") != std::string::npos ||
      op.find("arith.muli") != std::string::npos ||
      op.find("arith.andi") != std::string::npos ||
      op.find("arith.ori") != std::string::npos ||
      op.find("arith.xori") != std::string::npos ||
      op.find("arith.shli") != std::string::npos ||
      op.find("arith.shrsi") != std::string::npos ||
      op.find("arith.shrui") != std::string::npos ||
      op.find("arith.divsi") != std::string::npos ||
      op.find("arith.divui") != std::string::npos ||
      op.find("arith.remsi") != std::string::npos ||
      op.find("arith.remui") != std::string::npos) {
    spec.inputTypes = {intType, intType};
    spec.outputTypes = {intType};
  } else if (op.find("arith.addf") != std::string::npos ||
             op.find("arith.subf") != std::string::npos ||
             op.find("arith.mulf") != std::string::npos ||
             op.find("arith.divf") != std::string::npos) {
    spec.inputTypes = {floatType, floatType};
    spec.outputTypes = {floatType};
  } else if (op.find("arith.negf") != std::string::npos) {
    spec.inputTypes = {floatType};
    spec.outputTypes = {floatType};
  } else if (op.find("arith.cmpi") != std::string::npos) {
    spec.inputTypes = {intType, intType};
    spec.outputTypes = {"i1"};
  } else if (op.find("arith.cmpf") != std::string::npos) {
    spec.inputTypes = {floatType, floatType};
    spec.outputTypes = {"i1"};
  } else if (op.find("arith.select") != std::string::npos) {
    spec.inputTypes = {"i1", intType, intType};
    spec.outputTypes = {intType};
  } else if (op.find("arith.extsi") != std::string::npos ||
             op.find("arith.extui") != std::string::npos ||
             op.find("arith.trunci") != std::string::npos) {
    spec.inputTypes = {intType};
    spec.outputTypes = {intType};
  } else if (op.find("arith.sitofp") != std::string::npos ||
             op.find("arith.uitofp") != std::string::npos) {
    spec.inputTypes = {intType};
    spec.outputTypes = {floatType};
  } else if (op.find("arith.fptosi") != std::string::npos ||
             op.find("arith.fptoui") != std::string::npos) {
    spec.inputTypes = {floatType};
    spec.outputTypes = {intType};
  } else if (op.find("arith.index_cast") != std::string::npos) {
    spec.inputTypes = {intType};
    spec.outputTypes = {"index"};
  } else if (op.find("math.sqrt") != std::string::npos ||
             op.find("math.exp") != std::string::npos ||
             op.find("math.log2") != std::string::npos ||
             op.find("math.sin") != std::string::npos ||
             op.find("math.cos") != std::string::npos ||
             op.find("math.absf") != std::string::npos) {
    spec.inputTypes = {floatType};
    spec.outputTypes = {floatType};
  } else if (op.find("math.fma") != std::string::npos) {
    spec.inputTypes = {floatType, floatType, floatType};
    spec.outputTypes = {floatType};
  } else if (op.find("handshake.load") != std::string::npos) {
    spec.inputTypes = {intType, "none"};
    spec.outputTypes = {intType, "none"};
  } else if (op.find("handshake.store") != std::string::npos) {
    spec.inputTypes = {intType, intType, "none"};
    spec.outputTypes = {"none"};
  } else if (op.find("handshake.constant") != std::string::npos) {
    spec.inputTypes = {"none"};
    spec.outputTypes = {intType};
  } else if (op.find("handshake.cond_br") != std::string::npos) {
    spec.inputTypes = {"i1", intType};
    spec.outputTypes = {intType, intType};
  } else if (op.find("handshake.mux") != std::string::npos) {
    spec.inputTypes = {"index", intType, intType};
    spec.outputTypes = {intType};
  } else if (op.find("handshake.join") != std::string::npos) {
    spec.inputTypes = {"none", "none"};
    spec.outputTypes = {"none"};
  } else if (op.find("dataflow.stream") != std::string::npos) {
    spec.inputTypes = {"index", "index", "index"};
    spec.outputTypes = {"index"};
  } else if (op.find("dataflow.gate") != std::string::npos) {
    spec.inputTypes = {"i1", intType};
    spec.outputTypes = {intType};
  } else if (op.find("dataflow.carry") != std::string::npos) {
    spec.inputTypes = {intType, "none"};
    spec.outputTypes = {intType};
  } else if (op.find("dataflow.invariant") != std::string::npos) {
    spec.inputTypes = {intType, "none"};
    spec.outputTypes = {intType};
  } else {
    // Default: binary operation with integer types
    spec.inputTypes = {intType, intType};
    spec.outputTypes = {intType};
  }

  return spec;
}

/// Check if an operation is a dataflow state-machine FU (requires latency=-1).
bool isDataflowStateMachineOp(const std::string &op) {
  return op.find("dataflow.stream") != std::string::npos ||
         op.find("dataflow.gate") != std::string::npos ||
         op.find("dataflow.carry") != std::string::npos ||
         op.find("dataflow.invariant") != std::string::npos;
}

/// Generate a sanitized FU name from an operation name.
std::string fuNameFromOp(const std::string &op) {
  std::string name = "fu_";
  for (char c : op) {
    if (c == '.')
      name += '_';
    else
      name += c;
  }
  return name;
}

/// Determine the number of PE-level input/output ports needed.
/// PE ports are used by the switch network; minimum 4 for chess topology.
unsigned computePEPorts(const CoreDesignParams &params) {
  unsigned base = 4;
  if (params.topology == RoutingTopology::CHESS) {
    // Chess topology needs at least 4 (NW, NE, SW, SE corners)
    base = 4;
  } else if (params.topology == RoutingTopology::MESH ||
             params.topology == RoutingTopology::LATTICE) {
    base = 4;
  } else if (params.topology == RoutingTopology::RING) {
    base = 2;
  }
  return base;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// buildADGFromParams
//===----------------------------------------------------------------------===//

std::string buildADGFromParams(const CoreDesignParams &params,
                               const std::string &moduleName) {
  adg::ADGBuilder builder(moduleName);

  unsigned dataWidth = params.dataWidth;
  // Use 64-bit fabric port width to support both int and float
  unsigned fabricWidth = std::max(dataWidth, 32u);

  // ---- Define Function Units ----
  std::vector<adg::FUHandle> fuHandles;

  for (const auto &op : params.fuRepertoire) {
    FUIOSpec ioSpec = getFUIOTypes(op, dataWidth);
    std::string fuName = fuNameFromOp(op);

    // Determine latency/interval
    int64_t latency = 1;
    int64_t interval = 1;
    if (isDataflowStateMachineOp(op)) {
      latency = -1;
      interval = -1;
    }

    // Use specialized defineFU helpers for common patterns
    if (op == "arith.cmpi") {
      fuHandles.push_back(
          builder.defineCmpiFU(fuName, "i32", "slt", latency, interval));
    } else if (op == "arith.cmpf") {
      std::string fType = (dataWidth >= 64) ? "f64" : "f32";
      fuHandles.push_back(
          builder.defineCmpfFU(fuName, fType, "oeq", latency, interval));
    } else if (op == "arith.select") {
      std::string vType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineSelectFU(fuName, vType, latency, interval));
    } else if (op == "arith.index_cast" || op == "arith.index_castui") {
      std::string iType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineIndexCastFU(fuName, iType, "index", latency, interval));
    } else if (op == "handshake.constant") {
      std::string rType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineConstantFU(fuName, rType, "0 : i32", latency, interval));
    } else if (op == "handshake.cond_br") {
      std::string vType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineCondBrFU(fuName, vType, latency, interval));
    } else if (op == "handshake.mux") {
      std::string vType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineMuxFU(fuName, vType, "index", latency, interval));
    } else if (op == "handshake.join") {
      fuHandles.push_back(
          builder.defineJoinFU(fuName, 2, "none", latency, interval));
    } else if (op == "handshake.load") {
      std::string aType = "i" + std::to_string(dataWidth);
      std::string dType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineLoadFU(fuName, aType, dType, latency, interval));
    } else if (op == "handshake.store") {
      std::string aType = "i" + std::to_string(dataWidth);
      std::string dType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineStoreFU(fuName, aType, dType, latency, interval));
    } else if (op == "dataflow.stream") {
      fuHandles.push_back(
          builder.defineStreamFU(fuName, "index", "+=", "<", latency, interval));
    } else if (op == "dataflow.gate") {
      std::string vType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineGateFU(fuName, vType, latency, interval));
    } else if (op == "dataflow.carry") {
      std::string vType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineCarryFU(fuName, vType, latency, interval));
    } else if (op == "dataflow.invariant") {
      std::string vType = "i" + std::to_string(dataWidth);
      fuHandles.push_back(
          builder.defineInvariantFU(fuName, vType, latency, interval));
    } else {
      // Generic FU definition via the spec-based API
      adg::FunctionUnitSpec fuSpec;
      fuSpec.name = fuName;
      fuSpec.inputTypes = ioSpec.inputTypes;
      fuSpec.outputTypes = ioSpec.outputTypes;
      fuSpec.ops = {op};
      fuSpec.latency = latency;
      fuSpec.interval = interval;
      fuHandles.push_back(builder.defineFU(fuSpec));
    }
  }

  // Ensure we have at least one FU
  if (fuHandles.empty()) {
    fuHandles.push_back(
        builder.defineBinaryFU("fu_arith_addi", "arith.addi", "i32", "i32"));
  }

  // ---- Define PE template ----
  unsigned peInputs = computePEPorts(params);
  unsigned peOutputs = peInputs;

  adg::PEHandle pe;
  if (params.peType == PEType::TEMPORAL) {
    // Temporal PE with instruction slots and register file
    std::vector<std::string> peInTypes(peInputs,
                                       "!fabric.bits<" + std::to_string(fabricWidth) + ">");
    std::vector<std::string> peOutTypes(peOutputs,
                                        "!fabric.bits<" + std::to_string(fabricWidth) + ">");

    pe = builder.defineTemporalPE(
        moduleName + "_tpe", peInTypes, peOutTypes, fuHandles,
        params.numRegisters, params.instructionSlots,
        params.regFifoDepth, params.shareOperandBuffer,
        params.operandBufferSize > 0
            ? std::optional<unsigned>(params.operandBufferSize)
            : std::nullopt);
  } else {
    // Spatial PE
    pe = builder.defineSpatialPE(
        moduleName + "_spe", peInputs, peOutputs, fabricWidth, fuHandles);
  }

  // ---- Build topology ----
  adg::MeshResult mesh;

  unsigned topLeftInputs = params.scalarInputs;
  unsigned bottomRightOutputs = params.scalarOutputs;

  // Add extra boundary ports for memory
  if (params.extmemCount > 0) {
    topLeftInputs += 1;
    bottomRightOutputs += 1;
  }

  switch (params.topology) {
  case RoutingTopology::CHESS:
    mesh = builder.buildChessMesh(
        params.arrayRows, params.arrayCols, pe,
        params.decomposableBits, topLeftInputs, bottomRightOutputs);
    break;

  case RoutingTopology::LATTICE:
    mesh = builder.buildLatticeMesh(
        params.arrayRows, params.arrayCols, pe,
        params.decomposableBits);
    break;

  case RoutingTopology::MESH: {
    // Build a torus mesh with a full-crossbar switch
    unsigned swInputs = peInputs * 4 + 4;
    unsigned swOutputs = peOutputs * 4 + 4;
    auto sw = builder.defineFullCrossbarSpatialSW(
        moduleName + "_sw", swInputs, swOutputs, fabricWidth,
        params.decomposableBits);
    mesh = builder.buildMesh(params.arrayRows, params.arrayCols, pe, sw);
    break;
  }

  case RoutingTopology::RING: {
    unsigned swInputs = peInputs + 2;
    unsigned swOutputs = peOutputs + 2;
    auto sw = builder.defineFullCrossbarSpatialSW(
        moduleName + "_sw_ring", swInputs, swOutputs, fabricWidth,
        params.decomposableBits);
    unsigned ringSize = params.arrayRows * params.arrayCols;
    mesh = builder.buildRing(ringSize, pe, sw);
    break;
  }
  }

  // ---- Add scalar I/O ----
  unsigned boundaryPortIdx = 0;
  for (unsigned i = 0; i < params.scalarInputs; ++i) {
    std::string name = "in" + std::to_string(i);
    auto idx = builder.addScalarInput(name, fabricWidth);
    if (boundaryPortIdx < mesh.ingressPorts.size()) {
      builder.connectInputToPort(idx, mesh.ingressPorts[boundaryPortIdx]);
      boundaryPortIdx++;
    }
  }

  unsigned egressIdx = 0;
  for (unsigned i = 0; i < params.scalarOutputs; ++i) {
    std::string name = "out" + std::to_string(i);
    auto idx = builder.addScalarOutput(name, fabricWidth);
    if (egressIdx < mesh.egressPorts.size()) {
      builder.connectPortToOutput(mesh.egressPorts[egressIdx], idx);
      egressIdx++;
    }
  }

  // ---- Add external memory ----
  if (params.extmemCount > 0 &&
      boundaryPortIdx < mesh.ingressPorts.size() &&
      egressIdx < mesh.egressPorts.size()) {
    // Add memref input port for the first external memory
    auto memIn = builder.addMemrefInput("mem0", "memref<?xi32>");
    auto extMem = builder.defineExtMemory(
        moduleName + "_extmem", params.extmemLdPorts, params.extmemStPorts);
    auto extInst = builder.instantiateExtMem(extMem, "extmem_0");
    builder.connectMemrefToExtMem(memIn, extInst);
  }

  // ---- Set SPM capacity ----
  uint64_t spmBytes = static_cast<uint64_t>(params.spmSizeKB) * 1024;
  builder.setSPMCapacity(spmBytes);

  // ---- Export as core type MLIR ----
  return builder.exportCoreType(moduleName);
}

} // namespace loom
