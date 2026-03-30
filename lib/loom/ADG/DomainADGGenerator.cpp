//===-- DomainADGGenerator.cpp - Domain-specific core type ADG gen --*- C++ -*-//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements parametric generation of the 6 domain-specific core types (D1-D6)
// using the ADGBuilder API. Follows the same proven pattern as KHGGenerator:
// chess mesh topology with boundary-distributed external memory and scalar I/O.
//
// Each domain type has hand-tuned parameters from the Python core_type_library
// (scripts/dse/core_type_library.py _DOMAIN_SPECIFIC_DEFS).
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/DomainADGGenerator.h"
#include "loom/ADG/ADGBuilder.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Domain Type Definitions (mirror of Python _DOMAIN_SPECIFIC_DEFS)
//===----------------------------------------------------------------------===//

namespace {

/// Static table of the 6 domain-specific types. Kept in sync with the Python
/// core_type_library._DOMAIN_SPECIFIC_DEFS.
static const DomainTypeParams kDomainTypes[] = {
    // D1: LLM (FP-heavy, large)
    {"D1", "LLM",     6, 6, 4, 4, 4, 2, 64, false, 0,  0, 32},
    // D2: CV (mixed, medium)
    {"D2", "CV",      4, 4, 4, 3, 3, 2, 32, false, 0,  0, 32},
    // D3: Signal (temporal, multiply)
    {"D3", "Signal",  4, 4, 3, 4, 2, 2, 16, true,  8,  8, 32},
    // D4: Crypto (INT-heavy, bitwise, 64-bit datapath)
    {"D4", "Crypto",  4, 4, 6, 4, 0, 2,  8, false, 0,  0, 64},
    // D5: Sensor (temporal, control)
    {"D5", "Sensor",  4, 4, 4, 2, 1, 2,  8, true, 16,  8, 32},
    // D6: Control (spatial, balanced)
    {"D6", "Control", 4, 4, 4, 2, 0, 2,  4, false, 0,  0, 32},
};

constexpr unsigned kNumDomainTypes = 6;

static const SciCompTypeParams kSciCompTypes[] = {
    {"SC-FP", "ScientificFP", 12, 12, 64, false, 8, 6, 1, 4, 1, true, 32, 32,
     2, 2, 2, 1, 0, 0, 0, 4, 2, "CHESS", true, true, true, false, false,
     false},
    {"SC-SPM", "ScientificSPM", 8, 8, 64, false, 4, 4, 1, 6, 1, true, 32, 64,
     4, 2, 4, 2, 0, 0, 0, 4, 2, "CHESS", true, false, false, true, false,
     false},
    {"SC-CTRL", "ScientificCTRL", 8, 8, 32, true, 2, 2, 0, 6, 2, false, 0, 32,
     2, 2, 2, 2, 16, 8, 4, 4, 2, "MESH", false, false, false, false, true,
     true},
};

constexpr unsigned kNumSciCompTypes = 3;

//===----------------------------------------------------------------------===//
// Constants (same conventions as KHGGenerator)
//===----------------------------------------------------------------------===//

constexpr unsigned kPEInputs = 4;
constexpr unsigned kPEOutputs = 4;
constexpr unsigned kExtMemOutputs = 3;     // ld_data, ld_done, st_done
constexpr unsigned kExtMemDataInputs = 3;  // ld_addr, st_addr, st_data
constexpr unsigned kNumExtMems = 2;
constexpr unsigned kNumScalarInputs = 4;
constexpr unsigned kNumScalarOutputs = 2;

static std::string bitsType(unsigned width) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

static std::string boolAttr(bool value) { return value ? "true" : "false"; }

/// Determine the integer type string for a given data width.
/// Returns "i32" for width <= 32, "i64" for width <= 64.
static std::string intTypeStr(unsigned width) {
  return (width > 32) ? "i64" : "i32";
}

/// Determine the float type string matching the data width.
static std::string floatTypeStr(unsigned width) {
  return (width > 32) ? "f64" : "f32";
}

//===----------------------------------------------------------------------===//
// FU Definition Helpers
//===----------------------------------------------------------------------===//

/// Define the baseline dataflow FUs needed for kernel mapping.
/// Mirrors the baseline set from KHGGenerator/tapestry_adg_gen.
/// Adapts types according to the core's data width.
static std::vector<FUHandle>
defineBaselineFUs(ADGBuilder &builder, const std::string &prefix,
                  unsigned dataWidth) {
  std::vector<FUHandle> fus;
  std::string itype = intTypeStr(dataWidth);
  std::string ftype = floatTypeStr(dataWidth);

  // Constants
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_int", itype, "0 : " + itype));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_index", "index", "0 : index"));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_float", ftype, "0.000000e+00 : " + ftype));

  // Index casts
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_index_to_int", "index", itype));
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_int_to_index", itype, "index"));

  // Dataflow control FUs
  fus.push_back(builder.defineStreamFU(prefix + "_stream"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_int", itype));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_none", "none"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_index", "index"));
  fus.push_back(builder.defineJoinFU(prefix + "_join", 4));
  fus.push_back(builder.defineGateFU(prefix + "_gate_int", itype));
  fus.push_back(builder.defineGateFU(prefix + "_gate_index", "index"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_float", ftype));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i1", "i1"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_int", itype));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_none", "none"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_float", ftype));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_int", itype));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_none", "none"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_float", ftype));
  fus.push_back(builder.defineInvariantFU(
      prefix + "_invariant_int", itype));
  fus.push_back(builder.defineInvariantFU(
      prefix + "_invariant_index", "index"));
  fus.push_back(builder.defineInvariantFU(
      prefix + "_invariant_float", ftype));
  fus.push_back(builder.defineInvariantFU(
      prefix + "_invariant_none", "none"));
  fus.push_back(builder.defineInvariantFU(
      prefix + "_invariant_i1", "i1"));

  // Memory access FUs
  fus.push_back(builder.defineLoadFU(prefix + "_load", "index", itype));
  fus.push_back(builder.defineStoreFU(prefix + "_store", "index", itype));

  // Comparison and selection
  fus.push_back(builder.defineSelectFU(prefix + "_select_int", itype));
  fus.push_back(builder.defineSelectFU(prefix + "_select_index", "index"));
  fus.push_back(builder.defineCmpiFU(prefix + "_cmpi_int", itype, "slt"));

  // Index-typed arithmetic
  fus.push_back(builder.defineBinaryFU(
      prefix + "_addi_index", "arith.addi", "index", "index"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_muli_index", "arith.muli", "index", "index"));

  return fus;
}

/// Define ALU FUs (integer arithmetic + bitwise).
static void defineALUFUs(ADGBuilder &builder, const std::string &prefix,
                         unsigned count, unsigned dataWidth,
                         std::vector<FUHandle> &fus) {
  std::string itype = intTypeStr(dataWidth);
  for (unsigned i = 0; i < count; ++i) {
    std::string suffix = "_alu" + std::to_string(i);
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_addi", "arith.addi", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_subi", "arith.subi", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_andi", "arith.andi", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_ori", "arith.ori", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_xori", "arith.xori", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_shli", "arith.shli", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_shrsi", "arith.shrsi", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_shrui", "arith.shrui", itype, itype));
  }
}

/// Define MUL FUs (integer multiply + divide).
static void defineMulFUs(ADGBuilder &builder, const std::string &prefix,
                         unsigned count, unsigned dataWidth,
                         std::vector<FUHandle> &fus) {
  std::string itype = intTypeStr(dataWidth);
  for (unsigned i = 0; i < count; ++i) {
    std::string suffix = "_mul" + std::to_string(i);
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_muli", "arith.muli", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_divsi", "arith.divsi", itype, itype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_remsi", "arith.remsi", itype, itype));
  }
}

/// Define FP FUs (floating-point arithmetic + conversion).
static void defineFPFUs(ADGBuilder &builder, const std::string &prefix,
                        unsigned count, unsigned dataWidth,
                        std::vector<FUHandle> &fus) {
  if (count == 0)
    return;

  std::string itype = intTypeStr(dataWidth);
  std::string ftype = floatTypeStr(dataWidth);
  for (unsigned i = 0; i < count; ++i) {
    std::string suffix = "_fp" + std::to_string(i);
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_addf", "arith.addf", ftype, ftype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_subf", "arith.subf", ftype, ftype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_mulf", "arith.mulf", ftype, ftype));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_divf", "arith.divf", ftype, ftype));
    fus.push_back(builder.defineCmpfFU(
        prefix + suffix + "_cmpf", ftype, "olt"));
    fus.push_back(builder.defineSelectFU(
        prefix + suffix + "_select_float", ftype));
    fus.push_back(builder.defineUnaryFU(
        prefix + suffix + "_sitofp", "arith.sitofp", itype, ftype));
    fus.push_back(builder.defineUnaryFU(
        prefix + suffix + "_fptosi", "arith.fptosi", ftype, itype));
    fus.push_back(builder.defineUnaryFU(
        prefix + suffix + "_negf", "arith.negf", ftype, ftype));
  }
}

static void defineSciCompSpecializedFUs(ADGBuilder &builder,
                                        const SciCompTypeParams &params,
                                        const std::string &prefix,
                                        std::vector<FUHandle> &fus) {
  std::string itype = intTypeStr(params.dataWidth);
  std::string ftype = floatTypeStr(params.dataWidth);

  if (params.hasFMA) {
    FunctionUnitSpec fmaSpec;
    fmaSpec.name = prefix + "_fp_fma";
    fmaSpec.inputTypes = {ftype, ftype, ftype};
    fmaSpec.outputTypes = {ftype};
    fmaSpec.ops = {"math.fma"};
    fus.push_back(builder.defineFU(fmaSpec));
  }
  if (params.hasRSQRT) {
    FunctionUnitSpec rsqrtSpec;
    rsqrtSpec.name = prefix + "_fp_rsqrt";
    rsqrtSpec.inputTypes = {ftype};
    rsqrtSpec.outputTypes = {ftype};
    rsqrtSpec.ops = {"math.rsqrt"};
    fus.push_back(builder.defineFU(rsqrtSpec));
  }
  if (params.hasFPMin) {
    FunctionUnitSpec minSpec;
    minSpec.name = prefix + "_fp_min";
    minSpec.inputTypes = {ftype, ftype};
    minSpec.outputTypes = {ftype};
    minSpec.ops = {"arith.minimumf"};
    fus.push_back(builder.defineFU(minSpec));
  }
  if (params.hasIndirectLoad) {
    FunctionUnitSpec indirectLoadSpec;
    indirectLoadSpec.name = prefix + "_indirect_load";
    indirectLoadSpec.inputTypes = {"index", itype, "none"};
    indirectLoadSpec.outputTypes = {itype, "index"};
    indirectLoadSpec.ops = {"handshake.load"};
    fus.push_back(builder.defineFU(indirectLoadSpec));
  }
  if (params.hasScatterStore) {
    FunctionUnitSpec scatterSpec;
    scatterSpec.name = prefix + "_scatter_store";
    scatterSpec.inputTypes = {"index", itype, "none"};
    scatterSpec.outputTypes = {"none"};
    scatterSpec.ops = {"handshake.store"};
    fus.push_back(builder.defineFU(scatterSpec));
  }
  if (params.hasBranch) {
    FunctionUnitSpec branchSpec;
    branchSpec.name = prefix + "_branch";
    branchSpec.inputTypes = {"i1", itype};
    branchSpec.outputTypes = {itype, itype};
    branchSpec.ops = {"handshake.cond_br"};
    fus.push_back(builder.defineFU(branchSpec));
  }
}

static std::string addModuleAttributes(
    std::string mlir, const std::map<std::string, std::string> &attrs) {
  std::size_t modulePos = mlir.find("fabric.module @");
  if (modulePos == std::string::npos || attrs.empty())
    return mlir;

  std::size_t bracePos = mlir.find(" {", modulePos);
  if (bracePos == std::string::npos)
    return mlir;

  std::size_t attrPos = mlir.rfind(" attributes {", bracePos);
  std::string attrText;
  bool hasAttrBlock = attrPos != std::string::npos && attrPos > modulePos;
  if (hasAttrBlock) {
    std::size_t attrEnd = mlir.find('}', attrPos);
    if (attrEnd == std::string::npos)
      return mlir;
    attrText = mlir.substr(attrPos + 13, attrEnd - (attrPos + 13));
    bracePos = attrEnd + 1;
  }

  std::string newAttrs = attrText;
  for (const auto &entry : attrs) {
    if (!newAttrs.empty())
      newAttrs += ", ";
    newAttrs += entry.first + " = " + entry.second;
  }

  if (hasAttrBlock) {
    std::size_t attrEnd = mlir.find('}', attrPos);
    mlir.replace(attrPos, attrEnd - attrPos + 1,
                 " attributes {" + newAttrs + "}");
    return mlir;
  }

  mlir.insert(bracePos, " attributes {" + newAttrs + "}");
  return mlir;
}

//===----------------------------------------------------------------------===//
// Core ADG Build Logic
//===----------------------------------------------------------------------===//

/// Core ADG build logic for domain-specific types. Follows the same proven
/// pattern as KHGGenerator::buildKHGADGImpl: chess mesh with boundary-
/// distributed external memory and scalar I/O.
static void buildDomainADGImpl(ADGBuilder &builder,
                               const DomainTypeParams &params,
                               const std::string &moduleName) {
  // Build FU list: baseline + compute-mix specific
  std::vector<FUHandle> fus =
      defineBaselineFUs(builder, moduleName, params.dataWidth);
  defineALUFUs(builder, moduleName, params.fuAluCount, params.dataWidth, fus);
  defineMulFUs(builder, moduleName, params.fuMulCount, params.dataWidth, fus);
  defineFPFUs(builder, moduleName, params.fuFpCount, params.dataWidth, fus);

  // Define PE template
  PEHandle pe;
  std::string portType = bitsType(params.dataWidth);
  std::vector<std::string> peInTypes(kPEInputs, portType);
  std::vector<std::string> peOutTypes(kPEOutputs, portType);

  if (params.isTemporal) {
    unsigned regFifoDepth = (params.numRegisters > 0) ? 1 : 0;
    pe = builder.defineTemporalPE(
        moduleName + "_tpe", peInTypes, peOutTypes, fus,
        params.numRegisters, params.instructionSlots,
        regFifoDepth, /*enableShareOperandBuffer=*/false,
        std::nullopt);

    std::vector<std::vector<InstanceHandle>> peGrid(
        params.arrayRows, std::vector<InstanceHandle>(params.arrayCols));
    for (unsigned r = 0; r < params.arrayRows; ++r) {
      for (unsigned c = 0; c < params.arrayCols; ++c) {
        peGrid[r][c] = builder.instantiatePE(
            pe, "pe_" + std::to_string(r) + "_" + std::to_string(c));
      }
    }
    for (unsigned r = 0; r < params.arrayRows; ++r) {
      for (unsigned c = 0; c < params.arrayCols; ++c) {
        unsigned north = (r > 0) ? r - 1 : params.arrayRows - 1;
        unsigned south = (r + 1 < params.arrayRows) ? r + 1 : 0;
        unsigned east = (c + 1 < params.arrayCols) ? c + 1 : 0;
        unsigned west = (c > 0) ? c - 1 : params.arrayCols - 1;
        builder.connect(peGrid[r][c], 0, peGrid[north][c], 1);
        builder.connect(peGrid[r][c], 1, peGrid[south][c], 0);
        builder.connect(peGrid[r][c], 2, peGrid[r][east], 3);
        builder.connect(peGrid[r][c], 3, peGrid[r][west], 2);
      }
    }

    uint64_t spmBytes = static_cast<uint64_t>(params.spmSizeKB) * 1024;
    builder.setSPMCapacity(spmBytes);
    return;
  } else {
    pe = builder.defineSpatialPE(
        moduleName + "_spe", kPEInputs, kPEOutputs, params.dataWidth, fus);
  }

  // Compute boundary port layout (same pattern as KHGGenerator)
  unsigned leftIngressMems = (kNumExtMems + 1) / 2;
  unsigned rightIngressMems = kNumExtMems / 2;
  unsigned leftEgressMems = (kNumExtMems + 1) / 2;
  unsigned rightEgressMems = kNumExtMems / 2;

  ChessMeshOptions meshOpts;
  meshOpts.topLeftExtraInputs =
      leftIngressMems * kExtMemOutputs + kNumScalarInputs;
  meshOpts.topRightExtraInputs = rightIngressMems * kExtMemOutputs;
  meshOpts.bottomLeftExtraOutputs = leftEgressMems * kExtMemDataInputs;
  meshOpts.bottomRightExtraOutputs =
      rightEgressMems * kExtMemDataInputs + kNumScalarOutputs;

  // Build chess mesh topology
  auto mesh = builder.buildChessMesh(
      params.arrayRows, params.arrayCols,
      [&](unsigned, unsigned) { return pe; },
      meshOpts);

  // Define and instantiate external memory
  std::string itype = intTypeStr(params.dataWidth);
  std::string memrefType = "memref<?" + ("x" + itype) + ">";

  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_extmem";
  extMemSpec.ldPorts = 1;
  extMemSpec.stPorts = 1;
  extMemSpec.memrefType = memrefType;
  extMemSpec.numRegion = 1;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMems = builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs =
      builder.addMemrefInputs("buffer", kNumExtMems, memrefType);
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  // Wire external memory to boundary ports (round-robin left/right)
  unsigned leftIngressIdx = 0;
  unsigned rightIngressIdx = meshOpts.topLeftExtraInputs;
  unsigned leftEgressIdx = 0;
  unsigned rightEgressIdx = meshOpts.bottomLeftExtraOutputs;

  for (unsigned memIdx = 0; memIdx < extMems.size(); ++memIdx) {
    InstanceHandle mem = extMems[memIdx];
    unsigned &ingressIdx =
        (memIdx % 2 == 0) ? leftIngressIdx : rightIngressIdx;
    unsigned &egressIdx =
        (memIdx % 2 == 0) ? leftEgressIdx : rightEgressIdx;

    for (unsigned outPort = 0; outPort < kExtMemOutputs; ++outPort) {
      builder.connect(mem, outPort,
                      mesh.ingressPorts[ingressIdx].instance,
                      mesh.ingressPorts[ingressIdx].port);
      ++ingressIdx;
    }
    for (unsigned inPort = 0; inPort < kExtMemDataInputs; ++inPort) {
      builder.connect(mesh.egressPorts[egressIdx].instance,
                      mesh.egressPorts[egressIdx].port,
                      mem, 1 + inPort);
      ++egressIdx;
    }
  }

  // Wire scalar I/O through remaining boundary ports
  std::vector<unsigned> scalarIns = builder.addInputs(
      "scalar",
      std::vector<std::string>(kNumScalarInputs, bitsType(params.dataWidth)));
  std::vector<unsigned> scalarOuts = builder.addOutputs(
      "scalar_out",
      std::vector<std::string>(kNumScalarOutputs, bitsType(params.dataWidth)));

  unsigned scalarIngressIdx = leftIngressMems * kExtMemOutputs;
  for (unsigned idx = 0; idx < scalarIns.size(); ++idx, ++scalarIngressIdx)
    builder.connectInputToPort(scalarIns[idx],
                               mesh.ingressPorts[scalarIngressIdx]);

  unsigned scalarEgressIdx = kNumExtMems * kExtMemDataInputs;
  for (unsigned idx = 0; idx < scalarOuts.size(); ++idx, ++scalarEgressIdx)
    builder.connectPortToOutput(mesh.egressPorts[scalarEgressIdx],
                                scalarOuts[idx]);

  // Set SPM capacity attribute
  uint64_t spmBytes = static_cast<uint64_t>(params.spmSizeKB) * 1024;
  builder.setSPMCapacity(spmBytes);
}

static void buildSciCompADGImpl(ADGBuilder &builder,
                                const SciCompTypeParams &params,
                                const std::string &moduleName) {
  std::vector<FUHandle> fus =
      defineBaselineFUs(builder, moduleName, params.dataWidth);
  defineALUFUs(builder, moduleName, params.intAluCount, params.dataWidth, fus);
  defineMulFUs(builder, moduleName, params.intMulCount, params.dataWidth, fus);
  defineFPFUs(builder, moduleName, params.totalFPUnits(), params.dataWidth,
              fus);
  defineSciCompSpecializedFUs(builder, params, moduleName, fus);

  PEHandle pe;
  std::string portType = bitsType(params.dataWidth);
  std::vector<std::string> peInTypes(kPEInputs, portType);
  std::vector<std::string> peOutTypes(kPEOutputs, portType);

  if (params.isTemporal) {
    pe = builder.defineTemporalPE(
        moduleName + "_tpe", peInTypes, peOutTypes, fus, params.numRegisters,
        params.instructionSlots, /*regFifoDepth=*/1,
        /*enableShareOperandBuffer=*/params.operandBufferSize > 0,
        params.operandBufferSize > 0
            ? std::optional<unsigned>(params.operandBufferSize)
            : std::nullopt);

    std::vector<std::vector<InstanceHandle>> peGrid(
        params.arrayRows, std::vector<InstanceHandle>(params.arrayCols));
    for (unsigned r = 0; r < params.arrayRows; ++r) {
      for (unsigned c = 0; c < params.arrayCols; ++c) {
        peGrid[r][c] = builder.instantiatePE(
            pe, "pe_" + std::to_string(r) + "_" + std::to_string(c));
      }
    }
    for (unsigned r = 0; r < params.arrayRows; ++r) {
      for (unsigned c = 0; c < params.arrayCols; ++c) {
        unsigned north = (r > 0) ? r - 1 : params.arrayRows - 1;
        unsigned south = (r + 1 < params.arrayRows) ? r + 1 : 0;
        unsigned east = (c + 1 < params.arrayCols) ? c + 1 : 0;
        unsigned west = (c > 0) ? c - 1 : params.arrayCols - 1;
        builder.connect(peGrid[r][c], 0, peGrid[north][c], 1);
        builder.connect(peGrid[r][c], 1, peGrid[south][c], 0);
        builder.connect(peGrid[r][c], 2, peGrid[r][east], 3);
        builder.connect(peGrid[r][c], 3, peGrid[r][west], 2);
      }
    }

    uint64_t spmBytes = static_cast<uint64_t>(params.spmSizeKB) * 1024;
    builder.setSPMCapacity(spmBytes);
    return;
  } else {
    pe = builder.defineSpatialPE(moduleName + "_spe", kPEInputs, kPEOutputs,
                                 params.dataWidth, fus);
  }

  ChessMeshOptions meshOpts;
  meshOpts.topLeftExtraInputs = kExtMemOutputs + params.scalarInputs;
  meshOpts.topRightExtraInputs = kExtMemOutputs;
  meshOpts.bottomLeftExtraOutputs = kExtMemDataInputs;
  meshOpts.bottomRightExtraOutputs =
      kExtMemDataInputs + params.scalarOutputs;

  auto mesh = builder.buildChessMesh(params.arrayRows, params.arrayCols,
                                     [&](unsigned, unsigned) { return pe; },
                                     meshOpts);

  std::string itype = intTypeStr(params.dataWidth);
  std::string memrefType = "memref<?x" + itype + ">";

  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_extmem";
  extMemSpec.ldPorts = std::max(1u, params.extMemLdPorts);
  extMemSpec.stPorts = std::max(1u, params.extMemStPorts);
  extMemSpec.memrefType = memrefType;
  extMemSpec.numRegion = 1;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMems = builder.instantiateExtMemArray(2, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", 2, memrefType);
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  unsigned leftIngressIdx = 0;
  unsigned rightIngressIdx = meshOpts.topLeftExtraInputs;
  unsigned leftEgressIdx = 0;
  unsigned rightEgressIdx = meshOpts.bottomLeftExtraOutputs;

  for (unsigned memIdx = 0; memIdx < extMems.size(); ++memIdx) {
    InstanceHandle mem = extMems[memIdx];
    unsigned &ingressIdx =
        (memIdx % 2 == 0) ? leftIngressIdx : rightIngressIdx;
    unsigned &egressIdx = (memIdx % 2 == 0) ? leftEgressIdx : rightEgressIdx;

    for (unsigned outPort = 0; outPort < kExtMemOutputs; ++outPort) {
      builder.connect(mem, outPort, mesh.ingressPorts[ingressIdx].instance,
                      mesh.ingressPorts[ingressIdx].port);
      ++ingressIdx;
    }
    for (unsigned inPort = 0; inPort < kExtMemDataInputs; ++inPort) {
      builder.connect(mesh.egressPorts[egressIdx].instance,
                      mesh.egressPorts[egressIdx].port, mem, 1 + inPort);
      ++egressIdx;
    }
  }

  std::vector<unsigned> scalarIns = builder.addInputs(
      "scalar",
      std::vector<std::string>(params.scalarInputs, bitsType(params.dataWidth)));
  std::vector<unsigned> scalarOuts = builder.addOutputs(
      "scalar_out", std::vector<std::string>(params.scalarOutputs,
                                             bitsType(params.dataWidth)));

  unsigned scalarIngressIdx = kExtMemOutputs;
  for (unsigned idx = 0; idx < scalarIns.size(); ++idx, ++scalarIngressIdx)
    builder.connectInputToPort(scalarIns[idx],
                               mesh.ingressPorts[scalarIngressIdx]);

  unsigned scalarEgressIdx = 2 * kExtMemDataInputs;
  for (unsigned idx = 0; idx < scalarOuts.size(); ++idx, ++scalarEgressIdx)
    builder.connectPortToOutput(mesh.egressPorts[scalarEgressIdx],
                                scalarOuts[idx]);

  uint64_t spmBytes = static_cast<uint64_t>(params.spmSizeKB) * 1024;
  builder.setSPMCapacity(spmBytes);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

bool isValidDomainTypeId(const std::string &typeId) {
  if (typeId.size() != 2 || typeId[0] != 'D')
    return false;
  char c = typeId[1];
  return c >= '1' && c <= '6';
}

bool isValidSciCompTypeId(const std::string &typeId) {
  for (unsigned i = 0; i < kNumSciCompTypes; ++i) {
    if (kSciCompTypes[i].typeId == typeId)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Parameter Construction
//===----------------------------------------------------------------------===//

DomainTypeParams domainParamsFromTypeId(const std::string &typeId) {
  if (!isValidDomainTypeId(typeId))
    return DomainTypeParams{};
  unsigned idx = static_cast<unsigned>(typeId[1] - '1');
  return kDomainTypes[idx];
}

SciCompTypeParams sciCompParamsFromTypeId(const std::string &typeId) {
  for (unsigned i = 0; i < kNumSciCompTypes; ++i) {
    if (kSciCompTypes[i].typeId == typeId)
      return kSciCompTypes[i];
  }
  return SciCompTypeParams{};
}

//===----------------------------------------------------------------------===//
// ADG Generation
//===----------------------------------------------------------------------===//

std::string generateDomainADG(const DomainTypeParams &params) {
  assert(!params.typeId.empty() && "Domain type ID must not be empty");

  const std::string moduleName = params.typeId + "_core";
  ADGBuilder builder(moduleName);
  buildDomainADGImpl(builder, params, moduleName);
  return builder.exportCoreType(moduleName);
}

static std::map<std::string, std::string>
getSciCompModuleAttrs(const SciCompTypeParams &params) {
  std::map<std::string, std::string> attrs;
  attrs["loom.scicomp_khg_type"] = "\"" + params.typeId + "\"";
  attrs["loom.routing_topology"] = "\"" + params.routingTopology + "\"";
  attrs["loom.decomposable"] = boolAttr(params.decomposable);
  attrs["loom.sub_lane_bits"] = std::to_string(params.subLaneBits);
  attrs["loom.spm_ld_ports"] = std::to_string(params.spmLdPorts);
  attrs["loom.spm_st_ports"] = std::to_string(params.spmStPorts);
  attrs["loom.extmem_ld_ports"] = std::to_string(params.extMemLdPorts);
  attrs["loom.extmem_st_ports"] = std::to_string(params.extMemStPorts);
  attrs["loom.fp_add_units"] = std::to_string(params.fpAddCount);
  attrs["loom.fp_mul_units"] = std::to_string(params.fpMulCount);
  attrs["loom.fp_div_units"] = std::to_string(params.fpDivCount);
  attrs["loom.int_alu_units"] = std::to_string(params.intAluCount);
  attrs["loom.int_mul_units"] = std::to_string(params.intMulCount);
  attrs["loom.has_fma"] = boolAttr(params.hasFMA);
  attrs["loom.has_rsqrt"] = boolAttr(params.hasRSQRT);
  attrs["loom.has_fp_min"] = boolAttr(params.hasFPMin);
  attrs["loom.has_indirect_load"] = boolAttr(params.hasIndirectLoad);
  attrs["loom.has_scatter_store"] = boolAttr(params.hasScatterStore);
  attrs["loom.has_branch"] = boolAttr(params.hasBranch);
  if (params.isTemporal) {
    attrs["loom.operand_buffer_size"] =
        std::to_string(params.operandBufferSize);
  }
  return attrs;
}

std::string generateSciCompADG(const SciCompTypeParams &params) {
  assert(!params.typeId.empty() && "SciComp type ID must not be empty");

  const std::string moduleName = params.typeId + "_core";
  ADGBuilder builder(moduleName);
  buildSciCompADGImpl(builder, params, moduleName);
  std::string mlir = builder.exportCoreType(moduleName);
  return addModuleAttributes(mlir, getSciCompModuleAttrs(params));
}

void exportDomainADG(const DomainTypeParams &params,
                     const std::string &outputPath) {
  assert(!params.typeId.empty() && "Domain type ID must not be empty");

  const std::string moduleName = params.typeId + "_core";
  ADGBuilder builder(moduleName);
  buildDomainADGImpl(builder, params, moduleName);
  builder.exportMLIR(outputPath);
}

void exportSciCompADG(const SciCompTypeParams &params,
                      const std::string &outputPath) {
  assert(!params.typeId.empty() && "SciComp type ID must not be empty");

  llvm::SmallString<128> dir(outputPath);
  llvm::sys::path::remove_filename(dir);
  if (!dir.empty()) {
    std::error_code ec = llvm::sys::fs::create_directories(dir);
    if (ec)
      llvm::report_fatal_error("exportSciCompADG: cannot create output dir");
  }

  std::ofstream os(outputPath, std::ios::out | std::ios::trunc);
  if (!os.is_open())
    llvm::report_fatal_error("exportSciCompADG: cannot open output file");
  std::string mlir = generateSciCompADG(params);
  os << mlir;
  os.flush();
}

//===----------------------------------------------------------------------===//
// Enumeration
//===----------------------------------------------------------------------===//

std::vector<std::string> allDomainTypeIds() {
  std::vector<std::string> ids;
  ids.reserve(kNumDomainTypes);
  for (unsigned i = 0; i < kNumDomainTypes; ++i)
    ids.push_back(kDomainTypes[i].typeId);
  return ids;
}

std::vector<DomainTypeParams> allDomainTypes() {
  return std::vector<DomainTypeParams>(
      kDomainTypes, kDomainTypes + kNumDomainTypes);
}

std::vector<std::string> allSciCompTypeIds() {
  std::vector<std::string> ids;
  ids.reserve(kNumSciCompTypes);
  for (unsigned i = 0; i < kNumSciCompTypes; ++i)
    ids.push_back(kSciCompTypes[i].typeId);
  return ids;
}

std::vector<SciCompTypeParams> allSciCompTypes() {
  return std::vector<SciCompTypeParams>(kSciCompTypes,
                                        kSciCompTypes + kNumSciCompTypes);
}

} // namespace adg
} // namespace loom
