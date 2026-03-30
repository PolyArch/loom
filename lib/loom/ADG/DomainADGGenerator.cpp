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

#include <cassert>
#include <string>
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

//===----------------------------------------------------------------------===//
// Parameter Construction
//===----------------------------------------------------------------------===//

DomainTypeParams domainParamsFromTypeId(const std::string &typeId) {
  if (!isValidDomainTypeId(typeId))
    return DomainTypeParams{};
  unsigned idx = static_cast<unsigned>(typeId[1] - '1');
  return kDomainTypes[idx];
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

void exportDomainADG(const DomainTypeParams &params,
                     const std::string &outputPath) {
  assert(!params.typeId.empty() && "Domain type ID must not be empty");

  const std::string moduleName = params.typeId + "_core";
  ADGBuilder builder(moduleName);
  buildDomainADGImpl(builder, params, moduleName);
  builder.exportMLIR(outputPath);
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

} // namespace adg
} // namespace loom
