//===-- KHGGenerator.cpp - Combinatorial KHG type ADG generator ---*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements parametric generation of all 24 combinatorial KHG types using
// the ADGBuilder API. Follows the same proven pattern as tapestry_adg_gen:
// chess mesh topology with boundary-distributed ext memory and scalar I/O.
//
// SPM presence is reflected via the spm_capacity_bytes attribute on the
// fabric.module, consistent with the existing ADG infrastructure.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/KHGGenerator.h"
#include "loom/ADG/ADGBuilder.h"

#include <cassert>
#include <regex>
#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Naming Convention
//===----------------------------------------------------------------------===//

static char computeMixChar(KHGComputeMix c) {
  switch (c) {
  case KHGComputeMix::INT_HEAVY: return 'I';
  case KHGComputeMix::FP_HEAVY:  return 'F';
  case KHGComputeMix::MIXED:     return 'M';
  }
  return '?';
}

static char peKindChar(KHGPEKind p) {
  switch (p) {
  case KHGPEKind::SPATIAL:  return 'S';
  case KHGPEKind::TEMPORAL: return 'T';
  }
  return '?';
}

static char spmChar(KHGSPMPresence s) {
  switch (s) {
  case KHGSPMPresence::WITH_SPM:    return 'Y';
  case KHGSPMPresence::WITHOUT_SPM: return 'N';
  }
  return '?';
}

static const char *sizeStr(KHGArraySize z) {
  switch (z) {
  case KHGArraySize::SIZE_8:  return "8";
  case KHGArraySize::SIZE_12: return "12";
  }
  return "?";
}

std::string encodeKHGTypeId(KHGComputeMix compute, KHGPEKind pe,
                            KHGSPMPresence spm, KHGArraySize size) {
  std::string id = "C";
  id += computeMixChar(compute);
  id += peKindChar(pe);
  id += spmChar(spm);
  id += sizeStr(size);
  return id;
}

bool decodeKHGTypeId(const std::string &typeId, KHGComputeMix &compute,
                     KHGPEKind &pe, KHGSPMPresence &spm, KHGArraySize &size) {
  static const std::regex pattern("^C([IFM])([ST])([YN])(8|12)$");
  std::smatch match;
  if (!std::regex_match(typeId, match, pattern))
    return false;

  char c = match[1].str()[0];
  switch (c) {
  case 'I': compute = KHGComputeMix::INT_HEAVY; break;
  case 'F': compute = KHGComputeMix::FP_HEAVY;  break;
  case 'M': compute = KHGComputeMix::MIXED;     break;
  default: return false;
  }

  char p = match[2].str()[0];
  switch (p) {
  case 'S': pe = KHGPEKind::SPATIAL;  break;
  case 'T': pe = KHGPEKind::TEMPORAL; break;
  default: return false;
  }

  char s = match[3].str()[0];
  switch (s) {
  case 'Y': spm = KHGSPMPresence::WITH_SPM;    break;
  case 'N': spm = KHGSPMPresence::WITHOUT_SPM;  break;
  default: return false;
  }

  std::string sz = match[4].str();
  if (sz == "8")       size = KHGArraySize::SIZE_8;
  else if (sz == "12") size = KHGArraySize::SIZE_12;
  else return false;

  return true;
}

bool isValidKHGTypeId(const std::string &typeId) {
  KHGComputeMix c;
  KHGPEKind p;
  KHGSPMPresence s;
  KHGArraySize z;
  return decodeKHGTypeId(typeId, c, p, s, z);
}

//===----------------------------------------------------------------------===//
// Parameter Construction
//===----------------------------------------------------------------------===//

KHGTypeParams makeKHGParams(KHGComputeMix compute, KHGPEKind pe,
                            KHGSPMPresence spm, KHGArraySize size) {
  KHGTypeParams params;
  params.typeId = encodeKHGTypeId(compute, pe, spm, size);
  params.computeMix = compute;
  params.peKind = pe;
  params.spmPresence = spm;
  params.arraySize = size;

  // Array dimensions
  switch (size) {
  case KHGArraySize::SIZE_8:  params.arrayRows = 8;  params.arrayCols = 8;  break;
  case KHGArraySize::SIZE_12: params.arrayRows = 12; params.arrayCols = 12; break;
  }

  // FU counts from compute mix
  switch (compute) {
  case KHGComputeMix::INT_HEAVY:
    params.fuAluCount = 6; params.fuMulCount = 4; params.fuFpCount = 1;
    break;
  case KHGComputeMix::FP_HEAVY:
    params.fuAluCount = 2; params.fuMulCount = 2; params.fuFpCount = 6;
    break;
  case KHGComputeMix::MIXED:
    params.fuAluCount = 4; params.fuMulCount = 3; params.fuFpCount = 3;
    break;
  }

  // SPM parameters
  switch (spm) {
  case KHGSPMPresence::WITH_SPM:
    params.spmSizeKB = 16; params.spmLdPorts = 2; params.spmStPorts = 2;
    break;
  case KHGSPMPresence::WITHOUT_SPM:
    params.spmSizeKB = 0; params.spmLdPorts = 0; params.spmStPorts = 0;
    break;
  }

  // Temporal PE parameters
  switch (pe) {
  case KHGPEKind::TEMPORAL:
    params.instructionSlots = 8; params.numRegisters = 8;
    break;
  case KHGPEKind::SPATIAL:
    params.instructionSlots = 0; params.numRegisters = 0;
    break;
  }

  params.dataWidth = 32;
  return params;
}

KHGTypeParams paramsFromTypeId(const std::string &typeId) {
  KHGComputeMix c;
  KHGPEKind p;
  KHGSPMPresence s;
  KHGArraySize z;
  if (!decodeKHGTypeId(typeId, c, p, s, z))
    return KHGTypeParams{};
  return makeKHGParams(c, p, s, z);
}

//===----------------------------------------------------------------------===//
// FU Definition Helpers
//===----------------------------------------------------------------------===//

namespace {

constexpr unsigned kDataWidth = 32;
constexpr unsigned kPEInputs = 4;
constexpr unsigned kPEOutputs = 4;
constexpr unsigned kExtMemOutputs = 3;     // ld_data, ld_done, st_done
constexpr unsigned kExtMemDataInputs = 3;  // ld_addr, st_addr, st_data
constexpr unsigned kNumExtMems = 2;
constexpr unsigned kNumScalarInputs = 4;
constexpr unsigned kNumScalarOutputs = 2;

static std::string bitsType(unsigned width = kDataWidth) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

/// Define the baseline dataflow FUs needed for kernel mapping.
/// Mirrors the baseline set from tapestry_adg_gen.
static std::vector<FUHandle>
defineBaselineFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;

  // Constants
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_index", "index", "0 : index"));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_f32", "f32", "0.000000e+00 : f32"));

  // Index casts
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_index_to_i32", "index", "i32"));
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_i32_to_index", "i32", "index"));

  // Dataflow control FUs
  fus.push_back(builder.defineStreamFU(prefix + "_stream"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_i32", "i32"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_none", "none"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_index", "index"));
  fus.push_back(builder.defineJoinFU(prefix + "_join", 4));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i32", "i32"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_index", "index"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_f32", "f32"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i1", "i1"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_i32", "i32"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_none", "none"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_f32", "f32"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_i32", "i32"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_none", "none"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_f32", "f32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i32", "i32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_index", "index"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_f32", "f32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_none", "none"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i1", "i1"));

  // Memory access FUs
  fus.push_back(builder.defineLoadFU(prefix + "_load", "index", "i32"));
  fus.push_back(builder.defineStoreFU(prefix + "_store", "index", "i32"));

  // Comparison and selection
  fus.push_back(builder.defineSelectFU(prefix + "_select_i32", "i32"));
  fus.push_back(builder.defineSelectFU(prefix + "_select_index", "index"));
  fus.push_back(builder.defineCmpiFU(prefix + "_cmpi_i32", "i32", "slt"));

  // Index-typed arithmetic
  fus.push_back(builder.defineBinaryFU(
      prefix + "_addi_index", "arith.addi", "index", "index"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_muli_index", "arith.muli", "index", "index"));

  return fus;
}

/// Define ALU FUs (integer arithmetic + bitwise): count instances.
static void defineALUFUs(ADGBuilder &builder, const std::string &prefix,
                         unsigned count, std::vector<FUHandle> &fus) {
  for (unsigned i = 0; i < count; ++i) {
    std::string suffix = "_alu" + std::to_string(i);
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_addi", "arith.addi", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_subi", "arith.subi", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_andi", "arith.andi", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_ori", "arith.ori", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_xori", "arith.xori", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_shli", "arith.shli", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_shrsi", "arith.shrsi", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_shrui", "arith.shrui", "i32", "i32"));
  }
}

/// Define MUL FUs (integer multiply + divide): count instances.
static void defineMulFUs(ADGBuilder &builder, const std::string &prefix,
                         unsigned count, std::vector<FUHandle> &fus) {
  for (unsigned i = 0; i < count; ++i) {
    std::string suffix = "_mul" + std::to_string(i);
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_muli", "arith.muli", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_divsi", "arith.divsi", "i32", "i32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_remsi", "arith.remsi", "i32", "i32"));
  }
}

/// Define FP FUs (floating-point arithmetic + conversion): count instances.
static void defineFPFUs(ADGBuilder &builder, const std::string &prefix,
                        unsigned count, std::vector<FUHandle> &fus) {
  for (unsigned i = 0; i < count; ++i) {
    std::string suffix = "_fp" + std::to_string(i);
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_addf", "arith.addf", "f32", "f32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_subf", "arith.subf", "f32", "f32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_mulf", "arith.mulf", "f32", "f32"));
    fus.push_back(builder.defineBinaryFU(
        prefix + suffix + "_divf", "arith.divf", "f32", "f32"));
    fus.push_back(builder.defineCmpfFU(
        prefix + suffix + "_cmpf", "f32", "olt"));
    fus.push_back(builder.defineSelectFU(
        prefix + suffix + "_select_f32", "f32"));
    fus.push_back(builder.defineUnaryFU(
        prefix + suffix + "_sitofp", "arith.sitofp", "i32", "f32"));
    fus.push_back(builder.defineUnaryFU(
        prefix + suffix + "_fptosi", "arith.fptosi", "f32", "i32"));
    fus.push_back(builder.defineUnaryFU(
        prefix + suffix + "_negf", "arith.negf", "f32", "f32"));
  }
}

/// Core ADG build logic shared by generateKHGADG and exportKHGADG.
/// Populates the builder with all components and wiring.
/// Follows the same proven pattern as tapestry_adg_gen: chess mesh with
/// boundary-distributed ext memory and scalar I/O.
static void buildKHGADGImpl(ADGBuilder &builder, const KHGTypeParams &params,
                            const std::string &moduleName) {
  // Build FU list: baseline + compute-mix specific
  std::vector<FUHandle> fus = defineBaselineFUs(builder, moduleName);
  defineALUFUs(builder, moduleName, params.fuAluCount, fus);
  defineMulFUs(builder, moduleName, params.fuMulCount, fus);
  defineFPFUs(builder, moduleName, params.fuFpCount, fus);

  // Define PE template. Both spatial and temporal PEs use !fabric.bits<N>
  // port types for compatibility with the chess mesh topology builder, which
  // infers switch port widths from PE port types.
  PEHandle pe;
  std::string portType = bitsType();
  std::vector<std::string> peInTypes(kPEInputs, portType);
  std::vector<std::string> peOutTypes(kPEOutputs, portType);

  if (params.isTemporal()) {
    // reg_fifo_depth must be >= 1 when num_register > 0
    unsigned regFifoDepth = (params.numRegisters > 0) ? 1 : 0;
    pe = builder.defineTemporalPE(
        moduleName + "_tpe", peInTypes, peOutTypes, fus,
        params.numRegisters, params.instructionSlots,
        regFifoDepth, /*enableShareOperandBuffer=*/false,
        std::nullopt);
  } else {
    pe = builder.defineSpatialPE(
        moduleName + "_spe", kPEInputs, kPEOutputs, kDataWidth, fus);
  }

  // Compute boundary port layout (same pattern as tapestry_adg_gen)
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

  // Build chess mesh topology (all KHG types use chess)
  auto mesh = builder.buildChessMesh(
      params.arrayRows, params.arrayCols,
      [&](unsigned, unsigned) { return pe; },
      meshOpts);

  // Define and instantiate external memory
  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_extmem";
  extMemSpec.ldPorts = 1;
  extMemSpec.stPorts = 1;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 1;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMems = builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", kNumExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  // Wire ext memory to boundary ports (round-robin left/right, same as
  // tapestry_adg_gen)
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
      "scalar", std::vector<std::string>(kNumScalarInputs, bitsType()));
  std::vector<unsigned> scalarOuts = builder.addOutputs(
      "scalar_out", std::vector<std::string>(kNumScalarOutputs, bitsType()));

  unsigned scalarIngressIdx = leftIngressMems * kExtMemOutputs;
  for (unsigned idx = 0; idx < scalarIns.size(); ++idx, ++scalarIngressIdx)
    builder.connectInputToPort(scalarIns[idx],
                               mesh.ingressPorts[scalarIngressIdx]);

  unsigned scalarEgressIdx = kNumExtMems * kExtMemDataInputs;
  for (unsigned idx = 0; idx < scalarOuts.size(); ++idx, ++scalarEgressIdx)
    builder.connectPortToOutput(mesh.egressPorts[scalarEgressIdx],
                                scalarOuts[idx]);

  // Set SPM capacity attribute (the standard way to declare SPM presence)
  uint64_t spmBytes = static_cast<uint64_t>(params.spmSizeKB) * 1024;
  builder.setSPMCapacity(spmBytes);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ADG Generation
//===----------------------------------------------------------------------===//

std::string generateKHGADG(const KHGTypeParams &params) {
  assert(!params.typeId.empty() && "KHG type ID must not be empty");

  const std::string moduleName = params.typeId + "_core";
  ADGBuilder builder(moduleName);
  buildKHGADGImpl(builder, params, moduleName);
  return builder.exportCoreType(moduleName);
}

void exportKHGADG(const KHGTypeParams &params, const std::string &outputPath) {
  assert(!params.typeId.empty() && "KHG type ID must not be empty");

  const std::string moduleName = params.typeId + "_core";
  ADGBuilder builder(moduleName);
  buildKHGADGImpl(builder, params, moduleName);
  builder.exportMLIR(outputPath);
}

//===----------------------------------------------------------------------===//
// Enumeration
//===----------------------------------------------------------------------===//

std::vector<std::string> allKHGTypeIds() {
  std::vector<std::string> ids;
  ids.reserve(24);
  KHGComputeMix computes[] = {
      KHGComputeMix::INT_HEAVY, KHGComputeMix::FP_HEAVY, KHGComputeMix::MIXED};
  KHGPEKind pes[] = {KHGPEKind::SPATIAL, KHGPEKind::TEMPORAL};
  KHGSPMPresence spms[] = {KHGSPMPresence::WITH_SPM, KHGSPMPresence::WITHOUT_SPM};
  KHGArraySize sizes[] = {KHGArraySize::SIZE_8, KHGArraySize::SIZE_12};

  for (auto c : computes)
    for (auto p : pes)
      for (auto s : spms)
        for (auto z : sizes)
          ids.push_back(encodeKHGTypeId(c, p, s, z));

  return ids;
}

std::vector<KHGTypeParams> allKHGTypes() {
  auto ids = allKHGTypeIds();
  std::vector<KHGTypeParams> params;
  params.reserve(ids.size());
  for (const auto &id : ids)
    params.push_back(paramsFromTypeId(id));
  return params;
}

} // namespace adg
} // namespace loom
