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

#include <algorithm>
#include <cassert>
#include <cmath>
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

  unsigned total = params.totalPEs();

  // PE category counts by compute mix. Percentages:
  //   INT_HEAVY: 38% arithInt, 6% arithFp, 25% control, 19% memory, rest stream
  //   FP_HEAVY:  12% arithInt, 31% arithFp, 25% control, 19% memory, rest stream
  //   MIXED:     25% arithInt, 19% arithFp, 25% control, 19% memory, rest stream
  auto pct = [&](unsigned percent) -> unsigned {
    return static_cast<unsigned>(
        std::llround(static_cast<double>(total) *
                     static_cast<double>(percent) / 100.0));
  };
  switch (compute) {
  case KHGComputeMix::INT_HEAVY:
    params.peArithInt = pct(38);
    params.peArithFp  = pct(6);
    break;
  case KHGComputeMix::FP_HEAVY:
    params.peArithInt = pct(12);
    params.peArithFp  = pct(31);
    break;
  case KHGComputeMix::MIXED:
    params.peArithInt = pct(25);
    params.peArithFp  = pct(19);
    break;
  }
  params.peControl = pct(25);
  params.peMemory  = pct(19);
  // Stream gets the remainder
  unsigned assigned = params.peArithInt + params.peArithFp +
                      params.peControl + params.peMemory;
  params.peStream = (assigned < total) ? (total - assigned) : 0;

  // Spatial fraction
  switch (pe) {
  // Temporal PE mixing disabled pending temporal PE mapper/simulator
  // validation. Restore to 0.80/0.25 after temporal PE tests pass.
  case KHGPEKind::SPATIAL:  params.spatialFraction = 1.0f; break;
  case KHGPEKind::TEMPORAL: params.spatialFraction = 1.0f; break;
  }

  // SPM parameters
  switch (spm) {
  case KHGSPMPresence::WITH_SPM:
    params.spmCount = 8;
    params.spmSizePerUnit = 4096;
    break;
  case KHGSPMPresence::WITHOUT_SPM:
    params.spmCount = 0;
    params.spmSizePerUnit = 0;
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
// Specialized PE FU Builders
//===----------------------------------------------------------------------===//

namespace {

constexpr unsigned kDataWidth = 32;
constexpr unsigned kPEInputs = 4;
constexpr unsigned kPEOutputs = 4;
constexpr unsigned kNumExtMems = 8;
constexpr unsigned kExtMemsPerEdge = 2;
// Module I/O port counts depend on array size.
// 8x8: 10 inputs (5 top + 5 left), 10 outputs (5 right + 5 bottom)
// 12x12: 18 inputs (9 top + 9 left), 18 outputs (9 right + 9 bottom)
static unsigned computeModuleIOPerEdge(unsigned arrayDim) {
  unsigned switchCount = arrayDim + 1; // chess mesh: N+1 switches per edge
  if (switchCount <= 6)
    return switchCount - 1; // skip corner switches
  return switchCount - 4;   // skip 2 corner switches on each end
}

// Tagged bridge port counts per extmem/spm (ldPorts=2, stPorts=2):
//   Ingress (mem -> mesh): 2 ld_data + 2 ld_done + 2 st_done = 6
//   Egress  (mesh -> mem): 2 ld_addr + 2 st_addr + 2 st_data = 6
constexpr unsigned kBridgeIngressPorts = 6;
constexpr unsigned kBridgeEgressPorts = 6;

static std::string bitsType(unsigned width = kDataWidth) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

static std::string taggedBitsType(unsigned tagWidth,
                                  unsigned width = kDataWidth * 2) {
  return "!fabric.tagged<" + bitsType(width) + ", i" +
         std::to_string(tagWidth) + ">";
}

//===----------------------------------------------------------------------===//
// Tagged bridge templates for multi-port memories
//===----------------------------------------------------------------------===//

/// Bridge switch templates for a memory with ldPorts=2, stPorts=2.
/// Muxes merge two tagged streams into one; demuxes split one tagged stream
/// into two. Defined once and instantiated per memory.
struct TaggedMemBridgeTemplates {
  SWHandle ldAddrMux;   // 2 tagged in -> 1 tagged out (spatial)
  SWHandle stAddrMux;   // 2 tagged in -> 1 tagged out (spatial)
  SWHandle stDataMux;   // 2 tagged in -> 1 tagged out (spatial)
  SWHandle ldDataDemux; // 1 tagged in -> 2 tagged out (temporal)
  SWHandle ldDoneDemux; // 1 tagged in -> 2 tagged out (temporal)
  SWHandle stDoneDemux; // 1 tagged in -> 2 tagged out (temporal)
};

static TaggedMemBridgeTemplates
defineTaggedMemBridgeTemplates(ADGBuilder &builder,
                               const std::string &prefix) {
  const std::string taggedTy = taggedBitsType(1);
  TaggedMemBridgeTemplates t;
  t.ldAddrMux = builder.defineSpatialSW(
      prefix + "_ld_addr_mux", {taggedTy, taggedTy}, {taggedTy},
      {{true, true}});
  t.stAddrMux = builder.defineSpatialSW(
      prefix + "_st_addr_mux", {taggedTy, taggedTy}, {taggedTy},
      {{true, true}});
  t.stDataMux = builder.defineSpatialSW(
      prefix + "_st_data_mux", {taggedTy, taggedTy}, {taggedTy},
      {{true, true}});
  t.ldDataDemux = builder.defineTemporalSW(
      prefix + "_ld_data_demux", {taggedTy}, {taggedTy, taggedTy},
      {{true}, {true}}, 2);
  t.ldDoneDemux = builder.defineTemporalSW(
      prefix + "_ld_done_demux", {taggedTy}, {taggedTy, taggedTy},
      {{true}, {true}}, 2);
  t.stDoneDemux = builder.defineTemporalSW(
      prefix + "_st_done_demux", {taggedTy}, {taggedTy, taggedTy},
      {{true}, {true}}, 2);
  return t;
}

/// Wire a multi-port memory (ldPorts=2, stPorts=2) through tagged bridges to
/// 6 ingress and 6 egress boundary ports.
///
/// Egress (mesh -> mem): 6 boundary ports
///   [0] -> add_tag(ld_addr, 0) -> ld_addr_mux in 0
///   [1] -> add_tag(ld_addr, 1) -> ld_addr_mux in 1
///   [2] -> add_tag(st_addr, 0) -> st_addr_mux in 0
///   [3] -> add_tag(st_addr, 1) -> st_addr_mux in 1
///   [4] -> add_tag(st_data, 0) -> st_data_mux in 0
///   [5] -> add_tag(st_data, 1) -> st_data_mux in 1
///   ld_addr_mux out 0 -> mem in 1 (ld_addr)
///   st_addr_mux out 0 -> mem in 2 (st_addr)
///   st_data_mux out 0 -> mem in 3 (st_data)
///
/// Ingress (mem -> mesh): 6 boundary ports
///   mem out 0 (ld_data)  -> ld_data_demux -> del_tag -> [0],[1]
///   mem out 1 (ld_done)  -> ld_done_demux -> del_tag -> [2],[3]
///   mem out 2 (st_done)  -> st_done_demux -> del_tag -> [4],[5]
/// inputPortBase: first data input port index on the memory instance.
///   For extmem, port 0 is memref, so inputPortBase = 1.
///   For on-chip memory, inputPortBase = 0.
static void wireTaggedMemBridge(ADGBuilder &builder,
                                InstanceHandle mem,
                                const std::vector<PortRef> &ingressPorts,
                                unsigned &ingressIdx,
                                const std::vector<PortRef> &egressPorts,
                                unsigned &egressIdx,
                                const TaggedMemBridgeTemplates &templates,
                                unsigned memIdx,
                                const std::string &prefix,
                                unsigned inputPortBase = 1) {
  const std::string taggedTy = taggedBitsType(1);
  const std::string meshTy = bitsType();
  const std::string suffix = "_" + std::to_string(memIdx);

  // Create add_tag instances for egress (mesh -> mem) path
  auto ldAddrTags = builder.createAddTagBank(meshTy, taggedTy, {0, 1});
  auto stAddrTags = builder.createAddTagBank(meshTy, taggedTy, {0, 1});
  auto stDataTags = builder.createAddTagBank(meshTy, taggedTy, {0, 1});

  // Create del_tag instances for ingress (mem -> mesh) path
  auto ldDataDrops = builder.createDelTagBank(taggedTy, meshTy, 2);
  auto ldDoneDrops = builder.createDelTagBank(taggedTy, meshTy, 2);
  auto stDoneDrops = builder.createDelTagBank(taggedTy, meshTy, 2);

  // Instantiate mux/demux switches
  auto ldAddrMux = builder.instantiateSW(
      templates.ldAddrMux, prefix + "_ld_addr_mux" + suffix);
  auto stAddrMux = builder.instantiateSW(
      templates.stAddrMux, prefix + "_st_addr_mux" + suffix);
  auto stDataMux = builder.instantiateSW(
      templates.stDataMux, prefix + "_st_data_mux" + suffix);
  auto ldDataDemux = builder.instantiateSW(
      templates.ldDataDemux, prefix + "_ld_data_demux" + suffix);
  auto ldDoneDemux = builder.instantiateSW(
      templates.ldDoneDemux, prefix + "_ld_done_demux" + suffix);
  auto stDoneDemux = builder.instantiateSW(
      templates.stDoneDemux, prefix + "_st_done_demux" + suffix);

  // Egress wiring: mesh boundary -> add_tag -> mux -> mem input
  // ld_addr: egress[0..1] -> add_tag(0,1) -> ld_addr_mux -> mem in 1
  builder.connect(egressPorts[egressIdx].instance,
                  egressPorts[egressIdx].port, ldAddrTags[0], 0);
  ++egressIdx;
  builder.connect(egressPorts[egressIdx].instance,
                  egressPorts[egressIdx].port, ldAddrTags[1], 0);
  ++egressIdx;
  // st_addr: egress[2..3] -> add_tag(0,1) -> st_addr_mux -> mem in 2
  builder.connect(egressPorts[egressIdx].instance,
                  egressPorts[egressIdx].port, stAddrTags[0], 0);
  ++egressIdx;
  builder.connect(egressPorts[egressIdx].instance,
                  egressPorts[egressIdx].port, stAddrTags[1], 0);
  ++egressIdx;
  // st_data: egress[4..5] -> add_tag(0,1) -> st_data_mux -> mem in 3
  builder.connect(egressPorts[egressIdx].instance,
                  egressPorts[egressIdx].port, stDataTags[0], 0);
  ++egressIdx;
  builder.connect(egressPorts[egressIdx].instance,
                  egressPorts[egressIdx].port, stDataTags[1], 0);
  ++egressIdx;

  // add_tag -> mux
  builder.connect(ldAddrTags[0], 0, ldAddrMux, 0);
  builder.connect(ldAddrTags[1], 0, ldAddrMux, 1);
  builder.connect(stAddrTags[0], 0, stAddrMux, 0);
  builder.connect(stAddrTags[1], 0, stAddrMux, 1);
  builder.connect(stDataTags[0], 0, stDataMux, 0);
  builder.connect(stDataTags[1], 0, stDataMux, 1);

  // mux -> mem data inputs (ld_addr, st_addr, st_data)
  builder.connect(ldAddrMux, 0, mem, inputPortBase + 0);
  builder.connect(stAddrMux, 0, mem, inputPortBase + 1);
  builder.connect(stDataMux, 0, mem, inputPortBase + 2);

  // Ingress wiring: mem output -> demux -> del_tag -> mesh boundary
  // ld_data: mem out 0 -> ld_data_demux -> del_tag -> ingress[0..1]
  builder.connect(mem, 0, ldDataDemux, 0);
  builder.connect(ldDataDemux, 0, ldDataDrops[0], 0);
  builder.connect(ldDataDemux, 1, ldDataDrops[1], 0);
  builder.connect(ldDataDrops[0], 0,
                  ingressPorts[ingressIdx].instance,
                  ingressPorts[ingressIdx].port);
  ++ingressIdx;
  builder.connect(ldDataDrops[1], 0,
                  ingressPorts[ingressIdx].instance,
                  ingressPorts[ingressIdx].port);
  ++ingressIdx;

  // ld_done: mem out 1 -> ld_done_demux -> del_tag -> ingress[2..3]
  builder.connect(mem, 1, ldDoneDemux, 0);
  builder.connect(ldDoneDemux, 0, ldDoneDrops[0], 0);
  builder.connect(ldDoneDemux, 1, ldDoneDrops[1], 0);
  builder.connect(ldDoneDrops[0], 0,
                  ingressPorts[ingressIdx].instance,
                  ingressPorts[ingressIdx].port);
  ++ingressIdx;
  builder.connect(ldDoneDrops[1], 0,
                  ingressPorts[ingressIdx].instance,
                  ingressPorts[ingressIdx].port);
  ++ingressIdx;

  // st_done: mem out 2 -> st_done_demux -> del_tag -> ingress[4..5]
  builder.connect(mem, 2, stDoneDemux, 0);
  builder.connect(stDoneDemux, 0, stDoneDrops[0], 0);
  builder.connect(stDoneDemux, 1, stDoneDrops[1], 0);
  builder.connect(stDoneDrops[0], 0,
                  ingressPorts[ingressIdx].instance,
                  ingressPorts[ingressIdx].port);
  ++ingressIdx;
  builder.connect(stDoneDrops[1], 0,
                  ingressPorts[ingressIdx].instance,
                  ingressPorts[ingressIdx].port);
  ++ingressIdx;
}

/// Distribute totalPorts across sideSwitchCount positions as evenly as
/// possible, spreading from center outward.
static std::vector<unsigned> buildSpreadSideCounts(unsigned totalPorts,
                                                   unsigned sideSwitchCount) {
  std::vector<unsigned> counts(sideSwitchCount, 0);
  if (totalPorts == 0 || sideSwitchCount == 0)
    return counts;

  if (totalPorts <= sideSwitchCount) {
    if (totalPorts == 1) {
      counts[sideSwitchCount / 2] = 1;
      return counts;
    }
    for (unsigned idx = 0; idx < totalPorts; ++idx) {
      unsigned sideIdx = static_cast<unsigned>(
          std::llround(static_cast<double>(idx) *
                       static_cast<double>(sideSwitchCount - 1) /
                       static_cast<double>(totalPorts - 1)));
      counts[std::min(sideIdx, sideSwitchCount - 1)] += 1;
    }
    return counts;
  }

  std::fill(counts.begin(), counts.end(), 1);
  unsigned remaining = totalPorts - sideSwitchCount;
  int center = static_cast<int>(sideSwitchCount / 2);
  for (unsigned extra = 0; extra < remaining; ++extra) {
    int delta = static_cast<int>((extra + 1) / 2);
    int sideIdx = center;
    if ((extra % 2) == 0) {
      sideIdx = center - delta;
    } else {
      sideIdx = center + delta;
    }
    sideIdx = std::max(0, std::min(static_cast<int>(sideSwitchCount) - 1,
                                   sideIdx));
    counts[static_cast<unsigned>(sideIdx)] += 1;
  }
  return counts;
}

//===----------------------------------------------------------------------===//
// Per-category FU definitions
//===----------------------------------------------------------------------===//

/// ArithInt: integer arithmetic, comparison, select, index ops.
static std::vector<FUHandle>
defineArithIntFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;
  // Integer arithmetic (i32)
  fus.push_back(builder.defineBinaryFU(prefix + "_addi", "arith.addi", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_subi", "arith.subi", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_muli", "arith.muli", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_divi", "arith.divsi", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_remsi", "arith.remsi", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_shli", "arith.shli", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_shrsi", "arith.shrsi", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_shrui", "arith.shrui", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_andi", "arith.andi", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_ori", "arith.ori", "i32", "i32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_xori", "arith.xori", "i32", "i32"));
  // Comparison and selection
  fus.push_back(builder.defineCmpiFU(prefix + "_cmpi", "i32", "slt"));
  fus.push_back(builder.defineSelectFU(prefix + "_select_i32", "i32"));
  // Constants and invariants
  fus.push_back(builder.defineConstantFU(prefix + "_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineConstantFU(prefix + "_const_index", "index", "0 : index"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i32", "i32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_index", "index"));
  // Index casts
  fus.push_back(builder.defineIndexCastFU(prefix + "_index_to_i32", "index", "i32"));
  fus.push_back(builder.defineIndexCastFU(prefix + "_i32_to_index", "i32", "index"));
  // Index-typed arithmetic
  fus.push_back(builder.defineBinaryFU(prefix + "_addi_index", "arith.addi", "index", "index"));
  fus.push_back(builder.defineBinaryFU(prefix + "_muli_index", "arith.muli", "index", "index"));
  return fus;
}

/// ArithFp: floating-point arithmetic, comparison, conversion.
static std::vector<FUHandle>
defineArithFpFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;
  // FP arithmetic (f32)
  fus.push_back(builder.defineBinaryFU(prefix + "_addf", "arith.addf", "f32", "f32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_subf", "arith.subf", "f32", "f32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_mulf", "arith.mulf", "f32", "f32"));
  fus.push_back(builder.defineBinaryFU(prefix + "_divf", "arith.divf", "f32", "f32"));
  fus.push_back(builder.defineUnaryFU(prefix + "_negf", "arith.negf", "f32", "f32"));
  // FP comparison and selection
  fus.push_back(builder.defineCmpfFU(prefix + "_cmpf", "f32", "olt"));
  fus.push_back(builder.defineSelectFU(prefix + "_select_f32", "f32"));
  // Conversion
  fus.push_back(builder.defineUnaryFU(prefix + "_sitofp", "arith.sitofp", "i32", "f32"));
  fus.push_back(builder.defineUnaryFU(prefix + "_fptosi", "arith.fptosi", "f32", "i32"));
  // Math
  fus.push_back(builder.defineUnaryFU(prefix + "_absf", "math.absf", "f32", "f32"));
  fus.push_back(builder.defineUnaryFU(prefix + "_sqrtf", "math.sqrt", "f32", "f32"));
  // Constants and invariants
  fus.push_back(builder.defineConstantFU(prefix + "_const_f32", "f32", "0.000000e+00 : f32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_f32", "f32"));
  return fus;
}

/// Control: mux, cond_br, join, select.
static std::vector<FUHandle>
defineControlFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;
  // Mux (multiple types)
  fus.push_back(builder.defineMuxFU(prefix + "_mux_i32", "i32"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_index", "index"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_f32", "f32"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_none", "none"));
  // Conditional branch (multiple types)
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_i32", "i32"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_index", "index"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_f32", "f32"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_none", "none"));
  // Join
  fus.push_back(builder.defineJoinFU(prefix + "_join", 4));
  // Select
  fus.push_back(builder.defineSelectFU(prefix + "_select_i32", "i32"));
  fus.push_back(builder.defineSelectFU(prefix + "_select_index", "index"));
  // Constant
  fus.push_back(builder.defineConstantFU(prefix + "_const_i1", "i1", "false"));
  return fus;
}

/// Memory: load, store, constants, invariants, index_cast.
static std::vector<FUHandle>
defineMemoryFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;
  // Load/store (i32 and f32)
  fus.push_back(builder.defineLoadFU(prefix + "_load_i32", "index", "i32"));
  fus.push_back(builder.defineStoreFU(prefix + "_store_i32", "index", "i32"));
  fus.push_back(builder.defineLoadFU(prefix + "_load_f32", "index", "f32"));
  fus.push_back(builder.defineStoreFU(prefix + "_store_f32", "index", "f32"));
  // Constants (all types)
  fus.push_back(builder.defineConstantFU(prefix + "_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineConstantFU(prefix + "_const_index", "index", "0 : index"));
  fus.push_back(builder.defineConstantFU(prefix + "_const_f32", "f32", "0.000000e+00 : f32"));
  fus.push_back(builder.defineConstantFU(prefix + "_const_i1", "i1", "false"));
  // Invariants (all types)
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i32", "i32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_index", "index"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_f32", "f32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_none", "none"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i1", "i1"));
  // Index casts
  fus.push_back(builder.defineIndexCastFU(prefix + "_index_to_i32", "index", "i32"));
  fus.push_back(builder.defineIndexCastFU(prefix + "_i32_to_index", "i32", "index"));
  return fus;
}

/// Stream: stream, gate, carry, index_cast, constants.
static std::vector<FUHandle>
defineStreamFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;
  // Stream
  fus.push_back(builder.defineStreamFU(prefix + "_stream"));
  // Gate (multiple types)
  fus.push_back(builder.defineGateFU(prefix + "_gate_index", "index"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i32", "i32"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_f32", "f32"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i1", "i1"));
  // Carry (multiple types)
  fus.push_back(builder.defineCarryFU(prefix + "_carry_index", "index"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_i32", "i32"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_f32", "f32"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_none", "none"));
  // Index casts
  fus.push_back(builder.defineIndexCastFU(prefix + "_index_to_i32", "index", "i32"));
  fus.push_back(builder.defineIndexCastFU(prefix + "_i32_to_index", "i32", "index"));
  // Constants and invariants
  fus.push_back(builder.defineConstantFU(prefix + "_const_index", "index", "0 : index"));
  fus.push_back(builder.defineConstantFU(prefix + "_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_index", "index"));
  return fus;
}

//===----------------------------------------------------------------------===//
// PE Category and PE Set
//===----------------------------------------------------------------------===//

enum class PECategory {
  ArithInt,
  ArithFp,
  Control,
  Memory,
  Stream,
};

/// Holds 10 PE templates: 5 categories x {spatial, temporal}.
struct PESet {
  PEHandle spatialArithInt;
  PEHandle spatialArithFp;
  PEHandle spatialControl;
  PEHandle spatialMemory;
  PEHandle spatialStream;
  PEHandle temporalArithInt;
  PEHandle temporalArithFp;
  PEHandle temporalControl;
  PEHandle temporalMemory;
  PEHandle temporalStream;

  PEHandle get(PECategory cat, bool spatial) const {
    switch (cat) {
    case PECategory::ArithInt: return spatial ? spatialArithInt : temporalArithInt;
    case PECategory::ArithFp:  return spatial ? spatialArithFp  : temporalArithFp;
    case PECategory::Control:  return spatial ? spatialControl  : temporalControl;
    case PECategory::Memory:   return spatial ? spatialMemory   : temporalMemory;
    case PECategory::Stream:   return spatial ? spatialStream   : temporalStream;
    }
    return spatialArithInt; // unreachable
  }
};

/// Create all 10 PE templates (5 categories x spatial/temporal).
static PESet definePESet(ADGBuilder &builder, const std::string &prefix) {
  std::string portType = bitsType();
  std::vector<std::string> inTypes(kPEInputs, portType);
  std::vector<std::string> outTypes(kPEOutputs, portType);

  constexpr unsigned kNumReg = 8;
  constexpr unsigned kNumInst = 8;
  constexpr unsigned kRegFifoDepth = 1;

  auto arithIntFUs = defineArithIntFUs(builder, prefix + "_arith_int");
  auto arithFpFUs  = defineArithFpFUs(builder, prefix + "_arith_fp");
  auto controlFUs  = defineControlFUs(builder, prefix + "_control");
  auto memoryFUs   = defineMemoryFUs(builder, prefix + "_memory");
  auto streamFUs   = defineStreamFUs(builder, prefix + "_stream");

  PESet ps;
  // Spatial PEs
  ps.spatialArithInt = builder.defineSpatialPE(
      prefix + "_spe_arith_int", kPEInputs, kPEOutputs, kDataWidth, arithIntFUs);
  ps.spatialArithFp = builder.defineSpatialPE(
      prefix + "_spe_arith_fp", kPEInputs, kPEOutputs, kDataWidth, arithFpFUs);
  ps.spatialControl = builder.defineSpatialPE(
      prefix + "_spe_control", kPEInputs, kPEOutputs, kDataWidth, controlFUs);
  ps.spatialMemory = builder.defineSpatialPE(
      prefix + "_spe_memory", kPEInputs, kPEOutputs, kDataWidth, memoryFUs);
  ps.spatialStream = builder.defineSpatialPE(
      prefix + "_spe_stream", kPEInputs, kPEOutputs, kDataWidth, streamFUs);

  // Temporal PEs
  ps.temporalArithInt = builder.defineTemporalPE(
      prefix + "_tpe_arith_int", inTypes, outTypes, arithIntFUs,
      kNumReg, kNumInst, kRegFifoDepth);
  ps.temporalArithFp = builder.defineTemporalPE(
      prefix + "_tpe_arith_fp", inTypes, outTypes, arithFpFUs,
      kNumReg, kNumInst, kRegFifoDepth);
  ps.temporalControl = builder.defineTemporalPE(
      prefix + "_tpe_control", inTypes, outTypes, controlFUs,
      kNumReg, kNumInst, kRegFifoDepth);
  ps.temporalMemory = builder.defineTemporalPE(
      prefix + "_tpe_memory", inTypes, outTypes, memoryFUs,
      kNumReg, kNumInst, kRegFifoDepth);
  ps.temporalStream = builder.defineTemporalPE(
      prefix + "_tpe_stream", inTypes, outTypes, streamFUs,
      kNumReg, kNumInst, kRegFifoDepth);

  return ps;
}

/// Build a flat vector of PEHandle assignments for all PEs in the mesh.
/// Distributes categories uniformly (not clustered) and mixes spatial/temporal.
static std::vector<PEHandle>
buildPEAssignment(const KHGTypeParams &params, const PESet &peSet) {
  unsigned total = params.totalPEs();

  // Build category sequence: repeat categories in round-robin order,
  // distributing them uniformly across the mesh.
  struct CatEntry { PECategory cat; unsigned remaining; };
  CatEntry entries[] = {
    {PECategory::ArithInt, params.peArithInt},
    {PECategory::ArithFp,  params.peArithFp},
    {PECategory::Control,  params.peControl},
    {PECategory::Memory,   params.peMemory},
    {PECategory::Stream,   params.peStream},
  };

  // Build the category assignment for each PE position using interleaving.
  // This avoids clustering same-type PEs together.
  std::vector<PECategory> catAssign;
  catAssign.reserve(total);
  while (catAssign.size() < total) {
    bool placed = false;
    for (auto &e : entries) {
      if (e.remaining > 0) {
        catAssign.push_back(e.cat);
        --e.remaining;
        placed = true;
        if (catAssign.size() >= total)
          break;
      }
    }
    if (!placed)
      break;
  }

  // Determine spatial/temporal assignment.
  std::vector<PEHandle> assignment;
  assignment.reserve(total);
  unsigned spatialCount =
      static_cast<unsigned>(std::llround(total * params.spatialFraction));
  if (spatialCount > total) spatialCount = total;
  for (unsigned idx = 0; idx < total; ++idx) {
    bool isSpatial;
    if (spatialCount == total) {
      isSpatial = true;
    } else if (spatialCount == 0) {
      isSpatial = false;
    } else if (spatialCount >= total / 2) {
      // Spatial-dominant: scatter temporal PEs at regular intervals
      unsigned temporalCount = total - spatialCount;
      unsigned interval = total / temporalCount;
      isSpatial = ((idx % interval) != (interval - 1));
    } else {
      // Temporal-dominant: scatter spatial PEs at regular intervals
      unsigned interval = total / spatialCount;
      isSpatial = ((idx % interval) == 0);
    }
    assignment.push_back(peSet.get(catAssign[idx], isSpatial));
  }
  return assignment;
}

//===----------------------------------------------------------------------===//
// Core ADG build logic
//===----------------------------------------------------------------------===//

/// Core ADG build logic shared by generateKHGADG and exportKHGADG.
/// Uses 5 specialized PE categories distributed uniformly across the mesh,
/// with spatial/temporal mixing. External memories (8 total) are distributed
/// across all 4 edges (2 per edge). Optional on-chip SPM (8 units) is also
/// distributed across 4 edges.
static void buildKHGADGImpl(ADGBuilder &builder, const KHGTypeParams &params,
                            const std::string &moduleName) {
  // Define all 10 PE templates and build the per-position assignment
  PESet peSet = definePESet(builder, moduleName);
  std::vector<PEHandle> peAssignment = buildPEAssignment(params, peSet);

  // Convert flat assignment to a row/col PE selector lambda.
  // The chess mesh calls peSelector(row, col) in row-major order.
  unsigned cols = params.arrayCols;
  auto peSelector = [&](unsigned row, unsigned col) -> PEHandle {
    unsigned flatIdx = row * cols + col;
    return peAssignment[flatIdx];
  };

  // Compute boundary port layout.
  // 8 extmemories: 2 per edge. Each uses tagged bridges.
  unsigned extMemIngressPerEdge = kExtMemsPerEdge * kBridgeIngressPorts;
  unsigned extMemEgressPerEdge  = kExtMemsPerEdge * kBridgeEgressPorts;

  // SPM ports on boundary (if present) -- same tagged bridge pattern
  unsigned spmPerEdge = params.spmCount / 4;
  unsigned spmIngressPerEdge = spmPerEdge * kBridgeIngressPorts;
  unsigned spmEgressPerEdge  = spmPerEdge * kBridgeEgressPorts;

  // Module I/O: inputs on top+left edges, outputs on right+bottom edges.
  // 8x8: 5 per edge = 10 total I/O. 12x12: 9 per edge = 18 total I/O.
  unsigned topIOCount = computeModuleIOPerEdge(params.arrayCols);
  unsigned leftIOCount = computeModuleIOPerEdge(params.arrayRows);
  unsigned rightIOCount = computeModuleIOPerEdge(params.arrayRows);
  unsigned bottomIOCount = computeModuleIOPerEdge(params.arrayCols);
  unsigned totalModuleInputs = topIOCount + leftIOCount;
  unsigned totalModuleOutputs = rightIOCount + bottomIOCount;

  unsigned vertSwitchCount  = params.arrayRows + 1;
  unsigned horizSwitchCount = params.arrayCols + 1;

  ChessMeshOptions meshOpts;

  // Top edge: extmem + spm + module inputs (ingress into mesh)
  meshOpts.topExtraInputsPerSwitch = buildSpreadSideCounts(
      extMemIngressPerEdge + spmIngressPerEdge + topIOCount, horizSwitchCount);
  meshOpts.topExtraOutputsPerSwitch = buildSpreadSideCounts(
      extMemEgressPerEdge + spmEgressPerEdge, horizSwitchCount);

  // Bottom edge: extmem + spm + module outputs (egress from mesh)
  meshOpts.bottomExtraInputsPerSwitch = buildSpreadSideCounts(
      extMemIngressPerEdge + spmIngressPerEdge, horizSwitchCount);
  meshOpts.bottomExtraOutputsPerSwitch = buildSpreadSideCounts(
      extMemEgressPerEdge + spmEgressPerEdge + bottomIOCount, horizSwitchCount);

  // Left edge: extmem + spm + module inputs (ingress into mesh)
  meshOpts.leftExtraInputsPerSwitch = buildSpreadSideCounts(
      extMemIngressPerEdge + spmIngressPerEdge + leftIOCount, vertSwitchCount);
  meshOpts.leftExtraOutputsPerSwitch = buildSpreadSideCounts(
      extMemEgressPerEdge + spmEgressPerEdge, vertSwitchCount);

  // Right edge: extmem + spm + module outputs (egress from mesh)
  meshOpts.rightExtraInputsPerSwitch = buildSpreadSideCounts(
      extMemIngressPerEdge + spmIngressPerEdge, vertSwitchCount);
  meshOpts.rightExtraOutputsPerSwitch = buildSpreadSideCounts(
      extMemEgressPerEdge + spmEgressPerEdge + rightIOCount, vertSwitchCount);

  // Build chess mesh topology
  auto mesh = builder.buildChessMesh(
      params.arrayRows, params.arrayCols, peSelector, meshOpts);

  // Define tagged bridge templates (shared across all extmem and spm)
  auto extMemBridgeTemplates =
      defineTaggedMemBridgeTemplates(builder, moduleName + "_extmem_bridge");

  // Define and instantiate 8 external memories (2 per edge)
  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_extmem";
  extMemSpec.ldPorts = 2;
  extMemSpec.stPorts = 2;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 4;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMems = builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", kNumExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  // Wire ext memory through tagged bridges to boundary ports: 2 per edge
  // Order: top[0..1], bottom[2..3], left[4..5], right[6..7]
  unsigned topIngressIdx = 0, topEgressIdx = 0;
  unsigned bottomIngressIdx = 0, bottomEgressIdx = 0;
  unsigned leftIngressIdx = 0, leftEgressIdx = 0;
  unsigned rightIngressIdx = 0, rightEgressIdx = 0;

  // Top edge extmem (indices 0..1)
  for (unsigned m = 0; m < kExtMemsPerEdge; ++m)
    wireTaggedMemBridge(builder, extMems[m],
                        mesh.topIngressPorts, topIngressIdx,
                        mesh.topEgressPorts, topEgressIdx,
                        extMemBridgeTemplates, m, "extmem");

  // Bottom edge extmem (indices 2..3)
  for (unsigned m = 0; m < kExtMemsPerEdge; ++m)
    wireTaggedMemBridge(builder, extMems[kExtMemsPerEdge + m],
                        mesh.bottomIngressPorts, bottomIngressIdx,
                        mesh.bottomEgressPorts, bottomEgressIdx,
                        extMemBridgeTemplates, kExtMemsPerEdge + m, "extmem");

  // Left edge extmem (indices 4..5)
  for (unsigned m = 0; m < kExtMemsPerEdge; ++m)
    wireTaggedMemBridge(builder, extMems[2 * kExtMemsPerEdge + m],
                        mesh.leftIngressPorts, leftIngressIdx,
                        mesh.leftEgressPorts, leftEgressIdx,
                        extMemBridgeTemplates, 2 * kExtMemsPerEdge + m,
                        "extmem");

  // Right edge extmem (indices 6..7)
  for (unsigned m = 0; m < kExtMemsPerEdge; ++m)
    wireTaggedMemBridge(builder, extMems[3 * kExtMemsPerEdge + m],
                        mesh.rightIngressPorts, rightIngressIdx,
                        mesh.rightEgressPorts, rightEgressIdx,
                        extMemBridgeTemplates, 3 * kExtMemsPerEdge + m,
                        "extmem");

  // Wire on-chip SPM if present (same tagged bridge pattern)
  if (params.hasSPM()) {
    auto spmBridgeTemplates =
        defineTaggedMemBridgeTemplates(builder, moduleName + "_spm_bridge");

    unsigned spmSizeBytes = params.spmSizePerUnit;
    std::string spmMemrefType =
        "memref<" + std::to_string(spmSizeBytes / 4) + "xi32>";
    MemorySpec spmSpec;
    spmSpec.name = moduleName + "_spm";
    spmSpec.ldPorts = 2;
    spmSpec.stPorts = 2;
    spmSpec.memrefType = spmMemrefType;
    spmSpec.numRegion = 1;
    spmSpec.isPrivate = true;
    auto spmDef = builder.defineMemory(spmSpec);
    auto spmInstances = builder.instantiateMemoryArray(
        params.spmCount, spmDef, "spm");

    // Wire 2 SPM per edge through tagged bridges, after extmem ports.
    // On-chip memory has no memref input port, so inputPortBase = 0.
    for (unsigned m = 0; m < spmPerEdge; ++m)
      wireTaggedMemBridge(builder, spmInstances[m],
                          mesh.topIngressPorts, topIngressIdx,
                          mesh.topEgressPorts, topEgressIdx,
                          spmBridgeTemplates, m, "spm", 0);
    for (unsigned m = 0; m < spmPerEdge; ++m)
      wireTaggedMemBridge(builder, spmInstances[spmPerEdge + m],
                          mesh.bottomIngressPorts, bottomIngressIdx,
                          mesh.bottomEgressPorts, bottomEgressIdx,
                          spmBridgeTemplates, spmPerEdge + m, "spm", 0);
    for (unsigned m = 0; m < spmPerEdge; ++m)
      wireTaggedMemBridge(builder, spmInstances[2 * spmPerEdge + m],
                          mesh.leftIngressPorts, leftIngressIdx,
                          mesh.leftEgressPorts, leftEgressIdx,
                          spmBridgeTemplates, 2 * spmPerEdge + m, "spm", 0);
    for (unsigned m = 0; m < spmPerEdge; ++m)
      wireTaggedMemBridge(builder, spmInstances[3 * spmPerEdge + m],
                          mesh.rightIngressPorts, rightIngressIdx,
                          mesh.rightEgressPorts, rightEgressIdx,
                          spmBridgeTemplates, 3 * spmPerEdge + m, "spm", 0);

    // Set total SPM capacity
    uint64_t totalSpmBytes =
        static_cast<uint64_t>(params.spmCount) * params.spmSizePerUnit;
    builder.setSPMCapacity(totalSpmBytes);
  } else {
    builder.setSPMCapacity(0);
  }

  // Wire module I/O: inputs on top + left edges, outputs on right + bottom.
  // All ports use !fabric.bits<32>.
  std::vector<unsigned> moduleIns = builder.addInputs(
      "in", std::vector<std::string>(totalModuleInputs, bitsType()));
  std::vector<unsigned> moduleOuts = builder.addOutputs(
      "out", std::vector<std::string>(totalModuleOutputs, bitsType()));

  // Connect first topIOCount inputs to top edge ingress (after extmem+spm)
  unsigned inIdx = 0;
  for (unsigned i = 0; i < topIOCount; ++i, ++inIdx) {
    builder.connectInputToPort(moduleIns[inIdx],
                               mesh.topIngressPorts[topIngressIdx]);
    ++topIngressIdx;
  }
  // Connect next leftIOCount inputs to left edge ingress (after extmem+spm)
  for (unsigned i = 0; i < leftIOCount; ++i, ++inIdx) {
    builder.connectInputToPort(moduleIns[inIdx],
                               mesh.leftIngressPorts[leftIngressIdx]);
    ++leftIngressIdx;
  }

  // Connect first rightIOCount outputs to right edge egress (after extmem+spm)
  unsigned outIdx = 0;
  for (unsigned i = 0; i < rightIOCount; ++i, ++outIdx) {
    builder.connectPortToOutput(mesh.rightEgressPorts[rightEgressIdx],
                                moduleOuts[outIdx]);
    ++rightEgressIdx;
  }
  // Connect next bottomIOCount outputs to bottom edge egress (after extmem+spm)
  for (unsigned i = 0; i < bottomIOCount; ++i, ++outIdx) {
    builder.connectPortToOutput(mesh.bottomEgressPorts[bottomEgressIdx],
                                moduleOuts[outIdx]);
    ++bottomEgressIdx;
  }
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
