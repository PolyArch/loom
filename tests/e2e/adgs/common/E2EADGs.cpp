#include "E2EADGs.h"

#include "loom/ADG/ADGBuilder.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace loom::adg;

namespace loom {
namespace e2e {

namespace {

constexpr unsigned kDataWidth = 64;
constexpr unsigned kDensePEInputs = 24;
constexpr unsigned kDensePEOutputs = 24;

std::string bitsType(unsigned width = kDataWidth) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

std::string taggedBitsType(unsigned tagWidth, unsigned width = kDataWidth) {
  return "!fabric.tagged<" + bitsType(width) + ", i" +
         std::to_string(tagWidth) + ">";
}

struct TaggedVecaddExtMemBridge {
  SWHandle ldAddrMux;
  SWHandle ldDataDemux;
  SWHandle ldDoneDemux;
};

struct TaggedVecaddExtMemBridgeInstances {
  std::vector<InstanceHandle> ldAddrTags;
  InstanceHandle stAddrTag;
  InstanceHandle stDataTag;
  SWHandle ldAddrMuxTemplate;
  SWHandle ldDataDemuxTemplate;
  SWHandle ldDoneDemuxTemplate;
  InstanceHandle ldAddrMux;
  InstanceHandle ldDataDemux;
  InstanceHandle ldDoneDemux;
  std::vector<InstanceHandle> ldDataDrops;
  std::vector<InstanceHandle> ldDoneDrops;
  InstanceHandle stDoneDrop;
};

TaggedVecaddExtMemBridge buildTaggedVecaddExtMemBridge(ADGBuilder &builder,
                                                       const std::string &prefix) {
  const std::string taggedTy = taggedBitsType(1);
  TaggedVecaddExtMemBridge bridge;
  bridge.ldAddrMux = builder.defineSpatialSW(
      prefix + "_ld_addr_mux", {taggedTy, taggedTy}, {taggedTy}, {{true, true}});
  bridge.ldDataDemux = builder.defineTemporalSW(
      prefix + "_ld_data_demux", {taggedTy}, {taggedTy, taggedTy},
      {{true}, {true}}, 2);
  bridge.ldDoneDemux = builder.defineTemporalSW(
      prefix + "_ld_done_demux", {taggedTy}, {taggedTy, taggedTy},
      {{true}, {true}}, 2);
  return bridge;
}

struct SpatialKernelPE {
  PEHandle pe;
  unsigned inputs = kDensePEInputs;
  unsigned outputs = kDensePEOutputs;
  unsigned fuCount = 0;
};

SpatialKernelPE buildSpatialKernelPE(ADGBuilder &builder,
                                     const std::string &prefix,
                                     unsigned numInputs = kDensePEInputs,
                                     unsigned numOutputs = kDensePEOutputs) {
  // --- i32 arithmetic ---
  auto fuAdd = builder.defineBinaryFU(prefix + "_add", "arith.addi", "i32",
                                      "i32");
  auto fuSub = builder.defineBinaryFU(prefix + "_sub", "arith.subi", "i32",
                                      "i32");
  auto fuMul = builder.defineBinaryFU(prefix + "_mul", "arith.muli", "i32",
                                      "i32");
  auto fuShli32 = builder.defineBinaryFU(prefix + "_shli32", "arith.shli",
                                         "i32", "i32");
  auto fuShrui32 = builder.defineBinaryFU(prefix + "_shrui32", "arith.shrui",
                                          "i32", "i32");
  auto fuAndi32 = builder.defineBinaryFU(prefix + "_andi32", "arith.andi",
                                         "i32", "i32");
  auto fuDivsi32 = builder.defineBinaryFU(prefix + "_divsi32", "arith.divsi",
                                          "i32", "i32");
  auto fuRemsi32 = builder.defineBinaryFU(prefix + "_remsi32", "arith.remsi",
                                          "i32", "i32");
  auto fuCmp = builder.defineCmpiFU(prefix + "_cmp", "i32", "slt");
  auto fuSelect = builder.defineSelectFU(prefix + "_select", "i32");

  // --- i64 arithmetic ---
  auto fuAdd64 = builder.defineBinaryFU(prefix + "_add64", "arith.addi",
                                        "i64", "i64");
  auto fuMul64 = builder.defineBinaryFU(prefix + "_mul64", "arith.muli",
                                        "i64", "i64");
  auto fuShli64 = builder.defineBinaryFU(prefix + "_shli64", "arith.shli",
                                         "i64", "i64");
  auto fuShrui64 = builder.defineBinaryFU(prefix + "_shrui64", "arith.shrui",
                                          "i64", "i64");
  auto fuCmp64 = builder.defineCmpiFU(prefix + "_cmp64", "i64", "slt");

  // --- i8 arithmetic ---
  auto fuXori8 = builder.defineBinaryFU(prefix + "_xori8", "arith.xori",
                                        "i8", "i8");

  // --- index arithmetic ---
  auto fuAddIndex = builder.defineBinaryFU(prefix + "_add_index", "arith.addi",
                                           "index", "index");
  auto fuMulIndex = builder.defineBinaryFU(prefix + "_mul_index", "arith.muli",
                                           "index", "index");
  auto fuDivuiIndex = builder.defineBinaryFU(
      prefix + "_divui_index", "arith.divui", "index", "index");

  // --- float arithmetic ---
  auto fuAddf = builder.defineBinaryFU(prefix + "_addf", "arith.addf",
                                       "f32", "f32");
  auto fuSubf = builder.defineBinaryFU(prefix + "_subf", "arith.subf",
                                       "f32", "f32");
  auto fuMulf = builder.defineBinaryFU(prefix + "_mulf", "arith.mulf",
                                       "f32", "f32");
  auto fuDivf = builder.defineBinaryFU(prefix + "_divf", "arith.divf",
                                       "f32", "f32");
  auto fuNegf = builder.defineUnaryFU(prefix + "_negf", "arith.negf",
                                      "f32", "f32");
  auto fuCmpf = builder.defineCmpfFU(prefix + "_cmpf", "f32", "oeq");
  auto fuSelectF32 = builder.defineSelectFU(prefix + "_select_f32", "f32");
  auto fuSelectIndex =
      builder.defineSelectFU(prefix + "_select_index", "index");
  auto fuSelectI1 = builder.defineSelectFU(prefix + "_select_i1", "i1");

  // --- math operations ---
  auto fuAbsf = builder.defineUnaryFU(prefix + "_absf", "math.absf",
                                      "f32", "f32");
  auto fuCos = builder.defineUnaryFU(prefix + "_cos", "math.cos",
                                     "f32", "f32");
  auto fuSin = builder.defineUnaryFU(prefix + "_sin", "math.sin",
                                     "f32", "f32");
  auto fuExp = builder.defineUnaryFU(prefix + "_exp", "math.exp",
                                     "f32", "f32");
  auto fuFloor = builder.defineUnaryFU(prefix + "_floor", "math.floor",
                                       "f32", "f32");
  auto fuSqrt = builder.defineUnaryFU(prefix + "_sqrt", "math.sqrt",
                                      "f32", "f32");
  auto fuFma = builder.defineFUWithBody(
      prefix + "_fma", {"f32", "f32", "f32"}, {"f32"},
      "%0 = math.fma %arg0, %arg1, %arg2 : f32\nfabric.yield %0 : f32");

  // --- type conversions ---
  auto fuIndexToI32 =
      builder.defineIndexCastFU(prefix + "_index_to_i32", "index", "i32");
  auto fuI32ToIndex =
      builder.defineIndexCastFU(prefix + "_i32_to_index", "i32", "index");
  auto fuIndexToI64 =
      builder.defineIndexCastFU(prefix + "_index_to_i64", "index", "i64");
  auto fuI64ToIndex =
      builder.defineIndexCastFU(prefix + "_i64_to_index", "i64", "index");
  auto fuI8ToIndex =
      builder.defineIndexCastFU(prefix + "_i8_to_index", "i8", "index");
  auto fuExtsi32to64 = builder.defineUnaryFU(prefix + "_extsi_32_64",
                                             "arith.extsi", "i32", "i64");
  auto fuExtui32to64 = builder.defineUnaryFU(prefix + "_extui_32_64",
                                             "arith.extui", "i32", "i64");
  auto fuExtui1to32 = builder.defineUnaryFU(prefix + "_extui_1_32",
                                            "arith.extui", "i1", "i32");
  auto fuExtui8to32 = builder.defineUnaryFU(prefix + "_extui_8_32",
                                            "arith.extui", "i8", "i32");
  auto fuExtui8to64 = builder.defineUnaryFU(prefix + "_extui_8_64",
                                            "arith.extui", "i8", "i64");
  auto fuTrunci64to32 = builder.defineUnaryFU(prefix + "_trunci_64_32",
                                              "arith.trunci", "i64", "i32");
  auto fuSitofp = builder.defineUnaryFU(prefix + "_sitofp", "arith.sitofp",
                                        "i32", "f32");
  auto fuFptosi = builder.defineUnaryFU(prefix + "_fptosi", "arith.fptosi",
                                        "f32", "i32");
  auto fuUitofp = builder.defineUnaryFU(prefix + "_uitofp", "arith.uitofp",
                                        "i32", "f32");

  // --- constants ---
  auto fuConst = builder.defineConstantFU(prefix + "_const1", "i32",
                                          "1 : i32");
  auto fuConstIndex =
      builder.defineConstantFU(prefix + "_const_index", "index", "0 : index");
  auto fuConstI64 =
      builder.defineConstantFU(prefix + "_const_i64", "i64", "0 : i64");
  auto fuConstF32 =
      builder.defineConstantFU(prefix + "_const_f32", "f32",
                               "0.000000e+00 : f32");
  auto fuConstI1 =
      builder.defineConstantFU(prefix + "_const_i1", "i1", "false");

  // --- dataflow control ---
  auto fuStream = builder.defineStreamFU(prefix + "_stream");
  auto fuMux = builder.defineMuxFU(prefix + "_mux", "i32");
  auto fuMuxNone = builder.defineMuxFU(prefix + "_mux_none", "none");
  auto fuMuxI64 = builder.defineMuxFU(prefix + "_mux_i64", "i64");
  auto fuMuxF32 = builder.defineMuxFU(prefix + "_mux_f32", "f32");
  auto fuMuxIndex = builder.defineMuxFU(prefix + "_mux_index", "index");
  auto fuJoin = builder.defineJoinFU(prefix + "_join", 4);
  auto fuGate = builder.defineGateFU(prefix + "_gate", "i32");
  auto fuGateIndex = builder.defineGateFU(prefix + "_gate_index", "index");
  auto fuGateI64 = builder.defineGateFU(prefix + "_gate_i64", "i64");
  auto fuGateF32 = builder.defineGateFU(prefix + "_gate_f32", "f32");
  auto fuCarry = builder.defineCarryFU(prefix + "_carry", "i32");
  auto fuCarryNone = builder.defineCarryFU(prefix + "_carry_none", "none");
  auto fuCarryI64 = builder.defineCarryFU(prefix + "_carry_i64", "i64");
  auto fuCarryF32 = builder.defineCarryFU(prefix + "_carry_f32", "f32");
  auto fuCarryIndex = builder.defineCarryFU(prefix + "_carry_index", "index");
  auto fuCondBr = builder.defineCondBrFU(prefix + "_cond_br", "i32");
  auto fuCondBrNone = builder.defineCondBrFU(prefix + "_cond_br_none", "none");
  auto fuCondBrI64 = builder.defineCondBrFU(prefix + "_cond_br_i64", "i64");
  auto fuCondBrF32 = builder.defineCondBrFU(prefix + "_cond_br_f32", "f32");
  auto fuCondBrIndex =
      builder.defineCondBrFU(prefix + "_cond_br_index", "index");

  // --- memory ---
  auto fuLoad = builder.defineLoadFU(prefix + "_load", "index", "i32");
  auto fuStore = builder.defineStoreFU(prefix + "_store", "index", "i32");
  auto fuLoadF32 = builder.defineLoadFU(prefix + "_load_f32", "index", "f32");
  auto fuStoreF32 =
      builder.defineStoreFU(prefix + "_store_f32", "index", "f32");
  auto fuLoadI64 = builder.defineLoadFU(prefix + "_load_i64", "index", "i64");
  auto fuStoreI64 =
      builder.defineStoreFU(prefix + "_store_i64", "index", "i64");
  auto fuLoadI8 = builder.defineLoadFU(prefix + "_load_i8", "index", "i8");
  auto fuStoreI8 = builder.defineStoreFU(prefix + "_store_i8", "index", "i8");

  // --- invariants ---
  auto fuInvariant = builder.defineInvariantFU(prefix + "_invariant", "i32");
  auto fuInvariantI64 =
      builder.defineInvariantFU(prefix + "_invariant_i64", "i64");
  auto fuInvariantF32 =
      builder.defineInvariantFU(prefix + "_invariant_f32", "f32");
  auto fuInvariantIndex =
      builder.defineInvariantFU(prefix + "_invariant_index", "index");
  auto fuInvariantI1 =
      builder.defineInvariantFU(prefix + "_invariant_i1", "i1");

  std::vector<FUHandle> fus = {
      // i32 arithmetic
      fuAdd, fuSub, fuMul, fuShli32, fuShrui32, fuAndi32, fuDivsi32,
      fuRemsi32, fuCmp, fuSelect,
      // i64 arithmetic
      fuAdd64, fuMul64, fuShli64, fuShrui64, fuCmp64,
      // i8 arithmetic
      fuXori8,
      // index arithmetic
      fuAddIndex, fuMulIndex, fuDivuiIndex,
      // float arithmetic + math
      fuAddf, fuSubf, fuMulf, fuDivf, fuNegf, fuCmpf, fuSelectF32,
      fuSelectIndex, fuSelectI1,
      fuAbsf, fuCos, fuSin, fuExp, fuFloor, fuSqrt, fuFma,
      // type conversions
      fuIndexToI32, fuI32ToIndex, fuIndexToI64, fuI64ToIndex, fuI8ToIndex,
      fuExtsi32to64, fuExtui32to64, fuExtui1to32, fuExtui8to32, fuExtui8to64,
      fuTrunci64to32, fuSitofp, fuFptosi, fuUitofp,
      // constants
      fuConst, fuConstIndex, fuConstI64, fuConstF32, fuConstI1,
      // dataflow control
      fuStream, fuMux, fuMuxNone, fuMuxI64, fuMuxF32, fuMuxIndex,
      fuJoin,
      fuGate, fuGateIndex, fuGateI64, fuGateF32,
      fuCarry, fuCarryNone, fuCarryI64, fuCarryF32, fuCarryIndex,
      fuCondBr, fuCondBrNone, fuCondBrI64, fuCondBrF32, fuCondBrIndex,
      // memory
      fuLoad, fuStore, fuLoadF32, fuStoreF32, fuLoadI64, fuStoreI64,
      fuLoadI8, fuStoreI8,
      // invariants
      fuInvariant, fuInvariantI64, fuInvariantF32, fuInvariantIndex,
      fuInvariantI1,
  };

  auto pe = builder.defineSpatialPE(
      prefix + "_pe",
      std::vector<std::string>(numInputs, bitsType()),
      std::vector<std::string>(numOutputs, bitsType()),
      fus);
  return {pe, numInputs, numOutputs, static_cast<unsigned>(fus.size())};
}

void applyStarLayout(ADGBuilder &builder, InstanceHandle sw,
                     const std::vector<InstanceHandle> &pes,
                     const std::vector<InstanceHandle> &extMems,
                     const SpatialKernelPE &kernelPE, unsigned swInputs,
                     unsigned swOutputs) {
  auto estimatePEWidth = [&](unsigned fuCount) {
    const double approxFuBoxW = 140.0;
    const double approxFuGap = 12.0;
    const double approxPEPadX = 60.0;
    return std::max(200.0,
                    static_cast<double>(fuCount) * approxFuBoxW +
                        std::max(0.0, static_cast<double>(fuCount) - 1.0) *
                            approxFuGap +
                        approxPEPadX);
  };
  auto estimateSwitchSide = [&](unsigned numInputs, unsigned numOutputs) {
    unsigned maxSideSlots =
        std::max((numInputs + 1) / 2, (numOutputs + 1) / 2);
    return std::max(84.0, 32.0 + (static_cast<double>(maxSideSlots) + 1.0) *
                                     24.0);
  };

  builder.setInstanceVizPosition(sw, 0.0, 0.0, 0, 0);

  const double peW = estimatePEWidth(kernelPE.fuCount);
  const double peH = 200.0;
  const double swSide = estimateSwitchSide(swInputs, swOutputs);
  const double peHalfDiag = std::hypot(peW / 2.0, peH / 2.0);
  const double swHalfDiag = std::hypot(swSide / 2.0, swSide / 2.0);
  const unsigned peCount = static_cast<unsigned>(pes.size());
  const double centerGap = peHalfDiag + swHalfDiag + 120.0;
  double peRadius = std::max(520.0, centerGap);
  if (peCount >= 2) {
    const double requiredAdjDistance = peHalfDiag * 2.0 + 100.0;
    const double denom =
        2.0 * std::sin(M_PI / static_cast<double>(peCount));
    if (denom > 0.0)
      peRadius = std::max(peRadius, requiredAdjDistance / denom);
  }
  const double angleStep =
      pes.empty() ? 0.0 : (2.0 * M_PI / static_cast<double>(pes.size()));
  for (unsigned idx = 0; idx < pes.size(); ++idx) {
    double angle = angleStep * static_cast<double>(idx);
    builder.setInstanceVizPosition(
        pes[idx], peRadius * std::cos(angle), peRadius * std::sin(angle), 1,
        static_cast<int>(idx));
  }

  const double memW = 170.0;
  const double memX = -peRadius - peW / 2.0 - memW / 2.0 - 180.0;
  const double memStep = 180.0;
  const double memY0 =
      -0.5 * memStep * static_cast<double>(std::max<size_t>(extMems.size(), 1) - 1);
  for (unsigned idx = 0; idx < extMems.size(); ++idx) {
    builder.setInstanceVizPosition(extMems[idx], memX,
                                   memY0 + memStep * static_cast<double>(idx),
                                   2, static_cast<int>(idx));
  }
}

void buildStarDomain(const std::string &moduleName,
                     const std::string &outputPath, unsigned numPEs,
                     unsigned numExtMems, unsigned scalarInputs,
                     unsigned scalarOutputs) {
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName);

  const unsigned extMemSWInputs = 2 + 1 + 2;
  const unsigned extMemSWOutputs = 2 + 2;
  const unsigned swInputs =
      numPEs * computePE.outputs + numExtMems * extMemSWInputs + scalarInputs;
  const unsigned swOutputs =
      numPEs * computePE.inputs + numExtMems * extMemSWOutputs + scalarOutputs;
  auto sw = builder.defineFullCrossbarSpatialSW(moduleName + "_sw", swInputs,
                                                swOutputs, kDataWidth);
  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_mem";
  extMemSpec.ldPorts = 2;
  extMemSpec.stPorts = 1;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 3;
  auto extMem = builder.defineExtMemory(extMemSpec);

  SwitchBankDomainSpec domain;
  domain.sw = sw;
  domain.switchInstanceName = "sw_center";
  domain.pe = computePE.pe;
  domain.numPEs = numPEs;
  domain.peInputCount = computePE.inputs;
  domain.peOutputCount = computePE.outputs;
  domain.pePrefix = "pe";
  domain.extMem = extMem;
  domain.numExtMems = numExtMems;
  domain.swInputPortsPerExtMem = extMemSWInputs;
  domain.swOutputPortsPerExtMem = extMemSWOutputs;
  domain.extMemPrefix = "extmem";
  domain.extMemrefPrefix = "buffer";
  domain.scalarInputTypes.assign(scalarInputs, bitsType());
  domain.scalarOutputTypes.assign(scalarOutputs, bitsType());

  auto result = builder.buildSwitchBankDomain(domain);
  applyStarLayout(builder, result.sw, result.peInstances,
                  result.extMemInstances, computePE, swInputs, swOutputs);
  builder.exportMLIR(outputPath);
}

void setExtMemStripLayout(ADGBuilder &builder,
                          const std::vector<InstanceHandle> &extMems,
                          double x, double y0, double step) {
  for (unsigned idx = 0; idx < extMems.size(); ++idx) {
    builder.setInstanceVizPosition(extMems[idx], x,
                                   y0 + step * static_cast<double>(idx), 8,
                                   static_cast<int>(idx));
  }
}

void wireExtMemToSidePorts(
    ADGBuilder &builder, InstanceHandle mem,
    const std::vector<PortRef> &ingressPorts,
    const std::vector<PortRef> &egressPorts,
    unsigned &ingressIdx, unsigned &egressIdx,
    unsigned memOutputPorts, unsigned memInputPorts) {
  for (unsigned outPort = 0; outPort < memOutputPorts; ++outPort) {
    builder.connect(mem, outPort, ingressPorts[ingressIdx].instance,
                    ingressPorts[ingressIdx].port);
    ++ingressIdx;
  }
  for (unsigned inPort = 0; inPort < memInputPorts; ++inPort) {
    builder.connect(egressPorts[egressIdx].instance,
                    egressPorts[egressIdx].port, mem, 1 + inPort);
    ++egressIdx;
  }
}

void wireRoundRobinExtMemIngressEgress(
    ADGBuilder &builder, const MeshResult &mesh,
    const std::vector<InstanceHandle> &extMems, unsigned topLeftIngressCount,
    unsigned bottomLeftEgressCount, unsigned memOutputPorts,
    unsigned memInputPorts) {
  unsigned leftIngressIdx = 0;
  unsigned rightIngressIdx = topLeftIngressCount;
  unsigned leftEgressIdx = 0;
  unsigned rightEgressIdx = bottomLeftEgressCount;

  for (unsigned memIdx = 0; memIdx < extMems.size(); ++memIdx) {
    InstanceHandle mem = extMems[memIdx];
    unsigned &ingressIdx = (memIdx % 2 == 0) ? leftIngressIdx : rightIngressIdx;
    unsigned &egressIdx = (memIdx % 2 == 0) ? leftEgressIdx : rightEgressIdx;

    for (unsigned outPort = 0; outPort < memOutputPorts; ++outPort) {
      builder.connect(mem, outPort, mesh.ingressPorts[ingressIdx].instance,
                      mesh.ingressPorts[ingressIdx].port);
      ++ingressIdx;
    }
    for (unsigned inPort = 0; inPort < memInputPorts; ++inPort) {
      builder.connect(mesh.egressPorts[egressIdx].instance,
                      mesh.egressPorts[egressIdx].port, mem, 1 + inPort);
      ++egressIdx;
    }
  }
}

std::vector<unsigned> buildSpreadSideCounts(unsigned totalPorts,
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

TaggedVecaddExtMemBridgeInstances wireTaggedVecaddExtMemSideBridge(
    ADGBuilder &builder, const std::vector<PortRef> &ingressPorts,
    const std::vector<PortRef> &egressPorts, InstanceHandle mem, unsigned memIdx,
    const TaggedVecaddExtMemBridge &bridgeTemplates) {
  assert(ingressPorts.size() >= 5 &&
         "vecadd extmemory side bridge expects five ingress boundary ports");
  assert(egressPorts.size() >= 4 &&
         "vecadd extmemory side bridge expects four egress boundary ports");

  const std::string taggedTy = taggedBitsType(1);
  TaggedVecaddExtMemBridgeInstances bridge;
  bridge.ldAddrMuxTemplate = bridgeTemplates.ldAddrMux;
  bridge.ldDataDemuxTemplate = bridgeTemplates.ldDataDemux;
  bridge.ldDoneDemuxTemplate = bridgeTemplates.ldDoneDemux;

  bridge.ldAddrTags = builder.createAddTagBank(bitsType(), taggedTy, {0, 1});
  bridge.stAddrTag = builder.createAddTag(bitsType(), taggedTy, 0);
  bridge.stDataTag = builder.createAddTag(bitsType(), taggedTy, 0);
  bridge.ldDataDrops = builder.createDelTagBank(taggedTy, bitsType(), 2);
  bridge.ldDoneDrops = builder.createDelTagBank(taggedTy, bitsType(), 2);
  bridge.stDoneDrop = builder.createDelTag(taggedTy, bitsType());

  bridge.ldAddrMux = builder.instantiateSW(
      bridgeTemplates.ldAddrMux, "vecadd_ld_addr_mux_" + std::to_string(memIdx));
  bridge.ldDataDemux = builder.instantiateSW(
      bridgeTemplates.ldDataDemux,
      "vecadd_ld_data_demux_" + std::to_string(memIdx));
  bridge.ldDoneDemux = builder.instantiateSW(
      bridgeTemplates.ldDoneDemux,
      "vecadd_ld_done_demux_" + std::to_string(memIdx));

  builder.connect(egressPorts[0].instance, egressPorts[0].port,
                  bridge.ldAddrTags[0], 0);
  builder.connect(egressPorts[1].instance, egressPorts[1].port,
                  bridge.ldAddrTags[1], 0);
  builder.connect(egressPorts[2].instance, egressPorts[2].port,
                  bridge.stAddrTag, 0);
  builder.connect(egressPorts[3].instance, egressPorts[3].port,
                  bridge.stDataTag, 0);

  builder.connect(bridge.ldAddrTags[0], 0, bridge.ldAddrMux, 0);
  builder.connect(bridge.ldAddrTags[1], 0, bridge.ldAddrMux, 1);
  builder.connect(bridge.ldAddrMux, 0, mem, 1);
  builder.connect(bridge.stAddrTag, 0, mem, 2);
  builder.connect(bridge.stDataTag, 0, mem, 3);

  builder.connect(mem, 0, bridge.ldDataDemux, 0);
  builder.connect(bridge.ldDataDemux, 0, bridge.ldDataDrops[0], 0);
  builder.connect(bridge.ldDataDemux, 1, bridge.ldDataDrops[1], 0);
  builder.connect(bridge.ldDataDrops[0], 0, ingressPorts[0].instance,
                  ingressPorts[0].port);
  builder.connect(bridge.ldDataDrops[1], 0, ingressPorts[1].instance,
                  ingressPorts[1].port);

  builder.connect(mem, 1, bridge.ldDoneDemux, 0);
  builder.connect(bridge.ldDoneDemux, 0, bridge.ldDoneDrops[0], 0);
  builder.connect(bridge.ldDoneDemux, 1, bridge.ldDoneDrops[1], 0);
  builder.connect(bridge.ldDoneDrops[0], 0, ingressPorts[2].instance,
                  ingressPorts[2].port);
  builder.connect(bridge.ldDoneDrops[1], 0, ingressPorts[3].instance,
                  ingressPorts[3].port);

  builder.connect(mem, 2, bridge.stDoneDrop, 0);
  builder.connect(bridge.stDoneDrop, 0, ingressPorts[4].instance,
                  ingressPorts[4].port);
  return bridge;
}

std::vector<TaggedVecaddExtMemBridgeInstances>
wireRoundRobinVecaddTaggedExtMemIngressEgress(
    ADGBuilder &builder, const MeshResult &mesh,
    const std::vector<InstanceHandle> &extMems, unsigned topLeftIngressCount,
    unsigned bottomLeftEgressCount,
    const TaggedVecaddExtMemBridge &bridgeTemplates) {
  unsigned leftIngressIdx = 0;
  unsigned rightIngressIdx = topLeftIngressCount;
  unsigned leftEgressIdx = 0;
  unsigned rightEgressIdx = bottomLeftEgressCount;

  const std::string taggedTy = taggedBitsType(1);
  std::vector<TaggedVecaddExtMemBridgeInstances> bridgeInstances;
  bridgeInstances.reserve(extMems.size());

  for (unsigned memIdx = 0; memIdx < extMems.size(); ++memIdx) {
    InstanceHandle mem = extMems[memIdx];
    bool useLeftIngress = (memIdx % 2 == 0);
    bool useLeftEgress = (memIdx % 2 == 0);
    if (extMems.size() == 3) {
      if (memIdx == 0) {
        useLeftIngress = true;
        useLeftEgress = true;
      } else if (memIdx == 1) {
        useLeftIngress = false;
        useLeftEgress = false;
      } else {
        useLeftIngress = true;
        useLeftEgress = false;
      }
    }
    unsigned &ingressIdx = useLeftIngress ? leftIngressIdx : rightIngressIdx;
    unsigned &egressIdx = useLeftEgress ? leftEgressIdx : rightEgressIdx;

    TaggedVecaddExtMemBridgeInstances bridge;
    bridge.ldAddrMuxTemplate = bridgeTemplates.ldAddrMux;
    bridge.ldDataDemuxTemplate = bridgeTemplates.ldDataDemux;
    bridge.ldDoneDemuxTemplate = bridgeTemplates.ldDoneDemux;

    bridge.ldAddrTags =
        builder.createAddTagBank(bitsType(), taggedTy, {0, 1});
    bridge.stAddrTag = builder.createAddTag(bitsType(), taggedTy, 0);
    bridge.stDataTag = builder.createAddTag(bitsType(), taggedTy, 0);
    bridge.ldDataDrops = builder.createDelTagBank(taggedTy, bitsType(), 2);
    bridge.ldDoneDrops = builder.createDelTagBank(taggedTy, bitsType(), 2);
    bridge.stDoneDrop = builder.createDelTag(taggedTy, bitsType());

    bridge.ldAddrMux = builder.instantiateSW(
        bridgeTemplates.ldAddrMux,
        "vecadd_ld_addr_mux_" + std::to_string(memIdx));
    bridge.ldDataDemux = builder.instantiateSW(
        bridgeTemplates.ldDataDemux,
        "vecadd_ld_data_demux_" + std::to_string(memIdx));
    bridge.ldDoneDemux = builder.instantiateSW(
        bridgeTemplates.ldDoneDemux,
        "vecadd_ld_done_demux_" + std::to_string(memIdx));

    builder.connect(mesh.egressPorts[egressIdx + 0].instance,
                    mesh.egressPorts[egressIdx + 0].port, bridge.ldAddrTags[0], 0);
    builder.connect(mesh.egressPorts[egressIdx + 1].instance,
                    mesh.egressPorts[egressIdx + 1].port, bridge.ldAddrTags[1], 0);
    builder.connect(mesh.egressPorts[egressIdx + 2].instance,
                    mesh.egressPorts[egressIdx + 2].port, bridge.stAddrTag, 0);
    builder.connect(mesh.egressPorts[egressIdx + 3].instance,
                    mesh.egressPorts[egressIdx + 3].port, bridge.stDataTag, 0);
    egressIdx += 4;

    builder.connect(bridge.ldAddrTags[0], 0, bridge.ldAddrMux, 0);
    builder.connect(bridge.ldAddrTags[1], 0, bridge.ldAddrMux, 1);
    builder.connect(bridge.ldAddrMux, 0, mem, 1);
    builder.connect(bridge.stAddrTag, 0, mem, 2);
    builder.connect(bridge.stDataTag, 0, mem, 3);

    builder.connect(mem, 0, bridge.ldDataDemux, 0);
    builder.connect(bridge.ldDataDemux, 0, bridge.ldDataDrops[0], 0);
    builder.connect(bridge.ldDataDemux, 1, bridge.ldDataDrops[1], 0);
    builder.connect(bridge.ldDataDrops[0], 0, mesh.ingressPorts[ingressIdx + 0].instance,
                    mesh.ingressPorts[ingressIdx + 0].port);
    builder.connect(bridge.ldDataDrops[1], 0, mesh.ingressPorts[ingressIdx + 1].instance,
                    mesh.ingressPorts[ingressIdx + 1].port);

    builder.connect(mem, 1, bridge.ldDoneDemux, 0);
    builder.connect(bridge.ldDoneDemux, 0, bridge.ldDoneDrops[0], 0);
    builder.connect(bridge.ldDoneDemux, 1, bridge.ldDoneDrops[1], 0);
    builder.connect(bridge.ldDoneDrops[0], 0, mesh.ingressPorts[ingressIdx + 2].instance,
                    mesh.ingressPorts[ingressIdx + 2].port);
    builder.connect(bridge.ldDoneDrops[1], 0, mesh.ingressPorts[ingressIdx + 3].instance,
                    mesh.ingressPorts[ingressIdx + 3].port);

    builder.connect(mem, 2, bridge.stDoneDrop, 0);
    builder.connect(bridge.stDoneDrop, 0, mesh.ingressPorts[ingressIdx + 4].instance,
                    mesh.ingressPorts[ingressIdx + 4].port);
    ingressIdx += 5;
    bridgeInstances.push_back(bridge);
  }
  return bridgeInstances;
}

void layoutTaggedVecaddExtMemBridges(
    ADGBuilder &builder, const std::vector<InstanceHandle> &extMems,
    const std::vector<TaggedVecaddExtMemBridgeInstances> &bridges,
    double memX, double memY0, double memStep) {
  const double muxX = memX - 145.0;
  const double tagX = memX - 255.0;
  const double demuxX = memX + 145.0;
  const double delTagX = memX + 255.0;
  for (unsigned idx = 0; idx < extMems.size() && idx < bridges.size(); ++idx) {
    double memY = memY0 + memStep * static_cast<double>(idx);
    builder.setInstanceVizPosition(extMems[idx], memX, memY, 8,
                                   static_cast<int>(idx));

    const auto &bridge = bridges[idx];
    builder.setInstanceVizPosition(bridge.ldAddrMux, muxX, memY - 62.0, 8,
                                   static_cast<int>(idx * 10 + 0));
    builder.setInstanceVizPosition(bridge.stAddrTag, tagX, memY + 6.0, 8,
                                   static_cast<int>(idx * 10 + 1));
    builder.setInstanceVizPosition(bridge.stDataTag, tagX, memY + 62.0, 8,
                                   static_cast<int>(idx * 10 + 2));
    if (bridge.ldAddrTags.size() >= 2) {
      builder.setInstanceVizPosition(bridge.ldAddrTags[0], tagX, memY - 96.0, 8,
                                     static_cast<int>(idx * 10 + 3));
      builder.setInstanceVizPosition(bridge.ldAddrTags[1], tagX, memY - 36.0, 8,
                                     static_cast<int>(idx * 10 + 4));
    }

    builder.setInstanceVizPosition(bridge.ldDataDemux, demuxX, memY - 54.0, 8,
                                   static_cast<int>(idx * 10 + 5));
    builder.setInstanceVizPosition(bridge.ldDoneDemux, demuxX, memY + 18.0, 8,
                                   static_cast<int>(idx * 10 + 6));
    builder.setInstanceVizPosition(bridge.stDoneDrop, delTagX, memY + 88.0, 8,
                                   static_cast<int>(idx * 10 + 7));
    if (bridge.ldDataDrops.size() >= 2) {
      builder.setInstanceVizPosition(bridge.ldDataDrops[0], delTagX, memY - 84.0, 8,
                                     static_cast<int>(idx * 10 + 8));
      builder.setInstanceVizPosition(bridge.ldDataDrops[1], delTagX, memY - 24.0, 8,
                                     static_cast<int>(idx * 10 + 9));
    }
    if (bridge.ldDoneDrops.size() >= 2) {
      builder.setInstanceVizPosition(bridge.ldDoneDrops[0], delTagX, memY + 28.0, 8,
                                     static_cast<int>(idx * 10 + 10));
      builder.setInstanceVizPosition(bridge.ldDoneDrops[1], delTagX, memY + 60.0, 8,
                                     static_cast<int>(idx * 10 + 11));
    }
  }
}

void layoutTaggedVecaddSideExtMemBridges(
    ADGBuilder &builder, const std::vector<InstanceHandle> &extMems,
    const std::vector<TaggedVecaddExtMemBridgeInstances> &bridges,
    unsigned rows, unsigned cols) {
  assert(extMems.size() == bridges.size() &&
         "vecadd extmemory layout expects one bridge bundle per extmemory");
  if (extMems.empty())
    return;

  // Keep this layout anchored to the actual chess-mesh visual scale used by
  // the current dense PE template, instead of the old 520x520 toy constants.
  const double switchOriginX = 215.0;
  const double switchOriginY = 215.0;
  const double switchStepX = 1226.0;
  const double switchStepY = 1288.0;
  const double meshLeft = switchOriginX;
  const double meshRight = switchOriginX + static_cast<double>(cols) * switchStepX;
  const double meshTop = switchOriginY;
  const double meshBottom = switchOriginY + static_cast<double>(rows) * switchStepY;
  const double meshMidY = (meshTop + meshBottom) / 2.0;
  const double memOffsetX = 620.0;
  const double leftMemX = meshLeft - memOffsetX;
  const double rightMemX = meshRight + memOffsetX;
  const double leftMemY = meshMidY - 180.0;
  const double rightMemY = meshMidY + 180.0;

  auto placeOne = [&](const TaggedVecaddExtMemBridgeInstances &bridge,
                      InstanceHandle mem, double memX, double memY,
                      int gridColBase) {
    const double muxX = memX - 145.0;
    const double tagX = memX - 255.0;
    const double demuxX = memX + 145.0;
    const double delTagX = memX + 255.0;

    builder.setInstanceVizPosition(mem, memX, memY, 8, gridColBase);
    builder.setInstanceVizPosition(bridge.ldAddrMux, muxX, memY - 62.0, 8,
                                   gridColBase + 1);
    builder.setInstanceVizPosition(bridge.stAddrTag, tagX, memY + 6.0, 8,
                                   gridColBase + 2);
    builder.setInstanceVizPosition(bridge.stDataTag, tagX, memY + 62.0, 8,
                                   gridColBase + 3);
    if (bridge.ldAddrTags.size() >= 2) {
      builder.setInstanceVizPosition(bridge.ldAddrTags[0], tagX, memY - 96.0, 8,
                                     gridColBase + 4);
      builder.setInstanceVizPosition(bridge.ldAddrTags[1], tagX, memY - 36.0, 8,
                                     gridColBase + 5);
    }
    builder.setInstanceVizPosition(bridge.ldDataDemux, demuxX, memY - 54.0, 8,
                                   gridColBase + 6);
    builder.setInstanceVizPosition(bridge.ldDoneDemux, demuxX, memY + 18.0, 8,
                                   gridColBase + 7);
    builder.setInstanceVizPosition(bridge.stDoneDrop, delTagX, memY + 88.0, 8,
                                   gridColBase + 8);
    if (bridge.ldDataDrops.size() >= 2) {
      builder.setInstanceVizPosition(bridge.ldDataDrops[0], delTagX, memY - 84.0, 8,
                                     gridColBase + 9);
      builder.setInstanceVizPosition(bridge.ldDataDrops[1], delTagX, memY - 24.0, 8,
                                     gridColBase + 10);
    }
    if (bridge.ldDoneDrops.size() >= 2) {
      builder.setInstanceVizPosition(bridge.ldDoneDrops[0], delTagX, memY + 28.0, 8,
                                     gridColBase + 11);
      builder.setInstanceVizPosition(bridge.ldDoneDrops[1], delTagX, memY + 60.0, 8,
                                     gridColBase + 12);
    }
  };

  placeOne(bridges[0], extMems[0], leftMemX, leftMemY, 0);
  if (extMems.size() >= 2)
    placeOne(bridges[1], extMems[1], rightMemX, rightMemY, 20);
}

void buildChessDomain(const std::string &moduleName,
                      const std::string &outputPath, unsigned rows,
                      unsigned cols, unsigned numExtMems,
                      unsigned scalarInputs, unsigned scalarOutputs) {
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName);

  constexpr unsigned kLdPorts = 2;
  constexpr unsigned kStPorts = 2;
  const unsigned memOutputPorts = (kLdPorts > 0 ? 2 : 0) + (kStPorts > 0 ? 1 : 0);
  const unsigned memInputPorts = (kLdPorts > 0 ? 1 : 0) + (kStPorts > 0 ? 2 : 0);

  ChessMeshOptions options;
  const unsigned leftIngressMems = (numExtMems + 1) / 2;
  const unsigned rightIngressMems = numExtMems / 2;
  const unsigned leftEgressMems = (numExtMems + 1) / 2;
  const unsigned rightEgressMems = numExtMems / 2;
  options.topLeftExtraInputs =
      leftIngressMems * memOutputPorts + scalarInputs;
  options.topRightExtraInputs = rightIngressMems * memOutputPorts;
  options.bottomLeftExtraOutputs = leftEgressMems * memInputPorts;
  options.bottomRightExtraOutputs =
      rightEgressMems * memInputPorts + scalarOutputs;
  auto mesh = builder.buildChessMesh(
      rows, cols,
      [&](unsigned, unsigned) { return computePE.pe; }, options);

  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_mem";
  extMemSpec.ldPorts = kLdPorts;
  extMemSpec.stPorts = kStPorts;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 3;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMems = builder.instantiateExtMemArray(numExtMems, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", numExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  wireRoundRobinExtMemIngressEgress(builder, mesh, extMems,
                                    options.topLeftExtraInputs,
                                    options.bottomLeftExtraOutputs,
                                    memOutputPorts, memInputPorts);

  std::vector<unsigned> inputs = builder.addInputs(
      "scalar", std::vector<std::string>(scalarInputs, bitsType()));
  std::vector<unsigned> outputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(scalarOutputs, bitsType()));

  unsigned ingressIdx = leftIngressMems * memOutputPorts;
  for (unsigned idx = 0; idx < inputs.size(); ++idx, ++ingressIdx)
    builder.connectInputToPort(inputs[idx], mesh.ingressPorts[ingressIdx]);

  unsigned egressIdx = numExtMems * memInputPorts;
  for (unsigned idx = 0; idx < outputs.size(); ++idx, ++egressIdx)
    builder.connectPortToOutput(mesh.egressPorts[egressIdx], outputs[idx]);

  setExtMemStripLayout(builder, extMems, -360.0,
                       -220.0 + static_cast<double>(rows) * 8.0, 180.0);
  builder.exportMLIR(outputPath);
}

} // namespace

void buildTinyAdd1PE(const std::string &outputPath) {
  const std::string moduleName = "tiny_add_1pe";
  ADGBuilder builder(moduleName);

  auto fuAdd =
      builder.defineBinaryFU(moduleName + "_add", "arith.addi", "i32", "i32");
  auto pe = builder.defineSingleFUSpatialPE(moduleName + "_pe", 2, 1,
                                            kDataWidth, fuAdd);
  auto peInst = builder.instantiatePE(pe, "pe_0");

  auto lhs = builder.addScalarInput("lhs", kDataWidth);
  auto rhs = builder.addScalarInput("rhs", kDataWidth);
  auto sum = builder.addScalarOutput("sum", kDataWidth);

  builder.connectInputToInstance(lhs, peInst, 0);
  builder.connectInputToInstance(rhs, peInst, 1);
  builder.connectInstanceToOutput(peInst, 0, sum);

  builder.setInstanceVizPosition(peInst, 0.0, 0.0, 0, 0);
  builder.exportMLIR(outputPath);
}

static void buildSumArrayDemoChess(const std::string &outputPath,
                                   const std::string &moduleName,
                                   unsigned rows, unsigned cols) {
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName, 4, 4);

  constexpr unsigned kNumExtMems = 1;
  constexpr unsigned kScalarInputs = 2;
  constexpr unsigned kScalarOutputs = 2;
  constexpr unsigned kExtMemMeshInputs = 2;
  constexpr unsigned kExtMemMeshOutputs = 1;

  ChessMeshOptions options;
  options.topLeftExtraInputs = kScalarInputs;
  options.topRightExtraInputs = kNumExtMems * kExtMemMeshInputs;
  options.topRightExtraOutputs = kNumExtMems * kExtMemMeshOutputs;
  options.bottomRightExtraOutputs = kScalarOutputs;
  auto mesh = builder.buildChessMesh(
      rows, cols,
      [&](unsigned, unsigned) { return computePE.pe; }, options);
  assert(mesh.ingressPorts.size() ==
             kScalarInputs + kNumExtMems * kExtMemMeshInputs &&
         "sum-array demo chess expects top-left and top-right ingress ports");
  assert(mesh.egressPorts.size() ==
             kNumExtMems * kExtMemMeshOutputs + kScalarOutputs &&
         "sum-array demo chess expects top-right extmem egress and bottom-right scalar egress");
  assert(mesh.egressPorts[0].instance.id == mesh.swGrid[0][cols].id &&
         "sum-array demo chess expects first egress at top-right switch");

  auto extMem = builder.defineExtMemory(moduleName + "_mem", 1, 0);
  auto extMems = builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs =
      builder.addMemrefInputs("buffer", kNumExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  std::vector<unsigned> inputs = builder.addInputs(
      "scalar", std::vector<std::string>(kScalarInputs, bitsType()));
  std::vector<unsigned> outputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(kScalarOutputs, bitsType()));

  unsigned ingressIdx = 0;
  for (unsigned idx = 0; idx < inputs.size(); ++idx, ++ingressIdx)
    builder.connectInputToPort(inputs[idx], mesh.ingressPorts[ingressIdx]);

  for (InstanceHandle mem : extMems) {
    for (unsigned outPort = 0; outPort < kExtMemMeshInputs; ++outPort) {
      builder.connect(mem, outPort, mesh.ingressPorts[ingressIdx].instance,
                      mesh.ingressPorts[ingressIdx].port);
      ++ingressIdx;
    }
  }

  auto switchDegree = [&](unsigned sr, unsigned sc) -> unsigned {
    unsigned degree = 0;
    if (sr > 0)
      degree++;
    if (sr + 1 < rows + 1)
      degree++;
    if (sc > 0)
      degree++;
    if (sc + 1 < cols + 1)
      degree++;
    if (sr > 0 && sc > 0)
      degree++;
    if (sr > 0 && sc < cols)
      degree++;
    if (sr < rows && sc > 0)
      degree++;
    if (sr < rows && sc < cols)
      degree++;
    return degree;
  };

  unsigned egressIdx = 0;
  unsigned topRightExtraOutputBase = switchDegree(0, cols);
  for (unsigned idx = 0; idx < kExtMemMeshOutputs; ++idx, ++egressIdx)
    builder.connect(mesh.swGrid[0][cols], topRightExtraOutputBase + idx,
                    extMems[0], 1 + idx);

  for (unsigned idx = 0; idx < outputs.size(); ++idx, ++egressIdx)
    builder.connectPortToOutput(mesh.egressPorts[egressIdx], outputs[idx]);

  setExtMemStripLayout(builder, extMems,
                       260.0 + static_cast<double>(cols) * 520.0,
                       -120.0 + static_cast<double>(rows) * 8.0, 180.0);
  builder.exportMLIR(outputPath);
}

void buildMesh6x6Extmem1(const std::string &outputPath) {
  buildSumArrayDemoChess(outputPath, "mesh_6x6_extmem_1", 6, 6);
}

void buildSumArrayDemoChess4x4(const std::string &outputPath) {
  buildSumArrayDemoChess(outputPath, "sum_array_demo_chess_4x4", 4, 4);
}

void buildSumArrayDemoChess5x5(const std::string &outputPath) {
  buildSumArrayDemoChess(outputPath, "sum_array_demo_chess_5x5", 5, 5);
}

void buildSumArrayDemoChess7x7(const std::string &outputPath) {
  buildSumArrayDemoChess(outputPath, "sum_array_demo_chess_7x7", 7, 7);
}

void buildTinyStar4PE(const std::string &outputPath) {
  const std::string moduleName = "tiny_star_4pe";
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName, 4, 4);

  constexpr unsigned kNumPEs = 4;
  constexpr unsigned kNumExtMems = 3;
  constexpr unsigned kScalarInputs = 4;
  constexpr unsigned kScalarOutputs = 2;
  constexpr unsigned kLeafLinkPorts = 1;
  constexpr unsigned kExtMemSWInputs = 2 + 1 + 2;
  constexpr unsigned kExtMemSWOutputs = 2 + 2;

  auto leafSW = builder.defineFullCrossbarSpatialSW(
      moduleName + "_leaf_sw", computePE.outputs + kLeafLinkPorts,
      computePE.inputs + kLeafLinkPorts, kDataWidth);
  auto centerSW = builder.defineFullCrossbarSpatialSW(
      moduleName + "_center_sw",
      kNumPEs * kLeafLinkPorts + kNumExtMems * kExtMemSWInputs +
          kScalarInputs,
      kNumPEs * kLeafLinkPorts + kNumExtMems * kExtMemSWOutputs +
          kScalarOutputs,
      kDataWidth);
  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_mem";
  extMemSpec.ldPorts = 2;
  extMemSpec.stPorts = 1;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 3;
  auto extMem = builder.defineExtMemory(extMemSpec);

  auto centerInst = builder.instantiateSW(centerSW, "sw_center");
  auto leafInsts = builder.instantiateSWArray(kNumPEs, leafSW, "sw_leaf");
  auto peInsts = builder.instantiatePEArray(kNumPEs, computePE.pe, "pe");

  for (unsigned idx = 0; idx < kNumPEs; ++idx) {
    builder.connectPEBankToSwitch(leafInsts[idx], {peInsts[idx]},
                                  computePE.inputs, computePE.outputs);
    builder.connect(leafInsts[idx], computePE.outputs, centerInst, idx);
    builder.connect(centerInst, idx, leafInsts[idx], computePE.inputs);
  }

  auto extMemInsts =
      builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs =
      builder.addMemrefInputs("buffer", kNumExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMemInsts.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMemInsts[idx]);

  unsigned centerInBase = kNumPEs * kLeafLinkPorts;
  unsigned centerOutBase = kNumPEs * kLeafLinkPorts;
  for (unsigned idx = 0; idx < extMemInsts.size(); ++idx) {
    builder.associateExtMemWithSW(extMemInsts[idx], centerInst, centerInBase,
                                  centerOutBase);
    centerInBase += kExtMemSWInputs;
    centerOutBase += kExtMemSWOutputs;
  }

  auto scalarInputs = builder.addInputs(
      "scalar", std::vector<std::string>(kScalarInputs, bitsType()));
  auto scalarOutputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(kScalarOutputs, bitsType()));
  builder.connectInputVectorToInstance(scalarInputs, centerInst, centerInBase);
  builder.connectInstanceToOutputVector(centerInst, centerOutBase,
                                        scalarOutputs);

  auto estimatePEWidth = [&](unsigned fuCount) {
    const double approxFuBoxW = 140.0;
    const double approxFuGap = 12.0;
    const double approxPEPadX = 60.0;
    return std::max(220.0,
                    static_cast<double>(fuCount) * approxFuBoxW +
                        std::max(0.0, static_cast<double>(fuCount) - 1.0) *
                            approxFuGap +
                        approxPEPadX);
  };
  auto estimateSwitchSide = [&](unsigned numInputs, unsigned numOutputs) {
    unsigned maxSideSlots =
        std::max((numInputs + 1) / 2, (numOutputs + 1) / 2);
    return std::max(96.0, 32.0 + (static_cast<double>(maxSideSlots) + 1.0) *
                                     24.0);
  };

  const double peW = estimatePEWidth(computePE.fuCount);
  const double peH = 220.0;
  const double leafSide =
      estimateSwitchSide(computePE.outputs + kLeafLinkPorts,
                         computePE.inputs + kLeafLinkPorts);
  const double centerSide = estimateSwitchSide(
      kNumPEs * kLeafLinkPorts + kNumExtMems * kExtMemSWInputs +
          kScalarInputs,
      kNumPEs * kLeafLinkPorts + kNumExtMems * kExtMemSWOutputs +
          kScalarOutputs);

  builder.setInstanceVizPosition(centerInst, 0.0, 0.0, 0, 0);

  const double peRadius =
      std::hypot(peW / 2.0, peH / 2.0) + leafSide / 2.0 + 320.0;
  const double leafRadius = centerSide / 2.0 + leafSide / 2.0 + 120.0;
  const double angleStep = 2.0 * M_PI / static_cast<double>(kNumPEs);
  for (unsigned idx = 0; idx < kNumPEs; ++idx) {
    const double angle = angleStep * static_cast<double>(idx);
    builder.setInstanceVizPosition(
        leafInsts[idx], leafRadius * std::cos(angle),
        leafRadius * std::sin(angle), 1, static_cast<int>(idx));
    builder.setInstanceVizPosition(
        peInsts[idx], peRadius * std::cos(angle), peRadius * std::sin(angle), 2,
        static_cast<int>(idx));
  }

  setExtMemStripLayout(builder, extMemInsts, -peRadius - 360.0, -180.0, 180.0);
  builder.exportMLIR(outputPath);
}

void buildMediumStar8PE(const std::string &outputPath) {
  buildStarDomain("medium_star_8pe", outputPath, 8, 4, 6, 4);
}

void buildWideStar16PE(const std::string &outputPath) {
  buildStarDomain("wide_star_16pe", outputPath, 16, 6, 8, 4);
}

void buildMesh6x6Extmem2(const std::string &outputPath) {
  const std::string moduleName = "mesh_6x6_extmem_2";
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName, 4, 4);

  constexpr unsigned kRows = 6;
  constexpr unsigned kCols = 6;
  constexpr unsigned kNumExtMems = 2;
  constexpr unsigned kScalarInputs = 4;
  constexpr unsigned kScalarOutputs = 2;
  constexpr unsigned kSideSwitchCount = kCols + 1;

  ChessMeshOptions options;
  options.leftExtraInputsPerSwitch =
      buildSpreadSideCounts(/*totalPorts=*/5, kSideSwitchCount);
  options.leftExtraOutputsPerSwitch =
      buildSpreadSideCounts(/*totalPorts=*/4, kSideSwitchCount);
  options.rightExtraInputsPerSwitch =
      buildSpreadSideCounts(/*totalPorts=*/5, kSideSwitchCount);
  options.rightExtraOutputsPerSwitch =
      buildSpreadSideCounts(/*totalPorts=*/4, kSideSwitchCount);
  options.topExtraInputsPerSwitch =
      buildSpreadSideCounts(kScalarInputs, kSideSwitchCount);
  options.bottomExtraOutputsPerSwitch =
      buildSpreadSideCounts(kScalarOutputs, kSideSwitchCount);
  auto mesh = builder.buildChessMesh(
      kRows, kCols,
      [&](unsigned, unsigned) { return computePE.pe; }, options);

  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_mem";
  extMemSpec.ldPorts = 2;
  extMemSpec.stPorts = 1;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 3;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMemBridge = buildTaggedVecaddExtMemBridge(builder, moduleName);
  auto extMems = builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs =
      builder.addMemrefInputs("buffer", kNumExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  std::vector<TaggedVecaddExtMemBridgeInstances> extMemBridges;
  extMemBridges.reserve(extMems.size());
  extMemBridges.push_back(wireTaggedVecaddExtMemSideBridge(
      builder, mesh.leftIngressPorts, mesh.leftEgressPorts, extMems[0], 0,
      extMemBridge));
  extMemBridges.push_back(wireTaggedVecaddExtMemSideBridge(
      builder, mesh.rightIngressPorts, mesh.rightEgressPorts, extMems[1], 1,
      extMemBridge));

  std::vector<unsigned> inputs = builder.addInputs(
      "scalar", std::vector<std::string>(kScalarInputs, bitsType()));
  std::vector<unsigned> outputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(kScalarOutputs, bitsType()));

  for (unsigned idx = 0; idx < inputs.size(); ++idx)
    builder.connectInputToPort(inputs[idx], mesh.topIngressPorts[idx]);

  for (unsigned idx = 0; idx < outputs.size(); ++idx)
    builder.connectPortToOutput(mesh.bottomEgressPorts[idx], outputs[idx]);

  layoutTaggedVecaddSideExtMemBridges(builder, extMems, extMemBridges, kRows,
                                      kCols);
  builder.exportMLIR(outputPath);
}

void buildMediumChess6x6(const std::string &outputPath) {
  buildChessDomain("medium_chess_6x6", outputPath, 6, 6, 2, 4, 2);
}

void buildMediumChess10x10(const std::string &outputPath) {
  buildChessDomain("medium_chess_10x10", outputPath, 10, 10, 4, 6, 4);
}

} // namespace e2e
} // namespace loom
