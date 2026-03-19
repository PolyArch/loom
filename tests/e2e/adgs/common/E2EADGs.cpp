#include "E2EADGs.h"

#include "fcc/ADG/ADGBuilder.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace fcc::adg;

namespace fcc {
namespace e2e {

namespace {

constexpr unsigned kDataWidth = 64;
constexpr unsigned kDensePEInputs = 24;
constexpr unsigned kDensePEOutputs = 24;

std::string bitsType(unsigned width = kDataWidth) {
  return "!fabric.bits<" + std::to_string(width) + ">";
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
  auto fuAdd = builder.defineBinaryFU(prefix + "_add", "arith.addi", "i32",
                                      "i32");
  auto fuSub = builder.defineBinaryFU(prefix + "_sub", "arith.subi", "i32",
                                      "i32");
  auto fuMul = builder.defineBinaryFU(prefix + "_mul", "arith.muli", "i32",
                                      "i32");
  auto fuCmp = builder.defineCmpiFU(prefix + "_cmp", "i32", "slt");
  auto fuSelect = builder.defineSelectFU(prefix + "_select", "i32");
  auto fuConst = builder.defineConstantFU(prefix + "_const1", "i32",
                                          "1 : i32");
  auto fuConstIndex =
      builder.defineConstantFU(prefix + "_const_index", "index", "0 : index");
  auto fuIndexToI32 =
      builder.defineIndexCastFU(prefix + "_index_to_i32", "index", "i32");
  auto fuI32ToIndex =
      builder.defineIndexCastFU(prefix + "_i32_to_index", "i32", "index");
  auto fuStream = builder.defineStreamFU(prefix + "_stream");
  auto fuMux = builder.defineMuxFU(prefix + "_mux", "i32");
  auto fuJoin = builder.defineJoinFU(prefix + "_join", 4);
  auto fuGate = builder.defineGateFU(prefix + "_gate", "i32");
  auto fuGateIndex = builder.defineGateFU(prefix + "_gate_index", "index");
  auto fuCarry = builder.defineCarryFU(prefix + "_carry", "none");
  auto fuCondBr = builder.defineCondBrFU(prefix + "_cond_br", "none");
  auto fuLoad = builder.defineLoadFU(prefix + "_load", "index", "i32");
  auto fuStore = builder.defineStoreFU(prefix + "_store", "index", "i32");
  auto fuInvariant = builder.defineInvariantFU(prefix + "_invariant", "i32");

  std::vector<FUHandle> fus = {
      fuAdd,        fuSub,        fuMul,         fuCmp,
      fuSelect,     fuConst,      fuConstIndex,  fuIndexToI32,
      fuI32ToIndex, fuStream,     fuMux,         fuJoin,
      fuGate,       fuGateIndex,  fuCarry,       fuCondBr,
      fuLoad,       fuStore,      fuInvariant,
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
  auto extMem = builder.defineExtMemory(moduleName + "_mem", 2, 1);

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

void wireRoundRobinExtMemIngressEgress(
    ADGBuilder &builder, const MeshResult &mesh,
    const std::vector<InstanceHandle> &extMems, unsigned topLeftIngressCount,
    unsigned bottomLeftEgressCount) {
  unsigned leftIngressIdx = 0;
  unsigned rightIngressIdx = topLeftIngressCount;
  unsigned leftEgressIdx = 0;
  unsigned rightEgressIdx = bottomLeftEgressCount;

  for (unsigned memIdx = 0; memIdx < extMems.size(); ++memIdx) {
    InstanceHandle mem = extMems[memIdx];
    unsigned &ingressIdx = (memIdx % 2 == 0) ? leftIngressIdx : rightIngressIdx;
    unsigned &egressIdx = (memIdx % 2 == 0) ? leftEgressIdx : rightEgressIdx;

    for (unsigned outPort = 0; outPort < 5; ++outPort) {
      builder.connect(mem, outPort, mesh.ingressPorts[ingressIdx].instance,
                      mesh.ingressPorts[ingressIdx].port);
      ++ingressIdx;
    }
    for (unsigned inPort = 0; inPort < 4; ++inPort) {
      builder.connect(mesh.egressPorts[egressIdx].instance,
                      mesh.egressPorts[egressIdx].port, mem, 1 + inPort);
      ++egressIdx;
    }
  }
}

void buildChessDomain(const std::string &moduleName,
                      const std::string &outputPath, unsigned rows,
                      unsigned cols, unsigned numExtMems,
                      unsigned scalarInputs, unsigned scalarOutputs) {
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName);

  ChessMeshOptions options;
  const unsigned leftIngressMems = (numExtMems + 1) / 2;
  const unsigned rightIngressMems = numExtMems / 2;
  const unsigned leftEgressMems = (numExtMems + 1) / 2;
  const unsigned rightEgressMems = numExtMems / 2;
  options.topLeftExtraInputs = leftIngressMems * 5 + scalarInputs;
  options.topRightExtraInputs = rightIngressMems * 5;
  options.bottomLeftExtraOutputs = leftEgressMems * 4;
  options.bottomRightExtraOutputs = rightEgressMems * 4 + scalarOutputs;
  auto mesh = builder.buildChessMesh(
      rows, cols,
      [&](unsigned, unsigned) { return computePE.pe; }, options);

  auto extMem = builder.defineExtMemory(moduleName + "_mem", 2, 1);
  auto extMems = builder.instantiateExtMemArray(numExtMems, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", numExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  wireRoundRobinExtMemIngressEgress(builder, mesh, extMems,
                                    options.topLeftExtraInputs,
                                    options.bottomLeftExtraOutputs);

  std::vector<unsigned> inputs = builder.addInputs(
      "scalar", std::vector<std::string>(scalarInputs, bitsType()));
  std::vector<unsigned> outputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(scalarOutputs, bitsType()));

  unsigned ingressIdx = leftIngressMems * 5;
  for (unsigned idx = 0; idx < inputs.size(); ++idx, ++ingressIdx)
    builder.connectInputToPort(inputs[idx], mesh.ingressPorts[ingressIdx]);

  unsigned egressIdx = numExtMems * 4;
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

void buildSumArrayDemoChess6x6(const std::string &outputPath) {
  buildSumArrayDemoChess(outputPath, "sum_array_demo_chess_6x6", 6, 6);
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
  auto extMem = builder.defineExtMemory(moduleName + "_mem", 2, 1);

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

void buildVecaddDemoChess6x6(const std::string &outputPath) {
  const std::string moduleName = "vecadd_demo_chess_6x6";
  ADGBuilder builder(moduleName);
  auto computePE = buildSpatialKernelPE(builder, moduleName, 4, 4);

  constexpr unsigned kRows = 6;
  constexpr unsigned kCols = 6;
  constexpr unsigned kNumExtMems = 3;
  constexpr unsigned kScalarInputs = 2;
  constexpr unsigned kScalarOutputs = 1;

  ChessMeshOptions options;
  const unsigned leftIngressMems = (kNumExtMems + 1) / 2;
  const unsigned rightIngressMems = kNumExtMems / 2;
  const unsigned leftEgressMems = (kNumExtMems + 1) / 2;
  const unsigned rightEgressMems = kNumExtMems / 2;
  options.topLeftExtraInputs = leftIngressMems * 5 + kScalarInputs;
  options.topRightExtraInputs = rightIngressMems * 5;
  options.bottomLeftExtraOutputs = leftEgressMems * 4;
  options.bottomRightExtraOutputs = rightEgressMems * 4 + kScalarOutputs;
  auto mesh = builder.buildChessMesh(
      kRows, kCols,
      [&](unsigned, unsigned) { return computePE.pe; }, options);

  auto extMem = builder.defineExtMemory(moduleName + "_mem", 2, 1);
  auto extMems = builder.instantiateExtMemArray(kNumExtMems, extMem, "extmem");
  auto memrefs =
      builder.addMemrefInputs("buffer", kNumExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  wireRoundRobinExtMemIngressEgress(builder, mesh, extMems,
                                    options.topLeftExtraInputs,
                                    options.bottomLeftExtraOutputs);

  std::vector<unsigned> inputs = builder.addInputs(
      "scalar", std::vector<std::string>(kScalarInputs, bitsType()));
  std::vector<unsigned> outputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(kScalarOutputs, bitsType()));

  unsigned ingressIdx = leftIngressMems * 5;
  for (unsigned idx = 0; idx < inputs.size(); ++idx, ++ingressIdx)
    builder.connectInputToPort(inputs[idx], mesh.ingressPorts[ingressIdx]);

  unsigned egressIdx = kNumExtMems * 4;
  for (unsigned idx = 0; idx < outputs.size(); ++idx, ++egressIdx)
    builder.connectPortToOutput(mesh.egressPorts[egressIdx], outputs[idx]);

  setExtMemStripLayout(builder, extMems, -420.0,
                       -220.0 + static_cast<double>(kRows) * 8.0, 180.0);
  builder.exportMLIR(outputPath);
}

void buildMediumChess6x6(const std::string &outputPath) {
  buildChessDomain("medium_chess_6x6", outputPath, 6, 6, 2, 4, 2);
}

void buildMediumChess10x10(const std::string &outputPath) {
  buildChessDomain("medium_chess_10x10", outputPath, 10, 10, 4, 6, 4);
}

} // namespace e2e
} // namespace fcc
