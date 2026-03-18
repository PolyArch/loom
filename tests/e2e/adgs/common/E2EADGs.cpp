#include "E2EADGs.h"

#include "fcc/ADG/ADGBuilder.h"

#include <cmath>
#include <string>
#include <vector>

using namespace fcc::adg;

namespace fcc {
namespace e2e {

namespace {

constexpr unsigned kDataWidth = 64;
constexpr unsigned kPEInputs = 24;
constexpr unsigned kPEOutputs = 24;

std::string bitsType(unsigned width = kDataWidth) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

struct SpatialKernelPE {
  PEHandle pe;
  unsigned inputs = kPEInputs;
  unsigned outputs = kPEOutputs;
  unsigned fuCount = 0;
};

SpatialKernelPE buildSpatialKernelPE(ADGBuilder &builder,
                                     const std::string &prefix) {
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
      std::vector<std::string>(kPEInputs, bitsType()),
      std::vector<std::string>(kPEOutputs, bitsType()),
      fus);
  return {pe, kPEInputs, kPEOutputs,
          static_cast<unsigned>(fus.size())};
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

void wireExtMemIngressEgress(ADGBuilder &builder, const MeshResult &mesh,
                             const std::vector<InstanceHandle> &extMems,
                             unsigned ingressBase, unsigned egressBase) {
  unsigned ingressIdx = ingressBase;
  unsigned egressIdx = egressBase;
  for (InstanceHandle mem : extMems) {
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
  options.topLeftExtraInputs = numExtMems * 5 + scalarInputs;
  options.bottomRightExtraOutputs = numExtMems * 4 + scalarOutputs;
  auto mesh = builder.buildChessMesh(
      rows, cols,
      [&](unsigned, unsigned) { return computePE.pe; }, options);

  auto extMem = builder.defineExtMemory(moduleName + "_mem", 2, 1);
  auto extMems = builder.instantiateExtMemArray(numExtMems, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", numExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  wireExtMemIngressEgress(builder, mesh, extMems, 0, 0);

  std::vector<unsigned> inputs = builder.addInputs(
      "scalar", std::vector<std::string>(scalarInputs, bitsType()));
  std::vector<unsigned> outputs = builder.addOutputs(
      "scalar_out", std::vector<std::string>(scalarOutputs, bitsType()));

  unsigned ingressIdx = numExtMems * 5;
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

void buildTinyStar4PE(const std::string &outputPath) {
  buildStarDomain("tiny_star_4pe", outputPath, 4, 3, 4, 2);
}

void buildMediumStar8PE(const std::string &outputPath) {
  buildStarDomain("medium_star_8pe", outputPath, 8, 4, 6, 4);
}

void buildWideStar16PE(const std::string &outputPath) {
  buildStarDomain("wide_star_16pe", outputPath, 16, 6, 8, 4);
}

void buildMediumChess6x6(const std::string &outputPath) {
  buildChessDomain("medium_chess_6x6", outputPath, 6, 6, 2, 4, 2);
}

void buildMediumChess10x10(const std::string &outputPath) {
  buildChessDomain("medium_chess_10x10", outputPath, 10, 10, 4, 6, 4);
}

} // namespace e2e
} // namespace fcc
