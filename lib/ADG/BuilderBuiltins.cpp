#include "BuilderInternal.h"

#include <system_error>

using namespace loom::adg;
using namespace loom::adg::detail;

ModuleBuilder loom::adg::buildFullSpatialCoreAdg() {
  ModuleBuilder module("full_spatialcore_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("mgr_aux", "memref<?x!fabric.bits<64>>")
      .addInput("lhs", "!fabric.bits<32>")
      .addInput("rhs", "!fabric.bits<32>")
      .addInput("addr", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>")
      .addInput("tag", "!fabric.bits<4>")
      .addInput("lhs_t", "!fabric.bits_tag<32, 4>")
      .addInput("rhs_t", "!fabric.bits_tag<32, 4>")
      .addInput("addr_t", "!fabric.bits_tag<32, 4>")
      .addInput("ctrl_t", "!fabric.bits_tag<0, 4>");

  module.addPe(makeMinimalAddPe(Schedule::Spatial, "!fabric.bits<32>",
                                "!fabric.bits<32>"));

  TemporalPeConfig temporal;
  temporal.tagWidth = 4;
  temporal.numInstruction = 2;
  temporal.fuConfigMode = "per_fu_config";
  temporal.operandBufferMode = "per_input_port";
  temporal.operandBufferSize = 2;
  temporal.numRegFifo = 2;
  temporal.regFifoDepth = 4;
  temporal.regFifoPorts = 1;
  module.addPe(makeMinimalAddPe(Schedule::Temporal, "lhs_t", "rhs_t",
                                "!fabric.bits_tag<32, 4>", "!fabric.bits<32>",
                                std::move(temporal)));

  module.addSwitch(SwitchSpec{Schedule::Spatial,
                              {"lhs", "rhs"},
                              {"!fabric.bits<32>", "!fabric.bits<32>"},
                              {"11", "11"},
                              0});
  module.addSwitch(
      SwitchSpec{Schedule::Temporal,
                 {"lhs_t", "rhs_t"},
                 {"!fabric.bits_tag<32, 4>", "!fabric.bits_tag<32, 4>"},
                 {"11", "11"},
                 2});

  MemSpec spatialMem(Schedule::Spatial, {"mgr"}, {},
                     MemDispatchEligibility{{{0}, {0}}, {}});
  spatialMem.loads = {{"addr", "ctrl"}};
  spatialMem.stores = {{"addr", "lhs", "ctrl"}};
  spatialMem.dataWidth = 32;
  module.addMem(std::move(spatialMem));

  MemSpec temporalMem(
      Schedule::Temporal, {"mgr", "mgr_aux"},
      {{"temporal_subordinate0", "memref<?x!fabric.bits<8>>"},
       {"temporal_subordinate1", "memref<?x!fabric.bits<16>>"}},
      MemDispatchEligibility{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}}});
  temporalMem.loads = {{"addr_t", "ctrl_t"}};
  temporalMem.stores = {{"addr_t", "lhs_t", "ctrl_t"}};
  temporalMem.dataWidth = 32;
  temporalMem.temporalTagWidth = 4;
  temporalMem.temporalOperationTableSize = 2;
  module.addMem(std::move(temporalMem));
  module.addOutput("temporal_subordinate1");

  std::vector<BodyResultSpec> taggedResults = {
      BodyResultSpec{"tagged", "!fabric.bits_tag<32, 4>"}};
  appendBodyOp(
      module,
      BodyOpSpec{taggedResults,
                 {directBodyLine({"fabric.boundary [s2t] ", ", ",
                                  " : (!fabric.bits<32>, !fabric.bits<4>) -> " +
                                      bodyResultTypes(taggedResults)},
                                 {"lhs", "tag"})}});
  addFifo(module, "queued", "tagged", "!fabric.bits_tag<32, 4>",
          "!fabric.bits_tag<32, 4>", 4, true);
  appendBodyOp(
      module,
      BodyOpSpec{
          {},
          {exactBodyLine("fabric.pe @ALU [spatial] (!fabric.bits<32>) -> "
                         "(!fabric.bits<32>) {"),
           nestedBodyLine("^bb0(%pa: !fabric.bits<32>):"),
           nestedBodyLine("  fabric.fu(%fa = %pa : !fabric.bits<32>) -> "
                          "(!fabric.bits<32>) {"),
           nestedBodyLine(
               "    %v = fabric.op [@arith.addi] (%fa, %fa) : "
               "(!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>"),
           nestedBodyLine("    fabric.yield %v : !fabric.bits<32>"),
           nestedBodyLine("  }"),
           nestedBodyLine("  fabric.yield %pa : !fabric.bits<32>"),
           nestedBodyLine("}")}});
  std::vector<BodyResultSpec> instantiateResults = {
      BodyResultSpec{"inst", "!fabric.bits<32>"}};
  appendBodyOp(
      module,
      BodyOpSpec{instantiateResults,
                 {directBodyLine({"fabric.instantiate @ALU(",
                                  " : !fabric.bits<32>) -> " +
                                      bodyResultTypes(instantiateResults)},
                                 {"lhs"})}});
  return module;
}

SystemBuilder loom::adg::buildHeterogeneousSocAdg() {
  SystemBuilder system("heterogeneous_dual_accel_soc", "sequential");
  system.addHostCore("host0", "rv64gc", axiManagerPort("mem"));
  system.addSpatialAccelerator("acc0", "shared_reduction_adg", "rv32im",
                               axiManagerPort("mem"));
  system.addFixedAccelerator("fft0", "fft", axiManagerPort("mem"));

  std::vector<std::string> cachePorts;
  appendPorts(cachePorts, axiSubordinatePort("host"));
  appendPorts(cachePorts, axiManagerPort("mem"));
  system.addCache("l1d0", 64, 32 * 1024, std::move(cachePorts));

  std::vector<std::string> dmaPorts;
  appendPorts(dmaPorts, axiSubordinatePort("ctrl"));
  appendPorts(dmaPorts, axiManagerPort("mem"));
  system.addDmaEngine("dma0", 4, std::move(dmaPorts));

  std::vector<std::string> dramPorts;
  appendPorts(dramPorts, axiSubordinatePort("cache"));
  appendPorts(dramPorts, axiSubordinatePort("acc0"));
  appendPorts(dramPorts, axiSubordinatePort("fft0"));
  appendPorts(dramPorts, axiSubordinatePort("dma0"));
  system.addMemory("dram0", 1024 * 1024, std::move(dramPorts));

  connectAxiMemoryPort(system, "host0", "mem", "l1d0", "host");
  connectAxiMemoryPort(system, "l1d0", "mem", "dram0", "cache");
  connectAxiMemoryPort(system, "acc0", "mem", "dram0", "acc0");
  connectAxiMemoryPort(system, "fft0", "mem", "dram0", "fft0");
  connectAxiMemoryPort(system, "dma0", "mem", "dram0", "dma0");
  return system;
}

llvm::Error loom::adg::writeMinimalSpatialAdg(llvm::raw_ostream &os) {
  return buildMinimalSpatialAdg().print(os);
}

llvm::Error loom::adg::writeMinimalTemporalAdg(llvm::raw_ostream &os) {
  return buildMinimalTemporalAdg().print(os);
}

llvm::Error loom::adg::writeSharedReductionAdg(llvm::raw_ostream &os) {
  return buildSharedReductionAdg().print(os);
}

llvm::Error loom::adg::writeSharedMemoryReductionAdg(llvm::raw_ostream &os) {
  return buildSharedMemoryReductionAdg().print(os);
}

llvm::Error loom::adg::writeSharedQuantizedWindowAdg(llvm::raw_ostream &os) {
  return buildSharedQuantizedWindowAdg().print(os);
}

llvm::Error loom::adg::writeSharedSignalWindowAdg(llvm::raw_ostream &os) {
  return buildSharedSignalWindowAdg().print(os);
}

llvm::Error loom::adg::writeSharedVectorAluAdg(llvm::raw_ostream &os) {
  return buildSharedVectorAluAdg().print(os);
}

llvm::Error loom::adg::writeSharedVectorMathAdg(llvm::raw_ostream &os) {
  return buildSharedVectorMathAdg().print(os);
}

llvm::Error loom::adg::writeSharedVectorMeshAdg(llvm::raw_ostream &os) {
  return buildSharedVectorMeshAdg().print(os);
}

llvm::Error loom::adg::writeFullSpatialCoreAdg(llvm::raw_ostream &os) {
  return buildFullSpatialCoreAdg().print(os);
}

llvm::Error loom::adg::writeHeterogeneousSocAdg(llvm::raw_ostream &os) {
  if (llvm::Error err = printReusableSpatialTemplates(os, false))
    return err;
  return buildHeterogeneousSocAdg().print(os);
}

llvm::Error loom::adg::writeSpatialTopologyMatrixAdg(llvm::raw_ostream &os,
                                                     llvm::StringRef family) {
  if (family == "chain-1d")
    return buildChain1DAdg().print(os);
  if (family == "mesh-2d")
    return buildMesh2DAdg().print(os);
  if (family == "torus-edge")
    return buildTorusEdgeAdg().print(os);
  if (family == "systolic-array")
    return buildSystolicArrayAdg().print(os);
  if (family == "clustered-array")
    return buildClusteredArrayAdg().print(os);
  if (family == "folded-ring")
    return buildFoldedRingAdg().print(os);
  if (family == "mesh-diagonal")
    return buildMeshDiagonalAdg().print(os);
  if (family == "multi-lane-pipeline")
    return buildMultiLanePipelineAdg().print(os);
  if (family == "reduction-tree")
    return buildReductionTreeAdg().print(os);
  if (family == "cross-coupled-switch")
    return buildCrossCoupledSwitchAdg().print(os);
  if (family == "diamond-bypass")
    return buildDiamondBypassAdg().print(os);
  if (family == "memory-fanout")
    return buildMemoryFanoutAdg().print(os);
  if (family == "mixed-temporal-bridge")
    return buildMixedTemporalBridgeAdg().print(os);
  if (family == "sparse-long-link")
    return buildSparseLongLinkAdg().print(os);
  if (family == "heterogeneous-islands")
    return buildHeterogeneousIslandsAdg().print(os);
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unknown topology matrix case %s",
                                 family.str().c_str());
}

llvm::Error loom::adg::writeSystemTopologyMatrixAdg(llvm::raw_ostream &os,
                                                    llvm::StringRef family) {
  if (family == "dual-spatial-shared-memory") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildDualSpatialSharedMemorySocAdg().print(os);
  }
  if (family == "cached-dual-accel") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildCachedDualAccelSocAdg().print(os);
  }
  if (family == "dma-scratchpad") {
    if (llvm::Error err = printReusableSpatialTemplates(os, false))
      return err;
    return buildDmaScratchpadSocAdg().print(os);
  }
  if (family == "fixed-and-spatial") {
    if (llvm::Error err = printReusableSpatialTemplates(os, false))
      return err;
    return buildFixedAndSpatialSocAdg().print(os);
  }
  if (family == "tri-spatial-shared-memory") {
    if (llvm::Error err =
            printReusableSpatialTemplates(os, true, true, false, false))
      return err;
    return buildTriSpatialSharedMemorySocAdg().print(os);
  }
  if (family == "dual-host-shared-memory") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildDualHostSharedMemorySocAdg().print(os);
  }
  if (family == "private-scratchpad-pair") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildPrivateScratchpadPairSocAdg().print(os);
  }
  if (family == "host-cache-dual-memory") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildHostCacheDualMemorySocAdg().print(os);
  }
  if (family == "dma-dual-memory") {
    if (llvm::Error err = printReusableSpatialTemplates(os, false))
      return err;
    return buildDmaDualMemorySocAdg().print(os);
  }
  if (family == "cached-accelerator-cluster") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildCachedAcceleratorClusterSocAdg().print(os);
  }
  if (family == "mixed-fixed-spatial-pipeline") {
    if (llvm::Error err = printReusableSpatialTemplates(os, true))
      return err;
    return buildMixedFixedSpatialPipelineSocAdg().print(os);
  }
  if (family == "signal-quantized-pair") {
    if (llvm::Error err =
            printReusableSpatialTemplates(os, false, false, true, true))
      return err;
    return buildSignalQuantizedPairSocAdg().print(os);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unknown system topology matrix case %s",
                                 family.str().c_str());
}
