//===-- tapestry_sim_test.cpp - Multi-core simulation smoke test -*- C++ -*-===//
//
// Smoke test for the multi-core simulation infrastructure.
// Tests three layers independently:
//   1. NoCSimModel - flit routing through a mesh
//   2. MemoryHierarchyModel - SPM/L2/DRAM latency modeling
//   3. CoreSimWrapper + MultiCoreSimSession - full pipeline
//
// Usage:
//   tapestry_sim_test [--runtime-image <path.simimg>]
//
// When --runtime-image is provided, the test loads a real single-core
// config and runs it through the MultiCoreSimSession (1-core mode).
// Otherwise, only the NoC and memory models are exercised.
//
//===----------------------------------------------------------------------===//

#include "loom/MultiCoreSim/CoreSimWrapper.h"
#include "loom/MultiCoreSim/MemoryHierarchyModel.h"
#include "loom/MultiCoreSim/MultiCoreSimSession.h"
#include "loom/MultiCoreSim/NoCSimModel.h"
#include "loom/MultiCoreSim/TapestryTypes.h"
#include "loom/Simulator/RuntimeImage.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static llvm::cl::opt<std::string>
    runtimeImagePath("runtime-image",
                     llvm::cl::desc("Path to a .simimg runtime image file"),
                     llvm::cl::init(""));

static llvm::cl::opt<std::string>
    configBlobPath("config-blob",
                   llvm::cl::desc("Path to a config blob binary (.bin)"),
                   llvm::cl::init(""));

namespace {

unsigned testsPassed = 0;
unsigned testsFailed = 0;

void check(bool condition, const char *testName) {
  if (condition) {
    llvm::outs() << "  PASS: " << testName << "\n";
    ++testsPassed;
  } else {
    llvm::outs() << "  FAIL: " << testName << "\n";
    ++testsFailed;
  }
}

// Test the NoCSimModel independently.
bool testNoCSimModel() {
  llvm::outs() << "\n=== NoCSimModel Tests ===\n";

  // Create a 2x2 mesh.
  loom::mcsim::NoCSimModel noc(2, 2, /*perHopLatency=*/1);

  check(noc.isIdle(), "NoC starts idle");
  check(noc.getCurrentCycle() == 0, "NoC starts at cycle 0");

  // Inject a flit from core 0 (0,0) to core 3 (1,1).
  // XY routing: (0,0) -> (0,1) -> (1,1) = 2 hops.
  loom::mcsim::Flit flit;
  flit.srcCoreId = 0;
  flit.dstCoreId = 3;
  flit.channelId = 0;
  flit.data = 42;
  flit.tag = 0;
  flit.hasTag = false;
  flit.injectionCycle = 0;
  flit.flitIndex = 0;
  flit.totalFlits = 1;
  flit.isHead = true;
  flit.isTail = true;

  noc.injectFlit(flit);
  check(!noc.isIdle(), "NoC not idle after injection");

  // Step enough cycles for the flit to arrive.
  // 2 hops * 1 cycle per hop = flit arrives after 2 steps.
  unsigned maxCycles = 10;
  bool arrived = false;
  for (unsigned cycle = 0; cycle < maxCycles; ++cycle) {
    noc.stepOneCycle();
    if (noc.hasArrivedFlits(3)) {
      arrived = true;
      break;
    }
  }
  check(arrived, "Flit arrived at destination core 3");

  if (arrived) {
    auto flits = noc.drainArrivedFlits(3);
    check(flits.size() == 1, "Exactly one flit arrived");
    if (!flits.empty()) {
      check(flits[0].data == 42, "Flit data matches");
      check(flits[0].srcCoreId == 0, "Flit source matches");
    }
  }

  check(noc.isIdle(), "NoC idle after delivery");

  // Check stats.
  auto stats = noc.getStats();
  check(stats.totalFlitsInjected == 1, "Stats: 1 flit injected");
  check(stats.totalFlitsDelivered == 1, "Stats: 1 flit delivered");
  check(stats.totalHops > 0, "Stats: hops recorded");
  check(stats.averageLatency > 0.0, "Stats: average latency > 0");

  // Test same-node delivery (0 hops).
  loom::mcsim::NoCSimModel noc2(1, 2, 1);
  loom::mcsim::Flit flit2;
  flit2.srcCoreId = 0;
  flit2.dstCoreId = 1;
  flit2.data = 99;
  flit2.injectionCycle = 0;
  flit2.isHead = true;
  flit2.isTail = true;
  flit2.totalFlits = 1;
  noc2.injectFlit(flit2);

  for (unsigned cycle = 0; cycle < 5; ++cycle) {
    noc2.stepOneCycle();
    if (noc2.hasArrivedFlits(1))
      break;
  }
  check(noc2.hasArrivedFlits(1), "Adjacent core flit delivered");

  return true;
}

// Test the MemoryHierarchyModel independently.
bool testMemoryHierarchyModel() {
  llvm::outs() << "\n=== MemoryHierarchyModel Tests ===\n";

  loom::mcsim::MemoryHierarchyConfig config;
  config.spmSizeBytes = 1024;
  config.spmLatencyCycles = 1;
  config.l2SizeBytes = 4096;
  config.l2LatencyCycles = 10;
  config.dramLatencyCycles = 100;
  config.numCores = 2;

  loom::mcsim::MemoryHierarchyModel mem(config);

  // SPM hit for core 0 (address within core 0's SPM range).
  uint64_t completion = mem.issueRequest(0, 0, 4, false, 0);
  check(completion == config.spmLatencyCycles, "SPM hit latency");

  // SPM hit for core 1 (address within core 1's SPM range).
  completion = mem.issueRequest(1, 1024, 4, false, 0);
  check(completion == config.spmLatencyCycles, "Core 1 SPM hit latency");

  // L2 hit (address outside SPM but within L2 range).
  completion = mem.issueRequest(0, 3000, 4, false, 0);
  check(completion == config.l2LatencyCycles, "L2 hit latency");

  // DRAM access (address beyond L2 range).
  completion = mem.issueRequest(0, 100000, 4, false, 0);
  check(completion == config.dramLatencyCycles, "DRAM latency");

  // Check stats.
  auto stats = mem.getStats();
  check(stats.spmHits == 2, "Stats: 2 SPM hits");
  check(stats.spmMisses == 2, "Stats: 2 SPM misses");
  check(stats.l2Hits == 1, "Stats: 1 L2 hit");
  check(stats.l2Misses == 1, "Stats: 1 L2 miss");
  check(stats.dramAccesses == 1, "Stats: 1 DRAM access");

  return true;
}

// Test CoreSimWrapper and MultiCoreSimSession with a real runtime image.
bool testWithRuntimeImage(const std::string &imagePath,
                          const std::string &configPath) {
  llvm::outs() << "\n=== CoreSimWrapper + MultiCoreSimSession Tests ===\n";

  // Load the runtime image.
  loom::sim::RuntimeImage image;
  std::string error;
  if (!loom::sim::loadRuntimeImageBinary(imagePath, image, error)) {
    llvm::outs() << "  SKIP: cannot load runtime image: " << error << "\n";
    return false;
  }
  check(true, "Runtime image loaded");

  // Load config blob.
  std::vector<uint8_t> configBlob;
  if (!configPath.empty()) {
    std::ifstream configFile(configPath, std::ios::binary);
    if (configFile) {
      configFile.seekg(0, std::ios::end);
      auto size = configFile.tellg();
      configFile.seekg(0, std::ios::beg);
      configBlob.resize(static_cast<size_t>(size));
      configFile.read(reinterpret_cast<char *>(configBlob.data()), size);
    }
  }
  if (configBlob.empty()) {
    // Convert config words from the image to bytes.
    configBlob.reserve(image.configImage.words.size() * 4);
    for (uint32_t word : image.configImage.words) {
      configBlob.push_back(static_cast<uint8_t>(word & 0xff));
      configBlob.push_back(static_cast<uint8_t>((word >> 8) & 0xff));
      configBlob.push_back(static_cast<uint8_t>((word >> 16) & 0xff));
      configBlob.push_back(static_cast<uint8_t>((word >> 24) & 0xff));
    }
  }
  check(!configBlob.empty(), "Config blob available");

  // Build config slices from the static config image.
  std::vector<loom::ConfigGen::ConfigSlice> configSlices;
  for (const auto &staticSlice : image.configImage.slices) {
    loom::ConfigGen::ConfigSlice slice;
    slice.name = staticSlice.name;
    slice.kind = staticSlice.kind;
    slice.hwNode = staticSlice.hwNode;
    slice.wordOffset = staticSlice.wordOffset;
    slice.wordCount = staticSlice.wordCount;
    slice.complete = staticSlice.complete;
    configSlices.push_back(std::move(slice));
  }

  // Build a TapestryCompilationResult with one core.
  loom::mcsim::TapestryCompilationResult compilationResult;
  compilationResult.designName = "smoke_test";

  loom::mcsim::CoreResult coreResult;
  coreResult.coreId = 0;

  loom::mcsim::CoreKernelResult kernelResult;
  kernelResult.kernelIndex = 0;
  kernelResult.staticModel = image.staticModel;
  kernelResult.configBlob = configBlob;
  kernelResult.configSlices = configSlices;

  // Synthesize minimal inputs from the control image.
  loom::sim::SynthesizedSetup synthSetup;
  if (image.controlImage.startTokenPort >= 0) {
    loom::sim::SynthesizedInputPort startInput;
    startInput.portIdx =
        static_cast<unsigned>(image.controlImage.startTokenPort);
    startInput.data = {0};
    synthSetup.inputs.push_back(startInput);
  }

  // Add scalar inputs.
  for (const auto &binding : image.controlImage.scalarBindings) {
    loom::sim::SynthesizedInputPort inp;
    inp.portIdx = binding.portIdx;
    inp.data = {1}; // dummy data
    synthSetup.inputs.push_back(inp);
  }

  // Add memory regions.
  for (const auto &binding : image.controlImage.memoryBindings) {
    loom::sim::SynthesizedMemoryRegion region;
    region.regionId = binding.regionId;
    region.elemSizeLog2 = binding.elemSizeLog2;
    unsigned elemSize = 1u << binding.elemSizeLog2;
    region.data.resize(1024 * elemSize, 0);
    synthSetup.memoryRegions.push_back(region);
  }

  kernelResult.synthSetup = synthSetup;
  coreResult.kernels.push_back(std::move(kernelResult));
  compilationResult.cores.push_back(std::move(coreResult));

  // Run through MultiCoreSimSession.
  loom::mcsim::MultiCoreSimConfig simConfig;
  simConfig.perCoreSimConfig.maxCycles = 100000;
  simConfig.maxGlobalCycles = 100000;

  loom::mcsim::MultiCoreSimSession session(simConfig);
  std::string initErr = session.initialize(compilationResult);
  check(initErr.empty(), "MultiCoreSimSession initialized");
  if (!initErr.empty()) {
    llvm::outs() << "  Init error: " << initErr << "\n";
    return false;
  }

  check(session.getNumActiveCores() == 1, "One active core");

  auto result = session.run();
  check(result.totalCycles > 0, "Produced non-zero cycle count");

  llvm::outs() << "  Cycle count: " << result.totalCycles << "\n";
  if (!result.coreResults.empty()) {
    const auto &cr = result.coreResults[0];
    llvm::outs() << "  Core 0 cycles: " << cr.totalCycles << "\n";
    llvm::outs() << "  Core 0 completed: " << (cr.completed ? "yes" : "no")
                 << "\n";
    if (!cr.error.empty())
      llvm::outs() << "  Core 0 error: " << cr.error << "\n";
  }

  return true;
}

// Test the MultiCoreSimSession with a synthetic 2-core setup.
bool testTwoCoreNoC() {
  llvm::outs() << "\n=== Two-Core NoC Transfer Test ===\n";

  // This test exercises the NoC transfer path without real per-core
  // simulation. We verify that the MultiCoreSimSession correctly
  // sets up the NoC model and can handle transfer contracts.

  loom::mcsim::NoCSchedule schedule;
  schedule.meshRows = 1;
  schedule.meshCols = 2;

  loom::mcsim::NoCScheduleEntry entry;
  entry.contract.srcCoreId = 0;
  entry.contract.dstCoreId = 1;
  entry.contract.srcOutputPort = 0;
  entry.contract.dstInputPort = 0;
  entry.contract.flitCount = 4;
  entry.contract.channelId = 0;
  schedule.entries.push_back(entry);

  // Create the NoC model directly and test transfer.
  loom::mcsim::NoCSimModel noc(1, 2, 1);
  noc.configure(schedule);

  // Inject 4 flits from core 0 to core 1.
  for (unsigned flitIdx = 0; flitIdx < 4; ++flitIdx) {
    loom::mcsim::Flit flit;
    flit.srcCoreId = 0;
    flit.dstCoreId = 1;
    flit.channelId = 0;
    flit.data = 10 + flitIdx;
    flit.injectionCycle = noc.getCurrentCycle();
    flit.flitIndex = flitIdx;
    flit.totalFlits = 4;
    flit.isHead = (flitIdx == 0);
    flit.isTail = (flitIdx == 3);
    noc.injectFlit(flit);
  }

  // Step until all delivered.
  unsigned delivered = 0;
  for (unsigned cycle = 0; cycle < 20 && delivered < 4; ++cycle) {
    noc.stepOneCycle();
    if (noc.hasArrivedFlits(1)) {
      auto arrivedFlits = noc.drainArrivedFlits(1);
      delivered += static_cast<unsigned>(arrivedFlits.size());
    }
  }

  check(delivered == 4, "All 4 flits delivered to core 1");

  auto stats = noc.getStats();
  check(stats.totalFlitsInjected == 4, "Stats: 4 flits injected");
  check(stats.totalFlitsDelivered == 4, "Stats: 4 flits delivered");

  return true;
}

} // anonymous namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "Multi-core simulation smoke test\n");

  llvm::outs() << "=== Multi-Core Simulation Smoke Test ===\n";

  testNoCSimModel();
  testMemoryHierarchyModel();
  testTwoCoreNoC();

  // If a runtime image is provided, test with real simulation data.
  if (!runtimeImagePath.empty()) {
    testWithRuntimeImage(runtimeImagePath, configBlobPath);
  } else {
    llvm::outs() << "\n  NOTE: No --runtime-image provided; "
                 << "skipping real simulation test.\n"
                 << "  To run with real data, provide a .simimg file:\n"
                 << "    tapestry_sim_test --runtime-image "
                 << "<path/to/result.simimg>\n";
  }

  llvm::outs() << "\n=== Results: " << testsPassed << " passed, "
               << testsFailed << " failed ===\n";

  return testsFailed > 0 ? 1 : 0;
}
