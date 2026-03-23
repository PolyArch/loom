#ifndef LOOM_MULTICORESIM_TAPESTRYTYPES_H
#define LOOM_MULTICORESIM_TAPESTRYTYPES_H

#include "loom/Mapper/ConfigGen.h"
#include "loom/Simulator/RuntimeImage.h"
#include "loom/Simulator/SimInputSynthesis.h"
#include "loom/Simulator/SimTypes.h"
#include "loom/Simulator/StaticModel.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

// Describes a single kernel mapped onto one core.
struct CoreKernelResult {
  unsigned kernelIndex = 0;
  sim::StaticMappedModel staticModel;
  std::vector<uint8_t> configBlob;
  std::vector<loom::ConfigGen::ConfigSlice> configSlices;
  sim::SynthesizedSetup synthSetup;
};

// Per-core compilation result: one ADG, zero or more mapped kernels.
struct CoreResult {
  unsigned coreId = 0;
  std::vector<CoreKernelResult> kernels;
};

// A single cross-core data transfer contract.
struct NoCTransferContract {
  unsigned srcCoreId = 0;
  unsigned dstCoreId = 0;
  unsigned srcOutputPort = 0;
  unsigned dstInputPort = 0;
  unsigned flitCount = 0;
  unsigned channelId = 0;
};

// Per-hop route through the mesh NoC.
struct NoCRouteHop {
  unsigned srcRow = 0;
  unsigned srcCol = 0;
  unsigned dstRow = 0;
  unsigned dstCol = 0;
  unsigned virtualChannel = 0;
};

// Complete schedule for one transfer through the NoC.
struct NoCScheduleEntry {
  NoCTransferContract contract;
  std::vector<NoCRouteHop> route;
};

// The complete schedule of all inter-core transfers.
struct NoCSchedule {
  unsigned meshRows = 1;
  unsigned meshCols = 1;
  std::vector<NoCScheduleEntry> entries;
};

// Complete multi-core compilation result.
struct TapestryCompilationResult {
  std::vector<CoreResult> cores;
  NoCSchedule nocSchedule;
  std::string designName;
};

// Result of multi-core simulation for one core.
struct CoreSimResult {
  unsigned coreId = 0;
  sim::SimResult simResult;
  uint64_t totalCycles = 0;
  bool completed = false;
  std::string error;
};

// NoC traffic statistics.
struct NoCStats {
  uint64_t totalFlitsInjected = 0;
  uint64_t totalFlitsDelivered = 0;
  uint64_t totalHops = 0;
  double averageLatency = 0.0;
  double linkUtilization = 0.0;
};

// Memory hierarchy statistics.
struct MemoryHierarchyStats {
  uint64_t spmHits = 0;
  uint64_t spmMisses = 0;
  uint64_t l2Hits = 0;
  uint64_t l2Misses = 0;
  uint64_t dramAccesses = 0;
};

// Complete multi-core simulation result.
struct MultiCoreSimResult {
  std::vector<CoreSimResult> coreResults;
  NoCStats nocStats;
  MemoryHierarchyStats memStats;
  uint64_t totalCycles = 0;
  bool allCoresCompleted = false;
  std::string error;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_TAPESTRYTYPES_H
