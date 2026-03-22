#ifndef LOOM_SIMULATOR_SIMINPUTSYNTHESIS_H
#define LOOM_SIMULATOR_SIMINPUTSYNTHESIS_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace sim {

struct SynthesizedInputPort {
  unsigned portIdx = 0;
  std::string type;
  std::vector<uint64_t> data;
  std::vector<uint16_t> tags;
};

struct SynthesizedMemoryRegion {
  unsigned regionId = 0;
  IdIndex hwNode = INVALID_ID;
  IdIndex swNode = INVALID_ID;
  int64_t memrefArgIndex = -1;
  uint32_t elemSizeLog2 = 0;
  std::vector<uint8_t> data;
};

struct SynthesizedSetup {
  std::vector<SynthesizedInputPort> inputs;
  std::vector<SynthesizedMemoryRegion> memoryRegions;
};

SynthesizedSetup synthesizeSimulationSetup(const Graph &dfg, const Graph &adg,
                                          const MappingState &mapping,
                                          unsigned vectorLength = 4);

bool writeSetupManifest(const SynthesizedSetup &setup,
                        const std::string &path);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMINPUTSYNTHESIS_H
