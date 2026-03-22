#ifndef LOOM_SIMULATOR_SIMBUNDLE_H
#define LOOM_SIMULATOR_SIMBUNDLE_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Simulator/SimInputSynthesis.h"
#include "loom/Simulator/SimSession.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace sim {

struct SimBundleScalarInput {
  int64_t argIndex = -1;
  std::vector<uint64_t> data;
  std::vector<uint16_t> tags;
};

struct SimBundleMemoryImage {
  int64_t memrefArgIndex = -1;
  uint32_t elemSizeBytes = 0;
  std::vector<uint64_t> values;
};

struct SimBundleExpectedOutput {
  int64_t resultIndex = -1;
  std::vector<uint64_t> data;
  std::vector<uint16_t> tags;
};

struct SimBundleExpectedMemory {
  int64_t memrefArgIndex = -1;
  uint32_t elemSizeBytes = 0;
  std::vector<uint64_t> values;
};

struct SimulationBundle {
  std::string caseName;
  unsigned startTokenCount = 1;
  std::vector<SimBundleScalarInput> scalarInputs;
  std::vector<SimBundleMemoryImage> memoryRegions;
  std::vector<SimBundleExpectedOutput> expectedOutputs;
  std::vector<SimBundleExpectedMemory> expectedMemoryRegions;
};

struct ResolvedExpectedOutput {
  unsigned portIdx = 0;
  int64_t resultIndex = -1;
  std::vector<uint64_t> data;
  std::vector<uint16_t> tags;
};

struct ResolvedExpectedMemory {
  unsigned regionId = 0;
  int64_t memrefArgIndex = -1;
  std::vector<uint8_t> data;
};

struct ResolvedSimulationBundle {
  std::string caseName;
  SynthesizedSetup setup;
  std::vector<ResolvedExpectedOutput> expectedOutputs;
  std::vector<ResolvedExpectedMemory> expectedMemoryRegions;
};

struct SimValidationReport {
  bool pass = false;
  unsigned totalChecks = 0;
  unsigned mismatches = 0;
  std::vector<std::string> diagnostics;
};

bool loadSimulationBundle(const std::string &path, SimulationBundle &bundle,
                          std::string &error);

bool resolveSimulationBundle(const SimulationBundle &bundle, const Graph &dfg,
                             const Graph &adg, const MappingState &mapping,
                             ResolvedSimulationBundle &resolved,
                             std::string &error);

SimValidationReport validateSimulationBundle(
    const SimSession &session, const ResolvedSimulationBundle &bundle);

bool writeValidationReport(const SimValidationReport &report,
                           const std::string &path);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMBUNDLE_H
