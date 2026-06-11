#ifndef LOOM_SIMULATOR_CGRA_SIMULATOR_H
#define LOOM_SIMULATOR_CGRA_SIMULATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace loom {
namespace sim {

struct CGRASimOptions {
  std::string dfgReportPath;
  std::string mappingArtifactPath;
  std::string hardwareMlirPath;
};

struct CGRASimReport {
  std::string workload;
  std::string hardware;
  std::string hardwareArtifact;
  std::string mappingId;
  std::string status;
  std::string diagnostic;
  std::string operationSemanticsSource;
  std::string operationCostModelSource;
  std::uint64_t dfgCycles = 0;
  std::uint64_t routeLatencyCycles = 0;
  std::uint64_t memoryLatencyCycles = 0;
  std::uint64_t temporalPenaltyCycles = 0;
  std::uint64_t performanceDeltaCycles = 0;
  std::uint64_t modeledLowerBoundCycles = 0;
  std::uint64_t hardwareAwareCycles = 0;
  std::uint64_t placedRecords = 0;
  std::uint64_t routedEdges = 0;
  std::uint64_t routeSegments = 0;
  std::uint64_t configRecords = 0;
  std::uint64_t spatialPlacements = 0;
  std::uint64_t temporalPlacements = 0;
  std::string functionalStateSource;
  std::vector<std::string> finalOutputs;
  std::map<std::string, std::vector<std::string>> finalMemoryState;
};

llvm::Expected<CGRASimReport> runCGRASimulation(const CGRASimOptions &options);

llvm::Error writeCGRASimReportJson(llvm::StringRef outputPath,
                                   const CGRASimReport &report);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_CGRA_SIMULATOR_H
