#ifndef LOOM_PNR_MAPPING_ESTIMATOR_H
#define LOOM_PNR_MAPPING_ESTIMATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom {
namespace pnr {

struct MappingEstimateOptions {
  std::string mappingArtifactPath;
  std::string hardwareMlirPath;
};

struct MappingEstimateReport {
  std::string workload;
  std::string hardware;
  std::string hardwareArtifact;
  std::string mappingId;
  std::string configId;
  std::string configFingerprint;
  std::string componentConfigView;
  std::string componentConfigFingerprint;
  std::string status;
  std::string diagnostic;
  std::uint64_t routeSegmentScore = 0;
  std::uint64_t memoryAccessScore = 0;
  std::uint64_t widthAdapterScore = 0;
  std::uint64_t functionalUnitScore = 0;
  std::uint64_t resourceMixScore = 0;
  std::uint64_t loadAddressScore = 0;
  std::uint64_t storeAddressScore = 0;
  std::uint64_t configLoadScore = 0;
  std::uint64_t temporalConflictScore = 0;
  std::uint64_t totalCostScore = 0;
  std::uint64_t placedRecords = 0;
  std::uint64_t routedEdges = 0;
  std::uint64_t routeSegments = 0;
  std::uint64_t configRecords = 0;
  std::uint64_t spatialPlacements = 0;
  std::uint64_t temporalPlacements = 0;
};

llvm::Expected<MappingEstimateReport>
estimateMapping(const MappingEstimateOptions &options);

llvm::Error writeMappingEstimateReportJson(llvm::StringRef outputPath,
                                           const MappingEstimateReport &report);

} // namespace pnr
} // namespace loom

#endif // LOOM_PNR_MAPPING_ESTIMATOR_H
