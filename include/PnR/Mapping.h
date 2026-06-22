#ifndef LOOM_PNR_MAPPING_H
#define LOOM_PNR_MAPPING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom {
namespace pnr {

struct MappingOptions {
  std::string dfgMlirPath;
  std::string graphName;
  std::string hardwareMlirPath;
  std::string hardwareName;
  std::string workload;
};

struct PlacementRecord {
  std::string softwareId;
  std::string operation;
  std::string resourceKind;
  std::string hardwareId;
  std::string schedule;
};

struct RouteSegment {
  std::string segmentId;
  std::string segmentKind;
  std::string sourceEndpoint;
  std::string sinkEndpoint;
  std::string hardwareRef;
};

struct RouteRecord {
  std::string recordId;
  std::string edgeRef;
  std::string producerBinding;
  std::string consumerBinding;
  std::string payloadKind;
  std::string fromSoftwareId;
  std::string toSoftwareId;
  llvm::SmallVector<RouteSegment, 1> segments;
};

struct UnroutedEdgeRecord {
  std::string edgeRef;
  std::string producerBinding;
  std::string consumerBinding;
  std::string payloadKind;
  std::string fromSoftwareId;
  std::string toSoftwareId;
  std::string sourceEndpoint;
  std::string sinkEndpoint;
  std::string diagnostic;
};

struct ConfigEntry {
  std::string target;
  std::string registerName;
  std::string value;
  std::string source;
};

struct ResourcePressureRecord {
  std::string resourceKind;
  std::string operation;
  std::uint64_t required = 0;
  std::uint64_t available = 0;
  std::uint64_t placed = 0;
  std::uint64_t missing = 0;
};

struct MappingSummary {
  std::string workload;
  std::string hardware;
  std::string graph;
  std::string mappingId;
  std::string configId;
  std::string configFingerprint;
  std::string componentConfigView;
  std::string componentConfigFingerprint;
  std::string status;
  std::string diagnostic;
  llvm::SmallVector<PlacementRecord> placements;
  llvm::SmallVector<RouteRecord, 0> routes;
  llvm::SmallVector<UnroutedEdgeRecord, 0> unroutedEdgeDetails;
  llvm::SmallVector<ConfigEntry> configEntries;
  llvm::SmallVector<ResourcePressureRecord, 0> resourcePressure;
  std::uint64_t unplacedRecords = 0;
  std::uint64_t unroutedEdges = 0;
};

llvm::Expected<MappingSummary> createMapping(const MappingOptions &options);

llvm::Error writeMappingCsv(llvm::StringRef outputPath,
                            llvm::ArrayRef<MappingSummary> summaries);

llvm::Error writeMappingJson(llvm::StringRef outputPath,
                             const MappingSummary &summary);

} // namespace pnr
} // namespace loom

#endif // LOOM_PNR_MAPPING_H
