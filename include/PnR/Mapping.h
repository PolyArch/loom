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

struct RouteRecord {
  std::string fromSoftwareId;
  std::string toSoftwareId;
};

struct ConfigEntry {
  std::string target;
  std::string registerName;
  std::string value;
  std::string source;
};

struct MappingSummary {
  std::string workload;
  std::string hardware;
  std::string graph;
  std::string mappingId;
  std::string status;
  std::string diagnostic;
  llvm::SmallVector<PlacementRecord> placements;
  llvm::SmallVector<RouteRecord> routes;
  llvm::SmallVector<ConfigEntry> configEntries;
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
