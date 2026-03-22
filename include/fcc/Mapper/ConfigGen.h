#ifndef FCC_MAPPER_CONFIGGEN_H
#define FCC_MAPPER_CONFIGGEN_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MapperTiming.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/TechMapper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <cstdint>
#include <vector>

namespace fcc {

struct MapperSearchSummary;

class ConfigGen {
public:
  struct ConfigSlice {
    std::string name;
    std::string kind;
    IdIndex hwNode = INVALID_ID;
    uint32_t wordOffset = 0;
    uint32_t wordCount = 0;
    bool complete = true;
  };

  /// Generate config artifacts together with mapping reports.
  bool generate(const MappingState &state, const Graph &dfg, const Graph &adg,
                const ADGFlattener &flattener,
                llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                llvm::ArrayRef<FUConfigSelection> fuConfigs,
                const std::string &basePath, int seed,
                const TechMapper::Plan *techMapPlan = nullptr,
                const TechMapper::PlanMetrics *techMapMetrics = nullptr,
                const MapperTimingSummary *timingSummary = nullptr,
                const MapperSearchSummary *searchSummary = nullptr,
                llvm::StringRef techMapDiagnostics = "");

  const std::vector<uint8_t> &getConfigBlob() const { return configBlob_; }
  const std::vector<uint32_t> &getConfigWords() const { return configWords_; }
  const std::vector<ConfigSlice> &getConfigSlices() const {
    return configSlices_;
  }
  bool isConfigComplete() const { return configComplete_; }
  uint32_t getConfigWordCount() const {
    return static_cast<uint32_t>(configWords_.size());
  }

private:
  struct NodeConfig {
    std::string name;
    std::string kind;
    IdIndex hwNode = INVALID_ID;
    bool complete = true;
    std::vector<uint32_t> words;
  };

  bool buildConfigArtifacts(const MappingState &state, const Graph &dfg,
                            const Graph &adg, const ADGFlattener &flattener,
                            llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                            llvm::ArrayRef<FUConfigSelection> fuConfigs);
  bool writeConfigBinary(const std::string &path) const;
  bool writeConfigJson(const std::string &path) const;
  bool writeConfigHeader(const std::string &path) const;
  bool writeMapJson(const MappingState &state, const Graph &dfg,
                    const Graph &adg, const ADGFlattener &flattener,
                    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                    llvm::ArrayRef<FUConfigSelection> fuConfigs,
                    const std::string &path, int seed,
                    const TechMapper::Plan *techMapPlan,
                    const TechMapper::PlanMetrics *techMapMetrics,
                    const MapperTimingSummary *timingSummary,
                    const MapperSearchSummary *searchSummary,
                    llvm::StringRef techMapDiagnostics);
  bool writeMapText(const MappingState &state, const Graph &dfg,
                    const Graph &adg, const ADGFlattener &flattener,
                    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                    const std::string &path,
                    const TechMapper::Plan *techMapPlan,
                    const TechMapper::PlanMetrics *techMapMetrics,
                    const MapperTimingSummary *timingSummary,
                    const MapperSearchSummary *searchSummary,
                    llvm::StringRef techMapDiagnostics);
  void writeMapJsonTailSections(llvm::raw_ostream &out,
                                const MappingState &state, const Graph &dfg,
                                const Graph &adg,
                                const ADGFlattener &flattener,
                                llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                llvm::ArrayRef<FUConfigSelection> fuConfigs,
                                const MapperTimingSummary *timingSummary,
                                const MapperSearchSummary *searchSummary);

  std::vector<NodeConfig> nodeConfigs_;
  std::vector<ConfigSlice> configSlices_;
  std::vector<uint32_t> configWords_;
  std::vector<uint8_t> configBlob_;
  bool configComplete_ = true;
};

} // namespace fcc

#endif // FCC_MAPPER_CONFIGGEN_H
