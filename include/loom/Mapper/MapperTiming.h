#ifndef LOOM_MAPPER_MAPPERTIMING_H
#define LOOM_MAPPER_MAPPERTIMING_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/MapperOptions.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <vector>

namespace loom {

struct MapperRecurrenceCycleSummary {
  unsigned cycleId = 0;
  llvm::SmallVector<IdIndex, 8> swNodes;
  llvm::SmallVector<IdIndex, 8> swEdges;
  unsigned recurrenceDistance = 1;
  unsigned sequentialLatencyCycles = 0;
  unsigned fifoStageCutContribution = 0;
  unsigned maxIntervalOnCycle = 1;
  unsigned estimatedCycleII = 1;
  double combinationalDelay = 0.0;
};

struct MapperTimingSummary {
  double estimatedCriticalPathDelay = 0.0;
  double estimatedClockPeriod = 0.0;
  unsigned estimatedInitiationInterval = 1;
  double estimatedThroughputCost = 0.0;
  double recurrencePressure = 0.0;
  llvm::SmallVector<IdIndex, 16> criticalPathEdges;
  unsigned fifoBufferCount = 0;
  unsigned forcedBufferedFifoCount = 0;
  llvm::SmallVector<IdIndex, 8> forcedBufferedFifoNodes;
  llvm::SmallVector<unsigned, 8> forcedBufferedFifoDepths;
  unsigned mapperSelectedBufferedFifoCount = 0;
  llvm::SmallVector<IdIndex, 8> mapperSelectedBufferedFifoNodes;
  llvm::SmallVector<unsigned, 8> mapperSelectedBufferedFifoDepths;
  llvm::SmallVector<IdIndex, 16> bufferizedEdges;
  std::vector<MapperRecurrenceCycleSummary> recurrenceCycles;
};

MapperTimingSummary analyzeMapperTiming(const MappingState &state,
                                       const Graph &dfg, const Graph &adg,
                                       llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                       const MapperTimingOptions &opts);

} // namespace loom

#endif // LOOM_MAPPER_MAPPERTIMING_H
