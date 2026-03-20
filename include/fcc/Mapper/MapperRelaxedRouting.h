#ifndef FCC_MAPPER_RELAXED_ROUTING_H
#define FCC_MAPPER_RELAXED_ROUTING_H

#include "fcc/Mapper/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace fcc {

class Graph;
class MappingState;
struct MapperRelaxedRoutingOptions;

class RelaxedRoutingState {
public:
  struct SourceUse {
    IdIndex logicalSourcePort = INVALID_ID;
    unsigned count = 0;

    bool operator==(const SourceUse &other) const {
      return logicalSourcePort == other.logicalSourcePort &&
             count == other.count;
    }
  };

  struct Snapshot {
    std::vector<uint8_t> trackedOutputs;
    std::vector<llvm::SmallVector<SourceUse, 2>> outputSourceUses;
    std::vector<llvm::SmallVector<IdIndex, 4>> outputUsingEdges;
  };

  void init(const Graph &adg);
  void clear();

  Snapshot save() const;
  void restore(const Snapshot &snapshot);

  bool isTrackedOutput(IdIndex outPortId) const;
  unsigned distinctSources(IdIndex outPortId) const;
  bool wouldConflict(IdIndex outPortId, IdIndex logicalSourcePort) const;
  double softConflictPenalty(IdIndex outPortId, IdIndex logicalSourcePort,
                             const MapperRelaxedRoutingOptions &opts) const;

  void commitRoute(IdIndex swEdgeId, IdIndex logicalSourcePort,
                   llvm::ArrayRef<IdIndex> path, const Graph &adg);
  void uncommitRoute(IdIndex swEdgeId, IdIndex logicalSourcePort,
                     llvm::ArrayRef<IdIndex> path, const Graph &adg);

  llvm::SmallVector<IdIndex, 8> collectOverusedOutputs() const;
  llvm::SmallVector<IdIndex, 16>
  collectEdgesTouchingOutputs(llvm::ArrayRef<IdIndex> outputs) const;

private:
  void updateRoute(IdIndex swEdgeId, IdIndex logicalSourcePort,
                   llvm::ArrayRef<IdIndex> path, const Graph &adg,
                   bool addRoute);

  std::vector<uint8_t> trackedOutputs;
  std::vector<llvm::SmallVector<SourceUse, 2>> outputSourceUses;
  std::vector<llvm::SmallVector<IdIndex, 4>> outputUsingEdges;
};

bool isRelaxableRoutingOutput(IdIndex outPortId, const Graph &adg);
unsigned countDistinctLogicalSourcesForOutput(IdIndex outPortId,
                                              const MappingState &state);
llvm::SmallVector<IdIndex, 8>
collectOverusedNonTaggedRoutingOutputs(const MappingState &state,
                                       const Graph &adg);
llvm::SmallVector<IdIndex, 16>
collectEdgesTouchingRoutingOutputs(const MappingState &state,
                                   llvm::ArrayRef<IdIndex> outputs);

} // namespace fcc

#endif
