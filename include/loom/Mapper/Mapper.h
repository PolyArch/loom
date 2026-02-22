//===-- Mapper.h - PnR mapper pipeline entry point ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Top-level Mapper class that orchestrates the full Place-and-Route pipeline:
// preprocessing, tech-mapping, placement, routing, refinement, temporal
// assignment, validation, and config generation.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_MAPPER_H
#define LOOM_MAPPER_MAPPER_H

#include "loom/Mapper/ConnectivityMatrix.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TechMapper.h"

#include <string>

namespace loom {

class Mapper {
public:
  struct Options {
    double budgetSeconds = 60.0;
    int seed = 0;
    std::string profile = "balanced";
    int maxLocalRepairs = 10;
    int maxGlobalRestarts = 3;
    int maxRefinementIterations = 50;
  };

  struct Result {
    bool success = false;
    MappingState state;
    std::string diagnostics;
  };

  /// Run the full PnR pipeline on the given DFG and ADG.
  Result run(const Graph &dfg, const Graph &adg, const Options &opts);

private:
  // --- Pipeline stages ---
  void preprocess(const Graph &adg);
  bool runPlacement(MappingState &state, const Graph &dfg, const Graph &adg,
                    const CandidateSet &candidates, const Options &opts);
  bool runRouting(MappingState &state, const Graph &dfg, const Graph &adg,
                  int seed = -1);
  bool runRefinement(MappingState &state, const Graph &dfg, const Graph &adg,
                     const CandidateSet &candidates, const Options &opts);
  bool runTemporalAssignment(MappingState &state, const Graph &dfg,
                             const Graph &adg);
  bool runValidation(const MappingState &state, const Graph &dfg,
                     const Graph &adg, std::string &diagnostics);

  // --- Placement helpers ---
  std::vector<IdIndex> computePlacementOrder(const Graph &dfg);
  void bindSentinelPorts(MappingState &state, const Graph &dfg,
                         const Graph &adg);
  double scorePlacement(IdIndex swNode, IdIndex hwNode,
                        const MappingState &state, const Graph &dfg,
                        const Graph &adg);

  // --- Routing helpers ---
  llvm::SmallVector<IdIndex, 8> findPath(IdIndex srcHwPort, IdIndex dstHwPort,
                                         const MappingState &state,
                                         const Graph &adg);
  bool isEdgeLegal(IdIndex srcPort, IdIndex dstPort,
                   const MappingState &state, const Graph &adg);
  IdIndex allocateTag(IdIndex hwPort, const MappingState &state,
                      const Graph &adg);

  // --- Cost helpers ---
  void computeCost(MappingState &state, const Graph &dfg, const Graph &adg,
                   const Options &opts);
  void getWeights(const std::string &profile, double weights[5]);

  // --- Validation helpers ---
  bool validateC1(const MappingState &state, const Graph &dfg,
                  const Graph &adg, std::string &diag);
  bool validateC2(const MappingState &state, const Graph &dfg,
                  const Graph &adg, std::string &diag);
  bool validateC3(const MappingState &state, const Graph &dfg,
                  const Graph &adg, std::string &diag);
  bool validateC4(const MappingState &state, const Graph &dfg,
                  const Graph &adg, std::string &diag);
  bool validateC5(const MappingState &state, const Graph &dfg,
                  const Graph &adg, std::string &diag);
  bool validateC6(const MappingState &state, const Graph &dfg,
                  const Graph &adg, std::string &diag);

  ConnectivityMatrix connectivity;
  /// Min-hop cost between pairs of ADG nodes (node ID -> node ID -> hops).
  llvm::DenseMap<IdIndex, llvm::DenseMap<IdIndex, unsigned>> minHopCosts;
};

} // namespace loom

#endif // LOOM_MAPPER_MAPPER_H
