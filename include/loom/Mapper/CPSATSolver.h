//===-- CPSATSolver.h - CP-SAT solver for mapper ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// CP-SAT solver integration with dual-mode architecture:
// - Full-problem mode for small DFGs (<= threshold nodes)
// - Sub-problem refinement mode for large DFGs
//
// Requires Google OR-Tools CP-SAT. Guarded by LOOM_HAS_ORTOOLS.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_CPSATSOLVER_H
#define LOOM_MAPPER_CPSATSOLVER_H

#include "loom/Mapper/ConnectivityMatrix.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TechMapper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace loom {

class CPSATSolver {
public:
  enum class Mode {
    FULL_PROBLEM,
    SUB_PROBLEM,
    DISABLED,
  };

  struct Options {
    Mode mode;
    double timeLimitSeconds;
    int subProblemMaxNodes;

    Options()
        : mode(Mode::FULL_PROBLEM), timeLimitSeconds(60.0),
          subProblemMaxNodes(50) {}
  };

  struct Result {
    bool success;
    MappingState state;
    std::string diagnostics;

    Result() : success(false) {}
  };

  /// Solve the full placement + routing problem using CP-SAT.
  /// If warmStart is provided, uses it to initialize the solver.
  Result solveFullProblem(const Graph &dfg, const Graph &adg,
                          const CandidateSet &candidates,
                          const ConnectivityMatrix &connectivity,
                          const MappingState *warmStart = nullptr,
                          const Options &opts = Options());

  /// Solve a local sub-problem: only re-map the given subset of DFG nodes.
  /// Placements outside the subset are fixed to their current values.
  Result solveSubProblem(const Graph &dfg, const Graph &adg,
                         llvm::ArrayRef<IdIndex> subgraphSwNodes,
                         const MappingState &currentState,
                         const CandidateSet &candidates,
                         const ConnectivityMatrix &connectivity,
                         const Options &opts = Options());

  /// Select the appropriate mode based on DFG size and profile.
  static Mode selectMode(const Graph &dfg, const std::string &profile,
                          int subProblemMaxNodes = 50);

  /// Extract a sub-problem region around the given conflict nodes.
  /// Expands to include immediate DFG neighbors up to maxNodes.
  static llvm::SmallVector<IdIndex, 16>
  extractSubProblem(const Graph &dfg, llvm::ArrayRef<IdIndex> conflictNodes,
                    int maxNodes);

  /// Check if CP-SAT support is compiled in.
  static bool isAvailable();
};

} // namespace loom

#endif // LOOM_MAPPER_CPSATSOLVER_H
