#ifndef FCC_MAPPER_ADGFLATTENER_H
#define FCC_MAPPER_ADGFLATTENER_H

#include "fcc/Mapper/Graph.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace fcc {

/// Connectivity matrix for routing queries.
struct ConnectivityMatrix {
  /// Physical edges: output port -> reachable input ports.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> outToIn;
  /// Routing node internal: input port -> reachable output port(s).
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> inToOut;
};

/// PE containment record: which FU nodes belong to which PE.
struct PEContainment {
  std::string peName;
  llvm::SmallVector<IdIndex, 8> fuNodeIds;
  /// Grid position for proximity scoring.
  int row = 0;
  int col = 0;
};

class ADGFlattener {
public:
  /// Flatten a fabric.module from an MLIR ModuleOp into an ADG graph.
  /// The fabricModulePath is the path to the .fabric.mlir file.
  /// Returns true on success.
  bool flatten(mlir::ModuleOp topModule, mlir::MLIRContext *ctx);

  /// Get the flattened ADG graph.
  const Graph &getADG() const { return adg; }
  Graph &getADG() { return adg; }

  /// Get the connectivity matrix built during flatten().
  const ConnectivityMatrix &getConnectivity() const { return connectivity; }

  /// Get the PE containment map (PE name -> list of FU node IDs).
  const std::vector<PEContainment> &getPEContainment() const {
    return peContainment;
  }

  /// Get node grid position (row, col) for proximity scoring. Returns (-1,-1)
  /// if node has no grid position.
  std::pair<int, int> getNodeGridPos(IdIndex nodeId) const;

private:
  Graph adg;
  ConnectivityMatrix connectivity;
  std::vector<PEContainment> peContainment;
  /// Node ID -> grid (row, col).
  llvm::DenseMap<IdIndex, std::pair<int, int>> nodeGridPos;
};

} // namespace fcc

#endif // FCC_MAPPER_ADGFLATTENER_H
