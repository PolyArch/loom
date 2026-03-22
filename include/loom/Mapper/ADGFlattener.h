#ifndef LOOM_MAPPER_ADGFLATTENER_H
#define LOOM_MAPPER_ADGFLATTENER_H

#include "loom/Mapper/Graph.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace loom {

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
  std::string peKind;
  llvm::SmallVector<IdIndex, 8> fuNodeIds;
  /// Grid position for proximity scoring.
  int row = 0;
  int col = 0;
  /// PE-level port counts (from the spatial_pe function type).
  unsigned numInputPorts = 0;
  unsigned numOutputPorts = 0;
  /// Temporal PE attributes. Spatial PE keeps these zeroed.
  unsigned numInstruction = 0;
  unsigned numRegister = 0;
  unsigned regFifoDepth = 0;
  unsigned tagWidth = 0;
  bool enableShareOperandBuffer = false;
  unsigned operandBufferSize = 0;
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
  /// Create ADG nodes for all hardware resources (Pass 1).
  void flattenCreateNodes(struct FlattenContext &fctx, mlir::Block &body);

  /// Wire edges from SSA def-use chains and legacy metadata (Pass 2).
  void flattenWireEdges(struct FlattenContext &fctx, mlir::Block &body);

  /// Analyze bridge port topology for multi-lane memory nodes.
  void flattenAnalyzeBridges(struct FlattenContext &fctx);

  Graph adg;
  ConnectivityMatrix connectivity;
  std::vector<PEContainment> peContainment;
  /// Node ID -> grid (row, col).
  llvm::DenseMap<IdIndex, std::pair<int, int>> nodeGridPos;
};

} // namespace loom

#endif // LOOM_MAPPER_ADGFLATTENER_H
