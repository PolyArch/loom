//===-- ADGFlattener.h - ADG extraction from Fabric MLIR ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Flattens a fabric::ModuleOp into a hardware architecture description graph
// (ADG). Resolves fabric.instance references, handles temporal_pe virtual
// node representation, and builds the ConnectivityMatrix.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_ADGFLATTENER_H
#define LOOM_MAPPER_ADGFLATTENER_H

#include "loom/Mapper/ConnectivityMatrix.h"
#include "loom/Mapper/Graph.h"

namespace loom {
namespace fabric {
class ModuleOp;
} // namespace fabric

class ADGFlattener {
public:
  /// Flatten a fabric::ModuleOp into an ADG graph.
  /// Also builds the connectivity matrix for routing queries.
  Graph flatten(fabric::ModuleOp moduleOp);

  /// Get the connectivity matrix built during flatten().
  const ConnectivityMatrix &getConnectivityMatrix() const { return matrix; }

private:
  ConnectivityMatrix matrix;
};

} // namespace loom

#endif // LOOM_MAPPER_ADGFLATTENER_H
