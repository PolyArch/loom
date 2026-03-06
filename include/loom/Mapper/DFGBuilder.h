//===-- DFGBuilder.h - DFG extraction from Handshake MLIR ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Extracts a software dataflow graph (DFG) from a handshake::FuncOp. The DFG
// is a Graph where each non-terminator operation becomes an OperationNode,
// block arguments become ModuleInputNode sentinels, and return operands
// become ModuleOutputNode sentinels. SSA value uses become edges.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_DFGBUILDER_H
#define LOOM_MAPPER_DFGBUILDER_H

#include "loom/Mapper/Graph.h"

namespace circt {
namespace handshake {
class FuncOp;
} // namespace handshake
} // namespace circt

namespace loom {

class DFGBuilder {
public:
  /// Build a DFG from a handshake::FuncOp.
  /// The returned Graph is immutable during mapping.
  Graph build(circt::handshake::FuncOp funcOp);
};

} // namespace loom

#endif // LOOM_MAPPER_DFGBUILDER_H
