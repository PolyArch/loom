#ifndef FCC_MAPPER_DFGBUILDER_H
#define FCC_MAPPER_DFGBUILDER_H

#include "fcc/Mapper/Graph.h"

namespace circt {
namespace handshake {
class FuncOp;
} // namespace handshake
} // namespace circt

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace fcc {

class DFGBuilder {
public:
  /// Build a DFG from the first handshake.func found in the module.
  /// Returns true on success.
  bool build(mlir::ModuleOp module, mlir::MLIRContext *ctx);

  /// Get the built DFG.
  const Graph &getDFG() const { return dfg; }
  Graph &getDFG() { return dfg; }

private:
  Graph dfg;
};

} // namespace fcc

#endif // FCC_MAPPER_DFGBUILDER_H
