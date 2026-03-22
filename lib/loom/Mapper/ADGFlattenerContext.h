#ifndef LOOM_MAPPER_ADGFLATTENER_CONTEXT_H
#define LOOM_MAPPER_ADGFLATTENER_CONTEXT_H

#include "loom/Mapper/Graph.h"

#include "loom/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
class Operation;
class Value;
class Block;
} // namespace mlir

namespace loom {

struct PEContainment;

/// Add a named attribute to a node.
inline void setNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute val,
                        mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), val));
}

/// Add a named attribute to an edge.
inline void setEdgeAttr(Edge *edge, llvm::StringRef key, mlir::Attribute val,
                        mlir::MLIRContext *ctx) {
  edge->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), val));
}

/// Source binding for wiring: associates a port with an optional PE output
/// index so that edges carry pe_output_index annotations.
struct SourceBinding {
  IdIndex portId = INVALID_ID;
  int peOutputIndex = -1;
};

/// PE info used during wiring: which MLIR operation corresponds to a PE and
/// which FU node IDs it contains.
struct PEWiringInfo {
  mlir::Operation *op = nullptr;
  std::vector<IdIndex> fuNodeIds;
};

/// Transient state shared between flatten() sub-routines.
/// Populated incrementally; each helper reads and/or extends it.
struct FlattenContext {
  mlir::MLIRContext *ctx = nullptr;
  loom::fabric::ModuleOp fabricMod;

  /// Map from SSA Value (result of each op) to ADG output port ID.
  llvm::DenseMap<mlir::Value, IdIndex> valueToOutputPort;

  /// Map from MLIR operation to its ADG node ID (single-node ops only;
  /// PE instances are tracked via peContainment instead).
  llvm::DenseMap<mlir::Operation *, IdIndex> opToNodeId;

  /// Definition symbol maps populated in the pre-pass.
  llvm::StringMap<loom::fabric::SpatialPEOp> peDefMap;
  llvm::StringMap<loom::fabric::TemporalPEOp> temporalPeDefMap;
  llvm::StringMap<loom::fabric::SpatialSwOp> swDefMap;
  llvm::StringMap<loom::fabric::TemporalSwOp> temporalSwDefMap;
  llvm::StringMap<loom::fabric::ExtMemoryOp> extMemoryDefMap;
  llvm::StringMap<loom::fabric::MemoryOp> memoryDefMap;
  llvm::StringMap<loom::fabric::FifoOp> fifoDefMap;
  llvm::StringMap<loom::fabric::FunctionUnitOp> functionUnitDefMap;

  /// Blocks that reference specific instance targets.
  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;

  /// Auto-naming counters for anonymous operations.
  unsigned autoTemporalSwCount = 0;
  unsigned autoExtMemCount = 0;
  unsigned autoMemCount = 0;
  unsigned autoFifoCount = 0;
  unsigned autoAddTagCount = 0;
  unsigned autoDelTagCount = 0;
  unsigned autoMapTagCount = 0;

  /// PE wiring info collected in pass 2.
  std::vector<PEWiringInfo> peInfos;

  /// Multi-source port map: Value -> multiple output port bindings.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<SourceBinding, 4>>
      valueSrcPorts;

  /// Total FU node count across all PEs.
  IdIndex totalFuNodes = 0;

  /// Returns or generates an instance name for the given operation.
  std::string getOrCreateOpName(mlir::Operation &op);

  /// Returns true when the operation is an unreferenced definition (not an
  /// instance target), meaning it should not generate a node on its own.
  bool isDefinitionOp(mlir::Operation *op, llvm::StringRef name) const;
};

/// Parse grid coordinates from a name like "pe_2_3" or "sw_0_1".
std::pair<int, int> parseGridPos(llvm::StringRef name);

/// Extract op names and internal DAG edges from an FU body.
/// Returns ops as ArrayAttr and edges as ArrayAttr of [srcIdx, dstIdx] pairs.
std::pair<mlir::ArrayAttr, mlir::ArrayAttr>
extractFUBodyDAG(loom::fabric::FunctionUnitOp fuOp, mlir::MLIRContext *ctx);

/// Extract config field widths from an FU body for configuration bits
/// computation.
mlir::DenseI64ArrayAttr
extractFUConfigFieldWidths(loom::fabric::FunctionUnitOp fuOp,
                           mlir::MLIRContext *ctx);

/// Infer grid positions for nodes that lack explicit coordinates by averaging
/// neighboring node positions until convergence.
void inferMissingNodeGridPositions(
    Graph &adg, llvm::DenseMap<IdIndex, std::pair<int, int>> &nodeGridPos);

} // namespace loom

#endif // LOOM_MAPPER_ADGFLATTENER_CONTEXT_H
