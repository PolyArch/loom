//===-- BridgeBinding.h - Bridge port binding utilities -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for classifying and binding bridge memory ports across the
// mapper pipeline. Consolidates the category-aware binding logic that was
// previously duplicated in Mapper, CPSATSolver, MapperRepair, TechMapper,
// and MapperValidation.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_BRIDGEBINDING_H
#define LOOM_MAPPER_BRIDGEBINDING_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/SmallVector.h"

namespace loom {

/// Category of a bridge boundary port.
enum class BridgePortCategory : uint8_t {
  StData = 0,
  StAddr = 1,
  LdAddr = 2,
  LdData = 3,
  LdDone = 4,
  StDone = 5,
};

/// Parsed bridge metadata from an ADG memory node.
struct BridgeInfo {
  bool hasBridge = false;

  llvm::SmallVector<IdIndex, 8> inputPorts;
  llvm::SmallVector<BridgePortCategory, 8> inputCategories;

  llvm::SmallVector<IdIndex, 8> outputPorts;
  llvm::SmallVector<BridgePortCategory, 8> outputCategories;

  llvm::SmallVector<IdIndex, 4> muxNodes;
  llvm::SmallVector<IdIndex, 4> demuxNodes;

  /// Extract bridge metadata from an ADG memory node.
  /// When bridge_input_categories / bridge_output_categories are present,
  /// uses them directly. Otherwise reconstructs categories from legacy
  /// split-point attributes (bridge_store_input_count, etc.).
  static BridgeInfo extract(const Node *hwNode);
};

/// Parsed DFG-side memory port structure.
struct DfgMemoryInfo {
  int64_t stCount = 0;
  int64_t ldCount = 0;
  unsigned swInSkip = 0; // 1 for extmemory (skip memref at input[0])

  /// Classify a DFG input port at relative index (after memref skip).
  BridgePortCategory classifyInput(unsigned relIdx) const;

  /// Classify a DFG output port at absolute index.
  BridgePortCategory classifyOutput(unsigned idx) const;

  /// Extract DFG memory info from a DFG node and its graph.
  static DfgMemoryInfo extract(const Node *swNode, const Graph &dfg,
                               bool isExtMem);
};

/// Read-only compatibility check for TechMapper. Returns true if all DFG
/// non-memref ports can be matched to bridge boundary ports by category
/// and type width.
bool isBridgeCompatible(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                        const Node *swNode, const Graph &dfg,
                        const Graph &adg);

/// Bind DFG input ports to bridge boundary input ports using strict
/// category matching. Returns true if all ports were bound.
bool bindBridgeInputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                      const Node *swNode, const Node *hwNode,
                      const Graph &dfg, const Graph &adg,
                      MappingState &state);

/// Bind DFG output ports to bridge boundary output ports using strict
/// category matching. Returns true if all ports were bound.
bool bindBridgeOutputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                       const Node *swNode, const Node *hwNode,
                       const Graph &dfg, const Graph &adg,
                       MappingState &state);

} // namespace loom

#endif // LOOM_MAPPER_BRIDGEBINDING_H
