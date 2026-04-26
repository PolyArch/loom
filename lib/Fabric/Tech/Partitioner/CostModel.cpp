#include "Fabric/Tech/Partitioner/CostModel.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>

namespace fabric {

namespace {

// Compute, for each rootOpName, the maximum bodyOpCount across templates
// that share that root. This caps "perfect density" in the cost formula:
// a block of size N with the largest available N-op template hits 1.0.
::llvm::StringMap<unsigned>
maxTemplateSizeByRoot(const TemplateLibrary &lib) {
  ::llvm::StringMap<unsigned> out;
  for (const FuTemplate &t : lib.templates()) {
    if (t.rootOpName.empty())
      continue;
    auto &slot = out[t.rootOpName];
    if (t.bodyOpCount > slot)
      slot = t.bodyOpCount;
  }
  return out;
}

} // namespace

double computeCost(const PartitionResult &partition, const TemplateLibrary &lib,
                   const ::loom::TechMapConfig &cfg) {
  // Map each op pointer to its block id, so we can detect cross-block edges
  // by comparing the producer's block to the consumer's block.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  for (const Block &b : partition.blocks)
    for (::mlir::Operation *op : b.ops)
      opToBlock[op] = b.id;

  // |blocks_with_template|.
  unsigned blocksWithTemplate = 0;
  for (const Block &b : partition.blocks)
    if (b.tpl != nullptr)
      ++blocksWithTemplate;

  // cross_edges: count (def, use) pairs where def and use are both in some
  // block and those blocks differ. Edges to / from ops outside the
  // partition (graph args, ops not enrolled, etc.) are ignored.
  unsigned crossEdges = 0;
  for (const Block &b : partition.blocks) {
    for (::mlir::Operation *op : b.ops) {
      for (::mlir::Value operand : op->getOperands()) {
        ::mlir::Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = opToBlock.find(def);
        if (it == opToBlock.end())
          continue;
        if (it->second != b.id)
          ++crossEdges;
      }
    }
  }

  // avg_density across blocks that have a bound template. Graph-level
  // (tpl == nullptr) blocks contribute neither numerator nor denominator.
  ::llvm::StringMap<unsigned> maxByRoot = maxTemplateSizeByRoot(lib);
  double densitySum = 0.0;
  unsigned densityCount = 0;
  for (const Block &b : partition.blocks) {
    if (b.tpl == nullptr)
      continue;
    unsigned cap = 1;
    auto it = maxByRoot.find(b.tpl->rootOpName);
    if (it != maxByRoot.end())
      cap = std::max<unsigned>(1u, it->second);
    densitySum += static_cast<double>(b.ops.size()) / static_cast<double>(cap);
    ++densityCount;
  }
  double avgDensity = densityCount == 0
                          ? 0.0
                          : densitySum / static_cast<double>(densityCount);

  return cfg.alpha * static_cast<double>(blocksWithTemplate)
       + cfg.beta * static_cast<double>(crossEdges)
       - cfg.gamma * avgDensity;
}

} // namespace fabric
