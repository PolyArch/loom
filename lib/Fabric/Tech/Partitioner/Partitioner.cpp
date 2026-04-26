#include "Fabric/Tech/Partitioner/Partitioner.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/BeamPartitioner.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"
#include "Fabric/Tech/Partitioner/ListPartitioner.h"
#include "Fabric/Tech/Partitioner/SAPartitioner.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace fabric {

PartitionResult buildSingletonPartition(::dataflow::GraphOp graph,
                                        const TemplateLibrary &lib) {
  PartitionResult result;
  unsigned nextId = 0;
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;

    Block block;
    block.id = nextId++;
    block.ops.push_back(&op);
    block.tpl = nullptr;

    ::llvm::StringRef name = op.getName().getStringRef();
    if (::fabric::isFabricOpSupported(name)) {
      auto ids = lib.templatesByRootOp(name);
      if (!ids.empty())
        block.tpl = &lib.templates()[ids.front()];
    }

    result.blocks.push_back(std::move(block));
  }
  return result;
}

PartitionResult buildSingletonPartition(::dataflow::GraphOp graph,
                                        const TemplateLibrary &lib,
                                        const CandidateCache &cache) {
  PartitionResult result;
  unsigned nextId = 0;
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;

    Block block;
    block.id = nextId++;
    block.ops.push_back(&op);
    block.tpl = nullptr;

    ::llvm::ArrayRef<unsigned> ids = cache.templatesForOp(&op);
    if (!ids.empty())
      block.tpl = &lib.templates()[ids.front()];

    result.blocks.push_back(std::move(block));
  }
  return result;
}

std::unique_ptr<IPartitioner>
createPartitioner(::llvm::StringRef algorithm) {
  if (algorithm == "list")
    return std::make_unique<ListPartitioner>();
  if (algorithm == "beam")
    return std::make_unique<BeamPartitioner>();
  if (algorithm == "sa")
    return std::make_unique<SAPartitioner>();
  // Default to "greedy" for unknown / empty values; Config validation
  // should already have rejected invalid algorithm names upstream.
  return std::make_unique<GreedyPartitioner>();
}

} // namespace fabric
