#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

namespace loom {

void ResourceTracker::addMapping(const MappingState &state, const Graph &adg) {
  // Record all hardware nodes that have at least one mapped software node.
  for (IdIndex hwId = 0;
       hwId < static_cast<IdIndex>(state.hwNodeToSwNodes.size()); ++hwId) {
    if (state.hwNodeToSwNodes[hwId].empty())
      continue;
    // Verify this is a valid ADG node.
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    usedNodes_.insert(hwId);
  }

  // Also accumulate SPM usage from memory nodes.
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    if (getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;
    if (hwId >= state.hwNodeToSwNodes.size() ||
        state.hwNodeToSwNodes[hwId].empty())
      continue;
    int64_t capacity = getNodeAttrInt(hwNode, "mem_size_bytes", 0);
    if (capacity > 0)
      spmBytesUsed_ += static_cast<uint64_t>(capacity);
  }
}

} // namespace loom
