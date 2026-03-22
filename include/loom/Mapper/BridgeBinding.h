#ifndef LOOM_MAPPER_BRIDGEBINDING_H
#define LOOM_MAPPER_BRIDGEBINDING_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace loom {

enum class BridgePortCategory : uint8_t {
  StData = 0,
  StAddr = 1,
  LdAddr = 2,
  LdData = 3,
  LdDone = 4,
  StDone = 5,
};

struct BridgeInfo {
  bool hasBridge = false;

  llvm::SmallVector<IdIndex, 8> inputPorts;
  llvm::SmallVector<BridgePortCategory, 8> inputCategories;
  llvm::SmallVector<unsigned, 8> inputLanes;

  llvm::SmallVector<IdIndex, 8> outputPorts;
  llvm::SmallVector<BridgePortCategory, 8> outputCategories;
  llvm::SmallVector<unsigned, 8> outputLanes;

  llvm::SmallVector<IdIndex, 4> muxNodes;
  llvm::SmallVector<IdIndex, 4> demuxNodes;

  static BridgeInfo extract(const Node *hwNode);
};

struct DfgMemoryInfo {
  int64_t stCount = 0;
  int64_t ldCount = 0;
  unsigned swInSkip = 0;

  BridgePortCategory classifyInput(unsigned relIdx) const;
  BridgePortCategory classifyOutput(unsigned idx) const;
  unsigned inputLocalLane(unsigned relIdx) const;
  unsigned outputLocalLane(unsigned idx) const;
  unsigned laneSpan() const;

  static DfgMemoryInfo extract(const Node *swNode, const Graph &dfg,
                               bool isExtMem);
};

struct BridgeLaneRange {
  unsigned start = 0;
  unsigned end = 0;
};

struct BridgeLaneUsage {
  unsigned base = 0;
  bool usesLoadFamily = false;
  unsigned loadStart = 0;
  unsigned loadEnd = 0;
  bool usesStoreFamily = false;
  unsigned storeStart = 0;
  unsigned storeEnd = 0;
};

BridgeLaneUsage computeBridgeLaneUsage(const DfgMemoryInfo &mem,
                                       unsigned baseLane);
bool bridgeLaneUsageConflicts(const BridgeLaneUsage &lhs,
                              const BridgeLaneUsage &rhs);

bool isBridgeCompatible(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                        const Node *swNode, const Node *hwNode,
                        const Graph &dfg, const Graph &adg);

bool bindBridgeInputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                      const Node *swNode, const Node *hwNode,
                      const Graph &dfg, const Graph &adg,
                      MappingState &state);

bool bindBridgeOutputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                       const Node *swNode, const Node *hwNode,
                       const Graph &dfg, const Graph &adg,
                       MappingState &state);

std::optional<unsigned> inferBridgeLane(const BridgeInfo &bridge,
                                        const DfgMemoryInfo &mem,
                                        const Node *swNode,
                                        const Graph &dfg,
                                        const MappingState &state);

std::optional<BridgeLaneRange>
inferBridgeLaneRange(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                     const Node *swNode, const Graph &dfg,
                     const MappingState &state);

} // namespace loom

#endif // LOOM_MAPPER_BRIDGEBINDING_H
