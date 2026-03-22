#ifndef LOOM_MAPPER_CONFIGGEN_INTERNAL_H
#define LOOM_MAPPER_CONFIGGEN_INTERNAL_H

// Internal header for ConfigGen split: shared struct types and helper
// function declarations used across ConfigGenHelpers.cpp,
// ConfigGenConfig.cpp, and ConfigGen.cpp.  This file lives under
// lib/loom/Mapper/ because the symbols are translation-unit-private.

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/BridgeBinding.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/TagRuntime.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace loom {
namespace configgen_detail {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t kConfigWordBits = 32;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct MemoryRegionEntry {
  IdIndex swNode = INVALID_ID;
  int64_t memrefArgIndex = -1;
  int64_t startLane = -1;
  int64_t endLane = -1;
  int64_t ldCount = 0;
  int64_t stCount = 0;
  int64_t elemSizeLog2 = 0;
};

struct MapTagTableEntry {
  bool valid = false;
  uint64_t srcTag = 0;
  uint64_t dstTag = 0;
};

struct GeneratedNodeConfig {
  std::vector<uint32_t> words;
  bool complete = true;
};

struct PERouteSummary {
  llvm::DenseMap<IdIndex, llvm::SmallVector<int, 4>> inputPortSelects;
  llvm::DenseMap<IdIndex, llvm::SmallVector<int, 4>> outputPortSelects;
  llvm::DenseMap<IdIndex, llvm::SmallVector<uint64_t, 4>> tagsByFU;
  bool complete = true;
};

struct TemporalRegisterBinding {
  IdIndex swEdgeId = INVALID_ID;
  IdIndex writerSwNode = INVALID_ID;
  IdIndex readerSwNode = INVALID_ID;
  IdIndex writerHwNode = INVALID_ID;
  IdIndex readerHwNode = INVALID_ID;
  IdIndex srcSwPort = INVALID_ID;
  unsigned writerOutputIndex = 0;
  unsigned readerInputIndex = 0;
  unsigned registerIndex = 0;
  std::string peName;
};

struct TemporalConfigPlan {
  llvm::SmallVector<std::pair<unsigned, IdIndex>, 8> usedFUs;
  llvm::DenseMap<IdIndex, unsigned> slotByFU;
  llvm::DenseMap<IdIndex, llvm::SmallVector<std::optional<unsigned>, 4>>
      operandRegsByFU;
  llvm::DenseMap<IdIndex, llvm::SmallVector<std::optional<unsigned>, 4>>
      resultRegsByFU;
  llvm::SmallVector<TemporalRegisterBinding, 8> registerBindings;
  bool complete = true;
};

// ---------------------------------------------------------------------------
// Report helpers
// ---------------------------------------------------------------------------
std::string escapeJsonString(llvm::StringRef text);
std::string summarizeConfigField(const FUConfigField &field);
bool getEffectiveFifoBypassed(const Node *hwNode, IdIndex hwId,
                              const MappingState &state);
void writeOptionalUIntJson(llvm::raw_ostream &out, unsigned value);
int64_t computeSelectedUnitTemporalPenalty(const TechMapper::Unit &unit);
int64_t computeFamilyScarcityPenalty(const TechMapper::Plan *techMapPlan,
                                     unsigned familyIndex);
void writeOptionalUIntText(llvm::raw_ostream &out, unsigned value);
llvm::StringRef lookupSupportClassKey(const TechMapper::Plan *techMapPlan,
                                      unsigned id);
llvm::StringRef lookupConfigClassKey(const TechMapper::Plan *techMapPlan,
                                     unsigned id);
llvm::StringRef lookupConfigClassReason(const TechMapper::Plan *techMapPlan,
                                        unsigned id);
llvm::StringRef lookupFamilySignature(const TechMapper::Plan *techMapPlan,
                                      unsigned id);
const TechMapper::CandidateSummaryInfo *
lookupCandidateSummary(const TechMapper::Plan *techMapPlan, unsigned id);
void writeCandidateFeedbackRefJson(llvm::raw_ostream &out,
                                   const TechMapper::Plan *techMapPlan,
                                   unsigned id);
void writeFamilyFeedbackRefJson(llvm::raw_ostream &out,
                                const TechMapper::Plan *techMapPlan,
                                unsigned id);
void writeConfigClassFeedbackRefJson(llvm::raw_ostream &out,
                                     const TechMapper::Plan *techMapPlan,
                                     unsigned id);
void writeCandidatePenaltyJson(llvm::raw_ostream &out,
                               const TechMapper::Plan *techMapPlan,
                               const TechMapper::WeightedIdPenalty &penalty);
void writeFamilyPenaltyJson(llvm::raw_ostream &out,
                            const TechMapper::Plan *techMapPlan,
                            const TechMapper::WeightedIdPenalty &penalty);
void writeConfigClassPenaltyJson(llvm::raw_ostream &out,
                                 const TechMapper::Plan *techMapPlan,
                                 const TechMapper::WeightedIdPenalty &penalty);
void writeCandidateFeedbackRefText(llvm::raw_ostream &out,
                                   const TechMapper::Plan *techMapPlan,
                                   unsigned id);
void writeFamilyFeedbackRefText(llvm::raw_ostream &out,
                                const TechMapper::Plan *techMapPlan,
                                unsigned id);
void writeConfigClassFeedbackRefText(llvm::raw_ostream &out,
                                     const TechMapper::Plan *techMapPlan,
                                     unsigned id);
std::string
sanitizeTechMapDiagnosticsForArtifact(llvm::StringRef diagnostics);

// ---------------------------------------------------------------------------
// Bit packing helpers
// ---------------------------------------------------------------------------
void packBits(std::vector<uint32_t> &words, uint32_t &bitPos, uint64_t value,
              unsigned width);

void packMuxField(std::vector<uint32_t> &words, uint32_t &bitPos,
                  unsigned selBits, uint64_t sel, bool discard,
                  bool disconnect);

// ---------------------------------------------------------------------------
// String / header helpers
// ---------------------------------------------------------------------------
std::string buildHeaderGuard(llvm::StringRef pathStem);
std::string getConfigHeaderFilename(llvm::StringRef basePath);
unsigned bitWidthForChoices(unsigned count);
llvm::StringRef configFieldKindName(FUConfigFieldKind kind);
std::string formatConfigFieldValue(const FUConfigField &field);

// ---------------------------------------------------------------------------
// Port / graph traversal
// ---------------------------------------------------------------------------
bool isConnectedPosition(const std::vector<std::string> &rows, unsigned outIdx,
                         unsigned inIdx, unsigned numIn);
unsigned countConnectedPositions(const std::vector<std::string> &rows,
                                 unsigned numIn, unsigned numOut);
unsigned connectedPositionOrdinal(const std::vector<std::string> &rows,
                                  unsigned numIn, unsigned numOut,
                                  unsigned inputIdx, unsigned outputIdx);
int findNodeInputIndex(const Node *node, IdIndex portId);
int findNodeOutputIndex(const Node *node, IdIndex portId);
mlir::DenseI64ArrayAttr getDenseI64NodeAttr(const Node *node,
                                            llvm::StringRef name);
std::optional<uint64_t> getUIntEdgeAttr(const Edge *edge,
                                        llvm::StringRef name);
mlir::Type getTypeNodeAttr(const Node *node, llvm::StringRef name);
IdIndex findFeedingPort(const Graph &graph, IdIndex inputPortId);
llvm::SmallVector<IdIndex, 4> findConsumingPorts(const Graph &graph,
                                                 IdIndex outputPortId);
IdIndex findMemoryInputFromSwitchOutput(const Graph &graph,
                                        IdIndex switchOutPort,
                                        IdIndex hwMemoryNodeId);
IdIndex findSwitchInputFromMemory(const Graph &graph, const Node *switchNode,
                                  IdIndex hwMemoryNodeId);
bool isBridgeTraversalNode(llvm::StringRef opKind);
llvm::SmallVector<IdIndex, 8>
findBridgePathForward(const Graph &graph, IdIndex startPortId,
                      IdIndex hwMemoryNodeId);
llvm::SmallVector<IdIndex, 8>
findBridgePathBackward(const Graph &graph, IdIndex startPortId,
                       IdIndex hwMemoryNodeId);

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------
bool isSoftwareMemoryInterfaceOp(llvm::StringRef opName);
int64_t findDfgMemrefArgIndex(const Node *swNode, const Graph &dfg);
std::optional<int64_t> getRegionElemSizeLog2(const Node *swNode,
                                             const Node *hwNode,
                                             const Graph &dfg,
                                             const Graph &adg);
std::vector<MemoryRegionEntry>
collectMemoryRegionsForNode(IdIndex hwId, const MappingState &state,
                            const Graph &dfg, const Graph &adg);
llvm::SmallVector<int64_t, 16>
buildAddrOffsetTable(const Node *hwNode, IdIndex hwId,
                     const MappingState &state, const Graph &dfg,
                     const Graph &adg);
std::optional<uint64_t>
findMemoryRegionStartLane(IdIndex swNodeId, IdIndex hwNodeId,
                          const MappingState &state, const Graph &dfg,
                          const Graph &adg);
std::optional<uint64_t>
computeSoftwareMemoryPortLane(IdIndex swNodeId, IdIndex swPortId, bool isOutput,
                              IdIndex hwNodeId, const MappingState &state,
                              const Graph &dfg, const Graph &adg);

// ---------------------------------------------------------------------------
// Tag helpers
// ---------------------------------------------------------------------------
std::optional<uint64_t> getUIntNodeAttr(const Node *node,
                                        llvm::StringRef name);
llvm::SmallVector<MapTagTableEntry, 8> getMapTagTableEntries(const Node *node);
std::pair<unsigned, unsigned> getMapTagTagWidths(const Node *node,
                                                 const Graph &adg);
std::vector<std::string> getBinaryRowsNodeAttr(const Node *node,
                                               llvm::StringRef name);
std::optional<uint64_t>
computeRuntimeTagValueAlongPath(IdIndex swEdgeId,
                                llvm::ArrayRef<IdIndex> hwPath,
                                size_t uptoIndex, const MappingState &state,
                                const Graph &dfg, const Graph &adg);
std::optional<size_t> findLastTaggedPortIndex(llvm::ArrayRef<IdIndex> hwPath,
                                              const Graph &adg);
std::optional<uint64_t>
computeTemporalNodeTagValue(IdIndex swNodeId, const MappingState &state,
                            const Graph &dfg, const Graph &adg);
std::optional<uint64_t>
computeTemporalNodeIngressTagValue(IdIndex swNodeId,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg);
std::optional<uint64_t>
computeTemporalRouteTagValue(IdIndex swEdgeId,
                             llvm::ArrayRef<IdIndex> hwPath,
                             size_t transitionIndex,
                             const MappingState &state, const Graph &dfg,
                             const Graph &adg);

// ---------------------------------------------------------------------------
// Export path helpers
// ---------------------------------------------------------------------------
llvm::SmallVector<IdIndex, 8>
buildBridgeInputSuffix(const Graph &adg, IdIndex boundaryInPortId,
                       IdIndex hwMemoryNodeId);
llvm::SmallVector<IdIndex, 8>
buildBridgeOutputPrefix(const Graph &adg, IdIndex boundaryOutPortId,
                        IdIndex hwMemoryNodeId);
llvm::SmallVector<IdIndex, 16>
buildExportPathForEdge(IdIndex edgeId, const MappingState &state,
                       const Graph &dfg, const Graph &adg);

// ---------------------------------------------------------------------------
// Config builder helpers (switch, routing, memory, FU, PE)
// ---------------------------------------------------------------------------
GeneratedNodeConfig buildSpatialSwitchConfig(const Node *hwNode, IdIndex hwId,
                                             const MappingState &state,
                                             const Graph &dfg,
                                             const Graph &adg);
GeneratedNodeConfig buildTemporalSwitchConfig(const Node *hwNode, IdIndex hwId,
                                              const MappingState &state,
                                              const Graph &dfg,
                                              const Graph &adg);
GeneratedNodeConfig buildAddTagConfig(const Node *hwNode);
GeneratedNodeConfig buildMapTagConfig(const Node *hwNode, const Graph &adg);
GeneratedNodeConfig buildMemoryConfig(const Node *hwNode, IdIndex hwId,
                                      const MappingState &state,
                                      const Graph &dfg, const Graph &adg);
GeneratedNodeConfig buildFifoConfig(const Node *hwNode, IdIndex hwId,
                                    const MappingState &state);

const FUConfigSelection *
findFUConfigSelection(llvm::ArrayRef<FUConfigSelection> fuConfigs,
                      IdIndex hwNodeId);
const Edge *findEdgeByPorts(const Graph &graph, IdIndex srcPortId,
                            IdIndex dstPortId);
llvm::SmallVector<unsigned, 4> getFUConfigFieldWidths(const Node *hwNode);
unsigned getFUConfigBitWidth(const Node *hwNode);
void packFUConfigBits(std::vector<uint32_t> &words, uint32_t &bitPos,
                      const Node *hwNode, const FUConfigSelection *selection,
                      bool &complete);

TemporalConfigPlan
buildTemporalConfigPlan(const PEContainment &pe, const MappingState &state,
                        const Graph &dfg, const Graph &adg,
                        llvm::ArrayRef<TechMappedEdgeKind> edgeKinds);
PERouteSummary collectPERouteSummary(const PEContainment &pe,
                                     const MappingState &state,
                                     const Graph &dfg, const Graph &adg);

GeneratedNodeConfig
buildFunctionUnitConfig(llvm::ArrayRef<FUConfigSelection> fuConfigs,
                        IdIndex hwId);
GeneratedNodeConfig
buildSpatialPEConfig(const PEContainment &pe, const MappingState &state,
                     const Graph &dfg, const Graph &adg,
                     llvm::ArrayRef<FUConfigSelection> fuConfigs,
                     bool &globalComplete);
GeneratedNodeConfig
buildTemporalPEConfig(const PEContainment &pe, const MappingState &state,
                      const Graph &dfg, const Graph &adg,
                      llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                      llvm::ArrayRef<FUConfigSelection> fuConfigs,
                      bool &globalComplete);

} // namespace configgen_detail
} // namespace loom

#endif // LOOM_MAPPER_CONFIGGEN_INTERNAL_H
