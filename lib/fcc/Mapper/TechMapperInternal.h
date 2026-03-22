#ifndef FCC_MAPPER_TECHMAPPER_INTERNAL_H
#define FCC_MAPPER_TECHMAPPER_INTERNAL_H

#include "fcc/Mapper/TechMapper.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace fcc {
namespace techmapper_detail {

constexpr unsigned kMaxHardwareJoinFanin = 64;

enum class RefKind : uint8_t {
  Input = 0,
  OpResult = 1,
};

struct ValueRef {
  RefKind kind = RefKind::Input;
  unsigned index = 0;
  unsigned resultIndex = 0;

  bool operator==(const ValueRef &other) const {
    return kind == other.kind && index == other.index &&
           resultIndex == other.resultIndex;
  }
};

struct TemplateOp {
  unsigned bodyOpIndex = 0;
  std::string opName;
  bool commutative = false;
  llvm::SmallVector<ValueRef, 4> operands;
};

struct VariantFamily {
  std::string signature;
  std::string hwName;
  llvm::SmallVector<IdIndex, 4> hwNodeIds;
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  llvm::SmallVector<mlir::Type, 4> outputTypes;
  llvm::SmallVector<TemplateOp, 4> ops;
  llvm::SmallVector<std::pair<unsigned, unsigned>, 4> edges;
  llvm::SmallVector<std::optional<ValueRef>, 4> outputs;
  llvm::SmallVector<FUConfigField, 2> configFields;
  bool configurable = false;

  bool isTechFamily() const { return !ops.empty(); }
};

struct Match {
  unsigned familyIndex = 0;
  llvm::SmallVector<IdIndex, 4> swNodesByOp;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> operandOrderByOp;
  llvm::SmallVector<TechMapper::PortBinding, 4> inputBindings;
  llvm::SmallVector<TechMapper::PortBinding, 4> outputBindings;
  llvm::SmallVector<IdIndex, 4> internalEdges;
  llvm::SmallVector<FUConfigField, 4> configFields;
};

struct FamilyMatch {
  VariantFamily family;
  Match match;
};

struct DemandMatchStats {
  unsigned structuralStateCount = 0;
  unsigned structuralStateCacheHitCount = 0;
  unsigned structuralStateCacheMissCount = 0;
};

// Locate the ADG node for a function unit identified by PE name and FU name.
IdIndex findFunctionUnitNode(const Graph &adg, llvm::StringRef peName,
                             llvm::StringRef fuName);

// Enumerate all mux/join variant families for a single FunctionUnitOp.
void collectVariantsForFU(fcc::fabric::FunctionUnitOp fuOp,
                          const Node *hwNode,
                          llvm::SmallVectorImpl<VariantFamily> &variants);

// Find all DFG match candidates for a given variant family.
std::vector<Match> findMatchesForFamily(const Graph &dfg,
                                        const VariantFamily &family,
                                        unsigned familyIndex);

// Find all DFG match candidates for a given FunctionUnitOp without globally
// enumerating every structural variant first.
std::vector<FamilyMatch>
findDemandDrivenMatchesForFU(const Graph &dfg, fcc::fabric::FunctionUnitOp fuOp,
                             const Node *hwNode,
                             DemandMatchStats *stats = nullptr);

} // namespace techmapper_detail

struct AggregatedMatch {
  unsigned familyIndex = 0;
  llvm::SmallVector<IdIndex, 4> swNodesByOp;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> operandOrderByOp;
  llvm::SmallVector<TechMapper::PortBinding, 4> inputBindings;
  llvm::SmallVector<TechMapper::PortBinding, 4> outputBindings;
  llvm::SmallVector<IdIndex, 4> internalEdges;
  llvm::SmallVector<FUConfigField, 4> configFields;
  llvm::SmallVector<IdIndex, 4> hwNodeIds;
  bool configurable = false;
  bool temporal = false;
  unsigned supportClassId = std::numeric_limits<unsigned>::max();
  unsigned supportClassCapacity = 0;
  unsigned configClassId = std::numeric_limits<unsigned>::max();
  int64_t selectionScore = 0;
  unsigned commutativeSwapCount = 0;
  bool hasDemandOrigin = false;
  bool hasLegacyOrigin = false;
};

void populateFallbackNodeSummary(
    const TechMapper::Plan &plan, IdIndex swNodeId,
    const llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    TechMapper::FallbackNodeInfo &fallbackInfo);
void markNodeAsConservativeFallback(TechMapper::NodeTechInfo &info,
                                    llvm::StringRef status);
void accumulateNodeTechCandidateCoverage(
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeSupportClassesBySwNode,
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeConfigClassesBySwNode,
    const AggregatedMatch &aggregated, unsigned candidateId,
    std::optional<unsigned> selectionComponentId);
void finalizeNodeTechCoverageSummaries(
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeSupportClassesBySwNode,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeConfigClassesBySwNode);
void collectCoveredNodes(const std::vector<AggregatedMatch> &matches,
                         llvm::ArrayRef<unsigned> selectedIndices,
                         llvm::DenseSet<IdIndex> &candidateCoveredNodes,
                         llvm::DenseSet<IdIndex> &selectedCoveredNodes);
void sortSelectedUnitIndices(TechMapper::Plan &plan);
TechMapper::Unit buildSelectedUnitFromAggregatedMatch(
    const AggregatedMatch &aggregated, unsigned selectedCandidateId,
    std::optional<unsigned> selectionComponentId);
void registerSelectedUnit(
    TechMapper::Plan &plan, const AggregatedMatch &aggregated,
    const TechMapper::Unit &unit, unsigned unitIndex,
    unsigned selectedCandidateId, llvm::DenseSet<unsigned> &selectedConfigClasses,
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode);
void accumulateLegacyDerivedCandidateMetrics(
    TechMapper::Plan &plan, bool legacyFallbackOrigin, bool mixedOrigin,
    llvm::ArrayRef<IdIndex> hwNodeIds,
    llvm::DenseSet<IdIndex> &legacyDerivedHwNodeIds);
void applyCandidateSelectionOutcome(TechMapper::Plan &plan,
                                    TechMapper::CandidateSummaryInfo &summary,
                                    bool selected, llvm::StringRef status);
void markFeedbackFilteredCandidate(TechMapper::Plan &plan,
                                   TechMapper::CandidateSummaryInfo &summary,
                                   llvm::StringRef status);
void accumulateConservativeFallbackCandidate(TechMapper::Plan &plan,
                                             IdIndex swNodeId,
                                             const AggregatedMatch &aggregated);
void rebuildPreferredConservativeFallbackCandidates(TechMapper::Plan &plan);
std::string inferCandidateStatus(unsigned candidateIdx,
                                 const std::vector<AggregatedMatch> &matches,
                                 llvm::ArrayRef<unsigned> selectedMatches,
                                 llvm::ArrayRef<unsigned> matchComponentIds,
                                 const Graph &adg);
std::string inferRejectedReason(IdIndex swNodeId,
                                const std::vector<AggregatedMatch> &matches,
                                llvm::ArrayRef<unsigned> selectedMatches,
                                llvm::ArrayRef<unsigned> matchComponentIds,
                                const Graph &adg);
bool selectMatchesByComponent(const std::vector<AggregatedMatch> &matches,
                              const Graph &adg,
                              llvm::SmallVectorImpl<TechMapper::SelectionComponentInfo>
                                  *componentInfos,
                              std::vector<unsigned> *matchComponentIds,
                              TechMapper::PlanMetrics *metrics,
                              std::vector<unsigned> &selectedIndices);
bool selectMatchesByCachedComponents(
    const std::vector<AggregatedMatch> &matches,
    llvm::ArrayRef<unsigned> candidateIds,
    llvm::ArrayRef<unsigned> cachedComponentIds, unsigned baseComponentCount,
    const Graph &adg,
    llvm::SmallVectorImpl<TechMapper::SelectionComponentInfo> *componentInfos,
    std::vector<unsigned> *matchComponentIds, TechMapper::PlanMetrics *metrics,
    std::vector<unsigned> &selectedIndices);
AggregatedMatch
buildAggregatedMatchFromSummary(const TechMapper::CandidateSummaryInfo &summary);
int64_t lookupPenalty(llvm::ArrayRef<TechMapper::WeightedIdPenalty> penalties,
                      unsigned id);
void resetSelectionMetrics(TechMapper::PlanMetrics &metrics);
void restoreReselectionBaselineMetrics(TechMapper::PlanMetrics &metrics,
                                       const TechMapper::PlanMetrics &seed);
bool finalizePlanGraphs(const Graph &dfg, const Graph &adg,
                        TechMapper::Plan &plan);
} // namespace fcc

#endif // FCC_MAPPER_TECHMAPPER_INTERNAL_H
