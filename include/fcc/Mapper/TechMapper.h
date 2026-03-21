#ifndef FCC_MAPPER_TECHMAPPER_H
#define FCC_MAPPER_TECHMAPPER_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
}

namespace fcc {

enum class FUConfigFieldKind : uint8_t {
  Mux = 0,
  ConstantValue = 1,
  CmpIPredicate = 2,
  CmpFPredicate = 3,
  StreamContCond = 4,
  JoinMask = 5,
};

struct FUConfigField {
  FUConfigFieldKind kind = FUConfigFieldKind::Mux;
  unsigned opIndex = 0;
  unsigned templateOpIndex = std::numeric_limits<unsigned>::max();
  std::string opName;
  unsigned bitWidth = 0;
  uint64_t value = 0;
  uint64_t sel = 0;
  bool discard = false;
  bool disconnect = false;
};

struct FUConfigSelection {
  IdIndex hwNodeId = INVALID_ID;
  std::string hwName;
  std::string peName;
  unsigned supportClassId = std::numeric_limits<unsigned>::max();
  unsigned configClassId = std::numeric_limits<unsigned>::max();
  llvm::SmallVector<IdIndex, 4> swNodeIds;
  llvm::SmallVector<FUConfigField, 2> fields;
};

enum class TechMappedEdgeKind : uint8_t {
  Routed = 0,
  IntraFU = 1,
  TemporalReg = 2,
};

class TechMapper {
public:
  struct SupportClassInfo {
    unsigned id = std::numeric_limits<unsigned>::max();
    std::string key;
    std::string kind;
    bool temporal = false;
    unsigned capacity = 0;
    bool enforceHardCapacity = true;
    llvm::SmallVector<IdIndex, 4> hwNodeIds;
    llvm::SmallVector<std::string, 4> peNames;
  };

  struct ConfigClassInfo {
    unsigned id = std::numeric_limits<unsigned>::max();
    std::string key;
    std::string reason;
    bool temporal = false;
    llvm::SmallVector<unsigned, 4> compatibleConfigClassIds;
  };

  struct FallbackNodeInfo {
    IdIndex swNodeId = INVALID_ID;
    IdIndex contractedNodeId = INVALID_ID;
    unsigned selectionComponentId = std::numeric_limits<unsigned>::max();
    std::string reason;
    unsigned candidateHwNodeCount = 0;
    unsigned supportClassCount = 0;
    unsigned configClassCount = 0;
    llvm::SmallVector<unsigned, 8> candidateIds;
    llvm::SmallVector<unsigned, 8> supportClassIds;
    llvm::SmallVector<unsigned, 8> configClassIds;
  };

  struct TemporalIncompatibilityInfo {
    unsigned lhsConfigClassId = std::numeric_limits<unsigned>::max();
    unsigned rhsConfigClassId = std::numeric_limits<unsigned>::max();
    std::string reason;
  };

  struct NodeTechInfo {
    IdIndex swNodeId = INVALID_ID;
    IdIndex contractedNodeId = INVALID_ID;
    unsigned selectionComponentId = std::numeric_limits<unsigned>::max();
    unsigned selectedUnitIndex = std::numeric_limits<unsigned>::max();
    unsigned selectedCandidateId = std::numeric_limits<unsigned>::max();
    unsigned candidateCount = 0;
    unsigned supportClassCount = 0;
    unsigned configClassCount = 0;
    unsigned maxFusionSize = 0;
    llvm::SmallVector<unsigned, 8> candidateIds;
    llvm::SmallVector<unsigned, 8> supportClassIds;
    llvm::SmallVector<unsigned, 8> configClassIds;
    bool selected = false;
    bool selectedAsFusion = false;
    bool conservativeFallback = false;
    bool selectedFromLegacyFallback = false;
    bool selectedFromDemand = false;
    bool selectedFromMixedOrigin = false;
    std::string status;
  };

  struct LegacyOracleSampleInfo {
    std::string key;
    unsigned familyIndex = std::numeric_limits<unsigned>::max();
    std::string familySignature;
    IdIndex hwNodeId = INVALID_ID;
    std::string peName;
    std::string hwName;
  };

  struct SelectionComponentInfo {
    unsigned id = 0;
    unsigned candidateCount = 0;
    unsigned swNodeCount = 0;
    unsigned selectedCount = 0;
    bool containsTemporalCandidate = false;
    int64_t baseMaxCandidateScore = 0;
    int64_t maxCandidateScore = 0;
    int64_t baseSelectedScoreSum = 0;
    int64_t selectedScoreSum = 0;
    llvm::SmallVector<IdIndex, 8> swNodeIds;
    llvm::SmallVector<unsigned, 8> candidateIds;
    llvm::SmallVector<unsigned, 8> filteredCandidateIds;
    llvm::SmallVector<unsigned, 8> selectedCandidateIds;
    llvm::SmallVector<unsigned, 8> selectedUnitIndices;
    std::string solver;
  };

  struct FamilyTechInfo {
    unsigned familyIndex = 0;
    std::string signature;
    unsigned hwSupportCount = 0;
    unsigned materializedStateCount = 0;
    unsigned legacyMaterializedStateCount = 0;
    unsigned matchCount = 0;
    unsigned selectedCount = 0;
    unsigned maxFusionSize = 0;
    unsigned opCount = 0;
    bool configurable = false;
  };

  struct PortBinding {
    IdIndex swPortId = INVALID_ID;
    unsigned hwPortIndex = 0;
  };

  struct CandidateSummaryInfo {
    unsigned id = 0;
    unsigned familyIndex = std::numeric_limits<unsigned>::max();
    std::string familySignature;
    unsigned selectionComponentId = std::numeric_limits<unsigned>::max();
    unsigned selectedUnitIndex = std::numeric_limits<unsigned>::max();
    IdIndex contractedNodeId = INVALID_ID;
    llvm::SmallVector<IdIndex, 4> swNodeIds;
    llvm::SmallVector<IdIndex, 4> internalEdgeIds;
    llvm::SmallVector<PortBinding, 4> inputBindings;
    llvm::SmallVector<PortBinding, 4> outputBindings;
    llvm::SmallVector<IdIndex, 4> hwNodeIds;
    llvm::SmallVector<std::string, 4> peNames;
    unsigned supportClassId = std::numeric_limits<unsigned>::max();
    unsigned supportClassCapacity = 0;
    std::string supportClassKey;
    unsigned configClassId = std::numeric_limits<unsigned>::max();
    std::string configClassKey;
    std::string configClassReason;
    bool temporal = false;
    bool configurable = false;
    int64_t baseSelectionScore = 0;
    int64_t candidatePenalty = 0;
    int64_t familyPenalty = 0;
    int64_t configClassPenalty = 0;
    int64_t selectionScore = 0;
    bool selected = false;
    std::string status;
    bool demandOrigin = false;
    bool legacyFallbackOrigin = false;
    bool mixedOrigin = false;
    llvm::SmallVector<FUConfigField, 4> configFields;
  };

  struct WeightedIdPenalty {
    unsigned id = std::numeric_limits<unsigned>::max();
    int64_t penalty = 0;
  };

  struct Feedback {
    llvm::SmallVector<unsigned, 8> bannedCandidateIds;
    llvm::SmallVector<unsigned, 8> bannedFamilyIds;
    llvm::SmallVector<unsigned, 8> bannedConfigClassIds;
    llvm::SmallVector<unsigned, 8> splitCandidateIds;
    llvm::SmallVector<WeightedIdPenalty, 8> candidatePenalties;
    llvm::SmallVector<WeightedIdPenalty, 8> familyPenalties;
    llvm::SmallVector<WeightedIdPenalty, 8> configClassPenalties;
  };

  struct FeedbackResolutionInfo {
    llvm::SmallVector<unsigned, 8> unknownBannedCandidateIds;
    llvm::SmallVector<unsigned, 8> unknownBannedFamilyIds;
    llvm::SmallVector<unsigned, 8> unknownBannedConfigClassIds;
    llvm::SmallVector<unsigned, 8> unknownSplitCandidateIds;
    llvm::SmallVector<WeightedIdPenalty, 8> unknownCandidatePenalties;
    llvm::SmallVector<WeightedIdPenalty, 8> unknownFamilyPenalties;
    llvm::SmallVector<WeightedIdPenalty, 8> unknownConfigClassPenalties;
  };

  struct PlanMetrics {
    double coverageScore = 1.0;
    uint64_t totalLayer2TimeMicros = 0;
    uint64_t candidateGenerationTimeMicros = 0;
    uint64_t selectionTimeMicros = 0;
    unsigned opAliasPairCount = 0;
    unsigned demandCandidateCount = 0;
    unsigned structuralStateCount = 0;
    unsigned structuralStateCacheHitCount = 0;
    unsigned structuralStateCacheMissCount = 0;
    unsigned selectedCandidateCount = 0;
    unsigned rejectedOverlapCandidateCount = 0;
    unsigned rejectedTemporalCandidateCount = 0;
    unsigned rejectedSupportCapacityCandidateCount = 0;
    unsigned rejectedSpatialPoolCandidateCount = 0;
    unsigned objectiveDroppedCandidateCount = 0;
    unsigned conservativeFallbackCount = 0;
    unsigned overlapEdgeCount = 0;
    unsigned supportClassCount = 0;
    unsigned configClassCount = 0;
    unsigned temporalRiskCount = 0;
    unsigned selectedFusedOpCount = 0;
    unsigned selectedInternalEdgeCount = 0;
    unsigned selectedCandidateChoiceCount = 0;
    unsigned selectedConfigDiversityCount = 0;
    unsigned selectedLegacyFallbackCount = 0;
    unsigned selectedMixedOriginCount = 0;
    unsigned selectedLegacyDerivedCount = 0;
    unsigned selectionComponentCount = 0;
    unsigned exactComponentCount = 0;
    unsigned cpSatComponentCount = 0;
    unsigned greedyComponentCount = 0;
    unsigned fallbackNoCandidateCount = 0;
    unsigned fallbackRejectedCount = 0;
    unsigned conservativeFallbackCoveredCount = 0;
    unsigned conservativeFallbackMissingCount = 0;
    bool legacyOracleEnabled = false;
    bool legacyOracleRequired = false;
    unsigned legacyOracleCheckCount = 0;
    unsigned legacyOracleCandidateCount = 0;
    unsigned legacyOracleMissingCount = 0;
    unsigned legacyFallbackCount = 0;
    unsigned legacyFallbackCandidateCount = 0;
    unsigned legacyContaminatedCandidateCount = 0;
    unsigned legacyDerivedSourceCount = 0;
    unsigned feedbackReselectionCount = 0;
    unsigned feedbackFilteredCandidateCount = 0;
    unsigned feedbackPenaltyCount = 0;
    unsigned feedbackUnknownCandidateRefCount = 0;
    unsigned feedbackUnknownFamilyRefCount = 0;
    unsigned feedbackUnknownConfigClassRefCount = 0;
  };

  struct Candidate {
    IdIndex hwNodeId = INVALID_ID;
    unsigned supportClassId = std::numeric_limits<unsigned>::max();
    unsigned configClassId = std::numeric_limits<unsigned>::max();
    bool temporal = false;
    llvm::SmallVector<FUConfigField, 2> configFields;
  };

  struct Unit {
    IdIndex contractedNodeId = INVALID_ID;
    unsigned familyIndex = std::numeric_limits<unsigned>::max();
    unsigned selectedCandidateId = std::numeric_limits<unsigned>::max();
    unsigned selectionComponentId = std::numeric_limits<unsigned>::max();
    llvm::SmallVector<IdIndex, 4> swNodes;
    llvm::SmallVector<IdIndex, 4> internalEdges;
    llvm::SmallVector<IdIndex, 4> contractedInputPorts;
    llvm::SmallVector<IdIndex, 4> contractedOutputPorts;
    llvm::SmallVector<PortBinding, 4> inputBindings;
    llvm::SmallVector<PortBinding, 4> outputBindings;
    llvm::SmallVector<Candidate, 4> candidates;
    unsigned preferredCandidateIndex = 0;
    unsigned configClassId = std::numeric_limits<unsigned>::max();
    int64_t selectionScore = 0;
    bool configurable = false;
    bool conservativeFallback = false;
    bool demandOrigin = false;
    bool legacyFallbackOrigin = false;
    bool mixedOrigin = false;
  };

  struct Plan {
    Graph contractedDFG;
    Graph conservativeFallbackDFG;
    llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>
        contractedCandidates;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        contractedCandidateSupportClasses;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        contractedCandidateConfigClasses;
    llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>
        conservativeFallbackCandidates;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        conservativeFallbackCandidateSupportClasses;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        conservativeFallbackCandidateConfigClasses;
    llvm::DenseMap<IdIndex, llvm::SmallVector<Candidate, 4>>
        conservativeFallbackCandidateDetails;
    llvm::DenseMap<IdIndex, Candidate> conservativeFallbackPreferredCandidate;
    llvm::SmallVector<SupportClassInfo, 8> supportClasses;
    llvm::SmallVector<ConfigClassInfo, 8> configClasses;
    llvm::SmallVector<FallbackNodeInfo, 16> fallbackNodes;
    llvm::SmallVector<LegacyOracleSampleInfo, 8> legacyOracleMissingSamples;
    llvm::SmallVector<NodeTechInfo, 32> nodeTechInfos;
    llvm::SmallVector<SelectionComponentInfo, 16> selectionComponents;
    llvm::SmallVector<FamilyTechInfo, 16> familyTechInfos;
    llvm::SmallVector<CandidateSummaryInfo, 32> candidateSummaries;
    llvm::SmallVector<TemporalIncompatibilityInfo, 16> temporalIncompatibilities;
    llvm::SmallVector<IdIndex, 16> conservativeFallbackSwNodes;
    std::vector<Unit> units;
    std::vector<IdIndex> originalNodeToContractedNode;
    std::vector<IdIndex> originalPortToContractedPort;
    std::vector<IdIndex> originalEdgeToContractedEdge;
    std::vector<TechMappedEdgeKind> originalEdgeKinds;
    bool feedbackApplied = false;
    Feedback appliedFeedback;
    FeedbackResolutionInfo feedbackResolution;
    PlanMetrics metrics;
    double coverageScore = 1.0;
    std::string diagnostics;

    Plan() = default;

    Plan(const Plan &other)
        : contractedDFG(other.contractedDFG.clone()),
          conservativeFallbackDFG(other.conservativeFallbackDFG.clone()),
          contractedCandidates(other.contractedCandidates),
          contractedCandidateSupportClasses(
              other.contractedCandidateSupportClasses),
          contractedCandidateConfigClasses(other.contractedCandidateConfigClasses),
          conservativeFallbackCandidates(other.conservativeFallbackCandidates),
          conservativeFallbackCandidateSupportClasses(
              other.conservativeFallbackCandidateSupportClasses),
          conservativeFallbackCandidateConfigClasses(
              other.conservativeFallbackCandidateConfigClasses),
          conservativeFallbackCandidateDetails(
              other.conservativeFallbackCandidateDetails),
          conservativeFallbackPreferredCandidate(
              other.conservativeFallbackPreferredCandidate),
          supportClasses(other.supportClasses), configClasses(other.configClasses),
          fallbackNodes(other.fallbackNodes),
          legacyOracleMissingSamples(other.legacyOracleMissingSamples),
          nodeTechInfos(other.nodeTechInfos),
          selectionComponents(other.selectionComponents),
          familyTechInfos(other.familyTechInfos),
          candidateSummaries(other.candidateSummaries),
          temporalIncompatibilities(other.temporalIncompatibilities),
          conservativeFallbackSwNodes(other.conservativeFallbackSwNodes),
          units(other.units),
          originalNodeToContractedNode(other.originalNodeToContractedNode),
          originalPortToContractedPort(other.originalPortToContractedPort),
          originalEdgeToContractedEdge(other.originalEdgeToContractedEdge),
          originalEdgeKinds(other.originalEdgeKinds),
          feedbackApplied(other.feedbackApplied),
          appliedFeedback(other.appliedFeedback),
          feedbackResolution(other.feedbackResolution),
          metrics(other.metrics), coverageScore(other.coverageScore),
          diagnostics(other.diagnostics) {}

    Plan &operator=(const Plan &other) {
      if (this == &other)
        return *this;
      contractedDFG = other.contractedDFG.clone();
      conservativeFallbackDFG = other.conservativeFallbackDFG.clone();
      contractedCandidates = other.contractedCandidates;
      contractedCandidateSupportClasses =
          other.contractedCandidateSupportClasses;
      contractedCandidateConfigClasses = other.contractedCandidateConfigClasses;
      conservativeFallbackCandidates = other.conservativeFallbackCandidates;
      conservativeFallbackCandidateSupportClasses =
          other.conservativeFallbackCandidateSupportClasses;
      conservativeFallbackCandidateConfigClasses =
          other.conservativeFallbackCandidateConfigClasses;
      conservativeFallbackCandidateDetails =
          other.conservativeFallbackCandidateDetails;
      conservativeFallbackPreferredCandidate =
          other.conservativeFallbackPreferredCandidate;
      supportClasses = other.supportClasses;
      configClasses = other.configClasses;
      fallbackNodes = other.fallbackNodes;
      legacyOracleMissingSamples = other.legacyOracleMissingSamples;
      nodeTechInfos = other.nodeTechInfos;
      selectionComponents = other.selectionComponents;
      familyTechInfos = other.familyTechInfos;
      candidateSummaries = other.candidateSummaries;
      temporalIncompatibilities = other.temporalIncompatibilities;
      conservativeFallbackSwNodes = other.conservativeFallbackSwNodes;
      units = other.units;
      originalNodeToContractedNode = other.originalNodeToContractedNode;
      originalPortToContractedPort = other.originalPortToContractedPort;
      originalEdgeToContractedEdge = other.originalEdgeToContractedEdge;
      originalEdgeKinds = other.originalEdgeKinds;
      feedbackApplied = other.feedbackApplied;
      appliedFeedback = other.appliedFeedback;
      feedbackResolution = other.feedbackResolution;
      metrics = other.metrics;
      coverageScore = other.coverageScore;
      diagnostics = other.diagnostics;
      return *this;
    }

    Plan(Plan &&) = default;
    Plan &operator=(Plan &&) = default;
  };

  bool buildPlan(const Graph &dfg, mlir::ModuleOp adgModule, const Graph &adg,
                 Plan &plan);

  bool applyFeedback(const Graph &dfg, const Graph &adg, const Plan &seedPlan,
                     const Feedback &feedback, Plan &plan);

  bool expandPlanMapping(const Graph &originalDfg, const Graph &adg,
                         const Plan &plan, const MappingState &contractedState,
                         MappingState &expandedState,
                         llvm::SmallVectorImpl<FUConfigSelection> &fuConfigs);

  static const SupportClassInfo *findSupportClass(const Plan &plan,
                                                  unsigned id) {
    llvm::ArrayRef<SupportClassInfo> supportClasses = allSupportClasses(plan);
    if (id >= supportClasses.size())
      return nullptr;
    return &supportClasses[id];
  }

  static const SupportClassInfo *findSupportClass(const Plan &plan,
                                                  const Candidate &candidate) {
    return findSupportClass(plan, candidate.supportClassId);
  }

  static bool supportClassEnforcesHardCapacity(const Plan &plan, unsigned id) {
    const auto *info = findSupportClass(plan, id);
    return info ? info->enforceHardCapacity : false;
  }

  static unsigned supportClassCapacity(const Plan &plan, unsigned id) {
    const auto *info = findSupportClass(plan, id);
    return info ? info->capacity : 0;
  }

  static bool isTemporalSupportClass(const Plan &plan, unsigned id) {
    const auto *info = findSupportClass(plan, id);
    return info ? info->temporal : false;
  }

  static const ConfigClassInfo *findConfigClass(const Plan &plan, unsigned id) {
    llvm::ArrayRef<ConfigClassInfo> configClasses = allConfigClasses(plan);
    if (id >= configClasses.size())
      return nullptr;
    return &configClasses[id];
  }

  static const ConfigClassInfo *findConfigClass(const Plan &plan,
                                                const Candidate &candidate) {
    return findConfigClass(plan, candidate.configClassId);
  }

  static const ConfigClassInfo *findConfigClass(const Plan &plan,
                                                const Unit &unit) {
    return findConfigClass(plan, unit.configClassId);
  }

  static const ConfigClassInfo *findSelectedUnitConfigClass(const Plan &plan,
                                                            const Unit &unit) {
    return findConfigClass(plan, unit);
  }

  static bool areConfigClassesCompatible(const Plan &plan, unsigned lhs,
                                         unsigned rhs) {
    const auto *lhsInfo = findConfigClass(plan, lhs);
    const auto *rhsInfo = findConfigClass(plan, rhs);
    if (!lhsInfo || !rhsInfo)
      return false;
    if (!lhsInfo->temporal || !rhsInfo->temporal)
      return true;
    if (lhs == rhs)
      return true;
    return std::find(lhsInfo->compatibleConfigClassIds.begin(),
                     lhsInfo->compatibleConfigClassIds.end(),
                     rhs) != lhsInfo->compatibleConfigClassIds.end();
  }

  static bool isTemporalConfigClass(const Plan &plan, unsigned id) {
    const auto *info = findConfigClass(plan, id);
    return info ? info->temporal : false;
  }

  static llvm::ArrayRef<unsigned> compatibleConfigClasses(const Plan &plan,
                                                          unsigned id) {
    const auto *info = findConfigClass(plan, id);
    if (!info)
      return {};
    return info->compatibleConfigClassIds;
  }

  static bool hasLegacyOracleGap(const Plan &plan) {
    return plan.metrics.legacyOracleEnabled &&
           plan.metrics.legacyOracleMissingCount != 0;
  }

  static llvm::StringRef
  originKind(bool demandOrigin, bool legacyFallbackOrigin, bool mixedOrigin) {
    if (mixedOrigin)
      return "mixed";
    if (legacyFallbackOrigin)
      return "legacy_only";
    if (demandOrigin)
      return "demand";
    return "unknown";
  }

  static bool selectedPlanUsesLegacyFallback(const Plan &plan) {
    return plan.metrics.selectedLegacyFallbackCount != 0;
  }

  static unsigned feedbackUnknownCandidateRefCount(const Plan &plan) {
    if (!hasAppliedFeedback(plan))
      return plan.metrics.feedbackUnknownCandidateRefCount;
    const auto &resolution = feedbackResolution(plan);
    return resolution.unknownBannedCandidateIds.size() +
           resolution.unknownSplitCandidateIds.size() +
           resolution.unknownCandidatePenalties.size();
  }

  static unsigned feedbackUnknownFamilyRefCount(const Plan &plan) {
    if (!hasAppliedFeedback(plan))
      return plan.metrics.feedbackUnknownFamilyRefCount;
    const auto &resolution = feedbackResolution(plan);
    return resolution.unknownBannedFamilyIds.size() +
           resolution.unknownFamilyPenalties.size();
  }

  static unsigned feedbackUnknownConfigClassRefCount(const Plan &plan) {
    if (!hasAppliedFeedback(plan))
      return plan.metrics.feedbackUnknownConfigClassRefCount;
    const auto &resolution = feedbackResolution(plan);
    return resolution.unknownBannedConfigClassIds.size() +
           resolution.unknownConfigClassPenalties.size();
  }

  static bool selectedPlanUsesMixedOrigin(const Plan &plan) {
    return plan.metrics.selectedMixedOriginCount != 0;
  }

  static bool selectedPlanUsesLegacyDerivedSupport(const Plan &plan) {
    return plan.metrics.selectedLegacyDerivedCount != 0;
  }

  static bool candidatePoolUsesPureLegacyFallback(const Plan &plan) {
    return plan.metrics.legacyFallbackCandidateCount != 0;
  }

  static bool candidatePoolUsesLegacyDerivedSupport(const Plan &plan) {
    return plan.metrics.legacyContaminatedCandidateCount != 0;
  }

  static bool sourcePoolUsesPureLegacyFallback(const Plan &plan) {
    return plan.metrics.legacyFallbackCount != 0;
  }

  static bool sourcePoolUsesLegacyDerivedSupport(const Plan &plan) {
    return plan.metrics.legacyDerivedSourceCount != 0;
  }

  static bool isDemandDrivenPrimaryPlan(const Plan &plan) {
    return !candidatePoolUsesLegacyDerivedSupport(plan);
  }

  static bool demandDrivenPlanVerifiedForMapperHandoff(const Plan &plan) {
    return plan.metrics.legacyOracleEnabled &&
           plan.metrics.legacyOracleMissingCount == 0 &&
           !selectedPlanUsesLegacyDerivedSupport(plan) &&
           isDemandDrivenPrimaryPlan(plan);
  }

  static llvm::StringRef mapperHandoffStatus(const Plan &plan) {
    if (demandDrivenPlanVerifiedForMapperHandoff(plan))
      return "ready_for_mapper_contract";
    if (hasLegacyOracleGap(plan))
      return "blocked_legacy_oracle_gap";
    if (selectedPlanUsesLegacyFallback(plan))
      return "blocked_selected_legacy_fallback";
    if (selectedPlanUsesMixedOrigin(plan))
      return "blocked_selected_mixed_origin";
    if (candidatePoolUsesLegacyDerivedSupport(plan))
      return "blocked_legacy_derived_candidate_pool";
    return "unverified_oracle_disabled";
  }

  static void collectMapperHandoffBlockers(
      const Plan &plan, llvm::SmallVectorImpl<llvm::StringRef> &blockers) {
    blockers.clear();
    if (!plan.metrics.legacyOracleEnabled)
      blockers.push_back("legacy_oracle_disabled");
    if (hasLegacyOracleGap(plan))
      blockers.push_back("legacy_oracle_gap");
    if (selectedPlanUsesLegacyFallback(plan))
      blockers.push_back("selected_legacy_fallback");
    if (selectedPlanUsesMixedOrigin(plan))
      blockers.push_back("selected_mixed_origin");
    if (candidatePoolUsesLegacyDerivedSupport(plan))
      blockers.push_back("legacy_derived_candidate_pool");
  }

  static const TemporalIncompatibilityInfo *
  findTemporalIncompatibility(const Plan &plan, unsigned lhs, unsigned rhs) {
    for (const auto &info : temporalIncompatibilities(plan)) {
      if ((info.lhsConfigClassId == lhs && info.rhsConfigClassId == rhs) ||
          (info.lhsConfigClassId == rhs && info.rhsConfigClassId == lhs)) {
        return &info;
      }
    }
    return nullptr;
  }

  static llvm::ArrayRef<TemporalIncompatibilityInfo>
  temporalIncompatibilities(const Plan &plan) {
    return plan.temporalIncompatibilities;
  }

  static llvm::SmallVectorImpl<TemporalIncompatibilityInfo> &
  temporalIncompatibilities(Plan &plan) {
    return plan.temporalIncompatibilities;
  }

  static const CandidateSummaryInfo *findCandidateSummary(const Plan &plan,
                                                          unsigned id) {
    llvm::ArrayRef<CandidateSummaryInfo> candidateSummaries =
        allCandidateSummaries(plan);
    if (id >= candidateSummaries.size())
      return nullptr;
    return &candidateSummaries[id];
  }

  static llvm::ArrayRef<SupportClassInfo> allSupportClasses(const Plan &plan) {
    return plan.supportClasses;
  }

  static llvm::SmallVectorImpl<SupportClassInfo> &allSupportClasses(Plan &plan) {
    return plan.supportClasses;
  }

  static llvm::ArrayRef<ConfigClassInfo> allConfigClasses(const Plan &plan) {
    return plan.configClasses;
  }

  static llvm::SmallVectorImpl<ConfigClassInfo> &allConfigClasses(Plan &plan) {
    return plan.configClasses;
  }

  static llvm::ArrayRef<FallbackNodeInfo> allFallbackNodes(const Plan &plan) {
    return plan.fallbackNodes;
  }

  static llvm::SmallVectorImpl<FallbackNodeInfo> &allFallbackNodes(Plan &plan) {
    return plan.fallbackNodes;
  }

  static llvm::ArrayRef<SelectionComponentInfo>
  allSelectionComponents(const Plan &plan) {
    return plan.selectionComponents;
  }

  static llvm::SmallVectorImpl<SelectionComponentInfo> &
  allSelectionComponents(Plan &plan) {
    return plan.selectionComponents;
  }

  static llvm::ArrayRef<FamilyTechInfo> allFamilyTechInfos(const Plan &plan) {
    return plan.familyTechInfos;
  }

  static llvm::SmallVectorImpl<FamilyTechInfo> &allFamilyTechInfos(Plan &plan) {
    return plan.familyTechInfos;
  }

  static llvm::ArrayRef<CandidateSummaryInfo>
  allCandidateSummaries(const Plan &plan) {
    return plan.candidateSummaries;
  }

  static llvm::SmallVectorImpl<CandidateSummaryInfo> &
  allCandidateSummaries(Plan &plan) {
    return plan.candidateSummaries;
  }

  static llvm::ArrayRef<NodeTechInfo> allNodeTechInfos(const Plan &plan) {
    return plan.nodeTechInfos;
  }

  static llvm::SmallVectorImpl<NodeTechInfo> &allNodeTechInfos(Plan &plan) {
    return plan.nodeTechInfos;
  }

  static llvm::ArrayRef<Unit> allUnits(const Plan &plan) {
    return plan.units;
  }

  static std::vector<Unit> &allUnits(Plan &plan) {
    return plan.units;
  }

  static bool hasAppliedFeedback(const Plan &plan) {
    return plan.feedbackApplied;
  }

  static const Feedback &appliedFeedback(const Plan &plan) {
    return plan.appliedFeedback;
  }

  static Feedback &appliedFeedback(Plan &plan) {
    return plan.appliedFeedback;
  }

  static const FeedbackResolutionInfo &feedbackResolution(const Plan &plan) {
    return plan.feedbackResolution;
  }

  static FeedbackResolutionInfo &feedbackResolution(Plan &plan) {
    return plan.feedbackResolution;
  }

  static void markFeedbackApplied(Plan &plan, const Feedback &feedback) {
    plan.feedbackApplied = true;
    appliedFeedback(plan) = feedback;
    feedbackResolution(plan) = FeedbackResolutionInfo();
  }

  static llvm::ArrayRef<IdIndex>
  conservativeFallbackSwNodes(const Plan &plan) {
    return plan.conservativeFallbackSwNodes;
  }

  static llvm::SmallVectorImpl<IdIndex> &
  conservativeFallbackSwNodes(Plan &plan) {
    return plan.conservativeFallbackSwNodes;
  }

  static const Graph &conservativeFallbackDFG(const Plan &plan) {
    return plan.conservativeFallbackDFG;
  }

  static llvm::ArrayRef<LegacyOracleSampleInfo>
  legacyOracleMissingSamples(const Plan &plan) {
    return plan.legacyOracleMissingSamples;
  }

  static llvm::SmallVectorImpl<LegacyOracleSampleInfo> &
  legacyOracleMissingSamples(Plan &plan) {
    return plan.legacyOracleMissingSamples;
  }

  static const CandidateSummaryInfo *findSelectedCandidateSummary(
      const Plan &plan, const Unit &unit) {
    return findCandidateSummary(plan, unit.selectedCandidateId);
  }

  static CandidateSummaryInfo *findCandidateSummary(Plan &plan, unsigned id) {
    auto &candidateSummaries = allCandidateSummaries(plan);
    if (id >= candidateSummaries.size())
      return nullptr;
    return &candidateSummaries[id];
  }

  static CandidateSummaryInfo *findSelectedCandidateSummary(Plan &plan,
                                                            const Unit &unit) {
    return findCandidateSummary(plan, unit.selectedCandidateId);
  }

  static const llvm::SmallVector<IdIndex, 4> *
  findContractedCandidates(const Plan &plan, IdIndex contractedNodeId) {
    auto it = plan.contractedCandidates.find(contractedNodeId);
    if (it == plan.contractedCandidates.end())
      return nullptr;
    return &it->second;
  }

  static const llvm::SmallVector<unsigned, 4> *
  findContractedCandidateSupportClasses(const Plan &plan,
                                        IdIndex contractedNodeId) {
    auto it = plan.contractedCandidateSupportClasses.find(contractedNodeId);
    if (it == plan.contractedCandidateSupportClasses.end())
      return nullptr;
    return &it->second;
  }

  static const llvm::SmallVector<unsigned, 4> *
  findContractedCandidateConfigClasses(const Plan &plan,
                                       IdIndex contractedNodeId) {
    auto it = plan.contractedCandidateConfigClasses.find(contractedNodeId);
    if (it == plan.contractedCandidateConfigClasses.end())
      return nullptr;
    return &it->second;
  }

  static const llvm::SmallVector<IdIndex, 4> *
  findConservativeFallbackCandidates(const Plan &plan, IdIndex swNodeId) {
    auto it = plan.conservativeFallbackCandidates.find(swNodeId);
    if (it == plan.conservativeFallbackCandidates.end())
      return nullptr;
    return &it->second;
  }

  static const llvm::SmallVector<unsigned, 4> *
  findConservativeFallbackCandidateSupportClasses(const Plan &plan,
                                                  IdIndex swNodeId) {
    auto it = plan.conservativeFallbackCandidateSupportClasses.find(swNodeId);
    if (it == plan.conservativeFallbackCandidateSupportClasses.end())
      return nullptr;
    return &it->second;
  }

  static const llvm::SmallVector<unsigned, 4> *
  findConservativeFallbackCandidateConfigClasses(const Plan &plan,
                                                 IdIndex swNodeId) {
    auto it = plan.conservativeFallbackCandidateConfigClasses.find(swNodeId);
    if (it == plan.conservativeFallbackCandidateConfigClasses.end())
      return nullptr;
    return &it->second;
  }

  static const llvm::SmallVector<Candidate, 4> *
  findConservativeFallbackCandidateDetails(const Plan &plan, IdIndex swNodeId) {
    auto it = plan.conservativeFallbackCandidateDetails.find(swNodeId);
    if (it == plan.conservativeFallbackCandidateDetails.end())
      return nullptr;
    return &it->second;
  }

  static const Candidate *findConservativeFallbackPreferredCandidate(
      const Plan &plan, IdIndex swNodeId) {
    auto it = plan.conservativeFallbackPreferredCandidate.find(swNodeId);
    if (it == plan.conservativeFallbackPreferredCandidate.end())
      return nullptr;
    return &it->second;
  }

  static const SelectionComponentInfo *
  findSelectionComponent(const Plan &plan, unsigned id) {
    llvm::ArrayRef<SelectionComponentInfo> selectionComponents =
        allSelectionComponents(plan);
    if (id >= selectionComponents.size())
      return nullptr;
    return &selectionComponents[id];
  }

  static const SelectionComponentInfo *findSelectionComponent(
      const Plan &plan, const Unit &unit) {
    return findSelectionComponent(plan, unit.selectionComponentId);
  }

  static SelectionComponentInfo *findSelectionComponent(Plan &plan,
                                                        unsigned id) {
    auto &selectionComponents = allSelectionComponents(plan);
    if (id >= selectionComponents.size())
      return nullptr;
    return &selectionComponents[id];
  }

  static SelectionComponentInfo *findSelectionComponent(Plan &plan,
                                                        const Unit &unit) {
    return findSelectionComponent(plan, unit.selectionComponentId);
  }

  static const FamilyTechInfo *findFamilyTechInfo(const Plan &plan,
                                                  unsigned id) {
    llvm::ArrayRef<FamilyTechInfo> familyTechInfos = allFamilyTechInfos(plan);
    if (id >= familyTechInfos.size())
      return nullptr;
    return &familyTechInfos[id];
  }

  static const FamilyTechInfo *findFamilyTechInfo(const Plan &plan,
                                                  const Unit &unit) {
    return findFamilyTechInfo(plan, unit.familyIndex);
  }

  static FamilyTechInfo *findFamilyTechInfo(Plan &plan, unsigned id) {
    auto &familyTechInfos = allFamilyTechInfos(plan);
    if (id >= familyTechInfos.size())
      return nullptr;
    return &familyTechInfos[id];
  }

  static FamilyTechInfo *findFamilyTechInfo(Plan &plan, const Unit &unit) {
    return findFamilyTechInfo(plan, unit.familyIndex);
  }

  static const Unit *findUnit(const Plan &plan, unsigned unitIndex) {
    llvm::ArrayRef<Unit> units = allUnits(plan);
    if (unitIndex >= units.size())
      return nullptr;
    return &units[unitIndex];
  }

  static Unit *findUnit(Plan &plan, unsigned unitIndex) {
    auto &units = allUnits(plan);
    if (unitIndex >= units.size())
      return nullptr;
    return &units[unitIndex];
  }

  static std::optional<IdIndex> findForcedTemporalHwNodeId(
      const Unit &unit, const Graph &adg) {
    std::optional<IdIndex> forcedHwNodeId;
    if (unit.candidates.empty())
      return std::nullopt;
    for (const auto &candidate : unit.candidates) {
      const Node *hwNode = adg.getNode(candidate.hwNodeId);
      if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "functional" ||
          getNodeAttrStr(hwNode, "pe_kind") != "temporal_pe") {
        return std::nullopt;
      }
      if (!forcedHwNodeId) {
        forcedHwNodeId = candidate.hwNodeId;
        continue;
      }
      if (*forcedHwNodeId != candidate.hwNodeId)
        return std::nullopt;
    }
    return forcedHwNodeId;
  }

  static std::optional<IdIndex> findForcedTemporalHwNodeId(
      const Plan &plan, unsigned unitIndex, const Graph &adg) {
    const auto *unit = findUnit(plan, unitIndex);
    if (!unit)
      return std::nullopt;
    return findForcedTemporalHwNodeId(*unit, adg);
  }

  static const Candidate *findPreferredUnitCandidate(const Unit &unit) {
    if (unit.preferredCandidateIndex >= unit.candidates.size())
      return nullptr;
    return &unit.candidates[unit.preferredCandidateIndex];
  }

  static const Candidate *findPreferredUnitCandidate(const Plan &plan,
                                                     unsigned unitIndex) {
    const auto *unit = findUnit(plan, unitIndex);
    if (!unit)
      return nullptr;
    return findPreferredUnitCandidate(*unit);
  }

  static const SupportClassInfo *findPreferredUnitSupportClass(
      const Plan &plan, const Unit &unit) {
    const auto *candidate = findPreferredUnitCandidate(unit);
    if (!candidate)
      return nullptr;
    return findSupportClass(plan, *candidate);
  }

  static const SupportClassInfo *findPreferredUnitSupportClass(
      const Plan &plan, unsigned unitIndex) {
    const auto *unit = findUnit(plan, unitIndex);
    if (!unit)
      return nullptr;
    return findPreferredUnitSupportClass(plan, *unit);
  }

  static const ConfigClassInfo *findPreferredUnitConfigClass(
      const Plan &plan, const Unit &unit) {
    const auto *candidate = findPreferredUnitCandidate(unit);
    if (!candidate)
      return nullptr;
    return findConfigClass(plan, *candidate);
  }

  static const ConfigClassInfo *findPreferredUnitConfigClass(
      const Plan &plan, unsigned unitIndex) {
    const auto *unit = findUnit(plan, unitIndex);
    if (!unit)
      return nullptr;
    return findPreferredUnitConfigClass(plan, *unit);
  }

  static const Unit *findUnitForContractedNode(const Plan &plan,
                                               IdIndex contractedNodeId) {
    for (const auto &unit : allUnits(plan)) {
      if (unit.contractedNodeId == contractedNodeId)
        return &unit;
    }
    return nullptr;
  }

  static const NodeTechInfo *findNodeTechInfo(const Plan &plan,
                                              IdIndex swNodeId) {
    for (const auto &info : allNodeTechInfos(plan)) {
      if (info.swNodeId == swNodeId)
        return &info;
    }
    return nullptr;
  }

  static const FallbackNodeInfo *findFallbackNodeInfo(const Plan &plan,
                                                      IdIndex swNodeId) {
    for (const auto &info : allFallbackNodes(plan)) {
      if (info.swNodeId == swNodeId)
        return &info;
    }
    return nullptr;
  }

  static NodeTechInfo *findNodeTechInfo(Plan &plan, IdIndex swNodeId) {
    for (auto &info : allNodeTechInfos(plan)) {
      if (info.swNodeId == swNodeId)
        return &info;
    }
    return nullptr;
  }

  static FallbackNodeInfo *findFallbackNodeInfo(Plan &plan, IdIndex swNodeId) {
    for (auto &info : allFallbackNodes(plan)) {
      if (info.swNodeId == swNodeId)
        return &info;
    }
    return nullptr;
  }
};

} // namespace fcc

#endif // FCC_MAPPER_TECHMAPPER_H
