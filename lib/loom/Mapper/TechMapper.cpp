#include "TechMapperInternal.h"
#include "loom/Mapper/OpCompat.h"
#include "loom/Mapper/TypeCompat.h"

#include "loom/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef LOOM_HAVE_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#endif

namespace loom {

using techmapper_detail::FamilyMatch;
using techmapper_detail::findFunctionUnitNode;
using techmapper_detail::collectVariantsForFU;
using techmapper_detail::findDemandDrivenMatchesForFU;
using techmapper_detail::findMatchesForFamily;
using techmapper_detail::Match;
using techmapper_detail::VariantFamily;

#ifdef LOOM_HAVE_ORTOOLS
using namespace operations_research::sat;
#endif

namespace {

void addNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute value,
                 mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), value));
}

void addUIntNodeAttr(Node *node, llvm::StringRef key, uint64_t value,
                     mlir::MLIRContext *ctx) {
  addNodeAttr(node, key,
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value),
              ctx);
}

void addBoolNodeAttr(Node *node, llvm::StringRef key, bool value,
                     mlir::MLIRContext *ctx) {
  addNodeAttr(node, key, mlir::BoolAttr::get(ctx, value), ctx);
}

std::unique_ptr<Node> cloneNodeShell(const Node *src, mlir::MLIRContext *ctx) {
  auto node = std::make_unique<Node>();
  node->kind = src->kind;
  node->attributes = src->attributes;
  (void)ctx;
  return node;
}

std::unique_ptr<Port> clonePort(const Port *src) {
  auto port = std::make_unique<Port>();
  port->direction = src->direction;
  port->type = src->type;
  port->attributes = src->attributes;
  return port;
}

std::string serializeConfigFields(llvm::ArrayRef<FUConfigField> fields) {
  llvm::SmallVector<std::string, 4> tokens;
  tokens.reserve(fields.size());
  for (const auto &field : fields) {
    std::string token;
    llvm::raw_string_ostream os(token);
    os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
       << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
       << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
       << field.disconnect;
    tokens.push_back(os.str());
  }
  std::sort(tokens.begin(), tokens.end());
  std::string text;
  llvm::raw_string_ostream joined(text);
  for (size_t idx = 0; idx < tokens.size(); ++idx) {
    if (idx)
      joined << ";";
    joined << tokens[idx];
  }
  return text;
}

llvm::SmallVector<mlir::Attribute, 4>
buildConfigFieldSummaryAttrs(llvm::ArrayRef<FUConfigField> fields,
                             mlir::MLIRContext *ctx) {
  llvm::SmallVector<std::string, 4> tokens;
  tokens.reserve(fields.size());
  for (const auto &field : fields) {
    std::string summary;
    llvm::raw_string_ostream os(summary);
    os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
       << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
       << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
       << field.disconnect;
    tokens.push_back(os.str());
  }
  std::sort(tokens.begin(), tokens.end());
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (const auto &token : tokens) {
    attrs.push_back(mlir::StringAttr::get(ctx, token));
  }
  return attrs;
}

std::string buildAggregatedMatchKey(unsigned familyIndex, const Match &match) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << familyIndex << "|nodes(";
  for (size_t idx = 0; idx < match.swNodesByOp.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.swNodesByOp[idx];
  }
  os << ")|in(";
  for (size_t idx = 0; idx < match.inputBindings.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.inputBindings[idx].swPortId << "->"
       << match.inputBindings[idx].hwPortIndex;
  }
  os << ")|out(";
  for (size_t idx = 0; idx < match.outputBindings.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.outputBindings[idx].swPortId << "->"
       << match.outputBindings[idx].hwPortIndex;
  }
  os << ")|edge(";
  for (size_t idx = 0; idx < match.internalEdges.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.internalEdges[idx];
  }
  os << ")|cfg(" << serializeConfigFields(match.configFields) << ")";
  return text;
}

bool isTemporalCandidateList(llvm::ArrayRef<IdIndex> hwNodeIds, const Graph &adg) {
  if (hwNodeIds.empty())
    return false;
  for (IdIndex hwNodeId : hwNodeIds) {
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "temporal_pe")
      return false;
  }
  return true;
}

int64_t computeFamilyScarcityPenalty(size_t hwSupportCount) {
  if (hwSupportCount >= 4)
    return 0;
  return static_cast<int64_t>(4 - hwSupportCount) * 48;
}

unsigned countCommutativeSwaps(
    llvm::ArrayRef<llvm::SmallVector<unsigned, 4>> operandOrderByOp) {
  unsigned swaps = 0;
  for (const auto &operandOrder : operandOrderByOp) {
    for (unsigned operandIdx = 0; operandIdx < operandOrder.size();
         ++operandIdx) {
      if (operandOrder[operandIdx] != operandIdx) {
        ++swaps;
        break;
      }
    }
  }
  return swaps;
}

int64_t scoreAggregatedMatch(const AggregatedMatch &match,
                             size_t familyHwSupportCount) {
  int64_t score = 0;
  score += static_cast<int64_t>(match.swNodesByOp.size()) * 1024;
  score += static_cast<int64_t>(match.internalEdges.size()) * 192;
  score += static_cast<int64_t>(match.hwNodeIds.size()) * 32;
  score -= static_cast<int64_t>(match.inputBindings.size() + match.outputBindings.size()) *
           48;
  score -= static_cast<int64_t>(match.configFields.size()) * 12;
  score -= computeFamilyScarcityPenalty(familyHwSupportCount);
  score -= static_cast<int64_t>(match.commutativeSwapCount) * 16;
  if (match.temporal)
    score -= 64;
  return score;
}

int64_t scoreAggregatedMatch(const AggregatedMatch &match) {
  return scoreAggregatedMatch(match, match.hwNodeIds.size());
}

std::string buildTechmapDiagnostics(const TechMapper::Plan &plan) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "coverage=" << plan.coverageScore;
  os << ", total_layer2_us=" << plan.metrics.totalLayer2TimeMicros;
  os << ", candidate_gen_us=" << plan.metrics.candidateGenerationTimeMicros;
  os << ", selection_us=" << plan.metrics.selectionTimeMicros;
  os << ", op_alias_pairs=" << plan.metrics.opAliasPairCount;
  os << ", demand_candidates=" << plan.metrics.demandCandidateCount;
  os << ", structural_states=" << plan.metrics.structuralStateCount;
  os << ", structural_cache_hits="
     << plan.metrics.structuralStateCacheHitCount;
  os << ", structural_cache_misses="
     << plan.metrics.structuralStateCacheMissCount;
  os << ", selected_candidates=" << plan.metrics.selectedCandidateCount;
  if (plan.metrics.rejectedOverlapCandidateCount != 0)
    os << ", rejected_overlap_candidates="
       << plan.metrics.rejectedOverlapCandidateCount;
  if (plan.metrics.rejectedTemporalCandidateCount != 0)
    os << ", rejected_temporal_candidates="
       << plan.metrics.rejectedTemporalCandidateCount;
  if (plan.metrics.rejectedSupportCapacityCandidateCount != 0)
    os << ", rejected_support_capacity_candidates="
       << plan.metrics.rejectedSupportCapacityCandidateCount;
  if (plan.metrics.rejectedSpatialPoolCandidateCount != 0)
    os << ", rejected_spatial_pool_candidates="
       << plan.metrics.rejectedSpatialPoolCandidateCount;
  if (plan.metrics.objectiveDroppedCandidateCount != 0)
    os << ", objective_dropped_candidates="
       << plan.metrics.objectiveDroppedCandidateCount;
  os << ", fallback_nodes=" << plan.metrics.conservativeFallbackCount;
  os << ", support_classes=" << plan.metrics.supportClassCount;
  os << ", config_classes=" << plan.metrics.configClassCount;
  os << ", fused_ops=" << plan.metrics.selectedFusedOpCount;
  os << ", internal_edges=" << plan.metrics.selectedInternalEdgeCount;
  os << ", candidate_choices=" << plan.metrics.selectedCandidateChoiceCount;
  os << ", selected_config_diversity="
     << plan.metrics.selectedConfigDiversityCount;
  if (plan.metrics.selectedLegacyFallbackCount != 0)
    os << ", selected_legacy_fallback="
       << plan.metrics.selectedLegacyFallbackCount;
  if (plan.metrics.selectedMixedOriginCount != 0)
    os << ", selected_mixed_origin=" << plan.metrics.selectedMixedOriginCount;
  if (plan.metrics.selectedLegacyDerivedCount != 0)
    os << ", selected_legacy_derived="
       << plan.metrics.selectedLegacyDerivedCount;
  os << ", selection_components=" << plan.metrics.selectionComponentCount;
  if (plan.metrics.cpSatComponentCount != 0)
    os << ", cpsat_components=" << plan.metrics.cpSatComponentCount;
  if (plan.metrics.exactComponentCount != 0)
    os << ", exact_components=" << plan.metrics.exactComponentCount;
  if (plan.metrics.greedyComponentCount != 0)
    os << ", greedy_components=" << plan.metrics.greedyComponentCount;
  if (plan.metrics.temporalRiskCount != 0)
    os << ", temporal_risk=" << plan.metrics.temporalRiskCount;
  if (plan.metrics.legacyOracleCandidateCount != 0 ||
      plan.metrics.legacyOracleMissingCount != 0) {
    os << ", legacy_oracle_candidates="
       << plan.metrics.legacyOracleCandidateCount;
    os << ", legacy_oracle_missing=" << plan.metrics.legacyOracleMissingCount;
  }
  if (plan.metrics.legacyOracleEnabled)
    os << ", legacy_oracle_checks=" << plan.metrics.legacyOracleCheckCount;
  if (plan.metrics.legacyOracleRequired)
    os << ", legacy_oracle_required=1";
  if (plan.metrics.legacyFallbackCount != 0)
    os << ", legacy_fallback_sources=" << plan.metrics.legacyFallbackCount;
  if (plan.metrics.legacyFallbackCandidateCount != 0)
    os << ", legacy_fallback_candidates="
       << plan.metrics.legacyFallbackCandidateCount;
  if (plan.metrics.legacyContaminatedCandidateCount != 0)
    os << ", legacy_contaminated_candidates="
       << plan.metrics.legacyContaminatedCandidateCount;
  if (plan.metrics.legacyDerivedSourceCount != 0)
    os << ", legacy_derived_sources="
       << plan.metrics.legacyDerivedSourceCount;
  if (plan.metrics.feedbackReselectionCount != 0)
    os << ", feedback_reselection=" << plan.metrics.feedbackReselectionCount;
  if (plan.metrics.feedbackFilteredCandidateCount != 0)
    os << ", feedback_filtered_candidates="
       << plan.metrics.feedbackFilteredCandidateCount;
  if (plan.metrics.feedbackPenaltyCount != 0)
    os << ", feedback_penalty_terms=" << plan.metrics.feedbackPenaltyCount;
  if (TechMapper::feedbackUnknownCandidateRefCount(plan) != 0)
    os << ", feedback_unknown_candidate_refs="
       << TechMapper::feedbackUnknownCandidateRefCount(plan);
  if (TechMapper::feedbackUnknownFamilyRefCount(plan) != 0)
    os << ", feedback_unknown_family_refs="
       << TechMapper::feedbackUnknownFamilyRefCount(plan);
  if (TechMapper::feedbackUnknownConfigClassRefCount(plan) != 0)
    os << ", feedback_unknown_config_class_refs="
       << TechMapper::feedbackUnknownConfigClassRefCount(plan);
  if (plan.metrics.fallbackNoCandidateCount != 0 ||
      plan.metrics.fallbackRejectedCount != 0) {
    os << ", fallback_no_candidate=" << plan.metrics.fallbackNoCandidateCount;
    os << ", fallback_rejected=" << plan.metrics.fallbackRejectedCount;
  }
  os << ", conservative_fallback_covered="
     << plan.metrics.conservativeFallbackCoveredCount;
  if (plan.metrics.conservativeFallbackMissingCount != 0)
    os << ", conservative_fallback_missing="
       << plan.metrics.conservativeFallbackMissingCount;
  return text;
}

} // namespace

bool TechMapper::buildPlan(const Graph &dfg, mlir::ModuleOp adgModule,
                           const Graph &adg, Plan &plan) {
  const auto layer2StartTime = std::chrono::steady_clock::now();
  auto candidateGenerationStartTime = layer2StartTime;
  bool candidateGenerationTimed = false;
  bool selectionTimed = false;
  Plan freshPlan;
  plan = std::move(freshPlan);
  plan.originalNodeToContractedNode.assign(dfg.nodes.size(), INVALID_ID);
  plan.originalPortToContractedPort.assign(dfg.ports.size(), INVALID_ID);
  plan.originalEdgeToContractedEdge.assign(dfg.edges.size(), INVALID_ID);
  plan.originalEdgeKinds.assign(dfg.edges.size(), TechMappedEdgeKind::Routed);
  plan.contractedDFG = Graph(dfg.context);
  plan.conservativeFallbackDFG = dfg.clone();
  plan.metrics.opAliasPairCount = opcompat::getAliasPairs().size();
  plan.coverageScore = 1.0;
  unsigned totalOpCount = 0;
  for (const Node *swNode : dfg.nodeRange()) {
    if (swNode && swNode->kind == Node::OperationNode)
      ++totalOpCount;
  }

  llvm::SmallVector<VariantFamily, 16> familyList;

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  adgModule.walk([&](loom::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<loom::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<loom::fabric::SpatialPEOp, loom::fabric::TemporalPEOp>(op))
      return false;
    return !op->hasAttr("inline_instantiation");
  };

  llvm::StringMap<loom::fabric::SpatialPEOp> peDefs;
  llvm::StringMap<loom::fabric::TemporalPEOp> temporalPeDefs;
  llvm::StringMap<loom::fabric::FunctionUnitOp> functionUnitDefs;
  adgModule->walk([&](loom::fabric::SpatialPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      peDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](loom::fabric::TemporalPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      temporalPeDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](loom::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefs[symName] = fuOp;
  });

  bool builtTechUnits = false;
  if (auto fabricMod = [&]() -> loom::fabric::ModuleOp {
        loom::fabric::ModuleOp found;
        adgModule->walk([&](loom::fabric::ModuleOp op) {
          if (!found)
            found = op;
        });
        return found;
      }()) {
    auto visitPEFunctionUnits =
        [&](auto peOp, llvm::StringRef peName,
            auto &&visitor) {
          auto &peBody = peOp.getBody().front();
          auto referencedIt = referencedTargetsByBlock.find(&peBody);
          const llvm::DenseSet<llvm::StringRef> *referencedTargets =
              referencedIt != referencedTargetsByBlock.end()
                  ? &referencedIt->second
                  : nullptr;
          for (mlir::Operation &bodyOp : peBody.getOperations()) {
            if (auto fuOp = mlir::dyn_cast<loom::fabric::FunctionUnitOp>(bodyOp)) {
              llvm::StringRef symName = fuOp.getSymNameAttr().getValue();
              if (!symName.empty() && referencedTargets &&
                  referencedTargets->contains(symName))
                continue;
              visitor(fuOp, symName.str());
              continue;
            }
            auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(bodyOp);
            if (!instOp)
              continue;
            auto fuIt = functionUnitDefs.find(instOp.getModule());
            if (fuIt == functionUnitDefs.end())
              continue;
            visitor(fuIt->second,
                    instOp.getSymName().value_or(instOp.getModule()).str());
          }
        };

    std::map<std::string, unsigned> familyIndexBySignature;
    std::map<std::string, unsigned> demandMaterializedStateCountBySignature;
    std::map<std::string, unsigned> legacyMaterializedStateCountBySignature;
    std::map<std::string, unsigned> aggregatedIndexByKey;
    std::vector<AggregatedMatch> aggregatedMatches;
    llvm::SmallVector<TechMapper::LegacyOracleSampleInfo, 4>
        legacyOracleMissingSampleBuffer;
    const bool runLegacyOracle = []() {
      const char *env = std::getenv("LOOM_TECHMAP_RUN_LEGACY_ORACLE");
      return env && std::string(env) == "1";
    }();
    plan.metrics.legacyOracleEnabled = runLegacyOracle;
    const bool requireLegacyOracleSuperset = []() {
      const char *env = std::getenv("LOOM_TECHMAP_REQUIRE_LEGACY_ORACLE_SUPERSET");
      return env && std::string(env) == "1";
    }();
    plan.metrics.legacyOracleRequired = requireLegacyOracleSuperset;
    const bool allowLegacyFallback = []() {
      const char *env = std::getenv("LOOM_TECHMAP_ENABLE_LEGACY_FALLBACK");
      return env && std::string(env) == "1";
    }();

    auto registerFamily = [&](VariantFamily &&family, IdIndex hwNodeId) -> unsigned {
      auto found = familyIndexBySignature.find(family.signature);
      if (found != familyIndexBySignature.end()) {
        auto &existing = familyList[found->second];
        existing.hwNodeIds.push_back(hwNodeId);
        return found->second;
      }
      family.hwNodeIds.clear();
      family.hwNodeIds.push_back(hwNodeId);
      unsigned index = familyList.size();
      familyIndexBySignature[family.signature] = index;
      familyList.push_back(std::move(family));
      return index;
    };

    auto absorbMatch = [&](unsigned familyIndex, const Match &match,
                           IdIndex hwNodeId, bool fromLegacyFallback) {
      std::string key = buildAggregatedMatchKey(familyIndex, match);
      auto found = aggregatedIndexByKey.find(key);
      if (found == aggregatedIndexByKey.end()) {
        AggregatedMatch aggregated;
        aggregated.familyIndex = familyIndex;
        aggregated.swNodesByOp = match.swNodesByOp;
        aggregated.operandOrderByOp = match.operandOrderByOp;
        aggregated.inputBindings = match.inputBindings;
        aggregated.outputBindings = match.outputBindings;
        aggregated.internalEdges = match.internalEdges;
        aggregated.configFields = match.configFields;
        aggregated.configurable = familyList[familyIndex].configurable;
        aggregated.commutativeSwapCount =
            countCommutativeSwaps(match.operandOrderByOp);
        aggregated.hasDemandOrigin = !fromLegacyFallback;
        aggregated.hasLegacyOrigin = fromLegacyFallback;
        aggregated.hwNodeIds.push_back(hwNodeId);
        aggregatedMatches.push_back(std::move(aggregated));
        aggregatedIndexByKey[key] = aggregatedMatches.size() - 1;
        return;
      }

      auto &aggregated = aggregatedMatches[found->second];
      aggregated.hasDemandOrigin =
          aggregated.hasDemandOrigin || !fromLegacyFallback;
      aggregated.hasLegacyOrigin =
          aggregated.hasLegacyOrigin || fromLegacyFallback;
      if (std::find(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end(),
                    hwNodeId) == aggregated.hwNodeIds.end()) {
        aggregated.hwNodeIds.push_back(hwNodeId);
      }
    };

    auto recordMatchesForPE = [&](llvm::StringRef peName,
                                  loom::fabric::FunctionUnitOp fuOp,
                                  const std::string &fuName) {
      IdIndex hwNodeId = findFunctionUnitNode(adg, peName, fuName);
      if (hwNodeId == INVALID_ID)
        return;
      const Node *hwNode = adg.getNode(hwNodeId);
      if (!hwNode)
        return;

      techmapper_detail::DemandMatchStats demandStats;
      auto demandMatches =
          findDemandDrivenMatchesForFU(dfg, fuOp, hwNode, &demandStats);
      llvm::StringSet<> demandKeysForPE;
      llvm::StringSet<> demandFamilySignaturesForPE;
      llvm::StringSet<> legacyFamilySignaturesForPE;
      plan.metrics.demandCandidateCount += demandMatches.size();
      plan.metrics.structuralStateCount += demandStats.structuralStateCount;
      plan.metrics.structuralStateCacheHitCount +=
          demandStats.structuralStateCacheHitCount;
      plan.metrics.structuralStateCacheMissCount +=
          demandStats.structuralStateCacheMissCount;
      for (auto &familyMatch : demandMatches) {
        if (demandFamilySignaturesForPE.insert(familyMatch.family.signature)
                .second) {
          ++demandMaterializedStateCountBySignature[familyMatch.family.signature];
        }
        unsigned familyIndex =
            registerFamily(std::move(familyMatch.family), hwNodeId);
        familyMatch.match.familyIndex = familyIndex;
        demandKeysForPE.insert(
            buildAggregatedMatchKey(familyIndex, familyMatch.match));
        absorbMatch(familyIndex, familyMatch.match, hwNodeId, false);
      }

      bool usedLegacyFallbackForPE = false;
      if (allowLegacyFallback) {
        llvm::SmallVector<VariantFamily, 8> legacyVariants;
        collectVariantsForFU(fuOp, hwNode, legacyVariants);
        for (auto &variant : legacyVariants) {
          if (!variant.isTechFamily())
            continue;
          if (legacyFamilySignaturesForPE.insert(variant.signature).second) {
            ++legacyMaterializedStateCountBySignature[variant.signature];
          }
          unsigned familyIndex = registerFamily(std::move(variant), hwNodeId);
          auto legacyMatches =
              findMatchesForFamily(dfg, familyList[familyIndex], familyIndex);
          for (const auto &legacyMatch : legacyMatches) {
            std::string legacyKey =
                buildAggregatedMatchKey(familyIndex, legacyMatch);
            if (demandKeysForPE.contains(legacyKey))
              continue;
            absorbMatch(familyIndex, legacyMatch, hwNodeId, true);
            usedLegacyFallbackForPE = true;
          }
        }
        if (usedLegacyFallbackForPE)
          ++plan.metrics.legacyFallbackCount;
      }

      if (!runLegacyOracle)
        return;
      ++plan.metrics.legacyOracleCheckCount;

      llvm::SmallVector<VariantFamily, 8> variants;
      collectVariantsForFU(fuOp, hwNode, variants);
      for (auto &variant : variants) {
        if (!variant.isTechFamily())
          continue;
        if (legacyFamilySignaturesForPE.insert(variant.signature).second) {
          ++legacyMaterializedStateCountBySignature[variant.signature];
        }
        unsigned familyIndex = registerFamily(std::move(variant), hwNodeId);
        auto legacyMatches =
            findMatchesForFamily(dfg, familyList[familyIndex], familyIndex);
        plan.metrics.legacyOracleCandidateCount += legacyMatches.size();
        for (const auto &legacyMatch : legacyMatches) {
          std::string legacyKey = buildAggregatedMatchKey(familyIndex, legacyMatch);
          if (!demandKeysForPE.contains(legacyKey)) {
            ++plan.metrics.legacyOracleMissingCount;
            if (legacyOracleMissingSampleBuffer.size() < 4) {
              TechMapper::LegacyOracleSampleInfo sample;
              sample.key = legacyKey;
              sample.familyIndex = familyIndex;
              if (familyIndex < familyList.size())
                sample.familySignature = familyList[familyIndex].signature;
              sample.hwNodeId = hwNodeId;
              sample.peName = peName.str();
              sample.hwName = fuName;
              legacyOracleMissingSampleBuffer.push_back(std::move(sample));
            }
          }
        }
      }
    };

    for (mlir::Operation &op : fabricMod.getBody().front().getOperations()) {
      if (auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(op)) {
        std::string peName =
            instOp.getSymName().value_or(instOp.getModule()).str();
        auto peIt = peDefs.find(instOp.getModule());
        if (peIt != peDefs.end()) {
          visitPEFunctionUnits(
              peIt->second, peName,
              [&](loom::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordMatchesForPE(peName, fuOp, fuName);
              });
          continue;
        }
        auto temporalPeIt = temporalPeDefs.find(instOp.getModule());
        if (temporalPeIt != temporalPeDefs.end()) {
          visitPEFunctionUnits(
              temporalPeIt->second, peName,
              [&](loom::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordMatchesForPE(peName, fuOp, fuName);
              });
        }
        continue;
      }

      if (auto peOp = mlir::dyn_cast<loom::fabric::SpatialPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](loom::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordMatchesForPE(peName, fuOp, fuName);
            });
        continue;
      }

      if (auto peOp = mlir::dyn_cast<loom::fabric::TemporalPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](loom::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordMatchesForPE(peName, fuOp, fuName);
            });
      }
    }

    for (auto &family : familyList) {
      std::sort(family.hwNodeIds.begin(), family.hwNodeIds.end());
      family.hwNodeIds.erase(
          std::unique(family.hwNodeIds.begin(), family.hwNodeIds.end()),
          family.hwNodeIds.end());
    }

    std::map<std::string, unsigned> supportClassIds;
    std::map<std::string, unsigned> configClassIds;
    for (auto &aggregated : aggregatedMatches) {
      std::sort(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end());
      aggregated.hwNodeIds.erase(
          std::unique(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end()),
          aggregated.hwNodeIds.end());
      aggregated.temporal = isTemporalCandidateList(aggregated.hwNodeIds, adg);

      std::string supportKey = aggregated.temporal ? "temporal|" : "spatial|";
      for (IdIndex hwNodeId : aggregated.hwNodeIds) {
        if (supportKey.back() != '|')
          supportKey += ",";
        supportKey += std::to_string(hwNodeId);
      }
      auto supportInserted =
          supportClassIds.emplace(supportKey, supportClassIds.size());
      aggregated.supportClassId = supportInserted.first->second;

      std::string configKey =
          familyList[aggregated.familyIndex].signature + "|" +
          serializeConfigFields(aggregated.configFields);
      auto configInserted =
          configClassIds.emplace(configKey, configClassIds.size());
      aggregated.configClassId = configInserted.first->second;
      aggregated.selectionScore = scoreAggregatedMatch(
          aggregated, familyList[aggregated.familyIndex].hwNodeIds.size());
    }
    plan.metrics.supportClassCount = supportClassIds.size();
    plan.metrics.configClassCount = configClassIds.size();
    auto &supportClasses = TechMapper::allSupportClasses(plan);
    supportClasses.resize(supportClassIds.size());
    for (const auto &entry : supportClassIds) {
      if (entry.second >= supportClasses.size())
        continue;
      auto &info = supportClasses[entry.second];
      info.id = entry.second;
      info.key = entry.first;
    }
    for (const auto &aggregated : aggregatedMatches) {
      if (aggregated.supportClassId >= supportClasses.size())
        continue;
      auto &info = supportClasses[aggregated.supportClassId];
      info.temporal = aggregated.temporal;
      info.kind = aggregated.temporal ? "temporal" : "spatial";
      info.hwNodeIds.assign(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end());
      info.peNames.clear();
      for (IdIndex hwNodeId : info.hwNodeIds) {
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!hwNode)
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (peName.empty())
          continue;
        if (std::find(info.peNames.begin(), info.peNames.end(), peName.str()) ==
            info.peNames.end()) {
          info.peNames.push_back(peName.str());
        }
      }
      std::sort(info.peNames.begin(), info.peNames.end());
      info.capacity = info.hwNodeIds.size();
      info.enforceHardCapacity = !aggregated.temporal;
    }
    for (auto &aggregated : aggregatedMatches) {
      if (aggregated.supportClassId >= supportClasses.size())
        continue;
      aggregated.supportClassCapacity =
          supportClasses[aggregated.supportClassId].capacity;
    }
    auto &configClasses = TechMapper::allConfigClasses(plan);
    configClasses.resize(configClassIds.size());
    for (const auto &entry : configClassIds) {
      if (entry.second >= configClasses.size())
        continue;
      auto &info = configClasses[entry.second];
      info.id = entry.second;
      info.key = entry.first;
      info.reason = "same FU signature and exact config-field binding: " + entry.first;
      info.temporal = false;
      info.compatibleConfigClassIds.clear();
      info.compatibleConfigClassIds.push_back(entry.second);
    }
    for (const auto &aggregated : aggregatedMatches) {
      if (!aggregated.temporal || aggregated.configClassId >= configClasses.size())
        continue;
      configClasses[aggregated.configClassId].temporal = true;
    }
    for (size_t lhs = 0; lhs < configClasses.size(); ++lhs) {
      auto &lhsInfo = configClasses[lhs];
      lhsInfo.compatibleConfigClassIds.clear();
      lhsInfo.compatibleConfigClassIds.push_back(lhs);
      if (lhsInfo.temporal)
        continue;
      for (size_t rhs = 0; rhs < configClasses.size(); ++rhs) {
        if (rhs == lhs)
          continue;
        if (configClasses[rhs].temporal)
          continue;
        lhsInfo.compatibleConfigClassIds.push_back(rhs);
      }
      std::sort(lhsInfo.compatibleConfigClassIds.begin(),
                lhsInfo.compatibleConfigClassIds.end());
    }
    for (size_t lhs = 0; lhs < configClasses.size(); ++lhs) {
      if (!configClasses[lhs].temporal)
        continue;
      for (size_t rhs = lhs + 1; rhs < configClasses.size(); ++rhs) {
        if (!configClasses[rhs].temporal)
          continue;
        TechMapper::TemporalIncompatibilityInfo info;
        info.lhsConfigClassId = lhs;
        info.rhsConfigClassId = rhs;
        info.reason =
            "temporal reuse currently requires identical config classes; " +
            std::to_string(lhs) + " and " + std::to_string(rhs) +
            " differ";
        TechMapper::temporalIncompatibilities(plan).push_back(std::move(info));
      }
    }

    llvm::DenseSet<IdIndex> overlapNodes;
    for (const auto &aggregated : aggregatedMatches) {
      for (IdIndex swNodeId : aggregated.swNodesByOp) {
        if (!overlapNodes.insert(swNodeId).second)
          ++plan.metrics.overlapEdgeCount;
      }
    }
    plan.metrics.candidateGenerationTimeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - candidateGenerationStartTime)
            .count();
    candidateGenerationTimed = true;

    std::vector<unsigned> selectedMatches;
    std::vector<unsigned> matchComponentIds;
    const auto selectionStartTime = std::chrono::steady_clock::now();
    selectMatchesByComponent(aggregatedMatches, adg,
                             &TechMapper::allSelectionComponents(plan),
                             &matchComponentIds,
                             &plan.metrics,
                             selectedMatches);
    plan.metrics.selectionTimeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - selectionStartTime)
            .count();
    selectionTimed = true;
    llvm::DenseSet<unsigned> selectedMatchSet;
    for (unsigned selectedIdx : selectedMatches)
      selectedMatchSet.insert(selectedIdx);
    auto &familyTechInfos = TechMapper::allFamilyTechInfos(plan);
    familyTechInfos.clear();
    familyTechInfos.resize(familyList.size());
    for (unsigned familyIndex = 0; familyIndex < familyList.size();
         ++familyIndex) {
      auto &info = familyTechInfos[familyIndex];
      info.familyIndex = familyIndex;
      info.signature = familyList[familyIndex].signature;
      info.hwSupportCount = familyList[familyIndex].hwNodeIds.size();
      info.materializedStateCount =
          demandMaterializedStateCountBySignature[info.signature];
      info.legacyMaterializedStateCount =
          legacyMaterializedStateCountBySignature[info.signature];
      info.opCount = familyList[familyIndex].ops.size();
      info.configurable = familyList[familyIndex].configurable;
    }
    for (const auto &aggregated : aggregatedMatches) {
      auto *info = TechMapper::findFamilyTechInfo(plan, aggregated.familyIndex);
      if (!info)
        continue;
      ++info->matchCount;
      info->maxFusionSize =
          std::max<unsigned>(info->maxFusionSize, aggregated.swNodesByOp.size());
    }
    auto &candidateSummaries = TechMapper::allCandidateSummaries(plan);
    candidateSummaries.clear();
    candidateSummaries.reserve(aggregatedMatches.size());
    llvm::DenseSet<IdIndex> legacyDerivedHwNodeIds;
    for (unsigned matchIdx = 0; matchIdx < aggregatedMatches.size(); ++matchIdx) {
      const auto &aggregated = aggregatedMatches[matchIdx];
      TechMapper::CandidateSummaryInfo summary;
      summary.id = matchIdx;
      summary.familyIndex = aggregated.familyIndex;
      if (aggregated.familyIndex < familyList.size())
        summary.familySignature = familyList[aggregated.familyIndex].signature;
      if (matchIdx < matchComponentIds.size())
        summary.selectionComponentId = matchComponentIds[matchIdx];
      summary.swNodeIds.assign(aggregated.swNodesByOp.begin(),
                               aggregated.swNodesByOp.end());
      summary.internalEdgeIds.assign(aggregated.internalEdges.begin(),
                                     aggregated.internalEdges.end());
      summary.inputBindings.assign(aggregated.inputBindings.begin(),
                                   aggregated.inputBindings.end());
      summary.outputBindings.assign(aggregated.outputBindings.begin(),
                                    aggregated.outputBindings.end());
      summary.hwNodeIds.assign(aggregated.hwNodeIds.begin(),
                               aggregated.hwNodeIds.end());
      for (IdIndex hwNodeId : summary.hwNodeIds) {
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!hwNode)
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (peName.empty())
          continue;
        if (std::find(summary.peNames.begin(), summary.peNames.end(),
                      peName.str()) == summary.peNames.end()) {
          summary.peNames.push_back(peName.str());
        }
      }
      std::sort(summary.peNames.begin(), summary.peNames.end());
      summary.supportClassId = aggregated.supportClassId;
      summary.supportClassCapacity = aggregated.supportClassCapacity;
      if (const auto *supportClassInfo =
              TechMapper::findSupportClass(plan, aggregated.supportClassId)) {
        summary.supportClassKey = supportClassInfo->key;
      }
      summary.configClassId = aggregated.configClassId;
      if (const auto *configClassInfo =
              TechMapper::findConfigClass(plan, aggregated.configClassId)) {
        summary.configClassKey = configClassInfo->key;
        summary.configClassReason = configClassInfo->reason;
      }
      summary.temporal = aggregated.temporal;
      summary.configurable = aggregated.configurable;
      summary.baseSelectionScore = aggregated.selectionScore;
      summary.candidatePenalty = 0;
      summary.familyPenalty = 0;
      summary.configClassPenalty = 0;
      summary.selectionScore = aggregated.selectionScore;
      applyCandidateSelectionOutcome(
          plan, summary, selectedMatchSet.contains(matchIdx),
          inferCandidateStatus(matchIdx, aggregatedMatches, selectedMatches,
                               matchComponentIds, adg));
      summary.demandOrigin = aggregated.hasDemandOrigin;
      summary.legacyFallbackOrigin =
          aggregated.hasLegacyOrigin && !aggregated.hasDemandOrigin;
      summary.mixedOrigin =
          aggregated.hasDemandOrigin && aggregated.hasLegacyOrigin;
      accumulateLegacyDerivedCandidateMetrics(
          plan, summary.legacyFallbackOrigin, summary.mixedOrigin,
          aggregated.hwNodeIds, legacyDerivedHwNodeIds);
      summary.configFields.assign(aggregated.configFields.begin(),
                                  aggregated.configFields.end());
      candidateSummaries.push_back(std::move(summary));
    }
    plan.metrics.legacyDerivedSourceCount = legacyDerivedHwNodeIds.size();

    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> nodeInfoBySwNode;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        nodeSupportClassesBySwNode;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        nodeConfigClassesBySwNode;
    for (unsigned matchIdx = 0; matchIdx < aggregatedMatches.size(); ++matchIdx) {
      const auto &aggregated = aggregatedMatches[matchIdx];
      accumulateNodeTechCandidateCoverage(
          nodeInfoBySwNode, nodeSupportClassesBySwNode,
          nodeConfigClassesBySwNode, aggregated, matchIdx,
          matchIdx < matchComponentIds.size()
              ? std::optional<unsigned>(matchComponentIds[matchIdx])
              : std::nullopt);
    }

    for (const auto &aggregated : aggregatedMatches) {
      if (aggregated.swNodesByOp.size() != 1)
        continue;
      IdIndex swNodeId = aggregated.swNodesByOp.front();
      accumulateConservativeFallbackCandidate(plan, swNodeId, aggregated);
    }
    rebuildPreferredConservativeFallbackCandidates(plan);
    finalizeNodeTechCoverageSummaries(nodeInfoBySwNode,
                                      nodeSupportClassesBySwNode,
                                      nodeConfigClassesBySwNode);

    std::vector<int> nodeToUnit(dfg.nodes.size(), -1);
    unsigned techNodeCount = 0;
    llvm::DenseSet<unsigned> selectedConfigClasses;
    for (unsigned selectedIdx : selectedMatches) {
      const auto &aggregated = aggregatedMatches[selectedIdx];
      TechMapper::Unit unit = buildSelectedUnitFromAggregatedMatch(
          aggregated, selectedIdx,
          selectedIdx < matchComponentIds.size()
              ? std::optional<unsigned>(matchComponentIds[selectedIdx])
              : std::nullopt);
      int unitIndex = static_cast<int>(TechMapper::allUnits(plan).size());
      for (IdIndex swNodeId : unit.swNodes) {
        if (swNodeId < nodeToUnit.size()) {
          nodeToUnit[swNodeId] = unitIndex;
          ++techNodeCount;
        }
      }
      registerSelectedUnit(plan, aggregated, unit, unitIndex, selectedIdx,
                           selectedConfigClasses, nodeInfoBySwNode);
      TechMapper::allUnits(plan).push_back(std::move(unit));
    }
    plan.metrics.selectedConfigDiversityCount = selectedConfigClasses.size();
    plan.metrics.selectedCandidateCount = selectedMatches.size();
    sortSelectedUnitIndices(plan);
    if (totalOpCount > techNodeCount)
      plan.metrics.conservativeFallbackCount = totalOpCount - techNodeCount;

    llvm::DenseSet<IdIndex> candidateCoveredNodes;
    llvm::DenseSet<IdIndex> selectedCoveredNodes;
    collectCoveredNodes(aggregatedMatches, selectedMatches,
                        candidateCoveredNodes, selectedCoveredNodes);
    for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
         ++swNodeId) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        continue;
      if (selectedCoveredNodes.contains(swNodeId))
        continue;
      TechMapper::FallbackNodeInfo fallbackInfo;
      populateFallbackNodeSummary(plan, swNodeId, nodeInfoBySwNode,
                                  fallbackInfo);
      if (candidateCoveredNodes.contains(swNodeId)) {
        ++plan.metrics.fallbackRejectedCount;
        fallbackInfo.reason =
            inferRejectedReason(swNodeId, aggregatedMatches, selectedMatches,
                                matchComponentIds, adg);
      } else {
        ++plan.metrics.fallbackNoCandidateCount;
        fallbackInfo.reason = "no_candidate";
      }
      TechMapper::allFallbackNodes(plan).push_back(std::move(fallbackInfo));
      auto infoIt = nodeInfoBySwNode.find(swNodeId);
      if (infoIt != nodeInfoBySwNode.end()) {
        markNodeAsConservativeFallback(
            infoIt->second,
            candidateCoveredNodes.contains(swNodeId)
                ? inferRejectedReason(swNodeId, aggregatedMatches,
                                      selectedMatches, matchComponentIds, adg)
                : llvm::StringRef("fallback_no_candidate"));
      }
    }
    auto &nodeTechInfos = TechMapper::allNodeTechInfos(plan);
    for (auto &entry : nodeInfoBySwNode)
      nodeTechInfos.push_back(entry.second);
    std::sort(nodeTechInfos.begin(), nodeTechInfos.end(),
              [](const TechMapper::NodeTechInfo &lhs,
                 const TechMapper::NodeTechInfo &rhs) {
                return lhs.swNodeId < rhs.swNodeId;
              });

    if (totalOpCount > 0)
      plan.coverageScore = static_cast<double>(techNodeCount) /
                           static_cast<double>(totalOpCount);
    plan.metrics.coverageScore = plan.coverageScore;

    if (runLegacyOracle && plan.metrics.legacyOracleMissingCount != 0) {
      auto &oracleMissingSamples = TechMapper::legacyOracleMissingSamples(plan);
      oracleMissingSamples.assign(legacyOracleMissingSampleBuffer.begin(),
                                  legacyOracleMissingSampleBuffer.end());
      plan.diagnostics =
          "techmap legacy oracle found missing demand-driven candidates: " +
          std::to_string(plan.metrics.legacyOracleMissingCount);
      if (!oracleMissingSamples.empty()) {
        plan.diagnostics += " samples=";
        for (size_t idx = 0; idx < oracleMissingSamples.size(); ++idx) {
          if (idx != 0)
            plan.diagnostics += " || ";
          plan.diagnostics += oracleMissingSamples[idx].key;
        }
      }
      if (requireLegacyOracleSuperset)
        return false;
    }
    builtTechUnits = true;
  }
  if (!builtTechUnits) {
    unsigned totalOpCount = 0;
    for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
         ++swNodeId) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        continue;
      ++totalOpCount;
      TechMapper::NodeTechInfo info;
      info.swNodeId = swNodeId;
      info.candidateCount = 0;
      info.supportClassCount = 0;
      info.configClassCount = 0;
      info.maxFusionSize = 0;
      info.selected = false;
      info.selectedAsFusion = false;
      info.conservativeFallback = true;
      info.status = "fallback_no_candidate";
      TechMapper::allNodeTechInfos(plan).push_back(std::move(info));

      TechMapper::FallbackNodeInfo fallbackInfo;
      fallbackInfo.swNodeId = swNodeId;
      fallbackInfo.reason = "no_candidate";
      TechMapper::allFallbackNodes(plan).push_back(std::move(fallbackInfo));
      ++plan.metrics.fallbackNoCandidateCount;
    }
    plan.metrics.conservativeFallbackCount = totalOpCount;
    if (totalOpCount > 0)
      plan.coverageScore = 0.0;
    auto &nodeTechInfos = TechMapper::allNodeTechInfos(plan);
    std::sort(nodeTechInfos.begin(), nodeTechInfos.end(),
              [](const TechMapper::NodeTechInfo &lhs,
                 const TechMapper::NodeTechInfo &rhs) {
                return lhs.swNodeId < rhs.swNodeId;
              });
  }
  plan.metrics.coverageScore = plan.coverageScore;
  if (!candidateGenerationTimed) {
    plan.metrics.candidateGenerationTimeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - candidateGenerationStartTime)
            .count();
  }
  if (!selectionTimed)
    plan.metrics.selectionTimeMicros = 0;
  plan.metrics.totalLayer2TimeMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - layer2StartTime)
          .count();
  if (plan.diagnostics.empty())
    plan.diagnostics = buildTechmapDiagnostics(plan);
  else
    plan.diagnostics += "; " + buildTechmapDiagnostics(plan);
  return finalizePlanGraphs(dfg, adg, plan);
}

bool TechMapper::applyFeedback(const Graph &dfg, const Graph &adg,
                               const Plan &seedPlan,
                               const Feedback &feedback, Plan &plan) {
  const auto reselectionStartTime = std::chrono::steady_clock::now();
  plan = seedPlan;
  plan.contractedDFG = Graph(dfg.context);
  plan.conservativeFallbackDFG = dfg.clone();
  plan.contractedCandidates.clear();
  plan.contractedCandidateSupportClasses.clear();
  plan.contractedCandidateConfigClasses.clear();
  plan.conservativeFallbackCandidates.clear();
  plan.conservativeFallbackCandidateSupportClasses.clear();
  plan.conservativeFallbackCandidateConfigClasses.clear();
  plan.conservativeFallbackCandidateDetails.clear();
  plan.conservativeFallbackPreferredCandidate.clear();
  TechMapper::allFallbackNodes(plan).clear();
  TechMapper::allNodeTechInfos(plan).clear();
  TechMapper::allSelectionComponents(plan).clear();
  TechMapper::temporalIncompatibilities(plan).assign(
      TechMapper::temporalIncompatibilities(seedPlan).begin(),
      TechMapper::temporalIncompatibilities(seedPlan).end());
  TechMapper::conservativeFallbackSwNodes(plan).clear();
  TechMapper::allUnits(plan).clear();
  plan.originalNodeToContractedNode.assign(dfg.nodes.size(), INVALID_ID);
  plan.originalPortToContractedPort.assign(dfg.ports.size(), INVALID_ID);
  plan.originalEdgeToContractedEdge.assign(dfg.edges.size(), INVALID_ID);
  plan.originalEdgeKinds.assign(dfg.edges.size(), TechMappedEdgeKind::Routed);
  TechMapper::markFeedbackApplied(plan, feedback);
  auto &feedbackResolution = TechMapper::feedbackResolution(plan);
  resetSelectionMetrics(plan.metrics);
  restoreReselectionBaselineMetrics(plan.metrics, seedPlan.metrics);
  plan.metrics.feedbackReselectionCount =
      seedPlan.metrics.feedbackReselectionCount + 1;
  plan.metrics.feedbackPenaltyCount = 0;
  for (auto &familyInfo : TechMapper::allFamilyTechInfos(plan))
    familyInfo.selectedCount = 0;
  for (auto &summary : TechMapper::allCandidateSummaries(plan)) {
    summary.selected = false;
    summary.selectionComponentId = std::numeric_limits<unsigned>::max();
    summary.selectedUnitIndex = std::numeric_limits<unsigned>::max();
    summary.contractedNodeId = INVALID_ID;
    summary.status.clear();
    summary.candidatePenalty = 0;
    summary.familyPenalty = 0;
    summary.configClassPenalty = 0;
    summary.selectionScore = summary.baseSelectionScore;
  }

  llvm::DenseSet<unsigned> bannedCandidateIds;
  llvm::DenseSet<unsigned> bannedFamilyIds;
  llvm::DenseSet<unsigned> bannedConfigClassIds;
  llvm::DenseSet<unsigned> splitCandidateIds;
  llvm::SmallVector<TechMapper::WeightedIdPenalty, 8> validCandidatePenalties;
  llvm::SmallVector<TechMapper::WeightedIdPenalty, 8> validFamilyPenalties;
  llvm::SmallVector<TechMapper::WeightedIdPenalty, 8>
      validConfigClassPenalties;
  llvm::ArrayRef<TechMapper::CandidateSummaryInfo> seedCandidateSummaries =
      TechMapper::allCandidateSummaries(seedPlan);
  llvm::ArrayRef<TechMapper::FamilyTechInfo> seedFamilyTechInfos =
      TechMapper::allFamilyTechInfos(seedPlan);
  llvm::ArrayRef<TechMapper::ConfigClassInfo> seedConfigClasses =
      TechMapper::allConfigClasses(seedPlan);
  for (unsigned id : feedback.bannedCandidateIds) {
    if (id < seedCandidateSummaries.size()) {
      bannedCandidateIds.insert(id);
      continue;
    }
    feedbackResolution.unknownBannedCandidateIds.push_back(id);
    ++plan.metrics.feedbackUnknownCandidateRefCount;
  }
  for (unsigned id : feedback.bannedFamilyIds) {
    if (id < seedFamilyTechInfos.size()) {
      bannedFamilyIds.insert(id);
      continue;
    }
    feedbackResolution.unknownBannedFamilyIds.push_back(id);
    ++plan.metrics.feedbackUnknownFamilyRefCount;
  }
  for (unsigned id : feedback.bannedConfigClassIds) {
    if (id < seedConfigClasses.size()) {
      bannedConfigClassIds.insert(id);
      continue;
    }
    feedbackResolution.unknownBannedConfigClassIds.push_back(id);
    ++plan.metrics.feedbackUnknownConfigClassRefCount;
  }
  for (unsigned id : feedback.splitCandidateIds) {
    if (id < seedCandidateSummaries.size()) {
      splitCandidateIds.insert(id);
      continue;
    }
    feedbackResolution.unknownSplitCandidateIds.push_back(id);
    ++plan.metrics.feedbackUnknownCandidateRefCount;
  }
  for (const auto &penalty : feedback.candidatePenalties) {
    if (penalty.id < seedCandidateSummaries.size()) {
      validCandidatePenalties.push_back(penalty);
      continue;
    }
    feedbackResolution.unknownCandidatePenalties.push_back(penalty);
    ++plan.metrics.feedbackUnknownCandidateRefCount;
  }
  for (const auto &penalty : feedback.familyPenalties) {
    if (penalty.id < seedFamilyTechInfos.size()) {
      validFamilyPenalties.push_back(penalty);
      continue;
    }
    feedbackResolution.unknownFamilyPenalties.push_back(penalty);
    ++plan.metrics.feedbackUnknownFamilyRefCount;
  }
  for (const auto &penalty : feedback.configClassPenalties) {
    if (penalty.id < seedConfigClasses.size()) {
      validConfigClassPenalties.push_back(penalty);
      continue;
    }
    feedbackResolution.unknownConfigClassPenalties.push_back(penalty);
    ++plan.metrics.feedbackUnknownConfigClassRefCount;
  }
  plan.metrics.feedbackPenaltyCount = validCandidatePenalties.size() +
                                      validFamilyPenalties.size() +
                                      validConfigClassPenalties.size();

  std::vector<AggregatedMatch> filteredMatches;
  std::vector<unsigned> filteredCandidateIds;
  std::vector<unsigned> cachedComponentIds;
  llvm::DenseSet<IdIndex> legacyFallbackHwNodeIds;
  llvm::DenseSet<IdIndex> legacyDerivedHwNodeIds;
  filteredMatches.reserve(seedCandidateSummaries.size());
  filteredCandidateIds.reserve(seedCandidateSummaries.size());
  cachedComponentIds.reserve(seedCandidateSummaries.size());

  for (const auto &seedSummary : seedCandidateSummaries) {
    auto *summary = TechMapper::findCandidateSummary(plan, seedSummary.id);
    if (!summary)
      continue;
    if (bannedCandidateIds.contains(seedSummary.id)) {
      markFeedbackFilteredCandidate(plan, *summary, "feedback_banned_candidate");
      continue;
    }
    if (splitCandidateIds.contains(seedSummary.id)) {
      markFeedbackFilteredCandidate(plan, *summary, "feedback_split_requested");
      continue;
    }
    if (bannedFamilyIds.contains(seedSummary.familyIndex)) {
      markFeedbackFilteredCandidate(plan, *summary, "feedback_banned_family");
      continue;
    }
    if (bannedConfigClassIds.contains(seedSummary.configClassId)) {
      markFeedbackFilteredCandidate(plan, *summary,
                                    "feedback_banned_config_class");
      continue;
    }

    AggregatedMatch match = buildAggregatedMatchFromSummary(seedSummary);
    int64_t candidatePenalty =
        lookupPenalty(validCandidatePenalties, seedSummary.id);
    int64_t familyPenalty =
        lookupPenalty(validFamilyPenalties, seedSummary.familyIndex);
    int64_t configClassPenalty =
        lookupPenalty(validConfigClassPenalties, seedSummary.configClassId);
    match.selectionScore -= candidatePenalty;
    match.selectionScore -= familyPenalty;
    match.selectionScore -= configClassPenalty;
    summary->baseSelectionScore = seedSummary.baseSelectionScore;
    summary->candidatePenalty = candidatePenalty;
    summary->familyPenalty = familyPenalty;
    summary->configClassPenalty = configClassPenalty;
    summary->selectionScore = match.selectionScore;
    accumulateLegacyDerivedCandidateMetrics(
        plan, seedSummary.legacyFallbackOrigin, seedSummary.mixedOrigin,
        match.hwNodeIds, legacyDerivedHwNodeIds);
    if (seedSummary.legacyFallbackOrigin) {
      for (IdIndex hwNodeId : match.hwNodeIds)
        legacyFallbackHwNodeIds.insert(hwNodeId);
    }
    filteredMatches.push_back(std::move(match));
    filteredCandidateIds.push_back(seedSummary.id);
    cachedComponentIds.push_back(seedSummary.selectionComponentId);
  }
  plan.metrics.legacyFallbackCount = legacyFallbackHwNodeIds.size();
  plan.metrics.legacyDerivedSourceCount = legacyDerivedHwNodeIds.size();

  std::vector<unsigned> selectedFilteredMatches;
  std::vector<unsigned> filteredMatchComponentIds;
  const auto selectionStartTime = std::chrono::steady_clock::now();
  selectMatchesByCachedComponents(filteredMatches, filteredCandidateIds,
                                  cachedComponentIds,
                                  TechMapper::allSelectionComponents(seedPlan).size(),
                                  adg,
                                  &TechMapper::allSelectionComponents(plan),
                                  &filteredMatchComponentIds, &plan.metrics,
                                  selectedFilteredMatches);
  llvm::ArrayRef<TechMapper::SelectionComponentInfo> seedSelectionComponents =
      TechMapper::allSelectionComponents(seedPlan);
  if (!seedSelectionComponents.empty()) {
    for (auto &componentInfo : TechMapper::allSelectionComponents(plan)) {
      if (componentInfo.id >= seedSelectionComponents.size())
        continue;
      componentInfo.baseMaxCandidateScore =
          seedSelectionComponents[componentInfo.id].maxCandidateScore;
      componentInfo.baseSelectedScoreSum =
          seedSelectionComponents[componentInfo.id].selectedScoreSum;
      llvm::DenseSet<unsigned> retainedCandidateIds;
      for (unsigned candidateId : componentInfo.candidateIds)
        retainedCandidateIds.insert(candidateId);
      for (unsigned candidateId :
           seedSelectionComponents[componentInfo.id].candidateIds) {
        if (!retainedCandidateIds.contains(candidateId))
          componentInfo.filteredCandidateIds.push_back(candidateId);
      }
      std::sort(componentInfo.filteredCandidateIds.begin(),
                componentInfo.filteredCandidateIds.end());
    }
  }
  plan.metrics.selectionTimeMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - selectionStartTime)
          .count();

  llvm::DenseSet<unsigned> selectedFilteredSet;
  for (unsigned selectedIdx : selectedFilteredMatches)
    selectedFilteredSet.insert(selectedIdx);
  for (unsigned filteredIdx = 0; filteredIdx < filteredMatches.size();
       ++filteredIdx) {
    unsigned candidateId = filteredCandidateIds[filteredIdx];
    auto *summary = TechMapper::findCandidateSummary(plan, candidateId);
    if (!summary)
      continue;
    if (filteredIdx < filteredMatchComponentIds.size())
      summary->selectionComponentId = filteredMatchComponentIds[filteredIdx];
    applyCandidateSelectionOutcome(
        plan, *summary, selectedFilteredSet.contains(filteredIdx),
        selectedFilteredSet.contains(filteredIdx)
            ? llvm::StringRef("selected")
            : llvm::StringRef(inferCandidateStatus(
                  filteredIdx, filteredMatches, selectedFilteredMatches,
                  filteredMatchComponentIds, adg)));
  }

  llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> nodeInfoBySwNode;
  llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
      nodeSupportClassesBySwNode;
  llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
      nodeConfigClassesBySwNode;
  for (const auto &seedInfo : TechMapper::allNodeTechInfos(seedPlan)) {
    auto &info = nodeInfoBySwNode[seedInfo.swNodeId];
    info.swNodeId = seedInfo.swNodeId;
    info.contractedNodeId = INVALID_ID;
    info.selectionComponentId = std::numeric_limits<unsigned>::max();
    info.selectedUnitIndex = std::numeric_limits<unsigned>::max();
    info.selectedCandidateId = std::numeric_limits<unsigned>::max();
    info.candidateCount = 0;
    info.supportClassCount = 0;
    info.configClassCount = 0;
    info.maxFusionSize = 0;
    info.candidateIds.clear();
    info.supportClassIds.clear();
    info.configClassIds.clear();
    info.selected = false;
    info.selectedAsFusion = false;
    info.conservativeFallback = true;
    info.selectedFromLegacyFallback = false;
    info.selectedFromDemand = false;
    info.selectedFromMixedOrigin = false;
    info.status = "feedback_filtered_out";
  }

  auto hadSeedCandidates = [&](IdIndex swNodeId) {
    const auto *seedInfo = TechMapper::findNodeTechInfo(seedPlan, swNodeId);
    return seedInfo && seedInfo->candidateCount != 0;
  };

  for (unsigned filteredIdx = 0; filteredIdx < filteredMatches.size();
       ++filteredIdx) {
    const auto &aggregated = filteredMatches[filteredIdx];
    unsigned candidateId = filteredCandidateIds[filteredIdx];
    accumulateNodeTechCandidateCoverage(
        nodeInfoBySwNode, nodeSupportClassesBySwNode,
        nodeConfigClassesBySwNode, aggregated, candidateId,
        filteredIdx < filteredMatchComponentIds.size()
            ? std::optional<unsigned>(filteredMatchComponentIds[filteredIdx])
            : std::nullopt);
  }

  for (const auto &aggregated : filteredMatches) {
    if (aggregated.swNodesByOp.size() != 1)
      continue;
    IdIndex swNodeId = aggregated.swNodesByOp.front();
    accumulateConservativeFallbackCandidate(plan, swNodeId, aggregated);
  }
  rebuildPreferredConservativeFallbackCandidates(plan);
  finalizeNodeTechCoverageSummaries(nodeInfoBySwNode,
                                    nodeSupportClassesBySwNode,
                                    nodeConfigClassesBySwNode);

  unsigned techNodeCount = 0;
  llvm::DenseSet<unsigned> selectedConfigClasses;
  for (unsigned selectedFilteredIdx : selectedFilteredMatches) {
    const auto &aggregated = filteredMatches[selectedFilteredIdx];
    unsigned selectedCandidateId = filteredCandidateIds[selectedFilteredIdx];
    TechMapper::Unit unit = buildSelectedUnitFromAggregatedMatch(
        aggregated, selectedCandidateId,
        selectedFilteredIdx < filteredMatchComponentIds.size()
            ? std::optional<unsigned>(filteredMatchComponentIds[selectedFilteredIdx])
            : std::nullopt);
    int unitIndex = static_cast<int>(TechMapper::allUnits(plan).size());
    techNodeCount += unit.swNodes.size();
    registerSelectedUnit(plan, aggregated, unit, unitIndex,
                         selectedCandidateId, selectedConfigClasses,
                         nodeInfoBySwNode);
    TechMapper::allUnits(plan).push_back(std::move(unit));
  }

  plan.metrics.selectedConfigDiversityCount = selectedConfigClasses.size();
  plan.metrics.selectedCandidateCount = selectedFilteredMatches.size();
  sortSelectedUnitIndices(plan);

  llvm::DenseSet<IdIndex> candidateCoveredNodes;
  llvm::DenseSet<IdIndex> selectedCoveredNodes;
  collectCoveredNodes(filteredMatches, selectedFilteredMatches,
                      candidateCoveredNodes, selectedCoveredNodes);

  unsigned totalOpCount = 0;
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    ++totalOpCount;
    if (selectedCoveredNodes.contains(swNodeId))
      continue;
    TechMapper::FallbackNodeInfo fallbackInfo;
    populateFallbackNodeSummary(plan, swNodeId, nodeInfoBySwNode,
                                fallbackInfo);
    if (candidateCoveredNodes.contains(swNodeId)) {
      ++plan.metrics.fallbackRejectedCount;
      fallbackInfo.reason =
          inferRejectedReason(swNodeId, filteredMatches, selectedFilteredMatches,
                              filteredMatchComponentIds, adg);
    } else {
      ++plan.metrics.fallbackNoCandidateCount;
      fallbackInfo.reason =
          hadSeedCandidates(swNodeId) ? "feedback_filtered_out" : "no_candidate";
    }
    TechMapper::allFallbackNodes(plan).push_back(std::move(fallbackInfo));
    auto infoIt = nodeInfoBySwNode.find(swNodeId);
    if (infoIt != nodeInfoBySwNode.end() && !infoIt->second.selected) {
      markNodeAsConservativeFallback(
          infoIt->second,
          candidateCoveredNodes.contains(swNodeId)
              ? inferRejectedReason(swNodeId, filteredMatches,
                                    selectedFilteredMatches,
                                    filteredMatchComponentIds, adg)
              : (hadSeedCandidates(swNodeId)
                     ? llvm::StringRef("feedback_filtered_out")
                     : llvm::StringRef("fallback_no_candidate")));
    }
  }
  if (totalOpCount > techNodeCount)
    plan.metrics.conservativeFallbackCount = totalOpCount - techNodeCount;

  for (auto &entry : nodeInfoBySwNode)
    TechMapper::allNodeTechInfos(plan).push_back(entry.second);
  auto &nodeTechInfos = TechMapper::allNodeTechInfos(plan);
  std::sort(nodeTechInfos.begin(), nodeTechInfos.end(),
            [](const TechMapper::NodeTechInfo &lhs,
               const TechMapper::NodeTechInfo &rhs) {
              return lhs.swNodeId < rhs.swNodeId;
            });

  plan.coverageScore =
      totalOpCount == 0 ? 1.0
                        : static_cast<double>(techNodeCount) /
                              static_cast<double>(totalOpCount);
  plan.metrics.coverageScore = plan.coverageScore;
  plan.metrics.totalLayer2TimeMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - reselectionStartTime)
          .count();
  std::string reselectionDiagnostics =
      "techmap cached reselection applied: filtered_candidates=" +
      std::to_string(plan.metrics.feedbackFilteredCandidateCount) +
      ", penalty_terms=" + std::to_string(plan.metrics.feedbackPenaltyCount);
  if (TechMapper::feedbackUnknownCandidateRefCount(plan) != 0 ||
      TechMapper::feedbackUnknownFamilyRefCount(plan) != 0 ||
      TechMapper::feedbackUnknownConfigClassRefCount(plan) != 0) {
    reselectionDiagnostics +=
        ", unresolved(candidate=" +
        std::to_string(TechMapper::feedbackUnknownCandidateRefCount(plan)) +
        ", family=" +
        std::to_string(TechMapper::feedbackUnknownFamilyRefCount(plan)) +
        ", config=" +
        std::to_string(TechMapper::feedbackUnknownConfigClassRefCount(plan)) +
        ")";
  }
  plan.diagnostics = reselectionDiagnostics;
  plan.diagnostics += "; " + buildTechmapDiagnostics(plan);
  return finalizePlanGraphs(dfg, adg, plan);
}



} // namespace loom
