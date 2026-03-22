#include "TechMapperInternal.h"
#include "loom/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <vector>

namespace loom {

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
  for (const auto &token : tokens)
    attrs.push_back(mlir::StringAttr::get(ctx, token));
  return attrs;
}

} // namespace

bool finalizePlanGraphs(const Graph &dfg, const Graph &adg,
                        TechMapper::Plan &plan) {
  auto &fallbackSwNodes = TechMapper::conservativeFallbackSwNodes(plan);
  fallbackSwNodes.clear();
  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(plan.conservativeFallbackDFG.nodes.size());
       ++swNodeId) {
    Node *fallbackNode = plan.conservativeFallbackDFG.getNode(swNodeId);
    if (!fallbackNode || fallbackNode->kind != Node::OperationNode)
      continue;
    addBoolNodeAttr(fallbackNode, "techmap_conservative_fallback_plan", true,
                    dfg.context);
    const auto *hwNodes =
        TechMapper::findConservativeFallbackCandidates(plan, swNodeId);
    const auto *supportClasses =
        TechMapper::findConservativeFallbackCandidateSupportClasses(plan,
                                                                   swNodeId);
    const auto *configClasses =
        TechMapper::findConservativeFallbackCandidateConfigClasses(plan,
                                                                  swNodeId);
    const auto *candidateDetails =
        TechMapper::findConservativeFallbackCandidateDetails(plan, swNodeId);
    const auto *preferredCandidate =
        TechMapper::findConservativeFallbackPreferredCandidate(plan, swNodeId);
    llvm::SmallVector<mlir::Attribute, 4> hwAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportKeyAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportKindAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportTemporalAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportHardCapacityAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportCapacityAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configKeyAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configReasonAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configTemporalAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configFieldSetAttrs;
    if (hwNodes) {
      for (IdIndex hwNodeId : *hwNodes) {
        hwAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), hwNodeId));
      }
    }
    if (supportClasses) {
      for (unsigned supportClassId : *supportClasses) {
        const auto *supportClassInfo =
            TechMapper::findSupportClass(plan, supportClassId);
        supportAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), supportClassId));
        supportKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            supportClassInfo ? llvm::StringRef(supportClassInfo->key)
                             : llvm::StringRef()));
        supportKindAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            supportClassInfo ? llvm::StringRef(supportClassInfo->kind)
                             : llvm::StringRef()));
        supportTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, supportClassInfo && supportClassInfo->temporal));
        supportHardCapacityAttrs.push_back(mlir::BoolAttr::get(
            dfg.context,
            TechMapper::supportClassEnforcesHardCapacity(plan, supportClassId)));
        supportCapacityAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            TechMapper::supportClassCapacity(plan, supportClassId)));
      }
    }
    if (configClasses) {
      for (unsigned configClassId : *configClasses) {
        const auto *configClassInfo =
            TechMapper::findConfigClass(plan, configClassId);
        configAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), configClassId));
        configKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            configClassInfo ? llvm::StringRef(configClassInfo->key)
                            : llvm::StringRef()));
        configReasonAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            configClassInfo ? llvm::StringRef(configClassInfo->reason)
                            : llvm::StringRef()));
        configTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, TechMapper::isTemporalConfigClass(plan, configClassId)));
      }
    }
    if (candidateDetails) {
      for (const auto &candidate : *candidateDetails) {
        configFieldSetAttrs.push_back(mlir::StringAttr::get(
            dfg.context, serializeConfigFields(candidate.configFields)));
      }
    }
    addNodeAttr(fallbackNode, "techmap_candidate_hw_nodes",
                mlir::ArrayAttr::get(dfg.context, hwAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_classes",
                mlir::ArrayAttr::get(dfg.context, supportAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_keys",
                mlir::ArrayAttr::get(dfg.context, supportKeyAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_kinds",
                mlir::ArrayAttr::get(dfg.context, supportKindAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_temporal",
                mlir::ArrayAttr::get(dfg.context, supportTemporalAttrs),
                dfg.context);
    addNodeAttr(
        fallbackNode, "techmap_candidate_support_class_enforce_hard_capacity",
        mlir::ArrayAttr::get(dfg.context, supportHardCapacityAttrs),
        dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_capacities",
                mlir::ArrayAttr::get(dfg.context, supportCapacityAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_classes",
                mlir::ArrayAttr::get(dfg.context, configAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_class_keys",
                mlir::ArrayAttr::get(dfg.context, configKeyAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_class_reasons",
                mlir::ArrayAttr::get(dfg.context, configReasonAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_class_temporal",
                mlir::ArrayAttr::get(dfg.context, configTemporalAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_field_sets",
                mlir::ArrayAttr::get(dfg.context, configFieldSetAttrs),
                dfg.context);
    addBoolNodeAttr(fallbackNode, "techmap_conservative_fallback_covered",
                    !hwAttrs.empty(), dfg.context);
    addUIntNodeAttr(fallbackNode, "techmap_conservative_fallback_candidate_count",
                    configFieldSetAttrs.size(), dfg.context);
    addUIntNodeAttr(
        fallbackNode, "techmap_conservative_fallback_support_class_count",
        supportAttrs.size(), dfg.context);
    addUIntNodeAttr(
        fallbackNode, "techmap_conservative_fallback_config_class_count",
        configAttrs.size(), dfg.context);
    if (const auto *fallbackInfo =
            TechMapper::findFallbackNodeInfo(plan, swNodeId)) {
      addNodeAttr(fallbackNode, "techmap_fallback_reason",
                  mlir::StringAttr::get(dfg.context, fallbackInfo->reason),
                  dfg.context);
      llvm::SmallVector<mlir::Attribute, 8> candidateIdAttrs;
      for (unsigned candidateId : fallbackInfo->candidateIds) {
        candidateIdAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), candidateId));
      }
      addNodeAttr(fallbackNode, "techmap_fallback_candidate_ids",
                  mlir::ArrayAttr::get(dfg.context, candidateIdAttrs),
                  dfg.context);
    }
    if (preferredCandidate) {
      const auto *preferredSupportClass =
          TechMapper::findSupportClass(plan, preferredCandidate->supportClassId);
      const auto *preferredConfigClass =
          TechMapper::findConfigClass(plan, preferredCandidate->configClassId);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_hw_node",
                      preferredCandidate->hwNodeId, dfg.context);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_support_class_id",
                      preferredCandidate->supportClassId, dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_support_class_key",
                  mlir::StringAttr::get(
                      dfg.context, preferredSupportClass
                                       ? llvm::StringRef(preferredSupportClass->key)
                                       : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_support_class_kind",
                  mlir::StringAttr::get(
                      dfg.context, preferredSupportClass
                                       ? llvm::StringRef(preferredSupportClass->kind)
                                       : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(
          fallbackNode, "techmap_preferred_support_class_temporal",
          preferredSupportClass && preferredSupportClass->temporal,
          dfg.context);
      addBoolNodeAttr(
          fallbackNode, "techmap_preferred_support_class_enforce_hard_capacity",
          TechMapper::supportClassEnforcesHardCapacity(
              plan, preferredCandidate->supportClassId),
          dfg.context);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_support_class_capacity",
                      TechMapper::supportClassCapacity(
                          plan, preferredCandidate->supportClassId),
                      dfg.context);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_config_class_id",
                      preferredCandidate->configClassId, dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_config_class_key",
                  mlir::StringAttr::get(
                      dfg.context, preferredConfigClass
                                       ? llvm::StringRef(preferredConfigClass->key)
                                       : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_config_class_reason",
                  mlir::StringAttr::get(
                      dfg.context, preferredConfigClass
                                       ? llvm::StringRef(preferredConfigClass->reason)
                                       : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(
          fallbackNode, "techmap_preferred_config_class_temporal",
          TechMapper::isTemporalConfigClass(plan,
                                            preferredCandidate->configClassId),
          dfg.context);
      if (TechMapper::findConfigClass(plan, preferredCandidate->configClassId)) {
        llvm::SmallVector<mlir::Attribute, 4> compatIdAttrs;
        llvm::SmallVector<mlir::Attribute, 4> compatKeyAttrs;
        for (unsigned compatId :
             TechMapper::compatibleConfigClasses(
                 plan, preferredCandidate->configClassId)) {
          const auto *compatConfigClass =
              TechMapper::findConfigClass(plan, compatId);
          compatIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), compatId));
          compatKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context, compatConfigClass
                               ? llvm::StringRef(compatConfigClass->key)
                               : llvm::StringRef()));
        }
        addNodeAttr(fallbackNode, "techmap_preferred_config_class_compatible_with",
                    mlir::ArrayAttr::get(dfg.context, compatIdAttrs), dfg.context);
        addNodeAttr(
            fallbackNode, "techmap_preferred_config_class_compatible_with_keys",
            mlir::ArrayAttr::get(dfg.context, compatKeyAttrs), dfg.context);
      }
      addBoolNodeAttr(fallbackNode, "techmap_preferred_temporal",
                      preferredCandidate->temporal, dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_config_fields",
                  mlir::ArrayAttr::get(
                      dfg.context,
                      buildConfigFieldSummaryAttrs(preferredCandidate->configFields,
                                                  dfg.context)),
                  dfg.context);
    }
    if (const auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId);
        nodeInfo && nodeInfo->selectionComponentId !=
                        std::numeric_limits<unsigned>::max()) {
      addUIntNodeAttr(fallbackNode, "techmap_selection_component_id",
                      nodeInfo->selectionComponentId, dfg.context);
    }
    if (!hwAttrs.empty())
      ++plan.metrics.conservativeFallbackCoveredCount;
    else
      ++plan.metrics.conservativeFallbackMissingCount;
  }

  std::vector<int> nodeToUnit(dfg.nodes.size(), -1);
  for (unsigned unitIndex = 0; unitIndex < TechMapper::allUnits(plan).size();
       ++unitIndex) {
    const auto *unit = TechMapper::findUnit(plan, unitIndex);
    if (!unit)
      continue;
    for (IdIndex swNodeId : unit->swNodes) {
      if (swNodeId < nodeToUnit.size())
        nodeToUnit[swNodeId] = unitIndex;
    }
  }

  plan.contractedDFG = Graph(dfg.context);
  auto &contracted = plan.contractedDFG;
  contracted.reserve(dfg.countNodes(), dfg.countPorts(), dfg.countEdges());
  plan.contractedCandidates.clear();
  plan.contractedCandidateSupportClasses.clear();
  plan.contractedCandidateConfigClasses.clear();

  std::vector<bool> unitCreated(TechMapper::allUnits(plan).size(), false);
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode)
      continue;

    int unitIndex = swNodeId < nodeToUnit.size() ? nodeToUnit[swNodeId] : -1;
    if (unitIndex >= 0) {
      auto *unit = TechMapper::findUnit(plan, static_cast<unsigned>(unitIndex));
      if (!unit)
        return false;
      if (unitCreated[unitIndex]) {
        plan.originalNodeToContractedNode[swNodeId] =
            unit->contractedNodeId;
        continue;
      }

      const auto *preferredCandidate =
          TechMapper::findPreferredUnitCandidate(plan, unitIndex);
      if (!preferredCandidate)
        return false;
      const auto *selectionComponent = TechMapper::findSelectionComponent(plan, *unit);
      const auto *familyInfo = TechMapper::findFamilyTechInfo(plan, *unit);
      const auto *selectedCandidateSummary =
          TechMapper::findSelectedCandidateSummary(plan, *unit);
      const auto *selectedConfigClass = TechMapper::findSelectedUnitConfigClass(plan, *unit);
      const auto *preferredSupportClass =
          TechMapper::findPreferredUnitSupportClass(plan, unitIndex);
      const auto *preferredConfigClass =
          TechMapper::findPreferredUnitConfigClass(plan, unitIndex);
      const Node *hwNode = adg.getNode(preferredCandidate->hwNodeId);
      if (!hwNode)
        return false;

      auto node = std::make_unique<Node>();
      node->kind = Node::OperationNode;
      addNodeAttr(node.get(), "op_name",
                  mlir::StringAttr::get(dfg.context, "techmap_group"),
                  dfg.context);
      addNodeAttr(node.get(), "tech_group_size",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(dfg.context, 32),
                                         unit->swNodes.size()),
                  dfg.context);
      addUIntNodeAttr(node.get(), "techmap_preferred_candidate_index",
                      unit->preferredCandidateIndex,
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_selected_unit_index", unitIndex,
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_selected_candidate_id",
                      unit->selectedCandidateId, dfg.context);
      if (unit->selectionComponentId !=
          std::numeric_limits<unsigned>::max()) {
        addUIntNodeAttr(node.get(), "techmap_selection_component_id",
                        unit->selectionComponentId, dfg.context);
        if (selectionComponent) {
          addNodeAttr(
              node.get(), "techmap_selection_solver",
              mlir::StringAttr::get(
                  dfg.context, selectionComponent->solver),
              dfg.context);
        }
      }
      addUIntNodeAttr(node.get(), "techmap_family_index",
                      unit->familyIndex, dfg.context);
      if (familyInfo) {
        addNodeAttr(
            node.get(), "techmap_family_signature",
            mlir::StringAttr::get(dfg.context, familyInfo->signature),
            dfg.context);
      }
      addUIntNodeAttr(node.get(), "techmap_config_class_id",
                      unit->configClassId, dfg.context);
      addNodeAttr(node.get(), "techmap_config_class_key",
                  mlir::StringAttr::get(dfg.context, selectedConfigClass
                                                         ? selectedConfigClass
                                                               ->key
                                                         : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_config_class_reason",
                  mlir::StringAttr::get(dfg.context, selectedConfigClass
                                                         ? selectedConfigClass
                                                               ->reason
                                                         : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(node.get(), "techmap_config_class_temporal",
                      TechMapper::isTemporalConfigClass(plan, unit->configClassId),
                      dfg.context);
      if (TechMapper::findConfigClass(plan, unit->configClassId)) {
        llvm::SmallVector<mlir::Attribute, 4> compatIdAttrs;
        llvm::SmallVector<mlir::Attribute, 4> compatKeyAttrs;
        for (unsigned compatId :
             TechMapper::compatibleConfigClasses(plan, unit->configClassId)) {
          const auto *compatConfigClass =
              TechMapper::findConfigClass(plan, compatId);
          compatIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), compatId));
          compatKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context, compatConfigClass
                               ? llvm::StringRef(compatConfigClass->key)
                               : llvm::StringRef()));
        }
        addNodeAttr(node.get(), "techmap_config_class_compatible_with",
                    mlir::ArrayAttr::get(dfg.context, compatIdAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_compatible_with_keys",
                    mlir::ArrayAttr::get(dfg.context, compatKeyAttrs),
                    dfg.context);
      }
      addUIntNodeAttr(node.get(), "techmap_support_class_id",
                      preferredCandidate->supportClassId, dfg.context);
      addNodeAttr(node.get(), "techmap_support_class_key",
                  mlir::StringAttr::get(dfg.context, preferredSupportClass
                                                         ? preferredSupportClass
                                                               ->key
                                                         : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_support_class_kind",
                  mlir::StringAttr::get(dfg.context, preferredSupportClass
                                                         ? preferredSupportClass
                                                               ->kind
                                                         : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(node.get(), "techmap_support_class_temporal",
                      preferredSupportClass && preferredSupportClass->temporal,
                      dfg.context);
      addBoolNodeAttr(node.get(), "techmap_support_class_enforce_hard_capacity",
                      TechMapper::supportClassEnforcesHardCapacity(
                          plan, preferredCandidate->supportClassId),
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_support_class_capacity",
                      TechMapper::supportClassCapacity(plan,
                                                       preferredCandidate
                                                           ->supportClassId),
                      dfg.context);
      if (!unit->swNodes.empty() &&
          unit->selectionComponentId == std::numeric_limits<unsigned>::max()) {
        IdIndex reprSwNodeId = unit->swNodes.front();
        if (const auto *nodeInfo =
                TechMapper::findNodeTechInfo(plan, reprSwNodeId);
            nodeInfo && nodeInfo->selectionComponentId !=
                            std::numeric_limits<unsigned>::max()) {
          addUIntNodeAttr(node.get(), "techmap_selection_component_id",
                          nodeInfo->selectionComponentId, dfg.context);
        }
      }
      addNodeAttr(node.get(), "techmap_selected_config_fields",
                  mlir::ArrayAttr::get(
                      dfg.context,
                      buildConfigFieldSummaryAttrs(preferredCandidate->configFields,
                                                  dfg.context)),
                  dfg.context);
      if (selectedCandidateSummary) {
        addUIntNodeAttr(
            node.get(), "techmap_base_selection_score",
            static_cast<uint64_t>(std::max<int64_t>(
                0, selectedCandidateSummary->baseSelectionScore)),
            dfg.context);
        addUIntNodeAttr(node.get(), "techmap_candidate_penalty",
                        static_cast<uint64_t>(
                            std::max<int64_t>(0,
                                              selectedCandidateSummary
                                                  ->candidatePenalty)),
                        dfg.context);
        addUIntNodeAttr(node.get(), "techmap_family_penalty",
                        static_cast<uint64_t>(
                            std::max<int64_t>(0,
                                              selectedCandidateSummary
                                                  ->familyPenalty)),
                        dfg.context);
        addUIntNodeAttr(
            node.get(), "techmap_config_class_penalty",
            static_cast<uint64_t>(
                std::max<int64_t>(0,
                                  selectedCandidateSummary
                                      ->configClassPenalty)),
            dfg.context);
      }
      addUIntNodeAttr(node.get(), "techmap_selection_score",
                      static_cast<uint64_t>(
                          std::max<int64_t>(0, unit->selectionScore)),
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_candidate_count",
                      unit->candidates.size(), dfg.context);
      addBoolNodeAttr(node.get(), "techmap_conservative_fallback",
                      unit->conservativeFallback, dfg.context);
      addBoolNodeAttr(node.get(), "techmap_legacy_fallback_origin",
                      unit->legacyFallbackOrigin, dfg.context);
      addNodeAttr(node.get(), "techmap_origin_kind",
                  mlir::StringAttr::get(
                      dfg.context,
                      TechMapper::originKind(unit->demandOrigin,
                                             unit->legacyFallbackOrigin,
                                             unit->mixedOrigin)),
                  dfg.context);
      llvm::SmallVector<mlir::Attribute, 4> candidateClassAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateClassKeyAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateClassReasonAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateClassTemporalAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportKeyAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportKindAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportTemporalAttrs;
      llvm::SmallVector<mlir::Attribute, 4>
          candidateSupportHardCapacityAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportCapacityAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateHwAttrs;
      for (const auto &unitCandidate : unit->candidates) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(plan, unitCandidate);
        const auto *candidateConfigClass =
            TechMapper::findConfigClass(plan, unitCandidate);
        candidateHwAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), unitCandidate.hwNodeId));
        candidateSupportAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            unitCandidate.supportClassId));
        candidateSupportKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateSupportClass ? candidateSupportClass->key
                                               : llvm::StringRef()));
        candidateSupportKindAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateSupportClass ? candidateSupportClass->kind
                                               : llvm::StringRef()));
        candidateSupportTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, candidateSupportClass
                             ? candidateSupportClass->temporal
                             : false));
        candidateSupportHardCapacityAttrs.push_back(mlir::BoolAttr::get(
            dfg.context,
            candidateSupportClass
                ? candidateSupportClass->enforceHardCapacity
                : false));
        candidateSupportCapacityAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            TechMapper::supportClassCapacity(plan,
                                             unitCandidate.supportClassId)));
        candidateClassAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            unitCandidate.configClassId));
        candidateClassKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateConfigClass ? candidateConfigClass->key
                                              : llvm::StringRef()));
        candidateClassReasonAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateConfigClass ? candidateConfigClass->reason
                                              : llvm::StringRef()));
        candidateClassTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, candidateConfigClass ? candidateConfigClass->temporal
                                              : false));
      }
      addNodeAttr(node.get(), "techmap_candidate_hw_nodes",
                  mlir::ArrayAttr::get(dfg.context, candidateHwAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_classes",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_class_keys",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportKeyAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_class_kinds",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportKindAttrs),
                  dfg.context);
      addNodeAttr(
          node.get(), "techmap_candidate_support_class_temporal",
          mlir::ArrayAttr::get(dfg.context, candidateSupportTemporalAttrs),
          dfg.context);
      addNodeAttr(
          node.get(), "techmap_candidate_support_class_enforce_hard_capacity",
          mlir::ArrayAttr::get(dfg.context, candidateSupportHardCapacityAttrs),
          dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_class_capacities",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportCapacityAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_classes",
                  mlir::ArrayAttr::get(dfg.context, candidateClassAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_class_keys",
                  mlir::ArrayAttr::get(dfg.context, candidateClassKeyAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_class_reasons",
                  mlir::ArrayAttr::get(dfg.context, candidateClassReasonAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_class_temporal",
                  mlir::ArrayAttr::get(dfg.context, candidateClassTemporalAttrs),
                  dfg.context);
      IdIndex contractedNodeId = contracted.addNode(std::move(node));
      unit->contractedNodeId = contractedNodeId;
      if (auto *selectedCandidateSummary = TechMapper::findCandidateSummary(
              plan, unit->selectedCandidateId)) {
        selectedCandidateSummary->contractedNodeId = contractedNodeId;
      }
      for (IdIndex member : unit->swNodes) {
        plan.originalNodeToContractedNode[member] = contractedNodeId;
        if (auto *nodeInfo = TechMapper::findNodeTechInfo(plan, member))
          nodeInfo->contractedNodeId = contractedNodeId;
      }

      for (IdIndex hwPortId : hwNode->inputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->inputPorts.push_back(portId);
        unit->contractedInputPorts.push_back(portId);
      }
      for (IdIndex hwPortId : hwNode->outputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->outputPorts.push_back(portId);
        unit->contractedOutputPorts.push_back(portId);
      }
      for (const auto &unitCandidate : unit->candidates)
        plan.contractedCandidates[contractedNodeId].push_back(
            unitCandidate.hwNodeId);
      for (const auto &unitCandidate : unit->candidates)
        plan.contractedCandidateSupportClasses[contractedNodeId].push_back(
            unitCandidate.supportClassId);
      for (const auto &unitCandidate : unit->candidates)
        plan.contractedCandidateConfigClasses[contractedNodeId].push_back(
            unitCandidate.configClassId);
      unitCreated[unitIndex] = true;
      continue;
    }

    auto node = cloneNodeShell(swNode, dfg.context);
    if (swNode->kind == Node::OperationNode) {
      addBoolNodeAttr(node.get(), "techmap_conservative_fallback", true,
                      dfg.context);
      if (const auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId)) {
        if (nodeInfo->selectionComponentId !=
            std::numeric_limits<unsigned>::max()) {
          addUIntNodeAttr(node.get(), "techmap_selection_component_id",
                          nodeInfo->selectionComponentId, dfg.context);
        }
        addUIntNodeAttr(node.get(), "techmap_candidate_count",
                        nodeInfo->candidateCount, dfg.context);
        addUIntNodeAttr(node.get(), "techmap_support_class_count",
                        nodeInfo->supportClassCount, dfg.context);
        addUIntNodeAttr(node.get(), "techmap_config_class_count",
                        nodeInfo->configClassCount, dfg.context);
        addUIntNodeAttr(node.get(), "techmap_max_fusion_size",
                        nodeInfo->maxFusionSize, dfg.context);
        addNodeAttr(node.get(), "techmap_status",
                    mlir::StringAttr::get(dfg.context, nodeInfo->status),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> candidateIdAttrs;
        for (unsigned candidateId : nodeInfo->candidateIds) {
          candidateIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), candidateId));
        }
        addNodeAttr(node.get(), "techmap_candidate_ids",
                    mlir::ArrayAttr::get(dfg.context, candidateIdAttrs),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> supportClassAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassKeyAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassKindAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassTemporalAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassHardCapacityAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassCapacityAttrs;
        for (unsigned supportClassId : nodeInfo->supportClassIds) {
          const auto *supportClassInfo =
              TechMapper::findSupportClass(plan, supportClassId);
          supportClassAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), supportClassId));
          supportClassKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              supportClassInfo ? llvm::StringRef(supportClassInfo->key)
                               : llvm::StringRef()));
          supportClassKindAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              supportClassInfo ? llvm::StringRef(supportClassInfo->kind)
                               : llvm::StringRef()));
          supportClassTemporalAttrs.push_back(mlir::BoolAttr::get(
              dfg.context, supportClassInfo && supportClassInfo->temporal));
          supportClassHardCapacityAttrs.push_back(mlir::BoolAttr::get(
              dfg.context,
              TechMapper::supportClassEnforcesHardCapacity(plan, supportClassId)));
          supportClassCapacityAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64),
              TechMapper::supportClassCapacity(plan, supportClassId)));
        }
        addNodeAttr(node.get(), "techmap_support_class_ids",
                    mlir::ArrayAttr::get(dfg.context, supportClassAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_keys",
                    mlir::ArrayAttr::get(dfg.context, supportClassKeyAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_kinds",
                    mlir::ArrayAttr::get(dfg.context, supportClassKindAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_temporal",
                    mlir::ArrayAttr::get(dfg.context, supportClassTemporalAttrs),
                    dfg.context);
        addNodeAttr(
            node.get(), "techmap_support_class_enforce_hard_capacity",
            mlir::ArrayAttr::get(dfg.context, supportClassHardCapacityAttrs),
            dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_capacities",
                    mlir::ArrayAttr::get(dfg.context, supportClassCapacityAttrs),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> configClassAttrs;
        llvm::SmallVector<mlir::Attribute, 8> configClassKeyAttrs;
        llvm::SmallVector<mlir::Attribute, 8> configClassReasonAttrs;
        llvm::SmallVector<mlir::Attribute, 8> configClassTemporalAttrs;
        for (unsigned configClassId : nodeInfo->configClassIds) {
          const auto *configClassInfo =
              TechMapper::findConfigClass(plan, configClassId);
          configClassAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), configClassId));
          configClassKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              configClassInfo ? llvm::StringRef(configClassInfo->key)
                              : llvm::StringRef()));
          configClassReasonAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              configClassInfo ? llvm::StringRef(configClassInfo->reason)
                              : llvm::StringRef()));
          configClassTemporalAttrs.push_back(mlir::BoolAttr::get(
              dfg.context,
              TechMapper::isTemporalConfigClass(plan, configClassId)));
        }
        addNodeAttr(node.get(), "techmap_config_class_ids",
                    mlir::ArrayAttr::get(dfg.context, configClassAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_keys",
                    mlir::ArrayAttr::get(dfg.context, configClassKeyAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_reasons",
                    mlir::ArrayAttr::get(dfg.context, configClassReasonAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_temporal",
                    mlir::ArrayAttr::get(dfg.context, configClassTemporalAttrs),
                    dfg.context);
      }
      if (const auto *fallbackInfo =
              TechMapper::findFallbackNodeInfo(plan, swNodeId)) {
        addNodeAttr(node.get(), "techmap_fallback_reason",
                    mlir::StringAttr::get(dfg.context, fallbackInfo->reason),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> fallbackCandidateIdAttrs;
        for (unsigned candidateId : fallbackInfo->candidateIds) {
          fallbackCandidateIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), candidateId));
        }
        addNodeAttr(node.get(), "techmap_fallback_candidate_ids",
                    mlir::ArrayAttr::get(dfg.context, fallbackCandidateIdAttrs),
                    dfg.context);
      }
      fallbackSwNodes.push_back(swNodeId);
    }
    IdIndex contractedNodeId = contracted.addNode(std::move(node));
    plan.originalNodeToContractedNode[swNodeId] = contractedNodeId;
    if (auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId))
      nodeInfo->contractedNodeId = contractedNodeId;
    if (auto *fallbackInfo = TechMapper::findFallbackNodeInfo(plan, swNodeId))
      fallbackInfo->contractedNodeId = contractedNodeId;

    for (IdIndex swPortId : swNode->inputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->inputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
    for (IdIndex swPortId : swNode->outputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->outputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
  }

  for (auto &unit : TechMapper::allUnits(plan)) {
    for (const auto &binding : unit.inputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedInputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedInputPorts[binding.hwPortIndex];
    }
    for (const auto &binding : unit.outputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedOutputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedOutputPorts[binding.hwPortIndex];
    }
    for (IdIndex edgeId : unit.internalEdges) {
      if (edgeId < plan.originalEdgeKinds.size())
        plan.originalEdgeKinds[edgeId] = TechMappedEdgeKind::IntraFU;
    }
  }

  std::map<std::string, IdIndex> dedupEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (plan.originalEdgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcPort = plan.originalPortToContractedPort[edge->srcPort];
    IdIndex dstPort = plan.originalPortToContractedPort[edge->dstPort];
    if (srcPort == INVALID_ID || dstPort == INVALID_ID) {
      plan.diagnostics = "tech-mapping lost an external port binding";
      return false;
    }

    std::string key = std::to_string(srcPort) + ":" + std::to_string(dstPort);
    auto found = dedupEdges.find(key);
    if (found != dedupEdges.end()) {
      plan.originalEdgeToContractedEdge[edgeId] = found->second;
      continue;
    }

    auto newEdge = std::make_unique<Edge>();
    newEdge->srcPort = srcPort;
    newEdge->dstPort = dstPort;
    newEdge->attributes = edge->attributes;
    IdIndex contractedEdgeId = contracted.addEdge(std::move(newEdge));
    contracted.ports[srcPort]->connectedEdges.push_back(contractedEdgeId);
    contracted.ports[dstPort]->connectedEdges.push_back(contractedEdgeId);
    dedupEdges[key] = contractedEdgeId;
    plan.originalEdgeToContractedEdge[edgeId] = contractedEdgeId;
  }

  return true;
}

bool TechMapper::expandPlanMapping(
    const Graph &originalDfg, const Graph &adg, const Plan &plan,
    const MappingState &contractedState, MappingState &expandedState,
    llvm::SmallVectorImpl<FUConfigSelection> &fuConfigs) {
  expandedState.init(originalDfg, adg);
  expandedState.hwNodeFifoBypassedOverride =
      contractedState.hwNodeFifoBypassedOverride;
  fuConfigs.clear();

  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(plan.originalNodeToContractedNode.size());
       ++swNodeId) {
    IdIndex contractedNodeId = plan.originalNodeToContractedNode[swNodeId];
    if (contractedNodeId == INVALID_ID ||
        contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    expandedState.swNodeToHwNode[swNodeId] = hwNodeId;
    expandedState.hwNodeToSwNodes[hwNodeId].push_back(swNodeId);
  }

  for (IdIndex swPortId = 0;
       swPortId < static_cast<IdIndex>(plan.originalPortToContractedPort.size());
       ++swPortId) {
    IdIndex contractedPortId = plan.originalPortToContractedPort[swPortId];
    if (contractedPortId == INVALID_ID ||
        contractedPortId >= contractedState.swPortToHwPort.size())
      continue;
    IdIndex hwPortId = contractedState.swPortToHwPort[contractedPortId];
    if (hwPortId == INVALID_ID)
      continue;
    expandedState.swPortToHwPort[swPortId] = hwPortId;
    expandedState.hwPortToSwPorts[hwPortId].push_back(swPortId);
  }

  for (IdIndex swEdgeId = 0;
       swEdgeId < static_cast<IdIndex>(plan.originalEdgeToContractedEdge.size());
       ++swEdgeId) {
    if (plan.originalEdgeKinds[swEdgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    IdIndex contractedEdgeId = plan.originalEdgeToContractedEdge[swEdgeId];
    if (contractedEdgeId == INVALID_ID ||
        contractedEdgeId >= contractedState.swEdgeToHwPaths.size())
      continue;
    expandedState.swEdgeToHwPaths[swEdgeId] =
        contractedState.swEdgeToHwPaths[contractedEdgeId];
    llvm::ArrayRef<IdIndex> path = expandedState.swEdgeToHwPaths[swEdgeId];
    for (size_t i = 0; i + 1 < path.size(); i += 2) {
      IdIndex outPort = path[i];
      IdIndex inPort = path[i + 1];
      const Port *hwOut = adg.getPort(outPort);
      if (!hwOut)
        continue;
      for (IdIndex hwEdgeId : hwOut->connectedEdges) {
        const Edge *hwEdge = adg.getEdge(hwEdgeId);
        if (!hwEdge)
          continue;
        if (hwEdge->srcPort == outPort && hwEdge->dstPort == inPort) {
          expandedState.hwEdgeToSwEdges[hwEdgeId].push_back(swEdgeId);
          break;
        }
      }
    }
  }

  for (const Unit &unit : TechMapper::allUnits(plan)) {
    if (unit.contractedNodeId == INVALID_ID ||
        unit.contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[unit.contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    for (const Candidate &candidate : unit.candidates) {
      if (candidate.hwNodeId != hwNodeId || candidate.configFields.empty())
        continue;
      FUConfigSelection selection;
      selection.hwNodeId = hwNodeId;
      selection.supportClassId = candidate.supportClassId;
      selection.configClassId = candidate.configClassId;
      if (const Node *hwNode = adg.getNode(hwNodeId)) {
        selection.hwName = getNodeAttrStr(hwNode, "op_name").str();
        selection.peName = getNodeAttrStr(hwNode, "pe_name").str();
      }
      selection.swNodeIds.append(unit.swNodes.begin(), unit.swNodes.end());
      selection.fields.append(candidate.configFields.begin(),
                              candidate.configFields.end());
      fuConfigs.push_back(std::move(selection));
      break;
    }
  }

  return true;
}

} // namespace loom
