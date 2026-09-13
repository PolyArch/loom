#include "SpatialHandshakeCycleCoreConstraint.h"

#include "Common/MappingDebugLog.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Error.h"

#include <array>
#include <system_error>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;
using namespace operations_research::sat;

namespace {

llvm::Error coreConstraintError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial handshake cycle core constraint: %s",
      detail.str().c_str());
}

llvm::Expected<SpatialHandshakeCycleCoreConstraintResult> addEscapeConstraint(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> variables, llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> legalValueOffsets,
    llvm::ArrayRef<std::int64_t> legalValues,
    const SpatialHandshakeCoreCoPlacement &coPlacement) {
  SpatialHandshakeCycleCoreConstraintResult result;
  if (coPlacement.computeDecisions.empty())
    return result;
  const auto placements = candidate.problem().realizations().computePlacements();

  std::vector<BoolVar> escaped;
  escaped.reserve(coPlacement.computeDecisions.size());
  std::vector<::loom::fabric::FabricPeOccurrenceRef> escapeTargets;
  for (PnrIndex decision : coPlacement.computeDecisions) {
    if (decision >= decisionVariables.size())
      return coreConstraintError("class decision is out of range");
    const int local = decisionVariables[decision];
    if (local < 0) {
      // The class reaches a decision this bounded region pins. Its own
      // enumeration cannot state the class, so nothing is encoded.
      result.outsideRegion = true;
      return result;
    }
    if (static_cast<std::size_t>(local) >= variables.size() ||
        static_cast<std::size_t>(local) + 1 >= legalValueOffsets.size() ||
        legalValueOffsets[local] > legalValueOffsets[local + 1] ||
        legalValueOffsets[local + 1] > legalValues.size())
      return coreConstraintError("class decision domain is malformed");
    const auto choices = bindings.computeChoices(decision);
    const BoolVar decisionEscaped = model.NewBoolVar();
    TableConstraint table =
        model.AddAllowedAssignments({variables[local], IntVar(decisionEscaped)});
    for (std::int64_t choice :
         legalValues.slice(legalValueOffsets[local],
                           legalValueOffsets[local + 1] -
                               legalValueOffsets[local])) {
      if (choice < 0 || static_cast<std::size_t>(choice) >= choices.size())
        return coreConstraintError("class choice is out of range");
      const PnrIndex placement = choices[choice].placement;
      if (placement >= placements.size())
        return coreConstraintError("class placement is out of range");
      const bool choiceEscapes = spatialHandshakeCoreCoPlacementEscapes(
          coPlacement, placements[placement]);
      if (choiceEscapes) {
        ++result.escapingChoiceCount;
        if (!llvm::is_contained(escapeTargets, placements[placement].parentPe))
          escapeTargets.push_back(placements[placement].parentPe);
      }
      const std::array<std::int64_t, 2> tuple{choice, choiceEscapes ? 1 : 0};
      table.AddTuple(tuple);
    }
    escaped.push_back(decisionEscaped);
    ++result.classDecisionCount;
  }
  result.escapeTargetPeCount = static_cast<std::uint64_t>(escapeTargets.size());
  // With no escaping choice the clause would refuse the whole legal domain.
  // That is not a search fact but a Fabric fact, and its owner reports it.
  if (result.escapingChoiceCount == 0)
    return result;
  model.AddAtLeastOne(escaped);
  result.encoded = true;
  return result;
}

} // namespace

llvm::Expected<SpatialHandshakeCycleCoreConstraintResult>
loom::pnr::detail::stateSpatialHandshakeCycleCoreClass(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> variables, llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> legalValueOffsets,
    llvm::ArrayRef<std::int64_t> legalValues,
    const SpatialHandshakeCycleCore &core) {
  if (!core.established())
    return SpatialHandshakeCycleCoreConstraintResult{};
  auto coPlacement =
      projectSpatialHandshakeCoreCoPlacement(candidate, core.arcs());
  if (!coPlacement)
    return coPlacement.takeError();
  auto encoded =
      addEscapeConstraint(model, candidate, bindings, variables,
                          decisionVariables, legalValueOffsets, legalValues,
                          *coPlacement);
  if (!encoded)
    return encoded.takeError();
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::ActionProposal,
      [&](llvm::json::Object &fields) {
        fields["operation"] = "handshake_cycle_core_co_placement";
        fields["encoded"] = encoded->encoded;
        fields["outside_region"] = encoded->outsideRegion;
        fields["class_decision_count"] = encoded->classDecisionCount;
        fields["escaping_choice_count"] = encoded->escapingChoiceCount;
        fields["escape_target_pe_count"] = encoded->escapeTargetPeCount;
        fields["neighbourhood_pe_count"] =
            static_cast<std::uint64_t>(coPlacement->neighbourhood.size());
        fields["core_fu_occurrence_count"] = coPlacement->coreFuOccurrenceCount;
        fields["core_arc_count"] =
            static_cast<std::uint64_t>(core.arcs().size());
      });
  encoded->coPlacement = std::move(*coPlacement);
  return *encoded;
}
