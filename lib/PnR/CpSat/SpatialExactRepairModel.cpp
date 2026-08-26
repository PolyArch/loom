#include "SpatialExactRepairModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <climits>
#include <cstddef>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;
using namespace operations_research;
using namespace operations_research::sat;

namespace {

llvm::Error modelError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial exact-repair model: %s", detail.str().c_str());
}

llvm::Expected<std::optional<std::string>>
admitChoiceStorage(std::size_t legalValueCount, std::size_t &total,
                   llvm::StringRef emptyReason, llvm::StringRef storageReason) {
  if (legalValueCount == 0)
    return std::optional<std::string>(emptyReason.str());
  const std::size_t maximumLegalValueCount = getPnrIndexMax();
  if (legalValueCount > maximumLegalValueCount ||
      total > maximumLegalValueCount - legalValueCount)
    return std::optional<std::string>(storageReason.str());
  total += legalValueCount;
  return std::nullopt;
}

} // namespace

llvm::Expected<std::optional<std::string>>
loom::pnr::detail::admitAtomicExactRepairModel(
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<PnrIndex> decisions,
    llvm::ArrayRef<std::uint64_t> contextOveruse) {
  if (decisions.size() > static_cast<std::size_t>(INT_MAX))
    return std::optional<std::string>(
        "compute decision domain is not CP-SAT encodable");

  std::size_t legalValueCount = 0;
  for (PnrIndex decision : decisions) {
    const auto choices = bindings.computeChoices(decision);
    auto admitted =
        admitChoiceStorage(choices.size(), legalValueCount,
                           "compute choice domain is not CP-SAT encodable",
                           "flattened compute choice domain exceeds PnrIndex");
    if (!admitted)
      return admitted.takeError();
    if (*admitted)
      return std::move(*admitted);
    for (const SpatialComputeBindingChoice &choice : choices)
      if (choice.instructionContext >= contextOveruse.size())
        return modelError("compute choice has no capacity projection");
  }
  return std::nullopt;
}

llvm::Expected<std::optional<std::string>>
loom::pnr::detail::admitTransportExactRepairModel(
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<PnrIndex> decisions,
    llvm::ArrayRef<std::uint64_t> contextOveruse) {
  if (decisions.empty() || decisions.size() > static_cast<std::size_t>(INT_MAX))
    return std::optional<std::string>(
        "route repair decision domain is not CP-SAT encodable");

  const PnrIndex computeCount = bindings.computeDecisionCount();
  const PnrIndex portOffset = bindings.portDecisionOffset();
  const PnrIndex boundaryOffset = bindings.graphBoundaryDecisionOffset();
  std::size_t legalValueCount = 0;
  for (PnrIndex decision : decisions) {
    std::size_t decisionLegalValueCount = 0;
    if (decision < computeCount) {
      for (const SpatialComputeBindingChoice &choice :
           bindings.computeChoices(decision)) {
        if (choice.instructionContext >= contextOveruse.size())
          return modelError(
              "route repair compute choice has no capacity value");
        decisionLegalValueCount +=
            contextOveruse[choice.instructionContext] == 0 ? 1 : 0;
      }
    } else if (decision < portOffset) {
      decisionLegalValueCount =
          bindings.memoryChoices(decision - computeCount).size();
    } else if (decision < boundaryOffset) {
      decisionLegalValueCount =
          bindings.portAttachmentChoices(decision - portOffset).size();
    } else {
      if (decision - boundaryOffset >= bindings.graphBoundaryDecisionCount())
        return modelError("route repair binding decision is out of range");
      decisionLegalValueCount =
          bindings.graphBoundaryAttachmentChoices(decision - boundaryOffset)
              .size();
    }
    auto admitted =
        admitChoiceStorage(decisionLegalValueCount, legalValueCount,
                           "route repair decision has no legal choice",
                           "route repair choice storage exceeds PnrIndex");
    if (!admitted)
      return admitted.takeError();
    if (*admitted)
      return std::move(*admitted);
  }
  return std::nullopt;
}

llvm::Expected<std::uint64_t>
loom::pnr::detail::countExactRepairRegionDecisions(
    llvm::ArrayRef<PnrIndex> decisions, llvm::ArrayRef<PnrIndex> affectedNets,
    const FrozenSpatialPnrProblem &problem) {
  std::uint64_t count = decisions.size();
  const auto offsets = problem.ports().computeRealizationDemandOffsets();
  if (offsets.size() != problem.realizations().computeRealizations().size() + 1)
    return modelError("compute-demand reverse index is incomplete");
  for (PnrIndex decision : decisions) {
    if (decision >= offsets.size() - 1)
      return modelError("atomic repair decision is out of range");
    const std::uint64_t demandCount = offsets[decision + 1] - offsets[decision];
    if (demandCount > std::numeric_limits<std::uint64_t>::max() - count)
      return modelError("repair region decision count overflows");
    count += demandCount;
  }
  if (affectedNets.size() > std::numeric_limits<std::uint64_t>::max() - count)
    return modelError("repair region decision count overflows");
  return count + affectedNets.size();
}

llvm::Expected<PnrIndex> loom::pnr::detail::currentExactRepairBindingChoice(
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings, PnrIndex decision) {
  if (decision < bindings.computeDecisionCount()) {
    const SpatialComputeBindingSelection &selected =
        candidate.computeBinding(decision);
    const auto choice = bindings.computeChoiceOrdinal(
        decision, selected.placement, selected.instructionContext);
    if (!choice)
      return modelError("compute binding has no relation-domain choice");
    return *choice;
  }

  if (decision < bindings.portDecisionOffset()) {
    const PnrIndex realization = decision - bindings.computeDecisionCount();
    const auto choice = bindings.memoryChoiceOrdinal(
        realization, candidate.memoryBinding(realization).placement);
    if (!choice)
      return modelError("memory binding has no relation-domain choice");
    return *choice;
  }

  if (decision < bindings.graphBoundaryDecisionOffset()) {
    const PnrIndex demand = decision - bindings.portDecisionOffset();
    const auto choice = bindings.portAttachmentChoiceOrdinal(
        demand, candidate.portAttachment(demand));
    if (!choice)
      return modelError("PortDemand has no relation-domain choice");
    return *choice;
  }

  const PnrIndex boundary = decision - bindings.graphBoundaryDecisionOffset();
  if (boundary >= bindings.graphBoundaryDecisionCount())
    return modelError("binding decision is out of range");
  const auto choice = bindings.graphBoundaryAttachmentChoiceOrdinal(
      boundary, candidate.graphBoundaryAttachment(boundary));
  if (!choice)
    return modelError("graph boundary has no relation-domain choice");
  return *choice;
}

llvm::Expected<IntVar> loom::pnr::detail::addExactRepairMutationCountObjective(
    CpModelBuilder &model, llvm::ArrayRef<IntVar> variables,
    const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<PnrIndex> decisions,
    llvm::ArrayRef<int> mutationParentLocals,
    llvm::ArrayRef<IntVar> additionalVariables,
    llvm::ArrayRef<std::int64_t> additionalCurrentValues) {
  if (variables.size() != decisions.size())
    return modelError(
        "mutation objective variable and decision counts disagree");
  if (additionalVariables.size() != additionalCurrentValues.size())
    return modelError(
        "additional mutation variable and current-value counts disagree");
  if (!mutationParentLocals.empty() &&
      mutationParentLocals.size() != decisions.size())
    return modelError("mutation parent and decision counts disagree");
  if (decisions.size() > static_cast<std::size_t>(INT64_MAX) ||
      additionalVariables.size() >
          static_cast<std::size_t>(INT64_MAX) - decisions.size())
    return modelError("mutation objective domain is not encodable");

  std::vector<BoolVar> decisionChanged;
  decisionChanged.reserve(decisions.size());
  for (auto [local, decision] : llvm::enumerate(decisions)) {
    auto current =
        currentExactRepairBindingChoice(candidate, bindings, decision);
    if (!current)
      return current.takeError();
    const BoolVar differs = model.NewBoolVar();
    model.AddNotEqual(variables[local], *current).OnlyEnforceIf(differs);
    model.AddEquality(variables[local], *current).OnlyEnforceIf(differs.Not());
    decisionChanged.push_back(differs);
  }

  std::vector<BoolVar> changed;
  changed.reserve(decisions.size() + additionalVariables.size());
  for (std::size_t local = 0; local < decisionChanged.size(); ++local) {
    const int parent =
        mutationParentLocals.empty() ? -1 : mutationParentLocals[local];
    if (parent < 0) {
      changed.push_back(decisionChanged[local]);
      continue;
    }
    if (static_cast<std::size_t>(parent) >= decisionChanged.size() ||
        static_cast<std::size_t>(parent) == local)
      return modelError("mutation parent is invalid");
    const BoolVar independentlyChanged = model.NewBoolVar();
    model.AddBoolAnd({decisionChanged[local], decisionChanged[parent].Not()})
        .OnlyEnforceIf(independentlyChanged);
    model.AddBoolOr({decisionChanged[local].Not(), decisionChanged[parent]})
        .OnlyEnforceIf(independentlyChanged.Not());
    changed.push_back(independentlyChanged);
  }
  for (auto [variable, current] :
       llvm::zip_equal(additionalVariables, additionalCurrentValues)) {
    const BoolVar differs = model.NewBoolVar();
    model.AddNotEqual(variable, current).OnlyEnforceIf(differs);
    model.AddEquality(variable, current).OnlyEnforceIf(differs.Not());
    changed.push_back(differs);
  }

  const IntVar mutationCount =
      model.NewIntVar(Domain(0, static_cast<std::int64_t>(changed.size())));
  model.AddEquality(mutationCount, LinearExpr::Sum(changed));
  model.Minimize(mutationCount);
  return mutationCount;
}

void loom::pnr::detail::addExactRepairInitializerRelationConstraint(
    CpModelBuilder &model, const InitializerRelationModel &relationModel,
    const InitializerRelationRecord &record,
    llvm::ArrayRef<IntVar> projections) {
  if (record.kind == InitializerRelationKind::Equal) {
    for (std::size_t member = 1; member < projections.size(); ++member)
      model.AddEquality(projections[member], projections.front());
    return;
  }
  if (record.kind == InitializerRelationKind::Disjoint) {
    model.AddAllDifferent(projections);
    return;
  }

  const auto members = relationModel.members(record);
  const auto capacities = relationModel.valueCapacities(record);
  assert(members.size() == projections.size());
  std::vector<BoolVar> selected;
  std::vector<std::int64_t> demands;
  selected.reserve(members.size());
  demands.reserve(members.size());
  for (PnrIndex value = 0; value < capacities.size(); ++value) {
    selected.clear();
    demands.clear();
    for (auto [member, projection] : llvm::zip_equal(members, projections)) {
      const BoolVar usesValue = model.NewBoolVar();
      model.AddEquality(projection, value).OnlyEnforceIf(usesValue);
      model.AddNotEqual(projection, value).OnlyEnforceIf(usesValue.Not());
      selected.push_back(usesValue);
      demands.push_back(static_cast<std::int64_t>(member.demand));
    }
    model.AddLessOrEqual(LinearExpr::WeightedSum(selected, demands),
                         static_cast<std::int64_t>(capacities[value]));
  }
}
