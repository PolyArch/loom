#include "SpatialLocalDispositionModel.h"

#include "SpatialRouteConstraintModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <climits>
#include <map>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;
using operations_research::Domain;
using namespace operations_research::sat;

namespace {

using FifoKey =
    std::pair<::loom::fabric::FabricEntityId, ::loom::fabric::FabricOrdinal>;

llvm::Error dispositionError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial local-disposition model: %s", message.str().c_str());
}

FifoKey fifoKey(const FrozenSpatialRegisterFifoTransferOption &option) {
  return {option.pe.id(), option.registerFifo};
}

} // namespace

llvm::Expected<SpatialLocalDispositionModel>
SpatialLocalDispositionModel::build(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> bindingVariables,
    llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> affectedLogicalNets) {
  SpatialLocalDispositionModel result;
  result.problem_ = &candidate.problem();
  const auto &problem = candidate.problem();
  const auto domains = problem.localTransfers().domains();
  const auto options = problem.localTransfers().options();
  const PnrIndex logicalNetCount =
      static_cast<PnrIndex>(problem.transfers().logicalNets().size());
  if (domains.size() != logicalNetCount ||
      decisionVariables.size() != bindings.decisionCount())
    return dispositionError("frozen index dimensions are inconsistent");

  result.localByLogicalNet_.assign(logicalNetCount, -1);
  std::vector<std::uint8_t> affected(logicalNetCount, 0);
  for (auto [local, logicalNet] : llvm::enumerate(affectedLogicalNets)) {
    if (logicalNet >= logicalNetCount || affected[logicalNet])
      return dispositionError("affected logical nets are not unique");
    if (local > static_cast<std::size_t>(INT_MAX))
      return dispositionError("local-disposition index exceeds int");
    affected[logicalNet] = 1;
    result.localByLogicalNet_[logicalNet] = static_cast<int>(local);
  }

  std::set<FifoKey> occupied;
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount; ++logicalNet) {
    if (affected[logicalNet] || !candidate.usesRegisterFifo(logicalNet))
      continue;
    const PnrIndex selected = candidate.registerFifoTransfer(logicalNet);
    if (selected >= options.size())
      return dispositionError("candidate names a foreign local option");
    if (!occupied.insert(fifoKey(options[selected])).second)
      return dispositionError("candidate repeats an occupied register FIFO");
  }

  std::map<FifoKey, std::vector<BoolVar>> resourceSelections;
  result.variables_.reserve(affectedLogicalNets.size());
  result.localSelected_.reserve(affectedLogicalNets.size());
  result.logicalNets_.assign(affectedLogicalNets.begin(),
                             affectedLogicalNets.end());
  result.legalValues_.reserve(affectedLogicalNets.size());
  result.externalValues_.reserve(affectedLogicalNets.size());
  result.currentValues_.reserve(affectedLogicalNets.size());

  const auto constrainPlacement = [&](PnrIndex realization, PnrIndex placement,
                                      BoolVar selected) -> llvm::Error {
    if (realization >= bindings.computeDecisionCount())
      return dispositionError("local option has a foreign compute owner");
    const int local = decisionVariables[realization];
    if (local < 0) {
      if (candidate.computeBinding(realization).placement != placement)
        model.AddEquality(selected, 0);
      return llvm::Error::success();
    }
    if (static_cast<std::size_t>(local) >= bindingVariables.size())
      return dispositionError("compute binding variable is out of range");
    std::vector<std::int64_t> matchingChoices;
    for (auto [ordinal, choice] :
         llvm::enumerate(bindings.computeChoices(realization))) {
      if (choice.placement != placement)
        continue;
      matchingChoices.push_back(static_cast<std::int64_t>(ordinal));
    }
    if (!matchingChoices.empty()) {
      TableConstraint accepted =
          model.AddAllowedAssignments({bindingVariables[local]});
      for (std::int64_t choice : matchingChoices) {
        const std::array<std::int64_t, 1> tuple{choice};
        accepted.AddTuple(tuple);
      }
      accepted.OnlyEnforceIf(selected);
    } else {
      model.AddEquality(selected, 0);
    }
    return llvm::Error::success();
  };

  for (PnrIndex logicalNet : affectedLogicalNets) {
    const FrozenSpatialRegisterFifoTransferDomain &domain = domains[logicalNet];
    if (domain.optionOffset > options.size() ||
        domain.optionCount > options.size() - domain.optionOffset)
      return dispositionError("local option domain is out of range");
    const std::int64_t external = domain.optionCount;
    std::vector<std::int64_t> legal;
    if (!problem.routeConstraints().netHasConstraints(logicalNet)) {
      for (PnrIndex local = 0; local < domain.optionCount; ++local) {
        const auto &option = options[domain.optionOffset + local];
        if (option.logicalNet != logicalNet)
          return dispositionError("local option has a foreign net owner");
        if (occupied.find(fifoKey(option)) == occupied.end())
          legal.push_back(local);
      }
    }
    legal.push_back(external);
    const IntVar disposition = model.NewIntVar(Domain::FromValues(legal));
    const BoolVar localSelected = model.NewBoolVar();
    model.AddNotEqual(disposition, external).OnlyEnforceIf(localSelected);
    model.AddEquality(disposition, external).OnlyEnforceIf(localSelected.Not());

    for (std::int64_t local : llvm::ArrayRef<std::int64_t>(legal).drop_back()) {
      const auto &option = options[domain.optionOffset + local];
      const BoolVar selected = model.NewBoolVar();
      model.AddEquality(disposition, local).OnlyEnforceIf(selected);
      model.AddNotEqual(disposition, local).OnlyEnforceIf(selected.Not());
      if (llvm::Error error = constrainPlacement(
              option.producerRealization, option.producerPlacement, selected))
        return std::move(error);
      if (llvm::Error error = constrainPlacement(
              option.consumerRealization, option.consumerPlacement, selected))
        return std::move(error);
      resourceSelections[fifoKey(option)].push_back(selected);
    }

    std::int64_t current = external;
    if (candidate.usesRegisterFifo(logicalNet)) {
      const PnrIndex selected = candidate.registerFifoTransfer(logicalNet);
      if (selected < domain.optionOffset ||
          selected - domain.optionOffset >= domain.optionCount)
        return dispositionError("candidate local option is outside its domain");
      current = selected - domain.optionOffset;
      if (!llvm::is_contained(legal, current))
        return dispositionError("candidate local option is unavailable");
    }

    result.variables_.push_back(disposition);
    result.localSelected_.push_back(localSelected);
    result.legalValues_.push_back(std::move(legal));
    result.externalValues_.push_back(external);
    result.currentValues_.push_back(current);
  }

  for (auto &[resource, selections] : resourceSelections) {
    (void)resource;
    if (selections.size() > 1)
      model.AddAtMostOne(selections);
  }
  return result;
}

std::optional<PnrIndex>
SpatialLocalDispositionModel::localForLogicalNet(PnrIndex logicalNet) const {
  if (logicalNet >= localByLogicalNet_.size() ||
      localByLogicalNet_[logicalNet] < 0)
    return std::nullopt;
  return static_cast<PnrIndex>(localByLogicalNet_[logicalNet]);
}

std::optional<BoolVar>
SpatialLocalDispositionModel::localSelected(PnrIndex logicalNet) const {
  const auto local = localForLogicalNet(logicalNet);
  if (!local)
    return std::nullopt;
  return localSelected_[*local];
}

llvm::Expected<std::optional<PnrIndex>>
SpatialLocalDispositionModel::selectedOption(PnrIndex local,
                                             std::int64_t value) const {
  if (!problem_ || local >= logicalNets_.size() ||
      local >= legalValues_.size() || local >= externalValues_.size())
    return dispositionError("selected disposition is out of range");
  if (!llvm::is_contained(legalValues_[local], value))
    return dispositionError("selected disposition value is illegal");
  if (value == externalValues_[local])
    return std::optional<PnrIndex>{};
  if (value < 0)
    return dispositionError("selected local option is negative");
  const PnrIndex logicalNet = logicalNets_[local];
  const auto &domain = problem_->localTransfers().domains()[logicalNet];
  if (static_cast<std::uint64_t>(value) >= domain.optionCount)
    return dispositionError("selected local option exceeds its domain");
  return std::optional<PnrIndex>(domain.optionOffset +
                                 static_cast<PnrIndex>(value));
}
