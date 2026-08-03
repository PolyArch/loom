#include "PnR/SpatialExactRepair.h"

#include "CpSatExactProtocol.h"
#include "SpatialBindingRelationModel.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>

using namespace loom;
using namespace loom::pnr;
using namespace operations_research;
using namespace operations_research::sat;

namespace {

llvm::Error invocationError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial exact-repair invocation: %s", detail.str().c_str());
}

SpatialExactRepairResult result(SpatialExactRepairResultKind kind,
                                std::uint64_t regionDecisions,
                                std::uint64_t solverCalls = 0,
                                std::uint64_t actionCount = 0,
                                std::string detail = {}) {
  return {kind, regionDecisions, solverCalls, actionCount, std::move(detail)};
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Expected<std::uint64_t>
countRegionDecisions(llvm::ArrayRef<PnrIndex> decisions,
                     llvm::ArrayRef<PnrIndex> affectedNets,
                     const FrozenSpatialPnrProblem &problem) {
  std::uint64_t count = decisions.size();
  const auto offsets = problem.ports().computeRealizationDemandOffsets();
  if (offsets.size() != problem.realizations().computeRealizations().size() + 1)
    return invocationError("compute-demand reverse index is incomplete");
  for (PnrIndex decision : decisions) {
    const std::uint64_t demandCount = offsets[decision + 1] - offsets[decision];
    if (demandCount > std::numeric_limits<std::uint64_t>::max() - count)
      return invocationError("repair region decision count overflows");
    count += demandCount;
  }
  if (affectedNets.size() > std::numeric_limits<std::uint64_t>::max() - count)
    return invocationError("repair region decision count overflows");
  return count + affectedNets.size();
}

} // namespace

llvm::Expected<SpatialExactRepairResult>
SpatialExactRepairScratch::repairCapacityOveruse(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const ResolvedPnrExactRepairPolicy &policy =
      problem.config().policy().search.exactRepair;
  if (policy.kind != ResolvedPnrExactRepairKind::CpSat)
    return invocationError("CpSat_1_0 is not selected by SearchPolicy");
  if (candidate.atomicCapacityOveruse() == 0)
    return result(SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
                  "candidate has no CapacityOveruse witness");

  const detail::SpatialBindingRelationModel &bindings =
      problem.bindingRelations();
  const detail::InitializerRelationModel &relationModel = bindings.relations();
  const PnrIndex computeCount = bindings.computeDecisionCount();
  std::optional<PnrIndex> witness;
  const auto contextOveruse =
      problem.capacity().computeInstructionContextOveruse();
  for (PnrIndex decision = 0; decision < computeCount; ++decision) {
    const PnrIndex context =
        candidate.computeBinding(decision).instructionContext;
    if (context >= contextOveruse.size())
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    "selected compute context has no capacity projection");
    if (contextOveruse[context] != 0) {
      witness = decision;
      break;
    }
  }
  if (!witness)
    return result(SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
                  "first CapacityOveruse witness is not a compute binding");

  decisionIncluded_.assign(bindings.decisionCount(), 0);
  relationIncluded_.assign(relationModel.relations().size(), 0);
  decisionQueue_.clear();
  decisions_.clear();
  relations_.clear();
  decisionIncluded_[*witness] = 1;
  decisionQueue_.push_back(*witness);
  for (std::size_t cursor = 0; cursor < decisionQueue_.size(); ++cursor) {
    const PnrIndex decision = decisionQueue_[cursor];
    for (PnrIndex relation : bindings.decisionRelations(decision)) {
      if (relation >= relationIncluded_.size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "binding relation reverse index is invalid");
      if (relationIncluded_[relation])
        continue;
      relationIncluded_[relation] = 1;
      relations_.push_back(relation);
      const detail::InitializerRelationRecord &record =
          relationModel.relations()[relation];
      for (const detail::InitializerRelationMember &member :
           relationModel.members(record)) {
        if (member.decision >= decisionIncluded_.size())
          return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "binding relation member is out of range");
        if (!decisionIncluded_[member.decision]) {
          decisionIncluded_[member.decision] = 1;
          decisionQueue_.push_back(member.decision);
        }
      }
    }
  }
  decisions_ = decisionQueue_;
  llvm::sort(decisions_);
  llvm::sort(relations_);

  if (llvm::any_of(decisions_,
                   [&](PnrIndex decision) { return decision >= computeCount; }))
    return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                  decisions_.size(), 0, 0,
                  "capacity region includes a memory binding decision");
  if (bindings.deferredProjection())
    return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                  decisions_.size(), 0, 0,
                  "capacity region has an unencoded constraint projection");

  netIncluded_.assign(problem.transfers().logicalNets().size(), 0);
  affectedNets_.clear();
  const auto demandOffsets = problem.ports().computeRealizationDemandOffsets();
  const auto demands = problem.ports().computeRealizationDemands();
  if (demandOffsets.size() != computeCount + 1)
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "compute-demand reverse index is incomplete");
  for (PnrIndex decision : decisions_) {
    if (decision >= computeCount)
      continue;
    for (PnrIndex demand :
         demands.slice(demandOffsets[decision],
                       demandOffsets[decision + 1] - demandOffsets[decision])) {
      if (demand >= problem.ports().portDemands().size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "repair region contains an invalid PortDemand");
      const PnrIndex net = problem.ports().portDemands()[demand].logicalNet;
      if (net >= netIncluded_.size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "repair region contains an invalid logical net");
      if (!netIncluded_[net]) {
        netIncluded_[net] = 1;
        affectedNets_.push_back(net);
      }
    }
  }
  llvm::sort(affectedNets_);
  auto regionDecisionCount =
      countRegionDecisions(decisions_, affectedNets_, problem);
  if (!regionDecisionCount)
    return regionDecisionCount.takeError();
  if (*regionDecisionCount > policy.maxRegionDecisions)
    return result(SpatialExactRepairResultKind::RegionTooLarge,
                  *regionDecisionCount);
  if (decisions_.size() > static_cast<std::size_t>(INT_MAX))
    return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                  *regionDecisionCount, 0, 0,
                  "compute decision domain is not CP-SAT encodable");

  CpModelBuilder model;
  std::vector<IntVar> variables;
  variables.reserve(decisions_.size());
  decisionVariables_.assign(bindings.decisionCount(), -1);
  legalValueOffsets_.clear();
  legalValueOffsets_.reserve(decisions_.size() + 1);
  legalValueOffsets_.push_back(0);
  legalValues_.clear();
  for (PnrIndex decision : decisions_) {
    const auto choices = bindings.computeChoices(decision);
    if (choices.empty() || choices.size() > static_cast<std::size_t>(INT64_MAX))
      return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                    *regionDecisionCount, 0, 0,
                    "compute choice domain is not CP-SAT encodable");
    const IntVar variable = model.NewIntVar(
        Domain(0, static_cast<std::int64_t>(choices.size() - 1)));
    decisionVariables_[decision] = static_cast<int>(variables.size());
    variables.push_back(variable);
    elementValues_.clear();
    elementValues_.reserve(choices.size());
    for (auto [ordinal, choice] : llvm::enumerate(choices)) {
      legalValues_.push_back(static_cast<std::int64_t>(ordinal));
      if (choice.instructionContext >= contextOveruse.size())
        return result(SpatialExactRepairResultKind::InternalError,
                      *regionDecisionCount, 0, 0,
                      "compute choice has no capacity projection");
      elementValues_.push_back(
          contextOveruse[choice.instructionContext] == 0 ? 0 : 1);
    }
    if (legalValues_.size() > getPnrIndexMax())
      return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                    *regionDecisionCount, 0, 0,
                    "flattened compute choice domain exceeds PnrIndex");
    legalValueOffsets_.push_back(static_cast<PnrIndex>(legalValues_.size()));
    const IntVar localOveruse = model.NewIntVar(Domain(0, 1));
    model.AddElement(variable, elementValues_, localOveruse);
    model.AddEquality(localOveruse, 0);
  }
  std::vector<detail::CpSatCanonicalVariable> canonicalVariables;
  canonicalVariables.reserve(decisions_.size());
  for (std::size_t local = 0; local < decisions_.size(); ++local)
    canonicalVariables.push_back(
        {variables[local].index(),
         llvm::ArrayRef(legalValues_)
             .slice(legalValueOffsets_[local], legalValueOffsets_[local + 1] -
                                                   legalValueOffsets_[local])});

  for (PnrIndex relation : relations_) {
    const detail::InitializerRelationRecord &record =
        relationModel.relations()[relation];
    std::vector<IntVar> projections;
    projections.reserve(record.memberCount);
    for (const detail::InitializerRelationMember &member :
         relationModel.members(record)) {
      if (member.decision >= decisionVariables_.size() ||
          decisionVariables_[member.decision] < 0)
        return result(SpatialExactRepairResultKind::InternalError,
                      *regionDecisionCount, 0, 0,
                      "repair relation escaped its closed decision region");
      const auto offsets = relationModel.decisionChoiceOffsets();
      const PnrIndex choiceCount =
          offsets[member.decision + 1] - offsets[member.decision];
      elementValues_.clear();
      elementValues_.reserve(choiceCount);
      for (PnrIndex choice = 0; choice < choiceCount; ++choice)
        elementValues_.push_back(relationModel.projectedValue(member, choice));
      const IntVar projection =
          model.NewIntVar(Domain::FromValues(elementValues_));
      model.AddElement(variables[decisionVariables_[member.decision]],
                       elementValues_, projection);
      projections.push_back(projection);
    }
    if (record.kind == detail::InitializerRelationKind::Equal) {
      for (std::size_t member = 1; member < projections.size(); ++member)
        model.AddEquality(projections[member], projections.front());
    } else {
      model.AddAllDifferent(projections);
    }
  }

  DeterministicPnrRandomStream exactRepairStream =
      DeterministicPnrRandomStream::create(
          problem.config().policy().determinism.masterSeed, restartOrdinal,
          PnrRandomStreamPurpose::ExactRepair);
  const std::int32_t solverSeed =
      detail::projectCpSatRandomSeed(exactRepairStream.nextU64());
  auto solved = detail::solveCanonicalCpSat(model.Build(), canonicalVariables,
                                            std::nullopt, policy.maxSolverCalls,
                                            solverSeed);
  if (!solved)
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, 0, 0,
                  llvm::toString(solved.takeError()));
  if (solved->kind == detail::CpSatCanonicalResultKind::Infeasible)
    return result(
        SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
        *regionDecisionCount, solved->solverCalls);
  if (solved->kind == detail::CpSatCanonicalResultKind::UnknownBudgetExhausted)
    return result(SpatialExactRepairResultKind::UnknownBudgetExhausted,
                  *regionDecisionCount, solved->solverCalls);
  if (solved->assignment.size() != decisions_.size())
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, solved->solverCalls, 0,
                  "canonical solver assignment has the wrong size");

  actions_.clear();
  netIncluded_.assign(netIncluded_.size(), 0);
  for (auto [local, decision] : llvm::enumerate(decisions_)) {
    const std::int64_t selected = solved->assignment[local];
    const auto choices = bindings.computeChoices(decision);
    if (selected < 0 || static_cast<std::size_t>(selected) >= choices.size())
      return result(SpatialExactRepairResultKind::InternalError,
                    *regionDecisionCount, solved->solverCalls, 0,
                    "canonical solver assignment escaped its domain");
    const detail::SpatialComputeBindingChoice &choice = choices[selected];
    const SpatialComputeBindingSelection &current =
        candidate.computeBinding(decision);
    if (current.placement == choice.placement &&
        current.instructionContext == choice.instructionContext)
      continue;
    actions_.push_back(
        SpatialRealizationBindingAction{SpatialComputeBindingAction{
            decision, choice.placement, choice.instructionContext}});
    if (current.placement != choice.placement) {
      for (PnrIndex demand : demands.slice(demandOffsets[decision],
                                           demandOffsets[decision + 1] -
                                               demandOffsets[decision])) {
        const PnrIndex net = problem.ports().portDemands()[demand].logicalNet;
        netIncluded_[net] = 1;
      }
    }
  }
  for (PnrIndex net = 0; net < netIncluded_.size(); ++net)
    if (netIncluded_[net])
      actions_.push_back(
          SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{net}});
  if (actions_.empty())
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, solved->solverCalls, 0,
                  "exact repair produced an empty ActionBatch");

  if (llvm::Error error = actionExecutor_.prepare(candidate))
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, solved->solverCalls, actions_.size(),
                  llvm::toString(std::move(error)));
  const std::uint64_t initialOveruse = candidate.atomicCapacityOveruse();
  auto probe = actionExecutor_.probeBatch(candidate, actions_);
  if (!probe) {
    std::string detail;
    std::optional<SpatialActionTransitionFailureKind> transitionFailure;
    llvm::Error error = llvm::handleErrors(
        probe.takeError(),
        [&](const SpatialActionTransitionFailure &failure) -> llvm::Error {
          llvm::raw_string_ostream stream(detail);
          failure.log(stream);
          transitionFailure = failure.kind();
          return llvm::Error::success();
        });
    if (error)
      detail = llvm::toString(std::move(error));
    SpatialExactRepairResultKind kind =
        SpatialExactRepairResultKind::InternalError;
    if (transitionFailure)
      kind = *transitionFailure == SpatialActionTransitionFailureKind::WorkLimit
                 ? SpatialExactRepairResultKind::UnknownBudgetExhausted
                 : SpatialExactRepairResultKind::UnsupportedEncoding;
    return result(kind, *regionDecisionCount, solved->solverCalls,
                  actions_.size(), std::move(detail));
  }
  for (PnrIndex decision : decisions_) {
    const PnrIndex context =
        candidate.computeBinding(decision).instructionContext;
    if (context >= contextOveruse.size() || contextOveruse[context] != 0) {
      if (llvm::Error error = probe->discard())
        return result(SpatialExactRepairResultKind::InternalError,
                      *regionDecisionCount, solved->solverCalls,
                      actions_.size(), llvm::toString(std::move(error)));
      return result(
          SpatialExactRepairResultKind::InternalError, *regionDecisionCount,
          solved->solverCalls, actions_.size(),
          "ActionBatch did not realize the exact capacity assignment");
    }
  }
  if (candidate.atomicCapacityOveruse() >= initialOveruse) {
    if (llvm::Error error = probe->discard())
      return result(SpatialExactRepairResultKind::InternalError,
                    *regionDecisionCount, solved->solverCalls, actions_.size(),
                    llvm::toString(std::move(error)));
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, solved->solverCalls, actions_.size(),
                  "ActionBatch did not reduce CapacityOveruse");
  }
  if (llvm::Error error = probe->commit())
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, solved->solverCalls, actions_.size(),
                  llvm::toString(std::move(error)));
  return result(SpatialExactRepairResultKind::Repaired, *regionDecisionCount,
                solved->solverCalls, actions_.size());
}

std::size_t SpatialExactRepairScratch::retainedStorageBytes() const {
  return actionExecutor_.retainedStorageBytes() +
         retainedBytes(decisionIncluded_) + retainedBytes(relationIncluded_) +
         retainedBytes(netIncluded_) + retainedBytes(decisionQueue_) +
         retainedBytes(decisions_) + retainedBytes(relations_) +
         retainedBytes(affectedNets_) + retainedBytes(decisionVariables_) +
         retainedBytes(legalValueOffsets_) + retainedBytes(legalValues_) +
         retainedBytes(elementValues_) + retainedBytes(actions_);
}
