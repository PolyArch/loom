#include "PnR/SpatialExactRepair.h"

#include "Common/MappingDebugLog.h"
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

SpatialExactRepairResult
result(SpatialExactRepairResultKind kind, std::uint64_t regionDecisions,
       std::uint64_t solverCalls = 0, std::uint64_t actionCount = 0,
       std::string detail = {}, std::uint64_t endpointExpansions = 0,
       std::uint64_t negotiationIterations = 0) {
  return {kind,
          regionDecisions,
          solverCalls,
          actionCount,
          endpointExpansions,
          negotiationIterations,
          std::move(detail)};
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
  if (candidate.atomicCapacityOveruse() == 0 &&
      (candidate.routeCapacityOveruse() != 0 ||
       candidate.tagResidentCapacityOveruse() != 0))
    return repairRouteCapacityOveruse(candidate, restartOrdinal);
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
      if (!bindings.relationIsConstraint(relation))
        continue;
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
  const auto executedResult = [&](SpatialExactRepairResultKind kind,
                                  std::string detail = {}) {
    return result(kind, *regionDecisionCount, solved->solverCalls,
                  actions_.size(), std::move(detail),
                  actionExecutor_.endpointExpansionCount(),
                  actionExecutor_.negotiationIterationCount());
  };
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
    return executedResult(kind, std::move(detail));
  }
  for (PnrIndex decision : decisions_) {
    const PnrIndex context =
        candidate.computeBinding(decision).instructionContext;
    if (context >= contextOveruse.size() || contextOveruse[context] != 0) {
      if (llvm::Error error = probe->discard())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      return executedResult(
          SpatialExactRepairResultKind::InternalError,
          "ActionBatch did not realize the exact capacity assignment");
    }
  }
  if (candidate.atomicCapacityOveruse() >= initialOveruse) {
    if (llvm::Error error = probe->discard())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
    return executedResult(SpatialExactRepairResultKind::InternalError,
                          "ActionBatch did not reduce CapacityOveruse");
  }
  if (llvm::Error error = probe->commit())
    return executedResult(SpatialExactRepairResultKind::InternalError,
                          llvm::toString(std::move(error)));
  return executedResult(SpatialExactRepairResultKind::Repaired);
}

llvm::Expected<SpatialExactRepairResult>
SpatialExactRepairScratch::repairRouteCapacityOveruse(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const ResolvedPnrExactRepairPolicy &policy =
      problem.config().policy().search.exactRepair;
  const detail::SpatialBindingRelationModel &bindings =
      problem.bindingRelations();
  const detail::InitializerRelationModel &relationModel = bindings.relations();
  const auto &routing = problem.routing();
  const auto &ports = problem.ports();
  const auto &transfers = problem.transfers();

  capacityWitnesses_.clear();
  const PnrIndex capacityCount =
      static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
  for (PnrIndex capacity = 0;
       capacity < problem.resources().capacityDimensions().size(); ++capacity)
    if (candidate.routeCapacityOveruseRaw(capacity) != 0)
      capacityWitnesses_.push_back(capacity);
  for (PnrIndex domain = 0;
       domain < routing.tagContinuity().matchDomains().size(); ++domain)
    if (candidate.tagDomainResidentCapacityOveruse(domain) != 0) {
      auto witness =
          checkedPnrIndexAdd({"SpatialExactRepair", "capacityWitnesses",
                              "Action", PnrCapacityMeasure::Index},
                             capacityCount, domain);
      if (!witness)
        return witness.takeError();
      capacityWitnesses_.push_back(*witness);
    }
  if (capacityWitnesses_.empty())
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "resident CapacityOveruse has no canonical witness");

  decisionIncluded_.assign(bindings.decisionCount(), 0);
  relationIncluded_.assign(relationModel.relations().size(), 0);
  netIncluded_.assign(transfers.logicalNets().size(), 0);
  decisionQueue_.clear();
  decisions_.clear();
  relations_.clear();
  affectedNets_.clear();

  const auto addDecision = [&](PnrIndex decision) -> llvm::Error {
    if (decision >= decisionIncluded_.size())
      return invocationError("route repair decision is out of range");
    if (!decisionIncluded_[decision]) {
      decisionIncluded_[decision] = 1;
      decisionQueue_.push_back(decision);
    }
    return llvm::Error::success();
  };
  const auto addNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= netIncluded_.size())
      return invocationError("route repair logical net is out of range");
    if (!netIncluded_[logicalNet]) {
      netIncluded_[logicalNet] = 1;
      affectedNets_.push_back(logicalNet);
    }
    return llvm::Error::success();
  };
  const auto addTerminal =
      [&](FrozenSpatialTerminalBinding binding) -> llvm::Error {
    switch (binding.kind) {
    case FrozenSpatialTerminalBindingKind::PortDemand:
      if (binding.index >= ports.portDemands().size())
        return invocationError("route repair PortDemand is out of range");
      return addDecision(bindings.portDecisionOffset() + binding.index);
    case FrozenSpatialTerminalBindingKind::GraphBoundary:
      if (binding.index >= ports.graphBoundaries().size())
        return invocationError("route repair graph boundary is out of range");
      return addDecision(bindings.graphBoundaryDecisionOffset() +
                         binding.index);
    }
    llvm_unreachable("unknown frozen terminal binding kind");
  };

  const auto capacityClaimOffsets = routing.capacityRouteClaimOffsets();
  const auto capacityClaims = routing.capacityRouteClaims();
  for (PnrIndex capacity : capacityWitnesses_) {
    if (capacity >= capacityCount) {
      const PnrIndex domain = capacity - capacityCount;
      if (domain >= routing.tagContinuity().matchDomains().size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "tag-table witness is out of range");
      for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
           ++logicalNet) {
        const auto values = candidate.tagValues(logicalNet);
        for (PnrIndex segment = 0; segment < values.size(); ++segment)
          if (llvm::is_contained(
                  candidate.tagSegmentDomains(logicalNet, segment), domain)) {
            if (llvm::Error error = addNet(logicalNet))
              return std::move(error);
            break;
          }
      }
      continue;
    }
    if (capacity + 1 >= capacityClaimOffsets.size())
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    "route witness has no capacity-to-claim incidence");
    const auto witnessClaims = capacityClaims.slice(
        capacityClaimOffsets[capacity],
        capacityClaimOffsets[capacity + 1] - capacityClaimOffsets[capacity]);
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const bool contributes = llvm::any_of(witnessClaims, [&](PnrIndex claim) {
        return claim < routing.routeClaims().size() &&
               candidate.logicalNetRouteClaimRefcount(logicalNet, claim) != 0;
      });
      if (contributes)
        if (llvm::Error error = addNet(logicalNet))
          return std::move(error);
    }
  }
  if (affectedNets_.empty())
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "route-capacity witness has no contributing logical net");

  std::size_t netCursor = 0;
  std::size_t decisionCursor = 0;
  while (netCursor != affectedNets_.size() ||
         decisionCursor != decisionQueue_.size()) {
    while (netCursor != affectedNets_.size()) {
      const PnrIndex logicalNet = affectedNets_[netCursor++];
      if (llvm::Error error =
              addTerminal(transfers.logicalNetSourceBindings()[logicalNet]))
        return std::move(error);
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      for (FrozenSpatialTerminalBinding sink :
           transfers.logicalNetSinkBindings().slice(net.sinkOffset,
                                                    net.sinkCount))
        if (llvm::Error error = addTerminal(sink))
          return std::move(error);
    }
    while (decisionCursor != decisionQueue_.size()) {
      const PnrIndex decision = decisionQueue_[decisionCursor++];
      if (decision < bindings.portDecisionOffset())
        return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                      decisionQueue_.size(), 0, 0,
                      "route repair escaped the fixed realization boundary");
      if (decision < bindings.graphBoundaryDecisionOffset()) {
        const PnrIndex demand = decision - bindings.portDecisionOffset();
        if (llvm::Error error = addNet(ports.portDemands()[demand].logicalNet))
          return std::move(error);
      } else {
        const PnrIndex boundary =
            decision - bindings.graphBoundaryDecisionOffset();
        if (boundary >= ports.graphBoundaries().size())
          return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "route repair boundary decision is out of range");
        if (llvm::Error error =
                addNet(ports.graphBoundaries()[boundary].logicalNet))
          return std::move(error);
      }
      for (PnrIndex relation : bindings.decisionRelations(decision)) {
        if (!bindings.relationIsConstraint(relation))
          continue;
        if (relation >= relationIncluded_.size())
          return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "route repair relation is out of range");
        if (!relationIncluded_[relation]) {
          relationIncluded_[relation] = 1;
          relations_.push_back(relation);
        }
        const detail::InitializerRelationRecord &record =
            relationModel.relations()[relation];
        for (const detail::InitializerRelationMember &member :
             relationModel.members(record))
          if (llvm::Error error = addDecision(member.decision))
            return std::move(error);
      }
    }
  }
  decisions_ = decisionQueue_;
  llvm::sort(decisions_);
  llvm::sort(relations_);
  llvm::sort(affectedNets_);

  if (affectedNets_.size() >
      std::numeric_limits<std::uint64_t>::max() - decisions_.size())
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "route repair region decision count overflows");
  const std::uint64_t regionDecisionCount =
      decisions_.size() + affectedNets_.size();
  if (regionDecisionCount > policy.maxRegionDecisions)
    return result(SpatialExactRepairResultKind::RegionTooLarge,
                  regionDecisionCount);
  if (decisions_.empty() ||
      decisions_.size() > static_cast<std::size_t>(INT_MAX))
    return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                  regionDecisionCount, 0, 0,
                  "route attachment decision domain is not CP-SAT encodable");

  CpModelBuilder model;
  std::vector<IntVar> variables;
  variables.reserve(decisions_.size());
  decisionVariables_.assign(bindings.decisionCount(), -1);
  legalValueOffsets_.clear();
  legalValueOffsets_.reserve(decisions_.size() + 1);
  legalValueOffsets_.push_back(0);
  legalValues_.clear();

  for (PnrIndex decision : decisions_) {
    const std::size_t begin = legalValues_.size();
    if (decision < bindings.graphBoundaryDecisionOffset()) {
      const PnrIndex demand = decision - bindings.portDecisionOffset();
      const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
      const PnrIndex placement =
          record.kind == FrozenSpatialPortDemandKind::Compute
              ? candidate.computeBinding(record.realization).placement
              : candidate.memoryBinding(record.realization).placement;
      const auto choices = bindings.portAttachmentChoices(demand);
      for (auto [ordinal, option] : llvm::enumerate(choices)) {
        if (option >= ports.attachmentOptions().size())
          return result(SpatialExactRepairResultKind::InternalError,
                        regionDecisionCount, 0, 0,
                        "route repair attachment option is out of range");
        const FrozenSpatialAttachmentOption &attachment =
            ports.attachmentOptions()[option];
        if (attachment.ownerKind !=
                FrozenSpatialAttachmentOwnerKind::PlacementDomain ||
            attachment.owner >= ports.placementDomains().size())
          return result(SpatialExactRepairResultKind::InternalError,
                        regionDecisionCount, 0, 0,
                        "route repair attachment owner is malformed");
        if (ports.placementDomains()[attachment.owner].placement == placement)
          legalValues_.push_back(static_cast<std::int64_t>(ordinal));
      }
    } else {
      const PnrIndex boundary =
          decision - bindings.graphBoundaryDecisionOffset();
      const auto choices = bindings.graphBoundaryAttachmentChoices(boundary);
      for (std::size_t ordinal = 0; ordinal < choices.size(); ++ordinal)
        legalValues_.push_back(static_cast<std::int64_t>(ordinal));
    }
    if (legalValues_.size() == begin)
      return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                    regionDecisionCount, 0, 0,
                    "route repair attachment has no fixed-boundary choice");
    if (legalValues_.size() > getPnrIndexMax())
      return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                    regionDecisionCount, 0, 0,
                    "route repair choice storage exceeds PnrIndex");
    legalValueOffsets_.push_back(static_cast<PnrIndex>(legalValues_.size()));
    const Domain domain = Domain::FromValues(
        llvm::ArrayRef(legalValues_).slice(begin, legalValues_.size() - begin));
    decisionVariables_[decision] = static_cast<int>(variables.size());
    variables.push_back(model.NewIntVar(domain));
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
                      regionDecisionCount, 0, 0,
                      "route repair relation escaped its closed region");
      const std::size_t local =
          static_cast<std::size_t>(decisionVariables_[member.decision]);
      elementValues_.clear();
      const auto legal =
          llvm::ArrayRef(legalValues_)
              .slice(legalValueOffsets_[local],
                     legalValueOffsets_[local + 1] - legalValueOffsets_[local]);
      elementValues_.reserve(legal.size());
      for (std::int64_t choice : legal)
        elementValues_.push_back(relationModel.projectedValue(member, choice));
      const IntVar projection =
          model.NewIntVar(Domain::FromValues(elementValues_));
      model.AddElement(variables[local], elementValues_, projection);
      projections.push_back(projection);
    }
    if (record.kind == detail::InitializerRelationKind::Equal) {
      for (std::size_t member = 1; member < projections.size(); ++member)
        model.AddEquality(projections[member], projections.front());
    } else {
      model.AddAllDifferent(projections);
    }
  }

  if (llvm::Error error = actionExecutor_.prepare(candidate))
    return result(SpatialExactRepairResultKind::InternalError,
                  regionDecisionCount, 0, 0, llvm::toString(std::move(error)));
  DeterministicPnrRandomStream exactRepairStream =
      DeterministicPnrRandomStream::create(
          problem.config().policy().determinism.masterSeed, restartOrdinal,
          PnrRandomStreamPurpose::ExactRepair);
  const std::int32_t solverSeed =
      detail::projectCpSatRandomSeed(exactRepairStream.nextU64());
  std::uint64_t solverCalls = 0;
  std::uint64_t assignmentOrdinal = 0;
  std::uint64_t lastActionCount = 0;

  const auto executedResult = [&](SpatialExactRepairResultKind kind,
                                  std::string detail = {}) {
    return result(kind, regionDecisionCount, solverCalls, lastActionCount,
                  std::move(detail), actionExecutor_.endpointExpansionCount(),
                  actionExecutor_.negotiationIterationCount());
  };

  while (solverCalls < policy.maxSolverCalls) {
    auto solved = detail::solveCanonicalCpSat(
        model.Build(), canonicalVariables, std::nullopt,
        policy.maxSolverCalls - solverCalls, solverSeed);
    if (!solved)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(solved.takeError()));
    solverCalls += solved->solverCalls;
    if (solved->kind == detail::CpSatCanonicalResultKind::Infeasible)
      return executedResult(
          SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
          "fixed-boundary attachment assignments were exhausted without an "
          "exact route closure");
    if (solved->kind ==
        detail::CpSatCanonicalResultKind::UnknownBudgetExhausted)
      return executedResult(
          SpatialExactRepairResultKind::UnknownBudgetExhausted,
          "route exact repair exhausted its solver-call budget");
    if (solved->assignment.size() != decisions_.size())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair assignment has the wrong size");

    actions_.clear();
    actions_.push_back(
        SpatialTransportRoutingAction{SpatialGlobalRoutingAction{}});
    for (auto [local, decision] : llvm::enumerate(decisions_)) {
      const std::int64_t selected = solved->assignment[local];
      if (selected < 0)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              "route repair selected a negative choice");
      if (decision < bindings.graphBoundaryDecisionOffset()) {
        const PnrIndex demand = decision - bindings.portDecisionOffset();
        const auto choices = bindings.portAttachmentChoices(demand);
        if (static_cast<std::size_t>(selected) >= choices.size())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                "route repair PortDemand choice is invalid");
        const PnrIndex option = choices[selected];
        if (candidate.portAttachment(demand) != option)
          actions_.push_back(SpatialResourceAllocationAction{
              SpatialPortAttachmentAction{demand, option}});
      } else {
        const PnrIndex boundary =
            decision - bindings.graphBoundaryDecisionOffset();
        const auto choices = bindings.graphBoundaryAttachmentChoices(boundary);
        if (static_cast<std::size_t>(selected) >= choices.size())
          return executedResult(
              SpatialExactRepairResultKind::InternalError,
              "route repair graph-boundary choice is invalid");
        const PnrIndex option = choices[selected];
        if (candidate.graphBoundaryAttachment(boundary) != option)
          actions_.push_back(SpatialResourceAllocationAction{
              SpatialGraphBoundaryAttachmentAction{boundary, option}});
      }
    }
    lastActionCount = actions_.size();
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ActionProposal,
        [&](llvm::json::Object &fields) {
          fields["search_scope"] = "route_exact_repair";
          fields["restart"] = restartOrdinal;
          fields["assignment"] = assignmentOrdinal;
          fields["capacity_ref"] = capacityWitnesses_.front();
          fields["capacity_witness_count"] = capacityWitnesses_.size();
          fields["region_decisions"] = regionDecisionCount;
          fields["action_count"] = lastActionCount;
          fields["solver_calls"] = solverCalls;
        });
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Detail,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ContextChoice,
        [&](llvm::json::Object &fields) {
          llvm::json::Array selectedTerminals;
          for (auto [local, decision] : llvm::enumerate(decisions_)) {
            const std::int64_t selected = solved->assignment[local];
            llvm::json::Object terminal;
            terminal["decision"] = decision;
            terminal["choice"] = selected;
            PnrIndex option = getInvalidPnrIndex();
            if (decision < bindings.graphBoundaryDecisionOffset()) {
              const PnrIndex demand = decision - bindings.portDecisionOffset();
              terminal["kind"] = "port_attachment";
              terminal["anchor"] = demand;
              option = bindings.portAttachmentChoices(demand)[selected];
            } else {
              const PnrIndex boundary =
                  decision - bindings.graphBoundaryDecisionOffset();
              terminal["kind"] = "graph_boundary_attachment";
              terminal["anchor"] = boundary;
              option =
                  bindings.graphBoundaryAttachmentChoices(boundary)[selected];
            }
            terminal["attachment_option"] = option;
            terminal["endpoint"] = ports.attachmentOptions()[option].endpoint;
            selectedTerminals.push_back(std::move(terminal));
          }
          fields["search_scope"] = "route_exact_repair";
          fields["restart"] = restartOrdinal;
          fields["assignment"] = assignmentOrdinal;
          fields["terminals"] = std::move(selectedTerminals);
        });

    auto probe = actionExecutor_.probeBatch(
        candidate, actions_, SpatialActionExecutionContext::FinalClosure);
    bool rejectAssignment = false;
    bool fixedTerminalCut = false;
    routeCutLogicalNets_.clear();
    if (!probe) {
      std::string detail;
      std::optional<SpatialActionTransitionFailureKind> transitionFailure;
      llvm::Error unhandled = llvm::handleErrors(
          probe.takeError(),
          [&](const SpatialPathFinderClosureFailure &failure) -> llvm::Error {
            if (failure.kind() != SpatialPathFinderClosureFailure::Kind::
                                      FixedTerminalCapacityCut) {
              std::string message;
              llvm::raw_string_ostream stream(message);
              failure.log(stream);
              return llvm::make_error<SpatialPathFinderClosureFailure>(
                  failure.kind(), stream.str(), failure.certificateCapacity(),
                  failure.mandatoryUsage(), failure.physicalCapacity(),
                  std::vector<PnrIndex>(failure.forcedLogicalNets().begin(),
                                        failure.forcedLogicalNets().end()));
            }
            fixedTerminalCut = true;
            routeCutLogicalNets_.assign(failure.forcedLogicalNets().begin(),
                                        failure.forcedLogicalNets().end());
            return llvm::Error::success();
          },
          [&](const SpatialActionTransitionFailure &failure) -> llvm::Error {
            llvm::raw_string_ostream stream(detail);
            failure.log(stream);
            transitionFailure = failure.kind();
            return llvm::Error::success();
          });
      if (unhandled)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(unhandled)));
      if (transitionFailure &&
          *transitionFailure == SpatialActionTransitionFailureKind::WorkLimit)
        return executedResult(
            SpatialExactRepairResultKind::UnknownBudgetExhausted,
            std::move(detail));
      rejectAssignment = true;
    } else if (candidate.atomicCapacityOveruse() != 0 ||
               candidate.routeCapacityOveruse() != 0 ||
               candidate.tagResidentCapacityOveruse() != 0 ||
               candidate.unroutedObligationCount() != 0) {
      rejectAssignment = true;
      if (llvm::Error error = probe->discard())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
    } else {
      if (llvm::Error error = probe->commit())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      if (llvm::Error error = candidate.verify())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      loom::mapping_debug::emit(loom::mapping_debug::Level::Decision,
                                loom::mapping_debug::Stage::SpatialPnr,
                                loom::mapping_debug::Event::ActionOutcome,
                                [&](llvm::json::Object &fields) {
                                  fields["search_scope"] = "route_exact_repair";
                                  fields["restart"] = restartOrdinal;
                                  fields["assignment"] = assignmentOrdinal;
                                  fields["accepted"] = true;
                                  fields["capacity_ref"] =
                                      capacityWitnesses_.front();
                                  fields["capacity_witness_count"] =
                                      capacityWitnesses_.size();
                                  fields["solver_calls"] = solverCalls;
                                });
      return executedResult(SpatialExactRepairResultKind::Repaired);
    }

    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ActionOutcome,
        [&](llvm::json::Object &fields) {
          fields["search_scope"] = "route_exact_repair";
          fields["restart"] = restartOrdinal;
          fields["assignment"] = assignmentOrdinal;
          fields["accepted"] = false;
          fields["capacity_ref"] = capacityWitnesses_.front();
          fields["capacity_witness_count"] = capacityWitnesses_.size();
          fields["solver_calls"] = solverCalls;
          fields["fixed_terminal_cut"] = fixedTerminalCut;
          fields["cut_logical_net_count"] = routeCutLogicalNets_.size();
        });
    if (!rejectAssignment)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair lost its assignment outcome");
    if (fixedTerminalCut) {
      routeCutDecisionLocals_.clear();
      const auto addCutTerminal =
          [&](FrozenSpatialTerminalBinding binding) -> llvm::Error {
        const PnrIndex decision =
            binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
                ? bindings.portDecisionOffset() + binding.index
                : bindings.graphBoundaryDecisionOffset() + binding.index;
        const auto found = llvm::lower_bound(decisions_, decision);
        if (found == decisions_.end() || *found != decision)
          return invocationError(
              "fixed-terminal cut escaped the exact repair region");
        routeCutDecisionLocals_.push_back(
            static_cast<PnrIndex>(found - decisions_.begin()));
        return llvm::Error::success();
      };
      for (PnrIndex logicalNet : routeCutLogicalNets_) {
        if (logicalNet >= transfers.logicalNets().size())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                "fixed-terminal cut net is out of range");
        if (llvm::Error error = addCutTerminal(
                transfers.logicalNetSourceBindings()[logicalNet]))
          return executedResult(
              SpatialExactRepairResultKind::UnsupportedEncoding,
              llvm::toString(std::move(error)));
        const FrozenSpatialLogicalNet &net =
            transfers.logicalNets()[logicalNet];
        for (FrozenSpatialTerminalBinding sink :
             transfers.logicalNetSinkBindings().slice(net.sinkOffset,
                                                      net.sinkCount))
          if (llvm::Error error = addCutTerminal(sink))
            return executedResult(
                SpatialExactRepairResultKind::UnsupportedEncoding,
                llvm::toString(std::move(error)));
      }
      llvm::sort(routeCutDecisionLocals_);
      routeCutDecisionLocals_.erase(std::unique(routeCutDecisionLocals_.begin(),
                                                routeCutDecisionLocals_.end()),
                                    routeCutDecisionLocals_.end());
      std::vector<IntVar> cutVariables;
      cutVariables.reserve(routeCutDecisionLocals_.size());
      elementValues_.clear();
      elementValues_.reserve(routeCutDecisionLocals_.size());
      for (PnrIndex local : routeCutDecisionLocals_) {
        cutVariables.push_back(variables[local]);
        elementValues_.push_back(solved->assignment[local]);
      }
      if (cutVariables.empty())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              "fixed-terminal cut has no terminal decision");
      model.AddForbiddenAssignments(cutVariables).AddTuple(elementValues_);
    } else {
      model.AddForbiddenAssignments(variables).AddTuple(solved->assignment);
    }
    ++assignmentOrdinal;
  }
  return executedResult(SpatialExactRepairResultKind::UnknownBudgetExhausted,
                        "route exact repair exhausted its solver-call budget");
}

std::size_t SpatialExactRepairScratch::retainedStorageBytes() const {
  return actionExecutor_.retainedStorageBytes() +
         retainedBytes(decisionIncluded_) + retainedBytes(relationIncluded_) +
         retainedBytes(netIncluded_) + retainedBytes(decisionQueue_) +
         retainedBytes(decisions_) + retainedBytes(relations_) +
         retainedBytes(affectedNets_) + retainedBytes(capacityWitnesses_) +
         retainedBytes(routeCutLogicalNets_) +
         retainedBytes(routeCutDecisionLocals_) +
         retainedBytes(decisionVariables_) + retainedBytes(legalValueOffsets_) +
         retainedBytes(legalValues_) + retainedBytes(elementValues_) +
         retainedBytes(actions_);
}
