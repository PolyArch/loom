#include "PnR/SpatialExactRepair.h"

#include "Common/MappingDebugLog.h"
#include "CpSatExactProtocol.h"
#include "PnR/MappingObjective.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialExactRepairInternal.h"
#include "SpatialExactRepairModel.h"
#include "SpatialFixedTerminalCutConstraint.h"
#include "SpatialLocalDispositionModel.h"
#include "SpatialProgressIndex.h"
#include "SpatialRouteConstraintModel.h"
#include "SpatialRuntimeCounterexampleRepairModel.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;
using namespace operations_research;
using namespace operations_research::sat;

using loom::pnr::detail::repairError;
using loom::pnr::detail::repairResult;

llvm::Error loom::pnr::detail::repairError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial exact-repair invocation: %s", detail.str().c_str());
}

SpatialExactRepairResult loom::pnr::detail::repairResult(
    SpatialExactRepairResultKind kind, std::uint64_t regionDecisions,
    std::uint64_t solverCalls, std::uint64_t actionCount, std::string detail,
    std::uint64_t endpointExpansions, std::uint64_t negotiationIterations,
    std::uint64_t logicalSolverCalls) {
  return {kind,
          regionDecisions,
          solverCalls,
          actionCount,
          endpointExpansions,
          negotiationIterations,
          std::move(detail),
          logicalSolverCalls};
}

namespace {

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

std::size_t retainedCertificateBytes(
    const std::vector<SpatialFixedTerminalCutCertificate> &certificates) {
  std::size_t bytes =
      certificates.capacity() * sizeof(SpatialFixedTerminalCutCertificate);
  for (const SpatialFixedTerminalCutCertificate &certificate : certificates)
    bytes += retainedBytes(certificate.forcedNetCuts);
  return bytes;
}

} // namespace

llvm::Error SpatialExactRepairScratch::planRegionDecision() {
  if (pendingRegionDecisionCount_ == std::numeric_limits<std::uint64_t>::max())
    return repairError("pending region decision count overflows");
  if (llvm::Error error =
          workLedger_.plan(SpatialPnrWorkKind::ExactRepairRegionDecision))
    return error;
  ++pendingRegionDecisionCount_;
  return llvm::Error::success();
}

llvm::Error SpatialExactRepairScratch::consumePendingRegionDecisions() {
  if (pendingRegionDecisionCount_ >
      std::numeric_limits<std::uint64_t>::max() - accountedRegionDecisionCount_)
    return repairError("accounted region decision count overflows");
  if (llvm::Error error =
          workLedger_.consume(SpatialPnrWorkKind::ExactRepairRegionDecision,
                              pendingRegionDecisionCount_))
    return error;
  accountedRegionDecisionCount_ += pendingRegionDecisionCount_;
  pendingRegionDecisionCount_ = 0;
  return llvm::Error::success();
}

llvm::Expected<SpatialExactRepairResult> SpatialExactRepairScratch::repair(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
    std::uint64_t solverCallLimit,
    DeterministicPnrRandomStream &exactRepairStream,
    SpatialPnrWorkLedgerView workLedger,
    std::optional<PnrIndex> runtimeCounterexampleClause,
    ExecutionControlView executionControl) {
  workLedger_ = workLedger;
  executionControl_ = executionControl;
  accountedRegionDecisionCount_ = 0;
  pendingRegionDecisionCount_ = 0;
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  accountedRegionDecisions_.assign(problem.bindingRelations().decisionCount(),
                                   0);
  accountedRegionNets_.assign(problem.transfers().logicalNets().size(), 0);
  const ResolvedPnrExactRepairPolicy &policy =
      problem.config().policy().search.exactRepair;
  if (policy.kind != ResolvedPnrExactRepairKind::CpSat)
    return repairError("CpSat_3_0 is not selected by SearchPolicy");
  if (solverCallLimit == 0 || solverCallLimit > policy.maxSolverCalls)
    return repairError("solver-call limit exceeds SearchPolicy");
  if (executionControl_.stopRequested())
    return repairResult(SpatialExactRepairResultKind::TimedOut, 0, 0, 0,
                        "exact repair started after its execution deadline");
  if (runtimeCounterexampleClause) {
    if (*runtimeCounterexampleClause >=
            problem.constraints().resolvedNoGoods().size() ||
        !candidate.runtimeCounterexampleClauseViolated(
            *runtimeCounterexampleClause))
      return repairResult(
          SpatialExactRepairResultKind::ProofNotEstablished, 0, 0, 0,
          "requested runtime-counterexample clause is not live");
    if (candidate.atomicCapacityOveruse() != 0)
      return repairResult(
          SpatialExactRepairResultKind::ProofNotEstablished, 0, 0, 0,
          "runtime-counterexample repair parent has an unrelated atomic "
          "capacity violation");
    return repairTransportClosure(candidate, restartOrdinal, solverCallLimit,
                                  exactRepairStream,
                                  runtimeCounterexampleClause);
  }
  if (candidate.progressProofDebtWitnessCount() != 0 &&
      candidate.hardProgressViolation() == 0 &&
      candidate.unroutedObligationCount() == 0 &&
      candidate.routeCapacityOveruse() == 0 &&
      candidate.tagResidentCapacityOveruse() == 0 &&
      candidate.tagUnassignedCount() == 0 && candidate.tagConflictCount() == 0)
    return repairResult(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "capacity proof-debt repair requires an owner-local disjunctive "
        "route or hardware proposal");
  if (candidate.atomicCapacityOveruse() == 0 &&
      candidate.hasTransportClosureViolation())
    return repairTransportClosure(candidate, restartOrdinal, solverCallLimit,
                                  exactRepairStream, std::nullopt);
  if (candidate.atomicCapacityOveruse() == 0)
    return repairResult(SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0,
                        0, "candidate has no supported exact-repair witness");

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
      return repairResult(
          SpatialExactRepairResultKind::InternalError, 0, 0, 0,
          "selected compute context has no capacity projection");
    if (contextOveruse[context] != 0) {
      witness = decision;
      break;
    }
  }
  if (!witness)
    return repairResult(SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0,
                        0,
                        "atomic-capacity repair does not encode a non-compute "
                        "binding witness");

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
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, "binding relation reverse index is invalid");
      if (relationIncluded_[relation])
        continue;
      relationIncluded_[relation] = 1;
      relations_.push_back(relation);
      const detail::InitializerRelationRecord &record =
          relationModel.relations()[relation];
      for (const detail::InitializerRelationMember &member :
           relationModel.members(record)) {
        if (member.decision >= decisionIncluded_.size())
          return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                              0, "binding relation member is out of range");
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
    return repairResult(
        SpatialExactRepairResultKind::UnsupportedEncoding, decisions_.size(), 0,
        0,
        "atomic-capacity repair region includes a memory binding "
        "decision");
  netIncluded_.assign(problem.transfers().logicalNets().size(), 0);
  affectedNets_.clear();
  const auto demandOffsets = problem.ports().computeRealizationDemandOffsets();
  const auto demands = problem.ports().computeRealizationDemands();
  if (demandOffsets.size() != computeCount + 1)
    return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "compute-demand reverse index is incomplete");
  for (PnrIndex decision : decisions_) {
    if (decision >= computeCount)
      continue;
    for (PnrIndex demand :
         demands.slice(demandOffsets[decision],
                       demandOffsets[decision + 1] - demandOffsets[decision])) {
      if (demand >= problem.ports().portDemands().size())
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, "repair region contains an invalid PortDemand");
      const PnrIndex net = problem.ports().portDemands()[demand].logicalNet;
      if (net >= netIncluded_.size())
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, "repair region contains an invalid logical net");
      if (!netIncluded_[net]) {
        netIncluded_[net] = 1;
        affectedNets_.push_back(net);
      }
    }
  }
  llvm::sort(affectedNets_);
  auto regionDecisionCount = detail::countExactRepairRegionDecisions(
      decisions_, affectedNets_, problem);
  if (!regionDecisionCount)
    return regionDecisionCount.takeError();
  if (*regionDecisionCount > policy.maxRegionDecisions)
    return repairResult(SpatialExactRepairResultKind::RegionTooLarge,
                        *regionDecisionCount, 0, 0,
                        (llvm::Twine("exact-repair region has ") +
                         llvm::Twine(*regionDecisionCount) +
                         " decisions, exceeding policy limit " +
                         llvm::Twine(policy.maxRegionDecisions))
                            .str());
  auto modelAdmission =
      detail::admitAtomicExactRepairModel(bindings, decisions_, contextOveruse);
  if (!modelAdmission)
    return modelAdmission.takeError();
  if (*modelAdmission)
    return repairResult(SpatialExactRepairResultKind::UnsupportedEncoding,
                        *regionDecisionCount, 0, 0,
                        std::move(**modelAdmission));
  CpModelBuilder model;
  std::vector<IntVar> variables;
  variables.reserve(decisions_.size());
  decisionVariables_.assign(bindings.decisionCount(), -1);
  legalValueOffsets_.clear();
  legalValueOffsets_.reserve(decisions_.size() + 1);
  legalValueOffsets_.push_back(0);
  legalValues_.clear();
  for (PnrIndex decision : decisions_) {
    if (llvm::Error error = planRegionDecision())
      return error;
    const auto choices = bindings.computeChoices(decision);
    const IntVar variable = model.NewIntVar(
        Domain(0, static_cast<std::int64_t>(choices.size() - 1)));
    decisionVariables_[decision] = static_cast<int>(variables.size());
    variables.push_back(variable);
    elementValues_.clear();
    elementValues_.reserve(choices.size());
    for (auto [ordinal, choice] : llvm::enumerate(choices)) {
      legalValues_.push_back(static_cast<std::int64_t>(ordinal));
      elementValues_.push_back(
          contextOveruse[choice.instructionContext] == 0 ? 0 : 1);
    }
    legalValueOffsets_.push_back(static_cast<PnrIndex>(legalValues_.size()));
    const IntVar localOveruse = model.NewIntVar(Domain(0, 1));
    model.AddElement(variable, elementValues_, localOveruse);
    model.AddEquality(localOveruse, 0);
    for (PnrIndex demand = demandOffsets[decision];
         demand != demandOffsets[decision + 1]; ++demand)
      if (llvm::Error error = planRegionDecision())
        return error;
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
        return repairResult(
            SpatialExactRepairResultKind::InternalError, *regionDecisionCount,
            0, 0, "repair relation escaped its closed decision region");
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
    detail::addExactRepairInitializerRelationConstraint(model, relationModel,
                                                        record, projections);
  }

  auto mutationCount = detail::addExactRepairMutationCountObjective(
      model, variables, candidate, bindings, decisions_);
  if (!mutationCount)
    return mutationCount.takeError();
  for (PnrIndex logicalNet : affectedNets_) {
    (void)logicalNet;
    if (llvm::Error error = planRegionDecision())
      return error;
  }
  if (pendingRegionDecisionCount_ != *regionDecisionCount)
    return repairResult(
        SpatialExactRepairResultKind::InternalError, *regionDecisionCount, 0, 0,
        "exact-repair region work disagrees with its closed model");
  if (llvm::Error error = consumePendingRegionDecisions())
    return error;

  const std::int32_t solverSeed =
      detail::projectCpSatRandomSeed(exactRepairStream.nextU64());
  auto solved = detail::solveCanonicalCpSat(
      model.Build(), canonicalVariables, mutationCount->index(),
      solverCallLimit, solverSeed, workLedger_);
  if (!solved)
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        *regionDecisionCount, 0, 0,
                        llvm::toString(solved.takeError()));
  if (solved->kind == detail::CpSatCanonicalResultKind::Infeasible)
    return repairResult(
        SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
        *regionDecisionCount, solved->solverCalls);
  if (solved->kind == detail::CpSatCanonicalResultKind::UnknownBudgetExhausted)
    return repairResult(SpatialExactRepairResultKind::UnknownBudgetExhausted,
                        *regionDecisionCount, solved->solverCalls);
  if (solved->assignment.size() != decisions_.size())
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        *regionDecisionCount, solved->solverCalls, 0,
                        "canonical solver assignment has the wrong size");

  actions_.clear();
  netIncluded_.assign(netIncluded_.size(), 0);
  for (auto [local, decision] : llvm::enumerate(decisions_)) {
    const std::int64_t selected = solved->assignment[local];
    const auto choices = bindings.computeChoices(decision);
    if (selected < 0 || static_cast<std::size_t>(selected) >= choices.size())
      return repairResult(SpatialExactRepairResultKind::InternalError,
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
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        *regionDecisionCount, solved->solverCalls, 0,
                        "exact repair produced an empty ActionBatch");

  if (llvm::Error error =
          actionExecutor_.prepare(candidate, workLedger_, executionControl_))
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        *regionDecisionCount, solved->solverCalls,
                        actions_.size(), llvm::toString(std::move(error)));
  const auto executedResult = [&](SpatialExactRepairResultKind kind,
                                  std::string detail = {}) {
    return repairResult(kind, *regionDecisionCount, solved->solverCalls,
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
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
    SpatialExactRepairResultKind kind =
        SpatialExactRepairResultKind::InternalError;
    if (transitionFailure) {
      switch (*transitionFailure) {
      case SpatialActionTransitionFailureKind::WorkLimit:
        kind = SpatialExactRepairResultKind::UnknownBudgetExhausted;
        break;
      case SpatialActionTransitionFailureKind::Interrupted:
        kind = SpatialExactRepairResultKind::TimedOut;
        break;
      case SpatialActionTransitionFailureKind::IntrinsicInvalid:
        kind = SpatialExactRepairResultKind::UnsupportedEncoding;
        break;
      }
    }
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

std::size_t SpatialExactRepairScratch::retainedStorageBytes() const {
  return actionExecutor_.retainedStorageBytes() +
         retainedBytes(decisionIncluded_) + retainedBytes(relationIncluded_) +
         retainedBytes(netIncluded_) + retainedBytes(decisionQueue_) +
         retainedBytes(decisions_) + retainedBytes(relations_) +
         retainedBytes(affectedNets_) +
         retainedBytes(accountedRegionDecisions_) +
         retainedBytes(accountedRegionNets_) +
         retainedBytes(routeCutCertificate_.forcedNetCuts) +
         retainedCertificateBytes(learnedCutCertificates_) +
         retainedBytes(routeCutBlockedTraversals_) +
         retainedBytes(routeCutReachableEndpoints_) +
         retainedBytes(routeCutWorklist_) + retainedBytes(decisionVariables_) +
         retainedBytes(legalValueOffsets_) + retainedBytes(legalValues_) +
         retainedBytes(elementValues_) + retainedBytes(actions_);
}
