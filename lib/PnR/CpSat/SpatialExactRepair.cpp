#include "PnR/SpatialExactRepair.h"

#include "Common/MappingDebugLog.h"
#include "CpSatExactProtocol.h"
#include "PnR/MappingObjective.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialExactRepairModel.h"
#include "SpatialFixedTerminalCutConstraint.h"
#include "SpatialLocalDispositionModel.h"
#include "SpatialProgressIndex.h"
#include "SpatialRouteConstraintModel.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

bool containsOrdinal(PnrIndex offset, PnrIndex count, PnrIndex ordinal) {
  return ordinal >= offset && ordinal - offset < count;
}

bool cutNetLess(const SpatialFixedTerminalCutNet &left,
                const SpatialFixedTerminalCutNet &right) {
  return std::tie(left.logicalNet, left.unreachableSink) <
         std::tie(right.logicalNet, right.unreachableSink);
}

bool cutCertificateLess(const SpatialFixedTerminalCutCertificate &left,
                        const SpatialFixedTerminalCutCertificate &right) {
  if (left.capacity != right.capacity)
    return left.capacity < right.capacity;
  return std::lexicographical_compare(
      left.forcedNetCuts.begin(), left.forcedNetCuts.end(),
      right.forcedNetCuts.begin(), right.forcedNetCuts.end(), cutNetLess);
}

bool cutCertificateEqual(const SpatialFixedTerminalCutCertificate &left,
                         const SpatialFixedTerminalCutCertificate &right) {
  return !cutCertificateLess(left, right) && !cutCertificateLess(right, left);
}

bool insertCutCertificate(
    std::vector<SpatialFixedTerminalCutCertificate> &certificates,
    SpatialFixedTerminalCutCertificate certificate) {
  llvm::sort(certificate.forcedNetCuts, cutNetLess);
  certificate.forcedNetCuts.erase(
      std::unique(certificate.forcedNetCuts.begin(),
                  certificate.forcedNetCuts.end(),
                  [](const SpatialFixedTerminalCutNet &left,
                     const SpatialFixedTerminalCutNet &right) {
                    return !cutNetLess(left, right) && !cutNetLess(right, left);
                  }),
      certificate.forcedNetCuts.end());
  const auto found =
      llvm::lower_bound(certificates, certificate, cutCertificateLess);
  if (found != certificates.end() && cutCertificateEqual(*found, certificate))
    return false;
  certificates.insert(found, std::move(certificate));
  return true;
}

std::size_t retainedCertificateBytes(
    const std::vector<SpatialFixedTerminalCutCertificate> &certificates) {
  std::size_t bytes =
      certificates.capacity() * sizeof(SpatialFixedTerminalCutCertificate);
  for (const SpatialFixedTerminalCutCertificate &certificate : certificates)
    bytes += retainedBytes(certificate.forcedNetCuts);
  return bytes;
}

struct TransportWitness final {
  ResolvedPnrViolationKind kind;
  PnrIndex ordinal;
};

llvm::Expected<std::optional<TransportWitness>>
firstTransportWitness(const SpatialCandidateState &candidate) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &transfers = problem.transfers();
  const auto &routing = problem.routing();
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet)
    if (!candidate.usesRegisterFifo(logicalNet) &&
        !candidate.routeTree(logicalNet).isRouted())
      return TransportWitness{ResolvedPnrViolationKind::UnroutedObligation,
                              transfers.logicalNets()[logicalNet].sinkOffset};

  const PnrIndex capacityCount =
      static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
  for (PnrIndex capacity = 0;
       capacity < problem.resources().capacityDimensions().size(); ++capacity)
    if (candidate.routeCapacityOveruseRaw(capacity) != 0)
      return TransportWitness{ResolvedPnrViolationKind::CapacityOveruse,
                              capacity};
  for (PnrIndex domain = 0;
       domain < routing.tagContinuity().matchDomains().size(); ++domain) {
    if (candidate.tagDomainResidentCapacityOveruse(domain) == 0)
      continue;
    auto ordinal = checkedPnrIndexAdd({"SpatialExactRepair", "transportWitness",
                                       "Action", PnrCapacityMeasure::Index},
                                      capacityCount, domain);
    if (!ordinal)
      return ordinal.takeError();
    return TransportWitness{ResolvedPnrViolationKind::CapacityOveruse,
                            *ordinal};
  }

  PnrIndex globalSegment = 0;
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet)
    for (const auto &value : candidate.tagValues(logicalNet)) {
      if (!value)
        return TransportWitness{ResolvedPnrViolationKind::TagUnassigned,
                                globalSegment};
      if (globalSegment == getPnrIndexMax())
        return invocationError("Physical Tag segment ordinal overflows");
      ++globalSegment;
    }
  for (PnrIndex domain = 0;
       domain < routing.tagContinuity().matchDomains().size(); ++domain)
    if (candidate.tagDomainConflictCount(domain) != 0)
      return TransportWitness{ResolvedPnrViolationKind::TagConflict, domain};

  if (const auto owner = candidate.progress().firstCapacityShortfallOwner())
    return TransportWitness{ResolvedPnrViolationKind::HardProgressViolation,
                            *owner};
  if (const auto owner = candidate.progress().firstCapacityProofDebtOwner())
    return TransportWitness{ResolvedPnrViolationKind::ProgressProofDebt,
                            *owner};

  if (candidate.unroutedObligationCount() != 0 ||
      candidate.routeCapacityOveruse() != 0 ||
      candidate.tagResidentCapacityOveruse() != 0 ||
      candidate.tagUnassignedCount() != 0 || candidate.tagConflictCount() != 0 ||
      candidate.hardProgressViolation() != 0 ||
      candidate.progressProofDebtWitnessCount() != 0)
    return invocationError(
        "transport violation aggregates have no canonical witness");
  return std::optional<TransportWitness>();
}

llvm::Expected<bool>
transportWitnessIsLive(const SpatialCandidateState &candidate,
                       TransportWitness witness) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &transfers = problem.transfers();
  const auto &routing = problem.routing();
  switch (witness.kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      if (containsOrdinal(net.sinkOffset, net.sinkCount, witness.ordinal))
        return !candidate.usesRegisterFifo(logicalNet) &&
               candidate.routeTree(logicalNet).isUnrouted();
    }
    return invocationError("unrouted witness is out of range");
  case ResolvedPnrViolationKind::CapacityOveruse: {
    const PnrIndex capacityCount =
        static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
    if (witness.ordinal < capacityCount)
      return candidate.routeCapacityOveruseRaw(witness.ordinal) != 0;
    const PnrIndex domain = witness.ordinal - capacityCount;
    if (domain >= routing.tagContinuity().matchDomains().size())
      return invocationError("resident-row witness is out of range");
    return candidate.tagDomainResidentCapacityOveruse(domain) != 0;
  }
  case ResolvedPnrViolationKind::TagUnassigned: {
    PnrIndex ordinal = 0;
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet)
      for (const auto &value : candidate.tagValues(logicalNet)) {
        if (ordinal == witness.ordinal)
          return !value.has_value();
        if (ordinal == getPnrIndexMax())
          return invocationError("Physical Tag segment ordinal overflows");
        ++ordinal;
      }
    return invocationError("unassigned-tag witness is out of range");
  }
  case ResolvedPnrViolationKind::TagConflict:
    if (witness.ordinal >= routing.tagContinuity().matchDomains().size())
      return invocationError("tag-conflict witness is out of range");
    return candidate.tagDomainConflictCount(witness.ordinal) != 0;
  case ResolvedPnrViolationKind::HardProgressViolation:
    if (witness.ordinal >= problem.progressIndex().finiteBufferOwners().size())
      return invocationError("capacity-shortfall witness is out of range");
    return candidate.progress().capacityShortfallOwner(witness.ordinal);
  case ResolvedPnrViolationKind::ProgressProofDebt:
    if (witness.ordinal >= problem.progressIndex().finiteBufferOwners().size())
      return invocationError("capacity proof-debt witness is out of range");
    return candidate.progress().capacityProofDebtOwner(witness.ordinal);
  case ResolvedPnrViolationKind::RuntimeCounterexampleViolation:
    if (witness.ordinal >= problem.constraints().resolvedNoGoods().size())
      return invocationError("runtime-counterexample witness is out of range");
    return candidate.runtimeCounterexampleClauseViolated(witness.ordinal);
  }
  llvm_unreachable("unknown Spatial transport witness kind");
}

} // namespace

llvm::Error SpatialExactRepairScratch::planRegionDecision() {
  if (pendingRegionDecisionCount_ == std::numeric_limits<std::uint64_t>::max())
    return invocationError("pending region decision count overflows");
  if (llvm::Error error =
          workLedger_.plan(SpatialPnrWorkKind::ExactRepairRegionDecision))
    return error;
  ++pendingRegionDecisionCount_;
  return llvm::Error::success();
}

llvm::Error SpatialExactRepairScratch::consumePendingRegionDecisions() {
  if (pendingRegionDecisionCount_ >
      std::numeric_limits<std::uint64_t>::max() - accountedRegionDecisionCount_)
    return invocationError("accounted region decision count overflows");
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
    SpatialPnrWorkLedgerView workLedger) {
  workLedger_ = workLedger;
  accountedRegionDecisionCount_ = 0;
  pendingRegionDecisionCount_ = 0;
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  accountedRegionDecisions_.assign(problem.bindingRelations().decisionCount(),
                                   0);
  accountedRegionNets_.assign(problem.transfers().logicalNets().size(), 0);
  const ResolvedPnrExactRepairPolicy &policy =
      problem.config().policy().search.exactRepair;
  if (policy.kind != ResolvedPnrExactRepairKind::CpSat)
    return invocationError("CpSat_3_0 is not selected by SearchPolicy");
  if (solverCallLimit == 0 || solverCallLimit > policy.maxSolverCalls)
    return invocationError("solver-call limit exceeds SearchPolicy");
  if (candidate.runtimeCounterexampleViolation() != 0)
    return result(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "runtime-counterexample repair requires an exact literal-breaking "
        "disjunction");
  if (candidate.progressProofDebtWitnessCount() != 0 &&
      candidate.hardProgressViolation() == 0 &&
      candidate.unroutedObligationCount() == 0 &&
      candidate.routeCapacityOveruse() == 0 &&
      candidate.tagResidentCapacityOveruse() == 0 &&
      candidate.tagUnassignedCount() == 0 && candidate.tagConflictCount() == 0)
    return result(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "capacity proof-debt repair requires an owner-local disjunctive "
        "route no-good");
  if (candidate.progress().capacityShortfallOwnerCount() != 0 &&
      candidate.progress().routeDependencyViolationCount() == 0 &&
      candidate.unroutedObligationCount() == 0 &&
      candidate.routeCapacityOveruse() == 0 &&
      candidate.tagResidentCapacityOveruse() == 0 &&
      candidate.tagUnassignedCount() == 0 &&
      candidate.tagConflictCount() == 0)
    return result(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "capacity shortfall repair requires an owner-local disjunctive "
        "route or hardware no-good");
  if (candidate.atomicCapacityOveruse() == 0 &&
      candidate.hasTransportClosureViolation())
    return repairTransportClosure(candidate, restartOrdinal, solverCallLimit,
                                  exactRepairStream);
  if (candidate.atomicCapacityOveruse() == 0)
    return result(SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
                  "candidate has no supported exact-repair witness");

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
                  "atomic-capacity repair region includes a memory binding "
                  "decision");
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
  auto regionDecisionCount = detail::countExactRepairRegionDecisions(
      decisions_, affectedNets_, problem);
  if (!regionDecisionCount)
    return regionDecisionCount.takeError();
  if (*regionDecisionCount > policy.maxRegionDecisions)
    return result(SpatialExactRepairResultKind::RegionTooLarge,
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
    return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                  *regionDecisionCount, 0, 0, std::move(**modelAdmission));
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
    return result(SpatialExactRepairResultKind::InternalError,
                  *regionDecisionCount, 0, 0,
                  "exact-repair region work disagrees with its closed model");
  if (llvm::Error error = consumePendingRegionDecisions())
    return error;

  const std::int32_t solverSeed =
      detail::projectCpSatRandomSeed(exactRepairStream.nextU64());
  auto solved = detail::solveCanonicalCpSat(
      model.Build(), canonicalVariables, mutationCount->index(),
      solverCallLimit, solverSeed, workLedger_);
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

  if (llvm::Error error = actionExecutor_.prepare(candidate, workLedger_))
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
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
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
SpatialExactRepairScratch::repairTransportClosure(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
    std::uint64_t solverCallLimit,
    DeterministicPnrRandomStream &exactRepairStream) {
  const std::int32_t solverSeed =
      detail::projectCpSatRandomSeed(exactRepairStream.nextU64());
  std::vector<SpatialFixedTerminalCutCertificate> certificates;
  std::uint64_t maximumRegionDecisionCount = 0;
  std::uint64_t totalSolverCalls = 0;
  std::uint64_t totalEndpointExpansions = 0;
  std::uint64_t totalNegotiationIterations = 0;

  const auto accumulate = [](std::uint64_t value, std::uint64_t &total) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      return false;
    total += value;
    return true;
  };

  while (totalSolverCalls < solverCallLimit) {
    bool requiresRegionExpansion = false;
    auto attempt = repairTransportClosureRegion(
        candidate, restartOrdinal, solverCallLimit - totalSolverCalls,
        solverSeed, certificates, requiresRegionExpansion);
    if (!attempt)
      return attempt.takeError();
    maximumRegionDecisionCount =
        std::max(maximumRegionDecisionCount, attempt->regionDecisions);
    if (!accumulate(attempt->solverCalls, totalSolverCalls) ||
        !accumulate(attempt->endpointExpansions, totalEndpointExpansions) ||
        !accumulate(attempt->negotiationIterations, totalNegotiationIterations))
      return result(SpatialExactRepairResultKind::InternalError,
                    maximumRegionDecisionCount, totalSolverCalls,
                    attempt->actionCount,
                    "expanded route repair work accounting overflows",
                    totalEndpointExpansions, totalNegotiationIterations);

    if (!requiresRegionExpansion) {
      attempt->regionDecisions = maximumRegionDecisionCount;
      attempt->solverCalls = totalSolverCalls;
      attempt->endpointExpansions = totalEndpointExpansions;
      attempt->negotiationIterations = totalNegotiationIterations;
      return std::move(*attempt);
    }

    if (learnedCutCertificates_.size() <= certificates.size())
      return result(
          SpatialExactRepairResultKind::InternalError,
          maximumRegionDecisionCount, totalSolverCalls, attempt->actionCount,
          "fixed-terminal cut cannot expand its bounded repair region",
          totalEndpointExpansions, totalNegotiationIterations);
    certificates = learnedCutCertificates_;
  }

  return result(SpatialExactRepairResultKind::UnknownBudgetExhausted,
                maximumRegionDecisionCount, totalSolverCalls, 0,
                "route exact repair exhausted its solver-call budget before "
                "closing a certificate-expanded region",
                totalEndpointExpansions, totalNegotiationIterations);
}

llvm::Expected<SpatialExactRepairResult>
SpatialExactRepairScratch::repairTransportClosureRegion(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
    std::uint64_t solverCallLimit, std::int32_t solverSeed,
    llvm::ArrayRef<SpatialFixedTerminalCutCertificate> certificates,
    bool &requiresRegionExpansion) {
  requiresRegionExpansion = false;
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const ResolvedPnrExactRepairPolicy &policy =
      problem.config().policy().search.exactRepair;
  const detail::SpatialBindingRelationModel &bindings =
      problem.bindingRelations();
  const detail::InitializerRelationModel &relationModel = bindings.relations();
  const auto &routing = problem.routing();
  const auto &ports = problem.ports();
  const auto &transfers = problem.transfers();

  const PnrIndex capacityCount =
      static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
  auto primaryWitness = firstTransportWitness(candidate);
  if (!primaryWitness)
    return primaryWitness.takeError();
  if (!*primaryWitness) {
    if (candidate.hardProgressViolation() != 0)
      return result(
          SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
          "route-progress dependency violation has no typed repair witness");
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "transport repair could not locate its primary witness");
  }
  const ResolvedPnrViolationKind primaryWitnessKind = (*primaryWitness)->kind;
  const PnrIndex primaryWitnessOrdinal = (*primaryWitness)->ordinal;

  decisionIncluded_.assign(bindings.decisionCount(), 0);
  relationIncluded_.assign(relationModel.relations().size(), 0);
  netIncluded_.assign(transfers.logicalNets().size(), 0);
  decisionQueue_.clear();
  decisions_.clear();
  relations_.clear();
  affectedNets_.clear();
  learnedCutCertificates_.assign(certificates.begin(), certificates.end());

  const PnrIndex computeCount = bindings.computeDecisionCount();
  const PnrIndex memoryOffset = computeCount;
  const PnrIndex portOffset = bindings.portDecisionOffset();
  const PnrIndex boundaryOffset = bindings.graphBoundaryDecisionOffset();
  const auto computeDemandOffsets = ports.computeRealizationDemandOffsets();
  const auto memoryDemandOffsets = ports.memoryRealizationDemandOffsets();
  if (computeDemandOffsets.size() != computeCount + 1 ||
      memoryDemandOffsets.size() != bindings.memoryDecisionCount() + 1)
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "realization PortDemand reverse index is malformed");

  const auto portDemandOwnerDecision =
      [&](PnrIndex demand) -> llvm::Expected<PnrIndex> {
    if (demand >= ports.portDemands().size())
      return invocationError("route repair PortDemand is out of range");
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    if (record.kind == FrozenSpatialPortDemandKind::Compute) {
      if (record.realization >= bindings.computeDecisionCount())
        return invocationError(
            "route repair PortDemand has a foreign compute owner");
      return record.realization;
    }
    if (record.realization >= bindings.memoryDecisionCount())
      return invocationError(
          "route repair PortDemand has a foreign memory owner");
    return memoryOffset + record.realization;
  };

  const auto addDecision = [&](PnrIndex decision) -> llvm::Error {
    if (decision >= decisionIncluded_.size())
      return invocationError("route repair decision is out of range");
    if (!decisionIncluded_[decision]) {
      decisionIncluded_[decision] = 1;
      decisionQueue_.push_back(decision);
    }
    return llvm::Error::success();
  };
  const auto addRoutingNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= netIncluded_.size())
      return invocationError("route repair logical net is out of range");
    for (PnrIndex member :
         problem.routeConstraints().equalityClosure(logicalNet)) {
      if (member >= netIncluded_.size())
        return invocationError(
            "route equality closure contains a foreign logical net");
      if (!netIncluded_[member]) {
        netIncluded_[member] = 1;
        affectedNets_.push_back(member);
      }
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
  const auto addWitnessNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (llvm::Error error = addRoutingNet(logicalNet))
      return error;
    for (PnrIndex member :
         problem.routeConstraints().equalityClosure(logicalNet)) {
      if (llvm::Error error =
              addTerminal(transfers.logicalNetSourceBindings()[member]))
        return error;
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[member];
      for (FrozenSpatialTerminalBinding sink :
           transfers.logicalNetSinkBindings().slice(net.sinkOffset,
                                                    net.sinkCount))
        if (llvm::Error error = addTerminal(sink))
          return error;
    }
    return llvm::Error::success();
  };
  const auto addCertificateCut =
      [&](const SpatialFixedTerminalCutNet &cut) -> llvm::Error {
    if (cut.logicalNet >= transfers.logicalNets().size())
      return invocationError("route repair certificate net is out of range");
    const FrozenSpatialLogicalNet &net =
        transfers.logicalNets()[cut.logicalNet];
    if (cut.unreachableSink >= net.sinkCount)
      return invocationError("route repair certificate sink is out of range");
    if (llvm::Error error = addRoutingNet(cut.logicalNet))
      return error;
    if (llvm::Error error =
            addTerminal(transfers.logicalNetSourceBindings()[cut.logicalNet]))
      return error;
    return addTerminal(transfers.logicalNetSinkBindings()[net.sinkOffset +
                                                          cut.unreachableSink]);
  };

  const auto capacityClaimOffsets = routing.capacityRouteClaimOffsets();
  const auto capacityClaims = routing.capacityRouteClaims();
  const auto addCapacityWitness = [&](PnrIndex capacity) -> llvm::Error {
    if (capacity >= capacityCount) {
      const PnrIndex domain = capacity - capacityCount;
      if (domain >= routing.tagContinuity().matchDomains().size())
        return invocationError("tag-table witness is out of range");
      for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
           ++logicalNet) {
        const auto values = candidate.tagValues(logicalNet);
        for (PnrIndex segment = 0; segment < values.size(); ++segment)
          if (llvm::is_contained(
                  candidate.tagSegmentDomains(logicalNet, segment), domain)) {
            if (llvm::Error error = addWitnessNet(logicalNet))
              return error;
            break;
          }
      }
      return llvm::Error::success();
    }
    if (capacity + 1 >= capacityClaimOffsets.size())
      return invocationError(
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
        if (llvm::Error error = addWitnessNet(logicalNet))
          return error;
    }
    return llvm::Error::success();
  };
  switch (primaryWitnessKind) {
  case ResolvedPnrViolationKind::UnroutedObligation: {
    bool found = false;
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      if (!containsOrdinal(net.sinkOffset, net.sinkCount,
                           primaryWitnessOrdinal))
        continue;
      if (llvm::Error error = addWitnessNet(logicalNet))
        return std::move(error);
      found = true;
      break;
    }
    if (!found)
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    "unrouted witness is out of range");
    break;
  }
  case ResolvedPnrViolationKind::CapacityOveruse:
    if (llvm::Error error = addCapacityWitness(primaryWitnessOrdinal))
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    llvm::toString(std::move(error)));
    break;
  case ResolvedPnrViolationKind::TagUnassigned: {
    PnrIndex ordinal = 0;
    bool found = false;
    for (PnrIndex logicalNet = 0;
         logicalNet < transfers.logicalNets().size() && !found; ++logicalNet)
      for (const auto &value : candidate.tagValues(logicalNet)) {
        if (ordinal == primaryWitnessOrdinal) {
          if (value)
            return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                          "unassigned-tag witness is no longer live");
          if (llvm::Error error = addWitnessNet(logicalNet))
            return std::move(error);
          found = true;
          break;
        }
        ++ordinal;
      }
    if (!found)
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    "unassigned-tag witness is out of range");
    break;
  }
  case ResolvedPnrViolationKind::TagConflict: {
    const PnrIndex domain = primaryWitnessOrdinal;
    if (domain >= routing.tagContinuity().matchDomains().size())
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    "tag-conflict witness is out of range");
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const auto values = candidate.tagValues(logicalNet);
      for (PnrIndex segment = 0; segment < values.size(); ++segment) {
        if (!values[segment] ||
            !llvm::is_contained(
                candidate.tagSegmentDomains(logicalNet, segment), domain) ||
            !candidate.tagDomainValueConflicts(domain, *values[segment]))
          continue;
        if (llvm::Error error = addWitnessNet(logicalNet))
          return std::move(error);
        break;
      }
    }
    break;
  }
  case ResolvedPnrViolationKind::HardProgressViolation: {
    SpatialFiniteBufferConflictWitness witness;
    if (llvm::Error error = candidate.rebuildCapacityShortfallWitness(
            primaryWitnessOrdinal, witness))
      return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                    llvm::toString(std::move(error)));
    for (PnrIndex logicalNet : witness.competingLogicalNets)
      if (llvm::Error error = addWitnessNet(logicalNet))
        return std::move(error);
    break;
  }
  case ResolvedPnrViolationKind::ProgressProofDebt: {
    return result(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "capacity proof-debt repair requires an owner-local disjunctive "
        "route no-good");
  }
  }
  for (const SpatialFixedTerminalCutCertificate &certificate : certificates)
    for (const SpatialFixedTerminalCutNet &cut : certificate.forcedNetCuts)
      if (llvm::Error error = addCertificateCut(cut))
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      llvm::toString(std::move(error)));
  if (affectedNets_.empty())
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "transport witness has no contributing logical net");

  for (std::size_t cursor = 0; cursor < decisionQueue_.size(); ++cursor) {
    const PnrIndex decision = decisionQueue_[cursor];
    if (decision < portOffset) {
      llvm::ArrayRef<PnrIndex> ownedDemands;
      if (decision < computeCount) {
        ownedDemands = ports.computeRealizationDemands().slice(
            computeDemandOffsets[decision], computeDemandOffsets[decision + 1] -
                                                computeDemandOffsets[decision]);
      } else {
        const PnrIndex realization = decision - memoryOffset;
        ownedDemands = ports.memoryRealizationDemands().slice(
            memoryDemandOffsets[realization],
            memoryDemandOffsets[realization + 1] -
                memoryDemandOffsets[realization]);
      }
      for (PnrIndex demand : ownedDemands) {
        if (demand >= ports.portDemands().size())
          return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "realization owns a foreign PortDemand");
        if (llvm::Error error = addDecision(portOffset + demand))
          return std::move(error);
        if (llvm::Error error =
                addRoutingNet(ports.portDemands()[demand].logicalNet))
          return std::move(error);
      }
    } else if (decision < boundaryOffset) {
      const PnrIndex demand = decision - portOffset;
      auto owner = portDemandOwnerDecision(demand);
      if (!owner)
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      llvm::toString(owner.takeError()));
      if (llvm::Error error = addDecision(*owner))
        return std::move(error);
      if (llvm::Error error =
              addRoutingNet(ports.portDemands()[demand].logicalNet))
        return std::move(error);
    } else {
      const PnrIndex boundary = decision - boundaryOffset;
      if (boundary >= ports.graphBoundaries().size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "route repair boundary decision is out of range");
      if (llvm::Error error =
              addRoutingNet(ports.graphBoundaries()[boundary].logicalNet))
        return std::move(error);
    }

    for (PnrIndex relation : bindings.decisionRelations(decision)) {
      const bool closesRegion = decision < portOffset
                                    ? bindings.relationIsConstraint(relation)
                                    : !bindings.relationIsStructural(relation);
      if (!closesRegion)
        continue;
      if (relation >= relationModel.relations().size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "route repair relation is out of range");
      const detail::InitializerRelationRecord &record =
          relationModel.relations()[relation];
      for (const detail::InitializerRelationMember &member :
           relationModel.members(record))
        if (llvm::Error error = addDecision(member.decision))
          return std::move(error);
    }
  }
  decisions_ = decisionQueue_;
  llvm::sort(decisions_);
  llvm::sort(affectedNets_);

  for (PnrIndex decision : decisions_)
    for (PnrIndex relation : bindings.decisionRelations(decision)) {
      if (relation >= relationIncluded_.size())
        return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                      "route repair relation is out of range");
      if (!relationIncluded_[relation]) {
        relationIncluded_[relation] = 1;
        relations_.push_back(relation);
      }
    }
  llvm::sort(relations_);

  if (affectedNets_.size() >
      std::numeric_limits<std::uint64_t>::max() - decisions_.size())
    return result(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                  "route repair region decision count overflows");
  const std::uint64_t canonicalRegionDecisionCount =
      decisions_.size() + affectedNets_.size();
  const bool retainedDecisionMissing = llvm::any_of(
      llvm::enumerate(accountedRegionDecisions_), [&](const auto indexed) {
        return indexed.value() != 0 &&
               (indexed.index() >= decisionIncluded_.size() ||
                decisionIncluded_[indexed.index()] == 0);
      });
  const bool retainedNetMissing = llvm::any_of(
      llvm::enumerate(accountedRegionNets_), [&](const auto indexed) {
        return indexed.value() != 0 &&
               (indexed.index() >= netIncluded_.size() ||
                netIncluded_[indexed.index()] == 0);
      });
  if (retainedDecisionMissing || retainedNetMissing ||
      canonicalRegionDecisionCount < accountedRegionDecisionCount_)
    return result(SpatialExactRepairResultKind::InternalError,
                  canonicalRegionDecisionCount, 0, 0,
                  "route exact-repair region decision count regressed");
  if (canonicalRegionDecisionCount > policy.maxRegionDecisions)
    return result(SpatialExactRepairResultKind::RegionTooLarge,
                  canonicalRegionDecisionCount, 0, 0,
                  (llvm::Twine("exact-repair region has ") +
                   llvm::Twine(canonicalRegionDecisionCount) +
                   " decisions, exceeding policy limit " +
                   llvm::Twine(policy.maxRegionDecisions))
                      .str());
  const auto contextOveruse =
      problem.capacity().computeInstructionContextOveruse();
  auto modelAdmission = detail::admitTransportExactRepairModel(
      bindings, decisions_, contextOveruse);
  if (!modelAdmission)
    return modelAdmission.takeError();
  if (*modelAdmission)
    return result(SpatialExactRepairResultKind::UnsupportedEncoding,
                  canonicalRegionDecisionCount, 0, 0,
                  std::move(**modelAdmission));
  CpModelBuilder model;
  std::vector<IntVar> variables;
  variables.reserve(decisions_.size());
  std::vector<IntVar> transportObservationVariables;
  transportObservationVariables.reserve(decisions_.size());
  decisionVariables_.assign(bindings.decisionCount(), -1);
  legalValueOffsets_.clear();
  legalValueOffsets_.reserve(decisions_.size() + 1);
  legalValueOffsets_.push_back(0);
  legalValues_.clear();

  for (PnrIndex decision : decisions_) {
    if (!accountedRegionDecisions_[decision])
      if (llvm::Error error = planRegionDecision())
        return error;
    const std::size_t begin = legalValues_.size();
    if (decision < computeCount) {
      const auto choices = bindings.computeChoices(decision);
      for (auto [ordinal, choice] : llvm::enumerate(choices)) {
        if (contextOveruse[choice.instructionContext] == 0)
          legalValues_.push_back(static_cast<std::int64_t>(ordinal));
      }
    } else if (decision < portOffset) {
      const PnrIndex realization = decision - memoryOffset;
      const auto choices = bindings.memoryChoices(realization);
      for (std::size_t ordinal = 0; ordinal < choices.size(); ++ordinal)
        legalValues_.push_back(static_cast<std::int64_t>(ordinal));
    } else if (decision < boundaryOffset) {
      const PnrIndex demand = decision - portOffset;
      const auto choices = bindings.portAttachmentChoices(demand);
      for (std::size_t ordinal = 0; ordinal < choices.size(); ++ordinal)
        legalValues_.push_back(static_cast<std::int64_t>(ordinal));
    } else {
      const PnrIndex boundary = decision - boundaryOffset;
      const auto choices = bindings.graphBoundaryAttachmentChoices(boundary);
      for (std::size_t ordinal = 0; ordinal < choices.size(); ++ordinal)
        legalValues_.push_back(static_cast<std::int64_t>(ordinal));
    }
    legalValueOffsets_.push_back(static_cast<PnrIndex>(legalValues_.size()));
    const Domain domain = Domain::FromValues(
        llvm::ArrayRef(legalValues_).slice(begin, legalValues_.size() - begin));
    decisionVariables_[decision] = static_cast<int>(variables.size());
    variables.push_back(model.NewIntVar(domain));
    if (decision < computeCount) {
      const auto choices = bindings.computeChoices(decision);
      elementValues_.clear();
      elementValues_.reserve(choices.size());
      for (const detail::SpatialComputeBindingChoice &choice : choices)
        elementValues_.push_back(choice.placement);
      const IntVar placement =
          model.NewIntVar(Domain::FromValues(elementValues_));
      model.AddElement(variables.back(), elementValues_, placement);
      transportObservationVariables.push_back(placement);
    } else {
      transportObservationVariables.push_back(variables.back());
    }
  }

  std::vector<detail::CpSatCanonicalVariable> canonicalVariables;
  canonicalVariables.reserve(decisions_.size() + affectedNets_.size());
  for (std::size_t local = 0; local < decisions_.size(); ++local)
    canonicalVariables.push_back(
        {variables[local].index(),
         llvm::ArrayRef(legalValues_)
             .slice(legalValueOffsets_[local], legalValueOffsets_[local + 1] -
                                                   legalValueOffsets_[local])});

  // Encode every closed relation, including structural relations. Members
  // outside the mutable region are pinned to their current values above;
  // filtering structural relations here would loosen the exact model.
  for (PnrIndex relation : relations_) {
    const detail::InitializerRelationRecord &record =
        relationModel.relations()[relation];
    std::vector<IntVar> projections;
    projections.reserve(record.memberCount);
    for (const detail::InitializerRelationMember &member :
         relationModel.members(record)) {
      if (member.decision >= decisionVariables_.size())
        return result(SpatialExactRepairResultKind::InternalError,
                      canonicalRegionDecisionCount, 0, 0,
                      "route repair relation member is out of range");
      if (decisionVariables_[member.decision] < 0) {
        auto selected = detail::currentExactRepairBindingChoice(
            candidate, bindings, member.decision);
        if (!selected)
          return selected.takeError();
        projections.push_back(
            model.NewConstant(relationModel.projectedValue(member, *selected)));
        continue;
      }

      const std::size_t local =
          static_cast<std::size_t>(decisionVariables_[member.decision]);
      const auto offsets = relationModel.decisionChoiceOffsets();
      const PnrIndex choiceCount =
          offsets[member.decision + 1] - offsets[member.decision];
      elementValues_.clear();
      elementValues_.reserve(choiceCount);
      for (PnrIndex choice = 0; choice < choiceCount; ++choice)
        elementValues_.push_back(relationModel.projectedValue(member, choice));
      const IntVar projection =
          model.NewIntVar(Domain::FromValues(elementValues_));
      model.AddElement(variables[local], elementValues_, projection);
      projections.push_back(projection);
    }
    detail::addExactRepairInitializerRelationConstraint(model, relationModel,
                                                        record, projections);
  }

  for (PnrIndex logicalNet : affectedNets_)
    if (!accountedRegionNets_[logicalNet])
      if (llvm::Error error = planRegionDecision())
        return error;
  auto localDispositions = detail::SpatialLocalDispositionModel::build(
      model, candidate, bindings, variables, decisionVariables_, affectedNets_);
  if (!localDispositions)
    return result(SpatialExactRepairResultKind::InternalError,
                  canonicalRegionDecisionCount, 0, 0,
                  llvm::toString(localDispositions.takeError()));
  if (localDispositions->variables().size() != affectedNets_.size())
    return result(SpatialExactRepairResultKind::InternalError,
                  canonicalRegionDecisionCount, 0, 0,
                  "route repair omitted a local-disposition variable");
  for (PnrIndex local = 0; local < affectedNets_.size(); ++local) {
    canonicalVariables.push_back({localDispositions->variables()[local].index(),
                                  localDispositions->legalValues(local)});
    transportObservationVariables.push_back(
        localDispositions->variables()[local]);
  }
  std::vector<IntVar> canonicalAssignmentVariables = variables;
  canonicalAssignmentVariables.insert(canonicalAssignmentVariables.end(),
                                      localDispositions->variables().begin(),
                                      localDispositions->variables().end());

  std::vector<int> mutationParentLocals(decisions_.size(), -1);
  for (auto [local, decision] : llvm::enumerate(decisions_)) {
    if (decision < portOffset || decision >= boundaryOffset)
      continue;
    auto owner = portDemandOwnerDecision(decision - portOffset);
    if (!owner)
      return result(SpatialExactRepairResultKind::InternalError,
                    canonicalRegionDecisionCount, 0, 0,
                    llvm::toString(owner.takeError()));
    if (*owner >= decisionVariables_.size() ||
        decisionVariables_[*owner] < 0)
      return result(SpatialExactRepairResultKind::InternalError,
                    canonicalRegionDecisionCount, 0, 0,
                    "route repair omitted a PortDemand owner decision");
    mutationParentLocals[local] = decisionVariables_[*owner];
  }

  auto mutationCount = detail::addExactRepairMutationCountObjective(
      model, variables, candidate, bindings, decisions_, mutationParentLocals,
      localDispositions->variables(), localDispositions->currentValues());
  if (!mutationCount)
    return mutationCount.takeError();

  bool currentAssignmentSatisfiesCertificates = true;
  for (const SpatialFixedTerminalCutCertificate &certificate : certificates) {
    auto encoded = detail::addSpatialFixedTerminalCutEscapeConstraint(
        model, candidate, bindings, variables, decisionVariables_,
        legalValueOffsets_, legalValues_, *localDispositions, certificate,
        routeCutBlockedTraversals_, routeCutReachableEndpoints_,
        routeCutWorklist_);
    if (!encoded)
      return result(SpatialExactRepairResultKind::InternalError,
                    canonicalRegionDecisionCount, 0, 0,
                    llvm::toString(encoded.takeError()));
    if (!encoded->encoded)
      return result(SpatialExactRepairResultKind::InternalError,
                    canonicalRegionDecisionCount, 0, 0,
                    "retained fixed-terminal cut is outside its rebuilt "
                    "repair region");
    currentAssignmentSatisfiesCertificates &= encoded->currentAssignmentEscapes;
  }

  if (pendingRegionDecisionCount_ !=
      canonicalRegionDecisionCount - accountedRegionDecisionCount_)
    return result(SpatialExactRepairResultKind::InternalError,
                  canonicalRegionDecisionCount, 0, 0,
                  "route exact-repair work disagrees with its closed model");
  if (llvm::Error error = consumePendingRegionDecisions())
    return error;
  accountedRegionDecisions_ = decisionIncluded_;
  accountedRegionNets_ = netIncluded_;

  if (llvm::Error error = actionExecutor_.prepare(candidate, workLedger_))
    return result(SpatialExactRepairResultKind::InternalError,
                  canonicalRegionDecisionCount, 0, 0,
                  llvm::toString(std::move(error)));
  std::uint64_t solverCalls = 0;
  std::uint64_t assignmentOrdinal = 0;
  std::uint64_t lastActionCount = 0;
  std::uint64_t maximumObservedRegionDecisionCount =
      canonicalRegionDecisionCount;
  bool sawUnknownAssignment = false;
  bool proveCurrentAssignment = currentAssignmentSatisfiesCertificates;
  std::vector<std::int64_t> currentAssignment;
  currentAssignment.reserve(canonicalVariables.size());
  for (PnrIndex decision : decisions_) {
    auto current =
        detail::currentExactRepairBindingChoice(candidate, bindings, decision);
    if (!current)
      return result(SpatialExactRepairResultKind::InternalError,
                    canonicalRegionDecisionCount, 0, 0,
                    llvm::toString(current.takeError()));
    currentAssignment.push_back(static_cast<std::int64_t>(*current));
  }
  currentAssignment.insert(currentAssignment.end(),
                           localDispositions->currentValues().begin(),
                           localDispositions->currentValues().end());

  const auto executedResult = [&](SpatialExactRepairResultKind kind,
                                  std::string detail = {}) {
    return result(kind, maximumObservedRegionDecisionCount, solverCalls,
                  lastActionCount, std::move(detail),
                  actionExecutor_.endpointExpansionCount(),
                  actionExecutor_.negotiationIterationCount());
  };

  const dse::ObjectiveVector initialObjective =
      actionExecutor_.currentObjective();

  // A legal route closure may be worse than the current objective. Keep the
  // best such assignment outside the mutable CP-SAT model so search can look
  // for a better legal closure without turning the fallback into a proof of
  // illegality.
  std::vector<SpatialMappingAction> bestLegalFallbackActions;
  std::vector<std::int64_t> bestLegalFallbackAssignment;
  std::optional<dse::ObjectiveVector> bestLegalFallbackObjective;

  const auto commitBestLegalFallback = [&](std::string detail)
      -> llvm::Expected<SpatialExactRepairResult> {
    if (!bestLegalFallbackObjective || bestLegalFallbackActions.empty() ||
        bestLegalFallbackAssignment.size() != canonicalVariables.size())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair lost its legal fallback");

    actions_ = bestLegalFallbackActions;
    auto fallbackProbe = actionExecutor_.probeBatch(
        candidate, actions_, SpatialActionExecutionContext::ExactRepair,
        policy.maxRegionDecisions - decisions_.size());
    if (!fallbackProbe) {
      std::string failureDetail;
      std::optional<SpatialActionTransitionFailureKind> transitionFailure;
      llvm::Error unhandled = llvm::handleErrors(
          fallbackProbe.takeError(),
          [&](const SpatialActionTransitionFailure &failure) -> llvm::Error {
            llvm::raw_string_ostream stream(failureDetail);
            failure.log(stream);
            transitionFailure = failure.kind();
            return llvm::Error::success();
          });
      const bool hasUnhandled = static_cast<bool>(unhandled);
      if (hasUnhandled)
        failureDetail = llvm::toString(std::move(unhandled));
      const SpatialExactRepairResultKind kind =
          !hasUnhandled && transitionFailure &&
                  *transitionFailure ==
                      SpatialActionTransitionFailureKind::WorkLimit
              ? SpatialExactRepairResultKind::UnknownBudgetExhausted
              : SpatialExactRepairResultKind::InternalError;
      return executedResult(
          kind, (llvm::Twine("best legal route-repair fallback could not be "
                             "replayed: ") +
                 failureDetail)
                    .str());
    }

    bool assignmentRealized = true;
    for (auto [local, decision] : llvm::enumerate(decisions_)) {
      auto realized = detail::currentExactRepairBindingChoice(
          candidate, bindings, decision);
      if (!realized) {
        llvm::Error error = realized.takeError();
        if (llvm::Error discardError = fallbackProbe->discard())
          error = llvm::joinErrors(std::move(error), std::move(discardError));
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      }
      if (*realized !=
          static_cast<PnrIndex>(bestLegalFallbackAssignment[local])) {
        assignmentRealized = false;
        break;
      }
    }
    auto witnessLive = transportWitnessIsLive(candidate, **primaryWitness);
    if (!witnessLive) {
      llvm::Error error = witnessLive.takeError();
      if (llvm::Error discardError = fallbackProbe->discard())
        error = llvm::joinErrors(std::move(error), std::move(discardError));
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
    }
    if (!assignmentRealized || candidate.atomicCapacityOveruse() != 0 ||
        *witnessLive) {
      if (llvm::Error error = fallbackProbe->discard())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      return executedResult(
          SpatialExactRepairResultKind::InternalError,
          "recorded legal route-repair fallback failed its replay checks");
    }
    lastActionCount = actions_.size();
    if (llvm::Error error = fallbackProbe->commit())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
    if (llvm::Error error = candidate.verify())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ActionOutcome,
        [&](llvm::json::Object &fields) {
          fields["search_scope"] = "route_exact_repair";
          fields["restart"] = restartOrdinal;
          fields["assignment"] = assignmentOrdinal;
          fields["accepted"] = true;
          fields["legal_fallback"] = true;
          fields["witness_kind"] =
              static_cast<std::uint32_t>(primaryWitnessKind);
          fields["witness_ordinal"] = primaryWitnessOrdinal;
          fields["solver_calls"] = solverCalls;
        });
    return executedResult(SpatialExactRepairResultKind::Repaired,
                          std::move(detail));
  };

  while (solverCalls < solverCallLimit) {
    auto solved =
        proveCurrentAssignment
            ? detail::solveFixedCpSatAssignment(
                  model.Build(), canonicalVariables, currentAssignment,
                  mutationCount->index(), solverCallLimit - solverCalls,
                  solverSeed, workLedger_)
            : detail::solveCanonicalCpSat(
                  model.Build(), canonicalVariables, mutationCount->index(),
                  solverCallLimit - solverCalls, solverSeed, workLedger_);
    if (!solved)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(solved.takeError()));
    solverCalls += solved->solverCalls;
    if (solved->kind == detail::CpSatCanonicalResultKind::Infeasible) {
      if (proveCurrentAssignment)
        return executedResult(
            SpatialExactRepairResultKind::InternalError,
            "current route-repair assignment violates its exact model");
      if (bestLegalFallbackObjective)
        return commitBestLegalFallback(
            "committed the best legal route-repair fallback after search "
            "exhaustion");
      if (sawUnknownAssignment)
        return executedResult(
            SpatialExactRepairResultKind::UnknownBudgetExhausted,
            "bounded route-repair assignments were exhausted after at "
            "least one route probe reached its work limit");
      return executedResult(
          SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
          "bounded route-repair assignments were exhausted without an "
          "exact route closure");
    }
    if (solved->kind ==
        detail::CpSatCanonicalResultKind::UnknownBudgetExhausted) {
      if (bestLegalFallbackObjective) {
        return commitBestLegalFallback(
            "committed the best legal route-repair fallback at the solver "
            "budget boundary");
      }
      return executedResult(
          SpatialExactRepairResultKind::UnknownBudgetExhausted,
          "route exact repair exhausted its solver-call budget");
    }
    if (solved->assignment.size() != canonicalVariables.size())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair assignment has the wrong size");
    if (!solved->objectiveValue || *solved->objectiveValue < 0)
      return executedResult(
          SpatialExactRepairResultKind::InternalError,
          "route repair omitted its mutation-count objective");
    if (proveCurrentAssignment && *solved->objectiveValue != 0)
      return executedResult(
          SpatialExactRepairResultKind::InternalError,
          "current route-repair assignment has nonzero mutation count");

    actions_.clear();
    for (auto [local, decision] : llvm::enumerate(decisions_)) {
      const std::int64_t selected = solved->assignment[local];
      if (selected < 0)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              "route repair selected a negative choice");
      if (decision < computeCount) {
        const auto choices = bindings.computeChoices(decision);
        if (static_cast<std::size_t>(selected) >= choices.size())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                "route repair compute choice is invalid");
        const detail::SpatialComputeBindingChoice &choice = choices[selected];
        const SpatialComputeBindingSelection &current =
            candidate.computeBinding(decision);
        if (current.placement != choice.placement ||
            current.instructionContext != choice.instructionContext)
          actions_.push_back(
              SpatialRealizationBindingAction{SpatialComputeBindingAction{
                  decision, choice.placement, choice.instructionContext}});
      } else if (decision < portOffset) {
        const PnrIndex realization = decision - memoryOffset;
        const auto choices = bindings.memoryChoices(realization);
        if (static_cast<std::size_t>(selected) >= choices.size())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                "route repair memory choice is invalid");
        const detail::SpatialMemoryBindingChoice &choice = choices[selected];
        if (candidate.memoryBinding(realization).placement != choice.placement)
          actions_.push_back(SpatialRealizationBindingAction{
              SpatialMemoryBindingAction{realization, choice.placement}});
      }
    }
    for (PnrIndex local = 0; local < affectedNets_.size(); ++local) {
      const std::int64_t selectedDisposition =
          solved->assignment[decisions_.size() + local];
      if (selectedDisposition == localDispositions->currentValues()[local])
        continue;
      auto localOption = localDispositions->selectedOption(
          local, selectedDisposition);
      if (!localOption)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(localOption.takeError()));
      actions_.push_back(
          SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{
              affectedNets_[local],
              *localOption ? SpatialWholeNetDispositionKind::RegisterFifo
                           : SpatialWholeNetDispositionKind::External,
              *localOption ? **localOption : getInvalidPnrIndex()}});
    }
    actions_.push_back(SpatialTransportRoutingAction{
        SpatialWitnessRegionRoutingAction{primaryWitnessKind,
                                          primaryWitnessOrdinal}});
    for (auto [local, decision] : llvm::enumerate(decisions_)) {
      const std::int64_t selected = solved->assignment[local];
      if (decision < portOffset)
        continue;
      if (decision < boundaryOffset) {
        const PnrIndex demand = decision - portOffset;
        const auto choices = bindings.portAttachmentChoices(demand);
        if (static_cast<std::size_t>(selected) >= choices.size())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                "route repair PortDemand choice is invalid");
        const PnrIndex option = choices[selected];
        if (candidate.portAttachment(demand) != option)
          actions_.push_back(SpatialResourceAllocationAction{
              SpatialPortAttachmentAction{demand, option}});
      } else {
        const PnrIndex boundary = decision - boundaryOffset;
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
          fields["witness_kind"] =
              static_cast<std::uint32_t>(primaryWitnessKind);
          fields["witness_ordinal"] = primaryWitnessOrdinal;
          fields["region_decisions"] = canonicalRegionDecisionCount;
          fields["binding_decision_count"] = decisions_.size();
          fields["affected_logical_net_count"] = affectedNets_.size();
          fields["mutation_count"] = *solved->objectiveValue;
          fields["action_count"] = lastActionCount;
          fields["solver_calls"] = solverCalls;
        });
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Detail,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ContextChoice,
        [&](llvm::json::Object &fields) {
          llvm::json::Array selectedDecisions;
          for (auto [local, decision] : llvm::enumerate(decisions_)) {
            const std::int64_t selected = solved->assignment[local];
            llvm::json::Object selectedDecision;
            selectedDecision["decision"] = decision;
            selectedDecision["choice"] = selected;
            if (decision < computeCount) {
              const detail::SpatialComputeBindingChoice &choice =
                  bindings.computeChoices(decision)[selected];
              selectedDecision["kind"] = "compute_binding";
              selectedDecision["anchor"] = decision;
              selectedDecision["placement"] = choice.placement;
              selectedDecision["instruction_context"] =
                  choice.instructionContext;
            } else if (decision < portOffset) {
              const PnrIndex realization = decision - memoryOffset;
              const detail::SpatialMemoryBindingChoice &choice =
                  bindings.memoryChoices(realization)[selected];
              selectedDecision["kind"] = "memory_binding";
              selectedDecision["anchor"] = realization;
              selectedDecision["placement"] = choice.placement;
            } else if (decision < boundaryOffset) {
              const PnrIndex demand = decision - portOffset;
              const PnrIndex option =
                  bindings.portAttachmentChoices(demand)[selected];
              selectedDecision["kind"] = "port_attachment";
              selectedDecision["anchor"] = demand;
              selectedDecision["attachment_option"] = option;
              selectedDecision["endpoint"] =
                  ports.attachmentOptions()[option].endpoint;
            } else {
              const PnrIndex boundary = decision - boundaryOffset;
              const PnrIndex option =
                  bindings.graphBoundaryAttachmentChoices(boundary)[selected];
              selectedDecision["kind"] = "graph_boundary_attachment";
              selectedDecision["anchor"] = boundary;
              selectedDecision["attachment_option"] = option;
              selectedDecision["endpoint"] =
                  ports.attachmentOptions()[option].endpoint;
            }
            selectedDecisions.push_back(std::move(selectedDecision));
          }
          fields["search_scope"] = "route_exact_repair";
          fields["restart"] = restartOrdinal;
          fields["assignment"] = assignmentOrdinal;
          fields["decisions"] = std::move(selectedDecisions);
        });

    auto probe = actionExecutor_.probeBatch(
        candidate, actions_, SpatialActionExecutionContext::ExactRepair,
        policy.maxRegionDecisions - decisions_.size());
    const std::uint64_t regionalLogicalNetCount =
        actionExecutor_.regionalLogicalNetCount();
    for (PnrIndex logicalNet : actionExecutor_.regionalLogicalNets()) {
      if (logicalNet >= netIncluded_.size())
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              "route conflict closure contains a foreign "
                              "logical net");
    }
    if (regionalLogicalNetCount >
        std::numeric_limits<std::uint64_t>::max() - decisions_.size())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route conflict closure decision count "
                            "overflows");
    const std::uint64_t assignmentRegionDecisionCount =
        decisions_.size() + regionalLogicalNetCount;
    maximumObservedRegionDecisionCount = std::max(
        maximumObservedRegionDecisionCount, assignmentRegionDecisionCount);
    bool rejectAssignment = false;
    bool fixedTerminalCut = false;
    bool regionalLimit = false;
    std::uint64_t rejectedRegionDecisionCount = assignmentRegionDecisionCount;
    bool routeWorkUnknown = false;
    bool objectiveOnlyRejection = false;
    routeCutCertificate_ = {};
    if (!probe) {
      std::string detail;
      std::optional<SpatialActionTransitionFailureKind> transitionFailure;
      llvm::Error unhandled = llvm::handleErrors(
          probe.takeError(),
          [&](const SpatialPathFinderClosureFailure &failure) -> llvm::Error {
            if (failure.kind() ==
                SpatialPathFinderClosureFailure::Kind::RegionalLimit) {
              if (failure.regionalLogicalNetCount() >
                  std::numeric_limits<std::uint64_t>::max() - decisions_.size())
                return invocationError(
                    "route conflict closure decision count overflows");
              regionalLimit = true;
              rejectedRegionDecisionCount =
                  decisions_.size() + failure.regionalLogicalNetCount();
              return llvm::Error::success();
            }
            if (failure.kind() != SpatialPathFinderClosureFailure::Kind::
                                      FixedTerminalCapacityCut) {
              std::string message;
              llvm::raw_string_ostream stream(message);
              failure.log(stream);
              return llvm::make_error<SpatialPathFinderClosureFailure>(
                  failure.kind(), stream.str(), failure.certificate(),
                  failure.mandatoryUsage(), failure.physicalCapacity(),
                  failure.regionalLogicalNetCount(),
                  failure.regionalLogicalNetLimit());
            }
            fixedTerminalCut = true;
            routeCutCertificate_ = failure.certificate();
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
          *transitionFailure == SpatialActionTransitionFailureKind::WorkLimit) {
        routeWorkUnknown = true;
        sawUnknownAssignment = true;
      }
      rejectAssignment = true;
    } else {
      bool assignmentRealized = true;
      for (auto [local, decision] : llvm::enumerate(decisions_)) {
        auto realized = detail::currentExactRepairBindingChoice(
            candidate, bindings, decision);
        if (!realized) {
          if (llvm::Error error = probe->discard())
            return executedResult(SpatialExactRepairResultKind::InternalError,
                                  llvm::toString(std::move(error)));
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                llvm::toString(realized.takeError()));
        }
        if (*realized != static_cast<PnrIndex>(solved->assignment[local])) {
          assignmentRealized = false;
          break;
        }
      }
      auto primaryWitnessLive =
          transportWitnessIsLive(candidate, **primaryWitness);
      if (!primaryWitnessLive) {
        llvm::Error error = primaryWitnessLive.takeError();
        if (llvm::Error discardError = probe->discard())
          error = llvm::joinErrors(std::move(error), std::move(discardError));
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      }
      auto selectedRank =
          candidate.problem().objectiveProgram().compareSelectedRank(
              probe->objective(), {}, initialObjective, {});
      if (!selectedRank) {
        llvm::Error error = selectedRank.takeError();
        if (llvm::Error discardError = probe->discard())
          error = llvm::joinErrors(std::move(error), std::move(discardError));
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(std::move(error)));
      }
      const bool selectedRankImproved = *selectedRank < 0;
      const bool assignmentLegal =
          assignmentRealized && candidate.atomicCapacityOveruse() == 0 &&
          !*primaryWitnessLive;
      if (!assignmentLegal) {
        rejectAssignment = true;
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Decision,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              fields["search_scope"] = "route_exact_repair";
              fields["operation"] = "assignment_acceptance";
              fields["restart"] = restartOrdinal;
              fields["assignment"] = assignmentOrdinal;
              fields["assignment_realized"] = assignmentRealized;
              fields["atomic_capacity_overuse"] =
                  candidate.atomicCapacityOveruse();
              fields["primary_witness_eliminated"] = !*primaryWitnessLive;
              fields["selected_rank_improved"] = selectedRankImproved;
              llvm::json::Array initialCodes;
              for (std::uint64_t code : initialObjective.codes())
                initialCodes.push_back(code);
              fields["initial_objective_codes"] = std::move(initialCodes);
              llvm::json::Array candidateCodes;
              for (std::uint64_t code : probe->objective().codes())
                candidateCodes.push_back(code);
              fields["candidate_objective_codes"] = std::move(candidateCodes);
            });
        if (llvm::Error error = probe->discard())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                llvm::toString(std::move(error)));
      } else if (*selectedRank <= 0) {
        // Objective rank is a preference only. Equal-rank and improved legal
        // closures can be committed immediately; neither is a no-good.
        if (llvm::Error error = probe->commit())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                llvm::toString(std::move(error)));
        if (llvm::Error error = candidate.verify())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                llvm::toString(std::move(error)));
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Decision,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::ActionOutcome,
            [&](llvm::json::Object &fields) {
              fields["search_scope"] = "route_exact_repair";
              fields["restart"] = restartOrdinal;
              fields["assignment"] = assignmentOrdinal;
              fields["accepted"] = true;
              fields["witness_kind"] =
                  static_cast<std::uint32_t>(primaryWitnessKind);
              fields["witness_ordinal"] = primaryWitnessOrdinal;
              fields["solver_calls"] = solverCalls;
        });
        return executedResult(SpatialExactRepairResultKind::Repaired);
      } else {
        // Keep QoR-worse legal closures as a deterministic fallback while the
        // canonical model searches for a better legal assignment. The
        // fallback is copied before the provisional probe is discarded.
        bool replaceFallback = !bestLegalFallbackObjective;
        if (bestLegalFallbackObjective) {
          auto comparison =
              candidate.problem().objectiveProgram().compareSelectedRank(
                  probe->objective(), {}, *bestLegalFallbackObjective, {});
          if (!comparison) {
            llvm::Error error = comparison.takeError();
            if (llvm::Error discardError = probe->discard())
              error = llvm::joinErrors(std::move(error),
                                       std::move(discardError));
            return executedResult(SpatialExactRepairResultKind::InternalError,
                                  llvm::toString(std::move(error)));
          }
          replaceFallback = *comparison < 0;
          if (*comparison == 0 &&
              std::lexicographical_compare(
                  solved->assignment.begin(), solved->assignment.end(),
                  bestLegalFallbackAssignment.begin(),
                  bestLegalFallbackAssignment.end()))
            replaceFallback = true;
        }
        if (replaceFallback) {
          bestLegalFallbackActions = actions_;
          bestLegalFallbackAssignment = solved->assignment;
          bestLegalFallbackObjective = probe->objective();
        }
        objectiveOnlyRejection = true;
        rejectAssignment = true;
        if (llvm::Error error = probe->discard())
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                llvm::toString(std::move(error)));
      }
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
          fields["witness_kind"] =
              static_cast<std::uint32_t>(primaryWitnessKind);
          fields["witness_ordinal"] = primaryWitnessOrdinal;
          fields["solver_calls"] = solverCalls;
          fields["fixed_terminal_cut"] = fixedTerminalCut;
          fields["regional_limit"] = regionalLimit;
          fields["region_decisions"] = rejectedRegionDecisionCount;
          fields["route_work_unknown"] = routeWorkUnknown;
          fields["objective_only_rejection"] = objectiveOnlyRejection;
          fields["legal_fallback_recorded"] =
              objectiveOnlyRejection && bestLegalFallbackObjective.has_value();
          fields["cut_logical_net_count"] =
              routeCutCertificate_.forcedNetCuts.size();
        });
    if (!rejectAssignment)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair lost its assignment outcome");
    if (regionalLimit) {
      if (bestLegalFallbackObjective)
        return commitBestLegalFallback(
            "committed the best legal route-repair fallback before the "
            "regional closure limit");
      return result(SpatialExactRepairResultKind::RegionTooLarge,
                    rejectedRegionDecisionCount, solverCalls, lastActionCount,
                    "route conflict closure exceeds max_region_decisions",
                    actionExecutor_.endpointExpansionCount(),
                    actionExecutor_.negotiationIterationCount());
    }
    if (fixedTerminalCut && bestLegalFallbackObjective)
      return commitBestLegalFallback(
          "committed the best legal route-repair fallback before fixed "
          "terminal-region expansion");
    if (fixedTerminalCut) {
      if (!insertCutCertificate(learnedCutCertificates_, routeCutCertificate_))
        return executedResult(
            SpatialExactRepairResultKind::InternalError,
            "negotiated routing repeated an active fixed-terminal cut");
      auto encoded = detail::addSpatialFixedTerminalCutEscapeConstraint(
          model, candidate, bindings, variables, decisionVariables_,
          legalValueOffsets_, legalValues_, *localDispositions,
          routeCutCertificate_, routeCutBlockedTraversals_,
          routeCutReachableEndpoints_, routeCutWorklist_);
      if (!encoded)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(encoded.takeError()));
      if (!encoded->encoded) {
        requiresRegionExpansion = true;
        return executedResult(
            SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
            "fixed-terminal cut crosses the bounded repair region");
      }
    } else {
      elementValues_.clear();
      if (objectiveOnlyRejection) {
        // This exclusion only advances this invocation's canonical
        // enumeration. The legal tuple remains retained above and is
        // committed if no better legal closure is found; it is not evidence
        // that the hard domain is infeasible.
        elementValues_.assign(solved->assignment.begin(),
                              solved->assignment.end());
        model.AddForbiddenAssignments(canonicalAssignmentVariables)
            .AddTuple(elementValues_);
      } else {
        elementValues_.reserve(canonicalVariables.size());
        for (auto [local, decision] : llvm::enumerate(decisions_)) {
          const std::int64_t selected = solved->assignment[local];
          if (decision < computeCount) {
            const auto choices = bindings.computeChoices(decision);
            if (selected < 0 ||
                static_cast<std::size_t>(selected) >= choices.size())
              return executedResult(
                  SpatialExactRepairResultKind::InternalError,
                  "route repair cannot project a compute observation");
            elementValues_.push_back(choices[selected].placement);
          } else {
            elementValues_.push_back(selected);
          }
        }
        elementValues_.insert(elementValues_.end(),
                              solved->assignment.begin() + decisions_.size(),
                              solved->assignment.end());
        model.AddForbiddenAssignments(transportObservationVariables)
            .AddTuple(elementValues_);
      }
    }
    proveCurrentAssignment = false;
    ++assignmentOrdinal;
  }
  if (bestLegalFallbackObjective)
    return commitBestLegalFallback(
        "committed the best legal route-repair fallback after the solver "
        "budget was exhausted");
  return executedResult(SpatialExactRepairResultKind::UnknownBudgetExhausted,
                        "route exact repair exhausted its solver-call budget");
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
