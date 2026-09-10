#include "PnR/SpatialExactRepair.h"

#include "Common/MappingDebugLog.h"
#include "CpSatExactProtocol.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "PnR/MappingObjective.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialExactRepairInternal.h"
#include "SpatialExactRepairModel.h"
#include "SpatialFixedTerminalCutConstraint.h"
#include "SpatialLocalDispositionModel.h"
#include "SpatialRouteConstraintModel.h"
#include "SpatialRuntimeCounterexampleRepairModel.h"
#include "SpatialTransportWitness.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/SmallString.h"
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
using loom::pnr::detail::SpatialTransportWitness;
using loom::pnr::detail::firstSpatialTransportWitness;
using loom::pnr::detail::spatialTransportWitnessIsLive;

namespace {

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


} // namespace

llvm::Expected<SpatialExactRepairResult>
SpatialExactRepairScratch::repairTransportClosure(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
    std::uint64_t solverCallLimit,
    DeterministicPnrRandomStream &exactRepairStream,
    std::optional<PnrIndex> runtimeCounterexampleClause,
    llvm::ArrayRef<PnrIndex> preferredBindingDecisions) {
  const std::int32_t solverSeed =
      detail::projectCpSatRandomSeed(exactRepairStream.nextU64());
  std::uint64_t maximumRegionDecisionCount = 0;
  std::uint64_t totalSolverCalls = 0;
  std::uint64_t totalLogicalSolverCalls = 0;
  std::uint64_t totalEndpointExpansions = 0;
  std::uint64_t totalNegotiationIterations = 0;

  const auto accumulate = [](std::uint64_t value, std::uint64_t &total) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      return false;
    total += value;
    return true;
  };

  llvm::Expected<std::optional<SpatialTransportWitness>> primaryWitness =
      runtimeCounterexampleClause
          ? llvm::Expected<std::optional<SpatialTransportWitness>>(
                SpatialTransportWitness{
                    ResolvedPnrViolationKind::RuntimeCounterexampleViolation,
                    *runtimeCounterexampleClause})
          : firstSpatialTransportWitness(candidate);
  if (!primaryWitness)
    return primaryWitness.takeError();
  if (!*primaryWitness)
    return repairResult(
        SpatialExactRepairResultKind::InternalError, 0, 0, 0,
        "transport repair could not locate its primary witness");

  std::vector<detail::SpatialRuntimeCounterexampleBreaker> runtimeBreakers;
  if ((*primaryWitness)->kind ==
      ResolvedPnrViolationKind::RuntimeCounterexampleViolation) {
    auto derived = detail::enumerateSpatialRuntimeCounterexampleBreakers(
        candidate, (*primaryWitness)->ordinal, preferredBindingDecisions);
    if (!derived)
      return derived.takeError();
    runtimeBreakers = std::move(*derived);
    if (runtimeBreakers.empty())
      return repairResult(
          SpatialExactRepairResultKind::ProofNotEstablished, 0, 0, 0,
          "runtime-counterexample clause has no finite breaker");
  }

  const std::size_t branchCount =
      runtimeBreakers.empty() ? 1 : runtimeBreakers.size();
  std::uint64_t lastActionCount = 0;
  bool sawIncompleteBreakerRouting = false;
  bool sawIncompleteExactProof = false;
  bool sawOversizedRegion = false;
  for (std::size_t branch = 0; branch < branchCount; ++branch) {
    if (executionControl_.stopRequested())
      return repairResult(SpatialExactRepairResultKind::TimedOut,
                          maximumRegionDecisionCount, totalSolverCalls,
                          lastActionCount,
                          "runtime-counterexample repair reached its deadline "
                          "between finite breaker branches",
                          totalEndpointExpansions, totalNegotiationIterations,
                          totalLogicalSolverCalls);
    const detail::SpatialRuntimeCounterexampleBreaker *runtimeBreaker =
        runtimeBreakers.empty() ? nullptr : &runtimeBreakers[branch];
    if (runtimeBreaker &&
        runtimeBreaker->kind ==
            detail::SpatialRuntimeCounterexampleBreakerKind::MappingIdentity) {
      // Monotone expansion accounting is local to one owner region. The
      // invocation ledger still accounts for every branch's actual work.
      accountedRegionDecisionCount_ = 0;
      pendingRegionDecisionCount_ = 0;
      std::fill(accountedRegionDecisions_.begin(),
                accountedRegionDecisions_.end(), 0);
      std::fill(accountedRegionNets_.begin(), accountedRegionNets_.end(), 0);
    }
    std::vector<SpatialFixedTerminalCutCertificate> certificates;
    std::vector<PnrIndex> handshakeCycleNets;
    const std::uint64_t branchStartLogicalSolverCalls = totalLogicalSolverCalls;
    const std::uint64_t remainingBranches = branchCount - branch;
    const std::uint64_t remainingGlobalCalls =
        solverCallLimit - totalLogicalSolverCalls;
    const std::uint64_t branchSolverCallLimit =
        runtimeBreaker ? std::max<std::uint64_t>(1, remainingGlobalCalls /
                                                        remainingBranches)
                       : remainingGlobalCalls;
    while (totalLogicalSolverCalls - branchStartLogicalSolverCalls <
           branchSolverCallLimit) {
      if (executionControl_.stopRequested())
        return repairResult(
            SpatialExactRepairResultKind::TimedOut, maximumRegionDecisionCount,
            totalSolverCalls, lastActionCount,
            "runtime-counterexample repair reached its deadline "
            "between exact assignments",
            totalEndpointExpansions, totalNegotiationIterations,
            totalLogicalSolverCalls);
      bool requiresRegionExpansion = false;
      const std::size_t certificateCountBefore = certificates.size();
      const std::size_t handshakeCycleNetCountBefore =
          handshakeCycleNets.size();
      auto attempt = repairTransportClosureRegion(
          candidate, restartOrdinal,
          branchSolverCallLimit -
              (totalLogicalSolverCalls - branchStartLogicalSolverCalls),
          solverSeed, certificates, handshakeCycleNets, requiresRegionExpansion,
          runtimeBreaker);
      if (!attempt)
        return attempt.takeError();
      lastActionCount = attempt->actionCount;
      maximumRegionDecisionCount =
          std::max(maximumRegionDecisionCount, attempt->regionDecisions);
      if (!accumulate(attempt->solverCalls, totalSolverCalls) ||
          !accumulate(attempt->logicalSolverCalls, totalLogicalSolverCalls) ||
          !accumulate(attempt->endpointExpansions, totalEndpointExpansions) ||
          !accumulate(attempt->negotiationIterations,
                      totalNegotiationIterations))
        return repairResult(SpatialExactRepairResultKind::InternalError,
                            maximumRegionDecisionCount, totalSolverCalls,
                            attempt->actionCount,
                            "expanded route repair work accounting overflows",
                            totalEndpointExpansions, totalNegotiationIterations,
                            totalLogicalSolverCalls);

      if (attempt->kind == SpatialExactRepairResultKind::UnknownBudgetExhausted)
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Summary,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              fields["failure_scope"] = "route_exact_repair";
              fields["restart"] = restartOrdinal;
              fields["solver_seed"] = solverSeed;
              fields["branch"] = branch;
              fields["branch_max_solver_calls"] = branchSolverCallLimit;
              fields["solver_calls"] = totalSolverCalls;
              fields["logical_solver_calls"] = totalLogicalSolverCalls;
              fields["max_solver_calls"] = solverCallLimit;
              fields["diagnostic"] = attempt->detail;
            });

      if (requiresRegionExpansion) {
        const bool certificatesExpanded =
            learnedCutCertificates_.size() > certificateCountBefore;
        const bool handshakeCycleExpanded =
            handshakeCycleNets.size() > handshakeCycleNetCountBefore;
        if (!certificatesExpanded && !handshakeCycleExpanded)
          return repairResult(
              SpatialExactRepairResultKind::InternalError,
              maximumRegionDecisionCount, totalSolverCalls,
              attempt->actionCount,
              "route witness cannot expand its bounded repair region",
              totalEndpointExpansions, totalNegotiationIterations,
              totalLogicalSolverCalls);
        certificates = learnedCutCertificates_;
        continue;
      }

      if (runtimeBreaker &&
          runtimeBreaker->kind ==
              detail::SpatialRuntimeCounterexampleBreakerKind::MappingIdentity &&
          attempt->kind == SpatialExactRepairResultKind::RegionTooLarge) {
        sawOversizedRegion = true;
        break;
      }
      if (runtimeBreaker &&
          attempt->kind ==
              SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary)
        break;
      if (runtimeBreaker &&
          attempt->kind == SpatialExactRepairResultKind::RoutingIncomplete) {
        sawIncompleteBreakerRouting = true;
        break;
      }
      if (runtimeBreaker &&
          attempt->kind ==
              SpatialExactRepairResultKind::UnknownBudgetExhausted) {
        sawIncompleteExactProof = true;
        break;
      }
      attempt->regionDecisions = maximumRegionDecisionCount;
      attempt->solverCalls = totalSolverCalls;
      attempt->logicalSolverCalls = totalLogicalSolverCalls;
      attempt->endpointExpansions = totalEndpointExpansions;
      attempt->negotiationIterations = totalNegotiationIterations;
      return std::move(*attempt);
    }

    if (totalLogicalSolverCalls >= solverCallLimit)
      return repairResult(
          SpatialExactRepairResultKind::UnknownBudgetExhausted,
          maximumRegionDecisionCount, totalSolverCalls, lastActionCount,
          "runtime-counterexample repair exhausted its shared solver-call "
          "budget before every finite breaker branch was decided",
          totalEndpointExpansions, totalNegotiationIterations,
          totalLogicalSolverCalls);
  }

  if (!runtimeBreakers.empty())
    if (sawIncompleteExactProof)
      return repairResult(
          SpatialExactRepairResultKind::UnknownBudgetExhausted,
          maximumRegionDecisionCount, totalSolverCalls, lastActionCount,
          "no finite runtime-counterexample breaker repaired the candidate, "
          "and at least one branch did not complete its bounded exact proof",
          totalEndpointExpansions, totalNegotiationIterations,
          totalLogicalSolverCalls);
  if (!runtimeBreakers.empty())
    if (sawIncompleteBreakerRouting)
      return repairResult(
          SpatialExactRepairResultKind::RoutingIncomplete,
          maximumRegionDecisionCount, totalSolverCalls, lastActionCount,
          "no finite runtime-counterexample breaker repaired the candidate, "
          "and at least one branch had incomplete negotiated routing",
          totalEndpointExpansions, totalNegotiationIterations,
          totalLogicalSolverCalls);
  if (sawOversizedRegion)
    return repairResult(
        SpatialExactRepairResultKind::RegionTooLarge,
        maximumRegionDecisionCount, totalSolverCalls, lastActionCount,
        "no owner region repaired the candidate and at least one dependency "
        "closure exceeded the unchanged region limit",
        totalEndpointExpansions, totalNegotiationIterations,
        totalLogicalSolverCalls);
  if (!runtimeBreakers.empty())
    return repairResult(
        SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
        maximumRegionDecisionCount, totalSolverCalls, lastActionCount,
        "every finite runtime-counterexample breaker is infeasible under the "
        "fixed region boundary",
        totalEndpointExpansions, totalNegotiationIterations,
        totalLogicalSolverCalls);

  return repairResult(
      SpatialExactRepairResultKind::UnknownBudgetExhausted,
      maximumRegionDecisionCount, totalSolverCalls, 0,
      "route exact repair exhausted its solver-call budget before "
      "closing a certificate-expanded region",
      totalEndpointExpansions, totalNegotiationIterations,
      totalLogicalSolverCalls);
}

llvm::Expected<SpatialExactRepairResult>
SpatialExactRepairScratch::repairTransportClosureRegion(
    SpatialCandidateState &candidate, std::uint64_t restartOrdinal,
    std::uint64_t solverCallLimit, std::int32_t solverSeed,
    llvm::ArrayRef<SpatialFixedTerminalCutCertificate> certificates,
    std::vector<PnrIndex> &handshakeCycleNets,
    bool &requiresRegionExpansion,
    const detail::SpatialRuntimeCounterexampleBreaker *runtimeBreaker) {
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
  llvm::Expected<std::optional<SpatialTransportWitness>> primaryWitness =
      runtimeBreaker
          ? llvm::Expected<std::optional<SpatialTransportWitness>>(
                SpatialTransportWitness{
                    ResolvedPnrViolationKind::RuntimeCounterexampleViolation,
                    runtimeBreaker->clauseOrdinal})
          : firstSpatialTransportWitness(candidate);
  if (!primaryWitness)
    return primaryWitness.takeError();
  if (!*primaryWitness) {
    if (candidate.hardProgressViolation() != 0)
      return repairResult(
          SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
          "route-progress dependency violation has no typed repair witness");
    return repairResult(
        SpatialExactRepairResultKind::InternalError, 0, 0, 0,
        "transport repair could not locate its primary witness");
  }
  const ResolvedPnrViolationKind primaryWitnessKind = (*primaryWitness)->kind;
  const PnrIndex primaryWitnessOrdinal = (*primaryWitness)->ordinal;
  if (primaryWitnessKind ==
          ResolvedPnrViolationKind::RuntimeCounterexampleViolation &&
      (!runtimeBreaker ||
       runtimeBreaker->clauseOrdinal != primaryWitnessOrdinal))
    return repairResult(
        SpatialExactRepairResultKind::InternalError, 0, 0, 0,
        "runtime-counterexample repair has no exact breaker branch");
  if (primaryWitnessKind !=
          ResolvedPnrViolationKind::RuntimeCounterexampleViolation &&
      runtimeBreaker)
    return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "non-runtime repair received a counterexample breaker");

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
    return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "realization PortDemand reverse index is malformed");

  const auto portDemandOwnerDecision =
      [&](PnrIndex demand) -> llvm::Expected<PnrIndex> {
    if (demand >= ports.portDemands().size())
      return repairError("route repair PortDemand is out of range");
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    if (record.kind == FrozenSpatialPortDemandKind::Compute) {
      if (record.realization >= bindings.computeDecisionCount())
        return repairError(
            "route repair PortDemand has a foreign compute owner");
      return record.realization;
    }
    if (record.realization >= bindings.memoryDecisionCount())
      return repairError("route repair PortDemand has a foreign memory owner");
    return memoryOffset + record.realization;
  };

  const auto addDecision = [&](PnrIndex decision) -> llvm::Error {
    if (decision >= decisionIncluded_.size())
      return repairError("route repair decision is out of range");
    if (!decisionIncluded_[decision]) {
      decisionIncluded_[decision] = 1;
      decisionQueue_.push_back(decision);
    }
    return llvm::Error::success();
  };
  const auto addRoutingNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= netIncluded_.size())
      return repairError("route repair logical net is out of range");
    for (PnrIndex member :
         problem.routeConstraints().equalityClosure(logicalNet)) {
      if (member >= netIncluded_.size())
        return repairError(
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
        return repairError("route repair PortDemand is out of range");
      return addDecision(bindings.portDecisionOffset() + binding.index);
    case FrozenSpatialTerminalBindingKind::GraphBoundary:
      if (binding.index >= ports.graphBoundaries().size())
        return repairError("route repair graph boundary is out of range");
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
      return repairError("route repair certificate net is out of range");
    const FrozenSpatialLogicalNet &net =
        transfers.logicalNets()[cut.logicalNet];
    if (cut.unreachableSink >= net.sinkCount)
      return repairError("route repair certificate sink is out of range");
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
        return repairError("tag-table witness is out of range");
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
      return repairError("route witness has no capacity-to-claim incidence");
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
      if (primaryWitnessOrdinal < net.sinkOffset ||
          primaryWitnessOrdinal - net.sinkOffset >= net.sinkCount)
        continue;
      if (llvm::Error error = addWitnessNet(logicalNet))
        return std::move(error);
      found = true;
      break;
    }
    if (!found)
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                          "unrouted witness is out of range");
    break;
  }
  case ResolvedPnrViolationKind::CapacityOveruse:
    if (llvm::Error error = addCapacityWitness(primaryWitnessOrdinal))
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
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
            return repairResult(SpatialExactRepairResultKind::InternalError, 0,
                                0, 0,
                                "unassigned-tag witness is no longer live");
          if (llvm::Error error = addWitnessNet(logicalNet))
            return std::move(error);
          found = true;
          break;
        }
        ++ordinal;
      }
    if (!found)
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                          "unassigned-tag witness is out of range");
    break;
  }
  case ResolvedPnrViolationKind::TagConflict: {
    const PnrIndex domain = primaryWitnessOrdinal;
    if (domain >= routing.tagContinuity().matchDomains().size())
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
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
  case ResolvedPnrViolationKind::HardProgressViolation:
    return repairResult(SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0,
                        0, "hard progress witness has no transport encoding");
  case ResolvedPnrViolationKind::ProgressProofDebt: {
    if (candidate.progress().isComputeProofDebtWitness(primaryWitnessOrdinal) &&
        !candidate.progress().computeProgress()->witnessRealizations.empty()) {
      for (PnrIndex realization : candidate.progress().computeProgress()->witnessRealizations)
        if (llvm::Error error = addDecision(realization))
          return std::move(error);
      break;
    }
    return repairResult(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "capacity proof-debt repair requires an owner-local disjunctive "
        "route or hardware proposal");
  }
  case ResolvedPnrViolationKind::RuntimeCounterexampleViolation: {
    const auto clauses = problem.constraints().resolvedNoGoods();
    const auto literals = problem.constraints().resolvedNoGoodLiterals();
    if (primaryWitnessOrdinal >= clauses.size() ||
        !candidate.runtimeCounterexampleClauseViolated(primaryWitnessOrdinal))
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                          "runtime-counterexample witness is no longer live");
    const FrozenNoGoodResolvedClause &clause = clauses[primaryWitnessOrdinal];
    if (clause.literalOffset > literals.size() ||
        clause.literalCount > literals.size() - clause.literalOffset)
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                          "runtime-counterexample literal range is malformed");
    bool hasLocalAnchor = false;
    for (const FrozenNoGoodResolvedLiteral &literal :
         literals.slice(clause.literalOffset, clause.literalCount)) {
      if (literal.kind ==
          FrozenNoGoodResolvedLiteral::Kind::SpatialMappingIdentityEquals)
        continue;
      hasLocalAnchor = true;
      if (llvm::Error error = addWitnessNet(literal.logicalNet))
        return std::move(error);
    }
    if (!hasLocalAnchor) {
      if (!runtimeBreaker ||
          runtimeBreaker->kind !=
              detail::SpatialRuntimeCounterexampleBreakerKind::MappingIdentity ||
          runtimeBreaker->bindingDecision == getInvalidPnrIndex())
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                            "identity breaker has no exact owner region");
      if (llvm::Error error = addDecision(runtimeBreaker->bindingDecision))
        return std::move(error);
    }
    break;
  }
  case ResolvedPnrViolationKind::SelectedHandshakeViolation:
    return repairResult(
        SpatialExactRepairResultKind::UnsupportedEncoding, 0, 0, 0,
        "selected-handshake repair is owned by the ordinary external-route "
        "Action domain");
  }
  for (PnrIndex logicalNet : handshakeCycleNets)
    if (llvm::Error error = addWitnessNet(logicalNet))
      return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                          0, llvm::toString(std::move(error)));
  for (const SpatialFixedTerminalCutCertificate &certificate : certificates)
    for (const SpatialFixedTerminalCutNet &cut : certificate.forcedNetCuts)
      if (llvm::Error error = addCertificateCut(cut))
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, llvm::toString(std::move(error)));
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
          return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                              0, "realization owns a foreign PortDemand");
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
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, llvm::toString(owner.takeError()));
      if (llvm::Error error = addDecision(*owner))
        return std::move(error);
      if (llvm::Error error =
              addRoutingNet(ports.portDemands()[demand].logicalNet))
        return std::move(error);
    } else {
      const PnrIndex boundary = decision - boundaryOffset;
      if (boundary >= ports.graphBoundaries().size())
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0,
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
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, "route repair relation is out of range");
      const detail::InitializerRelationRecord &record =
          relationModel.relations()[relation];
      for (const detail::InitializerRelationMember &member :
           relationModel.members(record))
        if (llvm::Error error = addDecision(member.decision))
          return std::move(error);
    }
  }
  if (affectedNets_.empty() &&
      (!runtimeBreaker ||
       runtimeBreaker->kind !=
           detail::SpatialRuntimeCounterexampleBreakerKind::MappingIdentity))
    return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
                        "transport witness has no contributing logical net");

  decisions_ = decisionQueue_;
  llvm::sort(decisions_);
  llvm::sort(affectedNets_);

  for (PnrIndex decision : decisions_)
    for (PnrIndex relation : bindings.decisionRelations(decision)) {
      if (relation >= relationIncluded_.size())
        return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0,
                            0, "route repair relation is out of range");
      if (!relationIncluded_[relation]) {
        relationIncluded_[relation] = 1;
        relations_.push_back(relation);
      }
    }
  llvm::sort(relations_);

  if (affectedNets_.size() >
      std::numeric_limits<std::uint64_t>::max() - decisions_.size())
    return repairResult(SpatialExactRepairResultKind::InternalError, 0, 0, 0,
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
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        canonicalRegionDecisionCount, 0, 0,
                        "route exact-repair region decision count regressed");
  if (canonicalRegionDecisionCount > policy.maxRegionDecisions)
    return repairResult(SpatialExactRepairResultKind::RegionTooLarge,
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
    return repairResult(SpatialExactRepairResultKind::UnsupportedEncoding,
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
        return repairResult(SpatialExactRepairResultKind::InternalError,
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
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        canonicalRegionDecisionCount, 0, 0,
                        llvm::toString(localDispositions.takeError()));
  if (localDispositions->variables().size() != affectedNets_.size())
    return repairResult(SpatialExactRepairResultKind::InternalError,
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

  if (runtimeBreaker) {
    if (runtimeBreaker->kind !=
        detail::SpatialRuntimeCounterexampleBreakerKind::MappingIdentity)
      if (llvm::Error error =
              detail::addSpatialRuntimeCounterexampleBreakerConstraint(
                  model, candidate, bindings, variables, decisionVariables_,
                  *localDispositions, *runtimeBreaker))
        return repairResult(SpatialExactRepairResultKind::InternalError,
                            canonicalRegionDecisionCount, 0, 0,
                            llvm::toString(std::move(error)));
  }

  for (auto [local, decision] : llvm::enumerate(decisions_)) {
    if (decision < portOffset || decision >= boundaryOffset)
      continue;
    auto owner = portDemandOwnerDecision(decision - portOffset);
    if (!owner)
      return repairResult(SpatialExactRepairResultKind::InternalError,
                          canonicalRegionDecisionCount, 0, 0,
                          llvm::toString(owner.takeError()));
    if (*owner >= decisionVariables_.size() || decisionVariables_[*owner] < 0)
      return repairResult(SpatialExactRepairResultKind::InternalError,
                          canonicalRegionDecisionCount, 0, 0,
                          "route repair omitted a PortDemand owner decision");
    mutationParentLocals[local] = decisionVariables_[*owner];
  }

  auto mutationCount = detail::addExactRepairMutationCountObjective(
      model, variables, candidate, bindings, decisions_, mutationParentLocals,
      localDispositions->variables(), localDispositions->currentValues());
  if (!mutationCount)
    return mutationCount.takeError();
  if (runtimeBreaker &&
      runtimeBreaker->kind ==
          detail::SpatialRuntimeCounterexampleBreakerKind::MappingIdentity)
    model.AddGreaterOrEqual(mutationCount->objective, 1);

  bool currentAssignmentSatisfiesCertificates = true;
  for (const SpatialFixedTerminalCutCertificate &certificate : certificates) {
    auto encoded = detail::addSpatialFixedTerminalCutEscapeConstraint(
        model, candidate, bindings, variables, decisionVariables_,
        legalValueOffsets_, legalValues_, *localDispositions, certificate,
        routeCutBlockedTraversals_, routeCutReachableEndpoints_,
        routeCutWorklist_);
    if (!encoded)
      return repairResult(SpatialExactRepairResultKind::InternalError,
                          canonicalRegionDecisionCount, 0, 0,
                          llvm::toString(encoded.takeError()));
    if (!encoded->encoded)
      return repairResult(SpatialExactRepairResultKind::InternalError,
                          canonicalRegionDecisionCount, 0, 0,
                          "retained fixed-terminal cut is outside its rebuilt "
                          "repair region");
    currentAssignmentSatisfiesCertificates &= encoded->currentAssignmentEscapes;
  }

  if (pendingRegionDecisionCount_ !=
      canonicalRegionDecisionCount - accountedRegionDecisionCount_)
    return repairResult(
        SpatialExactRepairResultKind::InternalError,
        canonicalRegionDecisionCount, 0, 0,
        "route exact-repair work disagrees with its closed model");
  if (llvm::Error error = consumePendingRegionDecisions())
    return error;
  accountedRegionDecisions_ = decisionIncluded_;
  accountedRegionNets_ = netIncluded_;

  if (llvm::Error error =
          actionExecutor_.prepare(candidate, workLedger_, executionControl_))
    return repairResult(SpatialExactRepairResultKind::InternalError,
                        canonicalRegionDecisionCount, 0, 0,
                        llvm::toString(std::move(error)));
  std::uint64_t solverCalls = 0;
  std::uint64_t logicalSolverCalls = 0;
  std::uint64_t assignmentOrdinal = 0;
  std::uint64_t lastActionCount = 0;
  std::uint64_t maximumObservedRegionDecisionCount =
      canonicalRegionDecisionCount;
  bool sawUnknownAssignment = false;
  bool sawRoutingIncomplete = false;
  bool proveCurrentAssignment =
      currentAssignmentSatisfiesCertificates &&
      primaryWitnessKind !=
          ResolvedPnrViolationKind::RuntimeCounterexampleViolation;
  std::vector<std::int64_t> currentAssignment;
  currentAssignment.reserve(canonicalVariables.size());
  for (PnrIndex decision : decisions_) {
    auto current =
        detail::currentExactRepairBindingChoice(candidate, bindings, decision);
    if (!current)
      return repairResult(SpatialExactRepairResultKind::InternalError,
                          canonicalRegionDecisionCount, 0, 0,
                          llvm::toString(current.takeError()));
    currentAssignment.push_back(static_cast<std::int64_t>(*current));
  }
  currentAssignment.insert(currentAssignment.end(),
                           localDispositions->currentValues().begin(),
                           localDispositions->currentValues().end());

  const auto executedResult = [&](SpatialExactRepairResultKind kind,
                                  std::string detail = {}) {
    return repairResult(
        kind, maximumObservedRegionDecisionCount, solverCalls, lastActionCount,
        std::move(detail), actionExecutor_.endpointExpansionCount(),
        actionExecutor_.negotiationIterationCount(), logicalSolverCalls);
  };

  const dse::ObjectiveVector initialObjective =
      actionExecutor_.currentObjective();
  const std::uint64_t initialProofDebt =
      candidate.progressProofDebtWitnessCount();
  const auto hardTransportClosed = [&]() {
    return candidate.unroutedObligationCount() == 0 &&
           candidate.routeCapacityOveruse() == 0 &&
           candidate.tagResidentCapacityOveruse() == 0 &&
           candidate.tagUnassignedCount() == 0 &&
           candidate.tagConflictCount() == 0 &&
           candidate.hardProgressViolation() == 0 &&
           candidate.runtimeCounterexampleViolation() == 0 &&
           candidate.progressProofDebtWitnessCount() <= initialProofDebt;
  };
  std::optional<SpatialRuntimeLiteralBreaker> routeBreaker;
  if (runtimeBreaker &&
      runtimeBreaker->kind ==
          detail::SpatialRuntimeCounterexampleBreakerKind::NetTraversal) {
    auto required = detail::spatialRuntimeTraversalRequiresRouteCut(
        candidate, *runtimeBreaker);
    if (!required)
      return repairResult(SpatialExactRepairResultKind::InternalError,
                          canonicalRegionDecisionCount, 0, 0,
                          llvm::toString(required.takeError()));
    if (*required)
      routeBreaker = SpatialRuntimeLiteralBreaker{
          runtimeBreaker->clauseOrdinal,
          runtimeBreaker->clauseLocalLiteralOrdinal};
  }

  // A legal route closure may be worse than the current objective. Keep the
  // best such assignment outside the mutable CP-SAT model so search can look
  // for a better legal closure without turning the fallback into a proof of
  // illegality.
  std::vector<SpatialMappingAction> bestLegalFallbackActions;
  std::vector<std::int64_t> bestLegalFallbackAssignment;
  std::optional<dse::ObjectiveVector> bestLegalFallbackObjective;

  const auto commitBestLegalFallback =
      [&](std::string detail) -> llvm::Expected<SpatialExactRepairResult> {
    if (!bestLegalFallbackObjective || bestLegalFallbackActions.empty() ||
        bestLegalFallbackAssignment.size() != canonicalVariables.size())
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair lost its legal fallback");

    actions_ = bestLegalFallbackActions;
    auto fallbackProbe = actionExecutor_.probeBatch(
        candidate, actions_, SpatialActionExecutionContext::ExactRepair,
        policy.maxRegionDecisions - decisions_.size(), routeBreaker);
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
      SpatialExactRepairResultKind kind =
          SpatialExactRepairResultKind::InternalError;
      if (!hasUnhandled && transitionFailure) {
        if (*transitionFailure == SpatialActionTransitionFailureKind::WorkLimit)
          kind = SpatialExactRepairResultKind::UnknownBudgetExhausted;
        else if (*transitionFailure ==
                 SpatialActionTransitionFailureKind::Interrupted)
          kind = SpatialExactRepairResultKind::TimedOut;
      }
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
    auto witnessLive = spatialTransportWitnessIsLive(candidate, **primaryWitness);
    if (!witnessLive) {
      llvm::Error error = witnessLive.takeError();
      if (llvm::Error discardError = fallbackProbe->discard())
        error = llvm::joinErrors(std::move(error), std::move(discardError));
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(std::move(error)));
    }
    if (!assignmentRealized || fallbackProbe->isSemanticNoop() ||
        candidate.atomicCapacityOveruse() != 0 || !hardTransportClosed() ||
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
    auto committedWitness = spatialTransportWitnessIsLive(candidate, **primaryWitness);
    if (!committedWitness)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(committedWitness.takeError()));
    if (*committedWitness)
      return executedResult(
          SpatialExactRepairResultKind::InternalError,
          "committed route-repair fallback restored its primary witness");
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

  while (logicalSolverCalls < solverCallLimit) {
    if (executionControl_.stopRequested())
      return executedResult(
          SpatialExactRepairResultKind::TimedOut,
          "route exact repair reached its deadline between assignments");
    auto solved =
        proveCurrentAssignment
            ? detail::solveFixedCpSatAssignment(
                  model.Build(), canonicalVariables, currentAssignment,
                  mutationCount->objective.index(),
                  solverCallLimit - logicalSolverCalls, solverSeed, workLedger_,
                  mutationCount->proofPriorityVariables, executionControl_)
            : detail::solveCanonicalCpSat(model.Build(), canonicalVariables,
                                          mutationCount->objective.index(),
                                          solverCallLimit - logicalSolverCalls,
                                          solverSeed, workLedger_,
                                          mutationCount->proofPriorityVariables,
                                          executionControl_);
    if (!solved)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            llvm::toString(solved.takeError()));
    solverCalls += solved->solverCalls;
    logicalSolverCalls += solved->logicalSolverCalls;
    if (solved->kind == detail::CpSatCanonicalResultKind::Interrupted)
      return executedResult(
          SpatialExactRepairResultKind::TimedOut,
          "route exact repair reached its deadline between solver calls");
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
      if (sawRoutingIncomplete)
        return executedResult(
            SpatialExactRepairResultKind::RoutingIncomplete,
            "bounded route-repair assignments were exhausted after at least "
            "one negotiated route remained incomplete");
      return executedResult(
          SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
          "bounded route-repair assignments were exhausted without an "
          "exact route closure");
    }
    if (solved->kind != detail::CpSatCanonicalResultKind::Assignment) {
      if (bestLegalFallbackObjective) {
        return commitBestLegalFallback(
            "committed the best legal route-repair fallback after an "
            "incomplete exact proof");
      }
      return executedResult(
          SpatialExactRepairResultKind::UnknownBudgetExhausted,
          (llvm::Twine("route exact repair incomplete: ") +
           detail::cpSatCanonicalResultKindSpelling(solved->kind))
              .str());
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
      auto localOption =
          localDispositions->selectedOption(local, selectedDisposition);
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
    if (!runtimeBreaker)
      actions_.push_back(
          SpatialTransportRoutingAction{SpatialWitnessRegionRoutingAction{
              primaryWitnessKind, primaryWitnessOrdinal}});
    else if (runtimeBreaker->kind ==
                 detail::SpatialRuntimeCounterexampleBreakerKind::
                     NetTraversal &&
             routeBreaker) {
      auto literal = detail::resolveSpatialRuntimeCounterexampleBreaker(
          problem, *runtimeBreaker);
      if (!literal)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(literal.takeError()));
      if ((*literal)->sink) {
        const FrozenSpatialLogicalNet &net =
            problem.transfers().logicalNets()[(*literal)->logicalNet];
        if (*(*literal)->sink >= net.sinkCount)
          return executedResult(
              SpatialExactRepairResultKind::InternalError,
              "traversal breaker sink is outside its logical net");
        actions_.push_back(
            SpatialTransportRoutingAction{SpatialSingleSinkRoutingAction{
                (*literal)->logicalNet, net.sinkOffset + *(*literal)->sink}});
      } else {
        actions_.push_back(
            SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{
                (*literal)->logicalNet,
                SpatialWholeNetDispositionKind::External}});
      }
    }
    if (runtimeBreaker &&
        runtimeBreaker->kind ==
            detail::SpatialRuntimeCounterexampleBreakerKind::NetTag) {
      auto literal = detail::resolveSpatialRuntimeCounterexampleBreaker(
          problem, *runtimeBreaker);
      if (!literal)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              llvm::toString(literal.takeError()));
      if (!runtimeBreaker->physicalTagValue)
        return executedResult(SpatialExactRepairResultKind::InternalError,
                              "tag breaker has no exact replacement value");
      actions_.push_back(SpatialTransportRoutingAction{
          SpatialPhysicalTagAction{(*literal)->logicalNet, (*literal)->target,
                                   *runtimeBreaker->physicalTagValue}});
    }
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
          llvm::json::Array selectedDispositions;
          for (PnrIndex local = 0; local < affectedNets_.size(); ++local) {
            const std::int64_t selected =
                solved->assignment[decisions_.size() + local];
            selectedDispositions.push_back(llvm::json::Object{
                {"logical_net", affectedNets_[local]},
                {"choice", selected},
                {"current", localDispositions->currentValues()[local]},
                {"external", localDispositions->externalValue(local)}});
          }
          fields["search_scope"] = "route_exact_repair";
          fields["restart"] = restartOrdinal;
          fields["assignment"] = assignmentOrdinal;
          fields["decisions"] = std::move(selectedDecisions);
          fields["local_dispositions"] = std::move(selectedDispositions);
        });

    auto attemptedProbe = actionExecutor_.probeBatch(
        candidate, actions_, SpatialActionExecutionContext::ExactRepair,
        policy.maxRegionDecisions - decisions_.size(), routeBreaker);
    std::optional<SpatialActionProbe> activeProbe;
    if (attemptedProbe)
      activeProbe.emplace(std::move(*attemptedProbe));
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
    bool selectedHandshakeCycle = false;
    bool handshakeCycleRequiresRegionExpansion = false;
    std::vector<PnrIndex> handshakeCycleLogicalNets;
    std::vector<SpatialTraversalRouteCut> handshakeCycleRouteCuts;
    std::vector<SpatialHandshakeCycleTagSelection>
        handshakeCycleTagSelections;
    std::uint64_t handshakeCycleOutsideRepairRegionCount = 0;
    routeCutCertificate_ = {};
    if (!attemptedProbe) {
      std::string detail;
      std::optional<SpatialActionTransitionFailureKind> transitionFailure;
      llvm::Error unhandled = llvm::handleErrors(
          attemptedProbe.takeError(),
          [&](const SpatialPathFinderClosureFailure &failure) -> llvm::Error {
            if (failure.kind() ==
                SpatialPathFinderClosureFailure::Kind::RegionalLimit) {
              if (failure.regionalLogicalNetCount() >
                  std::numeric_limits<std::uint64_t>::max() - decisions_.size())
                return repairError(
                    "route conflict closure decision count overflows");
              regionalLimit = true;
              rejectedRegionDecisionCount =
                  decisions_.size() + failure.regionalLogicalNetCount();
              return llvm::Error::success();
            }
            if (failure.kind() != SpatialPathFinderClosureFailure::Kind::
                                      FixedTerminalCapacityCut) {
              if (failure.kind() ==
                      SpatialPathFinderClosureFailure::Kind::NonClosure ||
                  failure.kind() ==
                      SpatialPathFinderClosureFailure::Kind::NoProgress) {
                routeWorkUnknown = true;
                sawUnknownAssignment = true;
                return llvm::Error::success();
              }
              if (failure.kind() == SpatialPathFinderClosureFailure::Kind::
                                        SelectedCombinationalHandshakeCycle) {
                sawRoutingIncomplete = true;
                selectedHandshakeCycle = true;
                handshakeCycleLogicalNets.assign(
                    failure.handshakeCycleLogicalNets().begin(),
                    failure.handshakeCycleLogicalNets().end());
                handshakeCycleRouteCuts.assign(
                    failure.handshakeCycleRouteCuts().begin(),
                    failure.handshakeCycleRouteCuts().end());
                handshakeCycleTagSelections.assign(
                    failure.handshakeCycleTagSelections().begin(),
                    failure.handshakeCycleTagSelections().end());
                for (const SpatialTraversalRouteCut &cut :
                     handshakeCycleRouteCuts)
                  if (cut.logicalNet >= netIncluded_.size())
                    return repairError(
                        "handshake cycle route cut has a foreign logical net");
                for (const SpatialHandshakeCycleTagSelection &selection :
                     handshakeCycleTagSelections)
                  if (selection.logicalNet >= netIncluded_.size())
                    return repairError(
                        "handshake cycle tag selection has a foreign logical net");
                for (PnrIndex logicalNet : handshakeCycleLogicalNets) {
                  if (logicalNet >= netIncluded_.size())
                    return repairError(
                        "handshake cycle has a foreign logical net");
                  if (netIncluded_[logicalNet] != 0)
                    continue;
                  ++handshakeCycleOutsideRepairRegionCount;
                  const auto found = llvm::lower_bound(
                      handshakeCycleNets, logicalNet);
                  if (found == handshakeCycleNets.end() ||
                      *found != logicalNet) {
                    handshakeCycleNets.insert(found, logicalNet);
                    handshakeCycleRequiresRegionExpansion = true;
                  }
                }
                return llvm::Error::success();
              }
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
          *transitionFailure == SpatialActionTransitionFailureKind::Interrupted)
        return executedResult(SpatialExactRepairResultKind::TimedOut,
                              std::move(detail));
      if (transitionFailure &&
          *transitionFailure == SpatialActionTransitionFailureKind::WorkLimit) {
        routeWorkUnknown = true;
        sawUnknownAssignment = true;
      }
      if (selectedHandshakeCycle)
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Summary,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              llvm::json::Array logicalNets;
              for (PnrIndex logicalNet : handshakeCycleLogicalNets)
                logicalNets.push_back(logicalNet);
              llvm::json::Array routeCuts;
              for (const SpatialTraversalRouteCut &cut :
                   handshakeCycleRouteCuts)
                routeCuts.push_back(llvm::json::Object{
                    {"logical_net", cut.logicalNet},
                    {"traversal", cut.traversal}});
              llvm::json::Array tagSelections;
              for (const SpatialHandshakeCycleTagSelection &selection :
                   handshakeCycleTagSelections) {
                llvm::json::Object tagSelection{
                    {"logical_net", selection.logicalNet},
                    {"segment_ordinal", selection.segmentOrdinal},
                    {"match_domain", selection.matchDomain},
                    {"tag_encoding_width_bits", selection.tagWidthBits},
                    {"tag_assigned", selection.value.has_value()}};
                if (selection.value) {
                  llvm::SmallString<32> tagValue;
                  selection.value->toStringUnsigned(tagValue, 10);
                  tagSelection["tag_value_bit_width"] =
                      selection.value->getBitWidth();
                  tagSelection["tag_value"] = std::string(tagValue);
                }
                tagSelections.push_back(std::move(tagSelection));
              }
              llvm::json::Array cycleLocalDispositions;
              for (PnrIndex logicalNet : handshakeCycleLogicalNets) {
                const auto local =
                    localDispositions->localForLogicalNet(logicalNet);
                if (!local)
                  continue;
                const std::int64_t external =
                    localDispositions->externalValue(*local);
                const std::int64_t selected = solved->assignment[
                    decisions_.size() + static_cast<std::size_t>(*local)];
                const bool hasRegisterFifoAlternative = llvm::any_of(
                    localDispositions->legalValues(*local),
                    [&](std::int64_t value) { return value != external; });
                cycleLocalDispositions.push_back(llvm::json::Object{
                    {"logical_net", logicalNet},
                    {"choice", selected},
                    {"current", localDispositions->currentValues()[*local]},
                    {"external", external},
                    {"selected_register_fifo", selected != external},
                    {"register_fifo_alternative_available",
                     hasRegisterFifoAlternative},
                });
              }
              fields["search_scope"] = "route_exact_repair";
              fields["operation"] = "selected_handshake_cycle_region_scope";
              fields["restart"] = restartOrdinal;
              fields["assignment"] = assignmentOrdinal;
              fields["cycle_contributing_logical_net_count"] =
                  handshakeCycleLogicalNets.size();
              fields["outside_repair_region_logical_net_count"] =
                  handshakeCycleOutsideRepairRegionCount;
              fields["region_expanded"] =
                  handshakeCycleRequiresRegionExpansion;
              fields["cycle_contributing_logical_nets"] =
                  std::move(logicalNets);
              fields["cycle_contributing_route_cut_count"] =
                  handshakeCycleRouteCuts.size();
              fields["cycle_contributing_route_cuts"] = std::move(routeCuts);
              fields["cycle_contributing_tag_segment_count"] =
                  handshakeCycleTagSelections.size();
              fields["cycle_contributing_tag_segments"] =
                  std::move(tagSelections);
              fields["cycle_contributing_local_dispositions"] =
                  std::move(cycleLocalDispositions);
            });
    }
    SpatialActionProbe *probe = activeProbe ? &*activeProbe : nullptr;
    if (!probe) {
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
      if (assignmentRealized)
        for (PnrIndex local = 0; local < affectedNets_.size(); ++local) {
          auto selected = localDispositions->selectedOption(
              local, solved->assignment[decisions_.size() + local]);
          if (!selected) {
            if (llvm::Error error = probe->discard())
              return executedResult(SpatialExactRepairResultKind::InternalError,
                                    llvm::toString(std::move(error)));
            return executedResult(SpatialExactRepairResultKind::InternalError,
                                  llvm::toString(selected.takeError()));
          }
          const PnrIndex actual =
              candidate.registerFifoTransfer(affectedNets_[local]);
          const PnrIndex expected =
              *selected ? **selected : getInvalidPnrIndex();
          if (actual != expected) {
            assignmentRealized = false;
            break;
          }
        }
      auto primaryWitnessLive =
          spatialTransportWitnessIsLive(candidate, **primaryWitness);
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
      const bool globalTransportClosed = hardTransportClosed();
      // ExactRegional routing closes the affected region. Independent
      // transport witnesses can remain outside it, so requiring global
      // closure here would discard every repair of just one component.
      // Such partial repairs must strictly improve the existing global
      // ordering; only a complete transport closure may use the QoR fallback.
      const bool regionalProgress =
          selectedRankImproved && candidate.hardProgressViolation() == 0 &&
          candidate.progressProofDebtWitnessCount() <= initialProofDebt;
      const bool assignmentLegal =
          assignmentRealized && !probe->isSemanticNoop() &&
          candidate.atomicCapacityOveruse() == 0 &&
          (globalTransportClosed || regionalProgress) &&
          !*primaryWitnessLive;
      if (!assignmentLegal) {
        // The negotiated route selection failed acceptance; another route
        // under these terminals may still close. Its search exclusion cannot
        // contribute a proof that the fixed-boundary region is infeasible.
        sawRoutingIncomplete = true;
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
              fields["semantic_change"] = !probe->isSemanticNoop();
              fields["atomic_capacity_overuse"] =
                  candidate.atomicCapacityOveruse();
              fields["hard_transport_closed"] = globalTransportClosed;
              fields["regional_progress"] = regionalProgress;
              fields["hard_progress_violation"] =
                  candidate.hardProgressViolation();
              fields["progress_proof_debt"] =
                  candidate.progressProofDebtWitnessCount();
              fields["initial_progress_proof_debt"] = initialProofDebt;
              const auto &compute = *candidate.progress().computeProgress();
              fields["compute_progress_proof_debt"] =
                  compute.objective.proofDebtWitnessCount;
              fields["compute_progress_reason"] =
                  mapping::mappingProgressClosureReasonSpelling(
                      compute.closure.reason);
              llvm::json::Array computeWitness;
              for (PnrIndex realization : compute.witnessRealizations)
                computeWitness.push_back(realization);
              fields["compute_progress_realizations"] =
                  std::move(computeWitness);
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
        auto committedWitness =
            spatialTransportWitnessIsLive(candidate, **primaryWitness);
        if (!committedWitness)
          return executedResult(SpatialExactRepairResultKind::InternalError,
                                llvm::toString(committedWitness.takeError()));
        if (*committedWitness)
          return executedResult(
              SpatialExactRepairResultKind::InternalError,
              "committed route repair restored its primary witness");
        loom::mapping_debug::emit(
            loom::mapping_debug::Level::Decision,
            loom::mapping_debug::Stage::SpatialPnr,
            loom::mapping_debug::Event::ActionOutcome,
            [&](llvm::json::Object &fields) {
              fields["search_scope"] = "route_exact_repair";
              fields["restart"] = restartOrdinal;
              fields["assignment"] = assignmentOrdinal;
              fields["accepted"] = true;
              fields["hard_transport_closed"] = globalTransportClosed;
              fields["remaining_unrouted_obligations"] =
                  candidate.unroutedObligationCount();
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
              error =
                  llvm::joinErrors(std::move(error), std::move(discardError));
            return executedResult(SpatialExactRepairResultKind::InternalError,
                                  llvm::toString(std::move(error)));
          }
          replaceFallback = *comparison < 0;
          if (*comparison == 0 &&
              std::lexicographical_compare(solved->assignment.begin(),
                                           solved->assignment.end(),
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
          fields["handshake_cycle_region_expansion"] =
              handshakeCycleRequiresRegionExpansion;
          fields["objective_only_rejection"] = objectiveOnlyRejection;
          fields["legal_fallback_recorded"] =
              objectiveOnlyRejection && bestLegalFallbackObjective.has_value();
          fields["cut_logical_net_count"] =
              routeCutCertificate_.forcedNetCuts.size();
        });
    if (!rejectAssignment)
      return executedResult(SpatialExactRepairResultKind::InternalError,
                            "route repair lost its assignment outcome");
    if (handshakeCycleRequiresRegionExpansion) {
      requiresRegionExpansion = true;
      return executedResult(
          SpatialExactRepairResultKind::RegionInfeasibleUnderFixedBoundary,
          "selected handshake cycle reaches a route outside the bounded "
          "repair region");
    }
    if (regionalLimit) {
      if (bestLegalFallbackObjective)
        return commitBestLegalFallback(
            "committed the best legal route-repair fallback before the "
            "regional closure limit");
      return repairResult(SpatialExactRepairResultKind::RegionTooLarge,
                          rejectedRegionDecisionCount, solverCalls,
                          lastActionCount,
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
