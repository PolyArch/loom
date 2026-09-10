#include "SpatialTransportWitness.h"

#include "PnR/SpatialCandidateState.h"
#include "SpatialExactRepairInternal.h"
#include "SpatialProgressIndex.h"

namespace loom::pnr::detail {

llvm::Expected<std::optional<SpatialTransportWitness>>
firstSpatialTransportWitness(const SpatialCandidateState &candidate) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &transfers = problem.transfers();
  const auto &routing = problem.routing();
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet)
    if (!candidate.usesRegisterFifo(logicalNet) &&
        !candidate.routeTree(logicalNet).isRouted())
      return SpatialTransportWitness{
          ResolvedPnrViolationKind::UnroutedObligation,
          transfers.logicalNets()[logicalNet].sinkOffset};

  const PnrIndex capacityCount =
      static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
  for (PnrIndex capacity = 0;
       capacity < problem.resources().capacityDimensions().size(); ++capacity)
    if (candidate.routeCapacityOveruseRaw(capacity) != 0)
      return SpatialTransportWitness{ResolvedPnrViolationKind::CapacityOveruse,
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
    return SpatialTransportWitness{ResolvedPnrViolationKind::CapacityOveruse,
                                   *ordinal};
  }

  PnrIndex globalSegment = 0;
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet)
    for (const auto &value : candidate.tagValues(logicalNet)) {
      if (!value)
        return SpatialTransportWitness{ResolvedPnrViolationKind::TagUnassigned,
                                       globalSegment};
      if (globalSegment == getPnrIndexMax())
        return repairError("Physical Tag segment ordinal overflows");
      ++globalSegment;
    }
  for (PnrIndex domain = 0;
       domain < routing.tagContinuity().matchDomains().size(); ++domain)
    if (candidate.tagDomainConflictCount(domain) != 0)
      return SpatialTransportWitness{ResolvedPnrViolationKind::TagConflict,
                                     domain};

  if (const auto owner = candidate.progress().firstCapacityShortfallOwner())
    return SpatialTransportWitness{ResolvedPnrViolationKind::ProgressProofDebt,
                                   *owner};
  if (const auto owner = candidate.progress().firstCapacityProofDebtOwner())
    return SpatialTransportWitness{ResolvedPnrViolationKind::ProgressProofDebt,
                                   *owner};
  if (const auto witness = candidate.progress().computeProofDebtWitness())
    return SpatialTransportWitness{ResolvedPnrViolationKind::ProgressProofDebt,
                                   *witness};
  if (const auto clause = candidate.firstRuntimeCounterexampleViolation())
    return SpatialTransportWitness{
        ResolvedPnrViolationKind::RuntimeCounterexampleViolation, *clause};
  if (candidate.selectedHandshakeViolation() != 0)
    return SpatialTransportWitness{
        ResolvedPnrViolationKind::SelectedHandshakeViolation, 0};

  if (candidate.unroutedObligationCount() != 0 ||
      candidate.routeCapacityOveruse() != 0 ||
      candidate.tagResidentCapacityOveruse() != 0 ||
      candidate.tagUnassignedCount() != 0 ||
      candidate.tagConflictCount() != 0 ||
      candidate.hardProgressViolation() != 0 ||
      candidate.progressProofDebtWitnessCount() != 0 ||
      candidate.runtimeCounterexampleViolation() != 0 ||
      candidate.selectedHandshakeViolation() != 0)
    return repairError(
        "transport violation aggregates have no canonical witness");
  return std::optional<SpatialTransportWitness>();
}

llvm::Expected<bool>
spatialTransportWitnessIsLive(const SpatialCandidateState &candidate,
                              SpatialTransportWitness witness) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &transfers = problem.transfers();
  const auto &routing = problem.routing();
  switch (witness.kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      if (witness.ordinal >= net.sinkOffset &&
          witness.ordinal - net.sinkOffset < net.sinkCount)
        return !candidate.usesRegisterFifo(logicalNet) &&
               candidate.routeTree(logicalNet).isUnrouted();
    }
    return repairError("unrouted witness is out of range");
  case ResolvedPnrViolationKind::CapacityOveruse: {
    const PnrIndex capacityCount =
        static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
    if (witness.ordinal < capacityCount)
      return candidate.routeCapacityOveruseRaw(witness.ordinal) != 0;
    const PnrIndex domain = witness.ordinal - capacityCount;
    if (domain >= routing.tagContinuity().matchDomains().size())
      return repairError("resident-row witness is out of range");
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
          return repairError("Physical Tag segment ordinal overflows");
        ++ordinal;
      }
    return repairError("unassigned-tag witness is out of range");
  }
  case ResolvedPnrViolationKind::TagConflict:
    if (witness.ordinal >= routing.tagContinuity().matchDomains().size())
      return repairError("tag-conflict witness is out of range");
    return candidate.tagDomainConflictCount(witness.ordinal) != 0;
  case ResolvedPnrViolationKind::HardProgressViolation:
    return repairError("hard progress witness has no transport encoding");
  case ResolvedPnrViolationKind::ProgressProofDebt:
    if (witness.ordinal == problem.progressIndex().finiteBufferOwners().size())
      return candidate.progress().isComputeProofDebtWitness(witness.ordinal);
    if (witness.ordinal >= problem.progressIndex().finiteBufferOwners().size())
      return repairError("capacity proof-debt witness is out of range");
    return candidate.progress().capacityProofDebtOwner(witness.ordinal) ||
           candidate.progress().capacityShortfallOwner(witness.ordinal);
  case ResolvedPnrViolationKind::RuntimeCounterexampleViolation:
    if (witness.ordinal >= problem.constraints().resolvedNoGoods().size())
      return repairError("runtime-counterexample witness is out of range");
    return candidate.runtimeCounterexampleClauseViolated(witness.ordinal);
  case ResolvedPnrViolationKind::SelectedHandshakeViolation:
    if (witness.ordinal != 0)
      return repairError("selected-handshake witness is out of range");
    return candidate.selectedHandshakeViolation() != 0;
  }
  llvm_unreachable("unknown Spatial transport witness kind");
}

} // namespace loom::pnr::detail
