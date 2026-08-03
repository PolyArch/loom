#include "PnR/SpatialCandidateInitializer.h"

#include "llvm/Support/Error.h"

#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace {

llvm::Error initializerError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate initialization: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

bool hasUncompiledRelations(const FrozenConstraintIndex &constraints) {
  for (std::size_t ordinal = 0;
       ordinal != FrozenConstraintIndex::projectionCount; ++ordinal) {
    const auto projection =
        ::mapping::symbolizeSpatialConstraintProjection(ordinal);
    if (!projection)
      return true;
    const FrozenConstraintShard &shard = constraints.shard(*projection);
    if (!shard.equalityClasses().empty() || !shard.disjointGroups().empty())
      return true;
  }
  return false;
}

} // namespace

llvm::Expected<SpatialCandidateStateHandle>
loom::pnr::createCanonicalSpatialCandidate(
    FrozenSpatialPnrProblemHandle problem) {
  if (!problem)
    return initializerError("FrozenSpatialPnrProblem owner is null");
  if (hasUncompiledRelations(problem->constraints()))
    return initializerError(
        "hard equality or disjointness requires relation propagation");

  const FrozenSpatialRealizationIndex &realizations = problem->realizations();
  const FrozenSpatialPortIndex &ports = problem->ports();
  const FrozenSpatialHandshakeIndex &handshake = problem->handshake();

  std::vector<SpatialComputeBindingSelection> computeBindings;
  computeBindings.reserve(realizations.computeRealizations().size());
  for (const FrozenSpatialComputeRealization &realization :
       realizations.computeRealizations()) {
    if (realization.placementCount == 0)
      return initializerError("compute realization has an empty domain");
    const PnrIndex placement = realization.placementOffset;
    const FrozenSpatialComputePlacement &placementRecord =
        realizations.computePlacements()[placement];
    if (placementRecord.contextCount == 0)
      return initializerError("compute placement has an empty context domain");
    computeBindings.push_back({placement, placementRecord.contextOffset});
  }

  std::vector<SpatialMemoryBindingSelection> memoryBindings;
  memoryBindings.reserve(realizations.memoryRealizations().size());
  for (const FrozenSpatialMemoryRealization &realization :
       realizations.memoryRealizations()) {
    if (realization.placementCount == 0)
      return initializerError("memory realization has an empty domain");
    memoryBindings.push_back({realization.placementOffset});
  }

  std::vector<PnrIndex> portAttachments;
  portAttachments.reserve(ports.portDemands().size());
  for (const FrozenSpatialPortDemand &demand : ports.portDemands()) {
    const PnrIndex placement =
        demand.kind == FrozenSpatialPortDemandKind::Compute
            ? computeBindings[demand.realization].placement
            : memoryBindings[demand.realization].placement;
    const PnrIndex ownerOffset =
        demand.kind == FrozenSpatialPortDemandKind::Compute
            ? realizations.computeRealizations()[demand.realization]
                  .placementOffset
            : realizations.memoryRealizations()[demand.realization]
                  .placementOffset;
    const PnrIndex localPlacement = placement - ownerOffset;
    if (localPlacement >= demand.placementDomainCount)
      return initializerError(
          "PortDemand has no domain for its canonical placement");
    const FrozenSpatialPortPlacementDomain &domain =
        ports.placementDomains()[demand.placementDomainOffset + localPlacement];
    if (domain.attachmentOptionCount == 0)
      return initializerError("PortDemand has an empty attachment domain");
    portAttachments.push_back(domain.attachmentOptionOffset);
  }

  std::vector<PnrIndex> graphBoundaryAttachments;
  graphBoundaryAttachments.reserve(ports.graphBoundaries().size());
  for (const FrozenSpatialGraphBoundary &boundary : ports.graphBoundaries()) {
    if (boundary.attachmentOptionCount == 0)
      return initializerError("graph boundary has an empty attachment domain");
    graphBoundaryAttachments.push_back(boundary.attachmentOptionOffset);
  }

  std::vector<PnrIndex> memoryOperationPlans(realizations.memoryActors().size(),
                                             getInvalidPnrIndex());
  for (PnrIndex realizationOrdinal = 0;
       realizationOrdinal < realizations.memoryRealizations().size();
       ++realizationOrdinal) {
    const FrozenSpatialMemoryRealization &realization =
        realizations.memoryRealizations()[realizationOrdinal];
    const PnrIndex placement = memoryBindings[realizationOrdinal].placement;
    const PnrIndex domainOffset =
        handshake.memoryPlacementDomainOffsets()[placement];
    for (PnrIndex localActor = 0; localActor < realization.actorCount;
         ++localActor) {
      const FrozenSpatialMemoryOperationHandshakeDomain &domain =
          handshake.memoryOperationDomains()[domainOffset + localActor];
      if (domain.planCount == 0)
        return initializerError("memory actor has an empty operation domain");
      memoryOperationPlans[realization.actorOffset + localActor] =
          domain.planOffset;
    }
  }

  return SpatialCandidateState::create(
      std::move(problem), {computeBindings, memoryBindings, portAttachments,
                           graphBoundaryAttachments, memoryOperationPlans});
}
