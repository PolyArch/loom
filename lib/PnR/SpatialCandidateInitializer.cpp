#include "PnR/SpatialCandidateInitializer.h"

#include "InitializerRelationSolver.h"
#include "SpatialBindingRelationModel.h"

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

} // namespace

llvm::Expected<SpatialCandidateStateHandle>
loom::pnr::createCanonicalSpatialCandidate(
    FrozenSpatialPnrProblemHandle problem) {
  if (!problem)
    return initializerError("FrozenSpatialPnrProblem owner is null");

  const detail::SpatialBindingRelationModel &bindingRelations =
      problem->bindingRelations();
  if (const auto deferred = bindingRelations.deferredProjection())
    return initializerError(
        "hard equality or disjointness for projection '" +
        ::mapping::stringifySpatialConstraintProjection(*deferred) +
        "' requires its owning decision model");

  detail::InitializerRelationSolver relationSolver(
      bindingRelations.relations());
  auto relationChoices = relationSolver.solveCanonical(
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed);
  if (!relationChoices)
    return relationChoices.takeError();

  const FrozenSpatialRealizationIndex &realizations = problem->realizations();
  const FrozenSpatialPortIndex &ports = problem->ports();
  const FrozenSpatialHandshakeIndex &handshake = problem->handshake();

  std::vector<SpatialComputeBindingSelection> computeBindings;
  computeBindings.reserve(realizations.computeRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.computeRealizations().size(); ++realization) {
    const auto choices = bindingRelations.computeChoices(realization);
    const PnrIndex selected = relationChoices->choices[realization];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign compute choice");
    computeBindings.push_back(
        {choices[selected].placement, choices[selected].instructionContext});
  }

  std::vector<SpatialMemoryBindingSelection> memoryBindings;
  memoryBindings.reserve(realizations.memoryRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.memoryRealizations().size(); ++realization) {
    const auto choices = bindingRelations.memoryChoices(realization);
    const PnrIndex selected =
        relationChoices
            ->choices[bindingRelations.computeDecisionCount() + realization];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign memory choice");
    memoryBindings.push_back({choices[selected].placement});
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
