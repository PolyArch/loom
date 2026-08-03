#include "TechMappingCandidateTestSupport.h"

#include "ADG/FuLibrary.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "PnR/HandshakeCandidateState.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping candidate test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

} // namespace

loom::adg::FinalizedFabricDesign
loom::test::buildTemporalCapacityFabric(const ArtifactStore &store) {
  using namespace loom::adg;

  DesignBuilder design(store);
  const PortType bits128 = take(PortType::bits(128));
  const PortType tagged128 = take(PortType::taggedBits(128, 4));
  const std::vector<PortType> moduleInputs(10, tagged128);
  const std::vector<PortType> moduleOutputs(8, tagged128);
  auto spatial = take(design.createSpatialCore("capacity-envelope",
                                               moduleInputs, moduleOutputs));

  std::vector<SpatialValue> outputs;
  outputs.reserve(moduleOutputs.size());
  for (unsigned peOrdinal = 0; peOrdinal != 2; ++peOrdinal) {
    std::vector<SpatialValue> peInputs;
    peInputs.reserve(5);
    for (unsigned input = 0; input != 5; ++input)
      peInputs.push_back(take(spatial.input(peOrdinal * 5 + input)));
    const ::fabric::OperandBufferMode mode =
        peOrdinal == 0 ? ::fabric::OperandBufferMode::AllFuShare
                       : ::fabric::OperandBufferMode::PerInstruction;
    auto pe = take(spatial.addPe(
        peInputs, PeSpec::temporal(std::vector<PortType>(5, bits128),
                                   std::vector<PortType>(4, tagged128),
                                   TemporalPeParameters{
                                       2, FuConfigurationMode::PerInstruction,
                                       mode, 2, std::nullopt})));
    std::vector<PeValue> fuInputs;
    fuInputs.reserve(5);
    for (unsigned input = 0; input != 5; ++input)
      fuInputs.push_back(take(pe.input(input)));
    requireSuccess(addTokenControlFu(pe, fuInputs));
    requireSuccess(pe.close());
    for (unsigned output = 0; output != 4; ++output)
      outputs.push_back(take(pe.output(output)));
  }
  requireSuccess(spatial.close(outputs));
  return take(std::move(design).finalize());
}

void loom::test::exerciseHandshakeCandidateRefcounts(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  const auto &handshake = problem->handshake();
  auto owner = std::shared_ptr<const pnr::FrozenSpatialHandshakeIndex>(
      problem, &problem->handshake());
  auto candidate = take(pnr::HandshakeCandidateState::create(owner));
  requireSuccess(candidate->verify());

  pnr::HandshakeCandidateScratch scratch;
  requireSuccess(scratch.prepare(*owner));
  const std::size_t retainedScratchBytes = scratch.retainedStorageBytes();
  const auto offsets = handshake.computePlacementFragmentOffsets();
  const auto fragments = handshake.computePlacementFragments().slice(
      offsets.front(), offsets[1] - offsets.front());
  std::optional<pnr::PnrIndex> observedFragment;
  std::optional<pnr::PnrIndex> observedArc;
  for (pnr::PnrIndex fragment : fragments) {
    const auto record = handshake.fragments()[fragment];
    if (record.contributionCount == 0)
      continue;
    observedFragment = fragment;
    observedArc = handshake.fragmentArcOrdinals()[record.contributionOffset];
    break;
  }
  if (!observedFragment || !observedArc)
    fail("compute placement has no observable handshake contribution");

  const pnr::PnrIndex baseArcRefcount = candidate->arcRefcount(*observedArc);
  for (unsigned selection = 0; selection < 2; ++selection) {
    auto transaction = take(candidate->beginTransaction(scratch));
    requireSuccess(transaction.addFragments(fragments));
    if (!take(transaction.close()))
      fail("exact compute placement closed a handshake cycle");
    requireSuccess(transaction.commit());
  }
  if (candidate->fragmentRefcount(*observedFragment) != 2)
    fail("shared handshake fragment lost its decision refcount");
  const pnr::PnrIndex selectedArcRefcount =
      candidate->arcRefcount(*observedArc);
  if (selectedArcRefcount <= baseArcRefcount)
    fail("selected handshake fragment did not activate its arc");

  {
    auto transaction = take(candidate->beginTransaction(scratch));
    requireSuccess(transaction.removeFragments(fragments));
    if (!take(transaction.close()))
      fail("handshake deletion reported a cycle");
    transaction.rollback();
  }
  if (candidate->fragmentRefcount(*observedFragment) != 2 ||
      candidate->arcRefcount(*observedArc) != selectedArcRefcount)
    fail("handshake rollback changed the committed refcounts");

  for (unsigned selection = 0; selection < 2; ++selection) {
    auto transaction = take(candidate->beginTransaction(scratch));
    requireSuccess(transaction.removeFragments(fragments));
    if (!take(transaction.close()))
      fail("handshake deletion reported a cycle");
    requireSuccess(transaction.commit());
  }
  if (candidate->fragmentRefcount(*observedFragment) != 0 ||
      candidate->arcRefcount(*observedArc) != baseArcRefcount ||
      scratch.retainedStorageBytes() != retainedScratchBytes)
    fail("handshake selection removal retained state or expanded scratch");
  requireSuccess(candidate->verify());
}

void loom::test::exerciseCapacityOveruseCandidate(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  const auto &realizations = problem->realizations();
  if (realizations.computeRealizations().size() != 1 ||
      !realizations.memoryRealizations().empty())
    fail("capacity fixture does not contain one compute realization");

  const auto &realization = realizations.computeRealizations().front();
  std::optional<pnr::SpatialComputeBindingSelection> overused;
  std::optional<pnr::SpatialComputeBindingSelection> legal;
  for (pnr::PnrIndex placement = realization.placementOffset;
       placement != realization.placementOffset + realization.placementCount;
       ++placement) {
    const auto &placementRecord = realizations.computePlacements()[placement];
    for (pnr::PnrIndex context = placementRecord.contextOffset;
         context !=
         placementRecord.contextOffset + placementRecord.contextCount;
         ++context) {
      const std::uint64_t value =
          problem->capacity().computeInstructionContextOveruse()[context];
      if (value == 1 && !overused)
        overused = pnr::SpatialComputeBindingSelection{placement, context};
      if (value == 0 && !legal)
        legal = pnr::SpatialComputeBindingSelection{placement, context};
    }
  }
  if (!overused || !legal)
    fail("capacity fixture lacks exact overused and legal placements");

  auto attachmentsFor = [&](pnr::PnrIndex placement) {
    std::vector<pnr::PnrIndex> attachments;
    attachments.reserve(problem->ports().portDemands().size());
    for (const auto &demand : problem->ports().portDemands()) {
      if (demand.kind != pnr::FrozenSpatialPortDemandKind::Compute ||
          demand.realization != 0)
        fail("capacity fixture contains a foreign PortDemand");
      const auto &domain =
          problem->ports()
              .placementDomains()[demand.placementDomainOffset + placement -
                                  realization.placementOffset];
      attachments.push_back(domain.attachmentOptionOffset);
    }
    return attachments;
  };

  const std::vector<pnr::PnrIndex> initialAttachments =
      attachmentsFor(overused->placement);
  std::vector<pnr::PnrIndex> boundaryAttachments;
  boundaryAttachments.reserve(problem->ports().graphBoundaries().size());
  for (const auto &boundary : problem->ports().graphBoundaries())
    boundaryAttachments.push_back(boundary.attachmentOptionOffset);

  auto candidate =
      take(pnr::SpatialCandidateState::create(problem, {{*overused},
                                                        {},
                                                        initialAttachments,
                                                        boundaryAttachments,
                                                        {},
                                                        {},
                                                        {},
                                                        {}}));
  if (candidate->capacityOveruse() != 1 ||
      take(pnr::spatialMappingViolationValue(
          *candidate, ResolvedPnrViolationKind::CapacityOveruse)) != 1)
    fail("shared temporal operand service lost its exact overuse");

  pnr::SpatialCandidateScratch scratch;
  requireSuccess(scratch.prepare(*problem));
  const std::vector<pnr::PnrIndex> legalAttachments =
      attachmentsFor(legal->placement);
  {
    auto move = take(candidate->beginMove(scratch));
    requireSuccess(
        move.setComputeBinding(0, legal->placement, legal->instructionContext));
    for (auto [demand, attachment] : llvm::enumerate(legalAttachments))
      requireSuccess(move.setPortAttachment(demand, attachment));
    if (!take(move.close()))
      fail("legal capacity move closed a handshake cycle");
    requireSuccess(move.commit());
  }
  if (candidate->capacityOveruse() != 0)
    fail("legal temporal operand allocation retained capacity overuse");

  {
    auto move = take(candidate->beginMove(scratch));
    requireSuccess(move.setComputeBinding(0, overused->placement,
                                          overused->instructionContext));
    for (auto [demand, attachment] : llvm::enumerate(initialAttachments))
      requireSuccess(move.setPortAttachment(demand, attachment));
    if (!take(move.close()))
      fail("overused capacity move closed a handshake cycle");
    move.rollback();
  }
  if (candidate->capacityOveruse() != 0)
    fail("capacity rollback changed the committed objective value");
  requireSuccess(candidate->verify());
}

void loom::test::exerciseTemporalComputeUseProjection(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  const auto frozenRealizations = problem->realizations().computeRealizations();
  const auto placements = problem->realizations().computePlacements();
  const auto contexts = problem->realizations().computeInstructionContexts();
  if (frozenRealizations.size() != 1 ||
      techMapping.computeRealizations().size() != 1)
    fail("temporal ResourceUse fixture does not contain one realization");
  const auto &frozen = frozenRealizations.front();
  if (frozen.placementCount == 0)
    fail("temporal ResourceUse fixture has no placement");
  const auto &placement = placements[frozen.placementOffset];
  if (placement.contextCount == 0 ||
      fabric.peSchedule(placement.parentPe) != ::fabric::Schedule::Temporal)
    fail("temporal ResourceUse fixture has no temporal context");
  const auto context = contexts[placement.contextOffset];
  mapping::SpatialComputeBindingView selected;
  selected.realization = frozen.reference.entity;
  selected.occurrence = placement.fu;
  selected.context = context;
  const std::array<mapping::SpatialComputeBindingView, 1> bindings = {
      std::move(selected)};
  auto uses = take(mapping::deriveSpatialComputeUseRequirements(
      dataflow, techMapping, fabric, bindings));

  const auto &realization = techMapping.computeRealizations().front();
  std::size_t expectedEnqueues = 0;
  std::size_t expectedTransitionUses = 0;
  for (const auto &boundary : realization.boundaries)
    if (boundary.direction == fabric::FabricPortDirection::Input)
      ++expectedEnqueues;
  for (const auto &binding : realization.actors) {
    auto actor = take(dataflow.resolve(binding.actor));
    auto projection =
        take(dataflow::projectRegisteredActorSchemaProjection(actor.op));
    auto cases = take(dataflow::semantics::projectActorHandshakeCases(
        projection.schema, binding.operandPorts.size(),
        binding.resultPorts.size()));
    for (const auto &transition : cases) {
      ++expectedTransitionUses;
      for (std::uint32_t operand : transition.consumedInputs)
        if (llvm::any_of(realization.boundaries, [&](const auto &boundary) {
              return boundary.actor == binding.actor &&
                     boundary.direction == fabric::FabricPortDirection::Input &&
                     boundary.portOrdinal == operand;
            }))
          ++expectedTransitionUses;
    }
  }

  std::size_t enqueues = 0;
  std::size_t transitionUses = 0;
  for (const auto &use : uses) {
    if (std::holds_alternative<dataflow::CanonicalGraphConsumerEndpointRef>(
            use.trigger)) {
      ++enqueues;
      continue;
    }
    if (std::holds_alternative<mapping::SpatialActorTransitionEventRef>(
            use.trigger))
      ++transitionUses;
  }
  if (enqueues != expectedEnqueues)
    fail("temporal ResourceUse projection omitted operand enqueue events");
  if (transitionUses != expectedTransitionUses)
    fail("temporal ResourceUse projection omitted operation or dequeue events");

  const auto &capacity = problem->capacity();
  const auto offsets = capacity.computeInstructionContextEnvelopeOffsets();
  if (offsets.size() != contexts.size() + 1 ||
      capacity.resourceEvents().empty() || capacity.resourceUses().empty() ||
      capacity.resourceTimeEnvelopes().empty() ||
      capacity.resourceTimeSegments().empty())
    fail("temporal ResourceUse freeze omitted dense resource-time tables");
  for (std::size_t contextOrdinal = 0; contextOrdinal < contexts.size();
       ++contextOrdinal) {
    std::uint64_t overuse = 0;
    for (const auto &envelope : capacity.resourceTimeEnvelopes().slice(
             offsets[contextOrdinal],
             offsets[contextOrdinal + 1] - offsets[contextOrdinal])) {
      if (envelope.event >= capacity.resourceEvents().size() ||
          envelope.useCount == 0 || envelope.segmentCount == 0)
        fail("temporal resource-time envelope has an incomplete dense slice");
      overuse += envelope.capacityOveruse;
    }
    if (overuse != capacity.computeInstructionContextOveruse()[contextOrdinal])
      fail("dense resource-time envelopes disagree with capacity overuse");
  }
}

void loom::test::exerciseCanonicalCandidateInitialization(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  auto first = take(pnr::createCanonicalSpatialCandidate(problem));
  auto second = take(pnr::createCanonicalSpatialCandidate(problem));
  const auto &realizations = problem->realizations();

  for (pnr::PnrIndex index = 0;
       index < realizations.computeRealizations().size(); ++index) {
    const auto &record = realizations.computeRealizations()[index];
    const auto &binding = first->computeBinding(index);
    const auto &repeat = second->computeBinding(index);
    if (binding.placement != record.placementOffset ||
        binding.instructionContext !=
            realizations.computePlacements()[record.placementOffset]
                .contextOffset ||
        binding.placement != repeat.placement ||
        binding.instructionContext != repeat.instructionContext)
      fail("canonical initializer changed compute choice order");
  }
  for (pnr::PnrIndex index = 0;
       index < realizations.memoryRealizations().size(); ++index) {
    const auto &record = realizations.memoryRealizations()[index];
    if (first->memoryBinding(index).placement != record.placementOffset ||
        first->memoryBinding(index).placement !=
            second->memoryBinding(index).placement)
      fail("canonical initializer changed memory choice order");
  }
  for (pnr::PnrIndex demand = 0; demand < problem->ports().portDemands().size();
       ++demand) {
    const auto &record = problem->ports().portDemands()[demand];
    const pnr::PnrIndex placement =
        record.kind == pnr::FrozenSpatialPortDemandKind::Compute
            ? first->computeBinding(record.realization).placement
            : first->memoryBinding(record.realization).placement;
    const pnr::PnrIndex ownerOffset =
        record.kind == pnr::FrozenSpatialPortDemandKind::Compute
            ? realizations.computeRealizations()[record.realization]
                  .placementOffset
            : realizations.memoryRealizations()[record.realization]
                  .placementOffset;
    const auto &domain =
        problem->ports().placementDomains()[record.placementDomainOffset +
                                            placement - ownerOffset];
    if (first->portAttachment(demand) != domain.attachmentOptionOffset ||
        first->portAttachment(demand) != second->portAttachment(demand))
      fail("canonical initializer changed port attachment order");
  }
  for (pnr::PnrIndex boundary = 0;
       boundary < problem->ports().graphBoundaries().size(); ++boundary) {
    const auto &record = problem->ports().graphBoundaries()[boundary];
    if (first->graphBoundaryAttachment(boundary) !=
            record.attachmentOptionOffset ||
        first->graphBoundaryAttachment(boundary) !=
            second->graphBoundaryAttachment(boundary))
      fail("canonical initializer changed graph-boundary attachment order");
  }
  for (pnr::PnrIndex actor = 0; actor < realizations.memoryActors().size();
       ++actor) {
    const pnr::PnrIndex realization =
        realizations.memoryActorRealizations()[actor];
    const pnr::PnrIndex placement = first->memoryBinding(realization).placement;
    const auto &owner = realizations.memoryRealizations()[realization];
    const pnr::PnrIndex localActor = actor - owner.actorOffset;
    const pnr::PnrIndex domainOffset =
        problem->handshake().memoryPlacementDomainOffsets()[placement];
    const auto &domain =
        problem->handshake()
            .memoryOperationDomains()[domainOffset + localActor];
    if (first->memoryOperationPlan(actor) != domain.planOffset ||
        first->memoryOperationPlan(actor) != second->memoryOperationPlan(actor))
      fail("canonical initializer changed memory plan order");
  }
  for (pnr::PnrIndex net = 0; net < problem->transfers().logicalNets().size();
       ++net)
    if (!first->routeTree(net).isUnrouted() ||
        !second->routeTree(net).isUnrouted())
      fail("candidate initializer hid the explicit global routing action");
  requireSuccess(first->verify());
  requireSuccess(second->verify());
}
