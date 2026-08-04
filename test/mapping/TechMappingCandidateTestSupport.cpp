#include "TechMappingCandidateTestSupport.h"

#include "ADG/FuLibrary.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "PnR/HandshakeCandidateState.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialAnnealingSearch.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialObjective.h"
#include "SpatialMappingCapacityVerification.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
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

loom::ResolvedObjectiveCatalogs availableSpatialObjectiveCatalogs() {
  loom::ResolvedObjectiveCatalogs catalogs;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
      {loom::ResolvedObjectiveSourceKind::MappingViolation,
       static_cast<std::uint32_t>(
           loom::ResolvedPnrViolationKind::UnroutedObligation),
       loom::ResolvedObjectiveDirection::Minimize, 0, 1, 0, maximum},
      {loom::ResolvedObjectiveSourceKind::MappingViolation,
       static_cast<std::uint32_t>(
           loom::ResolvedPnrViolationKind::CapacityOveruse),
       loom::ResolvedObjectiveDirection::Minimize, 0, 1, 0, maximum},
      {loom::ResolvedObjectiveSourceKind::MappingMeasure,
       static_cast<std::uint32_t>(
           loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim),
       loom::ResolvedObjectiveDirection::Minimize, 0, 1, 0, maximum},
  };
  catalogs.weightedLevels = {
      {{{0, 1}, {1, 1}, {2, 1}}},
  };
  catalogs.totalOrderings = {{{0}}};
  return catalogs;
}

} // namespace

loom::ResolvedConfig loom::test::buildSpatialPnrTestResolvedConfig() {
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.objectiveCatalogs = availableSpatialObjectiveCatalogs();
  for (ResolvedPnrPolicyConfig *policy :
       {&config.dse.spatialPnr, &config.dse.systemPnr}) {
    policy->temporaryViolations.admitted = {
        ResolvedPnrViolationKind::UnroutedObligation,
        ResolvedPnrViolationKind::CapacityOveruse,
    };
    policy->objectiveSelection = {0, 0, {}};
  }
  return config;
}

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
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  const auto &realizations = problem->realizations();
  if (realizations.computeRealizations().size() != 1 ||
      !realizations.memoryRealizations().empty())
    fail("capacity fixture does not contain one compute realization");

  const auto &realization = realizations.computeRealizations().front();
  std::optional<pnr::SpatialComputeBindingSelection> overused;
  std::vector<pnr::SpatialComputeBindingSelection> legalBindings;
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
      if (value == 0)
        legalBindings.push_back(
            pnr::SpatialComputeBindingSelection{placement, context});
    }
  }
  if (!overused || legalBindings.empty())
    fail("capacity fixture lacks exact overused and legal placements");
  llvm::erase_if(legalBindings, [&](const auto &binding) {
    return binding.placement == overused->placement;
  });
  if (legalBindings.empty())
    fail("capacity fixture lacks a cross-placement legal Action");
  const pnr::SpatialComputeBindingSelection legal = legalBindings.front();

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
  auto repairCandidate =
      take(pnr::SpatialCandidateState::create(problem, {{*overused},
                                                        {},
                                                        initialAttachments,
                                                        boundaryAttachments,
                                                        {},
                                                        {},
                                                        {},
                                                        {}}));
  pnr::SpatialExactRepairScratch exactRepair;
  const pnr::SpatialExactRepairResult repaired =
      take(exactRepair.repairCapacityOveruse(*repairCandidate, 0));
  if (repaired.kind != pnr::SpatialExactRepairResultKind::Repaired ||
      repaired.regionDecisions == 0 || repaired.solverCalls == 0 ||
      repaired.actionCount == 0 ||
      repairCandidate->atomicCapacityOveruse() != 0)
    fail("CP-SAT capacity repair did not commit one exact ActionBatch");
  requireSuccess(repairCandidate->verify());
  const dse::ObjectiveVector overusedObjective =
      take(problem->objectiveProgram().evaluate(*candidate));
  if (candidate->atomicCapacityOveruse() != 1 ||
      take(pnr::spatialMappingViolationValue(
          *candidate, ResolvedPnrViolationKind::CapacityOveruse)) != 1)
    fail("shared temporal operand service lost its exact overuse");

  const auto &placement =
      problem->realizations().computePlacements()[overused->placement];
  const mapping::SpatialComputeBindingView selected{
      techMapping.computeRealizations().front().entityId,
      placement.fu,
      problem->realizations()
          .computeInstructionContexts()[overused->instructionContext],
      {}};
  const auto requirements =
      take(mapping::deriveSpatialComputeBindingUseRequirements(
          dataflow, techMapping.computeRealizations().front(), fabric,
          selected));
  std::vector<mapping::SpatialResourceUseView> persistentUses;
  persistentUses.reserve(requirements.size());
  for (const auto &requirement : requirements)
    persistentUses.push_back(
        {mapping::SpatialComputeResourceOwnerRef{requirement.realization},
         requirement.pattern,
         mapping::SpatialRelativeActivationView{
             mapping::SpatialEventPointView{requirement.trigger, std::nullopt},
             std::nullopt},
         {},
         {}});
  const auto coldOveruse = take(mapping::detail::deriveSpatialCapacityOveruse(
      fabric, dataflow.identity(), persistentUses, {}));
  if (coldOveruse.total != candidate->atomicCapacityOveruse() ||
      !coldOveruse.firstWitness ||
      coldOveruse.firstWitness->usage <= coldOveruse.firstWitness->capacity)
    fail("strict capacity reconstruction disagrees with Candidate state");
  const auto envelopeOffsets =
      problem->capacity().computeInstructionContextEnvelopeOffsets();
  const auto requireContextEnvelopeState =
      [&](const pnr::SpatialComputeBindingSelection &binding, bool active) {
        for (pnr::PnrIndex envelope =
                 envelopeOffsets[binding.instructionContext];
             envelope != envelopeOffsets[binding.instructionContext + 1];
             ++envelope)
          if (candidate->resourceTimeEnvelopeActive(envelope) != active ||
              candidate->resourceTimeEnvelopeRefcount(envelope) !=
                  (active ? 1U : 0U))
            fail("compute context selected the wrong resource-time envelope");
      };
  requireContextEnvelopeState(*overused, true);
  requireContextEnvelopeState(legal, false);
  std::vector<pnr::PnrIndex> incidentNets;
  for (const auto &demand : problem->ports().portDemands())
    if (std::find(incidentNets.begin(), incidentNets.end(),
                  demand.logicalNet) == incidentNets.end())
      incidentNets.push_back(demand.logicalNet);
  pnr::SpatialActionExecutorScratch actionExecutor;
  requireSuccess(actionExecutor.prepare(*candidate));
  const std::uint64_t initialUnroutedObligations =
      candidate->unroutedObligationCount();
  const pnr::SpatialMappingAction legalAction =
      pnr::SpatialRealizationBindingAction{pnr::SpatialComputeBindingAction{
          0, legal.placement, legal.instructionContext}};
  {
    auto probe = take(actionExecutor.probe(*candidate, legalAction));
    if (candidate->atomicCapacityOveruse() != 0)
      fail("Spatial Action probe did not update the shadow candidate");
    if (candidate->unroutedObligationCount() == 0)
      fail("unreachable binding Action lost its temporary route violation");
    requireSuccess(probe.discard());
  }
  if (candidate->unroutedObligationCount() != initialUnroutedObligations)
    fail("Spatial Action discard did not restore unrouted obligations");
  const std::size_t retainedActionExecutorBytes =
      actionExecutor.retainedStorageBytes();
  const std::vector<pnr::PnrIndex> legalAttachments =
      attachmentsFor(legal.placement);
  {
    const std::array<pnr::SpatialMappingAction, 2> malformedBatch{
        legalAction,
        pnr::SpatialResourceAllocationAction{pnr::SpatialPortAttachmentAction{
            static_cast<pnr::PnrIndex>(problem->ports().portDemands().size()),
            0}},
    };
    auto malformedProbe = actionExecutor.probeBatch(*candidate, malformedBatch);
    if (malformedProbe)
      fail("partially malformed Spatial ActionBatch produced a probe");
    const std::string failure = llvm::toString(malformedProbe.takeError());
    if (!llvm::StringRef(failure).contains(
            "port Action anchor is out of range"))
      fail("malformed Spatial ActionBatch returned the wrong failure");
  }
  if (candidate->atomicCapacityOveruse() != 1)
    fail("malformed Spatial ActionBatch retained its first Action");
  for (auto [demand, attachment] : llvm::enumerate(initialAttachments))
    if (candidate->portAttachment(demand) != attachment)
      fail("malformed Spatial ActionBatch retained a dependent attachment");
  {
    const pnr::SpatialMappingAction malformedAction =
        pnr::SpatialRealizationBindingAction{pnr::SpatialComputeBindingAction{
            static_cast<pnr::PnrIndex>(
                problem->realizations().computeRealizations().size()),
            legal.placement, legal.instructionContext}};
    auto malformedProbe = actionExecutor.probe(*candidate, malformedAction);
    if (malformedProbe)
      fail("out-of-range Spatial Action unexpectedly produced a probe");
    const std::string failure = llvm::toString(malformedProbe.takeError());
    if (!llvm::StringRef(failure).contains(
            "compute realization is out of range"))
      fail("out-of-range Spatial Action returned the wrong failure");
  }
  if (candidate->atomicCapacityOveruse() != 1)
    fail("Spatial Action discard did not restore the candidate");
  for (auto [demand, attachment] : llvm::enumerate(initialAttachments))
    if (candidate->portAttachment(demand) != attachment)
      fail("Spatial Action discard did not restore an attachment");
  for (pnr::PnrIndex logicalNet : incidentNets)
    if (!candidate->routeTree(logicalNet).isUnrouted())
      fail("Spatial Action discard did not restore an old RouteTree");
  {
    auto probe = take(actionExecutor.probe(*candidate, legalAction));
    pnr::DeterministicPnrRandomStream acceptanceStream =
        pnr::DeterministicPnrRandomStream::create(
            UINT64_C(0x0123456789abcdef), 0,
            pnr::PnrRandomStreamPurpose::Acceptance);
    pnr::DeterministicPnrRandomStream referenceStream =
        pnr::DeterministicPnrRandomStream::create(
            UINT64_C(0x0123456789abcdef), 0,
            pnr::PnrRandomStreamPurpose::Acceptance);
    const pnr::SpatialActionResolution resolution =
        take(probe.resolve(1, acceptanceStream));
    if (!resolution.accepted || resolution.objective.codes() !=
                                    actionExecutor.currentObjective().codes())
      fail("improving Spatial Action was not atomically accepted");
    if (acceptanceStream.nextU64() != referenceStream.nextU64())
      fail("improving Spatial Action consumed acceptance entropy");
  }
  if (candidate->atomicCapacityOveruse() != 0)
    fail("legal temporal operand allocation retained capacity overuse");
  for (auto [demand, attachment] : llvm::enumerate(legalAttachments))
    if (candidate->portAttachment(demand) != attachment)
      fail("Spatial Action did not rebuild a placement attachment");
  if (candidate->unroutedObligationCount() == 0)
    fail("committed binding Action lost its explicit route violation");
  requireSuccess(candidate->verify());
  const dse::ObjectiveVector legalObjective =
      take(problem->objectiveProgram().evaluate(*candidate));
  const dse::ObjectiveWideValue legalEnergy =
      take(problem->objectiveProgram().selectedEnergy(legalObjective));
  const dse::ObjectiveWideValue overusedEnergy =
      take(problem->objectiveProgram().selectedEnergy(overusedObjective));
  if (!(legalEnergy < overusedEnergy))
    fail("selected Spatial energy did not improve after legal placement");
  const dse::ObjectiveSignedDifference reward =
      take(problem->objectiveProgram().selectedEnergyDifference(
          overusedObjective, legalObjective));
  if (reward.sign != dse::ObjectiveDifferenceSign::Positive ||
      reward.magnitude == dse::ObjectiveWideValue{0, 0})
    fail("selected Spatial reward changed sign or magnitude");
  const std::array<std::uint8_t, 1> earlierKey = {0};
  const std::array<std::uint8_t, 1> laterKey = {1};
  if (take(problem->objectiveProgram().compareSelectedRank(
          legalObjective, laterKey, overusedObjective, earlierKey)) >= 0 ||
      take(problem->objectiveProgram().compareSelectedRank(
          legalObjective, earlierKey, legalObjective, laterKey)) >= 0)
    fail("selected Spatial rank lost objective or semantic-key ordering");
  requireContextEnvelopeState(*overused, false);
  requireContextEnvelopeState(legal, true);
  if (actionExecutor.retainedStorageBytes() != retainedActionExecutorBytes)
    fail("warmed Spatial Action execution grew worker-local storage");

  pnr::SpatialCandidateScratch scratch;
  requireSuccess(scratch.prepare(*problem));
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
  if (candidate->atomicCapacityOveruse() != 0)
    fail("capacity rollback changed the committed objective value");
  requireContextEnvelopeState(*overused, false);
  requireContextEnvelopeState(legal, true);
  requireSuccess(candidate->verify());
}

void loom::test::exerciseCapacityExactRepairNoMutation(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialExactRepairResultKind expected) {
  const auto &realizations = problem->realizations();
  if (realizations.computeRealizations().size() != 1 ||
      !realizations.memoryRealizations().empty())
    fail("exact-repair fixture does not contain one compute realization");
  const auto &realization = realizations.computeRealizations().front();
  std::optional<pnr::SpatialComputeBindingSelection> overused;
  for (pnr::PnrIndex placement = realization.placementOffset;
       placement != realization.placementOffset + realization.placementCount;
       ++placement) {
    const auto &record = realizations.computePlacements()[placement];
    for (pnr::PnrIndex context = record.contextOffset;
         context != record.contextOffset + record.contextCount; ++context)
      if (problem->capacity().computeInstructionContextOveruse()[context] !=
          0) {
        overused = pnr::SpatialComputeBindingSelection{placement, context};
        break;
      }
    if (overused)
      break;
  }
  if (!overused)
    fail("exact-repair fixture has no CapacityOveruse witness");

  std::vector<pnr::PnrIndex> attachments;
  attachments.reserve(problem->ports().portDemands().size());
  for (const auto &demand : problem->ports().portDemands()) {
    const auto &domain =
        problem->ports().placementDomains()[demand.placementDomainOffset +
                                            overused->placement -
                                            realization.placementOffset];
    attachments.push_back(domain.attachmentOptionOffset);
  }
  std::vector<pnr::PnrIndex> boundaries;
  boundaries.reserve(problem->ports().graphBoundaries().size());
  for (const auto &boundary : problem->ports().graphBoundaries())
    boundaries.push_back(boundary.attachmentOptionOffset);

  auto candidate = take(pnr::SpatialCandidateState::create(
      problem, {{*overused}, {}, attachments, boundaries, {}, {}, {}, {}}));
  const std::uint64_t initialOveruse = candidate->atomicCapacityOveruse();
  pnr::SpatialExactRepairScratch repair;
  const pnr::SpatialExactRepairResult outcome =
      take(repair.repairCapacityOveruse(*candidate, 0));
  if (outcome.kind != expected)
    fail("bounded exact repair returned the wrong non-repaired outcome");
  if (candidate->atomicCapacityOveruse() != initialOveruse)
    fail("non-repaired exact outcome changed the candidate");
  if (expected == pnr::SpatialExactRepairResultKind::RegionTooLarge &&
      outcome.solverCalls != 0)
    fail("oversized exact region entered CP-SAT");
  if (expected == pnr::SpatialExactRepairResultKind::UnknownBudgetExhausted &&
      outcome.solverCalls !=
          problem->config().policy().search.exactRepair.maxSolverCalls)
    fail("exact repair did not consume its solver-call budget");
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
  const auto canonicalAttempt =
      take(pnr::createSpatialCandidateInitializerAttempt(problem, 0));
  const auto &realizations = problem->realizations();

  for (pnr::PnrIndex index = 0;
       index < realizations.computeRealizations().size(); ++index) {
    const auto &record = realizations.computeRealizations()[index];
    const auto &binding = first->computeBinding(index);
    const auto &repeat = second->computeBinding(index);
    const auto &attemptZero = canonicalAttempt.candidate->computeBinding(index);
    if (binding.placement != record.placementOffset ||
        binding.instructionContext !=
            realizations.computePlacements()[record.placementOffset]
                .contextOffset ||
        binding.placement != repeat.placement ||
        binding.instructionContext != repeat.instructionContext ||
        binding.placement != attemptZero.placement ||
        binding.instructionContext != attemptZero.instructionContext)
      fail("canonical initializer changed compute choice order");
    const auto envelopeOffsets =
        problem->capacity().computeInstructionContextEnvelopeOffsets();
    for (pnr::PnrIndex envelope = envelopeOffsets[binding.instructionContext];
         envelope != envelopeOffsets[binding.instructionContext + 1];
         ++envelope)
      if (first->resourceTimeEnvelopeRefcount(envelope) != 1 ||
          !first->resourceTimeEnvelopeActive(envelope))
        fail("canonical initializer lost a compute resource-time envelope");
  }
  for (pnr::PnrIndex index = 0;
       index < realizations.memoryRealizations().size(); ++index) {
    const auto &record = realizations.memoryRealizations()[index];
    if (first->memoryBinding(index).placement != record.placementOffset ||
        first->memoryBinding(index).placement !=
            second->memoryBinding(index).placement ||
        first->memoryBinding(index).placement !=
            canonicalAttempt.candidate->memoryBinding(index).placement)
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
        first->portAttachment(demand) != second->portAttachment(demand) ||
        first->portAttachment(demand) !=
            canonicalAttempt.candidate->portAttachment(demand))
      fail("canonical initializer changed port attachment order");
  }
  for (pnr::PnrIndex boundary = 0;
       boundary < problem->ports().graphBoundaries().size(); ++boundary) {
    const auto &record = problem->ports().graphBoundaries()[boundary];
    if (first->graphBoundaryAttachment(boundary) !=
            record.attachmentOptionOffset ||
        first->graphBoundaryAttachment(boundary) !=
            second->graphBoundaryAttachment(boundary) ||
        first->graphBoundaryAttachment(boundary) !=
            canonicalAttempt.candidate->graphBoundaryAttachment(boundary))
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
        first->memoryOperationPlan(actor) !=
            second->memoryOperationPlan(actor) ||
        first->memoryOperationPlan(actor) !=
            canonicalAttempt.candidate->memoryOperationPlan(actor))
      fail("canonical initializer changed memory plan order");
  }
  for (pnr::PnrIndex net = 0; net < problem->transfers().logicalNets().size();
       ++net)
    if (!first->routeTree(net).isUnrouted() ||
        !second->routeTree(net).isUnrouted())
      fail("candidate initializer hid the explicit global routing action");
  requireSuccess(first->verify());
  requireSuccess(second->verify());
  requireSuccess(canonicalAttempt.candidate->verify());

  if (canonicalAttempt.assignmentAttempts >
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed)
    fail("Spatial initializer exceeded its assignment work limit");

  bool observedDependentDiversification = false;
  for (std::uint32_t attempt = 1;
       attempt < problem->config().policy().search.initializer.seedAttemptCount;
       ++attempt) {
    const auto diversified =
        take(pnr::createSpatialCandidateInitializerAttempt(problem, attempt));
    const auto replay =
        take(pnr::createSpatialCandidateInitializerAttempt(problem, attempt));
    requireSuccess(diversified.candidate->verify());
    requireSuccess(replay.candidate->verify());
    if (diversified.assignmentAttempts != replay.assignmentAttempts ||
        diversified.assignmentAttempts >
            problem->config()
                .policy()
                .search.initializer.assignmentAttemptLimitPerSeed)
      fail("fixed Spatial initializer slot changed its work accounting");

    for (pnr::PnrIndex realization = 0;
         realization < realizations.computeRealizations().size();
         ++realization) {
      const auto &selected = diversified.candidate->computeBinding(realization);
      const auto &repeated = replay.candidate->computeBinding(realization);
      if (selected.placement != repeated.placement ||
          selected.instructionContext != repeated.instructionContext)
        fail("fixed Spatial initializer slot changed a compute binding");
    }
    for (pnr::PnrIndex realization = 0;
         realization < realizations.memoryRealizations().size(); ++realization)
      if (diversified.candidate->memoryBinding(realization).placement !=
          replay.candidate->memoryBinding(realization).placement)
        fail("fixed Spatial initializer slot changed a memory binding");

    for (pnr::PnrIndex demand = 0;
         demand < problem->ports().portDemands().size(); ++demand) {
      if (diversified.candidate->portAttachment(demand) !=
          replay.candidate->portAttachment(demand))
        fail("fixed Spatial initializer slot changed a port attachment");
      observedDependentDiversification |=
          diversified.candidate->portAttachment(demand) !=
          canonicalAttempt.candidate->portAttachment(demand);
    }
    for (pnr::PnrIndex boundary = 0;
         boundary < problem->ports().graphBoundaries().size(); ++boundary) {
      if (diversified.candidate->graphBoundaryAttachment(boundary) !=
          replay.candidate->graphBoundaryAttachment(boundary))
        fail("fixed Spatial initializer slot changed a boundary attachment");
      observedDependentDiversification |=
          diversified.candidate->graphBoundaryAttachment(boundary) !=
          canonicalAttempt.candidate->graphBoundaryAttachment(boundary);
    }
    for (pnr::PnrIndex actor = 0; actor < realizations.memoryActors().size();
         ++actor) {
      if (diversified.candidate->memoryOperationPlan(actor) !=
          replay.candidate->memoryOperationPlan(actor))
        fail("fixed Spatial initializer slot changed a memory plan");
      observedDependentDiversification |=
          diversified.candidate->memoryOperationPlan(actor) !=
          canonicalAttempt.candidate->memoryOperationPlan(actor);
    }
    for (pnr::PnrIndex binding = 0;
         binding < problem->memory().logicalBindings().size(); ++binding) {
      const auto &selected =
          diversified.candidate->logicalMemoryBinding(binding);
      const auto &repeated = replay.candidate->logicalMemoryBinding(binding);
      if (selected.target != repeated.target ||
          selected.physicalOffsetBytes != repeated.physicalOffsetBytes)
        fail("fixed Spatial initializer slot changed a logical-memory binding");
      observedDependentDiversification |=
          selected.target !=
              canonicalAttempt.candidate->logicalMemoryBinding(binding)
                  .target ||
          selected.physicalOffsetBytes !=
              canonicalAttempt.candidate->logicalMemoryBinding(binding)
                  .physicalOffsetBytes;
    }
    for (pnr::PnrIndex use = 0; use < problem->memory().rootedUses().size();
         ++use) {
      if (diversified.candidate->memoryUseDispatch(use) !=
          replay.candidate->memoryUseDispatch(use))
        fail("fixed Spatial initializer slot changed a memory dispatch");
      observedDependentDiversification |=
          diversified.candidate->memoryUseDispatch(use) !=
          canonicalAttempt.candidate->memoryUseDispatch(use);
    }
    for (pnr::PnrIndex exposure = 0;
         exposure < problem->memory().exposures().size(); ++exposure) {
      if (diversified.candidate->memoryExposureSelection(exposure) !=
          replay.candidate->memoryExposureSelection(exposure))
        fail("fixed Spatial initializer slot changed a memory exposure");
      observedDependentDiversification |=
          diversified.candidate->memoryExposureSelection(exposure) !=
          canonicalAttempt.candidate->memoryExposureSelection(exposure);
    }
    for (pnr::PnrIndex net = 0; net < problem->transfers().logicalNets().size();
         ++net)
      if (!diversified.candidate->routeTree(net).isUnrouted() ||
          !replay.candidate->routeTree(net).isUnrouted())
        fail("Spatial initializer slot hid its global routing Action");
  }

  if (!observedDependentDiversification)
    fail("fixed Spatial initializer slots did not diversify dependent choices");

  auto foreignAttempt = pnr::createSpatialCandidateInitializerAttempt(
      problem, problem->config().policy().search.initializer.seedAttemptCount);
  if (foreignAttempt)
    fail("Spatial initializer accepted an out-of-range fixed slot");
  llvm::consumeError(foreignAttempt.takeError());

  pnr::SpatialActionDomainScratch actionDomain;
  requireSuccess(actionDomain.prepare(*problem));
  const std::size_t retainedActionDomainBytes =
      actionDomain.retainedStorageBytes();
  requireSuccess(actionDomain.rebuild(*first));
  const pnr::SpatialActionProposalDomain firstDomain = actionDomain.view();
  const std::uint64_t movableDecisionCount =
      firstDomain.realizationAnchors.size() +
      problem->transfers().logicalNets().size() +
      firstDomain.resourceAnchors.size();
  if (actionDomain.movableDecisionCount() != movableDecisionCount)
    fail("Spatial Action domain miscounted movable decisions");
  if (firstDomain.realizationChoices.empty() &&
      firstDomain.transportChoices.empty() &&
      firstDomain.resourceChoices.empty())
    fail("canonical candidate has no dynamic Spatial Action");
  pnr::DeterministicPnrRandomStream proposalStream =
      pnr::DeterministicPnrRandomStream::create(
          UINT64_C(0x0123456789abcdef), 0,
          pnr::PnrRandomStreamPurpose::ActionProposal);
  if (!take(pnr::proposeSpatialAction(ResolvedPnrActionProposalPolicy{1, 1, 1},
                                      firstDomain, proposalStream)))
    fail("nonempty dynamic domain produced no Spatial Action");
  for (const pnr::SpatialRealizationBindingAction &action :
       firstDomain.realizationChoices) {
    std::visit(
        [&](const auto &choice) {
          using T = std::decay_t<decltype(choice)>;
          if constexpr (std::is_same_v<T, pnr::SpatialComputeBindingAction>) {
            const auto &current = first->computeBinding(choice.realization);
            if (current.placement == choice.placement &&
                current.instructionContext == choice.instructionContext)
              fail("compute Action retained the current binding");
          } else {
            if (first->memoryBinding(choice.realization).placement ==
                choice.placement)
              fail("memory Action retained the current binding");
          }
        },
        action);
  }
  for (const pnr::SpatialResourceAllocationAction &action :
       firstDomain.resourceChoices) {
    std::visit(
        [&](const auto &choice) {
          using T = std::decay_t<decltype(choice)>;
          if constexpr (std::is_same_v<T, pnr::SpatialPortAttachmentAction>) {
            if (first->portAttachment(choice.demand) == choice.attachmentOption)
              fail("port Action retained the current attachment");
          } else if constexpr (std::is_same_v<
                                   T,
                                   pnr::SpatialGraphBoundaryAttachmentAction>) {
            if (first->graphBoundaryAttachment(choice.boundary) ==
                choice.attachmentOption)
              fail("graph-boundary Action retained the current attachment");
          } else if constexpr (std::is_same_v<
                                   T, pnr::SpatialMemoryOperationPlanAction>) {
            if (first->memoryOperationPlan(choice.actor) == choice.plan)
              fail("memory-plan Action retained the current plan");
          } else {
            fail("dynamic domain exposed an unimplemented resource Action");
          }
        },
        action);
  }
  if (actionDomain.retainedStorageBytes() != retainedActionDomainBytes)
    fail("Spatial Action domain allocated while rebuilding a candidate");
  requireSuccess(actionDomain.rebuild(*second));
  if (actionDomain.retainedStorageBytes() != retainedActionDomainBytes ||
      actionDomain.movableDecisionCount() != movableDecisionCount)
    fail(
        "warm Spatial Action-domain rebuild changed storage or decision count");

  const auto vector = take(problem->objectiveProgram().evaluate(*first));
  const std::uint64_t capacityOveruse = take(pnr::spatialMappingViolationValue(
      *first, ResolvedPnrViolationKind::CapacityOveruse));
  if (vector.codes() != llvm::ArrayRef<std::uint64_t>(
                            {first->unroutedObligationCount(), capacityOveruse,
                             first->totalSelectedTraversalClaim()}))
    fail("Spatial objective adapter changed a Mapping-owned value");

  auto annealedFirst = take(pnr::createCanonicalSpatialCandidate(problem));
  auto annealedSecond = take(pnr::createCanonicalSpatialCandidate(problem));
  pnr::SpatialAnnealingSearchScratch firstSearch;
  pnr::SpatialAnnealingSearchScratch secondSearch;
  const auto firstStatistics = take(firstSearch.run(*annealedFirst, 0));
  const auto secondStatistics = take(secondSearch.run(*annealedSecond, 0));
  if (!(firstStatistics == secondStatistics))
    fail("Spatial annealing replay changed its search statistics");
  if (firstStatistics.minimumTemperatureLevelCount != 1 ||
      firstStatistics.calibrationProposalSlots !=
          problem->config().policy().search.annealing.calibrationProposalCount)
    fail("Spatial annealing did not execute its exact fixed schedule");

  const auto requireSameCandidate = [&](const pnr::SpatialCandidateState &lhs,
                                        const pnr::SpatialCandidateState &rhs) {
    for (pnr::PnrIndex realization = 0;
         realization < realizations.computeRealizations().size();
         ++realization) {
      const auto &left = lhs.computeBinding(realization);
      const auto &right = rhs.computeBinding(realization);
      if (left.placement != right.placement ||
          left.instructionContext != right.instructionContext)
        fail("Spatial annealing replay changed a compute binding");
    }
    for (pnr::PnrIndex realization = 0;
         realization < realizations.memoryRealizations().size(); ++realization)
      if (lhs.memoryBinding(realization).placement !=
          rhs.memoryBinding(realization).placement)
        fail("Spatial annealing replay changed a memory binding");
    for (pnr::PnrIndex demand = 0;
         demand < problem->ports().portDemands().size(); ++demand)
      if (lhs.portAttachment(demand) != rhs.portAttachment(demand))
        fail("Spatial annealing replay changed a port attachment");
    for (pnr::PnrIndex boundary = 0;
         boundary < problem->ports().graphBoundaries().size(); ++boundary)
      if (lhs.graphBoundaryAttachment(boundary) !=
          rhs.graphBoundaryAttachment(boundary))
        fail("Spatial annealing replay changed a boundary attachment");
    for (pnr::PnrIndex actor = 0; actor < realizations.memoryActors().size();
         ++actor)
      if (lhs.memoryOperationPlan(actor) != rhs.memoryOperationPlan(actor))
        fail("Spatial annealing replay changed a memory operation plan");
    for (pnr::PnrIndex net = 0; net < problem->transfers().logicalNets().size();
         ++net) {
      const auto &left = lhs.routeTree(net);
      const auto &right = rhs.routeTree(net);
      if (left.sourceEndpoint() != right.sourceEndpoint() ||
          !llvm::equal(left.nodeStorage(), right.nodeStorage()))
        fail("Spatial annealing replay changed a RouteTree");
      for (pnr::PnrIndex sink = 0;
           sink < problem->transfers().logicalNets()[net].sinkCount; ++sink)
        if (left.sinkEndpoint(sink) != right.sinkEndpoint(sink))
          fail("Spatial annealing replay changed a route sink binding");
    }
    const auto leftObjective = take(problem->objectiveProgram().evaluate(lhs));
    const auto rightObjective = take(problem->objectiveProgram().evaluate(rhs));
    if (leftObjective.codes() != rightObjective.codes())
      fail("Spatial annealing replay changed its final objective");
  };
  requireSuccess(annealedFirst->verify());
  requireSuccess(annealedSecond->verify());
  requireSameCandidate(*annealedFirst, *annealedSecond);

  auto annealedReplay = take(pnr::createCanonicalSpatialCandidate(problem));
  const std::size_t warmStorage = firstSearch.retainedStorageBytes();
  const auto replayStatistics = take(firstSearch.run(*annealedReplay, 0));
  if (!(replayStatistics == firstStatistics) ||
      firstSearch.retainedStorageBytes() != warmStorage)
    fail("warm Spatial annealing replay changed statistics or storage");
  requireSameCandidate(*annealedFirst, *annealedReplay);

  auto foreignSeed = firstSearch.run(
      *annealedReplay,
      problem->config().policy().search.initializer.seedAttemptCount);
  if (foreignSeed)
    fail("Spatial annealing accepted an out-of-range seed ordinal");
  llvm::consumeError(foreignSeed.takeError());
}
