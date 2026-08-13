#include "TechMappingCandidateTestSupport.h"
#include "../TestAllocationProbe.h"

#include "ADG/FuLibrary.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/HandshakeCandidateState.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialAnnealingSearch.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialGlobalRoutingClosure.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialRouteCostState.h"
#include "ResourceCapacityVerification.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialMappingCapacityVerification.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
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

void verifyResidentAndEventCapacityComposition(
    const loom::fabric::FabricArtifactView &fabric) {
  const loom::fabric::FabricPhysicalTraversalView *selectedTraversal = nullptr;
  std::optional<loom::fabric::FabricUsePatternRef> selectedPattern;
  std::uint64_t eventMultiplicity = 0;
  for (const auto &traversal : fabric.physicalTraversals()) {
    for (const auto &use : traversal.impliedUses) {
      const auto *contract =
          fabric.resourceContract(use.pattern.owner.catalog());
      if (!contract || use.pattern.ordinal >= contract->usePatternCount())
        continue;
      const auto pattern =
          contract->usePattern(::fabric::UsePatternKey(use.pattern.ordinal));
      std::uint64_t maximumMultiplicity =
          std::numeric_limits<std::uint64_t>::max();
      for (const auto &claim : pattern.claims) {
        if (claim.state.ordinal() >= contract->stateCount()) {
          maximumMultiplicity = 0;
          break;
        }
        const auto dimensions = contract->capacityDimensions(claim.state);
        if (claim.dimension.ordinal() >= dimensions.size()) {
          maximumMultiplicity = 0;
          break;
        }
        const auto &dimension = dimensions[claim.dimension.ordinal()];
        const std::uint64_t available =
            dimension.capacity.value() - dimension.initialOccupancy.value();
        const std::uint64_t amount = claim.amount.value();
        if (amount != 0)
          maximumMultiplicity =
              std::min(maximumMultiplicity, available / amount);
      }
      if (maximumMultiplicity != 0 &&
          maximumMultiplicity != std::numeric_limits<std::uint64_t>::max() &&
          maximumMultiplicity <= 4096) {
        selectedTraversal = &traversal;
        selectedPattern = use.pattern;
        eventMultiplicity = maximumMultiplicity;
        break;
      }
    }
    if (selectedPattern)
      break;
  }
  if (!selectedTraversal || !selectedPattern)
    fail("capacity fixture lacks a composable resident traversal claim");

  using loom::mapping::detail::ResourceCapacityNamespaceView;
  using loom::mapping::detail::ResourceCapacityRouteProjection;
  using loom::mapping::detail::ResourceCapacityUseProjection;
  const ResourceCapacityNamespaceView shared{&fabric, {0}};
  const std::vector<ResourceCapacityUseProjection> eventUses(
      eventMultiplicity,
      ResourceCapacityUseProjection{0, *selectedPattern, "shared-event"});
  const ResourceCapacityRouteProjection residentRoute{
      0, {selectedTraversal->reference}};
  const auto eventOnly =
      take(loom::mapping::detail::deriveResourceCapacityOveruse(
          llvm::ArrayRef<ResourceCapacityNamespaceView>(shared), eventUses,
          {}));
  const auto residentOnly =
      take(loom::mapping::detail::deriveResourceCapacityOveruse(
          llvm::ArrayRef<ResourceCapacityNamespaceView>(shared), {},
          llvm::ArrayRef<ResourceCapacityRouteProjection>(residentRoute)));
  const auto combined =
      take(loom::mapping::detail::deriveResourceCapacityOveruse(
          llvm::ArrayRef<ResourceCapacityNamespaceView>(shared), eventUses,
          llvm::ArrayRef<ResourceCapacityRouteProjection>(residentRoute)));
  if (eventOnly.total != 0 || residentOnly.total != 0 || combined.total == 0 ||
      !combined.firstWitness)
    fail("capacity closure did not combine resident and event occupancy");

  const std::array<ResourceCapacityNamespaceView, 2> isolated = {
      ResourceCapacityNamespaceView{&fabric, {0}},
      ResourceCapacityNamespaceView{&fabric, {1}}};
  std::vector<ResourceCapacityUseProjection> isolatedUses(
      eventMultiplicity,
      ResourceCapacityUseProjection{1, *selectedPattern, "shared-event"});
  const auto separated =
      take(loom::mapping::detail::deriveResourceCapacityOveruse(
          isolated, isolatedUses,
          llvm::ArrayRef<ResourceCapacityRouteProjection>(residentRoute)));
  if (separated.total != 0)
    fail("capacity closure merged distinct physical occurrence namespaces");
}

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string text = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      text += ", ";
    text += std::to_string(static_cast<std::int8_t>(byte));
  }
  return text + "]";
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

template <typename Ref>
std::string dataflowAttr(llvm::StringRef spelling,
                         const loom::ArtifactIdentity &owner, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(take(dataflow::encodeDataflowReference(owner, ref))) + ">";
}

template <typename Ref>
std::string fabricAttr(llvm::StringRef spelling, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(loom::fabric::canonicalFabricBytes(ref)) + ">";
}

loom::ResolvedObjectiveCatalogs availableSpatialObjectiveCatalogs() {
  loom::ResolvedObjectiveCatalogs catalogs;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::CapacityOveruse},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingMeasureObjectiveSource{static_cast<std::uint32_t>(
           loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim)},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
  };
  catalogs.weightedLevels = {
      {{{0, 1}, {1, 1}, {2, 1}}},
  };
  catalogs.totalOrderings = {{{0}}};
  return catalogs;
}

} // namespace

loom::mapping::FinalizedSpatialMappingConstraintSet
loom::test::buildSpatialMappingConstraints(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store,
    bool restrictTagsToZero, bool rejectComputePlacement) {
  std::string clauses;
  if (restrictTagsToZero)
    for (const auto &net : techMapping.residualLogicalNets())
      clauses += "    mapping.constraint.domain_restriction "
                 "projection(net_assigned_tag_values) subject(" +
                 dataflowAttr("graph_producer_endpoint_ref",
                              dataflow.identity(), net.producer) +
                 ") admissible_domain(["
                 "#mapping.constraint_unsigned_interval<lower = 0 : ui8, "
                 "upper = 1 : ui8>])\n";
  if (rejectComputePlacement && !techMapping.computeRealizations().empty())
    clauses +=
        "    mapping.constraint.domain_restriction "
        "projection(compute_placement) subject("
        "#mapping.compute_realization_ref<" +
        std::to_string(techMapping.computeRealizations().front().entityId) +
        ">) admissible_domain([])\n";
  const std::string text =
      "module {\n  mapping.constraints.spatial dataflow(" +
      identityAttr(dataflow.identity()) + ") tech_mapping(" +
      identityAttr(techMapping.identity()) + ") fabric(" +
      identityAttr(fabric.identity()) + ") {\n" + clauses + "  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse MappingConstraintSet fixture");
  auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
  return take(mapping::finalizeSpatialMappingConstraintSet(
      *roots.begin(), dataflow, techMapping, fabric, store));
}

void loom::test::exerciseSpatialTagConstraintRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store) {
  const auto nets = techMapping.residualLogicalNets();
  if (nets.size() < 2)
    fail("tag relation fixture has fewer than two residual logical nets");

  const auto pnrConfig = take(pnr::projectResolvedSpatialPnrConfigView(
      buildSpatialPnrTestResolvedConfig()));
  const auto unconstrained = buildSpatialMappingConstraints(
      context, dataflow, techMapping, fabric, store);
  auto unconstrainedProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, unconstrained.view()));
  pnr::SpatialPathFinderSeedWorkSummary unconstrainedWork;
  auto unconstrainedSeed = take(pnr::createCanonicalPathFinderSpatialSeed(
      unconstrainedProblem, unconstrainedWork));
  const auto projectedValues = [&](pnr::PnrIndex net) {
    std::vector<llvm::APInt> projected;
    for (const auto &value : unconstrainedSeed.candidate->tagValues(net))
      if (value)
        projected.push_back(
            value->zextOrTrunc(std::max(1u, value->getActiveBits())));
    llvm::sort(projected, [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
      const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
      return lhs.zext(width).ult(rhs.zext(width));
    });
    projected.erase(std::unique(projected.begin(), projected.end()),
                    projected.end());
    return projected;
  };
  std::optional<std::array<pnr::PnrIndex, 2>> selectedNets;
  for (pnr::PnrIndex lhs = 0; lhs < nets.size() && !selectedNets; ++lhs) {
    const auto lhsValues = projectedValues(lhs);
    if (lhsValues.empty())
      continue;
    for (pnr::PnrIndex rhs = lhs + 1; rhs < nets.size(); ++rhs)
      if (lhsValues == projectedValues(rhs)) {
        selectedNets = std::array<pnr::PnrIndex, 2>{lhs, rhs};
        break;
      }
  }
  if (!selectedNets)
    fail("tag relation fixture has no proven-compatible net pair");
  const std::array<std::string, 2> subjects = {
      dataflowAttr("graph_producer_endpoint_ref", dataflow.identity(),
                   nets[(*selectedNets)[0]].producer),
      dataflowAttr("graph_producer_endpoint_ref", dataflow.identity(),
                   nets[(*selectedNets)[1]].producer),
  };
  const auto buildConstraints = [&](llvm::StringRef relation,
                                    bool singletonDomain) {
    std::string clauses;
    if (singletonDomain)
      for (const std::string &subject : subjects)
        clauses += "    mapping.constraint.domain_restriction "
                   "projection(net_assigned_tag_values) subject(" +
                   subject +
                   ") admissible_domain(["
                   "#mapping.constraint_unsigned_interval<"
                   "lower = 0 : ui8, upper = 1 : ui8>])\n";
    clauses += "    mapping.constraint." + relation.str() +
               " projection(net_assigned_tag_values) subjects([" + subjects[0] +
               ", " + subjects[1] + "])\n";
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(techMapping.identity()) + ") fabric(" +
        identityAttr(fabric.identity()) + ") {\n" + clauses + "  }\n}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      fail("cannot parse Physical Tag relation fixture");
    auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
    return take(mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, techMapping, fabric, store));
  };

  const auto equality = buildConstraints("equal", false);
  auto equalityProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, equality.view()));
  pnr::SpatialPathFinderSeedWorkSummary equalityWork;
  auto equalitySeed = take(pnr::createCanonicalPathFinderSpatialSeed(
      std::move(equalityProblem), equalityWork));
  requireSuccess(equalitySeed.candidate->verify());
  std::array<std::vector<llvm::APInt>, 2> projected;
  for (pnr::PnrIndex member = 0; member < projected.size(); ++member) {
    for (const auto &value :
         equalitySeed.candidate->tagValues((*selectedNets)[member]))
      if (value)
        projected[member].push_back(
            value->zextOrTrunc(std::max(1u, value->getActiveBits())));
    llvm::sort(
        projected[member], [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
          const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
          return lhs.zext(width).ult(rhs.zext(width));
        });
    projected[member].erase(
        std::unique(projected[member].begin(), projected[member].end()),
        projected[member].end());
    if (projected[member].empty())
      fail("tag equality fixture produced an empty routed projection");
  }
  if (projected[0] != projected[1])
    fail("tag equality relation did not constrain the selected value sets");

  const auto disjoint = buildConstraints("disjoint", true);
  auto disjointProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, disjoint.view()));
  pnr::SpatialPathFinderSeedWorkSummary disjointWork;
  auto disjointSeed = take(
      pnr::createCanonicalPathFinderSpatialSeed(disjointProblem, disjointWork));
  pnr::SpatialGlobalRoutingClosureScratch closure;
  llvm::Error rejected = closure.run(*disjointSeed.candidate);
  if (!rejected)
    fail("singleton Physical Tag domains satisfied a disjoint relation");
  bool observedUnassigned = false;
  llvm::handleAllErrors(
      std::move(rejected),
      [&](const pnr::SpatialGlobalRoutingClosureFailure &failure) {
        observedUnassigned =
            failure.kind() ==
            pnr::SpatialGlobalRoutingClosureFailureKind::TagUnassigned;
      });
  if (!observedUnassigned)
    fail("tag disjointness rejection lost its typed unassigned witness");
}

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
    requireSuccess(
        addTokenControlFu(pe, fuInputs, TokenControlFuParameters{128, 64}));
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
  pnr::DeterministicPnrRandomStream exactRepairStream =
      pnr::DeterministicPnrRandomStream::create(
          problem->config().policy().determinism.masterSeed, 0,
          pnr::PnrRandomStreamPurpose::ExactRepair);
  const pnr::SpatialExactRepairResult repaired = take(exactRepair.repair(
      *repairCandidate, 0,
      problem->config().policy().search.exactRepair.maxSolverCalls,
      exactRepairStream));
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
             [&] {
               std::vector<mapping::SpatialEventPointView> release;
               for (const auto &event : requirement.release)
                 release.push_back({event, std::nullopt});
               return release;
             }()},
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

void loom::test::exerciseCombinedCapacityProjection(
    const fabric::FabricArtifactView &fabric) {
  verifyResidentAndEventCapacityComposition(fabric);
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
  pnr::DeterministicPnrRandomStream exactRepairStream =
      pnr::DeterministicPnrRandomStream::create(
          problem->config().policy().determinism.masterSeed, 0,
          pnr::PnrRandomStreamPurpose::ExactRepair);
  const pnr::SpatialExactRepairResult outcome = take(repair.repair(
      *candidate, 0,
      problem->config().policy().search.exactRepair.maxSolverCalls,
      exactRepairStream));
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
  std::size_t expectedDequeues = 0;
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
            })) {
          ++expectedTransitionUses;
          ++expectedDequeues;
        }
    }
  }

  std::size_t enqueues = 0;
  std::size_t dequeues = 0;
  std::size_t transitionUses = 0;
  const auto peOwner = loom::fabric::FabricInventoryOwnerRef::of(context.pe);
  for (const auto &use : uses) {
    if (std::holds_alternative<dataflow::CanonicalGraphConsumerEndpointRef>(
            use.trigger)) {
      if (!use.release.empty())
        fail("temporal enqueue ResourceUse gained a causal release");
      ++enqueues;
      continue;
    }
    if (std::holds_alternative<mapping::SpatialActorTransitionEventRef>(
            use.trigger)) {
      ++transitionUses;
      if (use.pattern.owner.catalog() == peOwner) {
        if (!use.release.empty())
          fail("temporal dequeue ResourceUse gained a causal release");
        ++dequeues;
      }
    }
  }
  if (enqueues != expectedEnqueues)
    fail("temporal ResourceUse projection omitted operand enqueue events");
  if (transitionUses != expectedTransitionUses)
    fail("temporal ResourceUse projection omitted operation or dequeue events");
  if (dequeues != expectedDequeues)
    fail("temporal ResourceUse projection omitted operand dequeue events");

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
  std::uint64_t canonicalAssignmentAttempts = 0;
  const auto canonicalAttempt =
      take(pnr::createSpatialCandidateInitializerAttempt(
          problem, 0, canonicalAssignmentAttempts));
  const auto &realizations = problem->realizations();

  for (pnr::PnrIndex index = 0;
       index < realizations.computeRealizations().size(); ++index) {
    const auto &record = realizations.computeRealizations()[index];
    const auto &binding = first->computeBinding(index);
    const auto &repeat = second->computeBinding(index);
    const auto &attemptZero = canonicalAttempt.candidate->computeBinding(index);
    if (binding.placement < record.placementOffset ||
        binding.placement >= record.placementOffset + record.placementCount ||
        binding.placement != repeat.placement ||
        binding.instructionContext != repeat.instructionContext ||
        binding.placement != attemptZero.placement ||
        binding.instructionContext != attemptZero.instructionContext)
      fail("canonical initializer returned an unstable compute choice");
    const auto &placement = realizations.computePlacements()[binding.placement];
    if (binding.instructionContext < placement.contextOffset ||
        binding.instructionContext >=
            placement.contextOffset + placement.contextCount)
      fail("canonical initializer returned a foreign instruction context");
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
    const pnr::PnrIndex placement = first->memoryBinding(index).placement;
    if (placement < record.placementOffset ||
        placement >= record.placementOffset + record.placementCount ||
        placement != second->memoryBinding(index).placement ||
        placement != canonicalAttempt.candidate->memoryBinding(index).placement)
      fail("canonical initializer returned an unstable memory choice");
  }
  const auto attachmentOptions = problem->ports().attachmentOptions();
  const auto &relations = problem->bindingRelations();
  std::map<pnr::PnrIndex, std::uint64_t> endpointSelectionCounts;
  const auto participatesInHardRelation = [&](pnr::PnrIndex decision) {
    return llvm::any_of(relations.decisionRelations(decision),
                        [&](pnr::PnrIndex relation) {
                          return relations.relationIsConstraint(relation);
                        });
  };
  const auto verifyLeastSelected = [&](pnr::PnrIndex decision,
                                       pnr::PnrIndex selected,
                                       llvm::ArrayRef<pnr::PnrIndex> choices,
                                       const auto &isEligible) {
    if (!participatesInHardRelation(decision)) {
      std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
      for (pnr::PnrIndex option : choices) {
        if (option >= attachmentOptions.size() || !isEligible(option))
          continue;
        const auto found =
            endpointSelectionCounts.find(attachmentOptions[option].endpoint);
        minimum = std::min(minimum, found == endpointSelectionCounts.end()
                                        ? 0
                                        : found->second);
      }
      const auto found =
          endpointSelectionCounts.find(attachmentOptions[selected].endpoint);
      const std::uint64_t selectedCount =
          found == endpointSelectionCounts.end() ? 0 : found->second;
      if (minimum == std::numeric_limits<std::uint64_t>::max() ||
          selectedCount != minimum)
        fail("canonical initializer did not select a least-used endpoint");
    }
    ++endpointSelectionCounts[attachmentOptions[selected].endpoint];
  };
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
    const pnr::PnrIndex domainOrdinal =
        record.placementDomainOffset + placement - ownerOffset;
    const auto &domain = problem->ports().placementDomains()[domainOrdinal];
    const pnr::PnrIndex selected = first->portAttachment(demand);
    if (selected < domain.attachmentOptionOffset ||
        selected - domain.attachmentOptionOffset >=
            domain.attachmentOptionCount ||
        selected >= attachmentOptions.size() ||
        attachmentOptions[selected].ownerKind !=
            pnr::FrozenSpatialAttachmentOwnerKind::PlacementDomain ||
        attachmentOptions[selected].owner != domainOrdinal ||
        selected != second->portAttachment(demand) ||
        selected != canonicalAttempt.candidate->portAttachment(demand))
      fail("canonical initializer changed port attachment order");
    if (attachmentOptions[selected].endpoint >=
        problem->routing().routingEndpoints().size())
      fail("canonical initializer selected a foreign attachment endpoint");
    verifyLeastSelected(
        relations.portDecisionOffset() + demand, selected,
        relations.portAttachmentChoices(demand), [&](pnr::PnrIndex option) {
          const auto &candidate = attachmentOptions[option];
          return candidate.ownerKind ==
                     pnr::FrozenSpatialAttachmentOwnerKind::PlacementDomain &&
                 candidate.owner < problem->ports().placementDomains().size() &&
                 problem->ports()
                         .placementDomains()[candidate.owner]
                         .placement == placement;
        });
  }
  for (pnr::PnrIndex boundary = 0;
       boundary < problem->ports().graphBoundaries().size(); ++boundary) {
    const auto &record = problem->ports().graphBoundaries()[boundary];
    const pnr::PnrIndex selected = first->graphBoundaryAttachment(boundary);
    if (selected < record.attachmentOptionOffset ||
        selected - record.attachmentOptionOffset >=
            record.attachmentOptionCount ||
        selected >= attachmentOptions.size() ||
        attachmentOptions[selected].ownerKind !=
            pnr::FrozenSpatialAttachmentOwnerKind::GraphBoundary ||
        attachmentOptions[selected].owner != boundary ||
        selected != second->graphBoundaryAttachment(boundary) ||
        selected !=
            canonicalAttempt.candidate->graphBoundaryAttachment(boundary))
      fail("canonical initializer changed graph-boundary attachment order");
    if (attachmentOptions[selected].endpoint >=
        problem->routing().routingEndpoints().size())
      fail("canonical initializer selected a foreign boundary endpoint");
  }
  const auto &routing = problem->routing();
  const auto endpoints = routing.routingEndpoints();
  const auto arcs = routing.routingArcs();
  const auto adjacency = routing.adjacencyOffsets();
  const auto reachable = [&](pnr::PnrIndex source, pnr::PnrIndex target,
                             std::uint32_t payloadWidth) {
    if (source >= endpoints.size() || target >= endpoints.size() ||
        adjacency.size() != endpoints.size() + 1)
      fail("canonical initializer produced a malformed routing endpoint");
    if (source == target)
      return true;
    std::vector<std::uint8_t> visited(endpoints.size(), 0);
    std::vector<pnr::PnrIndex> worklist{source};
    visited[source] = 1;
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const pnr::PnrIndex endpoint = worklist[cursor];
      for (pnr::PnrIndex arc = adjacency[endpoint];
           arc < adjacency[endpoint + 1]; ++arc) {
        if (arc >= arcs.size() || arcs[arc].target >= endpoints.size())
          fail("canonical initializer routing graph is malformed");
        if (arcs[arc].payloadCapacityBits < payloadWidth)
          continue;
        const pnr::PnrIndex successor = arcs[arc].target;
        if (visited[successor])
          continue;
        if (successor == target)
          return true;
        visited[successor] = 1;
        worklist.push_back(successor);
      }
    }
    return false;
  };
  for (pnr::PnrIndex logicalNet = 0;
       logicalNet < problem->transfers().logicalNets().size(); ++logicalNet) {
    const pnr::PnrIndex source = first->logicalNetSourceEndpoint(logicalNet);
    const auto &net = problem->transfers().logicalNets()[logicalNet];
    for (pnr::PnrIndex sink = 0; sink < net.sinkCount; ++sink)
      if (!reachable(source, first->logicalNetSinkEndpoint(logicalNet, sink),
                     first->logicalNetPayloadWidth(logicalNet)))
        fail("canonical initializer selected an unreachable terminal pair");
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
  for (pnr::PnrIndex binding = 0;
       binding < problem->memory().logicalBindings().size(); ++binding) {
    const auto &selected = first->logicalMemoryBinding(binding);
    const auto &repeated = second->logicalMemoryBinding(binding);
    const auto &attempt =
        canonicalAttempt.candidate->logicalMemoryBinding(binding);
    if (selected.target != repeated.target ||
        selected.physicalOffsetBytes != repeated.physicalOffsetBytes ||
        selected.target != attempt.target ||
        selected.physicalOffsetBytes != attempt.physicalOffsetBytes)
      fail("canonical initializer changed a logical-memory binding");
  }
  for (pnr::PnrIndex use = 0; use < problem->memory().rootedUses().size();
       ++use)
    if (first->memoryUseDispatch(use) != second->memoryUseDispatch(use) ||
        first->memoryUseDispatch(use) !=
            canonicalAttempt.candidate->memoryUseDispatch(use))
      fail("canonical initializer changed a memory dispatch");
  for (pnr::PnrIndex exposure = 0;
       exposure < problem->memory().exposures().size(); ++exposure)
    if (first->memoryExposureSelection(exposure) !=
            second->memoryExposureSelection(exposure) ||
        first->memoryExposureSelection(exposure) !=
            canonicalAttempt.candidate->memoryExposureSelection(exposure))
      fail("canonical initializer changed a memory exposure");
  for (pnr::PnrIndex net = 0; net < problem->transfers().logicalNets().size();
       ++net)
    if (!first->routeTree(net).isUnrouted() ||
        !second->routeTree(net).isUnrouted())
      fail("candidate initializer hid the explicit global routing action");
  requireSuccess(first->verify());
  requireSuccess(second->verify());
  requireSuccess(canonicalAttempt.candidate->verify());

  if (canonicalAssignmentAttempts >
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed)
    fail("Spatial initializer exceeded its assignment work limit");
}

void loom::test::exerciseSpatialInitializerDiversification(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  std::uint64_t canonicalAssignmentAttempts = 0;
  const auto canonicalAttempt =
      take(pnr::createSpatialCandidateInitializerAttempt(
          problem, 0, canonicalAssignmentAttempts));
  const auto &realizations = problem->realizations();
  bool observedDependentDiversification = false;
  for (std::uint32_t attempt = 1;
       attempt < problem->config().policy().search.initializer.seedAttemptCount;
       ++attempt) {
    std::uint64_t diversifiedAssignmentAttempts = 0;
    std::uint64_t replayAssignmentAttempts = 0;
    const auto diversified = take(pnr::createSpatialCandidateInitializerAttempt(
        problem, attempt, diversifiedAssignmentAttempts));
    const auto replay = take(pnr::createSpatialCandidateInitializerAttempt(
        problem, attempt, replayAssignmentAttempts));
    requireSuccess(diversified.candidate->verify());
    requireSuccess(replay.candidate->verify());
    if (diversifiedAssignmentAttempts != replayAssignmentAttempts ||
        diversifiedAssignmentAttempts >
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

  std::uint64_t foreignAssignmentAttempts = 0;
  auto foreignAttempt = pnr::createSpatialCandidateInitializerAttempt(
      problem, problem->config().policy().search.initializer.seedAttemptCount,
      foreignAssignmentAttempts);
  if (foreignAttempt)
    fail("Spatial initializer accepted an out-of-range fixed slot");
  llvm::consumeError(foreignAttempt.takeError());
}

void loom::test::exerciseSpatialActionDomainAndObjective(
    const pnr::FrozenSpatialPnrProblemHandle &problem) {
  auto first = take(pnr::createCanonicalSpatialCandidate(problem));
  auto second = take(pnr::createCanonicalSpatialCandidate(problem));
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
          } else if constexpr (std::is_same_v<
                                   T, pnr::SpatialLogicalMemoryBindingAction>) {
            const auto &current = first->logicalMemoryBinding(choice.binding);
            if (current.target == choice.target &&
                current.physicalOffsetBytes == choice.physicalOffsetBytes)
              fail("logical-memory Action retained the current binding");
          } else if constexpr (std::is_same_v<
                                   T, pnr::SpatialMemoryUseDispatchAction>) {
            if (first->memoryUseDispatch(choice.use) == choice.dispatchOption)
              fail("memory-dispatch Action retained the current option");
          } else {
            if (first->memoryExposureSelection(choice.exposure) ==
                choice.exposureOption)
              fail("memory-exposure Action retained the current option");
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
}

void loom::test::exerciseSpatialAnnealingReplay(
    const pnr::FrozenSpatialPnrProblemHandle &problem, bool warmScratch) {
  const auto &realizations = problem->realizations();
  auto annealedFirst = take(pnr::createCanonicalSpatialCandidate(problem));
  auto annealedReplay = take(pnr::createCanonicalSpatialCandidate(problem));
  pnr::SpatialAnnealingSearchScratch firstSearch;
  const auto firstStatistics = take(firstSearch.run(*annealedFirst, 0));
  const std::size_t warmStorage = firstSearch.retainedStorageBytes();
  pnr::SpatialAnnealingSearchScratch independentSearch;
  const auto replayStatistics =
      warmScratch ? take(firstSearch.run(*annealedReplay, 0))
                  : take(independentSearch.run(*annealedReplay, 0));
  if (!(firstStatistics == replayStatistics))
    fail("Spatial annealing replay changed its search statistics");
  if (warmScratch && firstSearch.retainedStorageBytes() != warmStorage)
    fail("warm Spatial annealing replay changed retained storage");
  const std::uint64_t configuredCalibration =
      problem->config().policy().search.annealing.calibrationProposalCount;
  if (firstStatistics.calibrationProposalSlots != 0 &&
      firstStatistics.calibrationProposalSlots != configuredCalibration)
    fail("Spatial annealing changed its fixed calibration schedule");
  if (!firstStatistics.exactClosureReached &&
      (firstStatistics.minimumTemperatureLevelCount != 1 ||
       firstStatistics.calibrationProposalSlots != configuredCalibration))
    fail("Spatial annealing did not execute its exact fixed schedule");
  if (firstStatistics.calibrationProposalSlots == 0 &&
      (firstStatistics.minimumTemperatureLevelCount != 0 ||
       firstStatistics.annealingProposalSlots != 0))
    fail("entry-closed Spatial annealing consumed schedule work");

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
    for (pnr::PnrIndex binding = 0;
         binding < problem->memory().logicalBindings().size(); ++binding) {
      const auto &left = lhs.logicalMemoryBinding(binding);
      const auto &right = rhs.logicalMemoryBinding(binding);
      if (left.target != right.target ||
          left.physicalOffsetBytes != right.physicalOffsetBytes)
        fail("Spatial annealing replay changed a logical-memory binding");
    }
    for (pnr::PnrIndex use = 0; use < problem->memory().rootedUses().size();
         ++use)
      if (lhs.memoryUseDispatch(use) != rhs.memoryUseDispatch(use))
        fail("Spatial annealing replay changed a memory dispatch");
    for (pnr::PnrIndex exposure = 0;
         exposure < problem->memory().exposures().size(); ++exposure)
      if (lhs.memoryExposureSelection(exposure) !=
          rhs.memoryExposureSelection(exposure))
        fail("Spatial annealing replay changed a memory exposure");
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
  requireSuccess(annealedReplay->verify());
  requireSameCandidate(*annealedFirst, *annealedReplay);

  auto foreignSeed = firstSearch.run(
      *annealedReplay,
      problem->config().policy().search.initializer.seedAttemptCount);
  if (foreignSeed)
    fail("Spatial annealing accepted an out-of-range seed ordinal");
  llvm::consumeError(foreignSeed.takeError());
}

void loom::test::exercisePathFinderFixedTerminalCutRejection(
    pnr::SpatialCandidateState &candidate,
    pnr::SpatialCandidateScratch &candidateScratch) {
  auto routeCosts = take(pnr::SpatialRouteCostState::create(candidate));
  pnr::SpatialPathFinderRouterScratch router;
  requireSuccess(router.prepare(candidate.problem()));
  const std::uint64_t routeClaim = candidate.totalSelectedTraversalClaim();
  auto fixedCut =
      router.routeToClosure(candidate, candidateScratch, routeCosts,
                            {candidate.problem()
                                 .config()
                                 .policy()
                                 .search.routing.endpointExpansionLimit,
                             candidate.problem()
                                 .config()
                                 .policy()
                                 .search.routing.negotiationIterationLimit,
                             candidate.problem()
                                 .config()
                                 .policy()
                                 .search.routing.noProgressIterationLimit,
                             candidate.problem()
                                 .config()
                                 .policy()
                                 .search.routing.noProgressTrendWindow},
                            {});
  if (fixedCut)
    fail("PathFinder ignored a fixed-terminal capacity cut");
  bool observedFixedCut = false;
  llvm::handleAllErrors(
      fixedCut.takeError(),
      [&](const pnr::SpatialPathFinderClosureFailure &failure) {
        observedFixedCut =
            failure.kind() == pnr::SpatialPathFinderClosureFailure::Kind::
                                  FixedTerminalCapacityCut &&
            failure.certificateCapacity() != pnr::getInvalidPnrIndex() &&
            failure.mandatoryUsage() > failure.physicalCapacity() &&
            !failure.forcedLogicalNets().empty();
      });
  if (!observedFixedCut)
    fail("PathFinder lost its fixed-terminal capacity-cut certificate");
  if (candidate.totalSelectedTraversalClaim() != routeClaim)
    fail("fixed-terminal cut rejection changed the candidate route overlay");

  pnr::SpatialActionExecutorScratch executor;
  requireSuccess(executor.prepare(candidate));
  const pnr::SpatialMappingAction globalRouting =
      pnr::SpatialTransportRoutingAction{pnr::SpatialGlobalRoutingAction{}};
  auto searchProbe = executor.probe(candidate, globalRouting);
  if (searchProbe)
    fail("Spatial search accepted a fixed-terminal capacity cut");
  bool observedSearchRejection = false;
  llvm::handleAllErrors(
      searchProbe.takeError(),
      [&](const pnr::SpatialActionTransitionFailure &failure) {
        observedSearchRejection =
            failure.kind() ==
            pnr::SpatialActionTransitionFailureKind::IntrinsicInvalid;
      });
  if (!observedSearchRejection)
    fail("Spatial search promoted a fixed-terminal cut to an internal error");

  auto closureProbe =
      executor.probe(candidate, globalRouting,
                     pnr::SpatialActionExecutionContext::FinalClosure);
  if (closureProbe)
    fail("Spatial final closure accepted a fixed-terminal capacity cut");
  bool observedClosureCertificate = false;
  llvm::handleAllErrors(
      closureProbe.takeError(),
      [&](const pnr::SpatialPathFinderClosureFailure &failure) {
        observedClosureCertificate =
            failure.kind() == pnr::SpatialPathFinderClosureFailure::Kind::
                                  FixedTerminalCapacityCut &&
            failure.certificateCapacity() != pnr::getInvalidPnrIndex() &&
            failure.mandatoryUsage() > failure.physicalCapacity() &&
            !failure.forcedLogicalNets().empty();
      });
  if (!observedClosureCertificate)
    fail("Spatial final closure lost its fixed-terminal cut certificate");
  requireSuccess(routeCosts.resetFromCandidate());
  for (pnr::PnrIndex capacity = 0;
       capacity < candidate.problem().resources().capacityDimensions().size();
       ++capacity)
    if (routeCosts.workingCapacityUsageRaw(capacity) !=
        candidate.routeCapacityUsageRaw(capacity))
      fail("fixed-terminal cut rejection left stale route capacity");
  requireSuccess(candidate.verify());
}

void loom::test::exerciseSpatialAttachmentConstraintRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store) {
  std::vector<pnr::PnrIndex> sourceNets;
  std::vector<dataflow::CanonicalGraphProducerEndpointRef> sources;
  for (auto [ordinal, net] :
       llvm::enumerate(techMapping.residualLogicalNets())) {
    if (!std::holds_alternative<dataflow::GraphIngressTokenRef>(net.producer))
      continue;
    sourceNets.push_back(static_cast<pnr::PnrIndex>(ordinal));
    sources.push_back(net.producer);
    if (sources.size() == 2)
      break;
  }
  if (sources.size() != 2)
    fail("attachment relation fixture has fewer than two graph sources");

  const auto config = take(pnr::projectResolvedSpatialPnrConfigView(
      buildSpatialPnrTestResolvedConfig()));
  const auto emptyConstraints = buildSpatialMappingConstraints(
      context, dataflow, techMapping, fabric, store);
  const auto unconstrained = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, emptyConstraints.view()));
  std::array<pnr::PnrIndex, 2> boundaries;
  for (std::size_t index = 0; index < sourceNets.size(); ++index) {
    const auto binding = unconstrained->transfers()
                             .logicalNetSourceBindings()[sourceNets[index]];
    if (binding.kind != pnr::FrozenSpatialTerminalBindingKind::GraphBoundary)
      fail("attachment relation fixture source is not a graph boundary");
    boundaries[index] = binding.index;
  }
  const auto &firstBoundary =
      unconstrained->ports().graphBoundaries()[boundaries[0]];
  const auto &secondBoundary =
      unconstrained->ports().graphBoundaries()[boundaries[1]];
  std::optional<pnr::PnrIndex> requiredEndpoint;
  for (pnr::PnrIndex first = firstBoundary.attachmentOptionOffset;
       first != firstBoundary.attachmentOptionOffset +
                    firstBoundary.attachmentOptionCount;
       ++first) {
    const pnr::PnrIndex endpoint =
        unconstrained->ports().attachmentOptions()[first].endpoint;
    bool shared = false;
    bool secondHasAlternative = false;
    for (pnr::PnrIndex second = secondBoundary.attachmentOptionOffset;
         second != secondBoundary.attachmentOptionOffset +
                       secondBoundary.attachmentOptionCount;
         ++second) {
      const pnr::PnrIndex other =
          unconstrained->ports().attachmentOptions()[second].endpoint;
      shared |= other == endpoint;
      secondHasAlternative |= other != endpoint;
    }
    if (shared && secondHasAlternative) {
      requiredEndpoint = endpoint;
      break;
    }
  }
  if (!requiredEndpoint)
    fail("attachment relation fixture has no restricted shared endpoint");

  const auto terminal = [&](const auto &producer) {
    return "#mapping.spatial_transfer_terminal<producer = " +
           dataflowAttr("graph_producer_endpoint_ref", dataflow.identity(),
                        producer) +
           ">";
  };
  const std::string text =
      "module {\n  mapping.constraints.spatial dataflow(" +
      identityAttr(dataflow.identity()) + ") tech_mapping(" +
      identityAttr(techMapping.identity()) + ") fabric(" +
      identityAttr(fabric.identity()) +
      ") {\n    mapping.constraint.domain_restriction "
      "projection(spatial_transfer_attachment) subject(" +
      terminal(sources[0]) + ") admissible_domain([" +
      fabricAttr("fabric_transport_endpoint_ref",
                 unconstrained->routing()
                     .routingEndpoints()[*requiredEndpoint]
                     .reference) +
      "])\n    mapping.constraint.equal "
      "projection(spatial_transfer_attachment) subjects([" +
      terminal(sources[0]) + ", " + terminal(sources[1]) + "])\n  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse attachment relation constraint fixture");
  auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
  auto constraints = take(mapping::finalizeSpatialMappingConstraintSet(
      *roots.begin(), dataflow, techMapping, fabric, store));
  auto problem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, constraints.view()));
  auto candidate = take(pnr::createCanonicalSpatialCandidate(problem));
  const pnr::PnrIndex selected =
      candidate->logicalNetSourceEndpoint(sourceNets[0]);
  if (selected != *requiredEndpoint ||
      selected != candidate->logicalNetSourceEndpoint(sourceNets[1]))
    fail("initializer violated a Spatial attachment restriction or equality");

  const auto binding =
      problem->transfers().logicalNetSourceBindings()[sourceNets[1]];
  if (binding.kind != pnr::FrozenSpatialTerminalBindingKind::GraphBoundary)
    fail("attachment relation fixture source is not a graph boundary");
  const auto &boundary = problem->ports().graphBoundaries()[binding.index];
  std::optional<pnr::PnrIndex> forbidden;
  for (pnr::PnrIndex option = boundary.attachmentOptionOffset;
       option !=
       boundary.attachmentOptionOffset + boundary.attachmentOptionCount;
       ++option)
    if (problem->ports().attachmentOptions()[option].endpoint != selected) {
      forbidden = option;
      break;
    }
  if (!forbidden)
    fail("attachment relation fixture has no violating alternative");

  pnr::SpatialCandidateScratch candidateScratch;
  requireSuccess(candidateScratch.prepare(*problem));
  auto move = take(candidate->beginMove(candidateScratch));
  llvm::Error rejectedAttachment =
      move.setGraphBoundaryAttachment(binding.index, *forbidden);
  if (!rejectedAttachment)
    fail("candidate move accepted a restricted Spatial attachment");
  const std::string failure = llvm::toString(std::move(rejectedAttachment));
  if (!llvm::StringRef(failure).contains("relation-domain choice"))
    fail("attachment restriction failed for the wrong reason");
  move.rollback();
  requireSuccess(candidate->verify());

  pnr::SpatialActionDomainScratch actionDomain;
  requireSuccess(actionDomain.prepare(*problem));
  requireSuccess(actionDomain.rebuild(*candidate));
  for (const pnr::SpatialResourceAllocationAction &action :
       actionDomain.view().resourceChoices) {
    const auto *attachment =
        std::get_if<pnr::SpatialGraphBoundaryAttachmentAction>(&action);
    if (!attachment || attachment->boundary != binding.index)
      continue;
    if (problem->ports()
            .attachmentOptions()[attachment->attachmentOption]
            .endpoint != selected)
      fail("Action domain exposed an attachment relation violation");
  }
}
