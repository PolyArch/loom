#include "SpatialMemoryConstraintTestSupport.h"
#include "../TestAllocationProbe.h"

#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCandidateState.h"

#include "../../lib/PnR/SpatialMemoryConstraintModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial memory constraint test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
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

const loom::pnr::FrozenSpatialMemoryDispatchDomain *
dispatchDomain(const loom::pnr::FrozenSpatialPnrProblem &problem,
               const loom::pnr::SpatialCandidateState &candidate,
               loom::pnr::PnrIndex use) {
  const auto &memory = problem.memory();
  if (use >= memory.rootedUses().size())
    return nullptr;
  const auto &rootedUse = memory.rootedUses()[use];
  const auto &realizations = problem.realizations();
  if (rootedUse.actor >= realizations.memoryActorRealizations().size())
    return nullptr;
  const loom::pnr::PnrIndex realization =
      realizations.memoryActorRealizations()[rootedUse.actor];
  const loom::pnr::PnrIndex placement =
      candidate.memoryBinding(realization).placement;
  const auto found =
      llvm::find_if(memory.dispatchDomains(), [&](const auto &domain) {
        return domain.actor == rootedUse.actor && domain.placement == placement;
      });
  return found == memory.dispatchDomains().end() ? nullptr : &*found;
}

bool dispatchMatchesTarget(
    const loom::pnr::FrozenSpatialMemoryIndex &memory,
    const loom::pnr::FrozenSpatialMemoryDispatchOption &option,
    const loom::pnr::FrozenSpatialMemoryBindingTargetOption &target) {
  if (const auto *region =
          std::get_if<loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target)) {
    const auto *local =
        std::get_if<loom::fabric::LocalMemoryServiceRef>(&option.target);
    if (!local || local->underlying() != region->service)
      return false;
    const auto regions = memory.dispatchServiceRegionOrdinals().slice(
        option.serviceRegionOffset, option.serviceRegionCount);
    return std::binary_search(regions.begin(), regions.end(), region->ordinal);
  }
  return std::holds_alternative<loom::fabric::ManagerEndpointRef>(
      option.target);
}

} // namespace

bool loom::test::admitsCanonicalSpatialCandidate(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store) {
  auto constraints = mapping::finalizeEmptySpatialMappingConstraintSet(
      dataflow, techMapping, fabric, store);
  if (!constraints) {
    llvm::consumeError(constraints.takeError());
    return false;
  }
  auto config =
      pnr::projectResolvedSpatialPnrConfigView(loom::defaultResolvedConfig());
  if (!config) {
    llvm::consumeError(config.takeError());
    return false;
  }
  auto problem = pnr::freezeSpatialPnrProblem(dataflow, techMapping, fabric,
                                              *config, constraints->view());
  if (!problem) {
    llvm::consumeError(problem.takeError());
    return false;
  }
  auto candidate = pnr::createCanonicalSpatialCandidate(*problem);
  if (!candidate) {
    llvm::consumeError(candidate.takeError());
    return false;
  }
  if (llvm::Error error = (*candidate)->verify()) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

void loom::test::exerciseSpatialMemoryOperationPortRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store) {
  if (techMapping.memoryRealizations().size() != 1 ||
      techMapping.memoryRealizations().front().actors.size() != 2)
    fail("fixture is not one grouped two-actor memory realization");
  const auto &actors = techMapping.memoryRealizations().front().actors;
  const bool samePort =
      actors[0].operationPort.ordinal == actors[1].operationPort.ordinal;
  const auto buildConstraints = [&](llvm::StringRef relation) {
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(techMapping.identity()) + ") fabric(" +
        identityAttr(fabric.identity()) + ") {\n    mapping.constraint." +
        relation.str() + " projection(memory_operation_port) subjects([" +
        dataflowAttr("actor_ref", dataflow.identity(), actors[0].actor) + ", " +
        dataflowAttr("actor_ref", dataflow.identity(), actors[1].actor) +
        "])\n  }\n}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      fail("cannot parse memory operation-port relation fixture");
    auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
    return take(mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, techMapping, fabric, store));
  };

  const auto pnrConfig = take(
      pnr::projectResolvedSpatialPnrConfigView(loom::defaultResolvedConfig()));
  const auto feasibleConstraints =
      buildConstraints(samePort ? "equal" : "disjoint");
  auto feasibleProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, feasibleConstraints.view()));
  auto feasible = take(pnr::createCanonicalSpatialCandidate(feasibleProblem));
  if (llvm::Error error = feasible->verify())
    fail("memory operation-port relation failed cold verification: " +
         llvm::toString(std::move(error)));

  const auto impossibleConstraints =
      buildConstraints(samePort ? "disjoint" : "equal");
  auto impossibleProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, impossibleConstraints.view()));
  auto impossible = pnr::createCanonicalSpatialCandidate(impossibleProblem);
  if (impossible)
    fail("contradictory memory operation-port relation produced a candidate");
  llvm::consumeError(impossible.takeError());

  if (dataflow.logicalMemoryRoots().size() != 2)
    fail("fixture does not expose two independent logical memory roots");
  const auto memories = fabric.memoryOccurrences();
  if (memories.empty())
    fail("fixture Fabric has no local memory occurrence");
  const auto *service = fabric.localMemoryService(memories.front());
  if (!service || service->regions().empty())
    fail("fixture Fabric has no local memory service region");
  const fabric::FabricMemoryServiceRef serviceRef =
      fabric::FabricMemoryServiceRef::local(memories.front());
  const auto &region = service->regions().front();
  const std::uint64_t regionEnd = region.addressBaseBytes + region.sizeBytes;
  const std::string roots =
      dataflowAttr("logical_memory_root_ref", dataflow.identity(),
                   dataflow.logicalMemoryRoots()[0].ref) +
      ", " +
      dataflowAttr("logical_memory_root_ref", dataflow.identity(),
                   dataflow.logicalMemoryRoots()[1].ref);
  const std::string serviceValue =
      fabricAttr("fabric_memory_service_ref", serviceRef);
  const std::string addressValue =
      "#mapping.constraint_address_region<service = " + serviceValue +
      ", intervals = [#mapping.constraint_unsigned_interval<lower = " +
      std::to_string(region.addressBaseBytes) +
      " : ui64, upper = " + std::to_string(regionEnd) + " : ui64>]>";
  const auto buildMemoryRootConstraints = [&](llvm::StringRef serviceRelation,
                                              llvm::StringRef addressRelation,
                                              bool restrictDomains = true) {
    std::string clauses;
    if (restrictDomains) {
      for (const auto &root : dataflow.logicalMemoryRoots()) {
        const std::string subject = dataflowAttr("logical_memory_root_ref",
                                                 dataflow.identity(), root.ref);
        clauses += "    mapping.constraint.domain_restriction "
                   "projection(memory_bound_services) subject(" +
                   subject + ") admissible_domain([" + serviceValue + "])\n";
        clauses += "    mapping.constraint.domain_restriction "
                   "projection(memory_address_region) subject(" +
                   subject + ") admissible_domain([" + addressValue + "])\n";
      }
    }
    if (!serviceRelation.empty())
      clauses += "    mapping.constraint." + serviceRelation.str() +
                 " projection(memory_bound_services) subjects([" + roots +
                 "])\n";
    if (!addressRelation.empty())
      clauses += "    mapping.constraint." + addressRelation.str() +
                 " projection(memory_address_region) subjects([" + roots +
                 "])\n";
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(techMapping.identity()) + ") fabric(" +
        identityAttr(fabric.identity()) + ") {\n" + clauses + "  }\n}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      fail("cannot parse memory-root relation fixture");
    auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
    return take(mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, techMapping, fabric, store));
  };

  const auto feasibleMemoryRoots =
      buildMemoryRootConstraints("equal", "disjoint");
  auto feasibleMemoryProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, feasibleMemoryRoots.view()));
  auto feasibleMemoryCandidate =
      take(pnr::createCanonicalSpatialCandidate(feasibleMemoryProblem));
  if (llvm::Error error = feasibleMemoryCandidate->verify())
    fail("memory service/address relations failed cold verification: " +
         llvm::toString(std::move(error)));
  for (pnr::PnrIndex binding = 0; binding < 2; ++binding) {
    const auto target =
        feasibleMemoryCandidate->logicalMemoryBinding(binding).target;
    if (target >= feasibleMemoryProblem->memory().bindingTargets().size() ||
        !std::holds_alternative<fabric::FabricMemoryServiceRegionRef>(
            feasibleMemoryProblem->memory().bindingTargets()[target].target))
      fail("feasible memory-root relation escaped through BoundaryProxy");
  }

  pnr::PnrIndex boundaryTarget = pnr::getInvalidPnrIndex();
  for (auto [ordinal, target] :
       llvm::enumerate(feasibleMemoryProblem->memory().bindingTargets()))
    if (std::holds_alternative<pnr::FrozenSpatialMemoryBoundaryProxy>(
            target.target)) {
      boundaryTarget = static_cast<pnr::PnrIndex>(ordinal);
      break;
    }
  if (boundaryTarget == pnr::getInvalidPnrIndex())
    fail("memory relation fixture has no BoundaryProxy target");
  pnr::SpatialActionDomainScratch actionDomain;
  if (llvm::Error error = actionDomain.prepare(*feasibleMemoryProblem))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = actionDomain.rebuild(*feasibleMemoryCandidate))
    fail(llvm::toString(std::move(error)));
  std::optional<pnr::SpatialLogicalMemoryBindingAction> boundaryAction;
  for (const pnr::SpatialResourceAllocationAction &action :
       actionDomain.view().resourceChoices)
    if (const auto *logical =
            std::get_if<pnr::SpatialLogicalMemoryBindingAction>(&action);
        logical && logical->binding == 0 && logical->target == boundaryTarget) {
      boundaryAction = *logical;
      break;
    }
  if (!boundaryAction)
    fail("relation-closed logical-memory Action is absent from its domain");

  std::vector<pnr::SpatialLogicalMemoryBindingSelection> boundedSelections;
  for (pnr::PnrIndex binding = 0;
       binding < feasibleMemoryProblem->memory().logicalBindings().size();
       ++binding)
    boundedSelections.push_back(
        feasibleMemoryCandidate->logicalMemoryBinding(binding));
  const std::array<pnr::PnrIndex, 1> boundedBinding{boundaryAction->binding};
  const std::array<pnr::SpatialLogicalMemoryBindingSelection, 1>
      boundedSelection{pnr::SpatialLogicalMemoryBindingSelection{
          boundaryAction->target, boundaryAction->physicalOffsetBytes}};
  pnr::detail::SpatialMemoryConstraintScratch boundedScratch;
  if (llvm::Error error =
          feasibleMemoryProblem->memoryConstraints().prepareScratch(
              boundedScratch))
    fail(llvm::toString(std::move(error)));
  auto bounded =
      feasibleMemoryProblem->memoryConstraints().solveCanonicalClosure(
          boundedSelections, boundedBinding, boundedSelection, 1,
          [](pnr::PnrIndex, pnr::PnrIndex) -> llvm::Expected<bool> {
            return true;
          },
          boundedScratch);
  if (bounded)
    fail("memory relation closure ignored its assignment work limit");
  bool typedWorkLimit = false;
  llvm::Error unhandled = llvm::handleErrors(
      bounded.takeError(),
      [&](const pnr::detail::SpatialMemoryConstraintSolveFailure &)
          -> llvm::Error {
        typedWorkLimit = true;
        return llvm::Error::success();
      });
  if (unhandled)
    fail(llvm::toString(std::move(unhandled)));
  if (!typedWorkLimit)
    fail("memory relation closure lost its typed work-limit result");

  std::vector<pnr::SpatialLogicalMemoryBindingSelection> originalBindings;
  std::vector<pnr::PnrIndex> originalDispatches;
  std::vector<pnr::PnrIndex> originalExposures;
  for (pnr::PnrIndex binding = 0;
       binding < feasibleMemoryProblem->memory().logicalBindings().size();
       ++binding)
    originalBindings.push_back(
        feasibleMemoryCandidate->logicalMemoryBinding(binding));
  for (pnr::PnrIndex use = 0;
       use < feasibleMemoryProblem->memory().rootedUses().size(); ++use)
    originalDispatches.push_back(
        feasibleMemoryCandidate->memoryUseDispatch(use));
  for (pnr::PnrIndex exposure = 0;
       exposure < feasibleMemoryProblem->memory().exposures().size();
       ++exposure)
    originalExposures.push_back(
        feasibleMemoryCandidate->memoryExposureSelection(exposure));
  const std::vector<std::uint64_t> originalEnvelopeBits(
      feasibleMemoryCandidate->activeResourceTimeEnvelopeBits().begin(),
      feasibleMemoryCandidate->activeResourceTimeEnvelopeBits().end());
  const std::uint64_t originalCapacityOveruse =
      feasibleMemoryCandidate->atomicCapacityOveruse();
  const auto originalObjective =
      take(feasibleMemoryProblem->objectiveProgram().evaluate(
          *feasibleMemoryCandidate));

  pnr::SpatialActionExecutorScratch actionExecutor;
  if (llvm::Error error = actionExecutor.prepare(*feasibleMemoryCandidate))
    fail(llvm::toString(std::move(error)));
  const pnr::SpatialMappingAction action =
      pnr::SpatialResourceAllocationAction{*boundaryAction};
  auto probe = take(actionExecutor.probe(*feasibleMemoryCandidate, action));
  for (pnr::PnrIndex binding = 0; binding < 2; ++binding)
    if (feasibleMemoryCandidate->logicalMemoryBinding(binding).target !=
        boundaryTarget)
      fail("logical-memory Action did not close its Equal relation");
  if (llvm::Error error = probe.commit())
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = feasibleMemoryCandidate->verify())
    fail("relation-closed logical-memory Action failed verification: " +
         llvm::toString(std::move(error)));

  const pnr::SpatialMappingAction restoreAction =
      pnr::SpatialResourceAllocationAction{
          pnr::SpatialLogicalMemoryBindingAction{
              0, originalBindings[0].target,
              originalBindings[0].physicalOffsetBytes}};
  auto restore =
      take(actionExecutor.probe(*feasibleMemoryCandidate, restoreAction));
  if (llvm::Error error = restore.commit())
    fail(llvm::toString(std::move(error)));

  const auto requireRestored = [&]() {
    for (auto [binding, expected] : llvm::enumerate(originalBindings)) {
      const auto &actual =
          feasibleMemoryCandidate->logicalMemoryBinding(binding);
      if (actual.target != expected.target ||
          actual.physicalOffsetBytes != expected.physicalOffsetBytes)
        fail("logical-memory rollback changed a binding");
    }
    for (auto [use, expected] : llvm::enumerate(originalDispatches))
      if (feasibleMemoryCandidate->memoryUseDispatch(use) != expected)
        fail("logical-memory rollback changed a dispatch");
    for (auto [exposure, expected] : llvm::enumerate(originalExposures))
      if (feasibleMemoryCandidate->memoryExposureSelection(exposure) !=
          expected)
        fail("logical-memory rollback changed an exposure");
    if (!llvm::equal(feasibleMemoryCandidate->activeResourceTimeEnvelopeBits(),
                     originalEnvelopeBits) ||
        feasibleMemoryCandidate->atomicCapacityOveruse() !=
            originalCapacityOveruse)
      fail("logical-memory rollback changed derived resource state");
    const auto objective =
        take(feasibleMemoryProblem->objectiveProgram().evaluate(
            *feasibleMemoryCandidate));
    if (objective.codes() != originalObjective.codes())
      fail("logical-memory rollback changed the objective");
    if (llvm::Error error = feasibleMemoryCandidate->verify())
      fail(llvm::toString(std::move(error)));
  };
  requireRestored();

  auto discarded = take(actionExecutor.probe(*feasibleMemoryCandidate, action));
  if (llvm::Error error = discarded.discard())
    fail(llvm::toString(std::move(error)));
  requireRestored();

  for (std::uint64_t warm = 0; warm < 8; ++warm) {
    if (llvm::Error error = actionDomain.rebuild(*feasibleMemoryCandidate))
      fail(llvm::toString(std::move(error)));
    auto warmProbe =
        take(actionExecutor.probe(*feasibleMemoryCandidate, action));
    if (llvm::Error error = warmProbe.discard())
      fail(llvm::toString(std::move(error)));
  }
  const std::size_t retainedDomainBytes = actionDomain.retainedStorageBytes();
  const std::size_t retainedExecutorBytes =
      actionExecutor.retainedStorageBytes();
  startAllocationProbe();
  for (std::uint64_t replay = 0; replay < 32; ++replay) {
    if (llvm::Error error = actionDomain.rebuild(*feasibleMemoryCandidate))
      fail(llvm::toString(std::move(error)));
    auto replayProbe =
        take(actionExecutor.probe(*feasibleMemoryCandidate, action));
    if (llvm::Error error = replayProbe.discard())
      fail(llvm::toString(std::move(error)));
  }
  if (stopAllocationProbe() != 0)
    fail("warm constrained logical-memory Action allocated heap storage");
  if (actionDomain.retainedStorageBytes() != retainedDomainBytes ||
      actionExecutor.retainedStorageBytes() != retainedExecutorBytes)
    fail("warm constrained logical-memory Action grew retained storage");
  requireRestored();

  const auto alternateMemoryRoots =
      buildMemoryRootConstraints("", "disjoint", false);
  auto alternateProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, alternateMemoryRoots.view()));
  auto alternateCandidate =
      take(pnr::createCanonicalSpatialCandidate(alternateProblem));
  pnr::PnrIndex alternateBoundary = pnr::getInvalidPnrIndex();
  for (auto [ordinal, target] :
       llvm::enumerate(alternateProblem->memory().bindingTargets())) {
    if (std::holds_alternative<pnr::FrozenSpatialMemoryBoundaryProxy>(
            target.target))
      alternateBoundary = static_cast<pnr::PnrIndex>(ordinal);
  }
  const pnr::PnrIndex alternatePreferred =
      alternateCandidate->logicalMemoryBinding(1).target;
  if (alternateBoundary == pnr::getInvalidPnrIndex() ||
      alternatePreferred >=
          alternateProblem->memory().bindingTargets().size() ||
      !std::holds_alternative<fabric::FabricMemoryServiceRegionRef>(
          alternateProblem->memory()
              .bindingTargets()[alternatePreferred]
              .target))
    fail("alternate-solution fixture has no preferred local choice");
  pnr::SpatialActionDomainScratch alternateDomain;
  if (llvm::Error error = alternateDomain.prepare(*alternateProblem))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = alternateDomain.rebuild(*alternateCandidate))
    fail(llvm::toString(std::move(error)));
  std::optional<pnr::SpatialLogicalMemoryBindingAction> alternateBoundaryAction;
  for (const pnr::SpatialResourceAllocationAction &candidateAction :
       alternateDomain.view().resourceChoices)
    if (const auto *logical =
            std::get_if<pnr::SpatialLogicalMemoryBindingAction>(
                &candidateAction);
        logical && logical->binding == 0 &&
        logical->target == alternateBoundary) {
      alternateBoundaryAction = *logical;
      break;
    }
  if (!alternateBoundaryAction)
    fail("alternate-solution fixture has no BoundaryProxy Action");
  std::optional<pnr::SpatialMemoryUseDispatchAction> managerDispatch;
  for (auto [use, rootedUse] :
       llvm::enumerate(alternateProblem->memory().rootedUses())) {
    if (!rootedUse.logicalBinding || *rootedUse.logicalBinding != 1)
      continue;
    const auto *domain = dispatchDomain(*alternateProblem, *alternateCandidate,
                                        static_cast<pnr::PnrIndex>(use));
    if (!domain)
      continue;
    for (pnr::PnrIndex option = domain->optionOffset;
         option < domain->optionOffset + domain->optionCount; ++option)
      if (dispatchMatchesTarget(
              alternateProblem->memory(),
              alternateProblem->memory().dispatchOptions()[option],
              alternateProblem->memory().bindingTargets()[alternateBoundary])) {
        managerDispatch = {static_cast<pnr::PnrIndex>(use), option};
        break;
      }
    if (managerDispatch)
      break;
  }
  if (!managerDispatch)
    fail("alternate-solution fixture has no exact manager dispatch");
  const pnr::PnrIndex originalAlternateDispatch =
      alternateCandidate->memoryUseDispatch(managerDispatch->use);
  const std::array<pnr::SpatialMappingAction, 2> alternateBatch{
      pnr::SpatialResourceAllocationAction{*alternateBoundaryAction},
      pnr::SpatialResourceAllocationAction{*managerDispatch}};
  pnr::SpatialActionExecutorScratch alternateExecutor;
  if (llvm::Error error = alternateExecutor.prepare(*alternateCandidate))
    fail(llvm::toString(std::move(error)));
  auto alternateProbe =
      take(alternateExecutor.probeBatch(*alternateCandidate, alternateBatch));
  if (alternateCandidate->logicalMemoryBinding(1).target != alternateBoundary ||
      alternateCandidate->memoryUseDispatch(managerDispatch->use) !=
          managerDispatch->dispatchOption)
    fail("memory closure did not retry for an exact explicit dispatch");
  if (llvm::Error error = alternateProbe.discard())
    fail(llvm::toString(std::move(error)));
  if (alternateCandidate->logicalMemoryBinding(1).target !=
          alternatePreferred ||
      alternateCandidate->memoryUseDispatch(managerDispatch->use) !=
          originalAlternateDispatch)
    fail("alternate-solution Action did not roll back exactly");
  if (llvm::Error error = alternateCandidate->verify())
    fail(llvm::toString(std::move(error)));

  const auto impossibleMemoryRoots =
      buildMemoryRootConstraints("disjoint", "equal");
  auto impossibleMemoryProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, impossibleMemoryRoots.view()));
  auto emptyMemoryCandidate =
      take(pnr::createCanonicalSpatialCandidate(impossibleMemoryProblem));
  if (llvm::Error error = emptyMemoryCandidate->verify())
    fail("empty zero-or-more memory projections failed verification: " +
         llvm::toString(std::move(error)));
  pnr::PnrIndex localTarget = pnr::getInvalidPnrIndex();
  for (auto [ordinal, target] :
       llvm::enumerate(impossibleMemoryProblem->memory().bindingTargets()))
    if (std::holds_alternative<fabric::FabricMemoryServiceRegionRef>(
            target.target)) {
      localTarget = static_cast<pnr::PnrIndex>(ordinal);
      break;
    }
  if (localTarget == pnr::getInvalidPnrIndex())
    fail("memory relation fixture has no local target");
  pnr::SpatialCandidateScratch scratch;
  if (llvm::Error error = scratch.prepare(*impossibleMemoryProblem))
    fail(llvm::toString(std::move(error)));
  auto move = take(emptyMemoryCandidate->beginMove(scratch));
  if (llvm::Error error = move.setLogicalMemoryBinding(0, localTarget, 0))
    fail(llvm::toString(std::move(error)));
  auto closed = move.close();
  if (closed)
    fail("memory service/address relation accepted an unequal local move");
  llvm::consumeError(closed.takeError());
}
