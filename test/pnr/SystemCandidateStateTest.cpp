#include "SystemCandidateFixture.h"
#include "PnR/System/SystemCandidateState.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/ResourceTimeFrontier.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPhysicalTiming.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemMappingMigration.h"
#include "PnR/System/SystemPnrGenerator.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "SystemCandidateStateTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace loom::pnr::test::fixture;

void memoryServiceWorkflow();

void graphBindingWorkflow() {
  using loom::pnr::test::countOccurrences;
  using loom::pnr::test::rawSystemBytes;
  using loom::pnr::test::verifyFinalizedSystemMappingWorkflow;
  using loom::pnr::test::verifySystemResourceActionWorkflow;
  using loom::pnr::test::verifySystemServiceTargetRejections;
  using loom::pnr::test::withFirstCoordinateLowerBound;
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto baselineDesign = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  require(baselineDesign.roots().size() == 1,
          "builtin System fixture did not publish one System root");
  auto primaryDesign =
      loom::pnr::test::buildSystemCandidateSpatialModule(store, false);
  auto primaryModule = primaryDesign.roots().front();
  auto alternateDesign =
      loom::pnr::test::buildSystemCandidateSpatialModule(store, true);
  auto design = loom::pnr::test::buildHeterogeneousSystem(
      store, baselineDesign.roots().front(), primaryModule,
      alternateDesign.roots().front(), context);
  const auto &systemRoot = design.roots().front();
  auto system = take(loom::fabric::requireSystemRoot(systemRoot.view()));
  require(systemRoot.directDependencies().size() == 2,
          "heterogeneous System did not retain both SpatialCores");

  const loom::ResolvedConfig resolved =
      loom::pnr::test::buildSystemCandidateResolvedConfig();
  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));

  std::vector<loom::ArtifactRootReference> spatialMappings;
  for (const auto &dependency : systemRoot.directDependencies()) {
    auto module =
        take(loom::fabric::importEntireFabricRoot(dependency.root, store));
    auto spatialReference =
        generateSpatialMapping(dataflow, module, resolved, store);
    spatialMappings.push_back(spatialReference);
  }
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    roots.push_back(root.ref);
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflow, system, roots, store));
  auto partition = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflow, constraints.view().rootThreadLaunches()));
  auto searchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      loom::pnr::SystemHierarchicalGraphSearchInput{spatialMappings}, store));
  require(!searchDomain.serviceObligations().empty(),
          "System route fixture has no service obligation");
  auto physicalTimingProfiles =
      take(loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(system));
  auto problem = take(loom::pnr::freezeSystemPnrProblem(
      dataflow, system, physicalTimingProfiles, searchDomain, config,
      constraints, store));

  auto importedCapacitySearch =
      take(loom::pnr::searchSystemImportedCapacity(problem));
  const auto *capacityFit = std::get_if<loom::pnr::SystemImportedCapacityFit>(
      &importedCapacitySearch);
  require(capacityFit && capacityFit->assignmentAttempts != 0,
          "exact imported-capacity search found no fitting binding");
  auto capacityFitCandidate = take(
      loom::pnr::initializeSystemCandidateAttemptWithImportedCapacityClosure(
          problem, 0));
  require(capacityFitCandidate.state->capacityOveruse() == 0,
          "exact imported-capacity binding did not close capacity");
  if (llvm::Error error = capacityFitCandidate.state->verify())
    fail(llvm::toString(std::move(error)));
  const auto finalizedParentMapping =
      take(loom::pnr::finalizeSystemMappingCandidate(
          *capacityFitCandidate.state, dataflow, system, constraints.view(),
          store, context));

  auto pressureDataflowArtifact = buildCapacityPressureDataflow(context);
  const auto pressureDataflowReference =
      take(dataflow::publishCanonicalDataflow(pressureDataflowArtifact, store));
  auto pressureDataflow = take(pressureDataflowArtifact.view());
  std::vector<loom::ArtifactRootReference> pressureSpatialMappings;
  for (const auto &graph : pressureDataflow.graphs())
    pressureSpatialMappings.push_back(generateSpatialMapping(
        pressureDataflow, primaryModule, resolved, store, nullptr, graph.ref));
  require(pressureSpatialMappings.size() == 2 &&
              pressureSpatialMappings.front() != pressureSpatialMappings.back(),
          "capacity-pressure fixture did not publish two distinct Mappings");
  std::vector<dataflow::RootThreadLaunchRef> pressureRoots;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       pressureDataflow.rootThreadLaunches())
    pressureRoots.push_back(root.ref);
  require(pressureRoots.size() == 2,
          "capacity-pressure fixture does not have two execution roots");

  const auto transitionTrigger = [&]() {
    return dataflow::rootThreadCompletionEventFamily(pressureRoots.front());
  };
  const loom::fabric::FabricPhysicalOccurrenceOwnerRef transitionResource;
  const loom::ArtifactRootReference transitionDeployment{
      loom::deployment::deploymentSchema.identity.str(),
      loom::deployment::deploymentSchema.version,
      finalizedParentMapping.reference().artifact};
  loom::pnr::ResourceTimeTransition structuralTransition{
      transitionTrigger(),
      loom::pnr::ResourceTimeSafePointReference{
          pressureDataflowReference,
          loom::pnr::ResourceTimeSafePointKind::Completion},
      {finalizedParentMapping.reference(), transitionDeployment},
      {finalizedParentMapping.reference(), transitionDeployment},
      {{pressureRoots.front(), {transitionResource}}},
      {{pressureRoots.front(), {transitionResource}}},
      {},
      {},
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      loom::pnr::ResourceTimeTransitionStatus::ProofNotEstablished};
  if (llvm::Error error =
          loom::pnr::validateResourceTimeTransition(structuralTransition))
    fail(llvm::toString(std::move(error)));
  structuralTransition.status =
      loom::pnr::ResourceTimeTransitionStatus::Verified;
  structuralTransition.reprogrammingTimePicoseconds = 1;
  structuralTransition.migrationTimePicoseconds = 1;
  structuralTransition.resourceDeltaDigest = config.digest();
  structuralTransition.configurationDeltaDigest = config.digest();
  structuralTransition.routeDeltaDigest = config.digest();
  if (llvm::Error error =
          loom::pnr::validateResourceTimeTransition(structuralTransition))
    fail(llvm::toString(std::move(error)));
  const loom::dse::ResourceTimeTransitionCacheKeyInput transitionCacheInput{
      constraints.reference(), config.digest(), systemRoot.reference(),
      config.digest(), config.digest()};
  const auto transitionCacheKey =
      take(loom::dse::deriveResourceTimeTransitionCacheKey(
          structuralTransition, transitionCacheInput));
  auto changedTransition = structuralTransition;
  changedTransition.afterActive.front().resources = {
      take(loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
          loom::fabric::FabricInventoryOwnerRef::of(
              loom::fabric::HostCoreOccurrenceRef(1))))};
  const auto changedTransitionCacheKey =
      take(loom::dse::deriveResourceTimeTransitionCacheKey(
          changedTransition, transitionCacheInput));
  require(transitionCacheKey != changedTransitionCacheKey,
          "resource-time transition cache key ignored an allocation delta");
  changedTransition = structuralTransition;
  changedTransition.completedBefore = {pressureRoots.back()};
  require(transitionCacheKey !=
              take(loom::dse::deriveResourceTimeTransitionCacheKey(
                  changedTransition, transitionCacheInput)),
          "resource-time transition cache key ignored its completion frontier");
  changedTransition = structuralTransition;
  changedTransition.migrationTimePicoseconds = 2;
  require(transitionCacheKey ==
              take(loom::dse::deriveResourceTimeTransitionCacheKey(
                  changedTransition, transitionCacheInput)),
          "resource-time transition result changed its semantic cache key");
  changedTransition = structuralTransition;
  changedTransition.reprogrammingTimePicoseconds = 2;
  require(transitionCacheKey ==
              take(loom::dse::deriveResourceTimeTransitionCacheKey(
                  changedTransition, transitionCacheInput)),
          "resource-time reprogramming result changed its semantic cache key");
  changedTransition = structuralTransition;
  changedTransition.status =
      loom::pnr::ResourceTimeTransitionStatus::ProofNotEstablished;
  require(transitionCacheKey ==
              take(loom::dse::deriveResourceTimeTransitionCacheKey(
                  changedTransition, transitionCacheInput)),
          "resource-time transition status changed its semantic cache key");
  changedTransition = structuralTransition;
  changedTransition.child.deployment = loom::ArtifactRootReference{
      loom::deployment::deploymentSchema.identity.str(),
      loom::deployment::deploymentSchema.version,
      pressureDataflowReference.artifact};
  require(transitionCacheKey !=
              take(loom::dse::deriveResourceTimeTransitionCacheKey(
                  changedTransition, transitionCacheInput)),
          "resource-time transition cache key ignored its child Deployment");
  loom::pnr::ResourceTimeTransitionSequence transitionSequence{
      {structuralTransition, structuralTransition}};
  if (llvm::Error error =
          loom::pnr::validateResourceTimeTransitionSequence(transitionSequence))
    fail(llvm::toString(std::move(error)));
  const auto requireTransitionFailure = [&](llvm::Error error,
                                            llvm::StringRef fragment) {
    if (!error)
      fail("malformed resource-time transition unexpectedly validated");
    const std::string message = llvm::toString(std::move(error));
    require(llvm::StringRef(message).contains(fragment), message);
  };
  transitionSequence.transitions[1].parent.mapping =
      pressureSpatialMappings.front();
  requireTransitionFailure(
      loom::pnr::validateResourceTimeTransitionSequence(transitionSequence),
      "resource-time transition sequence is not chained");
  auto duplicateResource = structuralTransition;
  duplicateResource.beforeActive.push_back(
      {pressureRoots.back(), {transitionResource}});
  requireTransitionFailure(
      loom::pnr::validateResourceTimeTransition(duplicateResource),
      "assigns one physical resource");
  auto missingDeployment = structuralTransition;
  missingDeployment.child.deployment.reset();
  requireTransitionFailure(
      loom::pnr::validateResourceTimeTransition(missingDeployment),
      "no exact parent and child Deployment references");
  auto nonCompletionSafePoint = structuralTransition;
  nonCompletionSafePoint.trigger =
      dataflow::rootThreadStartEventFamily(pressureRoots.front());
  requireTransitionFailure(
      loom::pnr::validateResourceTimeTransition(nonCompletionSafePoint),
      "not the completion event of an active parent region");

  std::optional<loom::fabric::AccCoreOccurrenceRef> pressureCore;
  for (const auto core : system.artifact().accCoreOccurrences()) {
    const auto target = system.spatialCoreTarget(core);
    require(target && target->dependencyOrdinal <
                          system.artifact().importedModules().size(),
            "System AccCore has no exact Module target");
    if (system.artifact()
            .importedModules()[target->dependencyOrdinal]
            .identity() == primaryModule.reference().artifact) {
      pressureCore = core;
      break;
    }
  }
  require(pressureCore.has_value(),
          "System fixture has no pressure-test AccCore");
  auto pressureModule =
      buildSystemConstraintModule(context, pressureDataflow.identity(),
                                  system.artifact().identity(), pressureRoots);
  auto pressureRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      pressureModule->getBody()->front());
  mlir::OpBuilder pressureBuilder(&context);
  const std::array<mlir::Attribute, 1> pressureCoreDomain = {
      constraintFabricAttr<::mapping::FabricAccCoreOccurrenceRefAttr>(
          &context, *pressureCore)};
  for (const auto root : pressureRoots)
    addSystemRestriction(
        pressureBuilder, pressureRoot,
        ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
        constraintDataflowAttr<::mapping::RootThreadLaunchRefAttr>(
            &context, pressureDataflow.identity(), root),
        pressureCoreDomain);
  auto pressureConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          pressureRoot, pressureDataflow, system, store));
  auto pressurePartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          pressureDataflow, pressureConstraints.view().rootThreadLaunches()));
  auto pressureSearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      pressureDataflow, system, config, pressureConstraints, pressurePartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{pressureSpatialMappings},
      store));
  auto pressureProblem = take(loom::pnr::freezeSystemPnrProblem(
      pressureDataflow, system, physicalTimingProfiles, pressureSearchDomain,
      config, pressureConstraints, store));
  auto pressureSearch =
      take(loom::pnr::searchSystemImportedCapacity(pressureProblem));
  const auto *capacityPressure =
      std::get_if<loom::pnr::SystemImportedCapacityPressure>(&pressureSearch);
  require(
      capacityPressure &&
          capacityPressure->checkpointChoices.size() ==
              pressureProblem->threadDecisions().size() +
                  pressureProblem->graphDecisions().size() &&
          capacityPressure->witness.usage > capacityPressure->witness.capacity,
      "two-Mapping imported-capacity search did not produce a real witness");

  loom::BlobStore blobs(directory.path());
  require(capacityPressure->witness.namespaceOrdinal > 0 &&
              capacityPressure->witness.namespaceOrdinal <=
                  pressureProblem->accCores().size(),
          "imported-capacity witness names a foreign AccCore namespace");
  const auto checkpointWitness =
      pressureProblem
          ->accCores()[capacityPressure->witness.namespaceOrdinal - 1];
  require(checkpointWitness == *pressureCore,
          "imported-capacity witness selected the wrong Module target");
  const auto &pressureChoices = capacityPressure->checkpointChoices;
  std::vector<loom::mapping::SystemThreadExecutionCheckpoint> checkpointThreads;
  for (const auto &[decision, frozen] :
       llvm::enumerate(pressureProblem->threadDecisions())) {
    const auto catalog = pressureProblem->threadChoiceCatalogOrdinals(decision);
    require(pressureChoices[decision] < catalog.size(),
            "checkpoint thread choice is outside its exact H domain");
    checkpointThreads.push_back(
        {frozen.root, frozen.cell,
         pressureProblem->accCores()[catalog[pressureChoices[decision]]]});
  }
  std::vector<loom::mapping::SystemGraphExecutionCheckpoint> checkpointGraphs;
  for (const auto &[decision, frozen] :
       llvm::enumerate(pressureProblem->graphDecisions())) {
    const auto catalog = pressureProblem->graphChoiceCatalogOrdinals(decision);
    const std::size_t choiceOffset = pressureProblem->threadDecisions().size();
    require(pressureChoices[choiceOffset + decision] < catalog.size(),
            "checkpoint graph choice is outside its exact H domain");
    checkpointGraphs.push_back(
        {frozen.launch, frozen.cell,
         pressureProblem->spatialMappings()
             [catalog[pressureChoices[choiceOffset + decision]]]});
  }
  std::vector<dataflow::RootThreadLaunchRef> checkpointDependencyRoots;
  for (const auto &binding : checkpointThreads)
    if (binding.target == checkpointWitness)
      checkpointDependencyRoots.push_back(binding.root);
  llvm::sort(checkpointDependencyRoots, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  checkpointDependencyRoots.erase(std::unique(checkpointDependencyRoots.begin(),
                                              checkpointDependencyRoots.end()),
                                  checkpointDependencyRoots.end());
  require(!checkpointDependencyRoots.empty(),
          "capacity witness owns no checkpoint dependency root");
  auto checkpointSearchDomainDigest = loom::ComponentViewDigest::fromBytes(
      pressureSearchDomain.digest().bytes());
  require(static_cast<bool>(checkpointSearchDomainDigest),
          "System search-domain digest is malformed");
  const auto checkpoint =
      take(loom::mapping::finalizeSystemExecutionBindingCheckpoint(
          pressureDataflowReference, systemRoot.reference(),
          pressureConstraints.reference(), config.digest(),
          std::move(*checkpointSearchDomainDigest),
          {loom::mapping::SystemExecutionBindingCheckpointIncompleteKind::
               ImportedSpatialCapacity,
           checkpointWitness, capacityPressure->witness.usage,
           capacityPressure->witness.capacity, checkpointDependencyRoots},
          std::move(checkpointThreads), std::move(checkpointGraphs), store));
  const auto importedCheckpoint =
      take(loom::mapping::importSystemExecutionBindingCheckpoint(
          checkpoint.reference(), store));
  require(importedCheckpoint.dataflow() == pressureDataflowReference &&
              importedCheckpoint.system() == systemRoot.reference() &&
              importedCheckpoint.threadBindings().size() ==
                  pressureProblem->threadDecisions().size() &&
              importedCheckpoint.graphBindings().size() ==
                  pressureProblem->graphDecisions().size(),
          "System execution-binding checkpoint lost its exact owners");
  const auto capacityFeedback =
      loom::pnr::test::verifySystemCapacityPressureRoundTrip(
          store, systemRoot, system, primaryModule.reference(),
          importedCheckpoint, pressureDataflowReference,
          pressureSpatialMappings, capacityPressure->assignmentAttempts);
  const auto mismatchedCapacityFeedback =
      take(loom::mapping::SystemAccCoreCapacityPressure::get(
          capacityFeedback.system(), capacityFeedback.targetModule(),
          capacityFeedback.witnessAccCore(),
          capacityFeedback.spatialMappings().vec(),
          capacityFeedback.compatibleAccCoreCount(),
          capacityFeedback.assignmentAttempts(),
          capacityFeedback.witnessUsage() + 1,
          capacityFeedback.witnessCapacity(),
          capacityFeedback.executionBindingCheckpoint()));
  requireFailureContains(loom::mapping::adoptSystemAccCoreCapacityPressure(
                             loom::mapping::encodeSystemAccCoreCapacityPressure(
                                 mismatchedCapacityFeedback),
                             systemRoot.reference(), pressureDataflowReference,
                             pressureSpatialMappings, store),
                         "disagrees with its checkpoint witness");

  std::optional<loom::fabric::AccCoreOccurrenceRef> primaryPrototype;
  for (const auto core : system.artifact().accCoreOccurrences()) {
    const auto target = system.spatialCoreTarget(core);
    require(target && target->dependencyOrdinal <
                          system.artifact().importedModules().size(),
            "System AccCore has no exact Module target");
    if (system.artifact()
            .importedModules()[target->dependencyOrdinal]
            .identity() == primaryModule.reference().artifact) {
      if (!primaryPrototype)
        primaryPrototype = core;
    }
  }
  require(primaryPrototype.has_value(),
          "heterogeneous System has no primary AccCore prototype");
  const auto witnessAccCore = capacityFeedback.witnessAccCore();
  std::vector<loom::dse::SystemCompositionDecisionDomain> growthDomains = {
      loom::dse::AddAccCoreDomain{*primaryPrototype,
                                  {primaryModule.reference()}}};
  const auto growthConfig =
      take(loom::dse::resolveSystemCompositionRewriteConfig(growthDomains, 1));
  const auto growthInputs =
      take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
          {systemRoot.reference()}, {primaryModule.reference()}));
  const auto growthBinding =
      take(loom::dse::resolveSystemCompositionCandidateGeneratorBinding(
          growthConfig));
  const auto growthResult = take(loom::dse::invokeCandidateGenerator(
      growthInputs, growthBinding, store, blobs));
  const auto *growthCompleted =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &growthResult.outcome);
  require(growthCompleted && growthCompleted->outputBindings.size() == 1 &&
              growthCompleted->outputBindings.front().artifacts.size() == 1,
          "typed AddAccCore did not publish one child System");
  const auto childRoot = take(loom::fabric::importEntireFabricRoot(
      growthCompleted->outputBindings.front().artifacts.front(), store));
  const auto childSystem =
      take(loom::fabric::requireSystemRoot(childRoot.view()));
  require(childSystem.artifact().accCoreOccurrences().size() ==
              system.artifact().accCoreOccurrences().size() + 1,
          "typed AddAccCore child has the wrong occurrence count");
  require(growthCompleted->lineageEdges.size() == 1 &&
              growthCompleted->lineageEdges.front().output ==
                  childRoot.reference(),
          "typed AddAccCore child has no exact transformation lineage");
  const auto growthLineage = take(loom::dse::adoptSystemCompositionDecision(
      growthCompleted->lineageEdges.front().ownerPayload));
  require(growthLineage.parent == systemRoot.reference() &&
              llvm::count_if(
                  growthLineage.entities,
                  [](const auto &entry) {
                    return entry.source.kind ==
                           loom::fabric::FabricEntityKind::AccCoreOccurrence;
                  }) == system.artifact().accCoreOccurrences().size(),
          "typed AddAccCore lineage lost a parent AccCore");
  const auto correspondence =
      loom::pnr::test::verifySystemAccCoreCorrespondence(
          store, systemRoot, system, childRoot, growthLineage.entities,
          growthLineage.transferPatterns);
  const auto checkpointGrowth = loom::pnr::test::applySystemCompositionDecision(
      store, blobs, systemRoot.reference(), {primaryModule.reference()},
      loom::dse::AddAccCoreDomain{witnessAccCore, {primaryModule.reference()}});
  auto checkpointChildRoot =
      take(loom::fabric::importEntireFabricRoot(checkpointGrowth.child, store));
  auto checkpointChildSystem =
      take(loom::fabric::requireSystemRoot(checkpointChildRoot.view()));
  const auto checkpointCorrespondence =
      loom::pnr::test::verifySystemAccCoreCorrespondence(
          store, systemRoot, system, checkpointChildRoot,
          checkpointGrowth.lineage.entities,
          checkpointGrowth.lineage.transferPatterns);
  auto checkpointChildConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          pressureDataflow, checkpointChildSystem, pressureRoots, store));
  auto checkpointChildPartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          pressureDataflow,
          checkpointChildConstraints.view().rootThreadLaunches()));
  auto checkpointChildSearchDomain =
      take(loom::pnr::projectSystemPnrSearchDomain(
          pressureDataflow, checkpointChildSystem, config,
          checkpointChildConstraints, checkpointChildPartition,
          loom::pnr::SystemHierarchicalGraphSearchInput{
              pressureSpatialMappings},
          store));
  auto checkpointChildPhysicalTiming =
      take(loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(
          checkpointChildSystem));
  auto checkpointChildProblem = take(loom::pnr::freezeSystemPnrProblem(
      pressureDataflow, checkpointChildSystem, checkpointChildPhysicalTiming,
      checkpointChildSearchDomain, config, checkpointChildConstraints, store));
  const auto checkpointMigrationContext =
      take(loom::pnr::SystemMappingMigrationContext::get(
          checkpointChildConstraints.reference(), pressureSpatialMappings,
          config.digest()));
  auto childConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflow, childSystem, roots, store));
  const auto migrationContext =
      take(loom::pnr::SystemMappingMigrationContext::get(
          childConstraints.reference(), spatialMappings, config.digest()));
  const auto migrationSeed =
      take(loom::pnr::finalizeSystemMappingCheckpointMigrationSeed(
          checkpoint.reference(), checkpointCorrespondence,
          checkpointMigrationContext, witnessAccCore, store));
  const auto importedMigrationSeed =
      take(loom::pnr::importSystemMappingCheckpointMigrationSeed(
          migrationSeed.reference(), store));
  require(importedMigrationSeed.checkpoint().reference() ==
                  checkpoint.reference() &&
              importedMigrationSeed.correspondence().accCores().size() ==
                  system.artifact().accCoreOccurrences().size() &&
              importedMigrationSeed.context().childConstraints() ==
                  checkpointChildConstraints.reference() &&
              importedMigrationSeed.context().resolvedPnrConfigDigest() ==
                  config.digest() &&
              importedMigrationSeed.reopenedParentAccCore() == witnessAccCore,
          "System migration seed lost its exact problem closure");
  auto childPartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          dataflow, childConstraints.view().rootThreadLaunches()));
  auto childSearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, childSystem, config, childConstraints, childPartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{spatialMappings}, store));
  auto childPhysicalTiming = take(
      loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(childSystem));
  auto childProblem = take(loom::pnr::freezeSystemPnrProblem(
      dataflow, childSystem, childPhysicalTiming, childSearchDomain, config,
      childConstraints, store));
  const auto finalizedMigrationSeed =
      take(loom::pnr::finalizeSystemMappingMigrationSeed(
          finalizedParentMapping.reference(), correspondence, migrationContext,
          store));
  const auto importedFinalizedMigrationSeed =
      take(loom::pnr::importSystemMappingMigrationSeed(
          finalizedMigrationSeed.reference(), store));
  const auto fullyRebasedChild =
      loom::pnr::generateSystemMappings({dataflow,
                                         childSystem,
                                         childPhysicalTiming,
                                         childSearchDomain,
                                         config,
                                         childConstraints,
                                         store,
                                         {},
                                         nullptr,
                                         nullptr,
                                         &importedFinalizedMigrationSeed,
                                         nullptr});
  const auto *fullyRebasedMappings =
      std::get_if<loom::pnr::GeneratedSystemMappings>(&fullyRebasedChild);
  require(
      fullyRebasedMappings && fullyRebasedMappings->candidates.size() == 1 &&
          fullyRebasedMappings->accounting.migrationSeedPrepared == 1 &&
          fullyRebasedMappings->accounting.migrationSeedFallbacks == 0 &&
          fullyRebasedMappings->accounting.migrationPreservedThreadBindings ==
              childProblem->threadDecisions().size() &&
          fullyRebasedMappings->accounting.migrationPreservedGraphBindings ==
              childProblem->graphDecisions().size() &&
          fullyRebasedMappings->accounting.migrationPreservedServiceLegs ==
              childProblem->serviceLegs().size() &&
          fullyRebasedMappings->accounting.migrationReopenedThreadBindings ==
              0 &&
          fullyRebasedMappings->accounting.migrationReopenedGraphBindings ==
              0 &&
          fullyRebasedMappings->accounting.migrationReopenedServiceLegs == 0 &&
          fullyRebasedMappings->accounting.migrationReopenedResourceUses == 0,
      "finalized SystemMapping did not preserve its complete child closure");
  const auto fullyRebasedMapping = take(loom::mapping::importSystemMapping(
      fullyRebasedMappings->candidates.front(), store));
  require(fullyRebasedMapping.view().fabricIdentity() ==
                  childRoot.reference().artifact &&
              fullyRebasedMapping.view().serviceRealizations().size() ==
                  finalizedParentMapping.view().serviceRealizations().size() &&
              fullyRebasedMapping.view().resourceUses().size() ==
                  finalizedParentMapping.view().resourceUses().size(),
          "finalized SystemMapping rebase lost service or ResourceUse state");

  // A resource-time allocation change keeps the immutable System but releases
  // one exact root-owned decision cone. This is a real preserve-first PnR
  // attempt, not a copied Mapping label or a hardware-growth correspondence.
  const auto scheduleRoot = roots.back();
  std::vector<loom::fabric::AccCoreOccurrenceRef> parentScheduleTargets;
  for (const auto &binding :
       finalizedParentMapping.view().executionBindings().threadBindings()) {
    if (binding.key != scheduleRoot)
      continue;
    for (const auto &clause : binding.clauses)
      parentScheduleTargets.push_back(clause.target);
    if (binding.defaultTarget)
      parentScheduleTargets.push_back(*binding.defaultTarget);
  }
  llvm::sort(parentScheduleTargets, [](const auto lhs, const auto rhs) {
    return lhs.id() < rhs.id();
  });
  parentScheduleTargets.erase(
      std::unique(parentScheduleTargets.begin(), parentScheduleTargets.end()),
      parentScheduleTargets.end());
  require(parentScheduleTargets.size() == 1,
          "schedule migration fixture has no exact parent AccCore");
  const auto parentScheduleTarget = parentScheduleTargets.front();
  const auto parentModuleTarget =
      system.spatialCoreTarget(parentScheduleTarget);
  require(parentModuleTarget && parentModuleTarget->dependencyOrdinal <
                                    systemRoot.directDependencies().size(),
          "schedule migration parent has no exact Module target");
  const auto parentModuleReference =
      systemRoot.directDependencies()[parentModuleTarget->dependencyOrdinal]
          .root;
  std::optional<loom::fabric::AccCoreOccurrenceRef> alternateScheduleTarget;
  for (const auto core : system.artifact().accCoreOccurrences()) {
    if (core == parentScheduleTarget)
      continue;
    const auto target = system.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= systemRoot.directDependencies().size())
      continue;
    if (systemRoot.directDependencies()[target->dependencyOrdinal].root ==
        parentModuleReference) {
      alternateScheduleTarget = core;
      break;
    }
  }
  require(alternateScheduleTarget.has_value(),
          "schedule migration fixture has no compatible alternate AccCore");

  auto scheduleConstraintModule = buildSystemConstraintModule(
      context, dataflow.identity(), system.artifact().identity(), roots);
  auto scheduleConstraintRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      scheduleConstraintModule->getBody()->front());
  mlir::OpBuilder scheduleConstraintBuilder(&context);
  const std::array<mlir::Attribute, 1> scheduleTargetDomain = {
      constraintFabricAttr<::mapping::FabricAccCoreOccurrenceRefAttr>(
          &context, *alternateScheduleTarget)};
  addSystemRestriction(
      scheduleConstraintBuilder, scheduleConstraintRoot,
      ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
      constraintDataflowAttr<::mapping::RootThreadLaunchRefAttr>(
          &context, dataflow.identity(), scheduleRoot),
      scheduleTargetDomain);
  auto scheduleConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          scheduleConstraintRoot, dataflow, system, store));
  auto schedulePartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          dataflow, scheduleConstraints.view().rootThreadLaunches()));
  auto scheduleSearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, scheduleConstraints, schedulePartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{spatialMappings}, store));
  auto scheduleProblem = take(loom::pnr::freezeSystemPnrProblem(
      dataflow, system, physicalTimingProfiles, scheduleSearchDomain, config,
      scheduleConstraints, store));
  const auto identityCorrespondence =
      take(loom::pnr::SystemExecutionBindingCorrespondence::getIdentity(
          systemRoot.reference(), store));
  const auto scheduleMigrationContext =
      take(loom::pnr::SystemMappingMigrationContext::get(
          scheduleConstraints.reference(), spatialMappings, config.digest()));
  const std::array scheduleReopenedRoots{scheduleRoot};
  const auto scheduleMigrationSeed =
      take(loom::pnr::finalizeSystemMappingMigrationSeed(
          finalizedParentMapping.reference(), identityCorrespondence,
          scheduleMigrationContext, scheduleReopenedRoots, store));
  const auto importedScheduleMigrationSeed =
      take(loom::pnr::importSystemMappingMigrationSeed(
          scheduleMigrationSeed.reference(), store));
  require(importedScheduleMigrationSeed.reopenedRoots() ==
                  llvm::ArrayRef<dataflow::RootThreadLaunchRef>(
                      scheduleReopenedRoots) &&
              importedScheduleMigrationSeed.correspondence().parentSystem() ==
                  importedScheduleMigrationSeed.correspondence().childSystem(),
          "schedule migration seed lost its typed root delta");
  const auto scheduleRepair =
      loom::pnr::generateSystemMappings({dataflow,
                                         system,
                                         physicalTimingProfiles,
                                         scheduleSearchDomain,
                                         config,
                                         scheduleConstraints,
                                         store,
                                         {},
                                         nullptr,
                                         nullptr,
                                         &importedScheduleMigrationSeed,
                                         nullptr});
  const auto *scheduleMappings =
      std::get_if<loom::pnr::GeneratedSystemMappings>(&scheduleRepair);
  require(
      scheduleMappings && !scheduleMappings->candidates.empty() &&
          scheduleMappings->accounting.migrationSeedPrepared == 1 &&
          scheduleMappings->accounting.migrationSeedFallbacks == 0 &&
          scheduleMappings->accounting.migrationReopenedThreadBindings != 0 &&
          scheduleMappings->accounting.migrationReopenedGraphBindings != 0 &&
          scheduleMappings->accounting.migrationPreservedThreadBindings +
                  scheduleMappings->accounting
                      .migrationReopenedThreadBindings ==
              scheduleProblem->threadDecisions().size() &&
          scheduleMappings->accounting.migrationPreservedGraphBindings +
                  scheduleMappings->accounting.migrationReopenedGraphBindings ==
              scheduleProblem->graphDecisions().size() &&
          scheduleMappings->accounting.migrationReopenedServiceLegs != 0 &&
          scheduleMappings->accounting.migrationPreservedServiceLegs +
          scheduleMappings->accounting.migrationReopenedServiceLegs ==
              scheduleProblem->serviceLegs().size(),
      "resource-time root delta did not drive bounded System repair");
  const auto scheduleMapping = take(loom::mapping::importSystemMapping(
      scheduleMappings->candidates.front(), store));
  require(scheduleMapping.reference() != finalizedParentMapping.reference(),
          "resource-time repair reproduced the unchanged parent Mapping");
  require(roots.size() > 1,
          "resource-time preservation fixture requires a preserved root");
  require(take(loom::pnr::preservesSystemMappingMigrationCone(
              finalizedParentMapping.view(), scheduleMapping.view(),
              scheduleReopenedRoots, store)),
          "resource-time repair changed a cone-external System selection");
  const std::array incorrectlyReopenedRoots{roots.front()};
  require(!take(loom::pnr::preservesSystemMappingMigrationCone(
              finalizedParentMapping.view(), scheduleMapping.view(),
              incorrectlyReopenedRoots, store)),
          "preservation check admitted a changed cone-external root");
  std::vector<loom::fabric::AccCoreOccurrenceRef> repairedScheduleTargets;
  for (const auto &binding :
       scheduleMapping.view().executionBindings().threadBindings()) {
    if (binding.key != scheduleRoot)
      continue;
    for (const auto &clause : binding.clauses)
      repairedScheduleTargets.push_back(clause.target);
    if (binding.defaultTarget)
      repairedScheduleTargets.push_back(*binding.defaultTarget);
  }
  llvm::sort(repairedScheduleTargets, [](const auto lhs, const auto rhs) {
    return lhs.id() < rhs.id();
  });
  repairedScheduleTargets.erase(std::unique(repairedScheduleTargets.begin(),
                                            repairedScheduleTargets.end()),
                                repairedScheduleTargets.end());
  require(repairedScheduleTargets.size() == 1 &&
              repairedScheduleTargets.front() == *alternateScheduleTarget,
          "resource-time repair did not realize the requested allocation");
  const auto migrated = loom::pnr::projectSystemMappingMigrationSeed(
      importedMigrationSeed, *checkpointChildProblem);
  const auto *projection =
      std::get_if<loom::pnr::SystemMappingMigrationProjection>(&migrated);
  const std::uint64_t expectedReopenedThreads = static_cast<std::uint64_t>(
      llvm::count_if(checkpoint.threadBindings(), [&](const auto &row) {
        return row.target == witnessAccCore;
      }));
  require(projection &&
              projection->preservedThreadBindings ==
                  checkpointChildProblem->threadDecisions().size() -
                      expectedReopenedThreads &&
              projection->releasedChoices.size() == expectedReopenedThreads &&
              expectedReopenedThreads != 0 &&
              projection->preservedGraphBindings ==
                  checkpointChildProblem->graphDecisions().size(),
          "System migration did not isolate its capacity-pressure cone");
  auto migratedCandidate =
      take(loom::pnr::initializeSystemCandidateWithReleasedChoices(
          checkpointChildProblem, projection->fixedChoices,
          projection->releasedChoices));
  if (llvm::Error error = migratedCandidate.state->verify())
    fail(llvm::toString(std::move(error)));

  const auto generatedChild =
      loom::pnr::generateSystemMappings({pressureDataflow,
                                         checkpointChildSystem,
                                         checkpointChildPhysicalTiming,
                                         checkpointChildSearchDomain,
                                         config,
                                         checkpointChildConstraints,
                                         store,
                                         {},
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         &importedMigrationSeed});
  const auto *generatedMappings =
      std::get_if<loom::pnr::GeneratedSystemMappings>(&generatedChild);
  require(
      generatedMappings && !generatedMappings->candidates.empty() &&
          generatedMappings->accounting.migrationSeedPrepared == 1 &&
          generatedMappings->accounting.migrationSeedFallbacks == 0 &&
          generatedMappings->accounting.migrationPreservedThreadBindings ==
              checkpointChildProblem->threadDecisions().size() -
                  expectedReopenedThreads &&
          generatedMappings->accounting.migrationPreservedGraphBindings ==
              checkpointChildProblem->graphDecisions().size() &&
          generatedMappings->accounting.migrationReopenedThreadBindings ==
              expectedReopenedThreads &&
          generatedMappings->accounting.migrationReopenedGraphBindings == 0 &&
          generatedMappings->accounting.migrationReopenedServiceLegs == 0 &&
          generatedMappings->accounting.migrationNewServiceLegs ==
              checkpointChildProblem->serviceLegs().size() &&
          generatedMappings->accounting.migrationNewResourceUses ==
              checkpointChildProblem->instructionUsePatternDomains().size() +
                  checkpointChildProblem->consistencyUsePatternDomains().size(),
      "System PnR did not consume its preserve-first migration seed");
  const auto migratedMapping = take(loom::mapping::importSystemMapping(
      generatedMappings->candidates.front(), store));
  require(migratedMapping.view().fabricIdentity() ==
              checkpointChildRoot.reference().artifact,
          "migrated SystemMapping did not bind the child System");

  // This witness is deliberately explicit: it compares a resource-time
  // expansion against an idle alternative, then records a second migration
  // cost where the same expansion loses. It is schedule evidence only; the
  // endpoint classifier still requires a verified SystemMapping schedule.
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef> witnessResources;
  for (std::uint64_t ordinal = 0; ordinal != 5; ++ordinal)
    witnessResources.push_back(
        take(loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
            loom::fabric::FabricInventoryOwnerRef::of(
                loom::fabric::HostCoreOccurrenceRef(ordinal)))));
  const std::vector<dataflow::RootThreadLaunchRef> witnessRegions = {
      dataflow::RootThreadLaunchRef{pressureDataflow.identity(),
                                    dataflow::RootThreadLaunchId(0)},
      dataflow::RootThreadLaunchRef{pressureDataflow.identity(),
                                    dataflow::RootThreadLaunchId(1)},
      dataflow::RootThreadLaunchRef{pressureDataflow.identity(),
                                    dataflow::RootThreadLaunchId(2)},
      dataflow::RootThreadLaunchRef{pressureDataflow.identity(),
                                    dataflow::RootThreadLaunchId(3)},
      dataflow::RootThreadLaunchRef{pressureDataflow.identity(),
                                    dataflow::RootThreadLaunchId(4)}};
  const auto allocate = [&](dataflow::RootThreadLaunchRef region,
                            std::initializer_list<std::size_t> owners) {
    std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef> resources;
    for (std::size_t owner : owners)
      resources.push_back(witnessResources[owner]);
    return loom::pnr::ResourceTimeRegionAllocation{region,
                                                   std::move(resources)};
  };
  const auto makeScenario = [&](std::uint64_t migrationCost, bool expand,
                                bool fifo, std::uint64_t makespan) {
    const auto &r1 = witnessRegions[0];
    const auto &r2 = witnessRegions[1];
    const auto &r3 = witnessRegions[2];
    const auto &r4 = witnessRegions[3];
    const auto &r5 = witnessRegions[4];
    const std::uint64_t r4Completion = expand ? 25 : 40;
    const std::uint64_t r5Start = fifo ? 20 : (expand ? 35 : 40);
    const std::uint64_t r5Completion = fifo ? 30 : (expand ? 45 : 50);
    loom::pnr::ResourceTimeScheduleScenario scenario;
    scenario.executions = {
        {r1, {}, 0, 0, 10},
        {r2, {}, 0, 0, 20},
        {r3, {}, 0, 0, 15},
        {r4,
         {{r1, loom::pnr::ResourceTimeReadinessKind::Completion}},
         10,
         10,
         r4Completion},
        {r5,
         fifo
             ? std::vector<
                   loom::pnr::
                       ResourceTimeRegionPrerequisite>{{r2,
                                                        loom::pnr::
                                                            ResourceTimeReadinessKind::
                                                                FifoToken}}
             : std::vector<
                   loom::pnr::
                       ResourceTimeRegionPrerequisite>{{r2,
                                                        loom::pnr::
                                                            ResourceTimeReadinessKind::
                                                                Completion},
                                                       {r4,
                                                        loom::pnr::
                                                            ResourceTimeReadinessKind::
                                                                Completion}},
         fifo ? 20 : r4Completion, r5Start, r5Completion}};
    std::vector<std::uint64_t> times;
    for (const auto &execution : scenario.executions) {
      times.push_back(execution.startPicoseconds);
      times.push_back(execution.completionPicoseconds);
    }
    llvm::sort(times);
    times.erase(std::unique(times.begin(), times.end()), times.end());
    std::vector<std::size_t> activeRegions;
    bool migrated = false;
    const auto appendState = [&](std::size_t boundaryRegion, bool completion,
                                 std::uint64_t time) {
      std::vector<loom::pnr::ResourceTimeRegionAllocation> active;
      for (std::size_t region : activeRegions) {
        const auto &execution = scenario.executions[region];
        if (region == 0)
          active.push_back(allocate(execution.region, {0}));
        else if (region == 1)
          active.push_back(allocate(execution.region, {1}));
        else if (region == 2)
          active.push_back(allocate(execution.region, {2}));
        else if (region == 3)
          active.push_back(expand ? allocate(execution.region, {3, 4})
                                  : allocate(execution.region, {3}));
        else
          active.push_back(fifo ? allocate(execution.region, {0})
                                : allocate(execution.region, {4}));
      }
      const auto region = scenario.executions[boundaryRegion].region;
      const dataflow::EventFamilyKey event =
          completion ? dataflow::rootThreadCompletionEventFamily(region)
                     : dataflow::rootThreadStartEventFamily(region);
      scenario.states.push_back({migrated ? migratedMapping.reference()
                                          : finalizedParentMapping.reference(),
                                 event, time, std::move(active)});
    };
    for (std::uint64_t time : times) {
      for (std::size_t region = 0; region != scenario.executions.size();
           ++region) {
        const auto &execution = scenario.executions[region];
        if (execution.completionPicoseconds != time)
          continue;
        auto active = llvm::find(activeRegions, region);
        require(active != activeRegions.end(),
                "resource-time fixture completes an inactive region");
        activeRegions.erase(active);
        if (expand && region == 0)
          migrated = true;
        appendState(region, true, time);
      }
      for (std::size_t region = 0; region != scenario.executions.size();
           ++region) {
        const auto &execution = scenario.executions[region];
        if (execution.startPicoseconds != time)
          continue;
        activeRegions.push_back(region);
        appendState(region, false, time);
      }
    }
    if (expand) {
      auto transition = structuralTransition;
      transition.trigger = dataflow::rootThreadCompletionEventFamily(r1);
      transition.safePoint = loom::pnr::ResourceTimeSafePointReference{
          pressureDataflowReference,
          loom::pnr::ResourceTimeSafePointKind::Completion};
      transition.child.mapping = migratedMapping.reference();
      transition.beforeActive = {allocate(r1, {0}), allocate(r2, {1}),
                                 allocate(r3, {2})};
      transition.afterActive = {allocate(r2, {1}), allocate(r3, {2})};
      transition.reprogrammingTimePicoseconds = migrationCost;
      transition.migrationTimePicoseconds = migrationCost;
      scenario.transitions.transitions.push_back(std::move(transition));
    }
    scenario.makespanPicoseconds = makespan;
    return scenario;
  };
  loom::pnr::ResourceTimeScheduleWitness witness{
      witnessRegions,
      {makeScenario(2, true, false, 47), makeScenario(20, true, false, 65),
       makeScenario(0, false, false, 52), makeScenario(2, true, true, 32)},
      1,
      3,
      loom::pnr::ResourceTimeConcurrencyBoundStatus::Exact};
  if (llvm::Error error =
          loom::pnr::validateResourceTimeScheduleWitness(witness))
    fail(llvm::toString(std::move(error)));
  auto malformedWitness = witness;
  malformedWitness.scenarios[0].states[2].active.push_back(
      allocate(witnessRegions[4], {0}));
  llvm::Error malformedError =
      loom::pnr::validateResourceTimeScheduleWitness(malformedWitness);
  require(static_cast<bool>(malformedError),
          "resource-time witness accepted an interval-inconsistent active set");
  llvm::consumeError(std::move(malformedError));
  auto mismatchedTransitionEvidence = witness;
  mismatchedTransitionEvidence.scenarios[0]
      .transitions.transitions.front()
      .beforeActive = {allocate(witnessRegions[1], {1}),
                       allocate(witnessRegions[2], {2})};
  llvm::Error transitionEvidenceError =
      loom::pnr::validateResourceTimeScheduleWitness(
          mismatchedTransitionEvidence);
  require(static_cast<bool>(transitionEvidenceError),
          "resource-time witness accepted transition allocations that differ "
          "from adjacent states");
  llvm::consumeError(std::move(transitionEvidenceError));
  auto staleStateEvent = witness;
  staleStateEvent.scenarios[0].states[2].event = transitionTrigger();
  llvm::Error staleStateEventError =
      loom::pnr::validateResourceTimeScheduleWitness(staleStateEvent);
  require(static_cast<bool>(staleStateEventError),
          "resource-time witness accepted a stale event label");
  llvm::consumeError(std::move(staleStateEventError));
  require(witness.scenarios[0].makespanPicoseconds <
                  witness.scenarios[2].makespanPicoseconds &&
              witness.scenarios[1].makespanPicoseconds >
                  witness.scenarios[2].makespanPicoseconds,
          "resource-time witness did not expose both migration-cost outcomes");
  require(witness.scenarios[2].executions.back().startPicoseconds == 40 &&
              witness.scenarios[3].executions.back().startPicoseconds == 20,
          "FIFO readiness did not expose the early consumer distinction");
  require(llvm::none_of(witness.scenarios[0].states[2].active,
                        [&](const auto &allocation) {
                          return allocation.region == witnessRegions[4];
                        }),
          "resource-time witness incorrectly admitted R5 before readiness");

  require(problem->threadDecisions().size() == 2 &&
              problem->graphDecisions().size() == 4,
          "frozen System problem merged execution atoms");
  require(problem->accCores().size() == 5 &&
              problem->spatialMappings().size() == 2 &&
              problem->targetClasses().size() == 2,
          "frozen System target catalogs are incomplete");
  require(problem->spatialMappingWorstRouteArrivalDelayQuanta().size() ==
                  problem->spatialMappings().size() &&
              problem->spatialMappingTotalRouteNegativeSlackQuanta().size() ==
                  problem->spatialMappings().size() &&
              problem->spatialMappingPhysicalTimingProfileDigests().size() ==
                  problem->spatialMappings().size() &&
              problem->spatialMappingPhysicalTimingProfileKinds().size() ==
                  problem->spatialMappings().size(),
          "frozen System physical timing catalog is incomplete");
  for (loom::pnr::PnrIndex decision = 0;
       decision < problem->graphDecisions().size(); ++decision)
    require(
        problem->graphChoiceSharedOperandIngressPressures(decision).size() ==
            problem->graphChoiceCatalogOrdinals(decision).size(),
        "frozen System graph operand pressure is incomplete");
  for (const auto &[mappingOrdinal, reference] :
       llvm::enumerate(problem->spatialMappings())) {
    auto mapping = take(loom::mapping::importSpatialMapping(reference, store));
    const auto profile =
        llvm::find_if(physicalTimingProfiles, [&](const auto &candidate) {
          return candidate.fabricIdentity() == mapping.view().fabricIdentity();
        });
    require(profile != physicalTimingProfiles.end(),
            "persistent SpatialMapping has no matching timing profile");
    const auto cold =
        take(loom::pnr::detail::projectSpatialMappingPhysicalTiming(
            mapping.view(), *profile));
    require(cold.worstArrivalDelayQuanta ==
                    problem->spatialMappingWorstRouteArrivalDelayQuanta()
                        [mappingOrdinal] &&
                cold.totalNegativeSlackQuanta ==
                    problem->spatialMappingTotalRouteNegativeSlackQuanta()
                        [mappingOrdinal] &&
                profile->digest().bytes() ==
                    problem->spatialMappingPhysicalTimingProfileDigests()
                        [mappingOrdinal] &&
                profile->kind() ==
                    problem->spatialMappingPhysicalTimingProfileKinds()
                        [mappingOrdinal],
            "frozen System timing diverged from cold persistent replay");
  }
  require(!problem->serviceLegs().empty(),
          "frozen System problem lost its service legs");

  for (const auto &[terminalOrdinal, terminal] :
       llvm::enumerate(problem->serviceTerminals())) {
    const auto &leg =
        std::holds_alternative<loom::mapping::SystemTransferSourceTerminalKey>(
            terminal.key)
            ? std::get<loom::mapping::SystemTransferSourceTerminalKey>(
                  terminal.key)
                  .leg
            : std::get<loom::mapping::SystemTransferSinkTerminalKey>(
                  terminal.key)
                  .leg;
    if (!std::holds_alternative<loom::mapping::TransferObligationFamilyKey>(
            leg.obligation))
      continue;
    const auto domains = problem->serviceTerminalOwnerDomains(
        static_cast<loom::pnr::PnrIndex>(terminalOrdinal));
    if (terminal.fixedHostOwner) {
      require(domains.size() == 1 &&
                  std::holds_alternative<loom::fabric::HostCoreOccurrenceRef>(
                      domains.front().owner),
              "host message terminal lost its exact owner domain");
      continue;
    }
    require(terminal.ownerThreadDecision < problem->threadDecisions().size(),
            "message terminal has no valid owner decision");
    const auto ownerChoices =
        problem->threadChoiceCatalogOrdinals(terminal.ownerThreadDecision);
    require(domains.size() == ownerChoices.size(),
            "message terminal owner domains diverged from its thread domain");
    for (loom::pnr::PnrIndex coreOrdinal : ownerChoices) {
      require(coreOrdinal < problem->accCores().size(),
              "message terminal owner choice is outside the core catalog");
      const auto core = problem->accCores()[coreOrdinal];
      const auto domain = llvm::find_if(domains, [&](const auto &candidate) {
        const auto *owner =
            std::get_if<loom::fabric::AccCoreOccurrenceRef>(&candidate.owner);
        return owner && *owner == core;
      });
      require(domain != domains.end(),
              "message terminal omitted a legal execution owner");
    }
  }

  std::optional<loom::mapping::SystemTransferTerminalKey>
      restrictedMessageTerminal;
  for (const auto &service : searchDomain.serviceObligations()) {
    if (!std::holds_alternative<loom::mapping::TransferObligationFamilyKey>(
            service.key))
      continue;
    for (const auto &row : service.transferTerminalCompatibility) {
      const auto *bound = std::get_if<loom::pnr::SystemMessageTerminalEndpoint>(
          &row.boundEndpoint);
      if (!bound)
        continue;
      const auto *endpoint =
          std::get_if<loom::fabric::SystemServiceEndpointRef>(
              &bound->endpoint.owner.payload);
      if (!endpoint)
        continue;
      const auto *owner = system.serviceEndpointOwner(*endpoint);
      if (!owner || !std::holds_alternative<loom::fabric::AccCoreOccurrenceRef>(
                        owner->owner().payload))
        continue;
      restrictedMessageTerminal = row.terminal;
      break;
    }
    if (restrictedMessageTerminal)
      break;
  }
  require(restrictedMessageTerminal.has_value(),
          "message fixture has no AccCore-owned terminal row");
  auto restrictedModule = buildSystemConstraintModule(
      context, dataflow.identity(), system.artifact().identity(), roots);
  auto restrictedRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      restrictedModule->getBody()->front());
  mlir::OpBuilder restrictedBuilder(&context);
  addSystemRestriction(
      restrictedBuilder, restrictedRoot,
      ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
      transferTerminalAttr(&context, dataflow.identity(),
                           *restrictedMessageTerminal),
      {});
  auto restrictedConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          restrictedRoot, dataflow, system, store));
  auto restrictedPartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          dataflow, restrictedConstraints.view().rootThreadLaunches()));
  auto restrictedDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, config, restrictedConstraints, restrictedPartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{spatialMappings}, store));
  requireProvenInfeasibleFreeze(loom::pnr::freezeSystemPnrProblem(
                                    dataflow, system, physicalTimingProfiles,
                                    restrictedDomain, config,
                                    restrictedConstraints, store),
                                "payloads constraining AccCore choices: none");

  auto first = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  auto second = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  require(first.state->threadChoices() == second.state->threadChoices() &&
              first.state->graphChoices() == second.state->graphChoices() &&
              first.assignmentAttempts == second.assignmentAttempts,
          "canonical System initializer is not deterministic");
  require(first.state->serviceRoutes().size() == problem->serviceLegs().size(),
          "canonical System initializer did not route every service leg");
  for (const loom::pnr::SystemServiceRouteSelection &route :
       first.state->serviceRoutes()) {
    require(route.nodeCount != 0 && route.sinkCount != 0,
            "canonical System route is empty");
    require(route.rootEndpoint != loom::pnr::getInvalidPnrIndex(),
            "canonical System route has no root endpoint");
  }
  if (llvm::Error error = first.state->verify())
    fail(llvm::toString(std::move(error)));

  loom::pnr::test::verifySystemFixedTerminalCutAndAnnealing(problem,
                                                            first.state);

  std::vector<loom::pnr::SystemServiceRouteSelection> incompleteRoutes(
      first.state->serviceRoutes().begin(), first.state->serviceRoutes().end());
  incompleteRoutes.front().sinkCount = 0;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {first.state->threadChoices(), first.state->graphChoices(),
           incompleteRoutes, first.state->serviceRouteNodes(),
           first.state->serviceRouteSinks(), first.state->serviceTargets(),
           first.state->instructionResourceUses(),
           first.state->serviceResourceUses()}),
      "service route does not cover the applicable sink-owner set");

  std::vector<loom::pnr::SystemServiceRouteSinkSelection> foreignSinks(
      first.state->serviceRouteSinks().begin(),
      first.state->serviceRouteSinks().end());
  foreignSinks.front().terminal =
      problem->serviceLegs()[first.state->serviceRoutes().front().leg]
          .sourceTerminal;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {first.state->threadChoices(), first.state->graphChoices(),
           first.state->serviceRoutes(), first.state->serviceRouteNodes(),
           foreignSinks, first.state->serviceTargets(),
           first.state->instructionResourceUses(),
           first.state->serviceResourceUses()}),
      "service route sink is outside its exact H domain");

  auto withCanonicalRoutes =
      [&](llvm::ArrayRef<loom::pnr::PnrIndex> threadChoices,
          llvm::ArrayRef<loom::pnr::PnrIndex> graphChoices) {
        return loom::pnr::SystemCandidateInitialization{
            threadChoices,
            graphChoices,
            first.state->serviceRoutes(),
            first.state->serviceRouteNodes(),
            first.state->serviceRouteSinks(),
            first.state->serviceTargets(),
            {},
            {}};
      };

  auto firstDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*first.state, context));
  auto secondDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*first.state, context));
  auto firstRoot = mlir::cast<::mapping::SystemOp>(firstDraft.get());
  verifyFinalizedSystemMappingWorkflow(*first.state, dataflow, system,
                                       constraints.view(), store, context,
                                       problem->serviceDomains().size());
  std::size_t materializedRouteCount = 0;
  for (auto service :
       firstRoot.getBody().front().getOps<::mapping::ServiceRealizationOp>()) {
    auto selections =
        service.getBody().front().getOps<::mapping::ServicePlanSelectionOp>();
    require(selections.begin() != selections.end(),
            "materialized service has no contextual plan selection");
    for (auto selection : selections)
      take(loom::mapping::decodeServicePlanSelectionKey(
          unsignedBytes(selection.getKey().getRecord()),
          problem->dataflowIdentity()));
    require(llvm::hasSingleElement(
                service.getBody().front().getOps<::mapping::ServicePlanOp>()),
            "materialized service has more than one selected plan");
    auto plan =
        *service.getBody().front().getOps<::mapping::ServicePlanOp>().begin();
    require(plan.getPlanOrdinal() == 0,
            "materialized selected service plan has a nonzero ordinal");
    for (auto route :
         plan.getBody().front().getOps<::mapping::TransferLegRealizationOp>()) {
      const auto leg = take(loom::mapping::decodeCanonicalServiceLegKey(
          unsignedBytes(route.getLeg().getRecord()),
          problem->dataflowIdentity()));
      loom::pnr::PnrIndex selectedOrdinal = loom::pnr::getInvalidPnrIndex();
      for (const auto &[ordinal, selected] :
           llvm::enumerate(first.state->serviceRoutes()))
        if (problem->serviceLegs()[selected.leg].key == leg) {
          selectedOrdinal = static_cast<loom::pnr::PnrIndex>(ordinal);
          break;
        }
      require(selectedOrdinal != loom::pnr::getInvalidPnrIndex(),
              "materialized route has no selected Candidate route");
      const auto &selected = first.state->serviceRoutes()[selectedOrdinal];
      const auto expectedRoot = loom::fabric::canonicalFabricBytes(
          problem->routingTopology()
              .endpoints()[selected.rootEndpoint]
              .reference);
      require(unsignedBytes(route.getRootEndpoint().getRecord()) ==
                  std::vector<std::uint8_t>(expectedRoot.begin(),
                                            expectedRoot.end()),
              "materialized route changed its selected root endpoint");

      auto selectedNodes = first.state->serviceRouteNodes().slice(
          selected.nodeOffset, selected.nodeCount);
      auto materializedNodes =
          route.getBody().front().getOps<::mapping::SystemRouteNodeOp>();
      require(
          std::distance(materializedNodes.begin(), materializedNodes.end()) +
                  1 ==
              selectedNodes.size(),
          "materialized route changed its node count");
      for (const auto &[nodeOrdinal, node] :
           llvm::enumerate(materializedNodes)) {
        const auto &expected = selectedNodes[nodeOrdinal + 1];
        const auto expectedTraversal = loom::fabric::canonicalFabricBytes(
            problem->routingTopology()
                .traversals()[expected.incomingTraversal]
                .reference);
        require(node.getNodeOrdinal() == nodeOrdinal + 1 &&
                    node.getParentNodeOrdinal() == expected.parentNode &&
                    unsignedBytes(node.getIncomingTraversal().getRecord()) ==
                        std::vector<std::uint8_t>(expectedTraversal.begin(),
                                                  expectedTraversal.end()),
                "materialized route changed a selected traversal");
      }

      auto selectedSinks = first.state->serviceRouteSinks().slice(
          selected.sinkOffset, selected.sinkCount);
      auto materializedSinks =
          route.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
      require(std::distance(materializedSinks.begin(),
                            materializedSinks.end()) == selectedSinks.size(),
              "materialized route changed its sink count");
      for (const auto &[sinkOrdinal, sink] :
           llvm::enumerate(materializedSinks)) {
        const auto &expected = selectedSinks[sinkOrdinal];
        const auto expectedTerminal =
            take(loom::mapping::encodeSystemTransferTerminalKey(
                problem->dataflowIdentity(),
                problem->serviceTerminals()[expected.terminal].key));
        require(unsignedBytes(sink.getTerminal().getRecord()) ==
                        expectedTerminal &&
                    sink.getNodeOrdinal() == expected.node,
                "materialized route changed a selected sink attachment");
      }
      ++materializedRouteCount;
    }
  }
  require(materializedRouteCount == first.state->serviceRoutes().size(),
          "materializer omitted a selected service route");
  auto firstBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(firstRoot));
  auto secondBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(secondDraft.get())));
  require(firstBytes.bytes() == secondBytes.bytes(),
          "System execution materialization is not deterministic");

  mlir::OwningOpRef<mlir::Operation *> reordered(firstDraft->clone());
  auto reorderedRoot = mlir::cast<::mapping::SystemOp>(reordered.get());
  llvm::SmallVector<mlir::Attribute> reversedRoots(
      reorderedRoot.getRootThreadLaunches().begin(),
      reorderedRoot.getRootThreadLaunches().end());
  std::reverse(reversedRoots.begin(), reversedRoots.end());
  reorderedRoot.setRootThreadLaunchesAttr(
      mlir::ArrayAttr::get(&context, reversedRoots));
  auto reorderedBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(reorderedRoot));
  require(reorderedBytes.bytes() == firstBytes.bytes(),
          "System root authoring order changed canonical bytes");
  auto rawReorderedBytes = rawSystemBytes(reorderedRoot);
  require(rawReorderedBytes.bytes() != firstBytes.bytes(),
          "noncanonical System fixture accidentally matched canonical bytes");
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             rawReorderedBytes, dataflow, system, store),
                         "payload is not canonical");

  mlir::OwningOpRef<mlir::Operation *> missingThread(firstDraft->clone());
  auto missingRoot = mlir::cast<::mapping::SystemOp>(missingThread.get());
  auto missingBinding = *missingRoot.getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  missingBinding.erase();
  requireVerificationFailureContains(missingRoot,
                                     "exactly one ThreadExecutionBinding");

  mlir::OwningOpRef<mlir::Operation *> defaultOnly(firstDraft->clone());
  auto defaultRoot = mlir::cast<::mapping::SystemOp>(defaultOnly.get());
  auto defaultBinding = *defaultRoot.getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  auto defaultClause = *defaultBinding.getBody()
                            .front()
                            .getOps<::mapping::ThreadPresburgerClauseOp>()
                            .begin();
  defaultBinding->setAttr("default_target", defaultClause.getTarget());
  defaultClause.erase();
  auto defaultBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(defaultRoot));
  auto defaultExecution =
      take(loom::mapping::strictImportSystemExecutionBindings(
          defaultBytes, dataflow, system, store));
  require(defaultExecution.threadBindings().front().clauses.empty() &&
              defaultExecution.threadBindings().front().defaultTarget,
          "default-only whole-domain relation did not round trip");

  mlir::OwningOpRef<mlir::Operation *> graphDefaultOnly(firstDraft->clone());
  auto graphDefaultRoot =
      mlir::cast<::mapping::SystemOp>(graphDefaultOnly.get());
  auto graphDefaultBinding = *graphDefaultRoot.getBody()
                                  .front()
                                  .getOps<::mapping::GraphExecutionBindingOp>()
                                  .begin();
  auto graphDefaultClause = *graphDefaultBinding.getBody()
                                 .front()
                                 .getOps<::mapping::GraphPresburgerClauseOp>()
                                 .begin();
  graphDefaultBinding->setAttr("default_target",
                               graphDefaultClause.getTarget());
  graphDefaultClause.erase();
  auto graphDefaultBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(graphDefaultRoot));
  auto graphDefaultExecution =
      take(loom::mapping::strictImportSystemExecutionBindings(
          graphDefaultBytes, dataflow, system, store));
  require(graphDefaultExecution.graphBindings().front().clauses.empty() &&
              graphDefaultExecution.graphBindings().front().defaultTarget,
          "default-only graph relation did not round trip");

  mlir::OwningOpRef<mlir::Operation *> emptyThread(firstDraft->clone());
  auto emptyThreadBinding = *mlir::cast<::mapping::SystemOp>(emptyThread.get())
                                 .getBody()
                                 .front()
                                 .getOps<::mapping::ThreadExecutionBindingOp>()
                                 .begin();
  emptyThreadBinding.getBody().front().front().erase();
  requireVerificationFailureContains(emptyThreadBinding,
                                     "requires a clause or default target");

  mlir::OwningOpRef<mlir::Operation *> emptyGraph(firstDraft->clone());
  auto emptyGraphBinding = *mlir::cast<::mapping::SystemOp>(emptyGraph.get())
                                .getBody()
                                .front()
                                .getOps<::mapping::GraphExecutionBindingOp>()
                                .begin();
  emptyGraphBinding.getBody().front().front().erase();
  requireVerificationFailureContains(emptyGraphBinding,
                                     "requires a clause or default target");

  mlir::OwningOpRef<mlir::Operation *> domainGap(firstDraft->clone());
  auto gapBinding = *mlir::cast<::mapping::SystemOp>(domainGap.get())
                         .getBody()
                         .front()
                         .getOps<::mapping::ThreadExecutionBindingOp>()
                         .begin();
  auto gapClause = *gapBinding.getBody()
                        .front()
                        .getOps<::mapping::ThreadPresburgerClauseOp>()
                        .begin();
  auto wholeCell =
      mlir::cast<::mapping::SystemPresburgerCellAttr>(gapClause.getCells()[0]);
  auto partialCell = withFirstCoordinateLowerBound(wholeCell, 1);
  gapClause->setAttr("cells", mlir::ArrayAttr::get(&context, {partialCell}));
  auto gapBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(domainGap.get())));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             gapBytes, dataflow, system, store),
                         "does not cover its Dataflow may-domain");

  mlir::OwningOpRef<mlir::Operation *> domainOverlap(firstDraft->clone());
  auto overlapBinding = *mlir::cast<::mapping::SystemOp>(domainOverlap.get())
                             .getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  auto overlapClause = *overlapBinding.getBody()
                            .front()
                            .getOps<::mapping::ThreadPresburgerClauseOp>()
                            .begin();
  llvm::SmallVector<mlir::Attribute> overlappingCells = {wholeCell,
                                                         partialCell};
  overlapClause->setAttr("cells",
                         mlir::ArrayAttr::get(&context, overlappingCells));
  auto overlapBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(domainOverlap.get())));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             overlapBytes, dataflow, system, store),
                         "overlapping Presburger cells");

  mlir::OwningOpRef<mlir::Operation *> redundantDefault(firstDraft->clone());
  auto redundantBinding =
      *mlir::cast<::mapping::SystemOp>(redundantDefault.get())
           .getBody()
           .front()
           .getOps<::mapping::ThreadExecutionBindingOp>()
           .begin();
  auto redundantClause = *redundantBinding.getBody()
                              .front()
                              .getOps<::mapping::ThreadPresburgerClauseOp>()
                              .begin();
  redundantBinding->setAttr("default_target", redundantClause.getTarget());
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             rawSystemBytes(mlir::cast<::mapping::SystemOp>(
                                 redundantDefault.get())),
                             dataflow, system, store),
                         "default is forbidden for an empty complement");

  const auto selectedMapping = first.state->selectedSpatialMapping(0);
  const auto unselectedMapping = spatialMappings.front() == selectedMapping
                                     ? spatialMappings.back()
                                     : spatialMappings.front();
  mlir::OwningOpRef<mlir::Operation *> extraImport(firstDraft->clone());
  auto extraImportRoot = mlir::cast<::mapping::SystemOp>(extraImport.get());
  llvm::SmallVector<mlir::Attribute> imports(
      extraImportRoot.getSpatialMappingImports().begin(),
      extraImportRoot.getSpatialMappingImports().end());
  imports.push_back(rootReferenceAttr(&context, unselectedMapping));
  extraImportRoot.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(&context, imports));
  auto extraImportBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(extraImportRoot));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             extraImportBytes, dataflow, system, store),
                         "not the exact selected B_graph range");

  mlir::OwningOpRef<mlir::Operation *> incompatible(firstDraft->clone());
  auto incompatibleRoot = mlir::cast<::mapping::SystemOp>(incompatible.get());
  llvm::SmallVector<mlir::Attribute> incompatibleImports(
      incompatibleRoot.getSpatialMappingImports().begin(),
      incompatibleRoot.getSpatialMappingImports().end());
  incompatibleImports.push_back(rootReferenceAttr(&context, unselectedMapping));
  incompatibleRoot.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(&context, incompatibleImports));
  auto incompatibleBinding = *incompatibleRoot.getBody()
                                  .front()
                                  .getOps<::mapping::GraphExecutionBindingOp>()
                                  .begin();
  auto incompatibleClause = *incompatibleBinding.getBody()
                                 .front()
                                 .getOps<::mapping::GraphPresburgerClauseOp>()
                                 .begin();
  incompatibleClause->setAttr(
      "target", ::mapping::SpatialMappingImportRefAttr::get(&context, 1));
  auto incompatibleBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(incompatibleRoot));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             incompatibleBytes, dataflow, system, store),
                         "graph and thread targets are incompatible");

  auto execution = take(loom::mapping::strictImportSystemExecutionBindings(
      firstBytes, dataflow, system, store));
  require(execution.rootThreadLaunches().size() == 2 &&
              execution.threadBindings().size() == 2 &&
              execution.graphBindings().size() == 4,
          "strict execution import lost factorized binding keys");
  require(execution.spatialMappingImports().size() == 1,
          "System import table is not the exact selected B_graph range");
  for (const auto &binding : execution.threadBindings())
    require(binding.clauses.size() == 1 && !binding.defaultTarget,
            "whole-domain thread binding was not canonicalized");
  for (const auto &binding : execution.graphBindings())
    require(binding.clauses.size() == 1 && !binding.defaultTarget &&
                binding.clauses.front().target ==
                    execution.spatialMappingImports().front(),
            "whole-domain graph binding did not resolve its exact import");
  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->graphDecisions().size(); ++decision) {
    const auto graphDomain = problem->graphChoiceCatalogOrdinals(decision);
    const auto selectedMapping =
        graphDomain[first.state->graphChoice(decision)];
    const auto threadDomain = problem->threadChoiceCatalogOrdinals(
        problem->graphDecisions()[decision].launch.rootThreadLaunch ==
                problem->threadDecisions().front().root
            ? 0
            : 1);
    const auto selectedCore = threadDomain[first.state->threadChoice(
        problem->graphDecisions()[decision].launch.rootThreadLaunch ==
                problem->threadDecisions().front().root
            ? 0
            : 1)];
    require(problem->spatialMappingTargetClass(selectedMapping) ==
                problem->accCoreTargetClass(selectedCore),
            "canonical initializer selected incompatible execution targets");
  }

  std::vector<loom::pnr::PnrIndex> threadChoices(
      problem->threadDecisions().size(), 0);
  std::vector<loom::pnr::PnrIndex> graphChoices(
      problem->graphDecisions().size(), 0);
  require(problem->threadChoiceCatalogOrdinals(0).size() > 1,
          "fixture needs two compatible AccCore choices");
  const auto initialThreadDomain = problem->threadChoiceCatalogOrdinals(0);
  loom::pnr::PnrIndex sameClassFirst = 0;
  loom::pnr::PnrIndex sameClassSecond = 0;
  loom::pnr::PnrIndex sharedClass = 0;
  bool foundSameClassAlternative = false;
  for (loom::pnr::PnrIndex firstChoice = 0;
       firstChoice != initialThreadDomain.size() && !foundSameClassAlternative;
       ++firstChoice)
    for (loom::pnr::PnrIndex secondChoice = firstChoice + 1;
         secondChoice != initialThreadDomain.size(); ++secondChoice)
      if (problem->accCoreTargetClass(initialThreadDomain[firstChoice]) ==
          problem->accCoreTargetClass(initialThreadDomain[secondChoice])) {
        sameClassFirst = firstChoice;
        sameClassSecond = secondChoice;
        sharedClass =
            problem->accCoreTargetClass(initialThreadDomain[firstChoice]);
        foundSameClassAlternative = true;
        break;
      }
  require(foundSameClassAlternative,
          "fixture needs two AccCores in one SpatialCore target class");

  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->threadDecisions().size(); ++decision) {
    const auto domain = problem->threadChoiceCatalogOrdinals(decision);
    bool found = false;
    for (loom::pnr::PnrIndex choice = 0; choice != domain.size(); ++choice)
      if (problem->accCoreTargetClass(domain[choice]) == sharedClass) {
        threadChoices[decision] = choice;
        found = true;
        break;
      }
    require(found, "thread domain lost a compatible target class");
  }
  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->graphDecisions().size(); ++decision) {
    const auto domain = problem->graphChoiceCatalogOrdinals(decision);
    bool found = false;
    for (loom::pnr::PnrIndex choice = 0; choice != domain.size(); ++choice)
      if (problem->spatialMappingTargetClass(domain[choice]) == sharedClass) {
        graphChoices[decision] = choice;
        found = true;
        break;
      }
    require(found, "graph domain lost a compatible target class");
  }
  threadChoices[0] = sameClassFirst;
  auto sameClassBase = take(loom::pnr::initializeSystemCandidate(
      problem, threadChoices, graphChoices));
  threadChoices[0] = sameClassSecond;
  auto alternate = take(loom::pnr::initializeSystemCandidate(
      problem, threadChoices, graphChoices));
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem,
          {threadChoices, graphChoices, sameClassBase->serviceRoutes(),
           sameClassBase->serviceRouteNodes(),
           sameClassBase->serviceRouteSinks(), sameClassBase->serviceTargets(),
           alternate->instructionResourceUses(),
           alternate->serviceResourceUses()}),
      "is not admitted by H");
  if (llvm::Error error = alternate->verify())
    fail(llvm::toString(std::move(error)));
  require(alternate->selectedAccCore(0) != sameClassBase->selectedAccCore(0),
          "explicit thread choice did not change the selected AccCore");
  verifyFinalizedSystemMappingWorkflow(*alternate, dataflow, system,
                                       constraints.view(), store, context,
                                       problem->serviceDomains().size());
  auto cancelledDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*alternate, context));
  const auto alwaysStop = [](const void *) { return true; };
  auto cancelledFinalization = loom::mapping::finalizeSystemMapping(
      mlir::cast<::mapping::SystemOp>(cancelledDraft.get()), dataflow, system,
      constraints.view(), store, &problem->spatialMappingImports(),
      loom::ExecutionControlView(nullptr, alwaysStop));
  require(!cancelledFinalization,
          "cancelled System finalization published a Mapping");
  bool sawTypedCancellation = false;
  llvm::handleAllErrors(
      cancelledFinalization.takeError(),
      [&](const loom::mapping::SystemMappingIncompleteError &error) {
        sawTypedCancellation =
            error.reason() ==
            loom::mapping::SystemMappingIncompleteReason::CancelledOrTimeout;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail("cancelled System finalization returned an untyped error: " +
             error.message());
      });
  require(sawTypedCancellation,
          "cancelled System finalization lost its typed outcome");
  const auto spectrumMapping = take(loom::pnr::finalizeSystemMappingCandidate(
      *alternate, dataflow, system, constraints.view(), store, context));
  loom::pnr::test::verifyResourceTimeSpectrumWorkflow(dataflow, spectrumMapping,
                                                      roots, store);

  const auto firstThreadDomain = problem->threadChoiceCatalogOrdinals(0);
  const auto firstGraphDomain = problem->graphChoiceCatalogOrdinals(0);
  bool foundMismatch = false;
  for (loom::pnr::PnrIndex threadChoice = 0;
       threadChoice != firstThreadDomain.size() && !foundMismatch;
       ++threadChoice)
    for (loom::pnr::PnrIndex graphChoice = 0;
         graphChoice != firstGraphDomain.size() && !foundMismatch;
         ++graphChoice)
      if (problem->accCoreTargetClass(firstThreadDomain[threadChoice]) !=
          problem->spatialMappingTargetClass(firstGraphDomain[graphChoice])) {
        threadChoices.assign(problem->threadDecisions().size(), threadChoice);
        graphChoices.assign(problem->graphDecisions().size(), graphChoice);
        requireFailureContains(
            loom::pnr::SystemCandidateState::create(
                problem, withCanonicalRoutes(threadChoices, graphChoices)),
            "target classes are incompatible");
        foundMismatch = true;
      }
  require(foundMismatch,
          "heterogeneous fixture did not expose an incompatible target pair");

  threadChoices.assign(problem->threadDecisions().size(), 0);
  graphChoices.assign(problem->graphDecisions().size(), 0);
  threadChoices[0] = problem->threadChoiceCatalogOrdinals(0).size();
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, withCanonicalRoutes(threadChoices, graphChoices)),
      "thread choice is outside its H domain");
  threadChoices.pop_back();
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, withCanonicalRoutes(threadChoices, graphChoices)),
      "thread choice count does not match H");

  llvm::outs() << "System CandidateState graph-binding anchors passed\n";
}

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one workflow name");
  const llvm::StringRef workflow(argv[1]);
  if (workflow == "memory-service")
    memoryServiceWorkflow();
  else if (workflow == "graph-binding")
    graphBindingWorkflow();
  else
    fail("unknown workflow name");
  return EXIT_SUCCESS;
}
