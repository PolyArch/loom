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
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemMappingMigration.h"
#include "PnR/System/SystemPnrGenerator.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "SystemCandidateFixture.h"
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

void memoryServiceWorkflow() {
  using loom::pnr::test::countOccurrences;
  using loom::pnr::test::rawSystemBytes;
  using loom::pnr::test::verifyFinalizedSystemMappingWorkflow;
  using loom::pnr::test::verifySystemResourceActionWorkflow;
  using loom::pnr::test::verifySystemServiceTargetRejections;
  using loom::pnr::test::withFirstCoordinateLowerBound;
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto baselineDesign = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  require(baselineDesign.roots().size() == 1 &&
              baselineDesign.roots().front().directDependencies().size() == 1,
          "builtin System fixture did not publish one Module dependency");
  auto primaryModule = take(loom::fabric::importEntireFabricRoot(
      baselineDesign.roots().front().directDependencies().front().root, store));

  const loom::ResolvedConfig resolved =
      loom::pnr::test::buildSystemCandidateResolvedConfig();
  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));

  auto memoryDataflowArtifact = buildMemoryDataflow(context);
  take(dataflow::publishCanonicalDataflow(memoryDataflowArtifact, store));
  auto memoryDataflow = take(memoryDataflowArtifact.view());
  auto endpointDesign = loom::pnr::test::buildHeterogeneousSystem(
      store, baselineDesign.roots().front(), primaryModule, primaryModule,
      context, /*extraSupportsRead=*/false,
      /*routeExtraMemoryThroughTransform=*/true);
  auto endpointSystem = take(
      loom::fabric::requireSystemRoot(endpointDesign.roots().front().view()));
  loom::ResolvedConfig memoryResolved = resolved;
  memoryResolved.dse.spatialPnr.search = resolved.dse.spatialPnr.search;
  const auto memoryMapping = generateSpatialMapping(
      memoryDataflow, primaryModule, memoryResolved, store, &context);
  verifySystemResourceActionWorkflow(store, baselineDesign.roots().front(),
                                     primaryModule, memoryDataflow,
                                     memoryMapping, resolved, config, context);
  std::vector<dataflow::RootThreadLaunchRef> memoryRoots{
      memoryDataflow.rootThreadLaunches().front().ref};
  auto memoryConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          memoryDataflow, endpointSystem, memoryRoots, store));
  auto memoryPartition =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          memoryDataflow, memoryConstraints.view().rootThreadLaunches()));
  auto memorySearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      memoryDataflow, endpointSystem, config, memoryConstraints,
      memoryPartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{memoryMapping}}, store));
  const auto memoryService = llvm::find_if(
      memorySearchDomain.serviceObligations(), [](const auto &service) {
        const auto *operation =
            std::get_if<loom::mapping::OperationServiceObligationFamilyKey>(
                &service.key);
        return operation &&
               std::holds_alternative<dataflow::LogicalMemoryRootOrViewRef>(
                   *operation);
      });
  require(memoryService != memorySearchDomain.serviceObligations().end(),
          "endpoint-factorization fixture has no memory obligation");
  std::vector<const loom::pnr::SystemSearchServiceTargetCompatibility *>
      addressedRows;
  for (const auto &row : memoryService->targetCompatibility) {
    const auto *subject =
        std::get_if<loom::pnr::SystemServiceMemberTargetSubject>(&row.subject);
    if (subject &&
        std::holds_alternative<dataflow::AddressedMemoryActorMemberRef>(
            subject->member))
      addressedRows.push_back(&row);
  }
  require(addressedRows.size() == 4,
          "two memory subjects did not produce endpoint-factorized rows");
  const loom::fabric::SystemServiceEndpointRef *supportedEndpoint = nullptr;
  const loom::fabric::SystemServiceEndpointRef *unsupportedEndpoint = nullptr;
  const loom::fabric::SystemServiceEndpointRef *transformedEndpoint = nullptr;
  std::vector<loom::fabric::SystemServiceEndpointRef> targetEndpoints;
  for (const auto *row : addressedRows)
    if (!llvm::is_contained(targetEndpoints, row->boundEndpoint))
      targetEndpoints.push_back(row->boundEndpoint);
  require(targetEndpoints.size() == 2,
          "same Module path did not produce two exact endpoint keys");
  for (const auto &endpoint : targetEndpoints) {
    const auto targetPlans =
        take(loom::fabric::projectFabricMemoryServiceTargetPlans(endpointSystem,
                                                                 endpoint));
    if (llvm::any_of(targetPlans, [](const auto &plan) {
          return llvm::any_of(plan.branches, [](const auto &branch) {
            return !branch.transformPath.empty();
          });
        })) {
      require(!transformedEndpoint,
              "more than one endpoint unexpectedly uses a transform chain");
      transformedEndpoint = &endpoint;
    }
    std::size_t emptyRows = 0;
    std::size_t nonemptyRows = 0;
    for (const auto *row : addressedRows) {
      if (row->boundEndpoint != endpoint)
        continue;
      const auto *regions =
          std::get_if<std::vector<loom::fabric::FabricMemoryServiceRegionRef>>(
              &row->compatibleTargets);
      require(regions, "memory target row has a non-region domain");
      regions->empty() ? ++emptyRows : ++nonemptyRows;
    }
    require(emptyRows + nonemptyRows == 2,
            "one endpoint did not retain both memory subjects");
    if (emptyRows == 0)
      supportedEndpoint = &endpoint;
    else {
      require(emptyRows == 1 && nonemptyRows == 1,
              "adverse endpoint does not distinguish read from write");
      unsupportedEndpoint = &endpoint;
    }
  }
  require(supportedEndpoint && unsupportedEndpoint,
          "endpoint rows unioned or intersected distinct read capabilities");
  require(transformedEndpoint == unsupportedEndpoint,
          "adverse endpoint did not exercise the explicit transform closure");
  const auto transformedTargetPlans =
      take(loom::fabric::projectFabricMemoryServiceTargetPlans(
          endpointSystem, *transformedEndpoint));
  std::vector<loom::fabric::SystemServiceTransformRef> foreignTransformPath;
  std::optional<loom::fabric::FabricMemoryServiceRegionRef> otherEndpointRegion;
  for (const auto &plan : transformedTargetPlans)
    for (const auto &branch : plan.branches)
      if (!branch.transformPath.empty()) {
        foreignTransformPath = branch.transformPath;
        otherEndpointRegion = branch.region;
        break;
      }
  require(!foreignTransformPath.empty() && otherEndpointRegion,
          "adverse endpoint has no concrete transform path");

  const loom::fabric::FabricMemoryEndpointRef *unsupportedOccurrence = nullptr;
  for (const auto &attachment : endpointSystem.spatialAttachments()) {
    if (attachment.serviceEndpoint != *unsupportedEndpoint)
      continue;
    require(!unsupportedOccurrence && attachment.spatialEndpoint.memory(),
            "unsupported endpoint has an ambiguous memory attachment");
    unsupportedOccurrence = attachment.spatialEndpoint.memory();
  }
  require(unsupportedOccurrence,
          "unsupported endpoint has no exact occurrence attachment");
  const loom::fabric::FabricMemoryEndpointRef unsupportedSystemEndpoint{
      loom::fabric::FabricMemoryEndpointOwnerRef::of(*unsupportedEndpoint), 0};
  std::size_t unsupportedEmptyTerminalRows = 0;
  std::size_t unsupportedNonemptyTerminalRows = 0;
  for (const auto &row : memoryService->transferTerminalCompatibility) {
    const auto &bound =
        std::get<loom::pnr::SystemMemoryOrFenceTerminalEndpoint>(
            row.boundEndpoint)
            .endpoint;
    if (bound != unsupportedSystemEndpoint && bound != *unsupportedOccurrence)
      continue;
    row.compatibleTransportEndpoints.empty()
        ? ++unsupportedEmptyTerminalRows
        : ++unsupportedNonemptyTerminalRows;
  }
  require(unsupportedEmptyTerminalRows != 0 &&
              unsupportedNonemptyTerminalRows != 0,
          "memory terminal rows lost per-member endpoint compatibility");

  std::vector<loom::fabric::AccCoreOccurrenceRef> supportedCores;
  for (const auto &attachment : endpointSystem.spatialAttachments()) {
    if (attachment.serviceEndpoint != *supportedEndpoint)
      continue;
    const auto *occurrence = attachment.spatialEndpoint.memory();
    require(occurrence, "supported endpoint has an incomplete attachment");
    const auto *spatialCore =
        std::get_if<loom::fabric::SpatialCoreOccurrenceRef>(
            &occurrence->owner.payload);
    require(spatialCore,
            "supported memory attachment is not occurrence-qualified");
    if (!llvm::is_contained(supportedCores, spatialCore->core))
      supportedCores.push_back(spatialCore->core);
  }
  require(!supportedCores.empty(),
          "supported endpoint has no exact occurrence attachment");
  llvm::sort(supportedCores, [](const auto left, const auto right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });

  auto bindingProblem =
      take(loom::pnr::freezeSystemPnrProblemWithNormalizedTiming(
          memoryDataflow, endpointSystem, memorySearchDomain, config,
          memoryConstraints, store));
  std::vector<std::uint32_t> operationLegWidths;
  for (const auto &leg : bindingProblem->serviceLegs())
    if (std::holds_alternative<
            loom::mapping::OperationServiceObligationFamilyKey>(
            leg.key.obligation))
      operationLegWidths.push_back(leg.requiredPayloadWidthBits);
  llvm::sort(operationLegWidths);
  require(operationLegWidths == std::vector<std::uint32_t>({0, 32, 64, 64}),
          "operation service legs lost their maximum-width envelopes");
  std::vector<loom::pnr::PnrIndex> memoryContextOrdinals;
  for (const auto &[ordinal, serviceContext] :
       llvm::enumerate(bindingProblem->serviceContexts()))
    if (serviceContext.service < bindingProblem->serviceDomains().size() &&
        std::holds_alternative<
            loom::mapping::OperationServiceObligationFamilyKey>(
            bindingProblem->serviceDomains()[serviceContext.service].key))
      memoryContextOrdinals.push_back(
          static_cast<loom::pnr::PnrIndex>(ordinal));
  require(memoryContextOrdinals.size() == 1,
          "one graph-backed memory obligation did not form one context");
  const auto &memoryContext =
      bindingProblem->serviceContexts()[memoryContextOrdinals.front()];
  std::vector<loom::pnr::PnrIndex> memoryThreadChoices(
      bindingProblem->threadDecisions().size(), 0);
  std::vector<loom::pnr::PnrIndex> memoryGraphChoices(
      bindingProblem->graphDecisions().size(), 0);
  const auto memoryThreadDomain =
      bindingProblem->threadChoiceCatalogOrdinals(memoryContext.threadDecision);
  loom::pnr::PnrIndex supportedChoice = loom::pnr::getInvalidPnrIndex();
  loom::pnr::PnrIndex unsupportedChoice = loom::pnr::getInvalidPnrIndex();
  for (loom::pnr::PnrIndex choice = 0; choice != memoryThreadDomain.size();
       ++choice) {
    const auto core = bindingProblem->accCores()[memoryThreadDomain[choice]];
    if (llvm::is_contained(supportedCores, core))
      supportedChoice = choice;
    else
      unsupportedChoice = choice;
  }
  require(supportedChoice != loom::pnr::getInvalidPnrIndex() &&
              unsupportedChoice != loom::pnr::getInvalidPnrIndex(),
          "memory context did not retain both occurrence choices");
  memoryThreadChoices[memoryContext.threadDecision] = supportedChoice;
  auto supportedCandidate = take(loom::pnr::initializeSystemCandidate(
      bindingProblem, memoryThreadChoices, memoryGraphChoices));
  auto selectedTargetDomain = take(
      supportedCandidate->serviceTargetDomain(memoryContextOrdinals.front()));
  const auto *selectedDomains =
      std::get_if<loom::pnr::SystemMemoryServiceTargetDomain>(
          &selectedTargetDomain);
  const auto *selectedTargets =
      std::get_if<loom::pnr::SystemMemoryServiceTargetSelection>(
          &supportedCandidate->serviceTarget(memoryContextOrdinals.front()));
  require(selectedDomains && selectedTargets &&
              selectedDomains->plansBySubject.size() ==
                  memoryContext.subjects.size() &&
              selectedTargets->plansBySubject.size() ==
                  memoryContext.subjects.size(),
          "matching target rows did not retain per-subject domains");
  for (const auto &[selected, domain] : llvm::zip_equal(
           selectedTargets->plansBySubject, selectedDomains->plansBySubject))
    require(!domain.empty() && selected == domain.front() &&
                selected.branches.size() == 1,
            "canonical candidate did not select each first exact target");
  const auto selectedRegion =
      selectedTargets->plansBySubject.front().branches.front().region;

  std::vector<loom::pnr::SystemServiceTargetSelection> foreignTargets(
      supportedCandidate->serviceTargets().begin(),
      supportedCandidate->serviceTargets().end());
  auto foreignSelection = *selectedTargets;
  auto &foreignPlan = foreignSelection.plansBySubject.front();
  auto &foreignRegion = foreignPlan.branches.front().region;
  foreignRegion.ordinal += 1000;
  foreignTargets[memoryContextOrdinals.front()] = std::move(foreignSelection);
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem,
          {supportedCandidate->threadChoices(),
           supportedCandidate->graphChoices(),
           supportedCandidate->serviceRoutes(),
           supportedCandidate->serviceRouteNodes(),
           supportedCandidate->serviceRouteSinks(), foreignTargets,
           supportedCandidate->instructionResourceUses(),
           supportedCandidate->serviceResourceUses()}),
      "selected service target is outside its exact H domain");

  auto memoryDraft = take(
      loom::pnr::materializeSystemCandidateDraft(*supportedCandidate, context));
  auto memoryRoot = mlir::cast<::mapping::SystemOp>(memoryDraft.get());
  const auto selectedRegionBytes =
      loom::fabric::canonicalFabricBytes(selectedRegion);
  std::size_t memoryTargetCount = 0;
  ::mapping::ServiceRealizationOp selectedMemoryService;
  ::mapping::ServicePlanOp selectedMemoryPlan;
  for (auto service :
       memoryRoot.getBody().front().getOps<::mapping::ServiceRealizationOp>())
    for (auto plan :
         service.getBody().front().getOps<::mapping::ServicePlanOp>())
      for (auto target :
           plan.getBody().front().getOps<::mapping::MemoryRegionTargetOp>()) {
        selectedMemoryService = service;
        selectedMemoryPlan = plan;
        ++memoryTargetCount;
        require(unsignedBytes(target.getServiceRegion().getRecord()) ==
                    std::vector<std::uint8_t>(selectedRegionBytes.begin(),
                                              selectedRegionBytes.end()),
                "materialized memory target changed its selected region");
        require(target.getTransformPath().empty(),
                "direct service target gained a transform path");
      }
  std::size_t expectedMemoryTargetCount = 0;
  for (const auto &selected : selectedTargets->plansBySubject)
    expectedMemoryTargetCount += selected.branches.size();
  require(memoryTargetCount == expectedMemoryTargetCount,
          "memory service context did not materialize per-subject targets");

  verifySystemServiceTargetRejections(
      memoryRoot, memoryDataflow, endpointSystem, store, context,
      foreignTransformPath, *otherEndpointRegion);

  require(!supportedCandidate->instructionResourceUses().empty(),
          "candidate omitted InstructionCore occupancy choices");
  std::size_t expectedServiceUseCount = 0;
  for (const auto &[subjectOrdinal, subject] :
       llvm::enumerate(memoryContext.subjects))
    if (std::holds_alternative<loom::pnr::SystemServiceMemberTargetSubject>(
            subject))
      expectedServiceUseCount +=
          selectedTargets->plansBySubject[subjectOrdinal].branches.size();
  require(supportedCandidate->serviceResourceUses().size() ==
              expectedServiceUseCount,
          "addressed members and target branches did not select exact uses");
  std::vector<loom::pnr::SystemServiceResourceUseSelection> foreignUses(
      supportedCandidate->serviceResourceUses().begin(),
      supportedCandidate->serviceResourceUses().end());
  foreignUses.front().pattern.ordinal += 1000;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem,
          {supportedCandidate->threadChoices(),
           supportedCandidate->graphChoices(),
           supportedCandidate->serviceRoutes(),
           supportedCandidate->serviceRouteNodes(),
           supportedCandidate->serviceRouteSinks(),
           supportedCandidate->serviceTargets(),
           supportedCandidate->instructionResourceUses(), foreignUses}),
      "service ResourceUse is foreign or inadmissible");
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem, {supportedCandidate->threadChoices(),
                           supportedCandidate->graphChoices(),
                           supportedCandidate->serviceRoutes(),
                           supportedCandidate->serviceRouteNodes(),
                           supportedCandidate->serviceRouteSinks(),
                           supportedCandidate->serviceTargets(),
                           {},
                           supportedCandidate->serviceResourceUses()}),
      "InstructionCore ResourceUse count is incomplete");
  std::size_t instructionUseCount = 0;
  std::size_t serviceUseCount = 0;
  for (auto use :
       memoryRoot.getBody().front().getOps<::mapping::ResourceUseOp>()) {
    auto activation = mlir::dyn_cast<::mapping::SystemRelativeActivationAttr>(
        use.getActivation());
    require(static_cast<bool>(activation),
            "System ResourceUse lost its typed activation");
    if (mlir::isa<::mapping::InstructionExecutionResourceOwnerRefAttr>(
            use.getOwner())) {
      require(activation.getRelease().size() == 1,
              "InstructionCore occupancy lost root completion release");
      ++instructionUseCount;
      continue;
    }
    auto owner =
        mlir::dyn_cast<::mapping::ServicePlanElementRefAttr>(use.getOwner());
    require(owner && mlir::isa<::mapping::MemoryRegionElementKeyAttr>(
                         owner.getElement()),
            "addressed service use lost its exact MemoryRegion owner");
    require(activation.getRelease().empty(),
            "addressed service use gained a causal release");
    auto event =
        take(dataflow::decodeDataflowReference<dataflow::EventFamilyKey>(
            unsignedBytes(activation.getTrigger().getEvent().getRecord()),
            memoryDataflow.identity()));
    require(std::holds_alternative<dataflow::ContextualActorTransitionEventRef>(
                event) &&
                std::get<dataflow::ContextualActorTransitionEventRef>(event)
                        .transitionCaseOrdinal == 0,
            "addressed service use did not trigger on its issue transition");
    ++serviceUseCount;
  }
  require(instructionUseCount ==
                  supportedCandidate->instructionResourceUses().size() &&
              serviceUseCount ==
                  supportedCandidate->serviceResourceUses().size(),
          "materializer did not preserve the candidate ResourceUse closure");

  const auto canonicalMemoryDraft =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(memoryRoot));
  const llvm::StringRef canonicalMemoryText(
      reinterpret_cast<const char *>(canonicalMemoryDraft.bytes().data()),
      canonicalMemoryDraft.bytes().size());
  const std::size_t baselinePlanCount =
      countOccurrences(canonicalMemoryText, "mapping.service_plan ");
  mlir::OwningOpRef<mlir::Operation *> alternateTargetDraft(
      memoryDraft->clone());
  auto alternateTargetRoot =
      mlir::cast<::mapping::SystemOp>(alternateTargetDraft.get());
  ::mapping::ServiceRealizationOp alternateTargetService;
  ::mapping::ServicePlanOp alternateTargetPlan;
  for (auto service : alternateTargetRoot.getBody()
                          .front()
                          .getOps<::mapping::ServiceRealizationOp>())
    for (auto plan :
         service.getBody().front().getOps<::mapping::ServicePlanOp>())
      if (!plan.getBody()
               .front()
               .getOps<::mapping::MemoryRegionTargetOp>()
               .empty()) {
        alternateTargetService = service;
        alternateTargetPlan = plan;
      }
  require(selectedMemoryService && selectedMemoryPlan &&
              alternateTargetService && alternateTargetPlan,
          "memory target plan lookup failed");
  auto distinctPlan =
      mlir::cast<::mapping::ServicePlanOp>(alternateTargetPlan->clone());
  distinctPlan.setPlanOrdinalAttr(
      mlir::Builder(&context).getI64IntegerAttr(1000));
  auto distinctTarget = *distinctPlan.getBody()
                             .front()
                             .getOps<::mapping::MemoryRegionTargetOp>()
                             .begin();
  distinctTarget.setServiceRegionAttr(
      constraintFabricAttr<::mapping::FabricMemoryServiceRegionRefAttr>(
          &context, foreignRegion));
  alternateTargetService.getBody().front().push_back(distinctPlan);
  auto alternateSelection = *alternateTargetService.getBody()
                                 .front()
                                 .getOps<::mapping::ServicePlanSelectionOp>()
                                 .begin();
  mlir::OpBuilder alternateBuilder(&context);
  alternateBuilder.setInsertionPointToEnd(
      &alternateSelection.getBody().front());
  ::mapping::ServicePlanPresburgerClauseOp::create(
      alternateBuilder, alternateBuilder.getUnknownLoc(),
      alternateBuilder.getArrayAttr({::mapping::SystemPresburgerCellAttr::get(
          &context, 0, 0, 0, alternateBuilder.getArrayAttr({}),
          alternateBuilder.getArrayAttr({}))}),
      1000);
  const auto distinctTargetBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(alternateTargetRoot));
  const llvm::StringRef distinctTargetText(
      reinterpret_cast<const char *>(distinctTargetBytes.bytes().data()),
      distinctTargetBytes.bytes().size());
  require(countOccurrences(distinctTargetText, "mapping.service_plan ") ==
              baselinePlanCount + 1,
          "canonicalization merged plans with different service targets");

  memoryThreadChoices[memoryContext.threadDecision] = unsupportedChoice;
  requireFailureContains(
      loom::pnr::initializeSystemCandidate(bindingProblem, memoryThreadChoices,
                                           memoryGraphChoices),
      "has an empty exact H domain");
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          bindingProblem, {memoryThreadChoices, memoryGraphChoices,
                           supportedCandidate->serviceRoutes(),
                           supportedCandidate->serviceRouteNodes(),
                           supportedCandidate->serviceRouteSinks(),
                           supportedCandidate->serviceTargets(),
                           supportedCandidate->instructionResourceUses(),
                           supportedCandidate->serviceResourceUses()}),
      "has an empty exact H domain");

  const auto belongsToSupportedExecution = [&](const auto &row) {
    const auto &endpoint =
        std::get<loom::pnr::SystemMemoryOrFenceTerminalEndpoint>(
            row.boundEndpoint)
            .endpoint;
    if (const auto *system =
            std::get_if<loom::fabric::SystemServiceEndpointRef>(
                &endpoint.owner.payload))
      return *system == *supportedEndpoint;
    const auto *spatialCore =
        std::get_if<loom::fabric::SpatialCoreOccurrenceRef>(
            &endpoint.owner.payload);
    return spatialCore && llvm::is_contained(supportedCores, spatialCore->core);
  };
  const auto restrictedTerminalRow = llvm::find_if(
      memoryService->transferTerminalCompatibility, [&](const auto &row) {
        return belongsToSupportedExecution(row) &&
               !row.compatibleTransportEndpoints.empty();
      });
  require(restrictedTerminalRow !=
              memoryService->transferTerminalCompatibility.end(),
          "constraint fixture has no supported transfer terminal row");
  const auto restrictedTerminal = restrictedTerminalRow->terminal;
  const loom::mapping::SystemTransferTerminalKey peerTerminal = [&] {
    if (const auto *source =
            std::get_if<loom::mapping::SystemTransferSourceTerminalKey>(
                &restrictedTerminal))
      return loom::mapping::SystemTransferTerminalKey(
          loom::mapping::SystemTransferSinkTerminalKey{source->leg, 0});
    return loom::mapping::SystemTransferTerminalKey(
        loom::mapping::SystemTransferSourceTerminalKey{
            std::get<loom::mapping::SystemTransferSinkTerminalKey>(
                restrictedTerminal)
                .leg});
  }();
  const auto peerTerminalRow = llvm::find_if(
      memoryService->transferTerminalCompatibility, [&](const auto &row) {
        return row.terminal == peerTerminal &&
               belongsToSupportedExecution(row) &&
               !row.compatibleTransportEndpoints.empty();
      });
  require(peerTerminalRow != memoryService->transferTerminalCompatibility.end(),
          "constraint fixture has no supported peer terminal");
  std::size_t expectedConstrainedTerminalRows = 0;
  bool expectedUnrestrictedTerminal = false;
  for (const auto &row : memoryService->transferTerminalCompatibility) {
    if (!belongsToSupportedExecution(row))
      continue;
    if (row.terminal == restrictedTerminal || row.terminal == peerTerminal)
      ++expectedConstrainedTerminalRows;
    else if (!row.compatibleTransportEndpoints.empty())
      expectedUnrestrictedTerminal = true;
  }
  require(expectedConstrainedTerminalRows >= 2 && expectedUnrestrictedTerminal,
          "constraint fixture lacks retained terminal coverage");
  const auto memoryOperation =
      std::get<loom::mapping::OperationServiceObligationFamilyKey>(
          memoryService->key);

  auto constrainedModule = buildSystemConstraintModule(
      context, memoryDataflow.identity(), endpointSystem.artifact().identity(),
      memoryRoots);
  auto constrainedRoot = llvm::cast<::mapping::ConstraintsSystemOp>(
      constrainedModule->getBody()->front());
  mlir::OpBuilder constraintBuilder(&context);
  std::vector<mlir::Attribute> supportedCoreAttributes;
  supportedCoreAttributes.reserve(supportedCores.size());
  for (const auto core : supportedCores)
    supportedCoreAttributes.push_back(
        constraintFabricAttr<::mapping::FabricAccCoreOccurrenceRefAttr>(
            &context, core));
  addSystemRestriction(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
      constraintDataflowAttr<::mapping::RootThreadLaunchRefAttr>(
          &context, memoryDataflow.identity(), memoryRoots.front()),
      supportedCoreAttributes);
  const auto memorySubject = serviceObligationAttr(
      &context, memoryDataflow.identity(),
      loom::mapping::SystemServiceObligationKey{memoryOperation});
  addSystemRestriction(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::ServiceTargetRegion, memorySubject,
      {});
  const auto restrictedTerminalSubject = transferTerminalAttr(
      &context, memoryDataflow.identity(), restrictedTerminal);
  const auto peerTerminalSubject =
      transferTerminalAttr(&context, memoryDataflow.identity(), peerTerminal);
  addSystemEquality(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
      {restrictedTerminalSubject, peerTerminalSubject});
  addSystemRestriction(
      constraintBuilder, constrainedRoot,
      ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
      peerTerminalSubject, {});
  auto constrainedSystemConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          constrainedRoot, memoryDataflow, endpointSystem, store));
  auto constrainedSearchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      memoryDataflow, endpointSystem, config, constrainedSystemConstraints,
      memoryPartition,
      loom::pnr::SystemHierarchicalGraphSearchInput{{memoryMapping}}, store));
  require(constrainedSearchDomain.digest() != memorySearchDomain.digest(),
          "exact System K did not change the H digest");
  for (const auto &binding : constrainedSearchDomain.bindings()) {
    if (!std::holds_alternative<dataflow::RootThreadLaunchRef>(binding.key))
      continue;
    const auto *thread = std::get_if<loom::pnr::SystemThreadBindingDomain>(
        &binding.atoms.front().domain);
    require(thread && thread->compatibleAccCores == supportedCores,
            "thread constraint did not restrict the H atom domain");
  }
  const auto constrainedMemoryService = llvm::find_if(
      constrainedSearchDomain.serviceObligations(),
      [&](const auto &service) { return service.key == memoryService->key; });
  require(constrainedMemoryService !=
              constrainedSearchDomain.serviceObligations().end(),
          "constrained H lost the memory obligation");
  std::size_t constrainedAddressedRows = 0;
  for (const auto &row : constrainedMemoryService->targetCompatibility) {
    const auto *subject =
        std::get_if<loom::pnr::SystemServiceMemberTargetSubject>(&row.subject);
    if (!subject ||
        !std::holds_alternative<dataflow::AddressedMemoryActorMemberRef>(
            subject->member))
      continue;
    ++constrainedAddressedRows;
    const auto *regions =
        std::get_if<std::vector<loom::fabric::FabricMemoryServiceRegionRef>>(
            &row.compatibleTargets);
    require(regions && regions->empty(),
            "service target restriction was not folded into its H row");
  }
  require(constrainedAddressedRows == addressedRows.size() / 2,
          "thread constraint did not restrict service row key coverage");
  std::size_t constrainedTerminalRows = 0;
  bool retainedUnrestrictedTerminal = false;
  for (const auto &row :
       constrainedMemoryService->transferTerminalCompatibility) {
    if (row.terminal == restrictedTerminal || row.terminal == peerTerminal) {
      ++constrainedTerminalRows;
      require(row.compatibleTransportEndpoints.empty(),
              "terminal restriction was not folded into its H row");
    } else if (!row.compatibleTransportEndpoints.empty()) {
      retainedUnrestrictedTerminal = true;
    }
  }
  require(constrainedTerminalRows == expectedConstrainedTerminalRows &&
              retainedUnrestrictedTerminal,
          "terminal restriction removed a row or leaked across subjects");
  auto adoptedConstrained = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      constrainedSearchDomain.canonicalViewBytes(),
      constrainedSearchDomain.digest(), store));
  require(adoptedConstrained.canonicalViewBytes() ==
              constrainedSearchDomain.canonicalViewBytes(),
          "strict H adoption changed constraint-folded rows");

  llvm::outs() << "System CandidateState memory-service anchors passed\n";
}
