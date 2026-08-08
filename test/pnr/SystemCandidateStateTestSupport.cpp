#include "SystemCandidateStateTestSupport.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/System/SystemMappingMaterializer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System CandidateState Fabric fixture failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void verifySelectedRouteCapacity(
    const loom::pnr::SystemCandidateState &candidate) {
  const auto &topology = candidate.problem().routingTopology();
  std::vector<std::uint64_t> usage;
  usage.reserve(topology.capacityCells().size());
  for (const auto &cell : topology.capacityCells())
    usage.push_back(cell.initialOccupancy);

  for (const auto &route : candidate.serviceRoutes()) {
    require(route.nodeOffset <= candidate.serviceRouteNodes().size() &&
                route.nodeCount <=
                    candidate.serviceRouteNodes().size() - route.nodeOffset,
            "selected route node range is invalid");
    std::map<std::pair<loom::pnr::PnrIndex, loom::pnr::PnrIndex>, std::uint64_t>
        selected;
    for (const auto &node : candidate.serviceRouteNodes().slice(
             route.nodeOffset, route.nodeCount)) {
      if (node.incomingTraversal == loom::pnr::getInvalidPnrIndex())
        continue;
      require(node.incomingTraversal < topology.traversals().size(),
              "selected route has an invalid capacity traversal");
      const auto &traversal = topology.traversals()[node.incomingTraversal];
      require(
          traversal.capacityClaimOffset <= topology.capacityClaims().size() &&
              traversal.capacityClaimCount <= topology.capacityClaims().size() -
                                                  traversal.capacityClaimOffset,
          "selected route has an invalid capacity claim range");
      for (const auto &claim : topology.capacityClaims().slice(
               traversal.capacityClaimOffset, traversal.capacityClaimCount)) {
        auto [position, inserted] = selected.try_emplace(
            std::make_pair(claim.activation, claim.cell), claim.amount);
        require(inserted || position->second == claim.amount,
                "selected route has inconsistent atomic claims");
      }
    }
    std::map<loom::pnr::PnrIndex, std::uint64_t> additions;
    for (const auto &[key, amount] : selected)
      additions[key.second] += amount;
    for (const auto &[cell, amount] : additions) {
      require(cell < usage.size(),
              "selected route claim has an invalid capacity cell");
      require(usage[cell] <= topology.capacityCells()[cell].capacity &&
                  amount <=
                      topology.capacityCells()[cell].capacity - usage[cell],
              "selected route set exceeds frozen Fabric capacity");
      usage[cell] += amount;
    }
  }
  require(llvm::any_of(usage, [](std::uint64_t value) { return value > 1; }),
          "System workflow did not exercise shared route capacity");
}

void requireFailureContains(llvm::Error error, llvm::StringRef diagnostic) {
  if (!error)
    fail("adverse System Mapping input unexpectedly succeeded");
  const std::string actual = llvm::toString(std::move(error));
  require(llvm::StringRef(actual).contains(diagnostic),
          "adverse System Mapping diagnostic changed: " + actual);
}

template <typename Attr, typename Ref>
Attr fabricRefAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(context,
                   loom::pnr::test::bytesAttr(
                       context, loom::fabric::canonicalFabricBytes(reference)));
}

::fabric::ResourceContract exclusiveResourceContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return take(::fabric::ResourceContract::create(std::move(declaration)));
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
inOrderMicroarchitecture() {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 1, 2, 1}},
      exclusiveResourceContract()};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 4, 2};
  return take(
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
}

} // namespace

mlir::DenseI8ArrayAttr
loom::pnr::test::bytesAttr(mlir::MLIRContext *context,
                           llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

std::vector<std::uint8_t>
loom::pnr::test::unsignedBytes(mlir::DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::string loom::pnr::test::byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string result = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      result += ", ";
    result += std::to_string(static_cast<std::int8_t>(byte));
  }
  return result + "]";
}

loom::CanonicalSemanticBytes
loom::pnr::test::rawSystemBytes(::mapping::SystemOp root) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  root.print(stream, mlir::OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

std::size_t loom::pnr::test::countOccurrences(llvm::StringRef text,
                                              llvm::StringRef needle) {
  std::size_t count = 0;
  while (true) {
    const std::size_t found = text.find(needle);
    if (found == llvm::StringRef::npos)
      return count;
    ++count;
    text = text.drop_front(found + needle.size());
  }
}

::mapping::SystemPresburgerCellAttr
loom::pnr::test::withFirstCoordinateLowerBound(
    ::mapping::SystemPresburgerCellAttr cell, std::int64_t lowerBound) {
  if (cell.getDimensionCount() == 0)
    fail("Presburger test cell has no logical coordinate");
  llvm::SmallVector<mlir::Attribute> inequalities(
      cell.getInequalities().begin(), cell.getInequalities().end());
  std::vector<std::int64_t> row(
      static_cast<std::size_t>(cell.getDimensionCount()) +
          cell.getSymbolCount() + cell.getLocalCount() + 1,
      0);
  row.front() = 1;
  row.back() = -lowerBound;
  inequalities.push_back(mlir::DenseI64ArrayAttr::get(cell.getContext(), row));
  return ::mapping::SystemPresburgerCellAttr::get(
      cell.getContext(), cell.getDimensionCount(), cell.getSymbolCount(),
      cell.getLocalCount(), cell.getEqualities(),
      mlir::ArrayAttr::get(cell.getContext(), inequalities));
}

loom::adg::FinalizedFabricDesign loom::pnr::test::buildHeterogeneousSystem(
    loom::ArtifactStore &store,
    const loom::fabric::FinalizedFabricRoot &baselineSystem,
    const loom::fabric::FinalizedFabricRoot &primaryModule,
    const loom::fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context, bool extraSupportsRead,
    bool routeExtraMemoryThroughTransform) {
  auto baseline = take(loom::fabric::requireSystemRoot(baselineSystem.view()));
  require(!baseline.artifact().systemMemoryServices().empty() &&
              !baseline.artifact().systemServiceEndpoints().empty(),
          "builtin System has no memory service capability source");
  const auto *memoryContract = baseline.memoryService(
      baseline.artifact().systemMemoryServices().front());
  const auto *memoryCapabilities = baseline.serviceEndpointCapabilities(
      baseline.artifact().systemServiceEndpoints().front());
  require(memoryContract && memoryCapabilities,
          "builtin System memory service contract is incomplete");

  loom::adg::DesignBuilder design(store);
  auto system = take(loom::adg::expandBuiltinSystem(
      design, loom::adg::BuiltinTargetPreset::Small, primaryModule));
  auto imported = take(system.importSpatialCore(alternateModule));
  const auto architecture =
      take(loom::adg::getBuiltinInstructionCoreArchitecture());
  auto extraCore = take(
      system.addAccCore(architecture, inOrderMicroarchitecture(), imported));

  const auto bits128 = take(loom::adg::PortType::bits(128));
  const auto transportContract = exclusiveResourceContract();
  std::vector<loom::adg::HardwareDomainMember> domainMembers = {
      extraCore.instructionCoreDomainMember(),
      extraCore.spatialCoreDomainMember()};
  std::vector<loom::adg::SystemTransportEndpoint> requestCarriers;
  std::vector<loom::adg::SystemTransportEndpoint> responseCarriers;
  std::vector<loom::adg::SystemTransportEndpoint> occurrenceRequestCarriers;
  std::vector<loom::adg::SystemTransportEndpoint> occurrenceResponseCarriers;
  for (std::uint32_t gateway = 0; gateway != 2; ++gateway) {
    auto transport = take(
        system.addTransportResource({{bits128}, {bits128}, transportContract}));
    auto pattern = take(system.addTransferPattern(transport, 0, {0}, 0));
    auto input = take(transport.input(0));
    auto output = take(transport.output(0));
    requestCarriers.push_back(input);
    responseCarriers.push_back(output);
    auto occurrenceRequest = take(extraCore.spatialTransportOutput(gateway));
    auto occurrenceResponse = take(extraCore.spatialTransportInput(gateway));
    occurrenceRequestCarriers.push_back(occurrenceRequest);
    occurrenceResponseCarriers.push_back(occurrenceResponse);
    if (llvm::Error error = system.connect(occurrenceRequest, input))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = system.connect(output, occurrenceResponse))
      fail(llvm::toString(std::move(error)));
    domainMembers.push_back(transport.domainMember());
    domainMembers.push_back(pattern.domainMember());
  }
  auto domain = take(system.createHardwareDomain());
  auto rate = take(system.createServiceRate(
      domain, 1, 1, 4,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>)));
  std::vector<loom::fabric::CanonicalServiceCapabilityRecord>
      localMemoryCapabilities;
  localMemoryCapabilities.reserve(memoryCapabilities->capabilities().size());
  for (const auto &capability : memoryCapabilities->capabilities()) {
    if (!extraSupportsRead &&
        capability.kind() == dataflow::semantics::ServiceKind::MemoryRead)
      continue;
    localMemoryCapabilities.push_back(
        take(loom::fabric::CanonicalServiceCapabilityRecord::create(
            capability.kind(), capability.role(), capability.domain(), rate)));
  }
  auto memoryCapabilitySet =
      take(loom::fabric::CanonicalServiceCapabilitySet::create(
          std::move(localMemoryCapabilities)));
  auto memoryService = take(system.addMemoryService(*memoryContract));
  auto memoryEndpoint =
      take(system.addServiceEndpoint(memoryService, memoryCapabilitySet));
  std::optional<loom::adg::ServiceTransformBuilder> memoryTransform;
  std::optional<loom::adg::SystemServiceEndpoint> transformInitiate;
  std::optional<loom::adg::SystemServiceEndpoint> transformServe;
  std::optional<loom::fabric::CanonicalServiceCapabilitySet>
      initiateCapabilitySet;
  if (routeExtraMemoryThroughTransform) {
    std::vector<loom::fabric::CanonicalServiceCapabilityRecord>
        initiateCapabilities;
    initiateCapabilities.reserve(memoryCapabilitySet.capabilities().size());
    for (const auto &capability : memoryCapabilitySet.capabilities())
      initiateCapabilities.push_back(
          take(loom::fabric::CanonicalServiceCapabilityRecord::create(
              capability.kind(),
              loom::fabric::CanonicalServiceEndpointRole::Initiate,
              capability.domain(), rate)));
    initiateCapabilitySet.emplace(
        take(loom::fabric::CanonicalServiceCapabilitySet::create(
            std::move(initiateCapabilities))));
    memoryTransform.emplace(take(system.createServiceTransform()));
    transformInitiate.emplace(take(
        system.addServiceEndpoint(*memoryTransform, *initiateCapabilitySet)));
    transformServe.emplace(
        take(system.addServiceEndpoint(*memoryTransform, memoryCapabilitySet)));
    if (llvm::Error error = system.connect(take(transformInitiate->memory()),
                                           take(memoryEndpoint.memory())))
      fail(llvm::toString(std::move(error)));
  }
  auto spatialMemory = take(extraCore.spatialMemoryManager(0));
  if (llvm::Error error = system.attachSpatialMemory(
          spatialMemory,
          routeExtraMemoryThroughTransform ? *transformServe : memoryEndpoint))
    fail(llvm::toString(std::move(error)));
  auto memoryEndpointRef = take(memoryEndpoint.memory());
  const auto attachEndpointCarriers =
      [&](const loom::adg::SystemMemoryEndpoint &endpoint,
          const loom::fabric::CanonicalServiceCapabilitySet &capabilities)
      -> llvm::Error {
    for (const auto &capability : capabilities.capabilities()) {
      const auto legCount =
          dataflow::semantics::getCanonicalServiceLegCount(capability.kind());
      for (dataflow::StructuralOrdinal leg = 0; leg != legCount; ++leg) {
        const auto direction =
            take(dataflow::semantics::getCanonicalServiceLegDirection(
                capability.kind(), leg));
        const bool endpointIsInitiator =
            capability.role() ==
            loom::fabric::CanonicalServiceEndpointRole::Initiate;
        const bool legSourceIsInitiator =
            direction ==
            dataflow::semantics::ServiceLegDirection::InitiatorToServer;
        const auto &carriers = endpointIsInitiator == legSourceIsInitiator
                                   ? responseCarriers
                                   : requestCarriers;
        if (llvm::Error error = system.attachServiceLegCarriers(
                endpoint, capability.kind(), leg, carriers))
          return error;
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          attachEndpointCarriers(memoryEndpointRef, memoryCapabilitySet))
    fail(llvm::toString(std::move(error)));
  if (routeExtraMemoryThroughTransform) {
    if (llvm::Error error = attachEndpointCarriers(
            take(transformInitiate->memory()), *initiateCapabilitySet))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = attachEndpointCarriers(
            take(transformServe->memory()), memoryCapabilitySet))
      fail(llvm::toString(std::move(error)));
  }
  for (const auto &capability : memoryCapabilitySet.capabilities()) {
    const auto legCount =
        dataflow::semantics::getCanonicalServiceLegCount(capability.kind());
    for (dataflow::StructuralOrdinal leg = 0; leg != legCount; ++leg) {
      const auto direction =
          take(dataflow::semantics::getCanonicalServiceLegDirection(
              capability.kind(), leg));
      const bool endpointIsInitiator =
          capability.role() ==
          loom::fabric::CanonicalServiceEndpointRole::Initiate;
      const bool legSourceIsInitiator =
          direction ==
          dataflow::semantics::ServiceLegDirection::InitiatorToServer;
      const auto &occurrenceCarriers =
          endpointIsInitiator == legSourceIsInitiator
              ? occurrenceResponseCarriers
              : occurrenceRequestCarriers;
      if (llvm::Error error = system.attachServiceLegCarriers(
              spatialMemory, capability.kind(), leg, occurrenceCarriers))
        fail(llvm::toString(std::move(error)));
    }
  }
  if (routeExtraMemoryThroughTransform) {
    if (llvm::Error error = memoryTransform->close(
            {take(transformServe->memory())},
            {take(transformInitiate->memory())},
            loom::fabric::AddressMaskXorTransform{64, 4095, 1}))
      fail(llvm::toString(std::move(error)));
    domainMembers.push_back(memoryTransform->domainMember());
    domainMembers.push_back(transformInitiate->domainMember());
    domainMembers.push_back(transformServe->domainMember());
  }
  domainMembers.push_back(memoryService.domainMember());
  domainMembers.push_back(memoryEndpoint.domainMember());

  auto initiateCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Initiate,
          take(loom::fabric::MessageTransferCapabilityDomain::create(
              {mlir::NoneType::get(&context),
               mlir::IntegerType::get(&context, 32),
               mlir::IndexType::get(&context)})),
          rate));
  auto serveCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Serve,
          take(loom::fabric::MessageTransferCapabilityDomain::create(
              {mlir::NoneType::get(&context),
               mlir::IntegerType::get(&context, 32),
               mlir::IndexType::get(&context)})),
          rate));
  auto initiateSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(initiateCapability)}));
  auto serveSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(serveCapability)}));
  auto messageSource =
      take(system.addServiceEndpoint(extraCore, initiateSet, bits128));
  auto messageSink =
      take(system.addServiceEndpoint(extraCore, serveSet, bits128));
  auto messageTransport = take(
      system.addTransportResource({{bits128}, {bits128}, transportContract}));
  auto messagePattern =
      take(system.addTransferPattern(messageTransport, 0, {0}, 0));
  if (llvm::Error error = system.connect(take(messageSource.transport()),
                                         take(messageTransport.input(0))))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.connect(take(messageTransport.output(0)),
                                         take(messageSink.transport())))
    fail(llvm::toString(std::move(error)));
  domainMembers.push_back(messageSource.domainMember());
  domainMembers.push_back(messageSink.domainMember());
  domainMembers.push_back(messageTransport.domainMember());
  domainMembers.push_back(messagePattern.domainMember());
  auto clock = take(loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = domain.close(domainMembers, std::move(clock)))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "heterogeneous fixture did not publish one System root");
  return finalized;
}

void loom::pnr::test::verifyFinalizedSystemMappingWorkflow(
    const SystemCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::fabric::FabricSystemRootView &fabric,
    const loom::mapping::SystemMappingConstraintSetView &emptyConstraints,
    ArtifactStore &store, mlir::MLIRContext &context,
    std::size_t expectedServiceCount) {
  verifySelectedRouteCapacity(candidate);
  auto finalized = take(finalizeSystemMappingCandidate(
      candidate, dataflow, fabric, emptyConstraints, store, context));
  auto imported =
      take(loom::mapping::importSystemMapping(finalized.reference(), store));
  auto replayed =
      take(loom::mapping::importSystemMapping(finalized.reference(), store));
  require(imported.canonicalBytes().bytes() ==
                  finalized.canonicalBytes().bytes() &&
              replayed.canonicalBytes().bytes() ==
                  finalized.canonicalBytes().bytes() &&
              imported.view().serviceRealizations().size() ==
                  expectedServiceCount &&
              imported.view().resourceUses().size() ==
                  candidate.instructionResourceUses().size() +
                      candidate.serviceResourceUses().size(),
          "full SystemMapping finalization or replay lost closure");

  auto draft = take(materializeSystemCandidateDraft(candidate, context));
  mlir::OwningOpRef<mlir::Operation *> missingUse(draft->clone());
  auto missingUseRoot = mlir::cast<::mapping::SystemOp>(missingUse.get());
  auto omittedUse = *missingUseRoot.getBody()
                         .front()
                         .getOps<::mapping::ResourceUseOp>()
                         .begin();
  omittedUse.erase();
  llvm::Error missingError = loom::mapping::verifySystemMappingBase(
      missingUseRoot, dataflow, fabric, store);
  require(static_cast<bool>(missingError),
          "missing System ResourceUse unexpectedly verified");
  const std::string missingDiagnostic = llvm::toString(std::move(missingError));
  require(llvm::StringRef(missingDiagnostic)
              .contains("ResourceUse closure is incomplete"),
          "missing ResourceUse diagnostic changed: " + missingDiagnostic);

  mlir::OwningOpRef<mlir::Operation *> missingSelection(draft->clone());
  auto missingSelectionRoot =
      mlir::cast<::mapping::SystemOp>(missingSelection.get());
  auto service = *missingSelectionRoot.getBody()
                      .front()
                      .getOps<::mapping::ServiceRealizationOp>()
                      .begin();
  auto unreachableSelection = *service.getBody()
                                   .front()
                                   .getOps<::mapping::ServicePlanSelectionOp>()
                                   .begin();
  auto unreachableKey = take(loom::mapping::decodeServicePlanSelectionKey(
      unsignedBytes(unreachableSelection.getKey().getRecord()),
      dataflow.identity()));
  if (std::holds_alternative<loom::mapping::InstructionExecutionContextKey>(
          unreachableKey.context)) {
    require(!candidate.problem().spatialMappings().empty(),
            "workflow needs a SpatialMapping for an unreachable context");
    unreachableKey.context = loom::mapping::SpatialExecutionContextKey{
        candidate.selectedAccCore(0),
        candidate.problem().spatialMappings().front().artifact};
  } else {
    unreachableKey.context = loom::mapping::InstructionExecutionContextKey{
        candidate.selectedAccCore(0)};
  }
  auto unreachableKeyBytes = take(loom::mapping::encodeServicePlanSelectionKey(
      dataflow.identity(), unreachableKey));
  unreachableSelection->setAttr(
      "key", ::mapping::ServicePlanSelectionKeyAttr::get(
                 &context, bytesAttr(&context, unreachableKeyBytes)));
  llvm::Error missingSelectionError = loom::mapping::verifySystemMappingBase(
      missingSelectionRoot, dataflow, fabric, store);
  require(static_cast<bool>(missingSelectionError),
          "missing System plan selection unexpectedly verified");
  const std::string missingSelectionDiagnostic =
      llvm::toString(std::move(missingSelectionError));
  require(llvm::StringRef(missingSelectionDiagnostic)
              .contains("ServicePlanSelection closure is incomplete"),
          "missing selection diagnostic changed: " +
              missingSelectionDiagnostic);

  mlir::OwningOpRef<mlir::Operation *> disconnectedRoute(draft->clone());
  auto disconnectedRoot =
      mlir::cast<::mapping::SystemOp>(disconnectedRoute.get());
  ::mapping::TransferLegRealizationOp selectedRoute;
  ::mapping::SystemRouteNodeOp selectedNode;
  disconnectedRoot.walk([&](::mapping::TransferLegRealizationOp route) {
    if (selectedRoute)
      return;
    auto nodes = route.getBody().front().getOps<::mapping::SystemRouteNodeOp>();
    if (nodes.empty())
      return;
    selectedRoute = route;
    selectedNode = *nodes.begin();
  });
  require(selectedRoute && selectedNode,
          "workflow has no nontrivial System route to disconnect");
  auto selectedTraversal = take(
      loom::fabric::decodeFabricRef<loom::fabric::FabricPhysicalTraversalRef>(
          unsignedBytes(selectedNode.getIncomingTraversal().getRecord())));
  const loom::fabric::FabricPhysicalTraversalView *traversalView = nullptr;
  for (const auto &traversal : fabric.artifact().physicalTraversals())
    if (traversal.reference == selectedTraversal) {
      traversalView = &traversal;
      break;
    }
  require(traversalView, "selected System traversal is absent from Fabric");
  std::optional<loom::fabric::FabricTransportEndpointRef> disconnectedEndpoint;
  for (const auto endpoint : fabric.artifact().transportEndpoints())
    if (!llvm::is_contained(traversalView->sources, endpoint)) {
      disconnectedEndpoint = endpoint;
      break;
    }
  require(disconnectedEndpoint.has_value(),
          "workflow Fabric has no endpoint outside the selected traversal");
  selectedRoute->setAttr(
      "root_endpoint",
      ::mapping::FabricTransportEndpointRefAttr::get(
          &context, bytesAttr(&context, loom::fabric::canonicalFabricBytes(
                                            *disconnectedEndpoint))));
  llvm::Error disconnectedError = loom::mapping::verifySystemMappingBase(
      disconnectedRoot, dataflow, fabric, store);
  require(static_cast<bool>(disconnectedError),
          "disconnected System route unexpectedly verified");
  const std::string disconnectedDiagnostic =
      llvm::toString(std::move(disconnectedError));
  require(llvm::StringRef(disconnectedDiagnostic)
              .contains("service route traversal is discontinuous"),
          "disconnected route diagnostic changed: " + disconnectedDiagnostic);

  const auto &problem = candidate.problem();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  llvm::SmallVector<mlir::Attribute> rootAttrs;
  for (const auto root : problem.rootThreadLaunches()) {
    auto encoded = take(
        ::dataflow::encodeDataflowReference(problem.dataflowIdentity(), root));
    rootAttrs.push_back(::mapping::RootThreadLaunchRefAttr::get(
        &context, bytesAttr(&context, encoded)));
  }
  auto constraintRoot = ::mapping::ConstraintsSystemOp::create(
      builder, builder.getUnknownLoc(),
      ::mapping::ArtifactIdentityAttr::get(
          &context, bytesAttr(&context, problem.dataflowIdentity().bytes())),
      ::mapping::ArtifactIdentityAttr::get(
          &context, bytesAttr(&context, problem.fabricIdentity().bytes())),
      builder.getArrayAttr(rootAttrs), builder.getArrayAttr({}));
  constraintRoot.getBody().emplaceBlock();
  llvm::SmallVector<mlir::Attribute> selectedCores;
  const auto selectedRoot = problem.rootThreadLaunches().front();
  for (const auto &[ordinal, decision] :
       llvm::enumerate(problem.threadDecisions()))
    if (decision.root == selectedRoot)
      selectedCores.push_back(::mapping::FabricAccCoreOccurrenceRefAttr::get(
          &context,
          bytesAttr(&context, loom::fabric::canonicalFabricBytes(
                                  candidate.selectedAccCore(ordinal)))));
  builder.setInsertionPointToEnd(&constraintRoot.getBody().front());
  mlir::OperationState restriction(
      builder.getUnknownLoc(),
      ::mapping::ConstraintDomainRestrictionOp::getOperationName());
  restriction.addAttribute(
      "projection",
      ::mapping::SystemConstraintProjectionKeyAttr::get(
          &context,
          static_cast<std::uint32_t>(
              ::mapping::SystemConstraintProjection::ThreadTargetAccCore)));
  restriction.addAttribute("subject", rootAttrs.front());
  restriction.addAttribute("admissible_domain",
                           builder.getArrayAttr(selectedCores));
  builder.create(restriction);
  auto admittedConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          constraintRoot, dataflow, fabric, store));
  auto admitted = take(finalizeSystemMappingCandidate(
      candidate, dataflow, fabric, admittedConstraints.view(), store, context));
  require(admitted.reference() == finalized.reference(),
          "independent System constraint admission changed Mapping identity");

  mlir::OwningOpRef<mlir::Operation *> rejectedOwner(constraintRoot->clone());
  auto rejectedRoot =
      mlir::cast<::mapping::ConstraintsSystemOp>(rejectedOwner.get());
  auto rejectedRestriction =
      *rejectedRoot.getBody()
           .front()
           .getOps<::mapping::ConstraintDomainRestrictionOp>()
           .begin();
  rejectedRestriction->setAttr("admissible_domain", builder.getArrayAttr({}));
  auto rejectedConstraints =
      take(loom::mapping::finalizeSystemMappingConstraintSet(
          rejectedRoot, dataflow, fabric, store));
  auto rejected = finalizeSystemMappingCandidate(
      candidate, dataflow, fabric, rejectedConstraints.view(), store, context);
  require(!rejected, "rejecting System constraint unexpectedly admitted");
  require(llvm::StringRef(llvm::toString(rejected.takeError()))
              .contains("system_mapping_rejected_by_constraint_set"),
          "System constraint rejection lost its typed diagnostic");
}

void loom::pnr::test::verifySystemServiceTargetRejections(
    ::mapping::SystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FabricSystemRootView &fabric, ArtifactStore &store,
    mlir::MLIRContext &context,
    llvm::ArrayRef<fabric::SystemServiceTransformRef> foreignTransformPath,
    fabric::FabricMemoryServiceRegionRef foreignRegion) {
  const auto findMemoryTarget = [&](::mapping::SystemOp root) {
    ::mapping::MemoryRegionTargetOp result;
    for (auto service :
         root.getBody().front().getOps<::mapping::ServiceRealizationOp>())
      for (auto plan :
           service.getBody().front().getOps<::mapping::ServicePlanOp>()) {
        auto targets =
            plan.getBody().front().getOps<::mapping::MemoryRegionTargetOp>();
        if (!targets.empty() && !result)
          result = *targets.begin();
      }
    return result;
  };
  const auto updateResourceOwners = [&](::mapping::SystemOp root,
                                        ::mapping::MemoryRegionTargetOp target,
                                        bool matchRegion) {
    for (auto use : root.getBody().front().getOps<::mapping::ResourceUseOp>()) {
      auto owner =
          mlir::dyn_cast<::mapping::ServicePlanElementRefAttr>(use.getOwner());
      if (!owner)
        continue;
      auto element = mlir::dyn_cast<::mapping::MemoryRegionElementKeyAttr>(
          owner.getElement());
      if (!element || element.getLogicalMemory() != target.getLogicalMemory() ||
          element.getInterval() != target.getInterval() ||
          (matchRegion &&
           element.getServiceRegion() != target.getServiceRegion()))
        continue;
      auto replacedElement = ::mapping::MemoryRegionElementKeyAttr::get(
          &context, element.getLogicalMemory(), element.getInterval(),
          target.getServiceRegion(), target.getTransformPath());
      use->setAttr("owner", ::mapping::ServicePlanElementRefAttr::get(
                                &context, owner.getService(),
                                owner.getPlanOrdinal(), replacedElement));
    }
  };

  mlir::OwningOpRef<mlir::Operation *> foreignPathDraft(source->clone());
  auto foreignPathRoot =
      mlir::cast<::mapping::SystemOp>(foreignPathDraft.get());
  auto foreignPathTarget = findMemoryTarget(foreignPathRoot);
  require(static_cast<bool>(foreignPathTarget),
          "foreign-path fixture has no memory target");
  llvm::SmallVector<mlir::Attribute> foreignPathAttributes;
  for (const auto transform : foreignTransformPath)
    foreignPathAttributes.push_back(
        fabricRefAttr<::mapping::SystemServiceTransformRefAttr>(&context,
                                                                transform));
  foreignPathTarget.setTransformPathAttr(
      mlir::ArrayAttr::get(&context, foreignPathAttributes));
  updateResourceOwners(foreignPathRoot, foreignPathTarget, true);
  requireFailureContains(
      loom::mapping::verifySystemMappingBase(foreignPathRoot, dataflow, fabric,
                                             store),
      "uniquely derived service transform path must be omitted");

  mlir::OwningOpRef<mlir::Operation *> foreignRegionDraft(source->clone());
  auto foreignRegionRoot =
      mlir::cast<::mapping::SystemOp>(foreignRegionDraft.get());
  auto foreignRegionTarget = findMemoryTarget(foreignRegionRoot);
  require(static_cast<bool>(foreignRegionTarget),
          "foreign-region fixture has no memory target");
  foreignRegionTarget.setServiceRegionAttr(
      fabricRefAttr<::mapping::FabricMemoryServiceRegionRefAttr>(
          &context, foreignRegion));
  updateResourceOwners(foreignRegionRoot, foreignRegionTarget, false);
  requireFailureContains(
      loom::mapping::verifySystemMappingBase(foreignRegionRoot, dataflow,
                                             fabric, store),
      "selected service target is outside its attachment-bound closure");
}
