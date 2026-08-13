#include "SystemCandidateStateTestSupport.h"

#include "ADG/Builtin.h"
#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/MappingObjective.h"
#include "PnR/System/SystemActionDomain.h"
#include "PnR/System/SystemActionExecutor.h"
#include "PnR/System/SystemAnnealingSearch.h"
#include "PnR/System/SystemMappingMaterializer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
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

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
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

void requireFailureContains(
    const loom::mapping::SystemMappingBaseVerification &verification,
    llvm::StringRef expected) {
  if (std::holds_alternative<loom::mapping::VerifiedSystemMappingBase>(
          verification))
    fail("adverse System Mapping input unexpectedly succeeded");
  const std::string &actual = std::visit(
      [](const auto &result) -> const std::string & {
        using Result = std::decay_t<decltype(result)>;
        if constexpr (std::is_same_v<Result,
                                     loom::mapping::VerifiedSystemMappingBase>)
          llvm_unreachable("verified result has no diagnostic");
        else
          return result.diagnostic;
      },
      verification);
  require(llvm::StringRef(actual).contains(expected),
          "adverse System Mapping diagnostic changed: " + actual);
}

template <typename Attr, typename Ref>
Attr fabricRefAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(context,
                   loom::pnr::test::bytesAttr(
                       context, loom::fabric::canonicalFabricBytes(reference)));
}

::fabric::ResourceContract
singleRequesterResourceContract(std::uint32_t capacity = 1) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(capacity),
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

::fabric::ResourceContract selectableResourceContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0),
                            ::fabric::RequesterKey(1)};
  declaration.eligibilityCount = 2;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  for (std::uint32_t ordinal = 0; ordinal != 2; ++ordinal)
    declaration.usePatterns.push_back(
        {::fabric::UsePatternKey(ordinal),
         ::fabric::RequesterKey(ordinal),
         ::fabric::EligibilityKey(ordinal),
         ::fabric::EventKey(0),
         ::fabric::EventKey(1),
         std::nullopt,
         ::fabric::TimingContractKey(0),
         {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
           ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
         {{{::fabric::ClaimKey(0)}}}});
  declaration.grantPolicy = ::fabric::RoundRobinDeclaration{
      {::fabric::RequesterKey(0), ::fabric::RequesterKey(1)},
      ::fabric::RequesterKey(0)};
  return take(::fabric::ResourceContract::create(std::move(declaration)));
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
inOrderMicroarchitecture() {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 1, 2, 1}},
      singleRequesterResourceContract()};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 4, 2};
  return take(
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
}

} // namespace

loom::adg::FinalizedFabricDesign
loom::pnr::test::buildSystemCandidateSpatialModule(loom::ArtifactStore &store,
                                                   bool addBoundaryBuffer) {
  const std::uint32_t payloadWidth = 128;
  const auto payloadType = take(loom::adg::PortType::bits(payloadWidth));
  const auto byteType = take(loom::adg::PortType::bits(8));
  const auto managerType = take(loom::adg::PortType::memory(
      {loom::adg::PortType::kDynamicExtent}, byteType));
  const std::vector<loom::adg::PortType> boundaryTypes(4, payloadType);
  const std::vector<loom::adg::PortType> peInputTypes(5, payloadType);
  std::vector<loom::adg::PortType> moduleInputTypes = boundaryTypes;
  moduleInputTypes.push_back(managerType);
  loom::adg::DesignBuilder design(store);
  auto spatial = take(
      design.createSpatialCore(addBoundaryBuffer ? "system-candidate-buffered"
                                                 : "system-candidate-direct",
                               moduleInputTypes, boundaryTypes));
  auto network = take(spatial.addMeshSwitchNetwork(
      take(loom::adg::MeshSwitchNetworkSpec::spatial(
          2, 2, 2, payloadType,
          {{0, 0, {payloadType, payloadType}, {payloadType, payloadType}},
           {0, 1, {payloadType, payloadType}, {payloadType, payloadType}},
           {1, 0, peInputTypes, boundaryTypes},
           {1, 1, peInputTypes, boundaryTypes}}))));

  auto upperBoundary = take(network.attachment(0));
  auto lowerBoundary = take(network.attachment(1));
  requireSuccess(upperBoundary.connectOutputs(
      {take(spatial.input(0)), take(spatial.input(1))}));
  requireSuccess(lowerBoundary.connectOutputs(
      {take(spatial.input(2)), take(spatial.input(3))}));

  for (std::size_t attachmentOrdinal = 2; attachmentOrdinal != 4;
       ++attachmentOrdinal) {
    auto attachment = take(network.attachment(attachmentOrdinal));
    auto pe = take(
        spatial.addPe(attachment.inputs(),
                      loom::adg::PeSpec::spatial(peInputTypes, boundaryTypes)));
    std::vector<loom::adg::PeValue> peInputs;
    peInputs.reserve(peInputTypes.size());
    for (std::size_t ordinal = 0; ordinal != peInputTypes.size(); ++ordinal)
      peInputs.push_back(take(pe.input(ordinal)));
    requireSuccess(
        loom::adg::addTokenControlFu(pe, peInputs, {payloadWidth, 64}));
    requireSuccess(pe.close());
    std::vector<loom::adg::SpatialValue> peOutputs;
    peOutputs.reserve(boundaryTypes.size());
    for (std::size_t ordinal = 0; ordinal != boundaryTypes.size(); ++ordinal)
      peOutputs.push_back(take(pe.output(ordinal)));
    requireSuccess(attachment.connectOutputs(peOutputs));
  }

  std::vector<loom::adg::SpatialValue> outputs(upperBoundary.inputs().begin(),
                                               upperBoundary.inputs().end());
  outputs.insert(outputs.end(), lowerBoundary.inputs().begin(),
                 lowerBoundary.inputs().end());
  if (addBoundaryBuffer)
    outputs.front() =
        take(spatial.addFifo(outputs.front(),
                             loom::adg::FifoSpec{payloadType, 2, true}))
            .value();
  requireSuccess(spatial.close(outputs));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "SpatialCore fixture did not publish one Module root");
  return finalized;
}

loom::ResolvedConfig loom::pnr::test::buildSystemCandidateResolvedConfig() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.routing.negotiationIterationLimit = 8;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::CpSat, 256, 1024};
  resolved.dse.systemPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  auto &systemSearch = resolved.dse.systemPnr.search;
  systemSearch.initializer.seedAttemptCount = 1;
  systemSearch.routing.negotiationIterationLimit = 8;
  systemSearch.actionProposal = {0, 1, 0};
  systemSearch.annealing.calibrationProposalCount = 1;
  systemSearch.annealing.fallbackTemperature = 1;
  systemSearch.annealing.minimumTemperature = 1;
  systemSearch.annealing.coolingRatio = {1, 2};
  systemSearch.annealing.proposalsPerLevelBase = 1;
  systemSearch.annealing.proposalsPerMovableDecision = 0;
  systemSearch.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  return resolved;
}

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

static loom::adg::FinalizedFabricDesign buildHeterogeneousSystemImpl(
    loom::ArtifactStore &store,
    const loom::fabric::FinalizedFabricRoot &baselineSystem,
    const loom::fabric::FinalizedFabricRoot &primaryModule,
    const loom::fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context, bool extraSupportsRead,
    bool routeExtraMemoryThroughTransform, bool negotiatedRoutingMesh) {
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
  auto host = take(system.hostCore(0));
  auto missingHost = system.hostCore(1);
  require(!missingHost, "System Builder admitted a foreign HostCore ordinal");
  llvm::consumeError(missingHost.takeError());
  auto imported = take(system.importSpatialCore(alternateModule));
  const auto architecture =
      take(loom::adg::getBuiltinInstructionCoreArchitecture());
  auto extraCore = take(
      system.addAccCore(architecture, inOrderMicroarchitecture(), imported));

  const auto bits128 = take(loom::adg::PortType::bits(128));
  const auto transportContract = singleRequesterResourceContract();
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
  std::vector<::fabric::MemoryServiceCapabilityDeclaration>
      selectableCapabilities(memoryContract->capabilities().begin(),
                             memoryContract->capabilities().end());
  for (auto &capability : selectableCapabilities)
    capability.admissibleUsePatterns = {::fabric::UsePatternKey(0),
                                        ::fabric::UsePatternKey(1)};
  auto selectableMemoryContract =
      take(::fabric::MemoryServiceContractRecord::create(
          &context, ::fabric::MemoryServiceOwnerKind::System,
          {{memoryContract->regions().begin(), memoryContract->regions().end()},
           selectableResourceContract(),
           std::move(selectableCapabilities)}));
  auto memoryService = take(system.addMemoryService(selectableMemoryContract));
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
  auto hostMessageSource =
      take(system.addServiceEndpoint(host, initiateSet, bits128));
  auto hostMessageSink =
      take(system.addServiceEndpoint(host, serveSet, bits128));
  auto coreMessageSource =
      take(system.addServiceEndpoint(extraCore, initiateSet, bits128));
  auto coreMessageSink =
      take(system.addServiceEndpoint(extraCore, serveSet, bits128));
  const std::array messageSources{hostMessageSource, coreMessageSource};
  const std::array messageSinks{hostMessageSink, coreMessageSink};
  std::vector<loom::adg::SystemTransportResource> messageRouters;
  messageRouters.reserve(2);
  const auto &builtinScale = loom::adg::getBuiltinTargetDescriptor(
                                 loom::adg::BuiltinTargetPreset::Small)
                                 .scale;
  const auto messageTransportContract = singleRequesterResourceContract(
      builtinScale.accCoreCount * builtinScale.temporalResidentContexts);
  const std::array<std::vector<std::uint32_t>, 3> messagePatterns = {
      std::vector<std::uint32_t>{0}, std::vector<std::uint32_t>{1},
      std::vector<std::uint32_t>{0, 1}};
  for (std::size_t ordinal = 0; ordinal != messageSources.size(); ++ordinal) {
    messageRouters.push_back(take(system.addTransportResource(
        {{bits128, bits128}, {bits128, bits128}, messageTransportContract})));
    domainMembers.push_back(messageRouters.back().domainMember());
    for (std::size_t input = 0; input != 2; ++input)
      for (const auto &outputs : messagePatterns) {
        auto pattern = take(system.addTransferPattern(messageRouters[ordinal],
                                                      input, outputs, 0));
        domainMembers.push_back(pattern.domainMember());
      }
    if (llvm::Error error =
            system.connect(take(messageSources[ordinal].transport()),
                           take(messageRouters[ordinal].input(0))))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error =
            system.connect(take(messageRouters[ordinal].output(0)),
                           take(messageSinks[ordinal].transport())))
      fail(llvm::toString(std::move(error)));
  }
  for (std::size_t ordinal = 0; ordinal != messageRouters.size(); ++ordinal) {
    auto source = take(messageRouters[ordinal].output(1));
    auto sink =
        take(messageRouters[(ordinal + 1) % messageRouters.size()].input(1));
    if (!negotiatedRoutingMesh) {
      if (llvm::Error error = system.connect(source, sink))
        fail(llvm::toString(std::move(error)));
      continue;
    }

    auto split = take(system.addTransportResource(
        {{bits128}, {bits128, bits128}, messageTransportContract}));
    auto directBranch = take(system.addTransferPattern(split, 0, {0}, 0));
    auto bypassBranch = take(system.addTransferPattern(split, 0, {1}, 0));
    auto merge = take(system.addTransportResource(
        {{bits128, bits128}, {bits128}, messageTransportContract}));
    auto directMerge = take(system.addTransferPattern(merge, 0, {0}, 0));
    auto bypassMerge = take(system.addTransferPattern(merge, 1, {0}, 0));
    domainMembers.push_back(split.domainMember());
    domainMembers.push_back(directBranch.domainMember());
    domainMembers.push_back(bypassBranch.domainMember());
    domainMembers.push_back(merge.domainMember());
    domainMembers.push_back(directMerge.domainMember());
    domainMembers.push_back(bypassMerge.domainMember());

    auto trunk = take(
        system.addTransportResource({{bits128}, {bits128}, transportContract}));
    auto trunkPattern = take(system.addTransferPattern(trunk, 0, {0}, 0));
    domainMembers.push_back(trunk.domainMember());
    domainMembers.push_back(trunkPattern.domainMember());

    std::array<loom::adg::SystemTransportResource, 3> bypass = {
        take(system.addTransportResource(
            {{bits128}, {bits128}, transportContract})),
        take(system.addTransportResource(
            {{bits128}, {bits128}, transportContract})),
        take(system.addTransportResource(
            {{bits128}, {bits128}, transportContract}))};
    for (auto &hop : bypass) {
      auto pattern = take(system.addTransferPattern(hop, 0, {0}, 0));
      domainMembers.push_back(hop.domainMember());
      domainMembers.push_back(pattern.domainMember());
    }

    const std::array<std::pair<loom::adg::SystemTransportEndpoint,
                               loom::adg::SystemTransportEndpoint>,
                     8>
        connections = {
            std::pair{source, take(split.input(0))},
            std::pair{take(split.output(0)), take(trunk.input(0))},
            std::pair{take(trunk.output(0)), take(merge.input(0))},
            std::pair{take(split.output(1)), take(bypass[0].input(0))},
            std::pair{take(bypass[0].output(0)), take(bypass[1].input(0))},
            std::pair{take(bypass[1].output(0)), take(bypass[2].input(0))},
            std::pair{take(bypass[2].output(0)), take(merge.input(1))},
            std::pair{take(merge.output(0)), sink}};
    for (const auto &[connectionSource, connectionSink] : connections)
      if (llvm::Error error = system.connect(connectionSource, connectionSink))
        fail(llvm::toString(std::move(error)));
  }
  domainMembers.push_back(hostMessageSource.domainMember());
  domainMembers.push_back(hostMessageSink.domainMember());
  domainMembers.push_back(coreMessageSource.domainMember());
  domainMembers.push_back(coreMessageSink.domainMember());
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

loom::adg::FinalizedFabricDesign loom::pnr::test::buildHeterogeneousSystem(
    loom::ArtifactStore &store,
    const loom::fabric::FinalizedFabricRoot &baselineSystem,
    const loom::fabric::FinalizedFabricRoot &primaryModule,
    const loom::fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context, bool extraSupportsRead,
    bool routeExtraMemoryThroughTransform) {
  return buildHeterogeneousSystemImpl(
      store, baselineSystem, primaryModule, alternateModule, context,
      extraSupportsRead, routeExtraMemoryThroughTransform,
      /*negotiatedRoutingMesh=*/false);
}

loom::adg::FinalizedFabricDesign loom::pnr::test::buildNegotiatedRoutingSystem(
    loom::ArtifactStore &store,
    const loom::fabric::FinalizedFabricRoot &baselineSystem,
    const loom::fabric::FinalizedFabricRoot &primaryModule,
    mlir::MLIRContext &context) {
  return buildHeterogeneousSystemImpl(store, baselineSystem, primaryModule,
                                      primaryModule, context,
                                      /*extraSupportsRead=*/true,
                                      /*routeExtraMemoryThroughTransform=*/true,
                                      /*negotiatedRoutingMesh=*/true);
}

void loom::pnr::test::verifyFinalizedSystemMappingWorkflow(
    const SystemCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::fabric::FabricSystemRootView &fabric,
    const loom::mapping::SystemMappingConstraintSetView &emptyConstraints,
    ArtifactStore &store, mlir::MLIRContext &context,
    std::size_t expectedServiceCount) {
  verifySelectedRouteCapacity(candidate);
  verifySystemImportedCapacityWorkflow(candidate);
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
  require(std::holds_alternative<loom::mapping::VerifiedSystemMappingBase>(
              loom::mapping::verifySystemMappingBase(
                  mlir::cast<::mapping::SystemOp>(draft.get()), dataflow,
                  fabric, store)),
          "complete SystemMapping draft did not produce Verified");
  mlir::OwningOpRef<mlir::Operation *> missingUse(draft->clone());
  auto missingUseRoot = mlir::cast<::mapping::SystemOp>(missingUse.get());
  auto omittedUse = *missingUseRoot.getBody()
                         .front()
                         .getOps<::mapping::ResourceUseOp>()
                         .begin();
  omittedUse.erase();
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             missingUseRoot, dataflow, fabric, store),
                         "ResourceUse closure is incomplete");

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
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             missingSelectionRoot, dataflow, fabric, store),
                         "ServicePlanSelection closure is incomplete");

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
  requireFailureContains(loom::mapping::verifySystemMappingBase(
                             disconnectedRoot, dataflow, fabric, store),
                         "service route traversal is discontinuous");

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

void loom::pnr::test::verifySystemResourceAction(
    const SystemCandidateStateHandle &candidate) {
  SystemActionDomainScratch domain;
  if (llvm::Error error = domain.rebuild(*candidate))
    fail(llvm::toString(std::move(error)));
  require(!domain.view().resourceAnchors.empty(),
          "memory/service fixture exposes no resource Action choice");
  const auto routingChoices = domain.view().routingChoices;
  const auto single = llvm::find_if(routingChoices, [](const auto &action) {
    return std::holds_alternative<SystemSingleSinkRoutingAction>(action);
  });
  const auto subtree = llvm::find_if(routingChoices, [](const auto &action) {
    return std::holds_alternative<SystemRootedSubtreeRoutingAction>(action);
  });
  const auto global = llvm::find_if(routingChoices, [](const auto &action) {
    return std::holds_alternative<SystemGlobalRoutingAction>(action);
  });
  require(single != routingChoices.end() && subtree != routingChoices.end() &&
              global != routingChoices.end(),
          "System routing domain omitted a closed negotiated scope");

  const auto sameRoute = [](const SystemCandidateState &lhs,
                            const SystemCandidateState &rhs, PnrIndex leg) {
    const auto findRoute = [&](const SystemCandidateState &value) {
      return llvm::find_if(value.serviceRoutes(),
                           [&](const auto &route) { return route.leg == leg; });
    };
    const auto left = findRoute(lhs);
    const auto right = findRoute(rhs);
    if (left == lhs.serviceRoutes().end() ||
        right == rhs.serviceRoutes().end() ||
        left->rootEndpoint != right->rootEndpoint ||
        left->nodeCount != right->nodeCount ||
        left->sinkCount != right->sinkCount)
      return false;
    const auto leftNodes =
        lhs.serviceRouteNodes().slice(left->nodeOffset, left->nodeCount);
    const auto rightNodes =
        rhs.serviceRouteNodes().slice(right->nodeOffset, right->nodeCount);
    for (const auto &[index, node] : llvm::enumerate(leftNodes)) {
      const auto &other = rightNodes[index];
      if (node.endpoint != other.endpoint ||
          node.parentNode != other.parentNode ||
          node.incomingTraversal != other.incomingTraversal)
        return false;
    }
    const auto leftSinks =
        lhs.serviceRouteSinks().slice(left->sinkOffset, left->sinkCount);
    const auto rightSinks =
        rhs.serviceRouteSinks().slice(right->sinkOffset, right->sinkCount);
    for (const auto &[index, sink] : llvm::enumerate(leftSinks))
      if (sink.terminal != rightSinks[index].terminal ||
          sink.node != rightSinks[index].node)
        return false;
    return true;
  };
  const auto exerciseLocal = [&](const SystemTransportRoutingAction &action,
                                 PnrIndex leg) {
    const auto sinkPaths = [&](const SystemCandidateState &state) {
      std::map<PnrIndex, std::vector<PnrIndex>> result;
      const auto route =
          llvm::find_if(state.serviceRoutes(), [&](const auto &candidateRoute) {
            return candidateRoute.leg == leg;
          });
      require(route != state.serviceRoutes().end(),
              "local routing path fixture lost its service leg");
      const auto nodes =
          state.serviceRouteNodes().slice(route->nodeOffset, route->nodeCount);
      for (const auto &sink : state.serviceRouteSinks().slice(
               route->sinkOffset, route->sinkCount)) {
        std::vector<PnrIndex> path;
        PnrIndex node = sink.node;
        while (node != 0) {
          require(node < nodes.size(),
                  "local routing path fixture has a foreign node");
          path.push_back(nodes[node].incomingTraversal);
          node = nodes[node].parentNode;
        }
        std::reverse(path.begin(), path.end());
        result.try_emplace(sink.terminal, std::move(path));
      }
      return result;
    };
    auto outsidePaths = sinkPaths(*candidate);
    const auto route = llvm::find_if(
        candidate->serviceRoutes(),
        [&](const auto &candidateRoute) { return candidateRoute.leg == leg; });
    const auto nodes = candidate->serviceRouteNodes().slice(route->nodeOffset,
                                                            route->nodeCount);
    const auto sinks = candidate->serviceRouteSinks().slice(route->sinkOffset,
                                                            route->sinkCount);
    if (const auto *value =
            std::get_if<SystemSingleSinkRoutingAction>(&action)) {
      outsidePaths.erase(sinks[value->sinkObligation].terminal);
    } else if (const auto *value =
                   std::get_if<SystemRootedSubtreeRoutingAction>(&action)) {
      const auto root = llvm::find_if(nodes, [&](const auto &node) {
        return node.endpoint == value->rootEndpoint;
      });
      require(root != nodes.end(),
              "RootedSubtree path fixture lost its anchor");
      const PnrIndex rootNode = static_cast<PnrIndex>(root - nodes.begin());
      for (const auto &sink : sinks) {
        PnrIndex node = sink.node;
        while (node != 0 && node != rootNode)
          node = nodes[node].parentNode;
        if (node == rootNode)
          outsidePaths.erase(sink.terminal);
      }
    }
    auto objective =
        take(candidate->problem().objectiveProgram().evaluate(*candidate));
    SystemActionProbeAccounting work;
    auto probe = probeSystemAction(candidate, objective,
                                   SystemMappingAction{action}, work);
    require(work.assignmentAttempts == 0 && work.negotiationIterations != 0,
            "local routing Action consumed the wrong work domain");
    if (!probe) {
      bool transition = false;
      llvm::Error remaining = llvm::handleErrors(
          probe.takeError(),
          [&](const SystemActionTransitionFailure &) { transition = true; });
      if (remaining)
        fail(llvm::toString(std::move(remaining)));
      require(transition,
              "local routing Action lost its typed rollback outcome");
      return false;
    }
    for (const auto &route : candidate->serviceRoutes())
      if (route.leg != leg)
        require(sameRoute(*candidate, *probe->candidate, route.leg),
                "local routing Action changed an unrelated service leg");
    const auto repairedPaths = sinkPaths(*probe->candidate);
    for (const auto &[terminal, path] : outsidePaths) {
      const auto found = repairedPaths.find(terminal);
      require(found != repairedPaths.end() && found->second == path,
              "local routing Action changed an outside-region sink path");
    }
    if (llvm::Error error = probe->candidate->verify())
      fail(llvm::toString(std::move(error)));
    return true;
  };
  bool singleClosed = false;
  bool subtreeClosed = false;
  for (const auto &routing : routingChoices) {
    if (!singleClosed)
      if (const auto *value =
              std::get_if<SystemSingleSinkRoutingAction>(&routing))
        singleClosed = exerciseLocal(routing, value->leg);
    if (!subtreeClosed)
      if (const auto *value =
              std::get_if<SystemRootedSubtreeRoutingAction>(&routing))
        subtreeClosed = exerciseLocal(routing, value->leg);
  }
  require(singleClosed && subtreeClosed,
          "pressure Fabric did not close both local routing scopes");

  auto objective =
      take(candidate->problem().objectiveProgram().evaluate(*candidate));
  SystemActionProbeAccounting globalWork;
  auto globalProbe = take(probeSystemAction(
      candidate, objective, SystemMappingAction{*global}, globalWork));
  require(globalWork.assignmentAttempts == 0 &&
              globalWork.negotiationIterations != 0,
          "Global routing Action did not consume negotiated routing work");
  if (llvm::Error error = globalProbe.candidate->verify())
    fail(llvm::toString(std::move(error)));

  const SystemResourceAllocationAction action =
      domain.view().resourceChoices.front();
  require(std::holds_alternative<SystemServiceUsePatternAction>(action),
          "memory fixture selected the wrong resource Action kind");
  auto resourceObjective =
      take(candidate->problem().objectiveProgram().evaluate(*candidate));
  SystemActionProbeAccounting accounting;
  auto probe = take(probeSystemAction(candidate, resourceObjective,
                                      SystemMappingAction{action}, accounting));
  require(accounting.assignmentAttempts == 0 &&
              accounting.endpointExpansions == 0 &&
              accounting.negotiationIterations == 0,
          "resource Action consumed unrelated binding or routing work");
  if (llvm::Error error = probe.candidate->verify())
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = candidate->verify())
    fail(llvm::toString(std::move(error)));
}

void loom::pnr::test::verifySystemFixedTerminalCutAndAnnealing(
    FrozenSystemPnrProblemHandle problem,
    const SystemCandidateStateHandle &baseline) {
  std::vector<PnrIndex> threadChoices(problem->threadDecisions().size(), 0);
  std::vector<PnrIndex> graphChoices(problem->graphDecisions().size(), 0);
  auto candidate =
      take(initializeSystemCandidate(problem, threadChoices, graphChoices));
  require(candidate->capacityOveruse() != 0,
          "fixed-terminal cut fixture unexpectedly closed capacity");

  auto objective =
      take(candidate->problem().objectiveProgram().evaluate(*candidate));
  SystemActionProbeAccounting work;
  auto probe =
      probeSystemAction(candidate, objective,
                        SystemMappingAction{SystemTransportRoutingAction{
                            SystemGlobalRoutingAction{}}},
                        work, SystemActionExecutionContext::FinalClosure);
  require(!probe, "strict global Action ignored a fixed-terminal cut");
  bool typedFixedTerminalCut = false;
  llvm::Error failure = llvm::handleErrors(
      probe.takeError(),
      [&](const SystemActionTransitionFailure &transitionFailure) {
        typedFixedTerminalCut =
            transitionFailure.kind() ==
            SystemActionTransitionFailureKind::IntrinsicInvalid;
      });
  if (failure)
    fail(llvm::toString(std::move(failure)));
  require(typedFixedTerminalCut && work.negotiationIterations == 1,
          "fixed-terminal cut lost its intrinsic Action failure kind");

  auto annealed = baseline;
  SystemAnnealingSearchScratch annealing;
  const auto statistics = take(annealing.run(annealed, 0));
  require(statistics.calibrationProposalSlots == 1 &&
              statistics.annealingBaseProposalSlots == 1,
          "System annealing diagnostic fixture consumed unexpected work");
  if (llvm::Error error = annealed->verify())
    fail(llvm::toString(std::move(error)));
}

void loom::pnr::test::verifySystemResourceActionWorkflow(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ArtifactRootReference &spatialMapping, const ResolvedConfig &resolved,
    const ResolvedPnrConfigView &config, mlir::MLIRContext &context) {
  verifySystemNegotiatedRoutingWorkflow(store, baselineSystem, primaryModule,
                                        dataflow, spatialMapping, resolved,
                                        context);
  auto design = buildHeterogeneousSystem(
      store, baselineSystem, primaryModule, primaryModule, context,
      /*extraSupportsRead=*/true,
      /*routeExtraMemoryThroughTransform=*/true);
  auto system = take(fabric::requireSystemRoot(design.roots().front().view()));
  std::vector<dataflow::RootThreadLaunchRef> roots{
      dataflow.rootThreadLaunches().front().ref};
  auto constraints = take(mapping::finalizeEmptySystemMappingConstraintSet(
      dataflow, system, roots, store));
  auto partition = take(projectWholeDomainPresburgerPartitionPlan(
      dataflow, constraints.view().rootThreadLaunches()));
  auto searchDomain = take(projectSystemPnrSearchDomain(
      dataflow, system, config, constraints, partition,
      SystemHierarchicalGraphSearchInput{{spatialMapping}}, store));
  auto problem = take(freezeSystemPnrProblem(dataflow, system, searchDomain,
                                             config, constraints, store));
  const auto selectable =
      llvm::find_if(problem->memoryServiceBindings(), [](const auto &binding) {
        return llvm::any_of(binding.usePatternDomains, [](const auto &domain) {
          return domain.patterns.size() > 1;
        });
      });
  require(selectable != problem->memoryServiceBindings().end(),
          "custom memory service exposes no resource Action choice");
  const auto corePosition =
      llvm::find(problem->accCores(), selectable->accCore);
  require(corePosition != problem->accCores().end(),
          "resource Action AccCore is absent from the frozen catalog");
  const PnrIndex targetCore =
      static_cast<PnrIndex>(corePosition - problem->accCores().begin());
  const PnrIndex targetClass = problem->accCoreTargetClass(targetCore);

  std::vector<PnrIndex> threadChoices(problem->threadDecisions().size(), 0);
  for (PnrIndex decision = 0; decision < problem->threadDecisions().size();
       ++decision) {
    const auto choices = problem->threadChoiceCatalogOrdinals(decision);
    const auto selected = llvm::find_if(choices, [&](PnrIndex core) {
      return problem->accCoreTargetClass(core) == targetClass;
    });
    require(selected != choices.end(),
            "resource Action target class is absent from a thread domain");
    threadChoices[decision] = static_cast<PnrIndex>(selected - choices.begin());
  }

  std::vector<PnrIndex> graphChoices(problem->graphDecisions().size(), 0);
  for (PnrIndex decision = 0; decision < problem->graphDecisions().size();
       ++decision) {
    const auto choices = problem->graphChoiceCatalogOrdinals(decision);
    const auto selected = llvm::find_if(choices, [&](PnrIndex mapping) {
      return problem->spatialMappingTargetClass(mapping) == targetClass;
    });
    require(selected != choices.end(),
            "resource Action target class is absent from a graph domain");
    graphChoices[decision] = static_cast<PnrIndex>(selected - choices.begin());
  }
  std::string lastDiagnostic;
  for (PnrIndex decision = 0; decision < problem->threadDecisions().size();
       ++decision) {
    const auto choices = problem->threadChoiceCatalogOrdinals(decision);
    const auto selected = llvm::find(choices, targetCore);
    if (selected == choices.end())
      continue;
    auto trial = threadChoices;
    trial[decision] = static_cast<PnrIndex>(selected - choices.begin());
    auto candidate = initializeSystemCandidate(problem, trial, graphChoices);
    if (!candidate) {
      lastDiagnostic = llvm::toString(candidate.takeError());
      continue;
    }
    SystemActionDomainScratch domain;
    if (llvm::Error error = domain.rebuild(**candidate))
      fail(llvm::toString(std::move(error)));
    if (domain.view().resourceAnchors.empty())
      continue;
    verifySystemResourceAction(*candidate);
    ResolvedConfig dualResolved = resolved;
    dualResolved.dse.systemPnr.search.routing.negotiation =
        ResolvedDualSubgradientPolicy{
            ResolvedDualDirectionKernel::ProjectedSigned,
            std::nullopt,
            {ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
    const auto dualConfig =
        take(projectResolvedSystemPnrConfigView(dualResolved));
    auto dualSearchDomain = take(projectSystemPnrSearchDomain(
        dataflow, system, dualConfig, constraints, partition,
        SystemHierarchicalGraphSearchInput{{spatialMapping}}, store));
    auto dualProblem = take(freezeSystemPnrProblem(
        dataflow, system, dualSearchDomain, dualConfig, constraints, store));
    auto dualCandidate =
        take(initializeSystemCandidate(dualProblem, trial, graphChoices));
    verifySystemResourceAction(dualCandidate);
    return;
  }
  fail("no finite-degree resource Action candidate is routable: " +
       lastDiagnostic);
}
