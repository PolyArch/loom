#include "SystemCandidateStateTestSupport.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
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
