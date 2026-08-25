#include "DeploymentTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/MemoryLibrary.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/DeploymentPipeline.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/Implementation/RepresentationFormat.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/PnrConfig.h"
#include "PnR/System/SystemPnrGenerator.h"
#include "PnR/System/SystemPnrSearchDomain.h"
#include "Runtime/InProcessPlatform.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::deployment::test {
namespace {

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    ::mapping::MappingDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

CanonicalTypeBytes typeBytes(llvm::StringRef test, mlir::Type type) {
  auto encoded = take(test, dataflow::encodeCanonicalType(type));
  return CanonicalTypeBytes(encoded.bytes().begin(), encoded.bytes().end());
}

struct HostCatalog final {
  std::vector<HostProgramEntry> entries;
  std::vector<HostExternalInterface> interfaces;
};

HostCatalog systemArtifactHostCatalog(llvm::StringRef test) {
  const CanonicalTypeBytes i32 =
      typeBytes(test, mlir::IntegerType::get(&context(), 32));
  const CanonicalTypeBytes i8 =
      typeBytes(test, mlir::IntegerType::get(&context(), 8));
  const CanonicalTypeBytes pointer =
      typeBytes(test, mlir::LLVM::LLVMPointerType::get(&context()));
  const CanonicalTypeBytes memory = typeBytes(
      test, mlir::MemRefType::get({16}, mlir::IntegerType::get(&context(), 8)));
  return {{{0, "loom_host_entry", {i32}, {i32}, {0, 1, 2, 3, 4, 5, 6}}},
          {{0, HostExternalInterfaceKind::Value,
            HostExternalInterfaceDirection::Input, pointer},
           {1, HostExternalInterfaceKind::Stream,
            HostExternalInterfaceDirection::Input, i8},
           {2, HostExternalInterfaceKind::Memory,
            HostExternalInterfaceDirection::InOut, memory},
           {3, HostExternalInterfaceKind::Value,
            HostExternalInterfaceDirection::Output, i32},
           {4, HostExternalInterfaceKind::Stream,
            HostExternalInterfaceDirection::Output, i8},
           {5, HostExternalInterfaceKind::Memory,
            HostExternalInterfaceDirection::InOut, memory},
           {6, HostExternalInterfaceKind::Memory,
            HostExternalInterfaceDirection::InOut, memory}}};
}

fabric::InstructionCoreMicroarchitecturalRealization
microarchitecture(llvm::StringRef test) {
  fabric::InstructionCoreCommonDeclaration common{
      1,
      {{fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      ::fabric::oneCycleElasticOperationResourceContract()};
  fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1, 1, 1, 2, 1};
  return take(
      test, fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
                std::move(common), pipeline));
}

fabric::InstructionCoreMicroarchitecturalRealization
outOfOrderMicroarchitecture(llvm::StringRef test) {
  fabric::InstructionCoreCommonDeclaration common{
      1,
      {{fabric::InstructionOperationClass::IntegerAlu, 2, 1, 1},
       {fabric::InstructionOperationClass::LoadStore, 2, 2, 1}},
      ::fabric::oneCycleElasticOperationResourceContract()};
  fabric::OutOfOrderMicroarchitectureDeclaration pipeline{
      2, 2, 2, 2, 2, 2, 2, 32, 16, 8, 8, 64, 64, 64};
  return take(
      test,
      fabric::InstructionCoreMicroarchitecturalRealization::createOutOfOrder(
          std::move(common), pipeline));
}

::fabric::ResourceContract
sharedTransportResourceContract(llvm::StringRef test,
                                std::uint32_t residentRouteCapacity) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {{::fabric::StateKey(0),
                         {{::fabric::CapacityDimensionKey(0),
                           ::fabric::CapacityUnits(residentRouteCapacity),
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
  return take(test, ::fabric::ResourceContract::create(std::move(declaration)));
}

fabric::FinalizedFabricRoot buildModule(llvm::StringRef test,
                                        const ArtifactStore &artifacts) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @configured(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>, %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir",
                                                        &context());
  require(test, static_cast<bool>(source), "cannot parse Fabric fixture");
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), signedContract));
  });
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no Module root");
  return take(test, fabric::finalizeFabricRoot(root, artifacts));
}

fabric::FinalizedFabricRoot
buildSystem(llvm::StringRef test, const fabric::FinalizedFabricRoot &module,
            const ArtifactStore &artifacts,
            llvm::ArrayRef<mlir::Type> messagePayloads = {},
            MappedSpatialSystemSpec spec = {}) {
  require(test, spec.accCoreCount != 0,
          "mapped System requires at least one AccCore");
  adg::DesignBuilder design(artifacts);
  auto system = take(test, design.createSystem("deployment-system"));
  auto imported = take(test, system.importSpatialCore(module));
  const auto architecture =
      take(test, adg::getBuiltinInstructionCoreArchitecture());
  const auto inOrder = microarchitecture(test);
  const auto outOfOrder = outOfOrderMicroarchitecture(test);
  auto host = take(test, system.addHostCore(architecture, inOrder));
  std::vector<adg::AccCore> cores;
  for (std::uint64_t ordinal = 0; ordinal != spec.accCoreCount; ++ordinal) {
    const auto &microarchitecture =
        spec.alternateInstructionMicroarchitectures && ordinal % 2 != 0
            ? outOfOrder
            : inOrder;
    cores.push_back(take(
        test, system.addAccCore(architecture, microarchitecture, imported)));
  }

  auto clock = take(test, system.createHardwareDomain());
  const auto rate = take(
      test,
      system.createServiceRate(
          clock, 1, 1, 1,
          fabric::ServiceProgress(std::in_place_type<::fabric::FairEventual>)));
  const std::array<mlir::Type, 1> defaultMessagePayloads{
      mlir::NoneType::get(&context())};
  if (messagePayloads.empty())
    messagePayloads = defaultMessagePayloads;
  const auto messageDomain = take(
      test, fabric::MessageTransferCapabilityDomain::create(messagePayloads));
  const auto initiateCapability =
      take(test, fabric::CanonicalServiceCapabilityRecord::create(
                     dataflow::semantics::ServiceKind::MessageTransfer,
                     fabric::CanonicalServiceEndpointRole::Initiate,
                     messageDomain, rate));
  const auto serveCapability =
      take(test, fabric::CanonicalServiceCapabilityRecord::create(
                     dataflow::semantics::ServiceKind::MessageTransfer,
                     fabric::CanonicalServiceEndpointRole::Serve, messageDomain,
                     rate));
  const auto initiateSet =
      take(test,
           fabric::CanonicalServiceCapabilitySet::create({initiateCapability}));
  const auto serveSet = take(
      test, fabric::CanonicalServiceCapabilitySet::create({serveCapability}));
  const auto carrier = take(test, adg::PortType::bits(128));
  std::vector<adg::SystemServiceEndpoint> sources;
  std::vector<adg::SystemServiceEndpoint> sinks;
  sources.push_back(
      take(test, system.addServiceEndpoint(host, initiateSet, carrier)));
  sinks.push_back(
      take(test, system.addServiceEndpoint(host, serveSet, carrier)));
  for (const auto &core : cores) {
    sources.push_back(
        take(test, system.addServiceEndpoint(core, initiateSet, carrier)));
    sinks.push_back(
        take(test, system.addServiceEndpoint(core, serveSet, carrier)));
  }

  constexpr std::uint32_t mappedTransportResidentRouteCapacity = 32;
  const auto transportContract = sharedTransportResourceContract(
      test, mappedTransportResidentRouteCapacity);
  const std::array<std::vector<std::uint32_t>, 3> patterns = {
      std::vector<std::uint32_t>{0}, std::vector<std::uint32_t>{1},
      std::vector<std::uint32_t>{0, 1}};
  std::vector<adg::SystemTransportResource> routers;
  routers.reserve(sources.size());
  std::vector<adg::HardwareDomainMember> clockMembers{host.domainMember()};
  for (const auto &core : cores) {
    clockMembers.push_back(core.instructionCoreDomainMember());
    clockMembers.push_back(core.spatialCoreDomainMember());
  }
  for (std::size_t ordinal = 0; ordinal != sources.size(); ++ordinal) {
    clockMembers.push_back(sources[ordinal].domainMember());
    clockMembers.push_back(sinks[ordinal].domainMember());
    routers.push_back(
        take(test,
             system.addTransportResource(
                 {{carrier, carrier}, {carrier, carrier}, transportContract})));
    clockMembers.push_back(routers.back().domainMember());
    for (std::size_t input = 0; input != 2; ++input)
      for (const auto &outputs : patterns) {
        const auto pattern = take(
            test, system.addTransferPattern(routers.back(), input, outputs, 0));
        clockMembers.push_back(pattern.domainMember());
      }
    requireSuccess(test,
                   system.connect(take(test, sources[ordinal].transport()),
                                  take(test, routers[ordinal].input(0))));
    requireSuccess(test,
                   system.connect(take(test, routers[ordinal].output(0)),
                                  take(test, sinks[ordinal].transport())));
  }
  for (std::size_t ordinal = 0; ordinal != routers.size(); ++ordinal)
    requireSuccess(
        test,
        system.connect(
            take(test, routers[ordinal].output(1)),
            take(test, routers[(ordinal + 1) % routers.size()].input(1))));

  if (spec.attachSystemMemory) {
    auto indexWidths = take(
        test, ::fabric::UnsignedDomain::fromCanonical({{32, 32}, {64, 64}}));
    auto memory =
        take(test, adg::makeHybrid32SystemMemory(
                       {0, 4096,
                        adg::MemoryAccessDomainParameters{
                            128, std::nullopt, 4, std::move(indexWidths)},
                        128},
                       rate));
    auto memoryService = take(test, system.addMemoryService(memory.contract));
    auto memoryEndpoint = take(
        test, system.addServiceEndpoint(memoryService, memory.capabilities));
    clockMembers.push_back(memoryService.domainMember());
    clockMembers.push_back(memoryEndpoint.domainMember());

    const auto moduleTemplate = module.view().moduleRootTemplate();
    require(test, moduleTemplate.has_value(),
            "mapped memory fixture is not a Module root");
    const auto widestTransportOrdinal =
        [&](fabric::FabricPortDirection direction) {
          std::optional<std::pair<std::size_t, std::uint32_t>> widest;
          std::size_t transportOrdinal = 0;
          const std::uint64_t boundaryCount =
              module.view().moduleBoundaryEndpointCount(*moduleTemplate,
                                                        direction);
          for (std::uint64_t ordinal = 0; ordinal != boundaryCount; ++ordinal) {
            const fabric::FabricModuleBoundaryEndpointRef boundary{
                *moduleTemplate, direction, ordinal};
            const auto plane =
                module.view().moduleBoundaryEndpointPlane(boundary);
            require(test, plane.has_value(),
                    "mapped memory boundary has no endpoint plane");
            if (*plane !=
                fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
              continue;
            const auto dataPath =
                module.view().moduleBoundaryEndpointDataPath(boundary);
            require(test, dataPath.has_value(),
                    "mapped memory transport boundary has no data path");
            if (!widest || dataPath->payloadWidthBits > widest->second)
              widest = std::pair{transportOrdinal, dataPath->payloadWidthBits};
            ++transportOrdinal;
          }
          require(test, widest.has_value(),
                  "mapped memory fixture has no service-leg carrier");
          return widest->first;
        };
    const std::size_t responseOrdinal =
        widestTransportOrdinal(fabric::FabricPortDirection::Input);
    const std::size_t requestOrdinal =
        widestTransportOrdinal(fabric::FabricPortDirection::Output);

    std::vector<adg::SystemTransportEndpoint> providerRequests;
    std::vector<adg::SystemTransportEndpoint> providerResponses;
    for (const adg::AccCore &core : cores) {
      auto manager = take(test, core.spatialMemoryManager(0));
      requireSuccess(test, system.attachSpatialMemory(manager, memoryEndpoint));
      auto transport =
          take(test, system.addTransportResource(
                         {{carrier}, {carrier}, transportContract}));
      auto pattern =
          take(test, system.addTransferPattern(transport, 0, {0}, 0));
      clockMembers.push_back(transport.domainMember());
      clockMembers.push_back(pattern.domainMember());
      auto request = take(test, core.spatialTransportOutput(requestOrdinal));
      auto response = take(test, core.spatialTransportInput(responseOrdinal));
      auto providerRequest = take(test, transport.input(0));
      auto providerResponse = take(test, transport.output(0));
      requireSuccess(test, system.connect(request, providerRequest));
      requireSuccess(test, system.connect(providerResponse, response));
      for (auto kind : {dataflow::semantics::ServiceKind::MemoryRead,
                        dataflow::semantics::ServiceKind::MemoryWrite}) {
        requireSuccess(
            test, system.attachServiceLegCarriers(manager, kind, 0, {request}));
        requireSuccess(test, system.attachServiceLegCarriers(manager, kind, 1,
                                                             {response}));
      }
      providerRequests.push_back(providerRequest);
      providerResponses.push_back(providerResponse);
    }
    auto subordinate = take(test, memoryEndpoint.memory());
    for (auto kind : {dataflow::semantics::ServiceKind::MemoryRead,
                      dataflow::semantics::ServiceKind::MemoryWrite}) {
      requireSuccess(test, system.attachServiceLegCarriers(subordinate, kind, 0,
                                                           providerRequests));
      requireSuccess(test, system.attachServiceLegCarriers(subordinate, kind, 1,
                                                           providerResponses));
    }
  }

  requireSuccess(
      test, clock.close(clockMembers,
                        take(test, fabric::ClockDomainContractRecord::create(
                                       1'000, 0))));
  auto reset = take(test, system.createHardwareDomain());
  std::vector<adg::HardwareDomainMember> resetMembers{host.domainMember()};
  for (const auto &core : cores) {
    resetMembers.push_back(core.instructionCoreDomainMember());
    resetMembers.push_back(core.spatialCoreResetDomainMember());
  }
  requireSuccess(
      test, reset.close(resetMembers,
                        take(test, fabric::ResetDomainContractRecord::create(
                                       fabric::ResetPolarity::ActiveHigh,
                                       fabric::ResetTiming::Asynchronous,
                                       fabric::ResetTiming::Asynchronous,
                                       fabric::ResetInitialState::Asserted,
                                       std::nullopt, 0))));
  requireSuccess(test, system.close());
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "system builder did not publish one root");
  return take(test, fabric::importEntireFabricRoot(
                        finalized.roots().front().reference(), artifacts));
}

dataflow::CanonicalDataflowArtifact buildDataflow(llvm::StringRef test,
                                                  ArtifactStore &artifacts) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  dataflow.thread private @worker_a domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %lhs = arith.constant 1 : i32
    %rhs = arith.constant 2 : i32
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.thread.yield
  }
  dataflow.thread private @worker_b domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %lhs = arith.constant 3 : i32
    %rhs = arith.constant 4 : i32
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.thread.yield
  }
  func.func private @host() {
    %completion_a = dataflow.thread.launch @worker_a()
        : () -> !dataflow.thread_token
    %completion_b = dataflow.thread.launch @worker_b()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  require(test, static_cast<bool>(module), "cannot parse Dataflow fixture");
  auto dataflow = take(test, dataflow::finalizeCanonicalDataflow(*module));
  (void)take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  return dataflow;
}

mapping::FinalizedSystemMapping buildSystemMapping(
    llvm::StringRef test,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FinalizedFabricRoot &system, ArtifactStore &artifacts,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings = {},
    llvm::ArrayRef<fabric::AccCoreOccurrenceRef> rootThreadTargets = {}) {
  auto systemView = take(test, fabric::requireSystemRoot(system.view()));
  require(test, !dataflow.rootThreadLaunches().empty(),
          "Dataflow fixture did not produce a root thread launch");
  require(test, !systemView.artifact().accCoreOccurrences().empty(),
          "System fixture has no AccCore occurrence");
  std::vector<dataflow::RootThreadLaunchRef> rootThreads;
  rootThreads.reserve(dataflow.rootThreadLaunches().size());
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    rootThreads.push_back(root.ref);
  mapping::FinalizedSystemMappingConstraintSet constraints = [&] {
    if (rootThreadTargets.empty())
      return take(test, mapping::finalizeEmptySystemMappingConstraintSet(
                            dataflow, systemView, rootThreads, artifacts));
    require(test, rootThreadTargets.size() == rootThreads.size(),
            "root thread target count does not match the root launch set");
    mlir::MLIRContext constraintContext;
    constraintContext.loadDialect<::mapping::MappingDialect>();
    mlir::OpBuilder builder(&constraintContext);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    const auto bytesAttr = [&](llvm::ArrayRef<std::uint8_t> bytes) {
      std::vector<std::int8_t> signedBytes;
      signedBytes.reserve(bytes.size());
      for (std::uint8_t byte : bytes)
        signedBytes.push_back(static_cast<std::int8_t>(byte));
      return mlir::DenseI8ArrayAttr::get(&constraintContext, signedBytes);
    };
    std::vector<mlir::Attribute> rootAttrs;
    rootAttrs.reserve(rootThreads.size());
    for (const auto root : rootThreads)
      rootAttrs.push_back(::mapping::RootThreadLaunchRefAttr::get(
          &constraintContext,
          bytesAttr(take(test, dataflow::encodeDataflowReference(
                                   dataflow.identity(), root)))));
    auto root = ::mapping::ConstraintsSystemOp::create(
        builder, builder.getUnknownLoc(),
        ::mapping::ArtifactIdentityAttr::get(
            &constraintContext, bytesAttr(dataflow.identity().bytes())),
        ::mapping::ArtifactIdentityAttr::get(
            &constraintContext,
            bytesAttr(systemView.artifact().identity().bytes())),
        builder.getArrayAttr(rootAttrs), builder.getArrayAttr({}));
    root.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&root.getBody().front());
    for (const auto [rootAttr, core] :
         llvm::zip_equal(rootAttrs, rootThreadTargets)) {
      mlir::OperationState restriction(
          builder.getUnknownLoc(),
          ::mapping::ConstraintDomainRestrictionOp::getOperationName());
      restriction.addAttribute(
          "projection",
          ::mapping::SystemConstraintProjectionKeyAttr::get(
              &constraintContext,
              static_cast<std::uint32_t>(
                  ::mapping::SystemConstraintProjection::ThreadTargetAccCore)));
      restriction.addAttribute("subject", rootAttr);
      restriction.addAttribute(
          "admissible_domain",
          builder.getArrayAttr({::mapping::FabricAccCoreOccurrenceRefAttr::get(
              &constraintContext,
              bytesAttr(fabric::canonicalFabricBytes(core)))}));
      builder.create(restriction);
    }
    return take(test, mapping::finalizeSystemMappingConstraintSet(
                          root, dataflow, systemView, artifacts));
  }();
  const auto partition =
      take(test, pnr::projectWholeDomainPresburgerPartitionPlan(
                     dataflow, constraints.view().rootThreadLaunches()));
  ResolvedConfig resolved = defaultResolvedConfig();
  auto &search = resolved.dse.systemPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.routing.negotiationIterationLimit = 8;
  search.actionProposal = {1, 0, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {ResolvedPnrExactRepairKind::Disabled, 0, 0};
  const auto config =
      take(test, pnr::projectResolvedSystemPnrConfigView(resolved));
  const auto searchDomain =
      take(test, pnr::projectSystemPnrSearchDomain(
                     dataflow, systemView, config, constraints, partition,
                     pnr::SystemHierarchicalGraphSearchInput{
                         std::vector<ArtifactRootReference>(
                             spatialMappings.begin(), spatialMappings.end())},
                     artifacts));
  const auto physicalTiming = take(
      test, fabric::projectNormalizedSystemPhysicalTimingProfiles(systemView));
  auto outcome = pnr::generateSystemMappings({dataflow, systemView,
                                              physicalTiming, searchDomain,
                                              config, constraints, artifacts});
  const auto *generated = std::get_if<pnr::GeneratedSystemMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1) {
    const std::string diagnostic = std::visit(
        [](const auto &result) {
          using Outcome = std::decay_t<decltype(result)>;
          if constexpr (std::is_same_v<Outcome, pnr::GeneratedSystemMappings>)
            return std::string("unexpected candidate count");
          else if constexpr (std::is_same_v<
                                 Outcome, pnr::InterruptedSystemPnrGeneration>)
            return (llvm::Twine("interrupted at ") +
                    pnr::systemPnrInterruptionStageSpelling(
                        result.snapshot.stage))
                .str();
          else
            return result.diagnostic;
        },
        outcome);
    fail(test, "System PnR did not produce one candidate: " + diagnostic);
  }
  return take(test, mapping::importSystemMapping(generated->candidates.front(),
                                                 artifacts));
}

CompilerTargetPolicy targetPolicy() {
  return {fabric::RiscVAbi::Lp64d,
          fabric::RiscVCodeModel::MediumAny,
          fabric::RelocationModel::Static,
          "generic-rv64",
          {}};
}

void writeBytes(llvm::StringRef test, llvm::StringRef path,
                llvm::ArrayRef<std::uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    fail(test, error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error())
    fail(test, "cannot write host object");
}

std::vector<std::uint8_t> linkedExecutable(llvm::StringRef test,
                                           const CompilerTargetBinding &target,
                                           const TemporaryTree &tree,
                                           llvm::StringRef stem,
                                           llvm::StringRef entrySymbol) {
  llvm::LLVMContext llvmContext;
  auto module = std::make_unique<llvm::Module>(stem, llvmContext);
  module->setTargetTriple(llvm::Triple(target.targetTriple()));
  module->setDataLayout(target.dataLayout());
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), false),
      llvm::GlobalValue::ExternalLinkage, entrySymbol, *module);
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(llvmContext, "entry", entry));
  builder.CreateRetVoid();
  const std::vector<std::uint8_t> object =
      take(test, emitCompilerTargetObject(std::move(module), target));
  const std::string objectPath = tree.path((stem + ".o").str());
  const std::string executablePath = tree.path((stem + ".elf").str());
  writeBytes(test, objectPath, object);
  const std::string entryArgument = ("--entry=" + entrySymbol).str();
  const llvm::SmallVector<llvm::StringRef, 12> arguments = {
      LOOM_TEST_LLD_PATH,
      "-m",
      "elf64lriscv",
      entryArgument,
      "-Ttext=0x10000",
      "--no-dynamic-linker",
      "-o",
      executablePath,
      objectPath};
  std::string error;
  bool executionFailed = false;
  const int result =
      llvm::sys::ExecuteAndWait(LOOM_TEST_LLD_PATH, arguments, std::nullopt, {},
                                30, 1024, &error, &executionFailed);
  require(test, !executionFailed && result == 0,
          "ld.lld did not produce the host executable");
  auto buffer = llvm::MemoryBuffer::getFile(executablePath, false, false);
  if (!buffer)
    fail(test, buffer.getError().message());
  return std::vector<std::uint8_t>((*buffer)->getBuffer().bytes_begin(),
                                   (*buffer)->getBuffer().bytes_end());
}

std::unique_ptr<llvm::Module> linkedModule(llvm::LLVMContext &llvmContext,
                                           const CompilerTargetBinding &target,
                                           llvm::StringRef stem,
                                           llvm::StringRef entrySymbol) {
  auto module = std::make_unique<llvm::Module>(stem, llvmContext);
  module->setTargetTriple(llvm::Triple(target.targetTriple()));
  module->setDataLayout(target.dataLayout());
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), false),
      llvm::GlobalValue::ExternalLinkage, entrySymbol, *module);
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(llvmContext, "entry", entry));
  builder.CreateRetVoid();
  return module;
}

std::vector<std::uint8_t> hostExecutable(llvm::StringRef test,
                                         const CompilerTargetBinding &target,
                                         const TemporaryTree &tree) {
  return linkedExecutable(test, target, tree, "host", "loom_host_entry");
}

hardware::ImplementationRepresentationRoot
representation(llvm::StringRef test, BlobStore &blobs,
               std::size_t configurationPortCount) {
  std::string rtl = "module top(";
  for (std::size_t index = 0; index != configurationPortCount; ++index) {
    if (index != 0)
      rtl += ", ";
    rtl += "input logic cfg_" + std::to_string(index);
  }
  rtl += "); endmodule\n";
  const BlobDigest source = take(
      test,
      blobs.put(llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(rtl.data()), rtl.size())));
  const auto format =
      take(test, hardware::RepresentationFormatDescriptorRef::get(
                     hardware::RepresentationFormatKind::SystemVerilogRtl));
  return take(test,
              hardware::createImplementationRepresentationRoot(
                  hardware::RepresentationRootVariant::Rtl, std::nullopt,
                  format, {hardware::RepresentationObjectKind::Module, "top"},
                  {{hardware::PayloadRole::RtlSource, "rtl/top.sv", source}}));
}

} // namespace

TemporaryTree::TemporaryTree(llvm::StringRef label) {
  llvm::SmallString<128> root;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          ("loom-deployment-" + label).str(), root))
    fail(label, error.message());
  root_ = root.str().str();
  std::filesystem::create_directories(path("artifacts"));
  std::filesystem::create_directories(path("blobs"));
}

TemporaryTree::~TemporaryTree() { std::filesystem::remove_all(root_); }

std::string TemporaryTree::path(llvm::StringRef leaf) const {
  llvm::SmallString<256> result(root_);
  llvm::sys::path::append(result, leaf);
  return result.str().str();
}

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

llvm::Expected<FinalizedDeployment> tryBuildMinimalDeploymentImpl(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree, llvm::StringRef finalLinkedTriple,
    bool trustedIdentity, bool shareProgrammingEndpoint,
    bool systemArtifactInterfaces, bool reverseRootTargets,
    const runtime::RuntimeProviderDescriptor &runtimeProvider) {
  const auto module = buildModule(test, artifacts);
  const auto system = buildSystem(test, module, artifacts);
  auto dataflowArtifact = buildDataflow(test, artifacts);
  auto dataflow = take(test, dataflowArtifact.view());
  auto systemView = take(test, fabric::requireSystemRoot(system.view()));
  const auto cores = systemView.artifact().accCoreOccurrences();
  require(test, cores.size() >= dataflow.rootThreadLaunches().size(),
          "minimal fixture has fewer AccCores than root threads");
  std::vector<fabric::AccCoreOccurrenceRef> rootTargets;
  rootTargets.reserve(dataflow.rootThreadLaunches().size());
  if (reverseRootTargets)
    rootTargets.insert(rootTargets.end(),
                       cores.end() - dataflow.rootThreadLaunches().size(),
                       cores.end());
  else
    rootTargets.insert(rootTargets.end(), cores.begin(),
                       cores.begin() + dataflow.rootThreadLaunches().size());
  if (reverseRootTargets)
    std::reverse(rootTargets.begin(), rootTargets.end());
  const auto systemMapping =
      buildSystemMapping(test, dataflow, system, artifacts, {}, rootTargets);
  auto abiDraft = take(
      test, hardware::derivePackedConfigurationABIDraft(system, context()));
  const auto abi = take(
      test, hardware::finalizeConfigurationABI(std::move(abiDraft), artifacts));
  require(test, abi.abi().programmingUnits().size() == 2,
          "fixture did not produce one programming unit per SpatialCore");
  requireSuccess(test, runtime::registerRuntimeProvider(runtimeProvider));
  std::optional<BlobDigest> trustedAttestation;
  if (trustedIdentity) {
    constexpr llvm::StringLiteral attestation =
        "deployment trusted implementation";
    trustedAttestation = take(
        test, blobs.put(llvm::ArrayRef<std::uint8_t>(
                  reinterpret_cast<const std::uint8_t *>(attestation.data()),
                  attestation.size())));
  }
  std::vector<DeploymentHardwareBinding> hardwareBindings;
  hardwareBindings.reserve(rootTargets.size());
  for (const auto indexedCore : llvm::enumerate(rootTargets)) {
    const fabric::SpatialCoreOccurrenceRef subject{indexedCore.value()};
    std::vector<hardware::ImplementationInterface> interfaces;
    for (const hardware::ProgrammingUnit &unit : abi.abi().programmingUnits()) {
      const hardware::ProgrammingUnitOccurrenceScope scope =
          hardware::deriveProgrammingUnitOccurrenceScope(unit);
      if (scope.includesDirectSystemResources || scope.spatialCores.size() != 1 ||
          scope.spatialCores.front() != subject)
        continue;
      interfaces.push_back(
          {hardware::ImplementationConfigurationInterfaceRef{
               hardware::ProgrammingUnitRef{abi.reference(), unit.id}},
           {hardware::RepresentationObjectKind::Port,
            "top.cfg_" + std::to_string(interfaces.size())},
           std::nullopt});
    }
    const auto implementation =
        take(test, hardware::finalizeHardwareImplementation(
                       hardware::HardwareImplementationDraft{
                           system.reference(), subject, abi.reference(),
                           representation(test, blobs, interfaces.size()),
                           std::nullopt, std::move(interfaces), {}, {}, {}},
                       artifacts, blobs));
    std::vector<runtime::RuntimeProgrammingBinding> programmingBindings;
    for (const auto indexedInterface :
         llvm::enumerate(implementation.implementation().interfaces())) {
      const auto *configuration =
          std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
              &indexedInterface.value().semanticRef);
      require(test, configuration != nullptr,
              "fixture implementation has a non-configuration interface");
      programmingBindings.push_back(
          {configuration->programmingUnit,
           {implementation.reference().artifact,
            hardware::HardwareImplementationInterfaceRef{
                indexedInterface.index()}},
           runtime::inProcessRuntimeEndpoint(
               runtime::RuntimeEndpointClass::Programming,
               shareProgrammingEndpoint
                   ? 0
                   : configuration->programmingUnit.unitId)});
    }
    runtime::RuntimeIdentityVerification identityVerification =
        trustedAttestation
            ? runtime::RuntimeIdentityVerification(
                  runtime::TrustedImmutableIdentity{*trustedAttestation})
            : runtime::RuntimeIdentityVerification(
                  runtime::HardwareReportedIdentity{
                      runtime::inProcessRuntimeEndpoint(
                          runtime::RuntimeEndpointClass::Identity,
                          indexedCore.value().id())});
    const auto runtimeBinding =
        take(test, runtime::finalizeRuntimePlatformBinding(
                       runtime::RuntimePlatformBindingDraft{
                           implementation.reference(),
                           runtime::runtimeProviderDescriptorRef(
                               runtimeProvider),
                           std::move(identityVerification),
                           std::move(programmingBindings), {}, {}},
                       artifacts, blobs));
    hardwareBindings.push_back(
        {implementation.reference(), runtimeBinding.reference()});
  }
  auto targets = take(test, resolveSystemCompilerTargetBindings(
                                system, targetPolicy(), artifacts));
  require(test, targets.instructionGroups().size() == 1,
          "fixture did not resolve one InstructionCore target group");
  auto hostTarget = take(
      test, importCompilerTargetBinding(targets.host().reference(), artifacts));
  const auto &instructionTarget = targets.instructionGroups().front().binding();
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  std::vector<ThreadEntryBinding> threadEntries;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    threadEntries.push_back({root.ref, 0});
  const auto instructionBinary =
      take(test, finalizeInstructionCoreBinary(
                     {dataflowReference,
                      instructionTarget.reference(),
                      linkedExecutable(test, instructionTarget.binding(), tree,
                                       "instruction", "__loom_thread_entry_0"),
                      std::move(threadEntries),
                      {}},
                     artifacts, blobs));
  HostCatalog hostCatalog;
  if (systemArtifactInterfaces)
    hostCatalog = systemArtifactHostCatalog(test);
  else
    hostCatalog.entries = {{0, "loom_host_entry", {}, {}, {}}};
  const HostProgramLeaf host = take(
      test,
      finalizeHostProgramLeaf(
          HostProgramLeafDraft{hostTarget.reference(),
                               hostExecutable(test, hostTarget.binding(), tree),
                               std::move(hostCatalog.entries),
                               std::move(hostCatalog.interfaces),
                               {}},
          artifacts, blobs));
  llvm::LLVMContext linkedContext;
  auto finalLinkedModule = linkedModule(linkedContext, hostTarget.binding(),
                                        "linked-host", "loom_host_entry");
  if (!finalLinkedTriple.empty())
    finalLinkedModule->setTargetTriple(llvm::Triple(finalLinkedTriple));
  return buildDeploymentFromLinkedProgram(
      DeploymentPipelineInputs{
          systemMapping.reference(),
          host,
          {instructionBinary.reference()},
          std::move(hardwareBindings)},
      *finalLinkedModule, artifacts, blobs);
}

FinalizedDeployment buildMappedSpatialDeployment(
    llvm::StringRef test,
    const dataflow::CanonicalDataflowArtifact &dataflowArtifact,
    const fabric::FinalizedFabricRoot &system,
    const mapping::FinalizedSpatialMapping &spatialMapping,
    llvm::ArrayRef<hardware::FinalizedHardwareImplementation> implementations,
    ArtifactStore &artifacts, BlobStore &blobs, const TemporaryTree &tree) {
  auto dataflow = take(test, dataflowArtifact.view());
  const auto systemMapping = buildSystemMapping(
      test, dataflow, system, artifacts, {spatialMapping.reference()});
  return buildMappedSystemDeployment(test, dataflowArtifact, system,
                                     systemMapping, implementations, {},
                                     artifacts, blobs, tree);
}

mapping::FinalizedSystemMapping buildMappedSystemMapping(
    llvm::StringRef test,
    const dataflow::CanonicalDataflowArtifact &dataflowArtifact,
    const fabric::FinalizedFabricRoot &system,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    ArtifactStore &artifacts,
    llvm::ArrayRef<fabric::AccCoreOccurrenceRef> rootThreadTargets) {
  auto dataflow = take(test, dataflowArtifact.view());
  return buildSystemMapping(test, dataflow, system, artifacts, spatialMappings,
                            rootThreadTargets);
}

FinalizedDeployment buildMappedSystemDeployment(
    llvm::StringRef test,
    const dataflow::CanonicalDataflowArtifact &dataflowArtifact,
    const fabric::FinalizedFabricRoot &system,
    const mapping::FinalizedSystemMapping &systemMapping,
    llvm::ArrayRef<hardware::FinalizedHardwareImplementation> implementations,
    MappedSystemExecutablePrograms programs, ArtifactStore &artifacts,
    BlobStore &blobs, const TemporaryTree &tree) {
  auto dataflow = take(test, dataflowArtifact.view());
  require(test, !dataflow.rootThreadLaunches().empty(),
          "mapped fixture did not contain a root thread launch");
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, dataflow.identity()};

  auto requiredSubjects = take(
      test, mapping::projectSystemExecutionSpatialCoreSubjects(
                dataflow, systemMapping.view().executionBindings()));
  std::vector<const hardware::FinalizedHardwareImplementation *>
      selectedImplementations;
  selectedImplementations.reserve(requiredSubjects.size());
  for (fabric::SpatialCoreOccurrenceRef subject : requiredSubjects) {
    const hardware::FinalizedHardwareImplementation *selected = nullptr;
    for (const hardware::FinalizedHardwareImplementation &implementation :
         implementations) {
      if (implementation.implementation().subject() != subject)
        continue;
      require(test, selected == nullptr,
              "mapped fixture repeats a SpatialCore implementation");
      selected = &implementation;
    }
    require(test, selected != nullptr,
            "mapped fixture omits a required SpatialCore implementation");
    selectedImplementations.push_back(selected);
  }

  const runtime::RuntimeProviderDescriptor &provider =
      runtime::inProcessRuntimeProviderDescriptor();
  requireSuccess(test, runtime::registerRuntimeProvider(provider));
  std::vector<DeploymentHardwareBinding> hardwareBindings;
  hardwareBindings.reserve(selectedImplementations.size());
  std::uint64_t programmingOrdinal = 0;
  std::uint64_t memoryOrdinal = 0;
  std::uint64_t completionOrdinal = 0;
  for (const auto [implementationOrdinal, selected] :
       llvm::enumerate(selectedImplementations)) {
    const hardware::FinalizedHardwareImplementation &implementation =
        *selected;
    std::vector<runtime::RuntimeProgrammingBinding> programmingBindings;
    std::vector<runtime::RuntimeInterfaceBinding> memoryBindings;
    std::vector<runtime::RuntimeInterfaceBinding> completionBindings;
    for (const auto &[interfaceOrdinal, interface] :
         llvm::enumerate(implementation.implementation().interfaces())) {
      const ArtifactReference<hardware::HardwareImplementationInterfaceRef>
          reference{
              implementation.reference().artifact,
              hardware::HardwareImplementationInterfaceRef{interfaceOrdinal}};
      if (const auto *configuration =
              std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
                  &interface.semanticRef)) {
        programmingBindings.push_back(
            {configuration->programmingUnit, reference,
             runtime::inProcessRuntimeEndpoint(
                 runtime::RuntimeEndpointClass::Programming,
                 programmingOrdinal++)});
      } else if (std::holds_alternative<
                     hardware::ImplementationMemoryInterfaceRef>(
                     interface.semanticRef)) {
        memoryBindings.push_back(
            {reference,
             runtime::inProcessRuntimeEndpoint(
                 runtime::RuntimeEndpointClass::Memory, memoryOrdinal++)});
      } else if (std::holds_alternative<
                     hardware::ImplementationDataInterfaceRef>(
                     interface.semanticRef) ||
                 std::holds_alternative<
                     hardware::ImplementationExternalProtocolInterfaceRef>(
                     interface.semanticRef)) {
        completionBindings.push_back(
            {reference, runtime::inProcessRuntimeEndpoint(
                            runtime::RuntimeEndpointClass::Completion,
                            completionOrdinal++)});
      }
    }
    const auto runtimeBinding = take(
        test, runtime::finalizeRuntimePlatformBinding(
                  runtime::RuntimePlatformBindingDraft{
                      implementation.reference(),
                      runtime::runtimeProviderDescriptorRef(provider),
                      runtime::HardwareReportedIdentity{
                          runtime::inProcessRuntimeEndpoint(
                              runtime::RuntimeEndpointClass::Identity,
                              implementationOrdinal)},
                      std::move(programmingBindings),
                      std::move(memoryBindings),
                      std::move(completionBindings)},
                  artifacts, blobs));
    hardwareBindings.push_back(
        {implementation.reference(), runtimeBinding.reference()});
  }

  auto targets = take(test, resolveSystemCompilerTargetBindings(
                                system, targetPolicy(), artifacts));
  require(test, targets.instructionGroups().size() == 1,
          "mapped fixture did not resolve one InstructionCore target group");
  auto hostTarget = take(
      test, importCompilerTargetBinding(targets.host().reference(), artifacts));
  const auto &instructionTarget = targets.instructionGroups().front().binding();
  if (programs.instructionProgramBytes.empty())
    programs.instructionProgramBytes =
        linkedExecutable(test, instructionTarget.binding(), tree, "instruction",
                         programs.instructionEntrySymbol);
  std::vector<ThreadEntryBinding> threadEntries;
  threadEntries.reserve(dataflow.rootThreadLaunches().size());
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    threadEntries.push_back({root.ref, 0});
  const auto instructionBinary =
      take(test, finalizeInstructionCoreBinary(
                     {dataflowReference,
                      instructionTarget.reference(),
                      std::move(programs.instructionProgramBytes),
                      std::move(threadEntries),
                      {}},
                     artifacts, blobs));
  if (programs.hostProgramBytes.empty())
    programs.hostProgramBytes =
        hostExecutable(test, hostTarget.binding(), tree);
  if (programs.hostEntries.empty())
    programs.hostEntries = {{0, "loom_host_entry", {}, {}, {}}};
  require(test, programs.hostEntries.size() == 1,
          "mapped fixture requires one host program entry");
  const std::string hostEntrySymbol = programs.hostEntries.front().abiSymbol;
  const HostProgramLeaf host =
      take(test, finalizeHostProgramLeaf(
                     HostProgramLeafDraft{hostTarget.reference(),
                                          std::move(programs.hostProgramBytes),
                                          std::move(programs.hostEntries),
                                          std::move(programs.hostInterfaces),
                                          {}},
                     artifacts, blobs));
  llvm::LLVMContext linkedContext;
  auto finalLinkedModule = linkedModule(linkedContext, hostTarget.binding(),
                                        "linked-host", hostEntrySymbol);
  auto deployment =
      take(test, buildDeploymentFromLinkedProgram(
                     DeploymentPipelineInputs{systemMapping.reference(),
                                              host,
                                              {instructionBinary.reference()},
                                              std::move(hardwareBindings)},
                     *finalLinkedModule, artifacts, blobs));
  return deployment;
}

fabric::FinalizedFabricRoot buildMappedSpatialSystem(
    llvm::StringRef test, const fabric::FinalizedFabricRoot &module,
    llvm::ArrayRef<mlir::Type> messagePayloads, const ArtifactStore &artifacts,
    bool attachSystemMemory) {
  return buildMappedSpatialSystem(
      test, module, messagePayloads, artifacts,
      MappedSpatialSystemSpec{2, false, attachSystemMemory});
}

fabric::FinalizedFabricRoot buildMappedSpatialSystem(
    llvm::StringRef test, const fabric::FinalizedFabricRoot &module,
    llvm::ArrayRef<mlir::Type> messagePayloads, const ArtifactStore &artifacts,
    MappedSpatialSystemSpec spec) {
  return buildSystem(test, module, artifacts, messagePayloads, spec);
}

llvm::Expected<FinalizedDeployment>
tryBuildMinimalDeployment(llvm::StringRef test, ArtifactStore &artifacts,
                          BlobStore &blobs, const TemporaryTree &tree,
                          llvm::StringRef finalLinkedTriple) {
  return tryBuildMinimalDeploymentImpl(
      test, artifacts, blobs, tree, finalLinkedTriple, false, false, false,
      false, runtime::inProcessRuntimeProviderDescriptor());
}

FinalizedDeployment buildMinimalDeployment(llvm::StringRef test,
                                           ArtifactStore &artifacts,
                                           BlobStore &blobs,
                                           const TemporaryTree &tree) {
  return take(test, tryBuildMinimalDeployment(test, artifacts, blobs, tree,
                                              llvm::StringRef()));
}

FinalizedDeployment
buildRetargetedMinimalDeployment(llvm::StringRef test, ArtifactStore &artifacts,
                                 BlobStore &blobs, const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), false, false,
                  false, true, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildRetargetedSharedProgrammingEndpointDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), false, true,
                  false, true, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildSystemArtifactDeployment(llvm::StringRef test,
                                                  ArtifactStore &artifacts,
                                                  BlobStore &blobs,
                                                  const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), false, false,
                  true, false, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildTrustedIdentityDeployment(llvm::StringRef test,
                                                   ArtifactStore &artifacts,
                                                   BlobStore &blobs,
                                                   const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), true, false,
                  false, false, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildSharedProgrammingEndpointDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), false, true,
                  false, false, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildRuntimeProviderDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree,
    const runtime::RuntimeProviderDescriptor &provider) {
  return take(test, tryBuildMinimalDeploymentImpl(
                        test, artifacts, blobs, tree, llvm::StringRef(), false,
                        false, false, false, provider));
}

} // namespace loom::deployment::test
