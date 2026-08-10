#include "DeploymentTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/DeploymentPipeline.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/Implementation/RepresentationFormat.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
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
            const ArtifactStore &artifacts) {
  adg::DesignBuilder design(artifacts);
  auto system = take(test, design.createSystem("deployment-system"));
  auto imported = take(test, system.importSpatialCore(module));
  const auto architecture =
      take(test, adg::getBuiltinInstructionCoreArchitecture());
  const auto micro = microarchitecture(test);
  auto host = take(test, system.addHostCore(architecture, micro));
  std::vector<adg::AccCore> cores;
  for (std::uint64_t ordinal = 0; ordinal != 2; ++ordinal) {
    cores.push_back(
        take(test, system.addAccCore(architecture, micro, imported)));
  }

  auto clock = take(test, system.createHardwareDomain());
  const auto rate = take(
      test,
      system.createServiceRate(
          clock, 1, 1, 1,
          fabric::ServiceProgress(std::in_place_type<::fabric::FairEventual>)));
  const auto messageDomain =
      take(test, fabric::MessageTransferCapabilityDomain::create(
                     {mlir::NoneType::get(&context())}));
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

  const auto transportContract = sharedTransportResourceContract(test, 16);
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

  requireSuccess(
      test, clock.close(clockMembers,
                        take(test, fabric::ClockDomainContractRecord::create(
                                       1'000, 0))));
  auto reset = take(test, system.createHardwareDomain());
  std::vector<adg::HardwareDomainMember> resetMembers{host.domainMember()};
  for (const auto &core : cores) {
    resetMembers.push_back(core.instructionCoreDomainMember());
    resetMembers.push_back(core.spatialCoreDomainMember());
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
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  func.func private @host() {
    %completion = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  require(test, static_cast<bool>(module), "cannot parse Dataflow fixture");
  auto dataflow = take(test, dataflow::finalizeCanonicalDataflow(*module));
  (void)take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  return dataflow;
}

mapping::FinalizedSystemMapping
buildSystemMapping(llvm::StringRef test,
                   const dataflow::CanonicalDataflowProgramView &dataflow,
                   const fabric::FinalizedFabricRoot &system,
                   ArtifactStore &artifacts) {
  auto systemView = take(test, fabric::requireSystemRoot(system.view()));
  require(test, dataflow.rootThreadLaunches().size() == 1,
          "Dataflow fixture did not produce one root thread launch");
  require(test, !systemView.artifact().accCoreOccurrences().empty(),
          "System fixture has no AccCore occurrence");
  const dataflow::RootThreadLaunchRef rootThread =
      dataflow.rootThreadLaunches().front().ref;
  auto constraints =
      take(test, mapping::finalizeEmptySystemMappingConstraintSet(
                     dataflow, systemView, {rootThread}, artifacts));
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
                     pnr::SystemHierarchicalGraphSearchInput{{}}, artifacts));
  auto outcome = pnr::generateSystemMappings(
      {dataflow, systemView, searchDomain, config, constraints, artifacts});
  const auto *generated = std::get_if<pnr::GeneratedSystemMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1) {
    const std::string diagnostic = std::visit(
        [](const auto &result) {
          using Outcome = std::decay_t<decltype(result)>;
          if constexpr (std::is_same_v<Outcome, pnr::GeneratedSystemMappings>)
            return std::string("unexpected candidate count");
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
    bool systemArtifactInterfaces,
    const runtime::RuntimeProviderDescriptor &runtimeProvider) {
  const auto module = buildModule(test, artifacts);
  const auto system = buildSystem(test, module, artifacts);
  auto dataflowArtifact = buildDataflow(test, artifacts);
  auto dataflow = take(test, dataflowArtifact.view());
  const auto systemMapping =
      buildSystemMapping(test, dataflow, system, artifacts);
  auto abiDraft = take(
      test, hardware::derivePackedConfigurationABIDraft(system, context()));
  const auto abi = take(
      test, hardware::finalizeConfigurationABI(std::move(abiDraft), artifacts));
  require(test, abi.abi().programmingUnits().size() == 2,
          "fixture did not produce one programming unit per SpatialCore");
  std::vector<hardware::ImplementationInterface> interfaces;
  interfaces.reserve(abi.abi().programmingUnits().size());
  for (std::size_t index = 0; index != abi.abi().programmingUnits().size();
       ++index) {
    const hardware::ProgrammingUnit &unit = abi.abi().programmingUnits()[index];
    interfaces.push_back(
        {hardware::ImplementationConfigurationInterfaceRef{
             hardware::ProgrammingUnitRef{abi.reference(), unit.id}},
         {hardware::RepresentationObjectKind::Port,
          "top.cfg_" + std::to_string(index)},
         std::nullopt});
  }
  const auto implementation =
      take(test, hardware::finalizeHardwareImplementation(
                     hardware::HardwareImplementationDraft{
                         system.reference(),
                         abi.reference(),
                         {},
                         representation(test, blobs, interfaces.size()),
                         std::nullopt,
                         std::move(interfaces),
                         {},
                         {},
                         {}},
                     artifacts, blobs));
  requireSuccess(test, runtime::registerRuntimeProvider(runtimeProvider));
  std::vector<runtime::RuntimeProgrammingBinding> programmingBindings;
  for (std::size_t index = 0;
       index != implementation.implementation().interfaces().size(); ++index) {
    const auto &interface = implementation.implementation().interfaces()[index];
    const auto *configuration =
        std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
            &interface.semanticRef);
    require(test, configuration != nullptr,
            "fixture implementation has a non-configuration interface");
    programmingBindings.push_back(
        {configuration->programmingUnit,
         {implementation.reference().artifact,
          hardware::HardwareImplementationInterfaceRef{index}},
         runtime::inProcessRuntimeEndpoint(
             runtime::RuntimeEndpointClass::Programming,
             shareProgrammingEndpoint ? 0 : index)});
  }
  runtime::RuntimeIdentityVerification identityVerification =
      runtime::HardwareReportedIdentity{runtime::inProcessRuntimeEndpoint(
          runtime::RuntimeEndpointClass::Identity, 0)};
  if (trustedIdentity) {
    constexpr llvm::StringLiteral attestation =
        "deployment trusted implementation";
    identityVerification = runtime::TrustedImmutableIdentity{take(
        test, blobs.put(llvm::ArrayRef<std::uint8_t>(
                  reinterpret_cast<const std::uint8_t *>(attestation.data()),
                  attestation.size())))};
  }
  const auto runtimeBinding =
      take(test, runtime::finalizeRuntimePlatformBinding(
                     runtime::RuntimePlatformBindingDraft{
                         implementation.reference(),
                         runtime::runtimeProviderDescriptorRef(runtimeProvider),
                         std::move(identityVerification),
                         std::move(programmingBindings),
                         {},
                         {}},
                     artifacts, blobs));
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
  const auto instructionBinary =
      take(test, finalizeInstructionCoreBinary(
                     {dataflowReference,
                      instructionTarget.reference(),
                      linkedExecutable(test, instructionTarget.binding(), tree,
                                       "instruction", "__loom_thread_entry_0"),
                      {{dataflow.rootThreadLaunches().front().ref, 0}},
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
          {{implementation.reference(), runtimeBinding.reference()}}},
      *finalLinkedModule, artifacts, blobs);
}

llvm::Expected<FinalizedDeployment>
tryBuildMinimalDeployment(llvm::StringRef test, ArtifactStore &artifacts,
                          BlobStore &blobs, const TemporaryTree &tree,
                          llvm::StringRef finalLinkedTriple) {
  return tryBuildMinimalDeploymentImpl(
      test, artifacts, blobs, tree, finalLinkedTriple, false, false, false,
      runtime::inProcessRuntimeProviderDescriptor());
}

FinalizedDeployment buildMinimalDeployment(llvm::StringRef test,
                                           ArtifactStore &artifacts,
                                           BlobStore &blobs,
                                           const TemporaryTree &tree) {
  return take(test, tryBuildMinimalDeployment(test, artifacts, blobs, tree,
                                              llvm::StringRef()));
}

FinalizedDeployment buildSystemArtifactDeployment(llvm::StringRef test,
                                                  ArtifactStore &artifacts,
                                                  BlobStore &blobs,
                                                  const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), false, false,
                  true, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildTrustedIdentityDeployment(llvm::StringRef test,
                                                   ArtifactStore &artifacts,
                                                   BlobStore &blobs,
                                                   const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), true, false,
                  false, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildSharedProgrammingEndpointDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree) {
  return take(test,
              tryBuildMinimalDeploymentImpl(
                  test, artifacts, blobs, tree, llvm::StringRef(), false, true,
                  false, runtime::inProcessRuntimeProviderDescriptor()));
}

FinalizedDeployment buildRuntimeProviderDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree,
    const runtime::RuntimeProviderDescriptor &provider) {
  return take(test, tryBuildMinimalDeploymentImpl(test, artifacts, blobs, tree,
                                                  llvm::StringRef(), false,
                                                  false, false, provider));
}

} // namespace loom::deployment::test
