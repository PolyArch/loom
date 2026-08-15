#include "Hardware/RTL/CommonSkeleton.h"

#include "ConfigurationABITestSupport.h"
#include "ConfigurationTransportTestSupport.h"
#include "../InternalMemoryEdgeTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/MemoryLibrary.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricMemoryConfiguration.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

::fabric::UnsignedDomain singleton(std::uint64_t value) {
  return take(::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

loom::fabric::FinalizedFabricRoot
makeMemoryModule(const loom::ArtifactStore &store, bool temporal) {
  loom::adg::LocalMemoryParameters parameters;
  parameters.capacityBytes = 256;
  parameters.interface = {
      loom::adg::MemoryAccessDomainParameters{64, 128, 8, singleton(32)}, 64,
      32};
  if (temporal)
    parameters.temporal = loom::adg::TemporalMemoryParameters{2, 2};
  parameters.managerEndpoint = temporal;
  auto memory = take(loom::adg::makeGeneral64LocalMemory(parameters));
  auto bits128 = take(loom::adg::PortType::bits(128));
  std::vector<loom::adg::PortType> moduleInputs(memory.inputTypes().begin(),
                                                memory.inputTypes().end());
  std::vector<loom::adg::PortType> moduleOutputs(memory.outputTypes().begin(),
                                                 memory.outputTypes().end());
  moduleInputs.push_back(bits128);
  moduleOutputs.push_back(bits128);

  loom::adg::DesignBuilder design(store);
  auto spatial = take(
      design.createSpatialCore(temporal ? "temporal-memory" : "spatial-memory",
                               moduleInputs, moduleOutputs));
  std::vector<loom::adg::SpatialValue> inputs;
  for (std::size_t ordinal = 0; ordinal != memory.inputTypes().size();
       ++ordinal)
    inputs.push_back(take(spatial.input(ordinal)));
  auto outputs = take(spatial.addMemory(inputs, memory));
  std::vector<loom::adg::SpatialValue> moduleResults(outputs.values().begin(),
                                                     outputs.values().end());
  moduleResults.push_back(take(spatial.input(memory.inputTypes().size())));
  if (llvm::Error error = spatial.close(moduleResults))
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "memory fixture published the wrong root count");
  return std::move(finalized.roots().front());
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
makeInstructionCoreMicroarchitecture() {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      ::fabric::oneCycleElasticOperationResourceContract()};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 2, 1};
  return take(
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
}

loom::fabric::FinalizedFabricRoot
makeMemorySystem(const loom::fabric::FinalizedFabricRoot &module,
                 const loom::ArtifactStore &store, bool attachManager) {
  loom::adg::DesignBuilder design(store);
  auto system = take(design.createSystem("memory-hierarchy-system"));
  auto imported = take(system.importSpatialCore(module));
  auto architecture = take(loom::adg::getBuiltinInstructionCoreArchitecture());
  auto microarchitecture = makeInstructionCoreMicroarchitecture();
  auto host = take(system.addHostCore(architecture, microarchitecture));
  auto core =
      take(system.addAccCore(architecture, microarchitecture, imported));

  auto clock = take(system.createHardwareDomain());
  std::vector<loom::adg::HardwareDomainMember> clockMembers{
      host.domainMember(), core.instructionCoreDomainMember(),
      core.spatialCoreDomainMember()};
  if (attachManager) {
    auto rate = take(system.createServiceRate(
        clock, 1, 1, 1,
        loom::fabric::ServiceProgress(
            std::in_place_type<::fabric::FairEventual>)));
    auto indexWidths =
        take(::fabric::UnsignedDomain::fromCanonical({{32, 32}}));
    auto service = take(loom::adg::makeGeneral64SystemMemory(
        {0, 4096,
         loom::adg::MemoryAccessDomainParameters{64, 128, 8,
                                                 std::move(indexWidths)},
         32},
        std::move(rate)));
    auto memoryService = take(system.addMemoryService(service.contract));
    auto endpoint =
        take(system.addServiceEndpoint(memoryService, service.capabilities));
    auto manager = take(core.spatialMemoryManager(0));
    if (llvm::Error error = system.attachSpatialMemory(manager, endpoint))
      fail(llvm::toString(std::move(error)));
    auto bits128 = take(loom::adg::PortType::bits(128));
    auto transport = take(system.addTransportResource(
        {{bits128},
         {bits128},
         ::fabric::oneCycleElasticOperationResourceContract()}));
    auto pattern = take(system.addTransferPattern(transport, 0, {0}, 0));
    auto requestCarrier = take(transport.input(0));
    auto responseCarrier = take(transport.output(0));
    const auto moduleTemplate = module.view().moduleRootTemplate();
    require(moduleTemplate.has_value(), "memory fixture is not a Module root");
    auto transportCount = [&](loom::fabric::FabricPortDirection direction) {
      std::size_t count = 0;
      const std::uint64_t boundaryCount =
          module.view().moduleBoundaryEndpointCount(*moduleTemplate, direction);
      for (std::uint64_t ordinal = 0; ordinal != boundaryCount; ++ordinal) {
        loom::fabric::FabricModuleBoundaryEndpointRef boundary{
            *moduleTemplate, direction, ordinal};
        auto plane = module.view().moduleBoundaryEndpointPlane(boundary);
        require(plane.has_value(), "memory boundary has no endpoint plane");
        count +=
            *plane ==
            loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport;
      }
      return count;
    };
    const std::size_t inputCount =
        transportCount(loom::fabric::FabricPortDirection::Input);
    const std::size_t outputCount =
        transportCount(loom::fabric::FabricPortDirection::Output);
    require(inputCount != 0 && outputCount != 0,
            "memory fixture has no transport carrier boundary");
    auto managerRequest = take(core.spatialTransportOutput(outputCount - 1));
    auto managerResponse = take(core.spatialTransportInput(inputCount - 1));
    if (llvm::Error error = system.connect(managerRequest, requestCarrier))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = system.connect(responseCarrier, managerResponse))
      fail(llvm::toString(std::move(error)));
    auto provider = take(endpoint.memory());
    for (auto kind : {::dataflow::semantics::ServiceKind::MemoryRead,
                      ::dataflow::semantics::ServiceKind::MemoryWrite}) {
      if (llvm::Error error = system.attachServiceLegCarriers(manager, kind, 0,
                                                              {managerRequest}))
        fail(llvm::toString(std::move(error)));
      if (llvm::Error error = system.attachServiceLegCarriers(
              manager, kind, 1, {managerResponse}))
        fail(llvm::toString(std::move(error)));
      if (llvm::Error error = system.attachServiceLegCarriers(provider, kind, 0,
                                                              {requestCarrier}))
        fail(llvm::toString(std::move(error)));
      if (llvm::Error error = system.attachServiceLegCarriers(
              provider, kind, 1, {responseCarrier}))
        fail(llvm::toString(std::move(error)));
    }
    clockMembers.push_back(transport.domainMember());
    clockMembers.push_back(pattern.domainMember());
    clockMembers.push_back(memoryService.domainMember());
    clockMembers.push_back(endpoint.domainMember());
  }
  auto clockContract =
      take(loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = clock.close(clockMembers, std::move(clockContract)))
    fail(llvm::toString(std::move(error)));

  auto reset = take(system.createHardwareDomain());
  auto resetContract = take(loom::fabric::ResetDomainContractRecord::create(
      loom::fabric::ResetPolarity::ActiveHigh,
      loom::fabric::ResetTiming::Asynchronous,
      loom::fabric::ResetTiming::Asynchronous,
      loom::fabric::ResetInitialState::Asserted, std::nullopt, 0));
  if (llvm::Error error =
          reset.close({host.domainMember(), core.instructionCoreDomainMember(),
                       core.spatialCoreDomainMember()},
                      std::move(resetContract)))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "memory System published the wrong root count");
  return std::move(finalized.roots().front());
}

loom::hardware::FinalizedConfigurationABI
makeAbi(const loom::ArtifactStore &store,
        const loom::fabric::FinalizedFabricRoot &module,
        const loom::fabric::FinalizedFabricRoot &system,
        loom::fabric::SpatialCoreOccurrenceRef &spatialCore) {
  auto systemView = take(loom::fabric::requireSystemRoot(system.view()));
  require(systemView.artifact().accCoreOccurrences().size() == 1,
          "memory fixture System changed its core count");
  spatialCore = loom::fabric::SpatialCoreOccurrenceRef{
      systemView.artifact().accCoreOccurrences().front()};
  require(module.view().memoryOccurrences().size() == 1,
          "memory fixture changed its occurrence count");
  auto schema = take(module.view().memoryConfigurationSchema(
      module.view().memoryOccurrences().front()));
  auto target =
      take(loom::fabric::FabricModulePhysicalTargetRef::create(schema.field()));
  auto physical =
      take(loom::fabric::FabricPhysicalConfigurationFieldRef::create(
          loom::fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                                         std::move(target)}));
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides{
          {physical,
           loom::hardware::DirectBitsEncoding{schema.layout().carrierBitCount},
           std::vector<std::uint8_t>((schema.layout().carrierBitCount + 7) / 8,
                                     0)}};
  auto draft = take(loom::hardware::test::makeCompleteConfigurationABIDraft(
      system, overrides));
  return take(
      loom::hardware::finalizeConfigurationABI(std::move(draft), store));
}

struct MemoryActors final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *load = nullptr;
  mlir::Operation *store = nullptr;
};

MemoryActors makeMemoryActors(mlir::MLIRContext &context,
                              unsigned indexWidth) {
  std::string text;
  llvm::raw_string_ostream source(text);
  source << R"mlir(module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, )mlir"
         << indexWidth << R"mlir(>>} {
  func.func private @memory_actors(
      %memory: memref<64xi32>, %address: index, %data: i32, %ctrl: none) {
    %loaded, %load_done = dataflow.load %memory[%address] %ctrl
        : memref<64xi32>
    %store_done = dataflow.store %memory[%address] %data %ctrl
        : memref<64xi32>
    return
  }
}
)mlir";
  source.flush();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse memory actor fixture");
  MemoryActors result{std::move(module)};
  result.module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "dataflow.load")
      result.load = operation;
    else if (operation->getName().getStringRef() == "dataflow.store")
      result.store = operation;
  });
  require(result.load && result.store,
          "memory actor fixture omitted load or store");
  return result;
}

llvm::Expected<loom::fabric::FabricMemoryOperationRow> projectMemoryRow(
    const loom::fabric::FabricArtifactView &fabric,
    const loom::fabric::FabricMemoryConfigurationSchemaView &schema,
    loom::fabric::FabricOrdinal physicalPort, mlir::Operation *actor,
    ::fabric::MemoryDispatchTarget serviceTarget) {
  auto actorProjection =
      dataflow::projectRegisteredActorSchemaProjection(actor);
  if (!actorProjection)
    return actorProjection.takeError();
  auto access = dataflow::semantics::getCanonicalMemoryAccessView(actor);
  if (!access)
    return access.takeError();
  auto service = dataflow::semantics::CanonicalService::forActor(actor);
  if (!service)
    return service.takeError();
  const auto *port =
      fabric.memoryOperationPort(loom::fabric::FabricMemoryOperationPortRef{
          schema.memory(), physicalPort});
  if (!port)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "memory operation port does not resolve");
  const auto arguments = service->arguments();
  const auto results = service->results();
  const unsigned tagWidth = std::max(1U, schema.layout().tagWidthBits);
  std::string rejection;
  for (auto [capabilityOrdinal, capability] :
       llvm::enumerate(port->capabilityAlternatives())) {
    std::vector<std::optional<loom::fabric::FabricMemoryRoleSource>> sources(
        schema.layout().roleCount);
    std::vector<std::optional<loom::fabric::FabricMemoryRoleDestination>>
        destinations(schema.layout().roleCount);
    for (const auto &binding : capability.roleToEndpoint) {
      const auto argument = llvm::find_if(arguments, [&](const auto &value) {
        return value.role == binding.role;
      });
      if (argument != arguments.end())
        sources[static_cast<unsigned>(binding.role)] =
            loom::fabric::FabricMemoryExternalRoleSource{
                binding.endpointOrdinal, llvm::APInt(tagWidth, 0)};
      const auto result = llvm::find_if(results, [&](const auto &value) {
        return value.role == binding.role;
      });
      if (result != results.end())
        destinations[static_cast<unsigned>(binding.role)] =
            loom::fabric::FabricMemoryRoleDestination{
                loom::fabric::FabricMemoryExternalRoleSource{
                    binding.endpointOrdinal, llvm::APInt(tagWidth, 0)},
                {}};
    }
    for (::fabric::UsePatternKey pattern : capability.admissibleUsePatterns) {
      auto row = schema.projectOperationRow(
          physicalPort, capabilityOrdinal, pattern.ordinal(), *actorProjection,
          std::optional<dataflow::semantics::CanonicalMemoryAccessView>(
              *access),
          0, sources, destinations, serviceTarget);
      if (row)
        return std::move(*row);
      rejection = llvm::toString(row.takeError());
    }
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "memory actor has no compatible physical operation row: " + rejection);
}

llvm::Expected<loom::CanonicalSemanticBytes>
makeActiveMemoryConfiguration(const loom::fabric::FabricArtifactView &fabric,
                              loom::fabric::FabricMemoryOccurrenceRef memory,
                              bool temporal, mlir::MLIRContext &actorContext) {
  auto schema = fabric.memoryConfigurationSchema(memory);
  if (!schema)
    return schema.takeError();
  MemoryActors actors = makeMemoryActors(actorContext, 32);
  const ::fabric::MemoryDispatchTarget localTarget(
      std::in_place_type<::fabric::LocalMemoryDispatchTarget>);
  auto load = projectMemoryRow(fabric, *schema, 0, actors.load, localTarget);
  if (!load)
    return load.takeError();
  auto store = projectMemoryRow(fabric, *schema, 1, actors.store, localTarget);
  if (!store)
    return store.takeError();
  loom::fabric::FabricMemoryActive active;
  active.operationRows.resize(schema->layout().operationRows.size());
  active.providerDecodeRows.resize(schema->layout().providerRows.size());
  for (auto [ordinal, rows] : llvm::enumerate(schema->layout().providerRows))
    active.providerDecodeRows[ordinal].resize(rows.size());
  if (active.operationRows.size() < 2)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "memory fixture has fewer than two rows");
  if (temporal) {
    active.operationRows[0] = std::move(*load);
    active.operationRows[1] = std::move(*store);
  } else {
    active.operationRows[load->physicalPort] = std::move(*load);
    active.operationRows[store->physicalPort] = std::move(*store);
  }
  return schema->encode(
      loom::fabric::FabricMemoryConfigurationValue{std::move(active)});
}

llvm::Expected<loom::CanonicalSemanticBytes>
makeInternalMemoryConfiguration(
    const loom::fabric::FabricArtifactView &fabric,
    loom::fabric::FabricMemoryOccurrenceRef memory,
    mlir::MLIRContext &actorContext) {
  auto schema = fabric.memoryConfigurationSchema(memory);
  if (!schema)
    return schema.takeError();
  MemoryActors actors = makeMemoryActors(actorContext, 64);
  const ::fabric::MemoryDispatchTarget managerTarget(
      std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
      ::fabric::ManagerMemoryDispatchTarget{0});
  auto load =
      projectMemoryRow(fabric, *schema, 0, actors.load, managerTarget);
  if (!load)
    return load.takeError();
  auto store =
      projectMemoryRow(fabric, *schema, 1, actors.store, managerTarget);
  if (!store)
    return store.takeError();

  using Role = ::dataflow::semantics::ServiceValueRole;
  const std::size_t control = static_cast<std::size_t>(Role::Control);
  const std::size_t completion = static_cast<std::size_t>(Role::Completion);
  if (!load->roleDestinations[completion] ||
      !store->roleSources[control])
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "internal-edge fixture omitted completion or control");
  load->roleDestinations[completion]->internalConnections = {0};
  store->roleSources[control] =
      loom::fabric::FabricMemoryInternalRoleSource{0};

  loom::fabric::FabricMemoryActive active;
  active.operationRows.resize(schema->layout().operationRows.size());
  active.providerDecodeRows.resize(schema->layout().providerRows.size());
  for (auto [ordinal, rows] : llvm::enumerate(schema->layout().providerRows))
    active.providerDecodeRows[ordinal].resize(rows.size());
  active.operationRows[load->physicalPort] = std::move(*load);
  active.operationRows[store->physicalPort] = std::move(*store);

  loom::fabric::FabricMemoryActive openConsumer = active;
  openConsumer.operationRows[0]
      ->roleDestinations[completion]
      ->internalConnections.clear();
  auto rejectedConsumer = schema->encode(
      loom::fabric::FabricMemoryConfigurationValue{openConsumer});
  if (rejectedConsumer)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "memory configuration accepted an internal consumer without a "
        "producer");
  const std::string consumerMessage =
      llvm::toString(rejectedConsumer.takeError());
  if (!llvm::StringRef(consumerMessage).contains(
          "internal connection is not closed"))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   consumerMessage);

  loom::fabric::FabricMemoryActive openProducer = active;
  openProducer.operationRows[1]->roleSources[control] =
      loom::fabric::FabricMemoryExternalRoleSource{
          6, llvm::APInt(std::max(1U, schema->layout().tagWidthBits), 0)};
  auto rejectedProducer = schema->encode(
      loom::fabric::FabricMemoryConfigurationValue{openProducer});
  if (rejectedProducer)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "memory configuration accepted an internal producer without a "
        "consumer");
  const std::string producerMessage =
      llvm::toString(rejectedProducer.takeError());
  if (!llvm::StringRef(producerMessage).contains(
          "internal connection is not closed"))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   producerMessage);

  return schema->encode(
      loom::fabric::FabricMemoryConfigurationValue{std::move(active)});
}

struct MemoryConfigurationImage final {
  loom::hardware::test::PortableConfigurationTarget target;
  std::vector<std::uint8_t> image;
};

MemoryConfigurationImage makeMemoryConfigurationImage(
    const loom::hardware::FinalizedConfigurationABI &abi,
    loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricMemoryConfigurationSchemaView &schema,
    const loom::CanonicalSemanticBytes &semantic) {
  auto target =
      take(loom::fabric::FabricModulePhysicalTargetRef::create(schema.field()));
  auto physical =
      take(loom::fabric::FabricPhysicalConfigurationFieldRef::create(
          loom::fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                                         std::move(target)}));
  auto slot = take(loom::fabric::qualifyFabricConfigurationSlot(
      physical, loom::fabric::FabricStaticConfigurationResidency{}));
  const loom::hardware::ProgrammingUnit *owner = nullptr;
  for (const auto &unit : abi.abi().programmingUnits())
    for (const auto &field : unit.fields)
      if (field.slot == slot) {
        require(owner == nullptr,
                "memory field has multiple programming owners");
        owner = &unit;
      }
  require(owner != nullptr, "memory field has no programming owner");
  const std::vector<loom::hardware::SemanticConfigurationValue> values = {
      {slot, std::vector<std::uint8_t>(semantic.bytes().begin(),
                                       semantic.bytes().end())}};
  return {take(loom::hardware::test::derivePortableConfigurationTarget(
              abi, spatialCore, owner->id)),
          take(abi.abi().encode(owner->id, values))};
}

void writeSpatialToolArtifacts(const std::filesystem::path &output,
                               const MemoryConfigurationImage &configuration) {
  std::ofstream testbench(output / "spatial_memory_testbench.sv");
  testbench << R"sv(
module testbench;
  logic         clock;
  logic         reset;
  logic [63:0]  input_0_data;
  logic         input_0_valid;
  logic         input_0_ready;
  logic [127:0] input_1_data;
  logic         input_1_valid;
  logic         input_1_ready;
  logic [7:0]   input_2_data;
  logic         input_2_valid;
  logic         input_2_ready;
  logic         input_3_valid;
  logic         input_3_ready;
  logic [63:0]  input_4_data;
  logic         input_4_valid;
  logic         input_4_ready;
  logic [127:0] input_5_data;
  logic         input_5_valid;
  logic         input_5_ready;
  logic [63:0]  input_6_data;
  logic         input_6_valid;
  logic         input_6_ready;
  logic [7:0]   input_7_data;
  logic         input_7_valid;
  logic         input_7_ready;
  logic         input_8_valid;
  logic         input_8_ready;
  logic [127:0] input_9_data;
  logic         input_9_valid;
  logic         input_9_ready;
  logic [63:0]  output_0_data;
  logic         output_0_valid;
  logic         output_0_ready;
  logic         output_1_valid;
  logic         output_1_ready;
  logic         output_2_valid;
  logic         output_2_ready;
  logic [127:0] output_3_data;
  logic         output_3_valid;
  logic         output_3_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations();
  testbench << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 0;
    input_0_valid = 0;
    input_1_data = 0;
    input_1_valid = 0;
    input_2_data = 0;
    input_2_valid = 0;
    input_3_valid = 0;
    input_4_data = 0;
    input_4_valid = 0;
    input_5_data = 0;
    input_5_valid = 0;
    input_6_data = 0;
    input_6_valid = 0;
    input_7_data = 0;
    input_7_valid = 0;
    input_8_valid = 0;
    input_9_data = 0;
    input_9_valid = 0;
    output_0_ready = 0;
    output_1_ready = 0;
    output_2_ready = 0;
    output_3_ready = 0;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
    input_0_data = 64'd3;
    input_0_valid = 1;
    input_3_valid = 1;
    #1;
    check(!input_0_ready && !input_3_ready && !output_0_valid &&
              !output_1_valid,
          "Disabled Spatial memory exchanged a token");
    input_0_valid = 0;
    input_3_valid = 0;

)sv";
  testbench << take(loom::hardware::test::portableAxiLiteProgramAndVerify(
      configuration.target, configuration.image));
  testbench << R"sv(
    cfg_read(32'hffff_fffc, cfg_readback, cfg_read_response);
    check(cfg_read_response === 2'b11,
          "Illegal configuration read did not return DECERR");

    input_9_data = 128'h0123456789abcdef_fedcba9876543210;
    input_9_valid = 1;
    output_3_ready = 1;
    #1;
    check(input_9_ready && output_3_valid &&
              output_3_data == input_9_data,
          "Spatial memory fixture changed boundary passthrough");
    input_9_valid = 0;
    output_3_ready = 0;

    @(negedge clock);
    input_4_data = 64'd3;
    input_6_data = 64'h00000000deadbeef;
    input_4_valid = 1;
    input_6_valid = 1;
    input_8_valid = 1;
    #1;
    check(input_4_ready && input_6_ready && input_8_ready,
          "Spatial store tuple was not accepted atomically");
    @(posedge clock);
    @(negedge clock);
    input_4_valid = 0;
    input_6_valid = 0;
    input_8_valid = 0;
    #1;
    check(output_2_valid,
          "Spatial store did not publish completion");
    repeat (2) begin
      @(negedge clock);
      #1;
      check(output_2_valid,
            "Stalled Spatial store completion was not stable");
    end
    output_2_ready = 1;
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_2_valid,
          "Consumed Spatial store completion remained valid");
    output_2_ready = 0;

    input_0_data = 64'd3;
    input_0_valid = 1;
    input_3_valid = 1;
    #1;
    check(input_0_ready && input_3_ready,
          "Spatial load tuple was not accepted atomically");
    @(posedge clock);
    @(negedge clock);
    input_0_valid = 0;
    input_3_valid = 0;
    #1;
    check(!output_0_valid && !output_1_valid,
          "Atomic load result escaped before all sinks were ready");
    output_0_ready = 1;
    #1;
    check(!output_0_valid && output_1_valid,
          "Atomic load readiness did not expose the blocked obligation");
    repeat (2) begin
      @(negedge clock);
      #1;
      check(!output_0_valid && output_1_valid,
            "Stalled load obligation changed before atomic release");
    end
    output_1_ready = 1;
    #1;
    check(output_0_valid && output_1_valid &&
              output_0_data[31:0] == 32'hdeadbeef,
          "Spatial load returned the wrong value or result tuple");
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_0_valid && !output_1_valid,
          "Consumed Spatial load result remained valid");
    output_0_ready = 0;
    output_1_ready = 0;

    input_0_data = 64'd64;
    input_0_valid = 1;
    input_3_valid = 1;
    repeat (2) begin
      #1;
      check(!input_0_ready && !input_3_ready && !output_0_valid &&
                !output_1_valid,
            "Out-of-range Spatial request aliased into local storage");
      @(negedge clock);
    end
    input_0_valid = 0;
    input_3_valid = 0;

    reset = 1;
    #1;
    check(!input_0_ready && !input_3_ready && !output_0_valid &&
              !output_1_valid && !output_2_valid,
          "Reset did not quiesce Spatial memory");
    @(posedge clock);
    @(negedge clock);
    reset = 0;
    input_0_data = 64'd3;
    input_0_valid = 1;
    input_3_valid = 1;
    #1;
    check(!input_0_ready && !input_3_ready,
          "Reset did not restore Disabled memory configuration");
    $finish;
  end
endmodule
)sv";
  require(static_cast<bool>(testbench),
          "could not write Spatial memory testbench");

  std::ofstream synthesis(output / "spatial_memory.ys");
  synthesis << R"ys(read_verilog -sv spatial_memory.sv
hierarchy -check -top loom_module
check -assert
proc
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
  require(static_cast<bool>(synthesis),
          "could not write Spatial memory synthesis script");
}

void writeTemporalToolArtifacts(const std::filesystem::path &output,
                                const MemoryConfigurationImage &configuration) {
  std::ofstream testbench(output / "temporal_memory_testbench.sv");
  testbench << R"sv(
module testbench;
  logic         clock;
  logic         reset;
  logic [63:0]  input_1_data;
  logic [1:0]   input_1_tag;
  logic         input_1_valid;
  logic         input_1_ready;
  logic [127:0] input_2_data;
  logic [1:0]   input_2_tag;
  logic         input_2_valid;
  logic         input_2_ready;
  logic [7:0]   input_3_data;
  logic [1:0]   input_3_tag;
  logic         input_3_valid;
  logic         input_3_ready;
  logic [1:0]   input_4_tag;
  logic         input_4_valid;
  logic         input_4_ready;
  logic [63:0]  input_5_data;
  logic [1:0]   input_5_tag;
  logic         input_5_valid;
  logic         input_5_ready;
  logic [127:0] input_6_data;
  logic [1:0]   input_6_tag;
  logic         input_6_valid;
  logic         input_6_ready;
  logic [63:0]  input_7_data;
  logic [1:0]   input_7_tag;
  logic         input_7_valid;
  logic         input_7_ready;
  logic [7:0]   input_8_data;
  logic [1:0]   input_8_tag;
  logic         input_8_valid;
  logic         input_8_ready;
  logic [1:0]   input_9_tag;
  logic         input_9_valid;
  logic         input_9_ready;
  logic [127:0] input_10_data;
  logic         input_10_valid;
  logic         input_10_ready;
  logic [63:0]  output_0_data;
  logic [1:0]   output_0_tag;
  logic         output_0_valid;
  logic         output_0_ready;
  logic [1:0]   output_1_tag;
  logic         output_1_valid;
  logic         output_1_ready;
  logic [1:0]   output_2_tag;
  logic         output_2_valid;
  logic         output_2_ready;
  logic [127:0] output_3_data;
  logic         output_3_valid;
  logic         output_3_ready;
  logic         memory_input_0_request_ready;
  logic [63:0]  memory_input_0_response_data;
  logic         memory_input_0_response_valid;
  logic         memory_input_0_request_kind;
  logic [127:0] memory_input_0_request_address;
  logic [63:0]  memory_input_0_request_data;
  logic [7:0]   memory_input_0_request_mask;
  logic         memory_input_0_request_active_lanes_kind;
  logic [1:0]   memory_input_0_request_access_form;
  logic         memory_input_0_request_address_form;
  logic [63:0]  memory_input_0_request_element_width;
  logic [63:0]  memory_input_0_request_lane_count;
  logic [31:0]  memory_input_0_request_address_lane_width;
  logic [63:0]  memory_input_0_request_base_address;
  logic [63:0]  memory_input_0_request_context;
  logic         memory_input_0_request_valid;
  logic         memory_input_0_response_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations();
  testbench << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
    input_1_data = 0;
    input_1_tag = 0;
    input_1_valid = 0;
    input_2_data = 0;
    input_2_tag = 0;
    input_2_valid = 0;
    input_3_data = 0;
    input_3_tag = 0;
    input_3_valid = 0;
    input_4_tag = 0;
    input_4_valid = 0;
    input_5_data = 0;
    input_5_tag = 0;
    input_5_valid = 0;
    input_6_data = 0;
    input_6_tag = 0;
    input_6_valid = 0;
    input_7_data = 0;
    input_7_tag = 0;
    input_7_valid = 0;
    input_8_data = 0;
    input_8_tag = 0;
    input_8_valid = 0;
    input_9_tag = 0;
    input_9_valid = 0;
    input_10_data = 0;
    input_10_valid = 0;
    output_0_ready = 0;
    output_1_ready = 0;
    output_2_ready = 0;
    output_3_ready = 0;
    memory_input_0_request_ready = 0;
    memory_input_0_response_data = 0;
    memory_input_0_response_valid = 0;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
    input_1_data = 64'd3;
    input_1_valid = 1;
    input_4_valid = 1;
    #1;
    check(!input_1_ready && !input_4_ready && !output_0_valid &&
              !output_1_valid,
          "Disabled Temporal memory exchanged a token");
    input_1_valid = 0;
    input_4_valid = 0;

)sv";
  testbench << take(loom::hardware::test::portableAxiLiteProgramAndVerify(
      configuration.target, configuration.image));
  testbench << R"sv(
    input_10_data = 128'h89abcdef01234567_76543210fedcba98;
    input_10_valid = 1;
    output_3_ready = 1;
    #1;
    check(input_10_ready && output_3_valid &&
              output_3_data == input_10_data,
          "Temporal memory fixture changed boundary passthrough");
    input_10_valid = 0;
    output_3_ready = 0;

    input_5_data = 64'd3;
    input_5_tag = 2'd3;
    input_5_valid = 1;
    #1;
    check(!input_5_ready,
          "Temporal memory accepted an unconfigured Physical Tag");
    input_5_valid = 0;
    input_5_tag = 0;

    @(negedge clock);
    input_5_data = 64'd3;
    input_7_data = 64'h0000000011223344;
    input_5_valid = 1;
    input_7_valid = 1;
    input_9_valid = 1;
    #1;
    check(input_5_ready && input_7_ready && input_9_ready,
          "Temporal store operand queues rejected the first tuple");
    @(posedge clock);
    @(negedge clock);
    input_5_data = 64'd4;
    input_7_data = 64'h0000000055667788;
    #1;
    check(!input_5_ready && !input_7_ready && !input_9_ready,
          "Occupied Temporal queues admitted a replacement token");
    @(posedge clock);
    @(negedge clock);
    #1;
    check(input_5_ready && input_7_ready && input_9_ready,
          "Temporal store queues did not reopen after dequeue");
    @(posedge clock);
    @(negedge clock);
    input_5_valid = 0;
    input_7_valid = 0;
    input_9_valid = 0;
    #1;
    check(output_2_valid && output_2_tag == 0,
          "Temporal store did not publish its first completion");
    output_2_ready = 1;
    @(posedge clock);
    @(negedge clock);
    #1;
    check(output_2_valid && output_2_tag == 0,
          "Temporal store pipeline did not publish its second completion");
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_2_valid,
          "Temporal store pipeline published an extra completion");
    output_2_ready = 0;

    input_1_data = 64'd4;
    input_1_tag = 0;
    input_1_valid = 1;
    input_4_tag = 0;
    input_4_valid = 1;
    #1;
    check(input_1_ready && input_4_ready,
          "Temporal load operand queues rejected a complete tuple");
    @(posedge clock);
    @(negedge clock);
    input_1_valid = 0;
    input_4_valid = 0;
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_0_valid && !output_1_valid,
          "Temporal load escaped before atomic readiness");
    output_0_ready = 1;
    #1;
    check(!output_0_valid && output_1_valid && output_1_tag == 0,
          "Temporal load did not retain its blocked completion");
    output_1_ready = 1;
    #1;
    check(output_0_valid && output_1_valid && output_0_tag == 0 &&
              output_1_tag == 0 &&
              output_0_data[31:0] == 32'h55667788,
          "Temporal load returned the wrong value or tag");
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_0_valid && !output_1_valid,
          "Consumed Temporal load result remained valid");

    reset = 1;
    #1;
    check(!input_1_ready && !input_4_ready && !output_0_valid &&
              !output_1_valid && !output_2_valid,
          "Reset did not quiesce Temporal memory");
    @(posedge clock);
    @(negedge clock);
    reset = 0;
    input_1_valid = 1;
    input_4_valid = 1;
    #1;
    check(!input_1_ready && !input_4_ready,
          "Reset did not restore Disabled Temporal memory configuration");
    $finish;
  end
endmodule
)sv";
  require(static_cast<bool>(testbench),
          "could not write Temporal memory testbench");

  std::ofstream synthesis(output / "temporal_memory.ys");
  synthesis << R"ys(read_verilog -sv temporal_memory.sv
hierarchy -check -top loom_module
check -assert
proc
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
  require(static_cast<bool>(synthesis),
          "could not write Temporal memory synthesis script");
}

void writeInternalMemoryToolArtifacts(
    const std::filesystem::path &output,
    const MemoryConfigurationImage &configuration) {
  std::ofstream testbench(output / "internal_memory_testbench.sv");
  testbench << R"sv(
module testbench;
  logic         clock;
  logic         reset;
  logic [63:0]  input_1_data;
  logic [3:0]   input_1_tag;
  logic         input_1_valid;
  logic         input_1_ready;
  logic [3:0]   input_2_data;
  logic [3:0]   input_2_tag;
  logic         input_2_valid;
  logic         input_2_ready;
  logic [3:0]   input_3_tag;
  logic         input_3_valid;
  logic         input_3_ready;
  logic [63:0]  input_4_data;
  logic [3:0]   input_4_tag;
  logic         input_4_valid;
  logic         input_4_ready;
  logic [127:0] input_5_data;
  logic [3:0]   input_5_tag;
  logic         input_5_valid;
  logic         input_5_ready;
  logic [3:0]   input_6_data;
  logic [3:0]   input_6_tag;
  logic         input_6_valid;
  logic         input_6_ready;
  logic [3:0]   input_7_tag;
  logic         input_7_valid;
  logic         input_7_ready;
  logic [127:0] input_8_data;
  logic         input_8_valid;
  logic         input_8_ready;
  logic [127:0] output_0_data;
  logic [3:0]   output_0_tag;
  logic         output_0_valid;
  logic         output_0_ready;
  logic [3:0]   output_1_tag;
  logic         output_1_valid;
  logic         output_1_ready;
  logic [3:0]   output_2_tag;
  logic         output_2_valid;
  logic         output_2_ready;
  logic [127:0] output_3_data;
  logic         output_3_valid;
  logic         output_3_ready;
  logic         memory_input_0_request_ready;
  logic [127:0] memory_input_0_response_data;
  logic         memory_input_0_response_valid;
  logic         memory_input_0_request_kind;
  logic [63:0]  memory_input_0_request_address;
  logic [127:0] memory_input_0_request_data;
  logic [3:0]   memory_input_0_request_mask;
  logic         memory_input_0_request_active_lanes_kind;
  logic [1:0]   memory_input_0_request_access_form;
  logic         memory_input_0_request_address_form;
  logic [63:0]  memory_input_0_request_element_width;
  logic [63:0]  memory_input_0_request_lane_count;
  logic [31:0]  memory_input_0_request_address_lane_width;
  logic [63:0]  memory_input_0_request_base_address;
  logic [63:0]  memory_input_0_request_context;
  logic         memory_input_0_request_valid;
  logic         memory_input_0_response_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations();
  testbench << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  integer store_request_count;
  integer store_completion_count;

  always @(posedge clock or posedge reset) begin
    if (reset) begin
      store_request_count <= 0;
      store_completion_count <= 0;
    end else begin
      if (memory_input_0_request_valid &&
          memory_input_0_request_ready && memory_input_0_request_kind)
        store_request_count <= store_request_count + 1;
      if (output_2_valid && output_2_ready)
        store_completion_count <= store_completion_count + 1;
    end
  end

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  task automatic accept_request(input bit expected_kind);
    integer cycles;
    begin
      cycles = 0;
      while (!memory_input_0_request_valid && cycles != 20) begin
        @(negedge clock);
        cycles = cycles + 1;
      end
      check(memory_input_0_request_valid,
            "Memory manager request did not become valid");
      check(memory_input_0_request_kind == expected_kind,
            "Memory manager request had the wrong operation kind");
      memory_input_0_request_ready = 1;
      @(posedge clock);
      @(negedge clock);
      memory_input_0_request_ready = 0;
    end
  endtask

  task automatic accept_response(input logic [127:0] data);
    integer cycles;
    begin
      memory_input_0_response_data = data;
      memory_input_0_response_valid = 1;
      cycles = 0;
      while (!memory_input_0_response_ready && cycles != 20) begin
        @(negedge clock);
        cycles = cycles + 1;
      end
      check(memory_input_0_response_ready,
            "Memory manager response was not accepted");
      @(posedge clock);
      @(negedge clock);
      memory_input_0_response_valid = 0;
      memory_input_0_response_data = 0;
    end
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
    input_1_data = 0;
    input_1_tag = 0;
    input_1_valid = 0;
    input_2_data = 0;
    input_2_tag = 0;
    input_2_valid = 0;
    input_3_tag = 0;
    input_3_valid = 0;
    input_4_data = 0;
    input_4_tag = 0;
    input_4_valid = 0;
    input_5_data = 0;
    input_5_tag = 0;
    input_5_valid = 0;
    input_6_data = 0;
    input_6_tag = 0;
    input_6_valid = 0;
    input_7_tag = 0;
    input_7_valid = 0;
    input_8_data = 0;
    input_8_valid = 0;
    output_0_ready = 0;
    output_1_ready = 0;
    output_2_ready = 0;
    output_3_ready = 0;
    memory_input_0_request_ready = 0;
    memory_input_0_response_data = 0;
    memory_input_0_response_valid = 0;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
    input_1_data = 64'd1;
    input_1_valid = 1;
    input_3_valid = 1;
    #1;
    check(!input_1_ready && !input_3_ready,
          "Disabled internal-edge memory accepted a load");
    input_1_valid = 0;
    input_3_valid = 0;

)sv";
  testbench << take(loom::hardware::test::portableAxiLiteProgramAndVerify(
      configuration.target, configuration.image));
  testbench << R"sv(
    input_8_data = 128'h0123456789abcdef_fedcba9876543210;
    input_8_valid = 1;
    output_3_ready = 1;
    #1;
    check(input_8_ready && output_3_valid &&
              output_3_data == input_8_data,
          "Internal-edge fixture changed boundary passthrough");
    input_8_valid = 0;
    output_3_ready = 0;

    input_1_data = 64'd1;
    input_1_valid = 1;
    input_3_valid = 1;
    #1;
    check(input_1_ready && input_3_ready,
          "First load operands were not accepted");
    @(posedge clock);
    @(negedge clock);
    input_1_valid = 0;
    input_3_valid = 0;
    accept_request(0);

    output_0_ready = 1;
    output_1_ready = 0;
    accept_response(128'h1111222233334444_5555666677778888);
    #1;
    check(!output_0_valid && output_1_valid,
          "Blocked external load result was not retained atomically");
    repeat (2) begin
      @(negedge clock);
      #1;
      check(!memory_input_0_request_valid,
            "Internal store control was published before external fanout");
    end

    output_1_ready = 1;
    #1;
    check(output_0_valid && output_1_valid &&
              output_0_data ==
                  128'h1111222233334444_5555666677778888,
          "First load result did not release as one fanout");
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_0_valid && !output_1_valid,
          "First load result was published more than once");

    input_1_data = 64'd2;
    input_1_valid = 1;
    input_3_valid = 1;
    #1;
    check(input_1_ready && input_3_ready,
          "Second load operands were not accepted");
    @(posedge clock);
    @(negedge clock);
    input_1_valid = 0;
    input_3_valid = 0;
    accept_request(0);
    accept_response(128'h9999aaaabbbbcccc_ddddeeeeffff0000);
    #1;
    check(!output_0_valid && !output_1_valid,
          "Full internal queue did not backpressure the second load result");

    input_4_data = 64'd3;
    input_5_data = 128'h0000000000000000_00000000deadbeef;
    input_4_valid = 1;
    input_5_valid = 1;
    #1;
    check(input_4_ready && input_5_ready,
          "First store operands were not accepted");
    @(posedge clock);
    @(negedge clock);
    input_4_valid = 0;
    input_5_valid = 0;
    #1;
    check(!output_0_valid && !output_1_valid,
          "Internal queue allowed same-cycle replacement");
    output_1_ready = 0;
    accept_request(1);
    #1;
    check(!output_0_valid && output_1_valid,
          "Second load did not reach its external fanout after store issue");
    output_1_ready = 1;
    #1;
    check(output_0_valid && output_1_valid &&
              output_0_data ==
                  128'h9999aaaabbbbcccc_ddddeeeeffff0000,
          "Second load result did not release after store issue");
    @(posedge clock);
    @(negedge clock);
    #1;
    check(!output_0_valid && !output_1_valid,
          "Second load result was published more than once");

    input_4_data = 64'd4;
    input_5_data = 128'h0000000000000000_00000000cafef00d;
    input_4_valid = 1;
    input_5_valid = 1;
    #1;
    check(input_4_ready && input_5_ready,
          "Second store operands were not accepted");
    @(posedge clock);
    @(negedge clock);
    input_4_valid = 0;
    input_5_valid = 0;

    output_2_ready = 1;
    accept_response(0);
    accept_request(1);
    accept_response(0);
    repeat (2) @(posedge clock);
    @(negedge clock);
    #1;
    check(store_request_count == 2,
          "Internal controls did not produce exactly two stores");
    check(store_completion_count == 2,
          "Internal stores did not produce exactly two completions");
    check(!memory_input_0_request_valid && !output_0_valid &&
              !output_1_valid && !output_2_valid,
          "Internal connection duplicated a token");
    $finish;
  end
endmodule
)sv";
  require(static_cast<bool>(testbench),
          "could not write internal-edge memory testbench");

  std::ofstream synthesis(output / "internal_memory.ys");
  synthesis << R"ys(read_verilog -sv internal_memory.sv
hierarchy -check -top loom_module
check -assert
proc
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
  require(static_cast<bool>(synthesis),
          "could not write internal-edge memory synthesis script");
}

void verifySchedule(const loom::ArtifactStore &store,
                    const std::filesystem::path &output, bool temporal) {
  auto module = makeMemoryModule(store, temporal);
  auto system = makeMemorySystem(module, store, temporal);
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
  auto abi = makeAbi(store, module, system, spatialCore);
  mlir::DialectRegistry actorRegistry;
  actorRegistry.insert<dataflow::DataflowDialect, mlir::DLTIDialect,
                       mlir::func::FuncDialect>();
  mlir::MLIRContext actorContext(actorRegistry,
                                 mlir::MLIRContext::Threading::DISABLED);
  const auto memory = module.view().memoryOccurrences().front();
  const auto schema = take(module.view().memoryConfigurationSchema(memory));
  const auto semantic = take(makeActiveMemoryConfiguration(
      module.view(), memory, temporal, actorContext));
  const auto configuration =
      makeMemoryConfigurationImage(abi, spatialCore, schema, semantic);
  require(configuration.image.size() == configuration.target.payloadByteCount,
          "memory programming image has the wrong byte count");

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(loom::hardware::rtl::buildModuleRootCirctSkeleton(
      context, spatialCore, abi));
  require(skeleton.operationLeaves.empty(),
          "memory-only hierarchy unexpectedly created operation leaves");
  std::string text;
  llvm::raw_string_ostream(text) << *skeleton.module;
  const llvm::StringRef ir(text);
  require(ir.contains("loom_memory_") && ir.contains("seq.firmem") &&
              ir.contains("result_occupied_"),
          "memory hierarchy omitted its controller or local storage");
  require(ir.contains("local_service_cursor") && ir.contains("result_cursor"),
          "memory hierarchy omitted its configured schedule state");
  if (temporal)
    require(ir.contains("memory_input_0_request_valid") &&
                ir.contains("memory_input_0_response_valid"),
            "memory hierarchy omitted its manager boundary profile");
  const std::string rtl =
      take(loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
          *skeleton.module));
  require(llvm::StringRef(rtl).contains("module loom_memory_") &&
              llvm::StringRef(rtl).contains("module loom_module"),
          "memory hierarchy did not export complete SystemVerilog");
  std::ofstream(output /
                (temporal ? "temporal_memory.sv" : "spatial_memory.sv"))
      << rtl;
  if (temporal)
    writeTemporalToolArtifacts(output, configuration);
  else
    writeSpatialToolArtifacts(output, configuration);
}

void verifyInternalEdge(loom::ArtifactStore &store,
                        const std::filesystem::path &output) {
  auto module = loom::test::buildInternalMemoryEdgeFabric(
      store, ::fabric::Schedule::Temporal);
  auto system = makeMemorySystem(module, store, true);
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
  auto abi = makeAbi(store, module, system, spatialCore);
  mlir::DialectRegistry actorRegistry;
  actorRegistry.insert<dataflow::DataflowDialect, mlir::DLTIDialect,
                       mlir::func::FuncDialect>();
  mlir::MLIRContext actorContext(actorRegistry,
                                 mlir::MLIRContext::Threading::DISABLED);
  const auto memory = module.view().memoryOccurrences().front();
  const auto schema = take(module.view().memoryConfigurationSchema(memory));
  const auto semantic = take(makeInternalMemoryConfiguration(
      module.view(), memory, actorContext));
  const auto configuration =
      makeMemoryConfigurationImage(abi, spatialCore, schema, semantic);

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(loom::hardware::rtl::buildModuleRootCirctSkeleton(
      context, spatialCore, abi));
  std::string text;
  llvm::raw_string_ostream(text) << *skeleton.module;
  const llvm::StringRef ir(text);
  require(ir.contains("operand_occupied_") &&
              ir.contains("result_occupied_"),
          "internal memory omitted registered operand or result state");
  const std::string rtl =
      take(loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
          *skeleton.module));
  std::ofstream(output / "internal_memory.sv") << rtl;
  std::ofstream image(output / "internal_memory_configuration.txt");
  for (std::uint8_t byte : configuration.image)
    image << static_cast<unsigned>(byte) << '\n';
  writeInternalMemoryToolArtifacts(output, configuration);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one artifact directory");
  std::filesystem::create_directories(argv[1]);
  loom::ArtifactStore store(argv[1]);
  verifySchedule(store, argv[1], false);
  verifySchedule(store, argv[1], true);
  verifyInternalEdge(store, argv[1]);
  return EXIT_SUCCESS;
}
