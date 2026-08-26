#include "Runtime/Gem5SimulationBinding.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Artifact/InterconnectImplementation.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Runtime/Gem5BridgeABI.h"

#include "FabricArtifactBytecodeInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::runtime;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid binding");
  const std::string diagnostic = llvm::toString(value.takeError());
  require(test, llvm::StringRef(diagnostic).contains(expected), diagnostic);
}

llvm::Error validateByte(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "payload must contain one byte");
  return llvm::Error::success();
}

llvm::Error validateProcessor(
    llvm::ArrayRef<std::uint8_t> payload,
    const loom::fabric::InstructionCoreArchitecturalContract &,
    const loom::fabric::InstructionCoreMicroarchitecturalRealization &) {
  return validateByte(payload);
}

const Gem5ModelContractDescriptor &processorDescriptor() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.test_processor", {1, 0}},
      "loom.gem5.test_processor.v1",
      "TimingSimpleCPU",
      Gem5ModelObjectClass::Processor,
      false,
      validateByte,
      validateProcessor,
      {}};
  return descriptor;
}

const Gem5ModelContractDescriptor &spatialBridgeDescriptor() {
  static const Gem5ModelPortKindDescriptor ports[] = {
      {0, "spatial_boundary", Gem5ModelPortClass::SpatialBoundary, false,
       validateByte}};
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.test_spatial_bridge", {1, 0}},
      "loom.gem5.test_spatial_bridge.v1",
      "LoomSpatialBridge",
      Gem5ModelObjectClass::SpatialBridge,
      false,
      validateByte,
      nullptr,
      ports};
  return descriptor;
}

const Gem5ModelContractDescriptor &memoryDescriptor() {
  static const Gem5ModelPortKindDescriptor ports[] = {
      {0, "memory", Gem5ModelPortClass::MemoryOrService, false,
       validateByte}};
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.test_memory", {1, 0}},
      "loom.gem5.test_memory.v1",
      "SimpleMemory",
      Gem5ModelObjectClass::MemoryOrService,
      false,
      validateByte,
      nullptr,
      ports};
  return descriptor;
}

const Gem5ModelContractDescriptor &transportDescriptor() {
  static const Gem5ModelPortKindDescriptor ports[] = {
      {0, "transport", Gem5ModelPortClass::Transport, false, validateByte}};
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.test_transport", {1, 0}},
      "loom.gem5.test_transport.v1",
      "SystemXBar",
      Gem5ModelObjectClass::Transport,
      false,
      validateByte,
      nullptr,
      ports};
  return descriptor;
}

const Gem5ModelContractDescriptor &externalDescriptor() {
  static const Gem5ModelPortKindDescriptor ports[] = {
      {0, "external", Gem5ModelPortClass::ExternalEndpoint, false,
       validateByte}};
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.test_external", {1, 0}},
      "loom.gem5.test_external.v1",
      "ExternalEndpoint",
      Gem5ModelObjectClass::ExternalEndpoint,
      false,
      validateByte,
      nullptr,
      ports};
  return descriptor;
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

loom::fabric::FinalizedFabricRoot makeModule(llvm::StringRef test,
                                             const ArtifactStore &store) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @gem5_fixture(%input: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%arg = %input : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fu_arg = %arg : !fabric.bits<32>) -> !fabric.bits<32> {
            %value = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
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
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  const auto encodedContract = take(
      test, ::fabric::encodeResourceContractRecord(
                ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> contract(encodedContract.begin(),
                                          encodedContract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), contract));
  });
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no Module root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

loom::fabric::FinalizedFabricRoot
makeCatalogSystem(llvm::StringRef test,
                  const loom::fabric::FinalizedFabricRoot &module,
                  const ArtifactStore &store) {
  loom::adg::DesignBuilder design(store);
  auto system = take(test, design.createSystem("gem5-catalog-system"));
  auto imported = take(test, system.importSpatialCore(module));
  const auto architecture =
      take(test, loom::adg::getBuiltinInstructionCoreArchitecture());
  const auto microarchitecture = take(
      test, loom::adg::getBuiltinInOrderInstructionCoreMicroarchitecture());
  const auto host =
      take(test, system.addHostCore(architecture, microarchitecture));
  const auto source =
      take(test, system.addAccCore(architecture, microarchitecture, imported));
  const auto destination =
      take(test, system.addAccCore(architecture, microarchitecture, imported));

  const auto bits32 = take(test, loom::adg::PortType::bits(32));
  const auto transport = take(
      test, system.addTransportResource(
                {{bits32},
                 {bits32, bits32},
                 ::fabric::oneCycleElasticOperationResourceContract(),
                 loom::adg::SystemTransferPatternSelection::Configuration}));
  const auto firstPattern =
      take(test, system.addTransferPattern(transport, 0, {0}, 0));
  const auto secondPattern =
      take(test, system.addTransferPattern(transport, 0, {1}, 0));
  const auto dynamicTransport =
      take(test, system.addTransportResource(
                     {{bits32},
                      {bits32, bits32},
                      ::fabric::oneCycleElasticOperationResourceContract(),
                      loom::adg::SystemTransferPatternSelection::Dynamic}));
  const auto dynamicFirstPattern =
      take(test, system.addTransferPattern(dynamicTransport, 0, {0}, 0));
  const auto dynamicSecondPattern =
      take(test, system.addTransferPattern(dynamicTransport, 0, {1}, 0));
  const auto sourceOutput = take(test, source.spatialTransportOutput(0));
  const auto transportInput = take(test, transport.input(0));
  const auto firstOutput = take(test, transport.output(0));
  const auto secondOutput = take(test, transport.output(1));
  const auto sourceInput = take(test, source.spatialTransportInput(0));
  const auto destinationInput =
      take(test, destination.spatialTransportInput(0));
  if (llvm::Error error = system.connect(sourceOutput, transportInput))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.connect(firstOutput, sourceInput))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.connect(secondOutput, destinationInput))
    fail(test, llvm::toString(std::move(error)));

  const std::vector<loom::adg::HardwareDomainMember> clockMembers = {
      host.domainMember(),
      source.instructionCoreDomainMember(),
      source.spatialCoreDomainMember(loom::fabric::FabricClockResetKind::Clock),
      destination.instructionCoreDomainMember(),
      destination.spatialCoreDomainMember(
          loom::fabric::FabricClockResetKind::Clock),
      transport.domainMember(),
      firstPattern.domainMember(),
      secondPattern.domainMember(),
      dynamicTransport.domainMember(),
      dynamicFirstPattern.domainMember(),
      dynamicSecondPattern.domainMember()};
  const std::vector<loom::adg::HardwareDomainMember> resetMembers = {
      host.domainMember(),
      source.instructionCoreDomainMember(),
      source.spatialCoreResetDomainMember(),
      destination.instructionCoreDomainMember(),
      destination.spatialCoreResetDomainMember(),
      transport.domainMember(),
      firstPattern.domainMember(),
      secondPattern.domainMember(),
      dynamicTransport.domainMember(),
      dynamicFirstPattern.domainMember(),
      dynamicSecondPattern.domainMember()};
  auto clock = take(test, system.createHardwareDomain());
  auto clockContract =
      take(test, loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = clock.close(clockMembers, std::move(clockContract)))
    fail(test, llvm::toString(std::move(error)));
  auto reset = take(test, system.createHardwareDomain());
  auto resetContract = take(
      test, loom::fabric::ResetDomainContractRecord::create(
                loom::fabric::ResetPolarity::ActiveHigh,
                loom::fabric::ResetTiming::Asynchronous,
                loom::fabric::ResetTiming::Asynchronous,
                loom::fabric::ResetInitialState::Asserted, std::nullopt, 0));
  if (llvm::Error error = reset.close(resetMembers, std::move(resetContract)))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "catalog System did not finalize exactly one root");
  return take(test, loom::fabric::importEntireFabricRoot(
                        finalized.roots().front().reference(), store));
}

std::uint64_t
configuredTransportCount(llvm::StringRef test,
                         const loom::fabric::FinalizedFabricRoot &system) {
  const auto view = take(test, loom::fabric::requireSystemRoot(system.view()));
  return llvm::count_if(view.transportResources(), [&](const auto resource) {
    return system.view().inventorySize(
               loom::fabric::FabricInventoryOwnerRef::of(resource),
               loom::fabric::FabricInventoryKind::SemanticConfigField) != 0;
  });
}

loom::fabric::FinalizedFabricRoot
replaceTransport(llvm::StringRef test,
                 const loom::fabric::FinalizedFabricRoot &parent,
                 loom::fabric::SystemTransportResourceRef target,
                 loom::fabric::SystemTransportResourceRef prototype,
                 const ArtifactStore &store) {
  loom::adg::DesignBuilder design(store);
  auto system = take(test, design.deriveSystem(parent, {}));
  if (llvm::Error error = system.replaceTransportResource(target, prototype))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "transport replacement did not finalize one System");
  return finalized.roots().front();
}

void transportReplacementPreservesSelection(
    llvm::StringRef test, const loom::fabric::FinalizedFabricRoot &system,
    const ArtifactStore &store) {
  const auto view = take(test, loom::fabric::requireSystemRoot(system.view()));
  std::optional<loom::fabric::SystemTransportResourceRef> configured;
  std::optional<loom::fabric::SystemTransportResourceRef> dynamic;
  for (const auto resource : view.transportResources()) {
    const bool hasField =
        system.view().inventorySize(
            loom::fabric::FabricInventoryOwnerRef::of(resource),
            loom::fabric::FabricInventoryKind::SemanticConfigField) != 0;
    auto &selection = hasField ? configured : dynamic;
    require(test, !selection.has_value(),
            "catalog System repeats one selection mode");
    selection = resource;
  }
  require(test, configured.has_value() && dynamic.has_value(),
          "catalog System does not exercise both selection modes");

  const auto promoted =
      replaceTransport(test, system, *dynamic, *configured, store);
  require(test, configuredTransportCount(test, promoted) == 2,
          "transport replacement did not copy Configuration selection");
  const auto demoted =
      replaceTransport(test, system, *configured, *dynamic, store);
  require(test, configuredTransportCount(test, demoted) == 0,
          "transport replacement did not copy Dynamic selection");
}

loom::fabric::FinalizedFabricRoot
makeInterconnect(llvm::StringRef test, const ArtifactRootReference &system,
                 const ArtifactStore &store) {
  auto refined =
      take(test, loom::fabric::importEntireFabricRoot(system, store));
  const auto systemView =
      take(test, loom::fabric::requireSystemRoot(refined.view()));
  require(test, systemView.transportResources().size() == 2,
          "catalog System does not contain both transport resources");
  const auto configured =
      llvm::find_if(systemView.transportResources(), [&](const auto candidate) {
        return refined.view().inventorySize(
                   loom::fabric::FabricInventoryOwnerRef::of(candidate),
                   loom::fabric::FabricInventoryKind::SemanticConfigField) != 0;
      });
  require(test, configured != systemView.transportResources().end(),
          "catalog System has no configured transport resource");
  const auto resource = *configured;
  const auto patterns = systemView.transferPatterns(resource);
  require(test, patterns.size() == 2,
          "catalog System does not contain two transfer patterns");
  const loom::fabric::FabricSemanticConfigFieldRef field{
      loom::fabric::FabricConfigurationOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(resource)),
      0};
  auto relation =
      take(test, refined.view().semanticFieldRelation(field, context()));
  require(test,
          relation.kind() ==
                  loom::fabric::FabricSemanticFieldRelationKind::Direct &&
              relation.directEncodedBitCount() == patterns.size(),
          "transport configuration field lost its pattern-control shape");
  const auto disabled =
      take(test, loom::fabric::encodeSystemTransportResourceConfiguration(
                     refined.view(), field, {}));
  const auto selected =
      take(test, loom::fabric::encodeSystemTransportResourceConfiguration(
                     refined.view(), field, patterns.take_front()));
  require(test, disabled.bytes() != selected.bytes(),
          "transport pattern selection did not change semantic configuration");
  auto builder = take(
      test,
      loom::fabric::InterconnectImplementationBuilder::create(refined, store));
  auto implementation = take(test, std::move(builder).finalize());
  auto summary = take(
      test, loom::fabric::inspectInterconnectImplementation(implementation));
  require(test, summary.endpointCount != 0,
          "interconnect builder emitted no protocol endpoints");
  require(test, summary.resourceStateCount != 0,
          "interconnect builder emitted no protocol resources");
  require(test, summary.transferPatternCount != 0,
          "interconnect builder emitted no protocol transfers");
  require(test, summary.configurationFieldCount != 0,
          "interconnect builder emitted no protocol configuration fields");
  require(test,
          summary.refinementCount == summary.endpointCount +
                                         summary.resourceStateCount +
                                         summary.transferPatternCount +
                                         summary.configurationFieldCount,
          "interconnect builder emitted an incomplete refinement relation");
  auto imported = take(test, loom::fabric::importEntireFabricRoot(
                                 implementation.reference(), store));
  require(test, imported.reference() == implementation.reference(),
          "strict interconnect re-import changed artifact identity");
  const auto importedSummary =
      take(test, loom::fabric::inspectInterconnectImplementation(imported));
  require(test,
          importedSummary.endpointCount == summary.endpointCount &&
              importedSummary.resourceStateCount ==
                  summary.resourceStateCount &&
              importedSummary.transferPatternCount ==
                  summary.transferPatternCount &&
              importedSummary.configurationFieldCount ==
                  summary.configurationFieldCount &&
              importedSummary.refinementCount == summary.refinementCount,
          "strict interconnect re-import changed the typed catalog");
  return imported;
}

ArtifactRootReference publishIncompleteInterconnect(
    llvm::StringRef test,
    const loom::fabric::FinalizedFabricRoot &implementation,
    const ArtifactStore &store) {
  const auto canonical = take(test, store.get(implementation.reference()));
  auto decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(canonical.bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  auto root = llvm::dyn_cast<::fabric::InterconnectImplementationOp>(
      &parsed.module->getBody()->front());
  require(test, static_cast<bool>(root),
          "interconnect payload has no implementation root");
  auto configuration = llvm::find_if(
      root.getImplementation().front(), [](mlir::Operation &operation) {
        return llvm::isa<::fabric::InterconnectGem5EventConfigurationFieldOp>(
            operation);
      });
  require(test, configuration != root.getImplementation().front().end(),
          "interconnect payload has no removable configuration field");
  configuration->erase();
  const auto bytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  const auto incomplete =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::InterconnectImplementation,
                     decoded.dependencies, bytecode));
  const ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, incomplete));
  return {loom::fabric::fabricArtifactSchema.identity.str(),
          loom::fabric::fabricArtifactSchema.version, identity};
}

Gem5SimObjectRef object(const Gem5ModelContractDescriptor &descriptor,
                        std::uint8_t ordinal) {
  return {gem5ModelContractDescriptorRef(descriptor), {ordinal}};
}

Gem5SimPortRef port(const Gem5ModelContractDescriptor &descriptor,
                    std::uint8_t ordinal) {
  auto owner = object(descriptor, ordinal);
  return {std::move(owner), 0, {ordinal}};
}

std::optional<loom::fabric::SpatialCoreOccurrenceRef> spatialCoreOf(
    const loom::fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport()) {
    if (transport->owner.kind() !=
        loom::fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return std::nullopt;
    return std::get<loom::fabric::SpatialCoreOccurrenceRef>(
        transport->owner.payload);
  }
  const auto *memory = endpoint.memory();
  if (!memory || memory->owner.kind() !=
                     loom::fabric::FabricMemoryEndpointOwnerKind::
                         SpatialCoreOccurrence)
    return std::nullopt;
  return std::get<loom::fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
}

Gem5SimulationBindingDraft makeDraft(
    llvm::StringRef test, const loom::fabric::FinalizedFabricRoot &systemRoot,
    const ArtifactRootReference &interconnect) {
  const auto system =
      take(test, loom::fabric::requireSystemRoot(systemRoot.view()));
  Gem5SimulationBindingDraft draft{
      systemRoot.reference(),
      interconnect,
      {"https://gem5.googlesource.com/public/gem5",
       "0123456789abcdef0123456789abcdef01234567",
       "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
       "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"},
      gem5BridgeAbiIdentity,
      {}};
  std::uint8_t ordinal = 1;
  for (const auto core : system.artifact().hostCoreOccurrences())
    draft.correspondences.push_back(Gem5ProcessorCorrespondence{
        Gem5ProcessorFabricRef(core), object(processorDescriptor(), ordinal++)});
  for (const auto core : system.artifact().accCoreOccurrences())
    draft.correspondences.push_back(Gem5ProcessorCorrespondence{
        Gem5ProcessorFabricRef(
            loom::fabric::InstructionCoreContextRef{core}),
        object(processorDescriptor(), ordinal++)});
  for (const auto &attachment : system.spatialAttachments()) {
    const auto core = spatialCoreOf(attachment.spatialEndpoint);
    require(test, core.has_value(), "attachment has no SpatialCore owner");
    draft.correspondences.push_back(Gem5SpatialBridgeCorrespondence{
        *core, attachment.spatialEndpoint,
        port(spatialBridgeDescriptor(), ordinal++)});
  }
  for (const auto service : system.artifact().systemMemoryServices()) {
    const auto simObject = object(memoryDescriptor(), ordinal);
    draft.correspondences.push_back(Gem5MemoryOrServiceCorrespondence{
        Gem5MemoryOrServiceFabricRef(service), simObject,
        Gem5SimPortRef{simObject, 0, {ordinal}}});
    ++ordinal;
  }
  for (const auto endpoint : system.artifact().systemServiceEndpoints()) {
    const auto simObject = object(memoryDescriptor(), ordinal);
    draft.correspondences.push_back(Gem5MemoryOrServiceCorrespondence{
        Gem5MemoryOrServiceFabricRef(endpoint), simObject,
        Gem5SimPortRef{simObject, 0, {ordinal}}});
    ++ordinal;
  }
  for (const auto resource : system.transportResources()) {
    const auto simObject = object(transportDescriptor(), ordinal);
    draft.correspondences.push_back(Gem5TransportCorrespondence{
        Gem5TransportFabricRef(resource), simObject,
        Gem5SimPortRef{simObject, 0, {ordinal}}});
    ++ordinal;
  }
  for (const auto &endpoint : system.artifact().transportEndpoints()) {
    const auto owner = endpoint.owner.kind();
    if (owner != loom::fabric::FabricTransportEndpointOwnerKind::
                     SystemServiceEndpoint &&
        owner != loom::fabric::FabricTransportEndpointOwnerKind::
                     SystemTransportResource)
      continue;
    const auto simObject = object(transportDescriptor(), ordinal);
    draft.correspondences.push_back(Gem5TransportCorrespondence{
        Gem5TransportFabricRef(endpoint), simObject,
        Gem5SimPortRef{simObject, 0, {ordinal}}});
    ++ordinal;
  }
  for (const auto boundary : system.artifact().externalBoundaries()) {
    const auto simObject = object(externalDescriptor(), ordinal);
    draft.correspondences.push_back(Gem5ExternalEndpointCorrespondence{
        boundary, simObject, Gem5SimPortRef{simObject, 0, {ordinal}}});
    ++ordinal;
  }
  return draft;
}

void roundTripAndRejectInvalid(const ArtifactStore &artifacts) {
  const auto module = makeModule(__func__, artifacts);
  const auto system = makeCatalogSystem(__func__, module, artifacts);
  transportReplacementPreservesSelection(__func__, system, artifacts);
  const auto interconnect =
      makeInterconnect(__func__, system.reference(), artifacts);
  auto refined = take(__func__, loom::fabric::importEntireFabricRoot(
                                     system.reference(), artifacts));
  auto invalidBuilder = take(
      __func__, loom::fabric::InterconnectImplementationBuilder::create(
                    refined, artifacts));
  if (llvm::Error error = invalidBuilder.setProtocolSchema(
          static_cast<::fabric::InterconnectProtocolSchema>(99))) {
    const std::string diagnostic = llvm::toString(std::move(error));
    require(__func__, llvm::StringRef(diagnostic).contains("not registered"),
            "invalid interconnect protocol lost its typed diagnostic");
  } else {
    fail(__func__, "accepted an unregistered interconnect protocol");
  }
  auto draft = makeDraft(__func__, system, interconnect.reference());
  require(__func__, draft.correspondences.size() >= 3,
          "fixture does not exercise multiple correspondence classes");
  auto reversed = draft;
  std::reverse(reversed.correspondences.begin(),
               reversed.correspondences.end());
  const auto forward = take(
      __func__, finalizeGem5SimulationBinding(std::move(draft), artifacts));
  const auto canonical = take(
      __func__, finalizeGem5SimulationBinding(std::move(reversed), artifacts));
  require(__func__, forward.reference() == canonical.reference(),
          "authoring order changed binding identity");
  const auto imported = take(
      __func__, importGem5SimulationBinding(forward.reference(), artifacts));
  require(__func__,
          imported.binding().correspondences().size() ==
              forward.binding().correspondences().size(),
          "strict import lost correspondence rows");

  const auto incomplete =
      publishIncompleteInterconnect(__func__, interconnect, artifacts);
  expectError(__func__,
              loom::fabric::importEntireFabricRoot(incomplete, artifacts),
              "omits a configuration field");

  auto missing = makeDraft(__func__, system, interconnect.reference());
  missing.correspondences.pop_back();
  expectError(__func__,
              finalizeGem5SimulationBinding(std::move(missing), artifacts),
              "does not exactly cover");

  auto shared = makeDraft(__func__, system, interconnect.reference());
  auto first = std::find_if(shared.correspondences.begin(),
                            shared.correspondences.end(), [](const auto &row) {
                              return std::holds_alternative<
                                  Gem5ProcessorCorrespondence>(row);
                            });
  auto second = first == shared.correspondences.end()
                    ? shared.correspondences.end()
                    : std::find_if(std::next(first),
                                   shared.correspondences.end(),
                                   [](const auto &row) {
                                     return std::holds_alternative<
                                         Gem5ProcessorCorrespondence>(row);
                                   });
  require(__func__, second != shared.correspondences.end(),
          "fixture has fewer than two processors");
  std::get<Gem5ProcessorCorrespondence>(*second).simObject =
      std::get<Gem5ProcessorCorrespondence>(*first).simObject;
  expectError(__func__,
              finalizeGem5SimulationBinding(std::move(shared), artifacts),
              "without declared sharing");
}

} // namespace

int main() {
  llvm::SmallString<128> root;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-gem5-simulation-binding-test", root))
    fail("main", error.message());
  const std::filesystem::path path(root.str().str());
  std::filesystem::create_directories(path / "artifacts");
  const ArtifactStore artifacts((path / "artifacts").string());

  for (const Gem5ModelContractDescriptor *descriptor :
       {&processorDescriptor(), &spatialBridgeDescriptor(),
        &memoryDescriptor(), &transportDescriptor(), &externalDescriptor()})
    if (llvm::Error error = registerGem5ModelContract(*descriptor))
      fail("main", llvm::toString(std::move(error)));
  roundTripAndRejectInvalid(artifacts);

  std::filesystem::remove_all(path);
  return 0;
}
