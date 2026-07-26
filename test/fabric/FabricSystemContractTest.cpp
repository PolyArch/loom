#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

namespace {

constexpr FabricEntityId kClockA = 21;
constexpr FabricEntityId kClockB = 22;
constexpr FabricEntityId kCarrierA = 11;

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    const llvm::Twine &message) {
  if (value)
    fail(test, message);
  llvm::consumeError(value.takeError());
}

ClockDomainRef clockA() { return ClockDomainRef(HardwareDomainRef(kClockA)); }
ClockDomainRef clockB() { return ClockDomainRef(HardwareDomainRef(kClockB)); }
FabricTransferPatternRef patternA() {
  return {SystemTransportResourceRef(kCarrierA), 0};
}

::fabric::ResourceContract instructionContextContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {::fabric::CapacityDimensionDeclaration{::fabric::CapacityDimensionKey(0),
                                              ::fabric::CapacityUnits(1),
                                              ::fabric::CapacityUnits(0)}}}};
  declaration.timingContracts = {::fabric::TimingContractDeclaration{
      ::fabric::TimingContractKey(0), {0, 1}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.usePatterns = {::fabric::UsePatternDeclaration{
      ::fabric::UsePatternKey(0),
      ::fabric::RequesterKey(0),
      ::fabric::EligibilityKey(0),
      ::fabric::EventKey(0),
      ::fabric::EventKey(1),
      std::nullopt,
      ::fabric::TimingContractKey(0),
      {::fabric::ClaimDeclaration{::fabric::ClaimKey(0), ::fabric::StateKey(0),
                                  ::fabric::CapacityDimensionKey(0),
                                  ::fabric::CapacityUnits(1)}},
      {::fabric::InternalTransactionDeclaration{{::fabric::ClaimKey(0)}}}}};
  return take("instruction context resource contract",
              ::fabric::ResourceContract::create(declaration));
}

InstructionCoreArchitecturalContract
representativeArchitecture(bool acceleratorDispatcher = true) {
  RiscVArchitectureDeclaration declaration;
  declaration.xlen = RiscVXLen::X64;
  declaration.base = RiscVBase::I;
  declaration.extensions = {RiscVExtension::M, RiscVExtension::F,
                            RiscVExtension::D, RiscVExtension::V,
                            RiscVExtension::Zicsr};
  declaration.endianness = InstructionEndianness::Little;
  declaration.physicalAddressWidthBits = 48;
  declaration.privilegeModes = {PrivilegeMode::User, PrivilegeMode::Machine};
  declaration.abiCapabilities = {RiscVAbi::Lp64d};
  declaration.memoryOrdering = RiscVMemoryOrdering::Rvwmo;
  declaration.syncScopes = {InstructionSyncScope::SingleThread,
                            InstructionSyncScope::Hart,
                            InstructionSyncScope::System};
  declaration.codeModels = {RiscVCodeModel::MediumLow,
                            RiscVCodeModel::MediumAny};
  declaration.relocationModels = {RelocationModel::Static,
                                  RelocationModel::PositionIndependent};
  if (acceleratorDispatcher)
    declaration.runtimeServices = {InstructionRuntimeService::ThreadDispatch,
                                   InstructionRuntimeService::SpatialLaunch};
  return take("representative architecture",
              InstructionCoreArchitecturalContract::create(declaration));
}

InstructionCoreMicroarchitecturalRealization representativeMicroarchitecture() {
  InstructionCoreCommonDeclaration common{
      2,
      {{InstructionOperationClass::LoadStore, 1, 3, 1},
       {InstructionOperationClass::IntegerAlu, 1, 1, 1},
       {InstructionOperationClass::IntegerAlu, 2, 1, 1}},
      instructionContextContract()};
  InOrderMicroarchitectureDeclaration pipeline{2, 2, 2, 2, 1, 1, 8, 4};
  return take("representative microarchitecture",
              InstructionCoreMicroarchitecturalRealization::createInOrder(
                  std::move(common), pipeline));
}

std::string denseI8Assembly(mlir::MLIRContext &context,
                            llvm::ArrayRef<std::uint8_t> bytes);

void checkInstructionArchitectureContract() {
  constexpr llvm::StringLiteral test = "instruction architecture contract";
  InstructionCoreArchitecturalContract architecture =
      representativeArchitecture();
  std::vector<std::uint8_t> encoded =
      take(test, encodeInstructionCoreArchitecturalContract(architecture));
  InstructionCoreArchitecturalContract decoded =
      take(test, decodeInstructionCoreArchitecturalContract(encoded));
  require(test,
          take(test, encodeInstructionCoreArchitecturalContract(decoded)) ==
              encoded,
          "architecture bytes changed after strict import");

  RiscVArchitectureDeclaration invalid;
  invalid.xlen = RiscVXLen::X64;
  invalid.base = RiscVBase::I;
  invalid.extensions = {RiscVExtension::D};
  invalid.endianness = InstructionEndianness::Little;
  invalid.physicalAddressWidthBits = 48;
  invalid.privilegeModes = {PrivilegeMode::Machine};
  invalid.abiCapabilities = {RiscVAbi::Lp64d};
  invalid.memoryOrdering = RiscVMemoryOrdering::Rvwmo;
  invalid.syncScopes = {InstructionSyncScope::System};
  invalid.codeModels = {RiscVCodeModel::MediumAny};
  invalid.relocationModels = {RelocationModel::Static};
  expectRejected(test, InstructionCoreArchitecturalContract::create(invalid),
                 "accepted D without F");
}

void checkImportedModuleTargetReference() {
  constexpr llvm::StringLiteral test = "ImportedModule target reference";
  const FabricImportedModuleTargetRef expected{7, FabricModuleTemplateRef(19)};
  const std::vector<std::uint8_t> encoded =
      encodeFabricImportedModuleTargetRef(expected);
  require(test, encoded.size() == 20,
          "target must contain one u64 and one typed entity reference");
  require(test,
          take(test, decodeFabricImportedModuleTargetRef(encoded)) == expected,
          "target changed during canonical round trip");

  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  expectRejected(test, decodeFabricImportedModuleTargetRef(trailing),
                 "accepted noncanonical trailing bytes");
}

void checkSystemStructuralRelations() {
  constexpr llvm::StringLiteral test = "System structural relations";
  const FabricImportedModuleBoundaryEndpointRef moduleEndpoint{
      3, FabricModuleBoundaryEndpointRef{FabricModuleTemplateRef(17),
                                         FabricPortDirection::Input, 0}};
  const std::vector<std::uint8_t> encodedModuleEndpoint =
      encodeFabricImportedModuleBoundaryEndpointRef(moduleEndpoint);
  require(test,
          take(test, decodeFabricImportedModuleBoundaryEndpointRef(
                         encodedModuleEndpoint)) == moduleEndpoint,
          "module boundary endpoint changed during canonical round trip");

  const FabricTransportEndpointRef localTransportEndpoint{
      FabricTransportEndpointOwnerRef::of(
          SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(23)}),
      0};
  const FabricSpatialAttachmentEndpointRef localEndpoint = take(
      test, FabricSpatialAttachmentEndpointRef::create(localTransportEndpoint));
  const std::vector<std::uint8_t> encodedLocalEndpoint =
      encodeFabricSpatialAttachmentEndpointRef(localEndpoint);
  require(test,
          take(test, decodeFabricSpatialAttachmentEndpointRef(
                         encodedLocalEndpoint)) == localEndpoint,
          "SpatialCore endpoint changed during canonical round trip");

  expectRejected(
      test,
      FabricSpatialAttachmentEndpointRef::create(FabricTransportEndpointRef{
          FabricTransportEndpointOwnerRef::of(AccCoreOccurrenceRef(23)), 0}),
      "accepted an AccCore endpoint outside its SpatialCore occurrence");

  const FabricTransportEndpointRef source{
      FabricTransportEndpointOwnerRef::of(SystemTransportResourceRef(31)), 0};
  const FabricTransportEndpointRef destination{
      FabricTransportEndpointOwnerRef::of(
          SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(23)}),
      1};

  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  const std::string sourceText =
      "module {\n"
      "  fabric.system @soc {\n"
      "    fabric.system.connection source = " +
      denseI8Assembly(context, canonicalFabricBytes(source)) +
      " destination = " +
      denseI8Assembly(context, canonicalFabricBytes(destination)) +
      "\n"
      "    fabric.system.spatial_attachment module_endpoint = " +
      denseI8Assembly(context, encodedModuleEndpoint) +
      " spatial_endpoint = " + denseI8Assembly(context, encodedLocalEndpoint) +
      "\n"
      "  }\n"
      "}\n";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  require(test, static_cast<bool>(module),
          "typed System relations did not parse:\n" + sourceText);
  require(test, mlir::succeeded(mlir::verify(*module)),
          "typed System relations did not verify");
}

void checkSystemTransportResource() {
  constexpr llvm::StringLiteral test = "System transport resource";
  const SystemTransportResourceRef resource(31);
  const FabricTransportEndpointOwnerRef owner =
      FabricTransportEndpointOwnerRef::of(resource);
  const FabricUsePatternOwnerRef useOwner(
      FabricInventoryOwnerRef::of(resource));
  const SystemTransferPatternRecord pattern =
      take(test, SystemTransferPatternRecord::create(
                     FabricTransferPatternRef{resource, 0},
                     FabricTransportEndpointRef{owner, 0},
                     {FabricTransportEndpointRef{owner, 1},
                      FabricTransportEndpointRef{owner, 2}},
                     FabricUsePatternRef{useOwner, 0}));
  const std::vector<std::uint8_t> encodedPattern =
      encodeSystemTransferPatternRecord(pattern);
  require(test,
          take(test, decodeSystemTransferPatternRecord(encodedPattern)) ==
              pattern,
          "transfer pattern changed during canonical round trip");

  expectRejected(test,
                 SystemTransferPatternRecord::create(
                     FabricTransferPatternRef{resource, 0},
                     FabricTransportEndpointRef{owner, 0},
                     {FabricTransportEndpointRef{owner, 1},
                      FabricTransportEndpointRef{owner, 1}},
                     FabricUsePatternRef{useOwner, 0}),
                 "accepted duplicate multicast egresses");

  const std::vector<std::uint8_t> resourceContract = take(
      test,
      ::fabric::encodeResourceContractRecord(instructionContextContract()));
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  const std::string sourceText =
      "module {\n"
      "  fabric.system @soc {\n"
      "    fabric.system.transport_resource ports = "
      "(!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) "
      "contract = " +
      denseI8Assembly(context, resourceContract) +
      " {entity_id = #fabric.entity_id<31>}\n"
      "    fabric.system.transfer_pattern contract = " +
      denseI8Assembly(context, encodedPattern) +
      "\n"
      "  }\n"
      "}\n";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  require(test, static_cast<bool>(module),
          "typed transport resource did not parse:\n" + sourceText);
  require(test, mlir::succeeded(mlir::verify(*module)),
          "typed transport resource did not verify");
}

void checkInstructionMicroarchitectureContract() {
  constexpr llvm::StringLiteral test = "instruction microarchitecture contract";
  InstructionCoreMicroarchitecturalRealization inOrder =
      representativeMicroarchitecture();
  require(test, inOrder.kind() == InstructionCoreRealizationKind::InOrder,
          "lost in-order realization variant");
  require(test,
          inOrder.executionUnits().size() == 2 &&
              inOrder.executionUnits().front().operationClass ==
                  InstructionOperationClass::IntegerAlu &&
              inOrder.executionUnits().front().count == 3,
          "execution-unit normalization is not canonical");

  std::vector<std::uint8_t> encoded =
      take(test, encodeInstructionCoreMicroarchitecturalRealization(inOrder));
  InstructionCoreMicroarchitecturalRealization decoded =
      take(test, decodeInstructionCoreMicroarchitecturalRealization(encoded));
  require(test,
          take(test, encodeInstructionCoreMicroarchitecturalRealization(
                         decoded)) == encoded,
          "microarchitecture bytes changed after strict import");

  OutOfOrderMicroarchitectureDeclaration outOfOrderPipeline{
      4, 4, 4, 4, 4, 4, 4, 64, 24, 16, 16, 96, 64, 64};
  InstructionCoreCommonDeclaration outOfOrderCommon{
      1,
      {{InstructionOperationClass::IntegerAlu, 4, 1, 1},
       {InstructionOperationClass::LoadStore, 2, 3, 1}},
      instructionContextContract()};
  InstructionCoreMicroarchitecturalRealization outOfOrder =
      take(test, InstructionCoreMicroarchitecturalRealization::createOutOfOrder(
                     std::move(outOfOrderCommon), outOfOrderPipeline));
  require(test,
          outOfOrder.kind() == InstructionCoreRealizationKind::OutOfOrder &&
              take(test, encodeInstructionCoreMicroarchitecturalRealization(
                             outOfOrder)) != encoded,
          "out-of-order realization did not retain distinct identity bytes");
}

std::string denseI8Assembly(mlir::MLIRContext &context,
                            llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::DenseI8ArrayAttr::get(&context, signedBytes).print(stream);
  return text;
}

void checkTypedSystemRoot() {
  constexpr llvm::StringLiteral test = "typed System root";
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry);

  const std::vector<std::uint8_t> architecture = take(
      test,
      encodeInstructionCoreArchitecturalContract(representativeArchitecture()));
  const std::vector<std::uint8_t> microarchitecture =
      take(test, encodeInstructionCoreMicroarchitecturalRealization(
                     representativeMicroarchitecture()));
  const std::string source =
      "module {\n"
      "  fabric.system @soc {\n"
      "    fabric.system.host_core architecture = " +
      denseI8Assembly(context, architecture) +
      " microarchitecture = " + denseI8Assembly(context, microarchitecture) +
      "\n"
      "  }\n"
      "}\n";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  require(test, static_cast<bool>(module),
          "typed System root did not parse:\n" + source);
  require(test, mlir::succeeded(mlir::verify(*module)),
          "typed System root did not verify");
}

void checkTypedSystemRejections() {
  constexpr llvm::StringLiteral test = "typed System rejection";
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  mlir::ScopedDiagnosticHandler diagnostics(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });

  const std::vector<std::uint8_t> dispatcherArchitecture = take(
      test,
      encodeInstructionCoreArchitecturalContract(representativeArchitecture()));
  std::vector<std::uint8_t> malformedArchitecture = dispatcherArchitecture;
  malformedArchitecture[3] = 1;
  const std::vector<std::uint8_t> hostArchitecture = take(
      test, encodeInstructionCoreArchitecturalContract(
                representativeArchitecture(/*acceleratorDispatcher=*/false)));
  const std::vector<std::uint8_t> microarchitecture =
      take(test, encodeInstructionCoreMicroarchitecturalRealization(
                     representativeMicroarchitecture()));
  const std::vector<std::uint8_t> spatialCore =
      encodeFabricImportedModuleTargetRef(
          FabricImportedModuleTargetRef{0, FabricModuleTemplateRef(1)});

  const std::string malformed =
      "module { fabric.system @soc { "
      "fabric.system.host_core architecture = " +
      denseI8Assembly(context, malformedArchitecture) +
      " microarchitecture = " + denseI8Assembly(context, microarchitecture) +
      " } }";
  require(test, !mlir::parseSourceString<mlir::ModuleOp>(malformed, &context),
          "accepted a malformed InstructionCore architecture record");

  const std::string missingServices =
      "module { fabric.system @soc { "
      "fabric.system.acc_core architecture = " +
      denseI8Assembly(context, hostArchitecture) +
      " microarchitecture = " + denseI8Assembly(context, microarchitecture) +
      " spatial_core = " + denseI8Assembly(context, spatialCore) + " } }";
  require(test,
          !mlir::parseSourceString<mlir::ModuleOp>(missingServices, &context),
          "accepted an AccCore without dispatch and launch services");

  const std::string host =
      "fabric.system.host_core architecture = " +
      denseI8Assembly(context, dispatcherArchitecture) +
      " microarchitecture = " + denseI8Assembly(context, microarchitecture) +
      " {entity_id = #fabric.entity_id<9>} ";
  const std::string duplicateIds =
      "module { fabric.system @soc { " + host + host + "} }";
  require(test,
          !mlir::parseSourceString<mlir::ModuleOp>(duplicateIds, &context),
          "accepted duplicate Fabric EntityIds");

  const std::string genericNode =
      "module { fabric.system @soc { \"fabric.node\"() : () -> () } }";
  require(test, !mlir::parseSourceString<mlir::ModuleOp>(genericNode, &context),
          "accepted the retired generic fabric.node operation");
}

void checkClockContract() {
  constexpr llvm::StringLiteral test = "clock contract";
  const ClockDomainContractRecord clock =
      take(test, ClockDomainContractRecord::create(1'000, 125));
  require(test, clock.periodFs() == 1'000 && clock.phaseFs() == 125,
          "lost clock fields");

  expectRejected(test, ClockDomainContractRecord::create(0, 0),
                 "accepted zero period");
  expectRejected(test, ClockDomainContractRecord::create(10, 10),
                 "accepted phase equal to period");

  const std::vector<std::uint8_t> encoded =
      take(test, encodeClockDomainContractRecord(clock));
  require(test, take(test, decodeClockDomainContractRecord(encoded)) == clock,
          "clock roundtrip changed the record");

  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  expectRejected(test, decodeClockDomainContractRecord(trailing),
                 "accepted trailing clock bytes");
}

void checkResetContract() {
  constexpr llvm::StringLiteral test = "reset contract";
  const ResetDomainContractRecord asynchronous =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveLow, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Asserted,
                     std::nullopt, 0));
  const ResetDomainContractRecord synchronousRelease =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Synchronous, ResetInitialState::Deasserted,
                     clockA(), 0));
  const ResetDomainContractRecord synchronousAssertion =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Synchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     clockA(), 2));

  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     clockA(), 0),
                 "accepted a clock on a fully asynchronous reset");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Synchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 0),
                 "accepted synchronous assertion without a clock");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 1),
                 "accepted clock-measured latency without a clock");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     static_cast<ResetPolarity>(2), ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 0),
                 "accepted an unknown reset polarity");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, static_cast<ResetTiming>(2),
                     ResetTiming::Asynchronous, ResetInitialState::Deasserted,
                     std::nullopt, 0),
                 "accepted an unknown reset timing");
  expectRejected(test,
                 ResetDomainContractRecord::create(
                     ResetPolarity::ActiveHigh, ResetTiming::Asynchronous,
                     ResetTiming::Asynchronous,
                     static_cast<ResetInitialState>(2), std::nullopt, 0),
                 "accepted an unknown reset initial state");

  const std::vector<std::uint8_t> encoded =
      take(test, encodeResetDomainContractRecord(synchronousRelease));
  require(test,
          take(test, decodeResetDomainContractRecord(encoded)) ==
              synchronousRelease,
          "reset roundtrip changed the record");

  const std::vector<std::uint8_t> asyncBytes =
      take(test, encodeResetDomainContractRecord(asynchronous));
  require(test,
          take(test, decodeResetDomainContractRecord(asyncBytes)) ==
              asynchronous,
          "asynchronous reset roundtrip changed the record");

  const std::vector<std::uint8_t> assertionBytes =
      take(test, encodeResetDomainContractRecord(synchronousAssertion));
  require(test,
          take(test, decodeResetDomainContractRecord(assertionBytes)) ==
              synchronousAssertion,
          "clocked release latency changed during roundtrip");
}

void checkClockCrossingContract() {
  constexpr llvm::StringLiteral test = "clock crossing contract";
  const ClockCrossingContractRecord crossing =
      take(test, ClockCrossingContractRecord::createAsyncFifo(
                     patternA(), clockA(), clockB(), 8, 2));

  expectRejected(test,
                 ClockCrossingContractRecord::createAsyncFifo(
                     patternA(), clockA(), clockB(), 0, 2),
                 "accepted zero FIFO depth");
  expectRejected(test,
                 ClockCrossingContractRecord::createAsyncFifo(
                     patternA(), clockA(), clockB(), 8, 0),
                 "accepted zero synchronizer stages");

  const std::vector<std::uint8_t> encoded =
      take(test, encodeClockCrossingContractRecord(crossing));
  require(test,
          take(test, decodeClockCrossingContractRecord(encoded)) == crossing,
          "crossing roundtrip changed the record");

  FabricByteWriter unknownVariant;
  unknownVariant.tag(1);
  expectRejected(test, decodeClockCrossingContractRecord(unknownVariant.take()),
                 "accepted an unknown crossing variant");

  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  expectRejected(test, decodeClockCrossingContractRecord(trailing),
                 "accepted trailing crossing bytes");
}

void checkHardwareDomainContract() {
  constexpr llvm::StringLiteral test = "hardware domain contract";
  const FabricInventoryOwnerRef host =
      FabricInventoryOwnerRef::of(HostCoreOccurrenceRef(8));
  const FabricInventoryOwnerRef accelerator =
      FabricInventoryOwnerRef::of(AccCoreOccurrenceRef(3));
  const std::vector<FabricInventoryOwnerRef> members = {host, accelerator};

  const ClockDomainContractRecord clock =
      take(test, ClockDomainContractRecord::create(1'000, 0));
  const ResetDomainContractRecord reset =
      take(test, ResetDomainContractRecord::create(
                     ResetPolarity::ActiveLow, ResetTiming::Asynchronous,
                     ResetTiming::Synchronous, ResetInitialState::Asserted,
                     ClockDomainRef(HardwareDomainRef(20)), 2));
  const PowerDomainContractRecord power =
      take(test, PowerDomainContractRecord::create(900'000));

  llvm::APInt fullAddressLimit(65, 1);
  fullAddressLimit <<= 64;
  const AddressDomainContractRecord address =
      take(test, AddressDomainContractRecord::create(
                     64, {{llvm::APInt(65, 0), fullAddressLimit}}));

  ::fabric::MemoryConsistencyContractDeclaration consistencyDeclaration{
      {::fabric::MemoryConsistencyParticipant::service(
          FabricMemoryServiceRef::system(SystemMemoryServiceRef(30)))},
      ::fabric::ReleaseVisibilityPoint::AtLinearization,
      ::fabric::BoundedCompletion{ClockDomainRef(HardwareDomainRef(20)), 16},
      instructionContextContract()};
  const ::fabric::MemoryConsistencyContract consistency =
      take(test, ::fabric::MemoryConsistencyContract::create(
                     std::move(consistencyDeclaration)));

  std::vector<std::pair<FabricHardwareDomainKind, HardwareDomainContract>>
      contracts;
  contracts.emplace_back(FabricHardwareDomainKind::Clock, clock);
  contracts.emplace_back(FabricHardwareDomainKind::Reset, reset);
  contracts.emplace_back(FabricHardwareDomainKind::Power, power);
  contracts.emplace_back(FabricHardwareDomainKind::Address, address);
  contracts.emplace_back(FabricHardwareDomainKind::MemoryConsistency,
                         consistency);

  std::vector<std::uint8_t> clockBytes;
  for (auto &[expectedKind, contract] : contracts) {
    HardwareDomainContractRecord record =
        take(test, HardwareDomainContractRecord::create(members,
                                                        std::move(contract)));
    require(test, record.kind() == expectedKind,
            "closed contract variant lost its domain kind");
    std::vector<std::uint8_t> encoded =
        take(test, encodeHardwareDomainContractRecord(record));
    HardwareDomainContractRecord decoded =
        take(test, decodeHardwareDomainContractRecord(encoded));
    require(test,
            decoded.kind() == expectedKind &&
                take(test, encodeHardwareDomainContractRecord(decoded)) ==
                    encoded,
            "hardware domain changed during strict roundtrip");
    if (expectedKind == FabricHardwareDomainKind::Clock)
      clockBytes = std::move(encoded);
  }

  expectRejected(test,
                 HardwareDomainContractRecord::create(
                     {host, host}, HardwareDomainContract(clock)),
                 "accepted duplicate domain membership");

  const AddressDomainContractRecord merged =
      take(test, AddressDomainContractRecord::create(
                     32, {{llvm::APInt(33, 16), llvm::APInt(33, 32)},
                          {llvm::APInt(33, 0), llvm::APInt(33, 16)}}));
  require(
      test,
      merged.ranges().size() == 1 && merged.ranges().front().lower.isZero() &&
          merged.ranges().front().upperExclusive == llvm::APInt(33, 32),
      "adjacent address ranges did not canonicalize to one half-open range");

  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  const std::string source = "module { fabric.system @soc { "
                             "fabric.system.hardware_domain contract = " +
                             denseI8Assembly(context, clockBytes) +
                             " {entity_id = #fabric.entity_id<20>} } }";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  require(test,
          static_cast<bool>(module) && mlir::succeeded(mlir::verify(*module)),
          "typed hardware-domain operation did not parse and verify");
}

} // namespace

int main() {
  checkImportedModuleTargetReference();
  checkSystemStructuralRelations();
  checkSystemTransportResource();
  checkInstructionArchitectureContract();
  checkInstructionMicroarchitectureContract();
  checkTypedSystemRoot();
  checkTypedSystemRejections();
  checkClockContract();
  checkResetContract();
  checkClockCrossingContract();
  checkHardwareDomainContract();
  return EXIT_SUCCESS;
}
