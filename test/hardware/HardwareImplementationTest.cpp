#include "Hardware/Implementation/HardwareImplementation.h"

#include "ConfigurationABI3TestSupport.h"

#include "ADG/Builder.h"
#include "ADG/MemoryLibrary.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementationLocalReference.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::hardware;

static_assert(implementationInterfaceSemanticRefKindOrdinal(
                  ImplementationInterfaceSemanticRefKind::Data) == 0);
static_assert(implementationInterfaceSemanticRefKindOrdinal(
                  ImplementationInterfaceSemanticRefKind::Memory) == 1);
static_assert(implementationInterfaceSemanticRefKindOrdinal(
                  ImplementationInterfaceSemanticRefKind::Clock) == 2);
static_assert(implementationInterfaceSemanticRefKindOrdinal(
                  ImplementationInterfaceSemanticRefKind::Reset) == 3);
static_assert(implementationInterfaceSemanticRefKindOrdinal(
                  ImplementationInterfaceSemanticRefKind::Configuration) == 4);
static_assert(implementationInterfaceSemanticRefKindOrdinal(
                  ImplementationInterfaceSemanticRefKind::ExternalProtocol) ==
              5);

static_assert(hardwareImplementationLocalReferenceKindOrdinal(
                  HardwareImplementationLocalReferenceKind::Interface) == 0);
static_assert(hardwareImplementationLocalReferenceKindOrdinal(
                  HardwareImplementationLocalReferenceKind::ActivityPoint) ==
              1);
static_assert(hardwareImplementationLocalReferenceKindOrdinal(
                  HardwareImplementationLocalReferenceKind::
                      ExternalImplementationBinding) == 2);
static_assert(hardwareImplementationLocalReferenceKindCount() == 3);

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
    fail(test, "accepted invalid input; expected error containing '" +
                   expected.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid input; expected error containing '" +
                   expected.str() + "'");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return std::vector<std::uint8_t>(value.bytes_begin(), value.bytes_end());
}

void writeFile(llvm::StringRef test, const std::filesystem::path &path,
               llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    fail(test, "could not open dependency fixture for writing");
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail(test, "could not write dependency fixture");
}

std::string readFile(llvm::StringRef test, const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(test, "could not open dependency fixture for reading");
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

ExternalFileFingerprint fingerprint(llvm::StringRef value) {
  const std::vector<std::uint8_t> input = bytes(value);
  return take(__func__,
              ExternalFileFingerprint::fromBytes(llvm::SHA256::hash(input)));
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
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
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
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

struct Fixture final {
  loom::fabric::FinalizedFabricRoot module;
  loom::fabric::FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef firstOwner;
  loom::fabric::FabricSpatialAttachmentEndpointRef firstDataEndpoint;
  ProgrammingUnitRef firstProgrammingUnit;
};

struct MemoryFixture final {
  loom::fabric::FinalizedFabricRoot module;
  loom::fabric::FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef memory;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef nonMemory;
};

::fabric::UnsignedDomain singletonUnsigned(std::uint64_t value) {
  return take(__func__,
              ::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

MemoryFixture makeMemoryFixture(llvm::StringRef test,
                                const ArtifactStore &artifacts,
                                std::uint64_t spatialCoreCount = 1) {
  loom::adg::LocalMemoryParameters parameters;
  parameters.capacityBytes = 4096;
  parameters.interface = {loom::adg::MemoryAccessDomainParameters{
                              128, 128, 16, singletonUnsigned(64)},
                          128, 128};
  auto memory =
      take(test, loom::adg::makeGeneral64LocalMemory(std::move(parameters)));
  loom::adg::DesignBuilder design(artifacts);
  auto spatial = take(test, design.createSpatialCore("implementation-memory",
                                                     memory.inputTypes(),
                                                     memory.outputTypes()));
  std::vector<loom::adg::SpatialValue> inputs;
  inputs.reserve(memory.inputTypes().size());
  for (std::size_t ordinal = 0; ordinal < memory.inputTypes().size(); ++ordinal)
    inputs.push_back(take(test, spatial.input(ordinal)));
  auto outputs = take(test, spatial.addMemory(inputs, memory));
  if (llvm::Error error = spatial.close(outputs.values()))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "memory fixture did not publish exactly one Module");
  auto module = std::move(finalized.roots().front());
  require(test, module.view().memoryOccurrences().size() == 1,
          "memory fixture did not publish exactly one memory occurrence");

  auto system = take(test, hardware::test::makeSpatialCoreSystem(
                               module, artifacts, spatialCoreCount));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test,
          systemView.artifact().accCoreOccurrences().size() == spatialCoreCount,
          "memory fixture System has the wrong accelerator core count");
  auto schema = take(test, module.view().memoryConfigurationSchema(
                               module.view().memoryOccurrences().front()));
  std::vector<hardware::test::ConfigurationFieldEncodingOverride> overrides;
  overrides.reserve(spatialCoreCount);
  for (loom::fabric::AccCoreOccurrenceRef core :
       systemView.artifact().accCoreOccurrences()) {
    auto target = take(
        test,
        loom::fabric::FabricModulePhysicalTargetRef::create(schema.field()));
    auto field =
        take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                       loom::fabric::SpatialCoreInternalOccurrenceRef{
                           loom::fabric::SpatialCoreOccurrenceRef{core},
                           std::move(target)}));
    overrides.push_back(
        {std::move(field), DirectBitsEncoding{schema.layout().carrierBitCount},
         std::vector<std::uint8_t>((schema.layout().carrierBitCount + 7) / 8,
                                   0)});
  }
  auto abiDraft = take(test, hardware::test::makeCompleteConfigurationABIDraft(
                                 system, overrides));
  auto abi =
      take(test, finalizeConfigurationABI(std::move(abiDraft), artifacts));
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().back()};
  auto localMemory =
      take(test, loom::fabric::FabricModulePhysicalOwnerRef::create(
                     module.view().memoryOccurrences().front()));
  auto target = take(test, loom::fabric::FabricModulePhysicalTargetRef::create(
                               std::move(localMemory)));
  auto physicalMemory =
      take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                     loom::fabric::SpatialCoreInternalOccurrenceRef{
                         spatialCore, std::move(target)}));
  auto nonMemory =
      take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                     loom::fabric::FabricInventoryOwnerRef::of(
                         systemView.artifact().accCoreOccurrences().front())));
  return MemoryFixture{std::move(module), std::move(system), std::move(abi),
                       std::move(physicalMemory), std::move(nonMemory)};
}

Fixture makeFixture(llvm::StringRef test, const ArtifactStore &artifacts,
                    std::uint64_t spatialCoreCount = 1) {
  auto module = makeModule(test, artifacts);
  auto system = take(test, hardware::test::makeSpatialCoreSystem(
                               module, artifacts, spatialCoreCount));
  auto abiDraft =
      take(test, hardware::test::makeCompleteConfigurationABIDraft(system));
  auto abi =
      take(test, finalizeConfigurationABI(std::move(abiDraft), artifacts));
  require(test, !abi.abi().programmingUnits().empty(),
          "fixture ABI has no programming units");
  const ProgrammingUnit &unit = abi.abi().programmingUnits().front();
  require(test, !unit.exactFabricResourceClosure.empty(),
          "fixture ABI has no physical owners");
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test, !systemView.spatialAttachments().empty(),
          "fixture System has no spatial attachment endpoints");
  const auto endpoint = systemView.spatialAttachments().front().spatialEndpoint;
  require(
      test,
      endpoint.plane() ==
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport,
      "fixture boundary endpoint is not transport");
  const auto firstOwner = unit.exactFabricResourceClosure.front();
  ProgrammingUnitRef programmingUnit{abi.reference(), unit.id};
  return Fixture{std::move(module), std::move(system),
                 std::move(abi),    firstOwner,
                 endpoint,          std::move(programmingUnit)};
}

ImplementationRepresentationRoot
makeRepresentation(llvm::StringRef test, const BlobStore &blobs,
                   bool withExternalDefinition = false) {
  const llvm::StringRef rtl = withExternalDefinition
                                  ? "module top(input logic a); vendor_cell "
                                    "u_vendor(); endmodule\n"
                                  : "module top(input logic a); endmodule\n";
  const BlobDigest rtlDigest = take(test, blobs.put(bytes(rtl)));
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::RtlSource, "rtl/top.sv", rtlDigest}};
  if (withExternalDefinition) {
    const BlobDigest contract =
        take(test, blobs.put(bytes("vendor_cell black box\n")));
    payloads.push_back(
        {PayloadRole::BlackBoxContract, "contracts/vendor_cell.txt", contract});
  }
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::Rtl, std::nullopt, format,
                        {RepresentationObjectKind::Module, "top"},
                        std::move(payloads)));
}

ImplementationRepresentationRoot
makeGateRepresentation(llvm::StringRef test, const BlobStore &blobs,
                       bool withExternalDefinition = false) {
  const llvm::StringRef netlist =
      withExternalDefinition
          ? "module top(a); input a; vendor_cell u_vendor(); endmodule\n"
          : "module top(a); input a; endmodule\n";
  const BlobDigest netlistDigest = take(test, blobs.put(bytes(netlist)));
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::Netlist, "netlist/top.v", netlistDigest}};
  if (withExternalDefinition) {
    const BlobDigest contract =
        take(test, blobs.put(bytes("vendor_cell black box\n")));
    payloads.push_back(
        {PayloadRole::BlackBoxContract, "contracts/vendor_cell.txt", contract});
  }
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::StructuralVerilogGateNetlist));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::GateNetlist, std::nullopt,
                        format, {RepresentationObjectKind::Module, "top"},
                        std::move(payloads)));
}

platform::FinalizedImplementationPlatform
makeAsicPlatform(llvm::StringRef test, const ArtifactStore &artifacts,
                 llvm::StringRef technology = "saed14",
                 llvm::StringRef release = "EDK_08_2025",
                 llvm::StringRef corner = "tt_0p80v_25c") {
  return take(test,
              platform::finalizeImplementationPlatform(
                  platform::ImplementationPlatformDraft{
                      platform::AsicTarget{technology.str(), release.str()},
                      {corner.str()}},
                  artifacts));
}

ImplementationRepresentationRoot
makeMemoryRepresentation(llvm::StringRef test, const BlobStore &blobs) {
  const BlobDigest rtlDigest =
      take(test, blobs.put(bytes("module top(input logic a); logic [31:0] mem "
                                 "[0:255]; endmodule\n")));
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::Rtl, std::nullopt, format,
                        {RepresentationObjectKind::Module, "top"},
                        {{PayloadRole::RtlSource, "rtl/top.sv", rtlDigest}}));
}

ImplementationRepresentationRoot
makeInstanceRepresentation(llvm::StringRef test, const BlobStore &blobs) {
  const BlobDigest rtlDigest =
      take(test, blobs.put(bytes("module leaf(); endmodule\n"
                                 "module top(input logic a); leaf u0(); leaf "
                                 "u1(); endmodule\n")));
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::Rtl, std::nullopt, format,
                        {RepresentationObjectKind::Module, "top"},
                        {{PayloadRole::RtlSource, "rtl/top.sv", rtlDigest}}));
}

ImplementationRepresentationRoot
makeAsicPhysicalRepresentation(llvm::StringRef test, const BlobStore &blobs) {
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::IndexedPhysical));
  const ImplementationPayload database{
      PayloadRole::PhysicalDatabase, "database/state.bin",
      take(test, blobs.put(bytes("authored synthetic physical state")))};
  const RepresentationLocator top{RepresentationObjectKind::PhysicalObject,
                                  "chip"};
  const PhysicalRepresentationIndexPayload index = take(
      test,
      createPhysicalRepresentationIndexPayload(
          format, RepresentationRootVariant::AsicPhysical,
          RepresentationPhysicalStage::Placed, top, "index/physical.json",
          {database},
          {{{RepresentationObjectKind::Port, "chip.data"},
            RepresentationSignalGeometry{RepresentationSignalDirection::Input,
                                         128}},
           {{RepresentationObjectKind::Net, "chip.activity"}, std::nullopt},
           {{RepresentationObjectKind::Memory, "chip.memory"}, std::nullopt},
           {{RepresentationObjectKind::Cell, "chip.external"}, std::nullopt},
           {{RepresentationObjectKind::PhysicalObject, "chip"}, std::nullopt}},
          {}));
  const std::string indexBytes =
      take(test, serializePhysicalRepresentationIndexPayloadJson(index));
  const ImplementationPayload indexPayload{
      PayloadRole::RepresentationIndex, index.indexLogicalName,
      take(test, blobs.put(bytes(indexBytes)))};
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::AsicPhysical,
                        RepresentationPhysicalStage::Placed, format, top,
                        {database, indexPayload}));
}

HardwareImplementationDraft basicDraft(llvm::StringRef test,
                                       const Fixture &fixture,
                                       const BlobStore &blobs) {
  return HardwareImplementationDraft{
      fixture.system.reference(),
      fixture.abi.reference(),
      {},
      makeRepresentation(test, blobs),
      std::nullopt,
      {{ImplementationDataInterfaceRef{fixture.firstDataEndpoint},
        {RepresentationObjectKind::Port, "top.a"},
        std::nullopt},
       {ImplementationConfigurationInterfaceRef{fixture.firstProgrammingUnit},
        {RepresentationObjectKind::Port, "top.a"},
        std::nullopt}},
      {{{RepresentationObjectKind::Module, "top"}, fixture.firstOwner}},
      {},
      {}};
}

ExternalImplementationContractCatalog makeExternalCatalog() {
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          "vendor.cell",
          {{"library",
            {ExternalDependencyKind::ExplicitFile,
             ExternalDependencyKind::ToolBundledResource}}},
          {RepresentationRootVariant::Rtl},
          true,
          false,
          nullptr}))
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          "vendor.memory",
          {{"library", {ExternalDependencyKind::ExplicitFile}}},
          {RepresentationRootVariant::Rtl},
          false,
          true,
          nullptr}))
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          "vendor.nonmemory",
          {{"library", {ExternalDependencyKind::ExplicitFile}}},
          {RepresentationRootVariant::Rtl},
          false,
          false,
          nullptr}))
    fail(__func__, llvm::toString(std::move(error)));
  return catalog;
}

HardwareImplementationDraft memoryDraft(llvm::StringRef test,
                                        const MemoryFixture &fixture,
                                        const BlobStore &blobs,
                                        bool reverseAuthoringOrder = false) {
  const RepresentationLocator port{RepresentationObjectKind::Port, "top.a"};
  const RepresentationLocator memory{RepresentationObjectKind::Memory,
                                     "top.mem"};
  std::vector<ExternalImplementationBindingDraft> bindings{
      {"vendor.memory",
       {{"library", ExplicitFileDependency{fingerprint("unused.lib")}}},
       {fixture.memory},
       {port},
       std::nullopt},
      {"vendor.memory",
       {{"library", ExplicitFileDependency{fingerprint("memory.lib")}}},
       {fixture.memory},
       {memory},
       std::nullopt}};
  std::uint64_t memoryBindingIndex = 1;
  if (reverseAuthoringOrder) {
    std::reverse(bindings.begin(), bindings.end());
    memoryBindingIndex = 0;
  }
  return HardwareImplementationDraft{
      fixture.system.reference(),
      fixture.abi.reference(),
      {},
      makeMemoryRepresentation(test, blobs),
      std::nullopt,
      {},
      {},
      {{fixture.memory, memoryBindingIndex, memory}},
      std::move(bindings)};
}

HardwareImplementationDraft externalDraft(llvm::StringRef test,
                                          const Fixture &fixture,
                                          const BlobStore &blobs) {
  HardwareImplementationDraft draft = basicDraft(test, fixture, blobs);
  draft.representationRoot = makeRepresentation(test, blobs, true);
  draft.externalImplementationBindings.push_back(
      ExternalImplementationBindingDraft{
          "vendor.cell",
          {{"library", ExplicitFileDependency{fingerprint("vendor.lib")}}},
          {fixture.firstOwner},
          {{RepresentationObjectKind::Module, "vendor_cell"}},
          ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                   "contracts/vendor_cell.txt"}});
  return draft;
}

void systemRootAndTypedRepresentationRoundTrip(const ArtifactStore &artifacts,
                                               const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  HardwareImplementationDraft firstDraft = basicDraft(__func__, fixture, blobs);
  HardwareImplementationDraft secondDraft =
      basicDraft(__func__, fixture, blobs);
  std::reverse(secondDraft.representationRoot.payloads.begin(),
               secondDraft.representationRoot.payloads.end());
  auto canonical = createImplementationRepresentationRoot(
      secondDraft.representationRoot.variant,
      secondDraft.representationRoot.stage,
      secondDraft.representationRoot.formatRef,
      secondDraft.representationRoot.top,
      secondDraft.representationRoot.payloads);
  secondDraft.representationRoot = take(__func__, std::move(canonical));

  const FinalizedHardwareImplementation first =
      take(__func__, finalizeHardwareImplementation(std::move(firstDraft),
                                                    artifacts, blobs));
  const FinalizedHardwareImplementation second =
      take(__func__, finalizeHardwareImplementation(std::move(secondDraft),
                                                    artifacts, blobs));
  require(__func__, first.reference() == second.reference(),
          "authoring order changed HardwareImplementation identity");
  require(__func__,
          first.reference().schemaVersion == SchemaVersion{3, 0} &&
              first.implementation().fabric() == fixture.system.reference() &&
              first.implementation().representationRoot().variant ==
                  RepresentationRootVariant::Rtl,
          "finalized root did not retain the exact schema-3.0 owners");
  const FinalizedHardwareImplementation imported =
      take(__func__,
           importHardwareImplementation(first.reference(), artifacts, blobs));
  require(__func__,
          imported.canonicalBytes().bytes() == first.canonicalBytes().bytes(),
          "strict import changed canonical bytes");
}

void nonRtlRepresentationRequiresPlatform(const ArtifactStore &artifacts,
                                          const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  HardwareImplementationDraft draft = basicDraft(__func__, fixture, blobs);
  draft.representationRoot = makeGateRepresentation(__func__, blobs);
  expectError(
      __func__,
      finalizeHardwareImplementation(std::move(draft), artifacts, blobs),
      "non-RTL representation requires an implementation platform");
}

void targetSpecializationChangesIdentity(const ArtifactStore &artifacts,
                                         const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const platform::FinalizedImplementationPlatform firstPlatform =
      makeAsicPlatform(__func__, artifacts);
  const platform::FinalizedImplementationPlatform secondPlatform =
      makeAsicPlatform(__func__, artifacts, "saed05", "EDK_06_2026",
                       "ss_0p65v_125c");
  require(__func__, firstPlatform.reference() != secondPlatform.reference(),
          "distinct platform fixture inputs converged to one identity");
  const FinalizedHardwareImplementation portable = take(
      __func__, finalizeHardwareImplementation(
                    basicDraft(__func__, fixture, blobs), artifacts, blobs));
  HardwareImplementationDraft firstDraft = basicDraft(__func__, fixture, blobs);
  firstDraft.implementationPlatform = firstPlatform.reference();
  const FinalizedHardwareImplementation first =
      take(__func__, finalizeHardwareImplementation(std::move(firstDraft),
                                                    artifacts, blobs));
  HardwareImplementationDraft secondDraft =
      basicDraft(__func__, fixture, blobs);
  secondDraft.implementationPlatform = secondPlatform.reference();
  const FinalizedHardwareImplementation second =
      take(__func__, finalizeHardwareImplementation(std::move(secondDraft),
                                                    artifacts, blobs));
  require(__func__,
          portable.reference() != first.reference() &&
              portable.reference() != second.reference() &&
              first.reference() != second.reference(),
          "exact target specialization did not distinguish implementation "
          "identity");
  require(__func__,
          first.implementation().implementationPlatform() ==
                  firstPlatform.reference() &&
              second.implementation().implementationPlatform() ==
                  secondPlatform.reference(),
          "target specialization did not retain its exact platform reference");
}

void configurationAbiRequiresExactFabric(const ArtifactStore &artifacts,
                                         const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const Fixture other = makeFixture(__func__, artifacts, 2);
  require(__func__, fixture.system.reference() != other.system.reference(),
          "mismatch fixture did not produce a distinct Fabric System");
  HardwareImplementationDraft mismatched = basicDraft(__func__, fixture, blobs);
  mismatched.configurationAbi = other.abi.reference();
  expectError(
      __func__,
      finalizeHardwareImplementation(std::move(mismatched), artifacts, blobs),
      "ConfigurationABI must describe the same Fabric System");
}

void typedInterfaceReferencesRemainPhysical(const ArtifactStore &artifacts,
                                            const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts, 2);
  HardwareImplementationDraft bare = basicDraft(__func__, fixture, blobs);
  bare.fabric = fixture.module.reference();
  expectError(__func__,
              finalizeHardwareImplementation(std::move(bare), artifacts, blobs),
              "System root");

  HardwareImplementationDraft foreign = basicDraft(__func__, fixture, blobs);
  require(__func__, fixture.abi.abi().programmingUnits().size() == 2,
          "two-core fixture did not produce two programming units");
  const auto &otherClosure =
      fixture.abi.abi().programmingUnits().back().exactFabricResourceClosure;
  require(__func__,
          !otherClosure.empty() && otherClosure.front() != fixture.firstOwner,
          "two-core fixture collapsed physical occurrences");
  auto systemView =
      take(__func__, loom::fabric::requireSystemRoot(fixture.system.view()));
  require(__func__, systemView.spatialAttachments().size() >= 2,
          "two-core fixture did not preserve physical boundary occurrences");
  const auto otherEndpoint =
      systemView.spatialAttachments().back().spatialEndpoint;
  foreign.interfaces.front().semanticRef =
      ImplementationDataInterfaceRef{otherEndpoint};
  const FinalizedHardwareImplementation distinct =
      take(__func__, finalizeHardwareImplementation(std::move(foreign),
                                                    artifacts, blobs));
  require(__func__,
          std::get<ImplementationDataInterfaceRef>(
              distinct.implementation().interfaces().front().semanticRef)
                  .endpoint == otherEndpoint,
          "physical boundary was reduced to a Module-local reference");

  HardwareImplementationDraft wrongPlane = basicDraft(__func__, fixture, blobs);
  wrongPlane.interfaces.front().semanticRef =
      ImplementationMemoryInterfaceRef{fixture.firstDataEndpoint};
  expectError(
      __func__,
      finalizeHardwareImplementation(std::move(wrongPlane), artifacts, blobs),
      "Memory interface target is not on the Memory plane");

  const Fixture otherFixture = makeFixture(__func__, artifacts);
  HardwareImplementationDraft foreignUnit =
      basicDraft(__func__, fixture, blobs);
  foreignUnit.interfaces.back().semanticRef =
      ImplementationConfigurationInterfaceRef{
          otherFixture.firstProgrammingUnit};
  expectError(
      __func__,
      finalizeHardwareImplementation(std::move(foreignUnit), artifacts, blobs),
      "Configuration interface references a foreign ABI");
}

void externalBindingsUseDerivedDenseIdentity(const ArtifactStore &artifacts,
                                             const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();
  HardwareImplementationDraft single = externalDraft(__func__, fixture, blobs);
  HardwareImplementationDraft duplicate =
      externalDraft(__func__, fixture, blobs);
  duplicate.externalImplementationBindings.push_back(
      duplicate.externalImplementationBindings.front());
  const FinalizedHardwareImplementation first =
      take(__func__, finalizeHardwareImplementation(std::move(single), catalog,
                                                    artifacts, blobs));
  const FinalizedHardwareImplementation second =
      take(__func__, finalizeHardwareImplementation(std::move(duplicate),
                                                    catalog, artifacts, blobs));
  require(__func__, first.reference() == second.reference(),
          "duplicate complete binding changed canonical identity");
  require(__func__,
          first.implementation().externalImplementationBindings().size() == 1 &&
              first.implementation()
                      .externalImplementationBindings()
                      .front()
                      .blackBoxContractPayloadRef ==
                  ImplementationPayloadRef{1},
          "external binding did not derive canonical dense references");
}

void externalBindingsRejectIncompleteContracts(const ArtifactStore &artifacts,
                                               const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();

  HardwareImplementationDraft missingSlot =
      externalDraft(__func__, fixture, blobs);
  missingSlot.externalImplementationBindings.front().externalInputs.clear();
  expectError(__func__,
              finalizeHardwareImplementation(std::move(missingSlot), catalog,
                                             artifacts, blobs),
              "provider input slot closure is incomplete");

  HardwareImplementationDraft missingBlackBox =
      externalDraft(__func__, fixture, blobs);
  missingBlackBox.externalImplementationBindings.front()
      .blackBoxContractPayload.reset();
  expectError(__func__,
              finalizeHardwareImplementation(std::move(missingBlackBox),
                                             catalog, artifacts, blobs),
              "provider contract requires a BlackBoxContract payload");

  const platform::FinalizedImplementationPlatform platform =
      makeAsicPlatform(__func__, artifacts);
  HardwareImplementationDraft incompatible =
      externalDraft(__func__, fixture, blobs);
  incompatible.representationRoot =
      makeGateRepresentation(__func__, blobs, true);
  incompatible.implementationPlatform = platform.reference();
  expectError(__func__,
              finalizeHardwareImplementation(std::move(incompatible), catalog,
                                             artifacts, blobs),
              "provider contract does not support the representation");

  HardwareImplementationDraft unusedBlackBox =
      externalDraft(__func__, fixture, blobs);
  ExternalImplementationBindingDraft extra =
      unusedBlackBox.externalImplementationBindings.front();
  extra.representationLocators = {{RepresentationObjectKind::Module, "top"}};
  unusedBlackBox.externalImplementationBindings.push_back(std::move(extra));
  expectError(__func__,
              finalizeHardwareImplementation(std::move(unusedBlackBox), catalog,
                                             artifacts, blobs),
              "black-box binding closes no indexed definition");
}

void externalDependencyIdentityExcludesMachinePaths(
    const std::filesystem::path &root, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();
  const std::filesystem::path firstPath = root / "host-a" / "vendor_cell.lib";
  const std::filesystem::path secondPath = root / "host-b" / "renamed.lib";
  const llvm::StringRef contents = "library bytes shared across hosts\n";
  writeFile(__func__, firstPath, contents);
  writeFile(__func__, secondPath, contents);

  HardwareImplementationDraft firstDraft =
      externalDraft(__func__, fixture, blobs);
  firstDraft.externalImplementationBindings.front()
      .externalInputs.front()
      .dependencyIdentity =
      ExplicitFileDependency{fingerprint(readFile(__func__, firstPath))};
  const FinalizedHardwareImplementation first =
      take(__func__, finalizeHardwareImplementation(std::move(firstDraft),
                                                    catalog, artifacts, blobs));
  HardwareImplementationDraft secondDraft =
      externalDraft(__func__, fixture, blobs);
  secondDraft.externalImplementationBindings.front()
      .externalInputs.front()
      .dependencyIdentity =
      ExplicitFileDependency{fingerprint(readFile(__func__, secondPath))};
  const FinalizedHardwareImplementation second =
      take(__func__, finalizeHardwareImplementation(std::move(secondDraft),
                                                    catalog, artifacts, blobs));
  require(__func__, first.reference() == second.reference(),
          "machine-local dependency paths changed implementation identity");
  const llvm::ArrayRef<std::uint8_t> explicitBytes =
      first.canonicalBytes().bytes();
  const llvm::StringRef explicitJson(
      reinterpret_cast<const char *>(explicitBytes.data()),
      explicitBytes.size());
  require(__func__,
          explicitJson.contains(
              formatExternalFileFingerprint(fingerprint(contents))) &&
              !explicitJson.contains(firstPath.string()) &&
              !explicitJson.contains(secondPath.string()),
          "explicit dependency did not retain only its content identity");

  writeFile(__func__, secondPath, "different library bytes\n");
  HardwareImplementationDraft changedDraft =
      externalDraft(__func__, fixture, blobs);
  changedDraft.externalImplementationBindings.front()
      .externalInputs.front()
      .dependencyIdentity =
      ExplicitFileDependency{fingerprint(readFile(__func__, secondPath))};
  const FinalizedHardwareImplementation changed =
      take(__func__, finalizeHardwareImplementation(std::move(changedDraft),
                                                    catalog, artifacts, blobs));
  require(__func__, first.reference() != changed.reference(),
          "changed dependency contents did not change implementation identity");

  HardwareImplementationDraft bundledDraft =
      externalDraft(__func__, fixture, blobs);
  bundledDraft.externalImplementationBindings.front()
      .externalInputs.front()
      .dependencyIdentity = ToolBundledResourceDependency{
      "synopsys.vcs:Y-2026.03-SP1", "designware:vendor_cell"};
  const FinalizedHardwareImplementation bundledImplementation =
      take(__func__, finalizeHardwareImplementation(std::move(bundledDraft),
                                                    catalog, artifacts, blobs));
  require(__func__, first.reference() != bundledImplementation.reference(),
          "explicit and tool-bundled dependencies converged to one identity");
}

template <typename Ref>
void requireLocalReferenceRoundTrip(
    llvm::StringRef test, const FinalizedHardwareImplementation &owner,
    Ref target, HardwareImplementationLocalReferenceKind expectedKind,
    llvm::ArrayRef<std::uint8_t> expectedPayload) {
  const ArtifactReference<Ref> typed{owner.reference().artifact, target};
  const EncodedArtifactLocalReference encoded =
      encodeHardwareImplementationLocalReference(typed);
  require(test, encoded.artifact == owner.reference(),
          "local reference lost its exact HardwareImplementation owner");
  require(test,
          encoded.ownerLocalKind ==
              hardwareImplementationLocalReferenceKindOrdinal(expectedKind),
          "local reference encoded the wrong owner-local kind");
  require(test,
          llvm::ArrayRef<std::uint8_t>(encoded.payload) == expectedPayload,
          "local reference ordinal is not canonical u64be");
  const ArtifactReference<Ref> decoded =
      take(test, decodeHardwareImplementationLocalReference<Ref>(encoded));
  require(test, decoded == typed,
          "typed HardwareImplementation local reference did not round-trip");
  if (llvm::Error error =
          validateHardwareImplementationLocalReference(owner, encoded))
    fail(test, llvm::toString(std::move(error)));
}

void ownerLocalReferencesAreExactAndBounded(const ArtifactStore &artifacts,
                                            const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();
  const FinalizedHardwareImplementation owner = take(
      __func__,
      finalizeHardwareImplementation(externalDraft(__func__, fixture, blobs),
                                     catalog, artifacts, blobs));

  const std::vector<std::uint8_t> zeroPayload(8, 0);
  const std::vector<std::uint8_t> onePayload{0, 0, 0, 0, 0, 0, 0, 1};
  requireLocalReferenceRoundTrip(
      __func__, owner, HardwareImplementationInterfaceRef{1},
      HardwareImplementationLocalReferenceKind::Interface, onePayload);
  requireLocalReferenceRoundTrip(
      __func__, owner, HardwareImplementationActivityPointRef{0},
      HardwareImplementationLocalReferenceKind::ActivityPoint, zeroPayload);
  requireLocalReferenceRoundTrip(
      __func__, owner, ExternalImplementationBindingRef{0},
      HardwareImplementationLocalReferenceKind::ExternalImplementationBinding,
      zeroPayload);

  const EncodedArtifactLocalReference endian =
      encodeHardwareImplementationLocalReference(
          ArtifactReference<HardwareImplementationInterfaceRef>{
              owner.reference().artifact,
              HardwareImplementationInterfaceRef{0x0102030405060708ULL}});
  require(__func__,
          endian.payload == std::vector<std::uint8_t>({1, 2, 3, 4, 5, 6, 7, 8}),
          "owner-local ordinal payload is not u64be");

  EncodedArtifactLocalReference wrongKind =
      encodeHardwareImplementationLocalReference(
          ArtifactReference<HardwareImplementationActivityPointRef>{
              owner.reference().artifact,
              HardwareImplementationActivityPointRef{0}});
  expectError(__func__,
              decodeHardwareImplementationLocalReference<
                  HardwareImplementationInterfaceRef>(wrongKind),
              "does not encode HardwareImplementationInterfaceRef");

  EncodedArtifactLocalReference unknownKind = wrongKind;
  unknownKind.ownerLocalKind = 3;
  expectError(__func__,
              validateHardwareImplementationLocalReference(owner, unknownKind),
              "unknown HardwareImplementation owner-local reference kind 3");

  EncodedArtifactLocalReference wrongSchema = wrongKind;
  wrongSchema.artifact.schemaVersion = SchemaVersion{1, 0};
  expectError(__func__,
              decodeHardwareImplementationLocalReference<
                  HardwareImplementationActivityPointRef>(wrongSchema),
              "loom.hardware_implementation 3.0");

  EncodedArtifactLocalReference malformed = wrongKind;
  malformed.payload.pop_back();
  expectError(__func__,
              decodeHardwareImplementationLocalReference<
                  HardwareImplementationActivityPointRef>(malformed),
              "exactly eight bytes");

  const FinalizedHardwareImplementation foreign = take(
      __func__, finalizeHardwareImplementation(
                    basicDraft(__func__, fixture, blobs), artifacts, blobs));
  require(__func__, foreign.reference() != owner.reference(),
          "foreign-reference fixture did not change implementation identity");
  EncodedArtifactLocalReference foreignReference = wrongKind;
  foreignReference.artifact = foreign.reference();
  expectError(
      __func__,
      validateHardwareImplementationLocalReference(owner, foreignReference),
      "foreign HardwareImplementation");

  const auto expectOutOfRange = [&](const auto &target) {
    expectError(
        __func__,
        validateHardwareImplementationLocalReference(
            owner, encodeHardwareImplementationLocalReference(
                       ArtifactReference<std::decay_t<decltype(target)>>{
                           owner.reference().artifact, target})),
        "out of range");
  };
  expectOutOfRange(HardwareImplementationInterfaceRef{
      static_cast<std::uint64_t>(owner.implementation().interfaces().size())});
  expectOutOfRange(
      HardwareImplementationActivityPointRef{static_cast<std::uint64_t>(
          owner.implementation().activityPoints().size())});
  expectOutOfRange(ExternalImplementationBindingRef{static_cast<std::uint64_t>(
      owner.implementation().externalImplementationBindings().size())});
}

void repeatedModuleImportsKeepPhysicalExternalBindings(
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts, 2);
  require(__func__, fixture.abi.abi().programmingUnits().size() == 2,
          "two-core fixture did not publish two programming units");
  const auto &firstClosure =
      fixture.abi.abi().programmingUnits().front().exactFabricResourceClosure;
  const auto &secondClosure =
      fixture.abi.abi().programmingUnits().back().exactFabricResourceClosure;
  require(__func__,
          !firstClosure.empty() && !secondClosure.empty() &&
              firstClosure.front() != secondClosure.front(),
          "two Module imports collapsed their physical resource owners");
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();
  HardwareImplementationDraft draft = basicDraft(__func__, fixture, blobs);
  draft.representationRoot = makeInstanceRepresentation(__func__, blobs);
  draft.externalImplementationBindings = {
      {"vendor.nonmemory",
       {{"library", ExplicitFileDependency{fingerprint("first.lib")}}},
       {firstClosure.front()},
       {{RepresentationObjectKind::Instance, "top.u0"}},
       std::nullopt},
      {"vendor.nonmemory",
       {{"library", ExplicitFileDependency{fingerprint("second.lib")}}},
       {secondClosure.front()},
       {{RepresentationObjectKind::Instance, "top.u1"}},
       std::nullopt}};
  HardwareImplementationDraft reordered = draft;
  std::reverse(reordered.externalImplementationBindings.begin(),
               reordered.externalImplementationBindings.end());
  HardwareImplementationDraft collapsed = draft;
  collapsed.externalImplementationBindings.back().fabricResourceRefs = {
      firstClosure.front()};

  const FinalizedHardwareImplementation first =
      take(__func__, finalizeHardwareImplementation(std::move(draft), catalog,
                                                    artifacts, blobs));
  const FinalizedHardwareImplementation second =
      take(__func__, finalizeHardwareImplementation(std::move(reordered),
                                                    catalog, artifacts, blobs));
  const FinalizedHardwareImplementation aliased =
      take(__func__, finalizeHardwareImplementation(std::move(collapsed),
                                                    catalog, artifacts, blobs));
  require(__func__, first.reference() == second.reference(),
          "external binding authoring order changed identity");
  require(__func__, first.reference() != aliased.reference(),
          "distinct physical Module imports collapsed to local identity");
  require(__func__,
          first.implementation().externalImplementationBindings().size() == 2,
          "distinct physical external bindings were deduplicated");
}

void memoryMacrosUsePhysicalOccurrencesAndDenseBindings(
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const MemoryFixture fixture = makeMemoryFixture(__func__, artifacts);
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();
  HardwareImplementationDraft forward =
      memoryDraft(__func__, fixture, blobs, false);
  HardwareImplementationDraft reverse =
      memoryDraft(__func__, fixture, blobs, true);
  const FinalizedHardwareImplementation first =
      take(__func__, finalizeHardwareImplementation(std::move(forward), catalog,
                                                    artifacts, blobs));
  const FinalizedHardwareImplementation second =
      take(__func__, finalizeHardwareImplementation(std::move(reverse), catalog,
                                                    artifacts, blobs));
  require(__func__, first.reference() == second.reference(),
          "external binding authoring order changed memory implementation "
          "identity");
  require(__func__, first.implementation().memoryMacroBindings().size() == 1,
          "memory macro binding did not remain unique");
  const MemoryMacroBinding &binding =
      first.implementation().memoryMacroBindings().front();
  require(__func__, binding.fabricMemoryRef == fixture.memory,
          "memory binding lost its physical occurrence");
  const auto external = first.implementation().externalImplementationBindings();
  require(
      __func__,
      binding.externalImplementationBindingRef.ordinal < external.size() &&
          llvm::is_contained(
              external[static_cast<std::size_t>(
                           binding.externalImplementationBindingRef.ordinal)]
                  .representationLocators,
              binding.representationLocator),
      "memory binding draft index was not remapped to the canonical "
      "external binding");

  HardwareImplementationDraft wrongOwner =
      memoryDraft(__func__, fixture, blobs);
  wrongOwner.memoryMacroBindings.front().fabricMemoryRef = fixture.nonMemory;
  expectError(__func__,
              finalizeHardwareImplementation(std::move(wrongOwner), catalog,
                                             artifacts, blobs),
              "not a memory occurrence");

  HardwareImplementationDraft wrongContract =
      memoryDraft(__func__, fixture, blobs);
  wrongContract.externalImplementationBindings[1].providerContractRef =
      "vendor.nonmemory";
  expectError(__func__,
              finalizeHardwareImplementation(std::move(wrongContract), catalog,
                                             artifacts, blobs),
              "provider contract is not memory-capable");
}

void foreignAndDuplicateMemoryBindingsAreRejected(
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const MemoryFixture fixture = makeMemoryFixture(__func__, artifacts);
  const MemoryFixture foreignFixture =
      makeMemoryFixture(__func__, artifacts, 2);
  require(__func__,
          fixture.system.reference() != foreignFixture.system.reference(),
          "foreign memory fixture did not produce a distinct Fabric System");
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();

  HardwareImplementationDraft foreign = memoryDraft(__func__, fixture, blobs);
  foreign.memoryMacroBindings.front().fabricMemoryRef = foreignFixture.memory;
  expectError(__func__,
              finalizeHardwareImplementation(std::move(foreign), catalog,
                                             artifacts, blobs),
              "SpatialCore has no imported Module target");

  HardwareImplementationDraft duplicate = memoryDraft(__func__, fixture, blobs);
  duplicate.memoryMacroBindings.push_back(
      duplicate.memoryMacroBindings.front());
  expectError(__func__,
              finalizeHardwareImplementation(std::move(duplicate), catalog,
                                             artifacts, blobs),
              "memory macro binding is duplicated");
}

void physicalIndexClosesEveryFinalizerLocator(const ArtifactStore &artifacts,
                                              const BlobStore &blobs) {
  const MemoryFixture fixture = makeMemoryFixture(__func__, artifacts);
  const platform::FinalizedImplementationPlatform platform =
      makeAsicPlatform(__func__, artifacts);
  auto systemView =
      take(__func__, loom::fabric::requireSystemRoot(fixture.system.view()));
  require(__func__, !systemView.spatialAttachments().empty(),
          "physical finalizer fixture has no data endpoint");
  const auto dataEndpoint =
      systemView.spatialAttachments().front().spatialEndpoint;

  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          "physical.cell",
          {{"library", {ExternalDependencyKind::ExplicitFile}}},
          {RepresentationRootVariant::AsicPhysical},
          false,
          false,
          nullptr}))
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          "physical.memory",
          {{"library", {ExternalDependencyKind::ExplicitFile}}},
          {RepresentationRootVariant::AsicPhysical},
          false,
          true,
          nullptr}))
    fail(__func__, llvm::toString(std::move(error)));

  const RepresentationLocator data{RepresentationObjectKind::Port, "chip.data"};
  const RepresentationLocator activity{RepresentationObjectKind::Net,
                                       "chip.activity"};
  const RepresentationLocator memory{RepresentationObjectKind::Memory,
                                     "chip.memory"};
  const RepresentationLocator external{RepresentationObjectKind::Cell,
                                       "chip.external"};
  HardwareImplementationDraft draft{
      fixture.system.reference(),
      fixture.abi.reference(),
      {},
      makeAsicPhysicalRepresentation(__func__, blobs),
      platform.reference(),
      {{ImplementationDataInterfaceRef{dataEndpoint}, data, std::nullopt}},
      {{activity, fixture.memory}},
      {{fixture.memory, 1, memory}},
      {{"physical.cell",
        {{"library", ExplicitFileDependency{fingerprint("cell.lib")}}},
        {fixture.nonMemory},
        {external},
        std::nullopt},
       {"physical.memory",
        {{"library", ExplicitFileDependency{fingerprint("memory.lib")}}},
        {fixture.memory},
        {memory},
        std::nullopt}}};

  HardwareImplementationDraft missingInterface = draft;
  missingInterface.interfaces.front().representationLocator.canonicalName =
      "chip.missing";
  expectError(__func__,
              finalizeHardwareImplementation(std::move(missingInterface),
                                             catalog, artifacts, blobs),
              "interface locator is absent");

  const FinalizedHardwareImplementation finalized =
      take(__func__, finalizeHardwareImplementation(std::move(draft), catalog,
                                                    artifacts, blobs));
  require(__func__,
          finalized.implementation().representationRoot().variant ==
                  RepresentationRootVariant::AsicPhysical &&
              finalized.implementation()
                      .interfaces()
                      .front()
                      .representationLocator == data &&
              finalized.implementation()
                      .activityPoints()
                      .front()
                      .representationLocator == activity &&
              finalized.implementation()
                      .memoryMacroBindings()
                      .front()
                      .representationLocator == memory &&
              finalized.implementation()
                      .externalImplementationBindings()
                      .front()
                      .representationLocators.front() == external,
          "physical finalization lost an indexed locator owner");
}

void oldCallerAuthoredBindingIdIsRejected(const ArtifactStore &artifacts,
                                          const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const ExternalImplementationContractCatalog catalog = makeExternalCatalog();
  const FinalizedHardwareImplementation implementation = take(
      __func__,
      finalizeHardwareImplementation(externalDraft(__func__, fixture, blobs),
                                     catalog, artifacts, blobs));
  std::string json(reinterpret_cast<const char *>(
                       implementation.canonicalBytes().bytes().data()),
                   implementation.canonicalBytes().bytes().size());
  const std::string needle = "\"provider_contract_ref\"";
  const std::size_t position = json.find(needle);
  require(__func__, position != std::string::npos,
          "canonical external binding is absent");
  json.insert(position, "\"binding_id\":\"legacy\",");
  CanonicalSemanticBytes mutated(bytes(json));
  const ArtifactIdentity identity =
      take(__func__, artifacts.put(hardwareImplementationSchema, mutated));
  expectError(__func__,
              importHardwareImplementation(
                  {hardwareImplementationSchema.identity.str(),
                   hardwareImplementationSchema.version, identity},
                  catalog, artifacts, blobs),
              "unknown field");
}

void oldCallerAuthoredInterfaceKeyIsRejected(const ArtifactStore &artifacts,
                                             const BlobStore &blobs) {
  const Fixture fixture = makeFixture(__func__, artifacts);
  const FinalizedHardwareImplementation implementation = take(
      __func__, finalizeHardwareImplementation(
                    basicDraft(__func__, fixture, blobs), artifacts, blobs));
  std::string json(reinterpret_cast<const char *>(
                       implementation.canonicalBytes().bytes().data()),
                   implementation.canonicalBytes().bytes().size());
  const std::string needle = "\"semantic_ref\"";
  const std::size_t position = json.find(needle);
  require(__func__, position != std::string::npos,
          "canonical interface is absent");
  json.insert(position, "\"interface_key\":\"legacy\",");
  CanonicalSemanticBytes mutated(bytes(json));
  const ArtifactIdentity identity =
      take(__func__, artifacts.put(hardwareImplementationSchema, mutated));
  expectError(__func__,
              importHardwareImplementation(
                  {hardwareImplementationSchema.identity.str(),
                   hardwareImplementationSchema.version, identity},
                  artifacts, blobs),
              "unknown field");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const ArtifactStore artifacts((root / "artifacts").string());
  const BlobStore blobs((root / "blobs").string());

  systemRootAndTypedRepresentationRoundTrip(artifacts, blobs);
  nonRtlRepresentationRequiresPlatform(artifacts, blobs);
  targetSpecializationChangesIdentity(artifacts, blobs);
  configurationAbiRequiresExactFabric(artifacts, blobs);
  typedInterfaceReferencesRemainPhysical(artifacts, blobs);
  externalBindingsUseDerivedDenseIdentity(artifacts, blobs);
  externalBindingsRejectIncompleteContracts(artifacts, blobs);
  externalDependencyIdentityExcludesMachinePaths(root, artifacts, blobs);
  ownerLocalReferencesAreExactAndBounded(artifacts, blobs);
  repeatedModuleImportsKeepPhysicalExternalBindings(artifacts, blobs);
  memoryMacrosUsePhysicalOccurrencesAndDenseBindings(artifacts, blobs);
  foreignAndDuplicateMemoryBindingsAreRejected(artifacts, blobs);
  physicalIndexClosesEveryFinalizerLocator(artifacts, blobs);
  oldCallerAuthoredBindingIdIsRejected(artifacts, blobs);
  oldCallerAuthoredInterfaceKeyIsRejected(artifacts, blobs);
  return EXIT_SUCCESS;
}
