#include "Runtime/Gem5SimulationBinding.h"

#include "ConfigurationABITestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Artifact/InterconnectImplementation.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Runtime/Gem5BridgeABI.h"

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

ArtifactRootReference makeInterconnect(
    llvm::StringRef test, const ArtifactRootReference &system,
    const ArtifactStore &store) {
  auto refined = take(test, loom::fabric::importEntireFabricRoot(system, store));
  auto builder = take(
      test, loom::fabric::InterconnectImplementationBuilder::create(refined,
                                                                    store));
  auto implementation = take(test, std::move(builder).finalize());
  auto summary = take(
      test, loom::fabric::inspectInterconnectImplementation(implementation));
  require(test, summary.endpointCount != 0,
          "interconnect builder emitted no protocol endpoints");
  require(test, summary.refinementCount != 0,
          "interconnect builder emitted no architecture refinements");
  return implementation.reference();
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
  const auto system = take(
      __func__, loom::hardware::test::makeSingleSpatialCoreSystem(module,
                                                                  artifacts));
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
  auto draft = makeDraft(__func__, system, interconnect);
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

  auto missing = makeDraft(__func__, system, interconnect);
  missing.correspondences.pop_back();
  expectError(__func__,
              finalizeGem5SimulationBinding(std::move(missing), artifacts),
              "does not exactly cover");

  auto shared = makeDraft(__func__, system, interconnect);
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
