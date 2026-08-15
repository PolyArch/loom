#include "Runtime/RuntimePlatformBinding.h"

#include "ConfigurationABITestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/Implementation/RepresentationFormat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::hardware;
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
    fail(test, "accepted invalid input; expected error containing '" +
                   expected.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return std::vector<std::uint8_t>(value.bytes_begin(), value.bytes_end());
}

llvm::Error validateOneByte(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "payload must contain one byte");
  return llvm::Error::success();
}

const RuntimeProviderDescriptor &provider() {
  static const RuntimeProviderEndpointKindDescriptor endpoints[] = {
      {0, "identity", RuntimeEndpointClass::Identity,
       RuntimeEndpointFlow::ImplementationToRuntime, false, validateOneByte},
      {1, "programming", RuntimeEndpointClass::Programming,
       RuntimeEndpointFlow::Bidirectional, false, validateOneByte},
      {2, "completion_input", RuntimeEndpointClass::Completion,
       RuntimeEndpointFlow::RuntimeToImplementation, false, validateOneByte},
      {3, "completion_output", RuntimeEndpointClass::Completion,
       RuntimeEndpointFlow::ImplementationToRuntime, false, validateOneByte},
  };
  static const RuntimeProviderDescriptor descriptor{
      {"loom.runtime.test_provider", SchemaVersion{1, 0}},
      "loom.runtime.test_implementation.v1",
      "loom.runtime_abi.v1",
      endpoints,
      true,
      true,
      false};
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

ImplementationRepresentationRoot makeRepresentation(llvm::StringRef test,
                                                    const BlobStore &blobs) {
  const BlobDigest source =
      take(test, blobs.put(bytes("module top(input logic a); endmodule\n")));
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::Rtl, std::nullopt, format,
                        {RepresentationObjectKind::Module, "top"},
                        {{PayloadRole::RtlSource, "rtl/top.sv", source}}));
}

struct Fixture final {
  loom::fabric::FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  FinalizedHardwareImplementation implementation;
};

Fixture makeFixture(llvm::StringRef test, const ArtifactStore &artifacts,
                    const BlobStore &blobs) {
  const auto module = makeModule(test, artifacts);
  auto system = take(
      test, hardware::test::makeSingleSpatialCoreSystem(module, artifacts));
  auto abiDraft =
      take(test, hardware::test::makeCompleteConfigurationABIDraft(system));
  auto abi =
      take(test, finalizeConfigurationABI(std::move(abiDraft), artifacts));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));

  std::vector<ImplementationInterface> interfaces;
  const RepresentationLocator port{RepresentationObjectKind::Port, "top.a"};
  for (const loom::fabric::FabricSpatialAttachmentRecordView &attachment :
       systemView.spatialAttachments()) {
    if (attachment.spatialEndpoint.plane() ==
        loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
      interfaces.push_back(
          {ImplementationDataInterfaceRef{attachment.spatialEndpoint}, port,
           std::nullopt});
  }
  for (const ProgrammingUnit &unit : abi.abi().programmingUnits())
    interfaces.push_back({ImplementationConfigurationInterfaceRef{
                              ProgrammingUnitRef{abi.reference(), unit.id}},
                          port, std::nullopt});
  auto implementation = take(
      test, finalizeHardwareImplementation(
                HardwareImplementationDraft{system.reference(),
                                            take(test,
                                                 hardware::test::
                                                     requireSingleSpatialCoreOccurrence(
                                                         system)),
                                            abi.reference(),
                                            makeRepresentation(test, blobs),
                                            std::nullopt,
                                            std::move(interfaces),
                                            {},
                                            {},
                                            {}},
                artifacts, blobs));
  return Fixture{std::move(system), std::move(abi), std::move(implementation)};
}

RuntimePlatformBindingDraft makeDraft(llvm::StringRef test,
                                      const Fixture &fixture,
                                      const ArtifactStore &artifacts,
                                      bool reverse = false) {
  auto system =
      take(test, loom::fabric::requireSystemRoot(fixture.system.view()));
  RuntimePlatformBindingDraft draft{fixture.implementation.reference(),
                                    runtimeProviderDescriptorRef(provider()),
                                    HardwareReportedIdentity{{0, {0x42}}},
                                    {},
                                    {},
                                    {}};
  for (const auto &[ordinal, interface] :
       llvm::enumerate(fixture.implementation.implementation().interfaces())) {
    ArtifactReference<HardwareImplementationInterfaceRef> reference{
        fixture.implementation.reference().artifact,
        HardwareImplementationInterfaceRef{ordinal}};
    const std::uint8_t payload = static_cast<std::uint8_t>(ordinal + 1);
    if (const auto *configuration =
            std::get_if<ImplementationConfigurationInterfaceRef>(
                &interface.semanticRef)) {
      draft.programmingBindings.push_back(
          {configuration->programmingUnit, reference, {1, {payload}}});
      continue;
    }
    if (const auto *data = std::get_if<ImplementationDataInterfaceRef>(
            &interface.semanticRef)) {
      const auto direction = system.artifact().transportEndpointDirection(
          *data->endpoint.transport());
      require(test, direction.has_value(), "Data endpoint has no direction");
      const std::uint32_t kind =
          *direction == loom::fabric::FabricPortDirection::Input ? 2 : 3;
      draft.completionInterfaceBindings.push_back(
          {reference, {kind, {payload}}});
    }
  }
  if (reverse) {
    std::reverse(draft.programmingBindings.begin(),
                 draft.programmingBindings.end());
    std::reverse(draft.completionInterfaceBindings.begin(),
                 draft.completionInterfaceBindings.end());
  }
  return draft;
}

void roundTripAndCanonicalOrder(const Fixture &fixture,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs) {
  const auto forward = take(
      __func__, finalizeRuntimePlatformBinding(
                    makeDraft(__func__, fixture, artifacts), artifacts, blobs));
  const auto reverse =
      take(__func__, finalizeRuntimePlatformBinding(
                         makeDraft(__func__, fixture, artifacts, true),
                         artifacts, blobs));
  require(__func__, forward.reference() == reverse.reference(),
          "authoring order changed RuntimePlatformBinding identity");
  const auto imported =
      take(__func__,
           importRuntimePlatformBinding(forward.reference(), artifacts, blobs));
  require(__func__,
          imported.binding().programmingBindings().size() ==
                  fixture.abi.abi().programmingUnits().size() &&
              imported.binding().completionInterfaceBindings().size() +
                      imported.binding().programmingBindings().size() ==
                  fixture.implementation.implementation().interfaces().size(),
          "roundtrip lost exact interface coverage");
}

void rejectsCoverageAndDirectionErrors(const Fixture &fixture,
                                       const ArtifactStore &artifacts,
                                       const BlobStore &blobs) {
  RuntimePlatformBindingDraft missing = makeDraft(__func__, fixture, artifacts);
  require(__func__, !missing.completionInterfaceBindings.empty(),
          "fixture has no completion interface");
  missing.completionInterfaceBindings.pop_back();
  expectError(
      __func__,
      finalizeRuntimePlatformBinding(std::move(missing), artifacts, blobs),
      "coverage is not exact");

  RuntimePlatformBindingDraft foreign = makeDraft(__func__, fixture, artifacts);
  foreign.programmingBindings.front().implementationInterface.artifact =
      fixture.abi.reference().artifact;
  expectError(
      __func__,
      finalizeRuntimePlatformBinding(std::move(foreign), artifacts, blobs),
      "foreign owner");

  RuntimePlatformBindingDraft wrongDirection =
      makeDraft(__func__, fixture, artifacts);
  RuntimeInterfaceBinding &binding =
      wrongDirection.completionInterfaceBindings.front();
  binding.providerEndpoint.kind = binding.providerEndpoint.kind == 2 ? 3 : 2;
  expectError(__func__,
              finalizeRuntimePlatformBinding(std::move(wrongDirection),
                                             artifacts, blobs),
              "flow does not match");

  RuntimePlatformBindingDraft shared = makeDraft(__func__, fixture, artifacts);
  require(__func__, shared.completionInterfaceBindings.size() > 1,
          "fixture cannot exercise endpoint sharing");
  const auto input = llvm::find_if(shared.completionInterfaceBindings,
                                   [](const RuntimeInterfaceBinding &item) {
                                     return item.providerEndpoint.kind == 2;
                                   });
  const auto secondInput =
      input == shared.completionInterfaceBindings.end()
          ? shared.completionInterfaceBindings.end()
          : std::find_if(std::next(input),
                         shared.completionInterfaceBindings.end(),
                         [](const RuntimeInterfaceBinding &item) {
                           return item.providerEndpoint.kind == 2;
                         });
  require(__func__, secondInput != shared.completionInterfaceBindings.end(),
          "fixture has fewer than two input interfaces");
  secondInput->providerEndpoint = input->providerEndpoint;
  expectError(
      __func__,
      finalizeRuntimePlatformBinding(std::move(shared), artifacts, blobs),
      "ambiguously more than once");
}

void validatesTrustedIdentityAndProviderProjection(
    const Fixture &fixture, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  RuntimePlatformBindingDraft trusted = makeDraft(__func__, fixture, artifacts);
  const BlobDigest attestation =
      take(__func__, blobs.put(bytes("trusted implementation attestation")));
  trusted.identityVerification = TrustedImmutableIdentity{attestation};
  (void)take(__func__, finalizeRuntimePlatformBinding(std::move(trusted),
                                                      artifacts, blobs));

  RuntimePlatformBindingDraft missing = makeDraft(__func__, fixture, artifacts);
  missing.identityVerification =
      TrustedImmutableIdentity{computeBlobDigest(bytes("missing"))};
  expectError(
      __func__,
      finalizeRuntimePlatformBinding(std::move(missing), artifacts, blobs),
      "attestation blob is unavailable");

  const auto valid = take(
      __func__, finalizeRuntimePlatformBinding(
                    makeDraft(__func__, fixture, artifacts), artifacts, blobs));
  std::string json(
      reinterpret_cast<const char *>(valid.canonicalBytes().bytes().data()),
      valid.canonicalBytes().bytes().size());
  const std::string oldIdentity = "loom.runtime.test_implementation.v1";
  const std::size_t position = json.find(oldIdentity);
  require(__func__, position != std::string::npos,
          "canonical provider projection is absent");
  json.replace(position, oldIdentity.size(),
               "loom.runtime.fake_implementation.v1");
  CanonicalSemanticBytes mutated(bytes(json));
  const ArtifactIdentity identity =
      take(__func__, artifacts.put(runtimePlatformBindingSchema, mutated));
  expectError(__func__,
              importRuntimePlatformBinding(
                  {runtimePlatformBindingSchema.identity.str(),
                   runtimePlatformBindingSchema.version, identity},
                  artifacts, blobs),
              "disagrees with its descriptor");
}

} // namespace

int main() {
  llvm::SmallString<128> root;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-runtime-platform-binding-test", root))
    fail("main", error.message());
  const std::filesystem::path path(root.str().str());
  std::filesystem::create_directories(path / "artifacts");
  std::filesystem::create_directories(path / "blobs");
  const ArtifactStore artifacts((path / "artifacts").string());
  const BlobStore blobs((path / "blobs").string());

  if (llvm::Error error = registerRuntimeProvider(provider()))
    fail("main", llvm::toString(std::move(error)));
  const Fixture fixture = makeFixture("main", artifacts, blobs);
  roundTripAndCanonicalOrder(fixture, artifacts, blobs);
  rejectsCoverageAndDirectionErrors(fixture, artifacts, blobs);
  validatesTrustedIdentityAndProviderProjection(fixture, artifacts, blobs);

  std::filesystem::remove_all(path);
  return 0;
}
