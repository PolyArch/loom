#include "Hardware/Implementation/HardwareImplementation.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactLocalReference.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
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

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition,
             const std::string &message) {
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

loom::fabric::FinalizedFabricRoot makeFabric(llvm::StringRef test,
                                             const ArtifactStore &store,
                                             bool alternate = false) {
  const llvm::StringRef sourceText = alternate ? R"mlir(
    module {
      fabric.module @configured_alternate(
          %a: !fabric.bits<64>, %b: !fabric.bits<64>) -> !fabric.bits<64> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
            -> !fabric.bits<64> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<64>, %fb = %pb : !fabric.bits<64>)
              -> !fabric.bits<64> {
            %value = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [64 : i32]}}
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
            fabric.yield %value : !fabric.bits<64>
          }
        }
        fabric.yield %pe : !fabric.bits<64>
      }
    }
  )mlir"
                                                     : R"mlir(
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
  )mlir";
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedContract(contract.begin(), contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), signedContract));
  });

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

ConfigurationABIDraft
configurationDraft(const loom::fabric::FinalizedFabricRoot &fabric) {
  const auto definitions = fabric.view().fuTemplates();
  if (definitions.size() != 1)
    fail(__func__, "fixture does not have one FU definition");
  const auto capabilities =
      fabric.view().resolvedFabricOpCapabilities(definitions.front());
  if (capabilities.size() != 1 ||
      capabilities.front().configurationFieldSchema.size() != 1)
    fail(__func__, "fixture does not have one configuration field");

  ConfigurationFieldEncoding field{
      capabilities.front().configurationFieldSchema.front(),
      FiniteCodebookEncoding{1, {{{0x00}, {0x00}}, {{0x01}, {0x01}}}},
      {{0, 0, 1}},
      {0x00}};
  ProgrammingUnitDraft unit{{field.field.owner.catalog()}, 8, {field}};
  return ConfigurationABIDraft{fabric.reference(), {std::move(unit)}};
}

struct Fixture final {
  loom::fabric::FinalizedFabricRoot fabric;
  FinalizedConfigurationABI abi;
  BlobDigest rtl;
  BlobDigest constraints;
  EncodedArtifactLocalReference fabricEndpoint;
};

Fixture makeFixture(llvm::StringRef test, const ArtifactStore &artifacts,
                    const BlobStore &blobs) {
  auto fabric = makeFabric(test, artifacts);
  auto abi = take(test,
                  finalizeConfigurationABI(configurationDraft(fabric),
                                           artifacts));
  auto rtl = take(test, blobs.put(bytes("module configured(); endmodule\n")));
  auto constraints = take(test, blobs.put(bytes("create_clock -period 1 clk\n")));
  const auto endpoints = fabric.view().transportEndpoints();
  require(test, !endpoints.empty(), "fixture has no transport endpoint");
  auto endpoint = loom::fabric::encodeFabricArtifactLocalReference(
      ArtifactReference<loom::fabric::FabricTransportEndpointRef>{
          fabric.reference().artifact, endpoints.front()});
  return Fixture{std::move(fabric), std::move(abi), rtl, constraints,
                 std::move(endpoint)};
}

HardwareImplementationDraft rtlDraft(const Fixture &fixture) {
  return HardwareImplementationDraft{
      fixture.fabric.reference(),
      fixture.abi.reference(),
      {},
      HardwareRepresentation::Rtl,
      std::nullopt,
      {{HardwarePayloadRole::RtlSource, "rtl/configured.sv",
        "text/x-systemverilog", fixture.rtl},
       {HardwarePayloadRole::GenerationConstraint, "constraints/main.sdc",
        "application/x-sdc", fixture.constraints}},
      {{"data.input", ImplementationInterfaceRole::Data,
        fixture.fabricEndpoint,
        {RepresentationObjectKind::Port, "configured.a"}, std::nullopt}},
      {{"activity.output", {RepresentationObjectKind::Net, "configured.y"},
        fixture.fabricEndpoint}},
      {},
      {}};
}

void portableRtlRoundTripsCanonically(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  ArtifactStore artifacts((root / "artifacts").string());
  BlobStore blobs((root / "blobs").string());
  Fixture fixture = makeFixture(test, artifacts, blobs);

  HardwareImplementationDraft firstDraft = rtlDraft(fixture);
  HardwareImplementationDraft secondDraft = rtlDraft(fixture);
  std::reverse(secondDraft.payloads.begin(), secondDraft.payloads.end());
  FinalizedHardwareImplementation first = take(
      test, finalizeHardwareImplementation(std::move(firstDraft), artifacts,
                                           blobs));
  FinalizedHardwareImplementation second = take(
      test, finalizeHardwareImplementation(std::move(secondDraft), artifacts,
                                           blobs));

  require(test, first.reference() == second.reference(),
          "payload authoring order changed HardwareImplementation identity");
  require(test, first.implementation().representation() ==
                    HardwareRepresentation::Rtl &&
                    !first.implementation().implementationPlatform(),
          "portable RTL did not preserve its representation boundary");
  require(test, first.implementation().payloads().size() == 2 &&
                    first.implementation().interfaces().size() == 1 &&
                    first.implementation().activityPoints().size() == 1,
          "portable RTL lost semantic closure catalogs");

  FinalizedHardwareImplementation imported = take(
      test, importHardwareImplementation(first.reference(), artifacts, blobs));
  require(test, imported.reference() == first.reference() &&
                    imported.canonicalBytes().bytes() ==
                        first.canonicalBytes().bytes(),
          "strict HardwareImplementation import changed canonical content");

  const llvm::ArrayRef<std::uint8_t> canonical =
      first.canonicalBytes().bytes();
  const llvm::StringRef json(reinterpret_cast<const char *>(canonical.data()),
                             canonical.size());
  require(test, !json.contains("parent") && !json.contains("generator") &&
                    !json.contains("report") && !json.contains("path"),
          "HardwareImplementation copied derivation or local execution state");
}

void platformAndDependencyRulesAreClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  ArtifactStore artifacts((root / "artifacts").string());
  BlobStore blobs((root / "blobs").string());
  Fixture fixture = makeFixture(test, artifacts, blobs);

  HardwareImplementationDraft gate = rtlDraft(fixture);
  gate.representation = HardwareRepresentation::GateNetlist;
  expectError(test,
              finalizeHardwareImplementation(gate, artifacts, blobs),
              "implementation platform");

  auto platform = take(
      test, platform::finalizeImplementationPlatform(
                platform::ImplementationPlatformDraft{
                    platform::AsicTarget{"saed14", "EDK_08_2025"},
                    {"tt_0p80v_25c"}},
                artifacts));
  gate.implementationPlatform = platform.reference();
  expectError(test,
              finalizeHardwareImplementation(gate, artifacts, blobs),
              "Netlist payload");

  HardwareImplementationDraft targetRtl = rtlDraft(fixture);
  targetRtl.implementationPlatform = platform.reference();
  FinalizedHardwareImplementation portable = take(
      test, finalizeHardwareImplementation(rtlDraft(fixture), artifacts,
                                           blobs));
  FinalizedHardwareImplementation targeted = take(
      test, finalizeHardwareImplementation(std::move(targetRtl), artifacts,
                                           blobs));
  require(test, portable.reference() != targeted.reference(),
          "target specialization did not change implementation identity");

  auto otherFabric = makeFabric(test, artifacts, true);
  require(test, otherFabric.reference() != fixture.fabric.reference(),
          "alternate Fabric fixture converged to the same identity");
  auto otherAbi = take(
      test, finalizeConfigurationABI(configurationDraft(otherFabric),
                                     artifacts));
  HardwareImplementationDraft mismatched = rtlDraft(fixture);
  mismatched.configurationAbi = otherAbi.reference();
  expectError(test,
              finalizeHardwareImplementation(std::move(mismatched), artifacts,
                                             blobs),
              "same Fabric");

  HardwareImplementationDraft foreignReference = rtlDraft(fixture);
  foreignReference.interfaces.front().semanticFabricRef.artifact.artifact =
      platform.reference().artifact;
  expectError(test,
              finalizeHardwareImplementation(std::move(foreignReference),
                                             artifacts, blobs),
              "foreign Fabric");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  portableRtlRoundTripsCanonically(root / "roundtrip");
  platformAndDependencyRulesAreClosed(root / "rules");
  return 0;
}
