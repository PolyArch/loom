#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"

#include "ConfigurationABI3TestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "ExternalTool/InvocationBundle.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::dse;
using namespace loom::eda::open_source;
using namespace loom::external_tool;
using namespace loom::hardware;

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
void expectFailure(llvm::StringRef test, llvm::Expected<T> value,
                   llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}
std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return std::vector<std::uint8_t>(value.bytes_begin(), value.bytes_end());
}
std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail(__func__, "cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}
void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail(__func__, "cannot write " + path.string());
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output)
    fail(__func__, "cannot finish writing " + path.string());
}
std::uint64_t regularFileCount(const std::filesystem::path &root) {
  std::uint64_t count = 0;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(root))
    if (entry.is_regular_file())
      ++count;
  return count;
}
ExternalFileFingerprint fingerprint(llvm::StringRef contents) {
  return take(__func__, ExternalFileFingerprint::fromBytes(
                            llvm::SHA256::hash(bytes(contents))));
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
loom::fabric::FinalizedFabricRoot makeModule(const ArtifactStore &store) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @yosys_fixture(
          %a: !fabric.bits<1>) -> !fabric.bits<1> {
        fabric.yield %a : !fabric.bits<1>
      }
    }
  )mlir",
                                                        &context());
  require(__func__, static_cast<bool>(source), "cannot parse Fabric fixture");
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(__func__, static_cast<bool>(root), "Fabric fixture has no root");
  return take(__func__, loom::fabric::finalizeFabricRoot(root, store));
}
struct SemanticFixture final {
  platform::FinalizedImplementationPlatform platform;
  FinalizedHardwareImplementation rtl;
  ResolvedYosysGateNetlistConfigView config;
  std::vector<CandidateGeneratorInputBinding> inputs;
  ResolvedCandidateGeneratorBinding binding;
  ExternalImplementationContractCatalog contracts;
};
SemanticFixture
makeSemanticFixture(const std::filesystem::path &fixtureRoot,
                    const ArtifactStore &artifacts, const BlobStore &blobs,
                    bool externalComponent = false,
                    llvm::StringRef providerBuild = "Yosys 0.67 fixture") {
  auto module = makeModule(artifacts);
  auto system = take(
      __func__, hardware::test::makeSingleSpatialCoreSystem(module, artifacts));
  auto abi = take(
      __func__, finalizeConfigurationABI(
                    ConfigurationABIDraft{system.reference(), {}}, artifacts));
  auto platform =
      take(__func__, platform::finalizeImplementationPlatform(
                         platform::ImplementationPlatformDraft{
                             platform::AsicTarget{"gpdk045", "synthetic-v1"},
                             {"synthetic_slow"}},
                         artifacts));

  std::string rtl = readFile(fixtureRoot / "rtl/top.sv");
  if (externalComponent) {
    const std::size_t insertion = rtl.find("  assign y = ~a;");
    require(__func__, insertion != std::string::npos,
            "RTL fixture has no component insertion anchor");
    rtl.insert(insertion, "  fixture_component u_component();\n");
  }
  const std::string sdc = readFile(fixtureRoot / "constraints/top.sdc");
  const BlobDigest rtlDigest = take(__func__, blobs.put(bytes(rtl)));
  const BlobDigest sdcDigest = take(__func__, blobs.put(bytes(sdc)));
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::RtlSource, "rtl/top.sv", rtlDigest},
      {PayloadRole::GenerationConstraint, "constraints/top.sdc", sdcDigest}};
  std::vector<ExternalImplementationBindingDraft> externalBindings;
  ExternalImplementationContractCatalog contracts;
  if (externalComponent) {
    const std::string blackBox = "fixture_component@1\n";
    payloads.push_back({PayloadRole::BlackBoxContract,
                        "blackbox/fixture-component.txt",
                        take(__func__, blobs.put(bytes(blackBox)))});
    if (llvm::Error error = contracts.add(ExternalImplementationContract{
            "fixture.external.component",
            {{"component_model",
              {ExternalDependencyKind::ToolBundledResource}}},
            {RepresentationRootVariant::Rtl},
            true,
            false,
            nullptr}))
      fail(__func__, llvm::toString(std::move(error)));
    externalBindings.push_back(
        {"fixture.external.component",
         {{"component_model",
           ToolBundledResourceDependency{"fixture_provider_build",
                                         "fixture_component"}}},
         {},
         {{RepresentationObjectKind::Module, "fixture_component"}},
         ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                  "blackbox/fixture-component.txt"}});
  }
  const auto format =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  auto representation =
      take(__func__,
           createImplementationRepresentationRoot(
               RepresentationRootVariant::Rtl, std::nullopt, format,
               {RepresentationObjectKind::Module, "top"}, std::move(payloads)));
  auto implementation =
      take(__func__,
           finalizeHardwareImplementation(
               HardwareImplementationDraft{
                   system.reference(),
                   abi.reference(),
                   {},
                   std::move(representation),
                   platform.reference(),
                   {},
                   {{{RepresentationObjectKind::Port, "top.a"}, std::nullopt}},
                   {},
                   std::move(externalBindings)},
               contracts, artifacts, blobs));

  const std::string liberty = readFile(fixtureRoot / "standard-cell.lib");
  const platform::TechnologyCornerRef corner{platform.reference().artifact,
                                             platform::TechnologyCornerId(0)};
  auto config =
      take(__func__, createResolvedYosysGateNetlistConfigView(
                         providerBuild, corner, fingerprint(liberty)));
  auto inputs =
      take(__func__, bindYosysGateNetlistInputs(implementation.reference(),
                                                platform.reference()));
  auto binding = take(__func__, resolveYosysGateNetlistBinding(config));
  return SemanticFixture{std::move(platform), std::move(implementation),
                         std::move(config),   std::move(inputs),
                         std::move(binding),  std::move(contracts)};
}
ExternalToolPreparationContext
makePreparationContext(const std::filesystem::path &bundle,
                       const std::filesystem::path &fixtureRoot,
                       std::filesystem::path executable = {}) {
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  if (executable.empty())
    executable = fixtureRoot / "fake-yosys";
  local.tools["yosys"].binding.executable =
      std::filesystem::canonical(executable).string();
  local.externalFiles["synthetic_standard_cell_liberty"] =
      std::filesystem::canonical(fixtureRoot / "standard-cell.lib").string();
  return ExternalToolPreparationContext{std::move(local), bundle.string()};
}
PreparedExternalToolInvocation prepare(const std::filesystem::path &bundle,
                                       const std::filesystem::path &fixtureRoot,
                                       const SemanticFixture &fixture,
                                       const ArtifactStore &artifacts,
                                       const BlobStore &blobs,
                                       std::filesystem::path executable = {}) {
  return take(__func__, prepareCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, artifacts, blobs,
                            makePreparationContext(bundle, fixtureRoot,
                                                   std::move(executable))));
}
void descriptorAndConfigAreExact() {
  require(__func__, !registerYosysGateNetlistCandidateGenerator(),
          "Yosys generator registration failed");
  const CandidateGeneratorDescriptor &descriptor =
      yosysGateNetlistCandidateGeneratorDescriptor();
  require(__func__,
          descriptor.kind == yosysGateNetlistCandidateGeneratorKind &&
              descriptor.providerForm == ProviderForm::ExternalPrepareImport &&
              descriptor.determinism ==
                  CandidateGeneratorDeterminism::Deterministic,
          "Yosys descriptor form changed");
  require(__func__,
          descriptor.inputSlots.size() == 2 &&
              descriptor.inputSlots[0].semanticRole ==
                  "finalized_rtl_with_generation_constraints" &&
              descriptor.inputSlots[0].schema ==
                  &hardwareImplementationSchema &&
              descriptor.inputSlots[1].semanticRole == "asic_target" &&
              descriptor.inputSlots[1].schema ==
                  &platform::implementationPlatformSchema,
          "Yosys exact input slots changed");
  require(__func__,
          descriptor.outputSlots.size() == 1 &&
              descriptor.outputSlots.front().schema ==
                  &hardwareImplementationSchema &&
              descriptor.outputSlots.front().semanticRole == "gate_netlist",
          "Yosys GateNetlist output slot changed");
  require(__func__,
          descriptor.workUnits.size() == 1 &&
              descriptor.workUnits.front().spelling == "synthesis_attempt",
          "Yosys work accounting changed");

  const std::string liberty = "synthetic-liberty";
  const platform::TechnologyCornerRef corner{
      take(__func__,
           ArtifactIdentity::fromBytes(
               std::array<std::uint8_t, ArtifactIdentity::byteSize>{})),
      platform::TechnologyCornerId(7)};
  auto config =
      take(__func__, createResolvedYosysGateNetlistConfigView(
                         "Yosys 0.67 fixture", corner, fingerprint(liberty)));
  require(__func__,
          config.stableProviderBuildIdentity() == "Yosys 0.67 fixture" &&
              config.technologyCorner() == corner &&
              config.standardCellLiberty() == fingerprint(liberty),
          "Yosys resolved config lost an exact semantic input");
  expectFailure(__func__,
                createResolvedYosysGateNetlistConfigView("", corner,
                                                         fingerprint(liberty)),
                "provider build identity");
}

void mismatchedBuildCannotPrepare(const std::filesystem::path &root,
                                  const std::filesystem::path &fixtureRoot,
                                  const SemanticFixture &fixture,
                                  const ArtifactStore &artifacts,
                                  const BlobStore &blobs) {
  auto wrongConfig = take(__func__, createResolvedYosysGateNetlistConfigView(
                                        "Yosys 0.67 different build",
                                        fixture.config.technologyCorner(),
                                        fixture.config.standardCellLiberty()));
  auto wrongBinding =
      take(__func__, resolveYosysGateNetlistBinding(wrongConfig));
  expectFailure(__func__,
                prepareCandidateGeneratorInvocation(
                    fixture.inputs, wrongBinding, artifacts, blobs,
                    makePreparationContext(root / "wrong-build", fixtureRoot)),
                "does not match semantic build");
}

void externalRtlIsRejectedBeforePreparation(
    const std::filesystem::path &root, const std::filesystem::path &fixtureRoot,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const SemanticFixture fixture =
      makeSemanticFixture(fixtureRoot, artifacts, blobs, true);
  expectFailure(__func__,
                prepareYosysGateNetlistInvocation(
                    fixture.inputs, fixture.binding, fixture.contracts,
                    artifacts, blobs,
                    makePreparationContext(root / "external-rtl", fixtureRoot)),
                "does not consume external RTL implementation bindings");
  require(__func__, !std::filesystem::exists(root / "external-rtl"),
          "unsupported external RTL mutated the bundle destination");
}

void successfulLifecyclePublishesGateNetlist(
    const std::filesystem::path &root, const std::filesystem::path &fixtureRoot,
    const SemanticFixture &fixture, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  const std::filesystem::path firstBundle = root / "bundle-a";
  const std::filesystem::path secondBundle = root / "bundle-b";
  const PreparedExternalToolInvocation first =
      prepare(firstBundle, fixtureRoot, fixture, artifacts, blobs);
  const PreparedExternalToolInvocation second =
      prepare(secondBundle, fixtureRoot, fixture, artifacts, blobs);
  require(__func__,
          first.manifestDigest == second.manifestDigest &&
              readFile(firstBundle / "drivers/synthesize.ys") ==
                  readFile(secondBundle / "drivers/synthesize.ys"),
          "equivalent preparation changed the bundle");
  require(__func__,
          !std::filesystem::exists(firstBundle / "outputs/netlist.v") &&
              !std::filesystem::exists(firstBundle / "outputs/completion.json"),
          "preparation executed Yosys");

  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(first)) == 0,
          "caller-owned fake Yosys execution failed");
  CandidateGeneratorProviderResult result = take(
      __func__, importCandidateGeneratorInvocation(
                    fixture.inputs, fixture.binding, first, artifacts, blobs));
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  require(__func__,
          completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().artifacts.size() == 1 &&
              completed->lineageEdges.size() == 1 &&
              completed->lineageEdges.front().kind ==
                  CandidateGeneratorLineageEdgeKind::MechanicalDerivation &&
              result.workSummary.size() == 1 &&
              result.workSummary.front().planned == 1 &&
              result.workSummary.front().consumed == 1,
          "Yosys provider result is not one complete mechanical derivation");

  const ArtifactRootReference output =
      completed->outputBindings.front().artifacts.front();
  auto implementation = take(
      __func__, importYosysGateNetlistImplementation(output, artifacts, blobs));
  const ImplementationRepresentationRoot &representation =
      implementation.implementation().representationRoot();
  require(
      __func__,
      representation.variant == RepresentationRootVariant::GateNetlist &&
          representation.formatRef.kind() ==
              RepresentationFormatKind::StructuralVerilogGateNetlist &&
          representation.top ==
              RepresentationLocator{RepresentationObjectKind::Module, "top"},
      "Yosys did not publish the exact GateNetlist state");
  require(__func__,
          representation.payloads.size() == 3 &&
              llvm::count_if(representation.payloads,
                             [](const auto &p) {
                               return p.role == PayloadRole::Netlist;
                             }) == 1 &&
              llvm::count_if(representation.payloads,
                             [](const auto &p) {
                               return p.role ==
                                      PayloadRole::GenerationConstraint;
                             }) == 1 &&
              llvm::count_if(representation.payloads,
                             [](const auto &p) {
                               return p.role == PayloadRole::BlackBoxContract;
                             }) == 1,
          "Yosys output payload closure is incomplete");
  const auto netlist =
      llvm::find_if(representation.payloads, [](const auto &p) {
        return p.role == PayloadRole::Netlist;
      });
  const std::vector<std::uint8_t> importedNetlist =
      take(__func__, blobs.get(netlist->blobDigest));
  require(
      __func__,
      llvm::StringRef(reinterpret_cast<const char *>(importedNetlist.data()),
                      importedNetlist.size()) ==
          readFile(fixtureRoot / "expected/top.v"),
      "published GateNetlist bytes differ from the strict snapshot");
  require(__func__,
          implementation.implementation().implementationPlatform() ==
                  std::optional<ArtifactRootReference>(
                      fixture.platform.reference()) &&
              implementation.implementation()
                      .externalImplementationBindings()
                      .size() == 1 &&
              implementation.implementation().activityPoints() ==
                  fixture.rtl.implementation().activityPoints(),
          "Yosys output lost an exact dependency or activity point");
}

void requireNoPublication(
    llvm::StringRef test,
    llvm::Expected<CandidateGeneratorProviderResult> value,
    const std::filesystem::path &artifactsRoot,
    const std::filesystem::path &blobsRoot, std::uint64_t artifactCount,
    std::uint64_t blobCount) {
  if (value)
    fail(test, "strict import published a result from invalid completion");
  llvm::consumeError(value.takeError());
  require(test,
          regularFileCount(artifactsRoot) == artifactCount &&
              regularFileCount(blobsRoot) == blobCount,
          "failed strict import changed persistent state");
}

void strictImportRejectsInvalidAttempts(
    const std::filesystem::path &root, const std::filesystem::path &fixtureRoot,
    const SemanticFixture &fixture, const ArtifactStore &artifacts,
    const BlobStore &blobs, const std::filesystem::path &artifactsRoot,
    const std::filesystem::path &blobsRoot) {
  const std::uint64_t artifactCount = regularFileCount(artifactsRoot);
  const std::uint64_t blobCount = regularFileCount(blobsRoot);

  const auto import = [&](const PreparedExternalToolInvocation &prepared) {
    return importCandidateGeneratorInvocation(fixture.inputs, fixture.binding,
                                              prepared, artifacts, blobs);
  };

  const auto stale =
      prepare(root / "stale", fixtureRoot, fixture, artifacts, blobs);
  writeFile(root / "stale/outputs/netlist.v",
            readFile(fixtureRoot / "expected/top.v"));
  auto staleResult = import(stale);
  require(__func__, !staleResult, "stale output imported without completion");
  bool incomplete = false;
  llvm::handleAllErrors(
      staleResult.takeError(),
      [&](const IncompleteExternalToolInvocationError &) { incomplete = true; },
      [&](const llvm::ErrorInfoBase &error) {
        fail(__func__, "stale output returned wrong error: " + error.message());
      });
  require(__func__, incomplete, "stale output lost incomplete classification");
  require(__func__,
          regularFileCount(artifactsRoot) == artifactCount &&
              regularFileCount(blobsRoot) == blobCount,
          "stale output changed persistent state");
  const auto tampered =
      prepare(root / "tampered", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(tampered));
  writeFile(root / "tampered/outputs/netlist.v",
            readFile(fixtureRoot / "expected/top.v") + "// tampered\n");
  requireNoPublication(__func__, import(tampered), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto missing =
      prepare(root / "missing", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(missing));
  std::filesystem::remove(root / "missing/outputs/netlist-structure.json");
  requireNoPublication(__func__, import(missing), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto partial =
      prepare(root / "partial", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(partial));
  writeFile(root / "partial/outputs/completion.json", "{");
  requireNoPublication(__func__, import(partial), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto manifest =
      prepare(root / "manifest", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(manifest));
  writeFile(root / "manifest/tool-invocation.json",
            readFile(root / "manifest/tool-invocation.json") + " ");
  requireNoPublication(__func__, import(manifest), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto undeclared =
      prepare(root / "undeclared", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(undeclared));
  std::string completion =
      readFile(root / "undeclared/outputs/completion.json");
  const std::size_t end = completion.find("]}");
  require(__func__, end != std::string::npos,
          "completion fixture has unexpected syntax");
  completion.insert(end, ",\"" + std::string(64, '0') + "\"");
  writeFile(root / "undeclared/outputs/completion.json", completion);
  requireNoPublication(__func__, import(undeclared), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto extra =
      prepare(root / "extra", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(extra));
  writeFile(root / "extra/outputs/undeclared.rpt", "not declared\n");
  requireNoPublication(__func__, import(extra), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);
}

void failedToolIsAnIncompleteCandidate(
    const std::filesystem::path &root, const std::filesystem::path &fixtureRoot,
    const SemanticFixture &fixture, const ArtifactStore &artifacts,
    const BlobStore &blobs, const std::filesystem::path &artifactsRoot,
    const std::filesystem::path &blobsRoot) {
  const std::uint64_t artifactCount = regularFileCount(artifactsRoot);
  const std::uint64_t blobCount = regularFileCount(blobsRoot);
  const PreparedExternalToolInvocation prepared = take(
      __func__, prepareCandidateGeneratorInvocation(
                    fixture.inputs, fixture.binding, artifacts, blobs,
                    makePreparationContext(root / "tool-failure", fixtureRoot,
                                           fixtureRoot / "fake-yosys-fail")));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 7,
          "authored failing Yosys returned the wrong status");
  CandidateGeneratorProviderResult result =
      take(__func__,
           importCandidateGeneratorInvocation(fixture.inputs, fixture.binding,
                                              prepared, artifacts, blobs));
  const auto *incomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&result.outcome);
  require(__func__,
          incomplete &&
              incomplete->reason ==
                  CandidateGeneratorIncompleteReason::ExecutionFailed &&
              incomplete->retainedOutputBindings.size() == 1 &&
              incomplete->retainedOutputBindings.front().artifacts.empty() &&
              result.workSummary.size() == 1 &&
              result.workSummary.front().consumed == 1,
          "failed Yosys did not produce the exact incomplete outcome");
  require(__func__,
          regularFileCount(artifactsRoot) == artifactCount &&
              regularFileCount(blobsRoot) == blobCount,
          "failed Yosys published persistent state");
}

void realYosysPublishesGateNetlist(const std::filesystem::path &root,
                                   const std::filesystem::path &fixtureRoot,
                                   const std::filesystem::path &executable,
                                   llvm::StringRef providerBuild,
                                   const ArtifactStore &artifacts,
                                   const BlobStore &blobs) {
  const SemanticFixture fixture =
      makeSemanticFixture(fixtureRoot, artifacts, blobs, false, providerBuild);
  const PreparedExternalToolInvocation prepared = prepare(
      root / "real-yosys", fixtureRoot, fixture, artifacts, blobs, executable);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "real Yosys synthesis failed");
  CandidateGeneratorProviderResult result =
      take(__func__,
           importCandidateGeneratorInvocation(fixture.inputs, fixture.binding,
                                              prepared, artifacts, blobs));
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  require(__func__,
          completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().artifacts.size() == 1,
          "real Yosys did not publish one GateNetlist candidate");
  const FinalizedHardwareImplementation implementation =
      take(__func__, importYosysGateNetlistImplementation(
                         completed->outputBindings.front().artifacts.front(),
                         artifacts, blobs));
  const ImplementationRepresentationRoot &representation =
      implementation.implementation().representationRoot();
  require(__func__,
          representation.variant == RepresentationRootVariant::GateNetlist &&
              representation.formatRef.kind() ==
                  RepresentationFormatKind::StructuralVerilogGateNetlist &&
              implementation.implementation().implementationPlatform() ==
                  std::optional<ArtifactRootReference>(
                      fixture.platform.reference()) &&
              implementation.implementation().activityPoints() ==
                  fixture.rtl.implementation().activityPoints(),
          "real Yosys publication lost GateNetlist semantic state");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 && argc != 5)
    fail("main", "expected scratch, fixture, and optional Yosys build");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  const std::filesystem::path fixtureRoot =
      std::filesystem::absolute(argv[2]).lexically_normal();
  const std::filesystem::path artifactsRoot = root / "artifacts";
  const std::filesystem::path blobsRoot = root / "blobs";
  std::filesystem::create_directories(artifactsRoot);
  std::filesystem::create_directories(blobsRoot);
  const ArtifactStore artifacts(artifactsRoot.string());
  const BlobStore blobs(blobsRoot.string());

  if (argc == 5) {
    realYosysPublishesGateNetlist(root, fixtureRoot, argv[3], argv[4],
                                  artifacts, blobs);
    return EXIT_SUCCESS;
  }

  descriptorAndConfigAreExact();
  const SemanticFixture fixture =
      makeSemanticFixture(fixtureRoot, artifacts, blobs);
  mismatchedBuildCannotPrepare(root, fixtureRoot, fixture, artifacts, blobs);
  externalRtlIsRejectedBeforePreparation(root, fixtureRoot, artifacts, blobs);
  successfulLifecyclePublishesGateNetlist(root, fixtureRoot, fixture, artifacts,
                                          blobs);
  strictImportRejectsInvalidAttempts(root, fixtureRoot, fixture, artifacts,
                                     blobs, artifactsRoot, blobsRoot);
  failedToolIsAnIncompleteCandidate(root, fixtureRoot, fixture, artifacts,
                                    blobs, artifactsRoot, blobsRoot);
  return EXIT_SUCCESS;
}
