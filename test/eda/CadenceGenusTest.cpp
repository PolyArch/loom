#include "EDA/Adapters/Cadence/Genus.h"
#include "EDA/Adapters/Cadence/Innovus.h"

#include "ConfigurationABITestSupport.h"

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
using namespace loom::eda::cadence;
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
      fabric.module @cadence_fixture(
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
  ResolvedGenusGateNetlistConfigView config;
  std::vector<CandidateGeneratorInputBinding> inputs;
  ResolvedCandidateGeneratorBinding binding;
  ExternalImplementationContractCatalog contracts;
};
SemanticFixture
makeSemanticFixture(const std::filesystem::path &fixtureRoot,
                    const ArtifactStore &artifacts, const BlobStore &blobs,
                    std::optional<std::string> componentBuild = std::nullopt) {
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
  if (componentBuild) {
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
  if (componentBuild) {
    const std::string blackBox = "fixture_component@1\n";
    payloads.push_back({PayloadRole::BlackBoxContract,
                        "blackbox/fixture-component.txt",
                        take(__func__, blobs.put(bytes(blackBox)))});
    if (llvm::Error error = contracts.add(ExternalImplementationContract{
            "fixture.genus.component",
            {{"component_model",
              {ExternalDependencyKind::ToolBundledResource}}},
            {RepresentationRootVariant::Rtl},
            true,
            false,
            nullptr}))
      fail(__func__, llvm::toString(std::move(error)));
    externalBindings.push_back(
        {"fixture.genus.component",
         {{"component_model",
           ToolBundledResourceDependency{
               genusToolBundledResourceProviderIdentity(*componentBuild),
               "chipware:fixture_component"}}},
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
  auto config = take(__func__, createResolvedGenusGateNetlistConfigView(
                                   "Program Name: Genus fixture 26.1", corner,
                                   fingerprint(liberty)));
  auto inputs =
      take(__func__, bindGenusGateNetlistInputs(implementation.reference(),
                                                platform.reference()));
  auto binding = take(__func__, resolveGenusGateNetlistBinding(config));
  return SemanticFixture{std::move(platform), std::move(implementation),
                         std::move(config),   std::move(inputs),
                         std::move(binding),  std::move(contracts)};
}
ExternalToolPreparationContext
makePreparationContext(const std::filesystem::path &bundle,
                       const std::filesystem::path &fixtureRoot) {
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  local.tools["genus"].binding.executable =
      std::filesystem::canonical(fixtureRoot / "fake-genus").string();
  local.externalFiles["synthetic_standard_cell_liberty"] =
      std::filesystem::canonical(fixtureRoot / "standard-cell.lib").string();
  return ExternalToolPreparationContext{std::move(local), bundle.string()};
}
PreparedExternalToolInvocation prepare(const std::filesystem::path &bundle,
                                       const std::filesystem::path &fixtureRoot,
                                       const SemanticFixture &fixture,
                                       const ArtifactStore &artifacts,
                                       const BlobStore &blobs) {
  return take(__func__, prepareCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, artifacts, blobs,
                            makePreparationContext(bundle, fixtureRoot)));
}
void descriptorAndConfigAreExact() {
  require(__func__, !registerGenusGateNetlistCandidateGenerator(),
          "Genus generator registration failed");
  const CandidateGeneratorDescriptor &descriptor =
      genusGateNetlistCandidateGeneratorDescriptor();
  require(__func__,
          descriptor.kind == genusGateNetlistCandidateGeneratorKind &&
              descriptor.providerForm == ProviderForm::ExternalPrepareImport &&
              descriptor.determinism ==
                  CandidateGeneratorDeterminism::Deterministic,
          "Genus descriptor form changed");
  require(__func__,
          descriptor.inputSlots.size() == 2 &&
              descriptor.inputSlots[0].semanticRole ==
                  "finalized_rtl_with_generation_constraints" &&
              descriptor.inputSlots[0].schema ==
                  &hardwareImplementationSchema &&
              descriptor.inputSlots[1].semanticRole == "asic_target" &&
              descriptor.inputSlots[1].schema ==
                  &platform::implementationPlatformSchema,
          "Genus exact input slots changed");
  require(__func__,
          descriptor.outputSlots.size() == 1 &&
              descriptor.outputSlots.front().schema ==
                  &hardwareImplementationSchema &&
              descriptor.outputSlots.front().semanticRole == "gate_netlist",
          "Genus GateNetlist output slot changed");
  require(__func__,
          descriptor.workUnits.size() == 1 &&
              descriptor.workUnits.front().spelling == "synthesis_attempt",
          "Genus work accounting changed");

  const std::string liberty = "synthetic-liberty";
  const platform::TechnologyCornerRef corner{
      take(__func__,
           ArtifactIdentity::fromBytes(
               std::array<std::uint8_t, ArtifactIdentity::byteSize>{})),
      platform::TechnologyCornerId(7)};
  auto config = take(__func__, createResolvedGenusGateNetlistConfigView(
                                   "Program Name: Genus fixture 26.1", corner,
                                   fingerprint(liberty)));
  require(__func__,
          config.stableProviderBuildIdentity() ==
                  "Program Name: Genus fixture 26.1" &&
              config.technologyCorner() == corner &&
              config.standardCellLiberty() == fingerprint(liberty),
          "Genus resolved config lost an exact semantic input");
  expectFailure(__func__,
                createResolvedGenusGateNetlistConfigView("", corner,
                                                         fingerprint(liberty)),
                "provider build identity");
}

void driverAndParserAreDeterministic(const std::filesystem::path &fixtureRoot) {
  const std::string expected =
      "proc loom_main {} {\n"
      "read_libs {/libraries/slow.lib}\n"
      "read_hdl -sv [list {inputs/implementation/rtl/package.sv} "
      "{inputs/implementation/rtl/top.sv}]\n"
      "elaborate {top}\n"
      "read_sdc {inputs/implementation/constraints/top.sdc}\n"
      "syn_generic\n"
      "syn_map\n"
      "syn_opt\n"
      "write_hdl > {outputs/genus-gate-netlist.v}\n"
      "}\n"
      "if {[catch {loom_main} loom_error]} {\n"
      "  puts stderr $loom_error\n"
      "  exit 1\n"
      "}\n"
      "exit 0\n";
  const std::string driver =
      take(__func__, renderGenusGateNetlistDriver(
                         "top",
                         {"inputs/implementation/rtl/package.sv",
                          "inputs/implementation/rtl/top.sv"},
                         {"inputs/implementation/constraints/top.sdc"},
                         "/libraries/slow.lib"));
  require(__func__,
          driver == expected &&
              driver == take(__func__,
                             renderGenusGateNetlistDriver(
                                 "top",
                                 {"inputs/implementation/rtl/package.sv",
                                  "inputs/implementation/rtl/top.sv"},
                                 {"inputs/implementation/constraints/top.sdc"},
                                 "/libraries/slow.lib")),
          "Genus driver is not byte deterministic");
  expectFailure(__func__,
                renderGenusGateNetlistDriver("bad top", {"inputs/rtl/top.sv"},
                                             {"inputs/top.sdc"},
                                             "/libraries/slow.lib"),
                "portable HDL identifier");
  expectFailure(__func__,
                renderGenusGateNetlistDriver("top", {"../top.sv"},
                                             {"inputs/top.sdc"},
                                             "/libraries/slow.lib"),
                "beneath inputs");
  expectFailure(__func__,
                renderGenusGateNetlistDriver("top", {"inputs/rtl/top.sv"}, {},
                                             "/libraries/slow.lib"),
                "constraint inventory is empty");

  const std::string gate = readFile(fixtureRoot / "expected/top.v");
  require(__func__,
          take(__func__, parseGenusGateNetlist(gate, "top")).verilog == gate,
          "Genus parser rewrote the exact netlist");
  expectFailure(__func__, parseGenusGateNetlist(gate, "other"), "exact top");
  expectFailure(
      __func__,
      parseGenusGateNetlist(std::string("module top;\0endmodule\n", 22), "top"),
      "LF text contract");
}

void mismatchedBuildCannotPrepare(const std::filesystem::path &root,
                                  const std::filesystem::path &fixtureRoot,
                                  const SemanticFixture &fixture,
                                  const ArtifactStore &artifacts,
                                  const BlobStore &blobs) {
  auto wrongConfig = take(__func__, createResolvedGenusGateNetlistConfigView(
                                        "Program Name: Genus different build",
                                        fixture.config.technologyCorner(),
                                        fixture.config.standardCellLiberty()));
  auto wrongBinding =
      take(__func__, resolveGenusGateNetlistBinding(wrongConfig));
  expectFailure(__func__,
                prepareCandidateGeneratorInvocation(
                    fixture.inputs, wrongBinding, artifacts, blobs,
                    makePreparationContext(root / "wrong-build", fixtureRoot)),
                "does not match semantic build");
}

void toolBundledRtlUsesExactProviderBuild(
    const std::filesystem::path &root, const std::filesystem::path &fixtureRoot,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  require(__func__,
          genusToolBundledResourceProviderIdentity("Build 1") ==
              "cadence_genus_build_4275696c642031",
          "tool-bundled resource provider identity is not injective bytes");
  const std::string build = "Program Name: Genus fixture 26.1";
  const SemanticFixture fixture =
      makeSemanticFixture(fixtureRoot, artifacts, blobs, build);
  const PreparedExternalToolInvocation prepared = take(
      __func__,
      prepareGenusGateNetlistInvocation(
          fixture.inputs, fixture.binding, fixture.contracts, artifacts, blobs,
          makePreparationContext(root / "tool-bundled", fixtureRoot)));
  const std::string driver = readFile(root / "tool-bundled/drivers/genus.tcl");
  require(__func__,
          driver.find("chipware:fixture_component") == std::string::npos,
          "Genus driver leaked a semantic resource key as a host path");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "caller-owned external-bound Genus execution failed");
  CandidateGeneratorProviderResult result =
      take(__func__, importGenusGateNetlistInvocation(
                         fixture.inputs, fixture.binding, prepared,
                         fixture.contracts, artifacts, blobs));
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  require(__func__, completed && completed->outputBindings.size() == 1,
          "external-bound Genus invocation did not publish one output");
  auto output =
      take(__func__, importGenusGateNetlistImplementation(
                         completed->outputBindings.front().artifacts.front(),
                         artifacts, blobs));
  require(__func__,
          output.implementation().externalImplementationBindings().size() ==
                  1 &&
              output.implementation()
                      .externalImplementationBindings()
                      .front()
                      .providerContractRef != "fixture.genus.component",
          "Genus retained the consumed RTL component binding");

  const SemanticFixture wrong = makeSemanticFixture(
      fixtureRoot, artifacts, blobs, "Program Name: Genus different build");
  expectFailure(
      __func__,
      prepareGenusGateNetlistInvocation(
          wrong.inputs, wrong.binding, wrong.contracts, artifacts, blobs,
          makePreparationContext(root / "wrong-component-build", fixtureRoot)),
      "another Genus build");
  require(__func__, !std::filesystem::exists(root / "wrong-component-build"),
          "wrong component build mutated the bundle destination");
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
              readFile(firstBundle / "drivers/genus.tcl") ==
                  readFile(secondBundle / "drivers/genus.tcl"),
          "equivalent preparation changed the bundle");
  require(
      __func__,
      !std::filesystem::exists(firstBundle / "outputs/genus-gate-netlist.v") &&
          !std::filesystem::exists(firstBundle / "outputs/completion.json"),
      "preparation executed Genus");

  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(first)) == 0,
          "caller-owned fake Genus execution failed");
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
          "Genus provider result is not one complete mechanical derivation");

  const ArtifactRootReference output =
      completed->outputBindings.front().artifacts.front();
  auto implementation = take(
      __func__, importGenusGateNetlistImplementation(output, artifacts, blobs));
  const ImplementationRepresentationRoot &representation =
      implementation.implementation().representationRoot();
  require(
      __func__,
      representation.variant == RepresentationRootVariant::GateNetlist &&
          representation.formatRef.kind() ==
              RepresentationFormatKind::StructuralVerilogGateNetlist &&
          representation.top ==
              RepresentationLocator{RepresentationObjectKind::Module, "top"},
      "Genus did not publish the exact GateNetlist state");
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
          "Genus output payload closure is incomplete");
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
              implementation.implementation().activityPoints().empty(),
          "Genus output lost an exact dependency or retained unowned activity");

  const InnovusPhysicalSnapshot physicalSnapshot =
      take(__func__, parseInnovusPhysicalSnapshot(
                         readFile(fixtureRoot / "expected/top.v"),
                         "VERSION 5.8 ;\nDESIGN top ;\nNETS 1 ;\n"
                         "- clk + ROUTED Metal2 ( 0 0 ) ( 100 0 ) ;\n"
                         "END NETS\nEND DESIGN\n",
                         readFile(fixtureRoot / "constraints/top.sdc"), "top",
                         RepresentationPhysicalStage::Routed));
  const FinalizedHardwareImplementation physical =
      take(__func__, publishInnovusPhysicalImplementation(
                         implementation, physicalSnapshot, artifacts, blobs));
  const ImplementationRepresentationRoot &physicalRoot =
      physical.implementation().representationRoot();
  require(
      __func__,
      physicalRoot.variant == RepresentationRootVariant::AsicPhysical &&
          physicalRoot.stage == RepresentationPhysicalStage::Routed &&
          physicalRoot.formatRef.kind() ==
              RepresentationFormatKind::IndexedDefPhysical &&
          llvm::count_if(physicalRoot.payloads,
                         [](const auto &payload) {
                           return payload.role == PayloadRole::Netlist;
                         }) == 1 &&
          physical.implementation().interfaces() ==
              implementation.implementation().interfaces() &&
          physical.implementation().externalImplementationBindings().size() ==
              1,
      "Innovus publication lost exact physical state or semantic ownership");
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
  writeFile(root / "stale/outputs/genus-gate-netlist.v",
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
  writeFile(root / "tampered/outputs/genus-gate-netlist.v",
            readFile(fixtureRoot / "expected/top.v") + "// tampered\n");
  requireNoPublication(__func__, import(tampered), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto missing =
      prepare(root / "missing", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(missing));
  std::filesystem::remove(root / "missing/outputs/genus-gate-netlist.v");
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

} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    fail("main", "expected scratch and fixture directory arguments");
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

  descriptorAndConfigAreExact();
  driverAndParserAreDeterministic(fixtureRoot);
  const SemanticFixture fixture =
      makeSemanticFixture(fixtureRoot, artifacts, blobs);
  mismatchedBuildCannotPrepare(root, fixtureRoot, fixture, artifacts, blobs);
  toolBundledRtlUsesExactProviderBuild(root, fixtureRoot, artifacts, blobs);
  successfulLifecyclePublishesGateNetlist(root, fixtureRoot, fixture, artifacts,
                                          blobs);
  strictImportRejectsInvalidAttempts(root, fixtureRoot, fixture, artifacts,
                                     blobs, artifactsRoot, blobsRoot);
  return EXIT_SUCCESS;
}
