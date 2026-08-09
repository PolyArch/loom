#include "ConfigurationABI3TestSupport.h"
#include "EDA/Adapters/Synopsys/DesignCompiler.h"
#include "EDA/Adapters/Synopsys/FusionCompiler.h"

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
#include "Hardware/Implementation/RepresentationIndex.h"
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
using namespace loom::eda::synopsys;
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
      fabric.module @synopsys_fixture(
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
  ResolvedDesignCompilerGateNetlistConfigView config;
  std::vector<CandidateGeneratorInputBinding> inputs;
  ResolvedCandidateGeneratorBinding binding;
  ExternalImplementationContractCatalog contracts;
};
SemanticFixture makeSemanticFixture(const std::filesystem::path &fixtureRoot,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs) {
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

  const std::string rtl = readFile(fixtureRoot / "rtl/top.sv");
  const std::string sdc = readFile(fixtureRoot / "constraints/top.sdc");
  const BlobDigest rtlDigest = take(__func__, blobs.put(bytes(rtl)));
  const BlobDigest sdcDigest = take(__func__, blobs.put(bytes(sdc)));
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::RtlSource, "rtl/top.sv", rtlDigest},
      {PayloadRole::GenerationConstraint, "constraints/top.sdc", sdcDigest}};
  ExternalImplementationContractCatalog contracts;
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
                   {}},
               contracts, artifacts, blobs));

  const std::string liberty = readFile(fixtureRoot / "standard-cell.lib");
  const platform::TechnologyCornerRef corner{platform.reference().artifact,
                                             platform::TechnologyCornerId(0)};
  auto config =
      take(__func__, createResolvedDesignCompilerGateNetlistConfigView(
                         "dc_shell version - Y-2026.03-SP2", corner,
                         fingerprint(liberty)));
  auto inputs =
      take(__func__, bindDesignCompilerGateNetlistInputs(
                         implementation.reference(), platform.reference()));
  auto binding =
      take(__func__, resolveDesignCompilerGateNetlistBinding(config));
  return SemanticFixture{std::move(platform), std::move(implementation),
                         std::move(config),   std::move(inputs),
                         std::move(binding),  std::move(contracts)};
}
ExternalToolPreparationContext
makePreparationContext(const std::filesystem::path &bundle,
                       const std::filesystem::path &fixtureRoot,
                       const std::filesystem::path &executable = {}) {
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  local.tools["dc_shell"].binding.executable =
      std::filesystem::canonical(
          executable.empty() ? fixtureRoot / "fake-dc_shell" : executable)
          .string();
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
  require(__func__, !registerDesignCompilerGateNetlistCandidateGenerator(),
          "DesignCompiler generator registration failed");
  const CandidateGeneratorDescriptor &descriptor =
      designCompilerGateNetlistCandidateGeneratorDescriptor();
  require(__func__,
          descriptor.kind == designCompilerGateNetlistCandidateGeneratorKind &&
              descriptor.providerForm == ProviderForm::ExternalPrepareImport &&
              descriptor.determinism ==
                  CandidateGeneratorDeterminism::Deterministic,
          "DesignCompiler descriptor form changed");
  require(__func__,
          descriptor.inputSlots.size() == 2 &&
              descriptor.inputSlots[0].semanticRole ==
                  "finalized_rtl_with_generation_constraints" &&
              descriptor.inputSlots[0].schema ==
                  &hardwareImplementationSchema &&
              descriptor.inputSlots[1].semanticRole == "asic_target" &&
              descriptor.inputSlots[1].schema ==
                  &platform::implementationPlatformSchema,
          "DesignCompiler exact input slots changed");
  require(__func__,
          descriptor.outputSlots.size() == 1 &&
              descriptor.outputSlots.front().schema ==
                  &hardwareImplementationSchema &&
              descriptor.outputSlots.front().semanticRole == "gate_netlist",
          "DesignCompiler GateNetlist output slot changed");
  require(__func__,
          descriptor.workUnits.size() == 1 &&
              descriptor.workUnits.front().spelling == "synthesis_attempt",
          "DesignCompiler work accounting changed");

  const std::string liberty = "synthetic-liberty";
  const platform::TechnologyCornerRef corner{
      take(__func__,
           ArtifactIdentity::fromBytes(
               std::array<std::uint8_t, ArtifactIdentity::byteSize>{})),
      platform::TechnologyCornerId(7)};
  auto config =
      take(__func__, createResolvedDesignCompilerGateNetlistConfigView(
                         "dc_shell version - Y-2026.03-SP2", corner,
                         fingerprint(liberty)));
  require(__func__,
          config.stableProviderBuildIdentity() ==
                  "dc_shell version - Y-2026.03-SP2" &&
              config.technologyCorner() == corner &&
              config.standardCellLiberty() == fingerprint(liberty),
          "DesignCompiler resolved config lost an exact semantic input");
  expectFailure(__func__,
                createResolvedDesignCompilerGateNetlistConfigView(
                    "", corner, fingerprint(liberty)),
                "provider build identity");
}

void driverAndParserAreDeterministic(const std::filesystem::path &fixtureRoot) {
  const std::string expected =
      "proc loom_main {} {\n"
      "set loom_target_library [list {/libraries/slow.lib}]\n"
      "set_app_var target_library $loom_target_library\n"
      "set_app_var link_library [concat {*} $loom_target_library]\n"
      "analyze -format sverilog [list "
      "{inputs/implementation/rtl/package.sv} "
      "{inputs/implementation/rtl/top.sv}]\n"
      "elaborate {top}\n"
      "current_design {top}\n"
      "link\n"
      "read_sdc {inputs/implementation/constraints/top.sdc}\n"
      "compile_ultra\n"
      "check_design\n"
      "if {[get_message_info -error_count] != 0} {\n"
      "  error {Synopsys tool emitted error diagnostics}\n"
      "}\n"
      "write -format verilog -hierarchy -output "
      "{outputs/design-compiler-gate-netlist.v}\n"
      "if {[get_message_info -error_count] != 0} {\n"
      "  error {Synopsys tool emitted error diagnostics}\n"
      "}\n"
      "}\n"
      "if {[catch {loom_main} loom_error]} {\n"
      "  puts stderr $loom_error\n"
      "  exit 1\n"
      "}\n"
      "exit 0\n";
  const std::string driver = take(
      __func__,
      renderDesignCompilerDriver("top",
                                 {"inputs/implementation/rtl/package.sv",
                                  "inputs/implementation/rtl/top.sv"},
                                 {"inputs/implementation/constraints/top.sdc"},
                                 "/libraries/slow.lib"));
  require(__func__,
          driver == expected &&
              driver == take(__func__,
                             renderDesignCompilerDriver(
                                 "top",
                                 {"inputs/implementation/rtl/package.sv",
                                  "inputs/implementation/rtl/top.sv"},
                                 {"inputs/implementation/constraints/top.sdc"},
                                 "/libraries/slow.lib")),
          "DesignCompiler driver is not byte deterministic");
  expectFailure(__func__,
                renderDesignCompilerDriver("bad top", {"inputs/rtl/top.sv"},
                                           {"inputs/top.sdc"},
                                           "/libraries/slow.lib"),
                "portable HDL identifier");
  expectFailure(__func__,
                renderDesignCompilerDriver("top", {"../top.sv"},
                                           {"inputs/top.sdc"},
                                           "/libraries/slow.lib"),
                "beneath inputs");
  expectFailure(__func__,
                renderDesignCompilerDriver("top", {"inputs/rtl/top.sv"}, {},
                                           "/libraries/slow.lib"),
                "constraint inventory is empty");

  const std::string gate = readFile(fixtureRoot / "expected/top.v");
  require(__func__,
          take(__func__, parseDesignCompilerGateNetlist(gate, "top")).verilog ==
              gate,
          "DesignCompiler parser rewrote the exact netlist");
  expectFailure(__func__, parseDesignCompilerGateNetlist(gate, "other"),
                "exact top");
  expectFailure(__func__,
                parseDesignCompilerGateNetlist(
                    std::string("module top;\0endmodule\n", 22), "top"),
                "LF text contract");
}

void mismatchedBuildCannotPrepare(const std::filesystem::path &root,
                                  const std::filesystem::path &fixtureRoot,
                                  const SemanticFixture &fixture,
                                  const ArtifactStore &artifacts,
                                  const BlobStore &blobs) {
  auto wrongConfig =
      take(__func__, createResolvedDesignCompilerGateNetlistConfigView(
                         "dc_shell version - X-2025.01",
                         fixture.config.technologyCorner(),
                         fixture.config.standardCellLiberty()));
  auto wrongBinding =
      take(__func__, resolveDesignCompilerGateNetlistBinding(wrongConfig));
  expectFailure(__func__,
                prepareCandidateGeneratorInvocation(
                    fixture.inputs, wrongBinding, artifacts, blobs,
                    makePreparationContext(root / "wrong-build", fixtureRoot)),
                "does not match semantic build");
}

FinalizedHardwareImplementation successfulLifecyclePublishesGateNetlist(
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
              readFile(firstBundle / "drivers/design-compiler.tcl") ==
                  readFile(secondBundle / "drivers/design-compiler.tcl"),
          "equivalent preparation changed the bundle");
  require(__func__,
          !std::filesystem::exists(firstBundle /
                                   "outputs/design-compiler-gate-netlist.v") &&
              !std::filesystem::exists(firstBundle / "outputs/completion.json"),
          "preparation executed DesignCompiler");

  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(first)) == 0,
          "caller-owned fake DesignCompiler execution failed");
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
          "DesignCompiler provider result is not one complete mechanical "
          "derivation");

  const ArtifactRootReference output =
      completed->outputBindings.front().artifacts.front();
  auto implementation = take(
      __func__,
      importDesignCompilerGateNetlistImplementation(output, artifacts, blobs));
  const ImplementationRepresentationRoot &representation =
      implementation.implementation().representationRoot();
  require(
      __func__,
      representation.variant == RepresentationRootVariant::GateNetlist &&
          representation.formatRef.kind() ==
              RepresentationFormatKind::StructuralVerilogGateNetlist &&
          representation.top ==
              RepresentationLocator{RepresentationObjectKind::Module, "top"},
      "DesignCompiler did not publish the exact GateNetlist state");
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
          "DesignCompiler output payload closure is incomplete");
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
          "DesignCompiler output lost an exact dependency or retained unowned "
          "activity");
  return implementation;
}

void fusionCompilerPublicationIsClosed(
    const std::filesystem::path &testRoot,
    const std::filesystem::path &fixtureRoot,
    const FinalizedHardwareImplementation &gate, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  const FusionCompilerPhysicalSnapshot snapshot{
      RepresentationPhysicalStage::Routed,
      readFile(fixtureRoot / "expected/fusion-routed.v"),
      "VERSION 5.8 ;\nDESIGN top ;\nNETS 1 ;\n"
      "- clk + ROUTED Metal2 ( 0 0 ) ( 100 0 ) ;\n"
      "END NETS\nEND DESIGN\n",
      "create_clock -period 1 clk\n"};
  auto physical = take(__func__, publishFusionCompilerPhysicalImplementation(
                                     gate, snapshot, artifacts, blobs));
  const ImplementationRepresentationRoot &root =
      physical.implementation().representationRoot();
  const RepresentationIndex sourceIndex =
      take(__func__, indexRepresentationRoot(
                         gate.implementation().representationRoot(), blobs));
  const RepresentationIndex physicalIndex =
      take(__func__, indexRepresentationRoot(root, blobs));
  require(
      __func__,
      root.variant == RepresentationRootVariant::AsicPhysical &&
          root.stage == RepresentationPhysicalStage::Routed &&
          root.formatRef.kind() ==
              RepresentationFormatKind::IndexedDefPhysical &&
          root.top ==
              RepresentationLocator{RepresentationObjectKind::PhysicalObject,
                                    "top"} &&
          llvm::count_if(root.payloads,
                         [](const auto &payload) {
                           return payload.role == PayloadRole::Netlist;
                         }) == 1 &&
          llvm::count_if(root.payloads,
                         [](const auto &payload) {
                           return payload.role == PayloadRole::PhysicalDatabase;
                         }) == 1 &&
          llvm::count_if(root.payloads,
                         [](const auto &payload) {
                           return payload.role ==
                                  PayloadRole::RepresentationIndex;
                         }) == 1 &&
          llvm::count_if(root.payloads,
                         [](const auto &payload) {
                           return payload.role ==
                                  PayloadRole::GenerationConstraint;
                         }) == 1 &&
          llvm::count_if(root.payloads,
                         [](const auto &payload) {
                           return payload.role == PayloadRole::BlackBoxContract;
                         }) == 1,
      "Fusion Compiler did not publish the exact routed physical closure");
  require(__func__,
          physicalIndex.rootBoundaryPorts() == sourceIndex.rootBoundaryPorts(),
          "Fusion Compiler physical index lost the exact top boundary");
  const auto sourceBindings =
      gate.implementation().externalImplementationBindings();
  const auto physicalBindings =
      physical.implementation().externalImplementationBindings();
  require(__func__,
          physical.implementation().implementationPlatform() ==
                  gate.implementation().implementationPlatform() &&
              physicalBindings.size() == 1 && sourceBindings.size() == 1 &&
              physicalBindings.front().providerContractRef ==
                  sourceBindings.front().providerContractRef &&
              physicalBindings.front().externalInputs ==
                  sourceBindings.front().externalInputs &&
              physicalBindings.front().fabricResourceRefs ==
                  sourceBindings.front().fabricResourceRefs &&
              physicalBindings.front().representationLocators ==
                  std::vector<RepresentationLocator>{
                      {RepresentationObjectKind::Module, "INVX0"}} &&
              sourceBindings.front().representationLocators ==
                  std::vector<RepresentationLocator>{
                      {RepresentationObjectKind::Module, "INVX1"}} &&
              physicalBindings.front().blackBoxContractPayloadRef.has_value(),
          "Fusion Compiler publication changed an exact implementation "
          "dependency");
  const ImplementationPayload &physicalContract =
      root.payloads[physicalBindings.front()
                        .blackBoxContractPayloadRef->ordinal];
  const std::vector<std::uint8_t> physicalContractBytes =
      take(__func__, blobs.get(physicalContract.blobDigest));
  const llvm::StringRef physicalContractText(
      reinterpret_cast<const char *>(physicalContractBytes.data()),
      physicalContractBytes.size());
  require(__func__,
          physicalContractText.contains("module=INVX0\n") &&
              !physicalContractText.contains("module=INVX1\n"),
          "Fusion Compiler did not derive the routed black-box closure");

  auto contracts = take(__func__, makeSynopsysStandardCellContractCatalog());
  auto strict =
      take(__func__, importHardwareImplementation(physical.reference(),
                                                  contracts, artifacts, blobs));
  require(__func__, strict.reference() == physical.reference(),
          "Fusion Compiler physical implementation did not strictly reimport");

  const std::uint64_t artifactCount = regularFileCount(testRoot / "artifacts");
  const std::uint64_t blobCount = regularFileCount(testRoot / "blobs");
  const ExternalImplementationBinding &sourceBinding =
      gate.implementation().externalImplementationBindings().front();
  ExternalImplementationContractCatalog gateOnly;
  if (llvm::Error error = gateOnly.add(ExternalImplementationContract{
          sourceBinding.providerContractRef,
          {{sourceBinding.externalInputs.front().providerInputSlotRef,
            {ExternalDependencyKind::ExplicitFile}}},
          {RepresentationRootVariant::GateNetlist},
          true,
          false,
          nullptr}))
    fail(__func__, llvm::toString(std::move(error)));
  expectFailure(__func__,
                publishFusionCompilerPhysicalImplementation(
                    gate, snapshot, gateOnly, artifacts, blobs),
                "does not support the representation");
  require(__func__,
          regularFileCount(testRoot / "artifacts") == artifactCount &&
              regularFileCount(testRoot / "blobs") == blobCount,
          "rejected external contract changed persistent state");

  FusionCompilerPhysicalSnapshot missingBoundary = snapshot;
  missingBoundary.netlistVerilog = "module top; endmodule\n";
  expectFailure(__func__,
                publishFusionCompilerPhysicalImplementation(
                    gate, missingBoundary, artifacts, blobs),
                "boundary");
  require(__func__,
          regularFileCount(testRoot / "artifacts") == artifactCount &&
              regularFileCount(testRoot / "blobs") == blobCount,
          "invalid routed boundary changed persistent state");
}

void fusionCompilerGeneratorIsClosed(
    const std::filesystem::path &root, const std::filesystem::path &fixtureRoot,
    const FinalizedHardwareImplementation &gate,
    const platform::FinalizedImplementationPlatform &platform,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const std::vector<ExternalFileTreeMember> referenceLibrary{
      {"parts/p0",
       fingerprint(readFile(fixtureRoot / "reference.ndm/parts/p0"))},
      {"pcat", fingerprint(readFile(fixtureRoot / "reference.ndm/pcat"))}};
  const platform::TechnologyCornerRef corner{platform.reference().artifact,
                                             platform::TechnologyCornerId(0)};
  const std::string early = readFile(fixtureRoot / "early.tluplus");
  const std::string late = readFile(fixtureRoot / "late.tluplus");
  const std::string layerMap = readFile(fixtureRoot / "layers.map");
  const std::string floorplan = readFile(fixtureRoot / "floorplan.def");
  auto config =
      take(__func__, createResolvedFusionCompilerRoutedConfigView(
                         "fc_shell version - Y-2026.03", corner,
                         referenceLibrary, fingerprint(early),
                         fingerprint(late), fingerprint(layerMap), floorplan));
  require(__func__,
          config.stableProviderBuildIdentity() ==
                  "fc_shell version - Y-2026.03" &&
              config.technologyCorner() == corner &&
              config.referenceLibraryMembers() ==
                  llvm::ArrayRef<ExternalFileTreeMember>(referenceLibrary) &&
              config.earlyParasiticTech() == fingerprint(early) &&
              config.lateParasiticTech() == fingerprint(late) &&
              config.parasiticLayerMap() == fingerprint(layerMap) &&
              config.floorplanDef() == floorplan,
          "Fusion Compiler config lost an exact semantic input");
  std::vector<ExternalFileTreeMember> reversed = referenceLibrary;
  std::reverse(reversed.begin(), reversed.end());
  expectFailure(__func__,
                createResolvedFusionCompilerRoutedConfigView(
                    "fc_shell version - Y-2026.03", corner, reversed,
                    fingerprint(early), fingerprint(late),
                    fingerprint(layerMap), floorplan),
                "sorted");

  require(__func__, !registerFusionCompilerRoutedCandidateGenerator(),
          "Fusion Compiler generator registration failed");
  const CandidateGeneratorDescriptor &descriptor =
      fusionCompilerRoutedCandidateGeneratorDescriptor();
  require(__func__,
          descriptor.kind == fusionCompilerRoutedCandidateGeneratorKind &&
              descriptor.providerForm == ProviderForm::ExternalPrepareImport &&
              descriptor.determinism ==
                  CandidateGeneratorDeterminism::IndependentReplicates &&
              descriptor.inputSlots.size() == 2 &&
              descriptor.outputSlots.size() == 1 &&
              descriptor.outputSlots.front().semanticRole ==
                  "routed_asic_physical",
          "Fusion Compiler descriptor lost its exact generator contract");

  const auto inputs =
      take(__func__, bindFusionCompilerRoutedInputs(gate.reference(),
                                                    platform.reference()));
  const auto binding =
      take(__func__, resolveFusionCompilerRoutedBinding(config));
  const auto makeLocal = [&](llvm::StringRef executable = "fake-fc_shell") {
    LocalToolConfig local;
    local.runtimePolicy = RuntimePolicy::Host;
    local.tools["fc_shell"].binding.executable =
        std::filesystem::canonical(fixtureRoot / executable.str()).string();
    local.externalFiles["early_parasitic_tech"] =
        std::filesystem::canonical(fixtureRoot / "early.tluplus").string();
    local.externalFiles["late_parasitic_tech"] =
        std::filesystem::canonical(fixtureRoot / "late.tluplus").string();
    local.externalFiles["parasitic_layer_map"] =
        std::filesystem::canonical(fixtureRoot / "layers.map").string();
    local.externalFileTrees["reference_library"] =
        std::filesystem::canonical(fixtureRoot / "reference.ndm").string();
    return local;
  };
  auto wrongConfig =
      take(__func__, createResolvedFusionCompilerRoutedConfigView(
                         "fc_shell version - Y-2026.04", corner,
                         referenceLibrary, fingerprint(early),
                         fingerprint(late), fingerprint(layerMap), floorplan));
  auto wrongBinding =
      take(__func__, resolveFusionCompilerRoutedBinding(wrongConfig));
  expectFailure(
      __func__,
      prepareCandidateGeneratorInvocation(
          inputs, wrongBinding, artifacts, blobs,
          ExternalToolPreparationContext{
              makeLocal(), (root / "fusion-generator-wrong-build").string()}),
      "does not match semantic build");

  const std::filesystem::path bundle = root / "fusion-generator";
  const PreparedExternalToolInvocation prepared =
      take(__func__,
           prepareCandidateGeneratorInvocation(
               inputs, binding, artifacts, blobs,
               ExternalToolPreparationContext{makeLocal(), bundle.string()}));
  const std::string manifest = readFile(bundle / "tool-invocation.json");
  require(__func__,
          llvm::StringRef(manifest).contains("\"version\": \"2.1\"") &&
              llvm::StringRef(manifest).contains("\"external_file_trees\"") &&
              llvm::StringRef(manifest).contains("\"path\": \"parts/p0\"") &&
              readFile(bundle / "drivers/fusion-floorplan.def") == floorplan,
          "Fusion Compiler bundle lost its exact tree or floorplan closure");
  const std::string driver = readFile(bundle / "drivers/fusion-compiler.tcl");
  require(
      __func__,
      llvm::StringRef(driver).contains(
          std::filesystem::canonical(fixtureRoot / "reference.ndm").string()) &&
          llvm::StringRef(driver).contains(
              std::filesystem::canonical(fixtureRoot / "early.tluplus")
                  .string()) &&
          llvm::StringRef(driver).contains(
              std::filesystem::canonical(fixtureRoot / "late.tluplus")
                  .string()) &&
          llvm::StringRef(driver).contains(
              std::filesystem::canonical(fixtureRoot / "layers.map").string()),
      "Fusion Compiler driver lost a resolved physical input");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "fake Fusion Compiler execution failed");
  CandidateGeneratorProviderResult result =
      take(__func__, importCandidateGeneratorInvocation(
                         inputs, binding, prepared, artifacts, blobs));
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  require(__func__,
          completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().artifacts.size() == 1 &&
              completed->lineageEdges.size() == 1 &&
              result.workSummary ==
                  std::vector<CandidateGeneratorWorkUnitSummary>{
                      {CandidateGeneratorWorkUnitRef(0), 1, 1}},
          "Fusion Compiler provider result is not one routed derivation");
  const auto physical =
      take(__func__, importFusionCompilerRoutedImplementation(
                         completed->outputBindings.front().artifacts.front(),
                         artifacts, blobs));
  require(__func__,
          physical.implementation().representationRoot().variant ==
                  RepresentationRootVariant::AsicPhysical &&
              physical.implementation().representationRoot().stage ==
                  RepresentationPhysicalStage::Routed &&
              physical.implementation().implementationPlatform() ==
                  gate.implementation().implementationPlatform(),
          "Fusion Compiler facade did not publish the exact routed state");
  const std::uint64_t publishedArtifactCount =
      regularFileCount(root / "artifacts");
  const std::uint64_t publishedBlobCount = regularFileCount(root / "blobs");
  expectFailure(__func__,
                importCandidateGeneratorInvocation(inputs, wrongBinding,
                                                   prepared, artifacts, blobs),
                "semantic closure");
  require(__func__,
          regularFileCount(root / "artifacts") == publishedArtifactCount &&
              regularFileCount(root / "blobs") == publishedBlobCount,
          "mismatched Fusion Compiler expectation changed persistent state");

  const PreparedExternalToolInvocation extra = take(
      __func__,
      prepareCandidateGeneratorInvocation(
          inputs, binding, artifacts, blobs,
          ExternalToolPreparationContext{
              makeLocal(), (root / "fusion-generator-extra-output").string()}));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(extra)) == 0,
          "fake Fusion Compiler extra-output execution failed");
  writeFile(std::filesystem::path(extra.bundleRoot) / "outputs/undeclared.txt",
            "undeclared\n");
  const std::uint64_t artifactCount = regularFileCount(root / "artifacts");
  const std::uint64_t blobCount = regularFileCount(root / "blobs");
  auto invalid = importCandidateGeneratorInvocation(inputs, binding, extra,
                                                    artifacts, blobs);
  expectFailure(__func__, std::move(invalid), "undeclared output");
  require(__func__,
          regularFileCount(root / "artifacts") == artifactCount &&
              regularFileCount(root / "blobs") == blobCount,
          "undeclared Fusion Compiler output changed persistent state");

  const PreparedExternalToolInvocation failed =
      take(__func__, prepareCandidateGeneratorInvocation(
                         inputs, binding, artifacts, blobs,
                         ExternalToolPreparationContext{
                             makeLocal("fake-fc_shell-fail"),
                             (root / "fusion-generator-tool-exit").string()}));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(failed)) == 7,
          "failing Fusion Compiler fixture returned the wrong exit status");
  CandidateGeneratorProviderResult failedResult =
      take(__func__, importCandidateGeneratorInvocation(inputs, binding, failed,
                                                        artifacts, blobs));
  const auto *incomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&failedResult.outcome);
  require(__func__,
          incomplete &&
              incomplete->reason ==
                  CandidateGeneratorIncompleteReason::ExecutionFailed &&
              incomplete->retainedOutputBindings.size() == 1 &&
              incomplete->retainedOutputBindings.front().artifacts.empty() &&
              regularFileCount(root / "artifacts") == artifactCount &&
              regularFileCount(root / "blobs") == blobCount,
          "Fusion Compiler ToolExit did not remain an unpublished incomplete "
          "candidate");

  const PreparedExternalToolInvocation missing = take(
      __func__, prepareCandidateGeneratorInvocation(
                    inputs, binding, artifacts, blobs,
                    ExternalToolPreparationContext{
                        makeLocal("fake-fc_shell-missing"),
                        (root / "fusion-generator-missing-output").string()}));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(missing)) != 0,
          "incomplete Fusion Compiler fixture returned success");
  CandidateGeneratorProviderResult missingResult =
      take(__func__, importCandidateGeneratorInvocation(
                         inputs, binding, missing, artifacts, blobs));
  const auto *missingOutput =
      std::get_if<IncompleteCandidateGeneratorResult>(&missingResult.outcome);
  require(__func__,
          missingOutput &&
              missingOutput->reason ==
                  CandidateGeneratorIncompleteReason::ExecutionFailed &&
              regularFileCount(root / "artifacts") == artifactCount &&
              regularFileCount(root / "blobs") == blobCount,
          "Fusion Compiler MissingOutput did not remain an unpublished "
          "incomplete candidate");
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
  writeFile(root / "stale/outputs/design-compiler-gate-netlist.v",
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
  writeFile(root / "tampered/outputs/design-compiler-gate-netlist.v",
            readFile(fixtureRoot / "expected/top.v") + "// tampered\n");
  requireNoPublication(__func__, import(tampered), artifactsRoot, blobsRoot,
                       artifactCount, blobCount);

  const auto missing =
      prepare(root / "missing", fixtureRoot, fixture, artifacts, blobs);
  take(__func__, executeExternalToolInvocationBundle(missing));
  std::filesystem::remove(root /
                          "missing/outputs/design-compiler-gate-netlist.v");
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
  const PreparedExternalToolInvocation prepared =
      take(__func__,
           prepareCandidateGeneratorInvocation(
               fixture.inputs, fixture.binding, artifacts, blobs,
               makePreparationContext(root / "tool-failure", fixtureRoot,
                                      fixtureRoot / "fake-dc_shell-fail")));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 7,
          "authored failing Design Compiler returned the wrong status");
  CandidateGeneratorProviderResult result =
      take(__func__,
           importCandidateGeneratorInvocation(fixture.inputs, fixture.binding,
                                              prepared, artifacts, blobs));
  const auto *incomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&result.outcome);
  require(
      __func__,
      incomplete &&
          incomplete->reason ==
              CandidateGeneratorIncompleteReason::ExecutionFailed &&
          incomplete->retainedOutputBindings.size() == 1 &&
          incomplete->retainedOutputBindings.front().artifacts.empty() &&
          result.workSummary.size() == 1 &&
          result.workSummary.front().consumed == 1,
      "failed Design Compiler did not produce the exact incomplete outcome");
  require(__func__,
          regularFileCount(artifactsRoot) == artifactCount &&
              regularFileCount(blobsRoot) == blobCount,
          "failed Design Compiler published persistent state");
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
  auto gate = successfulLifecyclePublishesGateNetlist(
      root, fixtureRoot, fixture, artifacts, blobs);
  fusionCompilerPublicationIsClosed(root, fixtureRoot, gate, artifacts, blobs);
  fusionCompilerGeneratorIsClosed(root, fixtureRoot, gate, fixture.platform,
                                  artifacts, blobs);
  strictImportRejectsInvalidAttempts(root, fixtureRoot, fixture, artifacts,
                                     blobs, artifactsRoot, blobsRoot);
  failedToolIsAnIncompleteCandidate(root, fixtureRoot, fixture, artifacts,
                                    blobs, artifactsRoot, blobsRoot);
  return EXIT_SUCCESS;
}
