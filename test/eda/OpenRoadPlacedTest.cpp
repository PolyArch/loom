#include "EDA/Adapters/OpenSource/OpenRoad.h"
#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "ConfigurationABI3TestSupport.h"

#include "ADG/Builder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::eda::open_source;

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
void expectInvalid(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted invalid OpenROAD input");
  llvm::consumeError(value.takeError());
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T>
void expectFailure(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted invalid or incomplete invocation");
  llvm::consumeError(value.takeError());
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return std::vector<std::uint8_t>(value.bytes_begin(), value.bytes_end());
}

void writeText(llvm::StringRef test, const std::filesystem::path &path,
               llvm::StringRef contents, bool executable = false) {
  std::error_code error;
  llvm::raw_fd_ostream output(path.string(), error);
  if (error)
    fail(test, "could not write " + path.string() + ": " + error.message());
  output << contents;
  output.close();
  if (output.has_error())
    fail(test, "could not finish writing " + path.string());
  if (executable) {
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec,
                                 std::filesystem::perm_options::replace, error);
    if (error)
      fail(test, "could not make test tool executable: " + error.message());
  }
}

std::string readText(llvm::StringRef test, const std::filesystem::path &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer)
    fail(test, "could not read " + path.string());
  return (*buffer)->getBuffer().str();
}

loom::ExternalFileFingerprint contentFingerprint(llvm::StringRef contents) {
  return take(__func__, loom::ExternalFileFingerprint::fromBytes(
                            llvm::SHA256::hash(bytes(contents))));
}

constexpr llvm::StringLiteral kTechnologyLef = R"lef(VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
UNITS
  DATABASE MICRONS 1000 ;
END UNITS
MANUFACTURINGGRID 0.001 ;
LAYER Metal2
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 0.20 ;
  WIDTH 0.10 ;
  SPACING 0.10 ;
END Metal2
LAYER Metal3
  TYPE ROUTING ;
  DIRECTION VERTICAL ;
  PITCH 0.20 ;
  WIDTH 0.10 ;
  SPACING 0.10 ;
END Metal3
END LIBRARY
)lef";

constexpr llvm::StringLiteral kCellLef = R"lef(VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
SITE CoreSite
  CLASS CORE ;
  SYMMETRY Y ;
  SIZE 0.2 BY 1.0 ;
END CoreSite
MACRO BUF_X1
  CLASS CORE ;
  ORIGIN 0 0 ;
  SIZE 0.4 BY 1.0 ;
  SYMMETRY X Y ;
  SITE CoreSite ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
    PORT
      LAYER Metal2 ;
      RECT 0.02 0.40 0.12 0.60 ;
    END
  END A
  PIN Y
    DIRECTION OUTPUT ;
    USE SIGNAL ;
    PORT
      LAYER Metal2 ;
      RECT 0.28 0.40 0.38 0.60 ;
    END
  END Y
END BUF_X1
END LIBRARY
)lef";

constexpr llvm::StringLiteral kLiberty = R"lib(library (synthetic) {
  delay_model : table_lookup;
  time_unit : "1ns";
  voltage_unit : "1V";
  current_unit : "1mA";
  capacitive_load_unit (1, pf);
  cell (BUF_X1) {
    area : 0.4;
    pin (A) {
      direction : input;
      capacitance : 0.001;
    }
    pin (Y) {
      direction : output;
      function : "A";
    }
  }
}
)lib";

constexpr llvm::StringLiteral kNetlist = R"verilog(module top(A, Y);
  input A;
  output Y;
  BUF_X1 u0 (.A(A), .Y(Y));
endmodule
)verilog";

constexpr llvm::StringLiteral kBlackBoxContract = "BUF_X1 input A output Y\n";
constexpr llvm::StringLiteral kSyntheticOpenRoadBuild =
    "OpenROAD synthetic b9a38929e342";

OpenRoadPlacementParameters parameters();

struct InvocationFixture final {
  loom::hardware::ExternalImplementationContractCatalog contracts;
  loom::hardware::FinalizedHardwareImplementation gate;
  loom::platform::FinalizedImplementationPlatform platform;
  OpenRoadPlacedConfig config;
};

InvocationFixture makeInvocationFixture(llvm::StringRef test,
                                        const loom::ArtifactStore &artifacts,
                                        const loom::BlobStore &blobs,
                                        llvm::StringRef providerBuild) {
  loom::adg::DesignBuilder design(artifacts);
  auto spatial =
      take(test, design.createSpatialCore("openroad-fixture", {}, {}));
  requireSuccess(test, spatial.close({}));
  auto moduleDesign = take(test, std::move(design).finalize());
  require(test, moduleDesign.roots().size() == 1,
          "fixture did not finalize one Fabric Module");
  auto module =
      take(test, loom::fabric::importEntireFabricRoot(
                     moduleDesign.roots().front().reference(), artifacts));
  auto system = take(test, loom::hardware::test::makeSingleSpatialCoreSystem(
                               module, artifacts));
  auto abi = take(
      test, loom::hardware::finalizeConfigurationABI(
                loom::hardware::ConfigurationABIDraft{system.reference(), {}},
                artifacts));
  auto platform = take(
      test, loom::platform::finalizeImplementationPlatform(
                loom::platform::ImplementationPlatformDraft{
                    loom::platform::AsicTarget{"synthetic-public", "2026.08"},
                    {"typical"}},
                artifacts));

  const loom::BlobDigest netlist = take(test, blobs.put(bytes(kNetlist)));
  const loom::BlobDigest contract =
      take(test, blobs.put(bytes(kBlackBoxContract)));
  const auto format =
      take(test, loom::hardware::RepresentationFormatDescriptorRef::get(
                     loom::hardware::RepresentationFormatKind::
                         StructuralVerilogGateNetlist));
  auto representation = take(
      test,
      loom::hardware::createImplementationRepresentationRoot(
          loom::hardware::RepresentationRootVariant::GateNetlist, std::nullopt,
          format, {loom::hardware::RepresentationObjectKind::Module, "top"},
          {{loom::hardware::PayloadRole::Netlist, "netlist/top.v", netlist},
           {loom::hardware::PayloadRole::BlackBoxContract,
            "contracts/yosys-standard-cells.txt", contract}}));

  loom::hardware::ExternalImplementationContractCatalog contracts =
      take(test, makeYosysStandardCellContractCatalog());
  const loom::ExternalFileFingerprint liberty = contentFingerprint(kLiberty);
  auto gate = take(
      test,
      loom::hardware::finalizeHardwareImplementation(
          loom::hardware::HardwareImplementationDraft{
              system.reference(),
              abi.reference(),
              {},
              std::move(representation),
              platform.reference(),
              {},
              {},
              {},
              {{"open_source.yosys.standard_cell_library",
                {{"standard_cell_liberty",
                  loom::hardware::ExplicitFileDependency{liberty}}},
                {},
                {{loom::hardware::RepresentationObjectKind::Module, "BUF_X1"}},
                loom::hardware::ImplementationPayloadKey{
                    loom::hardware::PayloadRole::BlackBoxContract,
                    "contracts/yosys-standard-cells.txt"}}}},
          contracts, artifacts, blobs));

  const loom::platform::TechnologyCornerRef exactCorner{
      platform.reference().artifact, loom::platform::TechnologyCornerId(0)};
  OpenRoadPlacedConfig placed{
      providerBuild.str(),
      exactCorner,
      parameters(),
      {{OpenRoadExternalFileKind::TechnologyLef, "technology",
        contentFingerprint(kTechnologyLef)},
       {OpenRoadExternalFileKind::CellLef, "cells",
        contentFingerprint(kCellLef)},
       {OpenRoadExternalFileKind::Liberty, "timing", liberty}}};
  return InvocationFixture{std::move(contracts), std::move(gate),
                           std::move(platform), std::move(placed)};
}

struct InvocationHarness final {
  std::vector<loom::dse::CandidateGeneratorInputBinding> inputs;
  loom::dse::ResolvedCandidateGeneratorBinding binding;
  OpenRoadResolvedExecution execution;
  loom::external_tool::ExternalToolPreparationContext context;
};

InvocationHarness makeInvocationHarness(llvm::StringRef test,
                                        const std::filesystem::path &root,
                                        const InvocationFixture &fixture,
                                        OpenRoadResolvedExecution execution) {
  const std::filesystem::path external = root / "external";
  std::filesystem::create_directories(external);
  const std::filesystem::path technology = external / "technology.lef";
  const std::filesystem::path cells = external / "cells.lef";
  const std::filesystem::path liberty = external / "cells.lib";
  writeText(test, technology, kTechnologyLef);
  writeText(test, cells, kCellLef);
  writeText(test, liberty, kLiberty);

  const std::vector<std::uint8_t> configBytes =
      take(test, encodeOpenRoadPlacedConfig(fixture.config));
  const loom::ComponentViewDigest configDigest =
      take(test, loom::computeComponentViewDigest(
                     openRoadPlacedConfigSchemaDescriptorBytes(), configBytes));
  auto binding =
      take(test, loom::dse::ResolvedCandidateGeneratorBinding::get(
                     openRoadPlacedCandidateGeneratorDescriptor().reference(),
                     configBytes, configDigest));
  std::vector<loom::dse::CandidateGeneratorInputBinding> inputs{
      {loom::dse::CandidateGeneratorInputSlotRef(0),
       {fixture.gate.reference()}}};

  loom::external_tool::LocalToolConfig local =
      loom::external_tool::defaultLocalToolConfig();
  local.runtimePolicy = loom::external_tool::RuntimePolicy::Host;
  local.externalFiles = {{"synthetic-tech", technology.string()},
                         {"synthetic-cells", cells.string()},
                         {"synthetic-liberty", liberty.string()}};
  return InvocationHarness{std::move(inputs), std::move(binding),
                           std::move(execution),
                           loom::external_tool::ExternalToolPreparationContext{
                               std::move(local), (root / "bundle").string()}};
}

InvocationHarness
makeSyntheticInvocationHarness(llvm::StringRef test,
                               const std::filesystem::path &root,
                               const InvocationFixture &fixture) {
  const std::filesystem::path probeMarker = root / "tool-was-probed";
  const std::filesystem::path tool = root / "synthetic-openroad";
  const std::string toolBody =
      "#!/usr/bin/env bash\n"
      "set -eu\n"
      "if [[ ${1-} == -version || ${1-} == --version ]]; then\n"
      "  : > \"" +
      probeMarker.string() +
      "\"\n"
      "  printf '%s\\n' '" +
      kSyntheticOpenRoadBuild.str() +
      "'\n"
      "  exit 0\n"
      "fi\n"
      "printf '%s\\n' 'synthetic placed database' > outputs/placed.odb\n"
      "printf '%s\\n' '{\"schema\":\"loom.openroad_physical_attempt\","
      "\"version\":\"1.0\",\"stage\":\"placed\",\"top\":\"top\"}' "
      "> outputs/placed-result.json\n";
  writeText(test, tool, toolBody, true);
  loom::external_tool::LocalToolConfig local =
      loom::external_tool::defaultLocalToolConfig();
  local.runtimePolicy = loom::external_tool::RuntimePolicy::Host;
  local.tools["openroad"].binding.executable = tool.string();

  loom::external_tool::ExternalToolProviderDescriptor provider{
      loom::external_tool::ToolProviderDescriptor{
          "openroad", {"openroad"}, {}, {}},
      loom::external_tool::ToolVersionProbe{
          {"--version"}, kSyntheticOpenRoadBuild.str(), {0}, std::nullopt},
      loom::external_tool::ToolRuntimeCompatibility{}};
  loom::external_tool::ResolvedToolBinding resolvedTool{
      "openroad",
      loom::external_tool::ToolBindingSource::Explicit,
      tool.string(),
      kSyntheticOpenRoadBuild.str(),
      {},
      {},
      std::nullopt,
      std::nullopt};
  loom::external_tool::InvocationRuntimeBinding runtime;
  runtime.kind = loom::external_tool::InvocationRuntimeKind::Host;
  InvocationHarness harness =
      makeInvocationHarness(test, root, fixture,
                            OpenRoadResolvedExecution{std::move(provider),
                                                      std::move(resolvedTool),
                                                      std::move(runtime),
                                                      {}});
  harness.context.localConfig.tools = std::move(local.tools);
  return harness;
}

void writeFailingSyntheticOpenRoad(llvm::StringRef test,
                                   const std::filesystem::path &root) {
  const std::string toolBody =
      "#!/usr/bin/env bash\n"
      "set -eu\n"
      "if [[ ${1-} == -version || ${1-} == --version ]]; then\n"
      "  printf '%s\\n' '" +
      kSyntheticOpenRoadBuild.str() +
      "'\n"
      "  exit 0\n"
      "fi\n"
      "exit 37\n";
  writeText(test, root / "synthetic-openroad", toolBody, true);
}

OpenRoadPlacementParameters parameters() {
  return OpenRoadPlacementParameters{
      {0, 0, 100000, 100000},
      {10000, 10000, 90000, 90000},
      "CoreSite",
      "Metal2",
      "Metal3",
      550000,
  };
}

OpenRoadPlacedDriverFiles files() {
  return OpenRoadPlacedDriverFiles{
      {"inputs/netlist/0000.v"}, {"inputs/constraints/0000.sdc"},
      "/public/tech.lef",        {"/public/cells.lef"},
      {"/public/cells.lib"},
  };
}

loom::ExternalFileFingerprint fingerprint(char digit) {
  return take(__func__,
              loom::parseExternalFileFingerprint(std::string(64, digit)));
}

loom::platform::TechnologyCornerRef corner() {
  return loom::platform::TechnologyCornerRef{
      take(__func__, loom::parseArtifactIdentityHex(std::string(64, 'a'))),
      loom::platform::TechnologyCornerId(2)};
}

OpenRoadPlacedConfig config() {
  return OpenRoadPlacedConfig{
      "OpenROAD authored test build",
      corner(),
      parameters(),
      {{OpenRoadExternalFileKind::TechnologyLef, "technology",
        fingerprint('1')},
       {OpenRoadExternalFileKind::CellLef, "cells", fingerprint('2')},
       {OpenRoadExternalFileKind::Liberty, "timing", fingerprint('3')}}};
}

void driverIsDeterministicAndPlacedOnly() {
  const std::string driver =
      take(__func__, renderOpenRoadPlacedDriver("top", parameters(), files()));
  require(__func__,
          take(__func__, renderOpenRoadPlacedDriver("top", parameters(),
                                                    files())) == driver,
          "identical typed inputs changed the driver");
  for (llvm::StringRef command :
       {"read_lef", "read_liberty", "read_verilog", "link_design top",
        "initialize_floorplan", "make_tracks", "place_pins", "global_placement",
        "detailed_placement", "check_placement", "write_db outputs/placed.odb"})
    require(__func__, llvm::StringRef(driver).contains(command),
            "driver omitted required command " + command.str());
  for (llvm::StringRef forbidden :
       {"write_def", "write_spef", "write_gds", "report_checks", "report_power",
        "detailed_route"})
    require(__func__, !llvm::StringRef(driver).contains(forbidden),
            "placed driver claimed or wrote a later-stage result");
  require(__func__,
          llvm::StringRef(driver).contains(
              "place_pins -hor_layers Metal2 -ver_layers Metal3\n"),
          "pin placement did not select deterministic matching");
  require(__func__,
          llvm::StringRef(driver).contains("outputs/placed-result.json"),
          "driver omitted the declared result protocol");
}

void driverRejectsAmbiguousInputs() {
  for (llvm::StringRef top : {"", "top name", "top;exit", "top\\name"})
    expectInvalid(__func__,
                  renderOpenRoadPlacedDriver(top, parameters(), files()));

  OpenRoadPlacementParameters invalid = parameters();
  invalid.coreArea.upperXNanometers = invalid.dieArea.upperXNanometers + 1;
  expectInvalid(__func__, renderOpenRoadPlacedDriver("top", invalid, files()));

  OpenRoadPlacedDriverFiles noNetlist = files();
  noNetlist.netlists.clear();
  expectInvalid(__func__,
                renderOpenRoadPlacedDriver("top", parameters(), noNetlist));
}

void resultProtocolIsCanonical() {
  const OpenRoadPlacedAttemptResult result =
      take(__func__, parseOpenRoadPlacedAttemptResult(
                         "{\"schema\":\"loom.openroad_physical_attempt\","
                         "\"version\":\"1.0\",\"stage\":\"placed\","
                         "\"top\":\"top\"}\n"));
  require(__func__, result.topModule == "top",
          "result parser lost the exact top module");
  for (llvm::StringRef invalid :
       {"{}", "not json",
        "{\"schema\":\"loom.openroad_physical_attempt\","
        "\"version\":\"1.0\",\"stage\":\"routed\","
        "\"top\":\"top\"}\n",
        "{\"schema\":\"loom.openroad_physical_attempt\","
        "\"version\":\"1.0\",\"stage\":\"placed\","
        "\"top\":\"top\",\"claim\":\"signoff\"}\n"})
    expectInvalid(__func__, parseOpenRoadPlacedAttemptResult(invalid));
}

void configAndDescriptorAreTypedAndCanonical() {
  const OpenRoadPlacedConfig authored = config();
  const std::vector<std::uint8_t> canonical =
      take(__func__, encodeOpenRoadPlacedConfig(authored));
  require(__func__,
          take(__func__, decodeOpenRoadPlacedConfig(canonical)) == authored,
          "config codec lost an exact typed value");

  OpenRoadPlacedConfig reordered = authored;
  std::reverse(reordered.externalFiles.begin(), reordered.externalFiles.end());
  require(__func__,
          take(__func__, encodeOpenRoadPlacedConfig(reordered)) == canonical,
          "external-file authoring order changed config identity");
  OpenRoadPlacedConfig anotherBuild = authored;
  anotherBuild.providerBuild = "OpenROAD another test build";
  require(__func__,
          take(__func__, encodeOpenRoadPlacedConfig(anotherBuild)) != canonical,
          "provider build did not contribute to candidate semantics");

  OpenRoadPlacedConfig incomplete = authored;
  incomplete.externalFiles.pop_back();
  expectInvalid(__func__, encodeOpenRoadPlacedConfig(incomplete));
  OpenRoadPlacedConfig duplicate = authored;
  duplicate.externalFiles.push_back(duplicate.externalFiles.front());
  expectInvalid(__func__, encodeOpenRoadPlacedConfig(duplicate));

  require(__func__, !openRoadPlacedConfigSchemaDescriptorBytes().empty(),
          "config schema descriptor is empty");
  const loom::ComponentViewDigest digest = take(
      __func__, loom::computeComponentViewDigest(
                    openRoadPlacedConfigSchemaDescriptorBytes(), canonical));
  require(__func__, !validateCanonicalOpenRoadPlacedConfig(canonical, digest),
          "canonical config did not validate against its derived digest");

  if (llvm::Error error = registerOpenRoadPlacedCandidateGenerator())
    fail(__func__, llvm::toString(std::move(error)));
  const loom::dse::CandidateGeneratorDescriptor &descriptor =
      openRoadPlacedCandidateGeneratorDescriptor();
  require(__func__, descriptor.kind.ordinal() == 11,
          "OpenROAD descriptor kind is not the next production ordinal");
  require(
      __func__,
      descriptor.inputSlots.size() == 1 && descriptor.outputSlots.size() == 1 &&
          descriptor.providerForm == loom::ProviderForm::ExternalPrepareImport,
      "OpenROAD descriptor lost its exact external generator shape");
  (void)take(__func__, loom::dse::ResolvedCandidateGeneratorBinding::get(
                           descriptor.reference(), canonical, digest));
}

void invocationBundleIsTheOnlyAttemptLifecycle(
    const std::filesystem::path &root) {
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const loom::ArtifactStore artifacts((root / "artifacts").string());
  const loom::BlobStore blobs((root / "blobs").string());
  const InvocationFixture fixture = makeInvocationFixture(
      __func__, artifacts, blobs, kSyntheticOpenRoadBuild);
  InvocationHarness harness =
      makeSyntheticInvocationHarness(__func__, root, fixture);

  const auto prepareAt = [&](llvm::StringRef name) {
    loom::external_tool::ExternalToolPreparationContext context =
        harness.context;
    context.bundleDestination = (root / name.str()).string();
    return take(__func__, loom::dse::prepareCandidateGeneratorInvocation(
                              harness.inputs, harness.binding, artifacts, blobs,
                              context));
  };
  const loom::external_tool::PreparedExternalToolInvocation first =
      prepareAt("bundle-a");
  const loom::external_tool::PreparedExternalToolInvocation second =
      prepareAt("bundle-b");
  require(__func__, first.manifestDigest == second.manifestDigest,
          "fresh destinations changed the deterministic manifest");
  require(__func__, std::filesystem::exists(root / "tool-was-probed"),
          "registered preparation did not resolve the OpenROAD tool");

  OpenRoadResolvedExecution mismatchedExecution = harness.execution;
  mismatchedExecution.tool.version += "-different";
  loom::external_tool::ExternalToolPreparationContext mismatchContext =
      harness.context;
  mismatchContext.bundleDestination =
      (root / "bundle-version-mismatch").string();
  expectFailure(__func__,
                prepareOpenRoadPlacedInvocation(
                    harness.inputs, harness.binding, fixture.contracts,
                    artifacts, blobs, mismatchedExecution, mismatchContext));
  require(__func__, !std::filesystem::exists(root / "bundle-version-mismatch"),
          "rejected preparation finalized a bundle");

  const std::string manifest =
      readText(__func__, root / "bundle-a" / "tool-invocation.json");
  const loom::external_tool::ExternalToolSemanticContract semanticContract =
      take(__func__, loom::dse::deriveExternalToolSemanticContract(
                         harness.inputs, harness.binding));
  require(__func__,
          llvm::StringRef(manifest).contains("\"provider_identity\": \"" +
                                             semanticContract.providerIdentity +
                                             "\"") &&
              llvm::StringRef(manifest).contains(
                  "\"result_importer_identity\": \"" +
                  semanticContract.resultImporterIdentity + "\""),
          "bundle did not transport the owner-derived semantic contract");
  require(__func__,
          llvm::StringRef(manifest).contains("[\n      \"" +
                                             harness.execution.tool.executable +
                                             "\",\n      \"-no_init\"") &&
              llvm::StringRef(manifest).contains("\"drivers/openroad.tcl\""),
          "OpenROAD command was not preserved as structured tokens");
  require(__func__,
          llvm::StringRef(manifest).contains(
              "\"typed_input_bindings\": \"000000000000000100000000"
              "00000000000000010000001c") &&
              llvm::StringRef(manifest).contains(
                  "\"resolved_binding\": \"0000000000000023"),
          "candidate closure did not use the central owner codec");
  for (llvm::StringRef slot :
       {"technology_lef", "cell_lef.cells", "liberty.timing"})
    require(__func__,
            llvm::StringRef(manifest).contains("\"provider_input_slot\": \"" +
                                               slot.str() + "\""),
            "manifest omitted exact external input slot " + slot.str());
  const std::string driver =
      readText(__func__, root / "bundle-a" / "drivers" / "openroad.tcl");
  require(__func__,
          llvm::StringRef(driver).contains(
              (root / "external" / "technology.lef").string()),
          "driver did not project the frozen external-file resolution");

  expectFailure(__func__, importOpenRoadPlacedInvocation(
                              harness.inputs, harness.binding, first,
                              fixture.contracts, artifacts, blobs));

  writeText(__func__, root / "bundle-a" / "outputs" / "placed.odb",
            "stale database\n");
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         first)) == 0,
      "synthetic OpenROAD invocation did not complete");
  const loom::dse::CandidateGeneratorProviderResult imported = take(
      __func__, loom::dse::importCandidateGeneratorInvocation(
                    harness.inputs, harness.binding, first, artifacts, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &imported.outcome);
  require(__func__,
          completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().slot ==
                  loom::dse::CandidateGeneratorOutputSlotRef(0) &&
              completed->outputBindings.front().artifacts.size() == 1 &&
              completed->lineageEdges.size() == 1 &&
              completed->lineageEdges.front().kind ==
                  loom::dse::CandidateGeneratorLineageEdgeKind::
                      MechanicalDerivation &&
              completed->lineageEdges.front().output ==
                  completed->outputBindings.front().artifacts.front(),
          "valid physical attempt did not publish one placed implementation");
  require(__func__,
          imported.workSummary ==
              std::vector<loom::dse::CandidateGeneratorWorkUnitSummary>{
                  {loom::dse::CandidateGeneratorWorkUnitRef(0), 1, 1}},
          "valid physical attempt lost exact work accounting");
  const loom::hardware::FinalizedHardwareImplementation placed =
      take(__func__, loom::hardware::importHardwareImplementation(
                         completed->outputBindings.front().artifacts.front(),
                         fixture.contracts, artifacts, blobs));
  const loom::hardware::ImplementationRepresentationRoot &placedRoot =
      placed.implementation().representationRoot();
  require(__func__,
          placed.implementation().fabric() ==
                  fixture.gate.implementation().fabric() &&
              placed.implementation().configurationAbi() ==
                  fixture.gate.implementation().configurationAbi() &&
              placed.implementation().implementationPlatform() ==
                  fixture.gate.implementation().implementationPlatform() &&
              placedRoot.variant ==
                  loom::hardware::RepresentationRootVariant::AsicPhysical &&
              placedRoot.stage ==
                  loom::hardware::RepresentationPhysicalStage::Placed &&
              placedRoot.formatRef.kind() ==
                  loom::hardware::RepresentationFormatKind::IndexedPhysical &&
              placedRoot.top ==
                  loom::hardware::RepresentationLocator{
                      loom::hardware::RepresentationObjectKind::PhysicalObject,
                      "top"},
          "placed implementation lost its exact source and platform state");
  const auto databasePayload = llvm::find_if(
      placedRoot.payloads, [](const loom::hardware::ImplementationPayload &p) {
        return p.role == loom::hardware::PayloadRole::PhysicalDatabase;
      });
  require(__func__, databasePayload != placedRoot.payloads.end(),
          "placed implementation omitted the physical database payload");
  require(__func__,
          take(__func__, blobs.get(databasePayload->blobDigest)) ==
              bytes("synthetic placed database\n"),
          "placed implementation did not publish the declared database");
  const auto placedExternal =
      placed.implementation().externalImplementationBindings();
  const auto gateExternal =
      fixture.gate.implementation().externalImplementationBindings();
  require(__func__,
          llvm::none_of(
              placedRoot.payloads,
              [](const loom::hardware::ImplementationPayload &payload) {
                return payload.role == loom::hardware::PayloadRole::Netlist;
              }) &&
              placedExternal.size() == 1 && gateExternal.size() == 1 &&
              placedExternal.front().providerContractRef ==
                  gateExternal.front().providerContractRef &&
              placedExternal.front().externalInputs ==
                  gateExternal.front().externalInputs &&
              placedExternal.front().representationLocators ==
                  gateExternal.front().representationLocators &&
              placedExternal.front().blackBoxContractPayloadRef ==
                  gateExternal.front().blackBoxContractPayloadRef,
          "placed implementation leaked the input netlist or lost external "
          "bindings");
  require(__func__,
          readText(__func__, root / "bundle-a" / "outputs" / "placed.odb") ==
              "synthetic placed database\n",
          "execution did not remove and replace stale declared output");

  const std::string workingSyntheticTool =
      readText(__func__, root / "synthetic-openroad");
  writeFailingSyntheticOpenRoad(__func__, root);
  const auto failed = prepareAt("bundle-failed");
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         failed)) == 37,
      "synthetic OpenROAD failure did not preserve its exit code");
  const loom::dse::CandidateGeneratorProviderResult failedImport = take(
      __func__, loom::dse::importCandidateGeneratorInvocation(
                    harness.inputs, harness.binding, failed, artifacts, blobs));
  const auto *executionFailed =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &failedImport.outcome);
  require(
      __func__,
      executionFailed &&
          executionFailed->reason ==
              loom::dse::CandidateGeneratorIncompleteReason::ExecutionFailed &&
          executionFailed->retainedOutputBindings.size() == 1 &&
          executionFailed->retainedOutputBindings.front().artifacts.empty() &&
          failedImport.workSummary ==
              std::vector<loom::dse::CandidateGeneratorWorkUnitSummary>{
                  {loom::dse::CandidateGeneratorWorkUnitRef(0), 1, 1}},
      "failed completion did not remain a typed non-publishing result");
  writeText(__func__, root / "synthetic-openroad", workingSyntheticTool, true);

  OpenRoadPlacedConfig changedConfig = fixture.config;
  changedConfig.placement.placementDensityPpm += 1000;
  const std::vector<std::uint8_t> changedBytes =
      take(__func__, encodeOpenRoadPlacedConfig(changedConfig));
  const loom::ComponentViewDigest changedDigest = take(
      __func__, loom::computeComponentViewDigest(
                    openRoadPlacedConfigSchemaDescriptorBytes(), changedBytes));
  const auto changedBinding = take(
      __func__, loom::dse::ResolvedCandidateGeneratorBinding::get(
                    openRoadPlacedCandidateGeneratorDescriptor().reference(),
                    changedBytes, changedDigest));
  expectFailure(__func__, importOpenRoadPlacedInvocation(
                              harness.inputs, changedBinding, first,
                              fixture.contracts, artifacts, blobs));

  const auto missing = prepareAt("bundle-missing");
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         missing)) == 0,
      "missing-output fixture did not execute");
  std::filesystem::remove(root / "bundle-missing" / "outputs" / "placed.odb");
  expectFailure(__func__, importOpenRoadPlacedInvocation(
                              harness.inputs, harness.binding, missing,
                              fixture.contracts, artifacts, blobs));

  const auto tampered = prepareAt("bundle-tampered");
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         tampered)) == 0,
      "tamper fixture did not execute");
  writeText(__func__,
            root / "bundle-tampered" / "outputs" / "placed-result.json",
            "{}\n");
  expectFailure(__func__, importOpenRoadPlacedInvocation(
                              harness.inputs, harness.binding, tampered,
                              fixture.contracts, artifacts, blobs));

  const auto partial = prepareAt("bundle-partial");
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         partial)) == 0,
      "partial fixture did not execute");
  std::filesystem::remove(root / "bundle-partial" / "outputs" /
                          "completion.json");
  expectFailure(__func__, importOpenRoadPlacedInvocation(
                              harness.inputs, harness.binding, partial,
                              fixture.contracts, artifacts, blobs));

  const auto undeclared = prepareAt("bundle-undeclared");
  const std::filesystem::path undeclaredManifest =
      root / "bundle-undeclared" / "tool-invocation.json";
  std::string altered = readText(__func__, undeclaredManifest);
  const std::string declaration =
      "    \"outputs/placed-result.json\"\n  ],\n  \"stdout\"";
  const std::string extraDeclaration =
      "    \"outputs/placed-result.json\",\n"
      "    \"outputs/undeclared.odb\"\n  ],\n  \"stdout\"";
  const std::size_t declarationPosition = altered.find(declaration);
  require(__func__, declarationPosition != std::string::npos,
          "could not locate canonical declared-output array");
  altered.replace(declarationPosition, declaration.size(), extraDeclaration);
  writeText(__func__, undeclaredManifest, altered);
  const loom::external_tool::PreparedExternalToolInvocation alteredHandle{
      undeclared.bundleRoot, loom::computeBlobDigest(bytes(altered))};
  expectFailure(__func__, importOpenRoadPlacedInvocation(
                              harness.inputs, harness.binding, alteredHandle,
                              fixture.contracts, artifacts, blobs));
}

void realOpenRoadPlacedSmoke(const std::filesystem::path &root,
                             llvm::StringRef executable,
                             llvm::StringRef version) {
  version = version.trim();
  require(__func__, std::filesystem::path(executable.str()).is_absolute(),
          "real OpenROAD executable is not absolute");
  require(__func__, version.contains("b9a38929e"),
          "real OpenROAD version is not the 2026-08-06 build");
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const loom::ArtifactStore artifacts((root / "artifacts").string());
  const loom::BlobStore blobs((root / "blobs").string());
  const InvocationFixture fixture =
      makeInvocationFixture(__func__, artifacts, blobs, version);
  requireSuccess(__func__,
                 registerOpenRoadPlacedCandidateGeneratorDescriptor());

  constexpr llvm::StringLiteral module = "openroad/2026.08.06-b9a38929e342";
  loom::external_tool::ExternalToolProviderDescriptor provider{
      loom::external_tool::ToolProviderDescriptor{
          "openroad", {"openroad"}, {}, {module.str()}},
      loom::external_tool::ToolVersionProbe{
          {"-version"}, "b9a38929e", {0}, std::nullopt},
      loom::external_tool::ToolRuntimeCompatibility{}};
  loom::external_tool::ResolvedToolBinding resolvedTool{
      "openroad",
      loom::external_tool::ToolBindingSource::Module,
      executable.str(),
      version.str(),
      {module.str()},
      {module.str()},
      "/etc/profile.d/modules.sh",
      std::nullopt};
  loom::external_tool::InvocationRuntimeBinding runtime;
  runtime.kind = loom::external_tool::InvocationRuntimeKind::Host;
  InvocationHarness harness =
      makeInvocationHarness(__func__, root, fixture,
                            OpenRoadResolvedExecution{std::move(provider),
                                                      std::move(resolvedTool),
                                                      std::move(runtime),
                                                      {}});

  const loom::external_tool::PreparedExternalToolInvocation prepared =
      take(__func__, prepareOpenRoadPlacedInvocation(
                         harness.inputs, harness.binding, fixture.contracts,
                         artifacts, blobs, harness.execution, harness.context));
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         prepared)) == 0,
      "real OpenROAD invocation did not complete");
  const loom::dse::CandidateGeneratorProviderResult imported = take(
      __func__,
      importOpenRoadPlacedInvocation(harness.inputs, harness.binding, prepared,
                                     fixture.contracts, artifacts, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &imported.outcome);
  require(__func__,
          completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().artifacts.size() == 1,
          "real placed attempt did not publish one implementation");
  const loom::hardware::FinalizedHardwareImplementation placed =
      take(__func__, loom::hardware::importHardwareImplementation(
                         completed->outputBindings.front().artifacts.front(),
                         fixture.contracts, artifacts, blobs));
  require(__func__,
          placed.implementation().representationRoot().variant ==
                  loom::hardware::RepresentationRootVariant::AsicPhysical &&
              placed.implementation().representationRoot().stage ==
                  loom::hardware::RepresentationPhysicalStage::Placed,
          "real OpenROAD output did not remain exactly placed");
  require(__func__,
          std::filesystem::file_size(root / "bundle" / "outputs" /
                                     "placed.odb") > 0,
          "real OpenROAD emitted an empty placed database");
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 5 && llvm::StringRef(argv[1]) == "--real-smoke") {
    realOpenRoadPlacedSmoke(
        std::filesystem::absolute(argv[2]).lexically_normal(), argv[3],
        argv[4]);
    return EXIT_SUCCESS;
  }
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  driverIsDeterministicAndPlacedOnly();
  driverRejectsAmbiguousInputs();
  resultProtocolIsCanonical();
  configAndDescriptorAreTypedAndCanonical();
  invocationBundleIsTheOnlyAttemptLifecycle(
      std::filesystem::absolute(argv[1]).lexically_normal());
  return EXIT_SUCCESS;
}
