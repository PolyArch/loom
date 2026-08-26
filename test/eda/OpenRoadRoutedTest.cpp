#include "OpenRoadPhysicalTestSupport.h"

#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "ExternalTool/InvocationBundle.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::dse;
using namespace loom::eda;
using namespace loom::eda::open_source;
using namespace loom::eda::open_source::test;
using namespace loom::external_tool;
using namespace loom::hardware;

namespace {

constexpr llvm::StringLiteral kSyntheticBuild =
    "OpenROAD synthetic 21512b0ab68c";

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

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void driverAndResultProtocolsAreStrict() {
  const OpenRoadTechnologyFixture technology =
      syntheticOpenRoadTechnologyFixture();
  const OpenRoadRoutedDriverFiles gate{OpenRoadRoutedInputKind::GateNetlist,
                                       {"inputs/netlist/0000.v"},
                                       std::nullopt,
                                       {"inputs/constraints/0000.sdc"},
                                       "/external/technology.lef",
                                       {"/external/cells.lef"},
                                       {"/external/cells.lib"}};
  const std::string driver = take(
      __func__, renderOpenRoadRoutedDriver("top", technology.placement, gate));
  require(__func__,
          take(__func__, renderOpenRoadRoutedDriver("top", technology.placement,
                                                    gate)) == driver,
          "identical inputs changed the routed driver");
  for (llvm::StringRef command :
       {"initialize_floorplan", "global_placement -density 0.55",
        "clock_tree_synthesis -repair_clock_nets", "detailed_placement",
        "global_route -congestion_iterations 30", "detailed_route -or_seed 1",
        "write_verilog -sort", "write_def -version 5.8"})
    require(__func__, llvm::StringRef(driver).contains(command),
            "routed driver omitted " + command.str());

  OpenRoadRoutedDriverFiles placed = gate;
  placed.inputKind = OpenRoadRoutedInputKind::PlacedDatabase;
  placed.netlists.clear();
  placed.placedDatabase = "inputs/database/placed.odb";
  const std::string placedDriver =
      take(__func__,
           renderOpenRoadRoutedDriver("top", technology.placement, placed));
  require(__func__,
          llvm::StringRef(placedDriver).contains("read_db") &&
              !llvm::StringRef(placedDriver).contains("initialize_floorplan") &&
              !llvm::StringRef(placedDriver).contains("global_placement"),
          "placed input was not consumed as an exact placed database");

  OpenRoadRoutedDriverFiles noConstraints = gate;
  noConstraints.constraints.clear();
  expectErrorContains(
      __func__,
      renderOpenRoadRoutedDriver("top", technology.placement, noConstraints),
      "constraint closure is empty");
  OpenRoadRoutedDriverFiles ambiguous = gate;
  ambiguous.placedDatabase = "placed.odb";
  expectErrorContains(
      __func__,
      renderOpenRoadRoutedDriver("top", technology.placement, ambiguous),
      "closure is inconsistent");

  const OpenRoadRoutedAttemptResult result = take(
      __func__, parseOpenRoadRoutedAttemptResult(
                    "{\"schema\":\"loom.openroad_routed_physical_attempt\","
                    "\"version\":\"1.0\",\"stage\":\"routed\","
                    "\"top\":\"top\"}\n"));
  require(__func__, result.topModule == "top", "result parser lost the top");
  for (llvm::StringRef invalid :
       {"{}\n",
        "{\"schema\":\"loom.openroad_routed_physical_attempt\","
        "\"version\":\"1.0\",\"stage\":\"placed\","
        "\"top\":\"top\"}\n",
        "{\"schema\":\"loom.openroad_routed_physical_attempt\","
        "\"version\":\"1.0\",\"stage\":\"routed\","
        "\"top\":\"top\",\"claim\":\"signoff\"}\n"})
    expectErrorContains(__func__, parseOpenRoadRoutedAttemptResult(invalid),
                        "invalid");
}

void authoredLifecyclePublishesOnlyThroughImporter(
    const std::filesystem::path &root) {
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const ArtifactStore artifacts((root / "artifacts").string());
  const BlobStore blobs((root / "blobs").string());
  const OpenRoadGateFixture fixture = take(
      __func__, makeOpenRoadGateFixture(root, artifacts, blobs, kSyntheticBuild,
                                        syntheticOpenRoadTechnologyFixture()));
  requireSuccess(__func__, registerOpenRoadRoutedCandidateGenerator());

  auto prepareAt = [&](llvm::StringRef name,
                       AuthoredOpenRoadRouteBehavior behavior) {
    const std::filesystem::path tool =
        take(__func__, writeAuthoredOpenRoadRouteTool(root, behavior));
    const LocalToolConfig local = makeOpenRoadLocalToolConfig(fixture, tool);
    OpenRoadRouteHarness harness = take(
        __func__, makeOpenRoadRouteHarness(root / name.str(), fixture, local));
    auto prepared = take(__func__, prepareCandidateGeneratorInvocation(
                                       harness.inputs, harness.binding,
                                       artifacts, blobs, harness.context));
    return std::pair<OpenRoadRouteHarness, PreparedExternalToolInvocation>{
        std::move(harness), std::move(prepared)};
  };

  auto [firstHarness, first] =
      prepareAt("route-complete", AuthoredOpenRoadRouteBehavior::Complete);
  const std::string driver =
      take(__func__, readText(root / "route-complete" / "drivers" /
                              "openroad-routed.tcl"));
  require(
      __func__,
      llvm::StringRef(driver).contains("inputs/netlist/0000.v") &&
          llvm::StringRef(driver).contains(fixture.technologyLefPath.string()),
      "prepared route did not bind the exact semantic and external inputs");
  expectErrorContains(__func__,
                      importOpenRoadRoutedInvocation(
                          firstHarness.inputs, firstHarness.binding, first,
                          fixture.contracts, artifacts, blobs),
                      "incomplete");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(first)) == 0,
          "authored route invocation failed");
  const CandidateGeneratorProviderResult imported =
      take(__func__, importCandidateGeneratorInvocation(
                         firstHarness.inputs, firstHarness.binding, first,
                         artifacts, blobs));
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&imported.outcome);
  require(__func__,
          completed && completed->outputBindings.size() == 1 &&
              completed->outputBindings.front().artifacts.size() == 1 &&
              completed->lineageEdges.size() == 1 &&
              completed->lineageEdges.front().kind ==
                  CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
          "route importer did not publish one mechanical output");
  const FinalizedHardwareImplementation routed =
      take(__func__, importHardwareImplementation(
                         completed->outputBindings.front().artifacts.front(),
                         fixture.contracts, artifacts, blobs));
  const HardwareImplementation &hardware = routed.implementation();
  const ImplementationRepresentationRoot &representation =
      hardware.representationRoot();
  require(__func__,
          hardware.fabric() == fixture.gate.implementation().fabric() &&
              hardware.configurationAbi() ==
                  fixture.gate.implementation().configurationAbi() &&
              hardware.implementationPlatform() ==
                  fixture.gate.implementation().implementationPlatform() &&
              representation.variant ==
                  RepresentationRootVariant::AsicPhysical &&
              representation.stage == RepresentationPhysicalStage::Routed &&
              representation.formatRef.kind() ==
                  RepresentationFormatKind::IndexedDefPhysical,
          "routed HImpl lost its exact parent or routed representation");
  for (PayloadRole role :
       {PayloadRole::Netlist, PayloadRole::PhysicalDatabase,
        PayloadRole::GenerationConstraint, PayloadRole::BlackBoxContract,
        PayloadRole::RepresentationIndex})
    require(__func__,
            llvm::any_of(representation.payloads,
                         [&](const ImplementationPayload &payload) {
                           return payload.role == role;
                         }),
            "routed HImpl omitted a required payload role");
  const auto external = hardware.externalImplementationBindings();
  require(__func__,
          external.size() == 1 &&
              external.front().providerContractRef ==
                  openRoadRoutedStandardCellContractRef &&
              external.front().externalInputs.size() == 3 &&
              external.front().blackBoxContractPayloadRef.has_value() &&
              !external.front().representationLocators.empty(),
          "routed HImpl did not replace the source standard-cell closure");

  auto [failedHarness, failed] =
      prepareAt("route-failed", AuthoredOpenRoadRouteBehavior::ToolFailure);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(failed)) == 37,
          "authored failure did not preserve the tool exit status");
  const CandidateGeneratorProviderResult failedResult =
      take(__func__, importCandidateGeneratorInvocation(
                         failedHarness.inputs, failedHarness.binding, failed,
                         artifacts, blobs));
  const auto *incomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&failedResult.outcome);
  require(__func__,
          incomplete && incomplete->reason ==
                            CandidateGeneratorIncompleteReason::ExecutionFailed,
          "tool failure did not remain a typed non-publishing result");

  auto [missingHarness, missing] =
      prepareAt("route-missing", AuthoredOpenRoadRouteBehavior::MissingOutput);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(missing)) == 122,
          "missing-output fixture lost the bundle status");
  const CandidateGeneratorProviderResult missingResult =
      take(__func__, importCandidateGeneratorInvocation(
                         missingHarness.inputs, missingHarness.binding, missing,
                         artifacts, blobs));
  const auto *missingOutput =
      std::get_if<IncompleteCandidateGeneratorResult>(&missingResult.outcome);
  require(__func__,
          missingOutput &&
              missingOutput->reason ==
                  CandidateGeneratorIncompleteReason::ExecutionFailed,
          "missing route output did not become typed execution failure");

  auto [tamperedHarness, tampered] =
      prepareAt("route-tampered", AuthoredOpenRoadRouteBehavior::Complete);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(tampered)) == 0,
          "tamper fixture failed before mutation");
  requireSuccess(__func__, writeText(root / "route-tampered" / "outputs" /
                                         "routed-result.json",
                                     "{}\n"));
  expectErrorContains(__func__,
                      importOpenRoadRoutedInvocation(
                          tamperedHarness.inputs, tamperedHarness.binding,
                          tampered, fixture.contracts, artifacts, blobs),
                      "completion digest");
}

void realOpenRoadRouteSmoke(const std::filesystem::path &root,
                            llvm::StringRef executable, llvm::StringRef version,
                            const std::filesystem::path &technologyLef,
                            const std::filesystem::path &cellLef,
                            const std::filesystem::path &liberty) {
  version = version.trim();
  require(__func__, std::filesystem::path(executable.str()).is_absolute(),
          "real OpenROAD executable is not absolute");
  require(__func__, version.contains("21512b0"),
          "real OpenROAD version is not the pinned build");
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const ArtifactStore artifacts((root / "artifacts").string());
  const BlobStore blobs((root / "blobs").string());
  const OpenRoadGateFixture fixture =
      take(__func__, makeOpenRoadGateFixture(
                         root, artifacts, blobs, version,
                         take(__func__, loadSaed32OpenRoadTechnologyFixture(
                                            technologyLef, cellLef, liberty))));
  requireSuccess(__func__,
                 registerOpenRoadRoutedCandidateGeneratorDescriptor());
  LocalToolConfig local =
      makeOpenRoadLocalToolConfig(fixture, executable.str());
  OpenRoadRouteHarness harness =
      take(__func__, makeOpenRoadRouteHarness(root / "bundle", fixture, local));
  const FinalizedHardwareImplementation routed =
      take(__func__, runOpenRoadRouteFixture(fixture, harness,
                                             makeOpenRoadResolvedExecution(
                                                 executable, version, true),
                                             artifacts, blobs));
  require(__func__,
          routed.implementation().representationRoot().stage ==
                  RepresentationPhysicalStage::Routed &&
              std::filesystem::file_size(root / "bundle" / "outputs" /
                                         "routed.def") > 0,
          "real OpenROAD route did not publish a nonempty routed HImpl");
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 8 && llvm::StringRef(argv[1]) == "--real-smoke") {
    realOpenRoadRouteSmoke(
        std::filesystem::absolute(argv[2]).lexically_normal(), argv[3], argv[4],
        argv[5], argv[6], argv[7]);
    return EXIT_SUCCESS;
  }
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  driverAndResultProtocolsAreStrict();
  authoredLifecyclePublishesOnlyThroughImporter(
      std::filesystem::absolute(argv[1]).lexically_normal());
  return EXIT_SUCCESS;
}
