#include "EDA/Adapters/Cadence/Innovus.h"
#include "EDA/Adapters/Cadence/Joules.h"
#include "EDA/Adapters/Cadence/Tempus.h"
#include "EDA/Adapters/Cadence/Voltus.h"
#include "EDA/Adapters/Cadence/Xcelium.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
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
void expectFailure(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted invalid adapter input");
  llvm::consumeError(value.takeError());
}

template <typename T>
void expectAdapterFailure(llvm::StringRef test, llvm::Expected<T> value,
                          CadenceAdapterFailureKind expected) {
  if (value)
    fail(test, "expected a typed adapter failure");
  bool matched = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const CadenceAdapterError &error) {
        matched = error.kind() == expected;
      },
      [&](const llvm::ErrorInfoBase &) {});
  require(test, matched, "adapter failure classification changed");
}

template <typename T>
void expectIncomplete(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invocation without completion");
  bool matched = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const IncompleteExternalToolInvocationError &) { matched = true; },
      [&](const llvm::ErrorInfoBase &) {});
  require(test, matched, "incomplete attempt lost its shared typed error");
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents,
               bool executable = false) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail(__func__, "could not write fixture file");
  stream.close();
  if (executable)
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream)
    fail(__func__, "could not read fixture file");
  return contents.str();
}

BlobDigest digest(llvm::StringRef contents) {
  return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(contents.data()),
      contents.size()));
}

ExternalFileFingerprint fingerprint(llvm::StringRef contents) {
  return take(__func__,
              ExternalFileFingerprint::fromBytes(digest(contents).bytes()));
}

ArtifactRootReference reference(llvm::StringRef schema, char digit) {
  return {schema.str(),
          {1, 0},
          take(__func__, parseArtifactIdentityHex(std::string(64, digit)))};
}

ImplementationRepresentationRoot rtlRepresentation() {
  const auto format =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::SystemVerilogRtl));
  return take(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::Rtl, std::nullopt, format,
                  {RepresentationObjectKind::Module, "top"},
                  {{PayloadRole::RtlSource, "rtl/top.sv", digest("rtl")},
                   {PayloadRole::GenerationConstraint, "constraints/top.sdc",
                    digest("sdc")}}));
}

ImplementationRepresentationRoot gateRepresentation() {
  const auto format = take(
      __func__, RepresentationFormatDescriptorRef::get(
                    RepresentationFormatKind::StructuralVerilogGateNetlist));
  return take(__func__,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::GateNetlist, std::nullopt, format,
                  {RepresentationObjectKind::Module, "top"},
                  {{PayloadRole::Netlist, "netlist/top.v", digest("gate")},
                   {PayloadRole::GenerationConstraint, "constraints/top.sdc",
                    digest("sdc")}}));
}

ImplementationRepresentationRoot
physicalRepresentation(const std::filesystem::path &root) {
  std::filesystem::create_directories(root);
  const BlobStore blobs(root.string());
  const auto format =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::IndexedPhysical));
  const ImplementationPayload database{
      PayloadRole::PhysicalDatabase, "database/top.def",
      take(__func__, blobs.put(llvm::ArrayRef<std::uint8_t>{1, 2, 3}))};
  const ImplementationPayload constraint{
      PayloadRole::GenerationConstraint, "constraints/top.sdc",
      take(__func__, blobs.put(llvm::ArrayRef<std::uint8_t>{4, 5, 6}))};
  const RepresentationLocator top{RepresentationObjectKind::PhysicalObject,
                                  "top"};
  auto index =
      take(__func__,
           createPhysicalRepresentationIndexPayload(
               format, RepresentationRootVariant::AsicPhysical,
               RepresentationPhysicalStage::Routed, top, "index/physical.json",
               {database, constraint}, {{top, std::nullopt}}, {}));
  const std::string indexBytes =
      take(__func__, serializePhysicalRepresentationIndexPayloadJson(index));
  const ImplementationPayload indexPayload{
      PayloadRole::RepresentationIndex, "index/physical.json",
      take(__func__,
           blobs.put(llvm::ArrayRef<std::uint8_t>(
               reinterpret_cast<const std::uint8_t *>(indexBytes.data()),
               indexBytes.size())))};
  return take(__func__, createImplementationRepresentationRoot(
                            RepresentationRootVariant::AsicPhysical,
                            RepresentationPhysicalStage::Routed, format, top,
                            {database, constraint, indexPayload}));
}

struct PlatformFixture final {
  ArtifactStore store;
  platform::FinalizedImplementationPlatform platform;
  EncodedArtifactLocalReference corner;
};

PlatformFixture makePlatform(const std::filesystem::path &root) {
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  auto platform = take(
      __func__,
      platform::finalizeImplementationPlatform(
          {platform::AsicTarget{"gpdk045", "fixture-v1"}, {"typical"}}, store));
  const platform::TechnologyCornerRef corner{platform.reference().artifact,
                                             platform::TechnologyCornerId(0)};
  return {std::move(store), std::move(platform),
          platform::encodeTechnologyCornerRef(corner)};
}

SemanticInvocationClosure generatorClosure() {
  CandidateGeneratorInvocationClosure closure;
  closure.typedInputBindings = {1, 2, 3};
  closure.resolvedBinding = {4, 5, 6};
  closure.bindingIdentity = digest("generator binding").bytes();
  return closure;
}

SemanticInvocationClosure evaluationClosure() {
  return reference("loom.evaluation_request", 'a');
}

ExternalToolSemanticContract
semanticContract(const CadenceInvocationDescriptor &descriptor,
                 SemanticInvocationClosure closure) {
  return {
      descriptor.implementationSemanticIdentity.str(), std::move(closure),
      formatBlobDigestHex(digest(descriptor.implementationSemanticIdentity))};
}

CadenceFrozenInvocation
frozen(const std::filesystem::path &tool, llvm::StringRef toolKey,
       llvm::ArrayRef<std::pair<std::string, std::filesystem::path>> external) {
  CadenceFrozenInvocation result;
  result.tool = {toolKey.str(),
                 ToolBindingSource::Explicit,
                 std::filesystem::canonical(tool).string(),
                 "fixture-tool 1.0",
                 {},
                 {},
                 std::nullopt,
                 std::nullopt};
  result.toolVersionProbe = {{"--version"}, "fixture-tool 1.0"};
  for (const auto &[slot, path] : external)
    result.externalFiles.push_back({slot, slot,
                                    std::filesystem::canonical(path).string(),
                                    fingerprint(readFile(path))});
  return result;
}

ResolvedExternalFileTree fileTree(const std::filesystem::path &root,
                                  llvm::StringRef slot) {
  writeFile(root / "cells/stdcells.cl", "standard cells\n");
  writeFile(root / "technology.cl", "technology\n");
  return {slot.str(),
          slot.str(),
          std::filesystem::canonical(root).string(),
          {{"cells/stdcells.cl", fingerprint("standard cells\n")},
           {"technology.cl", fingerprint("technology\n")}}};
}

std::vector<MaterializedBundleFile>
semanticFiles(llvm::ArrayRef<std::pair<std::string, std::string>> files) {
  std::vector<MaterializedBundleFile> result;
  const ArtifactRootReference source =
      reference("loom.hardware_implementation", 'b');
  for (const auto &[path, contents] : files)
    result.push_back({path, contents, source, false});
  return result;
}

CadenceBundleInputs
bundleInputs(const ImplementationRepresentationRoot &implementation,
             const CadenceInvocationDescriptor &descriptor,
             SemanticInvocationClosure closure, CadenceFrozenInvocation binding,
             std::vector<MaterializedBundleFile> files,
             const PlatformFixture *platform = nullptr) {
  CadenceBundleInputs result;
  result.semanticContract = semanticContract(descriptor, std::move(closure));
  result.implementation = &implementation;
  if (platform) {
    result.implementationPlatform = platform->platform.reference();
    result.platform = &platform->platform;
    result.technologyCorner = platform->corner;
  }
  result.frozen = std::move(binding);
  result.semanticInputs = std::move(files);
  return result;
}

evaluation::models::CompleteRailAnalysisConfiguration
railConfiguration(const PlatformFixture &platform,
                  const ResolvedExternalFileTree &powerGridLibrary) {
  const ArtifactRootReference hardware =
      reference("loom.hardware_implementation", 'b');
  const evaluation::SubjectTargetRef target{
      evaluation::models::hardwareImplementationPhysicalSubjectRole(), hardware,
      evaluation::SubjectTarget{hardware}};
  const evaluation::ExplicitAssumptionSource activity{
      target, take(__func__, evaluation::ExactRatio::get(1, 2)),
      take(__func__, evaluation::ExactRatio::get(1, 10))};
  return {evaluation::models::staticExplicitRailAnalysisModelConfig(),
          {"fixture-tool 1.0",
           powerGridLibrary.members,
           {"technology.cl", "cells/stdcells.cl"}},
          {target,
           {platform.platform.reference().artifact,
            platform::TechnologyCornerId(0)}},
          {target, take(__func__, evaluation::DecimalValue::get(9, -1))},
          {target, take(__func__, evaluation::DecimalValue::get(3, 2))},
          {target, take(__func__, evaluation::DecimalValue::get(2, -9))},
          {target, activity}};
}

bool hasStrictBatchEnvelope(llvm::StringRef driver) {
  return driver.starts_with("proc loom_main {} {\n") &&
         driver.contains("catch {loom_main} loom_error") &&
         !driver.contains("get_message_info") && driver.ends_with("exit 0\n");
}

void descriptorsAndParsersAreExact(const std::filesystem::path &root) {
  const auto &xcelium = xceliumFunctionalDescriptor();
  const auto &innovus = innovusDescriptor();
  const auto &joules = joulesPowerDescriptor();
  const auto &tempus = tempusTimingDescriptor();
  const auto &voltus = voltusRailDescriptor();
  require(__func__,
          xcelium.toolProvider == &external_tool::xceliumProvider() &&
              xcelium.operation == CadenceOperation::FunctionalEvaluation &&
              !xcelium.requiresAsicPlatform &&
              innovus.toolProvider == &external_tool::innovusProvider() &&
              innovus.operation == CadenceOperation::PhysicalImplementation &&
              innovus.requiredProviderInputs.size() == 4 &&
              joules.operation == CadenceOperation::PowerEvaluation &&
              tempus.operation == CadenceOperation::TimingEvaluation &&
              voltus.operation == CadenceOperation::RailEvaluation &&
              voltus.requiredProviderInputs.size() == 1 &&
              voltus.requiredProviderInputs.front() == "power_grid_library",
          "Cadence descriptor closure changed");

  const std::vector<std::string> command = take(
      __func__, renderXceliumFunctionalCommand(
                    "/tools/xrun", "loom_testbench",
                    {"inputs/rtl/top.sv", "inputs/workload/testbench.sv"}));
  require(__func__,
          command ==
              std::vector<std::string>({"/tools/xrun", "-64bit", "-sv", "-top",
                                        "loom_testbench", "inputs/rtl/top.sv",
                                        "inputs/workload/testbench.sv"}),
          "Xcelium command is not exact");
  expectFailure(__func__, renderXceliumFunctionalCommand("xrun", "top",
                                                         {"inputs/top.sv"}));
  const auto functional = take(
      __func__, parseXceliumFunctionalResult(
                    "{\"schema\":\"loom.cadence.xcelium_functional_result\","
                    "\"version\":\"1.0\",\"status\":\"passed\","
                    "\"completed_transactions\":3}\n"));
  require(__func__,
          functional.status == XceliumFunctionalStatus::Passed &&
              functional.completedTransactions == 3,
          "Xcelium result changed");

  const std::string innovusDriver =
      take(__func__, renderInnovusDriver("top", "inputs/netlist/top.v",
                                         "inputs/physical/floorplan.def",
                                         "/pdk/tech.lef", "/pdk/cells.lef"));
  const std::string innovusMmmc =
      take(__func__, renderInnovusMmmcDriver("inputs/constraints/top.sdc",
                                             "/pdk/slow.lib", "/pdk/qrc.tch"));
  require(
      __func__,
      hasStrictBatchEnvelope(innovusDriver) &&
          llvm::StringRef(innovusDriver)
              .contains("global init_top_cell init_verilog init_lef_file "
                        "init_mmmc_file\n") &&
          llvm::StringRef(innovusDriver)
              .contains("set init_mmmc_file "
                        "{drivers/innovus-mmmc.tcl}\n") &&
          llvm::StringRef(innovusMmmc).contains("set_analysis_view -setup") &&
          llvm::StringRef(innovusDriver).contains("routeDesign\n") &&
          !llvm::StringRef(innovusDriver).contains("optDesign -postRoute\n") &&
          llvm::StringRef(innovusDriver)
              .contains("defOut -routing {outputs/innovus-routed.def}"),
      "Innovus driver does not close routed state");
  const auto physical = take(
      __func__,
      parseInnovusPhysicalSnapshot("module top;\nendmodule\n",
                                   "VERSION 5.8 ;\nDESIGN top ;\nNETS 1 ;\n"
                                   "- clk + ROUTED Metal2 ( 0 0 ) "
                                   "( 100 0 ) ;\nEND NETS\nEND DESIGN\n",
                                   "create_clock -period 1 clk\n", "top",
                                   RepresentationPhysicalStage::Routed));
  require(__func__, physical.stage == RepresentationPhysicalStage::Routed,
          "Innovus stage changed");
  expectFailure(__func__, parseInnovusPhysicalSnapshot(
                              "module top;\nendmodule\n",
                              "VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n",
                              "create_clock -period 1 clk\n", "top",
                              RepresentationPhysicalStage::Routed));

  const std::string joulesDriver =
      take(__func__, renderJoulesPowerDriver("top", "inputs/netlist/top.v",
                                             "inputs/constraints/top.sdc",
                                             "inputs/activity/top.saif", "top",
                                             "/pdk/slow.lib"));
  require(__func__,
          hasStrictBatchEnvelope(joulesDriver) &&
              llvm::StringRef(joulesDriver)
                  .contains("read_stimulus -file {inputs/activity/top.saif} "
                            "-dut_instance {top} -format saif\n") &&
              llvm::StringRef(joulesDriver)
                  .contains("report_power -unit W -format %.17g -csv -out "
                            "{outputs/joules-power-result.csv}\n") &&
              !llvm::StringRef(joulesDriver).contains(".power_dynamic") &&
              !llvm::StringRef(joulesDriver).contains("read_saif"),
          "Joules driver does not project the exact SAIF binding");

  const auto timing =
      take(__func__, parseTempusTimingObservation(
                         "{\"schema\":\"loom.cadence.tempus_timing_result\","
                         "\"version\":\"1.0\",\"clock_period_seconds\":"
                         "\"1.25e-9\",\"limiting_clock_frequency_hz\":"
                         "\"8e8\"}\n"));
  require(__func__,
          timing.clockPeriodSeconds.coefficient() == 125 &&
              timing.clockPeriodSeconds.base10Exponent() == -11,
          "Tempus observation was not normalized");
  const auto power =
      take(__func__, parseJoulesPowerObservation(
                         "Instance: /top\n"
                         "Power Unit: W\n"
                         "PDB Frames: /stim#1/frame#0\n"
                         "Category,leakage,internal,switching,total,Row%\n"
                         "register,5.3692099999999995e-10,"
                         "1.5344022240000001e-06,4.9539598000000003e-08,"
                         "1.5844787429999999e-06,74.43%\n"
                         "Subtotal,8.2671900000000001e-10,"
                         "1.9205989290000001e-06,2.07318728e-07,"
                         "2.1287443760000001e-06,99.99%\n"
                         "Percentage,0.04%,90.22%,9.74%,100.00%,100.00%\n"));
  require(__func__,
          power.dynamicPowerWatts.coefficient() == 21279176570000001 &&
              power.dynamicPowerWatts.base10Exponent() == -22 &&
              power.leakagePowerWatts.coefficient() == 82671900000000001 &&
              power.leakagePowerWatts.base10Exponent() == -26,
          "Joules observation was not normalized");
  const auto rail =
      take(__func__, parseVoltusRailObservation(
                         "{\"schema\":\"loom.cadence.voltus_rail_result\","
                         "\"version\":\"1.0\",\"maximum_voltage_drop_volts\":"
                         "\"4.42943e-2\"}\n"));
  require(__func__,
          rail.maximumVoltageDropVolts.coefficient() == 442943 &&
              rail.maximumVoltageDropVolts.base10Exponent() == -7,
          "Voltus observation was not normalized");
  expectFailure(__func__,
                parseTempusTimingObservation("{\"schema\":\"other\"}\n"));
  expectFailure(__func__,
                parseJoulesPowerObservation(
                    "Instance: /top\n"
                    "Power Unit: mW\n"
                    "PDB Frames: /stim#1/frame#0\n"
                    "Category,leakage,internal,switching,total,Row%\n"
                    "Subtotal,5e-6,1e-3,1e-3,2.005e-3,100.00%\n"
                    "Percentage,0.25%,49.88%,49.88%,100.00%,100.00%\n"));
  expectFailure(__func__, parseJoulesPowerObservation(
                              "Instance: /top\n"
                              "Power Unit: W\n"
                              "PDB Frames: /stim#1/frame#0\n"
                              "Category,leakage,internal,switching,total,Row%\n"
                              "Subtotal,5e-6,1e-3,1e-3,2.005e-3,100.00%\n"
                              "Subtotal,5e-6,1e-3,1e-3,2.005e-3,100.00%\n"
                              "Percentage,0.25%,49.88%,49.88%,100.00%,"
                              "100.00%\n"));
  expectFailure(__func__,
                parseVoltusRailObservation("{\"schema\":\"other\"}\n"));

  auto routed = physicalRepresentation(root / "physical-blobs");
  require(__func__, !validateCadenceRepresentation(tempus, routed),
          "Tempus rejected exact routed physical state");
}

void lifecycleIsStrict(const std::filesystem::path &root) {
  std::filesystem::create_directories(root);
  const auto tool = root / "fixture-tool.sh";
  writeFile(tool, R"sh(#!/usr/bin/env bash
set -u
if [[ "${1-}" == "--version" ]]; then
  printf 'fixture-tool 1.0\n'
  exit 0
fi
mkdir -p scratch
: > scratch/tool-entered
args="$*"
if [[ "$args" == *"innovus.tcl"* ]]; then
  printf 'module top;\nendmodule\n' > outputs/innovus-routed.v
  printf 'VERSION 5.8 ;\nDESIGN top ;\nNETS 1 ;\n- clk + ROUTED Metal2 ( 0 0 ) ( 100 0 ) ;\nEND NETS\nEND DESIGN\n' > outputs/innovus-routed.def
  printf 'create_clock -period 1 clk\n' > outputs/innovus-routed.sdc
elif [[ "$args" == *"tempus.tcl"* ]]; then
  printf '{"schema":"loom.cadence.tempus_timing_result","version":"1.0","clock_period_seconds":"1e-9","limiting_clock_frequency_hz":"1e9"}\n' > outputs/tempus-timing-result.json
elif [[ "$args" == *"joules.tcl"* ]]; then
  printf 'Instance: /top\nPower Unit: W\nPDB Frames: /stim#1/frame#0\nCategory,leakage,internal,switching,total,Row%%\nSubtotal,5e-6,1.75e-3,2.5e-4,2.005e-3,100.00%%\nPercentage,0.25%%,87.28%%,12.47%%,100.00%%,100.00%%\n' > outputs/joules-power-result.csv
elif [[ "$args" == *"voltus-rail.tcl"* ]]; then
  grep -F 'set_rail_analysis_mode -method static' drivers/voltus-rail.tcl >/dev/null
  grep -F 'report_power_rail_results -plot ivdd' drivers/voltus-rail.tcl >/dev/null
  mkdir -p work
  printf '# domain voltage-drop report\n1.25e-2 instance0 power_main ground_main\n4.42943e-2 instance1 power_main ground_main\n3.7e-2 instance2 power_main ground_main\n' > work/voltus-ivdd.rpt
  tclsh drivers/voltus-rail-publish.tcl
elif [[ "$args" == *"-64bit"* ]]; then
  printf '{"schema":"loom.cadence.xcelium_functional_result","version":"1.0","status":"passed","completed_transactions":3}\n' > outputs/xcelium-functional-result.json
elif [[ "$args" == *"fail"* ]]; then
  exit 9
fi
)sh",
            true);

  const auto fixture = [&](llvm::StringRef name) {
    const auto path = root / "fixtures" / name.str();
    writeFile(path, "authored fixture\n");
    return path;
  };
  const auto technologyLef = fixture("technology.lef");
  const auto cellLef = fixture("cells.lef");
  const auto liberty = fixture("timing.lib");
  const auto qrc = fixture("technology.tch");
  PlatformFixture platform = makePlatform(root / "artifacts");
  const auto rtl = rtlRepresentation();
  const auto gate = gateRepresentation();
  const auto physical = physicalRepresentation(root / "physical");

  auto finalize = [&](llvm::StringRef name,
                      const ExternalToolInvocationBundleSpec &spec) {
    const auto bundle = root / name.str();
    auto prepared = take(
        __func__, finalizeExternalToolInvocationBundle(bundle.string(), spec));
    require(__func__, !std::filesystem::exists(bundle / "scratch/tool-entered"),
            "preparation executed the tool");
    return prepared;
  };
  auto execute = [&](const PreparedExternalToolInvocation &prepared) {
    require(__func__,
            take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
            "fixture execution failed");
  };

  CadenceBundleInputs xceliumInputs = bundleInputs(
      rtl, xceliumFunctionalDescriptor(), evaluationClosure(),
      frozen(tool, "xrun", {}),
      semanticFiles({{"inputs/rtl/top.sv", "module top; endmodule\n"},
                     {"inputs/workload/testbench.sv",
                      "module loom_testbench; initial $finish; endmodule\n"}}));
  const std::vector<std::string> xceliumSources{"inputs/rtl/top.sv",
                                                "inputs/workload/testbench.sv"};
  const auto xceliumSpec =
      take(__func__, makeXceliumFunctionalBundleSpec(
                         xceliumInputs, "loom_testbench", xceliumSources));
  const auto xceliumA = finalize("xcelium-a", xceliumSpec);
  const auto xceliumB = finalize("xcelium-b", xceliumSpec);
  require(__func__, xceliumA.manifestDigest == xceliumB.manifestDigest,
          "equivalent preparation changed the manifest");
  expectIncomplete(__func__,
                   importXceliumFunctionalResult(xceliumA, xceliumInputs));
  execute(xceliumA);
  require(__func__,
          take(__func__, importXceliumFunctionalResult(xceliumA, xceliumInputs))
                  .status == XceliumFunctionalStatus::Passed,
          "Xcelium lifecycle lost the result");

  const auto stale = finalize("xcelium-stale", xceliumSpec);
  writeFile(root / "xcelium-stale/outputs/xcelium-functional-result.json",
            "{\"schema\":\"loom.cadence.xcelium_functional_result\","
            "\"version\":\"1.0\",\"status\":\"passed\","
            "\"completed_transactions\":99}\n");
  expectIncomplete(__func__,
                   importXceliumFunctionalResult(stale, xceliumInputs));

  const auto partial = finalize("xcelium-partial", xceliumSpec);
  execute(partial);
  writeFile(root / "xcelium-partial/outputs/completion.json", "{\n");
  expectAdapterFailure(__func__,
                       importXceliumFunctionalResult(partial, xceliumInputs),
                       CadenceAdapterFailureKind::IntegrityFailure);

  const auto tampered = finalize("xcelium-tampered", xceliumSpec);
  execute(tampered);
  writeFile(root / "xcelium-tampered/outputs/xcelium-functional-result.json",
            "tampered\n");
  expectAdapterFailure(__func__,
                       importXceliumFunctionalResult(tampered, xceliumInputs),
                       CadenceAdapterFailureKind::IntegrityFailure);

  const auto expectationMismatch =
      finalize("xcelium-expectation-mismatch", xceliumSpec);
  execute(expectationMismatch);
  CadenceBundleInputs mismatchedInputs = xceliumInputs;
  mismatchedInputs.semanticContract.resultImporterIdentity =
      formatBlobDigestHex(digest("different importer"));
  expectAdapterFailure(
      __func__,
      importXceliumFunctionalResult(expectationMismatch, mismatchedInputs),
      CadenceAdapterFailureKind::IntegrityFailure);

  auto failedSpec = xceliumSpec;
  failedSpec.commands = {{tool.string(), "fail"}};
  const auto failed = finalize("xcelium-failed", failedSpec);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(failed)) == 9,
          "fixture tool exit was not preserved");
  expectAdapterFailure(__func__,
                       importXceliumFunctionalResult(failed, xceliumInputs),
                       CadenceAdapterFailureKind::ToolExecutionFailed);

  auto missingOutputSpec = xceliumSpec;
  missingOutputSpec.commands = {{tool.string(), "missing"}};
  const auto missingOutput = finalize("xcelium-missing", missingOutputSpec);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(missingOutput)) !=
              0,
          "missing declared output did not produce the wrapper status");
  expectAdapterFailure(
      __func__, importXceliumFunctionalResult(missingOutput, xceliumInputs),
      CadenceAdapterFailureKind::MissingDeclaredOutput);

  execute(xceliumB);
  writeFile(root / "xcelium-b/outputs/undeclared.rpt", "undeclared\n");
  expectAdapterFailure(__func__,
                       importXceliumFunctionalResult(xceliumB, xceliumInputs),
                       CadenceAdapterFailureKind::IntegrityFailure);

  CadenceBundleInputs innovusInputs = bundleInputs(
      gate, innovusDescriptor(), generatorClosure(),
      frozen(tool, "innovus",
             {{"cell_lef", cellLef},
              {"qrc_technology_file", qrc},
              {"technology_lef", technologyLef},
              {"timing_liberty", liberty}}),
      semanticFiles(
          {{"inputs/netlist/top.v", "module top; endmodule\n"},
           {"inputs/constraints/top.sdc", "create_clock -period 1 clk\n"},
           {"inputs/physical/floorplan.def",
            "VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n"}}),
      &platform);
  const auto innovusSpec =
      take(__func__,
           makeInnovusBundleSpec(innovusInputs, "top", "inputs/netlist/top.v",
                                 "inputs/constraints/top.sdc",
                                 "inputs/physical/floorplan.def"));
  require(__func__,
          innovusSpec.commands == std::vector<std::vector<std::string>>(
                                      {{tool.string(), "-no_gui", "-batch",
                                        "-files", "drivers/innovus.tcl"}}),
          "Innovus invocation does not use the installed batch interface");
  const auto innovus = finalize("innovus", innovusSpec);
  execute(innovus);
  require(__func__,
          take(__func__,
               importInnovusPhysicalSnapshot(innovus, innovusInputs, "top"))
                  .stage == RepresentationPhysicalStage::Routed,
          "Innovus lifecycle lost routed state");

  CadenceBundleInputs tempusInputs = bundleInputs(
      physical, tempusTimingDescriptor(), evaluationClosure(),
      frozen(tool, "tempus", {{"timing_liberty", liberty}}),
      semanticFiles(
          {{"inputs/netlist/top.v", "module top; endmodule\n"},
           {"inputs/constraints/top.sdc", "create_clock -period 1 clk\n"},
           {"inputs/physical/top.def",
            "VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n"}}),
      &platform);
  const auto tempusSpec =
      take(__func__, makeTempusTimingBundleSpec(tempusInputs, "top",
                                                "inputs/netlist/top.v",
                                                "inputs/constraints/top.sdc",
                                                "inputs/physical/top.def"));
  const auto tempus = finalize("tempus", tempusSpec);
  execute(tempus);
  require(__func__,
          take(__func__, importTempusTimingObservation(tempus, tempusInputs))
                  .limitingClockFrequencyHz ==
              take(__func__, evaluation::DecimalValue::get(1, 9)),
          "Tempus lifecycle lost the normalized observation");

  CadenceBundleInputs joulesInputs = bundleInputs(
      physical, joulesPowerDescriptor(), evaluationClosure(),
      frozen(tool, "joules", {{"timing_liberty", liberty}}),
      semanticFiles(
          {{"inputs/netlist/top.v", "module top; endmodule\n"},
           {"inputs/constraints/top.sdc", "create_clock -period 1 clk\n"},
           {"inputs/activity/top.saif", "(SAIFILE (SAIFVERSION \"2.0\"))\n"}}),
      &platform);
  const auto joulesSpec = take(
      __func__,
      makeJoulesPowerBundleSpec(joulesInputs, "top", "inputs/netlist/top.v",
                                "inputs/constraints/top.sdc",
                                "inputs/activity/top.saif", "tb/dut"));
  const auto joules = finalize("joules", joulesSpec);
  execute(joules);
  require(__func__,
          take(__func__, importJoulesPowerObservation(joules, joulesInputs))
                  .dynamicPowerWatts ==
              take(__func__, evaluation::DecimalValue::get(2, -3)),
          "Joules lifecycle lost the normalized observation");

  const ResolvedExternalFileTree voltusPgv =
      fileTree(root / "voltus-pgv", "power_grid_library");
  CadenceBundleInputs voltusInputs = bundleInputs(
      physical, voltusRailDescriptor(), evaluationClosure(),
      frozen(tool, "voltus", {}),
      semanticFiles(
          {{"inputs/netlist/top.v", "module top; endmodule\n"},
           {"inputs/constraints/top.sdc", "create_clock -period 1 clk\n"},
           {"inputs/physical/top.def",
            "VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n"}}),
      &platform);
  voltusInputs.frozen.externalFileTrees.push_back(voltusPgv);
  const VoltusRailInvocationConfiguration voltusConfiguration{
      "top",
      {"inputs/netlist/top.v"},
      {"inputs/constraints/top.sdc"},
      "inputs/physical/top.def",
      {"VDD", "VSS"},
      railConfiguration(platform, voltusPgv)};
  const auto voltusSpec = take(
      __func__, makeVoltusRailBundleSpec(voltusInputs, voltusConfiguration));
  require(__func__,
          voltusSpec.externalFiles.empty() &&
              voltusSpec.externalFileTrees.size() == 1 &&
              voltusSpec.externalFileTrees.front().providerInputSlot ==
                  "power_grid_library" &&
              voltusSpec.externalFileTrees.front().members ==
                  voltusInputs.frozen.externalFileTrees.front().members,
          "Cadence bundle did not retain the exact PGV tree");
  const auto voltusDriver =
      llvm::find_if(voltusSpec.files, [](const MaterializedBundleFile &file) {
        return file.relativePath == "drivers/voltus-rail.tcl";
      });
  require(
      __func__,
      voltusDriver != voltusSpec.files.end() &&
          llvm::StringRef(voltusDriver->contents)
              .contains("read_lib -pgv [list {") &&
          llvm::StringRef(voltusDriver->contents)
              .contains("technology.cl} {") &&
          llvm::StringRef(voltusDriver->contents)
              .contains("cells/stdcells.cl}") &&
          llvm::StringRef(voltusDriver->contents)
              .contains("set_default_switching_activity ") &&
          llvm::StringRef(voltusDriver->contents)
              .contains("-global_activity [expr {double(1) / 10}]") &&
          llvm::StringRef(voltusDriver->contents)
              .contains("-duty [expr {double(1) / 2}]") &&
          llvm::StringRef(voltusDriver->contents).contains("-period {2e-9s}"),
      "Voltus driver lost an exact projected rail input");
  const auto voltus = finalize("voltus", voltusSpec);
  execute(voltus);
  require(__func__,
          take(__func__, importVoltusRailObservation(voltus, voltusInputs))
                  .maximumVoltageDropVolts ==
              take(__func__, evaluation::DecimalValue::get(442943, -7)),
          "Cadence tree-bound import lost the normalized rail observation");

  CadenceBundleInputs changedPgv = voltusInputs;
  changedPgv.frozen.externalFileTrees.front().members.front().fingerprint =
      fingerprint("changed standard cells\n");
  expectAdapterFailure(__func__,
                       importVoltusRailObservation(voltus, changedPgv),
                       CadenceAdapterFailureKind::IntegrityFailure);

  CadenceBundleInputs missingPgv = voltusInputs;
  missingPgv.frozen.externalFileTrees.clear();
  expectAdapterFailure(
      __func__, makeVoltusRailBundleSpec(missingPgv, voltusConfiguration),
      CadenceAdapterFailureKind::MissingProviderInput);

  CadenceBundleInputs scalarPgv = missingPgv;
  scalarPgv.frozen.externalFiles.push_back(
      {"power_grid_library", "power_grid_library", liberty.string(),
       fingerprint(readFile(liberty))});
  expectAdapterFailure(__func__,
                       makeVoltusRailBundleSpec(scalarPgv, voltusConfiguration),
                       CadenceAdapterFailureKind::MissingProviderInput);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one bundle-root argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  descriptorsAndParsersAreExact(root);
  lifecycleIsStrict(root);
  return EXIT_SUCCESS;
}
