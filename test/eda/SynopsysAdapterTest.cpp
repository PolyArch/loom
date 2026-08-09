#include "EDA/Adapters/Synopsys/DesignCompiler.h"
#include "EDA/Adapters/Synopsys/FusionCompiler.h"
#include "EDA/Adapters/Synopsys/PrimePower.h"
#include "EDA/Adapters/Synopsys/PrimeTime.h"
#include "EDA/Adapters/Synopsys/Vcs.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"
#include "ExternalTool/InvocationBundle.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace loom::eda::synopsys;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectFailure(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid Synopsys adapter value");
  llvm::consumeError(value.takeError());
}

template <typename T>
void expectAdapterFailure(llvm::StringRef test, llvm::Expected<T> value,
                          SynopsysAdapterFailureKind expected) {
  if (value)
    fail(test, "expected a typed Synopsys adapter failure");
  bool matched = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const SynopsysAdapterError &error) {
        matched = error.kind() == expected;
        if (!matched)
          llvm::errs() << test << ": expected failure kind "
                       << static_cast<unsigned>(expected) << ", got "
                       << static_cast<unsigned>(error.kind()) << " ("
                       << error.detail() << ")\n";
      },
      [&](const llvm::ErrorInfoBase &) {});
  require(test, matched, "adapter failure kind changed");
}

void expectAdapterFailure(llvm::StringRef test, llvm::Error error,
                          SynopsysAdapterFailureKind expected) {
  if (!error)
    fail(test, "expected a typed Synopsys adapter failure");
  bool matched = false;
  llvm::handleAllErrors(
      std::move(error),
      [&](const SynopsysAdapterError &value) {
        matched = value.kind() == expected;
      },
      [&](const llvm::ErrorInfoBase &) {});
  require(test, matched, "adapter failure kind changed");
}

bool hasStrictBatchEnvelope(llvm::StringRef driver) {
  return driver.starts_with("proc loom_main {} {\n") &&
         driver.contains("if {[get_message_info -error_count] != 0} ") &&
         driver.contains("if {[catch {loom_main} loom_error]} {\n") &&
         driver.ends_with("exit 0\n");
}

template <typename T>
void expectIncomplete(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invocation without completion");
  bool matched = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const loom::external_tool::IncompleteExternalToolInvocationError &) {
        matched = true;
      },
      [&](const llvm::ErrorInfoBase &) {});
  require(test, matched, "incomplete invocation lost its shared typed error");
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents,
               bool executable = false) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail("writeFile", "could not write fixture file");
  stream.close();
  if (executable) {
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);
  }
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream bytes;
  bytes << stream.rdbuf();
  if (!stream)
    fail("readFile", "could not read fixture file");
  return bytes.str();
}

loom::ArtifactRootReference reference(llvm::StringRef schema, char digit) {
  return {schema.str(),
          {1, 0},
          take("reference",
               loom::parseArtifactIdentityHex(std::string(64, digit)))};
}

loom::BlobDigest digest(llvm::StringRef contents) {
  return loom::computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(contents.data()),
      contents.size()));
}

loom::ExternalFileFingerprint fingerprint(llvm::StringRef contents) {
  const loom::BlobDigest value = digest(contents);
  return take("fingerprint",
              loom::ExternalFileFingerprint::fromBytes(value.bytes()));
}

loom::hardware::ImplementationRepresentationRoot
rtlRepresentation(llvm::StringRef test) {
  using namespace loom::hardware;
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::Rtl, std::nullopt, format,
                        {RepresentationObjectKind::Module, "top"},
                        {{PayloadRole::RtlSource, "rtl/top.sv", digest("rtl")},
                         {PayloadRole::GenerationConstraint,
                          "constraints/top.sdc", digest("sdc")}}));
}

loom::hardware::ImplementationRepresentationRoot
gateRepresentation(llvm::StringRef test) {
  using namespace loom::hardware;
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::StructuralVerilogGateNetlist));
  return take(test,
              createImplementationRepresentationRoot(
                  RepresentationRootVariant::GateNetlist, std::nullopt, format,
                  {RepresentationObjectKind::Module, "top"},
                  {{PayloadRole::Netlist, "netlist/top.v", digest("gate")},
                   {PayloadRole::GenerationConstraint, "constraints/top.sdc",
                    digest("sdc")}}));
}

struct PlatformFixture final {
  loom::ArtifactStore store;
  loom::platform::FinalizedImplementationPlatform platform;
  loom::EncodedArtifactLocalReference corner;

  PlatformFixture(loom::ArtifactStore store,
                  loom::platform::FinalizedImplementationPlatform platform,
                  loom::EncodedArtifactLocalReference corner)
      : store(std::move(store)), platform(std::move(platform)),
        corner(std::move(corner)) {}
};

PlatformFixture makePlatform(const std::filesystem::path &root) {
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  auto platform =
      take("makePlatform",
           loom::platform::finalizeImplementationPlatform(
               {loom::platform::AsicTarget{"fixture-tech", "r1"}, {"typical"}},
               store));
  const loom::platform::TechnologyCornerRef corner{
      platform.reference().artifact, loom::platform::TechnologyCornerId(0)};
  return PlatformFixture(std::move(store), std::move(platform),
                         loom::platform::encodeTechnologyCornerRef(corner));
}

loom::external_tool::SemanticInvocationClosure generatorClosure() {
  loom::external_tool::CandidateGeneratorInvocationClosure closure;
  closure.typedInputBindings = {1, 2, 3};
  closure.resolvedBinding = {4, 5, 6};
  closure.bindingIdentity = digest("binding").bytes();
  return closure;
}

loom::external_tool::SemanticInvocationClosure evaluationClosure() {
  return reference("loom.evaluation_request", 'a');
}

loom::external_tool::ExternalToolSemanticContract
semanticContract(const SynopsysInvocationDescriptor &descriptor,
                 loom::external_tool::SemanticInvocationClosure closure) {
  return {descriptor.implementationSemanticIdentity.str(), std::move(closure),
          loom::formatBlobDigestHex(
              digest(descriptor.implementationSemanticIdentity))};
}

SynopsysFrozenInvocation
frozen(const std::filesystem::path &tool, llvm::StringRef toolKey,
       llvm::ArrayRef<std::pair<std::string, std::filesystem::path>> external) {
  SynopsysFrozenInvocation result;
  result.tool = {toolKey.str(),
                 loom::external_tool::ToolBindingSource::Explicit,
                 std::filesystem::canonical(tool).string(),
                 "fixture-tool 1.0",
                 {},
                 {},
                 std::nullopt,
                 std::nullopt};
  result.toolVersionProbe = {{"--version"}, "fixture-tool 1.0"};
  for (const auto &[slot, path] : external) {
    const std::string contents = readFile(path);
    result.externalFiles.push_back({slot, slot,
                                    std::filesystem::canonical(path).string(),
                                    fingerprint(contents)});
  }
  return result;
}

std::vector<loom::external_tool::MaterializedBundleFile>
semanticFiles(llvm::ArrayRef<std::pair<std::string, std::string>> files) {
  std::vector<loom::external_tool::MaterializedBundleFile> materialized;
  const loom::ArtifactRootReference source =
      reference("loom.hardware_implementation", 'b');
  for (const auto &[path, contents] : files)
    materialized.push_back({path, contents, source, false});
  return materialized;
}

SynopsysBundleInputs bundleInputs(
    const loom::hardware::ImplementationRepresentationRoot &implementation,
    const SynopsysInvocationDescriptor &descriptor,
    loom::external_tool::SemanticInvocationClosure closure,
    SynopsysFrozenInvocation binding,
    std::vector<loom::external_tool::MaterializedBundleFile> files,
    const PlatformFixture *platform = nullptr) {
  SynopsysBundleInputs inputs;
  inputs.semanticContract = semanticContract(descriptor, std::move(closure));
  inputs.implementation = &implementation;
  if (platform) {
    inputs.implementationPlatform = platform->platform.reference();
    inputs.platform = &platform->platform;
    inputs.technologyCorner = platform->corner;
  }
  inputs.frozen = std::move(binding);
  inputs.semanticInputs = std::move(files);
  return inputs;
}

void descriptorsOwnExactContracts() {
  const SynopsysInvocationDescriptor &vcs = vcsFunctionalDescriptor();
  require(__func__, vcs.operation == SynopsysOperation::FunctionalEvaluation,
          "VCS operation changed");
  require(__func__,
          vcs.acceptedStates.size() == 1 &&
              vcs.acceptedStates[0].variant ==
                  loom::hardware::RepresentationRootVariant::Rtl,
          "VCS accepted state is not exact");
  require(__func__,
          !vcs.requiresAsicPlatform && !vcs.requiresTechnologyCorner &&
              vcs.requiredProviderInputs.empty() &&
              vcs.declaredOutputs.size() == 1,
          "VCS acquired timing or platform authority");

  const SynopsysInvocationDescriptor &dc = designCompilerDescriptor();
  require(__func__,
          dc.operation == SynopsysOperation::LogicSynthesis &&
              dc.acceptedStates.size() == 1 &&
              dc.acceptedStates.front().variant ==
                  loom::hardware::RepresentationRootVariant::Rtl &&
              dc.requiresAsicPlatform && dc.requiresTechnologyCorner &&
              dc.requiresGenerationConstraint,
          "Design Compiler input contract changed");
  require(__func__,
          dc.requiredProviderInputs.size() == 1 &&
              dc.requiredProviderInputs.front() == "target_library" &&
              dc.declaredOutputs.size() == 1 &&
              dc.declaredOutputs.front() ==
                  "outputs/design-compiler-gate-netlist.v",
          "Design Compiler output or provider input changed");

  const SynopsysInvocationDescriptor &fc = fusionCompilerDescriptor();
  require(__func__,
          fc.operation == SynopsysOperation::PhysicalImplementation &&
              fc.acceptedStates.size() == 1 &&
              fc.acceptedStates.front().variant ==
                  loom::hardware::RepresentationRootVariant::GateNetlist &&
              fc.requiresAsicPlatform && fc.requiresTechnologyCorner &&
              fc.requiresGenerationConstraint,
          "Fusion Compiler input contract changed");
  require(
      __func__,
      fc.requiredProviderInputs.size() == 4 &&
          llvm::is_contained(fc.requiredProviderInputs, "reference_library") &&
          llvm::is_contained(fc.requiredProviderInputs,
                             "early_parasitic_tech") &&
          llvm::is_contained(fc.requiredProviderInputs,
                             "late_parasitic_tech") &&
          llvm::is_contained(fc.requiredProviderInputs,
                             "parasitic_layer_map") &&
          fc.declaredOutputs.size() == 3,
      "Fusion Compiler closure changed");

  const SynopsysInvocationDescriptor &timing = primeTimeDescriptor();
  require(__func__,
          timing.operation == SynopsysOperation::TimingEvaluation &&
              timing.acceptedStates.size() == 1 &&
              timing.acceptedStates.front().variant ==
                  loom::hardware::RepresentationRootVariant::GateNetlist &&
              timing.requiresAsicPlatform && timing.requiresTechnologyCorner &&
              timing.requiredProviderInputs.size() == 1 &&
              timing.requiredProviderInputs.front() == "timing_library",
          "PrimeTime input contract changed");

  const SynopsysInvocationDescriptor &power = primePowerDescriptor();
  require(__func__,
          power.operation == SynopsysOperation::PowerEvaluation &&
              power.acceptedStates.size() == 1 &&
              power.acceptedStates.front().variant ==
                  loom::hardware::RepresentationRootVariant::GateNetlist &&
              power.requiresAsicPlatform && power.requiresTechnologyCorner &&
              power.requiredProviderInputs.size() == 1 &&
              power.requiredProviderInputs.front() == "power_library",
          "PrimePower input contract changed");

  requireSuccess(__func__, validateSynopsysRepresentation(
                               dc, rtlRepresentation(__func__)));
  expectAdapterFailure(
      __func__,
      validateSynopsysRepresentation(dc, gateRepresentation(__func__)),
      SynopsysAdapterFailureKind::UnsupportedImplementation);
  expectAdapterFailure(
      __func__,
      validateSynopsysRepresentation(vcs, gateRepresentation(__func__)),
      SynopsysAdapterFailureKind::UnsupportedImplementation);
  requireSuccess(__func__, validateSynopsysRepresentation(
                               timing, gateRepresentation(__func__)));

  const auto rtlFormat =
      take(__func__,
           loom::hardware::RepresentationFormatDescriptorRef::get(
               loom::hardware::RepresentationFormatKind::SystemVerilogRtl));
  const auto unownedPhysical = take(
      __func__,
      loom::hardware::createImplementationRepresentationRoot(
          loom::hardware::RepresentationRootVariant::AsicPhysical,
          loom::hardware::RepresentationPhysicalStage::Routed, rtlFormat,
          {loom::hardware::RepresentationObjectKind::PhysicalObject, "top"},
          {{loom::hardware::PayloadRole::RtlSource, "rtl/top.sv",
            digest("rtl")},
           {loom::hardware::PayloadRole::GenerationConstraint,
            "constraints/top.sdc", digest("sdc")}}));
  expectAdapterFailure(__func__,
                       validateSynopsysRepresentation(timing, unownedPhysical),
                       SynopsysAdapterFailureKind::UnsupportedImplementation);

  const auto physicalFormat = take(
      __func__, loom::hardware::RepresentationFormatDescriptorRef::get(
                    loom::hardware::RepresentationFormatKind::IndexedPhysical));
  require(__func__,
          loom::hardware::admitsRepresentationRoot(
              loom::hardware::getRepresentationFormatDescriptor(physicalFormat),
              loom::hardware::RepresentationRootVariant::AsicPhysical,
              loom::hardware::RepresentationPhysicalStage::Routed),
          "HImpl no longer admits indexed routed ASIC physical state");
}

void vcsDriverAndResultAreExact() {
  const std::vector<std::string> command = take(
      __func__, renderVcsFunctionalCommand("/tools/vcs", "loom_testbench",
                                           {"inputs/implementation/rtl/top.sv",
                                            "inputs/workload/testbench.sv"}));
  const std::vector<std::string> expected{"/tools/vcs",
                                          "-full64",
                                          "-sverilog",
                                          "-top",
                                          "loom_testbench",
                                          "-Mdir=vcs-csrc",
                                          "-o",
                                          "vcs-simv",
                                          "-R",
                                          "inputs/implementation/rtl/top.sv",
                                          "inputs/workload/testbench.sv"};
  require(__func__, command == expected, "VCS command tokens changed");
  require(__func__,
          take(__func__, renderVcsFunctionalCommand(
                             "/tools/vcs", "loom_testbench",
                             {"inputs/implementation/rtl/top.sv",
                              "inputs/workload/testbench.sv"})) == command,
          "VCS command is not deterministic");
  expectFailure(__func__, renderVcsFunctionalCommand("/tools/vcs", "bad top",
                                                     {"inputs/top.sv"}));
  expectAdapterFailure(
      __func__, renderVcsFunctionalCommand("vcs", "top", {"inputs/top.sv"}),
      SynopsysAdapterFailureKind::ExecutableUnavailable);
  expectFailure(__func__, renderVcsFunctionalCommand("/tools/vcs", "top",
                                                     {"../outside.sv"}));

  const VcsFunctionalResult passed =
      take(__func__, parseVcsFunctionalResult(
                         "{\"schema\":\"loom.synopsys.vcs_functional_result\","
                         "\"version\":\"1.0\",\"status\":\"passed\","
                         "\"completed_transactions\":3}\n"));
  require(__func__,
          passed.status == VcsFunctionalStatus::Passed &&
              passed.completedTransactions == 3 &&
              !passed.firstFailingTransaction,
          "VCS passed result changed");

  const VcsFunctionalResult failed =
      take(__func__, parseVcsFunctionalResult(
                         "{\"schema\":\"loom.synopsys.vcs_functional_result\","
                         "\"version\":\"1.0\",\"status\":\"failed\","
                         "\"completed_transactions\":2,"
                         "\"first_failing_transaction\":1}\n"));
  require(__func__,
          failed.status == VcsFunctionalStatus::Failed &&
              failed.firstFailingTransaction == 1,
          "VCS adverse result changed");
  expectFailure(__func__, parseVcsFunctionalResult(
                              "{\"schema\":\"other\",\"version\":\"1.0\","
                              "\"status\":\"passed\","
                              "\"completed_transactions\":3}\n"));
}

void implementationDriversAreDeterministic() {
  const std::string dc = take(
      __func__,
      renderDesignCompilerDriver("top",
                                 {"inputs/implementation/rtl/package.sv",
                                  "inputs/implementation/rtl/top.sv"},
                                 {"inputs/implementation/constraints/top.sdc"},
                                 "/libraries/saed.db"));
  require(__func__,
          dc ==
              take(__func__, renderDesignCompilerDriver(
                                 "top",
                                 {"inputs/implementation/rtl/package.sv",
                                  "inputs/implementation/rtl/top.sv"},
                                 {"inputs/implementation/constraints/top.sdc"},
                                 "/libraries/saed.db")),
          "Design Compiler driver is not deterministic");
  require(
      __func__,
      hasStrictBatchEnvelope(dc) &&
          llvm::StringRef(dc).contains("compile_ultra\n") &&
          llvm::StringRef(dc).contains(
              "set_app_var target_library $loom_target_library\n") &&
          llvm::StringRef(dc).contains("set_app_var link_library [concat {*} "
                                       "$loom_target_library]\n") &&
          !llvm::StringRef(dc).contains("$target_library") &&
          llvm::StringRef(dc).contains(
              "outputs/design-compiler-gate-netlist.v") &&
          !llvm::StringRef(dc).contains("report_"),
      "Design Compiler driver acquired report semantics");

  const std::string fc =
      take(__func__, renderFusionCompilerDriver(
                         "top", "inputs/implementation/netlist/top.v",
                         "inputs/implementation/constraints/top.sdc",
                         "inputs/physical/floorplan.def", "/libraries/saed.ndm",
                         "/libraries/early.tluplus", "/libraries/late.tluplus",
                         "/libraries/layers.map"));
  require(
      __func__,
      hasStrictBatchEnvelope(fc) &&
          llvm::StringRef(fc).contains("create_lib {fusion.dlib}") &&
          llvm::StringRef(fc).contains(
              "read_parasitic_tech -tlup {/libraries/early.tluplus} "
              "-layermap {/libraries/layers.map} -name {loom_early}\n") &&
          llvm::StringRef(fc).contains(
              "set_parasitic_parameters -early_spec {loom_early} "
              "-late_spec {loom_late}\n") &&
          llvm::StringRef(fc).contains(
              "compile_fusion -from initial_map -to final_opto\n"
              "clock_opt\n"
              "route_auto\n"
              "route_opt\n") &&
          llvm::StringRef(fc).contains("outputs/fusion-compiler-routed.def") &&
          llvm::StringRef(fc).contains("outputs/fusion-compiler-routed.v") &&
          llvm::StringRef(fc).contains("outputs/fusion-compiler-routed.sdc"),
      "Fusion Compiler driver does not close one routed snapshot");

  expectFailure(__func__, renderDesignCompilerDriver(
                              "top", {}, {"inputs/top.sdc"}, "/library.db"));
  expectFailure(__func__,
                renderFusionCompilerDriver("top", "inputs/top.v",
                                           "inputs/top.sdc", "../floorplan.def",
                                           "/library.ndm", "/early.tluplus",
                                           "/late.tluplus", "/layers.map"));
}

void normalizedObservationsAreStrict() {
  const std::string timingDriver =
      take(__func__,
           renderPrimeTimeDriver("top", "inputs/implementation/netlist/top.v",
                                 "inputs/implementation/constraints/top.sdc",
                                 "/libraries/saed.db"));
  require(
      __func__,
      timingDriver ==
              take(__func__, renderPrimeTimeDriver(
                                 "top", "inputs/implementation/netlist/top.v",
                                 "inputs/implementation/constraints/top.sdc",
                                 "/libraries/saed.db")) &&
          hasStrictBatchEnvelope(timingDriver) &&
          llvm::StringRef(timingDriver).contains("update_timing\n") &&
          llvm::StringRef(timingDriver)
              .contains(
                  "get_timing_paths -delay_type max -max_paths 1 -nworst 1") &&
          llvm::StringRef(timingDriver)
              .contains("get_attribute $loom_limiting_path slack") &&
          !llvm::StringRef(timingDriver).contains("report_timing"),
      "PrimeTime driver is nondeterministic, ignores the limiting path, "
      "or parses a report");

  const PrimeTimeObservation timing = take(
      __func__, parsePrimeTimeObservation(
                    "{\"schema\":\"loom.synopsys.primetime_timing_result\","
                    "\"version\":\"1.0\","
                    "\"clock_period_seconds\":\"1.25e-9\","
                    "\"limiting_clock_frequency_hz\":\"8e8\"}\n"));
  require(__func__,
          timing.clockPeriodSeconds.coefficient() == 125 &&
              timing.clockPeriodSeconds.base10Exponent() == -11 &&
              timing.limitingClockFrequencyHz.coefficient() == 8 &&
              timing.limitingClockFrequencyHz.base10Exponent() == 8,
          "PrimeTime values were not normalized");

  const std::string powerDriver =
      take(__func__,
           renderPrimePowerDriver("top", "inputs/implementation/netlist/top.v",
                                  "inputs/implementation/constraints/top.sdc",
                                  "inputs/activity/top.saif", "tb/dut",
                                  "/libraries/saed.db"));
  require(__func__,
          powerDriver == take(__func__,
                              renderPrimePowerDriver(
                                  "top", "inputs/implementation/netlist/top.v",
                                  "inputs/implementation/constraints/top.sdc",
                                  "inputs/activity/top.saif", "tb/dut",
                                  "/libraries/saed.db")) &&
              hasStrictBatchEnvelope(powerDriver) &&
              llvm::StringRef(powerDriver)
                  .contains(
                      "if {![read_saif {inputs/activity/top.saif} "
                      "-strip_path {tb/dut}]} {error {SAIF annotated no design "
                      "objects}}\n") &&
              llvm::StringRef(powerDriver).contains("update_power\n") &&
              !llvm::StringRef(powerDriver).contains("report_power"),
          "PrimePower driver is nondeterministic or parses a report");

  const PrimePowerObservation power = take(
      __func__, parsePrimePowerObservation(
                    "{\"schema\":\"loom.synopsys.primepower_power_result\","
                    "\"version\":\"1.0\","
                    "\"dynamic_power_watts\":\"2.50e-3\","
                    "\"leakage_power_watts\":\"7.5e-6\"}\n"));
  require(__func__,
          power.dynamicPowerWatts.coefficient() == 25 &&
              power.dynamicPowerWatts.base10Exponent() == -4 &&
              power.leakagePowerWatts.coefficient() == 75 &&
              power.leakagePowerWatts.base10Exponent() == -7,
          "PrimePower values were not normalized");

  expectFailure(__func__,
                parsePrimeTimeObservation(
                    "{\"schema\":\"loom.synopsys.primetime_timing_result\","
                    "\"version\":\"1.0\","
                    "\"clock_period_seconds\":\"0\","
                    "\"limiting_clock_frequency_hz\":\"8e8\"}\n"));
  expectFailure(__func__,
                parsePrimePowerObservation(
                    "{\"schema\":\"loom.synopsys.primepower_power_result\","
                    "\"version\":\"1.0\","
                    "\"dynamic_power_watts\":\"nan\","
                    "\"leakage_power_watts\":\"7.5e-6\"}\n"));
}

void implementationOutputsRemainExact() {
  const std::string gate = "module top(input wire a, output wire y);\n"
                           "  BUFX1 u0 (.A(a), .Y(y));\n"
                           "endmodule\n";
  const DesignCompilerGateNetlist imported =
      take(__func__, parseDesignCompilerGateNetlist(gate, "top"));
  require(__func__, imported.verilog == gate,
          "Design Compiler importer rewrote the netlist");
  expectFailure(__func__, parseDesignCompilerGateNetlist(gate, "other"));
  expectFailure(__func__,
                parseDesignCompilerGateNetlist(
                    std::string("module top;\0endmodule\n", 22), "top"));

  const std::string def = "VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n";
  const std::string constraints = "create_clock -period 1 clk\n";
  const FusionCompilerPhysicalSnapshot physical =
      take(__func__, parseFusionCompilerPhysicalSnapshot(
                         gate, def, constraints, "top",
                         loom::hardware::RepresentationPhysicalStage::Routed));
  require(
      __func__,
      physical.netlistVerilog == gate && physical.designExchangeFormat == def &&
          physical.generationConstraints == constraints &&
          physical.stage == loom::hardware::RepresentationPhysicalStage::Routed,
      "Fusion Compiler importer did not preserve one exact snapshot");
  expectFailure(__func__,
                parseFusionCompilerPhysicalSnapshot(
                    gate, "", constraints, "top",
                    loom::hardware::RepresentationPhysicalStage::Routed));
  expectFailure(__func__,
                parseFusionCompilerPhysicalSnapshot(
                    gate, def, constraints, "top",
                    loom::hardware::RepresentationPhysicalStage::Placed));
}

void invocationLifecycleIsSingleAndStrict(const std::filesystem::path &root) {
  std::filesystem::create_directories(root);
  const std::filesystem::path tool = root / "fixture-tool.sh";
  writeFile(tool,
            R"sh(#!/usr/bin/env bash
set -u
if [[ "${1-}" == "--version" ]]; then
  printf 'fixture-tool 1.0\n'
  exit 0
fi
mkdir -p scratch
: > scratch/tool-entered
loom_args="$*"
if [[ "$loom_args" == *"design-compiler.tcl"* ]]; then
  printf 'module top(input wire a, output wire y);\n  assign y = a;\nendmodule\n' > outputs/design-compiler-gate-netlist.v
elif [[ "$loom_args" == *"fusion-compiler.tcl"* ]]; then
  printf 'module top(input wire a, output wire y);\n  assign y = a;\nendmodule\n' > outputs/fusion-compiler-routed.v
  printf 'VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n' > outputs/fusion-compiler-routed.def
  printf 'create_clock -period 1 clk\n' > outputs/fusion-compiler-routed.sdc
elif [[ "$loom_args" == *"primetime.tcl"* ]]; then
  printf '{"schema":"loom.synopsys.primetime_timing_result","version":"1.0","clock_period_seconds":"1e-9","limiting_clock_frequency_hz":"1e9"}\n' > outputs/primetime-timing-result.json
elif [[ "$loom_args" == *"primepower.tcl"* ]]; then
  printf '{"schema":"loom.synopsys.primepower_power_result","version":"1.0","dynamic_power_watts":"2e-3","leakage_power_watts":"5e-6"}\n' > outputs/primepower-power-result.json
elif [[ "$loom_args" == *"-R"* ]]; then
  printf '{"schema":"loom.synopsys.vcs_functional_result","version":"1.0","status":"passed","completed_transactions":3}\n' > outputs/vcs-functional-result.json
elif [[ "$loom_args" == *"fail"* ]]; then
  exit 9
fi
)sh",
            true);

  const std::filesystem::path targetLibrary = root / "fixtures" / "target.db";
  const std::filesystem::path referenceLibrary =
      root / "fixtures" / "reference.ndm";
  const std::filesystem::path earlyParasiticTech =
      root / "fixtures" / "early.tluplus";
  const std::filesystem::path lateParasiticTech =
      root / "fixtures" / "late.tluplus";
  const std::filesystem::path parasiticLayerMap =
      root / "fixtures" / "layers.map";
  const std::filesystem::path timingLibrary = root / "fixtures" / "timing.db";
  const std::filesystem::path powerLibrary = root / "fixtures" / "power.db";
  writeFile(targetLibrary, "authored target library fixture\n");
  writeFile(referenceLibrary / "pcat", "authored reference catalog fixture\n");
  writeFile(referenceLibrary / "parts" / "p0",
            "authored reference payload fixture\n");
  writeFile(earlyParasiticTech, "authored early parasitic fixture\n");
  writeFile(lateParasiticTech, "authored late parasitic fixture\n");
  writeFile(parasiticLayerMap, "authored parasitic map fixture\n");
  writeFile(timingLibrary, "authored timing library fixture\n");
  writeFile(powerLibrary, "authored power library fixture\n");

  PlatformFixture platform = makePlatform(root / "artifact-store");
  const auto rtl = rtlRepresentation(__func__);
  const auto gate = gateRepresentation(__func__);

  auto finalize =
      [&](const std::filesystem::path &bundle,
          const loom::external_tool::ExternalToolInvocationBundleSpec
              &specification) {
        auto prepared = take(
            __func__, loom::external_tool::finalizeExternalToolInvocationBundle(
                          bundle.string(), specification));
        require(__func__,
                !std::filesystem::exists(bundle / "scratch" / "tool-entered"),
                "bundle preparation executed the tool");
        return prepared;
      };
  auto execute =
      [&](const loom::external_tool::PreparedExternalToolInvocation &prepared) {
        require(__func__,
                take(__func__,
                     loom::external_tool::executeExternalToolInvocationBundle(
                         prepared)) == 0,
                "fixture bundle execution failed");
        require(
            __func__,
            std::filesystem::exists(std::filesystem::path(prepared.bundleRoot) /
                                    "scratch" / "tool-entered"),
            "caller-owned execution did not enter the tool");
      };

  SynopsysBundleInputs vcsInputs = bundleInputs(
      rtl, vcsFunctionalDescriptor(), evaluationClosure(),
      frozen(tool, "vcs", {}),
      semanticFiles(
          {{"inputs/implementation/rtl/top.sv",
            "module top(input logic a, output logic y); assign y = a; "
            "endmodule\n"},
           {"inputs/workload/testbench.sv",
            "module loom_testbench; initial begin $finish; end "
            "endmodule\n"}}));
  const std::vector<std::string> vcsSources{"inputs/implementation/rtl/top.sv",
                                            "inputs/workload/testbench.sv"};
  SynopsysBundleInputs missingVcsInput = vcsInputs;
  missingVcsInput.semanticInputs.pop_back();
  expectAdapterFailure(__func__,
                       makeVcsFunctionalBundleSpec(
                           missingVcsInput, "loom_testbench", vcsSources),
                       SynopsysAdapterFailureKind::MissingSemanticInput);
  SynopsysBundleInputs wrongVcsClosure = vcsInputs;
  wrongVcsClosure.semanticContract.semanticClosure = generatorClosure();
  expectAdapterFailure(__func__,
                       makeVcsFunctionalBundleSpec(
                           wrongVcsClosure, "loom_testbench", vcsSources),
                       SynopsysAdapterFailureKind::DescriptorMismatch);
  const auto vcsSpec =
      take(__func__, makeVcsFunctionalBundleSpec(vcsInputs, "loom_testbench",
                                                 vcsSources));
  require(__func__, vcsSpec.semanticContract == vcsInputs.semanticContract,
          "bundle specification re-encoded the semantic contract");
  const auto vcsA = finalize(root / "vcs-a", vcsSpec);
  const auto vcsB = finalize(root / "vcs-b", vcsSpec);
  const auto vcsC = finalize(root / "vcs-c", vcsSpec);
  require(__func__,
          readFile(root / "vcs-a" / "tool-invocation.json") ==
                  readFile(root / "vcs-b" / "tool-invocation.json") &&
              readFile(root / "vcs-a" / "tool-invocation.json") ==
                  readFile(root / "vcs-c" / "tool-invocation.json") &&
              readFile(root / "vcs-a" / "run.sh") ==
                  readFile(root / "vcs-b" / "run.sh"),
          "three preparations diverged");
  std::filesystem::create_directory(root / "vcs-a" / "outputs" /
                                    "vcs-functional-result.json");
  expectIncomplete(__func__, importVcsFunctionalResult(vcsA, vcsInputs));
  std::filesystem::remove(root / "vcs-a" / "outputs" /
                          "vcs-functional-result.json");
  execute(vcsA);
  const VcsFunctionalResult vcsResult =
      take(__func__, importVcsFunctionalResult(vcsA, vcsInputs));
  require(__func__, vcsResult.status == VcsFunctionalStatus::Passed,
          "VCS lifecycle lost the functional result");

  execute(vcsB);
  auto importedVcs =
      take(__func__, importSynopsysInvocation(vcsFunctionalDescriptor(), vcsB,
                                              vcsInputs));
  expectAdapterFailure(__func__,
                       readSynopsysDeclaredOutput(vcsFunctionalDescriptor(),
                                                  importedVcs,
                                                  "outputs/not-declared.json"),
                       SynopsysAdapterFailureKind::DescriptorMismatch);

  SynopsysBundleInputs staleInputs = vcsInputs;
  staleInputs.semanticInputs.front().contents += "// stale substitution\n";
  expectAdapterFailure(__func__, importVcsFunctionalResult(vcsB, staleInputs),
                       SynopsysAdapterFailureKind::IntegrityFailure);
  writeFile(root / "vcs-b" / "outputs" / "vcs-functional-result.json",
            "tampered\n");
  expectAdapterFailure(__func__, importVcsFunctionalResult(vcsB, vcsInputs),
                       SynopsysAdapterFailureKind::IntegrityFailure);

  execute(vcsC);
  SynopsysBundleInputs wrongProviderContract = vcsInputs;
  wrongProviderContract.semanticContract.providerIdentity += ".tampered";
  expectAdapterFailure(__func__,
                       importVcsFunctionalResult(vcsC, wrongProviderContract),
                       SynopsysAdapterFailureKind::DescriptorMismatch);
  SynopsysBundleInputs wrongImporterContract = vcsInputs;
  wrongImporterContract.semanticContract.resultImporterIdentity =
      loom::formatBlobDigestHex(digest("tampered importer"));
  expectAdapterFailure(__func__,
                       importVcsFunctionalResult(vcsC, wrongImporterContract),
                       SynopsysAdapterFailureKind::IntegrityFailure);
  SynopsysBundleInputs wrongClosureContract = vcsInputs;
  wrongClosureContract.semanticContract.semanticClosure =
      reference("loom.evaluation_request", 'c');
  expectAdapterFailure(__func__,
                       importVcsFunctionalResult(vcsC, wrongClosureContract),
                       SynopsysAdapterFailureKind::IntegrityFailure);

  SynopsysBundleInputs incompatibleVersionInputs = vcsInputs;
  incompatibleVersionInputs.frozen.tool.version = "fixture-tool 2.0";
  const auto incompatibleVersionSpec =
      take(__func__, makeVcsFunctionalBundleSpec(incompatibleVersionInputs,
                                                 "loom_testbench", vcsSources));
  const auto incompatibleVersion =
      finalize(root / "incompatible-version", incompatibleVersionSpec);
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         incompatibleVersion)) != 0,
      "incompatible tool version unexpectedly executed");
  expectAdapterFailure(
      __func__,
      importVcsFunctionalResult(incompatibleVersion, incompatibleVersionInputs),
      SynopsysAdapterFailureKind::IncompatibleVersion);

  require(__func__, ::unsetenv("LOOM_SYNOPSYS_TEST_ENV_DO_NOT_SET") == 0,
          "could not clear authored missing environment fixture");
  SynopsysBundleInputs missingEnvironmentInputs = vcsInputs;
  missingEnvironmentInputs.frozen.inheritEnvironment = {
      "LOOM_SYNOPSYS_TEST_ENV_DO_NOT_SET"};
  const auto missingEnvironmentSpec =
      take(__func__, makeVcsFunctionalBundleSpec(missingEnvironmentInputs,
                                                 "loom_testbench", vcsSources));
  const auto missingEnvironment =
      finalize(root / "missing-environment", missingEnvironmentSpec);
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         missingEnvironment)) != 0,
      "missing inherited environment unexpectedly executed");
  expectAdapterFailure(
      __func__,
      importVcsFunctionalResult(missingEnvironment, missingEnvironmentInputs),
      SynopsysAdapterFailureKind::ActivationUnavailable);

  const auto missingSpec =
      take(__func__, makeSynopsysInvocationBundleSpec(
                         vcsFunctionalDescriptor(), vcsInputs,
                         {{vcsInputs.frozen.tool.executable, "missing"}}, {}));
  const auto missing = finalize(root / "missing-output", missingSpec);
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         missing)) != 0,
      "missing output execution unexpectedly succeeded");
  std::filesystem::create_directory(root / "missing-output" / "outputs" /
                                    "vcs-functional-result.json");
  expectAdapterFailure(
      __func__,
      importSynopsysInvocation(vcsFunctionalDescriptor(), missing, vcsInputs),
      SynopsysAdapterFailureKind::MissingDeclaredOutput);

  const auto failedSpec =
      take(__func__, makeSynopsysInvocationBundleSpec(
                         vcsFunctionalDescriptor(), vcsInputs,
                         {{vcsInputs.frozen.tool.executable, "fail"}}, {}));
  const auto failed = finalize(root / "tool-failure", failedSpec);
  require(
      __func__,
      take(__func__, loom::external_tool::executeExternalToolInvocationBundle(
                         failed)) == 9,
      "tool failure status changed");
  std::filesystem::create_directory(root / "tool-failure" / "outputs" /
                                    "vcs-functional-result.json");
  expectAdapterFailure(
      __func__,
      importSynopsysInvocation(vcsFunctionalDescriptor(), failed, vcsInputs),
      SynopsysAdapterFailureKind::ToolExecutionFailed);

  SynopsysBundleInputs dcInputs = bundleInputs(
      rtl, designCompilerDescriptor(), generatorClosure(),
      frozen(tool, "dc_shell", {{"target_library", targetLibrary}}),
      semanticFiles(
          {{"inputs/implementation/rtl/top.sv",
            "module top(input logic a, output logic y); assign y = a; "
            "endmodule\n"},
           {"inputs/implementation/constraints/top.sdc",
            "create_clock -period 1 clk\n"}}),
      &platform);
  SynopsysBundleInputs wrongDcClosure = dcInputs;
  wrongDcClosure.semanticContract.semanticClosure = evaluationClosure();
  expectAdapterFailure(__func__,
                       makeDesignCompilerBundleSpec(
                           wrongDcClosure, "top",
                           {"inputs/implementation/rtl/top.sv"},
                           {"inputs/implementation/constraints/top.sdc"}),
                       SynopsysAdapterFailureKind::DescriptorMismatch);
  SynopsysBundleInputs missingDcTarget = dcInputs;
  missingDcTarget.implementationPlatform.reset();
  expectAdapterFailure(__func__,
                       makeDesignCompilerBundleSpec(
                           missingDcTarget, "top",
                           {"inputs/implementation/rtl/top.sv"},
                           {"inputs/implementation/constraints/top.sdc"}),
                       SynopsysAdapterFailureKind::MissingTarget);
  SynopsysBundleInputs missingDcCorner = dcInputs;
  missingDcCorner.technologyCorner.reset();
  expectAdapterFailure(__func__,
                       makeDesignCompilerBundleSpec(
                           missingDcCorner, "top",
                           {"inputs/implementation/rtl/top.sv"},
                           {"inputs/implementation/constraints/top.sdc"}),
                       SynopsysAdapterFailureKind::MissingCorner);
  SynopsysBundleInputs wrongDcTool = dcInputs;
  wrongDcTool.frozen.tool.toolKey = "fc_shell";
  expectAdapterFailure(__func__,
                       makeDesignCompilerBundleSpec(
                           wrongDcTool, "top",
                           {"inputs/implementation/rtl/top.sv"},
                           {"inputs/implementation/constraints/top.sdc"}),
                       SynopsysAdapterFailureKind::DescriptorMismatch);
  SynopsysBundleInputs missingDcProviderInput = dcInputs;
  missingDcProviderInput.frozen.externalFiles.front().providerInputSlot =
      "other_library";
  expectAdapterFailure(__func__,
                       makeDesignCompilerBundleSpec(
                           missingDcProviderInput, "top",
                           {"inputs/implementation/rtl/top.sv"},
                           {"inputs/implementation/constraints/top.sdc"}),
                       SynopsysAdapterFailureKind::MissingProviderInput);
  const auto dcSpec =
      take(__func__, makeDesignCompilerBundleSpec(
                         dcInputs, "top", {"inputs/implementation/rtl/top.sv"},
                         {"inputs/implementation/constraints/top.sdc"}));
  const auto dcPrepared = finalize(root / "dc", dcSpec);
  execute(dcPrepared);
  require(__func__,
          take(__func__,
               importDesignCompilerGateNetlist(dcPrepared, dcInputs, "top"))
                  .verilog.find("module top") != std::string::npos,
          "Design Compiler lifecycle lost the exact netlist");

  SynopsysFrozenInvocation fcFrozen =
      frozen(tool, "fc_shell",
             {{"early_parasitic_tech", earlyParasiticTech},
              {"late_parasitic_tech", lateParasiticTech},
              {"parasitic_layer_map", parasiticLayerMap}});
  fcFrozen.externalFileTrees.push_back(
      {"reference_library",
       "reference_library",
       std::filesystem::canonical(referenceLibrary).string(),
       {{"parts/p0", fingerprint("authored reference payload fixture\n")},
        {"pcat", fingerprint("authored reference catalog fixture\n")}}});
  SynopsysBundleInputs fcInputs = bundleInputs(
      gate, fusionCompilerDescriptor(), generatorClosure(), std::move(fcFrozen),
      semanticFiles({{"inputs/implementation/netlist/top.v",
                      "module top(input wire a, output wire y); assign y = a; "
                      "endmodule\n"},
                     {"inputs/implementation/constraints/top.sdc",
                      "create_clock -period 1 clk\n"},
                     {"inputs/physical/floorplan.def",
                      "VERSION 5.8 ;\nDESIGN top ;\nEND DESIGN\n"}}),
      &platform);
  const auto fcSpec =
      take(__func__, makeFusionCompilerBundleSpec(
                         fcInputs, "top", "inputs/implementation/netlist/top.v",
                         "inputs/implementation/constraints/top.sdc",
                         "inputs/physical/floorplan.def"));
  require(__func__,
          fcSpec.externalFiles.size() == 3 &&
              fcSpec.externalFileTrees.size() == 1 &&
              fcSpec.externalFileTrees.front().providerInputSlot ==
                  "reference_library",
          "Fusion Compiler did not preserve the exact reference-library tree");
  SynopsysBundleInputs missingFcTree = fcInputs;
  missingFcTree.frozen.externalFileTrees.clear();
  expectAdapterFailure(
      __func__,
      makeFusionCompilerBundleSpec(missingFcTree, "top",
                                   "inputs/implementation/netlist/top.v",
                                   "inputs/implementation/constraints/top.sdc",
                                   "inputs/physical/floorplan.def"),
      SynopsysAdapterFailureKind::MissingProviderInput);
  SynopsysBundleInputs wrongFcInputKind = missingFcTree;
  wrongFcInputKind.frozen.externalFiles.push_back(
      {"reference_library", "reference_library",
       std::filesystem::canonical(referenceLibrary / "pcat").string(),
       fingerprint("authored reference catalog fixture\n")});
  expectAdapterFailure(
      __func__,
      makeFusionCompilerBundleSpec(wrongFcInputKind, "top",
                                   "inputs/implementation/netlist/top.v",
                                   "inputs/implementation/constraints/top.sdc",
                                   "inputs/physical/floorplan.def"),
      SynopsysAdapterFailureKind::MissingProviderInput);
  const auto fcPrepared = finalize(root / "fc", fcSpec);
  execute(fcPrepared);
  SynopsysBundleInputs wrongFcTreeClosure = fcInputs;
  wrongFcTreeClosure.frozen.externalFileTrees.front()
      .members.front()
      .fingerprint = fingerprint("wrong");
  expectAdapterFailure(__func__,
                       importFusionCompilerPhysicalSnapshot(
                           fcPrepared, wrongFcTreeClosure, "top"),
                       SynopsysAdapterFailureKind::IntegrityFailure);
  require(__func__,
          take(__func__, importFusionCompilerPhysicalSnapshot(fcPrepared,
                                                              fcInputs, "top"))
                  .stage == loom::hardware::RepresentationPhysicalStage::Routed,
          "Fusion Compiler lifecycle lost routed state");

  SynopsysBundleInputs timingInputs = bundleInputs(
      gate, primeTimeDescriptor(), evaluationClosure(),
      frozen(tool, "pt_shell", {{"timing_library", timingLibrary}}),
      semanticFiles({{"inputs/implementation/netlist/top.v",
                      "module top(input wire a, output wire y); assign y = a; "
                      "endmodule\n"},
                     {"inputs/implementation/constraints/top.sdc",
                      "create_clock -period 1 clk\n"}}),
      &platform);
  const auto timingSpec = take(
      __func__, makePrimeTimeBundleSpec(
                    timingInputs, "top", "inputs/implementation/netlist/top.v",
                    "inputs/implementation/constraints/top.sdc"));
  const auto timingPrepared = finalize(root / "primetime", timingSpec);
  execute(timingPrepared);
  require(
      __func__,
      take(__func__, importPrimeTimeObservation(timingPrepared, timingInputs))
              .clockPeriodSeconds ==
          take(__func__, loom::evaluation::DecimalValue::get(1, -9)),
      "PrimeTime lifecycle changed the normalized observation");

  SynopsysBundleInputs powerInputs = bundleInputs(
      gate, primePowerDescriptor(), evaluationClosure(),
      frozen(tool, "pt_shell", {{"power_library", powerLibrary}}),
      semanticFiles(
          {{"inputs/implementation/netlist/top.v",
            "module top(input wire a, output wire y); assign y = a; "
            "endmodule\n"},
           {"inputs/implementation/constraints/top.sdc",
            "create_clock -period 1 clk\n"},
           {"inputs/activity/top.saif", "(SAIFILE (SAIFVERSION \"2.0\"))\n"}}),
      &platform);
  const auto powerSpec =
      take(__func__,
           makePrimePowerBundleSpec(powerInputs, "top",
                                    "inputs/implementation/netlist/top.v",
                                    "inputs/implementation/constraints/top.sdc",
                                    "inputs/activity/top.saif", "tb/dut"));
  const auto powerPrepared = finalize(root / "primepower", powerSpec);
  execute(powerPrepared);
  require(
      __func__,
      take(__func__, importPrimePowerObservation(powerPrepared, powerInputs))
              .dynamicPowerWatts ==
          take(__func__, loom::evaluation::DecimalValue::get(2, -3)),
      "PrimePower lifecycle changed the normalized observation");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || llvm::StringRef(argv[1]) != "--bundle-root")
    fail("main", "expected --bundle-root <path>");
  descriptorsOwnExactContracts();
  vcsDriverAndResultAreExact();
  implementationDriversAreDeterministic();
  normalizedObservationsAreStrict();
  implementationOutputsRemainExact();
  invocationLifecycleIsSingleAndStrict(argv[2]);
  return EXIT_SUCCESS;
}
