#include "EDA/Adapters/IntelAltera/Quartus.h"

#include "ConfigurationABI2TestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CandidateGenerator.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::dse;
using namespace loom::eda::intel_altera;
using namespace loom::external_tool;
using namespace loom::hardware;
using namespace loom::platform;

namespace {

constexpr llvm::StringLiteral kDevice = "AGIA040R39A1E1VC";
constexpr llvm::StringLiteral kPhysicalRoot =
    "device_41474941303430523339413145315643";
constexpr llvm::StringLiteral kProviderBuild =
    "altera.quartus-prime-pro:26.1.0-build-110";
constexpr llvm::StringLiteral kToolVersion =
    "Version 26.1.0 Build 110 03/26/2026 SC Pro Edition";
constexpr llvm::StringLiteral kTop = "loom_quartus_top";
constexpr CandidateGeneratorOutputSlotRef kPhysicalOutput(0);
constexpr CandidateGeneratorOutputSlotRef kImageOutput(1);

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
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
void expectFailureContains(llvm::StringRef test, llvm::Expected<T> value,
                           llvm::StringRef expected) {
  if (value)
    fail(test, "accepted input expected to fail with '" + expected.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectUnsupported(llvm::StringRef test, llvm::Expected<T> value,
                       QuartusPrimeUnsupportedReason expected) {
  if (value)
    fail(test, "accepted unsupported input");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const QuartusPrimeUnsupportedError &error) {
        matched = error.reason() == expected;
      });
  require(test, matched, "failure was not the expected typed Unsupported");
  llvm::consumeError(std::move(remainder));
}

template <typename T>
void expectUnavailable(llvm::StringRef test, llvm::Expected<T> value,
                       QuartusPrimeUnavailableReason expected) {
  if (value)
    fail(test, "accepted unavailable provider binding");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const QuartusPrimeUnavailableError &error) {
        matched = error.reason() == expected;
      });
  require(test, matched, "failure was not the expected typed Unavailable");
  llvm::consumeError(std::move(remainder));
}

template <typename T>
void expectIncomplete(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invocation without a completion record");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(),
      [&](const IncompleteExternalToolInvocationError &) { matched = true; });
  require(test, matched, "incomplete invocation error type was lost");
  llvm::consumeError(std::move(remainder));
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return std::vector<std::uint8_t>(value.bytes_begin(), value.bytes_end());
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not read " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail(__func__, "could not write " + path.string());
}

void writeExecutable(const std::filesystem::path &path,
                     llvm::StringRef contents) {
  writeFile(path, contents);
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::group_read |
                                   std::filesystem::perms::group_exec,
                               std::filesystem::perm_options::replace);
}

struct SemanticFixture final {
  loom::fabric::FinalizedFabricRoot module;
  loom::fabric::FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef firstOwner;
};

SemanticFixture makeSemanticFixture(llvm::StringRef test,
                                    const ArtifactStore &artifacts) {
  adg::DesignBuilder design(artifacts);
  const std::vector<adg::PortType> noTypes;
  auto spatial =
      take(test, design.createSpatialCore("quartus-static", noTypes, noTypes));
  const std::vector<adg::SpatialValue> noOutputs;
  if (llvm::Error error = spatial.close(noOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "fixture did not produce one Fabric Module");
  loom::fabric::FinalizedFabricRoot module =
      std::move(finalized.roots().front());
  loom::fabric::FinalizedFabricRoot system =
      take(test, hardware::test::makeSpatialCoreSystem(module, artifacts, 1));
  ConfigurationABIDraft abiDraft =
      take(test, hardware::test::makeCompleteConfigurationABIDraft(system));
  FinalizedConfigurationABI abi =
      take(test, finalizeConfigurationABI(std::move(abiDraft), artifacts));
  const auto cores = system.view().accCoreOccurrences();
  require(test, cores.size() == 1,
          "fixture did not produce one accelerator-core occurrence");
  const auto firstOwner = take(
      test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                loom::fabric::FabricInventoryOwnerRef::of(
                    loom::fabric::SpatialCoreOccurrenceRef{cores.front()})));
  return {std::move(module), std::move(system), std::move(abi), firstOwner};
}

FinalizedImplementationPlatform makePlatform(llvm::StringRef test,
                                             const ArtifactStore &artifacts,
                                             FpgaVendor vendor,
                                             llvm::StringRef device) {
  return take(test, finalizeImplementationPlatform(
                        ImplementationPlatformDraft{
                            FpgaTarget{vendor, device.str()}, {"default"}},
                        artifacts));
}

ImplementationRepresentationRoot
makeRepresentation(llvm::StringRef test, const BlobStore &blobs,
                   RepresentationRootVariant variant, bool withConstraint) {
  std::vector<ImplementationPayload> payloads;
  RepresentationFormatKind formatKind;
  if (variant == RepresentationRootVariant::Rtl) {
    formatKind = RepresentationFormatKind::SystemVerilogRtl;
    const BlobDigest helper =
        take(test, blobs.put(bytes("module loom_quartus_helper; endmodule\n")));
    const BlobDigest top = take(
        test, blobs.put(bytes("module loom_quartus_top; "
                              "loom_quartus_helper u_helper(); endmodule\n")));
    payloads = {{PayloadRole::RtlSource, "rtl/helper.sv", helper},
                {PayloadRole::RtlSource, "rtl/top.sv", top}};
    if (withConstraint) {
      const BlobDigest constraint =
          take(test, blobs.put(bytes("# exact constraint payload\n")));
      payloads.push_back({PayloadRole::GenerationConstraint,
                          "constraints/top.sdc", constraint});
    }
  } else {
    formatKind = RepresentationFormatKind::StructuralVerilogGateNetlist;
    const BlobDigest netlist =
        take(test, blobs.put(bytes("module loom_quartus_top; endmodule\n")));
    payloads = {{PayloadRole::Netlist, "netlist/top.v", netlist}};
  }
  const RepresentationFormatDescriptorRef format =
      take(test, RepresentationFormatDescriptorRef::get(formatKind));
  return take(test, createImplementationRepresentationRoot(
                        variant, std::nullopt, format,
                        {RepresentationObjectKind::Module, kTop.str()},
                        std::move(payloads)));
}

FinalizedHardwareImplementation makeImplementation(
    llvm::StringRef test, const SemanticFixture &fixture,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    std::optional<ArtifactRootReference> platformReference,
    RepresentationRootVariant variant = RepresentationRootVariant::Rtl,
    bool withConstraint = true) {
  HardwareImplementationDraft draft{
      fixture.system.reference(),
      fixture.abi.reference(),
      {},
      makeRepresentation(test, blobs, variant, withConstraint),
      std::move(platformReference),
      {},
      {{{RepresentationObjectKind::Module, kTop.str()}, fixture.firstOwner}},
      {},
      {}};
  return take(
      test, finalizeHardwareImplementation(std::move(draft), artifacts, blobs));
}

std::string physicalMetadata(llvm::StringRef providerBuild,
                             llvm::StringRef device) {
  return "{\"schema\":\"loom.quartus_prime_fpga_physical_attempt\","
         "\"version\":\"1.0\",\"provider_build_identity\":\"" +
         providerBuild.str() + "\",\"device_ordering_code\":\"" + device.str() +
         "\",\"device_resource_key\":\"" + device.str() + "\",\"top\":\"" +
         kTop.str() + "\"}\n";
}

std::string imageMetadata(llvm::StringRef providerBuild,
                          llvm::StringRef device) {
  return "{\"schema\":\"loom.quartus_prime_fpga_image_attempt\","
         "\"version\":\"1.0\",\"provider_build_identity\":\"" +
         providerBuild.str() + "\",\"device_ordering_code\":\"" + device.str() +
         "\",\"device_resource_key\":\"" + device.str() + "\",\"top\":\"" +
         kTop.str() +
         "\",\"input_physical_output\":\"outputs/fpga-physical.qar\"}\n";
}

void writeFakeQuartus(const std::filesystem::path &tool) {
  std::string script =
      "#!/usr/bin/env bash\n"
      "set -euo pipefail\n"
      "if [[ \"${1-}\" == --version ]]; then\n"
      "  printf '%s\\n' 'Quartus Prime Shell' '" +
      kToolVersion.str() +
      "'\n"
      "  exit 0\n"
      "fi\n"
      "[[ \"${1-}\" == -t && \"${3-}\" != '' ]] || exit 64\n"
      "case \"$3\" in\n"
      "  synthesis) : ;;\n"
      "  fitter)\n"
      "    printf '%s' 'physical-database' >outputs/fpga-physical.qar\n"
      "    printf '%s' '" +
      physicalMetadata(kProviderBuild, kDevice) +
      "' >outputs/fpga-physical.json\n"
      "    ;;\n"
      "  sta) : ;;\n"
      "  assembler)\n"
      "    if [[ -z \"${LOOM_QUARTUS_TEST_OMIT_IMAGE-}\" ]]; then\n"
      "      printf '%s' 'device-image' >outputs/device.sof\n"
      "    fi\n"
      "    if [[ -n \"${LOOM_QUARTUS_TEST_BAD_METADATA-}\" ]]; then\n"
      "      printf '%s' '" +
      imageMetadata(kProviderBuild, "1SM21BHN1F53E1VG") +
      "' >outputs/fpga-image.json\n"
      "    else\n"
      "      printf '%s' '" +
      imageMetadata(kProviderBuild, kDevice) +
      "' >outputs/fpga-image.json\n"
      "    fi\n"
      "    if [[ -n \"${LOOM_QUARTUS_TEST_UNDECLARED_OUTPUT-}\" ]]; then\n"
      "      printf '%s' 'undeclared' >outputs/undeclared.bin\n"
      "    fi\n"
      "    ;;\n"
      "  *) exit 64 ;;\n"
      "esac\n";
  writeExecutable(tool, script);
}

LocalToolConfig localConfig(const std::filesystem::path &tool) {
  LocalToolConfig config;
  config.runtimePolicy = RuntimePolicy::Host;
  config.tools["quartus_sh"].binding.executable = tool.string();
  return config;
}

ExternalToolPreparationContext
preparationContext(const std::filesystem::path &bundle,
                   const std::filesystem::path &tool) {
  return ExternalToolPreparationContext{localConfig(tool), bundle.string()};
}

void createStoreDirectories(const std::filesystem::path &root) {
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
}

struct LaneFixture final {
  ArtifactStore artifacts;
  BlobStore blobs;
  SemanticFixture semantics;
  FinalizedImplementationPlatform platform;
  FinalizedHardwareImplementation implementation;
  std::vector<CandidateGeneratorInputBinding> inputs;
  ResolvedQuartusPrimeStaticFullDeviceConfigView config;
  ResolvedCandidateGeneratorBinding binding;

  LaneFixture(const std::filesystem::path &root,
              FpgaVendor vendor = FpgaVendor::IntelAltera,
              llvm::StringRef device = kDevice,
              llvm::StringRef providerBuild = kProviderBuild,
              llvm::StringRef toolVersion = kToolVersion,
              bool withConstraint = true)
      : artifacts((root / "artifacts").string()),
        blobs((root / "blobs").string()),
        semantics(makeSemanticFixture(__func__, artifacts)),
        platform(makePlatform(__func__, artifacts, vendor, device)),
        implementation(makeImplementation(
            __func__, semantics, artifacts, blobs, platform.reference(),
            RepresentationRootVariant::Rtl, withConstraint)),
        inputs(take(__func__,
                    bindQuartusPrimeStaticFullDeviceCandidateGeneratorInputs(
                        implementation.reference(), platform.reference()))),
        config(take(__func__,
                    projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                        providerBuild, toolVersion, device))),
        binding(
            take(__func__,
                 resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(
                     config))) {}
};

PreparedExternalToolInvocation
prepareBundle(llvm::StringRef test, const LaneFixture &fixture,
              const std::filesystem::path &bundle,
              const std::filesystem::path &tool) {
  const ExternalImplementationContractCatalog contracts;
  return take(test,
              prepareQuartusPrimeStaticFullDeviceInvocation(
                  fixture.inputs, fixture.binding, contracts, fixture.artifacts,
                  fixture.blobs, preparationContext(bundle, tool)));
}

void descriptorAndConfigAreExact() {
  auto config =
      take(__func__, projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                         kProviderBuild, kToolVersion, kDevice));
  auto adopted = take(
      __func__, adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
                    resolvedQuartusPrimeStaticFullDeviceConfigSchemaBytes(),
                    config.canonicalViewBytes(), config.digest()));
  require(__func__,
          adopted.stableProviderBuildIdentity() == kProviderBuild &&
              adopted.verifiedToolVersion() == kToolVersion &&
              adopted.deviceResourceKey() == kDevice,
          "resolved Quartus config did not round-trip exactly");
  std::vector<std::uint8_t> changed(config.canonicalViewBytes().begin(),
                                    config.canonicalViewBytes().end());
  changed.back() ^= 1;
  expectFailureContains(
      __func__,
      adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
          resolvedQuartusPrimeStaticFullDeviceConfigSchemaBytes(), changed,
          config.digest()),
      "digest");
  expectFailureContains(__func__,
                        projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                            "amd.vivado:2026.1", kToolVersion, kDevice),
                        "provider build identity");

  requireSuccess(__func__,
                 registerQuartusPrimeStaticFullDeviceCandidateGenerator());
  const CandidateGeneratorDescriptor &descriptor =
      quartusPrimeStaticFullDeviceCandidateGeneratorDescriptor();
  require(__func__,
          descriptor.kind ==
                  quartusPrimeStaticFullDeviceCandidateGeneratorKind &&
              descriptor.inputSlots.size() == 2 &&
              descriptor.outputSlots.size() == 2 &&
              descriptor.providerForm == ProviderForm::ExternalPrepareImport &&
              descriptor.implementationSemanticIdentity ==
                  "loom.eda.intel_altera.quartus_prime_static_full_device."
                  "generator.v2" &&
              descriptor.determinism ==
                  CandidateGeneratorDeterminism::IndependentReplicates,
          "Quartus descriptor shape is not exact");
  require(
      __func__,
      *descriptor.inputSlots[0].schema == hardwareImplementationSchema &&
          *descriptor.inputSlots[1].schema == implementationPlatformSchema &&
          *descriptor.outputSlots[0].schema == hardwareImplementationSchema &&
          *descriptor.outputSlots[1].schema == hardwareImplementationSchema &&
          descriptor.inputSlots[0].cardinality ==
              PlanValueCardinality::ExactlyOne &&
          descriptor.inputSlots[1].cardinality ==
              PlanValueCardinality::ExactlyOne &&
          descriptor.outputSlots[0].cardinality ==
              PlanValueCardinality::ExactlyOne &&
          descriptor.outputSlots[1].cardinality ==
              PlanValueCardinality::ExactlyOne,
      "Quartus descriptor slots are not exact singleton Artifact slots");
}

void deterministicPreparationUsesExactInputs(
    const std::filesystem::path &root) {
  const std::filesystem::path data = root / "deterministic-data";
  createStoreDirectories(data);
  LaneFixture fixture(data);
  const std::filesystem::path tool = root / "tools" / "quartus_sh";
  writeFakeQuartus(tool);
  const PreparedExternalToolInvocation first =
      prepareBundle(__func__, fixture, root / "deterministic-a", tool);
  const PreparedExternalToolInvocation second =
      prepareBundle(__func__, fixture, root / "deterministic-b", tool);
  const PreparedExternalToolInvocation third =
      prepareBundle(__func__, fixture, root / "deterministic-c", tool);

  for (llvm::StringRef path :
       {"tool-invocation.json", "run.sh", "drivers/quartus-static.tcl",
        "inputs/platform.json", "inputs/rtl/0000000000000000.sv",
        "inputs/rtl/0000000000000001.sv",
        "inputs/constraints/0000000000000000.sdc"}) {
    const std::string a = readFile(root / "deterministic-a" / path.str());
    require(__func__,
            a == readFile(root / "deterministic-b" / path.str()) &&
                a == readFile(root / "deterministic-c" / path.str()),
            "three preparations diverged at " + path.str());
  }

  const std::string manifest =
      readFile(root / "deterministic-a" / "tool-invocation.json");
  std::size_t previous = 0;
  for (llvm::StringRef action : {"synthesis", "fitter", "sta", "assembler"}) {
    const std::size_t position = manifest.find(("\"" + action + "\"").str());
    require(__func__, position != std::string::npos && position >= previous,
            "structured command order changed at " + action.str());
    previous = position;
  }
  for (llvm::StringRef output :
       {"outputs/fpga-physical.qar", "outputs/fpga-physical.json",
        "outputs/device.sof", "outputs/fpga-image.json"})
    require(__func__, manifest.find(output.str()) != std::string::npos,
            "manifest lost declared output " + output.str());

  const std::string driver =
      readFile(root / "deterministic-a" / "drivers/quartus-static.tcl");
  require(__func__,
          driver.find("set loom_device {" + kDevice.str() + "}") !=
                  std::string::npos &&
              driver.find(kProviderBuild.str()) != std::string::npos &&
              driver.find("TOP_LEVEL_ENTITY {" + kTop.str() + "}") !=
                  std::string::npos &&
              driver.find("get_part_info -family $loom_device") !=
                  std::string::npos &&
              driver.find("inputs/constraints/0000000000000000.sdc") !=
                  std::string::npos,
          "driver lost an exact semantic input");
  require(__func__,
          driver.find("project_archive quartus-work/loom_quartus.qar "
                      "-include_outputs -overwrite") != std::string::npos &&
              driver.find("file copy -force quartus-work/loom_quartus.qar "
                          "outputs/fpga-physical.qar") != std::string::npos &&
              driver.find("file copy -force "
                          "quartus-work/output/loom_quartus.sof "
                          "outputs/device.sof") != std::string::npos &&
              driver.find("-version_compatible_database") == std::string::npos,
          "driver does not preserve the exact physical-to-image route");
  const std::string platformBytes =
      readFile(root / "deterministic-a" / "inputs/platform.json");
  require(__func__, platformBytes.find(kDevice.str()) != std::string::npos,
          "platform input lost the exact device ordering code");
  for (llvm::StringRef forbidden : {"create_clock", "-name FAMILY",
                                    "VIRTUAL_PIN", "altera_mult", "fallback"})
    require(__func__, driver.find(forbidden.str()) == std::string::npos,
            "driver inferred forbidden content: " + forbidden.str());
  require(__func__,
          !std::filesystem::exists(root / "deterministic-a" / "outputs" /
                                   "completion.json"),
          "preparation executed the tool");
  (void)first;
  (void)second;
  (void)third;
}

void exactAdmissionIsTyped(const std::filesystem::path &root) {
  const std::filesystem::path data = root / "admission-data";
  createStoreDirectories(data);
  LaneFixture fixture(data);
  const std::filesystem::path tool = root / "tools" / "quartus_sh";
  writeFakeQuartus(tool);

  auto wrongDeviceConfig =
      take(__func__, projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                         kProviderBuild, kToolVersion, "1SM21BHN1F53E1VG"));
  auto wrongDeviceBinding = take(
      __func__, resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(
                    wrongDeviceConfig));
  expectUnsupported(__func__,
                    prepareCandidateGeneratorInvocation(
                        fixture.inputs, wrongDeviceBinding, fixture.artifacts,
                        fixture.blobs,
                        preparationContext(root / "wrong-device", tool)),
                    QuartusPrimeUnsupportedReason::DeviceResourceBinding);

  auto wrongVersionConfig =
      take(__func__, projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                         kProviderBuild, "Version 26.2.0 Build 1", kDevice));
  auto wrongVersionBinding = take(
      __func__, resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(
                    wrongVersionConfig));
  expectUnavailable(__func__,
                    prepareCandidateGeneratorInvocation(
                        fixture.inputs, wrongVersionBinding, fixture.artifacts,
                        fixture.blobs,
                        preparationContext(root / "wrong-version", tool)),
                    QuartusPrimeUnavailableReason::ProviderBuild);

  expectUnavailable(__func__,
                    prepareCandidateGeneratorInvocation(
                        fixture.inputs, fixture.binding, fixture.artifacts,
                        fixture.blobs,
                        preparationContext(root / "missing-tool",
                                           root / "missing" / "quartus_sh")),
                    QuartusPrimeUnavailableReason::ToolResolution);

  const std::filesystem::path amdData = root / "amd-data";
  createStoreDirectories(amdData);
  LaneFixture amd(amdData, FpgaVendor::AmdXilinx, "xcvu13p-flga2577-3-e");
  expectUnsupported(__func__,
                    prepareCandidateGeneratorInvocation(
                        amd.inputs, amd.binding, amd.artifacts, amd.blobs,
                        preparationContext(root / "wrong-vendor", tool)),
                    QuartusPrimeUnsupportedReason::TargetVendor);

  FinalizedHardwareImplementation gate =
      makeImplementation(__func__, fixture.semantics, fixture.artifacts,
                         fixture.blobs, fixture.platform.reference(),
                         RepresentationRootVariant::GateNetlist, false);
  auto gateInputs =
      take(__func__, bindQuartusPrimeStaticFullDeviceCandidateGeneratorInputs(
                         gate.reference(), fixture.platform.reference()));
  expectUnsupported(
      __func__,
      prepareCandidateGeneratorInvocation(
          gateInputs, fixture.binding, fixture.artifacts, fixture.blobs,
          preparationContext(root / "wrong-representation", tool)),
      QuartusPrimeUnsupportedReason::InputRepresentation);

  FinalizedImplementationPlatform otherPlatform = makePlatform(
      __func__, fixture.artifacts, FpgaVendor::IntelAltera, "1SM21BHN1F53E1VG");
  auto mismatchedInputs =
      take(__func__,
           bindQuartusPrimeStaticFullDeviceCandidateGeneratorInputs(
               fixture.implementation.reference(), otherPlatform.reference()));
  expectUnsupported(__func__,
                    prepareCandidateGeneratorInvocation(
                        mismatchedInputs, fixture.binding, fixture.artifacts,
                        fixture.blobs,
                        preparationContext(root / "mismatched-platform", tool)),
                    QuartusPrimeUnsupportedReason::PlatformBinding);
}

void requireIncompleteResult(llvm::StringRef test,
                             CandidateGeneratorProviderResult result,
                             CandidateGeneratorIncompleteReason reason) {
  const auto *incomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&result.outcome);
  require(test,
          incomplete && incomplete->reason == reason &&
              incomplete->retainedOutputBindings.size() == 2 &&
              incomplete->retainedOutputBindings[0].artifacts.empty() &&
              incomplete->retainedOutputBindings[1].artifacts.empty(),
          "failed invocation changed its typed incomplete result");
  require(test,
          result.workSummary.size() == 3 &&
              result.workSummary[0].planned == 1 &&
              result.workSummary[0].consumed == 1 &&
              result.workSummary[1].planned == 1 &&
              result.workSummary[1].consumed == 0 &&
              result.workSummary[2].planned == 1 &&
              result.workSummary[2].consumed == 0,
          "failed invocation changed its exact work boundary");
}

void requirePublishedResult(llvm::StringRef test, const LaneFixture &fixture,
                            CandidateGeneratorProviderResult result) {
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  require(test,
          completed && completed->outputBindings.size() == 2 &&
              completed->outputBindings[0].slot == kPhysicalOutput &&
              completed->outputBindings[0].artifacts.size() == 1 &&
              completed->outputBindings[1].slot == kImageOutput &&
              completed->outputBindings[1].artifacts.size() == 1 &&
              completed->lineageEdges.size() == 2,
          "successful invocation did not publish both exact outputs");

  FinalizedHardwareImplementation physical =
      take(test, importHardwareImplementation(
                     completed->outputBindings[0].artifacts.front(),
                     fixture.artifacts, fixture.blobs));
  FinalizedHardwareImplementation image =
      take(test, importHardwareImplementation(
                     completed->outputBindings[1].artifacts.front(),
                     fixture.artifacts, fixture.blobs));
  const HardwareImplementation &source =
      fixture.implementation.implementation();
  require(test,
          physical.implementation().fabric() == source.fabric() &&
              physical.implementation().configurationAbi() ==
                  source.configurationAbi() &&
              physical.implementation().implementationPlatform() ==
                  source.implementationPlatform() &&
              physical.implementation().representationRoot().variant ==
                  RepresentationRootVariant::FpgaPhysical &&
              physical.implementation().representationRoot().stage ==
                  RepresentationPhysicalStage::Routed,
          "FpgaPhysical publication lost its exact semantic closure");
  require(test,
          image.implementation().fabric() == source.fabric() &&
              image.implementation().configurationAbi() ==
                  source.configurationAbi() &&
              image.implementation().implementationPlatform() ==
                  source.implementationPlatform() &&
              image.implementation().representationRoot().variant ==
                  RepresentationRootVariant::FpgaImage &&
              !image.implementation().representationRoot().stage,
          "FpgaImage publication lost its exact semantic closure");
  for (const HardwareImplementation *implementation :
       {&physical.implementation(), &image.implementation()})
    require(
        test,
        implementation->activityPoints().size() == 1 &&
            implementation->activityPoints().front().representationLocator ==
                RepresentationLocator{RepresentationObjectKind::DeviceResource,
                                      kPhysicalRoot.str()},
        "FPGA publication did not project the activity locator");
  require(test,
          completed->lineageEdges[0].output == physical.reference() &&
              completed->lineageEdges[0].parents.empty() &&
              completed->lineageEdges[1].output == image.reference() &&
              completed->lineageEdges[1].parents.empty(),
          "mechanical output lineage is not exact");
  require(test,
          result.workSummary.size() == 3 &&
              llvm::all_of(result.workSummary,
                           [](const auto &summary) {
                             return summary.planned == 1 &&
                                    summary.consumed == 1;
                           }),
          "successful publication did not consume all three work units");
}

void strictImportUsesOnlyTheCompletedDeclaredSnapshot(
    const std::filesystem::path &root) {
  const std::filesystem::path data = root / "strict-data";
  createStoreDirectories(data);
  LaneFixture fixture(data);
  const std::filesystem::path tool = root / "tools" / "quartus_sh";
  writeFakeQuartus(tool);
  const auto import = [&](const PreparedExternalToolInvocation &prepared,
                          const ResolvedCandidateGeneratorBinding &binding) {
    const ExternalImplementationContractCatalog contracts;
    return importQuartusPrimeStaticFullDeviceInvocation(
        fixture.inputs, binding, prepared, contracts, fixture.artifacts,
        fixture.blobs);
  };

  PreparedExternalToolInvocation incomplete =
      prepareBundle(__func__, fixture, root / "incomplete", tool);
  expectIncomplete(__func__, import(incomplete, fixture.binding));

  PreparedExternalToolInvocation valid =
      prepareBundle(__func__, fixture, root / "valid", tool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(valid)) == 0,
          "synthetic Quartus bundle execution failed");
  requirePublishedResult(__func__, fixture,
                         take(__func__, import(valid, fixture.binding)));

  PreparedExternalToolInvocation tampered =
      prepareBundle(__func__, fixture, root / "tampered", tool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(tampered)) == 0,
          "tamper fixture execution failed");
  writeFile(root / "tampered" / "outputs" / "fpga-physical.qar", "changed");
  expectFailureContains(__func__, import(tampered, fixture.binding), "digest");

  PreparedExternalToolInvocation partial =
      prepareBundle(__func__, fixture, root / "partial", tool);
  ::setenv("LOOM_QUARTUS_TEST_OMIT_IMAGE", "1", 1);
  const int partialExit =
      take(__func__, executeExternalToolInvocationBundle(partial));
  ::unsetenv("LOOM_QUARTUS_TEST_OMIT_IMAGE");
  require(__func__, partialExit != 0,
          "partial synthetic output was marked successful");
  requireIncompleteResult(__func__,
                          take(__func__, import(partial, fixture.binding)),
                          CandidateGeneratorIncompleteReason::ExecutionFailed);

  PreparedExternalToolInvocation badMetadata =
      prepareBundle(__func__, fixture, root / "bad-metadata", tool);
  ::setenv("LOOM_QUARTUS_TEST_BAD_METADATA", "1", 1);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(badMetadata)) == 0,
          "bad-metadata fixture execution failed");
  ::unsetenv("LOOM_QUARTUS_TEST_BAD_METADATA");
  expectFailureContains(__func__, import(badMetadata, fixture.binding),
                        "FpgaImage metadata");

  PreparedExternalToolInvocation undeclared =
      prepareBundle(__func__, fixture, root / "undeclared", tool);
  ::setenv("LOOM_QUARTUS_TEST_UNDECLARED_OUTPUT", "1", 1);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(undeclared)) == 0,
          "undeclared-output fixture execution failed");
  ::unsetenv("LOOM_QUARTUS_TEST_UNDECLARED_OUTPUT");
  expectFailureContains(__func__, import(undeclared, fixture.binding),
                        "undeclared output");

  PreparedExternalToolInvocation stale =
      prepareBundle(__func__, fixture, root / "stale", tool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(stale)) == 0,
          "stale fixture execution failed");
  auto changedConfig =
      take(__func__, projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                         "altera.quartus-prime-pro:26.1.0-build-110-repacked",
                         kToolVersion, kDevice));
  auto changedBinding = take(
      __func__, resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(
                    changedConfig));
  expectFailureContains(__func__, import(stale, changedBinding),
                        "semantic closure");
}

void runSynthetic(const std::filesystem::path &root) {
  descriptorAndConfigAreExact();
  deterministicPreparationUsesExactInputs(root);
  exactAdmissionIsTyped(root);
  strictImportUsesOnlyTheCompletedDeclaredSnapshot(root);
}

void runRealSmoke(const std::filesystem::path &root, llvm::StringRef module,
                  llvm::StringRef verifiedVersion,
                  llvm::StringRef providerBuild, llvm::StringRef device) {
  createStoreDirectories(root);
  ArtifactStore artifacts((root / "artifacts").string());
  BlobStore blobs((root / "blobs").string());
  SemanticFixture semantics = makeSemanticFixture(__func__, artifacts);
  FinalizedImplementationPlatform platform =
      makePlatform(__func__, artifacts, FpgaVendor::IntelAltera, device);
  FinalizedHardwareImplementation implementation = makeImplementation(
      __func__, semantics, artifacts, blobs, platform.reference(),
      RepresentationRootVariant::Rtl, false);
  auto inputs =
      take(__func__, bindQuartusPrimeStaticFullDeviceCandidateGeneratorInputs(
                         implementation.reference(), platform.reference()));
  auto config =
      take(__func__, projectResolvedQuartusPrimeStaticFullDeviceConfigView(
                         providerBuild, verifiedVersion, device));
  auto binding = take(
      __func__,
      resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(config));
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  local.moduleInit = "/etc/profile.d/modules.sh";
  local.tools["quartus_sh"].binding.modules = {module.str()};
  PreparedExternalToolInvocation prepared =
      take(__func__, prepareCandidateGeneratorInvocation(
                         inputs, binding, artifacts, blobs,
                         ExternalToolPreparationContext{
                             std::move(local), (root / "bundle").string()}));
  const int exitCode =
      take(__func__, executeExternalToolInvocationBundle(prepared));
  require(__func__, exitCode == 0,
          "real Quartus bundle exited with " + std::to_string(exitCode));
  CandidateGeneratorProviderResult result =
      take(__func__, importCandidateGeneratorInvocation(
                         inputs, binding, prepared, artifacts, blobs));
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  require(__func__, completed && completed->outputBindings.size() == 2,
          "real Quartus smoke did not publish both hardware states");
  llvm::outs() << "quartus_smoke version=\"" << verifiedVersion
               << "\" device=\"" << device << "\" module=\"" << module
               << "\" publication=complete\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2) {
    runSynthetic(std::filesystem::absolute(argv[1]).lexically_normal());
    return EXIT_SUCCESS;
  }
  if (argc == 7 && llvm::StringRef(argv[1]) == "--real-smoke") {
    runRealSmoke(std::filesystem::absolute(argv[2]).lexically_normal(), argv[3],
                 argv[4], argv[5], argv[6]);
    return EXIT_SUCCESS;
  }
  fail("main", "expected <root> or --real-smoke <root> <module> "
               "<version> <provider-build> <device>");
}
