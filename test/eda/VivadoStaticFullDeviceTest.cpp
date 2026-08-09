#include "EDA/Adapters/AMD/Vivado.h"

#include "ConfigurationABI2TestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::dse;
using namespace loom::eda::amd;
using namespace loom::external_tool;
using namespace loom::hardware;

namespace {

constexpr llvm::StringLiteral kSyntheticBuild =
    "SW Build 9000000 on Mon Jan 01 00:00:00 UTC 2099";
constexpr llvm::StringLiteral kSyntheticPart = "xcvh1782-lsva4737-3HP-e-S";
constexpr std::array<llvm::StringLiteral, 4> kSupportedParts = {
    "xcvh1782-lsva4737-3HP-e-S",
    "xcvp1802-vsva5601-3HP-e-S",
    "xcvu47p-fsvh2892-3-e",
    "xcvu13p-flga2577-3-e",
};
constexpr llvm::StringLiteral kRtl =
    "module top(input logic a, output logic y);\n"
    "  assign y = ~a;\n"
    "endmodule\n";
constexpr llvm::StringLiteral kConstraint =
    "set_property PACKAGE_PIN V17 [get_ports {a}]\n"
    "set_property PACKAGE_PIN U16 [get_ports {y}]\n"
    "set_property IOSTANDARD LVCMOS33 [get_ports {a y}]\n";

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
void expectFailureContains(llvm::StringRef test, llvm::Expected<T> value,
                           llvm::StringRef expected) {
  if (value)
    fail(test, "accepted input expected to fail with '" + expected.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename ErrorT, typename T>
void expectTypedFailure(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted input expected to produce a typed failure");
  bool matched = false;
  llvm::Error remainder = llvm::handleErrors(
      value.takeError(), [&](const ErrorT &) { matched = true; });
  require(test, matched, "failure did not preserve the expected error type");
  llvm::consumeError(std::move(remainder));
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return std::vector<std::uint8_t>(value.bytes_begin(), value.bytes_end());
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  stream << contents.str();
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

void replaceAll(llvm::StringRef test, std::string &contents,
                llvm::StringRef from, llvm::StringRef to) {
  std::size_t offset = 0;
  std::size_t replacements = 0;
  while ((offset = contents.find(from.str(), offset)) != std::string::npos) {
    contents.replace(offset, from.size(), to.str());
    offset += to.size();
    ++replacements;
  }
  require(test, replacements != 0, "fixture text did not contain replacement");
}

PreparedExternalToolInvocation
rewriteManifestText(llvm::StringRef test,
                    const PreparedExternalToolInvocation &original,
                    llvm::StringRef from, llvm::StringRef to,
                    bool mirrorReplacementInScript = false) {
  const std::filesystem::path bundle(original.bundleRoot);
  const std::filesystem::path manifestPath = bundle / "tool-invocation.json";
  std::string manifest = readFile(manifestPath);
  replaceAll(test, manifest, from, to);
  const BlobDigest changedDigest = computeBlobDigest(bytes(manifest));
  writeFile(manifestPath, manifest);

  const std::filesystem::path scriptPath = bundle / "run.sh";
  std::string script = readFile(scriptPath);
  if (mirrorReplacementInScript)
    replaceAll(test, script, from, to);
  replaceAll(test, script, formatBlobDigestHex(original.manifestDigest),
             formatBlobDigestHex(changedDigest));
  writeFile(scriptPath, script);
  return PreparedExternalToolInvocation{bundle.string(), changedDigest};
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

struct Fixture final {
  loom::fabric::FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef firstOwner;
  loom::fabric::FabricSpatialAttachmentEndpointRef firstDataEndpoint;
  ProgrammingUnitRef firstProgrammingUnit;
};

Fixture makeFixture(llvm::StringRef test, const ArtifactStore &artifacts) {
  auto module = makeModule(test, artifacts);
  auto system =
      take(test, hardware::test::makeSpatialCoreSystem(module, artifacts, 1));
  auto abiDraft =
      take(test, hardware::test::makeCompleteConfigurationABIDraft(system));
  auto abi =
      take(test, finalizeConfigurationABI(std::move(abiDraft), artifacts));
  const ProgrammingUnit &unit = abi.abi().programmingUnits().front();
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  const auto endpoint = systemView.spatialAttachments().front().spatialEndpoint;
  const auto firstOwner = unit.exactFabricResourceClosure.front();
  ProgrammingUnitRef programmingUnit{abi.reference(), unit.id};
  return Fixture{std::move(system), std::move(abi), firstOwner, endpoint,
                 std::move(programmingUnit)};
}

platform::FinalizedImplementationPlatform
makePlatform(llvm::StringRef test, const ArtifactStore &artifacts,
             platform::FpgaVendor vendor, llvm::StringRef part) {
  return take(test, platform::finalizeImplementationPlatform(
                        platform::ImplementationPlatformDraft{
                            platform::FpgaTarget{vendor, part.str()},
                            {"device_default"}},
                        artifacts));
}

FinalizedHardwareImplementation
makeImplementation(llvm::StringRef test, const Fixture &fixture,
                   const ArtifactStore &artifacts, const BlobStore &blobs,
                   llvm::StringRef rtl,
                   std::optional<llvm::StringRef> constraint,
                   std::optional<ArtifactRootReference> implementationPlatform =
                       std::nullopt) {
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::RtlSource, "rtl/top.sv",
       take(test, blobs.put(bytes(rtl)))}};
  if (constraint)
    payloads.push_back({PayloadRole::GenerationConstraint,
                        "constraints/top.sdc",
                        take(test, blobs.put(bytes(*constraint)))});
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  auto representation =
      take(test,
           createImplementationRepresentationRoot(
               RepresentationRootVariant::Rtl, std::nullopt, format,
               {RepresentationObjectKind::Module, "top"}, std::move(payloads)));
  HardwareImplementationDraft draft{
      fixture.system.reference(),
      fixture.abi.reference(),
      {},
      std::move(representation),
      std::move(implementationPlatform),
      {{ImplementationDataInterfaceRef{fixture.firstDataEndpoint},
        {RepresentationObjectKind::Port, "top.a"},
        std::nullopt},
       {ImplementationConfigurationInterfaceRef{fixture.firstProgrammingUnit},
        {RepresentationObjectKind::Port, "top.a"},
        std::nullopt}},
      {{{RepresentationObjectKind::Module, "top"}, fixture.firstOwner}},
      {},
      {}};
  return take(
      test, finalizeHardwareImplementation(std::move(draft), artifacts, blobs));
}

std::string synthesisMetadata(llvm::StringRef part) {
  return "{\"schema\":\"loom.vivado_synthesis_attempt\",\"version\":"
         "\"1.0\",\"top\":\"top\",\"device_ordering_code\":\"" +
         part.str() + "\"}\n";
}

std::string physicalMetadata(llvm::StringRef part) {
  return "{\"schema\":\"loom.vivado_fpga_physical_attempt\",\"version\":"
         "\"1.0\",\"stage\":\"routed\",\"top\":\"top\","
         "\"device_ordering_code\":\"" +
         part.str() + "\",\"input_checkpoint\":\"outputs/synthesized.dcp\"}\n";
}

std::string imageMetadata(llvm::StringRef part) {
  return "{\"schema\":\"loom.vivado_fpga_image_attempt\",\"version\":"
         "\"1.0\",\"top\":\"top\",\"device_ordering_code\":\"" +
         part.str() + "\",\"input_checkpoint\":\"outputs/routed.dcp\"}\n";
}

void makeFakeVivado(const std::filesystem::path &path, bool badMetadata,
                    bool deviceAvailable = true, bool failSynthesis = false) {
  std::string script =
      "#!/usr/bin/env bash\n"
      "set -euo pipefail\n"
      "if [[ \"${1-}\" == -version ]]; then\n"
      "  printf '%s\\n' 'vivado v2099.1 (64-bit)' "
      "'SW Build 9000000 on Mon Jan 01 00:00:00 UTC 2099'\n"
      "  exit 0\n"
      "fi\n"
      "loom_source=''\n"
      "while (($#)); do\n"
      "  if [[ \"$1\" == -source ]]; then shift; loom_source=\"$1\"; fi\n"
      "  shift\n"
      "done\n"
      "case \"$loom_source\" in\n"
      "  *drivers/validate-device.tcl)\n"
      "    [[ -f \"$loom_source\" ]]\n"
      "    printf '%s\\n' 'vivado v2099.1 (64-bit)' "
      "'**** SW Build 9000000 on Mon Jan 01 00:00:00 UTC 2099'\n" +
      (deviceAvailable ? "    printf '%s\\n' 'LOOM_VIVADO_DEVICE_AVAILABLE " +
                             kSyntheticPart.str() + "'\n    exit 0\n"
                       : "    exit 1\n") +
      "    ;;\n"
      "  drivers/synthesize.tcl)\n" +
      (failSynthesis ? "    exit 9\n" : "") +
      "    printf 'SYNTHETIC-DCP\\n' >outputs/synthesized.dcp\n"
      "    printf '%s' '" +
      synthesisMetadata(kSyntheticPart) +
      "' >outputs/synthesis.json\n"
      "    ;;\n"
      "  drivers/implement.tcl)\n"
      "    [[ -s outputs/synthesized.dcp ]]\n"
      "    printf 'ROUTED-SYNTHETIC-DCP\\n' >outputs/routed.dcp\n"
      "    printf '%s' '" +
      (badMetadata ? physicalMetadata("xc7a100tcsg324-1")
                   : physicalMetadata(kSyntheticPart)) +
      "' >outputs/fpga-physical.json\n"
      "    ;;\n"
      "  drivers/image.tcl)\n"
      "    [[ -s outputs/routed.dcp ]]\n"
      "    printf 'SYNTHETIC-BITSTREAM\\n' >outputs/device.bit\n"
      "    printf '%s' '" +
      imageMetadata(kSyntheticPart) +
      "' >outputs/fpga-image.json\n"
      "    ;;\n"
      "  *) exit 64 ;;\n"
      "esac\n";
  writeExecutable(path, script);
}

void makeFakeContainer(const std::filesystem::path &path) {
  writeExecutable(path, R"bash(#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == --version ]]; then
  printf '%s\n' 'PolyArch container v2099.1'
  exit 0
fi
[[ "${1-}" == run ]]
shift
loom_workdir=''
while (($#)); do
  if [[ "$1" == --workdir ]]; then
    shift
    loom_workdir="$1"
  fi
  if [[ "$1" == -- ]]; then
    shift
    [[ -n "$loom_workdir" ]]
    cd -- "$loom_workdir"
    exec "$@"
  fi
  shift
done
exit 64
)bash");
}

LocalToolConfig localConfig(const std::filesystem::path &tool) {
  LocalToolConfig config;
  config.runtimePolicy = RuntimePolicy::Host;
  config.tools["vivado"].binding.executable = tool.string();
  return config;
}

ExternalToolPreparationContext
preparationContext(const std::filesystem::path &bundle,
                   const std::filesystem::path &tool) {
  return ExternalToolPreparationContext{localConfig(tool), bundle.string()};
}

void descriptorAndConfigAreExact() {
  for (llvm::StringRef part : kSupportedParts)
    take(__func__, projectResolvedVivadoStaticFullDeviceConfigView(
                       kSyntheticBuild, part));
  expectFailureContains(__func__,
                        projectResolvedVivadoStaticFullDeviceConfigView(
                            kSyntheticBuild, "xc7a35tcpg236-1"),
                        "supported ordering code");

  auto config = take(__func__, projectResolvedVivadoStaticFullDeviceConfigView(
                                   kSyntheticBuild, kSyntheticPart));
  auto adopted =
      take(__func__, adoptResolvedVivadoStaticFullDeviceConfigView(
                         resolvedVivadoStaticFullDeviceConfigSchemaBytes(),
                         config.canonicalViewBytes(), config.digest()));
  require(__func__,
          adopted.stableProviderBuildIdentity() == kSyntheticBuild &&
              adopted.deviceResourceKey() == kSyntheticPart,
          "resolved Vivado config did not round-trip exactly");
  std::vector<std::uint8_t> changed(config.canonicalViewBytes().begin(),
                                    config.canonicalViewBytes().end());
  const auto buildDigit = std::find(changed.begin(), changed.end(), '9');
  require(__func__, buildDigit != changed.end(),
          "resolved config fixture has no mutable build digit");
  *buildDigit = '8';
  expectFailureContains(__func__,
                        adoptResolvedVivadoStaticFullDeviceConfigView(
                            resolvedVivadoStaticFullDeviceConfigSchemaBytes(),
                            changed, config.digest()),
                        "digest");

  requireSuccess(__func__, registerVivadoStaticFullDeviceCandidateGenerator());
  const CandidateGeneratorDescriptor &descriptor =
      vivadoStaticFullDeviceCandidateGeneratorDescriptor();
  require(__func__,
          descriptor.kind == vivadoStaticFullDeviceCandidateGeneratorKind &&
              descriptor.inputSlots.size() == 2 &&
              descriptor.outputSlots.size() == 2 &&
              descriptor.providerForm == ProviderForm::ExternalPrepareImport &&
              descriptor.determinism ==
                  CandidateGeneratorDeterminism::Deterministic,
          "Vivado descriptor shape is not exact");
  require(
      __func__,
      *descriptor.inputSlots[0].schema == hardwareImplementationSchema &&
          *descriptor.inputSlots[1].schema ==
              platform::implementationPlatformSchema &&
          *descriptor.outputSlots[0].schema == hardwareImplementationSchema &&
          *descriptor.outputSlots[1].schema == hardwareImplementationSchema &&
          descriptor.outputSlots[0].semanticRole == "fpga_physical" &&
          descriptor.outputSlots[1].semanticRole == "fpga_image" &&
          descriptor.inputSlots[0].cardinality ==
              PlanValueCardinality::ExactlyOne &&
          descriptor.inputSlots[1].cardinality ==
              PlanValueCardinality::ExactlyOne &&
          descriptor.outputSlots[0].cardinality ==
              PlanValueCardinality::ExactlyOne &&
          descriptor.outputSlots[1].cardinality ==
              PlanValueCardinality::ExactlyOne,
      "Vivado descriptor slots are not exact singleton Artifact slots");
}

void driversAreDeterministicAndDoNotInfer() {
  const std::vector<std::string> rtl{"inputs/rtl/0000000000000000.sv"};
  const std::vector<std::string> constraints{
      "inputs/constraints/0000000000000000.sdc"};
  const std::string synthesis =
      take(__func__, renderVivadoSynthesisDriver("top", kSyntheticPart, rtl,
                                                 constraints));
  require(__func__,
          synthesis ==
              take(__func__, renderVivadoSynthesisDriver("top", kSyntheticPart,
                                                         rtl, constraints)),
          "synthesis driver is nondeterministic");
  require(__func__,
          llvm::StringRef(synthesis).contains("get_parts -quiet") &&
              llvm::StringRef(synthesis).contains(kSyntheticPart) &&
              llvm::StringRef(synthesis).contains("read_verilog -sv") &&
              llvm::StringRef(synthesis).contains("read_xdc") &&
              llvm::StringRef(synthesis).contains(
                  "get_property TOP [current_design]") &&
              !llvm::StringRef(synthesis).contains("create_clock"),
          "synthesis driver omitted exact inputs or inferred a clock");
  const std::string implementation =
      take(__func__, renderVivadoImplementationDriver("top", kSyntheticPart));
  const std::string image =
      take(__func__, renderVivadoImageDriver("top", kSyntheticPart));
  require(
      __func__,
      llvm::StringRef(implementation)
              .contains("set_param general.maxThreads 1\nopen_checkpoint") &&
          llvm::StringRef(implementation).contains("place_design") &&
          llvm::StringRef(implementation).contains("route_design") &&
          llvm::StringRef(implementation).contains("outputs/synthesized.dcp") &&
          llvm::StringRef(image).contains("write_bitstream") &&
          llvm::StringRef(image).contains("outputs/routed.dcp"),
      "implementation and image drivers do not form the exact route");
  expectFailureContains(
      __func__,
      renderVivadoSynthesisDriver("bad top", kSyntheticPart, rtl, constraints),
      "top");
  expectFailureContains(__func__, renderVivadoImageDriver("top", "bad part"),
                        "device");
}

struct LaneFixture final {
  ArtifactStore artifacts;
  BlobStore blobs;
  Fixture fabric;
  platform::FinalizedImplementationPlatform platform;
  FinalizedHardwareImplementation implementation;
  std::vector<CandidateGeneratorInputBinding> inputs;
  ResolvedVivadoStaticFullDeviceConfigView config;
  ResolvedCandidateGeneratorBinding binding;

  LaneFixture(const std::filesystem::path &root,
              platform::FpgaVendor vendor = platform::FpgaVendor::AmdXilinx,
              llvm::StringRef part = kSyntheticPart)
      : artifacts((root / "artifacts").string()),
        blobs((root / "blobs").string()),
        fabric(makeFixture(__func__, artifacts)),
        platform(makePlatform(__func__, artifacts, vendor, part)),
        implementation(makeImplementation(__func__, fabric, artifacts, blobs,
                                          kRtl, kConstraint)),
        inputs(take(__func__,
                    bindVivadoStaticFullDeviceCandidateGeneratorInputs(
                        implementation.reference(), platform.reference()))),
        config(take(__func__, projectResolvedVivadoStaticFullDeviceConfigView(
                                  kSyntheticBuild, part))),
        binding(take(
            __func__,
            resolveVivadoStaticFullDeviceCandidateGeneratorBinding(config))) {}
};

void createStoreDirectories(const std::filesystem::path &root) {
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
}

PreparedExternalToolInvocation
prepareBundle(llvm::StringRef test, const LaneFixture &fixture,
              const std::filesystem::path &bundle,
              const std::filesystem::path &tool) {
  return take(test, prepareCandidateGeneratorInvocation(
                        fixture.inputs, fixture.binding, fixture.artifacts,
                        fixture.blobs, preparationContext(bundle, tool)));
}

void bundleLifecycleIsStrict(const std::filesystem::path &root) {
  const std::filesystem::path data = root / "lane";
  createStoreDirectories(data);
  LaneFixture fixture(data);
  const std::filesystem::path tool = root / "tools" / "vivado";
  makeFakeVivado(tool, false);

  const PreparedExternalToolInvocation first =
      prepareBundle(__func__, fixture, root / "bundle-a", tool);
  const PreparedExternalToolInvocation second =
      prepareBundle(__func__, fixture, root / "bundle-b", tool);
  const PreparedExternalToolInvocation third =
      prepareBundle(__func__, fixture, root / "bundle-c", tool);
  require(__func__,
          readFile(root / "bundle-a" / "tool-invocation.json") ==
                  readFile(root / "bundle-b" / "tool-invocation.json") &&
              readFile(root / "bundle-a" / "tool-invocation.json") ==
                  readFile(root / "bundle-c" / "tool-invocation.json") &&
              readFile(root / "bundle-a" / "run.sh") ==
                  readFile(root / "bundle-b" / "run.sh") &&
              readFile(root / "bundle-a" / "run.sh") ==
                  readFile(root / "bundle-c" / "run.sh") &&
              readFile(root / "bundle-a" / "drivers" / "synthesize.tcl") ==
                  readFile(root / "bundle-b" / "drivers" / "synthesize.tcl") &&
              readFile(root / "bundle-a" / "drivers" / "synthesize.tcl") ==
                  readFile(root / "bundle-c" / "drivers" / "synthesize.tcl"),
          "identical semantic inputs produced different bundle bytes");
  require(__func__,
          readFile(root / "bundle-a" / "inputs" / "rtl" /
                   "0000000000000000.sv") == kRtl &&
              readFile(root / "bundle-a" / "inputs" / "constraints" /
                       "0000000000000000.sdc") == kConstraint,
          "bundle did not materialize exact HImpl payload bytes");
  const std::string manifest =
      readFile(root / "bundle-a" / "tool-invocation.json");
  const std::size_t synthesis = manifest.find("drivers/synthesize.tcl");
  const std::size_t implementation = manifest.find("drivers/implement.tcl");
  const std::size_t image = manifest.find("drivers/image.tcl");
  require(__func__,
          synthesis != std::string::npos && implementation > synthesis &&
              image > implementation,
          "bundle command order is not synthesis, implementation, image");

  const PreparedExternalToolInvocation unexecuted =
      prepareBundle(__func__, fixture, root / "unexecuted", tool);
  auto staleConfig =
      take(__func__, projectResolvedVivadoStaticFullDeviceConfigView(
                         "SW Build 9000001 on Tue Jan 02 00:00:00 UTC 2099",
                         kSyntheticPart));
  auto staleBinding =
      take(__func__,
           resolveVivadoStaticFullDeviceCandidateGeneratorBinding(staleConfig));
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, staleBinding, unexecuted,
                            fixture.artifacts, fixture.blobs),
                        "semantic closure");
  expectTypedFailure<IncompleteExternalToolInvocationError>(
      __func__, importCandidateGeneratorInvocation(
                    fixture.inputs, fixture.binding, unexecuted,
                    fixture.artifacts, fixture.blobs));

  const std::filesystem::path failingTool = root / "tools" / "failing-vivado";
  makeFakeVivado(failingTool, false, true, true);
  const PreparedExternalToolInvocation failed =
      prepareBundle(__func__, fixture, root / "failed", failingTool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(failed)) == 9,
          "synthetic Vivado failure did not preserve the tool exit code");
  std::filesystem::create_directory(root / "failed" / "outputs" /
                                    "synthesized.dcp");
  expectFailureContains(
      __func__,
      importCandidateGeneratorInvocation(fixture.inputs, staleBinding, failed,
                                         fixture.artifacts, fixture.blobs),
      "semantic closure");
  CandidateGeneratorProviderResult failedResult =
      take(__func__, importCandidateGeneratorInvocation(
                         fixture.inputs, fixture.binding, failed,
                         fixture.artifacts, fixture.blobs));
  const auto *executionFailed =
      std::get_if<IncompleteCandidateGeneratorResult>(&failedResult.outcome);
  require(__func__,
          executionFailed &&
              executionFailed->reason ==
                  CandidateGeneratorIncompleteReason::ExecutionFailed &&
              executionFailed->retainedOutputBindings.size() == 2 &&
              executionFailed->retainedOutputBindings[0].slot ==
                  vivadoStaticFullDeviceCandidateGeneratorDescriptor()
                      .outputSlots[0]
                      .slot &&
              executionFailed->retainedOutputBindings[1].slot ==
                  vivadoStaticFullDeviceCandidateGeneratorDescriptor()
                      .outputSlots[1]
                      .slot &&
              executionFailed->retainedOutputBindings[0].artifacts.empty() &&
              executionFailed->retainedOutputBindings[1].artifacts.empty(),
          "tool failure did not remain an output-free typed result");

  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(first)) == 0,
          "synthetic Vivado invocation failed");
  CandidateGeneratorProviderResult result = take(
      __func__,
      importCandidateGeneratorInvocation(fixture.inputs, fixture.binding, first,
                                         fixture.artifacts, fixture.blobs));
  const auto *unsupported =
      std::get_if<IncompleteCandidateGeneratorResult>(&result.outcome);
  require(__func__,
          unsupported &&
              unsupported->reason ==
                  CandidateGeneratorIncompleteReason::Unsupported &&
              unsupported->retainedOutputBindings.size() == 2 &&
              unsupported->retainedOutputBindings[0].slot ==
                  vivadoStaticFullDeviceCandidateGeneratorDescriptor()
                      .outputSlots[0]
                      .slot &&
              unsupported->retainedOutputBindings[1].slot ==
                  vivadoStaticFullDeviceCandidateGeneratorDescriptor()
                      .outputSlots[1]
                      .slot &&
              unsupported->retainedOutputBindings[0].artifacts.empty() &&
              unsupported->retainedOutputBindings[1].artifacts.empty(),
          "schema-limited import did not preserve typed Unsupported");
  require(__func__,
          result.workSummary.size() == 3 &&
              result.workSummary[0].planned == 1 &&
              result.workSummary[0].consumed == 1 &&
              result.workSummary[1].planned == 1 &&
              result.workSummary[1].consumed == 0 &&
              result.workSummary[2].planned == 1 &&
              result.workSummary[2].consumed == 0,
          "Vivado work accounting is not exact");

  expectFailureContains(
      __func__,
      importCandidateGeneratorInvocation(fixture.inputs, staleBinding, first,
                                         fixture.artifacts, fixture.blobs),
      "semantic closure");

  const PreparedExternalToolInvocation originalSubstituted =
      prepareBundle(__func__, fixture, root / "substituted-driver", tool);
  const std::filesystem::path substitutedRoot = root / "substituted-driver";
  const std::filesystem::path substitutedDriver =
      substitutedRoot / "drivers" / "implement.tcl";
  const std::string originalDriver = readFile(substitutedDriver);
  const std::string changedDriver = originalDriver + "# substituted\n";
  const std::string originalDriverDigest =
      formatBlobDigestHex(computeBlobDigest(bytes(originalDriver)));
  const std::string changedDriverDigest =
      formatBlobDigestHex(computeBlobDigest(bytes(changedDriver)));
  writeFile(substitutedDriver, changedDriver);
  const PreparedExternalToolInvocation substituted =
      rewriteManifestText(__func__, originalSubstituted, originalDriverDigest,
                          changedDriverDigest, true);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(substituted)) == 0,
          "substituted-driver fixture did not execute");
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, substituted,
                            fixture.artifacts, fixture.blobs),
                        "driver contract");

  const PreparedExternalToolInvocation originalProbeMutation =
      prepareBundle(__func__, fixture, root / "mutated-probe", tool);
  const PreparedExternalToolInvocation probeMutation = rewriteManifestText(
      __func__, originalProbeMutation, "\"-version\"", "\"--version\"");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(probeMutation)) ==
              0,
          "provider-probe mutation fixture did not execute");
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, probeMutation,
                            fixture.artifacts, fixture.blobs),
                        "provider probe");

  const PreparedExternalToolInvocation originalCommandOrder =
      prepareBundle(__func__, fixture, root / "reordered-command", tool);
  const PreparedExternalToolInvocation temporaryCommandOrder =
      rewriteManifestText(__func__, originalCommandOrder,
                          "drivers/synthesize.tcl",
                          "drivers/reordered-command.tcl");
  const PreparedExternalToolInvocation reversedCommandOrder =
      rewriteManifestText(__func__, temporaryCommandOrder,
                          "drivers/implement.tcl", "drivers/synthesize.tcl");
  const PreparedExternalToolInvocation commandOrder = rewriteManifestText(
      __func__, reversedCommandOrder, "drivers/reordered-command.tcl",
      "drivers/implement.tcl");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(commandOrder)) ==
              0,
          "reordered-command fixture did not execute");
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, commandOrder,
                            fixture.artifacts, fixture.blobs),
                        "command contract");

  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(second)) == 0,
          "second synthetic Vivado invocation failed");
  writeFile(root / "bundle-b" / "outputs" / "routed.dcp", "tampered\n");
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, second,
                            fixture.artifacts, fixture.blobs),
                        "completion digest");

  const PreparedExternalToolInvocation missing =
      prepareBundle(__func__, fixture, root / "missing", tool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(missing)) == 0,
          "missing-output fixture did not execute");
  std::filesystem::remove(root / "missing" / "outputs" / "device.bit");
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, missing,
                            fixture.artifacts, fixture.blobs),
                        "device.bit");

  const PreparedExternalToolInvocation extra =
      prepareBundle(__func__, fixture, root / "extra", tool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(extra)) == 0,
          "undeclared-output fixture did not execute");
  writeFile(root / "extra" / "outputs" / "undeclared.rpt", "synthetic\n");
  expectFailureContains(
      __func__,
      importCandidateGeneratorInvocation(fixture.inputs, fixture.binding, extra,
                                         fixture.artifacts, fixture.blobs),
      "undeclared output");

  const std::filesystem::path badTool = root / "tools" / "bad-vivado";
  makeFakeVivado(badTool, true);
  const PreparedExternalToolInvocation incoherent =
      prepareBundle(__func__, fixture, root / "incoherent", badTool);
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(incoherent)) == 0,
          "incoherent-metadata fixture did not execute");
  expectFailureContains(__func__,
                        importCandidateGeneratorInvocation(
                            fixture.inputs, fixture.binding, incoherent,
                            fixture.artifacts, fixture.blobs),
                        "physical metadata");
}

void typedAdmissionFailuresArePreserved(const std::filesystem::path &root) {
  const std::filesystem::path amdData = root / "admission-amd";
  createStoreDirectories(amdData);
  LaneFixture amd(amdData);
  const std::filesystem::path tool = root / "tools" / "admission-vivado";
  makeFakeVivado(tool, false);

  auto wrongResourceConfig =
      take(__func__, projectResolvedVivadoStaticFullDeviceConfigView(
                         kSyntheticBuild, kSupportedParts[1]));
  auto wrongResourceBinding =
      take(__func__, resolveVivadoStaticFullDeviceCandidateGeneratorBinding(
                         wrongResourceConfig));
  expectTypedFailure<VivadoStaticFullDeviceUnsupportedError>(
      __func__, prepareCandidateGeneratorInvocation(
                    amd.inputs, wrongResourceBinding, amd.artifacts, amd.blobs,
                    preparationContext(root / "wrong-resource", tool)));

  auto wrongBuildConfig =
      take(__func__, projectResolvedVivadoStaticFullDeviceConfigView(
                         "SW Build 9000001 on Tue Jan 02 00:00:00 UTC 2099",
                         kSyntheticPart));
  auto wrongBuildBinding = take(
      __func__,
      resolveVivadoStaticFullDeviceCandidateGeneratorBinding(wrongBuildConfig));
  expectTypedFailure<VivadoStaticFullDeviceUnavailableError>(
      __func__, prepareCandidateGeneratorInvocation(
                    amd.inputs, wrongBuildBinding, amd.artifacts, amd.blobs,
                    preparationContext(root / "wrong-build", tool)));

  const std::filesystem::path missingDeviceTool =
      root / "tools" / "missing-device-vivado";
  makeFakeVivado(missingDeviceTool, false, false);
  expectTypedFailure<VivadoStaticFullDeviceUnavailableError>(
      __func__,
      prepareCandidateGeneratorInvocation(
          amd.inputs, amd.binding, amd.artifacts, amd.blobs,
          preparationContext(root / "missing-device", missingDeviceTool)));

  const std::filesystem::path firstCandidate =
      root / "tools" / "first-provider" / "vivado";
  const std::filesystem::path alternateCandidate =
      root / "tools" / "alternate-provider" / "vivado";
  makeFakeVivado(firstCandidate, false, false);
  makeFakeVivado(alternateCandidate, false, true);
  const char *oldPathValue = std::getenv("PATH");
  require(__func__, oldPathValue != nullptr, "PATH is unavailable");
  const std::string oldPath(oldPathValue);
  const std::string candidatePath =
      firstCandidate.parent_path().string() + ":" +
      alternateCandidate.parent_path().string() + ":" + oldPath;
  require(__func__, ::setenv("PATH", candidatePath.c_str(), 1) == 0,
          "could not install synthetic provider search path");
  LocalToolConfig discoveredConfig;
  discoveredConfig.runtimePolicy = RuntimePolicy::Host;
  expectTypedFailure<VivadoStaticFullDeviceUnavailableError>(
      __func__, prepareCandidateGeneratorInvocation(
                    amd.inputs, amd.binding, amd.artifacts, amd.blobs,
                    ExternalToolPreparationContext{
                        std::move(discoveredConfig),
                        (root / "alternate-provider").string()}));
  require(__func__, ::setenv("PATH", oldPath.c_str(), 1) == 0,
          "could not restore provider search path");

  const std::filesystem::path intelData = root / "admission-intel";
  createStoreDirectories(intelData);
  LaneFixture intel(intelData, platform::FpgaVendor::IntelAltera,
                    kSyntheticPart);
  expectTypedFailure<VivadoStaticFullDeviceUnsupportedError>(
      __func__, prepareCandidateGeneratorInvocation(
                    intel.inputs, intel.binding, intel.artifacts, intel.blobs,
                    preparationContext(root / "wrong-vendor", tool)));

  auto otherPlatform =
      makePlatform(__func__, amd.artifacts, platform::FpgaVendor::AmdXilinx,
                   kSupportedParts[1]);
  auto boundImplementation =
      makeImplementation(__func__, amd.fabric, amd.artifacts, amd.blobs, kRtl,
                         kConstraint, otherPlatform.reference());
  auto mismatchedInputs = take(
      __func__, bindVivadoStaticFullDeviceCandidateGeneratorInputs(
                    boundImplementation.reference(), amd.platform.reference()));
  expectTypedFailure<VivadoStaticFullDeviceUnsupportedError>(
      __func__, prepareCandidateGeneratorInvocation(
                    mismatchedInputs, amd.binding, amd.artifacts, amd.blobs,
                    preparationContext(root / "wrong-platform-owner", tool)));
}

void containerEnvironmentIsFrozen(const std::filesystem::path &root) {
  const std::filesystem::path data = root / "container-environment";
  createStoreDirectories(data);
  LaneFixture fixture(data);
  const std::filesystem::path tool = root / "tools" / "container-vivado";
  const std::filesystem::path container = root / "tools" / "polyarch-container";
  makeFakeVivado(tool, false);
  makeFakeContainer(container);

  constexpr const char *toolEnvironment = "LOOM_VIVADO_TEST_TOOL_ENV";
  constexpr const char *sharedEnvironment = "LOOM_VIVADO_TEST_SHARED_ENV";
  constexpr const char *containerEnvironment = "LOOM_VIVADO_TEST_CONTAINER_ENV";
  require(__func__, ::setenv(toolEnvironment, "synthetic", 1) == 0,
          "could not set synthetic tool environment");
  require(__func__, ::setenv(sharedEnvironment, "synthetic", 1) == 0,
          "could not set synthetic shared environment");
  require(__func__, ::setenv(containerEnvironment, "synthetic", 1) == 0,
          "could not set synthetic container environment");
  LocalToolConfig config = localConfig(tool);
  config.runtimePolicy = RuntimePolicy::PolyArchContainer;
  config.tools["vivado"].inheritEnvironment = {toolEnvironment,
                                               sharedEnvironment};
  config.polyArchContainer.binding.executable = container.string();
  config.polyArchContainer.os = "almalinux9";
  config.polyArchContainer.inheritEnvironment = {sharedEnvironment,
                                                 containerEnvironment};
  const PreparedExternalToolInvocation prepared = take(
      __func__,
      prepareCandidateGeneratorInvocation(
          fixture.inputs, fixture.binding, fixture.artifacts, fixture.blobs,
          ExternalToolPreparationContext{
              std::move(config), (root / "container-bundle").string()}));
  require(__func__,
          ::unsetenv(toolEnvironment) == 0 &&
              ::unsetenv(sharedEnvironment) == 0 &&
              ::unsetenv(containerEnvironment) == 0,
          "could not clear synthetic environment");
  const std::string manifest = readFile(
      std::filesystem::path(prepared.bundleRoot) / "tool-invocation.json");
  llvm::json::Value parsed = take(__func__, llvm::json::parse(manifest));
  const llvm::json::Object *object = parsed.getAsObject();
  const llvm::json::Array *environment =
      object ? object->getArray("inherit_environment") : nullptr;
  require(__func__, environment && environment->size() == 3,
          "container invocation did not deduplicate environment contracts");
  const std::array<llvm::StringRef, 3> expected{
      toolEnvironment, sharedEnvironment, containerEnvironment};
  for (std::size_t index = 0; index != expected.size(); ++index) {
    const std::optional<llvm::StringRef> actual =
        (*environment)[index].getAsString();
    require(__func__, actual && *actual == expected[index],
            "container invocation changed environment contract ordering");
  }
}

void runSyntheticTests(const std::filesystem::path &root) {
  descriptorAndConfigAreExact();
  driversAreDeterministicAndDoNotInfer();
  bundleLifecycleIsStrict(root);
  typedAdmissionFailuresArePreserved(root);
  containerEnvironmentIsFrozen(root);
}

void runRealSmoke(const std::filesystem::path &root, llvm::StringRef module,
                  llvm::StringRef part, llvm::StringRef build,
                  const std::filesystem::path &constraintPath) {
  createStoreDirectories(root);
  ArtifactStore artifacts((root / "artifacts").string());
  BlobStore blobs((root / "blobs").string());
  Fixture fabric = makeFixture(__func__, artifacts);
  auto platform =
      makePlatform(__func__, artifacts, platform::FpgaVendor::AmdXilinx, part);
  const std::string constraint = readFile(constraintPath);
  auto implementation =
      makeImplementation(__func__, fabric, artifacts, blobs, kRtl, constraint);
  auto inputs =
      take(__func__, bindVivadoStaticFullDeviceCandidateGeneratorInputs(
                         implementation.reference(), platform.reference()));
  auto config = take(
      __func__, projectResolvedVivadoStaticFullDeviceConfigView(build, part));
  auto binding = take(
      __func__, resolveVivadoStaticFullDeviceCandidateGeneratorBinding(config));
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  local.moduleInit = "/etc/profile.d/modules.sh";
  local.tools["vivado"].binding.modules = {module.str()};
  PreparedExternalToolInvocation prepared = take(
      __func__, prepareCandidateGeneratorInvocation(
                    inputs, binding, artifacts, blobs,
                    ExternalToolPreparationContext{
                        std::move(local), (root / "invocation").string()}));
  const int status =
      take(__func__, executeExternalToolInvocationBundle(prepared));
  require(__func__, status == 0,
          "real Vivado execution failed; inspect the ignored invocation logs");
  CandidateGeneratorProviderResult result =
      take(__func__, importCandidateGeneratorInvocation(
                         inputs, binding, prepared, artifacts, blobs));
  const auto *unsupported =
      std::get_if<IncompleteCandidateGeneratorResult>(&result.outcome);
  require(__func__,
          unsupported && unsupported->reason ==
                             CandidateGeneratorIncompleteReason::Unsupported,
          "real smoke crossed the schema-limited publication boundary");
  llvm::outs() << "vivado_smoke build=\"" << build << "\" device=\"" << part
               << "\" module=\"" << module << "\" publication=unsupported\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2) {
    const std::filesystem::path root =
        std::filesystem::absolute(argv[1]).lexically_normal();
    std::filesystem::create_directories(root);
    runSyntheticTests(root);
    return EXIT_SUCCESS;
  }
  if (argc == 7 && llvm::StringRef(argv[1]) == "--real-smoke") {
    const std::filesystem::path root =
        std::filesystem::absolute(argv[2]).lexically_normal();
    std::filesystem::create_directories(root);
    runRealSmoke(root, argv[3], argv[4], argv[5], argv[6]);
    return EXIT_SUCCESS;
  }
  llvm::errs() << "usage: " << argv[0]
               << " ROOT | --real-smoke ROOT MODULE PART BUILD XDC\n";
  return EXIT_FAILURE;
}
