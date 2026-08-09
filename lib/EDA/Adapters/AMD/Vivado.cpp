#include "EDA/Adapters/AMD/Vivado.h"

#include "EDA/Adapters/FpgaImplementationPublication.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::eda::amd {

char VivadoStaticFullDeviceUnavailableError::ID = 0;
char VivadoStaticFullDeviceUnsupportedError::ID = 0;

void VivadoStaticFullDeviceUnavailableError::log(
    llvm::raw_ostream &stream) const {
  stream << "vivado_static_full_device_unavailable: " << detail_;
}

std::error_code
VivadoStaticFullDeviceUnavailableError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

void VivadoStaticFullDeviceUnsupportedError::log(
    llvm::raw_ostream &stream) const {
  stream << "vivado_static_full_device_unsupported: " << detail_;
}

std::error_code
VivadoStaticFullDeviceUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace {

using namespace dse;
using namespace external_tool;
using namespace hardware;

constexpr llvm::StringLiteral configDescriptor =
    "loom.amd.vivado_static_full_device_generator.config.1.0";

constexpr std::array<llvm::StringLiteral, 4> supportedDeviceOrderingCodes = {
    "xcvh1782-lsva4737-3HP-e-S",
    "xcvp1802-vsva5601-3HP-e-S",
    "xcvu47p-fsvh2892-3-e",
    "xcvu13p-flga2577-3-e",
};

enum InputSlot : std::uint32_t {
  RtlImplementationInput,
  ImplementationPlatformInput,
  InputSlotCount,
};

enum OutputSlot : std::uint32_t {
  FpgaPhysicalOutput,
  FpgaImageOutput,
  OutputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(RtlImplementationInput),
         "rtl_implementation", PlanValueRole::CandidateSet,
         &hardwareImplementationSchema, PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(ImplementationPlatformInput),
         "implementation_platform", PlanValueRole::CandidateSet,
         &platform::implementationPlatformSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, OutputSlotCount>
    outputSlots = {{
        {CandidateGeneratorOutputSlotRef(FpgaPhysicalOutput), "fpga_physical",
         PlanValueRole::CandidateSet, &hardwareImplementationSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorOutputSlotRef(FpgaImageOutput), "fpga_image",
         PlanValueRole::CandidateSet, &hardwareImplementationSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 3> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "vivado_invocation"},
    {CandidateGeneratorWorkUnitRef(1), "fpga_physical_publication"},
    {CandidateGeneratorWorkUnitRef(2), "fpga_image_publication"},
}};

enum DeclaredOutput : std::size_t {
  SynthesizedCheckpointOutput,
  SynthesisMetadataOutput,
  RoutedCheckpointOutput,
  PhysicalMetadataOutput,
  DeviceImageOutput,
  ImageMetadataOutput,
  DeclaredOutputCount,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "vivado_static_full_device_invalid: " +
                                     message);
}

llvm::Error unavailable(const llvm::Twine &message) {
  return llvm::make_error<VivadoStaticFullDeviceUnavailableError>(
      message.str());
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::make_error<VivadoStaticFullDeviceUnsupportedError>(
      message.str());
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendFixed(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> value) {
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendBytes(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  appendFixed(bytes, value);
}

void appendText(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendBytes(bytes, llvm::ArrayRef<std::uint8_t>(
                         reinterpret_cast<const std::uint8_t *>(value.data()),
                         value.size()));
}

class ConfigReader final {
public:
  explicit ConfigReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::string> text(llvm::StringRef field) {
    if (remaining() < sizeof(std::uint64_t))
      return invalid("truncated " + field + " length");
    std::uint64_t size = 0;
    for (unsigned index = 0; index != sizeof(std::uint64_t); ++index)
      size = (size << 8) | bytes_[offset_++];
    if (size > std::numeric_limits<std::size_t>::max() || size > remaining())
      return invalid("truncated " + field);
    const auto value = bytes_.slice(offset_, static_cast<std::size_t>(size));
    offset_ += static_cast<std::size_t>(size);
    return std::string(value.begin(), value.end());
  }

  bool empty() const { return offset_ == bytes_.size(); }

private:
  std::size_t remaining() const { return bytes_.size() - offset_; }

  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Error validateBuildIdentity(llvm::StringRef value) {
  if (!value.starts_with("SW Build ") || value.trim() != value ||
      !llvm::all_of(value, [](unsigned char character) {
        return character >= 0x20 && character <= 0x7e;
      }))
    return invalid("stable provider build identity must be one exact printable "
                   "SW Build line");
  return llvm::Error::success();
}

bool isResourceCharacter(char character) {
  const unsigned char value = static_cast<unsigned char>(character);
  return std::isalnum(value) || character == '.' || character == '_' ||
         character == '-' || character == '+';
}

llvm::Error validateDeviceResourceKey(llvm::StringRef value) {
  if (value.empty() ||
      !std::isalnum(static_cast<unsigned char>(value.front())) ||
      !std::isalnum(static_cast<unsigned char>(value.back())) ||
      !llvm::all_of(value, isResourceCharacter))
    return invalid("device resource key is not a canonical FPGA ordering "
                   "code");
  if (!llvm::is_contained(supportedDeviceOrderingCodes, value))
    return invalid("device resource key is not a supported ordering code");
  return llvm::Error::success();
}

bool isPortableTop(llvm::StringRef value) {
  const auto first = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto rest = [&](char character) {
    return first(character) || (character >= '0' && character <= '9') ||
           character == '$';
  };
  return !value.empty() && first(value.front()) &&
         llvm::all_of(value.drop_front(), rest);
}

std::vector<std::uint8_t> encodeConfig(llvm::StringRef build,
                                       llvm::StringRef device) {
  std::vector<std::uint8_t> bytes;
  appendText(bytes, build);
  appendText(bytes, device);
  return bytes;
}

struct DecodedConfig final {
  std::string build;
  std::string device;
};

llvm::Expected<DecodedConfig> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  ConfigReader reader(bytes);
  auto build = reader.text("stable provider build identity");
  if (!build)
    return build.takeError();
  auto device = reader.text("device resource key");
  if (!device)
    return device.takeError();
  if (!reader.empty())
    return invalid("resolved config has trailing bytes");
  if (llvm::Error error = validateBuildIdentity(*build))
    return std::move(error);
  if (llvm::Error error = validateDeviceResourceKey(*device))
    return std::move(error);
  const std::vector<std::uint8_t> canonical = encodeConfig(*build, *device);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("resolved config is not canonical");
  return DecodedConfig{std::move(*build), std::move(*device)};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedVivadoStaticFullDeviceConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

std::string ordinalPath(std::uint64_t ordinal, llvm::StringRef prefix,
                        llvm::StringRef suffix) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string number(16, '0');
  for (std::size_t index = 0; index != number.size(); ++index) {
    number[number.size() - index - 1] = hex[ordinal & 0xf];
    ordinal >>= 4;
  }
  return prefix.str() + number + suffix.str();
}

llvm::Expected<std::string> tclPath(llvm::StringRef path,
                                    llvm::StringRef requiredDirectory) {
  if (path.empty() || path.contains('\0') || path.contains('\n') ||
      path.contains('\r'))
    return invalid("driver path is empty or contains a control character");
  const std::filesystem::path parsed(path.str());
  if (parsed.is_absolute() ||
      parsed.lexically_normal().generic_string() != path)
    return invalid("driver path is not a normalized relative path");
  const std::string prefix = requiredDirectory.str() + "/";
  if (!path.starts_with(prefix))
    return invalid("driver path is outside its exact input directory");

  std::string quoted = "\"";
  for (char character : path) {
    if (character == '\\' || character == '"' || character == '$' ||
        character == '[' || character == ']')
      quoted.push_back('\\');
    quoted.push_back(character);
  }
  quoted.push_back('"');
  return quoted;
}

std::string synthesisMetadata(llvm::StringRef top, llvm::StringRef device) {
  return "{\"schema\":\"loom.vivado_synthesis_attempt\",\"version\":"
         "\"1.0\",\"top\":\"" +
         top.str() + "\",\"device_ordering_code\":\"" + device.str() + "\"}\n";
}

std::string physicalMetadata(llvm::StringRef top, llvm::StringRef device) {
  return "{\"schema\":\"loom.vivado_fpga_physical_attempt\",\"version\":"
         "\"1.0\",\"stage\":\"routed\",\"top\":\"" +
         top.str() + "\",\"device_ordering_code\":\"" + device.str() +
         "\",\"input_checkpoint\":\"outputs/synthesized.dcp\"}\n";
}

std::string imageMetadata(llvm::StringRef top, llvm::StringRef device) {
  return "{\"schema\":\"loom.vivado_fpga_image_attempt\",\"version\":"
         "\"1.0\",\"top\":\"" +
         top.str() + "\",\"device_ordering_code\":\"" + device.str() +
         "\",\"input_checkpoint\":\"outputs/routed.dcp\"}\n";
}

std::string exactDevicePreamble(llvm::StringRef device) {
  return "set loom_device {" + device.str() +
         "}\nset loom_parts [get_parts -quiet $loom_device]\n"
         "if {[llength $loom_parts] != 1 || "
         "[lindex $loom_parts 0] ne $loom_device} {\n"
         "  error {exact device resource is unavailable}\n"
         "}\n";
}

std::string resourceProbeMarker(llvm::StringRef device) {
  return "LOOM_VIVADO_DEVICE_AVAILABLE " + device.str();
}

std::string renderVivadoResourceProbeDriver(llvm::StringRef device) {
  return exactDevicePreamble(device) + "puts {" + resourceProbeMarker(device) +
         "}\n";
}

std::string designCoherenceChecks(llvm::StringRef top) {
  return "if {[get_property TOP [current_design]] ne {" + top.str() +
         "}} { error {current design has the wrong top} }\n"
         "if {[get_property PART [current_design]] ne $loom_device} {\n"
         "  error {current design has the wrong device resource}\n"
         "}\n";
}

std::string metadataWriter(llvm::StringRef path, llvm::StringRef contents) {
  return "set loom_metadata [open {" + path.str() +
         "} {w}]\nputs $loom_metadata {" + contents.drop_back().str() +
         "}\nclose $loom_metadata\n";
}

BlobDigest digest(llvm::StringRef contents) {
  return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(contents.data()),
      contents.size()));
}

ExternalToolProviderDescriptor exactBuildVivadoProvider() {
  ExternalToolProviderDescriptor provider = vivadoProvider();
  provider.versionProbe =
      ToolVersionProbe{{"-version"}, "SW Build", {0}, "SW Build"};
  return provider;
}

ExternalToolProviderDescriptor
exactResourceVivadoProvider(llvm::StringRef device,
                            llvm::StringRef probeDriver) {
  ExternalToolProviderDescriptor provider = vivadoProvider();
  const std::string marker = resourceProbeMarker(device);
  provider.versionProbe =
      ToolVersionProbe{{"-mode", "batch", "-nojournal", "-nolog", "-notrace",
                        "-source", probeDriver.str()},
                       marker,
                       {0},
                       marker};
  return provider;
}

bool sameLocalToolBinding(const ResolvedToolBinding &lhs,
                          const ResolvedToolBinding &rhs) {
  return lhs.toolKey == rhs.toolKey && lhs.source == rhs.source &&
         lhs.executable == rhs.executable &&
         lhs.requestedModules == rhs.requestedModules &&
         lhs.loadedModules == rhs.loadedModules &&
         lhs.moduleInit == rhs.moduleInit &&
         lhs.environmentVariable == rhs.environmentVariable;
}

struct InvocationInputs final {
  std::string top;
  std::string device;
  std::array<std::string, DeclaredOutputCount> declaredOutputs;
  std::string imageLogicalName;
  std::vector<std::string> rtlSources;
  std::vector<std::string> constraints;
  std::vector<MaterializedBundleFile> files;
};

bool isVersalDevice(llvm::StringRef device) {
  return device.starts_with("xcvh") || device.starts_with("xcvp");
}

std::array<std::string, DeclaredOutputCount>
declaredOutputPaths(llvm::StringRef device) {
  return {"outputs/synthesized.dcp",
          "outputs/synthesis.json",
          "outputs/routed.dcp",
          "outputs/fpga-physical.json",
          isVersalDevice(device) ? "outputs/device.pdi" : "outputs/device.bit",
          "outputs/fpga-image.json"};
}

llvm::Expected<InvocationInputs> collectInvocationInputs(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<PreparedExternalToolInvocation> prepareProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context);

llvm::Expected<CandidateGeneratorProviderResult> importProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<PreparedExternalToolInvocation>
prepareProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
                const ResolvedCandidateGeneratorBinding &binding,
                const ArtifactStore &artifacts, const BlobStore &blobs,
                const ExternalToolPreparationContext &context) {
  static const ExternalImplementationContractCatalog contracts;
  return prepareProviderWithContracts(inputBindings, binding, contracts,
                                      artifacts, blobs, context);
}

llvm::Expected<CandidateGeneratorProviderResult>
importProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const PreparedExternalToolInvocation &prepared,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  static const ExternalImplementationContractCatalog contracts;
  return importProviderWithContracts(inputBindings, binding, prepared,
                                     contracts, artifacts, blobs);
}

llvm::Error validateExactManifestContract(
    const PreparedExternalToolInvocation &prepared,
    const InvocationInputs &inputs,
    const ResolvedVivadoStaticFullDeviceConfigView &config);

const CandidateGeneratorDescriptor descriptor{
    vivadoStaticFullDeviceCandidateGeneratorKind,
    "eda.amd.vivado_static_full_device",
    "loom.eda.amd.vivado_static_full_device.generator.v3",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::IndependentReplicates,
    workUnits,
    nullptr,
    ProviderForm::ExternalPrepareImport,
};

const CandidateGeneratorProvider provider{
    descriptor.reference(), CandidateGeneratorExternalPrepareImportProvider{
                                prepareProvider, importProvider}};

llvm::Expected<InvocationInputs> collectInvocationInputs(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (binding.descriptorRef() != descriptor.reference())
    return invalid("binding does not select the Vivado generator");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputBindings))
    return std::move(error);
  auto config = adoptResolvedVivadoStaticFullDeviceConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  auto implementation = importHardwareImplementation(
      inputBindings[RtlImplementationInput].artifacts.front(), contracts,
      artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  auto targetPlatform = platform::importImplementationPlatform(
      inputBindings[ImplementationPlatformInput].artifacts.front(), artifacts);
  if (!targetPlatform)
    return targetPlatform.takeError();

  const HardwareImplementation &hardware = implementation->implementation();
  const ImplementationRepresentationRoot &root = hardware.representationRoot();
  if (root.variant != RepresentationRootVariant::Rtl ||
      root.formatRef.kind() != RepresentationFormatKind::SystemVerilogRtl)
    return unsupported("input implementation is not exact SystemVerilog RTL");
  if (!hardware.memoryMacroBindings().empty())
    return unsupported("input RTL requires unsupported memory macro binding");
  for (const ExternalImplementationBinding &external :
       hardware.externalImplementationBindings()) {
    auto physicalInputs = contracts.canonicalizeAndValidateInputs(
        external.providerContractRef, external.externalInputs,
        RepresentationRootVariant::FpgaPhysical);
    if (!physicalInputs)
      return unsupported("external implementation cannot be retained in "
                         "FpgaPhysical: " +
                         llvm::toString(physicalInputs.takeError()));
    for (const ExternalInputBinding &input : external.externalInputs) {
      if (std::holds_alternative<ExplicitFileDependency>(
              input.dependencyIdentity))
        return unsupported(
            "occurrence-scoped explicit-file projection is unavailable");
      const auto &resource =
          std::get<ToolBundledResourceDependency>(input.dependencyIdentity);
      if (resource.stableProviderBuildIdentity !=
          vivadoToolBundledResourceProviderIdentity(
              config->stableProviderBuildIdentity()))
        return unsupported(
            "RTL bundled resource belongs to another provider build");
    }
  }
  if (hardware.implementationPlatform() &&
      *hardware.implementationPlatform() != targetPlatform->reference())
    return unsupported(
        "input RTL is bound to a different ImplementationPlatform");

  const auto *fpga =
      std::get_if<platform::FpgaTarget>(&targetPlatform->platform().target());
  if (!fpga || fpga->vendor != platform::FpgaVendor::AmdXilinx)
    return unsupported("ImplementationPlatform is not an AMD/Xilinx FPGA");
  if (fpga->deviceOrderingCode != config->deviceResourceKey())
    return unsupported(
        "device resource key does not match the exact vendor ordering code");
  if (!isPortableTop(root.top.canonicalName))
    return unsupported("RTL top is not a portable SystemVerilog identifier");

  InvocationInputs result;
  result.top = root.top.canonicalName;
  result.device = fpga->deviceOrderingCode;
  result.declaredOutputs = declaredOutputPaths(result.device);
  result.imageLogicalName =
      isVersalDevice(result.device) ? "image/device.pdi" : "image/device.bit";
  std::uint64_t rtlOrdinal = 0;
  std::uint64_t constraintOrdinal = 0;
  for (const ImplementationPayload &payload : root.payloads) {
    std::string path;
    if (payload.role == PayloadRole::RtlSource) {
      path = ordinalPath(rtlOrdinal++, "inputs/rtl/", ".sv");
      result.rtlSources.push_back(path);
    } else if (payload.role == PayloadRole::GenerationConstraint) {
      path = ordinalPath(constraintOrdinal++, "inputs/constraints/", ".sdc");
      result.constraints.push_back(path);
    } else if (payload.role == PayloadRole::BlackBoxContract) {
      continue;
    } else {
      return unsupported(
          "RTL payload closure contains an unsupported provider contract");
    }
    auto payloadBytes = blobs.get(payload.blobDigest);
    if (!payloadBytes)
      return payloadBytes.takeError();
    result.files.push_back(MaterializedBundleFile{
        std::move(path),
        std::string(payloadBytes->begin(), payloadBytes->end()),
        implementation->reference(), false});
  }
  if (result.rtlSources.empty())
    return invalid("RTL implementation contains no source payload");
  return result;
}

std::vector<std::string>
localInheritedEnvironment(const LocalToolConfig &config) {
  auto tool = config.tools.find("vivado");
  return tool == config.tools.end() ? std::vector<std::string>{}
                                    : tool->second.inheritEnvironment;
}

void appendInheritedEnvironment(std::vector<std::string> &destination,
                                llvm::ArrayRef<std::string> source) {
  for (const std::string &name : source)
    if (llvm::find(destination, name) == destination.end())
      destination.push_back(name);
}

llvm::Expected<PreparedExternalToolInvocation> prepareProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  auto inputs = collectInvocationInputs(inputBindings, binding, contracts,
                                        artifacts, blobs);
  if (!inputs)
    return inputs.takeError();
  auto semanticContract =
      deriveExternalToolSemanticContract(inputBindings, binding);
  if (!semanticContract)
    return semanticContract.takeError();
  auto config = adoptResolvedVivadoStaticFullDeviceConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();

  const std::filesystem::path bundle(context.bundleDestination);
  const std::filesystem::path bundleParent = bundle.parent_path();
  llvm::SmallString<256> probePath;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          (bundleParent / "loom-vivado-resource-probe").string(), probePath))
    return unavailable("could not create exact device probe directory: " +
                       error.message());
  const std::filesystem::path probeDirectory(probePath.str().str());
  llvm::scope_exit removeProbeDirectory([&] {
    std::error_code ignored;
    std::filesystem::remove_all(probeDirectory, ignored);
  });
  std::error_code probeError;
  std::filesystem::create_directories(probeDirectory / "drivers", probeError);
  if (probeError)
    return unavailable("could not create exact device probe layout: " +
                       probeError.message());
  const std::string resourceProbeDriver =
      renderVivadoResourceProbeDriver(inputs->device);
  {
    std::ofstream stream(probeDirectory / "drivers" / "validate-device.tcl",
                         std::ios::binary | std::ios::trunc);
    if (!stream)
      return unavailable("could not materialize exact device probe");
    stream << resourceProbeDriver;
    if (!stream)
      return unavailable("could not write exact device probe");
  }

  const ExternalToolProviderDescriptor buildVivado = exactBuildVivadoProvider();
  ShellToolBindingProbe buildProbe(probeDirectory.string(),
                                   buildVivado.versionProbe);
  auto tool = resolveToolBinding(buildVivado.binding, context.localConfig,
                                 captureToolEnvironment(buildVivado.binding),
                                 buildProbe);
  if (!tool)
    return unavailable(llvm::toString(tool.takeError()));
  if (tool->version != config->stableProviderBuildIdentity())
    return unavailable("resolved Vivado build '" + tool->version +
                       "' does not match semantic build '" +
                       config->stableProviderBuildIdentity() + "'");

  const ExternalToolProviderDescriptor hostResourceVivado =
      exactResourceVivadoProvider(
          inputs->device,
          (probeDirectory / "drivers" / "validate-device.tcl").string());
  ShellToolBindingProbe resourceBindingProbe(probeDirectory.string(),
                                             hostResourceVivado.versionProbe);
  auto resource = resolveToolBinding(
      hostResourceVivado.binding, context.localConfig,
      captureToolEnvironment(hostResourceVivado.binding), resourceBindingProbe);
  if (!resource)
    return unavailable(llvm::toString(resource.takeError()));
  if (resource->version != resourceProbeMarker(inputs->device) ||
      !sameLocalToolBinding(*tool, *resource))
    return unavailable(
        "exact device probe did not preserve the Vivado resource binding");
  const ExternalToolProviderDescriptor containerResourceVivado =
      exactResourceVivadoProvider(inputs->device,
                                  "drivers/validate-device.tcl");

  const std::vector<std::string> toolEnvironment =
      localInheritedEnvironment(context.localConfig);
  std::vector<std::string> compositionEnvironment = toolEnvironment;
  appendInheritedEnvironment(
      compositionEnvironment,
      context.localConfig.polyArchContainer.inheritEnvironment);
  const ExternalToolProviderDescriptor &container = polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeDirectory.string(),
                                       container.versionProbe);
  auto runtime = resolveInvocationRuntime(
      *tool, context.localConfig, container.binding,
      captureToolEnvironment(container.binding), containerProbe,
      buildVivado.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &resolvedContainer,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        ResolvedToolBinding resourceTool = resolvedTool;
        resourceTool.version = resourceProbeMarker(inputs->device);
        auto resourceResult = probeContainerToolComposition(
            probeDirectory.string(), resourceTool,
            containerResourceVivado.versionProbe, resolvedContainer, os,
            compositionEnvironment);
        if (!resourceResult)
          return resourceResult.takeError();
        if (*resourceResult)
          return std::move(*resourceResult);
        return probeContainerToolComposition(
            probeDirectory.string(), resolvedTool, buildVivado.versionProbe,
            resolvedContainer, os, compositionEnvironment);
      });
  if (!runtime)
    return unavailable(llvm::toString(runtime.takeError()));

  auto synthesis = renderVivadoSynthesisDriver(
      inputs->top, inputs->device, inputs->rtlSources, inputs->constraints);
  if (!synthesis)
    return synthesis.takeError();
  auto implementation =
      renderVivadoImplementationDriver(inputs->top, inputs->device);
  if (!implementation)
    return implementation.takeError();
  auto image = renderVivadoImageDriver(inputs->top, inputs->device);
  if (!image)
    return image.takeError();

  ExternalToolInvocationBundleSpec specification;
  specification.semanticContract = std::move(*semanticContract);
  specification.tool = std::move(*tool);
  specification.toolVersionProbe = buildVivado.versionProbe;
  specification.runtime = std::move(*runtime);
  specification.containerVersionProbe = container.versionProbe;
  specification.commands = {
      {specification.tool.executable, "-mode", "batch", "-nojournal", "-nolog",
       "-notrace", "-source", "drivers/synthesize.tcl"},
      {specification.tool.executable, "-mode", "batch", "-nojournal", "-nolog",
       "-notrace", "-source", "drivers/implement.tcl"},
      {specification.tool.executable, "-mode", "batch", "-nojournal", "-nolog",
       "-notrace", "-source", "drivers/image.tcl"},
  };
  specification.inheritEnvironment = toolEnvironment;
  if (specification.runtime.kind == InvocationRuntimeKind::PolyArchContainer)
    appendInheritedEnvironment(
        specification.inheritEnvironment,
        context.localConfig.polyArchContainer.inheritEnvironment);
  specification.declaredOutputs.assign(inputs->declaredOutputs.begin(),
                                       inputs->declaredOutputs.end());
  specification.files = std::move(inputs->files);
  specification.files.push_back(
      {"drivers/synthesize.tcl", std::move(*synthesis), std::nullopt, false});
  specification.files.push_back({"drivers/implement.tcl",
                                 std::move(*implementation), std::nullopt,
                                 false});
  specification.files.push_back(
      {"drivers/image.tcl", std::move(*image), std::nullopt, false});
  return finalizeExternalToolInvocationBundle(context.bundleDestination,
                                              specification);
}

llvm::Error rejectUndeclaredOutputs(
    llvm::StringRef bundleRoot,
    const std::array<std::string, DeclaredOutputCount> &declaredOutputs) {
  const std::filesystem::path outputs =
      std::filesystem::path(bundleRoot.str()) / "outputs";
  std::set<std::string> allowed{"completion.json", "stdout.log", "stderr.log"};
  for (const std::string &output : declaredOutputs)
    allowed.insert(std::filesystem::path(output).filename().string());
  std::set<std::string> found;
  std::error_code error;
  const std::filesystem::file_status rootStatus =
      std::filesystem::symlink_status(outputs, error);
  if (error || !std::filesystem::is_directory(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return invalid("outputs directory is missing or not an ordinary directory");
  for (std::filesystem::directory_iterator iterator(outputs, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::path path = iterator->path();
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error)
      break;
    const std::string name = path.filename().string();
    if (!std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status) || !allowed.count(name))
      return invalid("outputs directory contains undeclared output '" + name +
                     "'");
    found.insert(name);
  }
  if (error)
    return invalid("could not enumerate outputs directory: " + error.message());
  if (found != allowed)
    return invalid("outputs directory omits a lifecycle or declared output");
  return llvm::Error::success();
}

llvm::Expected<std::string>
readExactManifest(const PreparedExternalToolInvocation &prepared) {
  const std::filesystem::path path =
      std::filesystem::path(prepared.bundleRoot) / "tool-invocation.json";
  std::error_code statusError;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, statusError);
  if (statusError || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status))
    return invalid("prepared manifest is not an ordinary file");
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return invalid("prepared manifest cannot be opened");
  std::string contents{std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>()};
  if (stream.bad())
    return invalid("prepared manifest cannot be read");
  if (digest(contents) != prepared.manifestDigest)
    return invalid("prepared manifest does not match its exact handle");
  return contents;
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Array &array, llvm::StringRef field,
                 llvm::StringRef contract) {
  std::vector<std::string> result;
  result.reserve(array.size());
  for (const llvm::json::Value &value : array) {
    std::optional<llvm::StringRef> text = value.getAsString();
    if (!text)
      return invalid(contract + " field '" + field +
                     "' contains a non-string value");
    result.push_back(text->str());
  }
  return result;
}

llvm::Expected<std::vector<std::string>>
requireStringArray(const llvm::json::Object &object, llvm::StringRef field,
                   llvm::StringRef contract) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array)
    return invalid(contract + " requires array field '" + field + "'");
  return parseStringArray(*array, field, contract);
}

llvm::Error validateExactManifestContract(
    const PreparedExternalToolInvocation &prepared,
    const InvocationInputs &inputs,
    const ResolvedVivadoStaticFullDeviceConfigView &config) {
  auto manifestBytes = readExactManifest(prepared);
  if (!manifestBytes)
    return manifestBytes.takeError();
  auto parsed = llvm::json::parse(*manifestBytes);
  if (!parsed)
    return invalid("prepared manifest is malformed JSON");
  const llvm::json::Object *manifest = parsed->getAsObject();
  const llvm::json::Object *tool =
      manifest ? manifest->getObject("tool_binding") : nullptr;
  const std::optional<llvm::StringRef> toolKey =
      tool ? tool->getString("tool_key") : std::nullopt;
  const std::optional<llvm::StringRef> executable =
      tool ? tool->getString("executable") : std::nullopt;
  const std::optional<llvm::StringRef> version =
      tool ? tool->getString("version") : std::nullopt;
  if (!toolKey || *toolKey != "vivado" || !executable || executable->empty() ||
      !version || *version != config.stableProviderBuildIdentity())
    return invalid("prepared manifest violates the exact provider contract");

  const ToolVersionProbe expectedProbe =
      exactBuildVivadoProvider().versionProbe;
  const llvm::json::Object *probe = manifest->getObject("tool_version_probe");
  if (!probe)
    return invalid("provider contract requires a probe");
  auto arguments = requireStringArray(*probe, "arguments", "provider contract");
  if (!arguments)
    return arguments.takeError();
  const llvm::json::Array *exitCodes = probe->getArray("accepted_exit_codes");
  const std::optional<std::int64_t> exitCode =
      exitCodes && exitCodes->size() == 1 ? (*exitCodes)[0].getAsInteger()
                                          : std::nullopt;
  const std::optional<llvm::StringRef> required =
      probe->getString("required_output_substring");
  const std::optional<llvm::StringRef> selected =
      probe->getString("selected_output_line_substring");
  if (*arguments != expectedProbe.arguments || !exitCode || *exitCode != 0 ||
      !required || !expectedProbe.requiredOutputSubstring ||
      *required != *expectedProbe.requiredOutputSubstring || !selected ||
      !expectedProbe.selectedOutputLineSubstring ||
      *selected != *expectedProbe.selectedOutputLineSubstring)
    return invalid("prepared manifest violates the exact provider probe");

  const llvm::json::Array *commands = manifest->getArray("commands");
  const std::vector<std::vector<std::string>> expectedCommands{
      {executable->str(), "-mode", "batch", "-nojournal", "-nolog", "-notrace",
       "-source", "drivers/synthesize.tcl"},
      {executable->str(), "-mode", "batch", "-nojournal", "-nolog", "-notrace",
       "-source", "drivers/implement.tcl"},
      {executable->str(), "-mode", "batch", "-nojournal", "-nolog", "-notrace",
       "-source", "drivers/image.tcl"},
  };
  if (!commands || commands->size() != expectedCommands.size())
    return invalid("prepared manifest violates the exact command contract");
  for (std::size_t index = 0; index != commands->size(); ++index) {
    const llvm::json::Array *command = (*commands)[index].getAsArray();
    if (!command)
      return invalid("prepared manifest violates the exact command contract");
    auto tokens = parseStringArray(*command, "command", "command contract");
    if (!tokens)
      return tokens.takeError();
    if (*tokens != expectedCommands[index])
      return invalid("prepared manifest violates the exact command contract");
  }

  auto synthesis = renderVivadoSynthesisDriver(
      inputs.top, inputs.device, inputs.rtlSources, inputs.constraints);
  if (!synthesis)
    return synthesis.takeError();
  auto implementation =
      renderVivadoImplementationDriver(inputs.top, inputs.device);
  if (!implementation)
    return implementation.takeError();
  auto image = renderVivadoImageDriver(inputs.top, inputs.device);
  if (!image)
    return image.takeError();
  const std::array<std::pair<llvm::StringRef, llvm::StringRef>, 3> drivers{{
      {"drivers/synthesize.tcl", *synthesis},
      {"drivers/implement.tcl", *implementation},
      {"drivers/image.tcl", *image},
  }};
  const llvm::json::Array *files = manifest->getArray("materialized_files");
  if (!files || files->size() != inputs.files.size() + drivers.size())
    return invalid("prepared manifest violates the exact driver contract");
  for (const auto &[path, contents] : drivers) {
    const std::string expectedDigest = formatBlobDigestHex(digest(contents));
    bool matched = false;
    for (const llvm::json::Value &value : *files) {
      const llvm::json::Object *file = value.getAsObject();
      const std::optional<llvm::StringRef> actualPath =
          file ? file->getString("path") : std::nullopt;
      if (!actualPath || *actualPath != path)
        continue;
      const std::optional<bool> executableFile = file->getBoolean("executable");
      const std::optional<llvm::StringRef> contentDigest =
          file->getString("content_sha256");
      matched = executableFile && !*executableFile && contentDigest &&
                *contentDigest == expectedDigest &&
                !file->get("source_artifact_ref");
      break;
    }
    if (!matched)
      return invalid("prepared manifest violates the exact driver contract");
  }
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult> importProviderWithContracts(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto inputs = collectInvocationInputs(inputBindings, binding, contracts,
                                        artifacts, blobs);
  if (!inputs)
    return inputs.takeError();

  ExternalToolInvocationImportExpectation expectation;
  auto semanticContract =
      deriveExternalToolSemanticContract(inputBindings, binding);
  if (!semanticContract)
    return semanticContract.takeError();
  expectation.semanticContract = std::move(*semanticContract);
  for (const MaterializedBundleFile &file : inputs->files)
    expectation.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact, digest(file.contents)});
  expectation.declaredOutputs.assign(inputs->declaredOutputs.begin(),
                                     inputs->declaredOutputs.end());
  auto attempt = importExternalToolInvocationAttempt(prepared, expectation);
  if (!attempt)
    return attempt.takeError();
  auto config = adoptResolvedVivadoStaticFullDeviceConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  if (llvm::Error error =
          validateExactManifestContract(prepared, *inputs, *config))
    return std::move(error);
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt)) {
    CandidateGeneratorIncompleteReason reason;
    switch (failed->status) {
    case InvocationCompletionStatus::MissingEnvironment:
    case InvocationCompletionStatus::ModuleActivationFailed:
    case InvocationCompletionStatus::VersionMismatch:
      reason = CandidateGeneratorIncompleteReason::ProviderUnavailable;
      break;
    case InvocationCompletionStatus::ToolExit:
    case InvocationCompletionStatus::MissingOutput:
      reason = CandidateGeneratorIncompleteReason::ExecutionFailed;
      break;
    case InvocationCompletionStatus::BundleContentMismatch:
      return invalid("invocation bundle content changed before execution");
    case InvocationCompletionStatus::Success:
      return invalid("failed invocation outcome carries success status");
    }
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            reason,
            {{CandidateGeneratorOutputSlotRef(FpgaPhysicalOutput), {}},
             {CandidateGeneratorOutputSlotRef(FpgaImageOutput), {}}},
            {}},
        {{CandidateGeneratorWorkUnitRef(0), 1, 1},
         {CandidateGeneratorWorkUnitRef(1), 1, 0},
         {CandidateGeneratorWorkUnitRef(2), 1, 0}}};
  }
  ImportedExternalToolInvocationBundle imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  if (llvm::Error error =
          rejectUndeclaredOutputs(prepared.bundleRoot, inputs->declaredOutputs))
    return std::move(error);

  auto synthesized = readExternalToolInvocationDeclaredOutput(
      imported, inputs->declaredOutputs[SynthesizedCheckpointOutput]);
  if (!synthesized)
    return synthesized.takeError();
  auto synthesisFacts = readExternalToolInvocationDeclaredOutput(
      imported, inputs->declaredOutputs[SynthesisMetadataOutput]);
  if (!synthesisFacts)
    return synthesisFacts.takeError();
  if (synthesized->empty())
    return invalid("synthesized checkpoint is empty");
  if (*synthesisFacts != synthesisMetadata(inputs->top, inputs->device))
    return invalid("synthesis metadata is not exact");

  auto routed = readExternalToolInvocationDeclaredOutput(
      imported, inputs->declaredOutputs[RoutedCheckpointOutput]);
  if (!routed)
    return routed.takeError();
  auto physicalFacts = readExternalToolInvocationDeclaredOutput(
      imported, inputs->declaredOutputs[PhysicalMetadataOutput]);
  if (!physicalFacts)
    return physicalFacts.takeError();
  if (routed->empty())
    return invalid("routed checkpoint is empty");
  if (*physicalFacts != physicalMetadata(inputs->top, inputs->device))
    return invalid("physical metadata is not exact");

  auto bitstream = readExternalToolInvocationDeclaredOutput(
      imported, inputs->declaredOutputs[DeviceImageOutput]);
  if (!bitstream)
    return bitstream.takeError();
  auto imageFacts = readExternalToolInvocationDeclaredOutput(
      imported, inputs->declaredOutputs[ImageMetadataOutput]);
  if (!imageFacts)
    return imageFacts.takeError();
  if (bitstream->empty())
    return invalid("device image is empty");
  if (*imageFacts != imageMetadata(inputs->top, inputs->device))
    return invalid("image metadata is not exact");

  auto source = importHardwareImplementation(
      inputBindings[RtlImplementationInput].artifacts.front(), contracts,
      artifacts, blobs);
  if (!source)
    return source.takeError();
  auto publishedPhysical = publishRoutedFpgaPhysicalImplementation(
      *source, inputBindings[ImplementationPlatformInput].artifacts.front(),
      inputs->device, "database/vivado-routed.dcp", *routed, contracts,
      artifacts, blobs);
  if (!publishedPhysical)
    return publishedPhysical.takeError();
  auto publishedImage = publishFpgaImageImplementation(
      *publishedPhysical,
      inputBindings[ImplementationPlatformInput].artifacts.front(),
      inputs->device, inputs->imageLogicalName, *bitstream, contracts,
      artifacts, blobs);
  if (!publishedImage)
    return publishedImage.takeError();

  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(FpgaPhysicalOutput),
            {publishedPhysical->reference()}},
           {CandidateGeneratorOutputSlotRef(FpgaImageOutput),
            {publishedImage->reference()}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(FpgaPhysicalOutput),
            publishedPhysical->reference(),
            {},
            {}},
           {CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(FpgaImageOutput),
            publishedImage->reference(),
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1},
       {CandidateGeneratorWorkUnitRef(1), 1, 1},
       {CandidateGeneratorWorkUnitRef(2), 1, 1}}};
}

} // namespace

llvm::ArrayRef<std::uint8_t> resolvedVivadoStaticFullDeviceConfigSchemaBytes() {
  return descriptorBytes();
}

std::string vivadoToolBundledResourceProviderIdentity(
    llvm::StringRef stableProviderBuildIdentity) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result = "amd_vivado_build_";
  result.reserve(result.size() + stableProviderBuildIdentity.size() * 2);
  for (const unsigned char byte : stableProviderBuildIdentity.bytes()) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ResolvedVivadoStaticFullDeviceConfigView>
projectResolvedVivadoStaticFullDeviceConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    llvm::StringRef deviceResourceKey) {
  if (llvm::Error error = validateBuildIdentity(stableProviderBuildIdentity))
    return std::move(error);
  if (llvm::Error error = validateDeviceResourceKey(deviceResourceKey))
    return std::move(error);
  std::vector<std::uint8_t> canonical =
      encodeConfig(stableProviderBuildIdentity, deviceResourceKey);
  auto digest = computeComponentViewDigest(descriptorBytes(), canonical);
  if (!digest)
    return digest.takeError();
  return ResolvedVivadoStaticFullDeviceConfigView(
      stableProviderBuildIdentity.str(), deviceResourceKey.str(),
      std::move(canonical), std::move(*digest));
}

llvm::Expected<ResolvedVivadoStaticFullDeviceConfigView>
adoptResolvedVivadoStaticFullDeviceConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digestValue) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  auto decoded = decodeConfig(canonicalViewBytes);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digestValue))
    return std::move(error);
  return ResolvedVivadoStaticFullDeviceConfigView(
      std::move(decoded->build), std::move(decoded->device),
      canonicalViewBytes.vec(), digestValue);
}

const CandidateGeneratorDescriptor &
vivadoStaticFullDeviceCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerVivadoStaticFullDeviceCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindVivadoStaticFullDeviceCandidateGeneratorInputs(
    const ArtifactRootReference &rtlImplementation,
    const ArtifactRootReference &implementationPlatform) {
  if (llvm::Error error = registerVivadoStaticFullDeviceCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings{
      {CandidateGeneratorInputSlotRef(RtlImplementationInput),
       {rtlImplementation}},
      {CandidateGeneratorInputSlotRef(ImplementationPlatformInput),
       {implementationPlatform}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveVivadoStaticFullDeviceCandidateGeneratorBinding(
    const ResolvedVivadoStaticFullDeviceConfigView &config) {
  if (llvm::Error error = registerVivadoStaticFullDeviceCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<PreparedExternalToolInvocation>
prepareVivadoStaticFullDeviceInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  return prepareProviderWithContracts(inputs, binding, contracts, artifacts,
                                      blobs, context);
}

llvm::Expected<CandidateGeneratorProviderResult>
importVivadoStaticFullDeviceInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderWithContracts(inputs, binding, prepared, contracts,
                                     artifacts, blobs);
}

llvm::Expected<std::string>
renderVivadoSynthesisDriver(llvm::StringRef topModule,
                            llvm::StringRef deviceOrderingCode,
                            llvm::ArrayRef<std::string> rtlSources,
                            llvm::ArrayRef<std::string> generationConstraints) {
  if (!isPortableTop(topModule))
    return invalid("top is not a portable SystemVerilog identifier");
  if (llvm::Error error = validateDeviceResourceKey(deviceOrderingCode))
    return std::move(error);
  if (rtlSources.empty())
    return invalid("synthesis driver requires at least one RTL source");
  std::string driver = exactDevicePreamble(deviceOrderingCode);
  for (const std::string &source : rtlSources) {
    auto quoted = tclPath(source, "inputs/rtl");
    if (!quoted)
      return quoted.takeError();
    driver += "read_verilog -sv " + *quoted + "\n";
  }
  for (const std::string &constraint : generationConstraints) {
    auto quoted = tclPath(constraint, "inputs/constraints");
    if (!quoted)
      return quoted.takeError();
    driver += "read_xdc " + *quoted + "\n";
  }
  driver += "synth_design -top {" + topModule.str() + "} -part $loom_device\n";
  driver += designCoherenceChecks(topModule);
  driver += "write_checkpoint -force {outputs/synthesized.dcp}\n";
  driver += metadataWriter("outputs/synthesis.json",
                           synthesisMetadata(topModule, deviceOrderingCode));
  return driver;
}

llvm::Expected<std::string>
renderVivadoImplementationDriver(llvm::StringRef topModule,
                                 llvm::StringRef deviceOrderingCode) {
  if (!isPortableTop(topModule))
    return invalid("top is not a portable SystemVerilog identifier");
  if (llvm::Error error = validateDeviceResourceKey(deviceOrderingCode))
    return std::move(error);
  std::string driver = exactDevicePreamble(deviceOrderingCode);
  driver += "set_param general.maxThreads 1\n";
  driver += "open_checkpoint {outputs/synthesized.dcp}\n";
  driver += designCoherenceChecks(topModule);
  driver += "opt_design\nplace_design\nroute_design\n";
  driver += "write_checkpoint -force {outputs/routed.dcp}\n";
  driver += metadataWriter("outputs/fpga-physical.json",
                           physicalMetadata(topModule, deviceOrderingCode));
  return driver;
}

llvm::Expected<std::string>
renderVivadoImageDriver(llvm::StringRef topModule,
                        llvm::StringRef deviceOrderingCode) {
  if (!isPortableTop(topModule))
    return invalid("top is not a portable SystemVerilog identifier");
  if (llvm::Error error = validateDeviceResourceKey(deviceOrderingCode))
    return std::move(error);
  std::string driver = exactDevicePreamble(deviceOrderingCode);
  driver += "open_checkpoint {outputs/routed.dcp}\n";
  driver += designCoherenceChecks(topModule);
  driver += isVersalDevice(deviceOrderingCode)
                ? "write_device_image -force {outputs/device.pdi}\n"
                : "write_bitstream -force {outputs/device.bit}\n";
  driver += metadataWriter("outputs/fpga-image.json",
                           imageMetadata(topModule, deviceOrderingCode));
  return driver;
}

} // namespace loom::eda::amd
