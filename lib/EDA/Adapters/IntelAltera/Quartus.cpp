#include "EDA/Adapters/IntelAltera/Quartus.h"

#include "EDA/Adapters/FpgaImplementationPublication.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::eda::intel_altera {

char QuartusPrimeUnsupportedError::ID = 0;
char QuartusPrimeUnavailableError::ID = 0;

void QuartusPrimeUnsupportedError::log(llvm::raw_ostream &stream) const {
  stream << "quartus_prime_unsupported: " << detail_;
}

std::error_code QuartusPrimeUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

void QuartusPrimeUnavailableError::log(llvm::raw_ostream &stream) const {
  stream << "quartus_prime_unavailable: " << detail_;
}

std::error_code QuartusPrimeUnavailableError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace {

using namespace dse;
using namespace external_tool;
using namespace hardware;

constexpr llvm::StringLiteral kConfigDescriptor =
    "loom.intel_altera.quartus_prime_static_full_device_generator.config.1.0";
constexpr llvm::StringLiteral kDriverPath = "drivers/quartus-static.tcl";
constexpr llvm::StringLiteral kPlatformPath = "inputs/platform.json";
constexpr llvm::StringLiteral kPhysicalPath = "outputs/fpga-physical.qar";
constexpr llvm::StringLiteral kPhysicalMetadataPath =
    "outputs/fpga-physical.json";
constexpr llvm::StringLiteral kImagePath = "outputs/device.sof";
constexpr llvm::StringLiteral kImageMetadataPath = "outputs/fpga-image.json";
constexpr llvm::StringLiteral kScratchArchivePath =
    "quartus-work/loom_quartus.qar";
constexpr llvm::StringLiteral kScratchImagePath =
    "quartus-work/output/loom_quartus.sof";

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
    kInputSlots = {{
        {CandidateGeneratorInputSlotRef(RtlImplementationInput),
         "rtl_implementation", PlanValueRole::CandidateSet,
         &hardwareImplementationSchema, PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorInputSlotRef(ImplementationPlatformInput),
         "implementation_platform", PlanValueRole::CandidateSet,
         &platform::implementationPlatformSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, OutputSlotCount>
    kOutputSlots = {{
        {CandidateGeneratorOutputSlotRef(FpgaPhysicalOutput), "fpga_physical",
         PlanValueRole::CandidateSet, &hardwareImplementationSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorOutputSlotRef(FpgaImageOutput), "fpga_image",
         PlanValueRole::CandidateSet, &hardwareImplementationSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 3> kWorkUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "quartus_invocation"},
    {CandidateGeneratorWorkUnitRef(1), "fpga_physical_publication"},
    {CandidateGeneratorWorkUnitRef(2), "fpga_image_publication"},
}};

constexpr std::array<llvm::StringLiteral, 4> kDeclaredOutputPaths = {
    kPhysicalPath, kPhysicalMetadataPath, kImagePath, kImageMetadataPath};

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "quartus_prime_adapter_invalid: " + detail);
}

template <typename T>
llvm::Expected<T> unsupported(QuartusPrimeUnsupportedReason reason,
                              const llvm::Twine &detail) {
  return llvm::make_error<QuartusPrimeUnsupportedError>(reason, detail.str());
}

template <typename T>
llvm::Expected<T> unavailable(QuartusPrimeUnavailableReason reason,
                              const llvm::Twine &detail) {
  return llvm::make_error<QuartusPrimeUnavailableError>(reason, detail.str());
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(kConfigDescriptor.data()),
          kConfigDescriptor.size()};
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

bool isPrintableAscii(llvm::StringRef value) {
  return !value.empty() && value.trim() == value &&
         llvm::all_of(value, [](unsigned char character) {
           return character >= 0x20 && character <= 0x7e;
         });
}

bool isCanonicalProviderBuild(llvm::StringRef value) {
  if (!value.starts_with("altera.quartus-prime-pro:") ||
      !isPrintableAscii(value))
    return false;
  return llvm::all_of(value, [](char character) {
    const unsigned char byte = static_cast<unsigned char>(character);
    return (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
           (byte >= '0' && byte <= '9') || character == '.' ||
           character == '_' || character == '-' || character == ':' ||
           character == '+';
  });
}

bool isDeviceCharacter(char character) {
  const unsigned char byte = static_cast<unsigned char>(character);
  return (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
         (byte >= '0' && byte <= '9') || character == '.' || character == '_' ||
         character == '-' || character == '+';
}

llvm::Error validateConfigFields(llvm::StringRef providerBuild,
                                 llvm::StringRef toolVersion,
                                 llvm::StringRef device) {
  if (!isCanonicalProviderBuild(providerBuild))
    return invalid("stable provider build identity is not canonical");
  if (!toolVersion.starts_with("Version ") || !isPrintableAscii(toolVersion))
    return invalid("verified Quartus tool version is not one exact line");
  if (device.empty() || !isDeviceCharacter(device.front()) ||
      !isDeviceCharacter(device.back()) ||
      !llvm::all_of(device, isDeviceCharacter))
    return invalid("device resource key is not a canonical FPGA ordering code");
  return llvm::Error::success();
}

std::vector<std::uint8_t> encodeConfig(llvm::StringRef providerBuild,
                                       llvm::StringRef toolVersion,
                                       llvm::StringRef device) {
  std::vector<std::uint8_t> bytes;
  appendText(bytes, providerBuild);
  appendText(bytes, toolVersion);
  appendText(bytes, device);
  return bytes;
}

struct DecodedConfig final {
  std::string providerBuild;
  std::string toolVersion;
  std::string device;
};

llvm::Expected<DecodedConfig> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  ConfigReader reader(bytes);
  auto providerBuild = reader.text("stable provider build identity");
  if (!providerBuild)
    return providerBuild.takeError();
  auto toolVersion = reader.text("verified tool version");
  if (!toolVersion)
    return toolVersion.takeError();
  auto device = reader.text("device resource key");
  if (!device)
    return device.takeError();
  if (!reader.empty())
    return invalid("resolved config has trailing bytes");
  if (llvm::Error error =
          validateConfigFields(*providerBuild, *toolVersion, *device))
    return std::move(error);
  const std::vector<std::uint8_t> canonical =
      encodeConfig(*providerBuild, *toolVersion, *device);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("resolved config is not canonical");
  return DecodedConfig{std::move(*providerBuild), std::move(*toolVersion),
                       std::move(*device)};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

bool isPortableIdentifier(llvm::StringRef value) {
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

std::string canonicalMetadata(llvm::StringRef schema,
                              llvm::StringRef providerBuild,
                              llvm::StringRef device, llvm::StringRef top,
                              std::optional<llvm::StringRef> inputPhysical) {
  llvm::SmallString<512> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", schema);
    json.attribute("version", "1.0");
    json.attribute("provider_build_identity", providerBuild);
    json.attribute("device_ordering_code", device);
    json.attribute("device_resource_key", device);
    json.attribute("top", top);
    if (inputPhysical)
      json.attribute("input_physical_output", *inputPhysical);
  });
  output << '\n';
  return output.str().str();
}

std::string physicalMetadata(llvm::StringRef providerBuild,
                             llvm::StringRef device, llvm::StringRef top) {
  return canonicalMetadata("loom.quartus_prime_fpga_physical_attempt",
                           providerBuild, device, top, std::nullopt);
}

std::string imageMetadata(llvm::StringRef providerBuild, llvm::StringRef device,
                          llvm::StringRef top) {
  return canonicalMetadata("loom.quartus_prime_fpga_image_attempt",
                           providerBuild, device, top, kPhysicalPath);
}

std::string metadataWriter(llvm::StringRef path, llvm::StringRef contents) {
  return "    set loom_metadata [open {" + path.str() +
         "} {WRONLY CREAT TRUNC}]\n"
         "    puts -nonewline $loom_metadata {" +
         contents.str() +
         "}\n"
         "    close $loom_metadata\n";
}

std::string renderDriver(llvm::StringRef providerBuild, llvm::StringRef device,
                         llvm::StringRef top,
                         llvm::ArrayRef<std::string> sources,
                         llvm::ArrayRef<std::string> constraints) {
  std::string driver = "package require ::quartus::project\n"
                       "package require ::quartus::flow\n\n"
                       "if {$argc != 1} { error {expected one exact action} }\n"
                       "set action [lindex $argv 0]\n"
                       "set project_name loom_quartus\n"
                       "set loom_device {" +
                       device.str() +
                       "}\n"
                       "if {[catch {get_part_info -family $loom_device}]} {\n"
                       "  error {exact device resource is unavailable}\n"
                       "}\n\n"
                       "proc run_module {tool} {\n"
                       "  if {[catch {execute_module -tool $tool} detail]} {\n"
                       "    error $detail\n"
                       "  }\n"
                       "}\n\n"
                       "if {$action eq {synthesis}} {\n"
                       "  project_new $project_name -overwrite\n"
                       "  set_global_assignment -name DEVICE $loom_device\n"
                       "  set_global_assignment -name TOP_LEVEL_ENTITY {" +
                       top.str() +
                       "}\n"
                       "  set_global_assignment -name PROJECT_OUTPUT_DIRECTORY "
                       "{quartus-work/output}\n";
  for (const std::string &source : sources)
    driver +=
        "  set_global_assignment -name SYSTEMVERILOG_FILE {" + source + "}\n";
  for (const std::string &constraint : constraints)
    driver += "  set_global_assignment -name SDC_FILE {" + constraint + "}\n";
  driver += "  run_module syn\n"
            "  project_close\n"
            "  exit 0\n"
            "}\n\n"
            "project_open $project_name\n"
            "if {[get_global_assignment -name DEVICE] ne $loom_device} {\n"
            "  error {current project has the wrong device resource}\n"
            "}\n"
            "if {[get_global_assignment -name TOP_LEVEL_ENTITY] ne {" +
            top.str() +
            "}} {\n"
            "  error {current project has the wrong top}\n"
            "}\n"
            "switch -- $action {\n"
            "  fitter {\n"
            "    run_module fit\n"
            "    project_archive " +
            kScratchArchivePath.str() +
            " -include_outputs -overwrite\n"
            "    if {![file isfile " +
            kScratchArchivePath.str() +
            "]} { error {physical database archive is absent} }\n"
            "    file copy -force " +
            kScratchArchivePath.str() + " " + kPhysicalPath.str() + "\n" +
            metadataWriter(kPhysicalMetadataPath,
                           physicalMetadata(providerBuild, device, top)) +
            "  }\n"
            "  sta { run_module sta }\n"
            "  assembler {\n"
            "    run_module asm\n"
            "    if {![file isfile " +
            kPhysicalPath.str() +
            "]} { error {physical database archive is absent} }\n"
            "    if {![file isfile " +
            kPhysicalMetadataPath.str() +
            "]} { error {physical metadata is absent} }\n"
            "    if {![file isfile " +
            kScratchImagePath.str() +
            "]} { error {device image is absent} }\n"
            "    file copy -force " +
            kScratchImagePath.str() + " " + kImagePath.str() + "\n" +
            metadataWriter(kImageMetadataPath,
                           imageMetadata(providerBuild, device, top)) +
            "  }\n"
            "  default { error {unknown exact action} }\n"
            "}\n"
            "project_close\n";
  return driver;
}

BlobDigest digest(llvm::StringRef contents) {
  return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(contents.data()),
      contents.size()));
}

struct InvocationInputs final {
  std::string providerBuild;
  std::string toolVersion;
  std::string device;
  std::string top;
  std::vector<std::string> rtlSources;
  std::vector<std::string> constraints;
  std::vector<MaterializedBundleFile> files;
  ExternalToolSemanticContract semanticContract;
};

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

const CandidateGeneratorDescriptor kDescriptor{
    quartusPrimeStaticFullDeviceCandidateGeneratorKind,
    "eda.intel_altera.quartus_prime_static_full_device",
    "loom.eda.intel_altera.quartus_prime_static_full_device.generator.v2",
    kInputSlots,
    kOutputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::IndependentReplicates,
    kWorkUnits,
    nullptr,
    ProviderForm::ExternalPrepareImport,
};

const CandidateGeneratorProvider kProvider{
    kDescriptor.reference(), CandidateGeneratorExternalPrepareImportProvider{
                                 prepareProvider, importProvider}};

llvm::Expected<InvocationInputs> collectInvocationInputs(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (binding.descriptorRef() != kDescriptor.reference())
    return invalid("binding does not select the Quartus generator");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          kDescriptor.reference(), inputBindings))
    return std::move(error);
  auto semanticContract =
      deriveExternalToolSemanticContract(inputBindings, binding);
  if (!semanticContract)
    return semanticContract.takeError();
  auto config = adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
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
    return unsupported<InvocationInputs>(
        QuartusPrimeUnsupportedReason::InputRepresentation,
        "input must be an exact SystemVerilog RTL HImpl2");
  if (!hardware.implementationPlatform() ||
      *hardware.implementationPlatform() != targetPlatform->reference())
    return unsupported<InvocationInputs>(
        QuartusPrimeUnsupportedReason::PlatformBinding,
        "RTL HImpl2 does not bind the supplied ImplementationPlatform");

  const auto *target =
      std::get_if<platform::FpgaTarget>(&targetPlatform->platform().target());
  if (!target || target->vendor != platform::FpgaVendor::IntelAltera)
    return unsupported<InvocationInputs>(
        QuartusPrimeUnsupportedReason::TargetVendor,
        "ImplementationPlatform is not an Intel/Altera FPGA");
  if (target->deviceOrderingCode != config->deviceResourceKey())
    return unsupported<InvocationInputs>(
        QuartusPrimeUnsupportedReason::DeviceResourceBinding,
        "device resource key does not equal the platform ordering code");
  if (!hardware.memoryMacroBindings().empty())
    return unsupported<InvocationInputs>(
        QuartusPrimeUnsupportedReason::MemoryMacroBinding,
        "RTL HImpl2 requires an unsupported memory macro binding");
  for (const ExternalImplementationBinding &external :
       hardware.externalImplementationBindings()) {
    auto physicalInputs = contracts.canonicalizeAndValidateInputs(
        external.providerContractRef, external.externalInputs,
        RepresentationRootVariant::FpgaPhysical);
    if (!physicalInputs)
      return unsupported<InvocationInputs>(
          QuartusPrimeUnsupportedReason::InputRepresentation,
          "external implementation cannot be retained in FpgaPhysical: " +
              llvm::toString(physicalInputs.takeError()));
    for (const ExternalInputBinding &input : external.externalInputs) {
      if (std::holds_alternative<ExplicitFileDependency>(
              input.dependencyIdentity))
        return unsupported<InvocationInputs>(
            QuartusPrimeUnsupportedReason::ExplicitFileDependency,
            "occurrence-scoped explicit-file projection is unavailable");
      const auto &resource =
          std::get<ToolBundledResourceDependency>(input.dependencyIdentity);
      if (resource.stableProviderBuildIdentity !=
          config->stableProviderBuildIdentity())
        return unsupported<InvocationInputs>(
            QuartusPrimeUnsupportedReason::ProviderResourceBinding,
            "RTL bundled resource belongs to another provider build");
    }
  }
  if (root.top.kind != RepresentationObjectKind::Module ||
      !isPortableIdentifier(root.top.canonicalName))
    return unsupported<InvocationInputs>(
        QuartusPrimeUnsupportedReason::TopModule,
        "top is not a portable SystemVerilog module identifier");

  InvocationInputs result;
  result.providerBuild = config->stableProviderBuildIdentity().str();
  result.toolVersion = config->verifiedToolVersion().str();
  result.device = target->deviceOrderingCode;
  result.top = root.top.canonicalName;
  const llvm::ArrayRef<std::uint8_t> platformBytes =
      targetPlatform->canonicalBytes().bytes();
  result.files.push_back(
      {kPlatformPath.str(),
       std::string(platformBytes.begin(), platformBytes.end()),
       targetPlatform->reference(), false});
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
      return unsupported<InvocationInputs>(
          QuartusPrimeUnsupportedReason::PayloadRole,
          "RTL payload closure contains an unsupported provider payload");
    }
    auto contents = blobs.get(payload.blobDigest);
    if (!contents)
      return contents.takeError();
    result.files.push_back({std::move(path),
                            std::string(contents->begin(), contents->end()),
                            implementation->reference(), false});
  }
  if (result.rtlSources.empty())
    return invalid("RTL HImpl2 contains no RTL source payload");
  result.semanticContract = std::move(*semanticContract);
  return result;
}

std::vector<std::string>
localInheritedEnvironment(const LocalToolConfig &config,
                          llvm::StringRef toolKey) {
  auto tool = config.tools.find(toolKey.str());
  return tool == config.tools.end() ? std::vector<std::string>{}
                                    : tool->second.inheritEnvironment;
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

  const std::filesystem::path bundle(context.bundleDestination);
  const std::filesystem::path probeDirectory = bundle.parent_path();
  const ExternalToolProviderDescriptor &quartus = quartusPrimeProvider();
  ShellToolBindingProbe probe(probeDirectory.string(), quartus.versionProbe);
  auto tool =
      resolveToolBinding(quartus.binding, context.localConfig,
                         captureToolEnvironment(quartus.binding), probe);
  if (!tool)
    return unavailable<PreparedExternalToolInvocation>(
        QuartusPrimeUnavailableReason::ToolResolution,
        llvm::toString(tool.takeError()));
  if (tool->version != inputs->toolVersion)
    return unavailable<PreparedExternalToolInvocation>(
        QuartusPrimeUnavailableReason::ProviderBuild,
        "resolved Quartus version does not match the semantic tool build");

  const std::vector<std::string> inherit =
      localInheritedEnvironment(context.localConfig, quartus.binding.key);
  const ExternalToolProviderDescriptor &container = polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeDirectory.string(),
                                       container.versionProbe);
  auto runtime = resolveInvocationRuntime(
      *tool, context.localConfig, container.binding,
      captureToolEnvironment(container.binding), containerProbe,
      quartus.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &resolvedContainer, llvm::StringRef os) {
        return probeContainerToolComposition(probeDirectory.string(),
                                             resolvedTool, quartus.versionProbe,
                                             resolvedContainer, os, inherit);
      });
  if (!runtime)
    return unavailable<PreparedExternalToolInvocation>(
        QuartusPrimeUnavailableReason::RuntimeResolution,
        llvm::toString(runtime.takeError()));

  ExternalToolInvocationBundleSpec specification;
  specification.semanticContract = std::move(inputs->semanticContract);
  specification.tool = std::move(*tool);
  specification.toolVersionProbe = quartus.versionProbe;
  specification.runtime = std::move(*runtime);
  specification.containerVersionProbe = container.versionProbe;
  for (llvm::StringRef action : {"synthesis", "fitter", "sta", "assembler"})
    specification.commands.push_back(
        {specification.tool.executable, "-t", kDriverPath.str(), action.str()});
  specification.inheritEnvironment = inherit;
  for (llvm::StringRef output : kDeclaredOutputPaths)
    specification.declaredOutputs.push_back(output.str());
  specification.files = std::move(inputs->files);
  specification.files.push_back(
      {kDriverPath.str(),
       renderDriver(inputs->providerBuild, inputs->device, inputs->top,
                    inputs->rtlSources, inputs->constraints),
       std::nullopt, false});
  return finalizeExternalToolInvocationBundle(context.bundleDestination,
                                              specification);
}

llvm::Error rejectUndeclaredOutputs(llvm::StringRef bundleRoot) {
  const std::filesystem::path outputs =
      std::filesystem::path(bundleRoot.str()) / "outputs";
  const std::set<std::string> allowed{"fpga-physical.qar", "fpga-physical.json",
                                      "device.sof",        "fpga-image.json",
                                      "completion.json",   "stdout.log",
                                      "stderr.log"};
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

CandidateGeneratorProviderResult
incompleteResult(CandidateGeneratorIncompleteReason reason) {
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
  expectation.semanticContract = inputs->semanticContract;
  for (const MaterializedBundleFile &file : inputs->files) {
    if (!file.sourceArtifact)
      continue;
    expectation.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact, digest(file.contents)});
  }
  for (llvm::StringRef output : kDeclaredOutputPaths)
    expectation.declaredOutputs.push_back(output.str());
  auto attempt = importExternalToolInvocationAttempt(prepared, expectation);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt)) {
    switch (failed->status) {
    case InvocationCompletionStatus::Success:
      return invalid("failed invocation outcome carries success status");
    case InvocationCompletionStatus::MissingEnvironment:
    case InvocationCompletionStatus::ModuleActivationFailed:
    case InvocationCompletionStatus::VersionMismatch:
      return incompleteResult(
          CandidateGeneratorIncompleteReason::ProviderUnavailable);
    case InvocationCompletionStatus::BundleContentMismatch:
      return invalid("invocation bundle content changed before execution");
    case InvocationCompletionStatus::ToolExit:
    case InvocationCompletionStatus::MissingOutput:
      return incompleteResult(
          CandidateGeneratorIncompleteReason::ExecutionFailed);
    }
  }
  auto imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  if (llvm::Error error = rejectUndeclaredOutputs(prepared.bundleRoot))
    return std::move(error);

  auto physical =
      readExternalToolInvocationDeclaredOutput(imported, kPhysicalPath);
  if (!physical)
    return physical.takeError();
  auto physicalFacts =
      readExternalToolInvocationDeclaredOutput(imported, kPhysicalMetadataPath);
  if (!physicalFacts)
    return physicalFacts.takeError();
  if (physical->empty())
    return invalid("physical database archive is empty");
  if (*physicalFacts !=
      physicalMetadata(inputs->providerBuild, inputs->device, inputs->top))
    return invalid("FpgaPhysical metadata is not exact");

  auto image = readExternalToolInvocationDeclaredOutput(imported, kImagePath);
  if (!image)
    return image.takeError();
  auto imageFacts =
      readExternalToolInvocationDeclaredOutput(imported, kImageMetadataPath);
  if (!imageFacts)
    return imageFacts.takeError();
  if (image->empty())
    return invalid("device image is empty");
  if (*imageFacts !=
      imageMetadata(inputs->providerBuild, inputs->device, inputs->top))
    return invalid("FpgaImage metadata is not exact");

  auto source = importHardwareImplementation(
      inputBindings[RtlImplementationInput].artifacts.front(), contracts,
      artifacts, blobs);
  if (!source)
    return source.takeError();
  auto publishedPhysical = publishRoutedFpgaPhysicalImplementation(
      *source, inputBindings[ImplementationPlatformInput].artifacts.front(),
      inputs->device, "database/quartus.qar", *physical, contracts, artifacts,
      blobs);
  if (!publishedPhysical)
    return publishedPhysical.takeError();
  auto publishedImage = publishFpgaImageImplementation(
      *publishedPhysical,
      inputBindings[ImplementationPlatformInput].artifacts.front(),
      inputs->device, "image/device.sof", *image, contracts, artifacts, blobs);
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

llvm::ArrayRef<std::uint8_t>
resolvedQuartusPrimeStaticFullDeviceConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedQuartusPrimeStaticFullDeviceConfigView>
projectResolvedQuartusPrimeStaticFullDeviceConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    llvm::StringRef verifiedToolVersion, llvm::StringRef deviceResourceKey) {
  if (llvm::Error error = validateConfigFields(
          stableProviderBuildIdentity, verifiedToolVersion, deviceResourceKey))
    return std::move(error);
  std::vector<std::uint8_t> canonical = encodeConfig(
      stableProviderBuildIdentity, verifiedToolVersion, deviceResourceKey);
  auto digest = computeComponentViewDigest(descriptorBytes(), canonical);
  if (!digest)
    return digest.takeError();
  return ResolvedQuartusPrimeStaticFullDeviceConfigView(
      stableProviderBuildIdentity.str(), verifiedToolVersion.str(),
      deviceResourceKey.str(), std::move(canonical), std::move(*digest));
}

llvm::Expected<ResolvedQuartusPrimeStaticFullDeviceConfigView>
adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
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
  return ResolvedQuartusPrimeStaticFullDeviceConfigView(
      std::move(decoded->providerBuild), std::move(decoded->toolVersion),
      std::move(decoded->device), canonicalViewBytes.vec(), digestValue);
}

const CandidateGeneratorDescriptor &
quartusPrimeStaticFullDeviceCandidateGeneratorDescriptor() {
  return kDescriptor;
}

llvm::Error registerQuartusPrimeStaticFullDeviceCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(kDescriptor))
    return error;
  return registerCandidateGeneratorProvider(kProvider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindQuartusPrimeStaticFullDeviceCandidateGeneratorInputs(
    const ArtifactRootReference &rtlImplementation,
    const ArtifactRootReference &implementationPlatform) {
  if (llvm::Error error =
          registerQuartusPrimeStaticFullDeviceCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings{
      {CandidateGeneratorInputSlotRef(RtlImplementationInput),
       {rtlImplementation}},
      {CandidateGeneratorInputSlotRef(ImplementationPlatformInput),
       {implementationPlatform}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          kDescriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(
    const ResolvedQuartusPrimeStaticFullDeviceConfigView &config) {
  if (llvm::Error error =
          registerQuartusPrimeStaticFullDeviceCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      kDescriptor.reference(), config.canonicalViewBytes(), config.digest());
}

llvm::Expected<PreparedExternalToolInvocation>
prepareQuartusPrimeStaticFullDeviceInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  return prepareProviderWithContracts(inputs, binding, contracts, artifacts,
                                      blobs, context);
}

llvm::Expected<CandidateGeneratorProviderResult>
importQuartusPrimeStaticFullDeviceInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const PreparedExternalToolInvocation &prepared,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importProviderWithContracts(inputs, binding, prepared, contracts,
                                     artifacts, blobs);
}

} // namespace loom::eda::intel_altera
