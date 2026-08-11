#include "EDA/Adapters/OpenSource/OpenRoad.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::eda::open_source {
namespace {

constexpr llvm::StringLiteral kConfigSchema = "loom.openroad_placed_config";
constexpr llvm::StringLiteral kConfigVersion = "1.0";
constexpr llvm::StringLiteral kResultSchema = "loom.openroad_physical_attempt";
constexpr llvm::StringLiteral kResultVersion = "1.0";
constexpr llvm::StringLiteral kProviderIdentity =
    "loom.openroad.placed_asic_physical.v1";
constexpr llvm::StringLiteral kDatabaseOutput = "outputs/placed.odb";
constexpr llvm::StringLiteral kResultOutput = "outputs/placed-result.json";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "openroad_placed_invalid: " + message);
}

bool isPortableIdentifier(llvm::StringRef value) {
  const auto isFirst = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto isRest = [&](char character) {
    return isFirst(character) || (character >= '0' && character <= '9');
  };
  return !value.empty() && isFirst(value.front()) &&
         llvm::all_of(value.drop_front(), isRest);
}

bool isCanonicalKey(llvm::StringRef value) {
  if (value.empty())
    return false;
  return llvm::all_of(value, [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.';
  });
}

bool isCanonicalProviderBuild(llvm::StringRef value) {
  return !value.empty() && value == value.trim() &&
         llvm::all_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return byte >= 0x20 && byte <= 0x7e;
         });
}

llvm::Expected<std::string> tclString(llvm::StringRef value,
                                      llvm::StringRef description) {
  if (value.empty() || value.contains('\0') || value.contains('\n') ||
      value.contains('\r'))
    return invalid(description + " is empty or contains a line separator");
  std::string result = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"' || character == '$' ||
        character == '[' || character == ']')
      result.push_back('\\');
    result.push_back(character);
  }
  result.push_back('"');
  return result;
}

llvm::Error validateRectangle(const OpenRoadRectangleNanometers &rectangle,
                              llvm::StringRef description) {
  if (rectangle.lowerXNanometers >= rectangle.upperXNanometers ||
      rectangle.lowerYNanometers >= rectangle.upperYNanometers)
    return invalid(description + " is empty or inverted");
  return llvm::Error::success();
}

bool contains(const OpenRoadRectangleNanometers &outer,
              const OpenRoadRectangleNanometers &inner) {
  return outer.lowerXNanometers <= inner.lowerXNanometers &&
         outer.lowerYNanometers <= inner.lowerYNanometers &&
         outer.upperXNanometers >= inner.upperXNanometers &&
         outer.upperYNanometers >= inner.upperYNanometers;
}

llvm::Error
validatePlacementParameters(const OpenRoadPlacementParameters &parameters) {
  if (llvm::Error error = validateRectangle(parameters.dieArea, "die area"))
    return error;
  if (llvm::Error error = validateRectangle(parameters.coreArea, "core area"))
    return error;
  if (!contains(parameters.dieArea, parameters.coreArea))
    return invalid("core area is not contained by die area");
  for (const auto &[value, description] :
       {std::pair<llvm::StringRef, llvm::StringRef>(parameters.siteName,
                                                    "site name"),
        {parameters.horizontalPinLayer, "horizontal pin layer"},
        {parameters.verticalPinLayer, "vertical pin layer"}})
    if (!isPortableIdentifier(value))
      return invalid(description + " is not a portable identifier");
  if (parameters.placementDensityPpm < 100000 ||
      parameters.placementDensityPpm > 990000)
    return invalid("placement density is outside [0.1, 0.99]");
  return llvm::Error::success();
}

llvm::StringRef externalKindSpelling(OpenRoadExternalFileKind kind) {
  switch (kind) {
  case OpenRoadExternalFileKind::TechnologyLef:
    return "technology_lef";
  case OpenRoadExternalFileKind::CellLef:
    return "cell_lef";
  case OpenRoadExternalFileKind::Liberty:
    return "liberty";
  }
  llvm_unreachable("validated OpenROAD external-file kind is closed");
}

llvm::Expected<OpenRoadExternalFileKind>
parseExternalKind(llvm::StringRef spelling) {
  if (spelling == "technology_lef")
    return OpenRoadExternalFileKind::TechnologyLef;
  if (spelling == "cell_lef")
    return OpenRoadExternalFileKind::CellLef;
  if (spelling == "liberty")
    return OpenRoadExternalFileKind::Liberty;
  return invalid("external file has an unknown kind");
}

llvm::Expected<OpenRoadPlacedConfig>
canonicalizeConfig(OpenRoadPlacedConfig config) {
  if (!isCanonicalProviderBuild(config.providerBuild))
    return invalid("provider build is not canonical printable ASCII");
  if (llvm::Error error = validatePlacementParameters(config.placement))
    return std::move(error);
  for (const OpenRoadExternalFile &file : config.externalFiles) {
    if (static_cast<std::uint8_t>(file.kind) >
        static_cast<std::uint8_t>(OpenRoadExternalFileKind::Liberty))
      return invalid("external file has an invalid kind");
    if (!isCanonicalKey(file.logicalName))
      return invalid("external file logical name is not a canonical key");
  }
  llvm::sort(config.externalFiles, [](const OpenRoadExternalFile &lhs,
                                      const OpenRoadExternalFile &rhs) {
    return std::tie(lhs.kind, lhs.logicalName) <
           std::tie(rhs.kind, rhs.logicalName);
  });
  for (std::size_t index = 1; index < config.externalFiles.size(); ++index)
    if (config.externalFiles[index - 1].kind ==
            config.externalFiles[index].kind &&
        config.externalFiles[index - 1].logicalName ==
            config.externalFiles[index].logicalName)
      return invalid("external file role and logical name are duplicated");

  std::array<std::size_t, 3> counts{};
  for (const OpenRoadExternalFile &file : config.externalFiles)
    ++counts[static_cast<std::size_t>(file.kind)];
  if (counts[static_cast<std::size_t>(
          OpenRoadExternalFileKind::TechnologyLef)] != 1)
    return invalid("config requires exactly one technology LEF");
  if (counts[static_cast<std::size_t>(OpenRoadExternalFileKind::CellLef)] == 0)
    return invalid("config requires at least one cell LEF");
  if (counts[static_cast<std::size_t>(OpenRoadExternalFileKind::Liberty)] == 0)
    return invalid("config requires at least one Liberty file");
  return config;
}

void writeRectangle(llvm::json::OStream &json,
                    const OpenRoadRectangleNanometers &rectangle) {
  json.object([&] {
    json.attribute("lower_x_nm", rectangle.lowerXNanometers);
    json.attribute("lower_y_nm", rectangle.lowerYNanometers);
    json.attribute("upper_x_nm", rectangle.upperXNanometers);
    json.attribute("upper_y_nm", rectangle.upperYNanometers);
  });
}

std::string serializeConfig(const OpenRoadPlacedConfig &config) {
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", kConfigSchema);
    json.attribute("version", kConfigVersion);
    json.attribute("provider_build", config.providerBuild);
    json.attributeObject("corner", [&] {
      json.attribute("platform_artifact",
                     formatArtifactIdentityHex(config.corner.artifact));
      json.attribute("corner_id", config.corner.entity.value());
    });
    json.attributeObject("placement", [&] {
      json.attributeBegin("die_area_nm");
      writeRectangle(json, config.placement.dieArea);
      json.attributeEnd();
      json.attributeBegin("core_area_nm");
      writeRectangle(json, config.placement.coreArea);
      json.attributeEnd();
      json.attribute("site", config.placement.siteName);
      json.attribute("horizontal_pin_layer",
                     config.placement.horizontalPinLayer);
      json.attribute("vertical_pin_layer", config.placement.verticalPinLayer);
      json.attribute("placement_density_ppm",
                     config.placement.placementDensityPpm);
    });
    json.attributeArray("external_files", [&] {
      for (const OpenRoadExternalFile &file : config.externalFiles) {
        json.object([&] {
          json.attribute("kind", externalKindSpelling(file.kind));
          json.attribute("logical_name", file.logicalName);
          json.attribute("content_sha256",
                         formatExternalFileFingerprint(file.fingerprint));
        });
      }
    });
  });
  return output.str().str();
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &[key, value] : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(key)))
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(key) + "'");
  return llvm::Error::success();
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(field);
  if (!value)
    return invalid(context + " requires object field '" + field + "'");
  return value;
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return invalid(context + " requires string field '" + field + "'");
  return *value;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  const std::optional<std::uint64_t> integer =
      value ? value->getAsUINT64() : std::nullopt;
  if (!integer)
    return invalid(context + " requires unsigned integer field '" + field +
                   "'");
  return *integer;
}

llvm::Expected<OpenRoadRectangleNanometers>
parseRectangle(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"lower_x_nm", "lower_y_nm", "upper_x_nm", "upper_y_nm"}))
    return std::move(error);
  auto lowerX = requireUnsigned(object, "lower_x_nm", context);
  auto lowerY = requireUnsigned(object, "lower_y_nm", context);
  auto upperX = requireUnsigned(object, "upper_x_nm", context);
  auto upperY = requireUnsigned(object, "upper_y_nm", context);
  if (!lowerX)
    return lowerX.takeError();
  if (!lowerY)
    return lowerY.takeError();
  if (!upperX)
    return upperX.takeError();
  if (!upperY)
    return upperY.takeError();
  return OpenRoadRectangleNanometers{*lowerX, *lowerY, *upperX, *upperY};
}

std::string decimalMicrons(std::uint64_t nanometers) {
  std::string result = std::to_string(nanometers / 1000);
  std::uint64_t fraction = nanometers % 1000;
  if (fraction == 0)
    return result;
  std::string digits = std::to_string(fraction + 1000).substr(1);
  while (digits.back() == '0')
    digits.pop_back();
  return result + "." + digits;
}

std::string density(std::uint32_t partsPerMillion) {
  std::string digits = std::to_string(partsPerMillion + 1000000).substr(1);
  while (digits.size() > 1 && digits.back() == '0')
    digits.pop_back();
  return "0." + digits;
}

std::string rectangle(const OpenRoadRectangleNanometers &value) {
  return decimalMicrons(value.lowerXNanometers) + " " +
         decimalMicrons(value.lowerYNanometers) + " " +
         decimalMicrons(value.upperXNanometers) + " " +
         decimalMicrons(value.upperYNanometers);
}

llvm::Expected<std::string> tclList(llvm::ArrayRef<std::string> values,
                                    llvm::StringRef description) {
  std::string result = "[list";
  for (const std::string &value : values) {
    auto encoded = tclString(value, description);
    if (!encoded)
      return encoded.takeError();
    result += " " + *encoded;
  }
  result += "]";
  return result;
}

std::string canonicalResult(llvm::StringRef topModule) {
  return "{\"schema\":\"" + kResultSchema.str() + "\",\"version\":\"" +
         kResultVersion.str() + "\",\"stage\":\"placed\",\"top\":\"" +
         topModule.str() + "\"}\n";
}

std::string externalSlot(const OpenRoadExternalFile &file) {
  switch (file.kind) {
  case OpenRoadExternalFileKind::TechnologyLef:
    return "technology_lef";
  case OpenRoadExternalFileKind::CellLef:
    return "cell_lef." + file.logicalName;
  case OpenRoadExternalFileKind::Liberty:
    return "liberty." + file.logicalName;
  }
  llvm_unreachable("validated OpenROAD external-file kind is closed");
}

std::vector<external_tool::ExternalFileRequirement>
externalRequirements(const OpenRoadPlacedConfig &config) {
  std::vector<external_tool::ExternalFileRequirement> requirements;
  requirements.reserve(config.externalFiles.size());
  for (const OpenRoadExternalFile &file : config.externalFiles)
    requirements.push_back({externalSlot(file), file.fingerprint});
  return requirements;
}

std::string asString(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::string indexedPath(llvm::StringRef directory, std::size_t ordinal,
                        llvm::StringRef extension) {
  std::string index = std::to_string(ordinal);
  if (index.size() < 4)
    index.insert(index.begin(), 4 - index.size(), '0');
  return (directory + "/" + index + extension).str();
}

struct OpenRoadInvocationData final {
  OpenRoadPlacedConfig config;
  std::string topModule;
  external_tool::ExternalToolSemanticContract semanticContract;
  std::vector<external_tool::MaterializedBundleFile> semanticFiles;
  std::vector<external_tool::ExternalFileRequirement> externalRequirements;
  OpenRoadPlacedDriverFiles driverFiles;
};

llvm::Expected<OpenRoadInvocationData> makeInvocationData(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const dse::CandidateGeneratorDescriptor &descriptor =
      openRoadPlacedCandidateGeneratorDescriptor();
  if (binding.descriptorRef() != descriptor.reference())
    return invalid("resolved binding references another generator");
  if (llvm::Error error = dse::validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputs))
    return std::move(error);
  auto config = decodeOpenRoadPlacedConfig(binding.canonicalConfigBytes());
  if (!config)
    return config.takeError();
  auto semanticContract =
      dse::deriveExternalToolSemanticContract(inputs, binding);
  if (!semanticContract)
    return semanticContract.takeError();

  const ArtifactRootReference &inputReference =
      inputs.front().artifacts.front();
  auto gate = hardware::importHardwareImplementation(inputReference, contracts,
                                                     artifacts, blobs);
  if (!gate)
    return gate.takeError();
  const hardware::ImplementationRepresentationRoot &root =
      gate->implementation().representationRoot();
  if (root.variant != hardware::RepresentationRootVariant::GateNetlist ||
      root.stage ||
      root.formatRef.kind() !=
          hardware::RepresentationFormatKind::StructuralVerilogGateNetlist)
    return invalid("input implementation is not an exact structural "
                   "GateNetlist");
  if (root.top.kind != hardware::RepresentationObjectKind::Module ||
      !isPortableIdentifier(root.top.canonicalName))
    return invalid("GateNetlist top is not a portable module identifier");
  if (!gate->implementation().implementationPlatform())
    return invalid("GateNetlist has no exact implementation platform");
  const ArtifactRootReference platformReference{
      platform::implementationPlatformSchema.identity.str(),
      platform::implementationPlatformSchema.version, config->corner.artifact};
  if (*gate->implementation().implementationPlatform() != platformReference)
    return invalid("config corner does not belong to the GateNetlist platform");
  auto implementationPlatform =
      platform::importImplementationPlatform(platformReference, artifacts);
  if (!implementationPlatform)
    return implementationPlatform.takeError();
  if (!std::holds_alternative<platform::AsicTarget>(
          implementationPlatform->platform().target()))
    return invalid("OpenROAD requires an ASIC implementation platform");
  auto corner = platform::resolveTechnologyCorner(config->corner, artifacts);
  if (!corner)
    return corner.takeError();

  std::set<ExternalFileFingerprint::Storage> liberties;
  for (const OpenRoadExternalFile &file : config->externalFiles)
    if (file.kind == OpenRoadExternalFileKind::Liberty)
      liberties.insert(file.fingerprint.bytes());
  for (const hardware::ExternalImplementationBinding &external :
       gate->implementation().externalImplementationBindings()) {
    for (const hardware::ExternalInputBinding &input :
         external.externalInputs) {
      const auto *explicitFile = std::get_if<hardware::ExplicitFileDependency>(
          &input.dependencyIdentity);
      if (!explicitFile)
        return invalid("GateNetlist uses a tool-bundled external dependency "
                       "that OpenROAD cannot consume exactly");
      if (!liberties.count(explicitFile->contentSha256.bytes()))
        return invalid("GateNetlist external dependency is absent from the "
                       "declared Liberty closure");
    }
  }

  OpenRoadInvocationData data{std::move(*config),
                              root.top.canonicalName,
                              std::move(*semanticContract),
                              {},
                              {},
                              {}};
  data.semanticFiles.push_back(external_tool::MaterializedBundleFile{
      "inputs/hardware-implementation.json",
      asString(gate->canonicalBytes().bytes()), inputReference, false});
  data.semanticFiles.push_back(external_tool::MaterializedBundleFile{
      "inputs/implementation-platform.json",
      asString(implementationPlatform->canonicalBytes().bytes()),
      platformReference, false});

  std::size_t netlistOrdinal = 0;
  std::size_t constraintOrdinal = 0;
  std::size_t contractOrdinal = 0;
  for (const hardware::ImplementationPayload &payload : root.payloads) {
    auto contents = blobs.get(payload.blobDigest);
    if (!contents)
      return contents.takeError();
    std::string path;
    switch (payload.role) {
    case hardware::PayloadRole::Netlist:
      path = indexedPath("inputs/netlist", netlistOrdinal++, ".v");
      data.driverFiles.netlists.push_back(path);
      break;
    case hardware::PayloadRole::GenerationConstraint:
      path = indexedPath("inputs/constraints", constraintOrdinal++, ".sdc");
      data.driverFiles.constraints.push_back(path);
      break;
    case hardware::PayloadRole::BlackBoxContract:
      path = indexedPath("inputs/contracts", contractOrdinal++, ".txt");
      break;
    default:
      return invalid("GateNetlist contains a payload outside its exact "
                     "OpenROAD input closure");
    }
    data.semanticFiles.push_back(external_tool::MaterializedBundleFile{
        std::move(path), asString(*contents), inputReference, false});
  }
  if (data.driverFiles.netlists.empty())
    return invalid("GateNetlist has no materialized netlist payload");
  data.externalRequirements = externalRequirements(data.config);
  return data;
}

external_tool::ExternalToolInvocationImportExpectation
importExpectation(const OpenRoadInvocationData &data) {
  external_tool::ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = data.semanticContract;
  for (const external_tool::MaterializedBundleFile &file : data.semanticFiles)
    expectation.semanticInputs.push_back(
        external_tool::ExternalToolInvocationSemanticInput{
            file.relativePath, *file.sourceArtifact,
            computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
                reinterpret_cast<const std::uint8_t *>(file.contents.data()),
                file.contents.size()))});
  for (const external_tool::ExternalFileRequirement &file :
       data.externalRequirements)
    expectation.externalInputs.push_back(
        {file.providerInputSlot, file.fingerprint});
  expectation.declaredOutputs = {kDatabaseOutput.str(), kResultOutput.str()};
  return expectation;
}

hardware::RepresentationLocator
projectPhysicalLocator(const hardware::RepresentationLocator &locator,
                       const hardware::RepresentationLocator &sourceTop,
                       const hardware::RepresentationLocator &physicalTop) {
  return locator == sourceTop ? physicalTop : locator;
}

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishPlacedImplementation(
    const OpenRoadInvocationData &data,
    const ArtifactRootReference &inputReference, llvm::StringRef database,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto source = hardware::importHardwareImplementation(
      inputReference, contracts, artifacts, blobs);
  if (!source)
    return source.takeError();
  const hardware::ImplementationRepresentationRoot &sourceRoot =
      source->implementation().representationRoot();
  auto sourceIndex = hardware::indexRepresentationRoot(sourceRoot, blobs);
  if (!sourceIndex)
    return sourceIndex.takeError();

  auto format = hardware::RepresentationFormatDescriptorRef::get(
      hardware::RepresentationFormatKind::IndexedPhysical);
  if (!format)
    return format.takeError();
  const hardware::RepresentationLocator physicalTop{
      hardware::RepresentationObjectKind::PhysicalObject, data.topModule};

  std::vector<hardware::PhysicalRepresentationObject> objects{
      {physicalTop, std::nullopt}};
  std::vector<hardware::RepresentationLocator> unresolved;
  const auto addReferencedObject =
      [&](const hardware::RepresentationLocator &sourceLocator) -> llvm::Error {
    const hardware::RepresentationLocator locator =
        projectPhysicalLocator(sourceLocator, sourceRoot.top, physicalTop);
    if (llvm::any_of(objects, [&](const auto &object) {
          return object.locator == locator;
        }))
      return llvm::Error::success();
    auto facts = sourceIndex->lookup(sourceLocator);
    if (!facts)
      return facts.takeError();
    if (!*facts)
      return invalid("referenced GateNetlist object is absent from its exact "
                     "representation index");
    objects.push_back({locator, (*facts)->signalGeometry});
    if (locator.kind == hardware::RepresentationObjectKind::Module)
      unresolved.push_back(locator);
    return llvm::Error::success();
  };

  std::vector<hardware::ImplementationInterface> interfaces(
      source->implementation().interfaces().begin(),
      source->implementation().interfaces().end());
  for (hardware::ImplementationInterface &interface : interfaces) {
    if (llvm::Error error =
            addReferencedObject(interface.representationLocator))
      return std::move(error);
    interface.representationLocator = projectPhysicalLocator(
        interface.representationLocator, sourceRoot.top, physicalTop);
  }

  std::vector<hardware::ActivityPoint> activityPoints(
      source->implementation().activityPoints().begin(),
      source->implementation().activityPoints().end());
  for (hardware::ActivityPoint &point : activityPoints) {
    if (llvm::Error error = addReferencedObject(point.representationLocator))
      return std::move(error);
    point.representationLocator = projectPhysicalLocator(
        point.representationLocator, sourceRoot.top, physicalTop);
  }

  std::vector<hardware::ExternalImplementationBindingDraft> externalBindings;
  for (const hardware::ExternalImplementationBinding &binding :
       source->implementation().externalImplementationBindings()) {
    std::vector<hardware::RepresentationLocator> locators(
        binding.representationLocators.begin(),
        binding.representationLocators.end());
    for (hardware::RepresentationLocator &locator : locators) {
      if (llvm::Error error = addReferencedObject(locator))
        return std::move(error);
      locator = projectPhysicalLocator(locator, sourceRoot.top, physicalTop);
    }
    std::optional<hardware::ImplementationPayloadKey> blackBoxContract;
    if (binding.blackBoxContractPayloadRef) {
      if (binding.blackBoxContractPayloadRef->ordinal >=
          sourceRoot.payloads.size())
        return invalid("source black-box payload reference is out of range");
      const hardware::ImplementationPayload &payload =
          sourceRoot.payloads[binding.blackBoxContractPayloadRef->ordinal];
      blackBoxContract = hardware::ImplementationPayloadKey{
          payload.role, payload.canonicalLogicalName};
    }
    externalBindings.push_back(hardware::ExternalImplementationBindingDraft{
        binding.providerContractRef, binding.externalInputs,
        binding.fabricResourceRefs, std::move(locators),
        std::move(blackBoxContract)});
  }

  std::vector<hardware::MemoryMacroBindingDraft> memoryBindings;
  for (const hardware::MemoryMacroBinding &binding :
       source->implementation().memoryMacroBindings()) {
    if (binding.externalImplementationBindingRef.ordinal >=
        externalBindings.size())
      return invalid("source memory binding references an unknown external "
                     "implementation");
    if (llvm::Error error = addReferencedObject(binding.representationLocator))
      return std::move(error);
    memoryBindings.push_back(hardware::MemoryMacroBindingDraft{
        binding.fabricMemoryRef,
        binding.externalImplementationBindingRef.ordinal,
        projectPhysicalLocator(binding.representationLocator, sourceRoot.top,
                               physicalTop)});
  }

  for (const hardware::RepresentationLocator &locator :
       sourceIndex->unresolvedExternalDefinitions())
    if (llvm::Error error = addReferencedObject(locator))
      return std::move(error);

  std::vector<hardware::ImplementationPayload> payloads;
  for (const hardware::ImplementationPayload &payload : sourceRoot.payloads) {
    if (payload.role == hardware::PayloadRole::GenerationConstraint ||
        payload.role == hardware::PayloadRole::BlackBoxContract)
      payloads.push_back(payload);
  }
  auto databaseDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(database.data()),
      database.size()));
  if (!databaseDigest)
    return databaseDigest.takeError();
  payloads.push_back({hardware::PayloadRole::PhysicalDatabase,
                      "database/openroad.odb", *databaseDigest});

  auto index = hardware::createPhysicalRepresentationIndexPayload(
      *format, hardware::RepresentationRootVariant::AsicPhysical,
      hardware::RepresentationPhysicalStage::Placed, physicalTop,
      "index/physical.json", payloads, std::move(objects),
      std::move(unresolved));
  if (!index)
    return index.takeError();
  auto indexBytes =
      hardware::serializePhysicalRepresentationIndexPayloadJson(*index);
  if (!indexBytes)
    return indexBytes.takeError();
  auto indexDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(indexBytes->data()),
      indexBytes->size()));
  if (!indexDigest)
    return indexDigest.takeError();
  payloads.push_back({hardware::PayloadRole::RepresentationIndex,
                      index->indexLogicalName, *indexDigest});
  auto representation = hardware::createImplementationRepresentationRoot(
      hardware::RepresentationRootVariant::AsicPhysical,
      hardware::RepresentationPhysicalStage::Placed, *format, physicalTop,
      std::move(payloads));
  if (!representation)
    return representation.takeError();

  return hardware::finalizeHardwareImplementation(
      hardware::HardwareImplementationDraft{
          source->implementation().fabric(),
          source->implementation().configurationAbi(),
          source->implementation().interconnectImplementations().vec(),
          std::move(*representation),
          source->implementation().implementationPlatform(),
          std::move(interfaces), std::move(activityPoints),
          std::move(memoryBindings), std::move(externalBindings)},
      contracts, artifacts, blobs);
}

dse::CandidateGeneratorProviderResult
incompleteResult(dse::CandidateGeneratorIncompleteReason reason) {
  return dse::CandidateGeneratorProviderResult{
      dse::IncompleteCandidateGeneratorResult{
          reason, {{dse::CandidateGeneratorOutputSlotRef(0), {}}}, {}},
      {{dse::CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

llvm::Error validateExecution(const OpenRoadResolvedExecution &execution,
                              llvm::StringRef providerBuild) {
  const auto &provider = external_tool::openRoadProvider();
  if (execution.provider.binding.key != provider.binding.key ||
      execution.tool.toolKey != provider.binding.key)
    return invalid("resolved execution does not bind the OpenROAD provider");
  if (execution.tool.version != providerBuild)
    return invalid("resolved OpenROAD build does not match candidate config");
  if (execution.provider.versionProbe.requiredOutputSubstring &&
      !llvm::StringRef(execution.tool.version)
           .contains(*execution.provider.versionProbe.requiredOutputSubstring))
    return invalid("resolved OpenROAD version does not satisfy the provider");
  if (execution.runtime.kind ==
          external_tool::InvocationRuntimeKind::PolyArchContainer &&
      !execution.provider.runtimeCompatibility.supportsPolyArchContainer)
    return invalid("OpenROAD provider does not support the resolved runtime");
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<std::uint8_t> openRoadPlacedConfigSchemaDescriptorBytes() {
  static const std::string bytes = (kConfigSchema + ":" + kConfigVersion).str();
  return llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
}

llvm::Error validateOpenRoadPlacementParameters(
    const OpenRoadPlacementParameters &parameters) {
  return validatePlacementParameters(parameters);
}

std::string openRoadExternalFileInputSlot(const OpenRoadExternalFile &file) {
  return externalSlot(file);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeOpenRoadPlacedConfig(const OpenRoadPlacedConfig &config) {
  auto canonical = canonicalizeConfig(config);
  if (!canonical)
    return canonical.takeError();
  const std::string text = serializeConfig(*canonical);
  return std::vector<std::uint8_t>(text.begin(), text.end());
}

llvm::Expected<OpenRoadPlacedConfig>
decodeOpenRoadPlacedConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  const llvm::StringRef contents(reinterpret_cast<const char *>(bytes.data()),
                                 bytes.size());
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("config JSON is malformed: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("config JSON root is not an object");
  if (llvm::Error error =
          rejectUnknownFields(*root, "config JSON",
                              {"schema", "version", "provider_build", "corner",
                               "placement", "external_files"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "config JSON");
  auto version = requireString(*root, "version", "config JSON");
  auto providerBuild = requireString(*root, "provider_build", "config JSON");
  if (!schema)
    return schema.takeError();
  if (!version)
    return version.takeError();
  if (!providerBuild)
    return providerBuild.takeError();
  if (*schema != kConfigSchema || *version != kConfigVersion)
    return invalid("config schema or version is unsupported");

  auto cornerObject = requireObject(*root, "corner", "config JSON");
  if (!cornerObject)
    return cornerObject.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **cornerObject, "config corner", {"platform_artifact", "corner_id"}))
    return std::move(error);
  auto platformArtifact =
      requireString(**cornerObject, "platform_artifact", "config corner");
  auto cornerId = requireUnsigned(**cornerObject, "corner_id", "config corner");
  if (!platformArtifact)
    return platformArtifact.takeError();
  if (!cornerId)
    return cornerId.takeError();
  auto artifact = parseArtifactIdentityHex(*platformArtifact);
  if (!artifact)
    return artifact.takeError();

  auto placementObject = requireObject(*root, "placement", "config JSON");
  if (!placementObject)
    return placementObject.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **placementObject, "config placement",
          {"die_area_nm", "core_area_nm", "site", "horizontal_pin_layer",
           "vertical_pin_layer", "placement_density_ppm"}))
    return std::move(error);
  auto dieObject =
      requireObject(**placementObject, "die_area_nm", "config placement");
  auto coreObject =
      requireObject(**placementObject, "core_area_nm", "config placement");
  if (!dieObject)
    return dieObject.takeError();
  if (!coreObject)
    return coreObject.takeError();
  auto die = parseRectangle(**dieObject, "config die area");
  auto core = parseRectangle(**coreObject, "config core area");
  auto site = requireString(**placementObject, "site", "config placement");
  auto horizontal = requireString(**placementObject, "horizontal_pin_layer",
                                  "config placement");
  auto vertical = requireString(**placementObject, "vertical_pin_layer",
                                "config placement");
  auto densityValue = requireUnsigned(
      **placementObject, "placement_density_ppm", "config placement");
  if (!die)
    return die.takeError();
  if (!core)
    return core.takeError();
  if (!site)
    return site.takeError();
  if (!horizontal)
    return horizontal.takeError();
  if (!vertical)
    return vertical.takeError();
  if (!densityValue)
    return densityValue.takeError();
  if (*densityValue > std::numeric_limits<std::uint32_t>::max())
    return invalid("placement density is outside uint32 range");

  const llvm::json::Array *externalArray = root->getArray("external_files");
  if (!externalArray)
    return invalid("config JSON requires array field 'external_files'");
  std::vector<OpenRoadExternalFile> externalFiles;
  externalFiles.reserve(externalArray->size());
  for (const llvm::json::Value &value : *externalArray) {
    const llvm::json::Object *file = value.getAsObject();
    if (!file)
      return invalid("config external file is not an object");
    if (llvm::Error error =
            rejectUnknownFields(*file, "config external file",
                                {"kind", "logical_name", "content_sha256"}))
      return std::move(error);
    auto kindText = requireString(*file, "kind", "config external file");
    auto logicalName =
        requireString(*file, "logical_name", "config external file");
    auto digestText =
        requireString(*file, "content_sha256", "config external file");
    if (!kindText)
      return kindText.takeError();
    if (!logicalName)
      return logicalName.takeError();
    if (!digestText)
      return digestText.takeError();
    auto kind = parseExternalKind(*kindText);
    auto digest = parseExternalFileFingerprint(*digestText);
    if (!kind)
      return kind.takeError();
    if (!digest)
      return digest.takeError();
    externalFiles.push_back(
        OpenRoadExternalFile{*kind, logicalName->str(), std::move(*digest)});
  }

  auto canonical = canonicalizeConfig(OpenRoadPlacedConfig{
      providerBuild->str(),
      platform::TechnologyCornerRef{std::move(*artifact),
                                    platform::TechnologyCornerId(*cornerId)},
      OpenRoadPlacementParameters{*die, *core, site->str(), horizontal->str(),
                                  vertical->str(),
                                  static_cast<std::uint32_t>(*densityValue)},
      std::move(externalFiles)});
  if (!canonical)
    return canonical.takeError();
  if (contents != serializeConfig(*canonical))
    return invalid("config JSON is not canonical");
  return std::move(*canonical);
}

llvm::Error validateCanonicalOpenRoadPlacedConfig(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ComponentViewDigest &suppliedDigest) {
  auto config = decodeOpenRoadPlacedConfig(bytes);
  if (!config)
    return config.takeError();
  return validateComponentViewDigest(
      openRoadPlacedConfigSchemaDescriptorBytes(), bytes, suppliedDigest);
}

const dse::CandidateGeneratorDescriptor &
openRoadPlacedCandidateGeneratorDescriptor() {
  static const std::array<dse::CandidateGeneratorInputSlotDescriptor, 1> inputs{
      {{dse::CandidateGeneratorInputSlotRef(0), "gate_netlist_implementation",
        dse::PlanValueRole::CandidateSet,
        &hardware::hardwareImplementationSchema,
        dse::PlanValueCardinality::ExactlyOne}}};
  static const std::array<dse::CandidateGeneratorOutputSlotDescriptor, 1>
      outputs{{{dse::CandidateGeneratorOutputSlotRef(0),
                "placed_asic_physical_implementation",
                dse::PlanValueRole::CandidateSet,
                &hardware::hardwareImplementationSchema,
                dse::PlanValueCardinality::ExactlyOne}}};
  static const std::array<dse::CandidateGeneratorWorkUnitDescriptor, 1>
      workUnits{
          {{dse::CandidateGeneratorWorkUnitRef(0), "openroad_placed_attempt"}}};
  static const dse::CandidateGeneratorDescriptor descriptor{
      openRoadPlacedCandidateGeneratorKind,
      "openroad.placed_asic_physical",
      kProviderIdentity,
      inputs,
      outputs,
      dse::ResolvedDseConfigViewContract{
          openRoadPlacedConfigSchemaDescriptorBytes(),
          validateCanonicalOpenRoadPlacedConfig},
      dse::CandidateGeneratorDeterminism::Deterministic,
      workUnits,
      nullptr,
      ProviderForm::ExternalPrepareImport};
  return descriptor;
}

llvm::Error registerOpenRoadPlacedCandidateGeneratorDescriptor() {
  return dse::registerCandidateGeneratorDescriptor(
      openRoadPlacedCandidateGeneratorDescriptor());
}

namespace {

llvm::Expected<OpenRoadResolvedExecution> resolveOpenRoadExecutionImpl(
    llvm::StringRef providerBuild,
    const external_tool::ExternalToolPreparationContext &context) {
  const external_tool::ExternalToolProviderDescriptor &toolProvider =
      external_tool::openRoadProvider();
  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  external_tool::ShellToolBindingProbe toolProbe(probeRoot.string(),
                                                 toolProvider.versionProbe);
  const external_tool::ToolEnvironment toolEnvironment =
      external_tool::captureToolEnvironment(toolProvider.binding);
  auto tool = external_tool::resolveToolBinding(
      toolProvider.binding, context.localConfig, toolEnvironment, toolProbe);
  if (!tool)
    return tool.takeError();
  if (tool->version != providerBuild)
    return invalid(llvm::Twine("resolved OpenROAD build '") + tool->version +
                   "' does not match semantic build '" + providerBuild + "'");

  std::vector<std::string> inheritEnvironment;
  const auto configured =
      context.localConfig.tools.find(toolProvider.binding.key);
  if (configured != context.localConfig.tools.end())
    inheritEnvironment = configured->second.inheritEnvironment;

  const external_tool::ExternalToolProviderDescriptor &containerProvider =
      external_tool::polyArchContainerProvider();
  external_tool::ShellToolBindingProbe containerProbe(
      probeRoot.string(), containerProvider.versionProbe);
  const external_tool::ToolEnvironment containerEnvironment =
      external_tool::captureToolEnvironment(containerProvider.binding);
  auto runtime = external_tool::resolveInvocationRuntime(
      *tool, context.localConfig, containerProvider.binding,
      containerEnvironment, containerProbe, toolProvider.runtimeCompatibility,
      [&](const external_tool::ResolvedToolBinding &resolvedTool,
          const external_tool::ResolvedToolBinding &container,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return external_tool::probeContainerToolComposition(
            probeRoot.string(), resolvedTool, toolProvider.versionProbe,
            container, os, inheritEnvironment);
      });
  if (!runtime)
    return runtime.takeError();

  return OpenRoadResolvedExecution{toolProvider, std::move(*tool),
                                   std::move(*runtime),
                                   containerProvider.versionProbe};
}

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareRegisteredOpenRoad(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context) {
  auto config = decodeOpenRoadPlacedConfig(binding.canonicalConfigBytes());
  if (!config)
    return config.takeError();
  auto execution = resolveOpenRoadExecutionImpl(config->providerBuild, context);
  if (!execution)
    return execution.takeError();
  auto contracts = makeYosysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return prepareOpenRoadPlacedInvocation(inputs, binding, *contracts, artifacts,
                                         blobs, *execution, context);
}

llvm::Expected<dse::CandidateGeneratorProviderResult> importRegisteredOpenRoad(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto contracts = makeYosysStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return importOpenRoadPlacedInvocation(inputs, binding, prepared, *contracts,
                                        artifacts, blobs);
}

} // namespace

llvm::Expected<OpenRoadResolvedExecution> resolveOpenRoadExecution(
    llvm::StringRef providerBuild,
    const external_tool::ExternalToolPreparationContext &context) {
  return resolveOpenRoadExecutionImpl(providerBuild, context);
}

llvm::Error
validateOpenRoadResolvedExecution(const OpenRoadResolvedExecution &execution,
                                  llvm::StringRef providerBuild) {
  return validateExecution(execution, providerBuild);
}

llvm::Error registerOpenRoadPlacedCandidateGenerator() {
  const dse::CandidateGeneratorDescriptor &descriptor =
      openRoadPlacedCandidateGeneratorDescriptor();
  static const dse::CandidateGeneratorProvider provider{
      descriptor.reference(),
      dse::CandidateGeneratorExternalPrepareImportProvider{
          prepareRegisteredOpenRoad, importRegisteredOpenRoad}};
  if (llvm::Error error = dse::registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return dse::registerCandidateGeneratorProvider(provider);
}

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareOpenRoadPlacedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const OpenRoadResolvedExecution &execution,
    const external_tool::ExternalToolPreparationContext &context) {
  auto data = makeInvocationData(inputs, binding, contracts, artifacts, blobs);
  if (!data)
    return data.takeError();
  if (llvm::Error error =
          validateExecution(execution, data->config.providerBuild))
    return std::move(error);
  auto externalFiles = external_tool::resolveExternalFiles(
      data->externalRequirements, context.localConfig);
  if (!externalFiles)
    return externalFiles.takeError();
  if (externalFiles->size() != data->config.externalFiles.size())
    return invalid("resolved external-file closure has the wrong size");
  for (std::size_t index = 0; index < externalFiles->size(); ++index) {
    const OpenRoadExternalFile &semantic = data->config.externalFiles[index];
    const external_tool::ResolvedExternalFile &resolved =
        (*externalFiles)[index];
    if (resolved.providerInputSlot != externalSlot(semantic) ||
        resolved.fingerprint != semantic.fingerprint)
      return invalid("resolved external-file closure does not match config");
    switch (semantic.kind) {
    case OpenRoadExternalFileKind::TechnologyLef:
      data->driverFiles.technologyLef = resolved.absolutePath;
      break;
    case OpenRoadExternalFileKind::CellLef:
      data->driverFiles.cellLefs.push_back(resolved.absolutePath);
      break;
    case OpenRoadExternalFileKind::Liberty:
      data->driverFiles.libertyFiles.push_back(resolved.absolutePath);
      break;
    }
  }
  auto driver = renderOpenRoadPlacedDriver(
      data->topModule, data->config.placement, data->driverFiles);
  if (!driver)
    return driver.takeError();

  external_tool::ExternalToolInvocationBundleSpec specification;
  specification.semanticContract = std::move(data->semanticContract);
  specification.tool = execution.tool;
  specification.toolVersionProbe = execution.provider.versionProbe;
  specification.runtime = execution.runtime;
  specification.containerVersionProbe = execution.containerVersionProbe;
  specification.commands = {{execution.tool.executable, "-no_init",
                             "-no_splash", "-no_settings", "-threads", "1",
                             "-exit", "drivers/openroad.tcl"}};
  specification.declaredOutputs = {kDatabaseOutput.str(), kResultOutput.str()};
  specification.files.push_back(external_tool::MaterializedBundleFile{
      "drivers/openroad.tcl", std::move(*driver), std::nullopt, false});
  specification.files.insert(
      specification.files.end(),
      std::make_move_iterator(data->semanticFiles.begin()),
      std::make_move_iterator(data->semanticFiles.end()));
  specification.externalFiles = std::move(*externalFiles);
  return external_tool::finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
}

llvm::Expected<dse::CandidateGeneratorProviderResult>
importOpenRoadPlacedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto data = makeInvocationData(inputs, binding, contracts, artifacts, blobs);
  if (!data)
    return data.takeError();
  auto attempt = external_tool::importExternalToolInvocationAttempt(
      prepared, importExpectation(*data));
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<
          external_tool::IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<
        external_tool::IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<external_tool::FailedExternalToolInvocationAttempt>(
              &*attempt)) {
    using Status = external_tool::InvocationCompletionStatus;
    switch (failed->status) {
    case Status::Success:
      return invalid("failed invocation outcome carries success status");
    case Status::MissingEnvironment:
    case Status::ModuleActivationFailed:
    case Status::VersionMismatch:
      return incompleteResult(
          dse::CandidateGeneratorIncompleteReason::ProviderUnavailable);
    case Status::BundleContentMismatch:
      return invalid("invocation bundle content changed before execution");
    case Status::ToolExit:
    case Status::MissingOutput:
      return incompleteResult(
          dse::CandidateGeneratorIncompleteReason::ExecutionFailed);
    }
  }
  auto imported = std::get<external_tool::ImportedExternalToolInvocationBundle>(
      std::move(*attempt));
  auto database = external_tool::readExternalToolInvocationDeclaredOutput(
      imported, kDatabaseOutput);
  if (!database)
    return database.takeError();
  if (database->empty())
    return invalid("placed database output is empty");
  auto result = external_tool::readExternalToolInvocationDeclaredOutput(
      imported, kResultOutput);
  if (!result)
    return result.takeError();
  auto parsed = parseOpenRoadPlacedAttemptResult(*result);
  if (!parsed)
    return parsed.takeError();
  if (parsed->topModule != data->topModule)
    return invalid("placed result top does not match the exact GateNetlist");

  auto placed =
      publishPlacedImplementation(*data, inputs.front().artifacts.front(),
                                  *database, contracts, artifacts, blobs);
  if (!placed)
    return placed.takeError();

  return dse::CandidateGeneratorProviderResult{
      dse::CompletedCandidateGeneratorResult{
          {{dse::CandidateGeneratorOutputSlotRef(0), {placed->reference()}}},
          {{dse::CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            dse::CandidateGeneratorOutputSlotRef(0),
            placed->reference(),
            {},
            {}}}},
      {{dse::CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

llvm::Expected<std::string>
renderOpenRoadPlacedDriver(llvm::StringRef topModule,
                           const OpenRoadPlacementParameters &parameters,
                           const OpenRoadPlacedDriverFiles &files) {
  if (!isPortableIdentifier(topModule))
    return invalid("top module is not a portable identifier");
  if (llvm::Error error = validatePlacementParameters(parameters))
    return std::move(error);
  if (files.netlists.empty())
    return invalid("netlist closure is empty");
  if (files.cellLefs.empty())
    return invalid("cell LEF closure is empty");
  if (files.libertyFiles.empty())
    return invalid("Liberty closure is empty");

  auto technologyLef = tclString(files.technologyLef, "technology LEF path");
  if (!technologyLef)
    return technologyLef.takeError();
  auto cellLefs = tclList(files.cellLefs, "cell LEF path");
  if (!cellLefs)
    return cellLefs.takeError();
  auto libertyFiles = tclList(files.libertyFiles, "Liberty path");
  if (!libertyFiles)
    return libertyFiles.takeError();
  auto netlists = tclList(files.netlists, "netlist path");
  if (!netlists)
    return netlists.takeError();
  auto constraints = tclList(files.constraints, "constraint path");
  if (!constraints)
    return constraints.takeError();

  std::string driver;
  driver += "set loom_technology_lef " + *technologyLef + "\n";
  driver += "set loom_cell_lefs " + *cellLefs + "\n";
  driver += "set loom_liberty_files " + *libertyFiles + "\n";
  driver += "set loom_netlists " + *netlists + "\n";
  driver += "set loom_constraints " + *constraints + "\n";
  driver += "read_lef $loom_technology_lef\n";
  driver += "foreach loom_path $loom_cell_lefs { read_lef $loom_path }\n";
  driver +=
      "foreach loom_path $loom_liberty_files { read_liberty $loom_path }\n";
  driver += "foreach loom_path $loom_netlists { read_verilog $loom_path }\n";
  driver += "link_design " + topModule.str() + "\n";
  driver += "foreach loom_path $loom_constraints { read_sdc $loom_path }\n";
  driver += "initialize_floorplan -die_area {" + rectangle(parameters.dieArea) +
            "} -core_area {" + rectangle(parameters.coreArea) + "} -site " +
            parameters.siteName + "\n";
  driver += "make_tracks\n";
  driver += "place_pins -hor_layers " + parameters.horizontalPinLayer +
            " -ver_layers " + parameters.verticalPinLayer + "\n";
  driver += "global_placement -density " +
            density(parameters.placementDensityPpm) + " -random_seed 1\n";
  driver += "detailed_placement\n";
  driver += "check_placement -verbose\n";
  driver += "write_db outputs/placed.odb\n";
  driver += "set loom_result [open outputs/placed-result.json w]\n";
  std::string resultMarker = canonicalResult(topModule);
  resultMarker.pop_back();
  driver += "puts $loom_result {" + resultMarker + "}\n";
  driver += "close $loom_result\n";
  return driver;
}

llvm::Expected<OpenRoadPlacedAttemptResult>
parseOpenRoadPlacedAttemptResult(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("result JSON is malformed: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 4)
    return invalid("result JSON has an invalid shape");
  const std::optional<llvm::StringRef> top = object->getString("top");
  if (!top || !isPortableIdentifier(*top))
    return invalid("result JSON fields are invalid");
  if (contents != canonicalResult(*top))
    return invalid("result JSON is not canonical");
  return OpenRoadPlacedAttemptResult{top->str()};
}

} // namespace loom::eda::open_source
