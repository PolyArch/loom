#include "EDA/Adapters/OpenSource/OpenRoadRouted.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "ExternalTool/ExternalFile.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <filesystem>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::open_source {
namespace {

constexpr llvm::StringLiteral kProviderIdentity =
    "loom.openroad.routed_asic_physical.v1";
constexpr llvm::StringLiteral kNetlistOutput = "outputs/routed.v";
constexpr llvm::StringLiteral kDefOutput = "outputs/routed.def";
constexpr llvm::StringLiteral kResultOutput = "outputs/routed-result.json";
constexpr llvm::StringLiteral kResultSchema =
    "loom.openroad_routed_physical_attempt";
constexpr llvm::StringLiteral kResultVersion = "1.0";
constexpr llvm::StringLiteral kBlackBoxLogicalName =
    "contracts/openroad-routed-standard-cells.txt";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "openroad_routed_invalid: " + detail);
}

bool isPortableIdentifier(llvm::StringRef value) {
  const auto first = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto rest = [&](char character) {
    return first(character) || (character >= '0' && character <= '9');
  };
  return !value.empty() && first(value.front()) &&
         llvm::all_of(value.drop_front(), rest);
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

std::string decimalMicrons(std::uint64_t nanometers) {
  std::string result = std::to_string(nanometers / 1000);
  const std::uint64_t fraction = nanometers % 1000;
  if (fraction == 0)
    return result;
  std::string digits = std::to_string(fraction + 1000).substr(1);
  while (digits.back() == '0')
    digits.pop_back();
  return result + "." + digits;
}

std::string rectangle(const OpenRoadRectangleNanometers &value) {
  return decimalMicrons(value.lowerXNanometers) + " " +
         decimalMicrons(value.lowerYNanometers) + " " +
         decimalMicrons(value.upperXNanometers) + " " +
         decimalMicrons(value.upperYNanometers);
}

std::string density(std::uint32_t partsPerMillion) {
  std::string digits = std::to_string(partsPerMillion + 1000000).substr(1);
  while (digits.size() > 1 && digits.back() == '0')
    digits.pop_back();
  return "0." + digits;
}

std::string indexedPath(llvm::StringRef directory, std::size_t ordinal,
                        llvm::StringRef extension) {
  std::string index = std::to_string(ordinal);
  if (index.size() < 4)
    index.insert(index.begin(), 4 - index.size(), '0');
  return (directory + "/" + index + extension).str();
}

std::string asString(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

struct StableExternalClosure final {
  OpenRoadExternalFile technology;
  OpenRoadExternalFile cells;
  OpenRoadExternalFile liberty;
};

llvm::Expected<StableExternalClosure>
stableExternalClosure(const OpenRoadPlacedConfig &config) {
  if (config.externalFiles.size() != 3)
    return invalid("stable routed capability requires exactly three files");
  const OpenRoadExternalFile *technology = nullptr;
  const OpenRoadExternalFile *cells = nullptr;
  const OpenRoadExternalFile *liberty = nullptr;
  for (const OpenRoadExternalFile &file : config.externalFiles) {
    switch (file.kind) {
    case OpenRoadExternalFileKind::TechnologyLef:
      technology = &file;
      break;
    case OpenRoadExternalFileKind::CellLef:
      cells = &file;
      break;
    case OpenRoadExternalFileKind::Liberty:
      liberty = &file;
      break;
    }
  }
  if (!technology || !cells || !liberty ||
      technology->logicalName != "technology" ||
      cells->logicalName != "cells" || liberty->logicalName != "timing" ||
      openRoadExternalFileInputSlot(*technology) !=
          openRoadTechnologyLefInputSlot ||
      openRoadExternalFileInputSlot(*cells) != openRoadCellLefInputSlot ||
      openRoadExternalFileInputSlot(*liberty) != openRoadLibertyInputSlot)
    return invalid("stable routed file roles are not exact");
  return StableExternalClosure{*technology, *cells, *liberty};
}

std::vector<external_tool::ExternalFileRequirement>
externalRequirements(const OpenRoadPlacedConfig &config) {
  std::vector<external_tool::ExternalFileRequirement> requirements;
  requirements.reserve(config.externalFiles.size());
  for (const OpenRoadExternalFile &file : config.externalFiles)
    requirements.push_back(
        {openRoadExternalFileInputSlot(file), file.fingerprint});
  return requirements;
}

struct RoutedInvocationFacts final {
  OpenRoadPlacedConfig config;
  StableExternalClosure closure;
  hardware::FinalizedHardwareImplementation source;
  platform::FinalizedImplementationPlatform platform;
  std::string top;
  external_tool::ExternalToolSemanticContract semanticContract;
  std::vector<external_tool::MaterializedBundleFile> semanticFiles;
  std::vector<external_tool::ExternalFileRequirement> externalRequirements;
  OpenRoadRoutedDriverFiles driverFiles;
};

llvm::Error
validateSourceExternalClosure(const hardware::HardwareImplementation &source,
                              const StableExternalClosure &closure) {
  if (!source.memoryMacroBindings().empty())
    return invalid("stable routed capability does not admit memory macros");
  const auto bindings = source.externalImplementationBindings();
  if (bindings.size() != 1 ||
      bindings.front().providerContractRef !=
          openSourceYosysStandardCellContractRef ||
      bindings.front().externalInputs.size() != 1 ||
      !bindings.front().fabricResourceRefs.empty() ||
      !bindings.front().blackBoxContractPayloadRef)
    return invalid("input does not have one exact Yosys standard-cell closure");
  const hardware::ExternalInputBinding &input =
      bindings.front().externalInputs.front();
  const auto *file =
      std::get_if<hardware::ExplicitFileDependency>(&input.dependencyIdentity);
  if (input.providerInputSlotRef != asicStandardCellLibertyInputSlot || !file ||
      file->contentSha256 != closure.liberty.fingerprint)
    return invalid("input Liberty binding differs from routed configuration");
  return llvm::Error::success();
}

llvm::Expected<RoutedInvocationFacts> invocationFacts(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const dse::CandidateGeneratorDescriptor &descriptor =
      openRoadRoutedCandidateGeneratorDescriptor();
  if (binding.descriptorRef() != descriptor.reference())
    return invalid("resolved binding references another generator");
  if (llvm::Error error = dse::validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputs))
    return std::move(error);
  auto config = decodeOpenRoadPlacedConfig(binding.canonicalConfigBytes());
  if (!config)
    return config.takeError();
  auto closure = stableExternalClosure(*config);
  if (!closure)
    return closure.takeError();
  auto semanticContract =
      dse::deriveExternalToolSemanticContract(inputs, binding);
  if (!semanticContract)
    return semanticContract.takeError();

  const ArtifactRootReference &sourceReference =
      inputs.front().artifacts.front();
  auto source = hardware::importHardwareImplementation(
      sourceReference, contracts, artifacts, blobs);
  if (!source)
    return source.takeError();
  const hardware::HardwareImplementation &implementation =
      source->implementation();
  const hardware::ImplementationRepresentationRoot &root =
      implementation.representationRoot();
  OpenRoadRoutedInputKind inputKind;
  if (root.variant == hardware::RepresentationRootVariant::GateNetlist &&
      !root.stage &&
      root.formatRef.kind() ==
          hardware::RepresentationFormatKind::StructuralVerilogGateNetlist &&
      root.top.kind == hardware::RepresentationObjectKind::Module) {
    inputKind = OpenRoadRoutedInputKind::GateNetlist;
  } else if (root.variant ==
                 hardware::RepresentationRootVariant::AsicPhysical &&
             root.stage == hardware::RepresentationPhysicalStage::Placed &&
             root.formatRef.kind() ==
                 hardware::RepresentationFormatKind::IndexedPhysical &&
             root.top.kind ==
                 hardware::RepresentationObjectKind::PhysicalObject) {
    inputKind = OpenRoadRoutedInputKind::PlacedDatabase;
  } else {
    return invalid("input is not an exact gate or placed implementation");
  }
  if (!isPortableIdentifier(root.top.canonicalName))
    return invalid("input top is not a portable identifier");
  if (!implementation.implementationPlatform())
    return invalid("input has no exact implementation platform");
  const ArtifactRootReference platformReference{
      platform::implementationPlatformSchema.identity.str(),
      platform::implementationPlatformSchema.version, config->corner.artifact};
  if (*implementation.implementationPlatform() != platformReference)
    return invalid("configuration corner belongs to a foreign platform");
  auto target =
      platform::importImplementationPlatform(platformReference, artifacts);
  if (!target)
    return target.takeError();
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()) ||
      !target->platform().findTechnologyCorner(config->corner.entity))
    return invalid("OpenROAD route requires the selected ASIC corner");
  if (llvm::Error error =
          validateSourceExternalClosure(implementation, *closure))
    return std::move(error);
  const std::string top = root.top.canonicalName;

  RoutedInvocationFacts facts{
      std::move(*config),
      std::move(*closure),
      std::move(*source),
      std::move(*target),
      top,
      std::move(*semanticContract),
      {},
      {},
      OpenRoadRoutedDriverFiles{inputKind, {}, std::nullopt, {}, "", {}, {}}};
  const hardware::ImplementationRepresentationRoot &materializedRoot =
      facts.source.implementation().representationRoot();
  facts.semanticFiles.push_back(
      {"inputs/hardware-implementation.json",
       asString(facts.source.canonicalBytes().bytes()), sourceReference,
       false});
  facts.semanticFiles.push_back(
      {"inputs/implementation-platform.json",
       asString(facts.platform.canonicalBytes().bytes()), platformReference,
       false});

  std::size_t netlistOrdinal = 0;
  std::size_t constraintOrdinal = 0;
  std::size_t contractOrdinal = 0;
  std::size_t databaseOrdinal = 0;
  for (const hardware::ImplementationPayload &payload :
       materializedRoot.payloads) {
    if (payload.role == hardware::PayloadRole::RepresentationIndex)
      continue;
    auto contents = blobs.get(payload.blobDigest);
    if (!contents)
      return contents.takeError();
    std::string path;
    switch (payload.role) {
    case hardware::PayloadRole::Netlist:
      if (inputKind != OpenRoadRoutedInputKind::GateNetlist)
        return invalid("placed input unexpectedly retains a netlist");
      path = indexedPath("inputs/netlist", netlistOrdinal++, ".v");
      facts.driverFiles.netlists.push_back(path);
      break;
    case hardware::PayloadRole::GenerationConstraint:
      path = indexedPath("inputs/constraints", constraintOrdinal++, ".sdc");
      facts.driverFiles.constraints.push_back(path);
      break;
    case hardware::PayloadRole::BlackBoxContract:
      path = indexedPath("inputs/contracts", contractOrdinal++, ".txt");
      break;
    case hardware::PayloadRole::PhysicalDatabase:
      if (inputKind != OpenRoadRoutedInputKind::PlacedDatabase ||
          databaseOrdinal++ != 0)
        return invalid("input physical database closure is ambiguous");
      path = "inputs/database/placed.odb";
      facts.driverFiles.placedDatabase = path;
      break;
    default:
      return invalid("input contains a payload outside routed capability");
    }
    facts.semanticFiles.push_back(
        {std::move(path), asString(*contents), sourceReference, false});
  }
  if (facts.driverFiles.constraints.empty())
    return invalid("routed capability requires one or more SDC payloads");
  if (inputKind == OpenRoadRoutedInputKind::GateNetlist &&
      facts.driverFiles.netlists.empty())
    return invalid("gate input has no netlist payload");
  if (inputKind == OpenRoadRoutedInputKind::PlacedDatabase &&
      !facts.driverFiles.placedDatabase)
    return invalid("placed input has no database payload");
  facts.externalRequirements = externalRequirements(facts.config);
  return facts;
}

external_tool::ExternalToolInvocationImportExpectation
importExpectation(const RoutedInvocationFacts &facts) {
  external_tool::ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = facts.semanticContract;
  for (const external_tool::MaterializedBundleFile &file : facts.semanticFiles)
    expectation.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact,
         computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
             reinterpret_cast<const std::uint8_t *>(file.contents.data()),
             file.contents.size()))});
  for (const external_tool::ExternalFileRequirement &file :
       facts.externalRequirements)
    expectation.externalInputs.push_back(
        {file.providerInputSlot, file.fingerprint});
  expectation.declaredOutputs = {kNetlistOutput.str(), kDefOutput.str(),
                                 kResultOutput.str()};
  return expectation;
}

hardware::RepresentationLocator
projectPhysicalLocator(const hardware::RepresentationLocator &locator,
                       const hardware::RepresentationLocator &sourceTop,
                       const hardware::RepresentationLocator &physicalTop) {
  return locator == sourceTop ? physicalTop : locator;
}

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishRoutedImplementation(
    const RoutedInvocationFacts &facts, llvm::StringRef netlist,
    llvm::StringRef def,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  using namespace hardware;
  const HardwareImplementation &source = facts.source.implementation();
  const ImplementationRepresentationRoot &sourceRoot =
      source.representationRoot();
  auto sourceIndex = indexRepresentationRoot(sourceRoot, blobs);
  if (!sourceIndex)
    return sourceIndex.takeError();

  auto netlistDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(netlist.data()), netlist.size()));
  if (!netlistDigest)
    return netlistDigest.takeError();
  auto defDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(def.data()), def.size()));
  if (!defDigest)
    return defDigest.takeError();

  std::vector<ImplementationPayload> payloads{
      {PayloadRole::Netlist, "netlist/" + facts.top + ".v", *netlistDigest},
      {PayloadRole::PhysicalDatabase, "database/" + facts.top + ".def",
       *defDigest}};
  for (const ImplementationPayload &payload : sourceRoot.payloads)
    if (payload.role == PayloadRole::GenerationConstraint)
      payloads.push_back(payload);

  auto gateFormat = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::StructuralVerilogGateNetlist);
  if (!gateFormat)
    return gateFormat.takeError();
  std::vector<ImplementationPayload> logicalPayloads;
  for (const ImplementationPayload &payload : payloads)
    if (payload.role == PayloadRole::Netlist ||
        payload.role == PayloadRole::GenerationConstraint)
      logicalPayloads.push_back(payload);
  auto canonicalLogical =
      canonicalizeImplementationPayloadCatalog(std::move(logicalPayloads));
  if (!canonicalLogical)
    return canonicalLogical.takeError();
  const RepresentationLocator logicalTop{RepresentationObjectKind::Module,
                                         facts.top};
  auto logical =
      indexRepresentation(*gateFormat, logicalTop, *canonicalLogical, blobs);
  if (!logical)
    return logical.takeError();
  if (logical->unresolvedExternalDefinitions().empty())
    return invalid("routed netlist has no standard-cell definitions");

  const RepresentationLocator physicalTop{
      RepresentationObjectKind::PhysicalObject, facts.top};
  std::vector<PhysicalRepresentationObject> objects{
      {physicalTop, std::nullopt}};
  const auto addLogicalObject =
      [&](const RepresentationLocator &locator) -> llvm::Error {
    if (llvm::any_of(objects, [&](const auto &object) {
          return object.locator == locator;
        }))
      return llvm::Error::success();
    auto found = logical->lookup(locator);
    if (!found)
      return found.takeError();
    if (!*found)
      return invalid("routed netlist omits a retained representation locator");
    objects.push_back({locator, (*found)->signalGeometry});
    return llvm::Error::success();
  };
  for (const RepresentationBoundaryPort &port : logical->rootBoundaryPorts())
    if (llvm::Error error = addLogicalObject(port.locator))
      return std::move(error);
  for (const RepresentationLocator &locator :
       logical->unresolvedExternalDefinitions())
    if (llvm::Error error = addLogicalObject(locator))
      return std::move(error);

  std::vector<ImplementationInterface> interfaces(source.interfaces().begin(),
                                                  source.interfaces().end());
  for (ImplementationInterface &interface : interfaces) {
    const RepresentationLocator sourceLocator = interface.representationLocator;
    if (!(sourceLocator == sourceRoot.top)) {
      auto before = sourceIndex->lookup(sourceLocator);
      auto after = logical->lookup(sourceLocator);
      if (!before)
        return before.takeError();
      if (!after)
        return after.takeError();
      if (!*before || !*after || !(**before == **after))
        return invalid("route changed an exact interface locator or geometry");
      if (llvm::Error error = addLogicalObject(sourceLocator))
        return std::move(error);
    }
    interface.representationLocator =
        projectPhysicalLocator(sourceLocator, sourceRoot.top, physicalTop);
  }
  std::vector<ActivityPoint> activityPoints(source.activityPoints().begin(),
                                            source.activityPoints().end());
  for (ActivityPoint &point : activityPoints) {
    const RepresentationLocator sourceLocator = point.representationLocator;
    if (!(sourceLocator == sourceRoot.top)) {
      auto before = sourceIndex->lookup(sourceLocator);
      auto after = logical->lookup(sourceLocator);
      if (!before)
        return before.takeError();
      if (!after)
        return after.takeError();
      if (!*before || !*after || !(**before == **after))
        return invalid("route changed an exact activity locator or geometry");
      if (llvm::Error error = addLogicalObject(sourceLocator))
        return std::move(error);
    }
    point.representationLocator =
        projectPhysicalLocator(sourceLocator, sourceRoot.top, physicalTop);
  }

  std::string contract =
      "loom.open_source.openroad.routed_standard_cell_contract.1.0\n";
  contract +=
      "technology_lef_sha256=" +
      formatExternalFileFingerprint(facts.closure.technology.fingerprint) +
      "\n";
  contract += "cell_lef_sha256=" +
              formatExternalFileFingerprint(facts.closure.cells.fingerprint) +
              "\n";
  contract += "liberty_sha256=" +
              formatExternalFileFingerprint(facts.closure.liberty.fingerprint) +
              "\n";
  for (const RepresentationLocator &locator :
       logical->unresolvedExternalDefinitions())
    contract += "module=" + locator.canonicalName + "\n";
  auto contractDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(contract.data()),
      contract.size()));
  if (!contractDigest)
    return contractDigest.takeError();
  payloads.push_back({PayloadRole::BlackBoxContract, kBlackBoxLogicalName.str(),
                      *contractDigest});

  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::IndexedDefPhysical);
  if (!format)
    return format.takeError();
  auto index = createPhysicalRepresentationIndexPayload(
      *format, RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed, physicalTop, "index/physical.json",
      payloads, std::move(objects),
      logical->unresolvedExternalDefinitions().vec());
  if (!index)
    return index.takeError();
  auto indexBytes = serializePhysicalRepresentationIndexPayloadJson(*index);
  if (!indexBytes)
    return indexBytes.takeError();
  auto indexDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(indexBytes->data()),
      indexBytes->size()));
  if (!indexDigest)
    return indexDigest.takeError();
  payloads.push_back({PayloadRole::RepresentationIndex, index->indexLogicalName,
                      *indexDigest});
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::AsicPhysical,
      RepresentationPhysicalStage::Routed, *format, physicalTop,
      std::move(payloads));
  if (!representation)
    return representation.takeError();

  std::vector<ExternalInputBinding> externalInputs{
      {openRoadTechnologyLefInputSlot.str(),
       ExplicitFileDependency{facts.closure.technology.fingerprint}},
      {openRoadCellLefInputSlot.str(),
       ExplicitFileDependency{facts.closure.cells.fingerprint}},
      {openRoadLibertyInputSlot.str(),
       ExplicitFileDependency{facts.closure.liberty.fingerprint}}};
  return finalizeHardwareImplementation(
      HardwareImplementationDraft{
          source.fabric(),
          source.subject(),
          source.configurationAbi(),
          std::move(*representation),
          source.implementationPlatform(),
          std::move(interfaces),
          std::move(activityPoints),
          {},
          {ExternalImplementationBindingDraft{
              openRoadRoutedStandardCellContractRef.str(),
              std::move(externalInputs),
              {},
              logical->unresolvedExternalDefinitions().vec(),
              ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                       kBlackBoxLogicalName.str()}}}},
      contracts, artifacts, blobs);
}

dse::CandidateGeneratorProviderResult
incompleteResult(dse::CandidateGeneratorIncompleteReason reason) {
  return dse::CandidateGeneratorProviderResult{
      dse::IncompleteCandidateGeneratorResult{
          reason, {{dse::CandidateGeneratorOutputSlotRef(0), {}}}, {}},
      {{dse::CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

std::string canonicalResult(llvm::StringRef top) {
  return "{\"schema\":\"" + kResultSchema.str() + "\",\"version\":\"" +
         kResultVersion.str() + "\",\"stage\":\"routed\",\"top\":\"" +
         top.str() + "\"}\n";
}

llvm::Expected<external_tool::PreparedExternalToolInvocation> prepareRegistered(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context) {
  auto config = decodeOpenRoadPlacedConfig(binding.canonicalConfigBytes());
  if (!config)
    return config.takeError();
  auto execution = resolveOpenRoadExecution(config->providerBuild, context);
  if (!execution)
    return execution.takeError();
  auto contracts = makeKnownAsicStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return prepareOpenRoadRoutedInvocation(inputs, binding, *contracts, artifacts,
                                         blobs, *execution, context);
}

llvm::Expected<dse::CandidateGeneratorProviderResult>
importRegistered(llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
                 const dse::ResolvedCandidateGeneratorBinding &binding,
                 const external_tool::PreparedExternalToolInvocation &prepared,
                 const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto contracts = makeKnownAsicStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  return importOpenRoadRoutedInvocation(inputs, binding, prepared, *contracts,
                                        artifacts, blobs);
}

} // namespace

const dse::CandidateGeneratorDescriptor &
openRoadRoutedCandidateGeneratorDescriptor() {
  static const std::array<dse::CandidateGeneratorInputSlotDescriptor, 1> inputs{
      {{dse::CandidateGeneratorInputSlotRef(0),
        "gate_or_placed_hardware_implementation",
        dse::PlanValueRole::CandidateSet,
        &hardware::hardwareImplementationSchema,
        dse::PlanValueCardinality::ExactlyOne}}};
  static const std::array<dse::CandidateGeneratorOutputSlotDescriptor, 1>
      outputs{{{dse::CandidateGeneratorOutputSlotRef(0),
                "routed_asic_physical_implementation",
                dse::PlanValueRole::CandidateSet,
                &hardware::hardwareImplementationSchema,
                dse::PlanValueCardinality::ExactlyOne}}};
  static const std::array<dse::CandidateGeneratorWorkUnitDescriptor, 1>
      workUnits{
          {{dse::CandidateGeneratorWorkUnitRef(0), "openroad_routed_attempt"}}};
  static const dse::CandidateGeneratorDescriptor descriptor{
      openRoadRoutedCandidateGeneratorKind,
      "openroad.routed_asic_physical",
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

llvm::Error registerOpenRoadRoutedCandidateGeneratorDescriptor() {
  return dse::registerCandidateGeneratorDescriptor(
      openRoadRoutedCandidateGeneratorDescriptor());
}

llvm::Error registerOpenRoadRoutedCandidateGenerator() {
  const dse::CandidateGeneratorDescriptor &descriptor =
      openRoadRoutedCandidateGeneratorDescriptor();
  static const dse::CandidateGeneratorProvider provider{
      descriptor.reference(),
      dse::CandidateGeneratorExternalPrepareImportProvider{prepareRegistered,
                                                           importRegistered}};
  if (llvm::Error error = dse::registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return dse::registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::string>
renderOpenRoadRoutedDriver(llvm::StringRef topModule,
                           const OpenRoadPlacementParameters &parameters,
                           const OpenRoadRoutedDriverFiles &files) {
  if (!isPortableIdentifier(topModule))
    return invalid("top module is not a portable identifier");
  if (llvm::Error error = validateOpenRoadPlacementParameters(parameters))
    return std::move(error);
  if (files.constraints.empty())
    return invalid("constraint closure is empty");
  if (files.cellLefs.size() != 1 || files.libertyFiles.size() != 1)
    return invalid("stable route requires one cell LEF and one Liberty file");
  if (files.inputKind == OpenRoadRoutedInputKind::GateNetlist &&
      (files.netlists.empty() || files.placedDatabase))
    return invalid("gate driver input closure is inconsistent");
  if (files.inputKind == OpenRoadRoutedInputKind::PlacedDatabase &&
      (!files.netlists.empty() || !files.placedDatabase))
    return invalid("placed driver input closure is inconsistent");

  auto technology = tclString(files.technologyLef, "technology LEF path");
  auto cells = tclList(files.cellLefs, "cell LEF path");
  auto liberties = tclList(files.libertyFiles, "Liberty path");
  auto netlists = tclList(files.netlists, "netlist path");
  auto constraints = tclList(files.constraints, "constraint path");
  if (!technology)
    return technology.takeError();
  if (!cells)
    return cells.takeError();
  if (!liberties)
    return liberties.takeError();
  if (!netlists)
    return netlists.takeError();
  if (!constraints)
    return constraints.takeError();

  std::string driver;
  driver += "set loom_technology_lef " + *technology + "\n";
  driver += "set loom_cell_lefs " + *cells + "\n";
  driver += "set loom_liberty_files " + *liberties + "\n";
  driver += "set loom_netlists " + *netlists + "\n";
  driver += "set loom_constraints " + *constraints + "\n";
  if (files.inputKind == OpenRoadRoutedInputKind::GateNetlist) {
    driver += "read_lef $loom_technology_lef\n";
    driver += "foreach loom_path $loom_cell_lefs { read_lef $loom_path }\n";
    driver +=
        "foreach loom_path $loom_liberty_files { read_liberty $loom_path }\n";
    driver += "foreach loom_path $loom_netlists { read_verilog $loom_path }\n";
    driver += "link_design " + topModule.str() + "\n";
    driver += "foreach loom_path $loom_constraints { read_sdc $loom_path }\n";
    driver += "initialize_floorplan -die_area {" +
              rectangle(parameters.dieArea) + "} -core_area {" +
              rectangle(parameters.coreArea) + "} -site " +
              parameters.siteName + "\n";
    driver += "make_tracks\n";
    driver += "place_pins -hor_layers " + parameters.horizontalPinLayer +
              " -ver_layers " + parameters.verticalPinLayer + "\n";
    driver += "global_placement -density " +
              density(parameters.placementDensityPpm) + " -random_seed 1\n";
    driver += "detailed_placement\n";
    driver += "check_placement -verbose\n";
  } else {
    auto database = tclString(*files.placedDatabase, "placed database path");
    if (!database)
      return database.takeError();
    driver += "read_db " + *database + "\n";
    driver +=
        "foreach loom_path $loom_liberty_files { read_liberty $loom_path }\n";
    driver += "foreach loom_path $loom_constraints { read_sdc $loom_path }\n";
  }
  driver +=
      "if {[llength [all_clocks]] == 0} { error {route requires a clock} }\n";
  driver += "clock_tree_synthesis -repair_clock_nets\n";
  driver += "detailed_placement\n";
  driver += "global_route -congestion_iterations 30\n";
  driver += "detailed_route -or_seed 1\n";
  driver += "write_verilog -sort " + kNetlistOutput.str() + "\n";
  driver += "write_def -version 5.8 " + kDefOutput.str() + "\n";
  driver += "set loom_result [open " + kResultOutput.str() + " w]\n";
  std::string marker = canonicalResult(topModule);
  marker.pop_back();
  driver += "puts $loom_result {" + marker + "}\n";
  driver += "close $loom_result\n";
  return driver;
}

llvm::Expected<OpenRoadRoutedAttemptResult>
parseOpenRoadRoutedAttemptResult(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("result JSON is malformed: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 4)
    return invalid("result JSON has an invalid shape");
  const auto schema = object->getString("schema");
  const auto version = object->getString("version");
  const auto stage = object->getString("stage");
  const auto top = object->getString("top");
  if (!schema || *schema != kResultSchema || !version ||
      *version != kResultVersion || !stage || *stage != "routed" || !top ||
      !isPortableIdentifier(*top) || contents != canonicalResult(*top))
    return invalid("result JSON fields are invalid or noncanonical");
  return OpenRoadRoutedAttemptResult{top->str()};
}

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareOpenRoadRoutedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const OpenRoadResolvedExecution &execution,
    const external_tool::ExternalToolPreparationContext &context) {
  auto facts = invocationFacts(inputs, binding, contracts, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  if (llvm::Error error = validateOpenRoadResolvedExecution(
          execution, facts->config.providerBuild))
    return std::move(error);
  auto externalFiles = external_tool::resolveExternalFiles(
      facts->externalRequirements, context.localConfig);
  if (!externalFiles)
    return externalFiles.takeError();
  if (externalFiles->size() != facts->config.externalFiles.size())
    return invalid("resolved external-file closure has the wrong size");
  for (std::size_t index = 0; index < externalFiles->size(); ++index) {
    const OpenRoadExternalFile &semantic = facts->config.externalFiles[index];
    const external_tool::ResolvedExternalFile &resolved =
        (*externalFiles)[index];
    if (resolved.providerInputSlot != openRoadExternalFileInputSlot(semantic) ||
        resolved.fingerprint != semantic.fingerprint)
      return invalid("resolved external-file closure differs from config");
    switch (semantic.kind) {
    case OpenRoadExternalFileKind::TechnologyLef:
      facts->driverFiles.technologyLef = resolved.absolutePath;
      break;
    case OpenRoadExternalFileKind::CellLef:
      facts->driverFiles.cellLefs.push_back(resolved.absolutePath);
      break;
    case OpenRoadExternalFileKind::Liberty:
      facts->driverFiles.libertyFiles.push_back(resolved.absolutePath);
      break;
    }
  }
  auto driver = renderOpenRoadRoutedDriver(facts->top, facts->config.placement,
                                           facts->driverFiles);
  if (!driver)
    return driver.takeError();

  external_tool::ExternalToolInvocationBundleSpec specification;
  specification.semanticContract = std::move(facts->semanticContract);
  specification.tool = execution.tool;
  specification.toolVersionProbe = execution.provider.versionProbe;
  specification.runtime = execution.runtime;
  specification.containerVersionProbe = execution.containerVersionProbe;
  specification.commands = {{execution.tool.executable, "-no_init",
                             "-no_splash", "-no_settings", "-threads", "1",
                             "-exit", "drivers/openroad-routed.tcl"}};
  specification.declaredOutputs = {kNetlistOutput.str(), kDefOutput.str(),
                                   kResultOutput.str()};
  specification.files.push_back(
      {"drivers/openroad-routed.tcl", std::move(*driver), std::nullopt, false});
  specification.files.insert(
      specification.files.end(),
      std::make_move_iterator(facts->semanticFiles.begin()),
      std::make_move_iterator(facts->semanticFiles.end()));
  specification.externalFiles = std::move(*externalFiles);
  return external_tool::finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
}

llvm::Expected<dse::CandidateGeneratorProviderResult>
importOpenRoadRoutedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto facts = invocationFacts(inputs, binding, contracts, artifacts, blobs);
  if (!facts)
    return facts.takeError();
  auto attempt = external_tool::importExternalToolInvocationAttempt(
      prepared, importExpectation(*facts));
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
  auto netlist = external_tool::readExternalToolInvocationDeclaredOutput(
      imported, kNetlistOutput);
  auto def = external_tool::readExternalToolInvocationDeclaredOutput(
      imported, kDefOutput);
  auto result = external_tool::readExternalToolInvocationDeclaredOutput(
      imported, kResultOutput);
  if (!netlist)
    return netlist.takeError();
  if (!def)
    return def.takeError();
  if (!result)
    return result.takeError();
  if (netlist->empty() || def->empty())
    return invalid("routed output closure contains an empty payload");
  auto marker = parseOpenRoadRoutedAttemptResult(*result);
  if (!marker)
    return marker.takeError();
  if (marker->topModule != facts->top)
    return invalid("routed result top differs from the exact input");
  auto routed = publishRoutedImplementation(*facts, *netlist, *def, contracts,
                                            artifacts, blobs);
  if (!routed)
    return routed.takeError();
  return dse::CandidateGeneratorProviderResult{
      dse::CompletedCandidateGeneratorResult{
          {{dse::CandidateGeneratorOutputSlotRef(0), {routed->reference()}}},
          {{dse::CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            dse::CandidateGeneratorOutputSlotRef(0),
            routed->reference(),
            {},
            {}}}},
      {{dse::CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

} // namespace loom::eda::open_source
