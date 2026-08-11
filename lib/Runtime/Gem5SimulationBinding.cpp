#include "Gem5SimulationBindingInternal.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Artifact/InterconnectImplementation.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Runtime/Gem5BridgeABI.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::runtime {

namespace detail {

class Gem5SimulationBindingBuilder final {
public:
  static Gem5SimulationBinding create(
      ArtifactRootReference fabric,
      ArtifactRootReference interconnectImplementation,
      Gem5BuildIdentity gem5BuildIdentity, std::string bridgeAbiIdentity,
      std::vector<Gem5Correspondence> correspondences) {
    return Gem5SimulationBinding(
        std::move(fabric), std::move(interconnectImplementation),
        std::move(gem5BuildIdentity), std::move(bridgeAbiIdentity),
        std::move(correspondences));
  }
};

} // namespace detail

namespace {

using Key = std::vector<std::uint8_t>;

std::vector<const Gem5ModelContractDescriptor *> &modelContracts() {
  static std::vector<const Gem5ModelContractDescriptor *> descriptors;
  return descriptors;
}

std::mutex &modelContractMutex() {
  static std::mutex mutex;
  return mutex;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_simulation_binding_invalid: " + message);
}

void appendU32(Key &key, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    key.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(Key &key, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    key.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBytes(Key &key, llvm::ArrayRef<std::uint8_t> bytes) {
  appendU64(key, bytes.size());
  key.insert(key.end(), bytes.begin(), bytes.end());
}

void appendText(Key &key, llvm::StringRef text) {
  appendBytes(key, {reinterpret_cast<const std::uint8_t *>(text.data()),
                    text.size()});
}

template <typename Ref> void appendFabricRef(Key &key, const Ref &reference) {
  appendBytes(key, fabric::canonicalFabricBytes(reference));
}

void appendSpatialBoundary(
    Key &key, const fabric::FabricSpatialAttachmentEndpointRef &reference) {
  appendBytes(key,
              fabric::encodeFabricSpatialAttachmentEndpointRef(reference));
}

Key objectKey(const Gem5SimObjectRef &object) {
  Key key;
  appendText(key, object.contract.identity);
  appendU32(key, object.contract.version.major);
  appendU32(key, object.contract.version.minor);
  appendBytes(key, object.payload);
  return key;
}

Key portKey(const Gem5SimPortRef &port) {
  Key key = objectKey(port.object);
  appendU32(key, port.kind);
  appendBytes(key, port.payload);
  return key;
}

Key correspondenceKey(const Gem5Correspondence &correspondence) {
  Key key;
  appendU32(key, correspondence.index());
  if (const auto *processor =
          std::get_if<Gem5ProcessorCorrespondence>(&correspondence)) {
    appendU32(key, processor->processor.index());
    std::visit([&](const auto &ref) { appendFabricRef(key, ref); },
               processor->processor);
  } else if (const auto *bridge =
                 std::get_if<Gem5SpatialBridgeCorrespondence>(
                     &correspondence)) {
    appendFabricRef(key, bridge->spatialCore);
    appendSpatialBoundary(key, bridge->spatialBoundary);
  } else if (const auto *memory =
                 std::get_if<Gem5MemoryOrServiceCorrespondence>(
                     &correspondence)) {
    appendU32(key, memory->fabricRef.index());
    std::visit([&](const auto &ref) { appendFabricRef(key, ref); },
               memory->fabricRef);
  } else if (const auto *transport =
                 std::get_if<Gem5TransportCorrespondence>(&correspondence)) {
    appendU32(key, transport->fabricRef.index());
    std::visit([&](const auto &ref) { appendFabricRef(key, ref); },
               transport->fabricRef);
  } else {
    appendFabricRef(
        key,
        std::get<Gem5ExternalEndpointCorrespondence>(correspondence).fabricRef);
  }
  return key;
}

std::optional<fabric::SpatialCoreOccurrenceRef> spatialCoreOf(
    const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport()) {
    if (transport->owner.kind() !=
        fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return std::nullopt;
    return std::get<fabric::SpatialCoreOccurrenceRef>(
        transport->owner.payload);
  }
  const auto *memory = endpoint.memory();
  if (!memory || memory->owner.kind() !=
                     fabric::FabricMemoryEndpointOwnerKind::
                         SpatialCoreOccurrence)
    return std::nullopt;
  return std::get<fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
}

std::vector<Key>
requiredCorrespondenceKeys(const fabric::FabricSystemRootView &system) {
  std::vector<Key> keys;
  for (fabric::HostCoreOccurrenceRef core :
       system.artifact().hostCoreOccurrences()) {
    Gem5Correspondence row = Gem5ProcessorCorrespondence{
        Gem5ProcessorFabricRef(core), Gem5SimObjectRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    Gem5Correspondence row = Gem5ProcessorCorrespondence{
        Gem5ProcessorFabricRef(fabric::InstructionCoreContextRef{core}),
        Gem5SimObjectRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (const fabric::FabricSpatialAttachmentRecordView &attachment :
       system.spatialAttachments()) {
    const auto core = spatialCoreOf(attachment.spatialEndpoint);
    if (!core)
      llvm::report_fatal_error(
          "validated spatial attachment has no SpatialCore owner");
    Gem5Correspondence row = Gem5SpatialBridgeCorrespondence{
        *core, attachment.spatialEndpoint, Gem5SimPortRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (fabric::SystemMemoryServiceRef service :
       system.artifact().systemMemoryServices()) {
    Gem5Correspondence row = Gem5MemoryOrServiceCorrespondence{
        Gem5MemoryOrServiceFabricRef(service), Gem5SimObjectRef{},
        Gem5SimPortRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (fabric::SystemServiceEndpointRef endpoint :
       system.artifact().systemServiceEndpoints()) {
    Gem5Correspondence row = Gem5MemoryOrServiceCorrespondence{
        Gem5MemoryOrServiceFabricRef(endpoint), Gem5SimObjectRef{},
        Gem5SimPortRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (fabric::SystemTransportResourceRef resource :
       system.transportResources()) {
    Gem5Correspondence row = Gem5TransportCorrespondence{
        Gem5TransportFabricRef(resource), Gem5SimObjectRef{},
        Gem5SimPortRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (const fabric::FabricTransportEndpointRef &endpoint :
       system.artifact().transportEndpoints()) {
    const auto owner = endpoint.owner.kind();
    if (owner !=
            fabric::FabricTransportEndpointOwnerKind::SystemServiceEndpoint &&
        owner !=
            fabric::FabricTransportEndpointOwnerKind::SystemTransportResource)
      continue;
    Gem5Correspondence row = Gem5TransportCorrespondence{
        Gem5TransportFabricRef(endpoint), Gem5SimObjectRef{},
        Gem5SimPortRef{}};
    keys.push_back(correspondenceKey(row));
  }
  for (fabric::ExternalBoundaryRef boundary :
       system.artifact().externalBoundaries()) {
    Gem5Correspondence row = Gem5ExternalEndpointCorrespondence{
        boundary, Gem5SimObjectRef{}, Gem5SimPortRef{}};
    keys.push_back(correspondenceKey(row));
  }
  llvm::sort(keys);
  return keys;
}

bool isLowerHex(llvm::StringRef value, std::size_t length) {
  return value.size() == length &&
         llvm::all_of(value, [](char character) {
           return std::isdigit(static_cast<unsigned char>(character)) ||
                  (character >= 'a' && character <= 'f');
         });
}

llvm::Error validateBuildIdentity(const Gem5BuildIdentity &identity) {
  if (identity.repositoryIdentity.empty())
    return invalid("gem5 repository identity is empty");
  if (!isLowerHex(identity.fullCommitIdentity, 40) &&
      !isLowerHex(identity.fullCommitIdentity, 64))
    return invalid("gem5 full commit identity is not canonical lowercase hex");
  if (!isLowerHex(identity.buildConfigurationDigest, 64))
    return invalid(
        "gem5 build configuration digest is not a canonical SHA-256 hex");
  if (!isLowerHex(identity.binaryFingerprint, 64))
    return invalid("gem5 binary fingerprint is not a canonical SHA-256 hex");
  return llvm::Error::success();
}

llvm::Expected<const Gem5ModelContractDescriptor *>
validateObject(const Gem5SimObjectRef &object,
               std::optional<Gem5ModelObjectClass> expectedClass) {
  const Gem5ModelContractDescriptor *descriptor =
      findGem5ModelContract(object.contract);
  if (!descriptor)
    return invalid("SimObject references an unregistered model contract");
  if (expectedClass && descriptor->objectClass != *expectedClass)
    return invalid("SimObject class does not match its Fabric correspondence");
  if (llvm::Error error =
          descriptor->validateCanonicalObjectPayload(object.payload))
    return invalid("SimObject payload is not canonical: " +
                   llvm::toString(std::move(error)));
  return descriptor;
}

llvm::Expected<const Gem5ModelPortKindDescriptor *>
validatePort(const Gem5SimPortRef &port, Gem5ModelPortClass expectedClass,
             std::optional<Gem5ModelObjectClass> expectedObjectClass) {
  auto descriptor = validateObject(port.object, expectedObjectClass);
  if (!descriptor)
    return descriptor.takeError();
  const Gem5ModelPortKindDescriptor *kind =
      findGem5ModelPortKind(**descriptor, port.kind);
  if (!kind)
    return invalid("SimObject port references an unknown kind");
  if (kind->portClass != expectedClass)
    return invalid("SimObject port class does not match its Fabric reference");
  if (llvm::Error error = kind->validateCanonicalPayload(port.payload))
    return invalid("SimObject port payload is not canonical: " +
                   llvm::toString(std::move(error)));
  return kind;
}

llvm::Error validateProcessor(
    const Gem5ProcessorCorrespondence &row,
    const fabric::FabricSystemRootView &system,
    std::vector<std::pair<Key, const Gem5ModelContractDescriptor *>> &objects) {
  auto descriptor =
      validateObject(row.simObject, Gem5ModelObjectClass::Processor);
  if (!descriptor)
    return descriptor.takeError();
  if (!(**descriptor).validateProcessorCompatibility)
    return invalid("processor model contract has no compatibility owner");
  const fabric::InstructionCoreArchitecturalContract *architecture = nullptr;
  const fabric::InstructionCoreMicroarchitecturalRealization *microarchitecture =
      nullptr;
  std::visit(
      [&](const auto &core) {
        architecture = system.instructionCoreArchitecture(core);
        microarchitecture = system.instructionCoreMicroarchitecture(core);
      },
      row.processor);
  if (!architecture || !microarchitecture)
    return invalid("processor correspondence names an unknown core");
  if (llvm::Error error = (**descriptor).validateProcessorCompatibility(
          row.simObject.payload, *architecture, *microarchitecture))
    return invalid("processor model is incompatible with Fabric: " +
                   llvm::toString(std::move(error)));
  objects.emplace_back(objectKey(row.simObject), *descriptor);
  return llvm::Error::success();
}

llvm::Error validateCorrespondenceModels(
    llvm::ArrayRef<Gem5Correspondence> rows,
    const fabric::FabricSystemRootView &system) {
  std::vector<std::pair<Key, const Gem5ModelContractDescriptor *>> objects;
  std::vector<std::pair<Key, const Gem5ModelPortKindDescriptor *>> ports;
  objects.reserve(rows.size());
  ports.reserve(rows.size());
  for (const Gem5Correspondence &row : rows) {
    if (const auto *processor =
            std::get_if<Gem5ProcessorCorrespondence>(&row)) {
      if (llvm::Error error = validateProcessor(*processor, system, objects))
        return error;
      continue;
    }
    if (const auto *bridge =
            std::get_if<Gem5SpatialBridgeCorrespondence>(&row)) {
      if (spatialCoreOf(bridge->spatialBoundary) != bridge->spatialCore)
        return invalid("SpatialBridge core and boundary owners disagree");
      auto kind = validatePort(bridge->bridgeEndpoint,
                               Gem5ModelPortClass::SpatialBoundary,
                               Gem5ModelObjectClass::SpatialBridge);
      if (!kind)
        return kind.takeError();
      const auto *descriptor =
          findGem5ModelContract(bridge->bridgeEndpoint.object.contract);
      objects.emplace_back(objectKey(bridge->bridgeEndpoint.object), descriptor);
      ports.emplace_back(portKey(bridge->bridgeEndpoint), *kind);
      continue;
    }
    if (const auto *memory =
            std::get_if<Gem5MemoryOrServiceCorrespondence>(&row)) {
      if (!(memory->simObject == memory->simPort.object))
        return invalid("MemoryOrService object and port owners disagree");
      auto descriptor = validateObject(memory->simObject,
                                       Gem5ModelObjectClass::MemoryOrService);
      auto kind = validatePort(memory->simPort,
                               Gem5ModelPortClass::MemoryOrService,
                               Gem5ModelObjectClass::MemoryOrService);
      if (!descriptor)
        return descriptor.takeError();
      if (!kind)
        return kind.takeError();
      objects.emplace_back(objectKey(memory->simObject), *descriptor);
      ports.emplace_back(portKey(memory->simPort), *kind);
      continue;
    }
    if (const auto *transport =
            std::get_if<Gem5TransportCorrespondence>(&row)) {
      if (!(transport->simObject == transport->simPort.object))
        return invalid("Transport object and port owners disagree");
      const bool resource = std::holds_alternative<
          fabric::SystemTransportResourceRef>(transport->fabricRef);
      auto descriptor = validateObject(
          transport->simObject,
          resource ? std::optional(Gem5ModelObjectClass::Transport)
                   : std::nullopt);
      auto kind = validatePort(
          transport->simPort, Gem5ModelPortClass::Transport,
          resource ? std::optional(Gem5ModelObjectClass::Transport)
                   : std::nullopt);
      if (!descriptor)
        return descriptor.takeError();
      if (!kind)
        return kind.takeError();
      objects.emplace_back(objectKey(transport->simObject), *descriptor);
      ports.emplace_back(portKey(transport->simPort), *kind);
      continue;
    }
    const auto &external = std::get<Gem5ExternalEndpointCorrespondence>(row);
    if (!(external.simObject == external.simPort.object))
      return invalid("ExternalEndpoint object and port owners disagree");
    auto descriptor =
        validateObject(external.simObject,
                       Gem5ModelObjectClass::ExternalEndpoint);
    auto kind = validatePort(external.simPort,
                             Gem5ModelPortClass::ExternalEndpoint,
                             Gem5ModelObjectClass::ExternalEndpoint);
    if (!descriptor)
      return descriptor.takeError();
    if (!kind)
      return kind.takeError();
    objects.emplace_back(objectKey(external.simObject), *descriptor);
    ports.emplace_back(portKey(external.simPort), *kind);
  }

  llvm::sort(objects, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (std::size_t index = 1; index < objects.size(); ++index)
    if (objects[index - 1].first == objects[index].first &&
        !objects[index].second->allowsSharedBinding)
      return invalid("SimObject is bound more than once without declared "
                     "sharing support");

  llvm::sort(ports,
             [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  for (std::size_t index = 1; index < ports.size(); ++index)
    if (ports[index - 1].first == ports[index].first &&
        !ports[index].second->allowsSharedBinding)
      return invalid("SimObject port is bound more than once without declared "
                     "sharing support");
  return llvm::Error::success();
}

llvm::Expected<Gem5SimulationBinding>
canonicalize(Gem5SimulationBindingDraft draft,
             const ArtifactStore &artifacts) {
  if (draft.fabric.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      draft.fabric.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("Fabric reference has the wrong schema descriptor");
  auto fabricRoot = fabric::importEntireFabricRoot(draft.fabric, artifacts);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto system = fabric::requireSystemRoot(fabricRoot->view());
  if (!system)
    return system.takeError();

  if (draft.interconnectImplementation.schemaIdentity !=
          fabric::fabricArtifactSchema.identity ||
      draft.interconnectImplementation.schemaVersion !=
          fabric::fabricArtifactSchema.version)
    return invalid(
        "InterconnectImplementation reference has the wrong schema descriptor");
  auto interconnect = fabric::importEntireFabricRoot(
      draft.interconnectImplementation, artifacts);
  if (!interconnect)
    return interconnect.takeError();
  if (interconnect->view().rootKind() !=
      fabric::FabricRootKind::InterconnectImplementation)
    return invalid("interconnect reference has the wrong Fabric root kind");
  if (interconnect->directDependencies().size() != 1 ||
      interconnect->directDependencies().front().role !=
          fabric::FabricDependencyRole::RefinedSystem ||
      interconnect->directDependencies().front().root != draft.fabric)
    return invalid("InterconnectImplementation does not refine the exact "
                   "bound Fabric System");
  auto interconnectSchema = fabric::interconnectProtocolSchema(*interconnect);
  if (!interconnectSchema)
    return interconnectSchema.takeError();
  if (*interconnectSchema !=
      ::fabric::InterconnectProtocolSchema::Gem5EventTransportV1)
    return invalid("InterconnectImplementation protocol is unsupported by "
                   "the gem5 binding");

  if (llvm::Error error = validateBuildIdentity(draft.gem5BuildIdentity))
    return std::move(error);
  if (draft.bridgeAbiIdentity != gem5BridgeAbiIdentity)
    return invalid("Bridge ABI identity is unsupported");

  llvm::sort(draft.correspondences,
             [](const Gem5Correspondence &lhs,
                const Gem5Correspondence &rhs) {
               return correspondenceKey(lhs) < correspondenceKey(rhs);
             });
  std::vector<Key> actual;
  actual.reserve(draft.correspondences.size());
  for (const Gem5Correspondence &row : draft.correspondences)
    actual.push_back(correspondenceKey(row));
  if (std::adjacent_find(actual.begin(), actual.end()) != actual.end())
    return invalid("correspondence table contains a duplicate Fabric key");
  if (actual != requiredCorrespondenceKeys(*system))
    return invalid("correspondence table does not exactly cover the modeled "
                   "Fabric System objects and boundaries");
  if (llvm::Error error =
          validateCorrespondenceModels(draft.correspondences, *system))
    return std::move(error);

  return detail::Gem5SimulationBindingBuilder::create(
      std::move(draft.fabric), std::move(draft.interconnectImplementation),
      std::move(draft.gem5BuildIdentity), std::move(draft.bridgeAbiIdentity),
      std::move(draft.correspondences));
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                         bytes.size());
}

} // namespace

Gem5ModelContractDescriptorRef gem5ModelContractDescriptorRef(
    const Gem5ModelContractDescriptor &descriptor) {
  return {descriptor.descriptor.identity.str(), descriptor.descriptor.version};
}

llvm::Error
registerGem5ModelContract(const Gem5ModelContractDescriptor &descriptor) {
  if (descriptor.descriptor.identity.empty())
    return invalid("model contract identity is empty");
  if (descriptor.semanticIdentity.empty())
    return invalid("model contract semantic identity is empty");
  if (descriptor.simObjectClass.empty())
    return invalid("model contract SimObject class is empty");
  if (!descriptor.validateCanonicalObjectPayload)
    return invalid("model contract has no object payload validator");
  if (descriptor.objectClass == Gem5ModelObjectClass::Processor &&
      !descriptor.validateProcessorCompatibility)
    return invalid("processor model contract has no compatibility validator");
  if (descriptor.objectClass != Gem5ModelObjectClass::Processor &&
      descriptor.validateProcessorCompatibility)
    return invalid("non-processor model contract owns processor compatibility");
  std::set<std::uint32_t> kinds;
  std::set<std::string> names;
  for (const Gem5ModelPortKindDescriptor &port : descriptor.portKinds) {
    if (port.stableName.empty())
      return invalid("model contract port has an empty stable name");
    if (!port.validateCanonicalPayload)
      return invalid("model contract port has no payload validator");
    if (!kinds.insert(port.kind).second)
      return invalid("model contract port kind is duplicated");
    if (!names.insert(port.stableName.str()).second)
      return invalid("model contract port name is duplicated");
  }
  std::lock_guard<std::mutex> lock(modelContractMutex());
  const auto reference = gem5ModelContractDescriptorRef(descriptor);
  for (const Gem5ModelContractDescriptor *existing : modelContracts()) {
    if (existing == &descriptor)
      return llvm::Error::success();
    if (gem5ModelContractDescriptorRef(*existing) == reference)
      return invalid("an exact model contract already has an owner");
  }
  modelContracts().push_back(&descriptor);
  return llvm::Error::success();
}

const Gem5ModelContractDescriptor *findGem5ModelContract(
    const Gem5ModelContractDescriptorRef &reference) {
  std::lock_guard<std::mutex> lock(modelContractMutex());
  auto found = llvm::find_if(modelContracts(), [&](const auto *descriptor) {
    return gem5ModelContractDescriptorRef(*descriptor) == reference;
  });
  return found == modelContracts().end() ? nullptr : *found;
}

const Gem5ModelPortKindDescriptor *findGem5ModelPortKind(
    const Gem5ModelContractDescriptor &descriptor, std::uint32_t kind) {
  auto found = llvm::find_if(descriptor.portKinds,
                             [&](const auto &port) { return port.kind == kind; });
  return found == descriptor.portKinds.end() ? nullptr : &*found;
}

llvm::Expected<FinalizedGem5SimulationBinding>
finalizeGem5SimulationBinding(Gem5SimulationBindingDraft draft,
                              const ArtifactStore &artifacts) {
  auto binding = canonicalize(std::move(draft), artifacts);
  if (!binding)
    return binding.takeError();
  const std::string json = detail::serializeGem5SimulationBinding(*binding);
  auto parsed = detail::parseGem5SimulationBinding(json);
  if (!parsed)
    return parsed.takeError();
  auto strict = canonicalize(std::move(*parsed), artifacts);
  if (!strict)
    return strict.takeError();
  if (detail::serializeGem5SimulationBinding(*strict) != json)
    return invalid("canonical binding failed independent roundtrip");
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
  auto identity = artifacts.put(gem5SimulationBindingSchema, bytes);
  if (!identity)
    return identity.takeError();
  return importGem5SimulationBinding(
      {gem5SimulationBindingSchema.identity.str(),
       gem5SimulationBindingSchema.version, *identity},
      artifacts);
}

llvm::Expected<FinalizedGem5SimulationBinding>
importGem5SimulationBinding(const ArtifactRootReference &reference,
                            const ArtifactStore &artifacts) {
  if (reference.schemaIdentity != gem5SimulationBindingSchema.identity ||
      reference.schemaVersion != gem5SimulationBindingSchema.version)
    return invalid("reference has the wrong schema descriptor");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto draft = detail::parseGem5SimulationBinding(asText(bytes->bytes()));
  if (!draft)
    return draft.takeError();
  auto binding = canonicalize(std::move(*draft), artifacts);
  if (!binding)
    return binding.takeError();
  if (detail::serializeGem5SimulationBinding(*binding) !=
      asText(bytes->bytes()))
    return invalid("stored root is not canonical");
  return FinalizedGem5SimulationBinding(reference, std::move(*bytes),
                                        std::move(*binding));
}

} // namespace loom::runtime
