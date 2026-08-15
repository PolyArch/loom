#include "Runtime/RuntimePlatformBinding.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::runtime {

namespace detail {

class RuntimePlatformBindingBuilder final {
public:
  static RuntimePlatformBinding
  create(ArtifactRootReference hardwareImplementation,
         RuntimeProviderBinding providerBinding,
         RuntimeIdentityVerification identityVerification,
         std::vector<RuntimeProgrammingBinding> programmingBindings,
         std::vector<RuntimeInterfaceBinding> memoryInterfaceBindings,
         std::vector<RuntimeInterfaceBinding> completionInterfaceBindings);
};

} // namespace detail

namespace {

struct ParsedBinding final {
  RuntimePlatformBindingDraft draft;
  RuntimeProviderBinding providerBinding;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "runtime_platform_binding_invalid: " +
                                     message);
}

llvm::Error
rejectUnknownFields(const llvm::json::Object &object, llvm::StringRef context,
                    std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &entry : object) {
    if (std::find(allowed.begin(), allowed.end(),
                  llvm::StringRef(entry.first)) == allowed.end())
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(entry.first) + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  auto result = value->getAsString();
  if (!result)
    return invalid(context + " field '" + field + "' must be a string");
  return *result;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  auto result = value->getAsUINT64();
  if (!result)
    return invalid(context + " field '" + field +
                   "' must be an unsigned integer");
  return *result;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  const llvm::json::Object *result = value->getAsObject();
  if (!result)
    return invalid(context + " field '" + field + "' must be an object");
  return result;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field,
             llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  const llvm::json::Array *result = value->getAsArray();
  if (!result)
    return invalid(context + " field '" + field + "' must be an array");
  return result;
}

void writeRootReference(llvm::json::OStream &json,
                        const ArtifactRootReference &reference) {
  json.object([&] {
    json.attribute("schema", reference.schemaIdentity);
    json.attribute("version", formatSchemaVersion(reference.schemaVersion));
    json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
  });
}

llvm::Expected<ArtifactRootReference>
parseRootReference(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"schema", "version", "artifact"}))
    return std::move(error);
  auto schema = requireString(object, "schema", context);
  auto version = requireString(object, "version", context);
  auto artifact = requireString(object, "artifact", context);
  if (!schema)
    return schema.takeError();
  if (!version)
    return version.takeError();
  if (!artifact)
    return artifact.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

void writeProviderEndpoint(llvm::json::OStream &json,
                           const RuntimeProviderEndpointRef &endpoint) {
  json.object([&] {
    json.attribute("kind", endpoint.kind);
    json.attribute("payload", formatArtifactLocalPayloadHex(endpoint.payload));
  });
}

llvm::Expected<RuntimeProviderEndpointRef>
parseProviderEndpoint(const llvm::json::Object &object,
                      llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"kind", "payload"}))
    return std::move(error);
  auto kind = requireUnsigned(object, "kind", context);
  auto payload = requireString(object, "payload", context);
  if (!kind)
    return kind.takeError();
  if (*kind > std::numeric_limits<std::uint32_t>::max())
    return invalid(context + " endpoint kind is out of range");
  if (!payload)
    return payload.takeError();
  auto parsedPayload = parseArtifactLocalPayloadHex(*payload);
  if (!parsedPayload)
    return parsedPayload.takeError();
  return RuntimeProviderEndpointRef{static_cast<std::uint32_t>(*kind),
                                    std::move(*parsedPayload)};
}

void writeInterfaceReference(
    llvm::json::OStream &json,
    const ArtifactReference<hardware::HardwareImplementationInterfaceRef>
        &reference) {
  json.object([&] {
    json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
    json.attribute("ordinal", reference.entity.ordinal);
  });
}

llvm::Expected<ArtifactReference<hardware::HardwareImplementationInterfaceRef>>
parseInterfaceReference(const llvm::json::Object &object,
                        llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"artifact", "ordinal"}))
    return std::move(error);
  auto artifact = requireString(object, "artifact", context);
  auto ordinal = requireUnsigned(object, "ordinal", context);
  if (!artifact)
    return artifact.takeError();
  if (!ordinal)
    return ordinal.takeError();
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactReference<hardware::HardwareImplementationInterfaceRef>{
      std::move(*parsedArtifact),
      hardware::HardwareImplementationInterfaceRef{*ordinal}};
}

void writeProgrammingUnit(llvm::json::OStream &json,
                          const hardware::ProgrammingUnitRef &reference) {
  json.object([&] {
    json.attributeBegin("configuration_abi_ref");
    writeRootReference(json, reference.configurationAbi);
    json.attributeEnd();
    json.attribute("unit_id", reference.unitId);
  });
}

llvm::Expected<hardware::ProgrammingUnitRef>
parseProgrammingUnit(const llvm::json::Object &object,
                     llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"configuration_abi_ref", "unit_id"}))
    return std::move(error);
  auto abiObject = requireObject(object, "configuration_abi_ref", context);
  auto unit = requireUnsigned(object, "unit_id", context);
  if (!abiObject)
    return abiObject.takeError();
  if (!unit)
    return unit.takeError();
  auto abi = parseRootReference(**abiObject,
                                (context + ".configuration_abi_ref").str());
  if (!abi)
    return abi.takeError();
  return hardware::ProgrammingUnitRef{std::move(*abi), *unit};
}

void writeIdentityVerification(llvm::json::OStream &json,
                               const RuntimeIdentityVerification &identity) {
  json.object([&] {
    if (const auto *reported =
            std::get_if<HardwareReportedIdentity>(&identity)) {
      json.attribute("kind", "HardwareReported");
      json.attributeBegin("implementation_identity_endpoint_ref");
      writeProviderEndpoint(json, reported->implementationIdentityEndpoint);
      json.attributeEnd();
      return;
    }
    json.attribute("kind", "TrustedImmutable");
    json.attribute(
        "attestation_blob",
        formatBlobDigestHex(
            std::get<TrustedImmutableIdentity>(identity).attestationBlob));
  });
}

llvm::Expected<RuntimeIdentityVerification>
parseIdentityVerification(const llvm::json::Object &object) {
  auto kind = requireString(object, "kind", "identity_verification");
  if (!kind)
    return kind.takeError();
  if (*kind == "HardwareReported") {
    if (llvm::Error error = rejectUnknownFields(
            object, "identity_verification",
            {"kind", "implementation_identity_endpoint_ref"}))
      return std::move(error);
    auto endpointObject =
        requireObject(object, "implementation_identity_endpoint_ref",
                      "identity_verification");
    if (!endpointObject)
      return endpointObject.takeError();
    auto endpoint = parseProviderEndpoint(
        **endpointObject,
        "identity_verification.implementation_identity_endpoint_ref");
    if (!endpoint)
      return endpoint.takeError();
    return RuntimeIdentityVerification(
        HardwareReportedIdentity{std::move(*endpoint)});
  }
  if (*kind == "TrustedImmutable") {
    if (llvm::Error error = rejectUnknownFields(object, "identity_verification",
                                                {"kind", "attestation_blob"}))
      return std::move(error);
    auto digest =
        requireString(object, "attestation_blob", "identity_verification");
    if (!digest)
      return digest.takeError();
    auto parsed = parseBlobDigestHex(*digest);
    if (!parsed)
      return parsed.takeError();
    return RuntimeIdentityVerification(
        TrustedImmutableIdentity{std::move(*parsed)});
  }
  return invalid("identity_verification has unknown kind '" + *kind + "'");
}

void writeProgrammingBinding(llvm::json::OStream &json,
                             const RuntimeProgrammingBinding &binding) {
  json.object([&] {
    json.attributeBegin("programming_unit_ref");
    writeProgrammingUnit(json, binding.programmingUnit);
    json.attributeEnd();
    json.attributeBegin("implementation_interface_ref");
    writeInterfaceReference(json, binding.implementationInterface);
    json.attributeEnd();
    json.attributeBegin("provider_endpoint_ref");
    writeProviderEndpoint(json, binding.providerEndpoint);
    json.attributeEnd();
  });
}

llvm::Expected<RuntimeProgrammingBinding>
parseProgrammingBinding(const llvm::json::Object &object,
                        llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(object, context,
                                              {"programming_unit_ref",
                                               "implementation_interface_ref",
                                               "provider_endpoint_ref"}))
    return std::move(error);
  auto unitObject = requireObject(object, "programming_unit_ref", context);
  auto interfaceObject =
      requireObject(object, "implementation_interface_ref", context);
  auto endpointObject = requireObject(object, "provider_endpoint_ref", context);
  if (!unitObject)
    return unitObject.takeError();
  if (!interfaceObject)
    return interfaceObject.takeError();
  if (!endpointObject)
    return endpointObject.takeError();
  auto unit = parseProgrammingUnit(**unitObject,
                                   (context + ".programming_unit_ref").str());
  auto interface = parseInterfaceReference(
      **interfaceObject, (context + ".implementation_interface_ref").str());
  auto endpoint = parseProviderEndpoint(
      **endpointObject, (context + ".provider_endpoint_ref").str());
  if (!unit)
    return unit.takeError();
  if (!interface)
    return interface.takeError();
  if (!endpoint)
    return endpoint.takeError();
  return RuntimeProgrammingBinding{std::move(*unit), std::move(*interface),
                                   std::move(*endpoint)};
}

void writeInterfaceBinding(llvm::json::OStream &json,
                           const RuntimeInterfaceBinding &binding) {
  json.object([&] {
    json.attributeBegin("implementation_interface_ref");
    writeInterfaceReference(json, binding.implementationInterface);
    json.attributeEnd();
    json.attributeBegin("provider_endpoint_ref");
    writeProviderEndpoint(json, binding.providerEndpoint);
    json.attributeEnd();
  });
}

llvm::Expected<RuntimeInterfaceBinding>
parseInterfaceBinding(const llvm::json::Object &object,
                      llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"implementation_interface_ref", "provider_endpoint_ref"}))
    return std::move(error);
  auto interfaceObject =
      requireObject(object, "implementation_interface_ref", context);
  auto endpointObject = requireObject(object, "provider_endpoint_ref", context);
  if (!interfaceObject)
    return interfaceObject.takeError();
  if (!endpointObject)
    return endpointObject.takeError();
  auto interface = parseInterfaceReference(
      **interfaceObject, (context + ".implementation_interface_ref").str());
  auto endpoint = parseProviderEndpoint(
      **endpointObject, (context + ".provider_endpoint_ref").str());
  if (!interface)
    return interface.takeError();
  if (!endpoint)
    return endpoint.takeError();
  return RuntimeInterfaceBinding{std::move(*interface), std::move(*endpoint)};
}

std::string serialize(const RuntimePlatformBinding &binding) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", runtimePlatformBindingSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(runtimePlatformBindingSchema.version));
    json.attributeBegin("hardware_implementation_ref");
    writeRootReference(json, binding.hardwareImplementation());
    json.attributeEnd();
    json.attributeBegin("provider_binding");
    json.object([&] {
      json.attribute("descriptor_identity",
                     binding.providerBinding().descriptor.identity);
      json.attribute(
          "descriptor_version",
          formatSchemaVersion(binding.providerBinding().descriptor.version));
      json.attribute("implementation_semantic_identity",
                     binding.providerBinding().implementationSemanticIdentity);
      json.attribute("runtime_abi_identity",
                     binding.providerBinding().runtimeAbiIdentity);
    });
    json.attributeEnd();
    json.attributeBegin("identity_verification");
    writeIdentityVerification(json, binding.identityVerification());
    json.attributeEnd();
    json.attributeArray("programming_bindings", [&] {
      for (const RuntimeProgrammingBinding &item :
           binding.programmingBindings())
        writeProgrammingBinding(json, item);
    });
    json.attributeArray("memory_interface_bindings", [&] {
      for (const RuntimeInterfaceBinding &item :
           binding.memoryInterfaceBindings())
        writeInterfaceBinding(json, item);
    });
    json.attributeArray("completion_interface_bindings", [&] {
      for (const RuntimeInterfaceBinding &item :
           binding.completionInterfaceBindings())
        writeInterfaceBinding(json, item);
    });
  });
  return output.str().str();
}

llvm::Expected<ParsedBinding> parse(llvm::StringRef canonicalJson) {
  auto parsed = llvm::json::parse(canonicalJson);
  if (!parsed)
    return invalid("stored root is not valid JSON");
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("stored root must be a JSON object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "stored root",
          {"schema", "schema_version", "hardware_implementation_ref",
           "provider_binding", "identity_verification", "programming_bindings",
           "memory_interface_bindings", "completion_interface_bindings"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "stored root");
  auto version = requireString(*root, "schema_version", "stored root");
  if (!schema)
    return schema.takeError();
  if (!version)
    return version.takeError();
  if (*schema != runtimePlatformBindingSchema.identity ||
      *version != formatSchemaVersion(runtimePlatformBindingSchema.version))
    return invalid("stored root has the wrong schema descriptor");

  auto implementationObject =
      requireObject(*root, "hardware_implementation_ref", "stored root");
  auto providerObject = requireObject(*root, "provider_binding", "stored root");
  auto identityObject =
      requireObject(*root, "identity_verification", "stored root");
  auto programmingArray =
      requireArray(*root, "programming_bindings", "stored root");
  auto memoryArray =
      requireArray(*root, "memory_interface_bindings", "stored root");
  auto completionArray =
      requireArray(*root, "completion_interface_bindings", "stored root");
  if (!implementationObject)
    return implementationObject.takeError();
  if (!providerObject)
    return providerObject.takeError();
  if (!identityObject)
    return identityObject.takeError();
  if (!programmingArray)
    return programmingArray.takeError();
  if (!memoryArray)
    return memoryArray.takeError();
  if (!completionArray)
    return completionArray.takeError();

  auto implementation =
      parseRootReference(**implementationObject, "hardware_implementation_ref");
  if (!implementation)
    return implementation.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **providerObject, "provider_binding",
          {"descriptor_identity", "descriptor_version",
           "implementation_semantic_identity", "runtime_abi_identity"}))
    return std::move(error);
  auto descriptorIdentity = requireString(
      **providerObject, "descriptor_identity", "provider_binding");
  auto descriptorVersion =
      requireString(**providerObject, "descriptor_version", "provider_binding");
  auto implementationIdentity = requireString(
      **providerObject, "implementation_semantic_identity", "provider_binding");
  auto runtimeAbiIdentity = requireString(
      **providerObject, "runtime_abi_identity", "provider_binding");
  if (!descriptorIdentity)
    return descriptorIdentity.takeError();
  if (!descriptorVersion)
    return descriptorVersion.takeError();
  if (!implementationIdentity)
    return implementationIdentity.takeError();
  if (!runtimeAbiIdentity)
    return runtimeAbiIdentity.takeError();
  auto parsedDescriptorVersion = parseSchemaVersion(*descriptorVersion);
  if (!parsedDescriptorVersion)
    return parsedDescriptorVersion.takeError();
  RuntimeProviderBinding providerBinding{
      {descriptorIdentity->str(), *parsedDescriptorVersion},
      implementationIdentity->str(),
      runtimeAbiIdentity->str()};
  auto identity = parseIdentityVerification(**identityObject);
  if (!identity)
    return identity.takeError();

  std::vector<RuntimeProgrammingBinding> programming;
  programming.reserve((*programmingArray)->size());
  for (const auto &[index, value] : llvm::enumerate(**programmingArray)) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("programming_bindings entry must be an object");
    auto item = parseProgrammingBinding(
        *object, "programming_bindings[" + std::to_string(index) + "]");
    if (!item)
      return item.takeError();
    programming.push_back(std::move(*item));
  }
  auto parseInterfaces = [&](const llvm::json::Array &array,
                             llvm::StringRef name)
      -> llvm::Expected<std::vector<RuntimeInterfaceBinding>> {
    std::vector<RuntimeInterfaceBinding> result;
    result.reserve(array.size());
    for (const auto &[index, value] : llvm::enumerate(array)) {
      const llvm::json::Object *object = value.getAsObject();
      if (!object)
        return invalid(name + " entry must be an object");
      auto item = parseInterfaceBinding(
          *object, name.str() + "[" + std::to_string(index) + "]");
      if (!item)
        return item.takeError();
      result.push_back(std::move(*item));
    }
    return result;
  };
  auto memory = parseInterfaces(**memoryArray, "memory_interface_bindings");
  auto completion =
      parseInterfaces(**completionArray, "completion_interface_bindings");
  if (!memory)
    return memory.takeError();
  if (!completion)
    return completion.takeError();
  return ParsedBinding{
      RuntimePlatformBindingDraft{std::move(*implementation),
                                  providerBinding.descriptor,
                                  std::move(*identity), std::move(programming),
                                  std::move(*memory), std::move(*completion)},
      std::move(providerBinding)};
}

bool endpointLess(const RuntimeProviderEndpointRef &lhs,
                  const RuntimeProviderEndpointRef &rhs) {
  return std::tie(lhs.kind, lhs.payload) < std::tie(rhs.kind, rhs.payload);
}

template <typename Binding>
bool bindingLess(const Binding &lhs, const Binding &rhs) {
  if (lhs.implementationInterface.entity.ordinal !=
      rhs.implementationInterface.entity.ordinal)
    return lhs.implementationInterface.entity.ordinal <
           rhs.implementationInterface.entity.ordinal;
  return endpointLess(lhs.providerEndpoint, rhs.providerEndpoint);
}

llvm::Error validateInterfaceOwner(
    const ArtifactReference<hardware::HardwareImplementationInterfaceRef> &ref,
    const hardware::FinalizedHardwareImplementation &implementation) {
  if (ref.artifact != implementation.reference().artifact)
    return invalid("implementation interface reference has a foreign owner");
  if (ref.entity.ordinal >= implementation.implementation().interfaces().size())
    return invalid("implementation interface reference is out of range");
  return llvm::Error::success();
}

RuntimeEndpointFlow
expectedDataFlow(const hardware::ImplementationDataInterfaceRef &semantic,
                 const fabric::FabricSystemRootView &system) {
  const auto *transport = semantic.endpoint.transport();
  if (!transport)
    llvm::report_fatal_error("validated Data interface is not transport");
  const auto direction =
      system.artifact().transportEndpointDirection(*transport);
  if (!direction)
    llvm::report_fatal_error("validated Data interface has no direction");
  return *direction == fabric::FabricPortDirection::Input
             ? RuntimeEndpointFlow::RuntimeToImplementation
             : RuntimeEndpointFlow::ImplementationToRuntime;
}

llvm::Error validateProviderEndpointSharing(
    llvm::ArrayRef<RuntimeProviderEndpointRef> endpoints,
    const RuntimeProviderDescriptor &provider) {
  std::vector<RuntimeProviderEndpointRef> ordered(endpoints.begin(),
                                                  endpoints.end());
  llvm::sort(ordered, endpointLess);
  for (std::size_t index = 1; index < ordered.size(); ++index) {
    if (!(ordered[index - 1] == ordered[index]))
      continue;
    const RuntimeProviderEndpointKindDescriptor *kind =
        findRuntimeEndpointKind(provider, ordered[index].kind);
    if (!kind || !kind->allowsSharedBinding)
      return invalid("provider endpoint is bound ambiguously more than once");
  }
  return llvm::Error::success();
}

llvm::Expected<RuntimePlatformBinding>
canonicalize(RuntimePlatformBindingDraft draft, const ArtifactStore &artifacts,
             const BlobStore &blobs) {
  if (draft.hardwareImplementation.schemaIdentity !=
          hardware::hardwareImplementationSchema.identity ||
      draft.hardwareImplementation.schemaVersion !=
          hardware::hardwareImplementationSchema.version)
    return invalid(
        "hardware_implementation_ref requires loom.hardware_implementation "
        "4.0");
  auto implementation = hardware::importHardwareImplementation(
      draft.hardwareImplementation, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  const RuntimeProviderDescriptor *provider =
      findRuntimeProvider(draft.providerDescriptor);
  if (!provider)
    return invalid("provider descriptor is not registered");
  RuntimeProviderBinding providerBinding{
      runtimeProviderDescriptorRef(*provider),
      provider->implementationSemanticIdentity.str(),
      provider->runtimeAbiIdentity.str()};

  if (auto *reported =
          std::get_if<HardwareReportedIdentity>(&draft.identityVerification)) {
    if (!provider->supportsHardwareReportedIdentity)
      return invalid("provider does not support HardwareReported identity");
    if (llvm::Error error = validateRuntimeProviderEndpoint(
            *provider, reported->implementationIdentityEndpoint,
            RuntimeEndpointClass::Identity,
            RuntimeEndpointFlow::ImplementationToRuntime))
      return std::move(error);
  } else {
    if (!provider->supportsTrustedImmutableIdentity)
      return invalid("provider does not support TrustedImmutable identity");
    const BlobDigest &digest =
        std::get<TrustedImmutableIdentity>(draft.identityVerification)
            .attestationBlob;
    auto blob = blobs.get(digest);
    if (!blob)
      return invalid("trusted attestation blob is unavailable: " +
                     llvm::toString(blob.takeError()));
  }

  auto fabricRoot = fabric::importEntireFabricRoot(
      implementation->implementation().fabric(), artifacts);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto system = fabric::requireSystemRoot(fabricRoot->view());
  if (!system)
    return system.takeError();
  auto abi = hardware::importConfigurationABI(
      implementation->implementation().configurationAbi(), artifacts);
  if (!abi)
    return abi.takeError();

  const auto interfaces = implementation->implementation().interfaces();
  std::vector<std::uint64_t> requiredProgramming;
  std::vector<std::uint64_t> requiredMemory;
  std::vector<std::uint64_t> requiredCompletion;
  std::set<hardware::ProgrammingUnitId> exposedUnits;
  for (const auto &[ordinal, interface] : llvm::enumerate(interfaces)) {
    if (const auto *configuration =
            std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
                &interface.semanticRef)) {
      requiredProgramming.push_back(ordinal);
      exposedUnits.insert(configuration->programmingUnit.unitId);
    } else if (std::holds_alternative<
                   hardware::ImplementationMemoryInterfaceRef>(
                   interface.semanticRef)) {
      requiredMemory.push_back(ordinal);
    } else if (std::holds_alternative<hardware::ImplementationDataInterfaceRef>(
                   interface.semanticRef) ||
               std::holds_alternative<
                   hardware::ImplementationExternalProtocolInterfaceRef>(
                   interface.semanticRef)) {
      requiredCompletion.push_back(ordinal);
    }
  }
  for (const hardware::ProgrammingUnit &unit : abi->abi().programmingUnits()) {
    const hardware::ProgrammingUnitOccurrenceScope scope =
        hardware::deriveProgrammingUnitOccurrenceScope(unit);
    const bool belongsToSubject =
        !scope.includesDirectSystemResources && scope.spatialCores.size() == 1 &&
        scope.spatialCores.front() == implementation->implementation().subject();
    if (belongsToSubject && !exposedUnits.count(unit.id))
      return invalid("HardwareImplementation omits a Configuration interface "
                     "for a subject-local programming unit");
  }

  llvm::sort(draft.programmingBindings, bindingLess<RuntimeProgrammingBinding>);
  llvm::sort(draft.memoryInterfaceBindings,
             bindingLess<RuntimeInterfaceBinding>);
  llvm::sort(draft.completionInterfaceBindings,
             bindingLess<RuntimeInterfaceBinding>);
  std::vector<std::uint64_t> actualProgramming;
  std::vector<std::uint64_t> actualMemory;
  std::vector<std::uint64_t> actualCompletion;
  std::vector<RuntimeProviderEndpointRef> allBoundEndpoints;

  for (const RuntimeProgrammingBinding &binding : draft.programmingBindings) {
    if (llvm::Error error = validateInterfaceOwner(
            binding.implementationInterface, *implementation))
      return std::move(error);
    const std::uint64_t ordinal =
        binding.implementationInterface.entity.ordinal;
    const auto *semantic =
        std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
            &interfaces[ordinal].semanticRef);
    if (!semantic)
      return invalid("programming binding targets a non-Configuration "
                     "implementation interface");
    if (binding.programmingUnit != semantic->programmingUnit)
      return invalid("programming binding unit disagrees with its "
                     "implementation interface");
    if (llvm::Error error =
            validateRuntimeProviderEndpoint(*provider, binding.providerEndpoint,
                                            RuntimeEndpointClass::Programming,
                                            RuntimeEndpointFlow::Bidirectional))
      return std::move(error);
    actualProgramming.push_back(ordinal);
    allBoundEndpoints.push_back(binding.providerEndpoint);
  }
  for (const RuntimeInterfaceBinding &binding : draft.memoryInterfaceBindings) {
    if (llvm::Error error = validateInterfaceOwner(
            binding.implementationInterface, *implementation))
      return std::move(error);
    const std::uint64_t ordinal =
        binding.implementationInterface.entity.ordinal;
    if (!std::holds_alternative<hardware::ImplementationMemoryInterfaceRef>(
            interfaces[ordinal].semanticRef))
      return invalid("memory binding targets a non-Memory implementation "
                     "interface");
    if (llvm::Error error = validateRuntimeProviderEndpoint(
            *provider, binding.providerEndpoint, RuntimeEndpointClass::Memory,
            RuntimeEndpointFlow::Bidirectional))
      return std::move(error);
    actualMemory.push_back(ordinal);
    allBoundEndpoints.push_back(binding.providerEndpoint);
  }
  for (const RuntimeInterfaceBinding &binding :
       draft.completionInterfaceBindings) {
    if (llvm::Error error = validateInterfaceOwner(
            binding.implementationInterface, *implementation))
      return std::move(error);
    const std::uint64_t ordinal =
        binding.implementationInterface.entity.ordinal;
    const auto &semantic = interfaces[ordinal].semanticRef;
    RuntimeEndpointFlow flow = RuntimeEndpointFlow::Bidirectional;
    if (const auto *data =
            std::get_if<hardware::ImplementationDataInterfaceRef>(&semantic))
      flow = expectedDataFlow(*data, *system);
    else if (!std::holds_alternative<
                 hardware::ImplementationExternalProtocolInterfaceRef>(
                 semantic))
      return invalid("completion binding targets an incompatible "
                     "implementation interface");
    if (llvm::Error error = validateRuntimeProviderEndpoint(
            *provider, binding.providerEndpoint,
            RuntimeEndpointClass::Completion, flow))
      return std::move(error);
    actualCompletion.push_back(ordinal);
    allBoundEndpoints.push_back(binding.providerEndpoint);
  }

  const auto exactCoverage = [&](llvm::ArrayRef<std::uint64_t> expected,
                                 llvm::ArrayRef<std::uint64_t> actual,
                                 llvm::StringRef name) -> llvm::Error {
    if (expected != actual)
      return invalid(name + " interface coverage is not exact");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          exactCoverage(requiredProgramming, actualProgramming, "programming"))
    return std::move(error);
  if (llvm::Error error = exactCoverage(requiredMemory, actualMemory, "memory"))
    return std::move(error);
  if (llvm::Error error =
          exactCoverage(requiredCompletion, actualCompletion, "completion"))
    return std::move(error);
  if (llvm::Error error =
          validateProviderEndpointSharing(allBoundEndpoints, *provider))
    return std::move(error);

  return detail::RuntimePlatformBindingBuilder::create(
      std::move(draft.hardwareImplementation), std::move(providerBinding),
      std::move(draft.identityVerification),
      std::move(draft.programmingBindings),
      std::move(draft.memoryInterfaceBindings),
      std::move(draft.completionInterfaceBindings));
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                         bytes.size());
}

llvm::Expected<RuntimePlatformBinding> decode(llvm::StringRef canonicalJson,
                                              const ArtifactStore &artifacts,
                                              const BlobStore &blobs) {
  auto parsed = parse(canonicalJson);
  if (!parsed)
    return parsed.takeError();
  auto binding = canonicalize(std::move(parsed->draft), artifacts, blobs);
  if (!binding)
    return binding.takeError();
  if (!(binding->providerBinding() == parsed->providerBinding))
    return invalid("stored provider binding disagrees with its descriptor");
  if (serialize(*binding) != canonicalJson)
    return invalid("stored root is not canonical");
  return binding;
}

} // namespace

namespace detail {

RuntimePlatformBinding RuntimePlatformBindingBuilder::create(
    ArtifactRootReference hardwareImplementation,
    RuntimeProviderBinding providerBinding,
    RuntimeIdentityVerification identityVerification,
    std::vector<RuntimeProgrammingBinding> programmingBindings,
    std::vector<RuntimeInterfaceBinding> memoryInterfaceBindings,
    std::vector<RuntimeInterfaceBinding> completionInterfaceBindings) {
  return RuntimePlatformBinding(
      std::move(hardwareImplementation), std::move(providerBinding),
      std::move(identityVerification), std::move(programmingBindings),
      std::move(memoryInterfaceBindings),
      std::move(completionInterfaceBindings));
}

} // namespace detail

llvm::Expected<FinalizedRuntimePlatformBinding>
finalizeRuntimePlatformBinding(RuntimePlatformBindingDraft draft,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  auto binding = canonicalize(std::move(draft), artifacts, blobs);
  if (!binding)
    return binding.takeError();
  const std::string json = serialize(*binding);
  auto strict = decode(json, artifacts, blobs);
  if (!strict)
    return strict.takeError();
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
  auto identity = artifacts.put(runtimePlatformBindingSchema, bytes);
  if (!identity)
    return identity.takeError();
  return importRuntimePlatformBinding(
      {runtimePlatformBindingSchema.identity.str(),
       runtimePlatformBindingSchema.version, *identity},
      artifacts, blobs);
}

llvm::Expected<FinalizedRuntimePlatformBinding>
importRuntimePlatformBinding(const ArtifactRootReference &reference,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  if (reference.schemaIdentity != runtimePlatformBindingSchema.identity ||
      reference.schemaVersion != runtimePlatformBindingSchema.version)
    return invalid("reference has the wrong schema descriptor");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto binding = decode(asText(bytes->bytes()), artifacts, blobs);
  if (!binding)
    return binding.takeError();
  return FinalizedRuntimePlatformBinding(reference, std::move(*bytes),
                                         std::move(*binding));
}

} // namespace loom::runtime
