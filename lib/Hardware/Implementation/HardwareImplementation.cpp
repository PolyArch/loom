#include "Hardware/Implementation/HardwareImplementation.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "HardwareImplementationInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware {

class detail::HardwareImplementationBuilder final {
public:
  static HardwareImplementation
  create(HardwareImplementationDraft draft,
         std::vector<MemoryMacroBinding> memoryMacroBindings,
         std::vector<ExternalImplementationBinding> externalBindings) {
    return HardwareImplementation(
        std::move(draft.fabric), std::move(draft.configurationAbi),
        std::move(draft.interconnectImplementations),
        std::move(draft.representationRoot),
        std::move(draft.implementationPlatform), std::move(draft.interfaces),
        std::move(draft.activityPoints), std::move(memoryMacroBindings),
        std::move(externalBindings));
  }
};

namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_implementation_invalid: " + message);
}

llvm::Error validateKey(llvm::StringRef value, llvm::StringRef field) {
  if (value.empty())
    return invalid(field + " must be nonempty");
  const auto allowed = [](char character) {
    const unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '.' || character == '_' ||
           character == '-' || character == ':' || character == '/';
  };
  if (!std::isalnum(static_cast<unsigned char>(value.front())) ||
      !std::isalnum(static_cast<unsigned char>(value.back())) ||
      !llvm::all_of(value, allowed))
    return invalid(field + " is not a canonical ASCII key");
  return llvm::Error::success();
}

ImplementationInterfaceSemanticRefKind
interfaceSemanticRefKind(const ImplementationInterfaceSemanticRef &reference) {
  if (std::holds_alternative<ImplementationDataInterfaceRef>(reference))
    return ImplementationInterfaceSemanticRefKind::Data;
  if (std::holds_alternative<ImplementationMemoryInterfaceRef>(reference))
    return ImplementationInterfaceSemanticRefKind::Memory;
  if (std::holds_alternative<ImplementationClockInterfaceRef>(reference))
    return ImplementationInterfaceSemanticRefKind::Clock;
  if (std::holds_alternative<ImplementationResetInterfaceRef>(reference))
    return ImplementationInterfaceSemanticRefKind::Reset;
  if (std::holds_alternative<ImplementationConfigurationInterfaceRef>(
          reference))
    return ImplementationInterfaceSemanticRefKind::Configuration;
  if (std::holds_alternative<ImplementationExternalProtocolInterfaceRef>(
          reference))
    return ImplementationInterfaceSemanticRefKind::ExternalProtocol;
  llvm_unreachable("interface semantic reference variant is closed");
}

llvm::StringRef
interfaceSemanticKind(const ImplementationInterfaceSemanticRef &reference) {
  switch (interfaceSemanticRefKind(reference)) {
  case ImplementationInterfaceSemanticRefKind::Data:
    return "Data";
  case ImplementationInterfaceSemanticRefKind::Memory:
    return "Memory";
  case ImplementationInterfaceSemanticRefKind::Clock:
    return "Clock";
  case ImplementationInterfaceSemanticRefKind::Reset:
    return "Reset";
  case ImplementationInterfaceSemanticRefKind::Configuration:
    return "Configuration";
  case ImplementationInterfaceSemanticRefKind::ExternalProtocol:
    return "ExternalProtocol";
  }
  llvm_unreachable("interface semantic reference variant is closed");
}

ByteVector attachmentEndpointBytes(
    const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport())
    return fabric::canonicalFabricBytes(*transport);
  return fabric::canonicalFabricBytes(*endpoint.memory());
}

ByteVector interfaceSemanticTargetBytes(
    const ImplementationInterfaceSemanticRef &reference) {
  switch (interfaceSemanticRefKind(reference)) {
  case ImplementationInterfaceSemanticRefKind::Data:
    return attachmentEndpointBytes(
        std::get<ImplementationDataInterfaceRef>(reference).endpoint);
  case ImplementationInterfaceSemanticRefKind::Memory:
    return attachmentEndpointBytes(
        std::get<ImplementationMemoryInterfaceRef>(reference).endpoint);
  case ImplementationInterfaceSemanticRefKind::Clock:
    return fabric::canonicalFabricBytes(
        std::get<ImplementationClockInterfaceRef>(reference).domain);
  case ImplementationInterfaceSemanticRefKind::Reset:
    return fabric::canonicalFabricBytes(
        std::get<ImplementationResetInterfaceRef>(reference).domain);
  case ImplementationInterfaceSemanticRefKind::Configuration:
    return encodeProgrammingUnitRef(
        std::get<ImplementationConfigurationInterfaceRef>(reference)
            .programmingUnit);
  case ImplementationInterfaceSemanticRefKind::ExternalProtocol:
    return fabric::canonicalFabricBytes(
        std::get<ImplementationExternalProtocolInterfaceRef>(reference)
            .boundary);
  }
  llvm_unreachable("interface semantic reference variant is closed");
}

ByteVector canonicalInterfaceSemanticBytes(
    const ImplementationInterfaceSemanticRef &reference) {
  const std::uint32_t tag = implementationInterfaceSemanticRefKindOrdinal(
      interfaceSemanticRefKind(reference));
  ByteVector bytes{static_cast<std::uint8_t>(tag >> 24),
                   static_cast<std::uint8_t>(tag >> 16),
                   static_cast<std::uint8_t>(tag >> 8),
                   static_cast<std::uint8_t>(tag)};
  ByteVector target = interfaceSemanticTargetBytes(reference);
  bytes.insert(bytes.end(), target.begin(), target.end());
  return bytes;
}

llvm::StringRef dependencyKindSpelling(ExternalDependencyKind kind) {
  switch (kind) {
  case ExternalDependencyKind::ExplicitFile:
    return "ExplicitFile";
  case ExternalDependencyKind::ToolBundledResource:
    return "ToolBundledResource";
  }
  llvm_unreachable("validated dependency kind is closed");
}

void writeRootReference(llvm::json::OStream &json,
                        const ArtifactRootReference &reference) {
  json.object([&] {
    json.attribute("schema", reference.schemaIdentity);
    json.attribute("version", formatSchemaVersion(reference.schemaVersion));
    json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
  });
}

llvm::Expected<llvm::json::Value> parseJsonValue(llvm::StringRef spelling) {
  auto value = llvm::json::parse(spelling);
  if (!value)
    return invalid("nested canonical JSON could not be parsed: " +
                   llvm::toString(value.takeError()));
  return std::move(*value);
}

void writeLocator(llvm::json::OStream &json,
                  const RepresentationLocator &locator) {
  auto spelling = serializeRepresentationLocatorJson(locator);
  if (!spelling)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(spelling.takeError())));
  auto value = parseJsonValue(*spelling);
  if (!value)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(value.takeError())));
  json.value(*value);
}

void writeRepresentationRoot(llvm::json::OStream &json,
                             const ImplementationRepresentationRoot &root) {
  auto spelling = serializeImplementationRepresentationRootJson(root);
  if (!spelling)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(spelling.takeError())));
  auto value = parseJsonValue(*spelling);
  if (!value)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(value.takeError())));
  json.value(*value);
}

std::string
physicalOwnerSpelling(const fabric::FabricPhysicalOccurrenceOwnerRef &owner) {
  return formatArtifactLocalPayloadHex(fabric::canonicalFabricBytes(owner));
}

void writeInterfaceSemanticRef(
    llvm::json::OStream &json,
    const ImplementationInterfaceSemanticRef &reference) {
  json.object([&] {
    json.attribute("kind", interfaceSemanticKind(reference));
    json.attribute("target", formatArtifactLocalPayloadHex(
                                 interfaceSemanticTargetBytes(reference)));
  });
}

void writeDependency(llvm::json::OStream &json,
                     const ExternalDependencyIdentity &identity) {
  json.object([&] {
    if (const auto *file = std::get_if<ExplicitFileDependency>(&identity)) {
      json.attribute(
          "kind", dependencyKindSpelling(ExternalDependencyKind::ExplicitFile));
      json.attribute("content_sha256",
                     formatExternalFileFingerprint(file->contentSha256));
      return;
    }
    const auto &bundled = std::get<ToolBundledResourceDependency>(identity);
    json.attribute("kind", dependencyKindSpelling(
                               ExternalDependencyKind::ToolBundledResource));
    json.attribute("stable_provider_build_identity",
                   bundled.stableProviderBuildIdentity);
    json.attribute("resource_key", bundled.resourceKey);
  });
}

std::string serialize(const HardwareImplementation &implementation) {
  llvm::SmallString<8192> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", hardwareImplementationSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(hardwareImplementationSchema.version));
    json.attributeBegin("fabric_ref");
    writeRootReference(json, implementation.fabric());
    json.attributeEnd();
    json.attributeBegin("configuration_abi_ref");
    writeRootReference(json, implementation.configurationAbi());
    json.attributeEnd();
    json.attributeArray("interconnect_implementation_refs", [&] {
      for (const ArtifactRootReference &reference :
           implementation.interconnectImplementations())
        writeRootReference(json, reference);
    });
    json.attributeBegin("representation_root");
    writeRepresentationRoot(json, implementation.representationRoot());
    json.attributeEnd();
    if (implementation.implementationPlatform()) {
      json.attributeBegin("implementation_platform_ref");
      writeRootReference(json, *implementation.implementationPlatform());
      json.attributeEnd();
    }
    json.attributeArray("interfaces", [&] {
      for (const ImplementationInterface &interface :
           implementation.interfaces()) {
        json.object([&] {
          json.attributeBegin("semantic_ref");
          writeInterfaceSemanticRef(json, interface.semanticRef);
          json.attributeEnd();
          json.attributeBegin("representation_locator");
          writeLocator(json, interface.representationLocator);
          json.attributeEnd();
          if (interface.devicePinRef)
            json.attribute("device_pin_ref", *interface.devicePinRef);
        });
      }
    });
    json.attributeArray("activity_points", [&] {
      for (const ActivityPoint &point : implementation.activityPoints()) {
        json.object([&] {
          json.attributeBegin("representation_locator");
          writeLocator(json, point.representationLocator);
          json.attributeEnd();
          if (point.semanticFabricRef)
            json.attribute("semantic_fabric_ref",
                           physicalOwnerSpelling(*point.semanticFabricRef));
        });
      }
    });
    json.attributeArray("memory_macro_bindings", [&] {
      for (const MemoryMacroBinding &binding :
           implementation.memoryMacroBindings()) {
        json.object([&] {
          json.attribute("fabric_memory_ref",
                         physicalOwnerSpelling(binding.fabricMemoryRef));
          json.attribute("external_implementation_binding_ref",
                         binding.externalImplementationBindingRef.ordinal);
          json.attributeBegin("representation_locator");
          writeLocator(json, binding.representationLocator);
          json.attributeEnd();
        });
      }
    });
    json.attributeArray("external_implementation_bindings", [&] {
      for (const ExternalImplementationBinding &binding :
           implementation.externalImplementationBindings()) {
        json.object([&] {
          json.attribute("provider_contract_ref", binding.providerContractRef);
          json.attributeArray("external_inputs", [&] {
            for (const ExternalInputBinding &input : binding.externalInputs) {
              json.object([&] {
                json.attribute("provider_input_slot_ref",
                               input.providerInputSlotRef);
                json.attributeBegin("dependency_identity");
                writeDependency(json, input.dependencyIdentity);
                json.attributeEnd();
              });
            }
          });
          json.attributeArray("fabric_resource_refs", [&] {
            for (const fabric::FabricPhysicalOccurrenceOwnerRef &reference :
                 binding.fabricResourceRefs)
              json.value(physicalOwnerSpelling(reference));
          });
          json.attributeArray("representation_locators", [&] {
            for (const RepresentationLocator &locator :
                 binding.representationLocators)
              writeLocator(json, locator);
          });
          if (binding.blackBoxContractPayloadRef)
            json.attribute("black_box_contract_payload_ref",
                           binding.blackBoxContractPayloadRef->ordinal);
        });
      }
    });
  });
  return output.str().str();
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &field : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(field.getFirst())))
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(field.getFirst()) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return invalid(context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(key);
  if (!value)
    return invalid(context + " requires object field '" + key + "'");
  return value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return invalid(context + " requires array field '" + key + "'");
  return value;
}

llvm::Expected<std::uint64_t> requireOrdinal(const llvm::json::Object &object,
                                             llvm::StringRef key,
                                             llvm::StringRef context) {
  std::optional<std::int64_t> value = object.getInteger(key);
  if (!value || *value < 0)
    return invalid(context + " requires nonnegative integer field '" + key +
                   "'");
  return static_cast<std::uint64_t>(*value);
}

llvm::Expected<ArtifactRootReference>
parseRootReference(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"schema", "version", "artifact"}))
    return std::move(error);
  auto schema = requireString(object, "schema", context);
  if (!schema)
    return schema.takeError();
  auto version = requireString(object, "version", context);
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  auto artifact = requireString(object, "artifact", context);
  if (!artifact)
    return artifact.takeError();
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

llvm::Expected<fabric::FabricPhysicalOccurrenceOwnerRef>
parsePhysicalOwner(llvm::StringRef spelling, llvm::StringRef context) {
  auto bytes = parseArtifactLocalPayloadHex(spelling);
  if (!bytes)
    return invalid(
        context + " has malformed bytes: " + llvm::toString(bytes.takeError()));
  auto owner =
      fabric::decodeFabricRef<fabric::FabricPhysicalOccurrenceOwnerRef>(*bytes);
  if (!owner)
    return invalid(context + " has malformed Fabric reference: " +
                   llvm::toString(owner.takeError()));
  return std::move(*owner);
}

llvm::Expected<ImplementationInterfaceSemanticRef>
parseInterfaceSemanticRef(const llvm::json::Object &object) {
  if (llvm::Error error = rejectUnknownFields(
          object, "interface semantic reference", {"kind", "target"}))
    return std::move(error);
  auto kind = requireString(object, "kind", "interface semantic reference");
  auto target = requireString(object, "target", "interface semantic reference");
  if (!kind || !target)
    return invalid("interface semantic reference is incomplete");
  auto bytes = parseArtifactLocalPayloadHex(*target);
  if (!bytes)
    return invalid("interface semantic reference has malformed bytes: " +
                   llvm::toString(bytes.takeError()));

  if (*kind == "Data") {
    auto endpoint =
        fabric::decodeFabricRef<fabric::FabricTransportEndpointRef>(*bytes);
    if (!endpoint)
      return invalid("Data interface target is malformed: " +
                     llvm::toString(endpoint.takeError()));
    auto attachment =
        fabric::FabricSpatialAttachmentEndpointRef::create(*endpoint);
    if (!attachment)
      return invalid("Data interface target is invalid: " +
                     llvm::toString(attachment.takeError()));
    return ImplementationInterfaceSemanticRef(
        ImplementationDataInterfaceRef{std::move(*attachment)});
  }
  if (*kind == "Memory") {
    auto endpoint =
        fabric::decodeFabricRef<fabric::FabricMemoryEndpointRef>(*bytes);
    if (!endpoint)
      return invalid("Memory interface target is malformed: " +
                     llvm::toString(endpoint.takeError()));
    auto attachment =
        fabric::FabricSpatialAttachmentEndpointRef::create(*endpoint);
    if (!attachment)
      return invalid("Memory interface target is invalid: " +
                     llvm::toString(attachment.takeError()));
    return ImplementationInterfaceSemanticRef(
        ImplementationMemoryInterfaceRef{std::move(*attachment)});
  }
  if (*kind == "Clock" || *kind == "Reset") {
    auto domain = fabric::decodeFabricRef<fabric::HardwareDomainRef>(*bytes);
    if (!domain)
      return invalid(*kind + " interface target is malformed: " +
                     llvm::toString(domain.takeError()));
    if (*kind == "Clock")
      return ImplementationInterfaceSemanticRef(
          ImplementationClockInterfaceRef{*domain});
    return ImplementationInterfaceSemanticRef(
        ImplementationResetInterfaceRef{*domain});
  }
  if (*kind == "Configuration") {
    auto unit = detail::decodeProgrammingUnitRefFraming(*bytes);
    if (!unit)
      return invalid("Configuration interface target is malformed: " +
                     llvm::toString(unit.takeError()));
    return ImplementationInterfaceSemanticRef(
        ImplementationConfigurationInterfaceRef{std::move(*unit)});
  }
  if (*kind == "ExternalProtocol") {
    auto boundary =
        fabric::decodeFabricRef<fabric::ExternalBoundaryRef>(*bytes);
    if (!boundary)
      return invalid("ExternalProtocol interface target is malformed: " +
                     llvm::toString(boundary.takeError()));
    return ImplementationInterfaceSemanticRef(
        ImplementationExternalProtocolInterfaceRef{*boundary});
  }
  return invalid("interface semantic reference has an unknown kind");
}

llvm::Expected<RepresentationLocator>
parseLocator(const llvm::json::Object &object, llvm::StringRef context) {
  auto locator = parseRepresentationLocatorJsonValue(object);
  if (!locator)
    return invalid(context +
                   " is invalid: " + llvm::toString(locator.takeError()));
  return std::move(*locator);
}

llvm::Expected<ImplementationRepresentationRoot>
parseRepresentationRoot(const llvm::json::Object &object) {
  auto root = parseImplementationRepresentationRootJsonValue(object);
  if (!root)
    return invalid("representation_root is invalid: " +
                   llvm::toString(root.takeError()));
  return std::move(*root);
}

llvm::Expected<ExternalDependencyIdentity>
parseDependency(const llvm::json::Object &object) {
  auto kind = requireString(object, "kind", "dependency identity");
  if (!kind)
    return kind.takeError();
  if (*kind == dependencyKindSpelling(ExternalDependencyKind::ExplicitFile)) {
    if (llvm::Error error = rejectUnknownFields(object, "dependency identity",
                                                {"kind", "content_sha256"}))
      return std::move(error);
    auto spelling =
        requireString(object, "content_sha256", "dependency identity");
    if (!spelling)
      return spelling.takeError();
    auto fingerprint = parseExternalFileFingerprint(*spelling);
    if (!fingerprint)
      return fingerprint.takeError();
    return ExternalDependencyIdentity(ExplicitFileDependency{*fingerprint});
  }
  if (*kind !=
      dependencyKindSpelling(ExternalDependencyKind::ToolBundledResource))
    return invalid("dependency identity has an unknown kind");
  if (llvm::Error error = rejectUnknownFields(
          object, "dependency identity",
          {"kind", "stable_provider_build_identity", "resource_key"}))
    return std::move(error);
  auto build = requireString(object, "stable_provider_build_identity",
                             "dependency identity");
  if (!build)
    return build.takeError();
  auto resource = requireString(object, "resource_key", "dependency identity");
  if (!resource)
    return resource.takeError();
  return ExternalDependencyIdentity(
      ToolBundledResourceDependency{build->str(), resource->str()});
}

llvm::Expected<HardwareImplementationDraft> parse(llvm::StringRef body) {
  auto parsed = llvm::json::parse(body);
  if (!parsed)
    return invalid("root is not valid JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("root must be a JSON object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "root",
          {"schema", "schema_version", "fabric_ref", "configuration_abi_ref",
           "interconnect_implementation_refs", "representation_root",
           "implementation_platform_ref", "interfaces", "activity_points",
           "memory_macro_bindings", "external_implementation_bindings"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "root");
  auto version = requireString(*root, "schema_version", "root");
  if (!schema || !version)
    return !schema ? schema.takeError() : version.takeError();
  if (*schema != hardwareImplementationSchema.identity ||
      *version != formatSchemaVersion(hardwareImplementationSchema.version))
    return invalid("root schema is not loom.hardware_implementation 2.2");

  auto fabricObject = requireObject(*root, "fabric_ref", "root");
  auto abiObject = requireObject(*root, "configuration_abi_ref", "root");
  if (!fabricObject || !abiObject)
    return !fabricObject ? fabricObject.takeError() : abiObject.takeError();
  auto fabric = parseRootReference(**fabricObject, "fabric_ref");
  auto abi = parseRootReference(**abiObject, "configuration_abi_ref");
  if (!fabric || !abi)
    return !fabric ? fabric.takeError() : abi.takeError();

  std::vector<ArtifactRootReference> interconnects;
  auto interconnectArray =
      requireArray(*root, "interconnect_implementation_refs", "root");
  if (!interconnectArray)
    return interconnectArray.takeError();
  for (const llvm::json::Value &value : **interconnectArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("interconnect reference must be an object");
    auto reference = parseRootReference(*object, "interconnect reference");
    if (!reference)
      return reference.takeError();
    interconnects.push_back(std::move(*reference));
  }

  auto representationObject =
      requireObject(*root, "representation_root", "root");
  if (!representationObject)
    return representationObject.takeError();
  auto representation = parseRepresentationRoot(**representationObject);
  if (!representation)
    return representation.takeError();

  std::optional<ArtifactRootReference> platform;
  if (const llvm::json::Object *object =
          root->getObject("implementation_platform_ref")) {
    auto reference = parseRootReference(*object, "implementation_platform_ref");
    if (!reference)
      return reference.takeError();
    platform = std::move(*reference);
  }

  std::vector<ImplementationInterface> interfaces;
  auto interfaceArray = requireArray(*root, "interfaces", "root");
  if (!interfaceArray)
    return interfaceArray.takeError();
  for (const llvm::json::Value &value : **interfaceArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("interface must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, "interface",
            {"semantic_ref", "representation_locator", "device_pin_ref"}))
      return std::move(error);
    auto semanticObject = requireObject(*object, "semantic_ref", "interface");
    auto locatorObject =
        requireObject(*object, "representation_locator", "interface");
    if (!semanticObject || !locatorObject)
      return invalid("interface is incomplete");
    auto semantic = parseInterfaceSemanticRef(**semanticObject);
    auto locator = parseLocator(**locatorObject, "interface locator");
    if (!semantic || !locator)
      return !semantic ? semantic.takeError() : locator.takeError();
    std::optional<std::string> pin;
    if (std::optional<llvm::StringRef> value =
            object->getString("device_pin_ref"))
      pin = value->str();
    interfaces.push_back(ImplementationInterface{
        std::move(*semantic), std::move(*locator), std::move(pin)});
  }

  std::vector<ActivityPoint> activityPoints;
  auto activityArray = requireArray(*root, "activity_points", "root");
  if (!activityArray)
    return activityArray.takeError();
  for (const llvm::json::Value &value : **activityArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("activity point must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, "activity point",
            {"representation_locator", "semantic_fabric_ref"}))
      return std::move(error);
    auto locatorObject =
        requireObject(*object, "representation_locator", "activity point");
    if (!locatorObject)
      return locatorObject.takeError();
    auto locator = parseLocator(**locatorObject, "activity point locator");
    if (!locator)
      return locator.takeError();
    std::optional<fabric::FabricPhysicalOccurrenceOwnerRef> owner;
    if (std::optional<llvm::StringRef> spelling =
            object->getString("semantic_fabric_ref")) {
      auto parsedOwner = parsePhysicalOwner(*spelling, "activity semantic ref");
      if (!parsedOwner)
        return parsedOwner.takeError();
      owner = std::move(*parsedOwner);
    }
    activityPoints.push_back(
        ActivityPoint{std::move(*locator), std::move(owner)});
  }

  std::vector<ExternalImplementationBindingDraft> externalBindings;
  auto externalArray =
      requireArray(*root, "external_implementation_bindings", "root");
  if (!externalArray)
    return externalArray.takeError();
  for (const llvm::json::Value &value : **externalArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("external implementation binding must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, "external implementation binding",
            {"provider_contract_ref", "external_inputs", "fabric_resource_refs",
             "representation_locators", "black_box_contract_payload_ref"}))
      return std::move(error);
    auto contract = requireString(*object, "provider_contract_ref",
                                  "external implementation binding");
    auto inputArray = requireArray(*object, "external_inputs",
                                   "external implementation binding");
    auto ownerArray = requireArray(*object, "fabric_resource_refs",
                                   "external implementation binding");
    auto locatorArray = requireArray(*object, "representation_locators",
                                     "external implementation binding");
    if (!contract || !inputArray || !ownerArray || !locatorArray)
      return invalid("external implementation binding is incomplete");
    ExternalImplementationBindingDraft binding;
    binding.providerContractRef = contract->str();
    for (const llvm::json::Value &inputValue : **inputArray) {
      const llvm::json::Object *input = inputValue.getAsObject();
      if (!input)
        return invalid("external input must be an object");
      if (llvm::Error error = rejectUnknownFields(
              *input, "external input",
              {"provider_input_slot_ref", "dependency_identity"}))
        return std::move(error);
      auto slot =
          requireString(*input, "provider_input_slot_ref", "external input");
      auto dependencyObject =
          requireObject(*input, "dependency_identity", "external input");
      if (!slot || !dependencyObject)
        return invalid("external input is incomplete");
      auto dependency = parseDependency(**dependencyObject);
      if (!dependency)
        return dependency.takeError();
      binding.externalInputs.push_back(
          ExternalInputBinding{slot->str(), std::move(*dependency)});
    }
    for (const llvm::json::Value &ownerValue : **ownerArray) {
      std::optional<llvm::StringRef> spelling = ownerValue.getAsString();
      if (!spelling)
        return invalid("external Fabric resource ref must be a string");
      auto owner =
          parsePhysicalOwner(*spelling, "external Fabric resource ref");
      if (!owner)
        return owner.takeError();
      binding.fabricResourceRefs.push_back(std::move(*owner));
    }
    for (const llvm::json::Value &locatorValue : **locatorArray) {
      const llvm::json::Object *locatorObject = locatorValue.getAsObject();
      if (!locatorObject)
        return invalid("external locator must be an object");
      auto locator = parseLocator(*locatorObject, "external locator");
      if (!locator)
        return locator.takeError();
      binding.representationLocators.push_back(std::move(*locator));
    }
    if (object->get("black_box_contract_payload_ref")) {
      auto ordinal = requireOrdinal(*object, "black_box_contract_payload_ref",
                                    "external implementation binding");
      if (!ordinal)
        return ordinal.takeError();
      if (*ordinal >= representation->payloads.size())
        return invalid("black-box payload reference is out of range");
      const ImplementationPayload &payload =
          representation->payloads[static_cast<std::size_t>(*ordinal)];
      binding.blackBoxContractPayload =
          ImplementationPayloadKey{payload.role, payload.canonicalLogicalName};
    }
    externalBindings.push_back(std::move(binding));
  }

  std::vector<MemoryMacroBindingDraft> memoryBindings;
  auto memoryArray = requireArray(*root, "memory_macro_bindings", "root");
  if (!memoryArray)
    return memoryArray.takeError();
  for (const llvm::json::Value &value : **memoryArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("memory macro binding must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, "memory macro binding",
            {"fabric_memory_ref", "external_implementation_binding_ref",
             "representation_locator"}))
      return std::move(error);
    auto ownerText =
        requireString(*object, "fabric_memory_ref", "memory macro binding");
    auto ordinal = requireOrdinal(
        *object, "external_implementation_binding_ref", "memory macro binding");
    auto locatorObject = requireObject(*object, "representation_locator",
                                       "memory macro binding");
    if (!ownerText || !ordinal || !locatorObject)
      return invalid("memory macro binding is incomplete");
    auto owner = parsePhysicalOwner(*ownerText, "memory macro Fabric ref");
    auto locator = parseLocator(**locatorObject, "memory macro locator");
    if (!owner || !locator)
      return !owner ? owner.takeError() : locator.takeError();
    memoryBindings.push_back(MemoryMacroBindingDraft{
        std::move(*owner), *ordinal, std::move(*locator)});
  }

  return HardwareImplementationDraft{
      std::move(*fabric),         std::move(*abi),
      std::move(interconnects),   std::move(*representation),
      std::move(platform),        std::move(interfaces),
      std::move(activityPoints),  std::move(memoryBindings),
      std::move(externalBindings)};
}

bool rootReferenceLess(const ArtifactRootReference &lhs,
                       const ArtifactRootReference &rhs) {
  return std::tie(lhs.schemaIdentity, lhs.schemaVersion.major,
                  lhs.schemaVersion.minor, lhs.artifact.bytes()) <
         std::tie(rhs.schemaIdentity, rhs.schemaVersion.major,
                  rhs.schemaVersion.minor, rhs.artifact.bytes());
}

bool interfaceLess(const ImplementationInterface &lhs,
                   const ImplementationInterface &rhs) {
  return std::tuple(canonicalInterfaceSemanticBytes(lhs.semanticRef),
                    lhs.representationLocator.kind,
                    lhs.representationLocator.canonicalName, lhs.devicePinRef) <
         std::tuple(canonicalInterfaceSemanticBytes(rhs.semanticRef),
                    rhs.representationLocator.kind,
                    rhs.representationLocator.canonicalName, rhs.devicePinRef);
}

bool activityLess(const ActivityPoint &lhs, const ActivityPoint &rhs) {
  const ByteVector lhsOwner =
      lhs.semanticFabricRef
          ? fabric::canonicalFabricBytes(*lhs.semanticFabricRef)
          : ByteVector();
  const ByteVector rhsOwner =
      rhs.semanticFabricRef
          ? fabric::canonicalFabricBytes(*rhs.semanticFabricRef)
          : ByteVector();
  return std::tuple(lhs.representationLocator.kind,
                    lhs.representationLocator.canonicalName,
                    lhs.semanticFabricRef.has_value(), lhsOwner) <
         std::tuple(rhs.representationLocator.kind,
                    rhs.representationLocator.canonicalName,
                    rhs.semanticFabricRef.has_value(), rhsOwner);
}

llvm::Expected<ImplementationPayloadRef>
resolvePayload(const ImplementationRepresentationRoot &root,
               const ImplementationPayloadKey &key) {
  auto found =
      llvm::find_if(root.payloads, [&](const ImplementationPayload &p) {
        return p.role == key.role &&
               p.canonicalLogicalName == key.canonicalLogicalName;
      });
  if (found == root.payloads.end())
    return invalid("BlackBoxContract payload reference is unresolved");
  return ImplementationPayloadRef{
      static_cast<std::uint64_t>(found - root.payloads.begin())};
}

llvm::Expected<std::vector<ExternalImplementationBinding>>
finalizeExternalBindings(
    llvm::ArrayRef<ExternalImplementationBindingDraft> bindings,
    const ImplementationRepresentationRoot &root) {
  std::vector<ExternalImplementationBinding> finalized;
  finalized.reserve(bindings.size());
  for (const ExternalImplementationBindingDraft &binding : bindings) {
    std::optional<ImplementationPayloadRef> payload;
    if (binding.blackBoxContractPayload) {
      auto resolved = resolvePayload(root, *binding.blackBoxContractPayload);
      if (!resolved)
        return resolved.takeError();
      if (root.payloads[static_cast<std::size_t>(resolved->ordinal)].role !=
          PayloadRole::BlackBoxContract)
        return invalid("external payload reference is not a BlackBoxContract");
      payload = *resolved;
    }
    finalized.push_back(ExternalImplementationBinding{
        binding.providerContractRef, binding.externalInputs,
        binding.fabricResourceRefs, binding.representationLocators,
        std::move(payload)});
  }
  return finalized;
}

llvm::Error requireIndexedLocator(const RepresentationIndex &index,
                                  const RepresentationLocator &locator,
                                  llvm::StringRef context) {
  auto facts = index.lookup(locator);
  if (!facts)
    return facts.takeError();
  if (!*facts)
    return invalid(context + " is absent from the representation index");
  return llvm::Error::success();
}

bool hasSpatialAttachment(
    const fabric::FabricSystemRootView &system,
    const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  return llvm::any_of(system.spatialAttachments(), [&](const auto &attachment) {
    return attachment.spatialEndpoint == endpoint;
  });
}

llvm::Error validateInterfaceSemanticRef(
    const ImplementationInterfaceSemanticRef &reference,
    const fabric::FabricSystemRootView &system, const ConfigurationABI &abi,
    const ArtifactRootReference &abiReference) {
  switch (interfaceSemanticRefKind(reference)) {
  case ImplementationInterfaceSemanticRefKind::Data: {
    const auto &endpoint =
        std::get<ImplementationDataInterfaceRef>(reference).endpoint;
    if (endpoint.plane() !=
        fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalid("Data interface target is not on the Transport plane");
    if (!hasSpatialAttachment(system, endpoint))
      return invalid("Data interface target is absent from the exact System");
    return llvm::Error::success();
  }
  case ImplementationInterfaceSemanticRefKind::Memory: {
    const auto &endpoint =
        std::get<ImplementationMemoryInterfaceRef>(reference).endpoint;
    if (endpoint.plane() !=
        fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory)
      return invalid("Memory interface target is not on the Memory plane");
    if (!hasSpatialAttachment(system, endpoint))
      return invalid("Memory interface target is absent from the exact System");
    return llvm::Error::success();
  }
  case ImplementationInterfaceSemanticRefKind::Clock: {
    const auto domain =
        std::get<ImplementationClockInterfaceRef>(reference).domain;
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract ||
        contract->kind() != fabric::FabricHardwareDomainKind::Clock)
      return invalid("Clock interface target is not an exact Clock domain");
    return llvm::Error::success();
  }
  case ImplementationInterfaceSemanticRefKind::Reset: {
    const auto domain =
        std::get<ImplementationResetInterfaceRef>(reference).domain;
    const auto *contract = system.hardwareDomainContract(domain);
    if (!contract ||
        contract->kind() != fabric::FabricHardwareDomainKind::Reset)
      return invalid("Reset interface target is not an exact Reset domain");
    return llvm::Error::success();
  }
  case ImplementationInterfaceSemanticRefKind::Configuration: {
    const ProgrammingUnitRef &unit =
        std::get<ImplementationConfigurationInterfaceRef>(reference)
            .programmingUnit;
    if (unit.configurationAbi != abiReference)
      return invalid("Configuration interface references a foreign ABI");
    if (!abi.findProgrammingUnit(unit.unitId))
      return invalid("Configuration interface references an unknown unit");
    return llvm::Error::success();
  }
  case ImplementationInterfaceSemanticRefKind::ExternalProtocol: {
    const auto boundary =
        std::get<ImplementationExternalProtocolInterfaceRef>(reference)
            .boundary;
    if (!llvm::is_contained(system.artifact().externalBoundaries(), boundary))
      return invalid(
          "ExternalProtocol interface target is absent from the exact System");
    return llvm::Error::success();
  }
  }
  llvm_unreachable("interface semantic reference variant is closed");
}

llvm::Expected<HardwareImplementation>
canonicalize(HardwareImplementationDraft draft,
             const ExternalImplementationContractCatalog &contracts,
             const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (draft.fabric.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      draft.fabric.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("fabric_ref requires loom.fabric 4.0");
  auto importedFabric = fabric::importEntireFabricRoot(draft.fabric, artifacts);
  if (!importedFabric)
    return importedFabric.takeError();
  auto system = fabric::requireSystemRoot(importedFabric->view());
  if (!system) {
    llvm::consumeError(system.takeError());
    return invalid("fabric_ref requires a complete System root");
  }

  if (draft.configurationAbi.schemaIdentity !=
          configurationAbiSchema.identity ||
      draft.configurationAbi.schemaVersion != configurationAbiSchema.version)
    return invalid("configuration_abi_ref requires loom.configuration_abi 2.0");
  auto abi = importConfigurationABI(draft.configurationAbi, artifacts);
  if (!abi)
    return abi.takeError();
  if (abi->abi().fabric() != draft.fabric)
    return invalid("ConfigurationABI must describe the same Fabric System");

  llvm::sort(draft.interconnectImplementations, rootReferenceLess);
  if (std::adjacent_find(draft.interconnectImplementations.begin(),
                         draft.interconnectImplementations.end()) !=
      draft.interconnectImplementations.end())
    return invalid("interconnect implementation catalog contains a duplicate");
  for (const ArtifactRootReference &reference :
       draft.interconnectImplementations) {
    auto interconnect = fabric::importEntireFabricRoot(reference, artifacts);
    if (!interconnect)
      return interconnect.takeError();
    if (interconnect->view().rootKind() !=
        fabric::FabricRootKind::InterconnectImplementation)
      return invalid("interconnect reference has the wrong Fabric root kind");
    const auto dependencies = interconnect->directDependencies();
    if (dependencies.size() != 1 ||
        dependencies.front().role !=
            fabric::FabricDependencyRole::RefinedSystem ||
        dependencies.front().root != draft.fabric)
      return invalid("interconnect implementation does not refine fabric_ref");
  }

  auto representationIndex =
      indexRepresentationRoot(draft.representationRoot, blobs);
  if (!representationIndex)
    return representationIndex.takeError();

  std::optional<platform::FinalizedImplementationPlatform> platform;
  if (draft.implementationPlatform) {
    auto imported = platform::importImplementationPlatform(
        *draft.implementationPlatform, artifacts);
    if (!imported)
      return imported.takeError();
    platform = std::move(*imported);
  } else if (draft.representationRoot.variant !=
             RepresentationRootVariant::Rtl) {
    return invalid(
        "non-RTL representation requires an implementation platform");
  }

  llvm::sort(draft.interfaces, interfaceLess);
  for (std::size_t ordinal = 0; ordinal < draft.interfaces.size(); ++ordinal) {
    const ImplementationInterface &interface = draft.interfaces[ordinal];
    if (ordinal != 0 && draft.interfaces[ordinal - 1] == interface)
      return invalid("interface catalog contains a duplicate record");
    if (llvm::Error error = validateInterfaceSemanticRef(
            interface.semanticRef, *system, abi->abi(), draft.configurationAbi))
      return std::move(error);
    if (llvm::Error error = requireIndexedLocator(
            *representationIndex, interface.representationLocator,
            "interface locator"))
      return std::move(error);
    if (interface.devicePinRef) {
      if (!platform || (draft.representationRoot.variant !=
                            RepresentationRootVariant::FpgaPhysical &&
                        draft.representationRoot.variant !=
                            RepresentationRootVariant::FpgaImage))
        return invalid("device pin reference requires an FPGA representation");
      if (llvm::Error error =
              validateKey(*interface.devicePinRef, "device pin reference"))
        return std::move(error);
    }
  }

  llvm::sort(draft.activityPoints, activityLess);
  for (std::size_t ordinal = 0; ordinal < draft.activityPoints.size();
       ++ordinal) {
    const ActivityPoint &point = draft.activityPoints[ordinal];
    if (ordinal != 0 && draft.activityPoints[ordinal - 1] == point)
      return invalid("activity point catalog contains a duplicate record");
    if (llvm::Error error = requireIndexedLocator(*representationIndex,
                                                  point.representationLocator,
                                                  "activity point locator"))
      return std::move(error);
    if (point.semanticFabricRef) {
      auto resolved = system->resolvePhysicalOwner(*point.semanticFabricRef);
      if (!resolved)
        return resolved.takeError();
    }
  }

  const std::vector<ExternalImplementationBindingDraft> authoredBindings =
      draft.externalImplementationBindings;
  if (llvm::Error error = contracts.canonicalizeAndValidateBindings(
          draft.externalImplementationBindings, draft.representationRoot,
          platform ? &platform->platform() : nullptr, *system))
    return std::move(error);
  for (const ExternalImplementationBindingDraft &binding :
       draft.externalImplementationBindings)
    for (const RepresentationLocator &locator : binding.representationLocators)
      if (llvm::Error error = requireIndexedLocator(
              *representationIndex, locator, "external implementation locator"))
        return std::move(error);

  std::vector<std::uint64_t> authoredToCanonical;
  authoredToCanonical.reserve(authoredBindings.size());
  for (const ExternalImplementationBindingDraft &authored : authoredBindings) {
    std::vector<ExternalImplementationBindingDraft> singleton{authored};
    if (llvm::Error error = contracts.canonicalizeAndValidateBindings(
            singleton, draft.representationRoot,
            platform ? &platform->platform() : nullptr, *system))
      return std::move(error);
    auto found =
        llvm::find(draft.externalImplementationBindings, singleton.front());
    if (found == draft.externalImplementationBindings.end())
      return invalid("external binding canonical remap is unresolved");
    authoredToCanonical.push_back(static_cast<std::uint64_t>(
        found - draft.externalImplementationBindings.begin()));
  }

  auto externalBindings = finalizeExternalBindings(
      draft.externalImplementationBindings, draft.representationRoot);
  if (!externalBindings)
    return externalBindings.takeError();
  auto memoryBindings = detail::canonicalizeMemoryMacroBindings(
      draft.memoryMacroBindings, *externalBindings, authoredToCanonical,
      contracts, draft.representationRoot, *system);
  if (!memoryBindings)
    return memoryBindings.takeError();
  for (const MemoryMacroBinding &binding : *memoryBindings)
    if (llvm::Error error = requireIndexedLocator(*representationIndex,
                                                  binding.representationLocator,
                                                  "memory macro locator"))
      return std::move(error);

  const llvm::ArrayRef<RepresentationLocator> unresolvedDefinitions =
      representationIndex->unresolvedExternalDefinitions();
  std::vector<std::size_t> closureCounts(unresolvedDefinitions.size(), 0);
  for (const ExternalImplementationBinding &binding : *externalBindings) {
    if (!binding.blackBoxContractPayloadRef)
      continue;
    bool closesDefinition = false;
    for (std::size_t ordinal = 0; ordinal < unresolvedDefinitions.size();
         ++ordinal) {
      if (!llvm::is_contained(binding.representationLocators,
                              unresolvedDefinitions[ordinal]))
        continue;
      ++closureCounts[ordinal];
      closesDefinition = true;
    }
    if (!closesDefinition)
      return invalid("black-box binding closes no indexed definition");
  }
  for (std::size_t closures : closureCounts)
    if (closures != 1)
      return invalid("unresolved external definition does not have one exact "
                     "black-box binding");

  return detail::HardwareImplementationBuilder::create(
      std::move(draft), std::move(*memoryBindings),
      std::move(*externalBindings));
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                         bytes.size());
}

llvm::Expected<HardwareImplementation>
decode(llvm::StringRef canonicalJson,
       const ExternalImplementationContractCatalog &contracts,
       const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto draft = parse(canonicalJson);
  if (!draft)
    return draft.takeError();
  auto implementation =
      canonicalize(std::move(*draft), contracts, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  if (serialize(*implementation) != canonicalJson)
    return invalid("stored root is not canonical");
  return implementation;
}

} // namespace

llvm::Expected<FinalizedHardwareImplementation>
finalizeHardwareImplementation(HardwareImplementationDraft draft,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  const ExternalImplementationContractCatalog contracts;
  return finalizeHardwareImplementation(std::move(draft), contracts, artifacts,
                                        blobs);
}

llvm::Expected<FinalizedHardwareImplementation> finalizeHardwareImplementation(
    HardwareImplementationDraft draft,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto implementation =
      canonicalize(std::move(draft), contracts, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  const std::string json = serialize(*implementation);
  auto strict = decode(json, contracts, artifacts, blobs);
  if (!strict)
    return strict.takeError();
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
  auto identity = artifacts.put(hardwareImplementationSchema, bytes);
  if (!identity)
    return identity.takeError();
  return importHardwareImplementation(
      {hardwareImplementationSchema.identity.str(),
       hardwareImplementationSchema.version, *identity},
      contracts, artifacts, blobs);
}

llvm::Expected<FinalizedHardwareImplementation>
importHardwareImplementation(const ArtifactRootReference &reference,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  const ExternalImplementationContractCatalog contracts;
  return importHardwareImplementation(reference, contracts, artifacts, blobs);
}

llvm::Expected<FinalizedHardwareImplementation> importHardwareImplementation(
    const ArtifactRootReference &reference,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (reference.schemaIdentity != hardwareImplementationSchema.identity ||
      reference.schemaVersion != hardwareImplementationSchema.version)
    return invalid("reference schema is not loom.hardware_implementation 2.2");
  auto bytes = artifacts.get(hardwareImplementationSchema, reference.artifact);
  if (!bytes)
    return bytes.takeError();
  auto implementation =
      decode(asText(bytes->bytes()), contracts, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  return FinalizedHardwareImplementation(reference, std::move(*bytes),
                                         std::move(*implementation));
}

} // namespace loom::hardware
