#include "Hardware/Implementation/HardwareImplementation.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricArtifactLocalReference.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "HardwareImplementationInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware {

class detail::HardwareImplementationBuilder final {
public:
  static HardwareImplementation create(HardwareImplementationDraft draft) {
    return HardwareImplementation(
        std::move(draft.fabric), std::move(draft.configurationAbi),
        std::move(draft.interconnectImplementations), draft.representation,
        std::move(draft.implementationPlatform), std::move(draft.payloads),
        std::move(draft.interfaces), std::move(draft.activityPoints),
        std::move(draft.memoryMacroBindings),
        std::move(draft.externalImplementationBindings));
  }
};

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_implementation_invalid: " + message);
}

llvm::StringRef representationSpelling(HardwareRepresentation value) {
  switch (value) {
  case HardwareRepresentation::Rtl:
    return "rtl";
  case HardwareRepresentation::GateNetlist:
    return "gate_netlist";
  case HardwareRepresentation::AsicPlaced:
    return "asic_placed";
  case HardwareRepresentation::AsicRouted:
    return "asic_routed";
  case HardwareRepresentation::AsicExtracted:
    return "asic_extracted";
  case HardwareRepresentation::FpgaPlaced:
    return "fpga_placed";
  case HardwareRepresentation::FpgaRouted:
    return "fpga_routed";
  case HardwareRepresentation::FpgaImage:
    return "fpga_image";
  }
  llvm_unreachable("validated hardware representation is closed");
}

std::optional<HardwareRepresentation>
parseRepresentation(llvm::StringRef spelling) {
  if (spelling == "rtl")
    return HardwareRepresentation::Rtl;
  if (spelling == "gate_netlist")
    return HardwareRepresentation::GateNetlist;
  if (spelling == "asic_placed")
    return HardwareRepresentation::AsicPlaced;
  if (spelling == "asic_routed")
    return HardwareRepresentation::AsicRouted;
  if (spelling == "asic_extracted")
    return HardwareRepresentation::AsicExtracted;
  if (spelling == "fpga_placed")
    return HardwareRepresentation::FpgaPlaced;
  if (spelling == "fpga_routed")
    return HardwareRepresentation::FpgaRouted;
  if (spelling == "fpga_image")
    return HardwareRepresentation::FpgaImage;
  return std::nullopt;
}

llvm::StringRef payloadRoleSpelling(PayloadRole value) {
  switch (value) {
  case PayloadRole::RtlSource:
    return "rtl_source";
  case PayloadRole::Netlist:
    return "netlist";
  case PayloadRole::PhysicalDatabase:
    return "physical_database";
  case PayloadRole::Parasitics:
    return "parasitics";
  case PayloadRole::LayoutStream:
    return "layout_stream";
  case PayloadRole::DeviceImage:
    return "device_image";
  case PayloadRole::GenerationConstraint:
    return "generation_constraint";
  case PayloadRole::BlackBoxContract:
    return "black_box_contract";
  }
  llvm_unreachable("validated hardware payload role is closed");
}

std::optional<PayloadRole> parsePayloadRole(llvm::StringRef spelling) {
  if (spelling == "rtl_source")
    return PayloadRole::RtlSource;
  if (spelling == "netlist")
    return PayloadRole::Netlist;
  if (spelling == "physical_database")
    return PayloadRole::PhysicalDatabase;
  if (spelling == "parasitics")
    return PayloadRole::Parasitics;
  if (spelling == "layout_stream")
    return PayloadRole::LayoutStream;
  if (spelling == "device_image")
    return PayloadRole::DeviceImage;
  if (spelling == "generation_constraint")
    return PayloadRole::GenerationConstraint;
  if (spelling == "black_box_contract")
    return PayloadRole::BlackBoxContract;
  return std::nullopt;
}

llvm::StringRef interfaceRoleSpelling(ImplementationInterfaceRole value) {
  switch (value) {
  case ImplementationInterfaceRole::Data:
    return "data";
  case ImplementationInterfaceRole::Clock:
    return "clock";
  case ImplementationInterfaceRole::Reset:
    return "reset";
  case ImplementationInterfaceRole::Configuration:
    return "configuration";
  case ImplementationInterfaceRole::Memory:
    return "memory";
  case ImplementationInterfaceRole::ExternalProtocol:
    return "external_protocol";
  }
  llvm_unreachable("validated implementation interface role is closed");
}

std::optional<ImplementationInterfaceRole>
parseInterfaceRole(llvm::StringRef spelling) {
  if (spelling == "data")
    return ImplementationInterfaceRole::Data;
  if (spelling == "clock")
    return ImplementationInterfaceRole::Clock;
  if (spelling == "reset")
    return ImplementationInterfaceRole::Reset;
  if (spelling == "configuration")
    return ImplementationInterfaceRole::Configuration;
  if (spelling == "memory")
    return ImplementationInterfaceRole::Memory;
  if (spelling == "external_protocol")
    return ImplementationInterfaceRole::ExternalProtocol;
  return std::nullopt;
}

llvm::StringRef objectKindSpelling(RepresentationObjectKind value) {
  switch (value) {
  case RepresentationObjectKind::Module:
    return "module";
  case RepresentationObjectKind::Instance:
    return "instance";
  case RepresentationObjectKind::Port:
    return "port";
  case RepresentationObjectKind::Net:
    return "net";
  case RepresentationObjectKind::Register:
    return "register";
  case RepresentationObjectKind::Memory:
    return "memory";
  case RepresentationObjectKind::Cell:
    return "cell";
  case RepresentationObjectKind::Pin:
    return "pin";
  case RepresentationObjectKind::PhysicalObject:
    return "physical_object";
  case RepresentationObjectKind::DeviceResource:
    return "device_resource";
  }
  llvm_unreachable("validated representation object kind is closed");
}

llvm::StringRef dependencyKindSpelling(ExternalDependencyKind value) {
  switch (value) {
  case ExternalDependencyKind::ExplicitFile:
    return "explicit_file";
  case ExternalDependencyKind::ToolBundledResource:
    return "tool_bundled_resource";
  }
  llvm_unreachable("validated external dependency kind is closed");
}

std::optional<ExternalDependencyKind>
parseDependencyKind(llvm::StringRef spelling) {
  if (spelling == "explicit_file")
    return ExternalDependencyKind::ExplicitFile;
  if (spelling == "tool_bundled_resource")
    return ExternalDependencyKind::ToolBundledResource;
  return std::nullopt;
}

std::optional<RepresentationObjectKind>
parseObjectKind(llvm::StringRef spelling) {
  if (spelling == "module")
    return RepresentationObjectKind::Module;
  if (spelling == "instance")
    return RepresentationObjectKind::Instance;
  if (spelling == "port")
    return RepresentationObjectKind::Port;
  if (spelling == "net")
    return RepresentationObjectKind::Net;
  if (spelling == "register")
    return RepresentationObjectKind::Register;
  if (spelling == "memory")
    return RepresentationObjectKind::Memory;
  if (spelling == "cell")
    return RepresentationObjectKind::Cell;
  if (spelling == "pin")
    return RepresentationObjectKind::Pin;
  if (spelling == "physical_object")
    return RepresentationObjectKind::PhysicalObject;
  if (spelling == "device_resource")
    return RepresentationObjectKind::DeviceResource;
  return std::nullopt;
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

llvm::Error validateLogicalName(llvm::StringRef value) {
  if (value.empty() || value.find('\0') != llvm::StringRef::npos)
    return invalid("payload logical name is empty or contains NUL");
  const std::filesystem::path path(value.str());
  if (path.is_absolute() || path.lexically_normal() != path || path == ".")
    return invalid("payload logical name must be a canonical relative path");
  for (const std::filesystem::path &component : path)
    if (component.empty() || component == "." || component == "..")
      return invalid("payload logical name contains an invalid component");
  return llvm::Error::success();
}

llvm::Error validateMediaType(llvm::StringRef value) {
  const std::size_t slash = value.find('/');
  if (slash == llvm::StringRef::npos || slash == 0 ||
      slash + 1 == value.size() ||
      value.find('/', slash + 1) != llvm::StringRef::npos)
    return invalid("payload media type must contain one nonempty '/' pair");
  for (char character : value) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (!(std::isalnum(byte) || character == '/' || character == '.' ||
          character == '+' || character == '-'))
      return invalid("payload media type is not canonical ASCII");
  }
  return llvm::Error::success();
}

bool isFpgaRepresentation(HardwareRepresentation representation) {
  return representation == HardwareRepresentation::FpgaPlaced ||
         representation == HardwareRepresentation::FpgaRouted ||
         representation == HardwareRepresentation::FpgaImage;
}

void writeArtifactRootReference(llvm::json::OStream &json,
                                const ArtifactRootReference &reference) {
  json.object([&] {
    json.attribute("schema", reference.schemaIdentity);
    json.attribute("version", formatSchemaVersion(reference.schemaVersion));
    json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
  });
}

void writeLocalReference(llvm::json::OStream &json,
                         const EncodedArtifactLocalReference &reference) {
  json.object([&] {
    json.attributeBegin("artifact_ref");
    writeArtifactRootReference(json, reference.artifact);
    json.attributeEnd();
    json.attribute("owner_local_kind", reference.ownerLocalKind);
    json.attribute("payload", formatArtifactLocalPayloadHex(reference.payload));
  });
}

void writeLocator(llvm::json::OStream &json,
                  const RepresentationLocator &locator) {
  json.object([&] {
    json.attribute("kind", objectKindSpelling(locator.kind));
    json.attribute("name", locator.canonicalName);
  });
}

void writeDependencyIdentity(llvm::json::OStream &json,
                             const ExternalDependencyIdentity &identity) {
  json.object([&] {
    std::visit(
        [&](const auto &dependency) {
          using Dependency = std::decay_t<decltype(dependency)>;
          if constexpr (std::is_same_v<Dependency, ExplicitFileDependency>) {
            json.attribute("kind", dependencyKindSpelling(
                                       ExternalDependencyKind::ExplicitFile));
            json.attribute("content_sha256", formatExternalFileFingerprint(
                                                 dependency.contentSha256));
          } else {
            json.attribute("kind",
                           dependencyKindSpelling(
                               ExternalDependencyKind::ToolBundledResource));
            json.attribute("stable_provider_build_identity",
                           dependency.stableProviderBuildIdentity);
            json.attribute("resource_key", dependency.resourceKey);
          }
        },
        identity);
  });
}

void writePayloadReference(llvm::json::OStream &json,
                           const HardwarePayloadRef &reference) {
  json.object([&] {
    json.attribute("role", payloadRoleSpelling(reference.role));
    json.attribute("logical_name", reference.logicalName);
  });
}

std::string serialize(const HardwareImplementation &implementation) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", hardwareImplementationSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(hardwareImplementationSchema.version));
    json.attributeBegin("fabric_ref");
    writeArtifactRootReference(json, implementation.fabric());
    json.attributeEnd();
    json.attributeBegin("configuration_abi_ref");
    writeArtifactRootReference(json, implementation.configurationAbi());
    json.attributeEnd();
    json.attributeArray("interconnect_implementation_refs", [&] {
      for (const ArtifactRootReference &reference :
           implementation.interconnectImplementations())
        writeArtifactRootReference(json, reference);
    });
    json.attribute("representation",
                   representationSpelling(implementation.representation()));
    if (implementation.implementationPlatform()) {
      json.attributeBegin("implementation_platform_ref");
      writeArtifactRootReference(json,
                                 *implementation.implementationPlatform());
      json.attributeEnd();
    }
    json.attributeArray("payloads", [&] {
      for (const HardwarePayload &payload : implementation.payloads()) {
        json.object([&] {
          json.attribute("role", payloadRoleSpelling(payload.role));
          json.attribute("logical_name", payload.logicalName);
          json.attribute("media_type", payload.mediaType);
          json.attribute("blob_sha256", formatBlobDigestHex(payload.content));
        });
      }
    });
    json.attributeArray("interfaces", [&] {
      for (const ImplementationInterface &interface :
           implementation.interfaces()) {
        json.object([&] {
          json.attribute("interface_key", interface.interfaceKey);
          json.attribute("role", interfaceRoleSpelling(interface.role));
          json.attributeBegin("semantic_fabric_ref");
          writeLocalReference(json, interface.semanticFabricRef);
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
          json.attribute("activity_point_id", point.activityPointId);
          json.attributeBegin("representation_locator");
          writeLocator(json, point.representationLocator);
          json.attributeEnd();
          if (point.semanticFabricRef) {
            json.attributeBegin("semantic_fabric_ref");
            writeLocalReference(json, *point.semanticFabricRef);
            json.attributeEnd();
          }
        });
      }
    });
    json.attributeArray("memory_macro_bindings", [&] {
      for (const MemoryMacroBinding &binding :
           implementation.memoryMacroBindings()) {
        json.object([&] {
          json.attributeBegin("fabric_memory_ref");
          writeLocalReference(json, fabric::encodeFabricArtifactLocalReference(
                                        binding.fabricMemoryRef));
          json.attributeEnd();
          json.attribute("external_implementation_binding_id",
                         binding.externalImplementationBindingId);
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
          json.attribute("binding_id", binding.bindingId);
          json.attribute("provider_contract_ref", binding.providerContractRef);
          json.attributeArray("external_inputs", [&] {
            for (const ExternalInputBinding &input : binding.externalInputs) {
              json.object([&] {
                json.attribute("provider_input_slot_ref",
                               input.providerInputSlotRef);
                json.attributeBegin("dependency_identity");
                writeDependencyIdentity(json, input.dependencyIdentity);
                json.attributeEnd();
              });
            }
          });
          json.attributeArray("fabric_resource_refs", [&] {
            for (const EncodedArtifactLocalReference &reference :
                 binding.fabricResourceRefs)
              writeLocalReference(json, reference);
          });
          json.attributeArray("representation_locators", [&] {
            for (const RepresentationLocator &locator :
                 binding.representationLocators)
              writeLocator(json, locator);
          });
          if (binding.blackBoxContractPayloadRef) {
            json.attributeBegin("black_box_contract_payload_ref");
            writePayloadReference(json, *binding.blackBoxContractPayloadRef);
            json.attributeEnd();
          }
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

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return invalid(context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return invalid(context + " requires array field '" + key + "'");
  return value;
}

llvm::Expected<ArtifactRootReference>
parseArtifactRootReference(const llvm::json::Object &object,
                           llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"schema", "version", "artifact"}))
    return std::move(error);
  auto schema = requireString(object, "schema", context);
  if (!schema)
    return schema.takeError();
  auto versionText = requireString(object, "version", context);
  if (!versionText)
    return versionText.takeError();
  auto version = parseSchemaVersion(*versionText);
  if (!version)
    return version.takeError();
  auto artifactText = requireString(object, "artifact", context);
  if (!artifactText)
    return artifactText.takeError();
  auto artifact = parseArtifactIdentityHex(*artifactText);
  if (!artifact)
    return artifact.takeError();
  return ArtifactRootReference{schema->str(), *version, *artifact};
}

llvm::Expected<EncodedArtifactLocalReference>
parseLocalReference(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"artifact_ref", "owner_local_kind", "payload"}))
    return std::move(error);
  const llvm::json::Object *artifact = object.getObject("artifact_ref");
  if (!artifact)
    return invalid(context + " requires object field 'artifact_ref'");
  const std::string artifactContext = (context + ".artifact_ref").str();
  auto root = parseArtifactRootReference(*artifact, artifactContext);
  if (!root)
    return root.takeError();
  const llvm::json::Value *kindValue = object.get("owner_local_kind");
  const std::optional<std::uint64_t> kind =
      kindValue ? kindValue->getAsUINT64() : std::nullopt;
  if (!kind || *kind > std::numeric_limits<std::uint32_t>::max())
    return invalid(context + " requires uint32 field 'owner_local_kind'");
  auto payloadText = requireString(object, "payload", context);
  if (!payloadText)
    return payloadText.takeError();
  auto payload = parseArtifactLocalPayloadHex(*payloadText);
  if (!payload)
    return payload.takeError();
  return EncodedArtifactLocalReference{
      std::move(*root), static_cast<std::uint32_t>(*kind), std::move(*payload)};
}

llvm::Expected<RepresentationLocator>
parseLocator(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"kind", "name"}))
    return std::move(error);
  auto kindText = requireString(object, "kind", context);
  if (!kindText)
    return kindText.takeError();
  std::optional<RepresentationObjectKind> kind = parseObjectKind(*kindText);
  if (!kind)
    return invalid(context + " has unknown locator kind");
  auto name = requireString(object, "name", context);
  if (!name)
    return name.takeError();
  return RepresentationLocator{*kind, name->str()};
}

llvm::Expected<ExternalDependencyIdentity>
parseDependencyIdentity(const llvm::json::Object &object,
                        llvm::StringRef context) {
  auto kindText = requireString(object, "kind", context);
  if (!kindText)
    return kindText.takeError();
  std::optional<ExternalDependencyKind> kind = parseDependencyKind(*kindText);
  if (!kind)
    return invalid(context + " has unknown dependency kind");

  switch (*kind) {
  case ExternalDependencyKind::ExplicitFile: {
    if (llvm::Error error =
            rejectUnknownFields(object, context, {"kind", "content_sha256"}))
      return std::move(error);
    auto fingerprintText = requireString(object, "content_sha256", context);
    if (!fingerprintText)
      return fingerprintText.takeError();
    auto fingerprint = parseExternalFileFingerprint(*fingerprintText);
    if (!fingerprint)
      return fingerprint.takeError();
    return ExternalDependencyIdentity(
        ExplicitFileDependency{std::move(*fingerprint)});
  }
  case ExternalDependencyKind::ToolBundledResource: {
    if (llvm::Error error = rejectUnknownFields(
            object, context,
            {"kind", "stable_provider_build_identity", "resource_key"}))
      return std::move(error);
    auto build =
        requireString(object, "stable_provider_build_identity", context);
    auto resource = requireString(object, "resource_key", context);
    if (!build)
      return build.takeError();
    if (!resource)
      return resource.takeError();
    return ExternalDependencyIdentity(
        ToolBundledResourceDependency{build->str(), resource->str()});
  }
  }
  llvm_unreachable("parsed external dependency kind is closed");
}

llvm::Expected<HardwarePayloadRef>
parsePayloadReference(const llvm::json::Object &object,
                      llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"role", "logical_name"}))
    return std::move(error);
  auto roleText = requireString(object, "role", context);
  auto logicalName = requireString(object, "logical_name", context);
  if (!roleText)
    return roleText.takeError();
  if (!logicalName)
    return logicalName.takeError();
  std::optional<PayloadRole> role = parsePayloadRole(*roleText);
  if (!role)
    return invalid(context + " has unknown payload role");
  return HardwarePayloadRef{*role, logicalName->str()};
}

llvm::Expected<ExternalImplementationBinding>
parseExternalBinding(const llvm::json::Object &object,
                     llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"binding_id", "provider_contract_ref", "external_inputs",
           "fabric_resource_refs", "representation_locators",
           "black_box_contract_payload_ref"}))
    return std::move(error);
  auto bindingId = requireString(object, "binding_id", context);
  auto contractRef = requireString(object, "provider_contract_ref", context);
  auto inputArray = requireArray(object, "external_inputs", context);
  auto fabricArray = requireArray(object, "fabric_resource_refs", context);
  auto locatorArray = requireArray(object, "representation_locators", context);
  if (!bindingId)
    return bindingId.takeError();
  if (!contractRef)
    return contractRef.takeError();
  if (!inputArray)
    return inputArray.takeError();
  if (!fabricArray)
    return fabricArray.takeError();
  if (!locatorArray)
    return locatorArray.takeError();

  std::vector<ExternalInputBinding> inputs;
  for (const llvm::json::Value &value : **inputArray) {
    const llvm::json::Object *input = value.getAsObject();
    if (!input)
      return invalid(context + " input must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *input, "external input",
            {"provider_input_slot_ref", "dependency_identity"}))
      return std::move(error);
    auto slot =
        requireString(*input, "provider_input_slot_ref", "external input");
    if (!slot)
      return slot.takeError();
    const llvm::json::Object *dependency =
        input->getObject("dependency_identity");
    if (!dependency)
      return invalid("external input requires dependency identity");
    auto identity =
        parseDependencyIdentity(*dependency, "external dependency identity");
    if (!identity)
      return identity.takeError();
    inputs.push_back(ExternalInputBinding{slot->str(), std::move(*identity)});
  }

  std::vector<EncodedArtifactLocalReference> fabricRefs;
  for (const llvm::json::Value &value : **fabricArray) {
    const llvm::json::Object *reference = value.getAsObject();
    if (!reference)
      return invalid(context + " Fabric resource reference must be an object");
    auto parsed =
        parseLocalReference(*reference, "external Fabric resource reference");
    if (!parsed)
      return parsed.takeError();
    fabricRefs.push_back(std::move(*parsed));
  }

  std::vector<RepresentationLocator> locators;
  for (const llvm::json::Value &value : **locatorArray) {
    const llvm::json::Object *locator = value.getAsObject();
    if (!locator)
      return invalid(context + " representation locator must be an object");
    auto parsed = parseLocator(*locator, "external representation locator");
    if (!parsed)
      return parsed.takeError();
    locators.push_back(std::move(*parsed));
  }

  std::optional<HardwarePayloadRef> blackBox;
  if (const llvm::json::Value *value =
          object.get("black_box_contract_payload_ref")) {
    const llvm::json::Object *reference = value->getAsObject();
    if (!reference)
      return invalid(context + " BlackBoxContract reference must be an object");
    auto parsed =
        parsePayloadReference(*reference, "BlackBoxContract payload reference");
    if (!parsed)
      return parsed.takeError();
    blackBox = std::move(*parsed);
  }

  return ExternalImplementationBinding{
      bindingId->str(),      contractRef->str(),  std::move(inputs),
      std::move(fabricRefs), std::move(locators), std::move(blackBox)};
}

llvm::Expected<MemoryMacroBinding>
parseMemoryMacroBinding(const llvm::json::Object &object,
                        llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"fabric_memory_ref", "external_implementation_binding_id",
           "representation_locator"}))
    return std::move(error);
  const llvm::json::Object *memoryObject =
      object.getObject("fabric_memory_ref");
  const llvm::json::Object *locatorObject =
      object.getObject("representation_locator");
  auto bindingId =
      requireString(object, "external_implementation_binding_id", context);
  if (!memoryObject || !locatorObject)
    return invalid(context + " requires memory and locator objects");
  if (!bindingId)
    return bindingId.takeError();
  auto encodedMemory =
      parseLocalReference(*memoryObject, "memory macro Fabric reference");
  if (!encodedMemory)
    return encodedMemory.takeError();
  auto memory = fabric::decodeFabricArtifactLocalReference<
      fabric::FabricMemoryOccurrenceRef>(*encodedMemory);
  if (!memory)
    return memory.takeError();
  auto locator =
      parseLocator(*locatorObject, "memory macro representation locator");
  if (!locator)
    return locator.takeError();
  return MemoryMacroBinding{std::move(*memory), bindingId->str(),
                            std::move(*locator)};
}

llvm::Expected<HardwareImplementationDraft> parse(llvm::StringRef body) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(body);
  if (!parsed)
    return invalid(llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "root",
          {"schema", "schema_version", "fabric_ref", "configuration_abi_ref",
           "interconnect_implementation_refs", "representation",
           "implementation_platform_ref", "payloads", "interfaces",
           "activity_points", "memory_macro_bindings",
           "external_implementation_bindings"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "root");
  auto version = requireString(*root, "schema_version", "root");
  if (!schema)
    return schema.takeError();
  if (!version)
    return version.takeError();
  if (*schema != hardwareImplementationSchema.identity ||
      *version != formatSchemaVersion(hardwareImplementationSchema.version))
    return invalid("root requires loom.hardware_implementation 1.0");

  const llvm::json::Object *fabricObject = root->getObject("fabric_ref");
  const llvm::json::Object *abiObject =
      root->getObject("configuration_abi_ref");
  if (!fabricObject || !abiObject)
    return invalid("root requires Fabric and ConfigurationABI references");
  auto fabric = parseArtifactRootReference(*fabricObject, "fabric_ref");
  if (!fabric)
    return fabric.takeError();
  auto abi = parseArtifactRootReference(*abiObject, "configuration_abi_ref");
  if (!abi)
    return abi.takeError();

  auto interconnectArray =
      requireArray(*root, "interconnect_implementation_refs", "root");
  if (!interconnectArray)
    return interconnectArray.takeError();
  std::vector<ArtifactRootReference> interconnects;
  for (const llvm::json::Value &value : **interconnectArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("interconnect implementation reference must be an object");
    auto reference =
        parseArtifactRootReference(*object, "interconnect reference");
    if (!reference)
      return reference.takeError();
    interconnects.push_back(std::move(*reference));
  }

  auto representationText = requireString(*root, "representation", "root");
  if (!representationText)
    return representationText.takeError();
  std::optional<HardwareRepresentation> representation =
      parseRepresentation(*representationText);
  if (!representation)
    return invalid("root has unknown hardware representation");

  std::optional<ArtifactRootReference> platform;
  if (const llvm::json::Value *value =
          root->get("implementation_platform_ref")) {
    const llvm::json::Object *object = value->getAsObject();
    if (!object)
      return invalid("implementation_platform_ref must be an object");
    auto reference =
        parseArtifactRootReference(*object, "implementation_platform_ref");
    if (!reference)
      return reference.takeError();
    platform = std::move(*reference);
  }

  auto payloadArray = requireArray(*root, "payloads", "root");
  if (!payloadArray)
    return payloadArray.takeError();
  std::vector<HardwarePayload> payloads;
  for (const llvm::json::Value &value : **payloadArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("payload must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *object, "payload",
            {"role", "logical_name", "media_type", "blob_sha256"}))
      return std::move(error);
    auto roleText = requireString(*object, "role", "payload");
    auto name = requireString(*object, "logical_name", "payload");
    auto mediaType = requireString(*object, "media_type", "payload");
    auto digestText = requireString(*object, "blob_sha256", "payload");
    if (!roleText)
      return roleText.takeError();
    if (!name)
      return name.takeError();
    if (!mediaType)
      return mediaType.takeError();
    if (!digestText)
      return digestText.takeError();
    std::optional<PayloadRole> role = parsePayloadRole(*roleText);
    if (!role)
      return invalid("payload has unknown role");
    auto digest = parseBlobDigestHex(*digestText);
    if (!digest)
      return digest.takeError();
    payloads.push_back(
        HardwarePayload{*role, name->str(), mediaType->str(), *digest});
  }

  auto interfaceArray = requireArray(*root, "interfaces", "root");
  if (!interfaceArray)
    return interfaceArray.takeError();
  std::vector<ImplementationInterface> interfaces;
  for (const llvm::json::Value &value : **interfaceArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("implementation interface must be an object");
    if (llvm::Error error =
            rejectUnknownFields(*object, "implementation interface",
                                {"interface_key", "role", "semantic_fabric_ref",
                                 "representation_locator", "device_pin_ref"}))
      return std::move(error);
    auto key =
        requireString(*object, "interface_key", "implementation interface");
    auto roleText = requireString(*object, "role", "implementation interface");
    if (!key)
      return key.takeError();
    if (!roleText)
      return roleText.takeError();
    std::optional<ImplementationInterfaceRole> role =
        parseInterfaceRole(*roleText);
    if (!role)
      return invalid("implementation interface has unknown role");
    const llvm::json::Object *semantic =
        object->getObject("semantic_fabric_ref");
    const llvm::json::Object *locatorObject =
        object->getObject("representation_locator");
    if (!semantic || !locatorObject)
      return invalid("implementation interface requires references");
    auto semanticRef =
        parseLocalReference(*semantic, "interface semantic Fabric reference");
    if (!semanticRef)
      return semanticRef.takeError();
    auto locator =
        parseLocator(*locatorObject, "interface representation locator");
    if (!locator)
      return locator.takeError();
    std::optional<std::string> pin;
    if (std::optional<llvm::StringRef> value =
            object->getString("device_pin_ref"))
      pin = value->str();
    interfaces.push_back(
        ImplementationInterface{key->str(), *role, std::move(*semanticRef),
                                std::move(*locator), std::move(pin)});
  }

  auto activityArray = requireArray(*root, "activity_points", "root");
  if (!activityArray)
    return activityArray.takeError();
  std::vector<ActivityPoint> activityPoints;
  for (const llvm::json::Value &value : **activityArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("activity point must be an object");
    if (llvm::Error error =
            rejectUnknownFields(*object, "activity point",
                                {"activity_point_id", "representation_locator",
                                 "semantic_fabric_ref"}))
      return std::move(error);
    auto id = requireString(*object, "activity_point_id", "activity point");
    if (!id)
      return id.takeError();
    const llvm::json::Object *locatorObject =
        object->getObject("representation_locator");
    if (!locatorObject)
      return invalid("activity point requires representation locator");
    auto locator =
        parseLocator(*locatorObject, "activity representation locator");
    if (!locator)
      return locator.takeError();
    std::optional<EncodedArtifactLocalReference> semanticRef;
    if (const llvm::json::Value *semanticValue =
            object->get("semantic_fabric_ref")) {
      const llvm::json::Object *semantic = semanticValue->getAsObject();
      if (!semantic)
        return invalid("activity semantic Fabric reference must be an object");
      auto reference =
          parseLocalReference(*semantic, "activity semantic Fabric reference");
      if (!reference)
        return reference.takeError();
      semanticRef = std::move(*reference);
    }
    activityPoints.push_back(
        ActivityPoint{id->str(), std::move(*locator), std::move(semanticRef)});
  }

  auto memoryBindings = requireArray(*root, "memory_macro_bindings", "root");
  auto externalBindings =
      requireArray(*root, "external_implementation_bindings", "root");
  if (!memoryBindings)
    return memoryBindings.takeError();
  if (!externalBindings)
    return externalBindings.takeError();

  std::vector<MemoryMacroBinding> parsedMemoryBindings;
  for (const llvm::json::Value &value : **memoryBindings) {
    const llvm::json::Object *binding = value.getAsObject();
    if (!binding)
      return invalid("memory macro binding must be an object");
    auto parsed = parseMemoryMacroBinding(*binding, "memory macro binding");
    if (!parsed)
      return parsed.takeError();
    parsedMemoryBindings.push_back(std::move(*parsed));
  }

  std::vector<ExternalImplementationBinding> parsedExternalBindings;
  for (const llvm::json::Value &value : **externalBindings) {
    const llvm::json::Object *binding = value.getAsObject();
    if (!binding)
      return invalid("external implementation binding must be an object");
    auto parsed =
        parseExternalBinding(*binding, "external implementation binding");
    if (!parsed)
      return parsed.takeError();
    parsedExternalBindings.push_back(std::move(*parsed));
  }

  return HardwareImplementationDraft{std::move(*fabric),
                                     std::move(*abi),
                                     std::move(interconnects),
                                     *representation,
                                     std::move(platform),
                                     std::move(payloads),
                                     std::move(interfaces),
                                     std::move(activityPoints),
                                     std::move(parsedMemoryBindings),
                                     std::move(parsedExternalBindings)};
}

llvm::Error validatePayloadClosure(llvm::ArrayRef<HardwarePayload> payloads,
                                   HardwareRepresentation representation,
                                   const BlobStore &blobs) {
  if (payloads.empty())
    return invalid("payload catalog must be nonempty");
  auto hasRole = [&](PayloadRole role) {
    return llvm::any_of(payloads, [&](const HardwarePayload &payload) {
      return payload.role == role;
    });
  };
  switch (representation) {
  case HardwareRepresentation::Rtl:
    if (!hasRole(PayloadRole::RtlSource))
      return invalid("RTL representation requires an RtlSource payload");
    break;
  case HardwareRepresentation::GateNetlist:
    if (!hasRole(PayloadRole::Netlist))
      return invalid("GateNetlist representation requires a Netlist payload");
    break;
  case HardwareRepresentation::AsicPlaced:
  case HardwareRepresentation::AsicRouted:
  case HardwareRepresentation::FpgaPlaced:
  case HardwareRepresentation::FpgaRouted:
    if (!hasRole(PayloadRole::PhysicalDatabase))
      return invalid("physical representation requires a PhysicalDatabase "
                     "payload");
    break;
  case HardwareRepresentation::AsicExtracted:
    if (!hasRole(PayloadRole::PhysicalDatabase) ||
        !hasRole(PayloadRole::Parasitics))
      return invalid("extracted ASIC representation requires PhysicalDatabase "
                     "and Parasitics payloads");
    break;
  case HardwareRepresentation::FpgaImage:
    if (!hasRole(PayloadRole::DeviceImage))
      return invalid(
          "FPGA image representation requires a DeviceImage payload");
    break;
  }

  for (const HardwarePayload &payload : payloads) {
    (void)payloadRoleSpelling(payload.role);
    if (llvm::Error error = validateLogicalName(payload.logicalName))
      return error;
    if (llvm::Error error = validateMediaType(payload.mediaType))
      return error;
    auto contents = blobs.get(payload.content);
    if (!contents)
      return contents.takeError();
  }
  return llvm::Error::success();
}

llvm::Expected<HardwareImplementation>
canonicalize(HardwareImplementationDraft draft,
             const ExternalImplementationContractCatalog &contracts,
             const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (draft.fabric.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      draft.fabric.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("fabric_ref requires loom.fabric 3.0");
  auto importedFabric = fabric::importEntireFabricRoot(draft.fabric, artifacts);
  if (!importedFabric)
    return importedFabric.takeError();
  if (importedFabric->view().rootKind() ==
      fabric::FabricRootKind::InterconnectImplementation)
    return invalid("fabric_ref may not select an interconnect implementation");

  if (draft.configurationAbi.schemaIdentity !=
          configurationAbiSchema.identity ||
      draft.configurationAbi.schemaVersion != configurationAbiSchema.version)
    return invalid("configuration_abi_ref requires loom.configuration_abi 1.0");
  auto abi = importConfigurationABI(draft.configurationAbi, artifacts);
  if (!abi)
    return abi.takeError();
  if (abi->abi().fabric() != draft.fabric)
    return invalid("ConfigurationABI must describe the same Fabric");

  llvm::sort(draft.interconnectImplementations, artifactRootReferenceLess);
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

  (void)representationSpelling(draft.representation);
  std::optional<platform::FinalizedImplementationPlatform> platform;
  if (draft.implementationPlatform) {
    auto imported = platform::importImplementationPlatform(
        *draft.implementationPlatform, artifacts);
    if (!imported)
      return imported.takeError();
    platform = std::move(*imported);
  } else if (draft.representation != HardwareRepresentation::Rtl) {
    return invalid(
        "non-RTL representation requires an implementation platform");
  }

  llvm::sort(draft.payloads,
             [](const HardwarePayload &lhs, const HardwarePayload &rhs) {
               return std::tie(lhs.role, lhs.logicalName, lhs.mediaType,
                               lhs.content.bytes()) <
                      std::tie(rhs.role, rhs.logicalName, rhs.mediaType,
                               rhs.content.bytes());
             });
  for (std::size_t index = 1; index < draft.payloads.size(); ++index)
    if (draft.payloads[index - 1].role == draft.payloads[index].role &&
        draft.payloads[index - 1].logicalName ==
            draft.payloads[index].logicalName)
      return invalid("payload catalog contains a duplicate role/logical name");
  if (llvm::Error error =
          validatePayloadClosure(draft.payloads, draft.representation, blobs))
    return std::move(error);

  llvm::sort(draft.interfaces, [](const ImplementationInterface &lhs,
                                  const ImplementationInterface &rhs) {
    return lhs.interfaceKey < rhs.interfaceKey;
  });
  for (std::size_t index = 0; index < draft.interfaces.size(); ++index) {
    const ImplementationInterface &interface = draft.interfaces[index];
    if (index != 0 &&
        draft.interfaces[index - 1].interfaceKey == interface.interfaceKey)
      return invalid("interface catalog contains a duplicate key");
    if (llvm::Error error =
            validateKey(interface.interfaceKey, "interface key"))
      return std::move(error);
    (void)interfaceRoleSpelling(interface.role);
    if (llvm::Error error = fabric::validateFabricArtifactLocalReference(
            importedFabric->view(), interface.semanticFabricRef))
      return std::move(error);
    if (llvm::Error error = detail::validateRepresentationLocator(
            interface.representationLocator, draft.representation))
      return std::move(error);
    if (interface.devicePinRef) {
      if (!platform ||
          !std::holds_alternative<platform::FpgaTarget>(
              platform->platform().target()) ||
          !isFpgaRepresentation(draft.representation))
        return invalid("device pin reference requires an FPGA representation");
      if (llvm::Error error =
              validateKey(*interface.devicePinRef, "device pin reference"))
        return std::move(error);
    }
  }

  llvm::sort(draft.activityPoints,
             [](const ActivityPoint &lhs, const ActivityPoint &rhs) {
               return lhs.activityPointId < rhs.activityPointId;
             });
  for (std::size_t index = 0; index < draft.activityPoints.size(); ++index) {
    const ActivityPoint &point = draft.activityPoints[index];
    if (index != 0 && draft.activityPoints[index - 1].activityPointId ==
                          point.activityPointId)
      return invalid("activity point catalog contains a duplicate ID");
    if (llvm::Error error =
            validateKey(point.activityPointId, "activity point ID"))
      return std::move(error);
    if (llvm::Error error = detail::validateRepresentationLocator(
            point.representationLocator, draft.representation))
      return std::move(error);
    if (point.semanticFabricRef)
      if (llvm::Error error = fabric::validateFabricArtifactLocalReference(
              importedFabric->view(), *point.semanticFabricRef))
        return std::move(error);
  }

  if (llvm::Error error = contracts.canonicalizeAndValidateBindings(
          draft.externalImplementationBindings, draft.representation,
          platform ? &platform->platform() : nullptr, draft.payloads,
          importedFabric->view()))
    return std::move(error);
  if (llvm::Error error = detail::canonicalizeMemoryMacroBindings(
          draft.memoryMacroBindings, draft.externalImplementationBindings,
          contracts, draft.representation, importedFabric->view()))
    return std::move(error);

  return detail::HardwareImplementationBuilder::create(std::move(draft));
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
    return invalid("reference requires loom.hardware_implementation 1.0");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto implementation =
      decode(asText(bytes->bytes()), contracts, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  if (finalizeArtifactIdentity(hardwareImplementationSchema, *bytes) !=
      reference.artifact)
    return invalid("reference has a stale HardwareImplementation identity");
  return FinalizedHardwareImplementation(reference, std::move(*bytes),
                                         std::move(*implementation));
}

} // namespace loom::hardware
