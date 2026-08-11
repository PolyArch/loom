#include "Gem5SimulationBindingInternal.h"

#include "Common/ArtifactText.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::runtime::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_simulation_binding_invalid: " + message);
}

llvm::Error rejectUnknownFields(
    const llvm::json::Object &object, llvm::StringRef context,
    std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &entry : object)
    if (std::find(allowed.begin(), allowed.end(),
                  llvm::StringRef(entry.first)) == allowed.end())
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(entry.first) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  auto text = value->getAsString();
  if (!text)
    return invalid(context + " field '" + field + "' must be a string");
  return *text;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return invalid(context + " is missing field '" + field + "'");
  auto number = value->getAsUINT64();
  if (!number)
    return invalid(context + " field '" + field +
                   "' must be an unsigned integer");
  return *number;
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
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedVersion)
    return parsedVersion.takeError();
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

void writeContractRef(llvm::json::OStream &json,
                      const Gem5ModelContractDescriptorRef &reference) {
  json.object([&] {
    json.attribute("identity", reference.identity);
    json.attribute("version", formatSchemaVersion(reference.version));
  });
}

llvm::Expected<Gem5ModelContractDescriptorRef>
parseContractRef(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"identity", "version"}))
    return std::move(error);
  auto identity = requireString(object, "identity", context);
  auto version = requireString(object, "version", context);
  if (!identity)
    return identity.takeError();
  if (!version)
    return version.takeError();
  auto parsed = parseSchemaVersion(*version);
  if (!parsed)
    return parsed.takeError();
  return Gem5ModelContractDescriptorRef{identity->str(), *parsed};
}

void writeObject(llvm::json::OStream &json, const Gem5SimObjectRef &object) {
  json.object([&] {
    json.attributeBegin("contract");
    writeContractRef(json, object.contract);
    json.attributeEnd();
    json.attribute("payload", formatArtifactLocalPayloadHex(object.payload));
  });
}

llvm::Expected<Gem5SimObjectRef>
parseObject(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"contract", "payload"}))
    return std::move(error);
  auto contractObject = requireObject(object, "contract", context);
  auto payload = requireString(object, "payload", context);
  if (!contractObject)
    return contractObject.takeError();
  if (!payload)
    return payload.takeError();
  auto contract =
      parseContractRef(**contractObject, (context + ".contract").str());
  auto parsedPayload = parseArtifactLocalPayloadHex(*payload);
  if (!contract)
    return contract.takeError();
  if (!parsedPayload)
    return parsedPayload.takeError();
  return Gem5SimObjectRef{std::move(*contract), std::move(*parsedPayload)};
}

void writePort(llvm::json::OStream &json, const Gem5SimPortRef &port) {
  json.object([&] {
    json.attributeBegin("object");
    writeObject(json, port.object);
    json.attributeEnd();
    json.attribute("kind", port.kind);
    json.attribute("payload", formatArtifactLocalPayloadHex(port.payload));
  });
}

llvm::Expected<Gem5SimPortRef>
parsePort(const llvm::json::Object &object, llvm::StringRef context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"object", "kind", "payload"}))
    return std::move(error);
  auto objectValue = requireObject(object, "object", context);
  auto kind = requireUnsigned(object, "kind", context);
  auto payload = requireString(object, "payload", context);
  if (!objectValue)
    return objectValue.takeError();
  if (!kind)
    return kind.takeError();
  if (*kind > std::numeric_limits<std::uint32_t>::max())
    return invalid(context + " port kind is out of range");
  if (!payload)
    return payload.takeError();
  auto parsedObject =
      parseObject(**objectValue, (context + ".object").str());
  auto parsedPayload = parseArtifactLocalPayloadHex(*payload);
  if (!parsedObject)
    return parsedObject.takeError();
  if (!parsedPayload)
    return parsedPayload.takeError();
  return Gem5SimPortRef{std::move(*parsedObject),
                        static_cast<std::uint32_t>(*kind),
                        std::move(*parsedPayload)};
}

template <typename Ref>
void writeFabricRef(llvm::json::OStream &json, const Ref &reference) {
  json.value(formatArtifactLocalPayloadHex(
      fabric::canonicalFabricBytes(reference)));
}

void writeSpatialBoundary(
    llvm::json::OStream &json,
    const fabric::FabricSpatialAttachmentEndpointRef &reference) {
  json.value(formatArtifactLocalPayloadHex(
      fabric::encodeFabricSpatialAttachmentEndpointRef(reference)));
}

template <typename Ref>
llvm::Expected<Ref> parseFabricRef(llvm::StringRef spelling,
                                   llvm::StringRef context) {
  auto bytes = parseArtifactLocalPayloadHex(spelling);
  if (!bytes)
    return bytes.takeError();
  auto reference = fabric::decodeFabricRef<Ref>(*bytes);
  if (!reference)
    return invalid(context + " is not a canonical Fabric reference: " +
                   llvm::toString(reference.takeError()));
  return std::move(*reference);
}

llvm::Expected<fabric::FabricSpatialAttachmentEndpointRef>
parseSpatialBoundary(llvm::StringRef spelling, llvm::StringRef context) {
  auto bytes = parseArtifactLocalPayloadHex(spelling);
  if (!bytes)
    return bytes.takeError();
  auto reference =
      fabric::decodeFabricSpatialAttachmentEndpointRef(*bytes);
  if (!reference)
    return invalid(context + " is not a canonical spatial boundary: " +
                   llvm::toString(reference.takeError()));
  return std::move(*reference);
}

void writeCorrespondence(llvm::json::OStream &json,
                         const Gem5Correspondence &correspondence) {
  json.object([&] {
    if (const auto *processor =
            std::get_if<Gem5ProcessorCorrespondence>(&correspondence)) {
      json.attribute("kind", "processor");
      json.attribute("processor_kind",
                     std::holds_alternative<fabric::HostCoreOccurrenceRef>(
                         processor->processor)
                         ? "host_core"
                         : "instruction_core");
      json.attributeBegin("fabric_ref");
      std::visit([&](const auto &ref) { writeFabricRef(json, ref); },
                 processor->processor);
      json.attributeEnd();
      json.attributeBegin("sim_object_ref");
      writeObject(json, processor->simObject);
      json.attributeEnd();
      return;
    }
    if (const auto *bridge =
            std::get_if<Gem5SpatialBridgeCorrespondence>(&correspondence)) {
      json.attribute("kind", "spatial_bridge");
      json.attributeBegin("spatial_core_occurrence_ref");
      writeFabricRef(json, bridge->spatialCore);
      json.attributeEnd();
      json.attributeBegin("fabric_spatial_launch_boundary_ref");
      writeSpatialBoundary(json, bridge->spatialBoundary);
      json.attributeEnd();
      json.attributeBegin("bridge_endpoint_ref");
      writePort(json, bridge->bridgeEndpoint);
      json.attributeEnd();
      return;
    }
    if (const auto *memory =
            std::get_if<Gem5MemoryOrServiceCorrespondence>(&correspondence)) {
      json.attribute("kind", "memory_or_service");
      json.attribute("fabric_ref_kind",
                     std::holds_alternative<fabric::SystemMemoryServiceRef>(
                         memory->fabricRef)
                         ? "memory_service"
                         : "service_endpoint");
      json.attributeBegin("fabric_ref");
      std::visit([&](const auto &ref) { writeFabricRef(json, ref); },
                 memory->fabricRef);
      json.attributeEnd();
      json.attributeBegin("sim_object_ref");
      writeObject(json, memory->simObject);
      json.attributeEnd();
      json.attributeBegin("sim_port_ref");
      writePort(json, memory->simPort);
      json.attributeEnd();
      return;
    }
    if (const auto *transport =
            std::get_if<Gem5TransportCorrespondence>(&correspondence)) {
      json.attribute("kind", "transport");
      json.attribute("fabric_ref_kind",
                     std::holds_alternative<fabric::SystemTransportResourceRef>(
                         transport->fabricRef)
                         ? "resource"
                         : "endpoint");
      json.attributeBegin("fabric_ref");
      std::visit([&](const auto &ref) { writeFabricRef(json, ref); },
                 transport->fabricRef);
      json.attributeEnd();
      json.attributeBegin("sim_object_ref");
      writeObject(json, transport->simObject);
      json.attributeEnd();
      json.attributeBegin("sim_port_ref");
      writePort(json, transport->simPort);
      json.attributeEnd();
      return;
    }
    const auto &external =
        std::get<Gem5ExternalEndpointCorrespondence>(correspondence);
    json.attribute("kind", "external_endpoint");
    json.attributeBegin("fabric_external_endpoint_ref");
    writeFabricRef(json, external.fabricRef);
    json.attributeEnd();
    json.attributeBegin("sim_object_ref");
    writeObject(json, external.simObject);
    json.attributeEnd();
    json.attributeBegin("sim_port_ref");
    writePort(json, external.simPort);
    json.attributeEnd();
  });
}

llvm::Expected<Gem5Correspondence>
parseCorrespondence(const llvm::json::Object &object,
                    llvm::StringRef context) {
  auto kind = requireString(object, "kind", context);
  if (!kind)
    return kind.takeError();
  if (*kind == "processor") {
    if (llvm::Error error = rejectUnknownFields(
            object, context,
            {"kind", "processor_kind", "fabric_ref", "sim_object_ref"}))
      return std::move(error);
    auto processorKind = requireString(object, "processor_kind", context);
    auto fabricRef = requireString(object, "fabric_ref", context);
    auto simObject = requireObject(object, "sim_object_ref", context);
    if (!processorKind)
      return processorKind.takeError();
    if (!fabricRef)
      return fabricRef.takeError();
    if (!simObject)
      return simObject.takeError();
    Gem5ProcessorFabricRef processor;
    if (*processorKind == "host_core") {
      auto parsed = parseFabricRef<fabric::HostCoreOccurrenceRef>(
          *fabricRef, (context + ".fabric_ref").str());
      if (!parsed)
        return parsed.takeError();
      processor = *parsed;
    } else if (*processorKind == "instruction_core") {
      auto parsed = parseFabricRef<fabric::InstructionCoreContextRef>(
          *fabricRef, (context + ".fabric_ref").str());
      if (!parsed)
        return parsed.takeError();
      processor = *parsed;
    } else {
      return invalid(context + " has an unknown processor_kind");
    }
    auto parsedObject =
        parseObject(**simObject, (context + ".sim_object_ref").str());
    if (!parsedObject)
      return parsedObject.takeError();
    return Gem5Correspondence(Gem5ProcessorCorrespondence{
        std::move(processor), std::move(*parsedObject)});
  }
  if (*kind == "spatial_bridge") {
    if (llvm::Error error = rejectUnknownFields(
            object, context,
            {"kind", "spatial_core_occurrence_ref",
             "fabric_spatial_launch_boundary_ref", "bridge_endpoint_ref"}))
      return std::move(error);
    auto coreText =
        requireString(object, "spatial_core_occurrence_ref", context);
    auto boundaryText = requireString(
        object, "fabric_spatial_launch_boundary_ref", context);
    auto endpointObject = requireObject(object, "bridge_endpoint_ref", context);
    if (!coreText)
      return coreText.takeError();
    if (!boundaryText)
      return boundaryText.takeError();
    if (!endpointObject)
      return endpointObject.takeError();
    auto core = parseFabricRef<fabric::SpatialCoreOccurrenceRef>(
        *coreText, (context + ".spatial_core_occurrence_ref").str());
    auto boundary = parseSpatialBoundary(
        *boundaryText,
        (context + ".fabric_spatial_launch_boundary_ref").str());
    auto endpoint =
        parsePort(**endpointObject,
                  (context + ".bridge_endpoint_ref").str());
    if (!core)
      return core.takeError();
    if (!boundary)
      return boundary.takeError();
    if (!endpoint)
      return endpoint.takeError();
    return Gem5Correspondence(Gem5SpatialBridgeCorrespondence{
        *core, std::move(*boundary), std::move(*endpoint)});
  }
  if (*kind == "memory_or_service" || *kind == "transport") {
    if (llvm::Error error = rejectUnknownFields(
            object, context,
            {"kind", "fabric_ref_kind", "fabric_ref", "sim_object_ref",
             "sim_port_ref"}))
      return std::move(error);
    auto refKind = requireString(object, "fabric_ref_kind", context);
    auto fabricRef = requireString(object, "fabric_ref", context);
    auto simObject = requireObject(object, "sim_object_ref", context);
    auto simPort = requireObject(object, "sim_port_ref", context);
    if (!refKind)
      return refKind.takeError();
    if (!fabricRef)
      return fabricRef.takeError();
    if (!simObject)
      return simObject.takeError();
    if (!simPort)
      return simPort.takeError();
    auto parsedObject =
        parseObject(**simObject, (context + ".sim_object_ref").str());
    auto parsedPort =
        parsePort(**simPort, (context + ".sim_port_ref").str());
    if (!parsedObject)
      return parsedObject.takeError();
    if (!parsedPort)
      return parsedPort.takeError();
    if (*kind == "memory_or_service") {
      Gem5MemoryOrServiceFabricRef ref;
      if (*refKind == "memory_service") {
        auto parsed = parseFabricRef<fabric::SystemMemoryServiceRef>(
            *fabricRef, (context + ".fabric_ref").str());
        if (!parsed)
          return parsed.takeError();
        ref = *parsed;
      } else if (*refKind == "service_endpoint") {
        auto parsed = parseFabricRef<fabric::SystemServiceEndpointRef>(
            *fabricRef, (context + ".fabric_ref").str());
        if (!parsed)
          return parsed.takeError();
        ref = *parsed;
      } else {
        return invalid(context + " has an unknown memory fabric_ref_kind");
      }
      return Gem5Correspondence(Gem5MemoryOrServiceCorrespondence{
          std::move(ref), std::move(*parsedObject), std::move(*parsedPort)});
    }
    Gem5TransportFabricRef ref;
    if (*refKind == "resource") {
      auto parsed = parseFabricRef<fabric::SystemTransportResourceRef>(
          *fabricRef, (context + ".fabric_ref").str());
      if (!parsed)
        return parsed.takeError();
      ref = *parsed;
    } else if (*refKind == "endpoint") {
      auto parsed = parseFabricRef<fabric::FabricTransportEndpointRef>(
          *fabricRef, (context + ".fabric_ref").str());
      if (!parsed)
        return parsed.takeError();
      ref = *parsed;
    } else {
      return invalid(context + " has an unknown transport fabric_ref_kind");
    }
    return Gem5Correspondence(Gem5TransportCorrespondence{
        std::move(ref), std::move(*parsedObject), std::move(*parsedPort)});
  }
  if (*kind == "external_endpoint") {
    if (llvm::Error error = rejectUnknownFields(
            object, context,
            {"kind", "fabric_external_endpoint_ref", "sim_object_ref",
             "sim_port_ref"}))
      return std::move(error);
    auto fabricRef =
        requireString(object, "fabric_external_endpoint_ref", context);
    auto simObject = requireObject(object, "sim_object_ref", context);
    auto simPort = requireObject(object, "sim_port_ref", context);
    if (!fabricRef)
      return fabricRef.takeError();
    if (!simObject)
      return simObject.takeError();
    if (!simPort)
      return simPort.takeError();
    auto parsedRef = parseFabricRef<fabric::ExternalBoundaryRef>(
        *fabricRef, (context + ".fabric_external_endpoint_ref").str());
    auto parsedObject =
        parseObject(**simObject, (context + ".sim_object_ref").str());
    auto parsedPort =
        parsePort(**simPort, (context + ".sim_port_ref").str());
    if (!parsedRef)
      return parsedRef.takeError();
    if (!parsedObject)
      return parsedObject.takeError();
    if (!parsedPort)
      return parsedPort.takeError();
    return Gem5Correspondence(Gem5ExternalEndpointCorrespondence{
        *parsedRef, std::move(*parsedObject), std::move(*parsedPort)});
  }
  return invalid(context + " has an unknown correspondence kind");
}

} // namespace

std::string
serializeGem5SimulationBinding(const Gem5SimulationBinding &binding) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", gem5SimulationBindingSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(gem5SimulationBindingSchema.version));
    json.attributeBegin("fabric_ref");
    writeRootReference(json, binding.fabric());
    json.attributeEnd();
    json.attributeBegin("interconnect_implementation_ref");
    writeRootReference(json, binding.interconnectImplementation());
    json.attributeEnd();
    json.attributeBegin("gem5_build_identity");
    json.object([&] {
      json.attribute("repository_identity",
                     binding.gem5BuildIdentity().repositoryIdentity);
      json.attribute("full_commit_identity",
                     binding.gem5BuildIdentity().fullCommitIdentity);
      json.attribute("build_configuration_digest",
                     binding.gem5BuildIdentity().buildConfigurationDigest);
      json.attribute("binary_fingerprint",
                     binding.gem5BuildIdentity().binaryFingerprint);
    });
    json.attributeEnd();
    json.attribute("bridge_abi_identity", binding.bridgeAbiIdentity());
    json.attributeArray("correspondences", [&] {
      for (const Gem5Correspondence &correspondence :
           binding.correspondences())
        writeCorrespondence(json, correspondence);
    });
  });
  return output.str().str();
}

llvm::Expected<Gem5SimulationBindingDraft>
parseGem5SimulationBinding(llvm::StringRef canonicalJson) {
  auto parsed = llvm::json::parse(canonicalJson);
  if (!parsed)
    return invalid("stored root is not valid JSON");
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("stored root must be a JSON object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "stored root",
          {"schema", "schema_version", "fabric_ref",
           "interconnect_implementation_ref", "gem5_build_identity",
           "bridge_abi_identity", "correspondences"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "stored root");
  auto schemaVersion =
      requireString(*root, "schema_version", "stored root");
  if (!schema)
    return schema.takeError();
  if (!schemaVersion)
    return schemaVersion.takeError();
  if (*schema != gem5SimulationBindingSchema.identity ||
      *schemaVersion != formatSchemaVersion(gem5SimulationBindingSchema.version))
    return invalid("stored root has the wrong schema descriptor");
  auto fabricObject = requireObject(*root, "fabric_ref", "stored root");
  auto interconnectObject =
      requireObject(*root, "interconnect_implementation_ref", "stored root");
  auto buildObject =
      requireObject(*root, "gem5_build_identity", "stored root");
  auto bridge = requireString(*root, "bridge_abi_identity", "stored root");
  auto correspondences =
      requireArray(*root, "correspondences", "stored root");
  if (!fabricObject)
    return fabricObject.takeError();
  if (!interconnectObject)
    return interconnectObject.takeError();
  if (!buildObject)
    return buildObject.takeError();
  if (!bridge)
    return bridge.takeError();
  if (!correspondences)
    return correspondences.takeError();
  auto fabric = parseRootReference(**fabricObject, "fabric_ref");
  auto interconnect = parseRootReference(**interconnectObject,
                                         "interconnect_implementation_ref");
  if (!fabric)
    return fabric.takeError();
  if (!interconnect)
    return interconnect.takeError();
  if (llvm::Error error = rejectUnknownFields(
          **buildObject, "gem5_build_identity",
          {"repository_identity", "full_commit_identity",
           "build_configuration_digest", "binary_fingerprint"}))
    return std::move(error);
  auto repository = requireString(**buildObject, "repository_identity",
                                  "gem5_build_identity");
  auto commit = requireString(**buildObject, "full_commit_identity",
                              "gem5_build_identity");
  auto digest = requireString(**buildObject, "build_configuration_digest",
                              "gem5_build_identity");
  auto binaryFingerprint = requireString(
      **buildObject, "binary_fingerprint", "gem5_build_identity");
  if (!repository)
    return repository.takeError();
  if (!commit)
    return commit.takeError();
  if (!digest)
    return digest.takeError();
  if (!binaryFingerprint)
    return binaryFingerprint.takeError();
  std::vector<Gem5Correspondence> rows;
  rows.reserve((*correspondences)->size());
  for (const auto &[index, value] : llvm::enumerate(**correspondences)) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return invalid("correspondences entry must be an object");
    auto row = parseCorrespondence(
        *object, "correspondences[" + std::to_string(index) + "]");
    if (!row)
      return row.takeError();
    rows.push_back(std::move(*row));
  }
  return Gem5SimulationBindingDraft{
      std::move(*fabric), std::move(*interconnect),
      Gem5BuildIdentity{repository->str(), commit->str(), digest->str(),
                        binaryFingerprint->str()},
      bridge->str(), std::move(rows)};
}

} // namespace loom::runtime::detail
