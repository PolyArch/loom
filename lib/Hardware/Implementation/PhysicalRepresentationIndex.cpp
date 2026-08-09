#include "Hardware/Implementation/PhysicalRepresentationIndex.h"

#include "Common/BlobDigest.h"
#include "RepresentationIndexInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "physical_representation_index_invalid: " +
                                     message);
}

llvm::Expected<const llvm::json::Object &>
requireObject(const llvm::json::Object &object, llvm::StringRef key) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return invalid("field '" + key + "' is required");
  const llvm::json::Object *child = value->getAsObject();
  if (!child)
    return invalid("field '" + key + "' must be an object");
  return *child;
}

bool rootedAt(const RepresentationLocator &locator,
              const RepresentationLocator &top) {
  if (locator == top)
    return true;
  const llvm::StringRef name(locator.canonicalName);
  const llvm::StringRef root(top.canonicalName);
  return name.starts_with(root) && name.size() > root.size() &&
         name[root.size()] == '.';
}

llvm::Expected<llvm::StringRef>
directionSpelling(RepresentationSignalDirection direction) {
  switch (direction) {
  case RepresentationSignalDirection::Input:
    return "Input";
  case RepresentationSignalDirection::Output:
    return "Output";
  case RepresentationSignalDirection::Inout:
    return "Inout";
  }
  return invalid("signal geometry direction is unsupported");
}

std::optional<RepresentationSignalDirection>
parseDirection(llvm::StringRef spelling) {
  if (spelling == "Input")
    return RepresentationSignalDirection::Input;
  if (spelling == "Output")
    return RepresentationSignalDirection::Output;
  if (spelling == "Inout")
    return RepresentationSignalDirection::Inout;
  return std::nullopt;
}

llvm::Expected<RepresentationSignalGeometry>
parseGeometry(const llvm::json::Object &object) {
  constexpr std::array<llvm::StringLiteral, 2> fields{"direction", "bit_width"};
  for (const auto &field : object) {
    const llvm::StringRef key = field.getFirst();
    if (key != fields[0] && key != fields[1])
      return invalid("signal geometry has unknown field '" + key + "'");
  }
  if (object.size() != fields.size())
    return invalid(
        "signal geometry requires exactly direction and bit_width fields");
  const std::optional<llvm::StringRef> directionText =
      object.getString("direction");
  if (!directionText)
    return invalid("signal geometry direction must be a string");
  const std::optional<RepresentationSignalDirection> direction =
      parseDirection(*directionText);
  if (!direction)
    return invalid("signal geometry direction is unsupported");
  const llvm::json::Value *widthValue = object.get("bit_width");
  const std::optional<std::uint64_t> bitWidth =
      widthValue ? widthValue->getAsUINT64() : std::nullopt;
  if (!bitWidth)
    return invalid("signal geometry bit_width must be an unsigned integer");
  return RepresentationSignalGeometry{*direction, *bitWidth};
}

llvm::Expected<PhysicalRepresentationObject>
parsePhysicalObject(const llvm::json::Object &object) {
  for (const auto &field : object) {
    const llvm::StringRef key = field.getFirst();
    if (key != "locator" && key != "signal_geometry")
      return invalid("physical object has unknown field '" + key + "'");
  }
  if (object.size() < 1 || object.size() > 2)
    return invalid("physical object requires locator and optional "
                   "signal_geometry fields");
  auto locatorObject = requireObject(object, "locator");
  if (!locatorObject)
    return locatorObject.takeError();
  auto locator = parseRepresentationLocatorJsonValue(*locatorObject);
  if (!locator)
    return locator.takeError();
  std::optional<RepresentationSignalGeometry> geometry;
  if (object.get("signal_geometry")) {
    auto geometryObject = requireObject(object, "signal_geometry");
    if (!geometryObject)
      return geometryObject.takeError();
    auto parsed = parseGeometry(*geometryObject);
    if (!parsed)
      return parsed.takeError();
    geometry = *parsed;
  }
  return PhysicalRepresentationObject{std::move(*locator), geometry};
}

void writeLocator(llvm::json::OStream &json,
                  const RepresentationLocator &locator) {
  json.rawValue(llvm::cantFail(serializeRepresentationLocatorJson(locator)));
}

void writePayload(llvm::json::OStream &json,
                  const ImplementationPayload &payload) {
  json.rawValue(llvm::cantFail(serializeImplementationPayloadJson(payload)));
}

void writeObject(llvm::json::OStream &json,
                 const PhysicalRepresentationObject &object) {
  json.object([&] {
    json.attributeBegin("locator");
    writeLocator(json, object.locator);
    json.attributeEnd();
    if (object.signalGeometry) {
      json.attributeObject("signal_geometry", [&] {
        json.attribute("direction", llvm::cantFail(directionSpelling(
                                        object.signalGeometry->direction)));
        json.attribute("bit_width", object.signalGeometry->bitWidth);
      });
    }
  });
}

} // namespace

llvm::Expected<PhysicalRepresentationIndexPayload>
createPhysicalRepresentationIndexPayload(
    RepresentationFormatDescriptorRef formatRef,
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage, RepresentationLocator top,
    std::string indexLogicalName, std::vector<ImplementationPayload> payloads,
    std::vector<PhysicalRepresentationObject> objects,
    std::vector<RepresentationLocator> unresolvedExternalDefinitions) {
  auto canonicalPayloads = canonicalizeImplementationPayloadCatalog(payloads);
  if (!canonicalPayloads)
    return canonicalPayloads.takeError();
  llvm::sort(objects, [](const PhysicalRepresentationObject &lhs,
                         const PhysicalRepresentationObject &rhs) {
    return representationLocatorCanonicalLess(lhs.locator, rhs.locator);
  });
  llvm::sort(unresolvedExternalDefinitions, representationLocatorCanonicalLess);
  PhysicalRepresentationIndexPayload index{
      formatRef,
      variant,
      stage,
      std::move(top),
      std::move(indexLogicalName),
      std::move(*canonicalPayloads),
      std::move(objects),
      std::move(unresolvedExternalDefinitions)};
  if (llvm::Error error = validatePhysicalRepresentationIndexPayload(index))
    return std::move(error);
  return index;
}

llvm::Error validatePhysicalRepresentationIndexPayload(
    const PhysicalRepresentationIndexPayload &index) {
  if (index.formatRef.kind() != RepresentationFormatKind::IndexedPhysical)
    return invalid("format_ref is not indexed_physical");
  const RepresentationFormatDescriptor &descriptor =
      getRepresentationFormatDescriptor(index.formatRef);
  const RepresentationRootAdmission *admission =
      findRepresentationRootAdmission(descriptor, index.variant, index.stage);
  if (!admission)
    return invalid("variant and stage do not name a physical root admission");
  if (index.top.kind != admission->exactRootKind)
    return invalid("top kind does not match the physical root admission");
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(index.formatRef, index.top))
    return invalid("top locator is invalid: " +
                   llvm::toString(std::move(error)));
  if (llvm::StringRef(index.top.canonicalName).contains('.'))
    return invalid("top locator must be exactly one identifier");

  const BlobDigest emptyDigest = computeBlobDigest({});
  const ImplementationPayload syntheticIndex{
      PayloadRole::RepresentationIndex, index.indexLogicalName, emptyDigest};
  if (llvm::Error error = validateImplementationPayload(syntheticIndex))
    return invalid("index_logical_name is invalid: " +
                   llvm::toString(std::move(error)));
  auto canonicalPayloads =
      canonicalizeImplementationPayloadCatalog(index.payloads);
  if (!canonicalPayloads)
    return invalid("payload catalog is invalid: " +
                   llvm::toString(canonicalPayloads.takeError()));
  if (*canonicalPayloads != index.payloads)
    return invalid("payload catalog is not in canonical order");
  if (llvm::any_of(index.payloads, [](const ImplementationPayload &payload) {
        return payload.role == PayloadRole::RepresentationIndex;
      }))
    return invalid("payload catalog cannot contain RepresentationIndex");
  std::vector<ImplementationPayload> completePayloads(index.payloads.begin(),
                                                      index.payloads.end());
  completePayloads.push_back(syntheticIndex);
  if (llvm::Error error =
          validateRepresentationPayloadCatalog(*admission, completePayloads))
    return invalid("payload catalog violates the root admission: " +
                   llvm::toString(std::move(error)));

  if (index.objects.empty())
    return invalid("object catalog must be nonempty");
  std::vector<RepresentationLocator> objectLocators;
  objectLocators.reserve(index.objects.size());
  for (const PhysicalRepresentationObject &object : index.objects)
    objectLocators.push_back(object.locator);
  llvm::sort(objectLocators, representationLocatorCanonicalLess);
  if (std::adjacent_find(objectLocators.begin(), objectLocators.end()) !=
      objectLocators.end())
    return invalid("object catalog contains a duplicate locator");
  std::size_t topCount = 0;
  std::vector<RepresentationLocator> indexedModules;
  for (std::size_t ordinal = 0; ordinal < index.objects.size(); ++ordinal) {
    const PhysicalRepresentationObject &object = index.objects[ordinal];
    if (ordinal != 0) {
      const RepresentationLocator &previous =
          index.objects[ordinal - 1].locator;
      if (!representationLocatorCanonicalLess(previous, object.locator))
        return invalid("object catalog is not in canonical order between '" +
                       previous.canonicalName + "' and '" +
                       object.locator.canonicalName + "'");
    }
    if (!llvm::is_contained(admission->admittedObjectKinds,
                            object.locator.kind))
      return invalid("object kind is not admitted by the physical root");
    if (llvm::Error error = validateRepresentationLocatorSyntax(index.formatRef,
                                                                object.locator))
      return invalid("object locator is invalid: " +
                     llvm::toString(std::move(error)));
    if (object.locator == index.top)
      ++topCount;
    else if (object.locator.kind == RepresentationObjectKind::Module)
      indexedModules.push_back(object.locator);
    else if (!rootedAt(object.locator, index.top))
      return invalid("ordinary object locator is not rooted at top");

    const bool terminal =
        object.locator.kind == RepresentationObjectKind::Port ||
        object.locator.kind == RepresentationObjectKind::Pin;
    if (terminal != object.signalGeometry.has_value())
      return invalid("Port and Pin objects require signal geometry and every "
                     "other object must omit it");
    if (object.signalGeometry) {
      auto direction = directionSpelling(object.signalGeometry->direction);
      if (!direction)
        return direction.takeError();
      if (object.signalGeometry->bitWidth == 0)
        return invalid("signal geometry bit_width must be positive");
    }
  }
  if (topCount != 1)
    return invalid("object catalog must contain the exact top once");

  std::vector<RepresentationLocator> canonicalUnresolved(
      index.unresolvedExternalDefinitions.begin(),
      index.unresolvedExternalDefinitions.end());
  llvm::sort(canonicalUnresolved, representationLocatorCanonicalLess);
  if (std::adjacent_find(canonicalUnresolved.begin(),
                         canonicalUnresolved.end()) !=
      canonicalUnresolved.end())
    return invalid(
        "unresolved external definitions contain a duplicate locator");
  for (std::size_t ordinal = 0;
       ordinal < index.unresolvedExternalDefinitions.size(); ++ordinal) {
    const RepresentationLocator &locator =
        index.unresolvedExternalDefinitions[ordinal];
    if (locator.kind != RepresentationObjectKind::Module)
      return invalid("unresolved external definition must be a Module");
    if (llvm::Error error =
            validateRepresentationLocatorSyntax(index.formatRef, locator))
      return invalid("unresolved external definition is invalid: " +
                     llvm::toString(std::move(error)));
    if (ordinal != 0) {
      const RepresentationLocator &previous =
          index.unresolvedExternalDefinitions[ordinal - 1];
      if (!representationLocatorCanonicalLess(previous, locator))
        return invalid(
            "unresolved external definitions are not in canonical order");
    }
  }
  if (indexedModules != index.unresolvedExternalDefinitions)
    return invalid("unresolved external definitions do not equal the indexed "
                   "Module object set");
  return llvm::Error::success();
}

llvm::Expected<std::string> serializePhysicalRepresentationIndexPayloadJson(
    const PhysicalRepresentationIndexPayload &index) {
  if (llvm::Error error = validatePhysicalRepresentationIndexPayload(index))
    return std::move(error);
  const llvm::StringRef variant =
      llvm::cantFail(representationRootVariantSpelling(index.variant));
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attributeBegin("format_ref");
    json.rawValue(
        serializeRepresentationFormatDescriptorRefJson(index.formatRef));
    json.attributeEnd();
    json.attribute("variant", variant);
    if (index.stage)
      json.attribute(
          "stage",
          llvm::cantFail(representationPhysicalStageSpelling(*index.stage)));
    json.attributeBegin("top");
    writeLocator(json, index.top);
    json.attributeEnd();
    json.attribute("index_logical_name", index.indexLogicalName);
    json.attributeArray("payloads", [&] {
      for (const ImplementationPayload &payload : index.payloads)
        writePayload(json, payload);
    });
    json.attributeArray("objects", [&] {
      for (const PhysicalRepresentationObject &object : index.objects)
        writeObject(json, object);
    });
    json.attributeArray("unresolved_external_definitions", [&] {
      for (const RepresentationLocator &locator :
           index.unresolvedExternalDefinitions)
        writeLocator(json, locator);
    });
  });
  return storage.str().str();
}

llvm::Expected<PhysicalRepresentationIndexPayload>
parsePhysicalRepresentationIndexPayloadJson(llvm::StringRef bytes) {
  auto parsed = llvm::json::parse(bytes);
  if (!parsed)
    return invalid("invalid JSON: " + llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("JSON must be an object");
  constexpr std::array<llvm::StringLiteral, 8> fields{
      "format_ref",
      "variant",
      "stage",
      "top",
      "index_logical_name",
      "payloads",
      "objects",
      "unresolved_external_definitions"};
  for (const auto &field : *object) {
    const llvm::StringRef key = field.getFirst();
    if (std::find(fields.begin(), fields.end(), key) == fields.end())
      return invalid("index has unknown field '" + key + "'");
  }

  auto formatObject = requireObject(*object, "format_ref");
  if (!formatObject)
    return formatObject.takeError();
  auto formatRef =
      parseRepresentationFormatDescriptorRefJsonValue(*formatObject);
  if (!formatRef)
    return formatRef.takeError();
  const std::optional<llvm::StringRef> variantText =
      object->getString("variant");
  if (!variantText)
    return invalid("field 'variant' must be a string");
  const std::optional<RepresentationRootVariant> variant =
      parseRepresentationRootVariantSpelling(*variantText);
  if (!variant)
    return invalid("variant is unsupported");
  std::optional<RepresentationPhysicalStage> stage;
  if (const llvm::json::Value *stageValue = object->get("stage")) {
    const std::optional<llvm::StringRef> stageText = stageValue->getAsString();
    if (!stageText)
      return invalid("field 'stage' must be a string");
    stage = parseRepresentationPhysicalStageSpelling(*stageText);
    if (!stage)
      return invalid("stage is unsupported");
  }
  const bool staged = *variant == RepresentationRootVariant::AsicPhysical ||
                      *variant == RepresentationRootVariant::FpgaPhysical;
  if (staged != stage.has_value())
    return invalid("stage presence does not match the physical variant");
  if (object->size() != fields.size() - (staged ? 0 : 1))
    return invalid("index does not contain its exact canonical field set");

  auto topObject = requireObject(*object, "top");
  if (!topObject)
    return topObject.takeError();
  auto top = parseRepresentationLocatorJsonValue(*topObject);
  if (!top)
    return top.takeError();
  const std::optional<llvm::StringRef> indexLogicalName =
      object->getString("index_logical_name");
  if (!indexLogicalName)
    return invalid("field 'index_logical_name' must be a string");

  const llvm::json::Array *payloadArray = object->getArray("payloads");
  if (!payloadArray)
    return invalid("field 'payloads' must be an array");
  std::vector<ImplementationPayload> payloads;
  payloads.reserve(payloadArray->size());
  for (const llvm::json::Value &value : *payloadArray) {
    const llvm::json::Object *payloadObject = value.getAsObject();
    if (!payloadObject)
      return invalid("payload entry must be an object");
    auto payload = parseImplementationPayloadJsonValue(*payloadObject);
    if (!payload)
      return payload.takeError();
    payloads.push_back(std::move(*payload));
  }

  const llvm::json::Array *objectArray = object->getArray("objects");
  if (!objectArray)
    return invalid("field 'objects' must be an array");
  std::vector<PhysicalRepresentationObject> objects;
  objects.reserve(objectArray->size());
  for (const llvm::json::Value &value : *objectArray) {
    const llvm::json::Object *physicalObject = value.getAsObject();
    if (!physicalObject)
      return invalid("physical object entry must be an object");
    auto entry = parsePhysicalObject(*physicalObject);
    if (!entry)
      return entry.takeError();
    objects.push_back(std::move(*entry));
  }

  const llvm::json::Array *unresolvedArray =
      object->getArray("unresolved_external_definitions");
  if (!unresolvedArray)
    return invalid("field 'unresolved_external_definitions' must be an array");
  std::vector<RepresentationLocator> unresolved;
  unresolved.reserve(unresolvedArray->size());
  for (const llvm::json::Value &value : *unresolvedArray) {
    const llvm::json::Object *locatorObject = value.getAsObject();
    if (!locatorObject)
      return invalid("unresolved definition entry must be an object");
    auto locator = parseRepresentationLocatorJsonValue(*locatorObject);
    if (!locator)
      return locator.takeError();
    unresolved.push_back(std::move(*locator));
  }

  PhysicalRepresentationIndexPayload index{*formatRef,
                                           *variant,
                                           stage,
                                           std::move(*top),
                                           indexLogicalName->str(),
                                           std::move(payloads),
                                           std::move(objects),
                                           std::move(unresolved)};
  if (llvm::Error error = validatePhysicalRepresentationIndexPayload(index))
    return std::move(error);
  auto canonical = serializePhysicalRepresentationIndexPayloadJson(index);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != bytes)
    return invalid("JSON is not canonical");
  return index;
}

namespace detail {

llvm::Expected<RawIndex> indexPhysicalRepresentation(
    RepresentationFormatDescriptorRef formatRef,
    const RepresentationLocator &exactRoot,
    llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
    const BlobStore &blobs) {
  auto canonical = canonicalizeImplementationPayloadCatalog(canonicalPayloads);
  if (!canonical)
    return invalidIndex("payload closure is invalid: " +
                        llvm::toString(canonical.takeError()));
  if (!llvm::equal(*canonical, canonicalPayloads))
    return invalidIndex("payload closure is not in canonical order");

  const ImplementationPayload *indexDescriptor = nullptr;
  std::vector<ImplementationPayload> nonIndexPayloads;
  std::vector<std::uint8_t> indexBytes;
  for (const ImplementationPayload &payload : canonicalPayloads) {
    auto contents = blobs.get(payload.blobDigest);
    if (!contents)
      return invalidIndex(
          "payload '" + payload.canonicalLogicalName +
          "' could not be loaded: " + llvm::toString(contents.takeError()));
    if (payload.role == PayloadRole::RepresentationIndex) {
      if (indexDescriptor)
        return invalidIndex(
            "RepresentationIndex payload role cardinality is above its "
            "maximum");
      indexDescriptor = &payload;
      indexBytes.assign(contents->begin(), contents->end());
      if (llvm::Error error = validateRepresentationTextPolicy(
              RepresentationTextPolicy::Utf8LfNoNul, payload, *contents))
        return std::move(error);
    } else {
      nonIndexPayloads.push_back(payload);
    }
  }
  if (!indexDescriptor)
    return invalidIndex(
        "RepresentationIndex payload role cardinality is below its minimum");

  auto index = parsePhysicalRepresentationIndexPayloadJson(llvm::StringRef(
      reinterpret_cast<const char *>(indexBytes.data()), indexBytes.size()));
  if (!index)
    return invalidIndex("RepresentationIndex payload is invalid: " +
                        llvm::toString(index.takeError()));
  if (index->formatRef != formatRef)
    return invalidIndex("RepresentationIndex format claim is foreign");
  if (!(index->top == exactRoot))
    return invalidIndex("RepresentationIndex top claim is foreign");
  if (index->indexLogicalName != indexDescriptor->canonicalLogicalName)
    return invalidIndex("RepresentationIndex logical-name claim is foreign");
  if (index->payloads != nonIndexPayloads)
    return invalidIndex(
        "RepresentationIndex payload catalog does not match the outer "
        "payload catalog");

  const RepresentationFormatDescriptor &descriptor =
      getRepresentationFormatDescriptor(formatRef);
  const RepresentationRootAdmission *admission =
      findRepresentationRootAdmission(descriptor, index->variant, index->stage);
  if (!admission)
    return invalidIndex("RepresentationIndex root claim is not admitted");
  if (llvm::Error error =
          validateRepresentationPayloadCatalog(*admission, canonicalPayloads))
    return invalidIndex("payload catalog violates the root admission: " +
                        llvm::toString(std::move(error)));

  RawIndex raw;
  raw.rootVariant = index->variant;
  raw.stage = index->stage;
  raw.entries.reserve(index->objects.size());
  for (PhysicalRepresentationObject &object : index->objects) {
    RepresentationObjectFacts facts{object.locator.kind,
                                    std::move(object.signalGeometry)};
    raw.entries.push_back({std::move(object.locator), std::move(facts)});
  }
  raw.unresolved = std::move(index->unresolvedExternalDefinitions);
  return raw;
}

} // namespace detail
} // namespace loom::hardware
