#include "Hardware/Implementation/ImplementationRepresentationRoot.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "representation_root_invalid: " + message);
}

llvm::Expected<llvm::StringRef>
variantSpelling(RepresentationRootVariant variant) {
  switch (variant) {
  case RepresentationRootVariant::Rtl:
    return llvm::StringRef("Rtl");
  case RepresentationRootVariant::GateNetlist:
    return llvm::StringRef("GateNetlist");
  case RepresentationRootVariant::AsicPhysical:
    return llvm::StringRef("AsicPhysical");
  case RepresentationRootVariant::FpgaPhysical:
    return llvm::StringRef("FpgaPhysical");
  case RepresentationRootVariant::FpgaImage:
    return llvm::StringRef("FpgaImage");
  }
  return invalid("representation root variant is unsupported");
}

std::optional<RepresentationRootVariant>
parseVariant(llvm::StringRef spelling) {
  if (spelling == "Rtl")
    return RepresentationRootVariant::Rtl;
  if (spelling == "GateNetlist")
    return RepresentationRootVariant::GateNetlist;
  if (spelling == "AsicPhysical")
    return RepresentationRootVariant::AsicPhysical;
  if (spelling == "FpgaPhysical")
    return RepresentationRootVariant::FpgaPhysical;
  if (spelling == "FpgaImage")
    return RepresentationRootVariant::FpgaImage;
  return std::nullopt;
}

llvm::Expected<llvm::StringRef>
stageSpelling(RepresentationPhysicalStage stage) {
  switch (stage) {
  case RepresentationPhysicalStage::Placed:
    return llvm::StringRef("Placed");
  case RepresentationPhysicalStage::Routed:
    return llvm::StringRef("Routed");
  case RepresentationPhysicalStage::Extracted:
    return llvm::StringRef("Extracted");
  }
  return invalid("representation physical stage is unsupported");
}

std::optional<RepresentationPhysicalStage>
parseStage(llvm::StringRef spelling) {
  if (spelling == "Placed")
    return RepresentationPhysicalStage::Placed;
  if (spelling == "Routed")
    return RepresentationPhysicalStage::Routed;
  if (spelling == "Extracted")
    return RepresentationPhysicalStage::Extracted;
  return std::nullopt;
}

bool variantHasStage(RepresentationRootVariant variant) {
  return variant == RepresentationRootVariant::AsicPhysical ||
         variant == RepresentationRootVariant::FpgaPhysical;
}

llvm::Expected<RepresentationObjectKind>
expectedTopKind(RepresentationRootVariant variant) {
  switch (variant) {
  case RepresentationRootVariant::Rtl:
  case RepresentationRootVariant::GateNetlist:
    return RepresentationObjectKind::Module;
  case RepresentationRootVariant::AsicPhysical:
    return RepresentationObjectKind::PhysicalObject;
  case RepresentationRootVariant::FpgaPhysical:
  case RepresentationRootVariant::FpgaImage:
    return RepresentationObjectKind::DeviceResource;
  }
  return invalid("representation root variant is unsupported");
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

class BinaryReader final {
public:
  explicit BinaryReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> readU32() {
    if (bytes_.size() - offset_ < sizeof(std::uint32_t))
      return invalid("truncated representation root");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64() {
    if (bytes_.size() - offset_ < sizeof(std::uint64_t))
      return invalid("truncated representation root");
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> readBytes(std::uint64_t size) {
    if (size > std::numeric_limits<std::size_t>::max() ||
        bytes_.size() - offset_ < static_cast<std::size_t>(size))
      return invalid("truncated representation root");
    const auto value = bytes_.slice(offset_, static_cast<std::size_t>(size));
    offset_ += static_cast<std::size_t>(size);
    return value;
  }

  bool empty() const { return offset_ == bytes_.size(); }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<RepresentationFormatDescriptorRef>
readFormatRef(BinaryReader &reader) {
  auto identitySize = reader.readU64();
  if (!identitySize)
    return identitySize.takeError();
  if (*identitySize > std::numeric_limits<std::size_t>::max() -
                          2 * sizeof(std::uint32_t) - sizeof(std::uint64_t))
    return invalid("truncated representation root");
  const std::uint64_t total =
      sizeof(std::uint64_t) + *identitySize + 3 * sizeof(std::uint32_t);
  auto bytes = reader.readBytes(total - sizeof(std::uint64_t));
  if (!bytes)
    return bytes.takeError();
  std::vector<std::uint8_t> framed;
  appendU64Be(framed, *identitySize);
  framed.insert(framed.end(), bytes->begin(), bytes->end());
  return decodeRepresentationFormatDescriptorRef(framed);
}

llvm::Expected<RepresentationLocator> readLocator(BinaryReader &reader) {
  auto kind = reader.readU32();
  if (!kind)
    return kind.takeError();
  auto nameSize = reader.readU64();
  if (!nameSize)
    return nameSize.takeError();
  auto nameBytes = reader.readBytes(*nameSize);
  if (!nameBytes)
    return nameBytes.takeError();
  std::vector<std::uint8_t> framed;
  appendU32Be(framed, *kind);
  appendU64Be(framed, *nameSize);
  framed.insert(framed.end(), nameBytes->begin(), nameBytes->end());
  return decodeRepresentationLocator(framed);
}

llvm::Expected<ImplementationPayload> readPayload(BinaryReader &reader) {
  auto role = reader.readU32();
  if (!role)
    return role.takeError();
  auto nameSize = reader.readU64();
  if (!nameSize)
    return nameSize.takeError();
  auto rest = reader.readBytes(*nameSize + 32);
  if (!rest)
    return rest.takeError();
  std::vector<std::uint8_t> framed;
  appendU32Be(framed, *role);
  appendU64Be(framed, *nameSize);
  framed.insert(framed.end(), rest->begin(), rest->end());
  return decodeImplementationPayload(framed);
}

llvm::Expected<const llvm::json::Object &>
jsonFieldObject(const llvm::json::Object &object, llvm::StringRef key) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return invalid("representation root field '" + key + "' is required");
  const llvm::json::Object *child = value->getAsObject();
  if (!child)
    return invalid("representation root field '" + key + "' must be an object");
  return *child;
}

} // namespace

llvm::Expected<ImplementationRepresentationRoot>
createImplementationRepresentationRoot(
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage,
    RepresentationFormatDescriptorRef formatRef, RepresentationLocator top,
    std::vector<ImplementationPayload> payloads) {
  auto canonical = canonicalizeImplementationPayloadCatalog(payloads);
  if (!canonical)
    return canonical.takeError();
  ImplementationRepresentationRoot root{variant, stage, formatRef,
                                        std::move(top), std::move(*canonical)};
  if (llvm::Error error = validateImplementationRepresentationRoot(root))
    return std::move(error);
  return root;
}

llvm::Error validateImplementationRepresentationRoot(
    const ImplementationRepresentationRoot &root) {
  auto spelling = variantSpelling(root.variant);
  if (!spelling)
    return spelling.takeError();
  if (variantHasStage(root.variant)) {
    if (!root.stage)
      return invalid("representation root variant requires an exact stage");
    auto stageText = stageSpelling(*root.stage);
    if (!stageText)
      return stageText.takeError();
    if (root.variant == RepresentationRootVariant::FpgaPhysical &&
        *root.stage == RepresentationPhysicalStage::Extracted)
      return invalid("FpgaPhysical root has no Extracted stage");
  } else if (root.stage) {
    return invalid("representation root variant carries no stage");
  }
  auto topKind = expectedTopKind(root.variant);
  if (!topKind)
    return topKind.takeError();
  if (root.top.kind != *topKind)
    return invalid("representation root top kind does not match its variant");
  auto canonical = canonicalizeImplementationPayloadCatalog(root.payloads);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != root.payloads)
    return invalid("representation root payloads are not in canonical order");
  return llvm::Error::success();
}

llvm::Error validateRepresentationRootAdmission(
    const RepresentationFormatDescriptor &descriptor,
    const ImplementationRepresentationRoot &root) {
  if (llvm::Error error = validateImplementationRepresentationRoot(root))
    return error;
  if (root.formatRef != descriptor.formatRef)
    return invalid("representation root format reference does not match the "
                   "selected descriptor");
  if (!admitsRepresentationRoot(descriptor, root.variant, root.stage))
    return invalid("representation root variant or stage is not admitted by "
                   "the selected descriptor");
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
encodeImplementationRepresentationRoot(
    const ImplementationRepresentationRoot &root) {
  if (llvm::Error error = validateImplementationRepresentationRoot(root))
    return std::move(error);
  std::vector<std::uint8_t> bytes;
  appendU32Be(bytes, static_cast<std::uint32_t>(root.variant));
  if (root.stage)
    appendU32Be(bytes, static_cast<std::uint32_t>(*root.stage));
  const std::vector<std::uint8_t> formatBytes =
      encodeRepresentationFormatDescriptorRef(root.formatRef);
  bytes.insert(bytes.end(), formatBytes.begin(), formatBytes.end());
  auto topBytes = encodeRepresentationLocator(root.top);
  if (!topBytes)
    return topBytes.takeError();
  bytes.insert(bytes.end(), topBytes->begin(), topBytes->end());
  appendU64Be(bytes, root.payloads.size());
  for (const ImplementationPayload &payload : root.payloads) {
    auto payloadBytes = encodeImplementationPayload(payload);
    if (!payloadBytes)
      return payloadBytes.takeError();
    bytes.insert(bytes.end(), payloadBytes->begin(), payloadBytes->end());
  }
  return bytes;
}

llvm::Expected<ImplementationRepresentationRoot>
decodeImplementationRepresentationRoot(llvm::ArrayRef<std::uint8_t> bytes) {
  BinaryReader reader(bytes);
  auto variantTag = reader.readU32();
  if (!variantTag)
    return variantTag.takeError();
  const auto variant = static_cast<RepresentationRootVariant>(*variantTag);
  if (auto spelling = variantSpelling(variant); !spelling)
    return spelling.takeError();
  std::optional<RepresentationPhysicalStage> stage;
  if (variantHasStage(variant)) {
    auto stageTag = reader.readU32();
    if (!stageTag)
      return stageTag.takeError();
    const auto value = static_cast<RepresentationPhysicalStage>(*stageTag);
    if (auto spelling = stageSpelling(value); !spelling)
      return spelling.takeError();
    stage = value;
  }
  auto formatRef = readFormatRef(reader);
  if (!formatRef)
    return formatRef.takeError();
  auto top = readLocator(reader);
  if (!top)
    return top.takeError();
  auto count = reader.readU64();
  if (!count)
    return count.takeError();
  // Every payload record costs at least 44 bytes (u32be role, u64be name
  // length, and the 32 digest bytes), so an oversized count is truncation.
  if (*count > reader.remaining() / 44)
    return invalid("truncated representation root");
  std::vector<ImplementationPayload> payloads;
  payloads.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto payload = readPayload(reader);
    if (!payload)
      return payload.takeError();
    payloads.push_back(std::move(*payload));
  }
  if (!reader.empty())
    return invalid("representation root has trailing bytes");
  // Decoding must not silently canonicalize: a noncanonical record is
  // rejected rather than repaired.
  ImplementationRepresentationRoot root{variant, stage, *formatRef,
                                        std::move(*top), std::move(payloads)};
  if (llvm::Error error = validateImplementationRepresentationRoot(root))
    return std::move(error);
  return root;
}

llvm::Expected<std::string> serializeImplementationRepresentationRootJson(
    const ImplementationRepresentationRoot &root) {
  if (llvm::Error error = validateImplementationRepresentationRoot(root))
    return std::move(error);
  auto variantText = variantSpelling(root.variant);
  if (!variantText)
    return variantText.takeError();
  std::string result = "{\"variant\":\"" + variantText->str() + "\",";
  if (root.stage) {
    auto stageText = stageSpelling(*root.stage);
    if (!stageText)
      return stageText.takeError();
    result += "\"stage\":\"" + stageText->str() + "\",";
  }
  result += "\"format_ref\":";
  result += serializeRepresentationFormatDescriptorRefJson(root.formatRef);
  result += ",\"top\":";
  auto topJson = serializeRepresentationLocatorJson(root.top);
  if (!topJson)
    return topJson.takeError();
  result += *topJson;
  result += ",\"payloads\":[";
  bool first = true;
  for (const ImplementationPayload &payload : root.payloads) {
    if (!first)
      result += ",";
    first = false;
    auto payloadJson = serializeImplementationPayloadJson(payload);
    if (!payloadJson)
      return payloadJson.takeError();
    result += *payloadJson;
  }
  result += "]}";
  return result;
}

llvm::Expected<ImplementationRepresentationRoot>
parseImplementationRepresentationRootJsonValue(
    const llvm::json::Object &object) {
  for (const auto &field : object) {
    const llvm::StringRef key = field.getFirst();
    if (key != "variant" && key != "stage" && key != "format_ref" &&
        key != "top" && key != "payloads")
      return invalid("representation root has unknown field '" + key + "'");
  }

  const std::optional<llvm::StringRef> variantText =
      object.getString("variant");
  if (!variantText)
    return invalid("representation root field 'variant' must be a string");
  const std::optional<RepresentationRootVariant> variant =
      parseVariant(*variantText);
  if (!variant)
    return invalid("representation root variant is unsupported");

  std::optional<RepresentationPhysicalStage> stage;
  if (const llvm::json::Value *stageValue = object.get("stage")) {
    if (!variantHasStage(*variant))
      return invalid("representation root variant carries no stage");
    if (const std::optional<llvm::StringRef> stageText =
            stageValue->getAsString()) {
      stage = parseStage(*stageText);
    }
    if (!stage)
      return invalid("representation root stage is unsupported");
  } else if (variantHasStage(*variant)) {
    return invalid("representation root variant requires an exact stage");
  }

  auto formatObject = jsonFieldObject(object, "format_ref");
  if (!formatObject)
    return formatObject.takeError();
  auto formatRef =
      parseRepresentationFormatDescriptorRefJsonValue(*formatObject);
  if (!formatRef)
    return formatRef.takeError();

  auto topObject = jsonFieldObject(object, "top");
  if (!topObject)
    return topObject.takeError();
  auto top = parseRepresentationLocatorJsonValue(*topObject);
  if (!top)
    return top.takeError();

  const llvm::json::Array *payloads = object.getArray("payloads");
  if (!payloads)
    return invalid("representation root field 'payloads' must be an array");
  std::vector<ImplementationPayload> catalog;
  catalog.reserve(payloads->size());
  for (const llvm::json::Value &entry : *payloads) {
    const llvm::json::Object *payloadObject = entry.getAsObject();
    if (!payloadObject)
      return invalid("representation root payload must be an object");
    auto payload = parseImplementationPayloadJsonValue(*payloadObject);
    if (!payload)
      return payload.takeError();
    catalog.push_back(std::move(*payload));
  }

  auto root = createImplementationRepresentationRoot(
      *variant, stage, *formatRef, std::move(*top), std::move(catalog));
  if (!root)
    return root.takeError();
  return root;
}

llvm::Expected<ImplementationRepresentationRoot>
parseImplementationRepresentationRootJson(llvm::StringRef bytes) {
  auto parsed = llvm::json::parse(bytes);
  if (!parsed)
    return invalid("invalid representation root JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("representation root JSON must be an object");
  auto root = parseImplementationRepresentationRootJsonValue(*object);
  if (!root)
    return root.takeError();
  auto canonical = serializeImplementationRepresentationRootJson(*root);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != bytes)
    return invalid("representation root JSON is not canonical");
  return root;
}

} // namespace loom::hardware
