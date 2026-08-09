#include "Hardware/Implementation/ImplementationPayload.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
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
                                 "implementation_payload_invalid: " + message);
}

llvm::Expected<llvm::StringRef> roleSpelling(PayloadRole role) {
  switch (role) {
  case PayloadRole::RtlSource:
    return llvm::StringRef("RtlSource");
  case PayloadRole::Netlist:
    return llvm::StringRef("Netlist");
  case PayloadRole::PhysicalDatabase:
    return llvm::StringRef("PhysicalDatabase");
  case PayloadRole::Parasitics:
    return llvm::StringRef("Parasitics");
  case PayloadRole::LayoutStream:
    return llvm::StringRef("LayoutStream");
  case PayloadRole::DeviceImage:
    return llvm::StringRef("DeviceImage");
  case PayloadRole::GenerationConstraint:
    return llvm::StringRef("GenerationConstraint");
  case PayloadRole::BlackBoxContract:
    return llvm::StringRef("BlackBoxContract");
  case PayloadRole::RepresentationIndex:
    return llvm::StringRef("RepresentationIndex");
  }
  return invalid("payload role is unsupported");
}

std::optional<PayloadRole> parseRole(llvm::StringRef spelling) {
  if (spelling == "RtlSource")
    return PayloadRole::RtlSource;
  if (spelling == "Netlist")
    return PayloadRole::Netlist;
  if (spelling == "PhysicalDatabase")
    return PayloadRole::PhysicalDatabase;
  if (spelling == "Parasitics")
    return PayloadRole::Parasitics;
  if (spelling == "LayoutStream")
    return PayloadRole::LayoutStream;
  if (spelling == "DeviceImage")
    return PayloadRole::DeviceImage;
  if (spelling == "GenerationConstraint")
    return PayloadRole::GenerationConstraint;
  if (spelling == "BlackBoxContract")
    return PayloadRole::BlackBoxContract;
  if (spelling == "RepresentationIndex")
    return PayloadRole::RepresentationIndex;
  return std::nullopt;
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
      return invalid("truncated implementation payload");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64() {
    if (bytes_.size() - offset_ < sizeof(std::uint64_t))
      return invalid("truncated implementation payload");
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> readBytes(std::uint64_t size) {
    if (size > std::numeric_limits<std::size_t>::max() ||
        bytes_.size() - offset_ < static_cast<std::size_t>(size))
      return invalid("truncated implementation payload");
    const auto value = bytes_.slice(offset_, static_cast<std::size_t>(size));
    offset_ += static_cast<std::size_t>(size);
    return value;
  }

  bool empty() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

bool byteStringLess(llvm::StringRef lhs, llvm::StringRef rhs) {
  return std::lexicographical_compare(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
      [](char lhsByte, char rhsByte) {
        return static_cast<std::uint8_t>(lhsByte) <
               static_cast<std::uint8_t>(rhsByte);
      });
}

} // namespace

llvm::Error
validateImplementationPayload(const ImplementationPayload &payload) {
  auto spelling = roleSpelling(payload.role);
  if (!spelling)
    return spelling.takeError();
  const llvm::StringRef name(payload.canonicalLogicalName);
  if (name.empty() || name.contains('\0'))
    return invalid("payload logical name must be nonempty and contain no NUL");
  if (!llvm::json::isUTF8(name))
    return invalid("payload logical name must be valid UTF-8");

  std::size_t offset = 0;
  while (offset <= name.size()) {
    const std::size_t separator = name.find('/', offset);
    const std::size_t end =
        separator == llvm::StringRef::npos ? name.size() : separator;
    const llvm::StringRef segment = name.slice(offset, end);
    if (segment.empty() || segment == "." || segment == "..")
      return invalid("payload logical name is not a normalized relative path");
    if (separator == llvm::StringRef::npos)
      break;
    offset = separator + 1;
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
encodeImplementationPayload(const ImplementationPayload &payload) {
  if (llvm::Error error = validateImplementationPayload(payload))
    return std::move(error);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                payload.canonicalLogicalName.size() + BlobDigest::byteSize);
  appendU32Be(bytes, static_cast<std::uint32_t>(payload.role));
  appendU64Be(bytes, payload.canonicalLogicalName.size());
  bytes.insert(bytes.end(), payload.canonicalLogicalName.begin(),
               payload.canonicalLogicalName.end());
  bytes.insert(bytes.end(), payload.blobDigest.bytes().begin(),
               payload.blobDigest.bytes().end());
  return bytes;
}

llvm::Expected<ImplementationPayload>
decodeImplementationPayload(llvm::ArrayRef<std::uint8_t> bytes) {
  BinaryReader reader(bytes);
  auto roleTag = reader.readU32();
  if (!roleTag)
    return roleTag.takeError();
  const auto role = static_cast<PayloadRole>(*roleTag);
  auto spelling = roleSpelling(role);
  if (!spelling)
    return spelling.takeError();
  auto nameSize = reader.readU64();
  if (!nameSize)
    return nameSize.takeError();
  auto nameBytes = reader.readBytes(*nameSize);
  if (!nameBytes)
    return nameBytes.takeError();
  auto digestBytes = reader.readBytes(BlobDigest::byteSize);
  if (!digestBytes)
    return digestBytes.takeError();
  if (!reader.empty())
    return invalid("implementation payload has trailing bytes");
  auto blobDigest = BlobDigest::fromBytes(*digestBytes);
  if (!blobDigest)
    return blobDigest.takeError();
  ImplementationPayload payload{
      role,
      std::string(reinterpret_cast<const char *>(nameBytes->data()),
                  nameBytes->size()),
      *blobDigest,
  };
  if (llvm::Error error = validateImplementationPayload(payload))
    return std::move(error);
  return payload;
}

llvm::Expected<std::string>
serializeImplementationPayloadJson(const ImplementationPayload &payload) {
  if (llvm::Error error = validateImplementationPayload(payload))
    return std::move(error);
  auto spelling = roleSpelling(payload.role);
  if (!spelling)
    return spelling.takeError();
  llvm::SmallString<192> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("role", *spelling);
    json.attribute("canonical_logical_name", payload.canonicalLogicalName);
    json.attribute("blob_digest", formatBlobDigestHex(payload.blobDigest));
  });
  return storage.str().str();
}

llvm::Expected<ImplementationPayload>
parseImplementationPayloadJsonValue(const llvm::json::Object &object) {
  constexpr std::array<llvm::StringLiteral, 3> fields{
      "role", "canonical_logical_name", "blob_digest"};
  for (const auto &field : object) {
    const llvm::StringRef key = field.getFirst();
    if (std::find(fields.begin(), fields.end(), key) == fields.end())
      return invalid("implementation payload has unknown field '" + key + "'");
  }
  if (object.size() != fields.size())
    return invalid("implementation payload requires exactly role, "
                   "canonical_logical_name, and blob_digest fields");

  const std::optional<llvm::StringRef> roleText = object.getString("role");
  if (!roleText)
    return invalid("implementation payload field 'role' must be a string");
  const std::optional<PayloadRole> role = parseRole(*roleText);
  if (!role)
    return invalid("implementation payload role is unsupported");
  const std::optional<llvm::StringRef> name =
      object.getString("canonical_logical_name");
  if (!name)
    return invalid("implementation payload field 'canonical_logical_name' "
                   "must be a string");
  const std::optional<llvm::StringRef> digestText =
      object.getString("blob_digest");
  if (!digestText)
    return invalid(
        "implementation payload field 'blob_digest' must be a string");
  auto blobDigest = parseBlobDigestHex(*digestText);
  if (!blobDigest)
    return invalid("implementation payload blob digest is invalid: " +
                   llvm::toString(blobDigest.takeError()));
  return ImplementationPayload{*role, name->str(), *blobDigest};
}

llvm::Expected<ImplementationPayload>
parseImplementationPayloadJson(llvm::StringRef bytes) {
  auto parsed = llvm::json::parse(bytes);
  if (!parsed)
    return invalid("invalid implementation payload JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("implementation payload JSON must be an object");
  auto payload = parseImplementationPayloadJsonValue(*object);
  if (!payload)
    return payload.takeError();
  auto canonical = serializeImplementationPayloadJson(*payload);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != bytes)
    return invalid("implementation payload JSON is not canonical");
  return *payload;
}

llvm::Expected<std::vector<ImplementationPayload>>
canonicalizeImplementationPayloadCatalog(
    llvm::ArrayRef<ImplementationPayload> payloads) {
  if (payloads.empty())
    return invalid("implementation payload catalog must be nonempty");
  std::vector<ImplementationPayload> canonical(payloads.begin(),
                                               payloads.end());
  for (const ImplementationPayload &payload : canonical)
    if (llvm::Error error = validateImplementationPayload(payload))
      return std::move(error);
  llvm::sort(canonical, implementationPayloadCanonicalLess);
  for (std::size_t index = 1; index < canonical.size(); ++index)
    if (canonical[index - 1].role == canonical[index].role &&
        canonical[index - 1].canonicalLogicalName ==
            canonical[index].canonicalLogicalName)
      return invalid(
          "implementation payload catalog contains a duplicate role/name");
  return canonical;
}

bool implementationPayloadCanonicalLess(const ImplementationPayload &lhs,
                                        const ImplementationPayload &rhs) {
  const auto lhsRole = static_cast<std::uint32_t>(lhs.role);
  const auto rhsRole = static_cast<std::uint32_t>(rhs.role);
  if (lhsRole != rhsRole)
    return lhsRole < rhsRole;
  if (lhs.canonicalLogicalName != rhs.canonicalLogicalName)
    return byteStringLess(lhs.canonicalLogicalName, rhs.canonicalLogicalName);
  return lhs.blobDigest.bytes() < rhs.blobDigest.bytes();
}

} // namespace loom::hardware
