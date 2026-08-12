#include "Common/ArtifactText.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace loom {
namespace {

llvm::Error artifactTextError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<std::uint32_t>
parseSchemaVersionComponent(llvm::StringRef spelling) {
  if (spelling.empty() || (spelling.size() > 1 && spelling.front() == '0'))
    return artifactTextError(
        "schema version must use canonical X.Y uint32 spelling");

  std::uint32_t value = 0;
  for (char character : spelling) {
    if (character < '0' || character > '9')
      return artifactTextError(
          "schema version must use canonical X.Y uint32 spelling");
    const std::uint32_t digit = static_cast<std::uint32_t>(character - '0');
    if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10)
      return artifactTextError(
          "schema version must use canonical X.Y uint32 spelling");
    value = value * 10 + digit;
  }
  return value;
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return artifactTextError("artifact reference is missing field '" + key +
                             "'");
  auto string = value->getAsString();
  if (!string)
    return artifactTextError("artifact reference field '" + key +
                             "' must be a string");
  return *string;
}

} // namespace

std::string formatSchemaVersion(SchemaVersion version) {
  return std::to_string(version.major) + "." + std::to_string(version.minor);
}

llvm::Expected<SchemaVersion> parseSchemaVersion(llvm::StringRef spelling) {
  const std::size_t separator = spelling.find('.');
  if (separator == llvm::StringRef::npos || separator == 0 ||
      separator + 1 == spelling.size() ||
      spelling.find('.', separator + 1) != llvm::StringRef::npos)
    return artifactTextError(
        "schema version must use canonical X.Y uint32 spelling");

  auto major = parseSchemaVersionComponent(spelling.take_front(separator));
  if (!major)
    return major.takeError();
  auto minor = parseSchemaVersionComponent(spelling.drop_front(separator + 1));
  if (!minor)
    return minor.takeError();

  return SchemaVersion{*major, *minor};
}

std::string formatArtifactIdentityHex(const ArtifactIdentity &identity) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(ArtifactIdentity::byteSize * 2);
  for (std::uint8_t byte : identity.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ArtifactIdentity>
parseArtifactIdentityHex(llvm::StringRef spelling) {
  if (spelling.size() != ArtifactIdentity::byteSize * 2)
    return artifactTextError("artifact identity must use exactly 64 lowercase "
                             "hexadecimal characters");

  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes;
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return artifactTextError(
          "artifact identity must use lowercase hexadecimal");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return ArtifactIdentity::fromBytes(bytes);
}

void writeArtifactRootReferenceJsonFields(
    llvm::json::OStream &json, const ArtifactRootReference &reference) {
  json.attribute("schema", reference.schemaIdentity);
  json.attribute("schema_version",
                 formatSchemaVersion(reference.schemaVersion));
  json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
}

llvm::Expected<ArtifactRootReference>
parseArtifactRootReferenceJsonFields(const llvm::json::Object &object) {
  auto schema = requireString(object, "schema");
  if (!schema)
    return schema.takeError();
  auto version = requireString(object, "schema_version");
  if (!version)
    return version.takeError();
  auto parsedVersion = parseSchemaVersion(*version);
  if (!parsedVersion)
    return parsedVersion.takeError();
  auto artifact = requireString(object, "artifact");
  if (!artifact)
    return artifact.takeError();
  auto parsedArtifact = parseArtifactIdentityHex(*artifact);
  if (!parsedArtifact)
    return parsedArtifact.takeError();
  return ArtifactRootReference{schema->str(), *parsedVersion,
                               std::move(*parsedArtifact)};
}

void writeArtifactRootReferenceJson(llvm::json::OStream &json,
                                    const ArtifactRootReference &reference) {
  json.object([&] { writeArtifactRootReferenceJsonFields(json, reference); });
}

llvm::Expected<ArtifactRootReference>
parseArtifactRootReferenceJson(const llvm::json::Object &object) {
  for (const auto &field : object)
    if (field.first != "schema" && field.first != "schema_version" &&
        field.first != "artifact")
      return artifactTextError(llvm::Twine("artifact reference has unknown "
                                           "field '") +
                               field.first.str() + "'");
  return parseArtifactRootReferenceJsonFields(object);
}

std::string
formatArtifactRootReferenceJson(const ArtifactRootReference &reference) {
  std::string text;
  llvm::raw_string_ostream output(text);
  llvm::json::OStream json(output);
  writeArtifactRootReferenceJson(json, reference);
  output.flush();
  return text;
}

llvm::Error
writeArtifactRootReferenceJsonFile(llvm::StringRef path,
                                   const ArtifactRootReference &reference) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::errorCodeToError(error);
  output << formatArtifactRootReferenceJson(reference) << '\n';
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference>
loadArtifactRootReferenceJsonFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::errorCodeToError(buffer.getError());
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return artifactTextError("artifact reference file must contain one JSON "
                             "object");
  return parseArtifactRootReferenceJson(*object);
}

std::string
formatArtifactLocalPayloadHex(llvm::ArrayRef<std::uint8_t> payload) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(payload.size() * 2);
  for (std::uint8_t byte : payload) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
parseArtifactLocalPayloadHex(llvm::StringRef spelling) {
  auto parseNibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };

  if ((spelling.size() % 2) != 0)
    return artifactTextError("an artifact local-reference payload must use "
                             "paired lowercase hexadecimal characters");
  std::vector<std::uint8_t> payload;
  payload.reserve(spelling.size() / 2);
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = parseNibble(spelling[index]);
    const int low = parseNibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return artifactTextError("an artifact local-reference payload must use "
                               "paired lowercase hexadecimal characters");
    payload.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return payload;
}

} // namespace loom
