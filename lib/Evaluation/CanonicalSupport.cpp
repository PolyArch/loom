#include "CanonicalSupport.h"

#include <algorithm>
#include <optional>

namespace loom::evaluation::detail {

llvm::Error evaluationError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
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

void appendI64Be(std::vector<std::uint8_t> &bytes, std::int64_t value) {
  appendU64Be(bytes, static_cast<std::uint64_t>(value));
}

void appendFramedBytes(std::vector<std::uint8_t> &bytes,
                       llvm::ArrayRef<std::uint8_t> payload) {
  appendU64Be(bytes, payload.size());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
}

void appendFramedString(std::vector<std::uint8_t> &bytes,
                        llvm::StringRef text) {
  appendU64Be(bytes, text.size());
  bytes.insert(bytes.end(), text.bytes_begin(), text.bytes_end());
}

void appendArtifactIdentity(std::vector<std::uint8_t> &bytes,
                            const ArtifactIdentity &identity) {
  bytes.insert(bytes.end(), identity.bytes().begin(), identity.bytes().end());
}

void appendSchemaVersion(std::vector<std::uint8_t> &bytes,
                         SchemaVersion version) {
  appendU32Be(bytes, version.major);
  appendU32Be(bytes, version.minor);
}

void appendDecimalValue(std::vector<std::uint8_t> &bytes, DecimalValue value) {
  appendI64Be(bytes, value.coefficient());
  appendI64Be(bytes, value.base10Exponent());
}

void appendExactRatio(std::vector<std::uint8_t> &bytes, ExactRatio value) {
  appendU64Be(bytes, value.numerator());
  appendU64Be(bytes, value.denominator());
}

std::string formatPayloadHex(llvm::ArrayRef<std::uint8_t> payload) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string text;
  text.reserve(payload.size() * 2);
  for (std::uint8_t byte : payload) {
    text.push_back(digits[byte >> 4]);
    text.push_back(digits[byte & 0x0f]);
  }
  return text;
}

llvm::Expected<std::vector<std::uint8_t>>
parsePayloadHex(llvm::StringRef text) {
  const auto nibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return 10 + (character - 'a');
    return -1;
  };

  if ((text.size() % 2) != 0)
    return evaluationError("a local target payload must use paired lowercase "
                           "hexadecimal characters");
  std::vector<std::uint8_t> payload;
  payload.reserve(text.size() / 2);
  for (std::size_t index = 0; index < text.size(); index += 2) {
    const int high = nibble(text[index]);
    const int low = nibble(text[index + 1]);
    if (high < 0 || low < 0)
      return evaluationError("a local target payload must use paired lowercase "
                             "hexadecimal characters");
    payload.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return payload;
}

llvm::Error
rejectUnknownFields(const llvm::json::Object &object, llvm::StringRef context,
                    std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &field : object) {
    llvm::StringRef key = field.getFirst();
    if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
      return evaluationError(context + " has unknown field '" + key + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' must be a string");
  return *value;
}

llvm::Expected<std::int64_t> requireInteger(const llvm::json::Object &object,
                                            llvm::StringRef key,
                                            llvm::StringRef context) {
  std::optional<std::int64_t> value = object.getInteger(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' must be an integer");
  return *value;
}

llvm::Expected<std::uint64_t> requireUnsigned(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' is required");
  std::optional<std::uint64_t> integer = value->getAsUINT64();
  if (!integer)
    return evaluationError(context + " field '" + key +
                           "' must be an unsigned integer");
  return *integer;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef key,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' must be an object");
  return value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef key,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(key);
  if (!value)
    return evaluationError(context + " field '" + key + "' must be an array");
  return value;
}

} // namespace loom::evaluation::detail
