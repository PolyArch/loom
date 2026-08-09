#include "Hardware/Implementation/RepresentationLocator.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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
                                 "representation_locator_invalid: " + message);
}

llvm::Expected<llvm::StringRef>
objectKindSpelling(RepresentationObjectKind kind) {
  switch (kind) {
  case RepresentationObjectKind::Module:
    return llvm::StringRef("Module");
  case RepresentationObjectKind::Instance:
    return llvm::StringRef("Instance");
  case RepresentationObjectKind::Port:
    return llvm::StringRef("Port");
  case RepresentationObjectKind::Net:
    return llvm::StringRef("Net");
  case RepresentationObjectKind::Register:
    return llvm::StringRef("Register");
  case RepresentationObjectKind::Memory:
    return llvm::StringRef("Memory");
  case RepresentationObjectKind::Cell:
    return llvm::StringRef("Cell");
  case RepresentationObjectKind::Pin:
    return llvm::StringRef("Pin");
  case RepresentationObjectKind::PhysicalObject:
    return llvm::StringRef("PhysicalObject");
  case RepresentationObjectKind::DeviceResource:
    return llvm::StringRef("DeviceResource");
  }
  return invalid("representation object kind is unsupported");
}

std::optional<RepresentationObjectKind>
parseObjectKind(llvm::StringRef spelling) {
  if (spelling == "Module")
    return RepresentationObjectKind::Module;
  if (spelling == "Instance")
    return RepresentationObjectKind::Instance;
  if (spelling == "Port")
    return RepresentationObjectKind::Port;
  if (spelling == "Net")
    return RepresentationObjectKind::Net;
  if (spelling == "Register")
    return RepresentationObjectKind::Register;
  if (spelling == "Memory")
    return RepresentationObjectKind::Memory;
  if (spelling == "Cell")
    return RepresentationObjectKind::Cell;
  if (spelling == "Pin")
    return RepresentationObjectKind::Pin;
  if (spelling == "PhysicalObject")
    return RepresentationObjectKind::PhysicalObject;
  if (spelling == "DeviceResource")
    return RepresentationObjectKind::DeviceResource;
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
      return invalid("truncated representation locator");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64() {
    if (bytes_.size() - offset_ < sizeof(std::uint64_t))
      return invalid("truncated representation locator");
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::StringRef> readString(std::uint64_t size) {
    if (size > std::numeric_limits<std::size_t>::max() ||
        bytes_.size() - offset_ < static_cast<std::size_t>(size))
      return invalid("truncated representation locator");
    const auto value = bytes_.slice(offset_, static_cast<std::size_t>(size));
    offset_ += static_cast<std::size_t>(size);
    return llvm::StringRef(reinterpret_cast<const char *>(value.data()),
                           value.size());
  }

  bool empty() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

bool isIdentifierStart(char character) {
  return (character >= 'A' && character <= 'Z') ||
         (character >= 'a' && character <= 'z') || character == '_';
}

bool isIdentifierContinuation(char character) {
  return isIdentifierStart(character) ||
         (character >= '0' && character <= '9') || character == '$';
}

llvm::Expected<std::size_t> validateHdlPath(llvm::StringRef name) {
  if (name.empty())
    return invalid("locator name must be a nonempty HDL identifier path");
  std::size_t segmentCount = 0;
  std::size_t offset = 0;
  while (offset <= name.size()) {
    const std::size_t separator = name.find('.', offset);
    const std::size_t end =
        separator == llvm::StringRef::npos ? name.size() : separator;
    const llvm::StringRef segment = name.slice(offset, end);
    if (segment.empty() || !isIdentifierStart(segment.front()))
      return invalid("locator name is not a canonical HDL identifier path");
    for (char character : segment.drop_front())
      if (!isIdentifierContinuation(character))
        return invalid("locator name is not a canonical HDL identifier path");
    ++segmentCount;
    if (separator == llvm::StringRef::npos)
      break;
    offset = separator + 1;
  }
  return segmentCount;
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
encodeRepresentationLocator(const RepresentationLocator &locator) {
  auto spelling = objectKindSpelling(locator.kind);
  if (!spelling)
    return spelling.takeError();
  std::vector<std::uint8_t> bytes;
  bytes.reserve(sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                locator.canonicalName.size());
  appendU32Be(bytes, static_cast<std::uint32_t>(locator.kind));
  appendU64Be(bytes, locator.canonicalName.size());
  bytes.insert(bytes.end(), locator.canonicalName.begin(),
               locator.canonicalName.end());
  return bytes;
}

llvm::Expected<RepresentationLocator>
decodeRepresentationLocator(llvm::ArrayRef<std::uint8_t> bytes) {
  BinaryReader reader(bytes);
  auto kind = reader.readU32();
  if (!kind)
    return kind.takeError();
  auto nameSize = reader.readU64();
  if (!nameSize)
    return nameSize.takeError();
  auto name = reader.readString(*nameSize);
  if (!name)
    return name.takeError();
  if (!reader.empty())
    return invalid("representation locator has trailing bytes");
  const auto objectKind = static_cast<RepresentationObjectKind>(*kind);
  auto spelling = objectKindSpelling(objectKind);
  if (!spelling)
    return spelling.takeError();
  return RepresentationLocator{objectKind, name->str()};
}

llvm::Expected<std::string>
serializeRepresentationLocatorJson(const RepresentationLocator &locator) {
  auto spelling = objectKindSpelling(locator.kind);
  if (!spelling)
    return spelling.takeError();
  llvm::SmallString<128> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("object_kind", *spelling);
    json.attribute("canonical_name", locator.canonicalName);
  });
  return storage.str().str();
}

llvm::Expected<RepresentationLocator>
parseRepresentationLocatorJsonValue(const llvm::json::Object &object) {
  constexpr std::array<llvm::StringLiteral, 2> fields{"object_kind",
                                                      "canonical_name"};
  for (const auto &field : object) {
    const llvm::StringRef key = field.getFirst();
    if (key != fields[0] && key != fields[1])
      return invalid("representation locator has unknown field '" + key + "'");
  }
  if (object.size() != fields.size())
    return invalid("representation locator requires exactly object_kind and "
                   "canonical_name fields");

  const std::optional<llvm::StringRef> kindText =
      object.getString("object_kind");
  if (!kindText)
    return invalid("representation locator field 'object_kind' must be a "
                   "string");
  const std::optional<RepresentationObjectKind> kind =
      parseObjectKind(*kindText);
  if (!kind)
    return invalid("representation locator object kind is unsupported");
  const std::optional<llvm::StringRef> name =
      object.getString("canonical_name");
  if (!name)
    return invalid("representation locator field 'canonical_name' must be a "
                   "string");
  return RepresentationLocator{*kind, name->str()};
}

llvm::Expected<RepresentationLocator>
parseRepresentationLocatorJson(llvm::StringRef bytes) {
  auto parsed = llvm::json::parse(bytes);
  if (!parsed)
    return invalid("invalid representation locator JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("representation locator JSON must be an object");
  auto locator = parseRepresentationLocatorJsonValue(*object);
  if (!locator)
    return locator.takeError();
  auto canonical = serializeRepresentationLocatorJson(*locator);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != bytes)
    return invalid("representation locator JSON is not canonical");
  return locator;
}

bool representationLocatorCanonicalLess(const RepresentationLocator &lhs,
                                        const RepresentationLocator &rhs) {
  const auto lhsKind = static_cast<std::uint32_t>(lhs.kind);
  const auto rhsKind = static_cast<std::uint32_t>(rhs.kind);
  if (lhsKind != rhsKind)
    return lhsKind < rhsKind;
  if (lhs.canonicalName.size() != rhs.canonicalName.size())
    return lhs.canonicalName.size() < rhs.canonicalName.size();
  return llvm::StringRef(lhs.canonicalName).compare(rhs.canonicalName) < 0;
}

llvm::Error
validateRepresentationLocatorSyntax(RepresentationFormatDescriptorRef format,
                                    const RepresentationLocator &locator) {
  auto spelling = objectKindSpelling(locator.kind);
  if (!spelling)
    return spelling.takeError();
  const RepresentationFormatDescriptor &descriptor =
      getRepresentationFormatDescriptor(format);
  const bool admitted = llvm::any_of(
      descriptor.admittedRoots,
      [&](const RepresentationRootAdmission &admission) {
        return llvm::is_contained(admission.admittedObjectKinds, locator.kind);
      });
  if (!admitted)
    return invalid("locator kind is incompatible with the selected format");
  auto segmentCount = validateHdlPath(locator.canonicalName);
  if (!segmentCount)
    return segmentCount.takeError();
  if (locator.kind == RepresentationObjectKind::Module && *segmentCount != 1)
    return invalid("Module locator name must be one HDL identifier");
  if (locator.kind != RepresentationObjectKind::Module &&
      locator.kind != RepresentationObjectKind::PhysicalObject &&
      locator.kind != RepresentationObjectKind::DeviceResource &&
      *segmentCount < 2)
    return invalid("non-Module locator name must be top-rooted");
  if (locator.kind == RepresentationObjectKind::Pin && *segmentCount < 3)
    return invalid("Pin locator name must append a terminal identifier");
  return llvm::Error::success();
}

} // namespace loom::hardware
