#include "Hardware/Implementation/RepresentationFormat.h"

#include "RepresentationIndexInternal.h"

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

class BinaryReader final {
public:
  explicit BinaryReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> readU32() {
    if (bytes_.size() - offset_ < sizeof(std::uint32_t))
      return invalid("truncated representation format reference");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64() {
    if (bytes_.size() - offset_ < sizeof(std::uint64_t))
      return invalid("truncated representation format reference");
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::StringRef> readString(std::uint64_t size) {
    if (size > std::numeric_limits<std::size_t>::max() ||
        bytes_.size() - offset_ < static_cast<std::size_t>(size))
      return invalid("truncated representation format reference");
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

llvm::Expected<std::uint32_t> requireU32(const llvm::json::Object &object,
                                         llvm::StringRef key) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return invalid("representation format reference field '" + key +
                   "' is required");
  std::optional<std::uint64_t> integer = value->getAsUINT64();
  if (!integer || *integer > std::numeric_limits<std::uint32_t>::max())
    return invalid("representation format reference field '" + key +
                   "' must be an unsigned uint32 integer");
  return static_cast<std::uint32_t>(*integer);
}

RepresentationFormatDescriptorRef knownRef(RepresentationFormatKind kind) {
  auto reference = RepresentationFormatDescriptorRef::get(kind);
  if (!reference)
    llvm_unreachable("static representation format kind is valid");
  return *reference;
}

const RepresentationFormatDescriptorRef systemVerilogRtlRef =
    knownRef(RepresentationFormatKind::SystemVerilogRtl);
const RepresentationFormatDescriptorRef structuralVerilogGateNetlistRef =
    knownRef(RepresentationFormatKind::StructuralVerilogGateNetlist);

constexpr std::array<RepresentationPayloadContract, 3> rtlPayloadContracts{{
    {PayloadRole::RtlSource, "text/x-systemverilog; charset=utf-8", 1,
     std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
    {PayloadRole::GenerationConstraint, "application/x-sdc; charset=utf-8", 0,
     std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
    {PayloadRole::BlackBoxContract, "application/vnd.loom.black-box-contract",
     0, std::nullopt, RepresentationTextPolicy::Opaque},
}};

constexpr std::array<RepresentationPayloadContract, 3>
    gateNetlistPayloadContracts{{
        {PayloadRole::Netlist, "text/x-verilog; charset=utf-8", 1, std::nullopt,
         RepresentationTextPolicy::Utf8LfNoNul},
        {PayloadRole::GenerationConstraint, "application/x-sdc; charset=utf-8",
         0, std::nullopt, RepresentationTextPolicy::Utf8LfNoNul},
        {PayloadRole::BlackBoxContract,
         "application/vnd.loom.black-box-contract", 0, std::nullopt,
         RepresentationTextPolicy::Opaque},
    }};

constexpr std::array<RepresentationObjectKind, 6> rtlObjectKinds{
    RepresentationObjectKind::Module,   RepresentationObjectKind::Instance,
    RepresentationObjectKind::Port,     RepresentationObjectKind::Net,
    RepresentationObjectKind::Register, RepresentationObjectKind::Memory};

constexpr std::array<RepresentationObjectKind, 5> gateObjectKinds{
    RepresentationObjectKind::Module, RepresentationObjectKind::Cell,
    RepresentationObjectKind::Port, RepresentationObjectKind::Pin,
    RepresentationObjectKind::Net};

const std::array<detail::StaticRepresentationFormatEntry, 2>
    representationFormats{{
        {{systemVerilogRtlRef, RepresentationObjectKind::Module,
          rtlPayloadContracts, PayloadRole::RtlSource,
          RepresentationLanguageProfile::Ieee1800_2017, rtlObjectKinds},
         detail::BuiltinRepresentationIndexer::SystemVerilogRtl},
        {{structuralVerilogGateNetlistRef, RepresentationObjectKind::Module,
          gateNetlistPayloadContracts, PayloadRole::Netlist,
          RepresentationLanguageProfile::Ieee1364_2005, gateObjectKinds},
         detail::BuiltinRepresentationIndexer::StructuralVerilogGateNetlist},
    }};

} // namespace

llvm::Expected<RepresentationFormatDescriptorRef>
RepresentationFormatDescriptorRef::get(RepresentationFormatKind kind) {
  switch (kind) {
  case RepresentationFormatKind::SystemVerilogRtl:
  case RepresentationFormatKind::StructuralVerilogGateNetlist:
    return RepresentationFormatDescriptorRef(kind);
  }
  return invalid("representation format kind is unsupported");
}

const RepresentationFormatDescriptor &
getRepresentationFormatDescriptor(RepresentationFormatDescriptorRef reference) {
  return detail::getStaticRepresentationFormatEntry(reference).descriptor;
}

namespace detail {

const StaticRepresentationFormatEntry &getStaticRepresentationFormatEntry(
    RepresentationFormatDescriptorRef reference) {
  return representationFormats[static_cast<std::size_t>(reference.kind())];
}

} // namespace detail

std::vector<std::uint8_t> encodeRepresentationFormatDescriptorRef(
    RepresentationFormatDescriptorRef reference) {
  std::vector<std::uint8_t> bytes;
  const llvm::StringRef identity =
      hardwareRepresentationFormatRegistry.identity;
  bytes.reserve(sizeof(std::uint64_t) + identity.size() +
                3 * sizeof(std::uint32_t));
  appendU64Be(bytes, identity.size());
  bytes.insert(bytes.end(), identity.bytes_begin(), identity.bytes_end());
  appendU32Be(bytes, hardwareRepresentationFormatRegistry.version.major);
  appendU32Be(bytes, hardwareRepresentationFormatRegistry.version.minor);
  appendU32Be(bytes, static_cast<std::uint32_t>(reference.kind()));
  return bytes;
}

llvm::Expected<RepresentationFormatDescriptorRef>
decodeRepresentationFormatDescriptorRef(llvm::ArrayRef<std::uint8_t> bytes) {
  BinaryReader reader(bytes);
  auto identitySize = reader.readU64();
  if (!identitySize)
    return identitySize.takeError();
  auto identity = reader.readString(*identitySize);
  if (!identity)
    return identity.takeError();
  auto major = reader.readU32();
  if (!major)
    return major.takeError();
  auto minor = reader.readU32();
  if (!minor)
    return minor.takeError();
  auto kind = reader.readU32();
  if (!kind)
    return kind.takeError();
  if (!reader.empty())
    return invalid("representation format reference has trailing bytes");
  if (*identity != hardwareRepresentationFormatRegistry.identity)
    return invalid("representation format registry is unsupported");
  if (SchemaVersion{*major, *minor} !=
      hardwareRepresentationFormatRegistry.version)
    return invalid("representation format registry version is unsupported");
  return RepresentationFormatDescriptorRef::get(
      static_cast<RepresentationFormatKind>(*kind));
}

std::string serializeRepresentationFormatDescriptorRefJson(
    RepresentationFormatDescriptorRef reference) {
  llvm::SmallString<128> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("registry", hardwareRepresentationFormatRegistry.identity);
    json.attribute("major",
                   static_cast<std::uint64_t>(
                       hardwareRepresentationFormatRegistry.version.major));
    json.attribute("minor",
                   static_cast<std::uint64_t>(
                       hardwareRepresentationFormatRegistry.version.minor));
    json.attribute("kind", static_cast<std::uint64_t>(reference.kind()));
  });
  return storage.str().str();
}

llvm::Expected<RepresentationFormatDescriptorRef>
parseRepresentationFormatDescriptorRefJson(llvm::StringRef bytes) {
  auto parsed = llvm::json::parse(bytes);
  if (!parsed)
    return invalid("invalid representation format reference JSON: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("representation format reference JSON must be an object");

  constexpr std::array<llvm::StringLiteral, 4> fields{"registry", "major",
                                                      "minor", "kind"};
  for (const auto &field : *object) {
    const llvm::StringRef key = field.getFirst();
    bool known = false;
    for (llvm::StringRef expected : fields)
      known |= key == expected;
    if (!known)
      return invalid("representation format reference has unknown field '" +
                     key + "'");
  }
  if (object->size() != fields.size())
    return invalid("representation format reference requires exactly registry, "
                   "major, minor, and kind fields");

  std::optional<llvm::StringRef> registry = object->getString("registry");
  if (!registry)
    return invalid("representation format reference field 'registry' must be "
                   "a string");
  auto major = requireU32(*object, "major");
  if (!major)
    return major.takeError();
  auto minor = requireU32(*object, "minor");
  if (!minor)
    return minor.takeError();
  auto kind = requireU32(*object, "kind");
  if (!kind)
    return kind.takeError();

  if (*registry != hardwareRepresentationFormatRegistry.identity)
    return invalid("representation format registry is unsupported");
  if (SchemaVersion{*major, *minor} !=
      hardwareRepresentationFormatRegistry.version)
    return invalid("representation format registry version is unsupported");
  auto reference = RepresentationFormatDescriptorRef::get(
      static_cast<RepresentationFormatKind>(*kind));
  if (!reference)
    return reference.takeError();
  if (serializeRepresentationFormatDescriptorRefJson(*reference) != bytes)
    return invalid("representation format reference JSON is not canonical");
  return reference;
}

} // namespace loom::hardware
