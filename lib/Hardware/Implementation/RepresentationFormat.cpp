#include "Hardware/Implementation/RepresentationFormat.h"

#include "RepresentationIndexInternal.h"

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
const RepresentationFormatDescriptorRef indexedPhysicalRef =
    knownRef(RepresentationFormatKind::IndexedPhysical);
const RepresentationFormatDescriptorRef indexedDefPhysicalRef =
    knownRef(RepresentationFormatKind::IndexedDefPhysical);
const RepresentationFormatDescriptorRef fabricModelRef =
    knownRef(RepresentationFormatKind::FabricModel);

constexpr std::array<RepresentationPayloadContract, 0>
    fabricModelPayloadContracts{};

constexpr std::array<RepresentationObjectKind, 1> fabricModelObjectKinds{
    RepresentationObjectKind::Model};

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

constexpr std::array<RepresentationObjectKind, 9> asicPhysicalObjectKinds{
    RepresentationObjectKind::Module,        RepresentationObjectKind::Instance,
    RepresentationObjectKind::Port,          RepresentationObjectKind::Net,
    RepresentationObjectKind::Register,      RepresentationObjectKind::Memory,
    RepresentationObjectKind::Cell,          RepresentationObjectKind::Pin,
    RepresentationObjectKind::PhysicalObject};

constexpr std::array<RepresentationObjectKind, 9> fpgaPhysicalObjectKinds{
    RepresentationObjectKind::Module,        RepresentationObjectKind::Instance,
    RepresentationObjectKind::Port,          RepresentationObjectKind::Net,
    RepresentationObjectKind::Register,      RepresentationObjectKind::Memory,
    RepresentationObjectKind::Cell,          RepresentationObjectKind::Pin,
    RepresentationObjectKind::DeviceResource};

constexpr RepresentationPayloadContract physicalIndexContract{
    PayloadRole::RepresentationIndex,
    "application/vnd.loom.physical-representation-index+json", 1,
    std::optional<std::uint64_t>(1), RepresentationTextPolicy::Utf8LfNoNul};

constexpr RepresentationPayloadContract physicalDatabaseContract{
    PayloadRole::PhysicalDatabase, "application/octet-stream", 1, std::nullopt,
    RepresentationTextPolicy::Opaque};
constexpr RepresentationPayloadContract physicalNetlistContract{
    PayloadRole::Netlist, "application/octet-stream", 0, std::nullopt,
    RepresentationTextPolicy::Opaque};
constexpr RepresentationPayloadContract generationConstraintContract{
    PayloadRole::GenerationConstraint, "application/octet-stream", 0,
    std::nullopt, RepresentationTextPolicy::Opaque};
constexpr RepresentationPayloadContract physicalBlackBoxContract{
    PayloadRole::BlackBoxContract, "application/vnd.loom.black-box-contract", 0,
    std::nullopt, RepresentationTextPolicy::Opaque};
constexpr RepresentationPayloadContract layoutStreamContract{
    PayloadRole::LayoutStream, "application/octet-stream", 0, std::nullopt,
    RepresentationTextPolicy::Opaque};
constexpr RepresentationPayloadContract parasiticsContract{
    PayloadRole::Parasitics, "application/octet-stream", 1, std::nullopt,
    RepresentationTextPolicy::Opaque};
constexpr RepresentationPayloadContract deviceImageContract{
    PayloadRole::DeviceImage, "application/octet-stream", 1,
    std::optional<std::uint64_t>(1), RepresentationTextPolicy::Opaque};

constexpr RepresentationPayloadContract defNetlistContract{
    PayloadRole::Netlist, "text/x-verilog; charset=utf-8", 1, std::nullopt,
    RepresentationTextPolicy::Utf8LfNoNul};
constexpr RepresentationPayloadContract defDatabaseContract{
    PayloadRole::PhysicalDatabase, "application/vnd.eda.def; charset=utf-8", 1,
    std::optional<std::uint64_t>(1), RepresentationTextPolicy::Utf8LfNoNul};
constexpr RepresentationPayloadContract defConstraintContract{
    PayloadRole::GenerationConstraint, "application/x-sdc; charset=utf-8", 1,
    std::nullopt, RepresentationTextPolicy::Utf8LfNoNul};

constexpr std::array<RepresentationPayloadContract, 5>
    asicPlacedPayloadContracts{
        {physicalDatabaseContract, physicalNetlistContract,
         generationConstraintContract, physicalBlackBoxContract,
         physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 6>
    asicRoutedPayloadContracts{
        {physicalDatabaseContract, physicalNetlistContract,
         layoutStreamContract, generationConstraintContract,
         physicalBlackBoxContract, physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 7>
    asicExtractedPayloadContracts{
        {physicalDatabaseContract, physicalNetlistContract, parasiticsContract,
         layoutStreamContract, generationConstraintContract,
         physicalBlackBoxContract, physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 4>
    fpgaPhysicalPayloadContracts{
        {physicalDatabaseContract, generationConstraintContract,
         physicalBlackBoxContract, physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 2>
    fpgaImagePayloadContracts{{deviceImageContract, physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 5>
    defPlacedPayloadContracts{{defNetlistContract, defDatabaseContract,
                               defConstraintContract, physicalBlackBoxContract,
                               physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 6>
    defRoutedPayloadContracts{{defNetlistContract, defDatabaseContract,
                               defConstraintContract, layoutStreamContract,
                               physicalBlackBoxContract,
                               physicalIndexContract}};
constexpr std::array<RepresentationPayloadContract, 7>
    defExtractedPayloadContracts{
        {defNetlistContract, defDatabaseContract, defConstraintContract,
         parasiticsContract, layoutStreamContract, physicalBlackBoxContract,
         physicalIndexContract}};

constexpr std::array<RepresentationRootAdmission, 1> rtlRootAdmissions{{
    {RepresentationRootVariant::Rtl, std::nullopt,
     RepresentationObjectKind::Module, rtlPayloadContracts, rtlObjectKinds},
}};

constexpr std::array<RepresentationRootAdmission, 1> gateRootAdmissions{{
    {RepresentationRootVariant::GateNetlist, std::nullopt,
     RepresentationObjectKind::Module, gateNetlistPayloadContracts,
     gateObjectKinds},
}};

constexpr std::array<RepresentationRootAdmission, 1> fabricModelAdmissions{{
    {RepresentationRootVariant::FabricModel, std::nullopt,
     RepresentationObjectKind::Model, fabricModelPayloadContracts,
     fabricModelObjectKinds},
}};

constexpr std::array<RepresentationRootAdmission, 6> physicalRootAdmissions{{
    {RepresentationRootVariant::AsicPhysical,
     RepresentationPhysicalStage::Placed,
     RepresentationObjectKind::PhysicalObject, asicPlacedPayloadContracts,
     asicPhysicalObjectKinds},
    {RepresentationRootVariant::AsicPhysical,
     RepresentationPhysicalStage::Routed,
     RepresentationObjectKind::PhysicalObject, asicRoutedPayloadContracts,
     asicPhysicalObjectKinds},
    {RepresentationRootVariant::AsicPhysical,
     RepresentationPhysicalStage::Extracted,
     RepresentationObjectKind::PhysicalObject, asicExtractedPayloadContracts,
     asicPhysicalObjectKinds},
    {RepresentationRootVariant::FpgaPhysical,
     RepresentationPhysicalStage::Placed,
     RepresentationObjectKind::DeviceResource, fpgaPhysicalPayloadContracts,
     fpgaPhysicalObjectKinds},
    {RepresentationRootVariant::FpgaPhysical,
     RepresentationPhysicalStage::Routed,
     RepresentationObjectKind::DeviceResource, fpgaPhysicalPayloadContracts,
     fpgaPhysicalObjectKinds},
    {RepresentationRootVariant::FpgaImage, std::nullopt,
     RepresentationObjectKind::DeviceResource, fpgaImagePayloadContracts,
     fpgaPhysicalObjectKinds},
}};

constexpr std::array<RepresentationRootAdmission, 3> defPhysicalRootAdmissions{{
    {RepresentationRootVariant::AsicPhysical,
     RepresentationPhysicalStage::Placed,
     RepresentationObjectKind::PhysicalObject, defPlacedPayloadContracts,
     asicPhysicalObjectKinds},
    {RepresentationRootVariant::AsicPhysical,
     RepresentationPhysicalStage::Routed,
     RepresentationObjectKind::PhysicalObject, defRoutedPayloadContracts,
     asicPhysicalObjectKinds},
    {RepresentationRootVariant::AsicPhysical,
     RepresentationPhysicalStage::Extracted,
     RepresentationObjectKind::PhysicalObject, defExtractedPayloadContracts,
     asicPhysicalObjectKinds},
}};

const std::array<detail::StaticRepresentationFormatEntry, 5>
    representationFormats{{
        {{systemVerilogRtlRef, PayloadRole::RtlSource,
          RepresentationLanguageProfile::Ieee1800_2017, rtlRootAdmissions},
         detail::BuiltinRepresentationIndexer::SystemVerilogRtl},
        {{structuralVerilogGateNetlistRef, PayloadRole::Netlist,
          RepresentationLanguageProfile::Ieee1364_2005, gateRootAdmissions},
         detail::BuiltinRepresentationIndexer::StructuralVerilogGateNetlist},
        {{indexedPhysicalRef, std::nullopt, std::nullopt,
          physicalRootAdmissions},
         detail::BuiltinRepresentationIndexer::IndexedPhysical},
        {{indexedDefPhysicalRef, std::nullopt,
          RepresentationLanguageProfile::Ieee1364_2005,
          defPhysicalRootAdmissions},
         detail::BuiltinRepresentationIndexer::IndexedPhysical},
        {{fabricModelRef, std::nullopt, std::nullopt, fabricModelAdmissions},
         detail::BuiltinRepresentationIndexer::FabricModel},
    }};

} // namespace

llvm::Expected<RepresentationFormatDescriptorRef>
RepresentationFormatDescriptorRef::get(RepresentationFormatKind kind) {
  switch (kind) {
  case RepresentationFormatKind::SystemVerilogRtl:
  case RepresentationFormatKind::StructuralVerilogGateNetlist:
  case RepresentationFormatKind::IndexedPhysical:
  case RepresentationFormatKind::IndexedDefPhysical:
  case RepresentationFormatKind::FabricModel:
    return RepresentationFormatDescriptorRef(kind);
  }
  return invalid("representation format kind is unsupported");
}

const RepresentationFormatDescriptor &
getRepresentationFormatDescriptor(RepresentationFormatDescriptorRef reference) {
  return detail::getStaticRepresentationFormatEntry(reference).descriptor;
}

bool admitsRepresentationRoot(
    const RepresentationFormatDescriptor &descriptor,
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage) {
  return findRepresentationRootAdmission(descriptor, variant, stage) != nullptr;
}

const RepresentationRootAdmission *findRepresentationRootAdmission(
    const RepresentationFormatDescriptor &descriptor,
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage) {
  const auto admission = llvm::find_if(
      descriptor.admittedRoots,
      [&](const RepresentationRootAdmission &candidate) {
        return candidate.variant == variant && candidate.stage == stage;
      });
  return admission == descriptor.admittedRoots.end() ? nullptr : &*admission;
}

llvm::Error validateRepresentationPayloadCatalog(
    const RepresentationRootAdmission &admission,
    llvm::ArrayRef<ImplementationPayload> canonicalPayloads) {
  if (canonicalPayloads.empty()) {
    if (!admission.payloadContracts.empty())
      return invalid("payload catalog is empty");
  } else {
    auto canonical =
        canonicalizeImplementationPayloadCatalog(canonicalPayloads);
    if (!canonical)
      return invalid("payload catalog is invalid: " +
                     llvm::toString(canonical.takeError()));
    if (!llvm::equal(*canonical, canonicalPayloads))
      return invalid("payload catalog is not in canonical order");
  }

  std::vector<std::uint64_t> counts(admission.payloadContracts.size());
  for (const ImplementationPayload &payload : canonicalPayloads) {
    const auto contract =
        llvm::find_if(admission.payloadContracts,
                      [&](const RepresentationPayloadContract &candidate) {
                        return candidate.role == payload.role;
                      });
    if (contract == admission.payloadContracts.end())
      return invalid("payload role is not admitted by the selected root");
    ++counts[static_cast<std::size_t>(contract -
                                      admission.payloadContracts.begin())];
  }
  for (auto [contract, count] :
       llvm::zip_equal(admission.payloadContracts, counts)) {
    if (count < contract.minimumCount)
      return invalid("payload role cardinality is below its minimum");
    if (contract.maximumCount && count > *contract.maximumCount)
      return invalid("payload role cardinality is above its maximum");
  }
  return llvm::Error::success();
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
parseRepresentationFormatDescriptorRefJsonValue(
    const llvm::json::Object &object) {
  constexpr std::array<llvm::StringLiteral, 4> fields{"registry", "major",
                                                      "minor", "kind"};
  for (const auto &field : object) {
    const llvm::StringRef key = field.getFirst();
    bool known = false;
    for (llvm::StringRef expected : fields)
      known |= key == expected;
    if (!known)
      return invalid("representation format reference has unknown field '" +
                     key + "'");
  }
  if (object.size() != fields.size())
    return invalid("representation format reference requires exactly registry, "
                   "major, minor, and kind fields");

  std::optional<llvm::StringRef> registry = object.getString("registry");
  if (!registry)
    return invalid("representation format reference field 'registry' must be "
                   "a string");
  auto major = requireU32(object, "major");
  if (!major)
    return major.takeError();
  auto minor = requireU32(object, "minor");
  if (!minor)
    return minor.takeError();
  auto kind = requireU32(object, "kind");
  if (!kind)
    return kind.takeError();

  if (*registry != hardwareRepresentationFormatRegistry.identity)
    return invalid("representation format registry is unsupported");
  if (SchemaVersion{*major, *minor} !=
      hardwareRepresentationFormatRegistry.version)
    return invalid("representation format registry version is unsupported");
  return RepresentationFormatDescriptorRef::get(
      static_cast<RepresentationFormatKind>(*kind));
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
  auto reference = parseRepresentationFormatDescriptorRefJsonValue(*object);
  if (!reference)
    return reference.takeError();
  if (serializeRepresentationFormatDescriptorRefJson(*reference) != bytes)
    return invalid("representation format reference JSON is not canonical");
  return reference;
}

} // namespace loom::hardware
