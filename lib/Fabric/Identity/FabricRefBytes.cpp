#include "Fabric/Identity/FabricRefBytes.h"

using namespace loom;
using namespace loom::fabric;

namespace {

template <typename Payload, typename Union>
llvm::Error
decodeValidatedFabricUnion(FabricByteReader &reader, Union &value,
                           llvm::Expected<Union> (*create)(const Payload &)) {
  Payload payload;
  if (llvm::Error error = decodeFabricRefInto(reader, payload))
    return error;
  llvm::Expected<Union> created = create(payload);
  if (!created)
    return created.takeError();
  value = std::move(*created);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::uint32_t>
loom::fabric::readFabricClosedTag(FabricByteReader &reader, std::uint32_t bound,
                                  llvm::StringRef what) {
  llvm::Expected<std::uint32_t> raw = reader.tag();
  if (!raw)
    return raw.takeError();
  if (*raw >= bound)
    return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                              llvm::Twine("unknown ") + what +
                                  " discriminant " + llvm::Twine(*raw));
  return *raw;
}

//===---------------------------------------------------------------------===//
// Closed unions
//
// A union encodes its constructor discriminant and then its selected payload
// recursively. The owner facts the payload's own owner already carries are
// never repeated.
//===---------------------------------------------------------------------===//

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer, const FabricTransportEndpointOwnerRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return encodeFabricRef(writer, std::get<Type>(value.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricMemoryEndpointOwnerRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return encodeFabricRef(writer, std::get<Type>(value.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricInventoryOwnerRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  case FabricInventoryOwnerKind::Name:                                         \
    return encodeFabricRef(writer, std::get<Type>(value.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricModulePhysicalOwnerRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_MODULE_PHYSICAL_OWNER(Ordinal, Name, Type, Validator)      \
  case FabricModulePhysicalOwnerKind::Name:                                    \
    return encodeFabricRef(writer, std::get<Type>(value.payload()));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricModuleDomainMemberRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_MODULE_DOMAIN_MEMBER(Ordinal, Name, Type)                  \
  case FabricModuleDomainMemberKind::Name:                                     \
    return encodeFabricRef(writer, std::get<Type>(value.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricModulePhysicalTargetRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_MODULE_PHYSICAL_TARGET(Ordinal, Name, Type, Validator)     \
  case FabricModulePhysicalTargetKind::Name:                                   \
    return encodeFabricRef(writer, std::get<Type>(value.payload()));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer, const SpatialCorePhysicalDomainTargetRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_SPATIAL_CORE_DOMAIN_TARGET(Ordinal, Name, Type, Validator) \
  case SpatialCorePhysicalDomainTargetKind::Name:                              \
    return encodeFabricRef(writer, std::get<Type>(value.payload()));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer, const FabricPhysicalOccurrenceOwnerRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_PHYSICAL_OCCURRENCE_OWNER(Ordinal, Name, Type, Validator)  \
  case FabricPhysicalOccurrenceOwnerKind::Name:                                \
    return encodeFabricRef(writer, std::get<Type>(value.payload()));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer,
    const FabricPhysicalConfigurationFieldRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_PHYSICAL_CONFIGURATION_FIELD(Ordinal, Name, Type,          \
                                                 Validator)                    \
  case FabricPhysicalConfigurationFieldKind::Name:                             \
    return encodeFabricRef(writer, std::get<Type>(value.payload()));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricHardwareDomainMemberRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_HARDWARE_DOMAIN_MEMBER(Ordinal, Name, Type, Validator)     \
  case FabricHardwareDomainMemberKind::Name:                                   \
    return encodeFabricRef(writer, std::get<Type>(value.payload()));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer, const FabricClockResetDirectOwnerRef &value) {
  encodeFabricRef(writer, value.underlying());
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricMemoryServiceRef &value) {
  writer.tag(static_cast<std::uint32_t>(value.kind()));
  switch (value.kind()) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type)                        \
  case FabricMemoryServiceKind::Name:                                          \
    return encodeFabricRef(writer, std::get<Type>(value.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer, const FabricPhysicalTraversalRef &traversal) {
  writer.tag(static_cast<std::uint32_t>(traversal.kind()));
  FabricEncodeVisitor visitor{writer};
  switch (traversal.kind()) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type)                             \
  case FabricPhysicalTraversalKind::Name:                                      \
    return Type::visitFields(std::get<Type>(traversal.payload), visitor);
#include "Fabric/Identity/FabricRefs.def"
  }
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricTransportEndpointOwnerRef &value) {
  const FabricTransportEndpointOwnerKind bound =
      FabricTransportEndpointOwnerKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricTransportEndpointOwnerKind>(*tag)) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return decodeFabricRefInto(reader, value.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown transport endpoint owner ") +
                                llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricMemoryEndpointOwnerRef &value) {
  const FabricMemoryEndpointOwnerKind bound = FabricMemoryEndpointOwnerKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricMemoryEndpointOwnerKind>(*tag)) {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return decodeFabricRefInto(reader, value.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown memory endpoint owner ") +
                                llvm::Twine(*tag));
}

llvm::Error loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                              FabricInventoryOwnerRef &value) {
  const FabricInventoryOwnerKind bound = FabricInventoryOwnerKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricInventoryOwnerKind>(*tag)) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  case FabricInventoryOwnerKind::Name:                                         \
    return decodeFabricRefInto(reader, value.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricModulePhysicalOwnerRef &value) {
  const FabricModulePhysicalOwnerKind bound = FabricModulePhysicalOwnerKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricModulePhysicalOwnerKind>(*tag)) {
#define LOOM_FABRIC_MODULE_PHYSICAL_OWNER(Ordinal, Name, Type, Validator)      \
  case FabricModulePhysicalOwnerKind::Name: {                                  \
    Type payload;                                                              \
    if (llvm::Error error = decodeFabricRefInto(reader, payload))              \
      return error;                                                            \
    llvm::Expected<FabricModulePhysicalOwnerRef> created =                     \
        FabricModulePhysicalOwnerRef::create(payload);                         \
    if (!created)                                                              \
      return created.takeError();                                              \
    value = std::move(*created);                                               \
    return llvm::Error::success();                                             \
  }
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown Module physical owner ") +
                                llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricModuleDomainMemberRef &value) {
  const FabricModuleDomainMemberKind bound = FabricModuleDomainMemberKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricModuleDomainMemberKind>(*tag)) {
#define LOOM_FABRIC_MODULE_DOMAIN_MEMBER(Ordinal, Name, Type)                  \
  case FabricModuleDomainMemberKind::Name:                                     \
    return decodeFabricRefInto(reader, value.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown Module domain member ") +
                                llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricModulePhysicalTargetRef &value) {
  const FabricModulePhysicalTargetKind bound = FabricModulePhysicalTargetKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricModulePhysicalTargetKind>(*tag)) {
#define LOOM_FABRIC_MODULE_PHYSICAL_TARGET(Ordinal, Name, Type, Validator)     \
  case FabricModulePhysicalTargetKind::Name: {                                 \
    Type payload;                                                              \
    if (llvm::Error error = decodeFabricRefInto(reader, payload))              \
      return error;                                                            \
    llvm::Expected<FabricModulePhysicalTargetRef> created =                    \
        FabricModulePhysicalTargetRef::create(payload);                        \
    if (!created)                                                              \
      return created.takeError();                                              \
    value = std::move(*created);                                               \
    return llvm::Error::success();                                             \
  }
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown Module physical target ") +
                                llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  SpatialCorePhysicalDomainTargetRef &value) {
  const SpatialCorePhysicalDomainTargetKind bound =
      SpatialCorePhysicalDomainTargetKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<SpatialCorePhysicalDomainTargetKind>(*tag)) {
#define LOOM_FABRIC_SPATIAL_CORE_DOMAIN_TARGET(Ordinal, Name, Type, Validator) \
  case SpatialCorePhysicalDomainTargetKind::Name:                              \
    return decodeValidatedFabricUnion<Type>(                                   \
        reader, value, &SpatialCorePhysicalDomainTargetRef::create);
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(
      FabricRefErrorKind::MalformedSyntax,
      llvm::Twine("unknown SpatialCore physical domain target ") +
          llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricPhysicalOccurrenceOwnerRef &value) {
  const FabricPhysicalOccurrenceOwnerKind bound =
      FabricPhysicalOccurrenceOwnerKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricPhysicalOccurrenceOwnerKind>(*tag)) {
#define LOOM_FABRIC_PHYSICAL_OCCURRENCE_OWNER(Ordinal, Name, Type, Validator)  \
  case FabricPhysicalOccurrenceOwnerKind::Name:                                \
    return decodeValidatedFabricUnion<Type>(                                   \
        reader, value, &FabricPhysicalOccurrenceOwnerRef::create);
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown physical occurrence owner ") +
                                llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricPhysicalConfigurationFieldRef &value) {
  const FabricPhysicalConfigurationFieldKind bound =
      FabricPhysicalConfigurationFieldKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricPhysicalConfigurationFieldKind>(*tag)) {
#define LOOM_FABRIC_PHYSICAL_CONFIGURATION_FIELD(Ordinal, Name, Type,          \
                                                 Validator)                    \
  case FabricPhysicalConfigurationFieldKind::Name:                             \
    return decodeValidatedFabricUnion<Type>(                                   \
        reader, value, &FabricPhysicalConfigurationFieldRef::create);
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(
      FabricRefErrorKind::MalformedSyntax,
      llvm::Twine("unknown physical configuration field ") + llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricHardwareDomainMemberRef &value) {
  const FabricHardwareDomainMemberKind bound = FabricHardwareDomainMemberKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricHardwareDomainMemberKind>(*tag)) {
#define LOOM_FABRIC_HARDWARE_DOMAIN_MEMBER(Ordinal, Name, Type, Validator)     \
  case FabricHardwareDomainMemberKind::Name:                                   \
    return decodeValidatedFabricUnion<Type>(                                   \
        reader, value, &FabricHardwareDomainMemberRef::create);
#include "Fabric/Identity/FabricRefs.def"
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("unknown hardware domain member ") +
                                llvm::Twine(*tag));
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricClockResetDirectOwnerRef &value) {
  FabricInventoryOwnerRef owner;
  if (llvm::Error error = decodeFabricRefInto(reader, owner))
    return error;
  llvm::Expected<FabricClockResetDirectOwnerRef> created =
      FabricClockResetDirectOwnerRef::create(owner);
  if (!created)
    return created.takeError();
  value = std::move(*created);
  return llvm::Error::success();
}

llvm::Error loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                              FabricMemoryServiceRef &value) {
  const FabricMemoryServiceKind bound = FabricMemoryServiceKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  switch (static_cast<FabricMemoryServiceKind>(*tag)) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type)                        \
  case FabricMemoryServiceKind::Name:                                          \
    return decodeFabricRefInto(reader, value.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricPhysicalTraversalRef &traversal) {
  const FabricPhysicalTraversalKind bound = FabricPhysicalTraversalKind();
  llvm::Expected<std::uint32_t> tag = readFabricClosedTag(
      reader, fabricClosedBound(bound), fabricClosedName(bound));
  if (!tag)
    return tag.takeError();
  FabricDecodeVisitor visitor{reader};
  switch (static_cast<FabricPhysicalTraversalKind>(*tag)) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type)                             \
  case FabricPhysicalTraversalKind::Name:                                      \
    Type::visitFields(traversal.payload.emplace<Type>(), visitor);             \
    break;
#include "Fabric/Identity/FabricRefs.def"
  }
  return std::move(visitor.error);
}
