#include "Fabric/Identity/FabricRefBytes.h"

using namespace loom;
using namespace loom::fabric;

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
