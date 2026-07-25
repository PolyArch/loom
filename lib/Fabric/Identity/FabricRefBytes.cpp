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
                              llvm::Twine("unknown ") + what + " discriminant " +
                                  llvm::Twine(*raw));
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
    FabricByteWriter &writer, const FabricTransportEndpointOwnerRef &owner) {
  writer.tag(static_cast<std::uint32_t>(owner.kind));
  switch (owner.kind) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type)                        \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return encodeFabricRef(writer, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricMemoryEndpointOwnerRef &owner) {
  writer.tag(static_cast<std::uint32_t>(owner.kind));
  switch (owner.kind) {
#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type)                           \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return encodeFabricRef(writer, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricInventoryOwnerRef &owner) {
  writer.tag(static_cast<std::uint32_t>(owner.kind));
  switch (owner.kind) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type)                        \
  case FabricInventoryOwnerKind::Name:                                         \
    return encodeFabricRef(writer, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(FabricByteWriter &writer,
                                   const FabricMemoryServiceRef &service) {
  writer.tag(static_cast<std::uint32_t>(service.kind));
  switch (service.kind) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Member, Type)                \
  case FabricMemoryServiceKind::Name:                                          \
    return encodeFabricRef(writer, service.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::encodeFabricRef(
    FabricByteWriter &writer, const FabricPhysicalTraversalRef &traversal) {
  writer.tag(static_cast<std::uint32_t>(traversal.kind));
  FabricEncodeVisitor visitor{writer};
  switch (traversal.kind) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Member, Type)                     \
  case FabricPhysicalTraversalKind::Name:                                      \
    return Type::visitFields(traversal.payload.Member, visitor);
#include "Fabric/Identity/FabricRefs.def"
  }
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricTransportEndpointOwnerRef &owner) {
  llvm::Expected<std::uint32_t> kind = readFabricClosedTag(
      reader, fabricClosedBound(owner.kind), fabricClosedName(owner.kind));
  if (!kind)
    return kind.takeError();
  owner.kind = static_cast<FabricTransportEndpointOwnerKind>(*kind);
  switch (owner.kind) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type)                        \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    owner.payload.Member = Type();                                             \
    return decodeFabricRefInto(reader, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricMemoryEndpointOwnerRef &owner) {
  llvm::Expected<std::uint32_t> kind = readFabricClosedTag(
      reader, fabricClosedBound(owner.kind), fabricClosedName(owner.kind));
  if (!kind)
    return kind.takeError();
  owner.kind = static_cast<FabricMemoryEndpointOwnerKind>(*kind);
  switch (owner.kind) {
#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type)                           \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    owner.payload.Member = Type();                                             \
    return decodeFabricRefInto(reader, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                              FabricInventoryOwnerRef &owner) {
  llvm::Expected<std::uint32_t> kind = readFabricClosedTag(
      reader, fabricClosedBound(owner.kind), fabricClosedName(owner.kind));
  if (!kind)
    return kind.takeError();
  owner.kind = static_cast<FabricInventoryOwnerKind>(*kind);
  switch (owner.kind) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type)                        \
  case FabricInventoryOwnerKind::Name:                                         \
    owner.payload.Member = Type();                                             \
    return decodeFabricRefInto(reader, owner.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                              FabricMemoryServiceRef &service) {
  llvm::Expected<std::uint32_t> kind =
      readFabricClosedTag(reader, fabricClosedBound(service.kind),
                          fabricClosedName(service.kind));
  if (!kind)
    return kind.takeError();
  service.kind = static_cast<FabricMemoryServiceKind>(*kind);
  switch (service.kind) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Member, Type)                \
  case FabricMemoryServiceKind::Name:                                          \
    service.payload.Member = Type();                                           \
    return decodeFabricRefInto(reader, service.payload.Member);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::decodeFabricRefInto(FabricByteReader &reader,
                                  FabricPhysicalTraversalRef &traversal) {
  llvm::Expected<std::uint32_t> kind =
      readFabricClosedTag(reader, fabricClosedBound(traversal.kind),
                          fabricClosedName(traversal.kind));
  if (!kind)
    return kind.takeError();
  traversal.kind = static_cast<FabricPhysicalTraversalKind>(*kind);
  FabricDecodeVisitor visitor{reader};
  switch (traversal.kind) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Member, Type)                     \
  case FabricPhysicalTraversalKind::Name:                                      \
    traversal.payload.Member = Type();                                         \
    Type::visitFields(traversal.payload.Member, visitor);                      \
    break;
#include "Fabric/Identity/FabricRefs.def"
  }
  return std::move(visitor.error);
}
