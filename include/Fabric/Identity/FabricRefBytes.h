#ifndef LOOM_FABRIC_IDENTITY_FABRICREFBYTES_H
#define LOOM_FABRIC_IDENTITY_FABRICREFBYTES_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom {
namespace fabric {

/// Canonical bytes are unsigned 32-bit big-endian closed variant tags and
/// unsigned 64-bit big-endian identifiers and ordinals, emitted recursively in
/// declaration order. There are no optional fields, no padding, no native
/// layout, no duplicated owner facts, and no native PnR indices.
class FabricByteWriter {
public:
  void tag(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void field(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class FabricByteReader {
public:
  explicit FabricByteReader(llvm::ArrayRef<std::uint8_t> bytes)
      : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> tag() {
    if (bytes_.size() < 4)
      return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                                "truncated canonical variant tag");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }
  llvm::Expected<std::uint64_t> field() {
    if (bytes_.size() < 8)
      return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                                "truncated canonical field");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }
  bool empty() const { return bytes_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

//===---------------------------------------------------------------------===//
// Encoding
//===---------------------------------------------------------------------===//

template <FabricEntityKind Kind>
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricTypedEntityRef<Kind> &ref);
template <typename Ref>
void encodeFabricRef(FabricByteWriter &writer, const Ref &ref);

void encodeFabricRef(FabricByteWriter &writer,
                     const FabricTransportEndpointOwnerRef &owner);
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricMemoryEndpointOwnerRef &owner);
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricInventoryOwnerRef &owner);
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricMemoryServiceRef &service);
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricPhysicalTraversalRef &traversal);

/// A projection and a refinement encode exactly their underlying reference.
template <FabricInventoryKind Inventory>
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricOwnerProjection<Inventory> &owner) {
  encodeFabricRef(writer, owner.catalog());
}
template <FabricRefinementKind Refinement, typename Underlying>
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricRefinedRef<Refinement, Underlying> &ref) {
  encodeFabricRef(writer, ref.underlying());
}

/// Walks one family's single field declaration in order.
struct FabricEncodeVisitor {
  FabricByteWriter &writer;

  template <typename Enum> void tag(const Enum &value) {
    writer.tag(static_cast<std::uint32_t>(value));
  }
  void ordinal(const FabricOrdinal &value) { writer.field(value); }
  template <typename Ref> void ref(const Ref &value) {
    encodeFabricRef(writer, value);
  }
};

template <FabricEntityKind Kind>
void encodeFabricRef(FabricByteWriter &writer,
                     const FabricTypedEntityRef<Kind> &ref) {
  writer.tag(static_cast<std::uint32_t>(Kind));
  writer.field(ref.id());
}

template <typename Ref>
void encodeFabricRef(FabricByteWriter &writer, const Ref &ref) {
  FabricEncodeVisitor visitor{writer};
  Ref::visitFields(ref, visitor);
}

template <typename Ref>
std::vector<std::uint8_t> canonicalFabricBytes(const Ref &ref) {
  FabricByteWriter writer;
  encodeFabricRef(writer, ref);
  return writer.take();
}

//===---------------------------------------------------------------------===//
// Decoding
//===---------------------------------------------------------------------===//

template <FabricEntityKind Kind>
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricTypedEntityRef<Kind> &ref);
template <typename Ref>
llvm::Error decodeFabricRefInto(FabricByteReader &reader, Ref &ref);

llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricTransportEndpointOwnerRef &owner);
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricMemoryEndpointOwnerRef &owner);
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricInventoryOwnerRef &owner);
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricMemoryServiceRef &service);
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricPhysicalTraversalRef &traversal);

/// Reads a closed tag and rejects any discriminant outside the declared set.
llvm::Expected<std::uint32_t> readFabricClosedTag(FabricByteReader &reader,
                                                  std::uint32_t bound,
                                                  llvm::StringRef what);

template <FabricInventoryKind Inventory>
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricOwnerProjection<Inventory> &owner) {
  FabricInventoryOwnerRef catalog;
  if (llvm::Error error = decodeFabricRefInto(reader, catalog))
    return error;
  owner = FabricOwnerProjection<Inventory>(std::move(catalog));
  return llvm::Error::success();
}
template <FabricRefinementKind Refinement, typename Underlying>
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricRefinedRef<Refinement, Underlying> &ref) {
  Underlying underlying;
  if (llvm::Error error = decodeFabricRefInto(reader, underlying))
    return error;
  ref = FabricRefinedRef<Refinement, Underlying>(std::move(underlying));
  return llvm::Error::success();
}

struct FabricDecodeVisitor {
  FabricByteReader &reader;
  llvm::Error error = llvm::Error::success();

  template <typename Enum> void tag(Enum &value) {
    if (error)
      return;
    llvm::Expected<std::uint32_t> raw = readFabricClosedTag(
        reader, fabricClosedBound(value), fabricClosedName(value));
    if (!raw)
      error = raw.takeError();
    else
      value = static_cast<Enum>(*raw);
  }
  void ordinal(FabricOrdinal &value) {
    if (error)
      return;
    llvm::Expected<std::uint64_t> raw = reader.field();
    if (!raw)
      error = raw.takeError();
    else
      value = *raw;
  }
  template <typename Ref> void ref(Ref &value) {
    if (error)
      return;
    error = decodeFabricRefInto(reader, value);
  }
};

template <FabricEntityKind Kind>
llvm::Error decodeFabricRefInto(FabricByteReader &reader,
                                FabricTypedEntityRef<Kind> &ref) {
  // An out-of-catalog discriminant is unknown input, not a known entity of
  // the wrong kind.
  llvm::Expected<std::uint32_t> kind = readFabricClosedTag(
      reader, fabricClosedBound(Kind), fabricClosedName(Kind));
  if (!kind)
    return kind.takeError();
  if (*kind != static_cast<std::uint32_t>(Kind))
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              "canonical bytes name entity kind " +
                                  llvm::Twine(*kind) + " where " +
                                  fabricRefKeyword(Kind) + " is required");
  llvm::Expected<std::uint64_t> id = reader.field();
  if (!id)
    return id.takeError();
  ref = FabricTypedEntityRef<Kind>(*id);
  return llvm::Error::success();
}

template <typename Ref>
llvm::Error decodeFabricRefInto(FabricByteReader &reader, Ref &ref) {
  FabricDecodeVisitor visitor{reader};
  Ref::visitFields(ref, visitor);
  return std::move(visitor.error);
}

/// Decodes one complete reference. Trailing bytes are rejected: canonical
/// bytes are not a container format.
template <typename Ref>
llvm::Expected<Ref> decodeFabricRef(llvm::ArrayRef<std::uint8_t> bytes) {
  FabricByteReader reader(bytes);
  Ref ref;
  if (llvm::Error error = decodeFabricRefInto(reader, ref))
    return std::move(error);
  if (!reader.empty())
    return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                              "trailing canonical bytes");
  return ref;
}

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_IDENTITY_FABRICREFBYTES_H
