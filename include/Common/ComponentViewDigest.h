#ifndef LOOM_COMMON_COMPONENTVIEWDIGEST_H
#define LOOM_COMMON_COMPONENTVIEWDIGEST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace loom {

/// Compact integrity and dependency value mechanically derived from one
/// component view's schema descriptor bytes and canonical view bytes. It is a
/// distinct semantic type from ArtifactIdentity, cannot be separately authored,
/// and never replaces the source bytes it is derived from.
class ComponentViewDigest {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  /// Adopts an exact 32-byte digest carried alongside the source bytes it was
  /// derived from. The adopted value is never authority on its own; a reader
  /// still recomputes through validateComponentViewDigest.
  static llvm::Expected<ComponentViewDigest>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const ComponentViewDigest &lhs,
                         const ComponentViewDigest &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const ComponentViewDigest &lhs,
                         const ComponentViewDigest &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit ComponentViewDigest(Storage bytes) : bytes_(bytes) {}

  friend llvm::Expected<ComponentViewDigest>
  computeComponentViewDigest(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                             llvm::ArrayRef<std::uint8_t> canonicalViewBytes);

  Storage bytes_;
};

/// Derives the digest of the exact opaque descriptor and canonical view bytes.
/// Neither input is parsed or interpreted here. A descriptor whose length is
/// not representable in the framed u32 length is rejected instead of framed
/// with a truncated length.
llvm::Expected<ComponentViewDigest>
computeComponentViewDigest(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes);

/// Recomputes the digest from the authoritative source bytes and reports a
/// typed error when the supplied digest is not exactly that value.
llvm::Error
validateComponentViewDigest(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                            llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                            const ComponentViewDigest &suppliedDigest);

} // namespace loom

#endif // LOOM_COMMON_COMPONENTVIEWDIGEST_H
