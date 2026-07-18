#ifndef LOOM_COMMON_ARTIFACT_H
#define LOOM_COMMON_ARTIFACT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom {

struct SchemaVersion {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;

  friend bool operator==(SchemaVersion lhs, SchemaVersion rhs) {
    return lhs.major == rhs.major && lhs.minor == rhs.minor;
  }
  friend bool operator!=(SchemaVersion lhs, SchemaVersion rhs) {
    return !(lhs == rhs);
  }
};

struct ArtifactSchemaDescriptor {
  llvm::StringLiteral identity;
  SchemaVersion version;
};

class CanonicalSemanticBytes {
public:
  explicit CanonicalSemanticBytes(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  llvm::ArrayRef<std::uint8_t> bytes() const { return bytes_; }

private:
  std::vector<std::uint8_t> bytes_;
};

class ArtifactIdentity {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  ArtifactIdentity(const ArtifactIdentity &) = default;
  ArtifactIdentity(ArtifactIdentity &&) = default;
  ArtifactIdentity &operator=(const ArtifactIdentity &) = default;
  ArtifactIdentity &operator=(ArtifactIdentity &&) = default;

  static llvm::Expected<ArtifactIdentity>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const ArtifactIdentity &lhs,
                         const ArtifactIdentity &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const ArtifactIdentity &lhs,
                         const ArtifactIdentity &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit ArtifactIdentity(Storage bytes) : bytes_(bytes) {}

  friend ArtifactIdentity
  finalizeArtifactIdentity(const ArtifactSchemaDescriptor &schema,
                           const CanonicalSemanticBytes &canonicalBytes);

  Storage bytes_;
};

template <typename EntityId> struct ArtifactReference {
  ArtifactIdentity artifact;
  EntityId entity;

  friend bool operator==(const ArtifactReference &lhs,
                         const ArtifactReference &rhs) {
    return lhs.artifact == rhs.artifact && lhs.entity == rhs.entity;
  }
  friend bool operator!=(const ArtifactReference &lhs,
                         const ArtifactReference &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace loom

#endif // LOOM_COMMON_ARTIFACT_H
