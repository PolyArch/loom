#ifndef LOOM_COMMON_ARTIFACT_H
#define LOOM_COMMON_ARTIFACT_H

#include <cstdint>
#include <initializer_list>
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

class ArtifactIdentity {
public:
  ArtifactIdentity() = default;
  ArtifactIdentity(std::initializer_list<std::uint8_t> bytes) : bytes_(bytes) {}
  explicit ArtifactIdentity(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  bool empty() const { return bytes_.empty(); }
  const std::vector<std::uint8_t> &bytes() const { return bytes_; }

  friend bool operator==(const ArtifactIdentity &lhs,
                         const ArtifactIdentity &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const ArtifactIdentity &lhs,
                         const ArtifactIdentity &rhs) {
    return !(lhs == rhs);
  }

private:
  std::vector<std::uint8_t> bytes_;
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
