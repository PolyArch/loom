#ifndef LOOM_DEPLOYMENT_HARDWARECONFIGURATIONIMAGE_H
#define LOOM_DEPLOYMENT_HARDWARECONFIGURATIONIMAGE_H

#include "Common/Artifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::deployment {

namespace detail {
class ConfigurationImageProjectionSessionState;
}

enum class ConfigurationImageProjectionVerificationDomain : std::uint8_t {
  SourceInvocation,
  IndependentReplay,
};

struct ConfigurationImageProjectionSessionStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t entryCount = 0;
};

/// Bounded immutable projection cache for one ArtifactStore verification
/// domain. Entries are keyed only by complete artifact roots, source kind,
/// occurrence qualification, and the projection algorithm version.
class ConfigurationImageProjectionSession final {
public:
  ConfigurationImageProjectionSession(const ArtifactStore &store,
                                      std::size_t entryLimit);
  ~ConfigurationImageProjectionSession();

  ConfigurationImageProjectionSession(
      const ConfigurationImageProjectionSession &) = delete;
  ConfigurationImageProjectionSession &
  operator=(const ConfigurationImageProjectionSession &) = delete;
  ConfigurationImageProjectionSession(ConfigurationImageProjectionSession &&) =
      delete;
  ConfigurationImageProjectionSession &
  operator=(ConfigurationImageProjectionSession &&) = delete;

  ConfigurationImageProjectionSessionStatistics statistics() const;

private:
  std::unique_ptr<detail::ConfigurationImageProjectionSessionState> state_;
  detail::ConfigurationImageProjectionSessionState *previous_ = nullptr;
};

void emitConfigurationImageProjectionSessionStatistics(
    ConfigurationImageProjectionVerificationDomain domain,
    const ConfigurationImageProjectionSessionStatistics &statistics);

class FinalizedHardwareConfigurationImage;

inline constexpr ArtifactSchemaDescriptor hardwareConfigurationImageSchema{
    "loom.hardware_configuration_image", SchemaVersion{3, 0}};

enum class ConfigurationImageSourceKind : std::uint32_t {
  SpatialMapping = 0,
  SystemMapping = 1,
};

struct ConfigurationImageSourceRef final {
  ConfigurationImageSourceKind kind =
      ConfigurationImageSourceKind::SpatialMapping;
  ArtifactRootReference mapping;

  friend bool operator==(const ConfigurationImageSourceRef &lhs,
                         const ConfigurationImageSourceRef &rhs) {
    return lhs.kind == rhs.kind && lhs.mapping == rhs.mapping;
  }
};

struct HardwareConfigurationImageDraft final {
  ArtifactRootReference configurationAbi;
  hardware::ProgrammingUnitId programmingUnitId = 0;
  ConfigurationImageSourceRef sourceMapping;
};

class HardwareConfigurationImage final {
public:
  const ArtifactRootReference &configurationAbi() const {
    return configurationAbi_;
  }
  hardware::ProgrammingUnitId programmingUnitId() const {
    return programmingUnitId_;
  }
  const ConfigurationImageSourceRef &sourceMapping() const {
    return sourceMapping_;
  }
  std::uint64_t payloadBitCount() const { return payloadBitCount_; }
  llvm::ArrayRef<std::uint8_t> payload() const { return payload_; }

private:
  HardwareConfigurationImage(ArtifactRootReference configurationAbi,
                             hardware::ProgrammingUnitId programmingUnitId,
                             ConfigurationImageSourceRef sourceMapping,
                             std::uint64_t payloadBitCount,
                             std::vector<std::uint8_t> payload)
      : configurationAbi_(std::move(configurationAbi)),
        programmingUnitId_(programmingUnitId),
        sourceMapping_(std::move(sourceMapping)),
        payloadBitCount_(payloadBitCount), payload_(std::move(payload)) {}

  ArtifactRootReference configurationAbi_;
  hardware::ProgrammingUnitId programmingUnitId_ = 0;
  ConfigurationImageSourceRef sourceMapping_;
  std::uint64_t payloadBitCount_ = 0;
  std::vector<std::uint8_t> payload_;

  friend class FinalizedHardwareConfigurationImage;
  friend llvm::Expected<FinalizedHardwareConfigurationImage>
  importHardwareConfigurationImage(const ArtifactRootReference &,
                                   const ArtifactStore &);
};

class FinalizedHardwareConfigurationImage final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const HardwareConfigurationImage &image() const { return image_; }

private:
  FinalizedHardwareConfigurationImage(ArtifactRootReference reference,
                                      CanonicalSemanticBytes canonicalBytes,
                                      HardwareConfigurationImage image)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), image_(std::move(image)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  HardwareConfigurationImage image_;

  friend llvm::Expected<FinalizedHardwareConfigurationImage>
  importHardwareConfigurationImage(const ArtifactRootReference &,
                                   const ArtifactStore &);
};

llvm::Expected<FinalizedHardwareConfigurationImage>
finalizeHardwareConfigurationImage(HardwareConfigurationImageDraft draft,
                                   const ArtifactStore &store);

llvm::Expected<FinalizedHardwareConfigurationImage>
importHardwareConfigurationImage(const ArtifactRootReference &reference,
                                 const ArtifactStore &store);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_HARDWARECONFIGURATIONIMAGE_H
