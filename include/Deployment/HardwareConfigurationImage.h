#ifndef LOOM_DEPLOYMENT_HARDWARECONFIGURATIONIMAGE_H
#define LOOM_DEPLOYMENT_HARDWARECONFIGURATIONIMAGE_H

#include "Common/Artifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::deployment {

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
