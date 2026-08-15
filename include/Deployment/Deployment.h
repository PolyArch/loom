#ifndef LOOM_DEPLOYMENT_DEPLOYMENT_H
#define LOOM_DEPLOYMENT_DEPLOYMENT_H

#include "Common/Artifact.h"
#include "Deployment/DeploymentReference.h"
#include "Deployment/ExecutableLeaves.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::deployment {

namespace detail {
class DeploymentCodecAccess;
struct DerivedRuntimeImages;
} // namespace detail

inline constexpr ArtifactSchemaDescriptor deploymentSchema{"loom.deployment",
                                                           SchemaVersion{5, 0}};
inline constexpr ArtifactSchemaDescriptor threadDispatchImageSchema{
    "loom.thread_dispatch_image", SchemaVersion{1, 0}};
inline constexpr ArtifactSchemaDescriptor spatialLaunchImageSchema{
    "loom.spatial_launch_image", SchemaVersion{1, 0}};
inline constexpr ArtifactSchemaDescriptor admissionImageSchema{
    "loom.admission_image", SchemaVersion{1, 0}};

struct DeploymentHardwareBinding final {
  ArtifactRootReference hardwareImplementation;
  ArtifactRootReference runtimePlatformBinding;

  friend bool operator==(const DeploymentHardwareBinding &lhs,
                         const DeploymentHardwareBinding &rhs) {
    return lhs.hardwareImplementation == rhs.hardwareImplementation &&
           lhs.runtimePlatformBinding == rhs.runtimePlatformBinding;
  }
};

/// Canonical Deployment-local JSON. The descriptor is fixed by the containing
/// field and the child has no independent digest or ArtifactIdentity.
class InlineRuntimeImage final {
public:
  const ArtifactSchemaDescriptor &schema() const { return schema_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }

private:
  InlineRuntimeImage(ArtifactSchemaDescriptor schema,
                     CanonicalSemanticBytes canonicalBytes)
      : schema_(schema), canonicalBytes_(std::move(canonicalBytes)) {}

  ArtifactSchemaDescriptor schema_;
  CanonicalSemanticBytes canonicalBytes_;

  friend class detail::DeploymentCodecAccess;
  friend struct detail::DerivedRuntimeImages;
};

struct DeploymentDraft final {
  ArtifactRootReference systemMapping;
  HostProgramLeaf hostProgram;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  std::vector<DeploymentHardwareBinding> hardwareBindings;
  std::vector<ArtifactRootReference> configurationImages;
  std::vector<StaticMemoryImageLeaf> staticMemoryImages;
  CanonicalSemanticBytes threadDispatchImage;
  std::optional<CanonicalSemanticBytes> spatialLaunchImage;
  CanonicalSemanticBytes admissionImage;
};

/// Inputs whose selection belongs upstream of Deployment. Configuration
/// images and runtime-image children are derived mechanically by the owner.
struct ExactDeploymentInputs final {
  ArtifactRootReference systemMapping;
  HostProgramLeaf hostProgram;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  std::vector<DeploymentHardwareBinding> hardwareBindings;
  std::vector<StaticMemoryImageLeaf> staticMemoryImages;
};

class Deployment final {
public:
  const ArtifactRootReference &systemMapping() const { return systemMapping_; }
  const HostProgramLeaf &hostProgram() const { return hostProgram_; }
  llvm::ArrayRef<ArtifactRootReference> instructionCoreBinaries() const {
    return instructionCoreBinaries_;
  }
  llvm::ArrayRef<DeploymentHardwareBinding> hardwareBindings() const {
    return hardwareBindings_;
  }
  llvm::ArrayRef<ArtifactRootReference> configurationImages() const {
    return configurationImages_;
  }
  llvm::ArrayRef<StaticMemoryImageLeaf> staticMemoryImages() const {
    return staticMemoryImages_;
  }
  const InlineRuntimeImage &threadDispatchImage() const {
    return threadDispatchImage_;
  }
  const std::optional<InlineRuntimeImage> &spatialLaunchImage() const {
    return spatialLaunchImage_;
  }
  const InlineRuntimeImage &admissionImage() const { return admissionImage_; }

private:
  Deployment(ArtifactRootReference systemMapping, HostProgramLeaf hostProgram,
             std::vector<ArtifactRootReference> instructionCoreBinaries,
             std::vector<DeploymentHardwareBinding> hardwareBindings,
             std::vector<ArtifactRootReference> configurationImages,
             std::vector<StaticMemoryImageLeaf> staticMemoryImages,
             InlineRuntimeImage threadDispatchImage,
             std::optional<InlineRuntimeImage> spatialLaunchImage,
             InlineRuntimeImage admissionImage)
      : systemMapping_(std::move(systemMapping)),
        hostProgram_(std::move(hostProgram)),
        instructionCoreBinaries_(std::move(instructionCoreBinaries)),
        hardwareBindings_(std::move(hardwareBindings)),
        configurationImages_(std::move(configurationImages)),
        staticMemoryImages_(std::move(staticMemoryImages)),
        threadDispatchImage_(std::move(threadDispatchImage)),
        spatialLaunchImage_(std::move(spatialLaunchImage)),
        admissionImage_(std::move(admissionImage)) {}

  ArtifactRootReference systemMapping_;
  HostProgramLeaf hostProgram_;
  std::vector<ArtifactRootReference> instructionCoreBinaries_;
  std::vector<DeploymentHardwareBinding> hardwareBindings_;
  std::vector<ArtifactRootReference> configurationImages_;
  std::vector<StaticMemoryImageLeaf> staticMemoryImages_;
  InlineRuntimeImage threadDispatchImage_;
  std::optional<InlineRuntimeImage> spatialLaunchImage_;
  InlineRuntimeImage admissionImage_;

  friend class detail::DeploymentCodecAccess;
};

class FinalizedDeployment final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const Deployment &deployment() const { return deployment_; }

private:
  FinalizedDeployment(ArtifactRootReference reference,
                      CanonicalSemanticBytes canonicalBytes,
                      Deployment deployment)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        deployment_(std::move(deployment)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  Deployment deployment_;

  friend class detail::DeploymentCodecAccess;
};

llvm::Expected<FinalizedDeployment>
finalizeDeployment(DeploymentDraft draft, const ArtifactStore &artifacts,
                   const BlobStore &blobs);

llvm::Expected<FinalizedDeployment>
importDeployment(const ArtifactRootReference &reference,
                 const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<FinalizedDeployment>
buildDeployment(ExactDeploymentInputs inputs, const ArtifactStore &artifacts,
                const BlobStore &blobs);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENT_H
