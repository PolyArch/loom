#ifndef LOOM_LIB_DEPLOYMENT_DEPLOYMENTINTERNAL_H
#define LOOM_LIB_DEPLOYMENT_DEPLOYMENTINTERNAL_H

#include "Deployment/Deployment.h"

#include "llvm/Support/JSON.h"

namespace loom::deployment::detail {

struct DerivedRuntimeImages final {
  std::optional<CanonicalSemanticBytes> threadDispatch;
  std::optional<CanonicalSemanticBytes> spatialLaunch;
  std::optional<CanonicalSemanticBytes> admission;
};

struct ParsedDeployment final {
  DeploymentExecutionRoot executionRoot;
  HostProgramLeaf hostProgram;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  std::vector<DeploymentHardwareBinding> hardwareBindings;
  std::vector<ArtifactRootReference> configurationImages;
  std::vector<StaticMemoryImageLeaf> staticMemoryImages;
  std::optional<llvm::json::Value> threadDispatchImage;
  std::optional<llvm::json::Value> spatialLaunchImage;
  std::optional<llvm::json::Value> admissionImage;
};

class DeploymentCodecAccess final {
public:
  static HostProgramLeaf
  hostProgram(ArtifactRootReference compilerTargetBinding,
              BlobDigest programBlob,
              std::vector<HostProgramEntry> programEntries,
              std::vector<HostExternalInterface> externalInterfaces,
              BlobDigest registrationTableDigest,
              std::vector<std::uint64_t> supportComponentOrdinals);

  static StaticMemoryImageLeaf
  staticMemory(ArtifactRootReference canonicalDataflow,
               dataflow::RootedGraphLaunchRef rootedGraphLaunch,
               dataflow::LogicalMemoryRootRef logicalMemoryRoot,
               ArtifactRootReference layoutBinding, std::uint64_t sizeBytes,
               std::uint64_t alignmentBytes,
               frontend::StaticMemoryPermissions permissions,
               std::vector<StaticMemoryInitializedChunk> initializedChunks,
               std::vector<StaticMemoryZeroFillRange> zeroFillRanges);

  static InlineRuntimeImage runtimeImage(ArtifactSchemaDescriptor schema,
                                         CanonicalSemanticBytes bytes);

  static Deployment
  deployment(DeploymentExecutionRoot executionRoot, HostProgramLeaf hostProgram,
             std::vector<ArtifactRootReference> instructionCoreBinaries,
             std::vector<DeploymentHardwareBinding> hardwareBindings,
             std::vector<ArtifactRootReference> configurationImages,
             std::vector<StaticMemoryImageLeaf> staticMemoryImages,
             std::optional<InlineRuntimeImage> threadDispatchImage,
             std::optional<InlineRuntimeImage> spatialLaunchImage,
             std::optional<InlineRuntimeImage> admissionImage);

  static FinalizedDeployment finalized(ArtifactRootReference reference,
                                       CanonicalSemanticBytes canonicalBytes,
                                       Deployment deployment);
};

llvm::Expected<ParsedDeployment>
parseDeployment(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<CanonicalSemanticBytes>
serializeDeployment(const ParsedDeployment &deployment,
                    const DerivedRuntimeImages &images);

llvm::Expected<DerivedRuntimeImages> deriveRuntimeImages(
    const ArtifactRootReference &systemMapping,
    llvm::ArrayRef<ArtifactRootReference> instructionCoreBinaries,
    llvm::ArrayRef<ArtifactRootReference> configurationImages,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<DerivedRuntimeImages>
validateDeploymentClosure(const ParsedDeployment &deployment,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

Deployment materializeDeployment(ParsedDeployment deployment,
                                 DerivedRuntimeImages images);

} // namespace loom::deployment::detail

#endif // LOOM_LIB_DEPLOYMENT_DEPLOYMENTINTERNAL_H
