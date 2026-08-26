#ifndef LOOM_TEST_DEPLOYMENT_DEPLOYMENTTESTSUPPORT_H
#define LOOM_TEST_DEPLOYMENT_DEPLOYMENTTESTSUPPORT_H

#include "Deployment/Deployment.h"
#include "Deployment/ExecutableLeaves.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Types.h"

#include <string>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
namespace runtime {
struct RuntimeProviderDescriptor;
}
} // namespace loom

namespace loom::deployment::test {

enum class MappedSystemTransportMode : std::uint8_t {
  FixedLocal,
  MappingSelected,
};

struct MappedSpatialSystemSpec final {
  std::uint32_t accCoreCount = 2;
  bool alternateInstructionMicroarchitectures = false;
  bool attachSystemMemory = false;
  fabric::ResetInitialState resetInitialState =
      fabric::ResetInitialState::Asserted;
  MappedSystemTransportMode transportMode =
      MappedSystemTransportMode::MappingSelected;
};

struct MappedSystemExecutablePrograms final {
  std::vector<std::uint8_t> hostProgramBytes;
  std::vector<HostProgramEntry> hostEntries;
  std::vector<HostExternalInterface> hostInterfaces;
  std::vector<std::uint8_t> instructionProgramBytes;
  std::string instructionEntrySymbol = "__loom_thread_entry_0";
};

class TemporaryTree final {
public:
  explicit TemporaryTree(llvm::StringRef label);
  ~TemporaryTree();

  TemporaryTree(const TemporaryTree &) = delete;
  TemporaryTree &operator=(const TemporaryTree &) = delete;

  std::string path(llvm::StringRef leaf) const;

private:
  std::string root_;
};

[[noreturn]] void fail(llvm::StringRef test, const std::string &message);
void require(llvm::StringRef test, bool condition, llvm::StringRef message);

FinalizedDeployment buildMinimalDeployment(llvm::StringRef test,
                                           ArtifactStore &artifacts,
                                           BlobStore &blobs,
                                           const TemporaryTree &tree);

FinalizedDeployment buildDirectSystemConfigurationDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree);

FinalizedDeployment buildRetargetedMinimalDeployment(llvm::StringRef test,
                                                     ArtifactStore &artifacts,
                                                     BlobStore &blobs,
                                                     const TemporaryTree &tree);

FinalizedDeployment buildRetargetedSharedProgrammingEndpointDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree);

FinalizedDeployment buildSystemArtifactDeployment(llvm::StringRef test,
                                                  ArtifactStore &artifacts,
                                                  BlobStore &blobs,
                                                  const TemporaryTree &tree);

FinalizedDeployment buildTrustedIdentityDeployment(llvm::StringRef test,
                                                   ArtifactStore &artifacts,
                                                   BlobStore &blobs,
                                                   const TemporaryTree &tree);

FinalizedDeployment buildSharedProgrammingEndpointDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree);

FinalizedDeployment buildRuntimeProviderDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree,
    const runtime::RuntimeProviderDescriptor &provider);

FinalizedDeployment buildMappedSpatialDeployment(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const fabric::FinalizedFabricRoot &system,
    const mapping::FinalizedSpatialMapping &spatialMapping,
    llvm::ArrayRef<hardware::FinalizedHardwareImplementation> implementations,
    ArtifactStore &artifacts, BlobStore &blobs, const TemporaryTree &tree);

mapping::FinalizedSystemMapping buildMappedSystemMapping(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const fabric::FinalizedFabricRoot &system,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    ArtifactStore &artifacts,
    llvm::ArrayRef<fabric::AccCoreOccurrenceRef> rootThreadTargets = {});

FinalizedDeployment buildMappedSystemDeployment(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const fabric::FinalizedFabricRoot &system,
    const mapping::FinalizedSystemMapping &systemMapping,
    llvm::ArrayRef<hardware::FinalizedHardwareImplementation> implementations,
    MappedSystemExecutablePrograms programs, ArtifactStore &artifacts,
    BlobStore &blobs, const TemporaryTree &tree);

fabric::FinalizedFabricRoot buildMappedSpatialSystem(
    llvm::StringRef test, const fabric::FinalizedFabricRoot &module,
    llvm::ArrayRef<mlir::Type> messagePayloads, const ArtifactStore &artifacts,
    bool attachSystemMemory);

fabric::FinalizedFabricRoot buildMappedSpatialSystem(
    llvm::StringRef test, const fabric::FinalizedFabricRoot &module,
    llvm::ArrayRef<mlir::Type> messagePayloads, const ArtifactStore &artifacts,
    MappedSpatialSystemSpec spec);

llvm::Expected<FinalizedDeployment>
tryBuildMinimalDeployment(llvm::StringRef test, ArtifactStore &artifacts,
                          BlobStore &blobs, const TemporaryTree &tree,
                          llvm::StringRef finalLinkedTriple);

} // namespace loom::deployment::test

#endif // LOOM_TEST_DEPLOYMENT_DEPLOYMENTTESTSUPPORT_H
