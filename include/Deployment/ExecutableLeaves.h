#ifndef LOOM_DEPLOYMENT_EXECUTABLELEAVES_H
#define LOOM_DEPLOYMENT_EXECUTABLELEAVES_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Frontend/Compilation/StaticGlobalMemory.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
}

namespace loom::deployment {

namespace detail {
class DeploymentCodecAccess;
}

using CanonicalTypeBytes = std::vector<std::uint8_t>;

enum class HostExternalInterfaceKind : std::uint32_t {
  Value,
  Stream,
  Memory,
};

enum class HostExternalInterfaceDirection : std::uint32_t {
  Input,
  Output,
  InOut,
};

struct HostProgramEntry final {
  std::uint64_t entryOrdinal;
  std::string abiSymbol;
  std::vector<CanonicalTypeBytes> valueArgumentTypes;
  std::vector<CanonicalTypeBytes> valueResultTypes;
  std::vector<std::uint64_t> externalInterfaceOrdinals;

  friend bool operator==(const HostProgramEntry &lhs,
                         const HostProgramEntry &rhs) {
    return lhs.entryOrdinal == rhs.entryOrdinal &&
           lhs.abiSymbol == rhs.abiSymbol &&
           lhs.valueArgumentTypes == rhs.valueArgumentTypes &&
           lhs.valueResultTypes == rhs.valueResultTypes &&
           lhs.externalInterfaceOrdinals == rhs.externalInterfaceOrdinals;
  }
};

struct HostExternalInterface final {
  std::uint64_t interfaceOrdinal;
  HostExternalInterfaceKind kind;
  HostExternalInterfaceDirection direction;
  CanonicalTypeBytes semanticType;

  friend bool operator==(const HostExternalInterface &lhs,
                         const HostExternalInterface &rhs) {
    return lhs.interfaceOrdinal == rhs.interfaceOrdinal &&
           lhs.kind == rhs.kind && lhs.direction == rhs.direction &&
           lhs.semanticType == rhs.semanticType;
  }
};

struct HostProgramLeafDraft final {
  ArtifactRootReference compilerTargetBinding;
  std::vector<std::uint8_t> programBytes;
  std::vector<HostProgramEntry> programEntries;
  std::vector<HostExternalInterface> externalInterfaces;
  std::vector<std::uint64_t> supportComponentOrdinals;
};

class HostProgramLeaf final {
public:
  const ArtifactRootReference &compilerTargetBinding() const {
    return compilerTargetBinding_;
  }
  const BlobDigest &programBlob() const { return programBlob_; }
  llvm::ArrayRef<HostProgramEntry> programEntries() const {
    return programEntries_;
  }
  llvm::ArrayRef<HostExternalInterface> externalInterfaces() const {
    return externalInterfaces_;
  }
  const BlobDigest &registrationTableDigest() const {
    return registrationTableDigest_;
  }
  llvm::ArrayRef<std::uint64_t> supportComponentOrdinals() const {
    return supportComponentOrdinals_;
  }

private:
  HostProgramLeaf(ArtifactRootReference compilerTargetBinding,
                  BlobDigest programBlob,
                  std::vector<HostProgramEntry> programEntries,
                  std::vector<HostExternalInterface> externalInterfaces,
                  BlobDigest registrationTableDigest,
                  std::vector<std::uint64_t> supportComponentOrdinals)
      : compilerTargetBinding_(std::move(compilerTargetBinding)),
        programBlob_(programBlob), programEntries_(std::move(programEntries)),
        externalInterfaces_(std::move(externalInterfaces)),
        registrationTableDigest_(registrationTableDigest),
        supportComponentOrdinals_(std::move(supportComponentOrdinals)) {}

  ArtifactRootReference compilerTargetBinding_;
  BlobDigest programBlob_;
  std::vector<HostProgramEntry> programEntries_;
  std::vector<HostExternalInterface> externalInterfaces_;
  BlobDigest registrationTableDigest_;
  std::vector<std::uint64_t> supportComponentOrdinals_;

  friend class ExecutableLeafBuilder;
  friend class detail::DeploymentCodecAccess;
};

struct StaticMemoryInitializedChunk final {
  std::uint64_t byteOffset;
  std::uint64_t byteCount;
  BlobDigest blobDigest;

  friend bool operator==(const StaticMemoryInitializedChunk &lhs,
                         const StaticMemoryInitializedChunk &rhs) {
    return lhs.byteOffset == rhs.byteOffset &&
           lhs.byteCount == rhs.byteCount &&
           lhs.blobDigest == rhs.blobDigest;
  }
};

struct StaticMemoryZeroFillRange final {
  std::uint64_t byteOffset;
  std::uint64_t byteCount;

  friend bool operator==(const StaticMemoryZeroFillRange &lhs,
                         const StaticMemoryZeroFillRange &rhs) {
    return lhs.byteOffset == rhs.byteOffset && lhs.byteCount == rhs.byteCount;
  }
};

class StaticMemoryImageLeaf final {
public:
  const ArtifactRootReference &canonicalDataflow() const {
    return canonicalDataflow_;
  }
  dataflow::RootedGraphLaunchRef rootedGraphLaunch() const {
    return rootedGraphLaunch_;
  }
  dataflow::LogicalMemoryRootRef logicalMemoryRoot() const {
    return logicalMemoryRoot_;
  }
  const ArtifactRootReference &layoutBinding() const { return layoutBinding_; }
  std::uint64_t sizeBytes() const { return sizeBytes_; }
  std::uint64_t alignmentBytes() const { return alignmentBytes_; }
  frontend::StaticMemoryPermissions permissions() const {
    return permissions_;
  }
  llvm::ArrayRef<StaticMemoryInitializedChunk> initializedChunks() const {
    return initializedChunks_;
  }
  llvm::ArrayRef<StaticMemoryZeroFillRange> zeroFillRanges() const {
    return zeroFillRanges_;
  }

private:
  StaticMemoryImageLeaf(
      ArtifactRootReference canonicalDataflow,
      dataflow::RootedGraphLaunchRef rootedGraphLaunch,
      dataflow::LogicalMemoryRootRef logicalMemoryRoot,
      ArtifactRootReference layoutBinding, std::uint64_t sizeBytes,
      std::uint64_t alignmentBytes,
      frontend::StaticMemoryPermissions permissions,
      std::vector<StaticMemoryInitializedChunk> initializedChunks,
      std::vector<StaticMemoryZeroFillRange> zeroFillRanges)
      : canonicalDataflow_(std::move(canonicalDataflow)),
        rootedGraphLaunch_(rootedGraphLaunch),
        logicalMemoryRoot_(logicalMemoryRoot),
        layoutBinding_(std::move(layoutBinding)), sizeBytes_(sizeBytes),
        alignmentBytes_(alignmentBytes), permissions_(permissions),
        initializedChunks_(std::move(initializedChunks)),
        zeroFillRanges_(std::move(zeroFillRanges)) {}

  ArtifactRootReference canonicalDataflow_;
  dataflow::RootedGraphLaunchRef rootedGraphLaunch_;
  dataflow::LogicalMemoryRootRef logicalMemoryRoot_;
  ArtifactRootReference layoutBinding_;
  std::uint64_t sizeBytes_;
  std::uint64_t alignmentBytes_;
  frontend::StaticMemoryPermissions permissions_;
  std::vector<StaticMemoryInitializedChunk> initializedChunks_;
  std::vector<StaticMemoryZeroFillRange> zeroFillRanges_;

  friend class ExecutableLeafBuilder;
  friend class detail::DeploymentCodecAccess;
};

llvm::Expected<HostProgramLeaf>
finalizeHostProgramLeaf(HostProgramLeafDraft draft,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs);

llvm::Error validateHostProgramLeaf(const HostProgramLeaf &leaf,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs);

llvm::Expected<StaticMemoryImageLeaf>
buildStaticMemoryImageLeaf(const ArtifactRootReference &canonicalDataflow,
                           dataflow::RootedGraphLaunchRef rootedGraphLaunch,
                           dataflow::LogicalMemoryRootRef logicalMemoryRoot,
                           const ArtifactRootReference &layoutBinding,
                           const frontend::StaticGlobalMemoryCatalog &catalog,
                           std::uint64_t globalOrdinal,
                           const ArtifactStore &artifacts,
                           const BlobStore &blobs);

llvm::Error validateStaticMemoryImageLeaf(const StaticMemoryImageLeaf &leaf,
                                          const ArtifactStore &artifacts,
                                          const BlobStore &blobs);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_EXECUTABLELEAVES_H
