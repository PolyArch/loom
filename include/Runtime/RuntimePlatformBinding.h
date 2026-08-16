#ifndef LOOM_RUNTIME_RUNTIMEPLATFORMBINDING_H
#define LOOM_RUNTIME_RUNTIMEPLATFORMBINDING_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementationLocalReference.h"
#include "Runtime/RuntimeProvider.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::runtime {

namespace detail {
class RuntimePlatformBindingBuilder;
}

inline constexpr ArtifactSchemaDescriptor runtimePlatformBindingSchema{
    "loom.runtime_platform_binding", SchemaVersion{3, 1}};

struct HardwareReportedIdentity final {
  RuntimeProviderEndpointRef implementationIdentityEndpoint;

  friend bool operator==(const HardwareReportedIdentity &lhs,
                         const HardwareReportedIdentity &rhs) {
    return lhs.implementationIdentityEndpoint ==
           rhs.implementationIdentityEndpoint;
  }
};

struct TrustedImmutableIdentity final {
  BlobDigest attestationBlob;

  friend bool operator==(const TrustedImmutableIdentity &lhs,
                         const TrustedImmutableIdentity &rhs) {
    return lhs.attestationBlob == rhs.attestationBlob;
  }
};

using RuntimeIdentityVerification =
    std::variant<HardwareReportedIdentity, TrustedImmutableIdentity>;

struct RuntimeProgrammingBinding final {
  hardware::ProgrammingUnitRef programmingUnit;
  ArtifactReference<hardware::HardwareImplementationInterfaceRef>
      implementationInterface;
  RuntimeProviderEndpointRef providerEndpoint;

  friend bool operator==(const RuntimeProgrammingBinding &lhs,
                         const RuntimeProgrammingBinding &rhs) {
    return lhs.programmingUnit == rhs.programmingUnit &&
           lhs.implementationInterface == rhs.implementationInterface &&
           lhs.providerEndpoint == rhs.providerEndpoint;
  }
};

struct RuntimeInterfaceBinding final {
  ArtifactReference<hardware::HardwareImplementationInterfaceRef>
      implementationInterface;
  RuntimeProviderEndpointRef providerEndpoint;

  friend bool operator==(const RuntimeInterfaceBinding &lhs,
                         const RuntimeInterfaceBinding &rhs) {
    return lhs.implementationInterface == rhs.implementationInterface &&
           lhs.providerEndpoint == rhs.providerEndpoint;
  }
};

struct RuntimePlatformBindingDraft final {
  ArtifactRootReference hardwareImplementation;
  RuntimeProviderDescriptorRef providerDescriptor;
  RuntimeIdentityVerification identityVerification;
  std::vector<RuntimeProgrammingBinding> programmingBindings;
  std::vector<RuntimeInterfaceBinding> memoryInterfaceBindings;
  std::vector<RuntimeInterfaceBinding> completionInterfaceBindings;
};

struct RuntimeProviderBinding final {
  RuntimeProviderDescriptorRef descriptor;
  std::string implementationSemanticIdentity;
  std::string runtimeAbiIdentity;

  friend bool operator==(const RuntimeProviderBinding &lhs,
                         const RuntimeProviderBinding &rhs) {
    return lhs.descriptor == rhs.descriptor &&
           lhs.implementationSemanticIdentity ==
               rhs.implementationSemanticIdentity &&
           lhs.runtimeAbiIdentity == rhs.runtimeAbiIdentity;
  }
};

class RuntimePlatformBinding final {
public:
  const ArtifactRootReference &hardwareImplementation() const {
    return hardwareImplementation_;
  }
  const RuntimeProviderBinding &providerBinding() const {
    return providerBinding_;
  }
  const RuntimeIdentityVerification &identityVerification() const {
    return identityVerification_;
  }
  llvm::ArrayRef<RuntimeProgrammingBinding> programmingBindings() const {
    return programmingBindings_;
  }
  llvm::ArrayRef<RuntimeInterfaceBinding> memoryInterfaceBindings() const {
    return memoryInterfaceBindings_;
  }
  llvm::ArrayRef<RuntimeInterfaceBinding> completionInterfaceBindings() const {
    return completionInterfaceBindings_;
  }

private:
  RuntimePlatformBinding(
      ArtifactRootReference hardwareImplementation,
      RuntimeProviderBinding providerBinding,
      RuntimeIdentityVerification identityVerification,
      std::vector<RuntimeProgrammingBinding> programmingBindings,
      std::vector<RuntimeInterfaceBinding> memoryInterfaceBindings,
      std::vector<RuntimeInterfaceBinding> completionInterfaceBindings)
      : hardwareImplementation_(std::move(hardwareImplementation)),
        providerBinding_(std::move(providerBinding)),
        identityVerification_(std::move(identityVerification)),
        programmingBindings_(std::move(programmingBindings)),
        memoryInterfaceBindings_(std::move(memoryInterfaceBindings)),
        completionInterfaceBindings_(std::move(completionInterfaceBindings)) {}

  ArtifactRootReference hardwareImplementation_;
  RuntimeProviderBinding providerBinding_;
  RuntimeIdentityVerification identityVerification_;
  std::vector<RuntimeProgrammingBinding> programmingBindings_;
  std::vector<RuntimeInterfaceBinding> memoryInterfaceBindings_;
  std::vector<RuntimeInterfaceBinding> completionInterfaceBindings_;

  friend class detail::RuntimePlatformBindingBuilder;
  friend class FinalizedRuntimePlatformBinding;
  friend llvm::Expected<class FinalizedRuntimePlatformBinding>
  importRuntimePlatformBinding(const ArtifactRootReference &,
                               const ArtifactStore &, const BlobStore &);
};

class FinalizedRuntimePlatformBinding final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const RuntimePlatformBinding &binding() const { return binding_; }

private:
  FinalizedRuntimePlatformBinding(ArtifactRootReference reference,
                                  CanonicalSemanticBytes canonicalBytes,
                                  RuntimePlatformBinding binding)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        binding_(std::move(binding)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  RuntimePlatformBinding binding_;

  friend llvm::Expected<FinalizedRuntimePlatformBinding>
  importRuntimePlatformBinding(const ArtifactRootReference &,
                               const ArtifactStore &, const BlobStore &);
};

llvm::Expected<FinalizedRuntimePlatformBinding>
finalizeRuntimePlatformBinding(RuntimePlatformBindingDraft draft,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs);

llvm::Expected<FinalizedRuntimePlatformBinding>
importRuntimePlatformBinding(const ArtifactRootReference &reference,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_RUNTIMEPLATFORMBINDING_H
