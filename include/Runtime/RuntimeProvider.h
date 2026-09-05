#ifndef LOOM_RUNTIME_RUNTIMEPROVIDER_H
#define LOOM_RUNTIME_RUNTIMEPROVIDER_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::runtime {

/// Exact provider-owned timing model for the bounded resource-time transition
/// operations that can be prepared and committed atomically. The descriptor
/// is the semantic owner; RuntimePlatformBinding retains a derived copy so a
/// packaged Deployment does not depend on ambient machine configuration.
struct RuntimeResourceTimeCostModel final {
  std::uint64_t memoryCopySetupPicoseconds = 0;
  std::uint64_t memoryCopyBytePicoseconds = 0;
  std::uint64_t configurationWordPicoseconds = 0;
  std::uint64_t configurationCommitPicoseconds = 0;

  friend bool operator==(const RuntimeResourceTimeCostModel &lhs,
                         const RuntimeResourceTimeCostModel &rhs) {
    return lhs.memoryCopySetupPicoseconds == rhs.memoryCopySetupPicoseconds &&
           lhs.memoryCopyBytePicoseconds == rhs.memoryCopyBytePicoseconds &&
           lhs.configurationWordPicoseconds ==
               rhs.configurationWordPicoseconds &&
           lhs.configurationCommitPicoseconds ==
               rhs.configurationCommitPicoseconds;
  }
};

enum class RuntimeEndpointClass : std::uint32_t {
  Identity = 0,
  Programming = 1,
  Memory = 2,
  Completion = 3,
};

enum class RuntimeEndpointFlow : std::uint32_t {
  RuntimeToImplementation = 0,
  ImplementationToRuntime = 1,
  Bidirectional = 2,
};

using RuntimeEndpointPayloadValidator =
    llvm::Error (*)(llvm::ArrayRef<std::uint8_t> payload);

/// One endpoint kind in a static provider-owned schema. The kind ordinal and
/// payload codec are meaningful only under the enclosing exact descriptor.
struct RuntimeProviderEndpointKindDescriptor final {
  std::uint32_t kind = 0;
  llvm::StringLiteral stableName;
  RuntimeEndpointClass endpointClass = RuntimeEndpointClass::Identity;
  RuntimeEndpointFlow flow = RuntimeEndpointFlow::Bidirectional;
  bool allowsSharedBinding = false;
  RuntimeEndpointPayloadValidator validateCanonicalPayload = nullptr;
};

/// Static runtime contract. Instances, device paths, handles, leases, and
/// addresses are invocation state and never enter this descriptor.
struct RuntimeProviderDescriptor final {
  ArtifactSchemaDescriptor descriptor;
  llvm::StringLiteral implementationSemanticIdentity;
  llvm::StringLiteral runtimeAbiIdentity;
  llvm::ArrayRef<RuntimeProviderEndpointKindDescriptor> endpointKinds;
  bool supportsHardwareReportedIdentity = false;
  bool supportsTrustedImmutableIdentity = false;
  bool supportsAtomicProgrammingMulticast = false;
  bool supportsPreparedActivationReplacement = false;
  /// Presence admits provider-atomic configuration and logical-memory copy
  /// work during prepared activation replacement. Every component is
  /// strictly positive so nonempty work has a nonzero exact cost.
  std::optional<RuntimeResourceTimeCostModel> resourceTimeCostModel;
};

struct RuntimeProviderDescriptorRef final {
  std::string identity;
  SchemaVersion version;

  friend bool operator==(const RuntimeProviderDescriptorRef &lhs,
                         const RuntimeProviderDescriptorRef &rhs) {
    return lhs.identity == rhs.identity && lhs.version == rhs.version;
  }
};

struct RuntimeProviderEndpointRef final {
  std::uint32_t kind = 0;
  std::vector<std::uint8_t> payload;

  friend bool operator==(const RuntimeProviderEndpointRef &lhs,
                         const RuntimeProviderEndpointRef &rhs) {
    return lhs.kind == rhs.kind && lhs.payload == rhs.payload;
  }
};

RuntimeProviderDescriptorRef
runtimeProviderDescriptorRef(const RuntimeProviderDescriptor &descriptor);

/// Registers one descriptor with static storage duration. Re-registering the
/// same object is a no-op; a competing owner for an exact descriptor is an
/// error rather than an ambient selection rule.
llvm::Error registerRuntimeProvider(const RuntimeProviderDescriptor &provider);

const RuntimeProviderDescriptor *
findRuntimeProvider(const RuntimeProviderDescriptorRef &reference);

const RuntimeProviderEndpointKindDescriptor *
findRuntimeEndpointKind(const RuntimeProviderDescriptor &provider,
                        std::uint32_t kind);

llvm::Error
validateRuntimeProviderEndpoint(const RuntimeProviderDescriptor &provider,
                                const RuntimeProviderEndpointRef &endpoint,
                                RuntimeEndpointClass expectedClass,
                                RuntimeEndpointFlow expectedFlow);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_RUNTIMEPROVIDER_H
