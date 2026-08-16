#ifndef LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATION_H
#define LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATION_H

#include "Common/Artifact.h"
#include "Common/ExternalFileFingerprint.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::platform {
class ImplementationPlatform;
}

namespace loom::fabric {
class FabricSystemRootView;
}

namespace loom::hardware {

inline constexpr ArtifactSchemaDescriptor hardwareImplementationSchema{
    "loom.hardware_implementation", SchemaVersion{4, 1}};

struct ImplementationDataInterfaceRef final {
  fabric::FabricSpatialAttachmentEndpointRef endpoint;

  friend bool operator==(const ImplementationDataInterfaceRef &lhs,
                         const ImplementationDataInterfaceRef &rhs) {
    return lhs.endpoint == rhs.endpoint;
  }
};

struct ImplementationMemoryInterfaceRef final {
  fabric::FabricSpatialAttachmentEndpointRef endpoint;

  friend bool operator==(const ImplementationMemoryInterfaceRef &lhs,
                         const ImplementationMemoryInterfaceRef &rhs) {
    return lhs.endpoint == rhs.endpoint;
  }
};

struct ImplementationClockInterfaceRef final {
  fabric::HardwareDomainRef domain;

  friend bool operator==(const ImplementationClockInterfaceRef &lhs,
                         const ImplementationClockInterfaceRef &rhs) {
    return lhs.domain == rhs.domain;
  }
};

struct ImplementationResetInterfaceRef final {
  fabric::HardwareDomainRef domain;

  friend bool operator==(const ImplementationResetInterfaceRef &lhs,
                         const ImplementationResetInterfaceRef &rhs) {
    return lhs.domain == rhs.domain;
  }
};

struct ImplementationConfigurationInterfaceRef final {
  ProgrammingUnitRef programmingUnit;

  friend bool operator==(const ImplementationConfigurationInterfaceRef &lhs,
                         const ImplementationConfigurationInterfaceRef &rhs) {
    return lhs.programmingUnit == rhs.programmingUnit;
  }
};

struct ImplementationExternalProtocolInterfaceRef final {
  fabric::ExternalBoundaryRef boundary;

  friend bool
  operator==(const ImplementationExternalProtocolInterfaceRef &lhs,
             const ImplementationExternalProtocolInterfaceRef &rhs) {
    return lhs.boundary == rhs.boundary;
  }
};

enum class ImplementationInterfaceSemanticRefKind : std::uint32_t {
  Data = 0,
  Memory = 1,
  Clock = 2,
  Reset = 3,
  Configuration = 4,
  ExternalProtocol = 5,
};

constexpr std::uint32_t implementationInterfaceSemanticRefKindOrdinal(
    ImplementationInterfaceSemanticRefKind kind) {
  return static_cast<std::uint32_t>(kind);
}

using ImplementationInterfaceSemanticRef = std::variant<
    ImplementationDataInterfaceRef, ImplementationMemoryInterfaceRef,
    ImplementationClockInterfaceRef, ImplementationResetInterfaceRef,
    ImplementationConfigurationInterfaceRef,
    ImplementationExternalProtocolInterfaceRef>;

struct ImplementationInterface final {
  ImplementationInterfaceSemanticRef semanticRef;
  RepresentationLocator representationLocator;
  std::optional<std::string> devicePinRef;

  friend bool operator==(const ImplementationInterface &lhs,
                         const ImplementationInterface &rhs) {
    return lhs.semanticRef == rhs.semanticRef &&
           lhs.representationLocator == rhs.representationLocator &&
           lhs.devicePinRef == rhs.devicePinRef;
  }
};

struct ActivityPoint final {
  RepresentationLocator representationLocator;
  std::optional<fabric::FabricPhysicalOccurrenceOwnerRef> semanticFabricRef;

  friend bool operator==(const ActivityPoint &lhs, const ActivityPoint &rhs) {
    return lhs.representationLocator == rhs.representationLocator &&
           lhs.semanticFabricRef == rhs.semanticFabricRef;
  }
};

struct ExplicitFileDependency final {
  ExternalFileFingerprint contentSha256;

  friend bool operator==(const ExplicitFileDependency &lhs,
                         const ExplicitFileDependency &rhs) {
    return lhs.contentSha256 == rhs.contentSha256;
  }
};

struct ToolBundledResourceDependency final {
  std::string stableProviderBuildIdentity;
  std::string resourceKey;

  friend bool operator==(const ToolBundledResourceDependency &lhs,
                         const ToolBundledResourceDependency &rhs) {
    return lhs.stableProviderBuildIdentity == rhs.stableProviderBuildIdentity &&
           lhs.resourceKey == rhs.resourceKey;
  }
};

enum class ExternalDependencyKind : std::uint32_t {
  ExplicitFile = 0,
  ToolBundledResource = 1,
};

using ExternalDependencyIdentity =
    std::variant<ExplicitFileDependency, ToolBundledResourceDependency>;

struct ExternalInputBinding final {
  std::string providerInputSlotRef;
  ExternalDependencyIdentity dependencyIdentity;

  friend bool operator==(const ExternalInputBinding &lhs,
                         const ExternalInputBinding &rhs) {
    return lhs.providerInputSlotRef == rhs.providerInputSlotRef &&
           lhs.dependencyIdentity == rhs.dependencyIdentity;
  }
};

/// Authoring-time payload identity. Finalization resolves it to the dense
/// ordinal owned by the canonical representation-root payload catalog.
struct ImplementationPayloadKey final {
  PayloadRole role;
  std::string canonicalLogicalName;

  friend bool operator==(const ImplementationPayloadKey &lhs,
                         const ImplementationPayloadKey &rhs) {
    return lhs.role == rhs.role &&
           lhs.canonicalLogicalName == rhs.canonicalLogicalName;
  }
};

struct ImplementationPayloadRef final {
  std::uint64_t ordinal = 0;

  friend bool operator==(ImplementationPayloadRef lhs,
                         ImplementationPayloadRef rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

struct ExternalImplementationBindingDraft final {
  std::string providerContractRef;
  std::vector<ExternalInputBinding> externalInputs;
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef> fabricResourceRefs;
  std::vector<RepresentationLocator> representationLocators;
  std::optional<ImplementationPayloadKey> blackBoxContractPayload;

  friend bool operator==(const ExternalImplementationBindingDraft &lhs,
                         const ExternalImplementationBindingDraft &rhs) {
    return lhs.providerContractRef == rhs.providerContractRef &&
           lhs.externalInputs == rhs.externalInputs &&
           lhs.fabricResourceRefs == rhs.fabricResourceRefs &&
           lhs.representationLocators == rhs.representationLocators &&
           lhs.blackBoxContractPayload == rhs.blackBoxContractPayload;
  }
};

struct ExternalImplementationBinding final {
  std::string providerContractRef;
  std::vector<ExternalInputBinding> externalInputs;
  std::vector<fabric::FabricPhysicalOccurrenceOwnerRef> fabricResourceRefs;
  std::vector<RepresentationLocator> representationLocators;
  std::optional<ImplementationPayloadRef> blackBoxContractPayloadRef;
};

struct ExternalImplementationBindingRef final {
  std::uint64_t ordinal = 0;

  friend bool operator==(ExternalImplementationBindingRef lhs,
                         ExternalImplementationBindingRef rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

struct ExternalInputSlotContract final {
  std::string providerInputSlotRef;
  std::vector<ExternalDependencyKind> acceptedDependencyKinds;
};

using ExternalImplementationBindingValidator =
    llvm::Error (*)(const ExternalImplementationBindingDraft &,
                    const ImplementationRepresentationRoot &,
                    const platform::ImplementationPlatform *);

struct ExternalImplementationContract final {
  std::string contractRef;
  std::vector<ExternalInputSlotContract> inputSlots;
  std::vector<RepresentationRootVariant> supportedRepresentations;
  bool blackBoxContractRequired = false;
  bool memoryMacroCapable = false;
  ExternalImplementationBindingValidator validator = nullptr;
};

class ExternalImplementationContractCatalog final {
public:
  llvm::Error add(ExternalImplementationContract contract);
  std::optional<ExternalImplementationContract>
  find(llvm::StringRef contractRef) const;
  llvm::Expected<std::vector<ExternalInputBinding>>
  canonicalizeAndValidateInputs(
      llvm::StringRef contractRef,
      llvm::ArrayRef<ExternalInputBinding> externalInputs,
      RepresentationRootVariant representation) const;
  llvm::Error canonicalizeAndValidateBindings(
      std::vector<ExternalImplementationBindingDraft> &bindings,
      const ImplementationRepresentationRoot &representation,
      const platform::ImplementationPlatform *implementationPlatform,
      const fabric::FabricSystemRootView &fabric) const;

private:
  std::vector<ExternalImplementationContract> contracts_;
};

/// The draft index is ephemeral authoring state. Finalization remaps it after
/// canonical binding ordering and persists only the derived dense reference.
struct MemoryMacroBindingDraft final {
  fabric::FabricPhysicalOccurrenceOwnerRef fabricMemoryRef;
  std::uint64_t externalImplementationBindingDraftIndex = 0;
  RepresentationLocator representationLocator;
};

struct MemoryMacroBinding final {
  fabric::FabricPhysicalOccurrenceOwnerRef fabricMemoryRef;
  ExternalImplementationBindingRef externalImplementationBindingRef;
  RepresentationLocator representationLocator;
};

struct HardwareImplementationDraft final {
  ArtifactRootReference fabric;
  fabric::SpatialCoreOccurrenceRef subject;
  ArtifactRootReference configurationAbi;
  ImplementationRepresentationRoot representationRoot;
  std::optional<ArtifactRootReference> implementationPlatform;
  std::vector<ImplementationInterface> interfaces;
  std::vector<ActivityPoint> activityPoints;
  std::vector<MemoryMacroBindingDraft> memoryMacroBindings;
  std::vector<ExternalImplementationBindingDraft>
      externalImplementationBindings;
};

namespace detail {
class HardwareImplementationBuilder;
}

class HardwareImplementation final {
public:
  const ArtifactRootReference &fabric() const { return fabric_; }
  fabric::SpatialCoreOccurrenceRef subject() const { return subject_; }
  const ArtifactRootReference &configurationAbi() const {
    return configurationAbi_;
  }
  const ImplementationRepresentationRoot &representationRoot() const {
    return representationRoot_;
  }
  const std::optional<ArtifactRootReference> &implementationPlatform() const {
    return implementationPlatform_;
  }
  llvm::ArrayRef<ImplementationInterface> interfaces() const {
    return interfaces_;
  }
  llvm::ArrayRef<ActivityPoint> activityPoints() const {
    return activityPoints_;
  }
  llvm::ArrayRef<MemoryMacroBinding> memoryMacroBindings() const {
    return memoryMacroBindings_;
  }
  llvm::ArrayRef<ExternalImplementationBinding>
  externalImplementationBindings() const {
    return externalImplementationBindings_;
  }

private:
  HardwareImplementation(
      ArtifactRootReference fabric, fabric::SpatialCoreOccurrenceRef subject,
      ArtifactRootReference configurationAbi,
      ImplementationRepresentationRoot representationRoot,
      std::optional<ArtifactRootReference> implementationPlatform,
      std::vector<ImplementationInterface> interfaces,
      std::vector<ActivityPoint> activityPoints,
      std::vector<MemoryMacroBinding> memoryMacroBindings,
      std::vector<ExternalImplementationBinding> externalImplementationBindings)
      : fabric_(std::move(fabric)), subject_(subject),
        configurationAbi_(std::move(configurationAbi)),
        representationRoot_(std::move(representationRoot)),
        implementationPlatform_(std::move(implementationPlatform)),
        interfaces_(std::move(interfaces)),
        activityPoints_(std::move(activityPoints)),
        memoryMacroBindings_(std::move(memoryMacroBindings)),
        externalImplementationBindings_(
            std::move(externalImplementationBindings)) {}

  ArtifactRootReference fabric_;
  fabric::SpatialCoreOccurrenceRef subject_;
  ArtifactRootReference configurationAbi_;
  ImplementationRepresentationRoot representationRoot_;
  std::optional<ArtifactRootReference> implementationPlatform_;
  std::vector<ImplementationInterface> interfaces_;
  std::vector<ActivityPoint> activityPoints_;
  std::vector<MemoryMacroBinding> memoryMacroBindings_;
  std::vector<ExternalImplementationBinding> externalImplementationBindings_;

  friend class detail::HardwareImplementationBuilder;
};

class FinalizedHardwareImplementation final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const HardwareImplementation &implementation() const {
    return implementation_;
  }

private:
  FinalizedHardwareImplementation(ArtifactRootReference reference,
                                  CanonicalSemanticBytes canonicalBytes,
                                  HardwareImplementation implementation)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        implementation_(std::move(implementation)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  HardwareImplementation implementation_;

  friend llvm::Expected<FinalizedHardwareImplementation>
  finalizeHardwareImplementation(HardwareImplementationDraft,
                                 const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<FinalizedHardwareImplementation>
  finalizeHardwareImplementation(HardwareImplementationDraft,
                                 const ExternalImplementationContractCatalog &,
                                 const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<FinalizedHardwareImplementation>
  importHardwareImplementation(const ArtifactRootReference &,
                               const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<FinalizedHardwareImplementation>
  importHardwareImplementation(const ArtifactRootReference &,
                               const ExternalImplementationContractCatalog &,
                               const ArtifactStore &, const BlobStore &);
};

llvm::Expected<FinalizedHardwareImplementation>
finalizeHardwareImplementation(HardwareImplementationDraft draft,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs);

llvm::Expected<FinalizedHardwareImplementation> finalizeHardwareImplementation(
    HardwareImplementationDraft draft,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<FinalizedHardwareImplementation>
importHardwareImplementation(const ArtifactRootReference &reference,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs);

llvm::Expected<FinalizedHardwareImplementation> importHardwareImplementation(
    const ArtifactRootReference &reference,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATION_H
