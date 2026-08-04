#ifndef LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATION_H
#define LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATION_H

#include "Common/Artifact.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/BlobDigest.h"
#include "Common/ExternalFileFingerprint.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

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
class FabricArtifactView;
}

namespace loom::hardware {

inline constexpr ArtifactSchemaDescriptor hardwareImplementationSchema{
    "loom.hardware_implementation", SchemaVersion{1, 0}};

enum class HardwareRepresentation {
  Rtl,
  GateNetlist,
  AsicPlaced,
  AsicRouted,
  AsicExtracted,
  FpgaPlaced,
  FpgaRouted,
  FpgaImage,
};

enum class HardwarePayloadRole {
  RtlSource,
  Netlist,
  PhysicalDatabase,
  Parasitics,
  LayoutStream,
  DeviceImage,
  GenerationConstraint,
  BlackBoxContract,
};

enum class ImplementationInterfaceRole {
  Data,
  Clock,
  Reset,
  Configuration,
  Memory,
  ExternalProtocol,
};

enum class RepresentationObjectKind {
  Module,
  Instance,
  Port,
  Net,
  Register,
  Memory,
  Cell,
  Pin,
  PhysicalObject,
  DeviceResource,
};

struct HardwarePayload final {
  HardwarePayloadRole role;
  std::string logicalName;
  std::string mediaType;
  BlobDigest content;

  friend bool operator==(const HardwarePayload &lhs,
                         const HardwarePayload &rhs) {
    return lhs.role == rhs.role && lhs.logicalName == rhs.logicalName &&
           lhs.mediaType == rhs.mediaType && lhs.content == rhs.content;
  }
};

struct RepresentationLocator final {
  RepresentationObjectKind kind;
  std::string canonicalName;

  friend bool operator==(const RepresentationLocator &lhs,
                         const RepresentationLocator &rhs) {
    return lhs.kind == rhs.kind && lhs.canonicalName == rhs.canonicalName;
  }
};

struct ImplementationInterface final {
  std::string interfaceKey;
  ImplementationInterfaceRole role;
  EncodedArtifactLocalReference semanticFabricRef;
  RepresentationLocator representationLocator;
  std::optional<std::string> devicePinRef;

  friend bool operator==(const ImplementationInterface &lhs,
                         const ImplementationInterface &rhs) {
    return lhs.interfaceKey == rhs.interfaceKey && lhs.role == rhs.role &&
           lhs.semanticFabricRef == rhs.semanticFabricRef &&
           lhs.representationLocator == rhs.representationLocator &&
           lhs.devicePinRef == rhs.devicePinRef;
  }
};

struct ActivityPoint final {
  std::string activityPointId;
  RepresentationLocator representationLocator;
  std::optional<EncodedArtifactLocalReference> semanticFabricRef;

  friend bool operator==(const ActivityPoint &lhs, const ActivityPoint &rhs) {
    return lhs.activityPointId == rhs.activityPointId &&
           lhs.representationLocator == rhs.representationLocator &&
           lhs.semanticFabricRef == rhs.semanticFabricRef;
  }
};

struct ExplicitFileDependency final {
  ExternalFileFingerprint contentSha256;
};

struct ToolBundledResourceDependency final {
  std::string stableProviderBuildIdentity;
  std::string resourceKey;
};

enum class ExternalDependencyKind {
  ExplicitFile,
  ToolBundledResource,
};

using ExternalDependencyIdentity =
    std::variant<ExplicitFileDependency, ToolBundledResourceDependency>;

struct ExternalInputBinding final {
  std::string providerInputSlotRef;
  ExternalDependencyIdentity dependencyIdentity;
};

struct HardwarePayloadRef final {
  HardwarePayloadRole role;
  std::string logicalName;
};

struct ExternalImplementationBinding final {
  std::string bindingId;
  std::string providerContractRef;
  std::vector<ExternalInputBinding> externalInputs;
  std::vector<EncodedArtifactLocalReference> fabricResourceRefs;
  std::vector<RepresentationLocator> representationLocators;
  std::optional<HardwarePayloadRef> blackBoxContractPayloadRef;
};

struct ExternalInputSlotContract final {
  std::string providerInputSlotRef;
  std::vector<ExternalDependencyKind> acceptedDependencyKinds;
};

using ExternalImplementationBindingValidator = llvm::Error (*)(
    const ExternalImplementationBinding &, HardwareRepresentation,
    const platform::ImplementationPlatform *);

struct ExternalImplementationContract final {
  std::string contractRef;
  std::vector<ExternalInputSlotContract> inputSlots;
  std::vector<HardwareRepresentation> supportedRepresentations;
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
      HardwareRepresentation representation) const;
  llvm::Error canonicalizeAndValidateBindings(
      std::vector<ExternalImplementationBinding> &bindings,
      HardwareRepresentation representation,
      const platform::ImplementationPlatform *implementationPlatform,
      llvm::ArrayRef<HardwarePayload> payloads,
      const fabric::FabricArtifactView &fabric) const;

private:
  std::vector<ExternalImplementationContract> contracts_;
};

struct MemoryMacroBinding final {
  ArtifactReference<fabric::FabricMemoryOccurrenceRef> fabricMemoryRef;
  std::string externalImplementationBindingId;
  RepresentationLocator representationLocator;
};

struct HardwareImplementationDraft final {
  ArtifactRootReference fabric;
  ArtifactRootReference configurationAbi;
  std::vector<ArtifactRootReference> interconnectImplementations;
  HardwareRepresentation representation;
  std::optional<ArtifactRootReference> implementationPlatform;
  std::vector<HardwarePayload> payloads;
  std::vector<ImplementationInterface> interfaces;
  std::vector<ActivityPoint> activityPoints;
  std::vector<MemoryMacroBinding> memoryMacroBindings;
  std::vector<ExternalImplementationBinding> externalImplementationBindings;
};

namespace detail {
class HardwareImplementationBuilder;
}

class HardwareImplementation final {
public:
  const ArtifactRootReference &fabric() const { return fabric_; }
  const ArtifactRootReference &configurationAbi() const {
    return configurationAbi_;
  }
  llvm::ArrayRef<ArtifactRootReference> interconnectImplementations() const {
    return interconnectImplementations_;
  }
  HardwareRepresentation representation() const { return representation_; }
  const std::optional<ArtifactRootReference> &implementationPlatform() const {
    return implementationPlatform_;
  }
  llvm::ArrayRef<HardwarePayload> payloads() const { return payloads_; }
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
      ArtifactRootReference fabric, ArtifactRootReference configurationAbi,
      std::vector<ArtifactRootReference> interconnectImplementations,
      HardwareRepresentation representation,
      std::optional<ArtifactRootReference> implementationPlatform,
      std::vector<HardwarePayload> payloads,
      std::vector<ImplementationInterface> interfaces,
      std::vector<ActivityPoint> activityPoints,
      std::vector<MemoryMacroBinding> memoryMacroBindings,
      std::vector<ExternalImplementationBinding> externalImplementationBindings)
      : fabric_(std::move(fabric)),
        configurationAbi_(std::move(configurationAbi)),
        interconnectImplementations_(std::move(interconnectImplementations)),
        representation_(representation),
        implementationPlatform_(std::move(implementationPlatform)),
        payloads_(std::move(payloads)), interfaces_(std::move(interfaces)),
        activityPoints_(std::move(activityPoints)),
        memoryMacroBindings_(std::move(memoryMacroBindings)),
        externalImplementationBindings_(
            std::move(externalImplementationBindings)) {}

  ArtifactRootReference fabric_;
  ArtifactRootReference configurationAbi_;
  std::vector<ArtifactRootReference> interconnectImplementations_;
  HardwareRepresentation representation_;
  std::optional<ArtifactRootReference> implementationPlatform_;
  std::vector<HardwarePayload> payloads_;
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
