#ifndef LOOM_RUNTIME_GEM5SIMULATIONBINDING_H
#define LOOM_RUNTIME_GEM5SIMULATIONBINDING_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::fabric {
class InstructionCoreArchitecturalContract;
class InstructionCoreMicroarchitecturalRealization;
}

namespace loom::runtime {

inline constexpr ArtifactSchemaDescriptor gem5SimulationBindingSchema{
    "loom.gem5_simulation_binding", SchemaVersion{2, 0}};

enum class Gem5ModelObjectClass : std::uint32_t {
  Processor = 0,
  SpatialBridge = 1,
  MemoryOrService = 2,
  Transport = 3,
  ExternalEndpoint = 4,
};

enum class Gem5ModelPortClass : std::uint32_t {
  SpatialBoundary = 0,
  MemoryOrService = 1,
  Transport = 2,
  ExternalEndpoint = 3,
};

struct Gem5ModelContractDescriptorRef final {
  std::string identity;
  SchemaVersion version;

  friend bool operator==(const Gem5ModelContractDescriptorRef &lhs,
                         const Gem5ModelContractDescriptorRef &rhs) {
    return lhs.identity == rhs.identity && lhs.version == rhs.version;
  }
  friend bool operator!=(const Gem5ModelContractDescriptorRef &lhs,
                         const Gem5ModelContractDescriptorRef &rhs) {
    return !(lhs == rhs);
  }
};

using Gem5CanonicalPayloadValidator =
    llvm::Error (*)(llvm::ArrayRef<std::uint8_t> payload);
using Gem5ProcessorCompatibilityValidator = llvm::Error (*)(
    llvm::ArrayRef<std::uint8_t> objectPayload,
    const fabric::InstructionCoreArchitecturalContract &architecture,
    const fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture);

struct Gem5ModelPortKindDescriptor final {
  std::uint32_t kind = 0;
  llvm::StringLiteral stableName;
  Gem5ModelPortClass portClass = Gem5ModelPortClass::Transport;
  bool allowsSharedBinding = false;
  Gem5CanonicalPayloadValidator validateCanonicalPayload = nullptr;
};

/// A static owner contract for one exact gem5 SimObject family. Object paths,
/// instantiated handles, and generated Python remain derived execution state.
struct Gem5ModelContractDescriptor final {
  ArtifactSchemaDescriptor descriptor;
  llvm::StringLiteral semanticIdentity;
  llvm::StringLiteral simObjectClass;
  Gem5ModelObjectClass objectClass = Gem5ModelObjectClass::Transport;
  bool allowsSharedBinding = false;
  Gem5CanonicalPayloadValidator validateCanonicalObjectPayload = nullptr;
  Gem5ProcessorCompatibilityValidator validateProcessorCompatibility = nullptr;
  llvm::ArrayRef<Gem5ModelPortKindDescriptor> portKinds;
};

Gem5ModelContractDescriptorRef gem5ModelContractDescriptorRef(
    const Gem5ModelContractDescriptor &descriptor);

llvm::Error
registerGem5ModelContract(const Gem5ModelContractDescriptor &descriptor);

const Gem5ModelContractDescriptor *findGem5ModelContract(
    const Gem5ModelContractDescriptorRef &reference);

const Gem5ModelPortKindDescriptor *findGem5ModelPortKind(
    const Gem5ModelContractDescriptor &descriptor, std::uint32_t kind);

struct Gem5SimObjectRef final {
  Gem5ModelContractDescriptorRef contract;
  std::vector<std::uint8_t> payload;

  friend bool operator==(const Gem5SimObjectRef &lhs,
                         const Gem5SimObjectRef &rhs) {
    return lhs.contract == rhs.contract && lhs.payload == rhs.payload;
  }
};

struct Gem5SimPortRef final {
  Gem5SimObjectRef object;
  std::uint32_t kind = 0;
  std::vector<std::uint8_t> payload;

  friend bool operator==(const Gem5SimPortRef &lhs,
                         const Gem5SimPortRef &rhs) {
    return lhs.object == rhs.object && lhs.kind == rhs.kind &&
           lhs.payload == rhs.payload;
  }
};

struct Gem5BuildIdentity final {
  std::string repositoryIdentity;
  std::string fullCommitIdentity;
  std::string buildConfigurationDigest;
  std::string binaryFingerprint;

  friend bool operator==(const Gem5BuildIdentity &lhs,
                         const Gem5BuildIdentity &rhs) {
    return lhs.repositoryIdentity == rhs.repositoryIdentity &&
           lhs.fullCommitIdentity == rhs.fullCommitIdentity &&
           lhs.buildConfigurationDigest == rhs.buildConfigurationDigest &&
           lhs.binaryFingerprint == rhs.binaryFingerprint;
  }
};

using Gem5ProcessorFabricRef =
    std::variant<fabric::HostCoreOccurrenceRef,
                 fabric::InstructionCoreContextRef>;

struct Gem5ProcessorCorrespondence final {
  Gem5ProcessorFabricRef processor;
  Gem5SimObjectRef simObject;
};

struct Gem5SpatialBridgeCorrespondence final {
  fabric::SpatialCoreOccurrenceRef spatialCore;
  fabric::FabricSpatialAttachmentEndpointRef spatialBoundary;
  Gem5SimPortRef bridgeEndpoint;
};

using Gem5MemoryOrServiceFabricRef =
    std::variant<fabric::SystemMemoryServiceRef,
                 fabric::SystemServiceEndpointRef>;

struct Gem5MemoryOrServiceCorrespondence final {
  Gem5MemoryOrServiceFabricRef fabricRef;
  Gem5SimObjectRef simObject;
  Gem5SimPortRef simPort;
};

using Gem5TransportFabricRef =
    std::variant<fabric::SystemTransportResourceRef,
                 fabric::FabricTransportEndpointRef>;

struct Gem5TransportCorrespondence final {
  Gem5TransportFabricRef fabricRef;
  Gem5SimObjectRef simObject;
  Gem5SimPortRef simPort;
};

struct Gem5ExternalEndpointCorrespondence final {
  fabric::ExternalBoundaryRef fabricRef;
  Gem5SimObjectRef simObject;
  Gem5SimPortRef simPort;
};

using Gem5Correspondence =
    std::variant<Gem5ProcessorCorrespondence,
                 Gem5SpatialBridgeCorrespondence,
                 Gem5MemoryOrServiceCorrespondence,
                 Gem5TransportCorrespondence,
                 Gem5ExternalEndpointCorrespondence>;

struct Gem5SimulationBindingDraft final {
  ArtifactRootReference fabric;
  ArtifactRootReference interconnectImplementation;
  Gem5BuildIdentity gem5BuildIdentity;
  std::string bridgeAbiIdentity;
  std::vector<Gem5Correspondence> correspondences;
};

namespace detail {
class Gem5SimulationBindingBuilder;
}

class Gem5SimulationBinding final {
public:
  const ArtifactRootReference &fabric() const { return fabric_; }
  const ArtifactRootReference &interconnectImplementation() const {
    return interconnectImplementation_;
  }
  const Gem5BuildIdentity &gem5BuildIdentity() const {
    return gem5BuildIdentity_;
  }
  llvm::StringRef bridgeAbiIdentity() const { return bridgeAbiIdentity_; }
  llvm::ArrayRef<Gem5Correspondence> correspondences() const {
    return correspondences_;
  }

private:
  Gem5SimulationBinding(
      ArtifactRootReference fabric,
      ArtifactRootReference interconnectImplementation,
      Gem5BuildIdentity gem5BuildIdentity, std::string bridgeAbiIdentity,
      std::vector<Gem5Correspondence> correspondences)
      : fabric_(std::move(fabric)),
        interconnectImplementation_(std::move(interconnectImplementation)),
        gem5BuildIdentity_(std::move(gem5BuildIdentity)),
        bridgeAbiIdentity_(std::move(bridgeAbiIdentity)),
        correspondences_(std::move(correspondences)) {}

  ArtifactRootReference fabric_;
  ArtifactRootReference interconnectImplementation_;
  Gem5BuildIdentity gem5BuildIdentity_;
  std::string bridgeAbiIdentity_;
  std::vector<Gem5Correspondence> correspondences_;

  friend class detail::Gem5SimulationBindingBuilder;
};

class FinalizedGem5SimulationBinding final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const Gem5SimulationBinding &binding() const { return binding_; }

private:
  FinalizedGem5SimulationBinding(ArtifactRootReference reference,
                                 CanonicalSemanticBytes canonicalBytes,
                                 Gem5SimulationBinding binding)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        binding_(std::move(binding)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  Gem5SimulationBinding binding_;

  friend llvm::Expected<FinalizedGem5SimulationBinding>
  importGem5SimulationBinding(const ArtifactRootReference &,
                              const ArtifactStore &);
};

llvm::Expected<FinalizedGem5SimulationBinding>
finalizeGem5SimulationBinding(Gem5SimulationBindingDraft draft,
                              const ArtifactStore &artifacts);

llvm::Expected<FinalizedGem5SimulationBinding>
importGem5SimulationBinding(const ArtifactRootReference &reference,
                            const ArtifactStore &artifacts);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SIMULATIONBINDING_H
