#ifndef LOOM_MAPPING_ARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricFuCapabilityTemplate.h"
#include "Fabric/IR/BoundaryDataPath.h"
#include "Fabric/IR/ImplementationFamily.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

using ArtifactIdentity = ::loom::ArtifactIdentity;

enum class PortDirection { Input, Output };
enum class PortKind { Value, Stream, Memory };
enum class ComputeScheduleKind { Spatial, Temporal };
enum class TransportResourceKind { Switch, Fifo, Boundary };
enum class MemoryOperationKind { Load, Store };
enum class MemoryAccessPortRole {
  Address,
  Data,
  Mask,
  Control,
  Result,
  Done,
};

class TypeKey {
public:
  explicit constexpr TypeKey(std::uint64_t value) : value_(value) {}

  friend constexpr bool operator==(TypeKey lhs, TypeKey rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(TypeKey lhs, TypeKey rhs) {
    return !(lhs == rhs);
  }

  constexpr std::uint64_t value() const { return value_; }

private:
  std::uint64_t value_;
};

class PortRoleKey {
public:
  explicit constexpr PortRoleKey(std::uint64_t value) : value_(value) {}

  friend constexpr bool operator==(PortRoleKey lhs, PortRoleKey rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(PortRoleKey lhs, PortRoleKey rhs) {
    return !(lhs == rhs);
  }

  constexpr std::uint64_t value() const { return value_; }

private:
  std::uint64_t value_;
};

struct PortDescriptor {
  PortKind kind;
  TypeKey type;
  std::uint32_t payloadWidthBits = 0;
  std::uint32_t tagWidthBits = 0;
  PortRoleKey role = PortRoleKey(0);

  friend bool operator==(const PortDescriptor &lhs, const PortDescriptor &rhs) {
    return lhs.kind == rhs.kind && lhs.type == rhs.type &&
           lhs.payloadWidthBits == rhs.payloadWidthBits &&
           lhs.tagWidthBits == rhs.tagWidthBits && lhs.role == rhs.role;
  }
  friend bool operator!=(const PortDescriptor &lhs, const PortDescriptor &rhs) {
    return !(lhs == rhs);
  }
};

template <typename Tag> class TypedEntityId {
public:
  explicit constexpr TypedEntityId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(TypedEntityId lhs, TypedEntityId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(TypedEntityId lhs, TypedEntityId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

struct ComputeOccurrenceIdTag;
struct MemoryOccurrenceIdTag;
struct TransportEndpointIdTag;
struct TransportResourceIdTag;
struct ComputeRealizationIdTag;
struct MemoryServiceDomainIdTag;
struct MemoryImplementationIdTag;
struct MemoryOperationPortTemplateIdTag;
struct MemoryInternalConnectionIdTag;
struct MemorySemanticEncodingIdTag;
struct MemoryRealizationIdTag;

using GraphId = ::dataflow::GraphId;
using ActorId = ::dataflow::ActorId;
using LogicalMemoryRootId = ::dataflow::LogicalMemoryRootId;
using ComputeOccurrenceId = TypedEntityId<ComputeOccurrenceIdTag>;
using MemoryOccurrenceId = TypedEntityId<MemoryOccurrenceIdTag>;
using TransportEndpointId = TypedEntityId<TransportEndpointIdTag>;
using ComputeEndpointId = TransportEndpointId;
using MemoryEndpointId = TransportEndpointId;
using TransportResourceId = TypedEntityId<TransportResourceIdTag>;
using ComputeRealizationId = TypedEntityId<ComputeRealizationIdTag>;
using MemoryServiceDomainId = TypedEntityId<MemoryServiceDomainIdTag>;
using MemoryImplementationId = TypedEntityId<MemoryImplementationIdTag>;
using MemoryOperationPortTemplateId =
    TypedEntityId<MemoryOperationPortTemplateIdTag>;
using MemoryInternalConnectionId = TypedEntityId<MemoryInternalConnectionIdTag>;
using MemorySemanticEncodingId = TypedEntityId<MemorySemanticEncodingIdTag>;
using MemoryRealizationId = TypedEntityId<MemoryRealizationIdTag>;

template <typename EntityId>
using EntityReference = ::loom::ArtifactReference<EntityId>;

using GraphRef = ::dataflow::GraphRef;
using ActorRef = ::dataflow::ActorRef;
using LogicalMemoryRootRef = ::dataflow::LogicalMemoryRootRef;
using TransportEndpointRef = EntityReference<TransportEndpointId>;
using ComputeEndpointRef = TransportEndpointRef;
using MemoryEndpointRef = TransportEndpointRef;
using TransportResourceRef = EntityReference<TransportResourceId>;
using MemoryImplementationRef = EntityReference<MemoryImplementationId>;
using MemoryOperationPortTemplateRef =
    EntityReference<MemoryOperationPortTemplateId>;
using MemoryInternalConnectionRef = EntityReference<MemoryInternalConnectionId>;
using MemorySemanticEncodingRef = EntityReference<MemorySemanticEncodingId>;

struct GraphDescriptor {
  GraphId id;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
};

struct MemoryAccessPortDescriptor {
  MemoryAccessPortRole role;
  PortDirection direction;
  std::uint32_t index;
};

// Lossless importer-derived view of a canonical dataflow.load/store. The
// Mapping verifier validates this projection as one unit; Mapping records do
// not select or redefine these facts.
struct CanonicalMemoryActorView {
  MemoryOperationKind operation;
  LogicalMemoryRootId root;
  std::uint32_t accessWidthBits;
  std::uint32_t accessSizeBytes;
  std::uint32_t alignmentBytes;
  std::vector<MemoryAccessPortDescriptor> ports;
};

struct ActorDescriptor {
  ActorId id;
  GraphId graph;
  ::dataflow::CanonicalActorSchemaProjection semantics;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
  std::optional<CanonicalMemoryActorView> memory;
};

struct GraphPort {
  GraphId graph;
  PortDirection direction;
  std::uint32_t index;

  friend constexpr bool operator==(GraphPort lhs, GraphPort rhs) {
    return lhs.graph == rhs.graph && lhs.direction == rhs.direction &&
           lhs.index == rhs.index;
  }
  friend constexpr bool operator!=(GraphPort lhs, GraphPort rhs) {
    return !(lhs == rhs);
  }
};

struct LogicalMemoryRootDescriptor {
  LogicalMemoryRootId id;
  GraphId graph;
  std::vector<GraphPort> importPorts;
  std::vector<GraphPort> exportPorts;
};

struct ActorPort {
  ActorId actor;
  PortDirection direction;
  std::uint32_t index;

  friend constexpr bool operator==(ActorPort lhs, ActorPort rhs) {
    return lhs.actor == rhs.actor && lhs.direction == rhs.direction &&
           lhs.index == rhs.index;
  }
  friend constexpr bool operator!=(ActorPort lhs, ActorPort rhs) {
    return !(lhs == rhs);
  }
};

using DataflowEndpoint = std::variant<GraphPort, ActorPort>;

struct DataflowEdge {
  DataflowEndpoint source;
  DataflowEndpoint target;

  friend bool operator==(const DataflowEdge &lhs, const DataflowEdge &rhs) {
    return lhs.source == rhs.source && lhs.target == rhs.target;
  }
  friend bool operator!=(const DataflowEdge &lhs, const DataflowEdge &rhs) {
    return !(lhs == rhs);
  }
};

struct DataflowEdgeRef {
  ArtifactIdentity artifact;
  DataflowEdge edge;

  friend bool operator==(const DataflowEdgeRef &lhs,
                         const DataflowEdgeRef &rhs) {
    return lhs.artifact == rhs.artifact && lhs.edge == rhs.edge;
  }
  friend bool operator!=(const DataflowEdgeRef &lhs,
                         const DataflowEdgeRef &rhs) {
    return !(lhs == rhs);
  }
};

struct DataflowProgramView {
  ArtifactIdentity identity;
  std::vector<GraphDescriptor> graphs;
  std::vector<ActorDescriptor> actors;
  std::vector<DataflowEdge> edges;
  std::vector<LogicalMemoryRootDescriptor> logicalMemoryRoots;
};

struct FuDescriptor {
  ::loom::fabric::FabricFuTemplateRef id;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
  std::vector<::loom::fabric::FabricFuCapabilityTemplateRecord>
      capabilityTemplates;
};

struct ComputeEndpointDescriptor {
  ComputeEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  std::vector<TypeKey> compatibleTypes;
  PortRoleKey role;
  ::fabric::DataPathKind transportKind = ::fabric::DataPathKind::Bits;
};

struct ComputeLocalArcDescriptor {
  ::loom::fabric::FabricFuTemplatePortRef fuPort;
  ComputeEndpointRef endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ComputeOccurrenceDescriptor {
  ComputeOccurrenceId id;
  ComputeScheduleKind schedule;
  std::vector<::loom::fabric::FabricFuTemplateRef> functionalUnits;
  std::vector<ComputeEndpointDescriptor> endpoints;
  std::vector<ComputeLocalArcDescriptor> localArcs;
  // Fabric PE capability: one for Spatial, num_instruction for Temporal.
  std::int64_t instructionContextCapacity;
};

struct MemoryOperationPortRef {
  MemoryOperationPortTemplateRef operation;
  std::uint32_t index;
};

struct MemoryEndpointDescriptor {
  MemoryEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  std::vector<TypeKey> compatibleTypes;
  PortRoleKey role;
  ::fabric::DataPathKind transportKind = ::fabric::DataPathKind::Bits;
};

struct MemoryLocalArcDescriptor {
  MemoryOperationPortRef operationPort;
  MemoryEndpointRef endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct MemoryOccurrenceDescriptor {
  MemoryOccurrenceId id;
  MemoryImplementationRef implementation;
  std::vector<MemoryEndpointDescriptor> endpoints;
  std::vector<MemoryLocalArcDescriptor> localArcs;
};

struct TransportEndpointDescriptor {
  TransportEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  ::fabric::DataPathKind transportKind = ::fabric::DataPathKind::Bits;
};

struct TransportResourceDescriptor {
  TransportResourceId id;
  TransportResourceKind kind;
  std::vector<TransportEndpointDescriptor> endpoints;
  std::optional<::fabric::BoundaryDirection> boundaryDirection = std::nullopt;
};

struct TransportArcDescriptor {
  TransportEndpointRef source;
  TransportEndpointRef target;
};

struct TransportTraversalDescriptor {
  TransportResourceRef resource;
  TransportEndpointRef source;
  TransportEndpointRef target;
};

struct FabricOpDescriptor {
  ::loom::fabric::FabricFuTemplateNodeRef id;
  ::fabric::ImplementationFamilyId family;
  ::fabric::FamilyCapabilityParams capability;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
};

struct MemoryServiceDomainDescriptor {
  MemoryServiceDomainId id;
};

struct MemoryImplementationBoundaryPortDescriptor {
  PortDirection direction;
  PortDescriptor port;
  std::uint32_t maxInternalFanout;
};

struct MemoryImplementationDescriptor {
  MemoryImplementationId id;
  MemoryServiceDomainId service;
  std::vector<MemoryImplementationBoundaryPortDescriptor> boundaryPorts;
};

struct MemoryOperationPortDescriptor {
  MemoryAccessPortRole role;
  PortDirection direction;
  PortDescriptor port;
  std::uint32_t maxInternalFanout;
};

struct MemoryAccessCapability {
  std::uint32_t accessSizeBytes;
  std::uint32_t requiredAlignmentBytes;
};

struct MemoryOperationPortTemplateDescriptor {
  MemoryOperationPortTemplateId id;
  MemoryImplementationId implementation;
  MemoryOperationKind operation;
  std::vector<MemoryOperationPortDescriptor> ports;
  std::uint32_t physicalDataWidthBits;
  std::vector<MemoryAccessCapability> accessCapabilities;
};

struct MemoryOperationPort {
  MemoryOperationPortTemplateId operation;
  std::uint32_t index;

  friend bool operator==(MemoryOperationPort lhs, MemoryOperationPort rhs) {
    return lhs.operation == rhs.operation && lhs.index == rhs.index;
  }
};

struct MemoryImplementationBoundaryPort {
  std::uint32_t index;
};

using MemoryInternalEndpoint =
    std::variant<MemoryImplementationBoundaryPort, MemoryOperationPort>;

struct MemoryInternalConnectionDescriptor {
  MemoryInternalConnectionId id;
  MemoryImplementationId implementation;
  MemoryInternalEndpoint source;
  MemoryInternalEndpoint sink;
};

struct MemorySemanticEncodingDescriptor {
  MemorySemanticEncodingId id;
  MemoryImplementationId implementation;
  std::vector<MemoryOperationPortTemplateId> operationTemplates;
  std::vector<MemoryInternalConnectionId> internalConnections;
};

struct FabricHardwareView {
  ArtifactIdentity identity;
  std::vector<FuDescriptor> functionalUnits;
  std::vector<FabricOpDescriptor> operations;
  std::vector<MemoryServiceDomainDescriptor> memoryServiceDomains;
  std::vector<MemoryImplementationDescriptor> memoryImplementations;
  std::vector<MemoryOperationPortTemplateDescriptor>
      memoryOperationPortTemplates;
  std::vector<MemoryInternalConnectionDescriptor> memoryInternalConnections;
  std::vector<MemorySemanticEncodingDescriptor> memorySemanticEncodings;
  std::vector<ComputeOccurrenceDescriptor> computeOccurrences;
  std::vector<TransportResourceDescriptor> transportResources = {};
  std::vector<TransportArcDescriptor> transportArcs = {};
  std::vector<TransportTraversalDescriptor> transportTraversals = {};
  std::vector<MemoryOccurrenceDescriptor> memoryOccurrences = {};
};

struct ActorPortRef {
  ActorRef actor;
  PortDirection direction;
  std::uint32_t index;
};

struct GraphPortRef {
  GraphRef graph;
  PortDirection direction;
  std::uint32_t index;
};

struct ActorToFabricOp {
  ActorRef actor;
  ::loom::fabric::FabricFuTemplateNodeRef fabricOp;
  std::vector<std::uint32_t> operandPorts;
  std::vector<std::uint32_t> resultPorts;
};

struct BoundaryPortCorrespondence {
  ActorPortRef actorPort;
  ::loom::fabric::FabricFuTemplatePortRef fuPort;
};

struct MemoryImplementationBoundaryPortRef {
  MemoryImplementationRef implementation;
  std::uint32_t index;
};

struct ActorToMemoryOperation {
  ActorRef actor;
  MemoryOperationPortTemplateRef operation;
  LogicalMemoryRootRef root;
};

struct MemoryBoundaryPortCorrespondence {
  ActorPortRef actorPort;
  MemoryOperationPortRef operationPort;
};

struct MemoryGraphBoundaryPortCorrespondence {
  GraphPortRef graphPort;
  MemoryImplementationBoundaryPortRef implementationPort;
};

struct MemoryInternalEdgeWitness {
  DataflowEdgeRef edge;
  MemoryInternalConnectionRef connection;
};

struct ComputeRealizationDraft {
  ComputeRealizationId id;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<ActorToFabricOp> actorToOps;
  std::vector<BoundaryPortCorrespondence> boundaryPorts;
};

struct MemoryRealizationDraft {
  MemoryRealizationId id;
  std::vector<ActorRef> actors;
  std::vector<ActorToMemoryOperation> actorToOperations;
  std::vector<LogicalMemoryRootRef> roots;
  MemorySemanticEncodingRef encoding;
  std::vector<MemoryBoundaryPortCorrespondence> boundaryPorts;
  std::vector<MemoryGraphBoundaryPortCorrespondence> graphBoundaryPorts;
  std::vector<MemoryInternalEdgeWitness> internalEdges;
};

struct MappingDraftHeader {
  MappingDraftHeader() = delete;
  MappingDraftHeader(ArtifactIdentity dataflowIdentity,
                     ArtifactIdentity fabricIdentity)
      : dataflowIdentity(std::move(dataflowIdentity)),
        fabricIdentity(std::move(fabricIdentity)) {}

  ArtifactIdentity dataflowIdentity;
  ArtifactIdentity fabricIdentity;
};

struct TechMappingDraft {
  MappingDraftHeader header;
  std::vector<GraphRef> coveredGraphs;
  std::vector<ComputeRealizationDraft> realizations;
  std::vector<MemoryRealizationDraft> memoryRealizations;
};

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_H
