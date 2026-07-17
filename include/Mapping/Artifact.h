#ifndef LOOM_MAPPING_ARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_H

#include "Common/Artifact.h"
#include "Fabric/IR/BoundaryDataPath.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

using SchemaVersion = ::loom::SchemaVersion;
using ArtifactIdentity = ::loom::ArtifactIdentity;

enum class MappingProfile { TechMapping, PhysicalMapping };
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

class SemanticKey {
public:
  SemanticKey() = delete;
  SemanticKey(std::initializer_list<std::uint8_t> bytes) : bytes_(bytes) {}
  explicit SemanticKey(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  friend bool operator==(const SemanticKey &lhs, const SemanticKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const SemanticKey &lhs, const SemanticKey &rhs) {
    return !(lhs == rhs);
  }

private:
  // Lossless canonical bytes from the owning semantic artifact view. This is
  // an equality identity, not a truncated hash.
  std::vector<std::uint8_t> bytes_;
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

struct GraphIdTag;
struct ActorIdTag;
struct EdgeIdTag;
struct LogicalMemoryRootIdTag;
struct FuIdTag;
struct ComputeOccurrenceIdTag;
struct TransportEndpointIdTag;
struct TransportResourceIdTag;
struct FabricOpIdTag;
struct EncodingIdTag;
struct ComputeRealizationIdTag;
struct MemoryServiceDomainIdTag;
struct MemoryImplementationIdTag;
struct MemoryOperationPortTemplateIdTag;
struct MemoryInternalConnectionIdTag;
struct MemorySemanticEncodingIdTag;
struct MemoryRealizationIdTag;

using GraphId = TypedEntityId<GraphIdTag>;
using ActorId = TypedEntityId<ActorIdTag>;
using EdgeId = TypedEntityId<EdgeIdTag>;
using LogicalMemoryRootId = TypedEntityId<LogicalMemoryRootIdTag>;
using FuId = TypedEntityId<FuIdTag>;
using ComputeOccurrenceId = TypedEntityId<ComputeOccurrenceIdTag>;
using TransportEndpointId = TypedEntityId<TransportEndpointIdTag>;
using ComputeEndpointId = TransportEndpointId;
using TransportResourceId = TypedEntityId<TransportResourceIdTag>;
using FabricOpId = TypedEntityId<FabricOpIdTag>;
using EncodingId = TypedEntityId<EncodingIdTag>;
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

using GraphRef = EntityReference<GraphId>;
using ActorRef = EntityReference<ActorId>;
using EdgeRef = EntityReference<EdgeId>;
using LogicalMemoryRootRef = EntityReference<LogicalMemoryRootId>;
using FuRef = EntityReference<FuId>;
using TransportEndpointRef = EntityReference<TransportEndpointId>;
using ComputeEndpointRef = TransportEndpointRef;
using TransportResourceRef = EntityReference<TransportResourceId>;
using FabricOpRef = EntityReference<FabricOpId>;
using EncodingRef = EntityReference<EncodingId>;
using MemoryImplementationRef = EntityReference<MemoryImplementationId>;
using MemoryOperationPortTemplateRef =
    EntityReference<MemoryOperationPortTemplateId>;
using MemoryInternalConnectionRef = EntityReference<MemoryInternalConnectionId>;
using MemorySemanticEncodingRef = EntityReference<MemorySemanticEncodingId>;

struct FuPortRef {
  FuRef fu;
  PortDirection direction;
  std::uint32_t index;
};

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
  SemanticKey operation;
  SemanticKey attributes;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
  std::optional<CanonicalMemoryActorView> memory;
};

struct GraphPort {
  GraphId graph;
  PortDirection direction;
  std::uint32_t index;
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
};

using DataflowEndpoint = std::variant<GraphPort, ActorPort>;

struct DataflowEdge {
  EdgeId id;
  DataflowEndpoint source;
  DataflowEndpoint target;
};

struct DataflowProgramView {
  ArtifactIdentity identity;
  std::vector<GraphDescriptor> graphs;
  std::vector<ActorDescriptor> actors;
  std::vector<DataflowEdge> edges;
  std::vector<LogicalMemoryRootDescriptor> logicalMemoryRoots;
};

struct FuDescriptor {
  FuId id;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
};

struct ComputeEndpointDescriptor {
  ComputeEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  std::vector<TypeKey> compatibleTypes;
  PortRoleKey role;
  fabric::DataPathKind transportKind = fabric::DataPathKind::Bits;
};

struct ComputeLocalArcDescriptor {
  FuPortRef fuPort;
  ComputeEndpointRef endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ComputeOccurrenceDescriptor {
  ComputeOccurrenceId id;
  ComputeScheduleKind schedule;
  std::vector<FuRef> functionalUnits;
  std::vector<ComputeEndpointDescriptor> endpoints;
  std::vector<ComputeLocalArcDescriptor> localArcs;
};

struct TransportEndpointDescriptor {
  TransportEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  fabric::DataPathKind transportKind = fabric::DataPathKind::Bits;
};

struct TransportResourceDescriptor {
  TransportResourceId id;
  TransportResourceKind kind;
  std::vector<TransportEndpointDescriptor> endpoints;
  std::optional<fabric::BoundaryDirection> boundaryDirection = std::nullopt;
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
  FabricOpId id;
  FuId fu;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
};

struct FuInputValue {
  std::uint32_t index;

  friend bool operator==(FuInputValue lhs, FuInputValue rhs) {
    return lhs.index == rhs.index;
  }
  friend bool operator!=(FuInputValue lhs, FuInputValue rhs) {
    return !(lhs == rhs);
  }
};

struct FabricOpResultValue {
  FabricOpId operation;
  std::uint32_t index;

  friend bool operator==(FabricOpResultValue lhs, FabricOpResultValue rhs) {
    return lhs.operation == rhs.operation && lhs.index == rhs.index;
  }
  friend bool operator!=(FabricOpResultValue lhs, FabricOpResultValue rhs) {
    return !(lhs == rhs);
  }
};

using ConfiguredValue = std::variant<FuInputValue, FabricOpResultValue>;

struct ConfiguredFabricOpDescriptor {
  FabricOpId operation;
  SemanticKey semantics;
  SemanticKey attributes;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
  std::vector<ConfiguredValue> operands;
};

struct ConfiguredInputDescriptor {
  std::uint32_t fuPort;
  PortDescriptor port;
};

struct ConfiguredOutputDescriptor {
  std::uint32_t fuPort;
  PortDescriptor port;
  ConfiguredValue value;
};

struct EncodingDescriptor {
  EncodingId id;
  FuId fu;
  // Derived lossless projection of the selected canonical ConfiguredFunction.
  // The Mapping Artifact persists only the encoding reference and explicit
  // correspondences, never this projection as a second software graph.
  std::vector<ConfiguredInputDescriptor> inputs;
  std::vector<ConfiguredFabricOpDescriptor> operations;
  std::vector<ConfiguredOutputDescriptor> outputs;
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
  std::vector<EncodingDescriptor> encodings;
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
  FabricOpRef fabricOp;
};

struct BoundaryPortCorrespondence {
  ActorPortRef actorPort;
  FuPortRef fuPort;
};

struct MemoryOperationPortRef {
  MemoryOperationPortTemplateRef operation;
  std::uint32_t index;
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
  EdgeRef edge;
  MemoryInternalConnectionRef connection;
};

struct ComputeRealizationDraft {
  ComputeRealizationId id;
  std::vector<ActorRef> actors;
  FuRef fu;
  EncodingRef encoding;
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
  MappingDraftHeader(SchemaVersion schemaVersion, MappingProfile profile,
                     ArtifactIdentity dataflowIdentity,
                     ArtifactIdentity fabricIdentity)
      : schemaVersion(schemaVersion), profile(profile),
        dataflowIdentity(std::move(dataflowIdentity)),
        fabricIdentity(std::move(fabricIdentity)) {}

  SchemaVersion schemaVersion;
  MappingProfile profile;
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
