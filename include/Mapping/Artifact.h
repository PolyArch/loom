#ifndef LOOM_MAPPING_ARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_H

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

struct SchemaVersion {
  std::uint32_t major = 0;
  std::uint32_t minor = 0;

  friend bool operator==(SchemaVersion lhs, SchemaVersion rhs) {
    return lhs.major == rhs.major && lhs.minor == rhs.minor;
  }
  friend bool operator!=(SchemaVersion lhs, SchemaVersion rhs) {
    return !(lhs == rhs);
  }
};

enum class MappingProfile { TechMapping, PhysicalMapping };
enum class PortDirection { Input, Output };
enum class PortKind { Value, Stream, Memory };

class ArtifactIdentity {
public:
  ArtifactIdentity() = default;
  ArtifactIdentity(std::initializer_list<std::uint8_t> bytes) : bytes_(bytes) {}
  explicit ArtifactIdentity(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  bool empty() const { return bytes_.empty(); }

  friend bool operator==(const ArtifactIdentity &lhs,
                         const ArtifactIdentity &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const ArtifactIdentity &lhs,
                         const ArtifactIdentity &rhs) {
    return !(lhs == rhs);
  }

private:
  std::vector<std::uint8_t> bytes_;
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

  friend bool operator==(const PortDescriptor &lhs, const PortDescriptor &rhs) {
    return lhs.kind == rhs.kind && lhs.type == rhs.type;
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
struct FuIdTag;
struct FabricOpIdTag;
struct EncodingIdTag;
struct ComputeRealizationIdTag;

using GraphId = TypedEntityId<GraphIdTag>;
using ActorId = TypedEntityId<ActorIdTag>;
using FuId = TypedEntityId<FuIdTag>;
using FabricOpId = TypedEntityId<FabricOpIdTag>;
using EncodingId = TypedEntityId<EncodingIdTag>;
using ComputeRealizationId = TypedEntityId<ComputeRealizationIdTag>;

template <typename EntityId> struct EntityReference {
  ArtifactIdentity artifact;
  EntityId entity;
};

using GraphRef = EntityReference<GraphId>;
using ActorRef = EntityReference<ActorId>;
using FuRef = EntityReference<FuId>;
using FabricOpRef = EntityReference<FabricOpId>;
using EncodingRef = EntityReference<EncodingId>;

struct GraphDescriptor {
  GraphId id;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
};

struct ActorDescriptor {
  ActorId id;
  GraphId graph;
  SemanticKey operation;
  SemanticKey attributes;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
};

struct GraphPort {
  GraphId graph;
  PortDirection direction;
  std::uint32_t index;
};

struct ActorPort {
  ActorId actor;
  PortDirection direction;
  std::uint32_t index;
};

using DataflowEndpoint = std::variant<GraphPort, ActorPort>;

struct DataflowEdge {
  DataflowEndpoint source;
  DataflowEndpoint target;
};

struct DataflowProgramView {
  ArtifactIdentity identity;
  std::vector<GraphDescriptor> graphs;
  std::vector<ActorDescriptor> actors;
  std::vector<DataflowEdge> edges;
};

struct FuDescriptor {
  FuId id;
  std::vector<PortDescriptor> inputPorts;
  std::vector<PortDescriptor> outputPorts;
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

struct FabricHardwareView {
  ArtifactIdentity identity;
  std::vector<FuDescriptor> functionalUnits;
  std::vector<FabricOpDescriptor> operations;
  std::vector<EncodingDescriptor> encodings;
};

struct ActorPortRef {
  ActorRef actor;
  PortDirection direction;
  std::uint32_t index;
};

struct FuPortRef {
  FuRef fu;
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

struct ComputeRealizationDraft {
  ComputeRealizationId id;
  std::vector<ActorRef> actors;
  FuRef fu;
  EncodingRef encoding;
  std::vector<ActorToFabricOp> actorToOps;
  std::vector<BoundaryPortCorrespondence> boundaryPorts;
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
};

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_H
