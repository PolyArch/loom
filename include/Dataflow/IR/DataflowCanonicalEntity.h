#ifndef LOOM_DATAFLOW_IR_DATAFLOW_CANONICAL_ENTITY_H
#define LOOM_DATAFLOW_IR_DATAFLOW_CANONICAL_ENTITY_H

#include "Common/Artifact.h"

#include <cstdint>

// The Canonical Dataflow Program is the single semantic owner of artifact-local
// entity identity. This header is the MLIR-free vocabulary that downstream
// consumers (Mapping, simulators, Evaluation, visualization) import and use as
// the one owner of these types; they do not declare another tag, ID
// implementation, or catalog, and none may assign or reinterpret a Dataflow
// entity ID. The owner finalizer, canonical bytes, and importer consume this
// vocabulary without duplicating it.
namespace dataflow {

/// The finalizer-owned attribute that carries a derived artifact-local entity
/// ID. It is not actor semantics and therefore never enters an operation's
/// identity-critical schema projection.
inline constexpr char kEntityIdAttrName[] = "dataflow.entity_id";

/// The closed first-schema catalog of independently referenceable entities.
/// A future referenceable object requires an explicit catalog change here; a
/// consumer cannot mint an ID for convenience.
enum class CanonicalDataflowEntityKind : std::uint32_t {
  Graph = 0,
  Actor = 1,
  RootThreadLaunch = 2,
  StaticGraphLaunch = 3,
  LogicalMemoryRoot = 4,
};

/// One artifact-local entity ID, typed by its kind. All kinds share a single
/// artifact-global unsigned 64-bit namespace: zero is an ordinary valid value
/// and there is no reserved sentinel. The kind is a compile-time tag so that a
/// reference of one kind is a distinct C++ type from a reference of another.
template <CanonicalDataflowEntityKind Kind> class CanonicalDataflowEntityId {
public:
  explicit constexpr CanonicalDataflowEntityId(std::uint64_t value)
      : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(CanonicalDataflowEntityId lhs,
                                   CanonicalDataflowEntityId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(CanonicalDataflowEntityId lhs,
                                   CanonicalDataflowEntityId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

using GraphId = CanonicalDataflowEntityId<CanonicalDataflowEntityKind::Graph>;
using ActorId = CanonicalDataflowEntityId<CanonicalDataflowEntityKind::Actor>;
using RootThreadLaunchId =
    CanonicalDataflowEntityId<CanonicalDataflowEntityKind::RootThreadLaunch>;
using StaticGraphLaunchId =
    CanonicalDataflowEntityId<CanonicalDataflowEntityKind::StaticGraphLaunch>;
using LogicalMemoryRootId =
    CanonicalDataflowEntityId<CanonicalDataflowEntityKind::LogicalMemoryRoot>;

/// A complete persistent reference is the exact Canonical Dataflow
/// ArtifactIdentity paired with a typed EntityId. Resolution requires the exact
/// artifact identity so foreign-artifact and wrong-kind rejection are real.
using GraphRef = ::loom::ArtifactReference<GraphId>;
using ActorRef = ::loom::ArtifactReference<ActorId>;
using RootThreadLaunchRef = ::loom::ArtifactReference<RootThreadLaunchId>;
using StaticGraphLaunchRef = ::loom::ArtifactReference<StaticGraphLaunchId>;
using LogicalMemoryRootRef = ::loom::ArtifactReference<LogicalMemoryRootId>;

/// The declared Artifact family framed by Common and hashed with SHA-256 v1.
inline constexpr ::loom::ArtifactSchemaDescriptor canonicalDataflowSchema{
    "loom.canonical_dataflow", ::loom::SchemaVersion{3, 0}};

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOW_CANONICAL_ENTITY_H
