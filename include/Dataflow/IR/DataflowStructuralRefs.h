#ifndef LOOM_DATAFLOW_IR_DATAFLOW_STRUCTURAL_REFS_H
#define LOOM_DATAFLOW_IR_DATAFLOW_STRUCTURAL_REFS_H

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowStructuralRefUnions.def"

#include <cstdint>
#include <variant>
#include <vector>

// The Dataflow-owned closed structural-reference catalog. Every object below
// the five entity kinds is an owner-relative structural reference composed from
// the typed entity references plus role-specific canonical ordinals. None of
// these receive an EntityId, and none may be replaced by a symbol path,
// operation position, generic field path, or native dense index. Each catalog
// alternative is a distinct typed struct so no impossible value is
// representable; the exact variants and ordinal ownership are owned by
// docs/spec-compiler-part-3-dfg.md `Closed Structural Reference Catalog`.
namespace dataflow {

namespace detail {

template <typename, typename... Alternatives> struct ClosedUnionVariant {
  using type = std::variant<Alternatives...>;
};

} // namespace detail

#define LOOM_DATAFLOW_CLOSED_UNION_ALTERNATIVE(Union, WireTag, Type) , Type
#define LOOM_DATAFLOW_DECLARE_CLOSED_UNION(Name, Alternatives)                 \
  using Name = typename detail::ClosedUnionVariant<void Alternatives(          \
      LOOM_DATAFLOW_CLOSED_UNION_ALTERNATIVE, Name)>::type

/// Owner-relative structural ordinals use the persistent unsigned 64-bit
/// domain. Native containers may use narrower indices only as disposable
/// import caches after checked conversion.
using StructuralOrdinal = std::uint64_t;

//===----------------------------------------------------------------------===//
// Rooted graph launch
//===----------------------------------------------------------------------===//

/// A static graph launch interpreted in the context of one root thread launch.
struct RootedGraphLaunchRef {
  RootThreadLaunchRef rootThreadLaunch;
  StaticGraphLaunchRef staticGraphLaunch;

  friend bool operator==(const RootedGraphLaunchRef &lhs,
                         const RootedGraphLaunchRef &rhs) {
    return lhs.rootThreadLaunch == rhs.rootThreadLaunch &&
           lhs.staticGraphLaunch == rhs.staticGraphLaunch;
  }
  friend bool operator!=(const RootedGraphLaunchRef &lhs,
                         const RootedGraphLaunchRef &rhs) {
    return !(lhs == rhs);
  }
};

//===----------------------------------------------------------------------===//
// Graph-local and actor token endpoints (token plane only)
//===----------------------------------------------------------------------===//

struct GraphStartTokenRef {
  GraphRef graph;
  friend bool operator==(const GraphStartTokenRef &a,
                         const GraphStartTokenRef &b) {
    return a.graph == b.graph;
  }
  friend bool operator!=(const GraphStartTokenRef &a,
                         const GraphStartTokenRef &b) {
    return !(a == b);
  }
};
struct GraphValueInputTokenRef {
  GraphRef graph;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphValueInputTokenRef &a,
                         const GraphValueInputTokenRef &b) {
    return a.graph == b.graph && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphValueInputTokenRef &a,
                         const GraphValueInputTokenRef &b) {
    return !(a == b);
  }
};
struct GraphStreamInputTokenRef {
  GraphRef graph;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphStreamInputTokenRef &a,
                         const GraphStreamInputTokenRef &b) {
    return a.graph == b.graph && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphStreamInputTokenRef &a,
                         const GraphStreamInputTokenRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    GraphIngressTokenRef, LOOM_DATAFLOW_GRAPH_INGRESS_TOKEN_REF_ALTERNATIVES);

struct GraphValueOutputTokenRef {
  GraphRef graph;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphValueOutputTokenRef &a,
                         const GraphValueOutputTokenRef &b) {
    return a.graph == b.graph && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphValueOutputTokenRef &a,
                         const GraphValueOutputTokenRef &b) {
    return !(a == b);
  }
};
struct GraphStreamOutputTokenRef {
  GraphRef graph;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphStreamOutputTokenRef &a,
                         const GraphStreamOutputTokenRef &b) {
    return a.graph == b.graph && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphStreamOutputTokenRef &a,
                         const GraphStreamOutputTokenRef &b) {
    return !(a == b);
  }
};
struct GraphCompletionFrontierTokenRef {
  GraphRef graph;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphCompletionFrontierTokenRef &a,
                         const GraphCompletionFrontierTokenRef &b) {
    return a.graph == b.graph && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphCompletionFrontierTokenRef &a,
                         const GraphCompletionFrontierTokenRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    GraphEgressTokenRef, LOOM_DATAFLOW_GRAPH_EGRESS_TOKEN_REF_ALTERNATIVES);

struct ActorTokenResultRef {
  ActorRef actor;
  StructuralOrdinal ordinal;
  friend bool operator==(const ActorTokenResultRef &a,
                         const ActorTokenResultRef &b) {
    return a.actor == b.actor && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const ActorTokenResultRef &a,
                         const ActorTokenResultRef &b) {
    return !(a == b);
  }
};
struct ActorTokenOperandRef {
  ActorRef actor;
  StructuralOrdinal ordinal;
  friend bool operator==(const ActorTokenOperandRef &a,
                         const ActorTokenOperandRef &b) {
    return a.actor == b.actor && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const ActorTokenOperandRef &a,
                         const ActorTokenOperandRef &b) {
    return !(a == b);
  }
};

LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    CanonicalGraphProducerEndpointRef,
    LOOM_DATAFLOW_GRAPH_PRODUCER_ENDPOINT_REF_ALTERNATIVES);
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    CanonicalGraphConsumerEndpointRef,
    LOOM_DATAFLOW_GRAPH_CONSUMER_ENDPOINT_REF_ALTERNATIVES);

//===----------------------------------------------------------------------===//
// One-message boundary transfers (thread/graph ABI)
//===----------------------------------------------------------------------===//

struct RootThreadStartTransferRef {
  RootThreadLaunchRef launch;
  friend bool operator==(const RootThreadStartTransferRef &a,
                         const RootThreadStartTransferRef &b) {
    return a.launch == b.launch;
  }
  friend bool operator!=(const RootThreadStartTransferRef &a,
                         const RootThreadStartTransferRef &b) {
    return !(a == b);
  }
};
struct RootThreadValueInputTransferRef {
  RootThreadLaunchRef launch;
  StructuralOrdinal ordinal; // value body-operand ordinal
  friend bool operator==(const RootThreadValueInputTransferRef &a,
                         const RootThreadValueInputTransferRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const RootThreadValueInputTransferRef &a,
                         const RootThreadValueInputTransferRef &b) {
    return !(a == b);
  }
};
struct RootThreadCompletionTransferRef {
  RootThreadLaunchRef launch;
  friend bool operator==(const RootThreadCompletionTransferRef &a,
                         const RootThreadCompletionTransferRef &b) {
    return a.launch == b.launch;
  }
  friend bool operator!=(const RootThreadCompletionTransferRef &a,
                         const RootThreadCompletionTransferRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    RootThreadBoundaryTransferRef,
    LOOM_DATAFLOW_ROOT_THREAD_TRANSFER_REF_ALTERNATIVES);

struct GraphLaunchStartTransferRef {
  RootedGraphLaunchRef launch;
  friend bool operator==(const GraphLaunchStartTransferRef &a,
                         const GraphLaunchStartTransferRef &b) {
    return a.launch == b.launch;
  }
  friend bool operator!=(const GraphLaunchStartTransferRef &a,
                         const GraphLaunchStartTransferRef &b) {
    return !(a == b);
  }
};
struct GraphLaunchValueInputTransferRef {
  RootedGraphLaunchRef launch;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphLaunchValueInputTransferRef &a,
                         const GraphLaunchValueInputTransferRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphLaunchValueInputTransferRef &a,
                         const GraphLaunchValueInputTransferRef &b) {
    return !(a == b);
  }
};
struct GraphLaunchValueResultTransferRef {
  RootedGraphLaunchRef launch;
  StructuralOrdinal ordinal;
  friend bool operator==(const GraphLaunchValueResultTransferRef &a,
                         const GraphLaunchValueResultTransferRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphLaunchValueResultTransferRef &a,
                         const GraphLaunchValueResultTransferRef &b) {
    return !(a == b);
  }
};
struct GraphLaunchDoneTransferRef {
  RootedGraphLaunchRef launch;
  friend bool operator==(const GraphLaunchDoneTransferRef &a,
                         const GraphLaunchDoneTransferRef &b) {
    return a.launch == b.launch;
  }
  friend bool operator!=(const GraphLaunchDoneTransferRef &a,
                         const GraphLaunchDoneTransferRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    GraphLaunchBoundaryTransferRef,
    LOOM_DATAFLOW_GRAPH_LAUNCH_TRANSFER_REF_ALTERNATIVES);

//===----------------------------------------------------------------------===//
// Channel endpoints
//===----------------------------------------------------------------------===//

struct ThreadChannelSendSiteRef {
  RootThreadLaunchRef launch;
  StructuralOrdinal ordinal; // canonical send-site ordinal
  friend bool operator==(const ThreadChannelSendSiteRef &a,
                         const ThreadChannelSendSiteRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const ThreadChannelSendSiteRef &a,
                         const ThreadChannelSendSiteRef &b) {
    return !(a == b);
  }
};
struct ThreadChannelReceiveSiteRef {
  RootThreadLaunchRef launch;
  StructuralOrdinal ordinal; // canonical receive-site ordinal
  friend bool operator==(const ThreadChannelReceiveSiteRef &a,
                         const ThreadChannelReceiveSiteRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const ThreadChannelReceiveSiteRef &a,
                         const ThreadChannelReceiveSiteRef &b) {
    return !(a == b);
  }
};
struct GraphStreamOutputProducerRef {
  RootedGraphLaunchRef launch;
  StructuralOrdinal ordinal; // stream-output ordinal
  friend bool operator==(const GraphStreamOutputProducerRef &a,
                         const GraphStreamOutputProducerRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphStreamOutputProducerRef &a,
                         const GraphStreamOutputProducerRef &b) {
    return !(a == b);
  }
};
struct GraphStreamInputConsumerRef {
  RootedGraphLaunchRef launch;
  StructuralOrdinal ordinal; // stream-input ordinal
  friend bool operator==(const GraphStreamInputConsumerRef &a,
                         const GraphStreamInputConsumerRef &b) {
    return a.launch == b.launch && a.ordinal == b.ordinal;
  }
  friend bool operator!=(const GraphStreamInputConsumerRef &a,
                         const GraphStreamInputConsumerRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    ChannelProducerRef, LOOM_DATAFLOW_CHANNEL_PRODUCER_REF_ALTERNATIVES);
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    ChannelConsumerRef, LOOM_DATAFLOW_CHANNEL_CONSUMER_REF_ALTERNATIVES);

//===----------------------------------------------------------------------===//
// Transfer terminals
//===----------------------------------------------------------------------===//

struct RootThreadBoundarySourceRef {
  RootThreadBoundaryTransferRef transfer;
  friend bool operator==(const RootThreadBoundarySourceRef &a,
                         const RootThreadBoundarySourceRef &b) {
    return a.transfer == b.transfer;
  }
  friend bool operator!=(const RootThreadBoundarySourceRef &a,
                         const RootThreadBoundarySourceRef &b) {
    return !(a == b);
  }
};
struct GraphLaunchBoundarySourceRef {
  GraphLaunchBoundaryTransferRef transfer;
  friend bool operator==(const GraphLaunchBoundarySourceRef &a,
                         const GraphLaunchBoundarySourceRef &b) {
    return a.transfer == b.transfer;
  }
  friend bool operator!=(const GraphLaunchBoundarySourceRef &a,
                         const GraphLaunchBoundarySourceRef &b) {
    return !(a == b);
  }
};
struct ChannelProducerTerminalRef {
  ChannelProducerRef producer;
  friend bool operator==(const ChannelProducerTerminalRef &a,
                         const ChannelProducerTerminalRef &b) {
    return a.producer == b.producer;
  }
  friend bool operator!=(const ChannelProducerTerminalRef &a,
                         const ChannelProducerTerminalRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    CanonicalProducerTerminalRef,
    LOOM_DATAFLOW_PRODUCER_TERMINAL_REF_ALTERNATIVES);

struct RootThreadBoundarySinkRef {
  RootThreadBoundaryTransferRef transfer;
  friend bool operator==(const RootThreadBoundarySinkRef &a,
                         const RootThreadBoundarySinkRef &b) {
    return a.transfer == b.transfer;
  }
  friend bool operator!=(const RootThreadBoundarySinkRef &a,
                         const RootThreadBoundarySinkRef &b) {
    return !(a == b);
  }
};
struct GraphLaunchBoundarySinkRef {
  GraphLaunchBoundaryTransferRef transfer;
  friend bool operator==(const GraphLaunchBoundarySinkRef &a,
                         const GraphLaunchBoundarySinkRef &b) {
    return a.transfer == b.transfer;
  }
  friend bool operator!=(const GraphLaunchBoundarySinkRef &a,
                         const GraphLaunchBoundarySinkRef &b) {
    return !(a == b);
  }
};
struct ChannelConsumerTerminalRef {
  ChannelConsumerRef consumer;
  friend bool operator==(const ChannelConsumerTerminalRef &a,
                         const ChannelConsumerTerminalRef &b) {
    return a.consumer == b.consumer;
  }
  friend bool operator!=(const ChannelConsumerTerminalRef &a,
                         const ChannelConsumerTerminalRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    CanonicalSinkTerminalRef, LOOM_DATAFLOW_SINK_TERMINAL_REF_ALTERNATIVES);

//===----------------------------------------------------------------------===//
// Memory-plane references
//===----------------------------------------------------------------------===//

struct LogicalMemoryViewRef {
  LogicalMemoryRootRef root;
  StructuralOrdinal viewOrdinal; // canonical root-local view ordinal
  friend bool operator==(const LogicalMemoryViewRef &a,
                         const LogicalMemoryViewRef &b) {
    return a.root == b.root && a.viewOrdinal == b.viewOrdinal;
  }
  friend bool operator!=(const LogicalMemoryViewRef &a,
                         const LogicalMemoryViewRef &b) {
    return !(a == b);
  }
};

LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    LogicalMemoryRootOrViewRef,
    LOOM_DATAFLOW_MEMORY_ROOT_OR_VIEW_REF_ALTERNATIVES);

struct ContextualActorRef {
  RootedGraphLaunchRef launch;
  ActorRef actor;
  friend bool operator==(const ContextualActorRef &a,
                         const ContextualActorRef &b) {
    return a.launch == b.launch && a.actor == b.actor;
  }
  friend bool operator!=(const ContextualActorRef &a,
                         const ContextualActorRef &b) {
    return !(a == b);
  }
};

struct MemoryExposureRef {
  RootedGraphLaunchRef launch;
  StructuralOrdinal memoryResultOrdinal;
  friend bool operator==(const MemoryExposureRef &a,
                         const MemoryExposureRef &b) {
    return a.launch == b.launch &&
           a.memoryResultOrdinal == b.memoryResultOrdinal;
  }
  friend bool operator!=(const MemoryExposureRef &a,
                         const MemoryExposureRef &b) {
    return !(a == b);
  }
};

struct FenceActorFamilyRef {
  ActorRef actor; // validated as dataflow.fence
  friend bool operator==(const FenceActorFamilyRef &a,
                         const FenceActorFamilyRef &b) {
    return a.actor == b.actor;
  }
  friend bool operator!=(const FenceActorFamilyRef &a,
                         const FenceActorFamilyRef &b) {
    return !(a == b);
  }
};

//===----------------------------------------------------------------------===//
// Service members
//===----------------------------------------------------------------------===//

/// The singleton member of one transfer obligation, including multicast.
struct MessageTransferMemberRef {
  friend bool operator==(const MessageTransferMemberRef &,
                         const MessageTransferMemberRef &) {
    return true;
  }
  friend bool operator!=(const MessageTransferMemberRef &,
                         const MessageTransferMemberRef &) {
    return false;
  }
};
struct AddressedMemoryActorMemberRef {
  ContextualActorRef actor;
  friend bool operator==(const AddressedMemoryActorMemberRef &a,
                         const AddressedMemoryActorMemberRef &b) {
    return a.actor == b.actor;
  }
  friend bool operator!=(const AddressedMemoryActorMemberRef &a,
                         const AddressedMemoryActorMemberRef &b) {
    return !(a == b);
  }
};
struct FenceActorMemberRef {
  ContextualActorRef actor;
  friend bool operator==(const FenceActorMemberRef &a,
                         const FenceActorMemberRef &b) {
    return a.actor == b.actor;
  }
  friend bool operator!=(const FenceActorMemberRef &a,
                         const FenceActorMemberRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    ServiceMemberRef, LOOM_DATAFLOW_SERVICE_MEMBER_REF_ALTERNATIVES);

//===----------------------------------------------------------------------===//
// System-visible static transfer events (no EntityId)
//===----------------------------------------------------------------------===//

struct ProducedTransferEventRef {
  CanonicalProducerTerminalRef terminal;
  friend bool operator==(const ProducedTransferEventRef &a,
                         const ProducedTransferEventRef &b) {
    return a.terminal == b.terminal;
  }
  friend bool operator!=(const ProducedTransferEventRef &a,
                         const ProducedTransferEventRef &b) {
    return !(a == b);
  }
};
struct ConsumedTransferEventRef {
  CanonicalSinkTerminalRef terminal;
  friend bool operator==(const ConsumedTransferEventRef &a,
                         const ConsumedTransferEventRef &b) {
    return a.terminal == b.terminal;
  }
  friend bool operator!=(const ConsumedTransferEventRef &a,
                         const ConsumedTransferEventRef &b) {
    return !(a == b);
  }
};
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    StaticTransferEventRef,
    LOOM_DATAFLOW_STATIC_TRANSFER_EVENT_REF_ALTERNATIVES);

struct ContextualActorTransitionEventRef {
  ContextualActorRef actor;
  StructuralOrdinal transitionCaseOrdinal = 0;
  friend bool operator==(const ContextualActorTransitionEventRef &a,
                         const ContextualActorTransitionEventRef &b) {
    return a.actor == b.actor &&
           a.transitionCaseOrdinal == b.transitionCaseOrdinal;
  }
  friend bool operator!=(const ContextualActorTransitionEventRef &a,
                         const ContextualActorTransitionEventRef &b) {
    return !(a == b);
  }
};

/// One Dataflow-owned system resource-time anchor. It has no independently
/// assigned entity identifier or persisted logical projection.
LOOM_DATAFLOW_DECLARE_CLOSED_UNION(EventFamilyKey,
                                   LOOM_DATAFLOW_EVENT_FAMILY_KEY_ALTERNATIVES);

struct CoordinateSlot {
  StructuralOrdinal ordinal = 0;
  friend bool operator==(CoordinateSlot lhs, CoordinateSlot rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
  friend bool operator!=(CoordinateSlot lhs, CoordinateSlot rhs) {
    return !(lhs == rhs);
  }
};

struct LaunchParameterSlot {
  StructuralOrdinal ordinal = 0;
  friend bool operator==(LaunchParameterSlot lhs, LaunchParameterSlot rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
  friend bool operator!=(LaunchParameterSlot lhs, LaunchParameterSlot rhs) {
    return !(lhs == rhs);
  }
};

LOOM_DATAFLOW_DECLARE_CLOSED_UNION(
    EventLogicalInputSlot, LOOM_DATAFLOW_EVENT_LOGICAL_INPUT_SLOT_ALTERNATIVES);
using EventLogicalProjection = std::vector<EventLogicalInputSlot>;

#undef LOOM_DATAFLOW_DECLARE_CLOSED_UNION
#undef LOOM_DATAFLOW_CLOSED_UNION_ALTERNATIVE
#undef LOOM_DATAFLOW_CLOSED_UNIONS
#undef LOOM_DATAFLOW_STRUCTURAL_REFERENCE_UNIONS
#undef LOOM_DATAFLOW_EVENT_LOGICAL_INPUT_SLOT_ALTERNATIVES
#undef LOOM_DATAFLOW_STATIC_TRANSFER_EVENT_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_SERVICE_MEMBER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_MEMORY_ROOT_OR_VIEW_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_SINK_TERMINAL_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_PRODUCER_TERMINAL_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_CHANNEL_CONSUMER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_CHANNEL_PRODUCER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_LAUNCH_TRANSFER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_ROOT_THREAD_TRANSFER_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_CONSUMER_ENDPOINT_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_PRODUCER_ENDPOINT_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_EGRESS_TOKEN_REF_ALTERNATIVES
#undef LOOM_DATAFLOW_GRAPH_INGRESS_TOKEN_REF_ALTERNATIVES

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOW_STRUCTURAL_REFS_H
