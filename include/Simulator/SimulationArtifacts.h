#ifndef LOOM_SIMULATOR_SIMULATIONARTIFACTS_H
#define LOOM_SIMULATOR_SIMULATIONARTIFACTS_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace dataflow {
class CanonicalDataflowProgramView;
} // namespace dataflow

// The typed schema-1.0 models of the loom.simulation_workload and
// loom.simulation_runtime_input Artifact families owned by
// docs/spec-simulation-artifacts.md. RootedGraphLaunchRef is the only Dataflow
// root: value, stream, memory, and channel meanings are recovered through
// CanonicalDataflowProgramView and are never copied here. Each family has one
// typed C++ model and one strict canonical serializer/parser; the Common
// finalizer is the sole identity authority. The System root is fail-closed in
// schema 1.0 and is therefore not representable in these models.
namespace loom::sim {

/// The declared Artifact families framed by Common and hashed with SHA-256 v1.
inline constexpr ::loom::ArtifactSchemaDescriptor simulationWorkloadSchema{
    "loom.simulation_workload", ::loom::SchemaVersion{1, 0}};
inline constexpr ::loom::ArtifactSchemaDescriptor simulationRuntimeInputSchema{
    "loom.simulation_runtime_input", ::loom::SchemaVersion{1, 0}};

//===----------------------------------------------------------------------===//
// Canonical semantic values and memory bytes
//===----------------------------------------------------------------------===//

/// The closed semantic state shared by value lanes and memory bytes. Wire
/// discriminants are the zero-based declaration order.
enum class SemanticState : std::uint32_t { Defined = 0, Poison = 1, Undef = 2 };

/// One semantic lane. A Defined lane carries the exact fixed-width
/// software-semantic bits; host integers and doubles are never authorities.
struct SemanticLane {
  SemanticState state = SemanticState::Undef;
  llvm::APInt bits; // meaningful only when state == Defined

  static SemanticLane defined(llvm::APInt value) {
    SemanticLane lane;
    lane.state = SemanticState::Defined;
    lane.bits = std::move(value);
    return lane;
  }
  static SemanticLane poison() {
    return SemanticLane{SemanticState::Poison, llvm::APInt()};
  }
  static SemanticLane undef() {
    return SemanticLane{SemanticState::Undef, llvm::APInt()};
  }
};

/// One value representation shared by fixed workload inputs, runtime value
/// inputs, and runtime stream tokens. The target type is recovered from the
/// workload context and is never serialized: a scalar has one lane per token,
/// a fixed-ranked vector has the row-major product of its dimensions.
struct CanonicalValueSequence {
  std::uint64_t tokenCount = 0;
  std::vector<SemanticLane> lanes; // tokenCount * lanes-per-token entries
};

/// Stream termination within the owning observation horizon. For runtime
/// input the horizon is the complete supplied input: OpenAfterLast means no
/// later token or close exists; it is not a timed-arrival placeholder.
enum class StreamTermination : std::uint32_t {
  ClosedAfterLast = 0,
  OpenAfterLast = 1,
};

/// The shared stream sequence reused by the future execution schema.
struct CanonicalStreamSequence {
  CanonicalValueSequence values;
  StreamTermination termination = StreamTermination::ClosedAfterLast;
};

/// One byte of neutral byte-addressed software storage.
struct SemanticMemoryByte {
  SemanticState state = SemanticState::Undef;
  std::uint8_t value = 0; // meaningful only when state == Defined
};

//===----------------------------------------------------------------------===//
// SpatialSimulationWorkload
//===----------------------------------------------------------------------===//

/// The classification of one graph value input. A Fixed entry holds exactly
/// one semantic token and is part of workload identity; a Runtime entry
/// delegates that token to the exact SimulationRuntimeInput.
struct RuntimeValueInput {};
using SpatialValueInputSource =
    std::variant<CanonicalValueSequence, RuntimeValueInput>;

enum class MemoryObservationForm : std::uint32_t {
  FullState = 0,
  DiffFromRuntimeInput = 1,
};

/// A launch-contextual graph memory result; its complete meaning is the
/// corresponding MemoryExposureRef resolved through the rooted launch.
struct MemoryExposureTarget {
  std::uint32_t memoryResultOrdinal = 0;
};

using SpatialMemoryObservableTarget =
    std::variant<dataflow::LogicalMemoryRootOrViewRef, MemoryExposureTarget>;

struct SpatialMemoryObservable {
  SpatialMemoryObservableTarget target;
  MemoryObservationForm form = MemoryObservationForm::FullState;
};

/// The canonical observable sets. Each collection is sorted by its canonical
/// typed key and contains no duplicates; owner-relative ordinals resolve
// through the rooted launch.
struct SpatialObservableContract {
  std::vector<std::uint32_t> valueResults = {};
  std::vector<std::uint32_t> streamOutputs = {};
  std::vector<SpatialMemoryObservable> memories = {};
};

/// The schema-1.0 spatial workload: one rooted graph launch, the dense
/// logical coordinates of the root thread point, the total value-input
/// classification, and the observable contract. Stream inputs and imported
/// memory roots are always runtime-plane and appear only in the exact
/// SimulationRuntimeInput. The System root is fail-closed and unrepresentable.
struct SpatialSimulationWorkload {
  dataflow::RootedGraphLaunchRef launchRef;
  std::vector<std::uint64_t> denseCoordinates = {};
  // Exactly one entry per graph value-input ordinal, in ordinal order.
  std::vector<SpatialValueInputSource> valueInputPlan = {};
  SpatialObservableContract observableContract = {};

  /// Graph completion is mandatory for every workload and is mechanically
  /// derived from the rooted launch; it is never an optional observable
  /// selector and is never serialized.
  dataflow::GraphLaunchDoneTransferRef completion() const {
    return dataflow::GraphLaunchDoneTransferRef{launchRef};
  }
};

//===----------------------------------------------------------------------===//
// SpatialSimulationRuntimeInput
//===----------------------------------------------------------------------===//

struct RuntimeValueEntry {
  std::uint32_t valueInputOrdinal = 0;
  CanonicalValueSequence value; // exactly one token
};

/// Neutral byte-addressed software storage. The exact Canonical Dataflow
/// type, DataLayout, and root/view relations alone interpret typed accesses.
struct RuntimeMemoryObject {
  std::vector<SemanticMemoryByte> initialBytes; // nonempty
};

struct RuntimeMemoryRootBinding {
  std::uint64_t objectOrdinal = 0; // derived, never author-selected
  std::uint64_t byteOffset = 0;
};

struct MemoryRootBindingEntry {
  dataflow::LogicalMemoryRootRef root;
  RuntimeMemoryRootBinding binding;
};

/// The canonical schema-1.0 spatial runtime input. `runtimeValues` is sorted
/// by value-input ordinal and exactly complements the workload's Runtime
/// classifications; `runtimeStreams` is dense over graph stream-input
/// ordinals; `memoryObjects` is in canonical object-ordinal order;
/// `memoryRootBindings` is sorted by the typed root key and total over the
/// imported logical-memory roots reachable from the workload's launch.
struct SpatialSimulationRuntimeInput {
  ::loom::ArtifactIdentity workloadIdentity;
  std::vector<RuntimeValueEntry> runtimeValues = {};
  std::vector<CanonicalStreamSequence> runtimeStreams = {};
  std::vector<RuntimeMemoryObject> memoryObjects = {};
  std::vector<MemoryRootBindingEntry> memoryRootBindings = {};
};

/// The author-facing draft. Aliasing is expressed only by sharing one draft
/// object slot; the persistent zero-based object ordinal is derived at
/// finalization from each object's sorted nonempty (root, byte offset)
/// binding key, so no author-selected ID or alias graph is ever serialized.
struct RuntimeMemoryBindingDraft {
  dataflow::LogicalMemoryRootRef root;
  std::uint64_t authorObject = 0; // draft-local slot into memoryObjects
  std::uint64_t byteOffset = 0;
};

struct SpatialSimulationRuntimeInputDraft {
  ::loom::ArtifactIdentity workloadIdentity;
  std::vector<RuntimeValueEntry> runtimeValues = {};
  std::vector<CanonicalStreamSequence> runtimeStreams = {};
  std::vector<RuntimeMemoryObject> memoryObjects = {};
  std::vector<RuntimeMemoryBindingDraft> memoryRootBindings = {};
};

//===----------------------------------------------------------------------===//
// Canonical artifacts
//===----------------------------------------------------------------------===//

class CanonicalSimulationWorkload {
public:
  CanonicalSimulationWorkload(const CanonicalSimulationWorkload &) = delete;
  CanonicalSimulationWorkload(CanonicalSimulationWorkload &&) = default;
  CanonicalSimulationWorkload &
  operator=(const CanonicalSimulationWorkload &) = delete;
  CanonicalSimulationWorkload &
  operator=(CanonicalSimulationWorkload &&) = default;

  const ::loom::ArtifactIdentity &identity() const { return identity_; }
  const SpatialSimulationWorkload &model() const { return model_; }
  const ::loom::CanonicalSemanticBytes &canonicalBytes() const {
    return bytes_;
  }
  dataflow::GraphLaunchDoneTransferRef completion() const {
    return model_.completion();
  }

private:
  CanonicalSimulationWorkload(::loom::ArtifactIdentity identity,
                              SpatialSimulationWorkload model,
                              ::loom::CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}

  ::loom::ArtifactIdentity identity_;
  SpatialSimulationWorkload model_;
  ::loom::CanonicalSemanticBytes bytes_;

  friend llvm::Expected<CanonicalSimulationWorkload>
  finalizeSimulationWorkload(const SpatialSimulationWorkload &,
                             const dataflow::CanonicalDataflowProgramView &);
  friend llvm::Expected<CanonicalSimulationWorkload>
  importSimulationWorkload(llvm::ArrayRef<std::uint8_t>,
                           const dataflow::CanonicalDataflowProgramView &,
                           const ::loom::ArtifactIdentity &);
};

class CanonicalSimulationRuntimeInput {
public:
  CanonicalSimulationRuntimeInput(const CanonicalSimulationRuntimeInput &) =
      delete;
  CanonicalSimulationRuntimeInput(CanonicalSimulationRuntimeInput &&) = default;
  CanonicalSimulationRuntimeInput &
  operator=(const CanonicalSimulationRuntimeInput &) = delete;
  CanonicalSimulationRuntimeInput &
  operator=(CanonicalSimulationRuntimeInput &&) = default;

  const ::loom::ArtifactIdentity &identity() const { return identity_; }
  const SpatialSimulationRuntimeInput &model() const { return model_; }
  const ::loom::CanonicalSemanticBytes &canonicalBytes() const {
    return bytes_;
  }

private:
  CanonicalSimulationRuntimeInput(::loom::ArtifactIdentity identity,
                                  SpatialSimulationRuntimeInput model,
                                  ::loom::CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}

  ::loom::ArtifactIdentity identity_;
  SpatialSimulationRuntimeInput model_;
  ::loom::CanonicalSemanticBytes bytes_;

  friend llvm::Expected<CanonicalSimulationRuntimeInput>
  finalizeSimulationRuntimeInput(
      const SpatialSimulationRuntimeInputDraft &,
      const CanonicalSimulationWorkload &,
      const dataflow::CanonicalDataflowProgramView &);
  friend llvm::Expected<CanonicalSimulationRuntimeInput>
  importSimulationRuntimeInput(llvm::ArrayRef<std::uint8_t>,
                               const CanonicalSimulationWorkload &,
                               const dataflow::CanonicalDataflowProgramView &,
                               const ::loom::ArtifactIdentity &);
};

/// Failure-atomic finalization. Validates the complete spatial workload
/// against the exact Dataflow owner view -- rooted-launch ownership, dense
/// coordinate rank and static bounds, total Fixed/Runtime classification with
/// exact lane states, and the sorted observable contract -- then publishes
/// the canonical bytes framed by the Common finalizer.
llvm::Expected<CanonicalSimulationWorkload> finalizeSimulationWorkload(
    const SpatialSimulationWorkload &workload,
    const dataflow::CanonicalDataflowProgramView &program);

/// Strict canonical parse. Rejects trailing bytes, malformed or truncated
/// lengths, overflow, unknown variants, unsorted or duplicate keys, the
/// fail-closed System root, and every semantic violation the finalizer
/// rejects, then requires the recomputed identity to equal
/// `expectedIdentity`.
llvm::Expected<CanonicalSimulationWorkload>
importSimulationWorkload(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                         const dataflow::CanonicalDataflowProgramView &program,
                         const ::loom::ArtifactIdentity &expectedIdentity);

/// Failure-atomic finalization of the exact runtime input of one finalized
/// workload. Validates the total runtime-value complement, the total stream
/// table with its horizon state, and the neutral memory objects; derives the
/// canonical object ordinals from the sorted binding keys; and rejects
/// missing, duplicate, unrelated, unreferenced, empty, out-of-range, or
/// noncanonical objects and bindings. Overlap between root ranges bound to
/// one object is legal aliasing.
llvm::Expected<CanonicalSimulationRuntimeInput> finalizeSimulationRuntimeInput(
    const SpatialSimulationRuntimeInputDraft &draft,
    const CanonicalSimulationWorkload &workload,
    const dataflow::CanonicalDataflowProgramView &program);

/// Strict canonical parse under the same rules as the workload importer. The
/// serialized object ordinals must equal the canonical order derived from the
/// sorted binding keys.
llvm::Expected<CanonicalSimulationRuntimeInput> importSimulationRuntimeInput(
    llvm::ArrayRef<std::uint8_t> canonicalBytes,
    const CanonicalSimulationWorkload &workload,
    const dataflow::CanonicalDataflowProgramView &program,
    const ::loom::ArtifactIdentity &expectedIdentity);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONARTIFACTS_H
