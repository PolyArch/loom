#ifndef LOOM_SIMULATOR_SIMULATIONARTIFACTS_H
#define LOOM_SIMULATOR_SIMULATIONARTIFACTS_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Deployment/Deployment.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

// The typed schema models of the loom.simulation_workload and
// loom.simulation_runtime_input Artifact families owned by
// docs/spec-simulation-artifacts.md. Each root carries one owner reference;
// types and structural meanings are recovered from that exact owner and are
// never copied here. Each family has one closed typed C++ model and one strict
// canonical serializer/parser; the Common finalizer is the sole identity
// authority.
namespace loom::sim {

/// The declared Artifact families framed by Common and hashed with SHA-256 v1.
inline constexpr ::loom::ArtifactSchemaDescriptor simulationWorkloadSchema{
    "loom.simulation_workload", ::loom::SchemaVersion{1, 1}};
inline constexpr ::loom::ArtifactSchemaDescriptor simulationRuntimeInputSchema{
    "loom.simulation_runtime_input", ::loom::SchemaVersion{2, 0}};

enum class SimulationWorkloadKind : std::uint32_t {
  Spatial = 0,
  System = 1,
  StructuredProgram = 2,
};

//===----------------------------------------------------------------------===//
// Canonical semantic values and memory bytes
//===----------------------------------------------------------------------===//

/// The closed semantic state shared by value lanes and memory bytes. Wire
/// discriminants are the zero-based declaration order.
enum class SemanticState : std::uint32_t { Defined = 0, Poison = 1, Undef = 2 };

/// Object-relative provenance accompanying one defined first-class LLVM
/// pointer lane. The offset is the exact signed two's-complement A(AS)-bit
/// byte offset; the lane bits separately carry the complete P(AS)-bit pointer
/// representation.
struct PointerTarget {
  std::uint64_t objectOrdinal = 0;
  llvm::APInt byteOffset;

  friend bool operator==(const PointerTarget &lhs, const PointerTarget &rhs) {
    return lhs.objectOrdinal == rhs.objectOrdinal &&
           lhs.byteOffset == rhs.byteOffset;
  }
};

/// One semantic lane. A Defined lane carries the exact fixed-width
/// software-semantic bits; host integers and doubles are never authorities.
struct SemanticLane {
  SemanticState state = SemanticState::Undef;
  llvm::APInt bits; // meaningful only when state == Defined
  std::optional<PointerTarget> pointerTarget;

  static SemanticLane defined(llvm::APInt value) {
    SemanticLane lane;
    lane.state = SemanticState::Defined;
    lane.bits = std::move(value);
    return lane;
  }
  static SemanticLane poison() {
    return SemanticLane{SemanticState::Poison, llvm::APInt(), std::nullopt};
  }
  static SemanticLane undef() {
    return SemanticLane{SemanticState::Undef, llvm::APInt(), std::nullopt};
  }
  static SemanticLane definedPointer(llvm::APInt representation,
                                     std::uint64_t objectOrdinal,
                                     llvm::APInt byteOffset) {
    SemanticLane lane = defined(std::move(representation));
    lane.pointerTarget = PointerTarget{objectOrdinal, std::move(byteOffset)};
    return lane;
  }

  friend bool operator==(const SemanticLane &lhs, const SemanticLane &rhs) {
    return lhs.state == rhs.state && lhs.bits == rhs.bits &&
           lhs.pointerTarget == rhs.pointerTarget;
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

/// The shared stream sequence used by runtime inputs and execution
/// observations.
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
// Spatial functional observations
//===----------------------------------------------------------------------===//

struct PublishedValueResult {
  CanonicalValueSequence value;
};
struct NotPublishedValueResult {};
using ValueResultObservation =
    std::variant<PublishedValueResult, NotPublishedValueResult>;

struct FullMemoryObservation {
  // The vector length is the canonical byte_count field.
  std::vector<SemanticMemoryByte> bytes;
};

struct MemoryDiffRun {
  std::uint64_t byteOffset = 0;
  std::vector<SemanticMemoryByte> changedBytes; // nonempty
};

struct DiffMemoryObservation {
  std::uint64_t byteCount = 0;
  std::vector<MemoryDiffRun> runs;
};

using MemoryObservationPayload =
    std::variant<FullMemoryObservation, DiffMemoryObservation>;

/// Exact positional observations selected by SpatialObservableContract. The
/// contract remains the sole owner of target identity, order, and memory form;
/// these arrays therefore carry payloads only.
struct SpatialFunctionalObservations {
  std::vector<ValueResultObservation> valueResults;
  std::vector<CanonicalStreamSequence> streamOutputs;
  std::vector<MemoryObservationPayload> memories;
};

/// Transient semantic lane geometry derived from one exact Dataflow boundary
/// type. It is shared by workload validation and external execution harnesses;
/// no copy enters an Artifact.
struct SpatialSimulationValueShape final {
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
};

struct SpatialSimulationBoundaryShapes final {
  std::vector<SpatialSimulationValueShape> valueInputs;
  std::vector<SpatialSimulationValueShape> streamInputs;
  std::vector<SpatialSimulationValueShape> valueResults;
  std::vector<SpatialSimulationValueShape> streamOutputs;
};

llvm::Expected<SpatialSimulationBoundaryShapes>
projectSpatialSimulationBoundaryShapes(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch);

/// Packs or unpacks one Defined, non-pointer token with the exact low-lane-
/// first transport ordering used by the Dataflow and CGRA simulators.
llvm::Expected<llvm::APInt>
packDefinedSpatialSimulationToken(const CanonicalValueSequence &sequence,
                                  SpatialSimulationValueShape shape,
                                  std::uint64_t tokenOrdinal);
llvm::Expected<std::vector<SemanticLane>>
unpackDefinedSpatialSimulationToken(const llvm::APInt &bits,
                                    SpatialSimulationValueShape shape);

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
  std::uint64_t memoryResultOrdinal = 0;
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
  std::vector<std::uint64_t> valueResults = {};
  std::vector<std::uint64_t> streamOutputs = {};
  std::vector<SpatialMemoryObservable> memories = {};
};

/// The schema-1.1 spatial workload root: one rooted graph launch, the dense
/// logical coordinates of the root thread point, the total value-input
/// classification, and the observable contract. Stream inputs and imported
/// memory roots are always runtime-plane and appear only in the exact
/// SimulationRuntimeInput.
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
// SystemSimulationWorkload
//===----------------------------------------------------------------------===//

using SystemValueInputSource =
    std::variant<CanonicalValueSequence, RuntimeValueInput>;

struct SystemExternalValueInputPlanEntry {
  deployment::DeploymentExternalInterfaceRef interfaceRef;
  SystemValueInputSource source;
};

struct SystemMemoryObservable {
  deployment::DeploymentExternalInterfaceRef interfaceRef;
  MemoryObservationForm form = MemoryObservationForm::FullState;
};

struct SystemObservableContract {
  std::vector<std::uint64_t> valueResults = {};
  std::vector<deployment::DeploymentExternalInterfaceRef> externalValueOutputs =
      {};
  std::vector<deployment::DeploymentExternalInterfaceRef>
      externalStreamOutputs = {};
  std::vector<SystemMemoryObservable> memories = {};
};

/// One finalized Deployment entry, its total program-value and external-value
/// input classifications, and its exact selected observables. The entry
/// reference is the sole Deployment identity carried by the workload.
struct SystemSimulationWorkload {
  deployment::DeploymentProgramEntryRef programEntryRef;
  std::vector<SystemValueInputSource> valueInputPlan = {};
  std::vector<SystemExternalValueInputPlanEntry> externalValueInputPlan = {};
  SystemObservableContract observableContract = {};
};

/// Transient program-entry value geometry derived from the exact Deployment
/// and its HostCore CompilerTargetBinding. It is shared by execution providers
/// and the independent execution finalizer; no copy enters an Artifact.
struct SystemSimulationValueShape final {
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  bool pointerPayload = false;
};

struct SystemSimulationBoundaryShapes final {
  bool littleEndian = false;
  std::vector<SystemSimulationValueShape> valueArguments;
  std::vector<SystemSimulationValueShape> valueResults;
};

llvm::Expected<SystemSimulationBoundaryShapes>
projectSystemSimulationBoundaryShapes(
    const deployment::FinalizedDeployment &deployment,
    const deployment::DeploymentProgramEntryRef &entry,
    const ::loom::ArtifactStore &store);

//===----------------------------------------------------------------------===//
// SpatialSimulationRuntimeInput
//===----------------------------------------------------------------------===//

struct RuntimeValueEntry {
  std::uint64_t valueInputOrdinal = 0;
  CanonicalValueSequence value; // exactly one token
};

/// Neutral byte-addressed software storage. The exact Canonical Dataflow
/// type, DataLayout, and root/view relations alone interpret typed accesses.
struct RuntimeMemoryPointer {
  std::uint64_t storageByteOffset = 0;
  std::uint32_t addressSpace = 0;
  PointerTarget target;
};

struct RuntimeMemoryObject {
  RuntimeMemoryObject() = default;
  explicit RuntimeMemoryObject(std::vector<SemanticMemoryByte> bytes,
                               std::vector<RuntimeMemoryPointer> pointers = {})
      : initialBytes(std::move(bytes)), pointerValues(std::move(pointers)) {}

  std::vector<SemanticMemoryByte> initialBytes; // nonempty
  // Canonical sorted provenance for defined pointer payloads already stored
  // in initialBytes. Representation bits remain owned by those bytes.
  std::vector<RuntimeMemoryPointer> pointerValues;
};

struct RuntimeMemoryRootBinding {
  std::uint64_t objectOrdinal = 0; // derived, never author-selected
  std::uint64_t byteOffset = 0;
};

struct MemoryRootBindingEntry {
  dataflow::LogicalMemoryRootRef root;
  RuntimeMemoryRootBinding binding;
};

/// The canonical schema-2.0 spatial runtime-input root. `runtimeValues` is
/// sorted by value-input ordinal and exactly complements the workload's Runtime
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
// SystemSimulationRuntimeInput
//===----------------------------------------------------------------------===//

struct SystemRuntimeEntryValue {
  std::uint64_t valueArgumentOrdinal = 0;
  CanonicalValueSequence value;
};

struct SystemRuntimeExternalValue {
  deployment::DeploymentExternalInterfaceRef interfaceRef;
  CanonicalValueSequence value;
};

struct SystemExternalStreamInput {
  deployment::DeploymentExternalInterfaceRef interfaceRef;
  CanonicalStreamSequence stream;
};

struct SystemMemoryInterfaceBindingEntry {
  deployment::DeploymentExternalInterfaceRef interfaceRef;
  RuntimeMemoryRootBinding binding;
};

struct SystemSimulationRuntimeInput {
  ::loom::ArtifactIdentity workloadIdentity;
  std::vector<SystemRuntimeEntryValue> runtimeEntryValues = {};
  std::vector<SystemRuntimeExternalValue> runtimeExternalValues = {};
  std::vector<SystemExternalStreamInput> externalStreamInputs = {};
  std::vector<RuntimeMemoryObject> memoryObjects = {};
  std::vector<SystemMemoryInterfaceBindingEntry> memoryInterfaceBindings = {};
};

struct SystemMemoryInterfaceBindingDraft {
  deployment::DeploymentExternalInterfaceRef interfaceRef;
  std::uint64_t authorObject = 0;
  std::uint64_t byteOffset = 0;
};

struct SystemSimulationRuntimeInputDraft {
  ::loom::ArtifactIdentity workloadIdentity;
  std::vector<SystemRuntimeEntryValue> runtimeEntryValues = {};
  std::vector<SystemRuntimeExternalValue> runtimeExternalValues = {};
  std::vector<SystemExternalStreamInput> externalStreamInputs = {};
  std::vector<RuntimeMemoryObject> memoryObjects = {};
  std::vector<SystemMemoryInterfaceBindingDraft> memoryInterfaceBindings = {};
};

//===----------------------------------------------------------------------===//
// Structured Program workload and runtime input
//===----------------------------------------------------------------------===//

struct StructuredRuntimeValueInput {};
struct StructuredRuntimeMemoryInput {};
using StructuredProgramArgumentSource =
    std::variant<CanonicalValueSequence, StructuredRuntimeValueInput,
                 StructuredRuntimeMemoryInput>;

struct EntryPointerArgumentTarget {
  std::uint64_t argumentOrdinal = 0;
};

struct GlobalObjectTarget {
  frontend::StructuredEntityRef global;
};

using StructuredProgramMemoryTarget =
    std::variant<EntryPointerArgumentTarget, GlobalObjectTarget>;

struct StructuredProgramMemoryObservable {
  StructuredProgramMemoryTarget target;
  MemoryObservationForm form = MemoryObservationForm::FullState;
};

struct StructuredProgramObservableContract {
  bool returnValue = false;
  std::vector<StructuredProgramMemoryObservable> memories = {};
};

/// One source-program workload. The entry reference owns the exact S0
/// identity; ABI order and types are recovered through its parent view.
struct StructuredProgramSimulationWorkload {
  frontend::StructuredEntityRef entryRef;
  std::vector<StructuredProgramArgumentSource> argumentPlan = {};
  StructuredProgramObservableContract observableContract = {};
};

struct StructuredRuntimeValueEntry {
  std::uint64_t argumentOrdinal = 0;
  CanonicalValueSequence value;
};

struct StructuredPointerBindingEntry {
  std::uint64_t argumentOrdinal = 0;
  RuntimeMemoryRootBinding binding;
};

struct StructuredProgramSimulationRuntimeInput {
  ::loom::ArtifactIdentity workloadIdentity;
  std::vector<StructuredRuntimeValueEntry> runtimeValues = {};
  std::vector<RuntimeMemoryObject> memoryObjects = {};
  std::vector<StructuredPointerBindingEntry> pointerBindings = {};
};

struct StructuredPointerBindingDraft {
  std::uint64_t argumentOrdinal = 0;
  std::uint64_t authorObject = 0;
  std::uint64_t byteOffset = 0;
};

struct StructuredProgramSimulationRuntimeInputDraft {
  ::loom::ArtifactIdentity workloadIdentity;
  std::vector<StructuredRuntimeValueEntry> runtimeValues = {};
  std::vector<RuntimeMemoryObject> memoryObjects = {};
  std::vector<StructuredPointerBindingDraft> pointerBindings = {};
};

using SimulationWorkloadModel =
    std::variant<SpatialSimulationWorkload, SystemSimulationWorkload,
                 StructuredProgramSimulationWorkload>;
using SimulationRuntimeInputModel =
    std::variant<SpatialSimulationRuntimeInput, SystemSimulationRuntimeInput,
                 StructuredProgramSimulationRuntimeInput>;

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
  SimulationWorkloadKind kind() const {
    if (std::holds_alternative<SpatialSimulationWorkload>(model_))
      return SimulationWorkloadKind::Spatial;
    if (std::holds_alternative<SystemSimulationWorkload>(model_))
      return SimulationWorkloadKind::System;
    return SimulationWorkloadKind::StructuredProgram;
  }
  const SimulationWorkloadModel &root() const { return model_; }
  const SpatialSimulationWorkload *spatial() const {
    return std::get_if<SpatialSimulationWorkload>(&model_);
  }
  const SystemSimulationWorkload *system() const {
    return std::get_if<SystemSimulationWorkload>(&model_);
  }
  const StructuredProgramSimulationWorkload *structuredProgram() const {
    return std::get_if<StructuredProgramSimulationWorkload>(&model_);
  }
  const ::loom::CanonicalSemanticBytes &canonicalBytes() const {
    return bytes_;
  }

private:
  CanonicalSimulationWorkload(::loom::ArtifactIdentity identity,
                              SimulationWorkloadModel model,
                              ::loom::CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}

  ::loom::ArtifactIdentity identity_;
  SimulationWorkloadModel model_;
  ::loom::CanonicalSemanticBytes bytes_;

  friend llvm::Expected<CanonicalSimulationWorkload>
  finalizeSimulationWorkload(const SpatialSimulationWorkload &,
                             const dataflow::CanonicalDataflowProgramView &);
  friend llvm::Expected<CanonicalSimulationWorkload>
  importSimulationWorkload(llvm::ArrayRef<std::uint8_t>,
                           const dataflow::CanonicalDataflowProgramView &,
                           const ::loom::ArtifactIdentity &);
  friend llvm::Expected<CanonicalSimulationWorkload>
  finalizeSimulationWorkload(const StructuredProgramSimulationWorkload &,
                             const frontend::StructuredProgramCandidateView &);
  friend llvm::Expected<CanonicalSimulationWorkload>
  importSimulationWorkload(llvm::ArrayRef<std::uint8_t>,
                           const frontend::StructuredProgramCandidateView &,
                           const ::loom::ArtifactIdentity &);
  friend llvm::Expected<CanonicalSimulationWorkload>
  finalizeSimulationWorkload(const SystemSimulationWorkload &,
                             const deployment::FinalizedDeployment &,
                             const ::loom::ArtifactStore &);
  friend llvm::Expected<CanonicalSimulationWorkload> importSimulationWorkload(
      llvm::ArrayRef<std::uint8_t>, const deployment::FinalizedDeployment &,
      const ::loom::ArtifactStore &, const ::loom::ArtifactIdentity &);
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
  SimulationWorkloadKind kind() const {
    if (std::holds_alternative<SpatialSimulationRuntimeInput>(model_))
      return SimulationWorkloadKind::Spatial;
    if (std::holds_alternative<SystemSimulationRuntimeInput>(model_))
      return SimulationWorkloadKind::System;
    return SimulationWorkloadKind::StructuredProgram;
  }
  const SimulationRuntimeInputModel &root() const { return model_; }
  const SpatialSimulationRuntimeInput *spatial() const {
    return std::get_if<SpatialSimulationRuntimeInput>(&model_);
  }
  const SystemSimulationRuntimeInput *system() const {
    return std::get_if<SystemSimulationRuntimeInput>(&model_);
  }
  const StructuredProgramSimulationRuntimeInput *structuredProgram() const {
    return std::get_if<StructuredProgramSimulationRuntimeInput>(&model_);
  }
  const ::loom::CanonicalSemanticBytes &canonicalBytes() const {
    return bytes_;
  }

private:
  CanonicalSimulationRuntimeInput(::loom::ArtifactIdentity identity,
                                  SimulationRuntimeInputModel model,
                                  ::loom::CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}

  ::loom::ArtifactIdentity identity_;
  SimulationRuntimeInputModel model_;
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
  friend llvm::Expected<CanonicalSimulationRuntimeInput>
  finalizeSimulationRuntimeInput(
      const StructuredProgramSimulationRuntimeInputDraft &,
      const CanonicalSimulationWorkload &,
      const frontend::StructuredProgramCandidateView &);
  friend llvm::Expected<CanonicalSimulationRuntimeInput>
  importSimulationRuntimeInput(llvm::ArrayRef<std::uint8_t>,
                               const CanonicalSimulationWorkload &,
                               const frontend::StructuredProgramCandidateView &,
                               const ::loom::ArtifactIdentity &);
  friend llvm::Expected<CanonicalSimulationRuntimeInput>
  finalizeSimulationRuntimeInput(const SystemSimulationRuntimeInputDraft &,
                                 const CanonicalSimulationWorkload &,
                                 const deployment::FinalizedDeployment &,
                                 const ::loom::ArtifactStore &);
  friend llvm::Expected<CanonicalSimulationRuntimeInput>
  importSimulationRuntimeInput(llvm::ArrayRef<std::uint8_t>,
                               const CanonicalSimulationWorkload &,
                               const deployment::FinalizedDeployment &,
                               const ::loom::ArtifactStore &,
                               const ::loom::ArtifactIdentity &);
};

struct ImportedStructuredProgramSimulationInputs {
  frontend::StructuredProgramCandidate structuredProgram;
  CanonicalSimulationWorkload workload;
  CanonicalSimulationRuntimeInput runtimeInput;
};

struct ImportedSpatialSimulationInputs {
  dataflow::CanonicalDataflowArtifact dataflow;
  CanonicalSimulationWorkload workload;
  CanonicalSimulationRuntimeInput runtimeInput;
};

struct ImportedSpatialSimulationWorkload {
  dataflow::CanonicalDataflowArtifact dataflow;
  CanonicalSimulationWorkload workload;
};

struct ImportedSystemSimulationInputs {
  deployment::FinalizedDeployment deployment;
  CanonicalSimulationWorkload workload;
  CanonicalSimulationRuntimeInput runtimeInput;
};

llvm::Expected<::loom::ArtifactRootReference>
publishSimulationWorkload(const CanonicalSimulationWorkload &workload,
                          const ::loom::ArtifactStore &store);

llvm::Expected<::loom::ArtifactRootReference> publishSimulationRuntimeInput(
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ::loom::ArtifactStore &store);

/// Strictly imports one stored Structured Program workload/runtime pair and
/// its sole owner. The owner identity is recovered from the workload's typed
/// entry reference; no caller-supplied program, symbol, or path participates.
llvm::Expected<ImportedStructuredProgramSimulationInputs>
importStructuredProgramSimulationInputs(
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput,
    const ::loom::ArtifactStore &store);

/// Strictly imports one stored Spatial workload/runtime pair and recovers its
/// sole Canonical Dataflow owner from the workload's RootedGraphLaunchRef.
/// No caller-provided program reference or path participates.
llvm::Expected<ImportedSpatialSimulationInputs>
importSpatialSimulationInputs(const ::loom::ArtifactRootReference &workload,
                              const ::loom::ArtifactRootReference &runtimeInput,
                              const ::loom::ArtifactStore &store);

/// Strictly imports one stored Spatial workload and recovers its sole
/// Canonical Dataflow owner without requiring a runtime-input instance.
llvm::Expected<ImportedSpatialSimulationWorkload>
importSpatialSimulationWorkload(const ::loom::ArtifactRootReference &workload,
                                const ::loom::ArtifactStore &store);

/// Strictly imports one runtime-input root against an already imported exact
/// Spatial workload. The workload owner and its Dataflow projection are reused;
/// the runtime bytes are still rehashed and fully validated.
llvm::Expected<CanonicalSimulationRuntimeInput>
importSpatialSimulationRuntimeInput(
    const ::loom::ArtifactRootReference &runtimeInput,
    const ImportedSpatialSimulationWorkload &workload,
    const ::loom::ArtifactStore &store);

/// Strictly imports one stored System workload/runtime pair and its exact
/// Deployment owner. Deployment remains responsible for closure and blob
/// validation.
llvm::Expected<ImportedSystemSimulationInputs>
importSystemSimulationInputs(const ::loom::ArtifactRootReference &workload,
                             const ::loom::ArtifactRootReference &runtimeInput,
                             const ::loom::ArtifactStore &store,
                             const ::loom::BlobStore &blobs);

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
/// owner-mismatched root, and every semantic violation the finalizer rejects,
/// then requires the recomputed identity to equal `expectedIdentity`.
llvm::Expected<CanonicalSimulationWorkload>
importSimulationWorkload(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                         const dataflow::CanonicalDataflowProgramView &program,
                         const ::loom::ArtifactIdentity &expectedIdentity);

/// Failure-atomic finalization and strict import of one Deployment-owned
/// System workload. Entry and external-interface catalogs, types, directions,
/// and the target DataLayout are recovered from the exact owner.
llvm::Expected<CanonicalSimulationWorkload>
finalizeSimulationWorkload(const SystemSimulationWorkload &workload,
                           const deployment::FinalizedDeployment &deployment,
                           const ::loom::ArtifactStore &store);

llvm::Expected<CanonicalSimulationWorkload>
importSimulationWorkload(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                         const deployment::FinalizedDeployment &deployment,
                         const ::loom::ArtifactStore &store,
                         const ::loom::ArtifactIdentity &expectedIdentity);

/// Failure-atomic finalization of one source Structured Program workload.
/// Entry ownership, LLVM ABI argument classification, fixed value widths, and
/// observable targets are recovered from the exact Structured owner view.
llvm::Expected<CanonicalSimulationWorkload> finalizeSimulationWorkload(
    const StructuredProgramSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &program);

/// Strict import of the Structured Program root. Other roots are rejected by
/// this owner-specific overload.
llvm::Expected<CanonicalSimulationWorkload> importSimulationWorkload(
    llvm::ArrayRef<std::uint8_t> canonicalBytes,
    const frontend::StructuredProgramCandidateView &program,
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

/// Canonicalizes the System runtime complements and Deployment-interface
/// memory bindings. Sharing one draft object slot is the only aliasing
/// authority; persistent ordinals derive from sorted interface binding keys.
llvm::Expected<CanonicalSimulationRuntimeInput> finalizeSimulationRuntimeInput(
    const SystemSimulationRuntimeInputDraft &draft,
    const CanonicalSimulationWorkload &workload,
    const deployment::FinalizedDeployment &deployment,
    const ::loom::ArtifactStore &store);

llvm::Expected<CanonicalSimulationRuntimeInput>
importSimulationRuntimeInput(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                             const CanonicalSimulationWorkload &workload,
                             const deployment::FinalizedDeployment &deployment,
                             const ::loom::ArtifactStore &store,
                             const ::loom::ArtifactIdentity &expectedIdentity);

/// Canonicalizes source runtime values and finite pointer-backed objects.
/// Object ordinals derive from sorted pointer-binding keys; sharing one draft
/// object is the only aliasing authority.
llvm::Expected<CanonicalSimulationRuntimeInput> finalizeSimulationRuntimeInput(
    const StructuredProgramSimulationRuntimeInputDraft &draft,
    const CanonicalSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &program);

llvm::Expected<CanonicalSimulationRuntimeInput> importSimulationRuntimeInput(
    llvm::ArrayRef<std::uint8_t> canonicalBytes,
    const CanonicalSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &program,
    const ::loom::ArtifactIdentity &expectedIdentity);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONARTIFACTS_H
