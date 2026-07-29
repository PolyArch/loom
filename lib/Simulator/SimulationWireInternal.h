#ifndef LOOM_LIB_SIMULATOR_SIMULATIONWIREINTERNAL_H
#define LOOM_LIB_SIMULATOR_SIMULATIONWIREINTERNAL_H

#include "Simulator/SimulationArtifacts.h"

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Internal shared machinery of the simulation workload/runtime-input wire.
// The canonical framing is fixed by docs/spec-simulation-artifacts.md:
// declaration-order fields, zero-based u32be discriminants, u64be counts and
// ordinals, fixed 32-byte Artifact identities, sorted typed tables, and no
// field names, padding, native layout, JSON, or MLIR authority.
namespace loom::sim::detail {

llvm::Error invalid(const llvm::Twine &message);

//===----------------------------------------------------------------------===//
// Byte writer and checked reader
//===----------------------------------------------------------------------===//

class WireWriter {
public:
  void u32(std::uint32_t value) {
    bytes_.push_back(static_cast<std::uint8_t>(value >> 24));
    bytes_.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes_.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes_.push_back(static_cast<std::uint8_t>(value));
  }
  void u64(std::uint64_t value) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
    bytes_.push_back(static_cast<std::uint8_t>(value));
  }
  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void identity(const ::loom::ArtifactIdentity &value) { bytes(value.bytes()); }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class WireReader {
public:
  explicit WireReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (bytes_.size() - offset_ < 4)
      return invalid("truncated u32");
    const std::uint32_t value =
        (static_cast<std::uint32_t>(bytes_[offset_]) << 24) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 16) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 8) |
        static_cast<std::uint32_t>(bytes_[offset_ + 3]);
    offset_ += 4;
    return value;
  }
  llvm::Expected<std::uint64_t> u64() {
    if (bytes_.size() - offset_ < 8)
      return invalid("truncated u64");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_ + index];
    offset_ += 8;
    return value;
  }
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> bytes(std::size_t count) {
    if (count > bytes_.size() - offset_)
      return invalid("truncated byte field");
    llvm::ArrayRef<std::uint8_t> value = bytes_.slice(offset_, count);
    offset_ += count;
    return value;
  }
  llvm::Expected<::loom::ArtifactIdentity> identity() {
    llvm::Expected<llvm::ArrayRef<std::uint8_t>> raw =
        bytes(::loom::ArtifactIdentity::byteSize);
    if (!raw)
      return raw.takeError();
    return ::loom::ArtifactIdentity::fromBytes(*raw);
  }
  // Guard a count-sized allocation: every remaining element must be
  // representable in the bytes left, given each element costs at least
  // `minElementBytes` on the wire.
  llvm::Error guardCount(std::uint64_t count, std::uint64_t minElementBytes) {
    if (count > (bytes_.size() - offset_) / minElementBytes)
      return invalid("element count exceeds the remaining bytes");
    return llvm::Error::success();
  }
  bool atEnd() const { return offset_ == bytes_.size(); }
  std::size_t offset() const { return offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

//===----------------------------------------------------------------------===//
// Typed key order (canonical table order)
//===----------------------------------------------------------------------===//

// Three-way comparison; the wire byte order of each encoded key equals the
// typed order, so one comparator serves model and wire sortedness checks.
int compareIdentities(const ::loom::ArtifactIdentity &lhs,
                      const ::loom::ArtifactIdentity &rhs);
int compareRootKeys(const dataflow::LogicalMemoryRootRef &lhs,
                    const dataflow::LogicalMemoryRootRef &rhs);
int compareRootOrViewKeys(const dataflow::LogicalMemoryRootOrViewRef &lhs,
                          const dataflow::LogicalMemoryRootOrViewRef &rhs);
int compareObservableTargets(const SpatialMemoryObservableTarget &lhs,
                             const SpatialMemoryObservableTarget &rhs);
int compareStructuredMemoryTargets(const StructuredProgramMemoryTarget &lhs,
                                   const StructuredProgramMemoryTarget &rhs);

//===----------------------------------------------------------------------===//
// Resolved launch context
//===----------------------------------------------------------------------===//

// The lane shape of one value or stream port type: lanes per token and the
// exact semantic bit width of one lane. Scalar integer, index, and floating
// types have one lane; a fixed-ranked non-scalable vector flattens row-major
// into the checked product of its dimensions. Anything else has no canonical
// lane shape.
struct LaneShape {
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
};
llvm::Expected<LaneShape> laneShapeOf(mlir::Type type,
                                      mlir::Operation *contextOp);

// The context every workload/runtime-input consumer recovers from the exact
// Dataflow owner through one rooted launch. Nothing here copies Dataflow
// facts into a second authority; every field is borrowed or derived on this
// cold path from the resolved launch.
struct ResolvedLaunchContext {
  dataflow::GraphRef graph;
  dataflow::GraphOp graphOp = {};
  dataflow::ThreadOp thread = {};
  dataflow::ThreadLaunchOp rootLaunchOp = {};
  dataflow::GraphLaunchOp graphLaunchOp = {};
  std::uint64_t numValueInputs = 0;
  std::uint64_t numStreamInputs = 0;
  std::uint64_t numValueResults = 0;
  std::uint64_t numStreamOutputs = 0;
  unsigned threadRank = 0;
  std::vector<LaneShape> valueInputShapes = {};
  std::vector<LaneShape> streamInputShapes = {};
  std::vector<LaneShape> valueResultShapes = {};
  std::vector<LaneShape> streamOutputShapes = {};
  // Graph memory-input ordinal -> imported runtime root. A missing entry is a
  // fresh allocation or exposure owned by another graph activation and cannot
  // be seeded from SimulationRuntimeInput.
  std::vector<std::optional<dataflow::LogicalMemoryRootRef>> memoryInputRoots =
      {};
  // Sorted by the typed root key; the imported logical-memory roots reachable
  // from this launch through its graph memory-input bindings.
  std::vector<dataflow::LogicalMemoryRootRef> importedRoots = {};
  // Sorted by the typed root key; every root a direct memory observable may
  // name: the imported roots plus the fresh allocations owned by the called
  // graph.
  std::vector<dataflow::LogicalMemoryRootRef> observableRoots = {};
};

llvm::Expected<ResolvedLaunchContext>
resolveLaunchContext(const dataflow::CanonicalDataflowProgramView &view,
                     const dataflow::RootedGraphLaunchRef &launch);

struct ResolvedStructuredProgramContext {
  mlir::Operation *entryOp = nullptr;
  std::vector<mlir::Type> argumentTypes = {};
  mlir::Type returnType;
  std::vector<std::optional<LaneShape>> argumentShapes = {};
  std::optional<LaneShape> returnShape;
};

llvm::Expected<ResolvedStructuredProgramContext>
resolveStructuredProgramContext(
    const frontend::StructuredProgramCandidateView &view,
    const frontend::StructuredEntityRef &entry);

llvm::Error validateStructuredProgramWorkload(
    const StructuredProgramSimulationWorkload &workload,
    const ResolvedStructuredProgramContext &context,
    const frontend::StructuredProgramCandidateView &view);

/// Bootstrap only the Structured owner identity needed to perform a strict
/// stored import. The complete workload is subsequently decoded, validated,
/// and byte-reencoded by its normal owner importer.
llvm::Expected<::loom::ArtifactIdentity>
structuredProgramWorkloadOwnerIdentity(llvm::ArrayRef<std::uint8_t> bytes);

//===----------------------------------------------------------------------===//
// Semantic value, stream, and memory-byte validation and codec
//===----------------------------------------------------------------------===//

llvm::Error validateValueSequence(const CanonicalValueSequence &sequence,
                                  const LaneShape &shape,
                                  const llvm::Twine &what);

void encodeValueSequence(WireWriter &writer,
                         const CanonicalValueSequence &sequence);
llvm::Expected<CanonicalValueSequence>
decodeValueSequence(WireReader &reader, const LaneShape &shape);

void encodeStreamSequence(WireWriter &writer,
                          const CanonicalStreamSequence &sequence);
llvm::Expected<CanonicalStreamSequence>
decodeStreamSequence(WireReader &reader, const LaneShape &shape);

void encodeMemoryObject(WireWriter &writer, const RuntimeMemoryObject &object);
llvm::Expected<RuntimeMemoryObject> decodeMemoryObject(WireReader &reader);

struct RuntimeObjectBindingKey {
  std::uint64_t authorObject = 0;
  std::vector<std::uint8_t> targetAndOffset;
};

llvm::Error
validateRuntimeMemoryObjects(llvm::ArrayRef<RuntimeMemoryObject> objects);

/// Assign canonical object ordinals from each object's sorted nonempty list of
/// typed target-and-offset wire keys. Callers own target validation and must
/// supply entries in canonical target order.
llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>>
deriveCanonicalObjectOrdinals(llvm::ArrayRef<RuntimeObjectBindingKey> bindings);

//===----------------------------------------------------------------------===//
// Shared semantic validation (finalize, import, and admission)
//===----------------------------------------------------------------------===//

// Each public finalize/import/admit operation resolves exactly one
// ResolvedLaunchContext per call and threads it through decoding,
// canonicalization, and validation; nothing here persists or caches it.
llvm::Error
validateSpatialWorkload(const SpatialSimulationWorkload &workload,
                        const ResolvedLaunchContext &context,
                        const dataflow::CanonicalDataflowProgramView &view);

llvm::Error
validateSpatialRuntimeInput(const SpatialSimulationRuntimeInput &input,
                            const SpatialSimulationWorkload &workload,
                            const ::loom::ArtifactIdentity &workloadIdentity,
                            const ResolvedLaunchContext &context,
                            const dataflow::CanonicalDataflowProgramView &view);

// Derive the canonical runtime input from an author draft: validate every
// table and assign each object its zero-based ordinal from the sorted
// binding keys. Grouping is expressed only by sharing a draft object slot.
llvm::Expected<SpatialSimulationRuntimeInput> canonicalizeSpatialRuntimeInput(
    const SpatialSimulationRuntimeInputDraft &draft,
    const SpatialSimulationWorkload &workload,
    const ::loom::ArtifactIdentity &workloadIdentity,
    const ResolvedLaunchContext &context,
    const dataflow::CanonicalDataflowProgramView &view);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_SIMULATIONWIREINTERNAL_H
