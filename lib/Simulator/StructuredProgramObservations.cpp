#include "StructuredProgramNativeExecutionInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Runtime/ComputationBoundary.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

bool loom::sim::isNativeStructuredProfileBlock(mlir::Block *block) {
  mlir::Operation *owner = block ? block->getParentOp() : nullptr;
  if (!owner || llvm::isa<mlir::scf::InParallelOp>(owner) ||
      owner->getParentOfType<mlir::scf::InParallelOp>())
    return false;
  auto function = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(owner);
  if (!function && owner)
    function = owner->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (function && !function.getBody().empty())
    return true;
  return llvm::isa<dataflow::ThreadOp>(owner) ||
         static_cast<bool>(owner->getParentOfType<dataflow::ThreadOp>());
}

namespace loom::sim::native_detail {
namespace {

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("native_structured_program_unsupported: ") + message);
}

llvm::Error executionFailed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::io_error),
      llvm::Twine("native_structured_program_execution_failed: ") + message);
}

std::vector<SemanticMemoryByte>
definedSemanticBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<SemanticMemoryByte> result;
  result.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    result.push_back({SemanticState::Defined, byte});
  return result;
}

bool sameMemoryByte(const SemanticMemoryByte &lhs,
                    const SemanticMemoryByte &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.value == rhs.value);
}

MemoryObservationPayload
makeMemoryObservation(MemoryObservationForm form,
                      llvm::ArrayRef<SemanticMemoryByte> baseline,
                      llvm::ArrayRef<std::uint8_t> finalBytes) {
  std::vector<SemanticMemoryByte> final = definedSemanticBytes(finalBytes);
  if (form == MemoryObservationForm::FullState)
    return FullMemoryObservation{std::move(final)};

  DiffMemoryObservation diff;
  diff.byteCount = final.size();
  std::uint64_t offset = 0;
  while (offset < final.size()) {
    if (sameMemoryByte(baseline[offset], final[offset])) {
      ++offset;
      continue;
    }
    MemoryDiffRun run;
    run.byteOffset = offset;
    do {
      run.changedBytes.push_back(final[offset]);
      ++offset;
    } while (offset < final.size() &&
             !sameMemoryByte(baseline[offset], final[offset]));
    diff.runs.push_back(std::move(run));
  }
  return diff;
}

bool sameLane(const SemanticLane &lhs, const SemanticLane &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.bits == rhs.bits);
}

bool sameValueSequence(const CanonicalValueSequence &lhs,
                       const CanonicalValueSequence &rhs) {
  return lhs.tokenCount == rhs.tokenCount &&
         lhs.lanes.size() == rhs.lanes.size() &&
         llvm::equal(lhs.lanes, rhs.lanes, sameLane);
}

bool sameMemoryObservation(const MemoryObservationPayload &lhs,
                           const MemoryObservationPayload &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *lhsFull = std::get_if<FullMemoryObservation>(&lhs)) {
    const auto &rhsFull = std::get<FullMemoryObservation>(rhs);
    return lhsFull->bytes.size() == rhsFull.bytes.size() &&
           llvm::equal(lhsFull->bytes, rhsFull.bytes, sameMemoryByte);
  }
  const auto &lhsDiff = std::get<DiffMemoryObservation>(lhs);
  const auto &rhsDiff = std::get<DiffMemoryObservation>(rhs);
  if (lhsDiff.byteCount != rhsDiff.byteCount ||
      lhsDiff.runs.size() != rhsDiff.runs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhsDiff.runs, rhsDiff.runs))
    if (left.byteOffset != right.byteOffset ||
        left.changedBytes.size() != right.changedBytes.size() ||
        !llvm::equal(left.changedBytes, right.changedBytes, sameMemoryByte))
      return false;
  return true;
}

} // namespace

llvm::Expected<AlignedByteStorage>
AlignedByteStorage::create(llvm::ArrayRef<std::uint8_t> bytes,
                           llvm::Align alignment) {
  AlignedByteStorage storage;
  storage.size_ = bytes.size();
  storage.alignment_ = alignment.value();
  storage.data_ = static_cast<std::uint8_t *>(::operator new(
      storage.size_, std::align_val_t(storage.alignment_), std::nothrow));
  if (!storage.data_)
    return executionFailed("cannot allocate a runtime memory object");
  std::memcpy(storage.data_, bytes.data(), storage.size_);
  return std::move(storage);
}

AlignedByteStorage::AlignedByteStorage(AlignedByteStorage &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)), alignment_(other.alignment_) {}

AlignedByteStorage &
AlignedByteStorage::operator=(AlignedByteStorage &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  data_ = std::exchange(other.data_, nullptr);
  size_ = std::exchange(other.size_, 0);
  alignment_ = other.alignment_;
  return *this;
}

AlignedByteStorage::~AlignedByteStorage() { reset(); }

void AlignedByteStorage::reset() {
  if (data_)
    ::operator delete(data_, std::align_val_t(alignment_));
  data_ = nullptr;
}

std::string uniqueMlirSymbolName(mlir::ModuleOp module,
                                 llvm::StringRef prefix) {
  std::string candidate = prefix.str();
  std::uint64_t suffix = 0;
  while (mlir::SymbolTable::lookupSymbolIn(module, candidate))
    candidate = (prefix + "." + llvm::Twine(++suffix)).str();
  return candidate;
}

llvm::Expected<BlockActivationCallbackNames>
instrumentBlockActivations(mlir::ModuleOp module,
                           const ArtifactIdentity &identity,
                           NativeExecutionContext &capture) {
  auto view = frontend::buildStructuredProgramCandidateView(module, identity);
  if (!view)
    return view.takeError();

  struct ProfileSite final {
    frontend::StructuredEntityRef reference;
    mlir::Block *block = nullptr;
  };
  std::vector<ProfileSite> sites;
  for (const frontend::StructuredEntity &entity :
       view->entities(frontend::StructuredEntityKind::Block))
    if (isNativeStructuredProfileBlock(entity.block))
      sites.push_back({entity.reference, entity.block});

  const std::string callbackName =
      uniqueMlirSymbolName(module, "__loom_structured_block_activation");
  mlir::OpBuilder declarations(module.getContext());
  declarations.setInsertionPointToStart(module.getBody());
  const mlir::Type i64 = declarations.getI64Type();
  const mlir::Type callbackType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(module.getContext()), {i64});
  mlir::LLVM::LLVMFuncOp::create(declarations, module.getLoc(), callbackName,
                                 callbackType);

  // The source-declared computation interval is what System QoR measures, so
  // the same interval gates this projection. The markers are ordinary weak
  // calls in the module: give each defined marker a callback at its entry and
  // the counting rule follows the real control flow, including recursion,
  // repeated samples, and callees the source reaches only inside the
  // interval. A module that defines no marker keeps its complete execution.
  std::optional<std::string> boundaryCallbackName;
  for (loom::runtime::ComputationBoundary boundary :
       {loom::runtime::ComputationBoundary::Begin,
        loom::runtime::ComputationBoundary::End}) {
    auto marker = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
        mlir::SymbolTable::lookupSymbolIn(
            module, loom::runtime::computationBoundarySymbol(boundary)));
    if (!marker || marker.getBody().empty())
      continue;
    if (!boundaryCallbackName) {
      boundaryCallbackName =
          uniqueMlirSymbolName(module, "__loom_structured_computation_boundary");
      mlir::OpBuilder boundaryDeclaration(module.getContext());
      boundaryDeclaration.setInsertionPointToStart(module.getBody());
      mlir::LLVM::LLVMFuncOp::create(boundaryDeclaration, module.getLoc(),
                                     *boundaryCallbackName, callbackType);
    }
    mlir::Block &entry = marker.getBody().front();
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(&entry);
    const mlir::Location location =
        entry.empty() ? marker.getLoc() : entry.front().getLoc();
    mlir::Value actionValue = mlir::LLVM::ConstantOp::create(
        builder, location, i64,
        builder.getI64IntegerAttr(static_cast<std::uint64_t>(boundary)));
    mlir::LLVM::CallOp::create(builder, location, mlir::TypeRange{},
                               *boundaryCallbackName,
                               mlir::ValueRange{actionValue});
  }

  capture.profileBlocks.reserve(sites.size());
  capture.blockActivationCounts.assign(sites.size(), 0);
  capture.measuredBlockActivationCounts.assign(sites.size(), 0);
  for (auto [ordinal, site] : llvm::enumerate(sites)) {
    capture.profileBlocks.push_back(site.reference);
    mlir::OpBuilder builder(module.getContext());
    if (!site.block->empty() &&
        llvm::isa<mlir::LLVM::LandingpadOp>(site.block->front()))
      builder.setInsertionPointAfter(&site.block->front());
    else
      builder.setInsertionPointToStart(site.block);
    const mlir::Location location =
        site.block->empty() ? module.getLoc() : site.block->front().getLoc();
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        builder, location, i64, builder.getI64IntegerAttr(ordinal));
    mlir::LLVM::CallOp::create(builder, location, mlir::TypeRange{},
                               callbackName, mlir::ValueRange{ordinalValue});
  }
  if (mlir::failed(mlir::verify(module)))
    return unsupported(
        "native block activation provider cannot instrument this Structured "
        "control form");
  return BlockActivationCallbackNames{callbackName,
                                      std::move(boundaryCallbackName)};
}

llvm::Expected<NativeStructuredProgramObservations>
buildObservations(const StructuredProgramSimulationWorkload &workload,
                  const StructuredProgramSimulationRuntimeInput &input,
                  llvm::ArrayRef<MemoryTargetPlan> plans,
                  const NativeExecutionContext &capture) {
  NativeStructuredProgramObservations result;
  if (workload.observableContract.returnValue) {
    if (!capture.returnValue)
      return executionFailed("selected return value was not captured");
    result.returnValue = capture.returnValue;
  }
  result.memories.reserve(plans.size());
  for (std::uint64_t ordinal = 0; ordinal < plans.size(); ++ordinal) {
    const MemoryTargetPlan &plan = plans[ordinal];
    if (plan.objectOrdinal) {
      const RuntimeMemoryObject &baseline =
          input.memoryObjects[*plan.objectOrdinal];
      const AlignedByteStorage &final = capture.objects[*plan.objectOrdinal];
      result.memories.push_back(makeMemoryObservation(
          plan.form, baseline.initialBytes,
          llvm::ArrayRef<std::uint8_t>(final.data(), final.size())));
      continue;
    }
    if (!capture.sawGlobalBefore[ordinal] || !capture.sawGlobalAfter[ordinal])
      return executionFailed("selected global observation was not captured");
    std::vector<SemanticMemoryByte> baseline =
        definedSemanticBytes(capture.globalBefore[ordinal]);
    result.memories.push_back(makeMemoryObservation(
        plan.form, baseline, capture.globalAfter[ordinal]));
  }
  if (capture.profileBlocks.size() != capture.blockActivationCounts.size() ||
      capture.profileBlocks.size() !=
          capture.measuredBlockActivationCounts.size())
    return executionFailed("block activation projection is inconsistent");
  if (capture.computationBoundaryDepth != 0)
    return executionFailed(
        "the source-declared computation interval was left open");
  result.blockActivations.reserve(capture.profileBlocks.size());
  for (std::size_t ordinal = 0; ordinal < capture.profileBlocks.size();
       ++ordinal) {
    const std::uint64_t activations = capture.blockActivationCounts[ordinal];
    // A source that declares no interval, or a workload that never reaches
    // one, is measured whole. Every consumer then reads one total projection.
    result.blockActivations.push_back(
        {capture.profileBlocks[ordinal], activations,
         capture.computationIntervalObserved
             ? capture.measuredBlockActivationCounts[ordinal]
             : activations});
  }
  return result;
}

bool haveEquivalentFunctionalObservationsImpl(
    const NativeStructuredProgramObservations &reference,
    const NativeStructuredProgramObservations &candidate) {
  if (reference.returnValue.has_value() != candidate.returnValue.has_value())
    return false;
  if (reference.returnValue &&
      !sameValueSequence(*reference.returnValue, *candidate.returnValue))
    return false;
  if (reference.memories.size() != candidate.memories.size())
    return false;
  for (auto [left, right] :
       llvm::zip_equal(reference.memories, candidate.memories))
    if (!sameMemoryObservation(left, right))
      return false;
  return true;
}

} // namespace loom::sim::native_detail

namespace loom::sim {

bool haveEquivalentFunctionalObservations(
    const NativeStructuredProgramObservations &reference,
    const NativeStructuredProgramObservations &candidate) {
  return native_detail::haveEquivalentFunctionalObservationsImpl(reference,
                                                                 candidate);
}

} // namespace loom::sim
