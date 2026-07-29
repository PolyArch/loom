#include "Simulator/NativeSimulationOracle.h"

#include "NativeExecutionSupport.h"
#include "SimulationWireInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("native_structured_program_invalid: ") + message);
}

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

llvm::Error classifyJitError(llvm::Error error, llvm::StringRef operation) {
  bool missingSymbol = false;
  bool otherFailure = false;
  std::string detail;
  llvm::raw_string_ostream stream(detail);
  llvm::handleAllErrors(
      std::move(error),
      [&](const llvm::orc::SymbolsNotFound &missing) {
        missingSymbol = true;
        missing.log(stream);
      },
      [&](const llvm::ErrorInfoBase &other) {
        otherFailure = true;
        other.log(stream);
      });
  stream.flush();
  if (missingSymbol && !otherFailure)
    return unsupported(llvm::Twine(operation) + ": " + detail);
  return executionFailed(llvm::Twine(operation) + ": " + detail);
}

class AlignedByteStorage final {
public:
  static llvm::Expected<AlignedByteStorage>
  create(llvm::ArrayRef<std::uint8_t> bytes, llvm::Align alignment) {
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

  AlignedByteStorage(const AlignedByteStorage &) = delete;
  AlignedByteStorage &operator=(const AlignedByteStorage &) = delete;

  AlignedByteStorage(AlignedByteStorage &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)), alignment_(other.alignment_) {}

  AlignedByteStorage &operator=(AlignedByteStorage &&other) noexcept {
    if (this == &other)
      return *this;
    reset();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    alignment_ = other.alignment_;
    return *this;
  }

  ~AlignedByteStorage() { reset(); }

  std::uint8_t *data() { return data_; }
  const std::uint8_t *data() const { return data_; }
  std::size_t size() const { return size_; }

private:
  AlignedByteStorage() = default;

  void reset() {
    if (data_)
      ::operator delete(data_, std::align_val_t(alignment_));
    data_ = nullptr;
  }

  std::uint8_t *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t alignment_ = alignof(std::max_align_t);
};

struct MemoryTargetPlan {
  MemoryObservationForm form = MemoryObservationForm::FullState;
  std::optional<std::uint64_t> objectOrdinal;
  std::string globalSymbol;
  std::uint64_t byteCount = 0;
};

struct NativeExecutionContext {
  std::vector<AlignedByteStorage> objects;
  std::optional<detail::LaneShape> returnShape;
  std::uint64_t returnByteCount = 0;
  bool littleEndian = true;
  std::optional<CanonicalValueSequence> returnValue;
  std::vector<std::vector<std::uint8_t>> globalBefore;
  std::vector<std::vector<std::uint8_t>> globalAfter;
  std::vector<bool> sawGlobalBefore;
  std::vector<bool> sawGlobalAfter;
  std::vector<frontend::StructuredEntityRef> profileBlocks;
  std::vector<std::uint64_t> blockActivationCounts;
  std::optional<std::string> error;
};

struct WorkloadCaptureValueShape final {
  std::uint64_t graphOrdinal = 0;
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  std::uint64_t byteCount = 0;
};

struct WorkloadCaptureActiveCall final {
  std::size_t captureIndex = 0;
  std::uint64_t nextRoot = 0;
  std::uint64_t nextValue = 0;
  std::uint64_t nextResult = 0;
  std::vector<std::uint64_t> runtimeObjectOrdinals;
  std::vector<std::uint8_t *> objectBases;
};

struct WorkloadCaptureContext final {
  NativeSimulationInputCapture result;
  std::vector<WorkloadCaptureValueShape> runtimeValueShapes;
  std::vector<WorkloadCaptureValueShape> valueResultShapes;
  std::vector<std::pair<std::uint8_t *, std::size_t>> runtimeObjects;
  std::vector<WorkloadCaptureActiveCall> activeCalls;
  std::uint64_t rootCount = 0;
  bool littleEndian = true;
  std::optional<std::error_code> errorCode;
  std::optional<std::string> error;
};

struct WorkloadCaptureCallbackNames final {
  std::string begin;
  std::string end;
  std::optional<std::string> registerObject;
  std::optional<std::string> memoryRoot;
  std::optional<std::string> value;
  std::optional<std::string> result;
};

thread_local NativeExecutionContext *activeExecution = nullptr;
thread_local WorkloadCaptureContext *activeWorkloadCapture = nullptr;

void recordExecutionError(llvm::StringRef message) {
  if (activeExecution && !activeExecution->error)
    activeExecution->error = message.str();
}

void *nativeRuntimeObject(std::uint64_t ordinal) {
  if (!activeExecution || ordinal >= activeExecution->objects.size()) {
    recordExecutionError("runtime object callback has an invalid ordinal");
    return nullptr;
  }
  return activeExecution->objects[ordinal].data();
}

void nativeReturnValue(void *base, std::uint64_t byteCount) {
  if (!activeExecution || !activeExecution->returnShape || !base ||
      byteCount != activeExecution->returnByteCount ||
      activeExecution->returnValue) {
    recordExecutionError("return callback has an invalid projection");
    return;
  }
  activeExecution->returnValue = detail::readDefinedNativeValue(
      llvm::ArrayRef<std::uint8_t>(static_cast<const std::uint8_t *>(base),
                                   static_cast<std::size_t>(byteCount)),
      activeExecution->returnShape->lanesPerToken,
      activeExecution->returnShape->laneBitWidth,
      activeExecution->littleEndian);
}

void copyGlobalBytes(std::vector<std::uint8_t> &destination, void *base,
                     std::uint64_t byteCount) {
  destination.resize(static_cast<std::size_t>(byteCount));
  if (byteCount != 0)
    std::memcpy(destination.data(), base, destination.size());
}

void nativeGlobalBefore(std::uint64_t targetOrdinal, void *base,
                        std::uint64_t byteCount) {
  if (!activeExecution ||
      targetOrdinal >= activeExecution->globalBefore.size() || !base ||
      activeExecution->sawGlobalBefore[targetOrdinal]) {
    recordExecutionError("global-before callback has an invalid projection");
    return;
  }
  copyGlobalBytes(activeExecution->globalBefore[targetOrdinal], base,
                  byteCount);
  activeExecution->sawGlobalBefore[targetOrdinal] = true;
}

void nativeGlobalAfter(std::uint64_t targetOrdinal, void *base,
                       std::uint64_t byteCount) {
  if (!activeExecution ||
      targetOrdinal >= activeExecution->globalAfter.size() || !base ||
      activeExecution->sawGlobalAfter[targetOrdinal] ||
      !activeExecution->sawGlobalBefore[targetOrdinal] ||
      activeExecution->globalBefore[targetOrdinal].size() != byteCount) {
    recordExecutionError("global-after callback has an invalid projection");
    return;
  }
  copyGlobalBytes(activeExecution->globalAfter[targetOrdinal], base, byteCount);
  activeExecution->sawGlobalAfter[targetOrdinal] = true;
}

void nativeBlockActivation(std::uint64_t ordinal) {
  if (!activeExecution ||
      ordinal >= activeExecution->blockActivationCounts.size()) {
    recordExecutionError("block activation callback has an invalid ordinal");
    return;
  }
  std::uint64_t &count = activeExecution->blockActivationCounts[ordinal];
  if (count == std::numeric_limits<std::uint64_t>::max()) {
    recordExecutionError("block activation count overflowed");
    return;
  }
  ++count;
}

void recordWorkloadCaptureError(std::error_code code, llvm::StringRef message) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  activeWorkloadCapture->errorCode = code;
  activeWorkloadCapture->error = message.str();
}

void copyWorkloadCaptureBytes(std::vector<std::uint8_t> &destination,
                              const std::uint8_t *base, std::size_t byteCount) {
  destination.resize(byteCount);
  if (byteCount != 0)
    std::memcpy(destination.data(), base, byteCount);
}

void workloadCaptureBegin() {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  activeWorkloadCapture->result.calls.emplace_back();
  WorkloadCaptureActiveCall active;
  active.captureIndex = activeWorkloadCapture->result.calls.size() - 1;
  activeWorkloadCapture->activeCalls.push_back(std::move(active));
}

void workloadCaptureRegisterObject(void *base, std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  if (!base || byteCount == 0 ||
      byteCount > std::numeric_limits<std::size_t>::max()) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::io_error),
        "program object registration has an invalid extent");
    return;
  }
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  for (auto &object : context.runtimeObjects) {
    if (object.first != base)
      continue;
    object.second = static_cast<std::size_t>(byteCount);
    return;
  }
  context.runtimeObjects.push_back(
      {static_cast<std::uint8_t *>(base), static_cast<std::size_t>(byteCount)});
}

std::optional<std::pair<std::uint64_t, std::uint64_t>>
resolveRuntimeObject(void *pointer) {
  if (!activeWorkloadCapture || !pointer)
    return std::nullopt;
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  for (std::size_t ordinal = activeWorkloadCapture->runtimeObjects.size();
       ordinal != 0; --ordinal) {
    const auto &object = activeWorkloadCapture->runtimeObjects[ordinal - 1];
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(object.first);
    if (address < base || address - base >= object.second)
      continue;
    return std::pair<std::uint64_t, std::uint64_t>{ordinal - 1, address - base};
  }
  return std::nullopt;
}

void workloadCaptureMemoryRoot(std::uint64_t rootOrdinal, void *pointer) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() || rootOrdinal >= context.rootCount ||
      context.activeCalls.back().nextRoot != rootOrdinal) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::io_error),
        "memory-root callbacks are not in canonical order");
    return;
  }
  std::optional<std::pair<std::uint64_t, std::uint64_t>> resolved =
      resolveRuntimeObject(pointer);
  if (!resolved) {
    recordWorkloadCaptureError(
        std::make_error_code(std::errc::not_supported),
        "selected memory root is not owned by the runtime object registry");
    return;
  }

  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  NativeSimulationCallCapture &capture =
      context.result.calls[active.captureIndex];
  auto found = llvm::find(active.runtimeObjectOrdinals, resolved->first);
  std::uint64_t objectOrdinal = 0;
  if (found == active.runtimeObjectOrdinals.end()) {
    objectOrdinal = capture.objects.size();
    active.runtimeObjectOrdinals.push_back(resolved->first);
    const auto &[base, byteCount] = context.runtimeObjects[resolved->first];
    active.objectBases.push_back(base);
    capture.objects.emplace_back();
    copyWorkloadCaptureBytes(capture.objects.back().initialBytes, base,
                             byteCount);
  } else {
    objectOrdinal = static_cast<std::uint64_t>(
        std::distance(active.runtimeObjectOrdinals.begin(), found));
  }
  capture.memoryRootObjectOrdinals.push_back(objectOrdinal);
  capture.memoryRootByteOffsets.push_back(resolved->second);
  ++active.nextRoot;
}

CanonicalValueSequence
readWorkloadCaptureValue(void *base, const WorkloadCaptureValueShape &shape,
                         bool littleEndian) {
  return detail::readDefinedNativeValue(
      llvm::ArrayRef<std::uint8_t>(static_cast<std::uint8_t *>(base),
                                   static_cast<std::size_t>(shape.byteCount)),
      shape.lanesPerToken, shape.laneBitWidth, littleEndian);
}

void workloadCaptureValue(std::uint64_t ordinal, void *base,
                          std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() ||
      ordinal >= context.runtimeValueShapes.size() || !base ||
      context.runtimeValueShapes[ordinal].byteCount != byteCount ||
      context.activeCalls.back().nextRoot != context.rootCount ||
      context.activeCalls.back().nextValue != ordinal) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "runtime value callback is malformed");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  const WorkloadCaptureValueShape &shape = context.runtimeValueShapes[ordinal];
  context.result.calls[active.captureIndex].runtimeValues.push_back(
      RuntimeValueEntry{
          shape.graphOrdinal,
          readWorkloadCaptureValue(base, shape, context.littleEndian)});
  ++active.nextValue;
}

void workloadCaptureResult(std::uint64_t ordinal, void *base,
                           std::uint64_t byteCount) {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty() ||
      ordinal >= context.valueResultShapes.size() || !base ||
      context.valueResultShapes[ordinal].byteCount != byteCount ||
      context.activeCalls.back().nextRoot != context.rootCount ||
      context.activeCalls.back().nextValue !=
          context.runtimeValueShapes.size() ||
      context.activeCalls.back().nextResult != ordinal) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "value result callback is malformed");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  context.result.calls[active.captureIndex].valueResults.push_back(
      readWorkloadCaptureValue(base, context.valueResultShapes[ordinal],
                               context.littleEndian));
  ++active.nextResult;
}

void workloadCaptureEnd() {
  if (!activeWorkloadCapture || activeWorkloadCapture->error)
    return;
  WorkloadCaptureContext &context = *activeWorkloadCapture;
  if (context.activeCalls.empty()) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "capture end has no active invocation");
    return;
  }
  WorkloadCaptureActiveCall &active = context.activeCalls.back();
  if (active.nextRoot != context.rootCount ||
      active.nextValue != context.runtimeValueShapes.size() ||
      active.nextResult != context.valueResultShapes.size()) {
    recordWorkloadCaptureError(std::make_error_code(std::errc::io_error),
                               "capture end observed an incomplete boundary");
    return;
  }
  NativeSimulationCallCapture &capture =
      context.result.calls[active.captureIndex];
  for (auto [ordinal, base] : llvm::enumerate(active.objectBases)) {
    const std::uint64_t runtimeOrdinal = active.runtimeObjectOrdinals[ordinal];
    copyWorkloadCaptureBytes(capture.objects[ordinal].finalBytes, base,
                             context.runtimeObjects[runtimeOrdinal].second);
  }
  context.activeCalls.pop_back();
}

std::string uniqueName(const llvm::Module &module, llvm::StringRef prefix) {
  std::string candidate = prefix.str();
  std::uint64_t suffix = 0;
  while (module.getNamedValue(candidate))
    candidate = (prefix + "." + llvm::Twine(++suffix)).str();
  return candidate;
}

std::string uniqueMlirSymbolName(mlir::ModuleOp module,
                                 llvm::StringRef prefix) {
  std::string candidate = prefix.str();
  std::uint64_t suffix = 0;
  while (mlir::SymbolTable::lookupSymbolIn(module, candidate))
    candidate = (prefix + "." + llvm::Twine(++suffix)).str();
  return candidate;
}

mlir::LLVM::LLVMFuncOp enclosingDefinedLlvmFunction(mlir::Block *block) {
  mlir::Operation *owner = block ? block->getParentOp() : nullptr;
  auto function = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(owner);
  if (!function && owner)
    function = owner->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  return function && !function.getBody().empty() ? function
                                                 : mlir::LLVM::LLVMFuncOp{};
}

llvm::Expected<std::string>
instrumentBlockActivations(mlir::ModuleOp module,
                           const ArtifactIdentity &identity,
                           NativeExecutionContext &capture) {
  llvm::Expected<frontend::StructuredProgramCandidateView> view =
      frontend::buildStructuredProgramCandidateView(module, identity);
  if (!view)
    return view.takeError();

  struct ProfileSite {
    frontend::StructuredEntityRef reference;
    mlir::Block *block = nullptr;
  };
  std::vector<ProfileSite> sites;
  for (const frontend::StructuredEntity &entity :
       view->entities(frontend::StructuredEntityKind::Block)) {
    if (enclosingDefinedLlvmFunction(entity.block))
      sites.push_back({entity.reference, entity.block});
  }

  const std::string callbackName =
      uniqueMlirSymbolName(module, "__loom_structured_block_activation");
  mlir::OpBuilder declarations(module.getContext());
  declarations.setInsertionPointToStart(module.getBody());
  const mlir::Type i64 = declarations.getI64Type();
  const mlir::Type callbackType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(module.getContext()), {i64});
  mlir::LLVM::LLVMFuncOp::create(declarations, module.getLoc(), callbackName,
                                 callbackType);

  capture.profileBlocks.reserve(sites.size());
  capture.blockActivationCounts.assign(sites.size(), 0);
  for (auto [ordinal, site] : llvm::enumerate(sites)) {
    capture.profileBlocks.push_back(site.reference);
    mlir::OpBuilder builder(module.getContext());
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
  return callbackName;
}

struct ProgramObjectCaptureSite final {
  mlir::Value base;
  std::uint64_t byteCount = 0;
  mlir::Operation *registration = nullptr;
};

std::optional<std::uint64_t> constantUnsignedValue(mlir::Value value) {
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attribute);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 64)
    return std::nullopt;
  return integer.getValue().getZExtValue();
}

std::optional<std::uint64_t>
fixedTypeByteCountForCapture(mlir::Operation *scope, mlir::Type type) {
  llvm::TypeSize bytes = mlir::DataLayout::closest(scope).getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return std::nullopt;
  return bytes.getFixedValue();
}

std::optional<std::uint64_t>
fixedAllocationByteCountForCapture(mlir::LLVM::AllocaOp allocation) {
  std::optional<std::uint64_t> count =
      constantUnsignedValue(allocation.getArraySize());
  std::optional<std::uint64_t> elementBytes = fixedTypeByteCountForCapture(
      allocation.getOperation(), allocation.getElemType());
  if (!count || *count == 0 || !elementBytes ||
      *count > std::numeric_limits<std::uint64_t>::max() / *elementBytes)
    return std::nullopt;
  return *count * *elementBytes;
}

llvm::Expected<std::vector<ProgramObjectCaptureSite>>
collectProgramObjectCaptureSites(mlir::ModuleOp module) {
  std::vector<mlir::LLVM::AllocaOp> allocations;
  std::vector<mlir::LLVM::AddressOfOp> addresses;
  module.walk([&](mlir::LLVM::AllocaOp allocation) {
    allocations.push_back(allocation);
  });
  module.walk(
      [&](mlir::LLVM::AddressOfOp address) { addresses.push_back(address); });

  std::vector<ProgramObjectCaptureSite> sites;
  sites.reserve(allocations.size() + addresses.size());
  for (mlir::LLVM::AllocaOp allocation : allocations) {
    std::optional<std::uint64_t> byteCount =
        fixedAllocationByteCountForCapture(allocation);
    if (byteCount)
      sites.push_back({allocation.getRes(), *byteCount, nullptr});
  }
  for (mlir::LLVM::AddressOfOp address : addresses) {
    auto global =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::GlobalOp>(
            address, address.getGlobalNameAttr());
    if (!global)
      continue;
    std::optional<std::uint64_t> byteCount = fixedTypeByteCountForCapture(
        global.getOperation(), global.getGlobalType());
    if (byteCount)
      sites.push_back({address.getResult(), *byteCount, nullptr});
  }
  return sites;
}

mlir::Value programObjectBase(mlir::Value pointer) {
  while (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>())
    pointer = gep.getBase();
  if (pointer.getDefiningOp<mlir::LLVM::AllocaOp>() ||
      pointer.getDefiningOp<mlir::LLVM::AddressOfOp>())
    return pointer;
  return {};
}

llvm::Expected<WorkloadCaptureCallbackNames> instrumentWorkloadBackedCapture(
    mlir::ModuleOp module, mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan) {
  if (!selectedOperation ||
      selectedOperation->getParentOfType<mlir::ModuleOp>() != module)
    return invalid("selected capture boundary is not owned by its module");
  auto selectedFunction =
      llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(selectedOperation);
  auto callable =
      selectedFunction
          ? selectedFunction
          : selectedOperation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!callable || callable.getBody().empty())
    return invalid("selected capture boundary has no LLVM callable");
  if (selectedFunction && !callable.getBody().hasOneBlock())
    return unsupported(
        "workload-backed callable capture requires one structured block");

  auto objectSites = collectProgramObjectCaptureSites(module);
  if (!objectSites)
    return objectSites.takeError();

  mlir::MLIRContext *context = module.getContext();
  mlir::Location location = selectedOperation->getLoc();
  mlir::OpBuilder declarations(context);
  declarations.setInsertionPointToStart(module.getBody());
  const mlir::Type i64 = declarations.getI64Type();
  const mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type voidType = mlir::LLVM::LLVMVoidType::get(context);
  const mlir::Type lifecycleType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {});
  const mlir::Type memoryRootType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {i64, pointer});
  const mlir::Type objectRegistrationType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {pointer, i64});
  const mlir::Type valueType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {i64, pointer, i64});

  WorkloadCaptureCallbackNames names;
  names.begin = uniqueMlirSymbolName(module, "__loom_workload_capture_begin");
  names.end = uniqueMlirSymbolName(module, "__loom_workload_capture_end");
  if (!objectSites->empty())
    names.registerObject =
        uniqueMlirSymbolName(module, "__loom_workload_capture_register_object");
  if (!plan.memoryRoots.empty())
    names.memoryRoot =
        uniqueMlirSymbolName(module, "__loom_workload_capture_memory_root");
  if (llvm::any_of(plan.valueInputs, [](const auto &input) {
        return !input.fixedValue.has_value();
      }))
    names.value = uniqueMlirSymbolName(module, "__loom_workload_capture_value");
  if (!plan.valueResults.empty())
    names.result =
        uniqueMlirSymbolName(module, "__loom_workload_capture_result");

  mlir::LLVM::LLVMFuncOp::create(declarations, location, names.begin,
                                 lifecycleType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, names.end,
                                 lifecycleType);
  if (names.registerObject)
    mlir::LLVM::LLVMFuncOp::create(
        declarations, location, *names.registerObject, objectRegistrationType);
  if (names.memoryRoot)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.memoryRoot,
                                   memoryRootType);
  if (names.value)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.value,
                                   valueType);
  if (names.result)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.result,
                                   valueType);

  if (names.registerObject) {
    for (ProgramObjectCaptureSite &site : *objectSites) {
      mlir::OpBuilder registration(context);
      registration.setInsertionPointAfter(site.base.getDefiningOp());
      mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
          registration, site.base.getLoc(), i64,
          registration.getI64IntegerAttr(site.byteCount));
      auto call = mlir::LLVM::CallOp::create(
          registration, site.base.getLoc(), mlir::TypeRange{},
          *names.registerObject, mlir::ValueRange{site.base, byteCount});
      site.registration = call.getOperation();
    }
  }

  mlir::OpBuilder storage(context);
  storage.setInsertionPointToStart(&callable.getBody().front());
  mlir::Value one;
  mlir::Operation *lastStorage = nullptr;
  auto allocateSlot = [&](mlir::Type type) {
    if (!one) {
      one = mlir::LLVM::ConstantOp::create(storage, location, i64,
                                           storage.getI64IntegerAttr(1));
      lastStorage = one.getDefiningOp();
    }
    mlir::Value slot =
        mlir::LLVM::AllocaOp::create(storage, location, pointer, type, one)
            .getRes();
    lastStorage = slot.getDefiningOp();
    return slot;
  };

  std::vector<mlir::Value> inputSlots(plan.valueInputs.size());
  for (auto [ordinal, input] : llvm::enumerate(plan.valueInputs))
    if (!input.fixedValue)
      inputSlots[ordinal] = allocateSlot(input.boundaryValue.getType());
  std::vector<mlir::Value> resultSlots;
  resultSlots.reserve(plan.valueResults.size());
  for (const SimulationValueResultCapture &result : plan.valueResults)
    resultSlots.push_back(allocateSlot(result.boundaryValue.getType()));

  mlir::OpBuilder before(context);
  if (selectedFunction) {
    mlir::Operation *lastPrelude = lastStorage;
    mlir::Block *entry = &callable.getBody().front();
    auto includePrelude = [&](mlir::Operation *operation) {
      if (!operation || operation->getBlock() != entry)
        return;
      if (!lastPrelude || lastPrelude->isBeforeInBlock(operation))
        lastPrelude = operation;
    };
    for (const SimulationValueInputCapture &input : plan.valueInputs)
      includePrelude(input.boundaryValue.getDefiningOp());
    for (const WorkloadBackedMemoryRootCapture &root : plan.memoryRoots) {
      includePrelude(root.boundaryPointer.getDefiningOp());
      mlir::Value base = programObjectBase(root.boundaryPointer);
      if (!base)
        continue;
      for (const ProgramObjectCaptureSite &site : *objectSites)
        if (site.base == base)
          includePrelude(site.registration);
    }
    if (lastPrelude)
      before.setInsertionPointAfter(lastPrelude);
    else
      before.setInsertionPointToStart(entry);
  } else {
    before.setInsertionPoint(selectedOperation);
  }
  mlir::LLVM::CallOp::create(before, location, mlir::TypeRange{}, names.begin,
                             mlir::ValueRange{});
  for (auto [ordinal, root] : llvm::enumerate(plan.memoryRoots)) {
    if (!names.memoryRoot || !root.boundaryPointer ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(root.boundaryPointer.getType()))
      return invalid("memory root has no pointer-valued selected boundary");
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(ordinal));
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, *names.memoryRoot,
        mlir::ValueRange{ordinalValue, root.boundaryPointer});
  }
  std::uint64_t runtimeOrdinal = 0;
  for (auto [ordinal, input] : llvm::enumerate(plan.valueInputs)) {
    if (input.fixedValue)
      continue;
    if (!names.value || !input.boundaryValue || input.byteCount == 0)
      return invalid("runtime graph input has no finite selected boundary");
    mlir::LLVM::StoreOp::create(before, location, input.boundaryValue,
                                inputSlots[ordinal]);
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(runtimeOrdinal));
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(input.byteCount));
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, *names.value,
        mlir::ValueRange{ordinalValue, inputSlots[ordinal], byteCount});
    ++runtimeOrdinal;
  }

  auto emitResultsAndEnd = [&](mlir::OpBuilder &after) -> llvm::Error {
    for (auto [ordinal, result] : llvm::enumerate(plan.valueResults)) {
      if (!names.result || !result.boundaryValue || result.byteCount == 0)
        return invalid("graph result has no finite selected boundary");
      mlir::LLVM::StoreOp::create(after, location, result.boundaryValue,
                                  resultSlots[ordinal]);
      mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
          after, location, i64, after.getI64IntegerAttr(ordinal));
      mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
          after, location, i64, after.getI64IntegerAttr(result.byteCount));
      mlir::LLVM::CallOp::create(
          after, location, mlir::TypeRange{}, *names.result,
          mlir::ValueRange{ordinalValue, resultSlots[ordinal], byteCount});
    }
    mlir::LLVM::CallOp::create(after, location, mlir::TypeRange{}, names.end,
                               mlir::ValueRange{});
    return llvm::Error::success();
  };

  if (selectedFunction) {
    auto returnOp = llvm::dyn_cast<mlir::LLVM::ReturnOp>(
        callable.getBody().front().getTerminator());
    if (!returnOp)
      return unsupported(
          "workload-backed callable capture has no direct return");
    mlir::OpBuilder after(returnOp);
    if (llvm::Error error = emitResultsAndEnd(after))
      return std::move(error);
  } else {
    mlir::OpBuilder after(selectedOperation);
    after.setInsertionPointAfter(selectedOperation);
    if (llvm::Error error = emitResultsAndEnd(after))
      return std::move(error);
  }
  if (mlir::failed(mlir::verify(module)))
    return invalid("workload-backed capture instrumentation does not verify");
  return names;
}

llvm::Align requiredRuntimeObjectAlignment(const llvm::Module &module) {
  llvm::Align result(alignof(std::max_align_t));
  auto include = [&](llvm::MaybeAlign alignment) {
    if (alignment && alignment->value() > result.value())
      result = *alignment;
  };

  for (const llvm::Function &function : module) {
    for (unsigned ordinal = 0; ordinal < function.arg_size(); ++ordinal)
      include(function.getParamAlign(ordinal));
    for (const llvm::Instruction &instruction : llvm::instructions(function)) {
      if (const auto *load = llvm::dyn_cast<llvm::LoadInst>(&instruction))
        include(load->getAlign());
      else if (const auto *store =
                   llvm::dyn_cast<llvm::StoreInst>(&instruction))
        include(store->getAlign());
      else if (const auto *rmw =
                   llvm::dyn_cast<llvm::AtomicRMWInst>(&instruction))
        include(rmw->getAlign());
      else if (const auto *compare =
                   llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction))
        include(compare->getAlign());
      else if (const auto *memory =
                   llvm::dyn_cast<llvm::MemIntrinsic>(&instruction)) {
        include(memory->getDestAlign());
        if (const auto *transfer =
                llvm::dyn_cast<llvm::MemTransferInst>(memory))
          include(transfer->getSourceAlign());
      } else if (const auto *call =
                     llvm::dyn_cast<llvm::CallBase>(&instruction)) {
        for (unsigned ordinal = 0; ordinal < call->arg_size(); ++ordinal)
          include(call->getParamAlign(ordinal));
      }
    }
  }
  return result;
}

llvm::Expected<std::vector<std::uint8_t>>
definedMemoryBytes(const RuntimeMemoryObject &object) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(object.initialBytes.size());
  for (const SemanticMemoryByte &byte : object.initialBytes) {
    if (byte.state != SemanticState::Defined)
      return unsupported(
          "native execution requires Defined runtime memory bytes");
    bytes.push_back(byte.value);
  }
  return bytes;
}

llvm::Expected<llvm::Constant *>
definedScalarConstant(llvm::Type *type, const SemanticLane &lane) {
  if (lane.state != SemanticState::Defined)
    return unsupported(
        "native execution requires Defined scalar and vector inputs");
  if (auto *integer = llvm::dyn_cast<llvm::IntegerType>(type)) {
    if (integer->getBitWidth() != lane.bits.getBitWidth())
      return invalid("entry integer width differs from the workload ABI");
    return llvm::ConstantInt::get(integer, lane.bits);
  }
  if (type->isFloatingPointTy()) {
    if (type->getPrimitiveSizeInBits() != lane.bits.getBitWidth())
      return invalid("entry floating width differs from the workload ABI");
    return llvm::ConstantFP::get(
        type->getContext(), llvm::APFloat(type->getFltSemantics(), lane.bits));
  }
  return unsupported("entry value type has no native constant provider");
}

llvm::Expected<llvm::Constant *>
definedValueConstant(llvm::Type *type, const CanonicalValueSequence &sequence) {
  if (sequence.tokenCount != 1)
    return invalid("entry value input does not contain exactly one token");
  if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    if (sequence.lanes.size() != vector->getNumElements())
      return invalid("entry vector lane count differs from the workload ABI");
    std::vector<llvm::Constant *> elements;
    elements.reserve(sequence.lanes.size());
    for (const SemanticLane &lane : sequence.lanes) {
      llvm::Expected<llvm::Constant *> element =
          definedScalarConstant(vector->getElementType(), lane);
      if (!element)
        return element.takeError();
      elements.push_back(*element);
    }
    return llvm::ConstantVector::get(elements);
  }
  if (sequence.lanes.size() != 1)
    return invalid("entry scalar input has a non-scalar lane sequence");
  return definedScalarConstant(type, sequence.lanes.front());
}

const StructuredRuntimeValueEntry *
findRuntimeValue(const StructuredProgramSimulationRuntimeInput &input,
                 std::uint64_t argumentOrdinal) {
  auto found = llvm::lower_bound(
      input.runtimeValues, argumentOrdinal,
      [](const StructuredRuntimeValueEntry &entry, std::uint64_t ordinal) {
        return entry.argumentOrdinal < ordinal;
      });
  return found != input.runtimeValues.end() &&
                 found->argumentOrdinal == argumentOrdinal
             ? &*found
             : nullptr;
}

const StructuredPointerBindingEntry *
findPointerBinding(const StructuredProgramSimulationRuntimeInput &input,
                   std::uint64_t argumentOrdinal) {
  auto found = llvm::lower_bound(
      input.pointerBindings, argumentOrdinal,
      [](const StructuredPointerBindingEntry &entry, std::uint64_t ordinal) {
        return entry.argumentOrdinal < ordinal;
      });
  return found != input.pointerBindings.end() &&
                 found->argumentOrdinal == argumentOrdinal
             ? &*found
             : nullptr;
}

llvm::Expected<const CanonicalValueSequence *>
valueForArgument(const StructuredProgramSimulationWorkload &workload,
                 const StructuredProgramSimulationRuntimeInput &input,
                 std::uint64_t argumentOrdinal) {
  const StructuredProgramArgumentSource &source =
      workload.argumentPlan[argumentOrdinal];
  if (const auto *fixed = std::get_if<CanonicalValueSequence>(&source))
    return fixed;
  if (std::holds_alternative<StructuredRuntimeValueInput>(source)) {
    const StructuredRuntimeValueEntry *runtime =
        findRuntimeValue(input, argumentOrdinal);
    if (!runtime)
      return invalid("runtime value table is not total over the workload");
    return &runtime->value;
  }
  return invalid("a memory argument was requested as a value");
}

llvm::Value *castPointerTo(llvm::IRBuilder<> &builder, llvm::Value *pointer,
                           llvm::PointerType *target) {
  auto *source = llvm::cast<llvm::PointerType>(pointer->getType());
  if (source->getAddressSpace() == target->getAddressSpace())
    return pointer;
  return builder.CreateAddrSpaceCast(pointer, target);
}

llvm::Expected<std::uint64_t> fixedStoreBytes(const llvm::DataLayout &layout,
                                              llvm::Type *type,
                                              llvm::StringRef what) {
  const llvm::TypeSize size = layout.getTypeStoreSize(type);
  if (size.isScalable() || size.getFixedValue() == 0)
    return unsupported(what + " has no fixed nonzero native storage size");
  return size.getFixedValue();
}

std::vector<SemanticMemoryByte>
definedSemanticBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<SemanticMemoryByte> result;
  result.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    result.push_back({SemanticState::Defined, byte});
  return result;
}

bool sameByte(const SemanticMemoryByte &lhs, const SemanticMemoryByte &rhs) {
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
    if (sameByte(baseline[offset], final[offset])) {
      ++offset;
      continue;
    }
    MemoryDiffRun run;
    run.byteOffset = offset;
    do {
      run.changedBytes.push_back(final[offset]);
      ++offset;
    } while (offset < final.size() &&
             !sameByte(baseline[offset], final[offset]));
    diff.runs.push_back(std::move(run));
  }
  return diff;
}

llvm::Expected<std::vector<MemoryTargetPlan>>
buildMemoryPlans(const StructuredProgramSimulationWorkload &workload,
                 const StructuredProgramSimulationRuntimeInput &input,
                 const frontend::StructuredProgramCandidateView &view) {
  std::vector<MemoryTargetPlan> plans;
  plans.reserve(workload.observableContract.memories.size());
  for (const StructuredProgramMemoryObservable &observable :
       workload.observableContract.memories) {
    MemoryTargetPlan plan;
    plan.form = observable.form;
    if (const auto *argument =
            std::get_if<EntryPointerArgumentTarget>(&observable.target)) {
      const StructuredPointerBindingEntry *binding =
          findPointerBinding(input, argument->argumentOrdinal);
      if (!binding ||
          binding->binding.objectOrdinal >= input.memoryObjects.size())
        return invalid("memory observable has no exact runtime object");
      plan.objectOrdinal = binding->binding.objectOrdinal;
      plan.byteCount = input.memoryObjects[binding->binding.objectOrdinal]
                           .initialBytes.size();
    } else {
      const frontend::StructuredEntityRef &reference =
          std::get<GlobalObjectTarget>(observable.target).global;
      llvm::Expected<frontend::StructuredEntity> entity =
          view.resolve(reference);
      if (!entity)
        return entity.takeError();
      auto global =
          llvm::dyn_cast_or_null<mlir::LLVM::GlobalOp>(entity->operation);
      if (!global)
        return invalid(
            "global observable does not resolve to llvm.mlir.global");
      plan.globalSymbol = global.getSymName().str();
    }
    plans.push_back(std::move(plan));
  }
  return plans;
}

struct CallbackNames {
  std::string wrapper;
  std::string runtimeObject;
  std::optional<std::string> blockActivation;
  std::optional<std::string> returnValue;
  std::optional<std::string> globalBefore;
  std::optional<std::string> globalAfter;
};

llvm::Expected<CallbackNames> instrumentExecution(
    llvm::Module &module, llvm::StringRef entrySymbol,
    const StructuredProgramSimulationWorkload &workload,
    const StructuredProgramSimulationRuntimeInput &input,
    const detail::ResolvedStructuredProgramContext &sourceContext,
    std::vector<MemoryTargetPlan> &memoryPlans,
    NativeExecutionContext &capture) {
  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration())
    return invalid("exact entry is absent after Structured lowering");
  if (entry->isVarArg())
    return unsupported("variadic Structured entries lack a workload ABI");
  if (entry->arg_size() != workload.argumentPlan.size())
    return invalid("lowered entry arguments differ from the exact workload");

  const llvm::DataLayout &layout = module.getDataLayout();
  const llvm::Align objectAlignment = requiredRuntimeObjectAlignment(module);
  capture.objects.reserve(input.memoryObjects.size());
  for (const RuntimeMemoryObject &object : input.memoryObjects) {
    llvm::Expected<std::vector<std::uint8_t>> bytes =
        definedMemoryBytes(object);
    if (!bytes)
      return bytes.takeError();
    llvm::Expected<AlignedByteStorage> storage =
        AlignedByteStorage::create(*bytes, objectAlignment);
    if (!storage)
      return storage.takeError();
    capture.objects.push_back(std::move(*storage));
  }
  capture.littleEndian = layout.isLittleEndian();
  capture.globalBefore.resize(memoryPlans.size());
  capture.globalAfter.resize(memoryPlans.size());
  capture.sawGlobalBefore.resize(memoryPlans.size());
  capture.sawGlobalAfter.resize(memoryPlans.size());

  CallbackNames names;
  names.wrapper = uniqueName(module, "__loom_structured_entry");
  names.runtimeObject = uniqueName(module, "__loom_structured_runtime_object");
  if (workload.observableContract.returnValue)
    names.returnValue = uniqueName(module, "__loom_structured_return_value");
  if (llvm::any_of(memoryPlans, [](const MemoryTargetPlan &plan) {
        return !plan.globalSymbol.empty();
      })) {
    names.globalBefore = uniqueName(module, "__loom_structured_global_before");
    names.globalAfter = uniqueName(module, "__loom_structured_global_after");
  }

  llvm::LLVMContext &context = module.getContext();
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  llvm::Type *i64Type = llvm::Type::getInt64Ty(context);
  llvm::PointerType *pointerType = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *objectCallbackType =
      llvm::FunctionType::get(pointerType, {i64Type}, false);
  llvm::FunctionCallee objectCallback =
      module.getOrInsertFunction(names.runtimeObject, objectCallbackType);

  llvm::FunctionCallee returnCallback;
  if (names.returnValue) {
    llvm::FunctionType *type =
        llvm::FunctionType::get(voidType, {pointerType, i64Type}, false);
    returnCallback = module.getOrInsertFunction(*names.returnValue, type);
  }

  llvm::FunctionCallee beforeCallback;
  llvm::FunctionCallee afterCallback;
  if (names.globalBefore) {
    llvm::FunctionType *type = llvm::FunctionType::get(
        voidType, {i64Type, pointerType, i64Type}, false);
    beforeCallback = module.getOrInsertFunction(*names.globalBefore, type);
    afterCallback = module.getOrInsertFunction(*names.globalAfter, type);
  }

  llvm::Function *wrapper = llvm::Function::Create(
      llvm::FunctionType::get(voidType, false),
      llvm::GlobalValue::ExternalLinkage, names.wrapper, module);
  llvm::BasicBlock *block = llvm::BasicBlock::Create(context, "entry", wrapper);
  llvm::IRBuilder<> builder(block);

  std::vector<llvm::Value *> arguments;
  arguments.reserve(entry->arg_size());
  for (std::uint64_t ordinal = 0; ordinal < entry->arg_size(); ++ordinal) {
    llvm::Type *type = entry->getFunctionType()->getParamType(ordinal);
    if (auto *pointer = llvm::dyn_cast<llvm::PointerType>(type)) {
      const StructuredPointerBindingEntry *binding =
          findPointerBinding(input, ordinal);
      if (!binding)
        return invalid("entry pointer has no exact runtime binding");
      llvm::Value *objectOrdinal =
          llvm::ConstantInt::get(i64Type, binding->binding.objectOrdinal);
      llvm::Value *base = builder.CreateCall(objectCallback, {objectOrdinal});
      llvm::Value *view = builder.CreateConstGEP1_64(
          llvm::Type::getInt8Ty(context), base, binding->binding.byteOffset);
      arguments.push_back(castPointerTo(builder, view, pointer));
      continue;
    }
    llvm::Expected<const CanonicalValueSequence *> sequence =
        valueForArgument(workload, input, ordinal);
    if (!sequence)
      return sequence.takeError();
    llvm::Expected<llvm::Constant *> value =
        definedValueConstant(type, **sequence);
    if (!value)
      return value.takeError();
    arguments.push_back(*value);
  }

  for (std::uint64_t ordinal = 0; ordinal < memoryPlans.size(); ++ordinal) {
    MemoryTargetPlan &plan = memoryPlans[ordinal];
    if (plan.globalSymbol.empty())
      continue;
    llvm::GlobalVariable *global =
        module.getGlobalVariable(plan.globalSymbol, true);
    if (!global || global->isDeclaration())
      return unsupported("observed global has no native storage provider");
    llvm::Expected<std::uint64_t> bytes =
        fixedStoreBytes(layout, global->getValueType(), "observed global");
    if (!bytes)
      return bytes.takeError();
    plan.byteCount = *bytes;
    llvm::Value *pointer = castPointerTo(builder, global, pointerType);
    builder.CreateCall(beforeCallback,
                       {llvm::ConstantInt::get(i64Type, ordinal), pointer,
                        llvm::ConstantInt::get(i64Type, plan.byteCount)});
  }

  llvm::CallInst *call = builder.CreateCall(entry, arguments);
  call->setCallingConv(entry->getCallingConv());
  call->setAttributes(entry->getAttributes());
  if (workload.observableContract.returnValue) {
    if (entry->getReturnType()->isVoidTy() || !sourceContext.returnShape)
      return invalid("selected return observation has no concrete result");
    llvm::Expected<std::uint64_t> bytes = fixedStoreBytes(
        layout, entry->getReturnType(), "Structured entry return");
    if (!bytes)
      return bytes.takeError();
    capture.returnShape = sourceContext.returnShape;
    capture.returnByteCount = *bytes;
    llvm::AllocaInst *storage = builder.CreateAlloca(entry->getReturnType());
    storage->setAlignment(layout.getABITypeAlign(entry->getReturnType()));
    builder.CreateStore(call, storage);
    builder.CreateCall(returnCallback,
                       {storage, llvm::ConstantInt::get(i64Type, *bytes)});
  }

  for (std::uint64_t ordinal = 0; ordinal < memoryPlans.size(); ++ordinal) {
    const MemoryTargetPlan &plan = memoryPlans[ordinal];
    if (plan.globalSymbol.empty())
      continue;
    llvm::GlobalVariable *global =
        module.getGlobalVariable(plan.globalSymbol, true);
    llvm::Value *pointer = castPointerTo(builder, global, pointerType);
    builder.CreateCall(afterCallback,
                       {llvm::ConstantInt::get(i64Type, ordinal), pointer,
                        llvm::ConstantInt::get(i64Type, plan.byteCount)});
  }
  builder.CreateRetVoid();
  if (llvm::verifyModule(module, &llvm::errs()))
    return invalid("instrumented Structured execution module does not verify");
  return names;
}

llvm::Expected<WorkloadCaptureContext> prepareWorkloadCaptureContext(
    const WorkloadBackedSimulationInputCapturePlan &plan,
    NativeExecutionContext &execution) {
  WorkloadCaptureContext capture;
  capture.rootCount = plan.memoryRoots.size();
  capture.littleEndian = execution.littleEndian;
  capture.runtimeObjects.reserve(execution.objects.size());
  for (AlignedByteStorage &object : execution.objects)
    capture.runtimeObjects.push_back({object.data(), object.size()});
  for (const SimulationValueInputCapture &input : plan.valueInputs) {
    if (input.valueInputOrdinal >= plan.valueInputs.size())
      return invalid("graph value input ordinal is outside its capture plan");
    if (!input.fixedValue)
      capture.runtimeValueShapes.push_back(WorkloadCaptureValueShape{
          input.valueInputOrdinal, input.lanesPerToken, input.laneBitWidth,
          input.byteCount});
  }
  capture.valueResultShapes.reserve(plan.valueResults.size());
  for (auto [ordinal, result] : llvm::enumerate(plan.valueResults)) {
    if (result.valueResultOrdinal != ordinal)
      return invalid("graph value results are not dense in ABI order");
    capture.valueResultShapes.push_back(WorkloadCaptureValueShape{
        result.valueResultOrdinal, result.lanesPerToken, result.laneBitWidth,
        result.byteCount});
  }
  return capture;
}

llvm::Error runInstrumentedExecution(
    llvm::orc::ThreadSafeModule module, const CallbackNames &names,
    NativeExecutionContext &capture, std::unique_ptr<llvm::orc::LLJIT> jit,
    const WorkloadCaptureCallbackNames *workloadCaptureNames = nullptr,
    WorkloadCaptureContext *workloadCapture = nullptr) {
  if ((workloadCaptureNames == nullptr) != (workloadCapture == nullptr))
    return invalid("workload capture callback state is partial");
  llvm::orc::JITDylib &dylib = jit->getMainJITDylib();
  if (llvm::Expected<std::unique_ptr<llvm::orc::DynamicLibrarySearchGenerator>>
          generator =
              llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                  jit->getDataLayout().getGlobalPrefix()))
    dylib.addGenerator(std::move(*generator));
  else
    return classifyJitError(generator.takeError(),
                            "cannot bind host process symbols");

  llvm::orc::SymbolMap callbacks;
  callbacks[jit->mangleAndIntern(names.runtimeObject)] = {
      llvm::orc::ExecutorAddr::fromPtr(&nativeRuntimeObject),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.blockActivation)
    callbacks[jit->mangleAndIntern(*names.blockActivation)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeBlockActivation),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.returnValue)
    callbacks[jit->mangleAndIntern(*names.returnValue)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeReturnValue),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (names.globalBefore) {
    callbacks[jit->mangleAndIntern(*names.globalBefore)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeGlobalBefore),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(*names.globalAfter)] = {
        llvm::orc::ExecutorAddr::fromPtr(&nativeGlobalAfter),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  }
  if (workloadCaptureNames) {
    callbacks[jit->mangleAndIntern(workloadCaptureNames->begin)] = {
        llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureBegin),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(workloadCaptureNames->end)] = {
        llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureEnd),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->registerObject)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->registerObject)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureRegisterObject),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->memoryRoot)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->memoryRoot)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureMemoryRoot),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->value)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->value)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureValue),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    if (workloadCaptureNames->result)
      callbacks[jit->mangleAndIntern(*workloadCaptureNames->result)] = {
          llvm::orc::ExecutorAddr::fromPtr(&workloadCaptureResult),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  }
  if (llvm::Error error =
          dylib.define(llvm::orc::absoluteSymbols(std::move(callbacks))))
    return classifyJitError(std::move(error),
                            "cannot bind execution callbacks");
  if (llvm::Error error = jit->addIRModule(std::move(module)))
    return classifyJitError(std::move(error),
                            "cannot add the Structured execution module");
  if (llvm::Error error = jit->initialize(dylib))
    return classifyJitError(std::move(error),
                            "cannot initialize the Structured program");
  llvm::Expected<llvm::orc::ExecutorAddr> wrapper = jit->lookup(names.wrapper);
  if (!wrapper)
    return classifyJitError(wrapper.takeError(),
                            "cannot materialize the Structured entry");
  if (activeExecution || activeWorkloadCapture)
    return executionFailed("nested native Structured execution is unsupported");
  activeExecution = &capture;
  activeWorkloadCapture = workloadCapture;
  using Wrapper = void();
  wrapper->toPtr<Wrapper>()();
  activeWorkloadCapture = nullptr;
  activeExecution = nullptr;
  if (llvm::Error error = jit->deinitialize(dylib))
    return classifyJitError(std::move(error),
                            "cannot deinitialize the Structured program");
  if (capture.error)
    return executionFailed(*capture.error);
  if (workloadCapture) {
    if (workloadCapture->error) {
      const std::error_code code = workloadCapture->errorCode.value_or(
          std::make_error_code(std::errc::io_error));
      return llvm::createStringError(code, "native_workload_capture: %s",
                                     workloadCapture->error->c_str());
    }
    if (!workloadCapture->activeCalls.empty())
      return executionFailed(
          "workload-backed capture left an incomplete invocation");
    workloadCapture->result.entryResult = 0;
  }
  return llvm::Error::success();
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
  if (capture.profileBlocks.size() != capture.blockActivationCounts.size())
    return executionFailed("block activation projection is inconsistent");
  result.blockActivations.reserve(capture.profileBlocks.size());
  for (std::size_t ordinal = 0; ordinal < capture.profileBlocks.size();
       ++ordinal)
    result.blockActivations.push_back({capture.profileBlocks[ordinal],
                                       capture.blockActivationCounts[ordinal]});
  return result;
}

llvm::Error inlineSpatialOwnershipCarriers(mlir::ModuleOp module) {
  llvm::SmallVector<loom::SpatialRegionOp> regions;
  module.walk([&](loom::SpatialRegionOp region) { regions.push_back(region); });
  for (loom::SpatialRegionOp region : llvm::reverse(regions)) {
    if (!region.getStreamInputs().empty() || !region.getStreamOutputs().empty())
      return unsupported(
          "native selected execution does not support stream ownership "
          "carriers");
    if (!region.getBody().hasOneBlock())
      return invalid("selected spatial ownership carrier is not single-block");
    mlir::Block &body = region.getBody().front();
    auto yield = llvm::dyn_cast<loom::SpatialYieldOp>(body.getTerminator());
    if (!yield)
      return invalid("selected spatial ownership carrier has no typed yield");
    if (body.getNumArguments() != region->getNumOperands() ||
        yield->getNumOperands() != region->getNumResults())
      return invalid("selected spatial ownership carrier boundary is not "
                     "positional");

    mlir::IRMapping mapping;
    for (auto [argument, operand] :
         llvm::zip_equal(body.getArguments(), region->getOperands()))
      mapping.map(argument, operand);
    mlir::OpBuilder builder(region);
    for (mlir::Operation &operation : body.without_terminator())
      builder.clone(operation, mapping);

    llvm::SmallVector<mlir::Value> results;
    results.reserve(yield->getNumOperands());
    for (mlir::Value value : yield->getOperands())
      results.push_back(mapping.lookupOrDefault(value));
    region->replaceAllUsesWith(results);
    region.erase();
  }
  return llvm::Error::success();
}

llvm::Error inlineRankZeroThreadOwnershipCarriers(mlir::ModuleOp module) {
  llvm::SmallVector<dataflow::ThreadLaunchOp> launches;
  module.walk(
      [&](dataflow::ThreadLaunchOp launch) { launches.push_back(launch); });
  for (dataflow::ThreadLaunchOp launch : launches) {
    if (!launch.getGridUpperBounds().empty())
      return unsupported(
          "native selected execution does not yet project a dense thread "
          "domain");
    if (!launch.getAsyncDependencies().empty())
      return unsupported(
          "native selected execution does not project asynchronous thread "
          "dependencies");
    if (!launch.getAsyncToken().hasOneUse())
      return unsupported(
          "native selected execution requires one exact thread wait");
    auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
        *launch.getAsyncToken().getUsers().begin());
    if (!wait || wait->getNumOperands() != 1 ||
        launch->getNextNode() != wait.getOperation())
      return unsupported(
          "native selected execution requires an immediately joined thread "
          "launch");

    auto thread =
        mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
            launch, launch.getCalleeAttr());
    if (!thread || thread.isExternal())
      return invalid("selected thread launch has no exact definition");
    if (thread.getDomain().getKind() !=
        dataflow::ThreadDomainKind::DenseRectangular)
      return unsupported(
          "native selected execution does not support dynamic-work threads");
    mlir::Block &body = thread.getBody().front();
    const std::size_t inputCount = thread.getFunctionType().getNumInputs();
    if (body.getNumArguments() != inputCount + 1 ||
        launch.getBodyOperands().size() != inputCount)
      return invalid("selected rank-zero thread boundary is malformed");
    if (!body.getArgument(inputCount).use_empty())
      return unsupported(
          "native selected execution cannot erase a used thread control "
          "token");
    auto yield = llvm::dyn_cast<dataflow::ThreadYieldOp>(body.getTerminator());
    if (!yield || !yield.getCompletionFrontier().empty())
      return unsupported(
          "native selected execution cannot erase a completion frontier");

    mlir::IRMapping mapping;
    for (auto [argument, operand] :
         llvm::zip_equal(body.getArguments().take_front(inputCount),
                         launch.getBodyOperands()))
      mapping.map(argument, operand);
    mlir::OpBuilder builder(launch);
    for (mlir::Operation &operation : body.without_terminator())
      builder.clone(operation, mapping);
    wait.erase();
    launch.erase();
  }

  bool residualLaunch = false;
  module.walk([&](dataflow::ThreadLaunchOp) { residualLaunch = true; });
  if (residualLaunch)
    return invalid("selected thread projection left a residual launch");
  llvm::SmallVector<dataflow::ThreadOp> threads;
  module.walk([&](dataflow::ThreadOp thread) { threads.push_back(thread); });
  for (dataflow::ThreadOp thread : threads)
    thread.erase();
  return llvm::Error::success();
}

llvm::Error projectSelectedWholeProgram(mlir::ModuleOp module) {
  if (llvm::Error error = inlineSpatialOwnershipCarriers(module))
    return error;
  if (llvm::Error error = inlineRankZeroThreadOwnershipCarriers(module))
    return error;
  bool residualCarrier = false;
  module.walk([&](mlir::Operation *operation) {
    residualCarrier |=
        llvm::isa<loom::SpatialRegionOp, loom::SpatialYieldOp,
                  dataflow::ThreadOp, dataflow::ThreadLaunchOp,
                  dataflow::ThreadWaitOp, dataflow::ThreadYieldOp>(operation);
  });
  if (residualCarrier)
    return invalid("selected whole-program projection left an ownership "
                   "carrier");
  if (mlir::failed(mlir::verify(module)))
    return invalid("selected whole-program projection does not verify");
  return llvm::Error::success();
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

bool sameMemoryByte(const SemanticMemoryByte &lhs,
                    const SemanticMemoryByte &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.value == rhs.value);
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

struct NativeProgramExecutionResult final {
  NativeStructuredProgramObservations observations;
  std::optional<NativeSimulationInputCapture> boundaryCapture;
};

llvm::Expected<NativeProgramExecutionResult> executePreparedProgramModule(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    bool profileSourceBlocks, bool projectOwnership,
    mlir::Operation *selectedOperation = nullptr,
    const WorkloadBackedSimulationInputCapturePlan *capturePlan = nullptr) {
  if (!module)
    return invalid("native execution has no Structured module");
  if ((selectedOperation == nullptr) != (capturePlan == nullptr))
    return invalid("workload-backed capture plan is partial");
  if (projectOwnership && capturePlan)
    return invalid("capture cannot instrument a projected ownership carrier");
  const StructuredProgramSimulationWorkload *structured =
      workload.structuredProgram();
  const StructuredProgramSimulationRuntimeInput *input =
      runtimeInput.structuredProgram();
  if (!structured || !input)
    return invalid("native execution requires Structured workload roots");

  auto sourceView = sourceProgram.view();
  if (!sourceView)
    return sourceView.takeError();
  auto verifiedWorkload = importSimulationWorkload(
      workload.canonicalBytes().bytes(), *sourceView, workload.identity());
  if (!verifiedWorkload)
    return verifiedWorkload.takeError();
  auto verifiedInput = importSimulationRuntimeInput(
      runtimeInput.canonicalBytes().bytes(), workload, *sourceView,
      runtimeInput.identity());
  if (!verifiedInput)
    return verifiedInput.takeError();
  auto sourceContext = detail::resolveStructuredProgramContext(
      *sourceView, structured->entryRef);
  if (!sourceContext)
    return sourceContext.takeError();
  auto entry = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(sourceContext->entryOp);
  if (!entry)
    return invalid("exact workload entry is not llvm.func");
  auto plans = buildMemoryPlans(*structured, *input, *sourceView);
  if (!plans)
    return plans.takeError();

  if (projectOwnership)
    if (llvm::Error error = projectSelectedWholeProgram(*module))
      return std::move(error);
  NativeExecutionContext capture;
  std::optional<WorkloadCaptureCallbackNames> workloadCaptureNames;
  if (capturePlan) {
    auto names = instrumentWorkloadBackedCapture(*module, selectedOperation,
                                                 *capturePlan);
    if (!names)
      return names.takeError();
    workloadCaptureNames.emplace(std::move(*names));
  }
  std::optional<std::string> blockActivation;
  if (profileSourceBlocks) {
    auto callback =
        instrumentBlockActivations(*module, sourceProgram.identity(), capture);
    if (!callback)
      return callback.takeError();
    blockActivation = std::move(*callback);
  }
  auto native = detail::lowerStructuredModuleToLlvm(std::move(module));
  if (!native)
    return native.takeError();
  if (llvm::Error error = detail::initializeNativeTarget())
    return std::move(error);
  auto targetJitOrError = llvm::orc::LLJITBuilder().create();
  if (!targetJitOrError)
    return classifyJitError(targetJitOrError.takeError(),
                            "cannot create host JIT");
  std::unique_ptr<llvm::orc::LLJIT> targetJit = std::move(*targetJitOrError);

  CallbackNames callbackNames;
  std::optional<WorkloadCaptureContext> workloadCapture;
  llvm::Error preparation =
      native->withModuleDo([&](llvm::Module &module) -> llvm::Error {
        if (llvm::Error error =
                detail::retargetStructuredOracle(module, *targetJit))
          return error;
        auto names =
            instrumentExecution(module, entry.getSymName(), *structured, *input,
                                *sourceContext, *plans, capture);
        if (!names)
          return names.takeError();
        callbackNames = std::move(*names);
        callbackNames.blockActivation = std::move(blockActivation);
        if (capturePlan) {
          auto prepared = prepareWorkloadCaptureContext(*capturePlan, capture);
          if (!prepared)
            return prepared.takeError();
          workloadCapture.emplace(std::move(*prepared));
        }
        return llvm::Error::success();
      });
  if (preparation)
    return std::move(preparation);
  if (llvm::Error error = runInstrumentedExecution(
          std::move(*native), callbackNames, capture, std::move(targetJit),
          workloadCaptureNames ? &*workloadCaptureNames : nullptr,
          workloadCapture ? &*workloadCapture : nullptr))
    return std::move(error);
  auto observations = buildObservations(*structured, *input, *plans, capture);
  if (!observations)
    return observations.takeError();
  NativeProgramExecutionResult result{std::move(*observations), std::nullopt};
  if (workloadCapture)
    result.boundaryCapture.emplace(std::move(workloadCapture->result));
  return result;
}

llvm::Expected<NativeStructuredProgramObservations>
executeProgramModule(const frontend::StructuredProgramCandidate &program,
                     const frontend::StructuredProgramCandidate &sourceProgram,
                     const CanonicalSimulationWorkload &workload,
                     const CanonicalSimulationRuntimeInput &runtimeInput,
                     bool profileSourceBlocks, bool projectOwnership) {
  auto programView = program.view();
  if (!programView)
    return programView.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> cloned(
      llvm::cast<mlir::ModuleOp>(program.module()->clone()));
  auto result = executePreparedProgramModule(
      std::move(cloned), sourceProgram, workload, runtimeInput,
      profileSourceBlocks, projectOwnership);
  if (!result)
    return result.takeError();
  return std::move(result->observations);
}

} // namespace

llvm::Expected<NativeStructuredProgramObservations>
executeNativeStructuredProgram(
    const frontend::StructuredProgramCandidate &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  return executeProgramModule(program, program, workload, runtimeInput, true,
                              false);
}

llvm::Expected<NativeStructuredProgramObservations>
executeSelectedStructuredProgram(
    const frontend::StructuredProgramCandidate &selectedProgram,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  return executeProgramModule(selectedProgram, sourceProgram, workload,
                              runtimeInput, false, true);
}

llvm::Expected<NativeSimulationInputCapture>
executeWorkloadBackedSimulationInputCapture(
    mlir::OwningOpRef<mlir::ModuleOp> preparedModule,
    mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan,
    const frontend::StructuredProgramCandidate &sourceProgram,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  auto result = executePreparedProgramModule(
      std::move(preparedModule), sourceProgram, workload, runtimeInput, false,
      false, selectedOperation, &plan);
  if (!result)
    return result.takeError();
  if (!result->boundaryCapture)
    return executionFailed("workload-backed capture produced no result");
  return std::move(*result->boundaryCapture);
}

bool haveEquivalentFunctionalObservations(
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

} // namespace loom::sim
