#include "Simulator/NativeSimulationOracle.h"

#include "NativeExecutionSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("native_simulation_oracle_invalid: ") + message);
}

llvm::Error executionFailed(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::io_error),
      llvm::Twine("native_simulation_oracle_execution_failed: ") + message);
}

struct ActiveCall {
  std::size_t captureIndex = 0;
  std::uint64_t nextBefore = 0;
  std::uint64_t nextRoot = 0;
  std::uint64_t nextValue = 0;
  std::uint64_t nextResult = 0;
  std::uint64_t nextAfter = 0;
};

struct RuntimeValueCaptureShape {
  std::uint64_t graphInputOrdinal = 0;
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  std::uint64_t byteCount = 0;
};

struct ValueResultCaptureShape {
  std::uint64_t graphResultOrdinal = 0;
  std::uint64_t lanesPerToken = 0;
  std::uint32_t laneBitWidth = 0;
  std::uint64_t byteCount = 0;
};

struct ResolvedNativeObject {
  std::uint64_t byteCount = 0;
  std::uint64_t byteOffset = 0;
};

llvm::Expected<ResolvedNativeObject>
applyNativeGepOffset(const llvm::GEPOperator &gep, ResolvedNativeObject base,
                     const llvm::DataLayout &layout) {
  llvm::APInt offset(layout.getIndexSizeInBits(gep.getPointerAddressSpace()), 0,
                     true);
  if (!gep.accumulateConstantOffset(layout, offset) || offset.isNegative() ||
      offset.getActiveBits() > 64)
    return invalid("native call operand has no nonnegative constant offset");
  const std::uint64_t increment = offset.getZExtValue();
  if (base.byteOffset > std::numeric_limits<std::uint64_t>::max() - increment)
    return invalid("native call operand byte offset overflows uint64");
  base.byteOffset += increment;
  if (base.byteOffset >= base.byteCount)
    return invalid("native call operand points outside its allocation");
  return base;
}

llvm::Expected<ResolvedNativeObject>
resolveNativeObject(llvm::Value *pointer, const llvm::DataLayout &layout) {
  if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(pointer)) {
    llvm::Expected<ResolvedNativeObject> base =
        resolveNativeObject(gep->getPointerOperand(), layout);
    if (!base)
      return base.takeError();
    return applyNativeGepOffset(*gep, std::move(*base), layout);
  }
  auto *allocation = llvm::dyn_cast<llvm::AllocaInst>(pointer);
  llvm::Type *elementType = nullptr;
  std::uint64_t arrayCount = 1;
  if (allocation) {
    auto *count = llvm::dyn_cast<llvm::ConstantInt>(allocation->getArraySize());
    if (!count || count->isZero() || count->getValue().getActiveBits() > 64)
      return invalid("native alloca has no positive constant element count");
    arrayCount = count->getZExtValue();
    elementType = allocation->getAllocatedType();
  } else if (auto *global = llvm::dyn_cast<llvm::GlobalVariable>(pointer)) {
    elementType = global->getValueType();
  } else {
    return invalid("native call operand does not resolve to a finite object");
  }
  const llvm::TypeSize elementBytes = layout.getTypeAllocSize(elementType);
  if (elementBytes.isScalable() || elementBytes.getFixedValue() == 0)
    return invalid("native object has no fixed nonzero element size");
  if (arrayCount >
      std::numeric_limits<std::uint64_t>::max() / elementBytes.getFixedValue())
    return invalid("native object byte count overflows uint64");
  return ResolvedNativeObject{arrayCount * elementBytes.getFixedValue(), 0};
}

llvm::Expected<ResolvedNativeObject> resolveNativeObjectThroughCallPath(
    llvm::Value *pointer, llvm::Function &enclosingCallable,
    llvm::ArrayRef<llvm::CallInst *> invocationPath, std::size_t pathIndex,
    const llvm::DataLayout &layout) {
  if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(pointer)) {
    llvm::Expected<ResolvedNativeObject> base =
        resolveNativeObjectThroughCallPath(gep->getPointerOperand(),
                                           enclosingCallable, invocationPath,
                                           pathIndex, layout);
    if (!base)
      return base.takeError();
    return applyNativeGepOffset(*gep, std::move(*base), layout);
  }
  if (auto *argument = llvm::dyn_cast<llvm::Argument>(pointer)) {
    if (argument->getParent() != &enclosingCallable)
      return invalid("native memory operand has the wrong callable owner");
    if (pathIndex >= invocationPath.size())
      return invalid("native invocation path is incomplete");
    llvm::CallInst *hostCall = invocationPath[pathIndex];
    if (!hostCall || hostCall->getCalledFunction() != &enclosingCallable)
      return invalid("native invocation path has a broken callee edge");
    if (argument->getArgNo() >= hostCall->arg_size())
      return invalid("native callable argument exceeds host call operands");
    llvm::Value *callerValue = hostCall->getArgOperand(argument->getArgNo());
    if (pathIndex == 0)
      return resolveNativeObject(callerValue, layout);
    return resolveNativeObjectThroughCallPath(
        callerValue, *hostCall->getFunction(), invocationPath, pathIndex - 1,
        layout);
  }
  return resolveNativeObject(pointer, layout);
}

llvm::Expected<llvm::Value *>
resolveDirectCallMemorySource(llvm::Module &module, llvm::CallInst &selected,
                              const DirectCallMemorySource &source) {
  if (const auto *operand =
          std::get_if<DirectCallOperandMemorySource>(&source)) {
    if (operand->operandOrdinal >= selected.arg_size())
      return invalid("capture object exceeds native call operands");
    return selected.getArgOperand(operand->operandOrdinal);
  }
  const auto &global = std::get<DirectCallGlobalMemorySource>(source);
  llvm::GlobalVariable *resolved =
      module.getGlobalVariable(global.symbol, true);
  if (!resolved)
    return invalid("capture global is absent from the native module");
  return resolved;
}

struct CaptureContext {
  NativeSimulationInputCapture result;
  std::vector<std::uint64_t> byteCounts;
  std::vector<std::uint64_t> rootObjectOrdinals;
  std::vector<std::uint64_t> staticRootByteOffsets;
  std::vector<RuntimeValueCaptureShape> runtimeValueShapes;
  std::vector<ValueResultCaptureShape> valueResultShapes;
  bool littleEndian = true;
  bool captureDynamicRootOffsets = false;
  std::vector<ActiveCall> activeCalls;
  std::uint64_t requiredGateDepth = 0;
  std::uint64_t gateDepth = 0;
  std::optional<std::string> error;
};

thread_local CaptureContext *activeCapture = nullptr;

CaptureContext *enabledCapture() {
  if (!activeCapture ||
      (activeCapture->requiredGateDepth != 0 &&
       activeCapture->gateDepth != activeCapture->requiredGateDepth))
    return nullptr;
  return activeCapture;
}

void recordCaptureError(llvm::StringRef message) {
  if (activeCapture && !activeCapture->error)
    activeCapture->error = message.str();
}

void copyBytes(std::vector<std::uint8_t> &destination, void *base,
               std::uint64_t byteCount) {
  destination.resize(static_cast<std::size_t>(byteCount));
  if (byteCount != 0)
    std::memcpy(destination.data(), base, destination.size());
}

void captureBegin() {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  context.result.calls.emplace_back();
  NativeSimulationCallCapture &capture = context.result.calls.back();
  capture.objects.resize(context.byteCounts.size());
  capture.memoryRootByteOffsets = context.staticRootByteOffsets;
  context.activeCalls.push_back(ActiveCall{
      context.result.calls.size() - 1, 0,
      context.captureDynamicRootOffsets
          ? 0
          : static_cast<std::uint64_t>(context.rootObjectOrdinals.size()),
      0, 0, 0});
}

void captureBefore(std::uint64_t objectOrdinal, void *base,
                   std::uint64_t byteCount) {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  if (objectOrdinal >= context.byteCounts.size() || !base ||
      context.byteCounts[objectOrdinal] != byteCount) {
    recordCaptureError("before callback has an invalid object projection");
    return;
  }
  if (context.activeCalls.empty() ||
      context.activeCalls.back().nextBefore != objectOrdinal) {
    recordCaptureError("before callbacks are not in canonical object order");
    return;
  }
  ActiveCall &active = context.activeCalls.back();
  copyBytes(context.result.calls[active.captureIndex]
                .objects[objectOrdinal]
                .initialBytes,
            base, byteCount);
  ++active.nextBefore;
}

void captureMemoryRoot(std::uint64_t rootOrdinal,
                       std::uint64_t objectOrdinal, void *view,
                       void *objectBase, std::uint64_t byteCount) {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  if (rootOrdinal >= context.rootObjectOrdinals.size() ||
      objectOrdinal >= context.byteCounts.size() || !view || !objectBase ||
      context.rootObjectOrdinals[rootOrdinal] != objectOrdinal ||
      context.byteCounts[objectOrdinal] != byteCount ||
      context.activeCalls.empty()) {
    recordCaptureError("memory-root callback has an invalid projection");
    return;
  }
  ActiveCall &active = context.activeCalls.back();
  if (active.nextBefore != context.byteCounts.size() ||
      active.nextRoot != rootOrdinal) {
    recordCaptureError(
        "memory-root callbacks are not in canonical binding order");
    return;
  }
  const std::uintptr_t viewAddress = reinterpret_cast<std::uintptr_t>(view);
  const std::uintptr_t baseAddress =
      reinterpret_cast<std::uintptr_t>(objectBase);
  if (viewAddress < baseAddress ||
      viewAddress - baseAddress >= byteCount) {
    recordCaptureError(
        "memory-root view lies outside its finite backing object");
    return;
  }
  context.result.calls[active.captureIndex]
      .memoryRootByteOffsets[rootOrdinal] = viewAddress - baseAddress;
  ++active.nextRoot;
}

void captureAfter(std::uint64_t objectOrdinal, void *base,
                  std::uint64_t byteCount) {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  if (objectOrdinal >= context.byteCounts.size() || !base ||
      context.byteCounts[objectOrdinal] != byteCount ||
      context.activeCalls.empty()) {
    recordCaptureError("after callback has an invalid object projection");
    return;
  }
  ActiveCall &active = context.activeCalls.back();
  if (active.nextBefore != context.byteCounts.size() ||
      active.nextRoot != context.rootObjectOrdinals.size() ||
      active.nextValue != context.runtimeValueShapes.size() ||
      active.nextResult != context.valueResultShapes.size() ||
      active.nextAfter != objectOrdinal) {
    recordCaptureError("after callbacks do not close the active call");
    return;
  }
  copyBytes(context.result.calls[active.captureIndex]
                .objects[objectOrdinal]
                .finalBytes,
            base, byteCount);
  ++active.nextAfter;
}

CanonicalValueSequence readCapturedValue(void *base, std::uint64_t byteCount,
                                         std::uint64_t lanesPerToken,
                                         std::uint32_t laneBitWidth,
                                         bool littleEndian) {
  llvm::ArrayRef<std::uint8_t> bytes(static_cast<const std::uint8_t *>(base),
                                     static_cast<std::size_t>(byteCount));
  return detail::readDefinedNativeValue(bytes, lanesPerToken, laneBitWidth,
                                        littleEndian);
}

void captureValue(std::uint64_t valueOrdinal, void *base,
                  std::uint64_t byteCount) {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  if (valueOrdinal >= context.runtimeValueShapes.size() || !base ||
      context.runtimeValueShapes[valueOrdinal].byteCount != byteCount ||
      context.activeCalls.empty()) {
    recordCaptureError("value callback has an invalid input projection");
    return;
  }
  ActiveCall &active = context.activeCalls.back();
  if (active.nextBefore != context.byteCounts.size() ||
      active.nextRoot != context.rootObjectOrdinals.size() ||
      active.nextValue != valueOrdinal) {
    recordCaptureError("value callbacks are not in canonical input order");
    return;
  }

  const RuntimeValueCaptureShape &shape =
      context.runtimeValueShapes[valueOrdinal];
  context.result.calls[active.captureIndex].runtimeValues.push_back(
      RuntimeValueEntry{shape.graphInputOrdinal,
                        readCapturedValue(base, byteCount, shape.lanesPerToken,
                                          shape.laneBitWidth,
                                          context.littleEndian)});
  ++active.nextValue;
}

void captureResult(std::uint64_t resultOrdinal, void *base,
                   std::uint64_t byteCount) {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  if (resultOrdinal >= context.valueResultShapes.size() || !base ||
      context.valueResultShapes[resultOrdinal].byteCount != byteCount ||
      context.activeCalls.empty()) {
    recordCaptureError("result callback has an invalid output projection");
    return;
  }
  ActiveCall &active = context.activeCalls.back();
  if (active.nextBefore != context.byteCounts.size() ||
      active.nextRoot != context.rootObjectOrdinals.size() ||
      active.nextValue != context.runtimeValueShapes.size() ||
      active.nextResult != resultOrdinal) {
    recordCaptureError("result callbacks are not in canonical output order");
    return;
  }
  const ValueResultCaptureShape &shape =
      context.valueResultShapes[resultOrdinal];
  if (shape.graphResultOrdinal != resultOrdinal) {
    recordCaptureError("result shape does not use dense graph output order");
    return;
  }
  context.result.calls[active.captureIndex].valueResults.push_back(
      readCapturedValue(base, byteCount, shape.lanesPerToken,
                        shape.laneBitWidth, context.littleEndian));
  ++active.nextResult;
}

void captureEnd() {
  CaptureContext *enabled = enabledCapture();
  if (!enabled)
    return;
  CaptureContext &context = *enabled;
  if (context.error)
    return;
  if (context.activeCalls.empty()) {
    recordCaptureError("end callback has no active invocation");
    return;
  }
  const ActiveCall &active = context.activeCalls.back();
  if (active.nextBefore != context.byteCounts.size() ||
      active.nextRoot != context.rootObjectOrdinals.size() ||
      active.nextValue != context.runtimeValueShapes.size() ||
      active.nextResult != context.valueResultShapes.size() ||
      active.nextAfter != context.byteCounts.size()) {
    recordCaptureError("end callback observed an incomplete invocation");
    return;
  }
  context.activeCalls.pop_back();
}

void captureGateEnter() {
  if (!activeCapture)
    return;
  if (activeCapture->gateDepth >= activeCapture->requiredGateDepth) {
    recordCaptureError("capture invocation gate exceeded its exact path");
    return;
  }
  ++activeCapture->gateDepth;
}

void captureGateExit() {
  if (!activeCapture)
    return;
  if (activeCapture->gateDepth == 0) {
    recordCaptureError("capture invocation gate exited without entry");
    return;
  }
  --activeCapture->gateDepth;
}

struct CaptureCallbackNames {
  std::string begin;
  std::string end;
  std::string before;
  std::string after;
  std::optional<std::string> memoryRoot;
  std::optional<std::string> value;
  std::optional<std::string> result;
  std::optional<std::string> gateEnter;
  std::optional<std::string> gateExit;
  std::uint64_t requiredGateDepth = 0;
};

std::string uniqueCallbackName(llvm::Module &module, llvm::StringRef prefix) {
  std::string name = prefix.str();
  std::uint64_t suffix = 0;
  while (module.getNamedValue(name))
    name = (prefix + llvm::Twine(".") + llvm::Twine(++suffix)).str();
  return name;
}

llvm::CallInst *findSelectedCall(llvm::Module &module,
                                 const DirectCallCaptureSite &site) {
  llvm::Function *caller = module.getFunction(site.hostCallerSymbol);
  llvm::Function *callee = module.getFunction(site.hostCalleeSymbol);
  if (!caller || caller->isDeclaration() || !callee)
    return nullptr;
  std::uint64_t ordinal = 0;
  for (llvm::BasicBlock &block : *caller)
    for (llvm::Instruction &instruction : block) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
      if (!call || call->getCalledFunction() != callee)
        continue;
      if (ordinal++ == site.hostCallOrdinal)
        return call;
    }
  return nullptr;
}

llvm::Expected<CaptureCallbackNames>
instrumentCapture(llvm::Module &module,
                  const DirectCallSimulationInputCapturePlan &plan,
                  llvm::StringRef entrySymbol) {
  if (plan.invocationPath.empty())
    return invalid("capture plan has no exact invocation path");
  if (plan.invocationPath.front().hostCallerSymbol != entrySymbol)
    return invalid("capture invocation path does not start at the entry");

  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration() || entry->arg_size() != 0 ||
      !entry->getReturnType()->isIntegerTy(32))
    return invalid("entry must be a defined i32 () function");

  llvm::SmallVector<llvm::CallInst *, 4> invocationPath;
  invocationPath.reserve(plan.invocationPath.size());
  for (auto [ordinal, site] : llvm::enumerate(plan.invocationPath)) {
    if (site.hostCallerSymbol.empty() || site.hostCalleeSymbol.empty())
      return invalid("capture invocation path has an empty locator");
    if (ordinal != 0 && plan.invocationPath[ordinal - 1].hostCalleeSymbol !=
                            site.hostCallerSymbol)
      return invalid("capture invocation path is not contiguous");
    llvm::CallInst *call = findSelectedCall(module, site);
    if (!call || call->isMustTailCall())
      return invalid("selected direct host call is absent or musttail");
    call->setTailCallKind(llvm::CallInst::TCK_None);
    invocationPath.push_back(call);
  }
  llvm::CallInst *selected = invocationPath.back();

  if (plan.memoryObjectSources.size() != plan.input.objects.size())
    return invalid("direct-call memory source table is not total");

  for (auto [object, source] :
       llvm::zip_equal(plan.input.objects, plan.memoryObjectSources)) {
    llvm::Expected<llvm::Value *> pointer =
        resolveDirectCallMemorySource(module, *selected, source);
    if (!pointer)
      return pointer.takeError();
    llvm::Expected<ResolvedNativeObject> resolved =
        invocationPath.size() == 1
            ? resolveNativeObject(*pointer, module.getDataLayout())
            : resolveNativeObjectThroughCallPath(
                  *pointer, *selected->getFunction(), invocationPath,
                  invocationPath.size() - 2, module.getDataLayout());
    if (!resolved)
      return resolved.takeError();
    if (resolved->byteCount != object.byteCount ||
        resolved->byteOffset != object.operandByteOffset)
      return invalid("native allocation projection differs from target plan");
  }

  llvm::LLVMContext &context = module.getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *pointer = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *lifecycleType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), {}, false);
  llvm::FunctionType *callbackType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {i64, pointer, i64}, false);
  std::string beginName =
      uniqueCallbackName(module, "__loom_native_capture_begin");
  std::string endName = uniqueCallbackName(module, "__loom_native_capture_end");
  std::string beforeName =
      uniqueCallbackName(module, "__loom_native_capture_before");
  std::string afterName =
      uniqueCallbackName(module, "__loom_native_capture_after");
  llvm::FunctionCallee begin =
      module.getOrInsertFunction(beginName, lifecycleType);
  llvm::FunctionCallee end = module.getOrInsertFunction(endName, lifecycleType);
  llvm::FunctionCallee before =
      module.getOrInsertFunction(beforeName, callbackType);
  llvm::FunctionCallee after =
      module.getOrInsertFunction(afterName, callbackType);
  std::optional<std::string> valueName;
  std::optional<llvm::FunctionCallee> valueCallback;
  if (llvm::any_of(plan.input.valueInputs, [](const auto &input) {
        return !input.fixedValue.has_value();
      })) {
    valueName = uniqueCallbackName(module, "__loom_native_capture_value");
    valueCallback = module.getOrInsertFunction(*valueName, callbackType);
  }
  std::optional<std::string> resultName;
  std::optional<llvm::FunctionCallee> resultCallback;
  if (!plan.input.valueResults.empty()) {
    if (plan.input.valueResults.size() != 1 || selected->getType()->isVoidTy())
      return invalid("direct call has no exact scalar result projection");
    resultName = uniqueCallbackName(module, "__loom_native_capture_result");
    resultCallback = module.getOrInsertFunction(*resultName, callbackType);
  }
  std::optional<std::string> gateEnterName;
  std::optional<std::string> gateExitName;
  std::optional<llvm::FunctionCallee> gateEnter;
  std::optional<llvm::FunctionCallee> gateExit;
  if (invocationPath.size() > 1) {
    gateEnterName =
        uniqueCallbackName(module, "__loom_native_capture_gate_enter");
    gateExitName =
        uniqueCallbackName(module, "__loom_native_capture_gate_exit");
    gateEnter = module.getOrInsertFunction(*gateEnterName, lifecycleType);
    gateExit = module.getOrInsertFunction(*gateExitName, lifecycleType);
    for (llvm::CallInst *call : llvm::drop_end(invocationPath)) {
      llvm::IRBuilder<> gateBefore(call);
      llvm::Instruction *continuation = call->getNextNode();
      if (!continuation)
        return invalid("capture invocation gate has no continuation");
      llvm::IRBuilder<> gateAfter(continuation);
      gateBefore.CreateCall(*gateEnter);
      gateAfter.CreateCall(*gateExit);
    }
  }

  llvm::IRBuilder<> beforeBuilder(selected);
  llvm::Instruction *afterInsertion = selected->getNextNode();
  if (!afterInsertion)
    return invalid("selected host call has no continuation instruction");
  llvm::IRBuilder<> afterBuilder(afterInsertion);
  beforeBuilder.CreateCall(begin);
  struct DeferredMemoryCapture {
    llvm::Value *ordinal;
    llvm::Value *base;
    llvm::Value *byteCount;
  };
  llvm::SmallVector<DeferredMemoryCapture, 4> deferredMemoryCaptures;
  deferredMemoryCaptures.reserve(plan.input.objects.size());
  for (auto [ordinal, object] : llvm::enumerate(plan.input.objects)) {
    if (object.byteCount == 0 ||
        object.byteCount > std::numeric_limits<std::size_t>::max() ||
        object.operandByteOffset >= object.byteCount ||
        object.operandByteOffset >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
      return invalid("capture object has an invalid finite projection");
    llvm::Expected<llvm::Value *> pointer = resolveDirectCallMemorySource(
        module, *selected, plan.memoryObjectSources[ordinal]);
    if (!pointer)
      return pointer.takeError();
    llvm::Value *operand = *pointer;
    if (!operand->getType()->isPointerTy())
      return invalid("capture object representative is not a pointer operand");
    llvm::Value *base = operand;
    if (object.operandByteOffset != 0) {
      llvm::Value *negativeOffset = llvm::ConstantInt::getSigned(
          i64, -static_cast<std::int64_t>(object.operandByteOffset));
      base = beforeBuilder.CreateGEP(llvm::Type::getInt8Ty(context), operand,
                                     negativeOffset, "loom.capture.base");
    }
    llvm::Value *objectOrdinal =
        llvm::ConstantInt::get(i64, static_cast<std::uint64_t>(ordinal));
    llvm::Value *byteCount = llvm::ConstantInt::get(i64, object.byteCount);
    beforeBuilder.CreateCall(before, {objectOrdinal, base, byteCount});
    deferredMemoryCaptures.push_back({objectOrdinal, base, byteCount});
  }
  std::uint64_t runtimeOrdinal = 0;
  llvm::IRBuilder<> entryBuilder(
      &selected->getFunction()->getEntryBlock(),
      selected->getFunction()->getEntryBlock().getFirstInsertionPt());
  for (const SimulationValueInputCapture &input : plan.input.valueInputs) {
    if (input.fixedValue)
      continue;
    if (!valueCallback || !input.boundaryOperandOrdinal ||
        *input.boundaryOperandOrdinal >= selected->arg_size() ||
        input.byteCount == 0 ||
        input.byteCount > std::numeric_limits<std::size_t>::max())
      return invalid("runtime value input has no finite call projection");
    llvm::Value *operand =
        selected->getArgOperand(*input.boundaryOperandOrdinal);
    if (!operand->getType()->isSized())
      return invalid("runtime value input has no sized native type");
    llvm::TypeSize nativeBytes =
        module.getDataLayout().getTypeStoreSize(operand->getType());
    if (nativeBytes.isScalable() ||
        nativeBytes.getFixedValue() != input.byteCount)
      return invalid("runtime value input native extent differs from plan");
    llvm::AllocaInst *slot = entryBuilder.CreateAlloca(
        operand->getType(), nullptr, "loom.capture.value");
    beforeBuilder.CreateStore(operand, slot);
    llvm::Value *ordinalValue = llvm::ConstantInt::get(i64, runtimeOrdinal++);
    llvm::Value *byteCount = llvm::ConstantInt::get(i64, input.byteCount);
    beforeBuilder.CreateCall(*valueCallback, {ordinalValue, slot, byteCount});
  }
  for (const SimulationValueResultCapture &result : plan.input.valueResults) {
    if (!resultCallback || result.valueResultOrdinal != 0 ||
        !selected->getType()->isSized() || result.byteCount == 0 ||
        result.byteCount > std::numeric_limits<std::size_t>::max())
      return invalid("direct call result has no finite native projection");
    llvm::TypeSize nativeBytes =
        module.getDataLayout().getTypeStoreSize(selected->getType());
    if (nativeBytes.isScalable() ||
        nativeBytes.getFixedValue() != result.byteCount)
      return invalid("direct call result native extent differs from plan");
    llvm::AllocaInst *slot = entryBuilder.CreateAlloca(
        selected->getType(), nullptr, "loom.capture.result");
    afterBuilder.CreateStore(selected, slot);
    llvm::Value *ordinalValue = llvm::ConstantInt::get(i64, 0);
    llvm::Value *byteCount = llvm::ConstantInt::get(i64, result.byteCount);
    afterBuilder.CreateCall(*resultCallback, {ordinalValue, slot, byteCount});
  }
  for (const DeferredMemoryCapture &capture : deferredMemoryCaptures)
    afterBuilder.CreateCall(after,
                            {capture.ordinal, capture.base, capture.byteCount});
  afterBuilder.CreateCall(end);
  if (llvm::verifyModule(module, &llvm::errs()))
    return invalid("instrumented native LLVM module does not verify");
  return CaptureCallbackNames{
      std::move(beginName),     std::move(endName),
      std::move(beforeName),    std::move(afterName),
      std::nullopt,             std::move(valueName),
      std::move(resultName),    std::move(gateEnterName),
      std::move(gateExitName),  invocationPath.size() - 1};
}

std::string uniqueCallbackName(mlir::ModuleOp module, llvm::StringRef prefix) {
  std::string name = prefix.str();
  std::uint64_t suffix = 0;
  while (mlir::SymbolTable::lookupSymbolIn(module, name))
    name = (prefix + llvm::Twine(".") + llvm::Twine(++suffix)).str();
  return name;
}

llvm::Expected<CaptureCallbackNames>
instrumentStructuredCapture(mlir::ModuleOp module,
                            mlir::Operation *selectedOperation,
                            const OperationSimulationInputCapturePlan &plan) {
  if (!selectedOperation ||
      selectedOperation->getParentOfType<mlir::ModuleOp>() != module)
    return invalid("selected operation is not owned by the prepared module");
  auto callable = selectedOperation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!callable || callable.getBody().empty())
    return invalid("selected operation is not enclosed by an LLVM callable");
  for (auto [ordinal, site] : llvm::enumerate(plan.invocationPath)) {
    mlir::LLVM::CallOp hostCall = site.hostCall;
    if (!hostCall || hostCall->getParentOfType<mlir::ModuleOp>() != module ||
        !hostCall.getCalleeAttr() ||
        hostCall.getCalleeAttr().getValue() != site.hostCalleeSymbol)
      return invalid("operation capture invocation path has a malformed "
                     "call site");
    auto caller = hostCall->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!caller || caller.getSymName() != site.hostCallerSymbol)
      return invalid("operation capture invocation path has the wrong caller");
    if (ordinal != 0 && plan.invocationPath[ordinal - 1].hostCalleeSymbol !=
                            site.hostCallerSymbol)
      return invalid("operation capture invocation path is not contiguous");
  }
  if (!plan.invocationPath.empty() &&
      plan.invocationPath.back().hostCalleeSymbol != callable.getSymName())
    return invalid("operation capture invocation path does not reach the "
                   "selected callable");
  const SimulationInputCapturePlan &inputPlan = plan.input;

  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder declarations(context);
  declarations.setInsertionPointToStart(module.getBody());
  mlir::Location location = selectedOperation->getLoc();
  mlir::Type i64 = declarations.getI64Type();
  mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type lifecycleType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(context), {});
  mlir::Type callbackType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(context), {i64, pointer, i64});
  mlir::Type memoryRootCallbackType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(context),
      {i64, i64, pointer, pointer, i64});
  std::string beginName =
      uniqueCallbackName(module, "__loom_native_capture_begin");
  std::string endName = uniqueCallbackName(module, "__loom_native_capture_end");
  std::string beforeName =
      uniqueCallbackName(module, "__loom_native_capture_before");
  std::string afterName =
      uniqueCallbackName(module, "__loom_native_capture_after");
  std::optional<std::string> memoryRootName;
  if (!inputPlan.memoryRootBindings.empty())
    memoryRootName =
        uniqueCallbackName(module, "__loom_native_capture_memory_root");
  std::optional<std::string> valueName;
  if (llvm::any_of(inputPlan.valueInputs, [](const auto &input) {
        return !input.fixedValue.has_value();
      }))
    valueName = uniqueCallbackName(module, "__loom_native_capture_value");
  std::optional<std::string> resultName;
  if (!inputPlan.valueResults.empty())
    resultName = uniqueCallbackName(module, "__loom_native_capture_result");
  std::optional<std::string> gateEnterName;
  std::optional<std::string> gateExitName;
  if (!plan.invocationPath.empty()) {
    gateEnterName =
        uniqueCallbackName(module, "__loom_native_capture_gate_enter");
    gateExitName =
        uniqueCallbackName(module, "__loom_native_capture_gate_exit");
  }
  mlir::LLVM::LLVMFuncOp::create(declarations, location, beginName,
                                 lifecycleType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, endName,
                                 lifecycleType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, beforeName,
                                 callbackType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, afterName,
                                 callbackType);
  if (memoryRootName)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *memoryRootName,
                                   memoryRootCallbackType);
  if (valueName)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *valueName,
                                   callbackType);
  if (resultName)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *resultName,
                                   callbackType);
  if (gateEnterName) {
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *gateEnterName,
                                   lifecycleType);
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *gateExitName,
                                   lifecycleType);
    for (const DirectCallCaptureSite &site : plan.invocationPath) {
      mlir::OpBuilder gateBefore(site.hostCall);
      mlir::OpBuilder gateAfter(site.hostCall);
      gateAfter.setInsertionPointAfter(site.hostCall);
      mlir::LLVM::CallOp::create(gateBefore, location, mlir::TypeRange{},
                                 *gateEnterName, mlir::ValueRange{});
      mlir::LLVM::CallOp::create(gateAfter, location, mlir::TypeRange{},
                                 *gateExitName, mlir::ValueRange{});
    }
  }

  mlir::OpBuilder before(selectedOperation);
  mlir::OpBuilder after(selectedOperation);
  after.setInsertionPointAfter(selectedOperation);
  mlir::LLVM::CallOp::create(before, location, mlir::TypeRange{}, beginName,
                             mlir::ValueRange{});
  llvm::SmallVector<mlir::Value, 4> objectBases;
  objectBases.reserve(inputPlan.objects.size());
  for (auto [ordinal, object] : llvm::enumerate(inputPlan.objects)) {
    if (!object.base ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(object.base.getType()) ||
        object.byteCount == 0 ||
        object.byteCount > std::numeric_limits<std::size_t>::max() ||
        object.operandByteOffset >= object.byteCount ||
        object.operandByteOffset >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
      return invalid("capture object has an invalid finite projection");
    mlir::Value base = object.base;
    if (object.operandByteOffset != 0) {
      mlir::Value negativeOffset = mlir::LLVM::ConstantOp::create(
          before, location, i64,
          before.getI64IntegerAttr(
              -static_cast<std::int64_t>(object.operandByteOffset)));
      base = mlir::LLVM::GEPOp::create(before, location, pointer,
                                       before.getI8Type(), object.base,
                                       mlir::ValueRange{negativeOffset})
                 .getResult();
    }
    objectBases.push_back(base);
    auto ordinalAttr = before.getI64IntegerAttr(ordinal);
    auto byteCountAttr = before.getI64IntegerAttr(object.byteCount);
    mlir::Value beforeOrdinal =
        mlir::LLVM::ConstantOp::create(before, location, i64, ordinalAttr);
    mlir::Value beforeByteCount =
        mlir::LLVM::ConstantOp::create(before, location, i64, byteCountAttr);
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, beforeName,
        mlir::ValueRange{beforeOrdinal, base, beforeByteCount});
  }
  for (auto [ordinal, binding] :
       llvm::enumerate(inputPlan.memoryRootBindings)) {
    if (!memoryRootName || !binding.boundaryPointer ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(
            binding.boundaryPointer.getType()) ||
        binding.objectIndex >= inputPlan.objects.size())
      return invalid(
          "operation memory root has no invocation-local pointer projection");
    const SimulationMemoryCaptureObject &object =
        inputPlan.objects[binding.objectIndex];
    mlir::Value rootOrdinal = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(ordinal));
    mlir::Value objectOrdinal = mlir::LLVM::ConstantOp::create(
        before, location, i64,
        before.getI64IntegerAttr(binding.objectIndex));
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(object.byteCount));
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, *memoryRootName,
        mlir::ValueRange{rootOrdinal, objectOrdinal, binding.boundaryPointer,
                         objectBases[binding.objectIndex], byteCount});
  }
  mlir::OpBuilder storage(context);
  storage.setInsertionPointToStart(&callable.getBody().front());
  mlir::Value one;
  auto allocateSlot = [&](mlir::Type type) {
    if (!one)
      one = mlir::LLVM::ConstantOp::create(storage, location, i64,
                                           storage.getI64IntegerAttr(1));
    return mlir::LLVM::AllocaOp::create(storage, location, pointer, type, one)
        .getRes();
  };
  std::uint64_t runtimeOrdinal = 0;
  for (const SimulationValueInputCapture &input : inputPlan.valueInputs) {
    if (input.fixedValue)
      continue;
    if (!valueName || !input.boundaryValue || !input.boundaryOperandOrdinal ||
        input.byteCount == 0 ||
        input.byteCount > std::numeric_limits<std::size_t>::max())
      return invalid("runtime value input has no finite boundary projection");
    mlir::Value slot = allocateSlot(input.boundaryValue.getType());
    mlir::LLVM::StoreOp::create(before, location, input.boundaryValue, slot);
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(runtimeOrdinal));
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(input.byteCount));
    mlir::LLVM::CallOp::create(before, location, mlir::TypeRange{}, *valueName,
                               mlir::ValueRange{ordinalValue, slot, byteCount});
    ++runtimeOrdinal;
  }
  for (auto [ordinal, result] : llvm::enumerate(inputPlan.valueResults)) {
    if (!resultName || result.valueResultOrdinal != ordinal ||
        !result.boundaryValue ||
        result.boundaryValue.getDefiningOp() != selectedOperation ||
        result.byteCount == 0 ||
        result.byteCount > std::numeric_limits<std::size_t>::max())
      return invalid("graph value result has no exact boundary projection");
    mlir::Value slot = allocateSlot(result.boundaryValue.getType());
    mlir::LLVM::StoreOp::create(after, location, result.boundaryValue, slot);
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        after, location, i64, after.getI64IntegerAttr(ordinal));
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        after, location, i64, after.getI64IntegerAttr(result.byteCount));
    mlir::LLVM::CallOp::create(after, location, mlir::TypeRange{}, *resultName,
                               mlir::ValueRange{ordinalValue, slot, byteCount});
  }
  for (auto [ordinal, object] : llvm::enumerate(inputPlan.objects)) {
    mlir::Value afterOrdinal = mlir::LLVM::ConstantOp::create(
        after, location, i64, after.getI64IntegerAttr(ordinal));
    mlir::Value afterByteCount = mlir::LLVM::ConstantOp::create(
        after, location, i64, after.getI64IntegerAttr(object.byteCount));
    mlir::LLVM::CallOp::create(
        after, location, mlir::TypeRange{}, afterName,
        mlir::ValueRange{afterOrdinal, objectBases[ordinal], afterByteCount});
  }
  mlir::LLVM::CallOp::create(after, location, mlir::TypeRange{}, endName,
                             mlir::ValueRange{});
  if (mlir::failed(mlir::verify(module)))
    return invalid("instrumented Structured Program does not verify");
  return CaptureCallbackNames{
      std::move(beginName),      std::move(endName),
      std::move(beforeName),     std::move(afterName),
      std::move(memoryRootName), std::move(valueName),
      std::move(resultName),     std::move(gateEnterName),
      std::move(gateExitName),   plan.invocationPath.size()};
}

void collectReferencedGlobals(llvm::Value *value,
                              llvm::SmallPtrSetImpl<llvm::GlobalValue *> &out) {
  if (auto *global = llvm::dyn_cast<llvm::GlobalValue>(value)) {
    out.insert(global);
    return;
  }
  auto *constant = llvm::dyn_cast<llvm::Constant>(value);
  if (!constant)
    return;
  for (llvm::Value *operand : constant->operands())
    collectReferencedGlobals(operand, out);
}

llvm::Error replaceCallableBody(llvm::Module &host,
                                mlir::ModuleOp preparedModule,
                                llvm::StringRef callableSymbol) {
  std::unique_ptr<llvm::Module> prepared = mlir::translateModuleToLLVMIR(
      preparedModule.getOperation(), host.getContext(),
      "loom-selected-callable-oracle");
  if (!prepared)
    return invalid("prepared Structured Program cannot translate to LLVM");

  llvm::Function *source = prepared->getFunction(callableSymbol);
  llvm::Function *target = host.getFunction(callableSymbol);
  if (!source || source->isDeclaration() || !target || target->isDeclaration())
    return invalid("selected callable is absent from one native oracle module");
  if (source->getFunctionType() != target->getFunctionType())
    return invalid("selected callable changed its native ABI");

  llvm::ValueToValueMapTy values;
  values[source] = target;
  for (auto [sourceArgument, targetArgument] :
       llvm::zip_equal(source->args(), target->args()))
    values[&sourceArgument] = &targetArgument;

  llvm::SmallPtrSet<llvm::GlobalValue *, 16> dependencies;
  for (llvm::BasicBlock &block : *source)
    for (llvm::Instruction &instruction : block)
      for (llvm::Value *operand : instruction.operands())
        collectReferencedGlobals(operand, dependencies);
  if (source->hasPersonalityFn())
    collectReferencedGlobals(source->getPersonalityFn(), dependencies);
  if (source->hasPrefixData())
    collectReferencedGlobals(source->getPrefixData(), dependencies);
  if (source->hasPrologueData())
    collectReferencedGlobals(source->getPrologueData(), dependencies);

  for (llvm::GlobalValue *global : dependencies) {
    if (global == source)
      continue;
    if (llvm::GlobalValue *existing = host.getNamedValue(global->getName())) {
      if (existing->getValueType() != global->getValueType())
        return invalid("selected callable dependency changed native type: " +
                       global->getName());
      values[global] = existing;
      continue;
    }
    auto *function = llvm::dyn_cast<llvm::Function>(global);
    if (!function || !function->isDeclaration())
      return invalid("selected callable introduced an unknown native global");
    auto *declaration = llvm::Function::Create(
        function->getFunctionType(), function->getLinkage(),
        function->getAddressSpace(), function->getName(), &host);
    declaration->copyAttributesFrom(function);
    declaration->setCallingConv(function->getCallingConv());
    values[function] = declaration;
  }

  target->deleteBody();
  llvm::SmallVector<llvm::ReturnInst *> returns;
  llvm::CloneFunctionInto(target, source, values,
                          llvm::CloneFunctionChangeType::DifferentModule,
                          returns);
  if (llvm::verifyFunction(*target, &llvm::errs()) ||
      llvm::verifyModule(host, &llvm::errs()))
    return invalid("selected callable replacement does not verify");
  return llvm::Error::success();
}

using PrepareNativeModule =
    llvm::function_ref<llvm::Expected<CaptureCallbackNames>(
        llvm::Module &, const llvm::orc::LLJIT &)>;

llvm::Expected<NativeSimulationInputCapture>
runNativeCapture(llvm::orc::ThreadSafeModule module,
                 const SimulationInputCapturePlan &plan,
                 llvm::StringRef entrySymbol, PrepareNativeModule prepare) {
  if (llvm::Error error = detail::initializeNativeTarget())
    return std::move(error);
  llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jitOrError =
      llvm::orc::LLJITBuilder().create();
  if (!jitOrError)
    return jitOrError.takeError();
  std::unique_ptr<llvm::orc::LLJIT> jit = std::move(*jitOrError);

  CaptureCallbackNames callbackNames;
  llvm::Error preparation =
      module.withModuleDo([&](llvm::Module &native) -> llvm::Error {
        llvm::Expected<CaptureCallbackNames> names = prepare(native, *jit);
        if (!names)
          return names.takeError();
        callbackNames = std::move(*names);
        return llvm::Error::success();
      });
  if (preparation)
    return std::move(preparation);

  llvm::orc::JITDylib &dylib = jit->getMainJITDylib();
  if (llvm::Expected<std::unique_ptr<llvm::orc::DynamicLibrarySearchGenerator>>
          generator =
              llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                  jit->getDataLayout().getGlobalPrefix()))
    dylib.addGenerator(std::move(*generator));
  else
    return generator.takeError();

  llvm::orc::SymbolMap callbacks;
  callbacks[jit->mangleAndIntern(callbackNames.begin)] = {
      llvm::orc::ExecutorAddr::fromPtr(&captureBegin),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  callbacks[jit->mangleAndIntern(callbackNames.end)] = {
      llvm::orc::ExecutorAddr::fromPtr(&captureEnd),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  callbacks[jit->mangleAndIntern(callbackNames.before)] = {
      llvm::orc::ExecutorAddr::fromPtr(&captureBefore),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  callbacks[jit->mangleAndIntern(callbackNames.after)] = {
      llvm::orc::ExecutorAddr::fromPtr(&captureAfter),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (callbackNames.memoryRoot)
    callbacks[jit->mangleAndIntern(*callbackNames.memoryRoot)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureMemoryRoot),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (callbackNames.value)
    callbacks[jit->mangleAndIntern(*callbackNames.value)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureValue),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (callbackNames.result)
    callbacks[jit->mangleAndIntern(*callbackNames.result)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureResult),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (callbackNames.gateEnter.has_value() != callbackNames.gateExit.has_value())
    return invalid("capture invocation gate callback table is incomplete");
  if (callbackNames.gateEnter) {
    callbacks[jit->mangleAndIntern(*callbackNames.gateEnter)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureGateEnter),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
    callbacks[jit->mangleAndIntern(*callbackNames.gateExit)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureGateExit),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  }
  if (llvm::Error error =
          dylib.define(llvm::orc::absoluteSymbols(std::move(callbacks))))
    return std::move(error);
  if (llvm::Error error = jit->addIRModule(std::move(module)))
    return std::move(error);
  if (llvm::Error error = jit->initialize(dylib))
    return std::move(error);

  llvm::Expected<llvm::orc::ExecutorAddr> entry = jit->lookup(entrySymbol);
  if (!entry)
    return entry.takeError();

  CaptureContext capture;
  capture.requiredGateDepth = callbackNames.requiredGateDepth;
  if ((capture.requiredGateDepth == 0) != !callbackNames.gateEnter.has_value())
    return invalid("capture invocation gate depth is inconsistent");
  capture.byteCounts.reserve(plan.objects.size());
  for (const SimulationMemoryCaptureObject &object : plan.objects)
    capture.byteCounts.push_back(object.byteCount);
  capture.rootObjectOrdinals.reserve(plan.memoryRootBindings.size());
  capture.staticRootByteOffsets.reserve(plan.memoryRootBindings.size());
  for (const SimulationMemoryRootCapture &binding : plan.memoryRootBindings) {
    if (binding.objectIndex >= plan.objects.size())
      return invalid("memory root references an absent capture object");
    capture.rootObjectOrdinals.push_back(binding.objectIndex);
    capture.staticRootByteOffsets.push_back(binding.byteOffset);
  }
  capture.captureDynamicRootOffsets = callbackNames.memoryRoot.has_value();
  capture.littleEndian = jit->getDataLayout().isLittleEndian();
  for (const SimulationValueInputCapture &input : plan.valueInputs)
    if (!input.fixedValue)
      capture.runtimeValueShapes.push_back(
          RuntimeValueCaptureShape{input.valueInputOrdinal, input.lanesPerToken,
                                   input.laneBitWidth, input.byteCount});
  capture.valueResultShapes.reserve(plan.valueResults.size());
  for (const SimulationValueResultCapture &result : plan.valueResults)
    capture.valueResultShapes.push_back(
        ValueResultCaptureShape{result.valueResultOrdinal, result.lanesPerToken,
                                result.laneBitWidth, result.byteCount});
  if (activeCapture)
    return executionFailed("nested native oracle execution is unsupported");
  activeCapture = &capture;
  using EntryFunction = std::int32_t();
  capture.result.entryResult = entry->toPtr<EntryFunction>()();
  activeCapture = nullptr;

  if (llvm::Error error = jit->deinitialize(dylib))
    return std::move(error);
  if (capture.error)
    return executionFailed(*capture.error);
  if (capture.gateDepth != 0)
    return executionFailed("native execution left an open invocation gate");
  if (!capture.activeCalls.empty())
    return executionFailed("native execution left an incomplete call capture");
  return std::move(capture.result);
}

} // namespace

llvm::Expected<NativeSimulationInputCapture>
executeNativeSimulationInputCapture(
    llvm::orc::ThreadSafeModule module,
    const DirectCallSimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol) {
  return runNativeCapture(
      std::move(module), plan.input, entrySymbol,
      [&](llvm::Module &native,
          const llvm::orc::LLJIT &jit) -> llvm::Expected<CaptureCallbackNames> {
        if (llvm::Error error = detail::admitNativeHostModule(native, jit))
          return std::move(error);
        return instrumentCapture(native, plan, entrySymbol);
      });
}

llvm::Expected<NativeSimulationInputCapture>
executeStructuredDirectCallSimulationInputCapture(
    llvm::orc::ThreadSafeModule hostModule,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    const DirectCallSimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol) {
  if (!module)
    return invalid("prepared Structured Program module is absent");
  if (plan.invocationPath.empty())
    return invalid("direct-call capture has no exact invocation path");
  if (llvm::Error error = detail::lowerStructuredModuleToLlvmDialect(*module))
    return std::move(error);
  return runNativeCapture(
      std::move(hostModule), plan.input, entrySymbol,
      [&](llvm::Module &target,
          const llvm::orc::LLJIT &jit) -> llvm::Expected<CaptureCallbackNames> {
        if (llvm::Error error = detail::admitNativeHostModule(target, jit))
          return std::move(error);
        if (llvm::Error error = replaceCallableBody(
                target, *module, plan.invocationPath.back().hostCalleeSymbol))
          return std::move(error);
        return instrumentCapture(target, plan, entrySymbol);
      });
}

llvm::Expected<NativeSimulationInputCapture>
executeStructuredSimulationInputCapture(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    mlir::Operation *selectedOperation,
    const OperationSimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol) {
  if (!module)
    return invalid("prepared Structured Program module is absent");
  auto callable =
      selectedOperation
          ? selectedOperation->getParentOfType<mlir::LLVM::LLVMFuncOp>()
          : mlir::LLVM::LLVMFuncOp{};
  if (callable && callable.getSymName() != entrySymbol &&
      plan.invocationPath.empty())
    return invalid("nested operation capture has no exact invocation path");
  llvm::Expected<CaptureCallbackNames> callbackNames =
      instrumentStructuredCapture(*module, selectedOperation, plan);
  if (!callbackNames)
    return callbackNames.takeError();
  llvm::Expected<llvm::orc::ThreadSafeModule> native =
      detail::lowerStructuredModuleToLlvm(std::move(module));
  if (!native)
    return native.takeError();

  return runNativeCapture(
      std::move(*native), plan.input, entrySymbol,
      [&](llvm::Module &target,
          const llvm::orc::LLJIT &jit) -> llvm::Expected<CaptureCallbackNames> {
        if (llvm::Error error = detail::retargetStructuredOracle(target, jit))
          return std::move(error);
        return *callbackNames;
      });
}

} // namespace loom::sim
