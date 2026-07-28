#include "Simulator/NativeSimulationOracle.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstring>
#include <limits>
#include <mutex>
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

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("native_simulation_oracle_unsupported: ") + message);
}

struct ActiveCall {
  std::size_t captureIndex = 0;
  std::uint64_t nextBefore = 0;
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
resolveNativeObject(llvm::Value *pointer, const llvm::DataLayout &layout) {
  if (auto *gep = llvm::dyn_cast<llvm::GEPOperator>(pointer)) {
    llvm::Expected<ResolvedNativeObject> base =
        resolveNativeObject(gep->getPointerOperand(), layout);
    if (!base)
      return base.takeError();
    llvm::APInt offset(layout.getIndexSizeInBits(gep->getPointerAddressSpace()),
                       0, true);
    if (!gep->accumulateConstantOffset(layout, offset) || offset.isNegative() ||
        offset.getActiveBits() > 64)
      return invalid("native call operand has no nonnegative constant offset");
    const std::uint64_t increment = offset.getZExtValue();
    if (base->byteOffset >
        std::numeric_limits<std::uint64_t>::max() - increment)
      return invalid("native call operand byte offset overflows uint64");
    base->byteOffset += increment;
    if (base->byteOffset >= base->byteCount)
      return invalid("native call operand points outside its allocation");
    return *base;
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
  std::vector<RuntimeValueCaptureShape> runtimeValueShapes;
  std::vector<ValueResultCaptureShape> valueResultShapes;
  bool littleEndian = true;
  std::vector<ActiveCall> activeCalls;
  std::optional<std::string> error;
};

thread_local CaptureContext *activeCapture = nullptr;

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
  if (!activeCapture)
    return;
  CaptureContext &context = *activeCapture;
  if (context.error)
    return;
  context.result.calls.emplace_back();
  NativeSimulationCallCapture &capture = context.result.calls.back();
  capture.objects.resize(context.byteCounts.size());
  context.activeCalls.push_back(
      ActiveCall{context.result.calls.size() - 1, 0, 0, 0, 0});
}

void captureBefore(std::uint64_t objectOrdinal, void *base,
                   std::uint64_t byteCount) {
  if (!activeCapture)
    return;
  CaptureContext &context = *activeCapture;
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

void captureAfter(std::uint64_t objectOrdinal, void *base,
                  std::uint64_t byteCount) {
  if (!activeCapture)
    return;
  CaptureContext &context = *activeCapture;
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

llvm::APInt readLaneBits(llvm::ArrayRef<std::uint8_t> bytes,
                         std::uint64_t bitOffset, std::uint32_t bitWidth,
                         bool littleEndian) {
  llvm::APInt bits(bitWidth, 0);
  for (std::uint32_t bit = 0; bit < bitWidth; ++bit) {
    const std::uint64_t storageBit = bitOffset + bit;
    const std::uint64_t byteOrdinal = storageBit / 8;
    const std::uint32_t bitInByte = storageBit % 8;
    const std::uint64_t addressedByte =
        littleEndian ? byteOrdinal : bytes.size() - 1 - byteOrdinal;
    if ((bytes[addressedByte] >> bitInByte) & 1U)
      bits.setBit(bit);
  }
  return bits;
}

CanonicalValueSequence readCapturedValue(void *base, std::uint64_t byteCount,
                                         std::uint64_t lanesPerToken,
                                         std::uint32_t laneBitWidth,
                                         bool littleEndian) {
  llvm::ArrayRef<std::uint8_t> bytes(static_cast<const std::uint8_t *>(base),
                                     static_cast<std::size_t>(byteCount));
  CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes.reserve(lanesPerToken);
  for (std::uint64_t lane = 0; lane < lanesPerToken; ++lane)
    sequence.lanes.push_back(SemanticLane::defined(
        readLaneBits(bytes, lane * laneBitWidth, laneBitWidth, littleEndian)));
  return sequence;
}

void captureValue(std::uint64_t valueOrdinal, void *base,
                  std::uint64_t byteCount) {
  if (!activeCapture)
    return;
  CaptureContext &context = *activeCapture;
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
  if (!activeCapture)
    return;
  CaptureContext &context = *activeCapture;
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
  if (!activeCapture)
    return;
  CaptureContext &context = *activeCapture;
  if (context.error)
    return;
  if (context.activeCalls.empty()) {
    recordCaptureError("end callback has no active invocation");
    return;
  }
  const ActiveCall &active = context.activeCalls.back();
  if (active.nextBefore != context.byteCounts.size() ||
      active.nextValue != context.runtimeValueShapes.size() ||
      active.nextResult != context.valueResultShapes.size() ||
      active.nextAfter != context.byteCounts.size()) {
    recordCaptureError("end callback observed an incomplete invocation");
    return;
  }
  context.activeCalls.pop_back();
}

struct CaptureCallbackNames {
  std::string begin;
  std::string end;
  std::string before;
  std::string after;
  std::optional<std::string> value;
  std::optional<std::string> result;
};

std::string uniqueCallbackName(llvm::Module &module, llvm::StringRef prefix) {
  std::string name = prefix.str();
  std::uint64_t suffix = 0;
  while (module.getNamedValue(name))
    name = (prefix + llvm::Twine(".") + llvm::Twine(++suffix)).str();
  return name;
}

llvm::CallInst *
findSelectedCall(llvm::Module &module,
                 const DirectCallSimulationInputCapturePlan &plan) {
  llvm::Function *caller = module.getFunction(plan.hostCallerSymbol);
  llvm::Function *callee = module.getFunction(plan.hostCalleeSymbol);
  if (!caller || caller->isDeclaration() || !callee)
    return nullptr;
  std::uint64_t ordinal = 0;
  for (llvm::BasicBlock &block : *caller)
    for (llvm::Instruction &instruction : block) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
      if (!call || call->getCalledFunction() != callee)
        continue;
      if (ordinal++ == plan.hostCallOrdinal)
        return call;
    }
  return nullptr;
}

llvm::Expected<CaptureCallbackNames>
instrumentCapture(llvm::Module &module,
                  const DirectCallSimulationInputCapturePlan &plan,
                  llvm::StringRef entrySymbol) {
  if (plan.hostCallerSymbol.empty() || plan.hostCalleeSymbol.empty())
    return invalid("capture plan has no exact call-site locator");

  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration() || entry->arg_size() != 0 ||
      !entry->getReturnType()->isIntegerTy(32))
    return invalid("entry must be a defined i32 () function");

  llvm::CallInst *selected = findSelectedCall(module, plan);
  if (!selected || selected->isMustTailCall())
    return invalid("selected direct host call is absent or musttail");
  selected->setTailCallKind(llvm::CallInst::TCK_None);

  if (plan.memoryObjectSources.size() != plan.input.objects.size())
    return invalid("direct-call memory source table is not total");

  for (auto [object, source] :
       llvm::zip_equal(plan.input.objects, plan.memoryObjectSources)) {
    llvm::Expected<llvm::Value *> pointer =
        resolveDirectCallMemorySource(module, *selected, source);
    if (!pointer)
      return pointer.takeError();
    llvm::Expected<ResolvedNativeObject> resolved =
        resolveNativeObject(*pointer, module.getDataLayout());
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
  return CaptureCallbackNames{std::move(beginName),  std::move(endName),
                              std::move(beforeName), std::move(afterName),
                              std::move(valueName),  std::move(resultName)};
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
                            const SimulationInputCapturePlan &plan) {
  if (!selectedOperation ||
      selectedOperation->getParentOfType<mlir::ModuleOp>() != module)
    return invalid("selected operation is not owned by the prepared module");
  auto callable = selectedOperation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!callable || callable.getBody().empty())
    return invalid("selected operation is not enclosed by an LLVM callable");

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
  std::string beginName =
      uniqueCallbackName(module, "__loom_native_capture_begin");
  std::string endName = uniqueCallbackName(module, "__loom_native_capture_end");
  std::string beforeName =
      uniqueCallbackName(module, "__loom_native_capture_before");
  std::string afterName =
      uniqueCallbackName(module, "__loom_native_capture_after");
  std::optional<std::string> valueName;
  if (llvm::any_of(plan.valueInputs, [](const auto &input) {
        return !input.fixedValue.has_value();
      }))
    valueName = uniqueCallbackName(module, "__loom_native_capture_value");
  std::optional<std::string> resultName;
  if (!plan.valueResults.empty())
    resultName = uniqueCallbackName(module, "__loom_native_capture_result");
  mlir::LLVM::LLVMFuncOp::create(declarations, location, beginName,
                                 lifecycleType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, endName,
                                 lifecycleType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, beforeName,
                                 callbackType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, afterName,
                                 callbackType);
  if (valueName)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *valueName,
                                   callbackType);
  if (resultName)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *resultName,
                                   callbackType);

  mlir::OpBuilder before(selectedOperation);
  mlir::OpBuilder after(selectedOperation);
  after.setInsertionPointAfter(selectedOperation);
  mlir::LLVM::CallOp::create(before, location, mlir::TypeRange{}, beginName,
                             mlir::ValueRange{});
  for (auto [ordinal, object] : llvm::enumerate(plan.objects)) {
    if (!object.base ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(object.base.getType()) ||
        object.byteCount == 0 ||
        object.byteCount > std::numeric_limits<std::size_t>::max() ||
        object.operandByteOffset >= object.byteCount)
      return invalid("capture object has an invalid finite projection");
    auto ordinalAttr = before.getI64IntegerAttr(ordinal);
    auto byteCountAttr = before.getI64IntegerAttr(object.byteCount);
    mlir::Value beforeOrdinal =
        mlir::LLVM::ConstantOp::create(before, location, i64, ordinalAttr);
    mlir::Value beforeByteCount =
        mlir::LLVM::ConstantOp::create(before, location, i64, byteCountAttr);
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, beforeName,
        mlir::ValueRange{beforeOrdinal, object.base, beforeByteCount});
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
  for (const SimulationValueInputCapture &input : plan.valueInputs) {
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
  for (auto [ordinal, result] : llvm::enumerate(plan.valueResults)) {
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
  for (auto [ordinal, object] : llvm::enumerate(plan.objects)) {
    mlir::Value afterOrdinal = mlir::LLVM::ConstantOp::create(
        after, location, i64, after.getI64IntegerAttr(ordinal));
    mlir::Value afterByteCount = mlir::LLVM::ConstantOp::create(
        after, location, i64, after.getI64IntegerAttr(object.byteCount));
    mlir::LLVM::CallOp::create(
        after, location, mlir::TypeRange{}, afterName,
        mlir::ValueRange{afterOrdinal, object.base, afterByteCount});
  }
  mlir::LLVM::CallOp::create(after, location, mlir::TypeRange{}, endName,
                             mlir::ValueRange{});
  if (mlir::failed(mlir::verify(module)))
    return invalid("instrumented Structured Program does not verify");
  return CaptureCallbackNames{std::move(beginName),  std::move(endName),
                              std::move(beforeName), std::move(afterName),
                              std::move(valueName),  std::move(resultName)};
}

llvm::Error lowerStructuredOracleToLlvmDialect(mlir::ModuleOp module) {
  mlir::PassManager pipeline(module->getContext());
  pipeline.enableVerifier(true);
  pipeline.addPass(mlir::createSCFToControlFlowPass());
  pipeline.addPass(mlir::createConvertToLLVMPass());
  if (mlir::failed(pipeline.run(module)))
    return invalid("prepared Structured Program cannot lower to LLVM");
  return llvm::Error::success();
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

llvm::Expected<llvm::orc::ThreadSafeModule>
lowerStructuredOracleToLlvm(mlir::OwningOpRef<mlir::ModuleOp> module) {
  if (llvm::Error error = lowerStructuredOracleToLlvmDialect(*module))
    return std::move(error);

  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> translated = mlir::translateModuleToLLVMIR(
      module->getOperation(), *context, "loom-structured-native-oracle");
  if (!translated)
    return invalid("instrumented Structured Program cannot translate to LLVM");
  return llvm::orc::ThreadSafeModule(std::move(translated), std::move(context));
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

llvm::Error admitNativeHostModule(llvm::Module &module,
                                  const llvm::orc::LLJIT &jit) {
  if (module.getTargetTriple().empty())
    module.setTargetTriple(jit.getTargetTriple());
  else if (llvm::Triple(module.getTargetTriple()) != jit.getTargetTriple())
    return invalid("native module target triple does not match this host");
  if (module.getDataLayoutStr().empty())
    module.setDataLayout(jit.getDataLayout());
  else if (module.getDataLayout() != jit.getDataLayout())
    return invalid("native module data layout does not match this host");
  return llvm::Error::success();
}

void collectExecutionLayoutType(
    llvm::Type *type, llvm::SmallPtrSetImpl<llvm::Type *> &types,
    llvm::SmallDenseSet<unsigned, 8> &addressSpaces) {
  if (!type || !types.insert(type).second)
    return;
  if (auto *pointer = llvm::dyn_cast<llvm::PointerType>(type->getScalarType()))
    addressSpaces.insert(pointer->getAddressSpace());
  for (llvm::Type *subtype : type->subtypes())
    collectExecutionLayoutType(subtype, types, addressSpaces);
}

llvm::Error requireExecutionLayoutCompatibility(llvm::Module &module,
                                                const llvm::DataLayout &host) {
  if (module.getDataLayoutStr().empty())
    return unsupported("target module has no data-layout contract");
  const llvm::DataLayout &target = module.getDataLayout();
  if (target.isLittleEndian() != host.isLittleEndian() ||
      target.getStackAlignment() != host.getStackAlignment() ||
      target.getAllocaAddrSpace() != host.getAllocaAddrSpace() ||
      target.getProgramAddressSpace() != host.getProgramAddressSpace() ||
      target.getDefaultGlobalsAddressSpace() !=
          host.getDefaultGlobalsAddressSpace() ||
      target.getFunctionPtrAlign() != host.getFunctionPtrAlign() ||
      target.getFunctionPtrAlignType() != host.getFunctionPtrAlignType())
    return unsupported("target and host execution-layout roots differ");

  llvm::SmallPtrSet<llvm::Type *, 32> types;
  llvm::SmallDenseSet<unsigned, 8> addressSpaces;
  addressSpaces.insert(target.getAllocaAddrSpace());
  addressSpaces.insert(target.getProgramAddressSpace());
  addressSpaces.insert(target.getDefaultGlobalsAddressSpace());
  for (llvm::GlobalValue &global : module.global_values()) {
    collectExecutionLayoutType(global.getType(), types, addressSpaces);
    collectExecutionLayoutType(global.getValueType(), types, addressSpaces);
  }
  for (llvm::Function &function : module) {
    collectExecutionLayoutType(function.getFunctionType(), types,
                               addressSpaces);
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block) {
        collectExecutionLayoutType(instruction.getType(), types, addressSpaces);
        for (llvm::Value *operand : instruction.operands())
          collectExecutionLayoutType(operand->getType(), types, addressSpaces);
      }
  }

  for (unsigned addressSpace : addressSpaces) {
    if (target.getPointerSizeInBits(addressSpace) !=
            host.getPointerSizeInBits(addressSpace) ||
        target.getIndexSizeInBits(addressSpace) !=
            host.getIndexSizeInBits(addressSpace) ||
        target.getPointerABIAlignment(addressSpace) !=
            host.getPointerABIAlignment(addressSpace) ||
        target.getPointerPrefAlignment(addressSpace) !=
            host.getPointerPrefAlignment(addressSpace) ||
        target.isNonIntegralAddressSpace(addressSpace) !=
            host.isNonIntegralAddressSpace(addressSpace) ||
        target.hasUnstableRepresentation(addressSpace) !=
            host.hasUnstableRepresentation(addressSpace) ||
        target.hasExternalState(addressSpace) !=
            host.hasExternalState(addressSpace) ||
        target.getNullPtrValue(addressSpace) !=
            host.getNullPtrValue(addressSpace))
      return unsupported("target and host pointer layouts differ for a used "
                         "address space");
  }

  for (llvm::Type *type : types) {
    if (!type->isSized())
      continue;
    if (target.getTypeSizeInBits(type) != host.getTypeSizeInBits(type) ||
        target.getTypeStoreSize(type) != host.getTypeStoreSize(type) ||
        target.getTypeAllocSize(type) != host.getTypeAllocSize(type) ||
        target.getABITypeAlign(type) != host.getABITypeAlign(type) ||
        target.getPrefTypeAlign(type) != host.getPrefTypeAlign(type))
      return unsupported("target and host layouts differ for an executed type");
    auto *structure = llvm::dyn_cast<llvm::StructType>(type);
    if (!structure)
      continue;
    const llvm::StructLayout *targetLayout = target.getStructLayout(structure);
    const llvm::StructLayout *hostLayout = host.getStructLayout(structure);
    for (unsigned element = 0; element < structure->getNumElements(); ++element)
      if (targetLayout->getElementOffsetInBits(element) !=
          hostLayout->getElementOffsetInBits(element))
        return unsupported("target and host struct element layouts differ");
  }
  return llvm::Error::success();
}

llvm::Error retargetStructuredOracle(llvm::Module &module,
                                     const llvm::orc::LLJIT &jit) {
  if (!module.getModuleInlineAsm().empty())
    return unsupported("target module contains inline assembly");
  for (const llvm::Function &function : module)
    if (function.isTargetIntrinsic())
      return unsupported("target module contains a target-specific intrinsic");

  if (module.getTargetTriple().empty() ||
      llvm::Triple(module.getTargetTriple()) != jit.getTargetTriple())
    return unsupported("target module triple does not match this host");
  if (llvm::Error error =
          requireExecutionLayoutCompatibility(module, jit.getDataLayout()))
    return error;

  if (llvm::NamedMDNode *flags = module.getNamedMetadata("llvm.module.flags"))
    module.eraseNamedMetadata(flags);
  for (llvm::Function &function : module) {
    function.removeFnAttr("target-cpu");
    function.removeFnAttr("target-features");
    function.removeFnAttr("tune-cpu");
  }
  module.setTargetTriple(jit.getTargetTriple());
  module.setDataLayout(jit.getDataLayout());
  if (llvm::verifyModule(module, &llvm::errs()))
    return invalid("retargeted Structured oracle module does not verify");
  return llvm::Error::success();
}

llvm::Error initializeNativeTargetOnce() {
  static std::once_flag once;
  static bool targetFailed = false;
  static bool printerFailed = false;
  std::call_once(once, [] {
    targetFailed = llvm::InitializeNativeTarget();
    printerFailed = llvm::InitializeNativeTargetAsmPrinter();
  });
  if (targetFailed || printerFailed)
    return executionFailed("native LLVM target initialization failed");
  return llvm::Error::success();
}

using PrepareNativeModule =
    llvm::function_ref<llvm::Expected<CaptureCallbackNames>(
        llvm::Module &, const llvm::orc::LLJIT &)>;

llvm::Expected<NativeSimulationInputCapture>
runNativeCapture(llvm::orc::ThreadSafeModule module,
                 const SimulationInputCapturePlan &plan,
                 llvm::StringRef entrySymbol, PrepareNativeModule prepare) {
  if (llvm::Error error = initializeNativeTargetOnce())
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
  if (callbackNames.value)
    callbacks[jit->mangleAndIntern(*callbackNames.value)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureValue),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  if (callbackNames.result)
    callbacks[jit->mangleAndIntern(*callbackNames.result)] = {
        llvm::orc::ExecutorAddr::fromPtr(&captureResult),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
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
  capture.byteCounts.reserve(plan.objects.size());
  for (const SimulationMemoryCaptureObject &object : plan.objects)
    capture.byteCounts.push_back(object.byteCount);
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
        if (llvm::Error error = admitNativeHostModule(native, jit))
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
  if (llvm::Error error = lowerStructuredOracleToLlvmDialect(*module))
    return std::move(error);
  return runNativeCapture(
      std::move(hostModule), plan.input, entrySymbol,
      [&](llvm::Module &target,
          const llvm::orc::LLJIT &jit) -> llvm::Expected<CaptureCallbackNames> {
        if (llvm::Error error = admitNativeHostModule(target, jit))
          return std::move(error);
        if (llvm::Error error =
                replaceCallableBody(target, *module, plan.hostCalleeSymbol))
          return std::move(error);
        return instrumentCapture(target, plan, entrySymbol);
      });
}

llvm::Expected<NativeSimulationInputCapture>
executeStructuredSimulationInputCapture(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    mlir::Operation *selectedOperation, const SimulationInputCapturePlan &plan,
    llvm::StringRef entrySymbol) {
  if (!module)
    return invalid("prepared Structured Program module is absent");
  llvm::Expected<CaptureCallbackNames> callbackNames =
      instrumentStructuredCapture(*module, selectedOperation, plan);
  if (!callbackNames)
    return callbackNames.takeError();
  llvm::Expected<llvm::orc::ThreadSafeModule> native =
      lowerStructuredOracleToLlvm(std::move(module));
  if (!native)
    return native.takeError();

  return runNativeCapture(
      std::move(*native), plan, entrySymbol,
      [&](llvm::Module &target,
          const llvm::orc::LLJIT &jit) -> llvm::Expected<CaptureCallbackNames> {
        if (llvm::Error error = retargetStructuredOracle(target, jit))
          return std::move(error);
        return *callbackNames;
      });
}

} // namespace loom::sim
