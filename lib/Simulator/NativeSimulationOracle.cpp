#include "Simulator/NativeSimulationOracle.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"

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

struct ActiveCall {
  std::size_t captureIndex = 0;
  std::uint64_t nextBefore = 0;
  std::uint64_t nextAfter = 0;
};

struct CaptureContext {
  NativeSimulationMemoryCapture result;
  std::vector<std::uint64_t> byteCounts;
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
  if (objectOrdinal == 0) {
    context.result.calls.emplace_back();
    NativeSimulationCallCapture &capture = context.result.calls.back();
    capture.objects.resize(context.byteCounts.size());
    context.activeCalls.push_back(
        ActiveCall{context.result.calls.size() - 1, 0, 0});
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
      active.nextAfter != objectOrdinal) {
    recordCaptureError("after callbacks do not close the active call");
    return;
  }
  copyBytes(context.result.calls[active.captureIndex]
                .objects[objectOrdinal]
                .finalBytes,
            base, byteCount);
  ++active.nextAfter;
  if (active.nextAfter == context.byteCounts.size())
    context.activeCalls.pop_back();
}

std::string uniqueCallbackName(llvm::Module &module, llvm::StringRef prefix) {
  std::string name = prefix.str();
  std::uint64_t suffix = 0;
  while (module.getNamedValue(name))
    name = (prefix + llvm::Twine(".") + llvm::Twine(++suffix)).str();
  return name;
}

llvm::CallInst *findSelectedCall(llvm::Module &module,
                                 const SimulationMemoryCapturePlan &plan) {
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

llvm::Expected<std::pair<std::string, std::string>>
instrumentCapture(llvm::Module &module, const SimulationMemoryCapturePlan &plan,
                  llvm::StringRef entrySymbol) {
  if (plan.objects.empty())
    return invalid("memory capture plan has no finite objects");
  if (plan.hostCallerSymbol.empty() || plan.hostCalleeSymbol.empty())
    return invalid("memory capture plan has no exact call-site locator");

  llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration() || entry->arg_size() != 0 ||
      !entry->getReturnType()->isIntegerTy(32))
    return invalid("entry must be a defined i32 () function");

  llvm::CallInst *selected = findSelectedCall(module, plan);
  if (!selected || selected->isMustTailCall())
    return invalid("selected direct host call is absent or musttail");
  selected->setTailCallKind(llvm::CallInst::TCK_None);

  llvm::LLVMContext &context = module.getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Type *pointer = llvm::PointerType::getUnqual(context);
  llvm::FunctionType *callbackType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {i64, pointer, i64}, false);
  std::string beforeName =
      uniqueCallbackName(module, "__loom_native_capture_before");
  std::string afterName =
      uniqueCallbackName(module, "__loom_native_capture_after");
  llvm::FunctionCallee before =
      module.getOrInsertFunction(beforeName, callbackType);
  llvm::FunctionCallee after =
      module.getOrInsertFunction(afterName, callbackType);

  llvm::IRBuilder<> beforeBuilder(selected);
  llvm::Instruction *afterInsertion = selected->getNextNode();
  if (!afterInsertion)
    return invalid("selected host call has no continuation instruction");
  llvm::IRBuilder<> afterBuilder(afterInsertion);
  for (auto [ordinal, object] : llvm::enumerate(plan.objects)) {
    if (object.byteCount == 0 ||
        object.byteCount > std::numeric_limits<std::size_t>::max() ||
        object.callOperandOrdinal >= selected->arg_size() ||
        object.operandByteOffset >= object.byteCount ||
        object.operandByteOffset >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
      return invalid("capture object has an invalid finite projection");
    llvm::Value *operand = selected->getArgOperand(object.callOperandOrdinal);
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
    afterBuilder.CreateCall(after, {objectOrdinal, base, byteCount});
  }
  if (llvm::verifyModule(module, &llvm::errs()))
    return invalid("instrumented native LLVM module does not verify");
  return std::make_pair(std::move(beforeName), std::move(afterName));
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

} // namespace

llvm::Expected<NativeSimulationMemoryCapture>
executeNativeSimulationMemoryCapture(llvm::orc::ThreadSafeModule module,
                                     const SimulationMemoryCapturePlan &plan,
                                     llvm::StringRef entrySymbol) {
  if (llvm::Error error = initializeNativeTargetOnce())
    return std::move(error);
  llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jitOrError =
      llvm::orc::LLJITBuilder().create();
  if (!jitOrError)
    return jitOrError.takeError();
  std::unique_ptr<llvm::orc::LLJIT> jit = std::move(*jitOrError);

  std::pair<std::string, std::string> callbackNames;
  llvm::Error preparation = module.withModuleDo([&](llvm::Module &native)
                                                    -> llvm::Error {
    if (native.getTargetTriple().empty())
      native.setTargetTriple(jit->getTargetTriple());
    else if (llvm::Triple(native.getTargetTriple()) != jit->getTargetTriple())
      return invalid("native module target triple does not match this host");
    if (native.getDataLayoutStr().empty())
      native.setDataLayout(jit->getDataLayout());
    else if (native.getDataLayout() != jit->getDataLayout())
      return invalid("native module data layout does not match this host");
    llvm::Expected<std::pair<std::string, std::string>> names =
        instrumentCapture(native, plan, entrySymbol);
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
  callbacks[jit->mangleAndIntern(callbackNames.first)] = {
      llvm::orc::ExecutorAddr::fromPtr(&captureBefore),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  callbacks[jit->mangleAndIntern(callbackNames.second)] = {
      llvm::orc::ExecutorAddr::fromPtr(&captureAfter),
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

} // namespace loom::sim
