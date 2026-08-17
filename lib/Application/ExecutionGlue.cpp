#include "ExecutionGlue.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Runtime/Gem5DispatchABI.h"
#include "Runtime/Gem5SpatialBridgeABI.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <system_error>
#include <utility>

namespace loom::application::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_execution_glue_invalid: " + message);
}

llvm::Expected<std::uint32_t>
transportBitCount(sim::SpatialSimulationValueShape shape) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint32_t>::max() / shape.laneBitWidth)
    return invalid("Spatial value shape exceeds the invocation ABI");
  return static_cast<std::uint32_t>(shape.lanesPerToken) * shape.laneBitWidth;
}

llvm::Expected<std::vector<std::uint32_t>>
transportBitCounts(llvm::ArrayRef<sim::SpatialSimulationValueShape> shapes) {
  std::vector<std::uint32_t> result;
  result.reserve(shapes.size());
  for (sim::SpatialSimulationValueShape shape : shapes) {
    auto count = transportBitCount(shape);
    if (!count)
      return count.takeError();
    result.push_back(*count);
  }
  return result;
}

llvm::Error verifyCanonicalCallableBoundary(
    const dataflow::CanonicalRootThreadLaunchView &rootView,
    dataflow::GraphLaunchOp graphLaunch,
    llvm::ArrayRef<std::uint32_t> valueBitCounts,
    llvm::ArrayRef<std::uint32_t> resultBitCounts,
    std::string &sourceCallableSymbol) {
  auto rootLaunch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(rootView.op);
  auto thread = llvm::dyn_cast<dataflow::ThreadOp>(rootView.callee);
  auto callable = rootView.op
                      ? rootView.op->getParentOfType<mlir::LLVM::LLVMFuncOp>()
                      : mlir::LLVM::LLVMFuncOp{};
  if (!rootLaunch || !thread || !callable || callable.isExternal() ||
      !callable.getBody().hasOneBlock())
    return invalid("root launch is not owned by one defined LLVM callable");
  if (!rootLaunch.getGridUpperBounds().empty() ||
      !rootLaunch.getAsyncDependencies().empty())
    return invalid(
        "initial dispatch requires one dependency-free rank-zero root");
  const bool returnsValue = !mlir::isa<mlir::LLVM::LLVMVoidType>(
      callable.getFunctionType().getReturnType());
  if (callable.getFunctionType().isVarArg() ||
      callable.getFunctionType().getNumParams() != valueBitCounts.size() ||
      resultBitCounts.size() > 1 || returnsValue != !resultBitCounts.empty())
    return invalid("initial dispatch callable boundary is not scalar C ABI");
  if (rootLaunch.getBodyOperands().size() !=
          valueBitCounts.size() + resultBitCounts.size() ||
      graphLaunch.getValueInputs().size() != valueBitCounts.size() ||
      graphLaunch.getValueResults().size() != resultBitCounts.size())
    return invalid("root and graph value boundaries are not exact");

  mlir::Block &callableBlock = callable.getBody().front();
  mlir::Block &threadBlock = thread.getBody().front();
  for (std::size_t ordinal = 0; ordinal != valueBitCounts.size(); ++ordinal) {
    auto graphArgument = llvm::dyn_cast<mlir::BlockArgument>(
        graphLaunch.getValueInputs()[ordinal]);
    auto callableArgument = llvm::dyn_cast<mlir::BlockArgument>(
        rootLaunch.getBodyOperands()[ordinal]);
    if (!graphArgument || graphArgument.getOwner() != &threadBlock ||
        graphArgument.getArgNumber() != ordinal || !callableArgument ||
        callableArgument.getOwner() != &callableBlock ||
        callableArgument.getArgNumber() != ordinal)
      return invalid("graph value input is not the matching source argument");
  }

  auto returnOp =
      llvm::dyn_cast<mlir::LLVM::ReturnOp>(callableBlock.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != resultBitCounts.size())
    return invalid("source callable return differs from the graph boundary");
  if (!rootLaunch.getAsyncToken().hasOneUse())
    return invalid("source callable has no unique root retirement wait");
  auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
      rootLaunch.getAsyncToken().use_begin()->getOwner());
  if (!wait || wait->getBlock() != returnOp->getBlock() ||
      !wait->isBeforeInBlock(returnOp))
    return invalid("source callable returns before root retirement");

  if (!resultBitCounts.empty()) {
    mlir::Value graphResult = graphLaunch.getValueResults().front();
    if (!graphResult.hasOneUse())
      return invalid("graph value result has no unique publication");
    auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(
        graphResult.use_begin()->getOwner());
    auto threadResultSlot =
        store ? llvm::dyn_cast<mlir::BlockArgument>(store.getAddr())
              : mlir::BlockArgument{};
    if (!store || store.getValue() != graphResult || !threadResultSlot ||
        threadResultSlot.getOwner() != &threadBlock ||
        threadResultSlot.getArgNumber() != valueBitCounts.size())
      return invalid(
          "graph value result is not stored through its thread slot");
    mlir::Value callerResultSlot = rootLaunch.getBodyOperands().back();
    if (!callerResultSlot.getDefiningOp<mlir::LLVM::AllocaOp>() ||
        callerResultSlot !=
            rootLaunch.getBodyOperands()[threadResultSlot.getArgNumber()])
      return invalid("thread result is not owned by the source callable");
    auto load = returnOp.getOperand(0).getDefiningOp<mlir::LLVM::LoadOp>();
    if (!load || load.getAddr() != callerResultSlot ||
        !wait->isBeforeInBlock(load))
      return invalid("source callable does not return the retired graph slot");
  }

  sourceCallableSymbol = callable.getSymName().str();
  return llvm::Error::success();
}

llvm::Expected<std::vector<llvm::SmallVector<mlir::LLVM::CallOp, 4>>>
enumerateDirectInvocationPaths(mlir::ModuleOp module,
                               llvm::StringRef entrySymbol,
                               llvm::StringRef targetSymbol) {
  auto entry = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      mlir::SymbolTable::lookupSymbolIn(module, entrySymbol));
  auto target = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      mlir::SymbolTable::lookupSymbolIn(module, targetSymbol));
  if (!entry || entry.isExternal() || !target || target.isExternal())
    return invalid("dynamic invocation entry or target is not defined");

  std::vector<llvm::SmallVector<mlir::LLVM::CallOp, 4>> result;
  llvm::SmallVector<mlir::LLVM::CallOp, 4> path;
  llvm::DenseSet<mlir::Operation *> active;
  std::function<llvm::Error(mlir::LLVM::LLVMFuncOp)> visit =
      [&](mlir::LLVM::LLVMFuncOp function) -> llvm::Error {
    if (!active.insert(function.getOperation()).second)
      return invalid("dynamic invocation closure is recursive");
    llvm::Error error = llvm::Error::success();
    function.walk([&](mlir::LLVM::CallOp call) {
      if (error)
        return mlir::WalkResult::interrupt();
      if (!call.getCalleeAttr()) {
        error = invalid("dynamic invocation closure contains an indirect call");
        return mlir::WalkResult::interrupt();
      }
      auto callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        return mlir::WalkResult::advance();
      path.push_back(call);
      if (callee == target)
        result.push_back(path);
      else if (llvm::Error nested = visit(callee))
        error = std::move(nested);
      path.pop_back();
      return error ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
    });
    active.erase(function.getOperation());
    return error;
  };
  if (llvm::Error error = visit(entry))
    return std::move(error);
  if (result.empty())
    return invalid("dynamic invocation target has no direct call path");
  return result;
}

llvm::Value *addressAt(llvm::IRBuilder<> &builder, llvm::Value *base,
                       std::uint64_t offset) {
  llvm::Value *address =
      builder.CreateAdd(base, llvm::ConstantInt::get(base->getType(), offset));
  return builder.CreateIntToPtr(
      address, llvm::PointerType::get(builder.getContext(), 0));
}

void storeMmio32(llvm::IRBuilder<> &builder, llvm::Value *base,
                 std::uint64_t offset, llvm::Value *value) {
  llvm::StoreInst *store =
      builder.CreateStore(value, addressAt(builder, base, offset));
  store->setVolatile(true);
  store->setAlignment(llvm::Align(4));
}

void storeMmio32(llvm::IRBuilder<> &builder, llvm::Value *base,
                 std::uint64_t offset, std::uint32_t value) {
  storeMmio32(builder, base, offset,
              llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(builder.getContext()), value));
}

llvm::Value *loadMmio32(llvm::IRBuilder<> &builder, llvm::Value *base,
                        std::uint64_t offset) {
  llvm::LoadInst *load =
      builder.CreateLoad(llvm::Type::getInt32Ty(builder.getContext()),
                         addressAt(builder, base, offset));
  load->setVolatile(true);
  load->setAlignment(llvm::Align(4));
  return load;
}

void storeAddressDescriptor(llvm::IRBuilder<> &builder, llvm::Value *base,
                            std::uint64_t lowOffset, std::uint64_t highOffset,
                            llvm::Value *address) {
  llvm::Type *i32 = llvm::Type::getInt32Ty(builder.getContext());
  storeMmio32(builder, base, lowOffset, builder.CreateTrunc(address, i32));
  storeMmio32(builder, base, highOffset,
              builder.CreateTrunc(builder.CreateLShr(address, 32), i32));
}

void emitFence(llvm::IRBuilder<> &builder) {
  builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent);
}

llvm::Value *bytePointer(llvm::IRBuilder<> &builder, llvm::Value *storage,
                         llvm::ArrayType *storageType, std::size_t offset) {
  llvm::Type *i64 = llvm::Type::getInt64Ty(builder.getContext());
  return builder.CreateInBoundsGEP(
      storageType, storage,
      {llvm::ConstantInt::get(i64, 0), llvm::ConstantInt::get(i64, offset)});
}

void emitFixedMemoryCopy(llvm::IRBuilder<> &builder, llvm::Value *destination,
                         llvm::Value *source, std::uint64_t byteCount) {
  builder.CreateMemCpyInline(
      destination, llvm::MaybeAlign(1), source, llvm::MaybeAlign(1),
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder.getContext()),
                             byteCount));
}

llvm::CallInst *findInvocationCall(llvm::Module &module,
                                   const sim::DirectCallCaptureSite &site) {
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

llvm::Expected<llvm::Value *>
resolveHelperMemorySource(llvm::Module &module, llvm::Function &helper,
                          const sim::DirectCallMemorySource &source) {
  if (const auto *operand =
          std::get_if<sim::DirectCallOperandMemorySource>(&source)) {
    if (operand->operandOrdinal >= helper.arg_size())
      return invalid("invocation memory object exceeds callable arguments");
    return helper.getArg(operand->operandOrdinal);
  }
  const auto &global = std::get<sim::DirectCallGlobalMemorySource>(source);
  llvm::GlobalVariable *resolved =
      module.getGlobalVariable(global.symbol, true);
  if (!resolved)
    return invalid("invocation memory object global is absent");
  return resolved;
}

llvm::Expected<llvm::Value *> resolveHelperValueSource(
    llvm::Module &module, llvm::Function &helper,
    const sim::DirectCallSimulationInputCapturePlan &capture,
    const sim::SimulationValueInputCapture &input, llvm::IRBuilder<> &builder) {
  if (input.boundaryOperandOrdinal) {
    if (*input.boundaryOperandOrdinal >= helper.arg_size())
      return invalid("invocation value input exceeds callable arguments");
    return helper.getArg(*input.boundaryOperandOrdinal);
  }
  if (!input.pointerTarget)
    return invalid("invocation value input has no exact callable source");
  const std::uint64_t rootOrdinal =
      input.pointerTarget->memoryRootBindingOrdinal;
  if (rootOrdinal >= capture.input.memoryRootBindings.size())
    return invalid("invocation pointer target exceeds memory roots");
  const sim::SimulationMemoryRootCapture &root =
      capture.input.memoryRootBindings[rootOrdinal];
  if (root.objectIndex >= capture.input.objects.size() ||
      root.objectIndex >= capture.memoryObjectSources.size())
    return invalid("invocation pointer target names an absent object");
  auto source = resolveHelperMemorySource(
      module, helper, capture.memoryObjectSources[root.objectIndex]);
  if (!source)
    return source.takeError();
  const sim::SimulationMemoryCaptureObject &object =
      capture.input.objects[root.objectIndex];
  if (root.byteOffset == object.operandByteOffset)
    return *source;
  if (root.byteOffset > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max()) ||
      object.operandByteOffset >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("invocation pointer offset exceeds int64");
  const std::int64_t delta =
      static_cast<std::int64_t>(root.byteOffset) -
      static_cast<std::int64_t>(object.operandByteOffset);
  return builder.CreateGEP(
      llvm::Type::getInt8Ty(module.getContext()), *source,
      llvm::ConstantInt::getSigned(llvm::Type::getInt64Ty(module.getContext()),
                                   delta),
      "invocation.pointer");
}

llvm::Error materializeInvocationHelper(
    llvm::Module &module, const ApplicationSpatialInvocationPlan &plan,
    const ApplicationSpatialInvocationPlan::Site &site,
    llvm::CallInst &selected, llvm::GlobalVariable &dispatchBase,
    std::size_t siteOrdinal) {
  llvm::Function *callable = module.getFunction(plan.sourceCallableSymbol);
  if (!callable || callable->isDeclaration() || callable->isVarArg() ||
      callable->arg_size() != plan.valueBitCounts.size() ||
      selected.getCalledFunction() != callable)
    return invalid("final-linked invocation call differs from Dataflow");
  const llvm::DataLayout &layout = module.getDataLayout();
  if (!layout.isLittleEndian() || layout.getPointerSizeInBits() != 64)
    return invalid("dynamic invocation currently requires little-endian RV64");
  for (const auto indexed : llvm::enumerate(callable->args())) {
    llvm::TypeSize bits = layout.getTypeSizeInBits(indexed.value().getType());
    if (bits.isScalable() ||
        bits.getFixedValue() != plan.valueBitCounts[indexed.index()])
      return invalid("source callable argument shape differs from graph input");
  }
  if (plan.resultBitCounts.empty() != callable->getReturnType()->isVoidTy() ||
      plan.resultBitCounts.size() > 1)
    return invalid("source callable result arity differs from graph result");
  if (!plan.resultBitCounts.empty()) {
    llvm::TypeSize resultBits =
        layout.getTypeSizeInBits(callable->getReturnType());
    if (resultBits.isScalable() ||
        resultBits.getFixedValue() != plan.resultBitCounts.front())
      return invalid("source callable result shape differs from graph result");
  }
  const sim::SimulationInputCapturePlan &capture = site.capture.input;
  if (capture.valueInputs.size() != plan.valueBitCounts.size() ||
      capture.valueResults.size() != plan.resultBitCounts.size() ||
      capture.objects.size() != site.capture.memoryObjectSources.size() ||
      site.wireLayout.valuePayloadOffsets.size() !=
          capture.valueInputs.size() ||
      site.wireLayout.memoryAddressOffsets.size() != capture.objects.size() ||
      site.wireLayout.memoryPayloadOffsets.size() != capture.objects.size() ||
      site.wireLayout.resultAddressOffsets.size() !=
          plan.resultBitCounts.size())
    return invalid("invocation capture and wire layout are inconsistent");

  std::string helperName =
      "__loom_spatial_dispatch_" + std::to_string(siteOrdinal);
  if (module.getNamedValue(helperName))
    return invalid("final-linked module defines a reserved dispatch helper");
  llvm::Function *helper = llvm::Function::Create(
      callable->getFunctionType(), llvm::GlobalValue::InternalLinkage,
      helperName, module);
  helper->setCallingConv(selected.getCallingConv());
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(module.getContext(), "entry", helper);
  llvm::IRBuilder<> builder(entry);
  llvm::ArrayType *wireType =
      llvm::ArrayType::get(llvm::Type::getInt8Ty(module.getContext()),
                           site.wireLayout.templateBytes.size());
  auto *wire = new llvm::GlobalVariable(
      module, wireType, false, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantDataArray::get(module.getContext(),
                                   site.wireLayout.templateBytes),
      helperName + ".wire");
  wire->setAlignment(llvm::Align(8));

  for (const auto indexed : llvm::enumerate(capture.valueInputs)) {
    const sim::SimulationValueInputCapture &input = indexed.value();
    if (input.valueInputOrdinal != indexed.index() || input.fixedValue)
      return invalid("invocation value capture is not dense and runtime");
    auto source =
        resolveHelperValueSource(module, *helper, site.capture, input, builder);
    if (!source)
      return source.takeError();
    llvm::Type *wireInteger = llvm::IntegerType::get(
        module.getContext(), plan.valueBitCounts[indexed.index()]);
    llvm::Value *bits = *source;
    if (bits->getType()->isPointerTy())
      bits = builder.CreatePtrToInt(bits, wireInteger);
    else if (bits->getType() != wireInteger)
      bits = builder.CreateBitCast(bits, wireInteger);
    llvm::StoreInst *store = builder.CreateStore(
        bits,
        bytePointer(builder, wire, wireType,
                    site.wireLayout.valuePayloadOffsets[indexed.index()]));
    store->setAlignment(llvm::Align(1));
  }

  for (const auto indexed : llvm::enumerate(capture.objects)) {
    const sim::SimulationMemoryCaptureObject &object = indexed.value();
    if (object.byteCount == 0 || object.operandByteOffset >= object.byteCount ||
        object.operandByteOffset >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
      return invalid("invocation memory capture is not finite");
    auto source = resolveHelperMemorySource(
        module, *helper, site.capture.memoryObjectSources[indexed.index()]);
    if (!source)
      return source.takeError();
    if (!(*source)->getType()->isPointerTy())
      return invalid("invocation memory source is not a pointer");
    llvm::Value *base = *source;
    if (object.operandByteOffset != 0)
      base = builder.CreateGEP(
          llvm::Type::getInt8Ty(module.getContext()), base,
          llvm::ConstantInt::getSigned(
              llvm::Type::getInt64Ty(module.getContext()),
              -static_cast<std::int64_t>(object.operandByteOffset)),
          "invocation.base");
    llvm::Value *address = builder.CreatePtrToInt(
        base, llvm::Type::getInt64Ty(module.getContext()));
    llvm::StoreInst *addressStore = builder.CreateStore(
        address,
        bytePointer(builder, wire, wireType,
                    site.wireLayout.memoryAddressOffsets[indexed.index()]));
    addressStore->setAlignment(llvm::Align(1));
    emitFixedMemoryCopy(
        builder,
        bytePointer(builder, wire, wireType,
                    site.wireLayout.memoryPayloadOffsets[indexed.index()]),
        base, object.byteCount);
  }

  llvm::AllocaInst *result = nullptr;
  if (!plan.resultBitCounts.empty()) {
    result = builder.CreateAlloca(callable->getReturnType(), nullptr, "result");
    result->setAlignment(layout.getABITypeAlign(callable->getReturnType()));
    llvm::Value *resultAddress = builder.CreatePtrToInt(
        result, llvm::Type::getInt64Ty(module.getContext()));
    llvm::StoreInst *resultAddressStore = builder.CreateStore(
        resultAddress,
        bytePointer(builder, wire, wireType,
                    site.wireLayout.resultAddressOffsets.front()));
    resultAddressStore->setAlignment(llvm::Align(1));
  }

  llvm::Value *dispatch = builder.CreateLoad(
      llvm::Type::getInt64Ty(module.getContext()), &dispatchBase);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchControl,
              runtime::gem5ThreadDispatchReset);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchTargetLow,
              static_cast<std::uint32_t>(plan.dispatchTargetOrdinal));
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchTargetHigh,
              static_cast<std::uint32_t>(plan.dispatchTargetOrdinal >> 32));
  llvm::Value *wireAddress =
      builder.CreatePtrToInt(wire, llvm::Type::getInt64Ty(module.getContext()));
  storeAddressDescriptor(
      builder, dispatch, runtime::gem5ThreadDispatchInvocationLow,
      runtime::gem5ThreadDispatchInvocationHigh, wireAddress);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchInvocationSize,
              static_cast<std::uint32_t>(site.wireLayout.templateBytes.size()));
  emitFence(builder);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchControl,
              runtime::gem5ThreadDispatchStart);
  builder.CreateBr(
      llvm::BasicBlock::Create(module.getContext(), "poll", helper));

  llvm::BasicBlock *poll = &helper->back();
  llvm::BasicBlock *failed =
      llvm::BasicBlock::Create(module.getContext(), "failed", helper);
  llvm::BasicBlock *complete =
      llvm::BasicBlock::Create(module.getContext(), "complete", helper);
  builder.SetInsertPoint(poll);
  llvm::Value *status =
      loadMmio32(builder, dispatch, runtime::gem5ThreadDispatchStatus);
  llvm::Value *hasFailed = builder.CreateICmpNE(
      builder.CreateAnd(status, runtime::gem5ThreadDispatchFailed),
      llvm::ConstantInt::get(status->getType(), 0));
  llvm::BasicBlock *checkDone =
      llvm::BasicBlock::Create(module.getContext(), "check_done", helper);
  builder.CreateCondBr(hasFailed, failed, checkDone);
  builder.SetInsertPoint(checkDone);
  llvm::Value *isDone = builder.CreateICmpNE(
      builder.CreateAnd(status, runtime::gem5ThreadDispatchDone),
      llvm::ConstantInt::get(status->getType(), 0));
  builder.CreateCondBr(isDone, complete, poll);
  builder.SetInsertPoint(failed);
  builder.CreateBr(failed);
  builder.SetInsertPoint(complete);
  emitFence(builder);
  if (result) {
    llvm::LoadInst *loaded =
        builder.CreateLoad(callable->getReturnType(), result);
    loaded->setAlignment(layout.getABITypeAlign(callable->getReturnType()));
    builder.CreateRet(loaded);
  } else {
    builder.CreateRetVoid();
  }
  selected.setCalledFunction(helper);
  selected.removeFnAttr(llvm::Attribute::Memory);
  selected.removeFnAttr(llvm::Attribute::MustProgress);
  selected.removeFnAttr(llvm::Attribute::NoSync);
  selected.removeFnAttr(llvm::Attribute::WillReturn);
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> fixedStoreSize(const llvm::DataLayout &layout,
                                             llvm::Type *type) {
  llvm::TypeSize size = layout.getTypeStoreSize(type);
  if (size.isScalable() || size.getFixedValue() == 0)
    return invalid("program result has no fixed nonzero storage size");
  return size.getFixedValue();
}

llvm::Error addHostEntry(llvm::Module &module, llvm::StringRef applicationEntry,
                         llvm::GlobalVariable &dispatchBase,
                         std::uint64_t targetCount) {
  if (module.getFunction(applicationHostEntrySymbol))
    return invalid("final-linked module defines the reserved host entry");
  llvm::Function *application = module.getFunction(applicationEntry);
  if (!application || application->isDeclaration() || application->isVarArg() ||
      !application->arg_empty() || application->getReturnType()->isVoidTy())
    return invalid(
        "initial host entry requires a nullary value-returning function");
  auto resultBytes =
      fixedStoreSize(module.getDataLayout(), application->getReturnType());
  if (!resultBytes)
    return resultBytes.takeError();
  llvm::Type *i64 = llvm::Type::getInt64Ty(module.getContext());
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(i64, {i64, i64, i64, i64, i64, i64}, false),
      llvm::GlobalValue::ExternalLinkage, applicationHostEntrySymbol, module);
  auto argument = entry->arg_begin();
  llvm::Value *dispatch = &*argument++;
  dispatch->setName("dispatch");
  llvm::Value *actualTargets = &*argument++;
  actualTargets->setName("target_count");
  (&*argument++)->setName("memory_table");
  (&*argument++)->setName("memory_table_entries");
  llvm::Value *resultAddress = &*argument++;
  resultAddress->setName("result_address");
  llvm::Value *actualResultBytes = &*argument;
  actualResultBytes->setName("result_bytes");

  llvm::BasicBlock *begin =
      llvm::BasicBlock::Create(module.getContext(), "entry", entry);
  llvm::BasicBlock *run =
      llvm::BasicBlock::Create(module.getContext(), "run", entry);
  llvm::BasicBlock *failed =
      llvm::BasicBlock::Create(module.getContext(), "failed", entry);
  llvm::IRBuilder<> builder(begin);
  llvm::Value *valid = builder.CreateAnd(
      builder.CreateICmpEQ(actualTargets,
                           llvm::ConstantInt::get(i64, targetCount)),
      builder.CreateICmpNE(resultAddress, llvm::ConstantInt::get(i64, 0)));
  valid = builder.CreateAnd(
      valid, builder.CreateICmpEQ(actualResultBytes,
                                  llvm::ConstantInt::get(i64, *resultBytes)));
  builder.CreateCondBr(valid, run, failed);
  builder.SetInsertPoint(failed);
  builder.CreateBr(failed);
  builder.SetInsertPoint(run);
  builder.CreateStore(dispatch, &dispatchBase);
  llvm::CallInst *programResult = builder.CreateCall(application);
  llvm::StoreInst *store = builder.CreateStore(
      programResult,
      builder.CreateIntToPtr(resultAddress,
                             llvm::PointerType::get(module.getContext(), 0)));
  store->setAlignment(llvm::Align(1));
  builder.CreateRet(llvm::ConstantInt::get(i64, 0));
  return llvm::Error::success();
}

void emitInstructionEntry(llvm::Module &module, std::uint64_t ordinal) {
  llvm::LLVMContext &context = module.getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                              {i64, i64, i64, i64, i64, i64}, false),
      llvm::GlobalValue::ExternalLinkage,
      "__loom_thread_entry_" + std::to_string(ordinal), module);
  auto argument = entry->arg_begin();
  llvm::Value *bridge = &*argument++;
  llvm::Value *staticAddress = &*argument++;
  llvm::Value *staticSize = &*argument++;
  llvm::Value *dispatch = &*argument++;
  llvm::Value *invocationAddress = &*argument++;
  llvm::Value *invocationSize = &*argument;
  llvm::BasicBlock *begin = llvm::BasicBlock::Create(context, "entry", entry);
  llvm::BasicBlock *poll = llvm::BasicBlock::Create(context, "poll", entry);
  llvm::BasicBlock *checkDone =
      llvm::BasicBlock::Create(context, "check_done", entry);
  llvm::BasicBlock *complete =
      llvm::BasicBlock::Create(context, "complete", entry);
  llvm::BasicBlock *failed = llvm::BasicBlock::Create(context, "failed", entry);
  llvm::BasicBlock *suspend =
      llvm::BasicBlock::Create(context, "suspend", entry);
  llvm::IRBuilder<> builder(begin);
  storeMmio32(builder, bridge, runtime::gem5SpatialBridgeControl,
              runtime::gem5SpatialBridgeReset);
  storeAddressDescriptor(
      builder, bridge, runtime::gem5SpatialBridgeStaticLaunchLow,
      runtime::gem5SpatialBridgeStaticLaunchHigh, staticAddress);
  storeMmio32(builder, bridge, runtime::gem5SpatialBridgeStaticLaunchSize,
              builder.CreateTrunc(staticSize, llvm::Type::getInt32Ty(context)));
  storeAddressDescriptor(
      builder, bridge, runtime::gem5SpatialBridgeInvocationLow,
      runtime::gem5SpatialBridgeInvocationHigh, invocationAddress);
  storeMmio32(
      builder, bridge, runtime::gem5SpatialBridgeInvocationSize,
      builder.CreateTrunc(invocationSize, llvm::Type::getInt32Ty(context)));
  emitFence(builder);
  storeMmio32(builder, bridge, runtime::gem5SpatialBridgeControl,
              runtime::gem5SpatialBridgeStart);
  builder.CreateBr(poll);
  builder.SetInsertPoint(poll);
  llvm::Value *status =
      loadMmio32(builder, bridge, runtime::gem5SpatialBridgeStatus);
  llvm::Value *hasFailed = builder.CreateICmpNE(
      builder.CreateAnd(status, runtime::gem5SpatialBridgeFailed),
      llvm::ConstantInt::get(status->getType(), 0));
  builder.CreateCondBr(hasFailed, failed, checkDone);
  builder.SetInsertPoint(checkDone);
  llvm::Value *isDone = builder.CreateICmpNE(
      builder.CreateAnd(status, runtime::gem5SpatialBridgeDone),
      llvm::ConstantInt::get(status->getType(), 0));
  builder.CreateCondBr(isDone, complete, poll);
  builder.SetInsertPoint(complete);
  emitFence(builder);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchCompletion, 1);
  builder.CreateBr(suspend);
  builder.SetInsertPoint(failed);
  llvm::Value *error =
      loadMmio32(builder, bridge, runtime::gem5SpatialBridgeError);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchWorkerFailure,
              error);
  builder.CreateBr(suspend);
  builder.SetInsertPoint(suspend);
  llvm::InlineAsm *wfi = llvm::InlineAsm::get(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false), "wfi", "",
      true);
  builder.CreateCall(wfi);
  builder.CreateBr(suspend);
}

} // namespace

llvm::Expected<ApplicationSpatialInvocationPlan>
deriveApplicationSpatialInvocationPlan(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::StringRef entrySymbol) {
  auto roots =
      dataflow.projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!roots)
    return roots.takeError();
  if (roots->size() != 1)
    return invalid("initial dispatch requires exactly one reachable root");
  const dataflow::RootThreadLaunchRef root = roots->front();
  std::vector<dataflow::RootedGraphLaunchRef> graphs;
  dataflow.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef graph) {
    if (graph.rootThreadLaunch == root)
      graphs.push_back(graph);
  });
  if (graphs.size() != 1)
    return invalid("initial dispatch requires exactly one graph in the root");
  auto rootView = dataflow.resolve(root);
  if (!rootView)
    return rootView.takeError();
  auto graphView = dataflow.resolve(graphs.front().staticGraphLaunch);
  if (!graphView)
    return graphView.takeError();
  auto graphLaunch = llvm::dyn_cast<dataflow::GraphLaunchOp>(graphView->op);
  if (!graphLaunch || !graphLaunch.getStreamInputs().empty() ||
      !graphLaunch.getStreamOutputs().empty() ||
      !graphLaunch.getMemoryResults().empty())
    return invalid(
        "initial dynamic invocation requires a stream-free imported-memory "
        "graph");
  auto shapes =
      sim::projectSpatialSimulationBoundaryShapes(dataflow, graphs.front());
  if (!shapes)
    return shapes.takeError();
  auto valueBitCounts = transportBitCounts(shapes->valueInputs);
  auto resultBitCounts = transportBitCounts(shapes->valueResults);
  if (!valueBitCounts || !resultBitCounts)
    return llvm::joinErrors(
        valueBitCounts ? llvm::Error::success() : valueBitCounts.takeError(),
        resultBitCounts ? llvm::Error::success() : resultBitCounts.takeError());
  std::string sourceCallableSymbol;
  if (llvm::Error error = verifyCanonicalCallableBoundary(
          *rootView, graphLaunch, *valueBitCounts, *resultBitCounts,
          sourceCallableSymbol))
    return std::move(error);

  auto module = rootView->op->getParentOfType<mlir::ModuleOp>();
  auto paths =
      enumerateDirectInvocationPaths(module, entrySymbol, sourceCallableSymbol);
  if (!paths)
    return paths.takeError();
  std::vector<ApplicationSpatialInvocationPlan::Site> sites;
  sites.reserve(paths->size());
  std::vector<std::pair<std::string, std::uint64_t>> leafLocators;
  for (llvm::ArrayRef<mlir::LLVM::CallOp> path : *paths) {
    auto capture =
        sim::deriveSimulationInputCapturePlan(dataflow, graphs.front(), path);
    if (!capture)
      return capture.takeError();
    if (capture->invocationPath.empty())
      return invalid("dynamic invocation capture has no call locator");
    const sim::DirectCallCaptureSite &leaf = capture->invocationPath.back();
    const std::pair<std::string, std::uint64_t> leafLocator{
        leaf.hostCallerSymbol, leaf.hostCallOrdinal};
    if (llvm::is_contained(leafLocators, leafLocator))
      return invalid(
          "one dynamic invocation call is reachable through multiple paths");
    leafLocators.push_back(leafLocator);

    if (capture->input.valueInputs.size() != valueBitCounts->size() ||
        capture->input.valueResults.size() != resultBitCounts->size() ||
        capture->input.objects.size() != capture->memoryObjectSources.size())
      return invalid("dynamic invocation capture differs from graph boundary");
    std::vector<runtime::SpatialInvocationValueLayout> valueLayouts;
    valueLayouts.reserve(valueBitCounts->size());
    for (const auto indexed : llvm::enumerate(capture->input.valueInputs)) {
      const sim::SimulationValueInputCapture &input = indexed.value();
      if (input.valueInputOrdinal != indexed.index() || input.fixedValue ||
          input.byteCount != ((*valueBitCounts)[indexed.index()] + 7) / 8)
        return invalid("dynamic invocation value capture is not exact");
      std::optional<runtime::SpatialInvocationPointerTarget> pointerTarget;
      if (input.pointerTarget) {
        const std::uint64_t rootOrdinal =
            input.pointerTarget->memoryRootBindingOrdinal;
        if (rootOrdinal >= capture->input.memoryRootBindings.size())
          return invalid("dynamic invocation pointer target exceeds roots");
        const sim::SimulationMemoryRootCapture &binding =
            capture->input.memoryRootBindings[rootOrdinal];
        if (binding.objectIndex > std::numeric_limits<std::uint32_t>::max())
          return invalid("dynamic invocation object ordinal exceeds ABI");
        pointerTarget = runtime::SpatialInvocationPointerTarget{
            static_cast<std::uint32_t>(binding.objectIndex),
            binding.byteOffset};
      }
      valueLayouts.push_back(
          {(*valueBitCounts)[indexed.index()], pointerTarget});
    }
    std::vector<runtime::SpatialInvocationMemoryObjectLayout> objectLayouts;
    objectLayouts.reserve(capture->input.objects.size());
    for (const sim::SimulationMemoryCaptureObject &object :
         capture->input.objects)
      objectLayouts.push_back({object.byteCount});
    std::vector<runtime::SpatialInvocationMemoryRootBinding> rootBindings;
    rootBindings.reserve(capture->input.memoryRootBindings.size());
    for (const sim::SimulationMemoryRootCapture &binding :
         capture->input.memoryRootBindings) {
      if (binding.objectIndex > std::numeric_limits<std::uint32_t>::max())
        return invalid("dynamic invocation object ordinal exceeds ABI");
      rootBindings.push_back({binding.root.entity.value(),
                              static_cast<std::uint32_t>(binding.objectIndex),
                              binding.byteOffset});
    }
    runtime::SpatialInvocationWireLayout wireLayout;
    std::string diagnostic;
    if (!runtime::projectSpatialInvocationWireLayout(
            dataflow.identity().bytes(), root.entity.value(),
            graphs.front().staticGraphLaunch.entity.value(), {}, valueLayouts,
            objectLayouts, rootBindings, *resultBitCounts, wireLayout,
            diagnostic))
      return invalid(diagnostic);
    if (wireLayout.templateBytes.size() >
        std::numeric_limits<std::uint32_t>::max())
      return invalid("invocation wire exceeds the dispatch size register");
    sites.push_back({std::move(*capture), std::move(wireLayout)});
  }
  return ApplicationSpatialInvocationPlan{root,
                                          graphs.front(),
                                          std::move(sourceCallableSymbol),
                                          0,
                                          std::move(*valueBitCounts),
                                          std::move(*resultBitCounts),
                                          std::move(sites)};
}

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeHostDispatchModule(const llvm::Module &finalLinkedModule,
                              llvm::StringRef applicationEntry,
                              const ApplicationSpatialInvocationPlan &plan) {
  auto module = llvm::CloneModule(finalLinkedModule);
  llvm::Type *i64 = llvm::Type::getInt64Ty(module->getContext());
  if (module->getGlobalVariable("__loom_dispatch_base", true))
    return invalid("final-linked module defines the reserved dispatch base");
  auto *dispatchBase = new llvm::GlobalVariable(
      *module, i64, false, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantInt::get(i64, 0), "__loom_dispatch_base");
  std::vector<llvm::CallInst *> calls;
  calls.reserve(plan.sites.size());
  llvm::DenseSet<llvm::CallInst *> uniqueCalls;
  for (const ApplicationSpatialInvocationPlan::Site &site : plan.sites) {
    if (site.capture.invocationPath.empty())
      return invalid("dynamic invocation site has no leaf locator");
    llvm::CallInst *call =
        findInvocationCall(*module, site.capture.invocationPath.back());
    if (!call || !uniqueCalls.insert(call).second)
      return invalid(
          "final-linked dynamic invocation site is absent or reused");
    calls.push_back(call);
  }
  for (const auto indexed : llvm::enumerate(plan.sites))
    if (llvm::Error error = materializeInvocationHelper(
            *module, plan, indexed.value(), *calls[indexed.index()],
            *dispatchBase, indexed.index()))
      return std::move(error);
  if (llvm::Error error = addHostEntry(*module, applicationEntry, *dispatchBase,
                                       plan.dispatchTargetOrdinal + 1))
    return std::move(error);
  return module;
}

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeInstructionDispatchModule(const llvm::Module &finalLinkedModule,
                                     std::uint64_t entryCount) {
  if (entryCount == 0)
    return invalid("InstructionCore dispatch module has no entries");
  auto module = llvm::CloneModule(finalLinkedModule);
  std::vector<llvm::Function *> functions;
  for (llvm::Function &function : module->functions())
    functions.push_back(&function);
  for (llvm::Function *function : functions)
    function->eraseFromParent();
  std::vector<llvm::GlobalVariable *> globals;
  for (llvm::GlobalVariable &global : module->globals())
    globals.push_back(&global);
  for (llvm::GlobalVariable *global : globals)
    global->eraseFromParent();
  for (std::uint64_t ordinal = 0; ordinal != entryCount; ++ordinal)
    emitInstructionEntry(*module, ordinal);
  return module;
}

} // namespace loom::application::detail
