#include "ExecutionGlue.h"
#include "LoomFreestandingMathBitcode.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Runtime/Gem5DispatchABI.h"
#include "Runtime/Gem5SpatialBridgeABI.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
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

llvm::Error verifySelectedCallableBoundary(
    const dataflow::CanonicalRootThreadLaunchView &rootView,
    dataflow::GraphLaunchOp graphLaunch,
    llvm::ArrayRef<std::uint32_t> valueBitCounts,
    llvm::ArrayRef<std::uint32_t> resultBitCounts,
    std::vector<std::uint64_t> &resultRootOperandOrdinals,
    std::string &sourceCallableSymbol) {
  auto rootLaunch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(rootView.op);
  auto thread = llvm::dyn_cast<dataflow::ThreadOp>(rootView.callee);
  auto callable = rootView.op
                      ? rootView.op->getParentOfType<mlir::LLVM::LLVMFuncOp>()
                      : mlir::LLVM::LLVMFuncOp{};
  if (!rootLaunch || !thread || !callable || callable.isExternal() ||
      !callable.getBody().hasOneBlock())
    return invalid("root launch is not owned by one defined LLVM callable");
  if (!rootLaunch.getAsyncDependencies().empty())
    return invalid("initial dispatch requires a dependency-free root");
  if (callable.getFunctionType().isVarArg())
    return invalid("initial dispatch callable boundary is variadic");
  if (graphLaunch.getValueInputs().size() != valueBitCounts.size() ||
      graphLaunch.getValueResults().size() != resultBitCounts.size())
    return invalid("root and graph value boundaries are not exact");

  mlir::Block &threadBlock = thread.getBody().front();
  if (!rootLaunch.getAsyncToken().hasOneUse())
    return invalid("source callable has no unique root retirement wait");
  auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
      rootLaunch.getAsyncToken().use_begin()->getOwner());
  if (!wait || wait->getBlock() != rootLaunch->getBlock() ||
      !rootLaunch->isBeforeInBlock(wait))
    return invalid("source callable does not retire the selected root");

  resultRootOperandOrdinals.clear();
  resultRootOperandOrdinals.reserve(resultBitCounts.size());
  for (mlir::Value graphResult : graphLaunch.getValueResults()) {
    if (!graphResult.hasOneUse())
      return invalid("graph value result has no unique publication");
    auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(
        graphResult.use_begin()->getOwner());
    auto threadResultSlot =
        store ? llvm::dyn_cast<mlir::BlockArgument>(store.getAddr())
              : mlir::BlockArgument{};
    if (!store || store.getValue() != graphResult || !threadResultSlot ||
        threadResultSlot.getOwner() != &threadBlock ||
        threadResultSlot.getArgNumber() >= rootLaunch.getBodyOperands().size())
      return invalid(
          "graph value result is not stored through its thread slot");
    mlir::Value callerResultSlot =
        rootLaunch.getBodyOperands()[threadResultSlot.getArgNumber()];
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(callerResultSlot.getType()))
      return invalid("thread result slot is not a pointer");
    resultRootOperandOrdinals.push_back(threadResultSlot.getArgNumber());
  }

  sourceCallableSymbol = callable.getSymName().str();
  return llvm::Error::success();
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

void storeMmio64Descriptor(llvm::IRBuilder<> &builder, llvm::Value *base,
                           std::uint64_t lowOffset, std::uint64_t highOffset,
                           llvm::Value *value) {
  llvm::Type *i32 = llvm::Type::getInt32Ty(builder.getContext());
  storeMmio32(builder, base, lowOffset, builder.CreateTrunc(value, i32));
  storeMmio32(builder, base, highOffset,
              builder.CreateTrunc(builder.CreateLShr(value, 32), i32));
}

llvm::Value *loadMmio64Descriptor(llvm::IRBuilder<> &builder, llvm::Value *base,
                                  std::uint64_t lowOffset,
                                  std::uint64_t highOffset) {
  llvm::Type *i64 = llvm::Type::getInt64Ty(builder.getContext());
  llvm::Value *low =
      builder.CreateZExt(loadMmio32(builder, base, lowOffset), i64);
  llvm::Value *high =
      builder.CreateZExt(loadMmio32(builder, base, highOffset), i64);
  return builder.CreateOr(low, builder.CreateShl(high, 32));
}

void emitFence(llvm::IRBuilder<> &builder) {
  builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent);
}

void emitRootLifecycleEvent(llvm::IRBuilder<> &builder, llvm::Value *dispatch,
                            dataflow::RootThreadLaunchRef root,
                            llvm::Value *occurrence,
                            runtime::Gem5RootLifecycleAction action) {
  const std::uint64_t entity = root.entity.value();
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchRootEntityLow,
              static_cast<std::uint32_t>(entity));
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchRootEntityHigh,
              static_cast<std::uint32_t>(entity >> 32));
  storeMmio64Descriptor(
      builder, dispatch, runtime::gem5ThreadDispatchRootOccurrenceLow,
      runtime::gem5ThreadDispatchRootOccurrenceHigh, occurrence);
  emitFence(builder);
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchRootEvent,
              static_cast<std::uint32_t>(action));
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
  llvm::LLVMContext &context = builder.getContext();
  llvm::Type *i64 = llvm::Type::getInt64Ty(context);
  // The freestanding host image has no memcpy symbol, so the copy must be
  // materialized in place. Forced inline expansion turns every byte into a
  // straight-line unaligned access and a multi-kilobyte capture into one
  // giant basic block whose instruction selection cost is superlinear; a
  // compact byte loop keeps both code generation and the emitted image small.
  constexpr std::uint64_t inlineCopyLimitBytes = 64;
  if (byteCount <= inlineCopyLimitBytes) {
    builder.CreateMemCpyInline(destination, llvm::MaybeAlign(1), source,
                               llvm::MaybeAlign(1),
                               llvm::ConstantInt::get(i64, byteCount));
    return;
  }
  llvm::Type *i8 = llvm::Type::getInt8Ty(context);
  llvm::Function *function = builder.GetInsertBlock()->getParent();
  llvm::BasicBlock *preheader = builder.GetInsertBlock();
  llvm::BasicBlock *loop =
      llvm::BasicBlock::Create(context, "fixed.copy.loop", function);
  llvm::BasicBlock *done =
      llvm::BasicBlock::Create(context, "fixed.copy.done", function);
  builder.CreateBr(loop);
  builder.SetInsertPoint(loop);
  llvm::PHINode *index = builder.CreatePHI(i64, 2, "fixed.copy.index");
  index->addIncoming(llvm::ConstantInt::get(i64, 0), preheader);
  llvm::LoadInst *byte =
      builder.CreateLoad(i8, builder.CreateGEP(i8, source, index));
  byte->setAlignment(llvm::Align(1));
  llvm::StoreInst *store =
      builder.CreateStore(byte, builder.CreateGEP(i8, destination, index));
  store->setAlignment(llvm::Align(1));
  llvm::Value *next =
      builder.CreateAdd(index, llvm::ConstantInt::get(i64, 1));
  index->addIncoming(next, loop);
  builder.CreateCondBr(
      builder.CreateICmpULT(next, llvm::ConstantInt::get(i64, byteCount)),
      loop, done);
  builder.SetInsertPoint(done);
}

llvm::CallInst *findDirectCall(llvm::Function &caller,
                               llvm::StringRef calleeSymbol,
                               std::uint64_t callOrdinal) {
  if (caller.isDeclaration())
    return nullptr;
  std::uint64_t ordinal = 0;
  for (llvm::BasicBlock &block : caller)
    for (llvm::Instruction &instruction : block) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
      llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
      if (!callee || callee->getName() != calleeSymbol)
        continue;
      if (ordinal++ == callOrdinal)
        return call;
    }
  return nullptr;
}

llvm::CallInst *findInvocationCall(llvm::Module &module,
                                   const sim::DirectCallCaptureSite &site) {
  llvm::Function *caller = module.getFunction(site.hostCallerSymbol);
  if (!caller || !module.getFunction(site.hostCalleeSymbol))
    return nullptr;
  return findDirectCall(*caller, site.hostCalleeSymbol, site.hostCallOrdinal);
}

bool sameDirectCallSite(const sim::DirectCallCaptureSite &lhs,
                        const sim::DirectCallCaptureSite &rhs) {
  return lhs.hostCallerSymbol == rhs.hostCallerSymbol &&
         lhs.hostCalleeSymbol == rhs.hostCalleeSymbol &&
         lhs.hostCallOrdinal == rhs.hostCallOrdinal;
}

bool isDirectCallPathPrefix(llvm::ArrayRef<sim::DirectCallCaptureSite> prefix,
                            llvm::ArrayRef<sim::DirectCallCaptureSite> path) {
  return prefix.size() <= path.size() &&
         std::equal(prefix.begin(), prefix.end(), path.begin(),
                    sameDirectCallSite);
}

bool directCallPathLess(llvm::ArrayRef<sim::DirectCallCaptureSite> lhs,
                        llvm::ArrayRef<sim::DirectCallCaptureSite> rhs) {
  return std::lexicographical_compare(
      lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
      [](const sim::DirectCallCaptureSite &left,
         const sim::DirectCallCaptureSite &right) {
        return std::tuple(left.hostCallerSymbol, left.hostCalleeSymbol,
                          left.hostCallOrdinal) <
               std::tuple(right.hostCallerSymbol, right.hostCalleeSymbol,
                          right.hostCallOrdinal);
      });
}

llvm::Expected<std::uint64_t>
dispatchOperandOrdinal(const ApplicationSpatialInvocationPlan::Launch &launch,
                       std::uint64_t rootOperandOrdinal) {
  auto found =
      llvm::find(launch.dispatchRootOperandOrdinals, rootOperandOrdinal);
  if (found == launch.dispatchRootOperandOrdinals.end())
    return invalid("root operand is absent from the dispatch ABI");
  return static_cast<std::uint64_t>(
      std::distance(launch.dispatchRootOperandOrdinals.begin(), found));
}

struct MaterializedRootDispatch final {
  llvm::Function *start = nullptr;
  llvm::Function *wait = nullptr;
};

llvm::Expected<MaterializedRootDispatch> materializeRootDispatchHelpers(
    llvm::Module &module,
    const ApplicationSpatialInvocationPlan::Launch &launch,
    const ApplicationSpatialInvocationPlan::Site &site,
    llvm::FunctionType *startType, llvm::FunctionType *waitType,
    llvm::GlobalVariable &dispatchBase, std::size_t siteOrdinal,
    std::size_t launchOrdinal) {
  const llvm::DataLayout &layout = module.getDataLayout();
  if (!layout.isLittleEndian() || layout.getPointerSizeInBits() != 64)
    return invalid("dynamic invocation currently requires little-endian RV64");
  llvm::Type *i64 = llvm::Type::getInt64Ty(module.getContext());
  if (!startType || startType->getReturnType() != i64 ||
      startType->isVarArg() || !waitType || waitType->isVarArg() ||
      !waitType->getReturnType()->isVoidTy() || waitType->getNumParams() != 1 ||
      waitType->getParamType(0) != i64)
    return invalid("selected root dispatch helper types are inconsistent");
  const sim::SimulationInputCapturePlan &capture = site.capture.input;
  if (launch.points.empty() ||
      site.pointWireLayouts.size() != launch.points.size() ||
      capture.valueInputs.size() != launch.valueBitCounts.size() ||
      capture.valueResults.size() != launch.resultBitCounts.size() ||
      capture.objects.size() != site.memoryObjectSources.size() ||
      capture.memoryRootBindings.size() != site.memoryRootSources.size())
    return invalid("invocation capture and wire layout are inconsistent");
  for (const runtime::SpatialInvocationWireLayout &wireLayout :
       site.pointWireLayouts)
    if (wireLayout.valuePayloadOffsets.size() != capture.valueInputs.size() ||
        wireLayout.valuePointerTargetOffsetOffsets.size() !=
            capture.valueInputs.size() ||
        wireLayout.memoryAddressOffsets.size() != capture.objects.size() ||
        wireLayout.memoryPayloadOffsets.size() != capture.objects.size() ||
        wireLayout.memoryRootByteOffsetOffsets.size() !=
            capture.memoryRootBindings.size() ||
        wireLayout.resultAddressOffsets.size() != launch.resultBitCounts.size())
      return invalid("point invocation wire layout is inconsistent");

  std::string helperName = "__loom_spatial_dispatch_" +
                           std::to_string(siteOrdinal) + "_" +
                           std::to_string(launchOrdinal);
  const std::string waitName = helperName + "_wait";
  const std::string handlesName = helperName + ".occurrences";
  const std::string activeName = helperName + ".active_generation";
  const std::string rootOccurrenceName = helperName + ".root_occurrence";
  const std::string nextName = helperName + ".next_generation";
  if (module.getNamedValue(helperName) || module.getNamedValue(waitName) ||
      module.getNamedValue(handlesName) || module.getNamedValue(activeName) ||
      module.getNamedValue(rootOccurrenceName) ||
      module.getNamedValue(nextName))
    return invalid("final-linked module defines a reserved dispatch helper");
  llvm::ArrayType *handlesType =
      llvm::ArrayType::get(i64, launch.points.size());
  auto *handles = new llvm::GlobalVariable(
      module, handlesType, false, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantAggregateZero::get(handlesType), handlesName);
  handles->setAlignment(llvm::Align(8));
  auto *activeGeneration = new llvm::GlobalVariable(
      module, i64, false, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantInt::get(i64, 0), activeName);
  activeGeneration->setAlignment(llvm::Align(8));
  auto *activeRootOccurrence = new llvm::GlobalVariable(
      module, i64, false, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantInt::get(i64, 0), rootOccurrenceName);
  activeRootOccurrence->setAlignment(llvm::Align(8));
  auto *nextGeneration = new llvm::GlobalVariable(
      module, i64, false, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantInt::get(i64, 1), nextName);
  nextGeneration->setAlignment(llvm::Align(8));

  llvm::Function *helper = llvm::Function::Create(
      startType, llvm::GlobalValue::InternalLinkage, helperName, module);
  helper->setDoesNotThrow();
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(module.getContext(), "entry", helper);
  llvm::BasicBlock *prepare =
      llvm::BasicBlock::Create(module.getContext(), "prepare", helper);
  llvm::BasicBlock *begin =
      llvm::BasicBlock::Create(module.getContext(), "begin", helper);
  llvm::BasicBlock *failed =
      llvm::BasicBlock::Create(module.getContext(), "failed", helper);
  llvm::IRBuilder<> builder(entry);
  llvm::Value *inactive =
      builder.CreateICmpEQ(builder.CreateLoad(i64, activeGeneration),
                           llvm::ConstantInt::get(i64, 0));
  builder.CreateCondBr(inactive, prepare, failed);
  builder.SetInsertPoint(prepare);
  llvm::Value *generation = builder.CreateLoad(i64, nextGeneration);
  llvm::Value *validGeneration =
      builder.CreateICmpNE(generation, llvm::ConstantInt::get(i64, 0));
  builder.CreateCondBr(validGeneration, begin, failed);
  builder.SetInsertPoint(begin);
  builder.CreateStore(
      builder.CreateAdd(generation, llvm::ConstantInt::get(i64, 1)),
      nextGeneration);
  builder.CreateStore(generation, activeGeneration);
  for (const auto pointIndexed : llvm::enumerate(launch.points)) {
    const ApplicationSpatialInvocationPlan::Launch::Point &point =
        pointIndexed.value();
    const runtime::SpatialInvocationWireLayout &wireLayout =
        site.pointWireLayouts[pointIndexed.index()];
    llvm::ArrayType *wireType =
        llvm::ArrayType::get(llvm::Type::getInt8Ty(module.getContext()),
                             wireLayout.templateBytes.size());
    auto *wire = new llvm::GlobalVariable(
        module, wireType, false, llvm::GlobalValue::PrivateLinkage,
        llvm::ConstantDataArray::get(module.getContext(),
                                     wireLayout.templateBytes),
        helperName + ".wire." + std::to_string(pointIndexed.index()));
    wire->setAlignment(llvm::Align(8));

    for (const auto indexed : llvm::enumerate(capture.valueInputs)) {
      const sim::SimulationValueInputCapture &input = indexed.value();
      if (input.valueInputOrdinal != indexed.index())
        return invalid("invocation value capture is not dense");
      llvm::Type *wireInteger = llvm::IntegerType::get(
          module.getContext(), launch.valueBitCounts[indexed.index()]);
      llvm::Value *bits = nullptr;
      if (input.fixedValue) {
        if (input.boundaryOperandOrdinal || input.denseCoordinateDimension)
          return invalid("fixed invocation value has another source");
        auto packed = sim::packDefinedSpatialSimulationToken(
            *input.fixedValue, {input.lanesPerToken, input.laneBitWidth}, 0);
        if (!packed)
          return packed.takeError();
        if (packed->getBitWidth() != launch.valueBitCounts[indexed.index()])
          return invalid("fixed invocation value has the wrong bit width");
        bits = llvm::ConstantInt::get(module.getContext(), *packed);
      } else if (input.denseCoordinateDimension) {
        if (input.boundaryOperandOrdinal ||
            *input.denseCoordinateDimension >= point.denseCoordinates.size())
          return invalid("dense invocation coordinate has another source");
        const std::uint64_t coordinate =
            point.denseCoordinates[*input.denseCoordinateDimension];
        llvm::APInt coordinateBits(64, coordinate);
        if (coordinateBits.getActiveBits() >
            launch.valueBitCounts[indexed.index()])
          return invalid("dense invocation coordinate exceeds graph width");
        bits = llvm::ConstantInt::get(wireInteger, coordinate);
      } else {
        if (!input.boundaryOperandOrdinal)
          return invalid("runtime invocation value has no root operand");
        auto dispatchOrdinal =
            dispatchOperandOrdinal(launch, *input.boundaryOperandOrdinal);
        if (!dispatchOrdinal || *dispatchOrdinal >= helper->arg_size())
          return dispatchOrdinal
                     ? invalid("runtime invocation value exceeds ABI")
                     : dispatchOrdinal.takeError();
        bits = helper->getArg(*dispatchOrdinal);
        llvm::TypeSize sourceBits = layout.getTypeSizeInBits(bits->getType());
        if (sourceBits.isScalable() ||
            sourceBits.getFixedValue() !=
                launch.valueBitCounts[indexed.index()])
          return invalid("root value operand differs from graph input width");
        if (bits->getType()->isPointerTy())
          bits = builder.CreatePtrToInt(bits, wireInteger);
        else if (bits->getType() != wireInteger)
          bits = builder.CreateBitCast(bits, wireInteger);
      }
      llvm::StoreInst *store = builder.CreateStore(
          bits, bytePointer(builder, wire, wireType,
                            wireLayout.valuePayloadOffsets[indexed.index()]));
      store->setAlignment(llvm::Align(1));
    }

    std::vector<llvm::Value *> objectBases;
    objectBases.reserve(capture.objects.size());
    for (const auto indexed : llvm::enumerate(capture.objects)) {
      const sim::SimulationMemoryCaptureObject &object = indexed.value();
      const ApplicationSpatialInvocationPlan::MemoryObjectSource &source =
          site.memoryObjectSources[indexed.index()];
      if (object.byteCount == 0 || source.byteOffset >= object.byteCount ||
          source.byteOffset > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::int64_t>::max()))
        return invalid("invocation memory capture is not finite");
      if (source.dispatchArgumentOrdinal >= helper->arg_size())
        return invalid("invocation memory object exceeds ABI");
      llvm::Value *rootPointer = helper->getArg(source.dispatchArgumentOrdinal);
      if (!rootPointer->getType()->isPointerTy())
        return invalid("invocation memory source is not a pointer");
      llvm::Value *base = rootPointer;
      if (source.byteOffset != 0)
        base = builder.CreateGEP(
            llvm::Type::getInt8Ty(module.getContext()), base,
            llvm::ConstantInt::getSigned(
                llvm::Type::getInt64Ty(module.getContext()),
                -static_cast<std::int64_t>(source.byteOffset)),
            "invocation.base");
      objectBases.push_back(base);
      llvm::Value *address = builder.CreatePtrToInt(
          base, llvm::Type::getInt64Ty(module.getContext()));
      llvm::StoreInst *addressStore = builder.CreateStore(
          address,
          bytePointer(builder, wire, wireType,
                      wireLayout.memoryAddressOffsets[indexed.index()]));
      addressStore->setAlignment(llvm::Align(1));
      emitFixedMemoryCopy(
          builder,
          bytePointer(builder, wire, wireType,
                      wireLayout.memoryPayloadOffsets[indexed.index()]),
          base, object.byteCount);
    }

    std::vector<llvm::Value *> rootByteOffsets;
    rootByteOffsets.reserve(site.memoryRootSources.size());
    for (const auto rootIndexed : llvm::enumerate(site.memoryRootSources)) {
      const ApplicationSpatialInvocationPlan::MemoryRootSource &source =
          rootIndexed.value();
      if (source.dispatchArgumentOrdinal >= helper->arg_size() ||
          source.objectIndex >= objectBases.size())
        return invalid("invocation memory root source exceeds ABI");
      llvm::Value *rootPointer = helper->getArg(source.dispatchArgumentOrdinal);
      if (!rootPointer->getType()->isPointerTy())
        return invalid("invocation memory root source is not a pointer");
      llvm::Value *rootAddress = builder.CreatePtrToInt(
          rootPointer, llvm::Type::getInt64Ty(module.getContext()));
      llvm::Value *baseAddress =
          builder.CreatePtrToInt(objectBases[source.objectIndex],
                                 llvm::Type::getInt64Ty(module.getContext()));
      llvm::Value *byteOffset = builder.CreateSub(rootAddress, baseAddress);
      llvm::StoreInst *offsetStore = builder.CreateStore(
          byteOffset,
          bytePointer(
              builder, wire, wireType,
              wireLayout.memoryRootByteOffsetOffsets[rootIndexed.index()]));
      offsetStore->setAlignment(llvm::Align(1));
      rootByteOffsets.push_back(byteOffset);
    }

    for (const auto valueIndexed : llvm::enumerate(capture.valueInputs)) {
      const sim::SimulationValueInputCapture &input = valueIndexed.value();
      const std::optional<std::size_t> offsetPosition =
          wireLayout.valuePointerTargetOffsetOffsets[valueIndexed.index()];
      if (!input.pointerTarget) {
        if (offsetPosition)
          return invalid("non-pointer invocation value has an offset slot");
        continue;
      }
      const std::uint64_t rootOrdinal =
          input.pointerTarget->memoryRootBindingOrdinal;
      if (!offsetPosition || rootOrdinal >= rootByteOffsets.size())
        return invalid("invocation pointer target has no root offset");
      llvm::StoreInst *offsetStore = builder.CreateStore(
          rootByteOffsets[rootOrdinal],
          bytePointer(builder, wire, wireType, *offsetPosition));
      offsetStore->setAlignment(llvm::Align(1));
    }

    for (const auto indexed :
         llvm::enumerate(launch.resultRootOperandOrdinals)) {
      auto dispatchOrdinal = dispatchOperandOrdinal(launch, indexed.value());
      if (!dispatchOrdinal || *dispatchOrdinal >= helper->arg_size())
        return dispatchOrdinal ? invalid("invocation result exceeds ABI")
                               : dispatchOrdinal.takeError();
      llvm::Value *result = helper->getArg(*dispatchOrdinal);
      if (!result->getType()->isPointerTy())
        return invalid("invocation result root operand is not a pointer");
      llvm::Value *resultAddress = builder.CreatePtrToInt(
          result, llvm::Type::getInt64Ty(module.getContext()));
      llvm::StoreInst *resultAddressStore = builder.CreateStore(
          resultAddress,
          bytePointer(builder, wire, wireType,
                      wireLayout.resultAddressOffsets[indexed.index()]));
      resultAddressStore->setAlignment(llvm::Align(1));
    }

    llvm::Value *dispatch = builder.CreateLoad(
        llvm::Type::getInt64Ty(module.getContext()), &dispatchBase);
    storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchTargetLow,
                static_cast<std::uint32_t>(point.dispatchTargetOrdinal));
    storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchTargetHigh,
                static_cast<std::uint32_t>(point.dispatchTargetOrdinal >> 32));
    storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchControl,
                runtime::gem5ThreadDispatchReset);
    llvm::Value *wireAddress = builder.CreatePtrToInt(
        wire, llvm::Type::getInt64Ty(module.getContext()));
    storeMmio64Descriptor(
        builder, dispatch, runtime::gem5ThreadDispatchInvocationLow,
        runtime::gem5ThreadDispatchInvocationHigh, wireAddress);
    storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchInvocationSize,
                static_cast<std::uint32_t>(wireLayout.templateBytes.size()));
    emitFence(builder);
    storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchControl,
                runtime::gem5ThreadDispatchStart);

    llvm::Value *occurrence = loadMmio64Descriptor(
        builder, dispatch, runtime::gem5ThreadDispatchOccurrenceLow,
        runtime::gem5ThreadDispatchOccurrenceHigh);
    llvm::Value *status =
        loadMmio32(builder, dispatch, runtime::gem5ThreadDispatchStatus);
    llvm::Value *hasFailed = builder.CreateICmpNE(
        builder.CreateAnd(status, runtime::gem5ThreadDispatchFailed),
        llvm::ConstantInt::get(status->getType(), 0));
    llvm::Value *accepted = builder.CreateAnd(
        builder.CreateNot(hasFailed),
        builder.CreateICmpNE(occurrence, llvm::ConstantInt::get(i64, 0)));
    llvm::Value *handleSlot = builder.CreateInBoundsGEP(
        handlesType, handles,
        {llvm::ConstantInt::get(i64, 0),
         llvm::ConstantInt::get(i64, pointIndexed.index())});
    builder.CreateStore(occurrence, handleSlot);
    llvm::BasicBlock *next = llvm::BasicBlock::Create(
        module.getContext(),
        "submitted." + std::to_string(pointIndexed.index()), helper);
    builder.CreateCondBr(accepted, next, failed);
    builder.SetInsertPoint(next);
  }
  llvm::Value *startDispatch = builder.CreateLoad(
      llvm::Type::getInt64Ty(module.getContext()), &dispatchBase);
  emitRootLifecycleEvent(builder, startDispatch, launch.root,
                         llvm::ConstantInt::get(i64, 0),
                         runtime::Gem5RootLifecycleAction::Start);
  llvm::Value *startedRootOccurrence = loadMmio64Descriptor(
      builder, startDispatch, runtime::gem5ThreadDispatchRootOccurrenceLow,
      runtime::gem5ThreadDispatchRootOccurrenceHigh);
  builder.CreateStore(startedRootOccurrence, activeRootOccurrence);
  llvm::BasicBlock *started =
      llvm::BasicBlock::Create(module.getContext(), "started", helper);
  builder.CreateCondBr(builder.CreateICmpNE(startedRootOccurrence,
                                            llvm::ConstantInt::get(i64, 0)),
                       started, failed);
  builder.SetInsertPoint(started);
  builder.CreateRet(generation);
  builder.SetInsertPoint(failed);
  builder.CreateBr(failed);

  llvm::Function *wait = llvm::Function::Create(
      waitType, llvm::GlobalValue::InternalLinkage, waitName, module);
  wait->setDoesNotThrow();
  wait->getArg(0)->setName("generation");
  llvm::BasicBlock *waitEntry =
      llvm::BasicBlock::Create(module.getContext(), "entry", wait);
  llvm::BasicBlock *waitFailed =
      llvm::BasicBlock::Create(module.getContext(), "failed", wait);
  llvm::BasicBlock *waitNext =
      llvm::BasicBlock::Create(module.getContext(), "point.0", wait);
  llvm::IRBuilder<> waitBuilder(waitEntry);
  llvm::Value *requestedGeneration = wait->getArg(0);
  llvm::Value *active = waitBuilder.CreateLoad(i64, activeGeneration);
  llvm::Value *validHandle = waitBuilder.CreateAnd(
      waitBuilder.CreateICmpNE(requestedGeneration,
                               llvm::ConstantInt::get(i64, 0)),
      waitBuilder.CreateICmpEQ(requestedGeneration, active));
  waitBuilder.CreateCondBr(validHandle, waitNext, waitFailed);

  for (const auto pointIndexed : llvm::enumerate(launch.points)) {
    const ApplicationSpatialInvocationPlan::Launch::Point &point =
        pointIndexed.value();
    waitBuilder.SetInsertPoint(waitNext);
    llvm::Value *dispatch = waitBuilder.CreateLoad(i64, &dispatchBase);
    storeMmio32(waitBuilder, dispatch, runtime::gem5ThreadDispatchTargetLow,
                static_cast<std::uint32_t>(point.dispatchTargetOrdinal));
    storeMmio32(waitBuilder, dispatch, runtime::gem5ThreadDispatchTargetHigh,
                static_cast<std::uint32_t>(point.dispatchTargetOrdinal >> 32));
    llvm::Value *actualOccurrence = loadMmio64Descriptor(
        waitBuilder, dispatch, runtime::gem5ThreadDispatchOccurrenceLow,
        runtime::gem5ThreadDispatchOccurrenceHigh);
    llvm::Value *handleSlot = waitBuilder.CreateInBoundsGEP(
        handlesType, handles,
        {llvm::ConstantInt::get(i64, 0),
         llvm::ConstantInt::get(i64, pointIndexed.index())});
    llvm::Value *expectedOccurrence = waitBuilder.CreateLoad(i64, handleSlot);
    llvm::BasicBlock *poll = llvm::BasicBlock::Create(
        module.getContext(), "poll." + std::to_string(pointIndexed.index()),
        wait);
    llvm::Value *matchingOccurrence = waitBuilder.CreateAnd(
        waitBuilder.CreateICmpNE(expectedOccurrence,
                                 llvm::ConstantInt::get(i64, 0)),
        waitBuilder.CreateICmpEQ(actualOccurrence, expectedOccurrence));
    waitBuilder.CreateCondBr(matchingOccurrence, poll, waitFailed);
    waitBuilder.SetInsertPoint(poll);
    llvm::Value *status =
        loadMmio32(waitBuilder, dispatch, runtime::gem5ThreadDispatchStatus);
    llvm::Value *hasFailed = waitBuilder.CreateICmpNE(
        waitBuilder.CreateAnd(status, runtime::gem5ThreadDispatchFailed),
        llvm::ConstantInt::get(status->getType(), 0));
    llvm::BasicBlock *checkDone = llvm::BasicBlock::Create(
        module.getContext(),
        "check_done." + std::to_string(pointIndexed.index()), wait);
    waitBuilder.CreateCondBr(hasFailed, waitFailed, checkDone);
    waitBuilder.SetInsertPoint(checkDone);
    llvm::Value *isDone = waitBuilder.CreateICmpNE(
        waitBuilder.CreateAnd(status, runtime::gem5ThreadDispatchDone),
        llvm::ConstantInt::get(status->getType(), 0));
    waitNext = llvm::BasicBlock::Create(
        module.getContext(),
        "point." + std::to_string(pointIndexed.index() + 1), wait);
    waitBuilder.CreateCondBr(isDone, waitNext, poll);
  }

  waitBuilder.SetInsertPoint(waitNext);
  emitFence(waitBuilder);
  for (const ApplicationSpatialInvocationPlan::Launch::Point &point :
       launch.points) {
    llvm::Value *dispatch = waitBuilder.CreateLoad(i64, &dispatchBase);
    storeMmio32(waitBuilder, dispatch, runtime::gem5ThreadDispatchTargetLow,
                static_cast<std::uint32_t>(point.dispatchTargetOrdinal));
    storeMmio32(waitBuilder, dispatch, runtime::gem5ThreadDispatchTargetHigh,
                static_cast<std::uint32_t>(point.dispatchTargetOrdinal >> 32));
    storeMmio32(waitBuilder, dispatch, runtime::gem5ThreadDispatchControl,
                runtime::gem5ThreadDispatchReset);
  }
  llvm::Value *completedRootOccurrence =
      waitBuilder.CreateLoad(i64, activeRootOccurrence);
  llvm::Value *completionDispatch = waitBuilder.CreateLoad(i64, &dispatchBase);
  emitRootLifecycleEvent(waitBuilder, completionDispatch, launch.root,
                         completedRootOccurrence,
                         runtime::Gem5RootLifecycleAction::Completion);
  waitBuilder.CreateStore(llvm::ConstantInt::get(i64, 0), activeGeneration);
  waitBuilder.CreateStore(llvm::ConstantInt::get(i64, 0), activeRootOccurrence);
  waitBuilder.CreateRetVoid();
  waitBuilder.SetInsertPoint(waitFailed);
  waitBuilder.CreateBr(waitFailed);
  return MaterializedRootDispatch{helper, wait};
}

llvm::Error lowerSelectedHostControlModule(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  mlir::PassManager pipeline(module->getContext());
  pipeline.enableVerifier(true);
  pipeline.addPass(mlir::createSCFToControlFlowPass());
  pipeline.addPass(mlir::createConvertToLLVMPass());
  if (mlir::failed(pipeline.run(module)))
    return invalid("selected host control cannot lower to LLVM");
  return llvm::Error::success();
}

struct TranslatedSelectedCallable final {
  std::unique_ptr<llvm::Module> module;
  llvm::Function *callable = nullptr;
  std::vector<llvm::Function *> rootDispatches;
  std::vector<llvm::Function *> rootWaits;
};

llvm::Expected<TranslatedSelectedCallable> translateSelectedCallable(
    const dataflow::CanonicalDataflowArtifact &dataflow,
    const ApplicationSpatialInvocationPlan &plan,
    const ApplicationSpatialInvocationPlan::Callable &callablePlan,
    llvm::LLVMContext &context) {
  if (callablePlan.launchOrdinals.empty())
    return invalid("selected host control has no Spatial launches");
  auto view = dataflow.view();
  if (!view)
    return view.takeError();

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> selected(
      llvm::cast<mlir::ModuleOp>(dataflow.module()->clone(mapping)));
  mlir::LLVM::LLVMFuncOp selectedCallable;
  std::vector<std::string> rootDispatchSymbols;
  std::vector<std::string> rootWaitSymbols;
  rootDispatchSymbols.reserve(callablePlan.launchOrdinals.size());
  rootWaitSymbols.reserve(callablePlan.launchOrdinals.size());
  llvm::DenseMap<mlir::Value, std::pair<mlir::Value, std::string>>
      completionHandles;
  llvm::SmallVector<dataflow::ThreadLaunchOp> selectedLaunches;
  llvm::SmallVector<dataflow::ThreadWaitOp> selectedWaits;
  llvm::SmallPtrSet<mlir::Operation *, 8> uniqueWaits;
  mlir::OpBuilder declarations(selected->getContext());
  declarations.setInsertionPointToStart(selected->getBody());
  for (std::uint64_t launchOrdinal : callablePlan.launchOrdinals) {
    if (launchOrdinal >= plan.launches.size())
      return invalid("selected callable launch ordinal is out of range");
    const ApplicationSpatialInvocationPlan::Launch &launch =
        plan.launches[launchOrdinal];
    auto resolvedRoot = view->resolve(launch.root);
    if (!resolvedRoot)
      return resolvedRoot.takeError();
    auto rootLaunch = llvm::dyn_cast_or_null<dataflow::ThreadLaunchOp>(
        mapping.lookupOrNull(resolvedRoot->op));
    if (!rootLaunch)
      return invalid("selected host clone lost a root launch");
    auto owner = rootLaunch->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!owner || owner.getSymName() != callablePlan.sourceCallableSymbol ||
        (selectedCallable && owner != selectedCallable))
      return invalid("selected roots do not share their callable owner");
    selectedCallable = owner;
    if (!rootLaunch.getAsyncToken().hasOneUse())
      return invalid("selected host root has no unique wait");
    auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
        rootLaunch.getAsyncToken().use_begin()->getOwner());
    if (!wait || wait->getBlock() != rootLaunch->getBlock() ||
        !rootLaunch->isBeforeInBlock(wait))
      return invalid("selected host root wait is not local and ordered");

    std::string symbol =
        "__loom_selected_root_dispatch_" + std::to_string(launchOrdinal);
    std::string waitSymbol =
        "__loom_selected_root_wait_" + std::to_string(launchOrdinal);
    if (mlir::SymbolTable::lookupSymbolIn(*selected, symbol) ||
        mlir::SymbolTable::lookupSymbolIn(*selected, waitSymbol))
      return invalid("selected Dataflow defines a reserved root dispatch");
    llvm::SmallVector<mlir::Type> rootTypes;
    llvm::SmallVector<mlir::Value> rootOperands;
    rootTypes.reserve(launch.dispatchRootOperandOrdinals.size());
    rootOperands.reserve(launch.dispatchRootOperandOrdinals.size());
    for (std::uint64_t ordinal : launch.dispatchRootOperandOrdinals) {
      if (ordinal >= rootLaunch.getBodyOperands().size())
        return invalid("selected dispatch operand exceeds its root boundary");
      mlir::Value operand = rootLaunch.getBodyOperands()[ordinal];
      if (llvm::isa<dataflow::ChannelType>(operand.getType()))
        return invalid("selected dispatch ABI retains a channel handle");
      rootTypes.push_back(operand.getType());
      rootOperands.push_back(operand);
    }
    if (launch.sites.empty())
      return invalid("selected root has no invocation site");
    const ApplicationSpatialInvocationPlan::Site &prototype =
        launch.sites.front();
    if (prototype.capture.input.objects.size() !=
        prototype.memoryObjectSources.size())
      return invalid("selected root memory source table is inconsistent");
    for (const auto objectIndexed :
         llvm::enumerate(prototype.capture.input.objects)) {
      const ApplicationSpatialInvocationPlan::MemoryObjectSource &source =
          prototype.memoryObjectSources[objectIndexed.index()];
      if (source.dispatchArgumentOrdinal != rootOperands.size() ||
          !source.base || !mapping.contains(source.base))
        return invalid("selected root local memory base is not exact");
      mlir::Value base = mapping.lookupOrDefault(source.base);
      auto owner =
          base.getDefiningOp()
              ? base.getDefiningOp()->getParentOfType<mlir::LLVM::LLVMFuncOp>()
              : mlir::LLVM::LLVMFuncOp{};
      if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(base))
        owner = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
            argument.getOwner()->getParentOp());
      if (owner != selectedCallable ||
          !llvm::isa<mlir::LLVM::LLVMPointerType>(base.getType()))
        return invalid("selected root local memory base has a foreign owner");
      rootTypes.push_back(base.getType());
      rootOperands.push_back(base);
    }
    for (const ApplicationSpatialInvocationPlan::Site &site : launch.sites) {
      if (site.memoryObjectSources.size() !=
          prototype.memoryObjectSources.size())
        return invalid("selected root sites disagree on memory source count");
      for (const auto sourceIndexed : llvm::enumerate(site.memoryObjectSources))
        if (sourceIndexed.value().dispatchArgumentOrdinal !=
                prototype.memoryObjectSources[sourceIndexed.index()]
                    .dispatchArgumentOrdinal ||
            sourceIndexed.value().base !=
                prototype.memoryObjectSources[sourceIndexed.index()].base)
          return invalid("selected root sites disagree on memory source ABI");
    }
    mlir::Type handleType = mlir::IntegerType::get(selected->getContext(), 64);
    auto dispatchType = mlir::LLVM::LLVMFunctionType::get(handleType, rootTypes,
                                                          /*isVarArg=*/false);
    auto waitType = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(selected->getContext()), {handleType},
        /*isVarArg=*/false);
    mlir::LLVM::LLVMFuncOp::create(declarations, rootLaunch.getLoc(), symbol,
                                   dispatchType);
    mlir::LLVM::LLVMFuncOp::create(declarations, rootLaunch.getLoc(),
                                   waitSymbol, waitType);
    mlir::OpBuilder launchBuilder(rootLaunch);
    mlir::LLVM::CallOp dispatch = mlir::LLVM::CallOp::create(
        launchBuilder, rootLaunch.getLoc(), mlir::TypeRange{handleType}, symbol,
        rootOperands);
    completionHandles.try_emplace(rootLaunch.getAsyncToken(),
                                  dispatch.getResult(), waitSymbol);
    selectedLaunches.push_back(rootLaunch);
    if (uniqueWaits.insert(wait.getOperation()).second)
      selectedWaits.push_back(wait);
    rootDispatchSymbols.push_back(std::move(symbol));
    rootWaitSymbols.push_back(std::move(waitSymbol));
  }

  for (dataflow::ThreadWaitOp wait : selectedWaits) {
    mlir::OpBuilder waitBuilder(wait);
    for (mlir::Value token : wait.getAsyncDependencies()) {
      auto found = completionHandles.find(token);
      if (found == completionHandles.end())
        return invalid("selected host wait retains an unselected root");
      mlir::LLVM::CallOp::create(waitBuilder, wait.getLoc(), mlir::TypeRange{},
                                 found->second.second,
                                 mlir::ValueRange{found->second.first});
    }
    wait.erase();
  }
  for (dataflow::ThreadLaunchOp launch : selectedLaunches)
    launch.erase();

  llvm::SmallVector<mlir::LLVM::LLVMFuncOp> functions;
  selected->walk(
      [&](mlir::LLVM::LLVMFuncOp function) { functions.push_back(function); });
  for (mlir::LLVM::LLVMFuncOp function : functions) {
    if (function == selectedCallable || function.isExternal())
      continue;
    function.getBody().dropAllReferences();
    function.getBody().getBlocks().clear();
    function.setLinkage(mlir::LLVM::Linkage::External);
  }

  llvm::SmallVector<dataflow::GraphOp> graphs;
  llvm::SmallVector<dataflow::ThreadOp> threads;
  llvm::SmallVector<dataflow::ChannelCreateOp> channels;
  selected->walk([&](dataflow::GraphOp graph) { graphs.push_back(graph); });
  selected->walk([&](dataflow::ThreadOp thread) { threads.push_back(thread); });
  for (dataflow::GraphOp graph : graphs)
    graph.erase();
  for (dataflow::ThreadOp thread : threads)
    thread.erase();
  selected->walk(
      [&](dataflow::ChannelCreateOp channel) { channels.push_back(channel); });
  for (dataflow::ChannelCreateOp channel : channels)
    if (channel->use_empty())
      channel.erase();
  bool residualDataflow = false;
  selected->walk([&](mlir::Operation *operation) {
    residualDataflow |=
        operation->getDialect() &&
        llvm::isa<dataflow::DataflowDialect>(operation->getDialect());
  });
  if (residualDataflow)
    return invalid("selected host control retains a Dataflow operation");
  if (mlir::failed(mlir::verify(*selected)))
    return invalid("selected host control does not verify");
  if (llvm::Error error = lowerSelectedHostControlModule(*selected))
    return std::move(error);

  std::unique_ptr<llvm::Module> translated = mlir::translateModuleToLLVMIR(
      selected->getOperation(), context, "loom-selected-host-control");
  if (!translated)
    return invalid("selected host control cannot translate to LLVM IR");
  llvm::Function *translatedCallable =
      translated->getFunction(callablePlan.sourceCallableSymbol);
  if (!translatedCallable || translatedCallable->isDeclaration())
    return invalid("translated host control lost its callable");
  std::vector<llvm::Function *> translatedDispatches;
  std::vector<llvm::Function *> translatedWaits;
  translatedDispatches.reserve(rootDispatchSymbols.size());
  translatedWaits.reserve(rootWaitSymbols.size());
  for (auto [dispatchSymbol, waitSymbol] :
       llvm::zip_equal(rootDispatchSymbols, rootWaitSymbols)) {
    llvm::Function *dispatch = translated->getFunction(dispatchSymbol);
    llvm::Function *wait = translated->getFunction(waitSymbol);
    if (!dispatch || !dispatch->isDeclaration() || !wait ||
        !wait->isDeclaration())
      return invalid("translated host control lost a root dispatch helper");
    translatedDispatches.push_back(dispatch);
    translatedWaits.push_back(wait);
  }
  return TranslatedSelectedCallable{std::move(translated), translatedCallable,
                                    std::move(translatedDispatches),
                                    std::move(translatedWaits)};
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

void clearDispatchAttributes(llvm::Function &function) {
  function.removeFnAttr(llvm::Attribute::Memory);
  function.removeFnAttr(llvm::Attribute::MustProgress);
  function.removeFnAttr(llvm::Attribute::NoSync);
  function.removeFnAttr(llvm::Attribute::WillReturn);
}

void clearDispatchAttributes(llvm::CallBase &call) {
  call.removeFnAttr(llvm::Attribute::Memory);
  call.removeFnAttr(llvm::Attribute::MustProgress);
  call.removeFnAttr(llvm::Attribute::NoSync);
  call.removeFnAttr(llvm::Attribute::WillReturn);
}

llvm::Expected<llvm::Function *>
cloneSelectedCallable(llvm::Module &host, llvm::Function &source,
                      llvm::ArrayRef<llvm::Function *> sourceRootDispatches,
                      llvm::ArrayRef<llvm::Function *> siteRootDispatches,
                      llvm::ArrayRef<llvm::Function *> sourceRootWaits,
                      llvm::ArrayRef<llvm::Function *> siteRootWaits,
                      std::uint64_t helperOrdinal) {
  if (sourceRootDispatches.size() != siteRootDispatches.size() ||
      sourceRootWaits.size() != siteRootWaits.size() ||
      sourceRootDispatches.size() != sourceRootWaits.size() ||
      sourceRootDispatches.empty())
    return invalid("selected callable dispatch table is inconsistent");
  const std::string name =
      "__loom_selected_callable_" + std::to_string(helperOrdinal);
  if (host.getNamedValue(name))
    return invalid("final-linked module defines a selected callable helper");
  auto *target = llvm::Function::Create(source.getFunctionType(),
                                        llvm::GlobalValue::InternalLinkage,
                                        source.getAddressSpace(), name, &host);
  target->copyAttributesFrom(&source);
  target->setCallingConv(source.getCallingConv());
  clearDispatchAttributes(*target);

  llvm::ValueToValueMapTy values;
  values[&source] = target;
  for (auto [sourceDispatch, siteDispatch] :
       llvm::zip_equal(sourceRootDispatches, siteRootDispatches))
    values[sourceDispatch] = siteDispatch;
  for (auto [sourceWait, siteWait] :
       llvm::zip_equal(sourceRootWaits, siteRootWaits))
    values[sourceWait] = siteWait;
  for (auto [sourceArgument, targetArgument] :
       llvm::zip_equal(source.args(), target->args()))
    values[&sourceArgument] = &targetArgument;

  llvm::SmallPtrSet<llvm::GlobalValue *, 16> dependencies;
  for (llvm::BasicBlock &block : source)
    for (llvm::Instruction &instruction : block)
      for (llvm::Value *operand : instruction.operands())
        collectReferencedGlobals(operand, dependencies);
  if (source.hasPersonalityFn())
    collectReferencedGlobals(source.getPersonalityFn(), dependencies);
  for (llvm::GlobalValue *dependency : dependencies) {
    if (values.count(dependency))
      continue;
    llvm::GlobalValue *existing = host.getNamedValue(dependency->getName());
    if (existing) {
      if (existing->getValueType() != dependency->getValueType())
        return invalid("selected callable dependency changed native type: " +
                       dependency->getName());
      values[dependency] = existing;
      continue;
    }
    auto *function = llvm::dyn_cast<llvm::Function>(dependency);
    if (!function || !function->isDeclaration())
      return invalid("selected callable introduced an unknown native global");
    auto *declaration = llvm::Function::Create(
        function->getFunctionType(), function->getLinkage(),
        function->getAddressSpace(), function->getName(), &host);
    declaration->copyAttributesFrom(function);
    declaration->setCallingConv(function->getCallingConv());
    values[function] = declaration;
  }

  llvm::SmallVector<llvm::ReturnInst *> returns;
  llvm::CloneFunctionInto(target, &source, values,
                          llvm::CloneFunctionChangeType::DifferentModule,
                          returns);
  if (llvm::verifyFunction(*target, &llvm::errs()))
    return invalid("selected callable helper does not verify");
  return target;
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
  if (application->doesNotThrow())
    entry->setDoesNotThrow();
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

llvm::Error pruneHostExecutableClosure(llvm::Module &module) {
  llvm::Function *entry = module.getFunction(applicationHostEntrySymbol);
  if (!entry || entry->isDeclaration())
    return invalid("host executable closure has no defined ABI entry");
  llvm::internalizeModule(module, [&](const llvm::GlobalValue &global) {
    return &global == entry ||
           (global.getName() == "expf" && !global.isDeclaration());
  });
  llvm::ModuleAnalysisManager analyses;
  (void)llvm::GlobalDCEPass().run(module, analyses);
  if (module.getFunction(applicationHostEntrySymbol) != entry)
    return invalid("host executable closure removed its ABI entry");
  return llvm::Error::success();
}

llvm::Error linkFreestandingRuntime(llvm::Module &module) {
  llvm::Function *requiredExp = module.getFunction("expf");
  bool requiresExp =
      requiredExp && requiredExp->isDeclaration() && !requiredExp->use_empty();
  for (const llvm::Function &function : module)
    for (const llvm::BasicBlock &block : function)
      for (const llvm::Instruction &instruction : block) {
        const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        const llvm::Function *callee =
            call ? call->getCalledFunction() : nullptr;
        if (!callee || callee->getIntrinsicID() != llvm::Intrinsic::exp)
          continue;
        if (!call->getType()->isFloatTy())
          return invalid("freestanding runtime does not provide double exp");
        requiresExp = true;
      }
  if (!requiresExp)
    return llvm::Error::success();
  const llvm::StringRef bytes(
      reinterpret_cast<const char *>(freestandingMathBitcode),
      sizeof(freestandingMathBitcode));
  auto runtime = llvm::parseBitcodeFile(
      llvm::MemoryBufferRef(bytes, "loom-freestanding-math.bc"),
      module.getContext());
  if (!runtime)
    return invalid("cannot import the freestanding math runtime: " +
                   llvm::toString(runtime.takeError()));
  if ((*runtime)->getTargetTriple().normalize() !=
          module.getTargetTriple().normalize() ||
      (*runtime)->getDataLayout() != module.getDataLayout())
    return invalid("freestanding math runtime has a foreign compiler target");
  llvm::Linker linker(module);
  if (linker.linkInModule(std::move(*runtime)))
    return invalid("cannot link the freestanding math runtime");
  llvm::Function *linkedExp = module.getFunction("expf");
  if (!linkedExp || linkedExp->isDeclaration())
    return invalid("freestanding math runtime did not define required expf");
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
  entry->setDoesNotThrow();
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
  storeMmio64Descriptor(
      builder, bridge, runtime::gem5SpatialBridgeStaticLaunchLow,
      runtime::gem5SpatialBridgeStaticLaunchHigh, staticAddress);
  storeMmio32(builder, bridge, runtime::gem5SpatialBridgeStaticLaunchSize,
              builder.CreateTrunc(staticSize, llvm::Type::getInt32Ty(context)));
  storeMmio64Descriptor(
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
  storeMmio32(builder, dispatch, runtime::gem5ThreadDispatchWorkerCompletion,
              1);
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
  if (roots->empty())
    return invalid("initial dispatch requires at least one reachable root");

  struct LaunchBoundary final {
    ApplicationSpatialInvocationPlan::Launch launch;
    dataflow::ThreadLaunchOp rootLaunch;
    dataflow::GraphLaunchOp graphLaunch;
    std::string callableSymbol;
  };
  std::vector<LaunchBoundary> boundaries;
  boundaries.reserve(roots->size());
  for (dataflow::RootThreadLaunchRef root : *roots) {
    std::vector<dataflow::RootedGraphLaunchRef> graphs;
    dataflow.forEachRootedGraphLaunch(
        [&](dataflow::RootedGraphLaunchRef graph) {
          if (graph.rootThreadLaunch == root)
            graphs.push_back(graph);
        });
    if (graphs.size() != 1)
      return invalid("each dynamic root must own exactly one graph launch");
    auto rootView = dataflow.resolve(root);
    if (!rootView)
      return rootView.takeError();
    auto graphView = dataflow.resolve(graphs.front().staticGraphLaunch);
    if (!graphView)
      return graphView.takeError();
    auto rootLaunch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(rootView->op);
    auto graphLaunch = llvm::dyn_cast<dataflow::GraphLaunchOp>(graphView->op);
    if (!rootLaunch || !graphLaunch || !graphLaunch.getMemoryResults().empty())
      return invalid(
          "dynamic invocation requires an imported-memory graph boundary");
    auto shapes =
        sim::projectSpatialSimulationBoundaryShapes(dataflow, graphs.front());
    if (!shapes)
      return shapes.takeError();
    auto coordinates = dataflow.enumerateStaticDenseCoordinates(
        graphs.front(), runtime::gem5MaximumDynamicSpatialInvocations,
        entrySymbol);
    if (!coordinates)
      return coordinates.takeError();
    if (!*coordinates || (*coordinates)->empty())
      return invalid(
          "dynamic invocation requires a finite nonempty dense domain");
    auto valueBitCounts = transportBitCounts(shapes->valueInputs);
    auto resultBitCounts = transportBitCounts(shapes->valueResults);
    if (!valueBitCounts || !resultBitCounts)
      return llvm::joinErrors(valueBitCounts ? llvm::Error::success()
                                             : valueBitCounts.takeError(),
                              resultBitCounts ? llvm::Error::success()
                                              : resultBitCounts.takeError());
    std::vector<std::uint64_t> resultRootOperandOrdinals;
    std::string callableSymbol;
    if (llvm::Error error = verifySelectedCallableBoundary(
            *rootView, graphLaunch, *valueBitCounts, *resultBitCounts,
            resultRootOperandOrdinals, callableSymbol))
      return std::move(error);
    std::vector<std::uint64_t> dispatchRootOperandOrdinals;
    for (const auto operand : llvm::enumerate(rootLaunch.getBodyOperands()))
      if (!llvm::isa<dataflow::ChannelType>(operand.value().getType()))
        dispatchRootOperandOrdinals.push_back(operand.index());
    std::vector<ApplicationSpatialInvocationPlan::Launch::Point> points;
    points.reserve((*coordinates)->size());
    for (std::vector<std::uint64_t> &point : **coordinates)
      points.push_back({0, std::move(point)});
    boundaries.push_back({{root,
                           graphs.front(),
                           std::move(points),
                           std::move(dispatchRootOperandOrdinals),
                           std::move(*valueBitCounts),
                           std::move(*resultBitCounts),
                           std::move(resultRootOperandOrdinals),
                           {}},
                          rootLaunch,
                          graphLaunch,
                          std::move(callableSymbol)});
  }
  llvm::sort(
      boundaries, [](const LaunchBoundary &lhs, const LaunchBoundary &rhs) {
        return std::tuple(lhs.launch.root.entity.value(),
                          lhs.launch.graph.staticGraphLaunch.entity.value()) <
               std::tuple(rhs.launch.root.entity.value(),
                          rhs.launch.graph.staticGraphLaunch.entity.value());
      });
  std::uint64_t nextDispatchTarget = 0;
  for (LaunchBoundary &boundary : boundaries)
    for (ApplicationSpatialInvocationPlan::Launch::Point &point :
         boundary.launch.points)
      point.dispatchTargetOrdinal = nextDispatchTarget++;

  std::map<std::string, std::vector<std::uint64_t>> launchesByCallable;
  for (const auto indexed : llvm::enumerate(boundaries))
    launchesByCallable[indexed.value().callableSymbol].push_back(
        indexed.index());
  std::vector<ApplicationSpatialInvocationPlan::Callable> callables;
  callables.reserve(launchesByCallable.size());
  for (auto &[symbol, launchOrdinals] : launchesByCallable)
    callables.push_back({std::move(symbol), std::move(launchOrdinals)});

  for (const ApplicationSpatialInvocationPlan::Callable &callable : callables) {
    auto paths = dataflow.projectRootThreadInvocationPathsFromAbiEntry(
        entrySymbol, boundaries[callable.launchOrdinals.front()].launch.root);
    if (!paths)
      return paths.takeError();
    std::vector<std::pair<std::string, std::uint64_t>> leafLocators;
    for (const auto callableLaunchIndexed :
         llvm::enumerate(callable.launchOrdinals)) {
      LaunchBoundary &boundary = boundaries[callableLaunchIndexed.value()];
      ApplicationSpatialInvocationPlan::Launch &launch = boundary.launch;
      launch.sites.reserve(paths->size());
      for (const auto pathIndexed : llvm::enumerate(*paths)) {
        llvm::SmallVector<mlir::LLVM::CallOp, 4> path;
        path.reserve(pathIndexed.value().calls.size());
        for (mlir::Operation *operation : pathIndexed.value().calls) {
          auto call = llvm::dyn_cast_or_null<mlir::LLVM::CallOp>(operation);
          if (!call)
            return invalid("canonical invocation path contains a non-call");
          path.push_back(call);
        }
        auto capture = sim::deriveOperationSimulationInputCapturePlan(
            dataflow, launch.graph, boundary.rootLaunch.getBodyOperands(),
            boundary.graphLaunch.getValueResults(), path);
        if (!capture)
          return capture.takeError();
        if (capture->invocationPath.empty())
          return invalid("dynamic invocation capture has no call locator");
        const sim::DirectCallCaptureSite &leaf = capture->invocationPath.back();
        const std::pair<std::string, std::uint64_t> leafLocator{
            leaf.hostCallerSymbol, leaf.hostCallOrdinal};
        if (callableLaunchIndexed.index() == 0) {
          if (llvm::is_contained(leafLocators, leafLocator))
            return invalid(
                "one dynamic invocation call is reachable through multiple "
                "paths");
          leafLocators.push_back(leafLocator);
        } else if (pathIndexed.index() >= leafLocators.size() ||
                   leafLocators[pathIndexed.index()] != leafLocator) {
          return invalid("dynamic roots disagree on their invocation sites");
        }
        if (capture->input.valueInputs.size() != launch.valueBitCounts.size() ||
            capture->input.valueResults.size() != launch.resultBitCounts.size())
          return invalid(
              "dynamic invocation capture differs from graph boundary");
        for (const sim::SimulationValueInputCapture &input :
             capture->input.valueInputs) {
          if (input.unusedByGraph &&
              (!input.fixedValue ||
               llvm::any_of(input.fixedValue->lanes, [](const auto &lane) {
                 return lane.state != sim::SemanticState::Defined ||
                        lane.pointerTarget.has_value();
               })))
            return invalid(
                "graph-unobserved capture does not carry a defined scalar "
                "wire value");
          if (!input.fixedValue)
            continue;
          if (llvm::any_of(input.fixedValue->lanes, [](const auto &lane) {
                return lane.state != sim::SemanticState::Defined ||
                       lane.pointerTarget.has_value();
              }))
            return invalid(
                "fixed invocation value is not representable on the wire: "
                "undef, poison, or pointer lane");
        }

        std::vector<mlir::Value> boundObjectBases(
            capture->input.objects.size());
        std::vector<bool> objectHasRoot(capture->input.objects.size(), false);
        std::vector<ApplicationSpatialInvocationPlan::MemoryRootSource>
            memoryRootSources;
        memoryRootSources.reserve(capture->input.memoryRootBindings.size());
        for (const sim::SimulationMemoryRootCapture &binding :
             capture->input.memoryRootBindings) {
          if (binding.objectIndex >= boundObjectBases.size())
            return invalid("dynamic invocation memory root exceeds objects");
          objectHasRoot[binding.objectIndex] = true;
          const sim::SimulationMemoryCaptureObject &object =
              capture->input.objects[binding.objectIndex];
          auto operand = llvm::find(boundary.rootLaunch.getBodyOperands(),
                                    binding.boundaryPointer);
          if (operand == boundary.rootLaunch.getBodyOperands().end())
            return invalid(
                "dynamic invocation memory root is not a root operand");
          const std::uint64_t rootOperandOrdinal =
              static_cast<std::uint64_t>(std::distance(
                  boundary.rootLaunch.getBodyOperands().begin(), operand));
          auto dispatchOrdinal =
              dispatchOperandOrdinal(launch, rootOperandOrdinal);
          if (!dispatchOrdinal)
            return dispatchOrdinal.takeError();
          memoryRootSources.push_back({*dispatchOrdinal, binding.objectIndex});
          if (!object.baseBindingCallOrdinal)
            continue;
          mlir::Value base = binding.boundaryPointer;
          while (base) {
            if (auto gep = base.getDefiningOp<mlir::LLVM::GEPOp>()) {
              base = gep.getBase();
              continue;
            }
            if (auto cast = base.getDefiningOp<mlir::LLVM::BitcastOp>()) {
              base = cast.getArg();
              continue;
            }
            if (auto cast = base.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>()) {
              base = cast.getArg();
              continue;
            }
            if (auto cast =
                    base.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
              if (cast.getInputs().size() != 1)
                return invalid(
                    "bound invocation memory base has a non-unary cast");
              base = cast.getInputs().front();
              continue;
            }
            break;
          }
          auto argument = llvm::dyn_cast<mlir::BlockArgument>(base);
          auto owner = argument
                           ? llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
                                 argument.getOwner()->getParentOp())
                           : mlir::LLVM::LLVMFuncOp{};
          if (!argument || !owner ||
              owner.getSymName() != callable.sourceCallableSymbol)
            return invalid(
                "bound invocation memory base does not reach a selected "
                "callable argument");
          if (boundObjectBases[binding.objectIndex] &&
              boundObjectBases[binding.objectIndex] != base)
            return invalid(
                "bound invocation memory object has conflicting callable "
                "bases");
          boundObjectBases[binding.objectIndex] = base;
        }
        std::vector<ApplicationSpatialInvocationPlan::MemoryObjectSource>
            memoryObjectSources;
        memoryObjectSources.reserve(capture->input.objects.size());
        std::uint64_t nextLocalBaseArgument =
            launch.dispatchRootOperandOrdinals.size();
        for (const auto objectIndexed :
             llvm::enumerate(capture->input.objects)) {
          if (!objectHasRoot[objectIndexed.index()])
            return invalid(
                "dynamic invocation memory object has no logical root");
          const sim::SimulationMemoryCaptureObject &object =
              objectIndexed.value();
          mlir::Value base = object.baseBindingCallOrdinal
                                 ? boundObjectBases[objectIndexed.index()]
                                 : object.base;
          if (!base)
            return invalid(
                "dynamic invocation memory object has no callable base");
          memoryObjectSources.push_back(
              {nextLocalBaseArgument++, object.operandByteOffset, base});
        }
        std::vector<runtime::SpatialInvocationValueLayout> valueLayouts;
        valueLayouts.reserve(launch.valueBitCounts.size());
        for (const auto indexed : llvm::enumerate(capture->input.valueInputs)) {
          const sim::SimulationValueInputCapture &input = indexed.value();
          const std::uint64_t expectedByteCount =
              (launch.valueBitCounts[indexed.index()] + 7) / 8;
          const bool byteCountMatches =
              input.fixedValue ? input.byteCount == 0
                               : input.byteCount == expectedByteCount;
          if (input.valueInputOrdinal != indexed.index() || !byteCountMatches)
            return invalid(
                llvm::Twine("dynamic invocation value capture is not exact: ") +
                "root=" + llvm::Twine(boundary.launch.root.entity.value()) +
                " graph=" +
                llvm::Twine(launch.graph.staticGraphLaunch.entity.value()) +
                " path=" + llvm::Twine(pathIndexed.index()) +
                " input=" + llvm::Twine(indexed.index()) +
                " captured_ordinal=" + llvm::Twine(input.valueInputOrdinal) +
                " captured_bytes=" + llvm::Twine(input.byteCount) +
                " expected_bytes=" +
                llvm::Twine(input.fixedValue ? 0 : expectedByteCount));
          if (!input.fixedValue) {
            const bool rootOperand = input.boundaryOperandOrdinal.has_value();
            const bool coordinate = input.denseCoordinateDimension.has_value();
            if (rootOperand == coordinate)
              return invalid(
                  "dynamic invocation value does not have one source");
            if (rootOperand && *input.boundaryOperandOrdinal >=
                                   boundary.rootLaunch.getBodyOperands().size())
              return invalid("dynamic invocation value has no root operand");
            if (coordinate && *input.denseCoordinateDimension >=
                                  launch.points.front().denseCoordinates.size())
              return invalid("dynamic invocation coordinate is out of range");
          }
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
              {launch.valueBitCounts[indexed.index()], pointerTarget});
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
          rootBindings.push_back(
              {binding.root.entity.value(),
               static_cast<std::uint32_t>(binding.objectIndex),
               binding.byteOffset});
        }
        std::vector<runtime::SpatialInvocationWireLayout> pointWireLayouts;
        pointWireLayouts.reserve(launch.points.size());
        for (const ApplicationSpatialInvocationPlan::Launch::Point &point :
             launch.points) {
          runtime::SpatialInvocationWireLayout wireLayout;
          std::string diagnostic;
          if (!runtime::projectSpatialInvocationWireLayout(
                  dataflow.identity().bytes(), launch.root.entity.value(),
                  launch.graph.staticGraphLaunch.entity.value(),
                  point.denseCoordinates, valueLayouts, objectLayouts,
                  rootBindings, launch.resultBitCounts, wireLayout, diagnostic))
            return invalid(diagnostic);
          if (wireLayout.templateBytes.size() >
              std::numeric_limits<std::uint32_t>::max())
            return invalid(
                "invocation wire exceeds the dispatch size register");
          pointWireLayouts.push_back(std::move(wireLayout));
        }
        launch.sites.push_back(
            {std::move(*capture), std::move(memoryObjectSources),
             std::move(memoryRootSources), std::move(pointWireLayouts)});
      }
    }
  }
  std::vector<ApplicationSpatialInvocationPlan::Launch> launches;
  launches.reserve(boundaries.size());
  for (LaunchBoundary &boundary : boundaries)
    launches.push_back(std::move(boundary.launch));
  return ApplicationSpatialInvocationPlan{std::move(launches),
                                          std::move(callables)};
}

llvm::Expected<std::unique_ptr<llvm::Module>> materializeHostDispatchModule(
    const llvm::Module &finalLinkedModule,
    const dataflow::CanonicalDataflowArtifact &dataflow,
    llvm::StringRef applicationEntry,
    const ApplicationSpatialInvocationPlan &plan) {
  auto module = llvm::CloneModule(finalLinkedModule);
  llvm::Type *i64 = llvm::Type::getInt64Ty(module->getContext());
  if (module->getGlobalVariable("__loom_dispatch_base", true))
    return invalid("final-linked module defines the reserved dispatch base");
  auto *dispatchBase = new llvm::GlobalVariable(
      *module, i64, false, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantInt::get(i64, 0), "__loom_dispatch_base");
  if (plan.launches.empty() || plan.callables.empty())
    return invalid("dynamic invocation plan has no callable closure");
  std::uint64_t targetCount = 0;
  for (const ApplicationSpatialInvocationPlan::Launch &launch : plan.launches) {
    if (launch.points.empty() || launch.sites.empty())
      return invalid("dynamic invocation launch has no point or site");
    for (const ApplicationSpatialInvocationPlan::Launch::Point &point :
         launch.points)
      if (point.dispatchTargetOrdinal != targetCount++)
        return invalid("dynamic invocation point ordinals are not dense");
  }

  struct PreparedCallable final {
    const ApplicationSpatialInvocationPlan::Callable *plan = nullptr;
    TranslatedSelectedCallable translated;
    std::size_t siteCount = 0;
  };
  std::vector<PreparedCallable> prepared;
  prepared.reserve(plan.callables.size());
  std::vector<std::uint8_t> assignedLaunches(plan.launches.size(), 0);
  for (const ApplicationSpatialInvocationPlan::Callable &callablePlan :
       plan.callables) {
    if (callablePlan.launchOrdinals.empty() ||
        !llvm::is_sorted(callablePlan.launchOrdinals) ||
        std::adjacent_find(callablePlan.launchOrdinals.begin(),
                           callablePlan.launchOrdinals.end()) !=
            callablePlan.launchOrdinals.end())
      return invalid("callable launch set is empty or noncanonical");
    auto translated = translateSelectedCallable(dataflow, plan, callablePlan,
                                                module->getContext());
    if (!translated)
      return translated.takeError();
    if (translated->rootDispatches.size() !=
            callablePlan.launchOrdinals.size() ||
        translated->rootWaits.size() != callablePlan.launchOrdinals.size())
      return invalid("callable dispatch table is inconsistent");
    const std::uint64_t firstLaunch = callablePlan.launchOrdinals.front();
    if (firstLaunch >= plan.launches.size())
      return invalid("callable launch set exceeds the invocation plan");
    const std::size_t siteCount = plan.launches[firstLaunch].sites.size();
    for (std::uint64_t launchOrdinal : callablePlan.launchOrdinals) {
      if (launchOrdinal >= plan.launches.size() ||
          assignedLaunches[launchOrdinal]++ != 0 ||
          plan.launches[launchOrdinal].sites.size() != siteCount)
        return invalid("callable launch ownership is not exact");
    }
    prepared.push_back({&callablePlan, std::move(*translated), siteCount});
  }
  if (llvm::any_of(assignedLaunches,
                   [](std::uint8_t assigned) { return assigned != 1; }))
    return invalid("callable groups do not cover every invocation launch");

  struct PendingCallableSite final {
    std::size_t callableOrdinal = 0;
    std::size_t siteOrdinal = 0;
    const std::vector<sim::DirectCallCaptureSite> *path = nullptr;
    llvm::CallInst *hostCall = nullptr;
  };
  std::vector<PendingCallableSite> pendingSites;
  llvm::DenseSet<llvm::CallInst *> uniqueHostCalls;
  for (const auto callableIndexed : llvm::enumerate(prepared)) {
    const PreparedCallable &callable = callableIndexed.value();
    const auto &launch = plan.launches[callable.plan->launchOrdinals.front()];
    for (const auto siteIndexed : llvm::enumerate(launch.sites)) {
      const auto &path = siteIndexed.value().capture.invocationPath;
      if (path.empty())
        return invalid("dynamic invocation site has no leaf locator");
      llvm::CallInst *hostCall = findInvocationCall(*module, path.back());
      if (!hostCall || !uniqueHostCalls.insert(hostCall).second)
        return invalid(
            "final-linked dynamic invocation site is absent or reused");
      pendingSites.push_back(
          {callableIndexed.index(), siteIndexed.index(), &path, hostCall});
    }
  }
  llvm::sort(pendingSites, [](const PendingCallableSite &lhs,
                              const PendingCallableSite &rhs) {
    if (lhs.path->size() != rhs.path->size())
      return lhs.path->size() > rhs.path->size();
    if (directCallPathLess(*lhs.path, *rhs.path))
      return true;
    if (directCallPathLess(*rhs.path, *lhs.path))
      return false;
    return std::tie(lhs.callableOrdinal, lhs.siteOrdinal) <
           std::tie(rhs.callableOrdinal, rhs.siteOrdinal);
  });

  struct MaterializedCallableSite final {
    const std::vector<sim::DirectCallCaptureSite> *path = nullptr;
    llvm::Function *callable = nullptr;
  };
  std::vector<MaterializedCallableSite> materializedSites;
  std::uint64_t helperOrdinal = 0;
  for (const PendingCallableSite &pending : pendingSites) {
    PreparedCallable &preparedCallable = prepared[pending.callableOrdinal];
    const ApplicationSpatialInvocationPlan::Callable &callablePlan =
        *preparedCallable.plan;
    std::vector<llvm::Function *> dispatches;
    std::vector<llvm::Function *> waits;
    dispatches.reserve(callablePlan.launchOrdinals.size());
    waits.reserve(callablePlan.launchOrdinals.size());
    for (const auto launchIndexed :
         llvm::enumerate(callablePlan.launchOrdinals)) {
      const std::uint64_t launchOrdinal = launchIndexed.value();
      const ApplicationSpatialInvocationPlan::Launch &launch =
          plan.launches[launchOrdinal];
      const ApplicationSpatialInvocationPlan::Site &site =
          launch.sites[pending.siteOrdinal];
      if (site.capture.invocationPath.empty() ||
          site.capture.invocationPath.size() != pending.path->size() ||
          !isDirectCallPathPrefix(*pending.path, site.capture.invocationPath))
        return invalid("dynamic roots disagree on their invocation locator");
      auto dispatch = materializeRootDispatchHelpers(
          *module, launch, site,
          preparedCallable.translated.rootDispatches[launchIndexed.index()]
              ->getFunctionType(),
          preparedCallable.translated.rootWaits[launchIndexed.index()]
              ->getFunctionType(),
          *dispatchBase, helperOrdinal, launchOrdinal);
      if (!dispatch)
        return dispatch.takeError();
      dispatches.push_back(dispatch->start);
      waits.push_back(dispatch->wait);
    }
    auto callable = cloneSelectedCallable(
        *module, *preparedCallable.translated.callable,
        preparedCallable.translated.rootDispatches, dispatches,
        preparedCallable.translated.rootWaits, waits, helperOrdinal);
    if (!callable)
      return callable.takeError();

    std::vector<std::pair<llvm::CallInst *, llvm::Function *>> nestedBindings;
    llvm::DenseSet<llvm::CallInst *> nestedCalls;
    for (const MaterializedCallableSite &nested : materializedSites) {
      if (nested.path->size() != pending.path->size() + 1 ||
          !isDirectCallPathPrefix(*pending.path, *nested.path))
        continue;
      const sim::DirectCallCaptureSite &nestedLeaf = nested.path->back();
      if (nestedLeaf.hostCallerSymbol != callablePlan.sourceCallableSymbol)
        continue;
      llvm::CallInst *nestedCall = findDirectCall(
          **callable, nestedLeaf.hostCalleeSymbol, nestedLeaf.hostCallOrdinal);
      if (!nestedCall || !nestedCalls.insert(nestedCall).second ||
          nestedCall->getFunctionType() != nested.callable->getFunctionType())
        return invalid("nested dynamic invocation call is not exact");
      nestedBindings.push_back({nestedCall, nested.callable});
    }
    for (const auto &[nestedCall, nestedCallable] : nestedBindings) {
      nestedCall->setCalledFunction(nestedCallable);
      clearDispatchAttributes(*nestedCall);
    }

    llvm::CallInst *call = pending.hostCall;
    llvm::Function *selected =
        module->getFunction(callablePlan.sourceCallableSymbol);
    if (!selected)
      return invalid("final-linked module has no selected callable " +
                     callablePlan.sourceCallableSymbol);
    if (call->getCalledFunction() != selected)
      return invalid("final-linked invocation call target differs from " +
                     callablePlan.sourceCallableSymbol);
    if (call->getFunctionType() != (*callable)->getFunctionType())
      return invalid("final-linked invocation call type differs from Dataflow");
    call->setCalledFunction(*callable);
    clearDispatchAttributes(*call);
    materializedSites.push_back({pending.path, *callable});
    ++helperOrdinal;
  }
  if (llvm::Error error =
          addHostEntry(*module, applicationEntry, *dispatchBase, targetCount))
    return std::move(error);
  if (llvm::Error error = linkFreestandingRuntime(*module))
    return std::move(error);
  if (llvm::Error error = pruneHostExecutableClosure(*module))
    return std::move(error);
  if (llvm::verifyModule(*module, &llvm::errs()))
    return invalid("materialized host dispatch module does not verify");
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
  std::vector<llvm::GlobalVariable *> globals;
  for (llvm::GlobalVariable &global : module->globals())
    globals.push_back(&global);
  std::vector<llvm::GlobalAlias *> aliases;
  for (llvm::GlobalAlias &alias : module->aliases())
    aliases.push_back(&alias);
  std::vector<llvm::GlobalIFunc *> ifuncs;
  for (llvm::GlobalIFunc &ifunc : module->ifuncs())
    ifuncs.push_back(&ifunc);

  for (llvm::Function *function : functions)
    function->dropAllReferences();
  for (llvm::GlobalVariable *global : globals)
    global->dropAllReferences();
  for (llvm::GlobalAlias *alias : aliases)
    alias->dropAllReferences();
  for (llvm::GlobalIFunc *ifunc : ifuncs)
    ifunc->dropAllReferences();

  for (llvm::Function *function : functions)
    function->eraseFromParent();
  for (llvm::GlobalVariable *global : globals)
    global->eraseFromParent();
  for (llvm::GlobalAlias *alias : aliases)
    alias->eraseFromParent();
  for (llvm::GlobalIFunc *ifunc : ifuncs)
    ifunc->eraseFromParent();
  for (std::uint64_t ordinal = 0; ordinal != entryCount; ++ordinal)
    emitInstructionEntry(*module, ordinal);
  return module;
}

} // namespace loom::application::detail
