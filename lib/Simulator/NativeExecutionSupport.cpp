#include "NativeExecutionSupport.h"

#include "DeterministicTranscendental.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message);
llvm::Error executionFailed(const llvm::Twine &message);
llvm::Error unsupported(const llvm::Twine &message);

constexpr llvm::StringLiteral cosineF16Symbol =
    "__loom_native_math_cos_f16_1_0";
constexpr llvm::StringLiteral cosineBF16Symbol =
    "__loom_native_math_cos_bf16_1_0";
constexpr llvm::StringLiteral cosineF32Symbol =
    "__loom_native_math_cos_f32_1_0";
constexpr llvm::StringLiteral cosineF64Symbol =
    "__loom_native_math_cos_f64_1_0";

template <typename Bits>
Bits deterministicCosineBits(const llvm::fltSemantics &semantics, Bits bits) {
  llvm::APFloat input(semantics, llvm::APInt(sizeof(Bits) * 8,
                                             static_cast<std::uint64_t>(bits)));
  llvm::APFloat result = llvm::cantFail(evaluateDeterministicCosine(input));
  return static_cast<Bits>(result.bitcastToAPInt().getZExtValue());
}

std::uint16_t deterministicCosineF16(std::uint16_t bits) {
  return deterministicCosineBits(llvm::APFloat::IEEEhalf(), bits);
}

std::uint16_t deterministicCosineBF16(std::uint16_t bits) {
  return deterministicCosineBits(llvm::APFloat::BFloat(), bits);
}

std::uint32_t deterministicCosineF32(std::uint32_t bits) {
  return deterministicCosineBits(llvm::APFloat::IEEEsingle(), bits);
}

std::uint64_t deterministicCosineF64(std::uint64_t bits) {
  return deterministicCosineBits(llvm::APFloat::IEEEdouble(), bits);
}

struct DeterministicMathCallback final {
  llvm::StringLiteral symbol;
  llvm::orc::ExecutorAddr address;
  unsigned bitWidth = 0;
};

std::optional<DeterministicMathCallback>
cosineCallback(llvm::Type *scalarType) {
  if (scalarType->isHalfTy())
    return DeterministicMathCallback{
        cosineF16Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineF16), 16};
  if (scalarType->isBFloatTy())
    return DeterministicMathCallback{
        cosineBF16Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineBF16), 16};
  if (scalarType->isFloatTy())
    return DeterministicMathCallback{
        cosineF32Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineF32), 32};
  if (scalarType->isDoubleTy())
    return DeterministicMathCallback{
        cosineF64Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineF64), 64};
  return std::nullopt;
}

std::optional<DeterministicMathCallback> cosineCallback(mlir::Type scalarType) {
  if (llvm::isa<mlir::Float16Type>(scalarType))
    return DeterministicMathCallback{
        cosineF16Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineF16), 16};
  if (llvm::isa<mlir::BFloat16Type>(scalarType))
    return DeterministicMathCallback{
        cosineBF16Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineBF16), 16};
  if (llvm::isa<mlir::Float32Type>(scalarType))
    return DeterministicMathCallback{
        cosineF32Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineF32), 32};
  if (llvm::isa<mlir::Float64Type>(scalarType))
    return DeterministicMathCallback{
        cosineF64Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicCosineF64), 64};
  return std::nullopt;
}

llvm::Error materializeDeterministicMathMlir(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::math::CosOp> cosineOps;
  module.walk([&](mlir::math::CosOp cosine) { cosineOps.push_back(cosine); });
  if (cosineOps.empty())
    return llvm::Error::success();

  for (mlir::math::CosOp cosine : cosineOps) {
    mlir::Type type = cosine.getType();
    auto vector = llvm::dyn_cast<mlir::VectorType>(type);
    if (vector && vector.isScalable())
      return unsupported(
          "deterministic native cosine does not support scalable vectors");
    mlir::Type scalarType = vector ? vector.getElementType() : type;
    auto callback = cosineCallback(scalarType);
    if (!callback)
      return unsupported(
          "deterministic native cosine has an unsupported scalar type");
    mlir::OpBuilder builder(cosine);
    mlir::LLVM::CosOp replacement = mlir::LLVM::CosOp::create(
        builder, cosine.getLoc(), type, cosine.getOperand(),
        mlir::arith::convertArithFastMathAttrToLLVM(
            cosine.getFastMathFlagsAttr()));
    cosine.replaceAllUsesWith(replacement.getResult());
    cosine.erase();
  }
  if (mlir::failed(mlir::verify(module)))
    return invalid("deterministic native math projection does not verify");
  return llvm::Error::success();
}

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

} // namespace

llvm::Error initializeNativeTarget() {
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

llvm::Error lowerStructuredModuleToLlvmDialect(mlir::ModuleOp module) {
  if (llvm::Error error = materializeDeterministicMathMlir(module))
    return error;
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
    return invalid("prepared Structured Program cannot lower to LLVM");
  return llvm::Error::success();
}

llvm::Expected<llvm::orc::ThreadSafeModule>
lowerStructuredModuleToLlvm(mlir::OwningOpRef<mlir::ModuleOp> module) {
  if (llvm::Error error = lowerStructuredModuleToLlvmDialect(*module))
    return std::move(error);

  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> translated = mlir::translateModuleToLLVMIR(
      module->getOperation(), *context, "loom-structured-native-oracle");
  if (!translated)
    return invalid("instrumented Structured Program cannot translate to LLVM");
  return llvm::orc::ThreadSafeModule(std::move(translated), std::move(context));
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

llvm::Error retargetStructuredOracle(llvm::Module &module,
                                     const llvm::orc::LLJIT &jit) {
  if (!module.getModuleInlineAsm().empty())
    return unsupported("target module contains inline assembly");
  for (const llvm::Function &function : module)
    if (function.isTargetIntrinsic())
      return unsupported("target module contains a target-specific intrinsic");

  if (module.getTargetTriple().empty())
    return unsupported("target module has no target triple");
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

llvm::Error prepareDeterministicMathOracle(llvm::Module &module,
                                           llvm::orc::LLJIT &jit) {
  std::vector<llvm::IntrinsicInst *> cosineCalls;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block)
        if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
            intrinsic && intrinsic->getIntrinsicID() == llvm::Intrinsic::cos)
          cosineCalls.push_back(intrinsic);
  llvm::DenseMap<llvm::StringRef, DeterministicMathCallback> callbacks;
  for (llvm::IntrinsicInst *intrinsic : cosineCalls) {
    llvm::Type *type = intrinsic->getType();
    if (type->isScalableTy())
      return unsupported(
          "deterministic native cosine does not support scalable vectors");
    auto callback = cosineCallback(type->getScalarType());
    if (!callback)
      return unsupported(
          "deterministic native cosine has an unsupported scalar type");
    callbacks.try_emplace(callback->symbol, *callback);
  }
  if (callbacks.empty())
    return llvm::Error::success();
  for (const auto &[symbol, callback] : callbacks)
    if (module.getNamedValue(symbol))
      return invalid("native module defines a reserved math callback symbol");

  llvm::DenseMap<llvm::StringRef, llvm::FunctionCallee> declarations;
  for (const auto &[symbol, callback] : callbacks) {
    llvm::IntegerType *bits =
        llvm::IntegerType::get(module.getContext(), callback.bitWidth);
    llvm::FunctionCallee declaration = module.getOrInsertFunction(
        symbol, llvm::FunctionType::get(bits, {bits}, false));
    declarations.try_emplace(symbol, declaration);
  }

  for (llvm::IntrinsicInst *intrinsic : cosineCalls) {
    llvm::Type *type = intrinsic->getType();
    const DeterministicMathCallback callback =
        *cosineCallback(type->getScalarType());
    llvm::FunctionCallee declaration = declarations.lookup(callback.symbol);
    llvm::IRBuilder<> builder(intrinsic);
    builder.SetCurrentDebugLocation(intrinsic->getDebugLoc());
    auto evaluateScalar = [&](llvm::Value *value) {
      llvm::IntegerType *bits =
          llvm::IntegerType::get(module.getContext(), callback.bitWidth);
      llvm::Value *inputBits = builder.CreateBitCast(value, bits);
      llvm::Value *resultBits = builder.CreateCall(declaration, {inputBits});
      return builder.CreateBitCast(resultBits, value->getType());
    };

    llvm::Value *replacement = nullptr;
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
      replacement = llvm::PoisonValue::get(vector);
      for (unsigned lane = 0; lane < vector->getNumElements(); ++lane) {
        llvm::Value *element =
            builder.CreateExtractElement(intrinsic->getArgOperand(0), lane);
        replacement = builder.CreateInsertElement(
            replacement, evaluateScalar(element), lane);
      }
    } else {
      replacement = evaluateScalar(intrinsic->getArgOperand(0));
    }
    intrinsic->replaceAllUsesWith(replacement);
    intrinsic->eraseFromParent();
  }

  if (llvm::verifyModule(module, &llvm::errs()))
    return invalid("deterministic native math projection does not verify");

  llvm::orc::SymbolMap symbols;
  for (const auto &[symbol, callback] : callbacks)
    symbols[jit.mangleAndIntern(symbol)] = {callback.address,
                                            llvm::JITSymbolFlags::Exported |
                                                llvm::JITSymbolFlags::Callable};
  if (llvm::Error error = jit.getMainJITDylib().define(
          llvm::orc::absoluteSymbols(std::move(symbols))))
    return executionFailed("cannot bind deterministic native math callbacks: " +
                           llvm::toString(std::move(error)));
  return llvm::Error::success();
}

CanonicalValueSequence
readDefinedNativeValue(llvm::ArrayRef<std::uint8_t> bytes,
                       std::uint64_t lanesPerToken, std::uint32_t laneBitWidth,
                       bool littleEndian) {
  auto readLane = [&](std::uint64_t bitOffset) {
    llvm::APInt bits(laneBitWidth, 0);
    for (std::uint32_t bit = 0; bit < laneBitWidth; ++bit) {
      const std::uint64_t storageBit = bitOffset + bit;
      const std::uint64_t byteOrdinal = storageBit / 8;
      const std::uint32_t bitInByte = storageBit % 8;
      const std::uint64_t addressedByte =
          littleEndian ? byteOrdinal : bytes.size() - 1 - byteOrdinal;
      if ((bytes[addressedByte] >> bitInByte) & 1U)
        bits.setBit(bit);
    }
    return bits;
  };

  CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes.reserve(lanesPerToken);
  for (std::uint64_t lane = 0; lane < lanesPerToken; ++lane)
    sequence.lanes.push_back(
        SemanticLane::defined(readLane(lane * laneBitWidth)));
  return sequence;
}

} // namespace loom::sim::detail
