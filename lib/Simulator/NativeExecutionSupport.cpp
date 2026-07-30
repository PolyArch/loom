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

constexpr llvm::StringLiteral unaryMathF16Symbol =
    "__loom_native_unary_math_f16_1_0";
constexpr llvm::StringLiteral unaryMathBF16Symbol =
    "__loom_native_unary_math_bf16_1_0";
constexpr llvm::StringLiteral unaryMathF32Symbol =
    "__loom_native_unary_math_f32_1_0";
constexpr llvm::StringLiteral unaryMathF64Symbol =
    "__loom_native_unary_math_f64_1_0";

template <typename Bits>
Bits deterministicUnaryMathBits(const llvm::fltSemantics &semantics,
                                std::uint32_t schemaOrdinal, Bits bits) {
  llvm::APFloat input(semantics, llvm::APInt(sizeof(Bits) * 8,
                                             static_cast<std::uint64_t>(bits)));
  llvm::APFloat result = llvm::cantFail(evaluateDeterministicUnaryMath(
      static_cast<dataflow::OperationSchemaId>(schemaOrdinal), input));
  return static_cast<Bits>(result.bitcastToAPInt().getZExtValue());
}

std::uint16_t deterministicUnaryMathF16(std::uint32_t schema,
                                        std::uint16_t bits) {
  return deterministicUnaryMathBits(llvm::APFloat::IEEEhalf(), schema, bits);
}

std::uint16_t deterministicUnaryMathBF16(std::uint32_t schema,
                                         std::uint16_t bits) {
  return deterministicUnaryMathBits(llvm::APFloat::BFloat(), schema, bits);
}

std::uint32_t deterministicUnaryMathF32(std::uint32_t schema,
                                        std::uint32_t bits) {
  return deterministicUnaryMathBits(llvm::APFloat::IEEEsingle(), schema, bits);
}

std::uint64_t deterministicUnaryMathF64(std::uint32_t schema,
                                        std::uint64_t bits) {
  return deterministicUnaryMathBits(llvm::APFloat::IEEEdouble(), schema, bits);
}

struct DeterministicMathCallback final {
  llvm::StringLiteral symbol;
  llvm::orc::ExecutorAddr address;
  unsigned bitWidth = 0;
};

std::optional<DeterministicMathCallback>
unaryMathCallback(llvm::Type *scalarType) {
  if (scalarType->isHalfTy())
    return DeterministicMathCallback{
        unaryMathF16Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicUnaryMathF16), 16};
  if (scalarType->isBFloatTy())
    return DeterministicMathCallback{
        unaryMathBF16Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicUnaryMathBF16), 16};
  if (scalarType->isFloatTy())
    return DeterministicMathCallback{
        unaryMathF32Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicUnaryMathF32), 32};
  if (scalarType->isDoubleTy())
    return DeterministicMathCallback{
        unaryMathF64Symbol,
        llvm::orc::ExecutorAddr::fromPtr(&deterministicUnaryMathF64), 64};
  return std::nullopt;
}

std::optional<dataflow::OperationSchemaId>
unaryMathSchema(llvm::Intrinsic::ID intrinsic) {
  using Schema = dataflow::OperationSchemaId;
  switch (intrinsic) {
  case llvm::Intrinsic::sin:
    return Schema::MathSin;
  case llvm::Intrinsic::cos:
    return Schema::MathCos;
  case llvm::Intrinsic::tan:
    return Schema::MathTan;
  case llvm::Intrinsic::sinh:
    return Schema::MathSinh;
  case llvm::Intrinsic::cosh:
    return Schema::MathCosh;
  case llvm::Intrinsic::tanh:
    return Schema::MathTanh;
  case llvm::Intrinsic::exp:
    return Schema::MathExp;
  case llvm::Intrinsic::exp2:
    return Schema::MathExp2;
  case llvm::Intrinsic::log:
    return Schema::MathLog;
  case llvm::Intrinsic::log2:
    return Schema::MathLog2;
  case llvm::Intrinsic::log10:
    return Schema::MathLog10;
  case llvm::Intrinsic::sqrt:
    return Schema::MathSqrt;
  default:
    return std::nullopt;
  }
}

template <typename SourceOp, typename TargetOp>
llvm::Error materializeDeterministicUnaryMath(mlir::ModuleOp module) {
  llvm::SmallVector<SourceOp> operations;
  module.walk([&](SourceOp operation) { operations.push_back(operation); });
  for (SourceOp operation : operations) {
    mlir::Type type = operation.getType();
    auto vector = llvm::dyn_cast<mlir::VectorType>(type);
    if (vector && vector.isScalable())
      return unsupported(
          "deterministic native unary math does not support scalable vectors");
    mlir::Type scalarType = vector ? vector.getElementType() : type;
    if (!llvm::isa<mlir::Float16Type, mlir::BFloat16Type, mlir::Float32Type,
                   mlir::Float64Type>(scalarType))
      return unsupported(
          "deterministic native unary math has an unsupported scalar type");
    mlir::OpBuilder builder(operation);
    TargetOp replacement = TargetOp::create(
        builder, operation.getLoc(), type, operation.getOperand(),
        mlir::arith::convertArithFastMathAttrToLLVM(
            operation.getFastMathFlagsAttr()));
    operation.replaceAllUsesWith(replacement.getResult());
    operation.erase();
  }
  return llvm::Error::success();
}

llvm::Error materializeDeterministicMathMlir(mlir::ModuleOp module) {
#define LOOM_MATERIALIZE_UNARY_MATH(SourceOp, TargetOp)                        \
  if (llvm::Error error =                                                      \
          materializeDeterministicUnaryMath<SourceOp, TargetOp>(module))       \
  return error
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::SinOp, mlir::LLVM::SinOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::CosOp, mlir::LLVM::CosOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::TanOp, mlir::LLVM::TanOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::SinhOp, mlir::LLVM::SinhOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::CoshOp, mlir::LLVM::CoshOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::TanhOp, mlir::LLVM::TanhOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::ExpOp, mlir::LLVM::ExpOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::Exp2Op, mlir::LLVM::Exp2Op);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::LogOp, mlir::LLVM::LogOp);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::Log2Op, mlir::LLVM::Log2Op);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::Log10Op, mlir::LLVM::Log10Op);
  LOOM_MATERIALIZE_UNARY_MATH(mlir::math::SqrtOp, mlir::LLVM::SqrtOp);
#undef LOOM_MATERIALIZE_UNARY_MATH
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
  using MathCall =
      std::pair<llvm::IntrinsicInst *, dataflow::OperationSchemaId>;
  std::vector<MathCall> mathCalls;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block)
        if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
            intrinsic)
          if (auto schema = unaryMathSchema(intrinsic->getIntrinsicID()))
            mathCalls.emplace_back(intrinsic, *schema);
  llvm::DenseMap<llvm::StringRef, DeterministicMathCallback> callbacks;
  for (const auto &[intrinsic, schema] : mathCalls) {
    (void)schema;
    llvm::Type *type = intrinsic->getType();
    if (type->isScalableTy())
      return unsupported(
          "deterministic native unary math does not support scalable vectors");
    auto callback = unaryMathCallback(type->getScalarType());
    if (!callback)
      return unsupported(
          "deterministic native unary math has an unsupported scalar type");
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
    llvm::IntegerType *schema = llvm::Type::getInt32Ty(module.getContext());
    llvm::FunctionCallee declaration = module.getOrInsertFunction(
        symbol, llvm::FunctionType::get(bits, {schema, bits}, false));
    declarations.try_emplace(symbol, declaration);
  }

  for (const auto &[intrinsic, schema] : mathCalls) {
    llvm::Type *type = intrinsic->getType();
    const DeterministicMathCallback callback =
        *unaryMathCallback(type->getScalarType());
    const std::uint32_t schemaValue = static_cast<std::uint32_t>(schema);
    llvm::FunctionCallee declaration = declarations.lookup(callback.symbol);
    llvm::IRBuilder<> builder(intrinsic);
    builder.SetCurrentDebugLocation(intrinsic->getDebugLoc());
    auto evaluateScalar = [&](llvm::Value *value) {
      llvm::IntegerType *bits =
          llvm::IntegerType::get(module.getContext(), callback.bitWidth);
      llvm::Value *inputBits = builder.CreateBitCast(value, bits);
      llvm::Value *schemaOrdinal = llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(module.getContext()), schemaValue);
      llvm::Value *resultBits =
          builder.CreateCall(declaration, {schemaOrdinal, inputBits});
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
