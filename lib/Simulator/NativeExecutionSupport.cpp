#include "NativeExecutionSupport.h"

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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <system_error>
#include <utility>

namespace loom::sim::detail {
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
