#include "Dataflow/IR/GepAddressPlan.h"

#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"

#include <cstdint>
#include <system_error>

namespace dataflow::semantics {
namespace {

llvm::Expected<std::uint64_t>
fixedAllocByteSize(const llvm::DataLayout &layout,
                   mlir::LLVM::TypeToLLVMIRTranslator &translator,
                   mlir::Type type) {
  llvm::Type *translated = translator.translateType(type);
  if (!translated)
    return llvm::createStringError(
        std::errc::not_supported,
        "LLVM GEP source type has no LLVM IR projection");
  llvm::TypeSize size = layout.getTypeAllocSize(translated);
  if (size.isScalable() || size.getFixedValue() == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "LLVM GEP source type has no fixed nonzero allocation size");
  return size.getFixedValue();
}

} // namespace

llvm::Expected<GepAddressPlan> projectGepAddressPlan(mlir::LLVM::GEPOp op,
                                                  mlir::Operation *graphScope) {
  auto baseType =
      mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(op.getBase().getType());
  auto resultType =
      mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(op.getRes().getType());
  if (!baseType || !resultType || baseType != resultType)
    return llvm::createStringError(
        std::errc::not_supported,
        "DFG-sim supports scalar GEP with one exact pointer type");
  for (mlir::Value index : op.getDynamicIndices())
    if (!mlir::isa<mlir::IntegerType>(index.getType()))
      return llvm::createStringError(
          std::errc::not_supported,
          "DFG-sim supports scalar integer GEP indices");

  auto pointerLayout =
      ::loom::resolvePointerLayout(graphScope, baseType.getAddressSpace());
  if (!pointerLayout)
    return pointerLayout.takeError();
  if (pointerLayout->kind == ::loom::PointerLayoutKind::Unstable ||
      pointerLayout->kind == ::loom::PointerLayoutKind::ExternalState)
    return llvm::createStringError(
        std::errc::not_supported,
        "DFG-sim has no provider for this LLVM pointer representation kind");
  auto llvmLayout = ::loom::resolveLLVMDataLayout(graphScope);
  if (!llvmLayout)
    return llvmLayout.takeError();

  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator translator(llvmContext);
  GepAddressPlan plan{*pointerLayout, op.getNoWrapFlags(), {}};
  mlir::Type indexedType = op.getElemType();
  unsigned dynamicOrdinal = 0;
  for (auto [position, rawIndex] :
       llvm::enumerate(op.getRawConstantIndices())) {
    GepOffsetTerm term;
    if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex) {
      if (dynamicOrdinal >= op.getDynamicIndices().size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "LLVM GEP dynamic-index table is malformed");
      term.dynamicOperandOrdinal = 1 + dynamicOrdinal++;
    } else {
      term.constantIndex = llvm::APInt(
          32, static_cast<std::uint64_t>(static_cast<std::int64_t>(rawIndex)),
          /*isSigned=*/true);
    }

    std::uint64_t scale = 0;
    if (position == 0) {
      auto size = fixedAllocByteSize(*llvmLayout, translator, indexedType);
      if (!size)
        return size.takeError();
      scale = *size;
    } else if (auto array =
                   mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(indexedType)) {
      indexedType = array.getElementType();
      auto size = fixedAllocByteSize(*llvmLayout, translator, indexedType);
      if (!size)
        return size.takeError();
      scale = *size;
    } else if (auto vector = mlir::dyn_cast<mlir::VectorType>(indexedType)) {
      indexedType = vector.getElementType();
      auto size = fixedAllocByteSize(*llvmLayout, translator, indexedType);
      if (!size)
        return size.takeError();
      scale = *size;
    } else if (auto structure =
                   mlir::dyn_cast<mlir::LLVM::LLVMStructType>(indexedType)) {
      if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex || rawIndex < 0 ||
          static_cast<std::size_t>(rawIndex) >= structure.getBody().size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "LLVM GEP structure index is not a valid constant field");
      llvm::Type *translated = translator.translateType(structure);
      auto *llvmStruct = llvm::dyn_cast_or_null<llvm::StructType>(translated);
      if (!llvmStruct)
        return llvm::createStringError(
            std::errc::not_supported,
            "LLVM GEP structure has no LLVM IR layout projection");
      const std::uint64_t offset =
          llvmLayout->getStructLayout(llvmStruct)->getElementOffset(rawIndex);
      term.constantIndex = llvm::APInt(pointerLayout->addressBits, offset,
                                       /*isSigned=*/false,
                                       /*implicitTrunc=*/true);
      scale = 1;
      indexedType = structure.getBody()[rawIndex];
    } else {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "LLVM GEP index path does not match its source element type");
    }
    term.scale = llvm::APInt(pointerLayout->addressBits, scale,
                             /*isSigned=*/false,
                             /*implicitTrunc=*/true);
    plan.terms.push_back(std::move(term));
  }
  if (dynamicOrdinal != op.getDynamicIndices().size())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "LLVM GEP has unused dynamic indices");
  return plan;
}

} // namespace dataflow::semantics
