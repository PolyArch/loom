#include "Frontend/Analysis/MemoryAddressProjection.h"
#include "Common/PointerLayout.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace loom::frontend::analysis {
namespace {

std::optional<llvm::DataLayout>
getModuleLLVMDataLayout(mlir::Operation *scope) {
  auto layout = ::loom::resolveLLVMDataLayout(scope);
  if (!layout) {
    llvm::consumeError(layout.takeError());
    return std::nullopt;
  }
  return std::move(*layout);
}

std::optional<llvm::APInt> integerConstantValue(mlir::Value value) {
  llvm::APInt constant;
  if (mlir::matchPattern(value, mlir::m_ConstantInt(&constant)))
    return constant;
  if (auto operation = value.getDefiningOp<dataflow::ConstantOp>())
    if (auto integer =
            llvm::dyn_cast<mlir::IntegerAttr>(operation.getConstValue()))
      return integer.getValue();
  return std::nullopt;
}

class PowerOfTwoMultipleProof final {
public:
  explicit PowerOfTwoMultipleProof(unsigned shift) : shift(shift) {}

  bool prove(mlir::Value value) {
    if (shift == 0 || assumptions.contains(value))
      return true;
    if (!active.insert(value).second)
      return false;
    bool result = proveImpl(value);
    active.erase(value);
    return result;
  }

private:
  bool proveWhileInvariant(mlir::scf::WhileOp loop, unsigned ordinal,
                           mlir::BlockArgument beforeArgument) {
    if (ordinal >= loop.getInits().size() ||
        ordinal >= loop.getYieldOp().getNumOperands() ||
        !prove(loop.getInits()[ordinal]))
      return false;
    const bool inserted = assumptions.insert(beforeArgument).second;
    bool preserved = prove(loop.getYieldOp().getOperand(ordinal));
    if (inserted)
      assumptions.erase(beforeArgument);
    return preserved;
  }

  bool proveBlockArgument(mlir::BlockArgument argument) {
    auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(
        argument.getOwner()->getParentOp());
    if (!loop)
      return false;
    const unsigned ordinal = argument.getArgNumber();
    if (argument.getOwner() == loop.getBeforeBody())
      return proveWhileInvariant(loop, ordinal, argument);
    if (argument.getOwner() == loop.getAfterBody() &&
        ordinal < loop.getConditionOp().getArgs().size())
      return prove(loop.getConditionOp().getArgs()[ordinal]);
    return false;
  }

  bool proveSelectionResult(mlir::Operation *selection, unsigned ordinal) {
    if (!selection || selection->getNumRegions() == 0)
      return false;
    for (mlir::Region &region : selection->getRegions()) {
      if (!region.hasOneBlock())
        return false;
      auto yield =
          llvm::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
      if (!yield || ordinal >= yield.getNumOperands() ||
          !prove(yield.getOperand(ordinal)))
        return false;
    }
    return true;
  }

  bool proveImpl(mlir::Value value) {
    if (std::optional<llvm::APInt> constant = integerConstantValue(value))
      return constant->isZero() || constant->countTrailingZeros() >= shift;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value))
      return proveBlockArgument(argument);
    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
      return prove(add.getLhs()) && prove(add.getRhs());
    if (auto subtract = value.getDefiningOp<mlir::arith::SubIOp>())
      return prove(subtract.getLhs()) && prove(subtract.getRhs());
    if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>())
      return prove(multiply.getLhs()) || prove(multiply.getRhs());
    if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>())
      return prove(extension.getIn());
    if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>())
      return prove(extension.getIn());
    if (auto truncation = value.getDefiningOp<mlir::arith::TruncIOp>()) {
      auto resultType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
      return resultType && resultType.getWidth() >= shift &&
             prove(truncation.getIn());
    }
    if (auto leftShift = value.getDefiningOp<mlir::arith::ShLIOp>()) {
      if (prove(leftShift.getLhs()))
        return true;
      std::optional<llvm::APInt> amount =
          integerConstantValue(leftShift.getRhs());
      auto integerType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
      return amount && integerType && amount->ult(integerType.getWidth()) &&
             amount->getZExtValue() >= shift;
    }
    if (auto loop = value.getDefiningOp<mlir::scf::WhileOp>()) {
      const unsigned ordinal =
          llvm::cast<mlir::OpResult>(value).getResultNumber();
      if (ordinal >= loop.getBeforeBody()->getNumArguments() ||
          ordinal >= loop.getConditionOp().getArgs().size())
        return false;
      auto beforeArgument = loop.getBeforeBody()->getArgument(ordinal);
      return proveWhileInvariant(loop, ordinal, beforeArgument) &&
             prove(loop.getConditionOp().getArgs()[ordinal]);
    }
    if (auto selection = value.getDefiningOp<mlir::scf::IfOp>())
      return proveSelectionResult(
          selection, llvm::cast<mlir::OpResult>(value).getResultNumber());
    if (auto selection = value.getDefiningOp<mlir::scf::IndexSwitchOp>())
      return proveSelectionResult(
          selection, llvm::cast<mlir::OpResult>(value).getResultNumber());
    return false;
  }

  unsigned shift;
  llvm::SmallDenseSet<mlir::Value, 8> assumptions;
  llvm::SmallDenseSet<mlir::Value, 16> active;
};

bool isSupportedElementType(mlir::Type type) {
  if (llvm::isa<mlir::IntegerType, mlir::Float16Type, mlir::BFloat16Type,
                mlir::Float32Type, mlir::Float64Type, mlir::Float80Type,
                mlir::Float128Type, mlir::LLVM::LLVMPointerType>(type))
    return true;
  auto array = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(type);
  return array && array.getNumElements() != 0 &&
         isSupportedElementType(array.getElementType());
}

std::optional<std::uint64_t>
getMLIRAllocByteSize(const mlir::DataLayout &layout, mlir::Type type) {
  llvm::TypeSize bytes = layout.getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return std::nullopt;
  std::uint64_t alignment = layout.getTypeABIAlignment(type);
  if (alignment == 0 ||
      bytes.getFixedValue() >
          std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
    return std::nullopt;
  return llvm::alignTo(bytes.getFixedValue(), alignment);
}

std::optional<std::uint64_t>
getMLIRStoreByteSize(const mlir::DataLayout &layout, mlir::Type type) {
  if (!isSupportedElementType(type))
    return std::nullopt;
  llvm::TypeSize bytes = layout.getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return std::nullopt;
  return bytes.getFixedValue();
}

std::optional<std::uint64_t>
getLLVMAllocByteSize(const llvm::DataLayout &layout,
                     mlir::LLVM::TypeToLLVMIRTranslator &translator,
                     mlir::Type type) {
  llvm::Type *translated = translator.translateType(type);
  if (!translated || !translated->isSized())
    return std::nullopt;
  llvm::TypeSize bytes = layout.getTypeAllocSize(translated);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return std::nullopt;
  return bytes.getFixedValue();
}

std::optional<std::uint64_t>
getLLVMStoreByteSize(const llvm::DataLayout &layout,
                     mlir::LLVM::TypeToLLVMIRTranslator &translator,
                     mlir::Type type) {
  if (!isSupportedElementType(type))
    return std::nullopt;
  llvm::TypeSize bytes =
      layout.getTypeStoreSize(translator.translateType(type));
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return std::nullopt;
  return bytes.getFixedValue();
}

struct ResolvedPointerRoot {
  mlir::Value root;
  mlir::Operation *scope = nullptr;
  std::optional<llvm::DataLayout> llvmDataLayout;
  std::uint64_t indexBitWidth = 0;
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> gepsLeafToRoot;
};

std::optional<ResolvedPointerRoot>
resolvePointerRoot(mlir::Value pointer,
                   llvm::function_ref<bool(mlir::Value)> isBoundaryRoot) {
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> gepsLeafToRoot;
  mlir::Value root = pointer;
  while (!isBoundaryRoot(root)) {
    auto gep = root.getDefiningOp<mlir::LLVM::GEPOp>();
    if (!gep)
      break;
    gepsLeafToRoot.push_back(gep);
    root = gep.getBase();
  }
  auto rootType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(root.getType());
  if (!rootType)
    return std::nullopt;

  mlir::Operation *scope = pointer.getDefiningOp();
  if (!scope) {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer);
    scope = argument ? argument.getOwner()->getParentOp() : nullptr;
  }
  if (!scope)
    return std::nullopt;

  std::optional<llvm::DataLayout> llvmDataLayout =
      getModuleLLVMDataLayout(scope);
  std::optional<std::uint64_t> indexBitWidth;
  if (llvmDataLayout)
    indexBitWidth =
        llvmDataLayout->getIndexSizeInBits(rootType.getAddressSpace());
  else
    indexBitWidth =
        mlir::DataLayout::closest(scope).getTypeIndexBitwidth(root.getType());
  if (!indexBitWidth || *indexBitWidth == 0 || *indexBitWidth > 64)
    return std::nullopt;
  return ResolvedPointerRoot{root, scope, std::move(llvmDataLayout),
                             *indexBitWidth, std::move(gepsLeafToRoot)};
}

std::optional<unsigned> getIntegralIndexBitWidth(const mlir::DataLayout &layout,
                                                 mlir::Type type) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type)) {
    if (!integer.isSignless())
      return std::nullopt;
    return integer.getWidth();
  }
  if (!llvm::isa<mlir::IndexType>(type))
    return std::nullopt;
  llvm::TypeSize bits = layout.getTypeSizeInBits(type);
  if (bits.isScalable() || bits.getFixedValue() == 0)
    return std::nullopt;
  return static_cast<unsigned>(
      std::min<std::uint64_t>(bits.getFixedValue(), 64));
}

std::optional<std::uint64_t>
getStructElementOffset(const mlir::DataLayout &layout,
                       mlir::LLVM::LLVMStructType type, unsigned ordinal) {
  if (type.isOpaque() || ordinal >= type.getBody().size())
    return std::nullopt;
  std::uint64_t offset = 0;
  for (unsigned index = 0; index <= ordinal; ++index) {
    mlir::Type element = type.getBody()[index];
    std::optional<std::uint64_t> size = getMLIRAllocByteSize(layout, element);
    if (!size)
      return std::nullopt;
    std::uint64_t alignment =
        type.isPacked() ? 1 : layout.getTypeABIAlignment(element);
    if (alignment == 0 ||
        offset > std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
      return std::nullopt;
    offset = llvm::alignTo(offset, alignment);
    if (index == ordinal)
      return offset;
    if (offset > std::numeric_limits<std::uint64_t>::max() - *size)
      return std::nullopt;
    offset += *size;
  }
  llvm_unreachable("struct element loop must return at the selected ordinal");
}

} // namespace

std::optional<ExactElementStrideScale>
resolveExactElementStrideScale(mlir::Value index, std::uint64_t byteStride,
                               std::uint64_t elementBytes) {
  constexpr std::uint64_t maxSigned =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (!index || byteStride == 0 || byteStride > maxSigned ||
      elementBytes == 0 || elementBytes > maxSigned ||
      !llvm::isPowerOf2_64(elementBytes))
    return std::nullopt;
  const unsigned elementShift = llvm::Log2_64(elementBytes);
  const unsigned strideShift =
      std::min<unsigned>(elementShift, llvm::countr_zero(byteStride));
  const unsigned exactSignedDivideShift = elementShift - strideShift;
  if (exactSignedDivideShift != 0) {
    auto indexType = llvm::dyn_cast<mlir::IntegerType>(index.getType());
    if (!indexType || !indexType.isSignless() ||
        exactSignedDivideShift >= indexType.getWidth())
      return std::nullopt;
  }
  if (!PowerOfTwoMultipleProof(exactSignedDivideShift).prove(index))
    return std::nullopt;
  return ExactElementStrideScale{
      static_cast<std::int64_t>(byteStride >> strideShift),
      exactSignedDivideShift};
}

static std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddressImpl(
    mlir::Value pointer, mlir::Type accessType,
    std::optional<unsigned> canonicalIndexBits,
    llvm::function_ref<bool(mlir::Value)> isBoundaryRoot) {
  auto pointerRoot = resolvePointerRoot(pointer, isBoundaryRoot);
  if (!pointerRoot)
    return std::nullopt;
  const unsigned arithmeticBits =
      canonicalIndexBits ? *canonicalIndexBits : pointerRoot->indexBitWidth;
  if (arithmeticBits == 0 || arithmeticBits > mlir::IntegerType::kMaxWidth)
    return std::nullopt;
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> leafToRoot =
      std::move(pointerRoot->gepsLeafToRoot);
  mlir::Value root = pointerRoot->root;
  // The chain is discharged only when all pointer uses lower to logical
  // memory accesses. On an LLVM-defined execution every no-wrap condition
  // already holds; violating one poisons the source address and its consuming
  // load or store has undefined behavior. The linear address is therefore the
  // exact defined-domain projection regardless of the chain's flag spelling.

  mlir::Operation *scope = pointerRoot->scope;
  mlir::DataLayout dataLayout = mlir::DataLayout::closest(scope);
  std::optional<llvm::DataLayout> &llvmDataLayout = pointerRoot->llvmDataLayout;
  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator translator(llvmContext);
  const std::uint64_t pointerIndexBits = pointerRoot->indexBitWidth;

  auto getAllocBytes = [&](mlir::Type type) {
    if (llvmDataLayout)
      return getLLVMAllocByteSize(*llvmDataLayout, translator, type);
    return getMLIRAllocByteSize(dataLayout, type);
  };
  auto getStoreBytes = [&](mlir::Type type) {
    if (llvmDataLayout)
      return getLLVMStoreByteSize(*llvmDataLayout, translator, type);
    return getMLIRStoreByteSize(dataLayout, type);
  };
  std::optional<std::uint64_t> elementBytes = getAllocBytes(accessType);
  std::optional<std::uint64_t> accessBytes = getStoreBytes(accessType);
  constexpr std::uint64_t maxSigned =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (!elementBytes || !accessBytes || *elementBytes > maxSigned ||
      *accessBytes > maxSigned || !llvm::isPowerOf2_64(*elementBytes))
    return std::nullopt;
  unsigned elementShift = llvm::Log2_64(*elementBytes);
  if (elementShift >= pointerIndexBits)
    return std::nullopt;

  ResolvedLinearMemoryAddress result;
  result.root = root;
  result.byteToElementShift = elementShift;
  result.elementAllocByteCount = *elementBytes;
  result.accessByteCount = *accessBytes;
  std::int64_t constantByteOffset = 0;

  for (mlir::LLVM::GEPOp gep : llvm::reverse(leafToRoot)) {
    auto rawIndices = gep.getRawConstantIndices();
    auto dynamicIndices = gep.getDynamicIndices();
    if (rawIndices.empty())
      return std::nullopt;
    const std::int64_t enteringByteOffset = constantByteOffset;
    const std::size_t enteringTermCount = result.terms.size();
    std::size_t dynamicOrdinal = 0;
    mlir::Type indexedType = gep.getElemType();
    for (auto [position, rawIndex] : llvm::enumerate(rawIndices)) {
      std::optional<std::uint64_t> strideBytes;
      std::optional<std::uint64_t> fixedByteOffset;
      if (position == 0) {
        strideBytes = getAllocBytes(indexedType);
      } else if (auto array =
                     llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(indexedType)) {
        indexedType = array.getElementType();
        strideBytes = getAllocBytes(indexedType);
      } else if (auto vector = llvm::dyn_cast<mlir::VectorType>(indexedType)) {
        indexedType = vector.getElementType();
        strideBytes = getAllocBytes(indexedType);
      } else if (auto structure =
                     llvm::dyn_cast<mlir::LLVM::LLVMStructType>(indexedType)) {
        if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex || rawIndex < 0 ||
            static_cast<std::size_t>(rawIndex) >= structure.getBody().size())
          return std::nullopt;
        if (llvmDataLayout) {
          auto *translated = llvm::dyn_cast_or_null<llvm::StructType>(
              translator.translateType(structure));
          if (!translated)
            return std::nullopt;
          fixedByteOffset = llvmDataLayout->getStructLayout(translated)
                                ->getElementOffset(rawIndex);
        } else {
          fixedByteOffset =
              getStructElementOffset(dataLayout, structure, rawIndex);
        }
        indexedType = structure.getBody()[rawIndex];
      } else {
        return std::nullopt;
      }

      if (fixedByteOffset) {
        if (*fixedByteOffset > maxSigned ||
            llvm::AddOverflow(constantByteOffset,
                              static_cast<std::int64_t>(*fixedByteOffset),
                              constantByteOffset))
          return std::nullopt;
        continue;
      }
      if (!strideBytes || *strideBytes > maxSigned ||
          !llvm::isIntN(pointerIndexBits,
                        static_cast<std::int64_t>(*strideBytes)) ||
          !llvm::isIntN(arithmeticBits,
                        static_cast<std::int64_t>(*strideBytes)))
        return std::nullopt;

      if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex) {
        if (dynamicOrdinal >= dynamicIndices.size())
          return std::nullopt;
        mlir::Value index = dynamicIndices[dynamicOrdinal++];
        std::optional<unsigned> indexBits;
        if (llvmDataLayout) {
          auto integer = llvm::dyn_cast<mlir::IntegerType>(index.getType());
          if (!integer || !integer.isSignless())
            return std::nullopt;
          indexBits = integer.getWidth();
        } else {
          indexBits = getIntegralIndexBitWidth(dataLayout, index.getType());
        }
        if (!indexBits || *indexBits > arithmeticBits)
          return std::nullopt;
        result.indexType =
            mlir::IntegerType::get(pointer.getContext(), arithmeticBits);
        result.terms.push_back(
            {index, static_cast<std::int64_t>(*strideBytes)});
        if (canonicalIndexBits) {
          if (llvm::isa<mlir::IndexType>(index.getType()) && *elementBytes != 1)
            return std::nullopt;
          std::optional<ExactElementStrideScale> elementScale =
              resolveExactElementStrideScale(index, *strideBytes,
                                             *elementBytes);
          if (!elementScale)
            return std::nullopt;
          result.elementTerms.push_back({index, elementScale->scale,
                                         elementScale->exactSignedDivideShift});
        }
        continue;
      }

      std::int64_t term = 0;
      if (llvm::MulOverflow(static_cast<std::int64_t>(rawIndex),
                            static_cast<std::int64_t>(*strideBytes), term) ||
          llvm::AddOverflow(constantByteOffset, term, constantByteOffset))
        return std::nullopt;
    }
    if (dynamicOrdinal != dynamicIndices.size())
      return std::nullopt;
    // A no-unsigned-wrap step computes a non-negative byte offset. When that
    // whole offset is one positively scaled index, the index is non-negative.
    if (mlir::LLVM::bitEnumContainsAny(gep.getNoWrapFlags(),
                                       mlir::LLVM::GEPNoWrapFlags::nuw) &&
        constantByteOffset == enteringByteOffset &&
        result.terms.size() == enteringTermCount + 1 &&
        result.terms.back().byteStride > 0)
      result.terms.back().nonNegative = true;
  }

  const bool elementAligned =
      constantByteOffset % static_cast<std::int64_t>(*elementBytes) == 0;
  if ((canonicalIndexBits && !elementAligned) ||
      !llvm::isIntN(pointerIndexBits, constantByteOffset) ||
      !llvm::isIntN(arithmeticBits, constantByteOffset))
    return std::nullopt;

  if (!result.indexType)
    result.indexType =
        mlir::IntegerType::get(pointer.getContext(), arithmeticBits);
  result.byteBias = constantByteOffset;
  if (elementAligned)
    result.elementBias =
        constantByteOffset / static_cast<std::int64_t>(*elementBytes);
  result.addressBitWidth = arithmeticBits;
  for (mlir::LLVM::GEPOp gep : leafToRoot)
    result.gepsLeafToRoot.push_back(gep.getOperation());
  return result;
}

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, mlir::Type accessType,
                           unsigned canonicalIndexBits) {
  return resolveLinearMemoryAddressImpl(pointer, accessType, canonicalIndexBits,
                                        [](mlir::Value) { return false; });
}

std::optional<ResolvedLinearMemoryAddress> resolveLinearMemoryAddress(
    mlir::Value pointer, mlir::Type accessType, unsigned canonicalIndexBits,
    llvm::function_ref<bool(mlir::Value)> isBoundaryRoot) {
  return resolveLinearMemoryAddressImpl(pointer, accessType, canonicalIndexBits,
                                        isBoundaryRoot);
}

std::optional<ResolvedLinearMemoryAddress>
resolveLinearPointerAddress(mlir::Value pointer, mlir::Type accessType) {
  return resolveLinearMemoryAddressImpl(pointer, accessType, std::nullopt,
                                        [](mlir::Value) { return false; });
}

std::optional<ResolvedLinearMemoryAddress> resolveLinearPointerAddress(
    mlir::Value pointer, mlir::Type accessType,
    llvm::function_ref<bool(mlir::Value)> isBoundaryRoot) {
  return resolveLinearMemoryAddressImpl(pointer, accessType, std::nullopt,
                                        isBoundaryRoot);
}

std::optional<std::uint64_t>
projectFixedAllocationByteCount(mlir::LLVM::AllocaOp allocation) {
  auto count = integerConstantValue(allocation.getArraySize());
  if (!count || count->isZero() || count->isNegative() ||
      count->getActiveBits() > 64)
    return std::nullopt;
  std::optional<std::uint64_t> elementBytes;
  if (auto layout = getModuleLLVMDataLayout(allocation)) {
    llvm::LLVMContext context;
    mlir::LLVM::TypeToLLVMIRTranslator translator(context);
    elementBytes =
        getLLVMAllocByteSize(*layout, translator, allocation.getElemType());
  } else {
    elementBytes = getMLIRAllocByteSize(mlir::DataLayout::closest(allocation),
                                        allocation.getElemType());
  }
  if (!elementBytes ||
      count->getZExtValue() >
          std::numeric_limits<std::uint64_t>::max() / *elementBytes)
    return std::nullopt;
  return count->getZExtValue() * *elementBytes;
}

} // namespace loom::frontend::analysis
