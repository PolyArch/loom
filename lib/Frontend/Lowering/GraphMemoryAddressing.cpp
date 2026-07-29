#include "GraphMemoryAddressing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace loom::lowering {
namespace {

bool isGraphPointerArgument(mlir::Value value, dataflow::GraphOp graph) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner() == &graph.getBody().front() &&
         llvm::isa<mlir::LLVM::LLVMPointerType>(argument.getType());
}

std::optional<llvm::DataLayout>
getModuleLLVMDataLayout(mlir::Operation *scope) {
  auto module = scope->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return std::nullopt;
  auto layout = module->getAttrOfType<mlir::StringAttr>(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (!layout)
    return std::nullopt;
  return llvm::DataLayout(layout.getValue());
}

std::optional<llvm::APInt> integerConstantValue(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getValue();
  }
  if (auto constant = value.getDefiningOp<dataflow::ConstantOp>()) {
    if (auto integer =
            llvm::dyn_cast<mlir::IntegerAttr>(constant.getConstValue()))
      return integer.getValue();
  }
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
    return false;
  }

  unsigned shift;
  llvm::SmallDenseSet<mlir::Value, 8> assumptions;
  llvm::SmallDenseSet<mlir::Value, 16> active;
};

bool isKnownMultipleOfPowerOfTwo(mlir::Value value, unsigned shift) {
  return PowerOfTwoMultipleProof(shift).prove(value);
}

bool isSupportedElementType(mlir::Type type) {
  if (llvm::isa<mlir::IntegerType, mlir::Float16Type, mlir::BFloat16Type,
                mlir::Float32Type, mlir::Float64Type, mlir::Float80Type,
                mlir::Float128Type>(type))
    return true;
  auto array = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(type);
  return array && array.getNumElements() != 0 &&
         isSupportedElementType(array.getElementType());
}

std::optional<std::uint64_t>
getMLIRAllocByteSize(const mlir::DataLayout &layout, mlir::Type type) {
  if (!isSupportedElementType(type))
    return std::nullopt;
  llvm::TypeSize bits = layout.getTypeSizeInBits(type);
  if (bits.isScalable() || bits.getFixedValue() == 0 ||
      bits.getFixedValue() % 8 != 0)
    return std::nullopt;
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
  if (!isSupportedElementType(type))
    return std::nullopt;
  llvm::TypeSize bytes =
      layout.getTypeAllocSize(translator.translateType(type));
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

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, mlir::Type accessType,
                           unsigned canonicalIndexBits) {
  if (canonicalIndexBits == 0 ||
      canonicalIndexBits > mlir::IntegerType::kMaxWidth)
    return std::nullopt;
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> leafToRoot;
  mlir::Value root = pointer;
  while (auto current = root.getDefiningOp<mlir::LLVM::GEPOp>()) {
    leafToRoot.push_back(current);
    root = current.getBase();
  }
  auto rootType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(root.getType());
  if (!rootType)
    return std::nullopt;

  // The chain is discharged only when all pointer uses lower to logical
  // memory accesses. On an LLVM-defined execution every no-wrap condition
  // already holds; violating one poisons the source address and its consuming
  // load or store has undefined behavior. The linear address is therefore the
  // exact defined-domain projection regardless of the chain's flag spelling.

  mlir::Operation *scope = pointer.getDefiningOp();
  if (!scope) {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer);
    scope = argument ? argument.getOwner()->getParentOp() : nullptr;
  }
  if (!scope)
    return std::nullopt;
  mlir::DataLayout dataLayout = mlir::DataLayout::closest(scope);
  std::optional<llvm::DataLayout> llvmDataLayout =
      getModuleLLVMDataLayout(scope);
  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator translator(llvmContext);

  std::optional<std::uint64_t> pointerIndexBits;
  if (llvmDataLayout)
    pointerIndexBits =
        llvmDataLayout->getIndexSizeInBits(rootType.getAddressSpace());
  else
    pointerIndexBits = dataLayout.getTypeIndexBitwidth(root.getType());
  if (!pointerIndexBits || *pointerIndexBits == 0 || *pointerIndexBits > 64)
    return std::nullopt;

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
  if (elementShift >= *pointerIndexBits)
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
          !llvm::isIntN(*pointerIndexBits,
                        static_cast<std::int64_t>(*strideBytes)) ||
          !llvm::isIntN(canonicalIndexBits,
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
        if (!indexBits || *indexBits > canonicalIndexBits)
          return std::nullopt;
        result.indexType =
            mlir::IntegerType::get(pointer.getContext(), canonicalIndexBits);
        if (llvm::isa<mlir::IndexType>(index.getType()) && *elementBytes != 1)
          return std::nullopt;
        unsigned strideShift =
            std::min<unsigned>(elementShift, llvm::countr_zero(*strideBytes));
        if (!isKnownMultipleOfPowerOfTwo(index, elementShift - strideShift))
          return std::nullopt;
        result.terms.push_back(
            {index, static_cast<std::int64_t>(*strideBytes)});
        result.elementTerms.push_back(
            {index, static_cast<std::int64_t>(*strideBytes >> strideShift),
             elementShift - strideShift});
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
  }

  if (constantByteOffset % static_cast<std::int64_t>(*elementBytes) != 0 ||
      !llvm::isIntN(*pointerIndexBits, constantByteOffset) ||
      !llvm::isIntN(canonicalIndexBits, constantByteOffset))
    return std::nullopt;

  if (!result.indexType)
    result.indexType =
        mlir::IntegerType::get(pointer.getContext(), canonicalIndexBits);
  result.byteBias = constantByteOffset;
  result.elementBias =
      constantByteOffset / static_cast<std::int64_t>(*elementBytes);
  for (mlir::LLVM::GEPOp gep : leafToRoot)
    result.gepsLeafToRoot.push_back(gep.getOperation());
  return result;
}

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, dataflow::GraphOp graph,
                           mlir::Type accessType, unsigned canonicalIndexBits) {
  auto resolved =
      resolveLinearMemoryAddress(pointer, accessType, canonicalIndexBits);
  if (!resolved || !isGraphPointerArgument(resolved->root, graph))
    return std::nullopt;
  return resolved;
}

} // namespace loom::lowering
