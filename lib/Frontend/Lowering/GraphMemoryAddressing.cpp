#include "Frontend/Lowering/GraphMemoryAddressing.h"

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

struct ResolvedPointerRoot {
  mlir::Value root;
  mlir::Operation *scope = nullptr;
  std::optional<llvm::DataLayout> llvmDataLayout;
  std::uint64_t indexBitWidth = 0;
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> gepsLeafToRoot;
};

std::optional<ResolvedPointerRoot> resolvePointerRoot(mlir::Value pointer) {
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> gepsLeafToRoot;
  mlir::Value root = pointer;
  while (auto gep = root.getDefiningOp<mlir::LLVM::GEPOp>()) {
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

mlir::Value resolveMemoryServiceBoundaryRootImpl(
    mlir::Value pointer, llvm::function_ref<bool(mlir::Value)> isBoundaryRoot,
    llvm::DenseSet<mlir::Value> &visiting) {
  if (!pointer || !visiting.insert(pointer).second)
    return {};
  if (isBoundaryRoot(pointer))
    return pointer;
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
      if (argument.getOwner() == loop.getBody() &&
          argument.getArgNumber() > 0) {
        unsigned ordinal = argument.getArgNumber() - 1;
        if (ordinal < loop.getInitArgs().size())
          return resolveMemoryServiceBoundaryRootImpl(
              loop.getInitArgs()[ordinal], isBoundaryRoot, visiting);
      }
    }
    if (auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(parent)) {
      unsigned ordinal = argument.getArgNumber();
      if (argument.getOwner() == loop.getBeforeBody() &&
          ordinal < loop.getInits().size())
        return resolveMemoryServiceBoundaryRootImpl(loop.getInits()[ordinal],
                                                    isBoundaryRoot, visiting);
      if (argument.getOwner() == loop.getAfterBody() &&
          ordinal < loop.getConditionOp().getArgs().size())
        return resolveMemoryServiceBoundaryRootImpl(
            loop.getConditionOp().getArgs()[ordinal], isBoundaryRoot, visiting);
    }
    return {};
  }
  if (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>())
    return resolveMemoryServiceBoundaryRootImpl(gep.getBase(), isBoundaryRoot,
                                                visiting);
  if (auto carry = pointer.getDefiningOp<dataflow::CarryOp>())
    return resolveMemoryServiceBoundaryRootImpl(carry.getInit(), isBoundaryRoot,
                                                visiting);
  if (auto invariant = pointer.getDefiningOp<dataflow::InvariantOp>())
    return resolveMemoryServiceBoundaryRootImpl(invariant.getInit(),
                                                isBoundaryRoot, visiting);
  if (auto gate = pointer.getDefiningOp<dataflow::GateOp>())
    return resolveMemoryServiceBoundaryRootImpl(gate.getBeforeValue(),
                                                isBoundaryRoot, visiting);
  if (auto sync = pointer.getDefiningOp<dataflow::SyncOp>()) {
    unsigned ordinal = llvm::cast<mlir::OpResult>(pointer).getResultNumber();
    if (ordinal < sync.getInputs().size())
      return resolveMemoryServiceBoundaryRootImpl(sync.getInputs()[ordinal],
                                                  isBoundaryRoot, visiting);
  }
  if (auto select = pointer.getDefiningOp<mlir::arith::SelectOp>()) {
    llvm::DenseSet<mlir::Value> truePath = visiting;
    llvm::DenseSet<mlir::Value> falsePath = visiting;
    mlir::Value trueRoot = resolveMemoryServiceBoundaryRootImpl(
        select.getTrueValue(), isBoundaryRoot, truePath);
    mlir::Value falseRoot = resolveMemoryServiceBoundaryRootImpl(
        select.getFalseValue(), isBoundaryRoot, falsePath);
    if (trueRoot && trueRoot == falseRoot)
      return trueRoot;
  }
  if (auto result = llvm::dyn_cast<mlir::OpResult>(pointer)) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(result.getOwner())) {
      unsigned ordinal = result.getResultNumber();
      if (ordinal >= loop.getInitArgs().size() ||
          ordinal >= loop.getYieldedValues().size())
        return {};
      llvm::DenseSet<mlir::Value> initialPath = visiting;
      llvm::DenseSet<mlir::Value> yieldedPath = visiting;
      mlir::Value initial = resolveMemoryServiceBoundaryRootImpl(
          loop.getInitArgs()[ordinal], isBoundaryRoot, initialPath);
      mlir::Value yielded = resolveMemoryServiceBoundaryRootImpl(
          loop.getYieldedValues()[ordinal], isBoundaryRoot, yieldedPath);
      if (initial && initial == yielded)
        return initial;
    }
    if (auto loop = llvm::dyn_cast<mlir::scf::WhileOp>(result.getOwner())) {
      unsigned ordinal = result.getResultNumber();
      if (ordinal >= loop.getInits().size() ||
          ordinal >= loop.getYieldOp().getNumOperands())
        return {};
      llvm::DenseSet<mlir::Value> initialPath = visiting;
      llvm::DenseSet<mlir::Value> yieldedPath = visiting;
      mlir::Value initial = resolveMemoryServiceBoundaryRootImpl(
          loop.getInits()[ordinal], isBoundaryRoot, initialPath);
      mlir::Value yielded = resolveMemoryServiceBoundaryRootImpl(
          loop.getYieldOp().getOperand(ordinal), isBoundaryRoot, yieldedPath);
      if (initial && initial == yielded)
        return initial;
    }
    if (auto branch = llvm::dyn_cast<mlir::scf::IfOp>(result.getOwner())) {
      unsigned ordinal = result.getResultNumber();
      auto thenYield = llvm::dyn_cast<mlir::scf::YieldOp>(
          branch.getThenRegion().front().getTerminator());
      auto elseYield =
          branch.getElseRegion().empty()
              ? mlir::scf::YieldOp{}
              : llvm::dyn_cast<mlir::scf::YieldOp>(
                    branch.getElseRegion().front().getTerminator());
      if (!thenYield || !elseYield || ordinal >= thenYield.getNumOperands() ||
          ordinal >= elseYield.getNumOperands())
        return {};
      llvm::DenseSet<mlir::Value> thenPath = visiting;
      llvm::DenseSet<mlir::Value> elsePath = visiting;
      mlir::Value thenRoot = resolveMemoryServiceBoundaryRootImpl(
          thenYield.getOperand(ordinal), isBoundaryRoot, thenPath);
      mlir::Value elseRoot = resolveMemoryServiceBoundaryRootImpl(
          elseYield.getOperand(ordinal), isBoundaryRoot, elsePath);
      if (thenRoot && thenRoot == elseRoot)
        return thenRoot;
    }
  }
  return {};
}

} // namespace

mlir::Value resolveMemoryServiceBoundaryRoot(
    mlir::Value pointer, llvm::function_ref<bool(mlir::Value)> isBoundaryRoot) {
  llvm::DenseSet<mlir::Value> visiting;
  return resolveMemoryServiceBoundaryRootImpl(pointer, isBoundaryRoot,
                                              visiting);
}

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
resolveLinearMemoryAddressImpl(mlir::Value pointer, mlir::Type accessType,
                               std::optional<unsigned> canonicalIndexBits) {
  auto pointerRoot = resolvePointerRoot(pointer);
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
  return resolveLinearMemoryAddressImpl(pointer, accessType,
                                        canonicalIndexBits);
}

std::optional<ResolvedLinearMemoryAddress>
resolveLinearPointerAddress(mlir::Value pointer, mlir::Type accessType) {
  return resolveLinearMemoryAddressImpl(pointer, accessType, std::nullopt);
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
