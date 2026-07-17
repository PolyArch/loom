// Tokenize residual `llvm.load` / `llvm.store` ops inside
// `dataflow.graph` bodies into the streaming-memory primitives
// `dataflow.load` / `dataflow.store`. The resulting ops bind the
// pointer to a memref via `unrealized_conversion_cast`, drive the
// address port from an `index`-typed offset, and accept the graph
// body's `thread_ctrl` block argument as the `none`-typed firing
// token.
//
// Two recognition modes:
//   * Top-level (load/store sits directly in the graph entry block):
//     require the chain to terminate at a graph block argument so
//     the rewrite preserves the brief's "block-arg base" contract.
//        - gep %arg_ptr[%idx]   -> (arg_ptr, idx)
//        - dataflow.carry init=arg_ptr  -> (carry_result, null)
//        - graph block-arg !llvm.ptr   -> (block_arg, null)
//
//   * Nested (load/store sits inside an scf.for / scf.if region of
//     the graph body): apply a permissive fallback that accepts any
//     !llvm.ptr SSA value as a per-iteration bridge target. The
//     bridge cast is emitted at the load/store site rather than
//     hoisted, so the per-iteration pointer is reinterpreted as a
//     memref<?xT> and read at offset 0 (or at the gep's idx).
//
// Loads whose pointer is none of the above (e.g. derived from
// `llvm.alloca` or `llvm.mlir.addressof` at the top level) remain untouched
// during recognition and are rejected by the residual gate. Raw LLVM memory
// operations do not provide the canonical dataflow completion contract.
// Pointer-element loads are also skipped: bridging !llvm.ptr-of-ptr to
// memref<?x!llvm.ptr> trips the streaming load verifier.

#include "Frontend/Lowering/Passes.h"

#include "GraphRegionLowering.h"
#include "StreamOrdinal.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace {

bool isLLVMMemsetOp(::mlir::Operation *op) {
  return op->getName().getStringRef() == "llvm.intr.memset";
}

bool isVolatileMemIntrinsic(::mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<::mlir::BoolAttr>("isVolatile"))
    return attr.getValue();
  return false;
}

bool isGraphPtrBlockArg(::mlir::Value v, ::dataflow::GraphOp graph) {
  auto blockArg = ::llvm::dyn_cast<::mlir::BlockArgument>(v);
  if (!blockArg || blockArg.getOwner() != &graph.getBody().front())
    return false;
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(blockArg.getType());
}

std::optional<::llvm::DataLayout>
getModuleLLVMDataLayout(::mlir::Operation *scope) {
  auto module = scope->getParentOfType<::mlir::ModuleOp>();
  if (!module)
    return std::nullopt;
  auto layoutAttr = module->getAttrOfType<::mlir::StringAttr>(
      ::mlir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (!layoutAttr)
    return std::nullopt;
  return ::llvm::DataLayout(layoutAttr.getValue());
}

// The distinguished leading `none` block argument is the graph start firing
// token; it is separate from the payload-only FunctionType.
::mlir::Value getThreadCtrl(::dataflow::GraphOp graph) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0)
    return {};
  ::mlir::Value first = entry.getArgument(0);
  return ::llvm::isa<::mlir::NoneType>(first.getType()) ? first
                                                        : ::mlir::Value{};
}

struct BridgeKey {
  ::mlir::Value ptr;
  ::mlir::Type elem;
  bool operator==(const BridgeKey &o) const {
    return ptr == o.ptr && elem == o.elem;
  }
};

struct BridgeKeyInfo {
  static BridgeKey getEmptyKey() { return {{}, {}}; }
  static BridgeKey getTombstoneKey() {
    return {::mlir::Value::getFromOpaquePointer((void *)1),
            ::mlir::Type::getFromOpaquePointer((void *)1)};
  }
  static unsigned getHashValue(const BridgeKey &k) {
    return ::llvm::hash_combine(::mlir::hash_value(k.ptr),
                                ::mlir::hash_value(k.elem));
  }
  static bool isEqual(const BridgeKey &a, const BridgeKey &b) { return a == b; }
};

using ScopedValueCache =
    ::llvm::DenseMap<::mlir::Block *,
                     ::llvm::DenseMap<::mlir::Value, ::mlir::Value>>;
using ScopedBridgeCache =
    ::llvm::DenseMap<::mlir::Block *,
                     ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo>>;

// Result of resolving the (memref-base, optional-int-index) pair for
// one llvm.load / llvm.store. `intIndex` is null when the address
// port should be `0 : index`.
struct AddrResolution {
  ::mlir::Value ptr;
  ::mlir::Value intIndex;
  ::dataflow::StreamOp ordinalStream;
  std::int64_t ordinalElementBias = 0;
  unsigned byteToElementShift = 0;
  ::mlir::Operation *gepToErase = nullptr;
  ::mlir::Operation *baseGepToErase = nullptr;
  std::int64_t intIndexByteStride = 1;
  std::int64_t intIndexByteBias = 0;
};

struct MemcpyPtrResolution {
  ::mlir::Value basePtr;
  ::mlir::Value chunkStride;
  ::dataflow::StreamOp outerStream;
};

struct DirectMemcpyPtrResolution {
  ::mlir::Value basePtr;
  ::mlir::Value offset;
  ::mlir::Operation *gepToErase = nullptr;
};

unsigned getElementByteWidth(::mlir::Type elemTy);

std::optional<AddrResolution>
resolveNestedMemcpyStaticPointer(::mlir::Value ptr, ::dataflow::GraphOp graph) {
  if (isGraphPtrBlockArg(ptr, graph))
    return AddrResolution{ptr, {}, {}, 0, 0, nullptr};

  auto gep = ptr.getDefiningOp<::mlir::LLVM::GEPOp>();
  if (!gep || !isGraphPtrBlockArg(gep.getBase(), graph))
    return std::nullopt;
  auto rawIndices = gep.getRawConstantIndices();
  auto dynamicIndices = gep.getDynamicIndices();
  if (rawIndices.size() != 1 || dynamicIndices.size() != 1 ||
      rawIndices.front() != ::mlir::LLVM::GEPOp::kDynamicIndex)
    return std::nullopt;
  ::mlir::Value dynamicIndex = dynamicIndices.front();
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(
          dynamicIndex.getType()))
    return std::nullopt;
  unsigned byteStride = getElementByteWidth(gep.getElemType());
  if (byteStride == 0)
    return std::nullopt;

  AddrResolution resolution{gep.getBase(),     dynamicIndex, {}, 0, 0,
                            gep.getOperation()};
  resolution.intIndexByteStride = byteStride;
  return resolution;
}

bool pointerUsesBase(::mlir::Value ptr, ::mlir::Value base) {
  if (ptr == base)
    return true;
  auto gep = ptr.getDefiningOp<::mlir::LLVM::GEPOp>();
  return gep && gep.getBase() == base;
}

std::optional<::mlir::Type>
inferMemsetElementType(::dataflow::GraphOp graph, ::mlir::Value basePtr,
                       ::mlir::Operation *memsetOp) {
  ::mlir::Type inferred;
  bool conflict = false;
  graph.walk([&](::mlir::Operation *op) -> ::mlir::WalkResult {
    if (op == memsetOp)
      return ::mlir::WalkResult::advance();
    ::mlir::Value ptr;
    ::mlir::Type elemTy;
    if (auto load = ::llvm::dyn_cast<::mlir::LLVM::LoadOp>(op)) {
      ptr = load.getAddr();
      elemTy = load.getResult().getType();
    } else if (auto store = ::llvm::dyn_cast<::mlir::LLVM::StoreOp>(op)) {
      ptr = store.getAddr();
      elemTy = store.getValue().getType();
    } else {
      return ::mlir::WalkResult::advance();
    }
    if (::llvm::isa<::mlir::LLVM::LLVMPointerType>(elemTy))
      return ::mlir::WalkResult::advance();
    if (!pointerUsesBase(ptr, basePtr))
      return ::mlir::WalkResult::advance();
    if (!inferred) {
      inferred = elemTy;
      return ::mlir::WalkResult::advance();
    }
    if (inferred != elemTy) {
      conflict = true;
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  if (conflict)
    return std::nullopt;
  if (inferred)
    return inferred;
  return ::mlir::IntegerType::get(graph.getContext(), 8);
}

unsigned getElementByteWidth(::mlir::Type elemTy) {
  unsigned bitWidth = 0;
  if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(elemTy))
    bitWidth = intTy.getWidth();
  else if (auto floatTy = ::llvm::dyn_cast<::mlir::FloatType>(elemTy))
    bitWidth = floatTy.getWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return 0;
  return bitWidth / 8;
}

bool isPowerOfTwo(unsigned value) {
  return value != 0 && (value & (value - 1)) == 0;
}

unsigned log2PowerOfTwo(unsigned value) {
  unsigned shift = 0;
  while (value > 1) {
    value >>= 1;
    ++shift;
  }
  return shift;
}

std::optional<llvm::APInt> integerConstantValue(::mlir::Value value) {
  auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>();
  if (constant) {
    auto intAttr = ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getValue());
    if (intAttr)
      return intAttr.getValue();
  }
  if (auto dataflowConst = value.getDefiningOp<::dataflow::ConstantOp>()) {
    auto intAttr =
        ::llvm::dyn_cast<::mlir::IntegerAttr>(dataflowConst.getConstValue());
    if (intAttr)
      return intAttr.getValue();
  }
  return std::nullopt;
}

bool isKnownMultipleOfPowerOfTwo(::mlir::Value value, unsigned shift) {
  if (shift == 0)
    return true;
  if (std::optional<llvm::APInt> constant = integerConstantValue(value))
    return constant->countTrailingZeros() >= shift;
  auto shl = value.getDefiningOp<::mlir::arith::ShLIOp>();
  if (!shl)
    return false;
  std::optional<llvm::APInt> amount = integerConstantValue(shl.getRhs());
  auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(value.getType());
  return amount && intTy && amount->ult(intTy.getWidth()) &&
         amount->getZExtValue() >= shift;
}

bool isZeroIntegerValue(::mlir::Value value) {
  std::optional<llvm::APInt> constant = integerConstantValue(value);
  return constant && constant->isZero();
}

unsigned getByteToElementShift(::mlir::LLVM::GEPOp gep, ::mlir::Type elemTy) {
  auto gepElemTy = ::llvm::dyn_cast<::mlir::IntegerType>(gep.getElemType());
  if (!gepElemTy || gepElemTy.getWidth() != 8)
    return 0;
  unsigned byteWidth = getElementByteWidth(elemTy);
  if (byteWidth <= 1 || !isPowerOfTwo(byteWidth))
    return 0;
  return log2PowerOfTwo(byteWidth);
}

std::optional<std::int64_t> getSingleIndexElementStride(::mlir::LLVM::GEPOp gep,
                                                        ::mlir::Type elemTy) {
  if (!gep.getDynamicIndices().empty())
    return std::nullopt;
  auto rawIndices = gep.getRawConstantIndices();
  if (rawIndices.size() != 1)
    return std::nullopt;
  unsigned gepByteWidth = getElementByteWidth(gep.getElemType());
  unsigned elementByteWidth = getElementByteWidth(elemTy);
  if (gepByteWidth == 0 || elementByteWidth == 0)
    return std::nullopt;
  std::int64_t byteOffset =
      static_cast<std::int64_t>(rawIndices.front()) * gepByteWidth;
  if (byteOffset % static_cast<std::int64_t>(elementByteWidth) != 0)
    return std::nullopt;
  return byteOffset / static_cast<std::int64_t>(elementByteWidth);
}

bool isSupportedLinearGepElementType(::mlir::Type type) {
  if (::llvm::isa<::mlir::IntegerType, ::mlir::Float16Type,
                  ::mlir::BFloat16Type, ::mlir::Float32Type,
                  ::mlir::Float64Type, ::mlir::Float80Type,
                  ::mlir::Float128Type>(type))
    return true;
  auto arrayTy = ::llvm::dyn_cast<::mlir::LLVM::LLVMArrayType>(type);
  return arrayTy && arrayTy.getNumElements() != 0 &&
         isSupportedLinearGepElementType(arrayTy.getElementType());
}

std::optional<std::uint64_t>
getMLIRFixedAllocByteSize(const ::mlir::DataLayout &dataLayout,
                          ::mlir::Type type) {
  if (!isSupportedLinearGepElementType(type))
    return std::nullopt;
  ::llvm::TypeSize bitSize = dataLayout.getTypeSizeInBits(type);
  if (bitSize.isScalable())
    return std::nullopt;
  std::uint64_t fixedBits = bitSize.getFixedValue();
  if (fixedBits == 0 || fixedBits % 8 != 0)
    return std::nullopt;
  ::llvm::TypeSize byteSize = dataLayout.getTypeSize(type);
  if (byteSize.isScalable() || byteSize.getFixedValue() == 0)
    return std::nullopt;
  std::uint64_t alignment = dataLayout.getTypeABIAlignment(type);
  if (alignment == 0 ||
      byteSize.getFixedValue() >
          std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
    return std::nullopt;
  return ::llvm::alignTo(byteSize.getFixedValue(), alignment);
}

std::optional<std::uint64_t>
getLLVMFixedAllocByteSize(const ::llvm::DataLayout &dataLayout,
                          ::mlir::LLVM::TypeToLLVMIRTranslator &typeTranslator,
                          ::mlir::Type type) {
  if (!isSupportedLinearGepElementType(type))
    return std::nullopt;
  ::llvm::TypeSize byteSize =
      dataLayout.getTypeAllocSize(typeTranslator.translateType(type));
  if (byteSize.isScalable() || byteSize.getFixedValue() == 0)
    return std::nullopt;
  return byteSize.getFixedValue();
}

std::optional<unsigned>
getIntegralIndexBitWidth(const ::mlir::DataLayout &dataLayout,
                         ::mlir::Type type) {
  if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
    if (!intTy.isSignless())
      return std::nullopt;
    return intTy.getWidth();
  }
  if (!::llvm::isa<::mlir::IndexType>(type))
    return std::nullopt;
  ::llvm::TypeSize bitSize = dataLayout.getTypeSizeInBits(type);
  if (bitSize.isScalable() || bitSize.getFixedValue() == 0)
    return std::nullopt;
  return static_cast<unsigned>(
      std::min<std::uint64_t>(bitSize.getFixedValue(), 64));
}

std::optional<AddrResolution>
resolveLinearTopLevelGep(::mlir::LLVM::GEPOp gep, ::dataflow::GraphOp graph,
                         ::mlir::Type elemTy) {
  ::mlir::LLVM::GEPOp dynamicGep = gep;
  ::mlir::Operation *baseGepToErase = nullptr;
  std::optional<std::int64_t> companionIndex;
  auto rawIndices = gep.getRawConstantIndices();
  auto dynamicIndices = gep.getDynamicIndices();
  if (rawIndices.size() == 1 && dynamicIndices.size() == 1 &&
      rawIndices.front() == ::mlir::LLVM::GEPOp::kDynamicIndex) {
    // The access is formed directly by the dynamic GEP.
  } else if (rawIndices.size() == 1 && dynamicIndices.empty() &&
             rawIndices.front() != ::mlir::LLVM::GEPOp::kDynamicIndex) {
    dynamicGep = gep.getBase().getDefiningOp<::mlir::LLVM::GEPOp>();
    if (!dynamicGep)
      return std::nullopt;
    baseGepToErase = dynamicGep.getOperation();
    companionIndex = rawIndices.front();
    rawIndices = dynamicGep.getRawConstantIndices();
    dynamicIndices = dynamicGep.getDynamicIndices();
  } else {
    return std::nullopt;
  }

  if (!isGraphPtrBlockArg(dynamicGep.getBase(), graph) ||
      rawIndices.size() != 1 || dynamicIndices.size() != 1 ||
      rawIndices.front() != ::mlir::LLVM::GEPOp::kDynamicIndex)
    return std::nullopt;

  ::mlir::DataLayout dataLayout =
      ::mlir::DataLayout::closest(gep.getOperation());
  std::optional<::llvm::DataLayout> llvmDataLayout =
      getModuleLLVMDataLayout(gep.getOperation());
  ::llvm::LLVMContext llvmContext;
  ::mlir::LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
  ::mlir::Value dynamicIndex = dynamicIndices.front();
  std::optional<unsigned> indexBitWidth;
  std::optional<std::uint64_t> pointerIndexBitWidth;
  if (llvmDataLayout) {
    auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(dynamicIndex.getType());
    if (!intTy || !intTy.isSignless())
      return std::nullopt;
    indexBitWidth = intTy.getWidth();
    auto ptrTy = ::llvm::dyn_cast<::mlir::LLVM::LLVMPointerType>(
        dynamicGep.getBase().getType());
    if (!ptrTy)
      return std::nullopt;
    pointerIndexBitWidth =
        llvmDataLayout->getIndexSizeInBits(ptrTy.getAddressSpace());
  } else {
    indexBitWidth =
        getIntegralIndexBitWidth(dataLayout, dynamicIndex.getType());
    pointerIndexBitWidth =
        dataLayout.getTypeIndexBitwidth(dynamicGep.getBase().getType());
  }
  if (!indexBitWidth || !pointerIndexBitWidth ||
      *pointerIndexBitWidth != *indexBitWidth)
    return std::nullopt;

  auto getAllocBytes = [&](::mlir::Type type) {
    if (llvmDataLayout)
      return getLLVMFixedAllocByteSize(*llvmDataLayout, typeTranslator, type);
    return getMLIRFixedAllocByteSize(dataLayout, type);
  };
  std::optional<std::uint64_t> elementBytes = getAllocBytes(elemTy);
  std::optional<std::uint64_t> baseStrideBytes =
      getAllocBytes(dynamicGep.getElemType());
  constexpr std::uint64_t maxSigned =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  unsigned elementShift = elementBytes ? ::llvm::Log2_64(*elementBytes) : 0;
  if (!elementBytes || !baseStrideBytes || *elementBytes > maxSigned ||
      *baseStrideBytes > maxSigned || !::llvm::isPowerOf2_64(*elementBytes) ||
      elementShift >= *indexBitWidth ||
      (::llvm::isa<::mlir::IndexType>(dynamicIndex.getType()) &&
       *elementBytes != 1))
    return std::nullopt;
  unsigned strideShift =
      std::min<unsigned>(elementShift, ::llvm::countr_zero(*baseStrideBytes));
  if (!isKnownMultipleOfPowerOfTwo(dynamicIndex, elementShift - strideShift))
    return std::nullopt;

  std::int64_t constantByteOffset = 0;
  if (companionIndex) {
    std::optional<std::uint64_t> companionStrideBytes =
        getAllocBytes(gep.getElemType());
    if (!companionStrideBytes || *companionStrideBytes > maxSigned ||
        ::llvm::MulOverflow(*companionIndex,
                            static_cast<std::int64_t>(*companionStrideBytes),
                            constantByteOffset))
      return std::nullopt;
  }
  if (constantByteOffset % static_cast<std::int64_t>(*elementBytes) != 0 ||
      !::llvm::isIntN(*indexBitWidth,
                      static_cast<std::int64_t>(*baseStrideBytes)) ||
      !::llvm::isIntN(*indexBitWidth, constantByteOffset))
    return std::nullopt;

  bool preserveElementIndex =
      !companionIndex && constantByteOffset == 0 &&
      *baseStrideBytes == *elementBytes &&
      ::mlir::LLVM::bitEnumContainsAny(dynamicGep.getNoWrapFlags(),
                                       ::mlir::LLVM::GEPNoWrapFlags::nusw);
  unsigned byteToElementShift = preserveElementIndex ? 0 : elementShift;
  std::int64_t intIndexByteStride =
      preserveElementIndex ? 1 : static_cast<std::int64_t>(*baseStrideBytes);

  AddrResolution resolution{dynamicGep.getBase(), dynamicIndex,      {}, 0,
                            byteToElementShift,   gep.getOperation()};
  resolution.baseGepToErase = baseGepToErase;
  resolution.intIndexByteStride = intIndexByteStride;
  resolution.intIndexByteBias = constantByteOffset;
  return resolution;
}

::dataflow::StreamOp getUnitStridePointerCarryStream(::dataflow::CarryOp carry,
                                                     ::dataflow::GraphOp graph,
                                                     ::mlir::Type elemTy) {
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(carry.getOutput().getType()))
    return {};
  if (!isGraphPtrBlockArg(carry.getInit(), graph))
    return {};
  auto gep = carry.getCarry().getDefiningOp<::mlir::LLVM::GEPOp>();
  if (!gep)
    return {};
  if (getModuleLLVMDataLayout(gep))
    return {};
  ::mlir::Value gepBase = gep.getBase();
  if (gepBase != carry.getOutput()) {
    auto gate = gepBase.getDefiningOp<::dataflow::GateOp>();
    if (!gate || gate.getAfterValue() != gepBase ||
        gate.getBeforeValue() != carry.getOutput() ||
        gate.getBeforeCond() != carry.getCond())
      return {};
  }
  std::optional<std::int64_t> elementStride =
      getSingleIndexElementStride(gep, elemTy);
  if (!elementStride || *elementStride != 1)
    return {};
  auto stream = carry.getCond().getDefiningOp<::dataflow::StreamOp>();
  if (!stream || stream.getPhase() != carry.getCond() ||
      !::loom::lowering::isZeroBasedUnitOrdinalStream(stream, gep))
    return {};
  return stream;
}

std::optional<::dataflow::CarryOp> getGatedPointerCarry(::mlir::Value value) {
  auto gate = value.getDefiningOp<::dataflow::GateOp>();
  if (!gate || gate.getAfterValue() != value)
    return std::nullopt;
  auto carry = gate.getBeforeValue().getDefiningOp<::dataflow::CarryOp>();
  if (!carry || gate.getBeforeCond() != carry.getCond() ||
      !::llvm::isa<::mlir::LLVM::LLVMPointerType>(carry.getOutput().getType()))
    return std::nullopt;
  return carry;
}

std::optional<::dataflow::CarryOp> getPointerCarry(::mlir::Value value) {
  if (auto carry = value.getDefiningOp<::dataflow::CarryOp>()) {
    if (carry.getOutput() == value &&
        ::llvm::isa<::mlir::LLVM::LLVMPointerType>(carry.getOutput().getType()))
      return carry;
  }
  return getGatedPointerCarry(value);
}

std::optional<::mlir::Value> getSingleDynamicIndex(::mlir::LLVM::GEPOp gep) {
  if (getModuleLLVMDataLayout(gep))
    return std::nullopt;
  auto rawIndices = gep.getRawConstantIndices();
  auto dynIndices = gep.getDynamicIndices();
  if (rawIndices.size() != 1 || dynIndices.size() != 1 ||
      rawIndices.front() != ::mlir::LLVM::GEPOp::kDynamicIndex)
    return std::nullopt;
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(
          dynIndices.front().getType()))
    return std::nullopt;
  auto gepElemTy = ::llvm::dyn_cast<::mlir::IntegerType>(gep.getElemType());
  if (!gepElemTy || gepElemTy.getWidth() != 8)
    return std::nullopt;
  return dynIndices.front();
}

std::optional<MemcpyPtrResolution>
resolveMemcpyPointer(::mlir::Value ptr, ::dataflow::GraphOp graph) {
  auto carry = getPointerCarry(ptr);
  if (!carry || !isGraphPtrBlockArg(carry->getInit(), graph))
    return std::nullopt;

  auto gep = carry->getCarry().getDefiningOp<::mlir::LLVM::GEPOp>();
  if (!gep)
    return std::nullopt;

  ::mlir::Value gepBase = gep.getBase();
  if (gepBase != ptr) {
    auto gate = gepBase.getDefiningOp<::dataflow::GateOp>();
    if (!gate || gate.getAfterValue() != gepBase ||
        gate.getBeforeValue() != carry->getOutput() ||
        gate.getBeforeCond() != carry->getCond())
      return std::nullopt;
  }

  std::optional<::mlir::Value> stride = getSingleDynamicIndex(gep);
  if (!stride)
    return std::nullopt;

  auto stream = carry->getCond().getDefiningOp<::dataflow::StreamOp>();
  if (!stream || stream.getPhase() != carry->getCond() ||
      stream.getStepKind() != ::dataflow::StreamStepKind::Add ||
      stream.getPredicate() != ::mlir::arith::CmpIPredicate::slt)
    return std::nullopt;

  return MemcpyPtrResolution{carry->getInit(), *stride, stream};
}

std::optional<DirectMemcpyPtrResolution>
resolveDirectMemcpyPointer(::mlir::Value ptr, ::dataflow::GraphOp graph) {
  if (isGraphPtrBlockArg(ptr, graph))
    return DirectMemcpyPtrResolution{ptr, {}, nullptr};

  auto gep = ptr.getDefiningOp<::mlir::LLVM::GEPOp>();
  if (!gep)
    return std::nullopt;
  if (!isGraphPtrBlockArg(gep.getBase(), graph))
    return std::nullopt;
  std::optional<::mlir::Value> offset = getSingleDynamicIndex(gep);
  if (!offset)
    return std::nullopt;
  return DirectMemcpyPtrResolution{gep.getBase(), *offset, gep.getOperation()};
}

std::optional<AddrResolution> resolvePointer(::mlir::Value loadStorePtr,
                                             ::dataflow::GraphOp graph,
                                             bool topLevel,
                                             ::mlir::Type elemTy) {
  if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
    if (std::optional<AddrResolution> linear =
            resolveLinearTopLevelGep(gep, graph, elemTy))
      return linear;
  }
  if (topLevel) {
    if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
      ::mlir::Value base = gep.getBase();
      if (auto carry = getPointerCarry(base)) {
        if (::dataflow::StreamOp stream =
                getUnitStridePointerCarryStream(*carry, graph, elemTy)) {
          std::optional<std::int64_t> bias =
              getSingleIndexElementStride(gep, elemTy);
          if (!bias)
            return std::nullopt;
          return AddrResolution{carry->getInit(), {}, stream, *bias, 0,
                                nullptr};
        }
      }
      return std::nullopt;
    }
    if (auto carry = loadStorePtr.getDefiningOp<::dataflow::CarryOp>()) {
      if (::dataflow::StreamOp stream =
              getUnitStridePointerCarryStream(carry, graph, elemTy))
        return AddrResolution{carry.getInit(), {}, stream, 0, 0, nullptr};
      return std::nullopt;
    }
    if (auto carry = getGatedPointerCarry(loadStorePtr)) {
      if (::dataflow::StreamOp stream =
              getUnitStridePointerCarryStream(*carry, graph, elemTy))
        return AddrResolution{carry->getInit(), {}, stream, 0, 0, nullptr};
      return std::nullopt;
    }
    if (isGraphPtrBlockArg(loadStorePtr, graph))
      return AddrResolution{loadStorePtr, {}, {}, 0, 0, nullptr};
    return std::nullopt;
  }
  // Nested permissive fallback.
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(loadStorePtr.getType()))
    return std::nullopt;
  if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
    if (getModuleLLVMDataLayout(gep))
      return std::nullopt;
    auto dynIdxs = gep.getDynamicIndices();
    if (dynIdxs.size() == 1) {
      ::mlir::Value idx = dynIdxs.front();
      unsigned byteToElementShift = getByteToElementShift(gep, elemTy);
      if (byteToElementShift == 0 &&
          ::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(idx.getType()))
        return AddrResolution{gep.getBase(), idx, {}, 0, 0, gep.getOperation()};
    }
  }
  return AddrResolution{loadStorePtr, {}, {}, 0, 0, nullptr};
}

// Materialize (or look up) an unrealized_conversion_cast bridging
// `ptr : !llvm.ptr` to `memref<?xElem>`. Graph-root pointers are hoisted and
// cached at graph entry; genuinely dynamic pointers are bridged at the access.
bool isSupportedBridgePointer(::mlir::Value ptr, ::dataflow::GraphOp graph) {
  auto ptrTy = ::llvm::dyn_cast<::mlir::LLVM::LLVMPointerType>(ptr.getType());
  if (!ptrTy)
    return false;
  unsigned addressSpace = ptrTy.getAddressSpace();
  std::optional<::llvm::DataLayout> llvmDataLayout =
      getModuleLLVMDataLayout(graph);
  if (llvmDataLayout && llvmDataLayout->isNonIntegralAddressSpace(addressSpace))
    return false;
  // The bridge does not preserve LLVM address spaces in the memref type.
  return addressSpace == 0;
}

::mlir::Value getMemrefBridge(
    ::mlir::OpBuilder &builder, ::dataflow::GraphOp graph,
    ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo> &cache,
    ::mlir::Value ptr, ::mlir::Type elem, ::mlir::Location loc, bool topLevel,
    ::mlir::Operation *insertBeforeIfNested) {
  if (!isSupportedBridgePointer(ptr, graph))
    return {};
  if (topLevel) {
    BridgeKey key{ptr, elem};
    if (auto it = cache.find(key); it != cache.end())
      return it->second;
  }
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (topLevel)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  auto memrefTy = ::mlir::MemRefType::get({::mlir::ShapedType::kDynamic}, elem);
  ::mlir::Value bridge =
      ::mlir::UnrealizedConversionCastOp::create(
          builder, loc, ::mlir::TypeRange{memrefTy}, ::mlir::ValueRange{ptr})
          .getResult(0);
  if (topLevel)
    cache.try_emplace(BridgeKey{ptr, elem}, bridge);
  return bridge;
}

bool canMaterializeIndex(::mlir::Type type, ::mlir::Operation *scope) {
  if (::llvm::isa<::mlir::IndexType>(type))
    return true;
  auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type);
  if (!intTy || !intTy.isSignless())
    return false;
  ::mlir::DataLayout dataLayout = ::mlir::DataLayout::closest(scope);
  ::llvm::TypeSize indexBits =
      dataLayout.getTypeSizeInBits(::mlir::IndexType::get(type.getContext()));
  return !indexBits.isScalable() &&
         intTy.getWidth() <= indexBits.getFixedValue();
}

::mlir::Value getIndexCast(::mlir::OpBuilder &builder,
                           ::dataflow::GraphOp graph, ScopedValueCache &cache,
                           ::mlir::Value iv, ::mlir::Location loc,
                           bool topLevel,
                           ::mlir::Operation *insertBeforeIfNested) {
  if (::llvm::isa<::mlir::IndexType>(iv.getType()))
    return iv;
  if (!canMaterializeIndex(iv.getType(), insertBeforeIfNested))
    return {};
  auto &blockCache =
      cache.try_emplace(insertBeforeIfNested->getBlock()).first->second;
  if (auto it = blockCache.find(iv); it != blockCache.end())
    return it->second;
  bool hoist = topLevel && ::llvm::isa<::mlir::BlockArgument>(iv);
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (hoist)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  ::mlir::Value out = ::mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), iv)
                          .getResult();
  blockCache.try_emplace(iv, out);
  return out;
}

::mlir::TypedAttr getIntegerLikeAttr(::mlir::OpBuilder &builder,
                                     ::mlir::Type type, std::int64_t value) {
  if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type))
    return builder.getIntegerAttr(intTy, value);
  if (::llvm::isa<::mlir::IndexType>(type))
    return builder.getIndexAttr(value);
  return {};
}

::mlir::Value getLinearByteIndex(::mlir::OpBuilder &builder,
                                 ::mlir::Value index, std::int64_t byteStride,
                                 std::int64_t byteBias, ::mlir::Location loc,
                                 ::mlir::Operation *insertBefore) {
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(insertBefore);
  ::mlir::Value result = index;
  if (byteStride != 1) {
    ::mlir::TypedAttr strideAttr =
        getIntegerLikeAttr(builder, index.getType(), byteStride);
    if (!strideAttr)
      return {};
    ::mlir::Value stride = ::mlir::arith::ConstantOp::create(
                               builder, loc, index.getType(), strideAttr)
                               .getResult();
    result =
        ::mlir::arith::MulIOp::create(builder, loc, result, stride).getResult();
  }
  if (byteBias != 0) {
    ::mlir::TypedAttr biasAttr =
        getIntegerLikeAttr(builder, index.getType(), byteBias);
    if (!biasAttr)
      return {};
    ::mlir::Value bias = ::mlir::arith::ConstantOp::create(
                             builder, loc, index.getType(), biasAttr)
                             .getResult();
    result =
        ::mlir::arith::AddIOp::create(builder, loc, result, bias).getResult();
  }
  return result;
}

::mlir::Value getDataflowConstant(::mlir::OpBuilder &builder, ::mlir::Type type,
                                  ::mlir::Value ctrl, std::int64_t value,
                                  ::mlir::Location loc) {
  ::mlir::TypedAttr attr = getIntegerLikeAttr(builder, type, value);
  if (!attr)
    return {};
  return ::dataflow::ConstantOp::create(builder, loc, type, ctrl, attr)
      .getValue();
}

::mlir::Value getMemsetElementCount(::mlir::OpBuilder &builder,
                                    ::mlir::Value byteCount,
                                    unsigned elementByteWidth,
                                    ::mlir::Value ctrl, ::mlir::Location loc) {
  if (elementByteWidth == 1)
    return byteCount;
  if (!isPowerOfTwo(elementByteWidth))
    return {};
  unsigned shiftAmount = log2PowerOfTwo(elementByteWidth);
  if (auto shl = byteCount.getDefiningOp<::mlir::arith::ShLIOp>()) {
    std::optional<llvm::APInt> shift = integerConstantValue(shl.getRhs());
    if (shift && shift->getLimitedValue() == shiftAmount)
      return shl.getLhs();
  }
  auto countTy = ::llvm::dyn_cast<::mlir::IntegerType>(byteCount.getType());
  if (!countTy)
    return {};
  ::mlir::Value shift =
      getDataflowConstant(builder, countTy, ctrl, shiftAmount, loc);
  if (!shift)
    return {};
  return ::mlir::arith::ShRUIOp::create(builder, loc, byteCount, shift)
      .getResult();
}

::mlir::Value projectStreamValue(::mlir::OpBuilder &builder,
                                 ::dataflow::StreamOp stream,
                                 ::mlir::Value value, ::mlir::Location loc) {
  return ::dataflow::GateOp::create(builder, loc, builder.getI1Type(),
                                    value.getType(), stream.getPhase(), value)
      .getAfterValue();
}

::mlir::Value getBiasedStreamOrdinal(::mlir::OpBuilder &builder,
                                     ::dataflow::StreamOp stream,
                                     ::mlir::Value ordinal, std::int64_t bias,
                                     ::mlir::Value ctrl, ::mlir::Location loc,
                                     ::mlir::Operation *insertBefore) {
  if (bias == 0)
    return ordinal;

  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(insertBefore);
  ::mlir::TypedAttr biasAttr =
      getIntegerLikeAttr(builder, ordinal.getType(), bias);
  if (!biasAttr)
    return {};
  ::mlir::Value biasValue = ::dataflow::ConstantOp::create(
                                builder, loc, ordinal.getType(), ctrl, biasAttr)
                                .getValue();
  ::mlir::Value stableBiasRaw =
      ::dataflow::InvariantOp::create(builder, loc, ordinal.getType(),
                                      stream.getPhase(), biasValue)
          .getOutput();
  ::mlir::Value stableBias =
      projectStreamValue(builder, stream, stableBiasRaw, loc);
  return ::mlir::arith::AddIOp::create(builder, loc, ordinal, stableBias)
      .getResult();
}

::mlir::Value getElementIndex(::mlir::OpBuilder &builder,
                              ::dataflow::GraphOp graph,
                              ScopedValueCache &cache, ::mlir::Value intIndex,
                              unsigned byteToElementShift, ::mlir::Value ctrl,
                              ::mlir::Location loc, bool topLevel,
                              ::mlir::Operation *insertBeforeIfNested) {
  if (byteToElementShift == 0)
    return getIndexCast(builder, graph, cache, intIndex, loc, topLevel,
                        insertBeforeIfNested);
  if (!canMaterializeIndex(intIndex.getType(), insertBeforeIfNested))
    return {};
  auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(intIndex.getType());
  if (!intTy)
    return {};

  ::mlir::OpBuilder::InsertionGuard g(builder);
  bool hoist = topLevel && ::llvm::isa<::mlir::BlockArgument>(intIndex);
  if (hoist)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  auto shiftAttr = builder.getIntegerAttr(intTy, byteToElementShift);
  ::mlir::Value shiftAmount =
      ::dataflow::ConstantOp::create(builder, loc, intTy, ctrl, shiftAttr)
          .getValue();
  ::mlir::Value elemIndex =
      ::mlir::arith::ShRSIOp::create(builder, loc, intIndex, shiftAmount)
          .getResult();
  return getIndexCast(builder, graph, cache, elemIndex, loc, topLevel,
                      insertBeforeIfNested);
}

::mlir::Value getZeroIndex(::mlir::OpBuilder &builder,
                           ::dataflow::GraphOp graph, ::mlir::Value &cached,
                           ::mlir::Location loc, bool topLevel,
                           ::mlir::Operation *insertBeforeIfNested) {
  if (topLevel && cached)
    return cached;
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (topLevel)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  ::mlir::Value c0 =
      ::mlir::arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                        builder.getIndexAttr(0))
          .getResult();
  if (topLevel)
    cached = c0;
  return c0;
}

::mlir::Value getIndexFromInt(::mlir::OpBuilder &builder,
                              ScopedValueCache &cache, ::mlir::Value value,
                              ::mlir::Location loc,
                              ::mlir::Operation *insertBefore,
                              bool cacheResult = true) {
  if (::llvm::isa<::mlir::IndexType>(value.getType()))
    return value;
  if (!canMaterializeIndex(value.getType(), insertBefore))
    return {};
  auto &blockCache = cache.try_emplace(insertBefore->getBlock()).first->second;
  if (cacheResult) {
    if (auto it = blockCache.find(value); it != blockCache.end())
      return it->second;
  }
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(insertBefore);
  ::mlir::Value idx = ::mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), value)
                          .getResult();
  if (cacheResult)
    blockCache.try_emplace(value, idx);
  return idx;
}

::mlir::Value makeChunkedAddress(::mlir::OpBuilder &builder,
                                 ::mlir::Value chunk, ::mlir::Value stride,
                                 ::mlir::Value offset, ::mlir::Location loc) {
  ::mlir::Value base =
      ::mlir::arith::MulIOp::create(builder, loc, chunk, stride).getResult();
  return ::mlir::arith::AddIOp::create(builder, loc, base, offset).getResult();
}

// Per-graph rewrite state bundle, threaded through tryRewriteOne to
// keep the call sites concise.
struct RewriteCtx {
  ::dataflow::GraphOp graph;
  ::mlir::Value ctrl;
  ::llvm::DenseMap<BridgeKey, ::mlir::Value, BridgeKeyInfo> bridgeCache;
  ScopedValueCache indexCastCache;
  ScopedBridgeCache addressCache;
  ::mlir::Value zeroIdx;
  ::llvm::SmallVector<::mlir::Operation *, 8> deadGeps;
};

bool tryRewriteNestedMemcpy(::mlir::LLVM::MemcpyOp memcpy,
                            ::mlir::OpBuilder &builder, RewriteCtx &ctx) {
  if (memcpy.getIsVolatile())
    return false;

  ::mlir::Type lenType = memcpy.getLen().getType();
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(lenType))
    return false;
  if (!canMaterializeIndex(memcpy.getLen().getType(), memcpy))
    return false;

  ::mlir::Type byteTy = builder.getI8Type();
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(memcpy.getSrc().getType()) ||
      !::llvm::isa<::mlir::LLVM::LLVMPointerType>(memcpy.getDst().getType()))
    return false;
  if (!isSupportedBridgePointer(memcpy.getSrc(), ctx.graph) ||
      !isSupportedBridgePointer(memcpy.getDst(), ctx.graph))
    return false;

  ::mlir::Location loc = memcpy.getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(memcpy);

  std::optional<AddrResolution> src =
      resolveNestedMemcpyStaticPointer(memcpy.getSrc(), ctx.graph);
  std::optional<AddrResolution> dst =
      resolveNestedMemcpyStaticPointer(memcpy.getDst(), ctx.graph);
  bool useStaticBindings =
      src && dst && !src->ordinalStream && !dst->ordinalStream;

  ::mlir::Value zero = getDataflowConstant(builder, lenType, ctx.ctrl, 0, loc);
  ::mlir::Value one = getDataflowConstant(builder, lenType, ctx.ctrl, 1, loc);
  if (!zero || !one)
    return false;

  ::mlir::Value srcMem = getMemrefBridge(
      builder, ctx.graph, ctx.bridgeCache,
      useStaticBindings ? src->ptr : memcpy.getSrc(), byteTy, loc,
      /*topLevel=*/useStaticBindings, memcpy);
  ::mlir::Value dstMem = getMemrefBridge(
      builder, ctx.graph, ctx.bridgeCache,
      useStaticBindings ? dst->ptr : memcpy.getDst(), byteTy, loc,
      /*topLevel=*/useStaticBindings, memcpy);
  if (!srcMem || !dstMem)
    return false;

  auto materializeOffset =
      [&](const AddrResolution &resolution) -> ::mlir::Value {
    if (!resolution.intIndex)
      return {};
    ::mlir::Value byteOffset = getLinearByteIndex(
        builder, resolution.intIndex, resolution.intIndexByteStride,
        resolution.intIndexByteBias, loc, memcpy);
    if (!byteOffset)
      return {};
    return getIndexFromInt(builder, ctx.indexCastCache, byteOffset, loc, memcpy,
                           /*cacheResult=*/false);
  };
  ::mlir::Value srcOffset =
      useStaticBindings ? materializeOffset(*src) : ::mlir::Value{};
  ::mlir::Value dstOffset =
      useStaticBindings ? materializeOffset(*dst) : ::mlir::Value{};
  if (useStaticBindings &&
      ((src->intIndex && !srcOffset) || (dst->intIndex && !dstOffset)))
    return false;

  auto buildBefore = [&](::mlir::OpBuilder &bodyBuilder,
                         ::mlir::Location bodyLoc,
                         ::mlir::ValueRange arguments) {
    ::mlir::Value iv = arguments.front();
    ::mlir::Value execution = arguments[1];
    auto synchronized = ::dataflow::SyncOp::create(
        bodyBuilder, bodyLoc,
        ::mlir::TypeRange{iv.getType(), bodyBuilder.getNoneType()},
        ::mlir::ValueRange{iv, execution});
    iv = synchronized.getOutputs()[0];
    execution = synchronized.getOutputs()[1];
    ::mlir::Value active = ::mlir::arith::CmpIOp::create(
        bodyBuilder, bodyLoc, ::mlir::arith::CmpIPredicate::ult, iv,
        memcpy.getLen());
    ::mlir::scf::ConditionOp::create(bodyBuilder, bodyLoc, active,
                                     ::mlir::ValueRange{iv, execution});
  };
  auto buildAfter = [&](::mlir::OpBuilder &bodyBuilder,
                        ::mlir::Location bodyLoc,
                        ::mlir::ValueRange arguments) {
    ::mlir::Value iv = arguments.front();
    ::mlir::Value execution = arguments[1];
    ::mlir::Value address = iv;
    if (!::llvm::isa<::mlir::IndexType>(iv.getType()))
      address = ::mlir::arith::IndexCastUIOp::create(
                    bodyBuilder, bodyLoc, bodyBuilder.getIndexType(), iv)
                    .getResult();
    auto addOffset = [&](::mlir::Value offset) {
      if (!offset)
        return address;
      return ::mlir::arith::AddIOp::create(bodyBuilder, bodyLoc, address,
                                           offset)
          .getResult();
    };
    ::mlir::Value srcAddress = addOffset(srcOffset);
    ::mlir::Value dstAddress = addOffset(dstOffset);
    auto load = ::dataflow::LoadOp::create(
        bodyBuilder, bodyLoc, /*data=*/byteTy,
        /*done=*/bodyBuilder.getNoneType(), /*mem=*/srcMem,
        /*addr=*/srcAddress, /*ctrl=*/execution);
    auto store = ::dataflow::StoreOp::create(
        bodyBuilder, bodyLoc, /*done=*/bodyBuilder.getNoneType(),
        /*mem=*/dstMem, /*addr=*/dstAddress, /*data=*/load.getData(),
        /*ctrl=*/load.getDone());
    ::mlir::Value next =
        ::mlir::arith::AddIOp::create(bodyBuilder, bodyLoc, iv, one);
    ::mlir::scf::YieldOp::create(bodyBuilder, bodyLoc,
                                 ::mlir::ValueRange{next, store.getDone()});
  };
  ::mlir::scf::WhileOp::create(
      builder, loc, ::mlir::TypeRange{lenType, builder.getNoneType()},
      ::mlir::ValueRange{zero, ctx.ctrl}, buildBefore, buildAfter);
  memcpy->erase();
  if (useStaticBindings) {
    if (src->gepToErase)
      ctx.deadGeps.push_back(src->gepToErase);
    if (src->baseGepToErase)
      ctx.deadGeps.push_back(src->baseGepToErase);
    if (dst->gepToErase)
      ctx.deadGeps.push_back(dst->gepToErase);
    if (dst->baseGepToErase)
      ctx.deadGeps.push_back(dst->baseGepToErase);
  }
  return true;
}

bool tryRewriteDirectMemcpy(::mlir::LLVM::MemcpyOp memcpy,
                            ::mlir::OpBuilder &builder, RewriteCtx &ctx) {
  ::mlir::Type lenType = memcpy.getLen().getType();
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(lenType))
    return false;

  auto pointerRoot = [](mlir::Value pointer) {
    if (auto gep = pointer.getDefiningOp<::mlir::LLVM::GEPOp>())
      return gep.getBase();
    return pointer;
  };
  if (pointerRoot(memcpy.getSrc()) == pointerRoot(memcpy.getDst())) {
    memcpy.emitError("llvm.intr.memcpy overlapping ranges are unsupported");
    return false;
  }

  std::optional<DirectMemcpyPtrResolution> src =
      resolveDirectMemcpyPointer(memcpy.getSrc(), ctx.graph);
  std::optional<DirectMemcpyPtrResolution> dst =
      resolveDirectMemcpyPointer(memcpy.getDst(), ctx.graph);
  if (!src || !dst)
    return false;
  if (!isSupportedBridgePointer(src->basePtr, ctx.graph) ||
      !isSupportedBridgePointer(dst->basePtr, ctx.graph))
    return false;
  if ((src->offset && !canMaterializeIndex(src->offset.getType(), memcpy)) ||
      (dst->offset && !canMaterializeIndex(dst->offset.getType(), memcpy)))
    return false;

  ::mlir::Location loc = memcpy.getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(memcpy);

  ::mlir::Type byteTy = builder.getI8Type();
  ::mlir::Value srcMem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, src->basePtr, byteTy,
                      loc, /*topLevel=*/true, memcpy);
  ::mlir::Value dstMem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, dst->basePtr, byteTy,
                      loc, /*topLevel=*/true, memcpy);
  ::mlir::Value zero = getDataflowConstant(builder, lenType, ctx.ctrl, 0, loc);
  ::mlir::Value one = getDataflowConstant(builder, lenType, ctx.ctrl, 1, loc);
  auto materializeOffset = [&](::mlir::Value offset) -> ::mlir::Value {
    if (!offset)
      return {};
    return getIndexFromInt(builder, ctx.indexCastCache, offset, loc, memcpy,
                           /*cacheResult=*/false);
  };
  ::mlir::Value srcOffset = materializeOffset(src->offset);
  ::mlir::Value dstOffset = materializeOffset(dst->offset);
  if (!srcMem || !dstMem || !zero || !one || (src->offset && !srcOffset) ||
      (dst->offset && !dstOffset))
    return false;

  auto buildBefore = [&](::mlir::OpBuilder &bodyBuilder,
                         ::mlir::Location bodyLoc,
                         ::mlir::ValueRange arguments) {
    ::mlir::Value iv = arguments.front();
    ::mlir::Value execution = arguments[1];
    auto synchronized = ::dataflow::SyncOp::create(
        bodyBuilder, bodyLoc,
        ::mlir::TypeRange{iv.getType(), bodyBuilder.getNoneType()},
        ::mlir::ValueRange{iv, execution});
    iv = synchronized.getOutputs()[0];
    execution = synchronized.getOutputs()[1];
    ::mlir::Value active = ::mlir::arith::CmpIOp::create(
        bodyBuilder, bodyLoc, ::mlir::arith::CmpIPredicate::ult, iv,
        memcpy.getLen());
    ::mlir::scf::ConditionOp::create(bodyBuilder, bodyLoc, active,
                                     ::mlir::ValueRange{iv, execution});
  };
  auto buildAfter = [&](::mlir::OpBuilder &bodyBuilder,
                        ::mlir::Location bodyLoc,
                        ::mlir::ValueRange arguments) {
    ::mlir::Value iv = arguments.front();
    ::mlir::Value execution = arguments[1];
    ::mlir::Value address = iv;
    if (!::llvm::isa<::mlir::IndexType>(iv.getType()))
      address = ::mlir::arith::IndexCastUIOp::create(
                    bodyBuilder, bodyLoc, bodyBuilder.getIndexType(), iv)
                    .getResult();
    auto addOffset = [&](::mlir::Value offset) {
      if (!offset)
        return address;
      return ::mlir::arith::AddIOp::create(bodyBuilder, bodyLoc, address,
                                           offset)
          .getResult();
    };
    auto load = ::dataflow::LoadOp::create(
        bodyBuilder, bodyLoc, /*data=*/byteTy,
        /*done=*/bodyBuilder.getNoneType(), /*mem=*/srcMem,
        /*addr=*/addOffset(srcOffset), /*ctrl=*/execution);
    auto store = ::dataflow::StoreOp::create(
        bodyBuilder, bodyLoc, /*done=*/bodyBuilder.getNoneType(),
        /*mem=*/dstMem, /*addr=*/addOffset(dstOffset),
        /*data=*/load.getData(), /*ctrl=*/load.getDone());
    ::mlir::Value next =
        ::mlir::arith::AddIOp::create(bodyBuilder, bodyLoc, iv, one);
    ::mlir::scf::YieldOp::create(bodyBuilder, bodyLoc,
                                 ::mlir::ValueRange{next, store.getDone()});
  };
  ::mlir::scf::WhileOp::create(
      builder, loc, ::mlir::TypeRange{lenType, builder.getNoneType()},
      ::mlir::ValueRange{zero, ctx.ctrl}, buildBefore, buildAfter);
  memcpy->erase();
  if (src->gepToErase && src->gepToErase->use_empty())
    src->gepToErase->erase();
  if (dst->gepToErase && dst->gepToErase->use_empty())
    dst->gepToErase->erase();
  return true;
}

// Attempt to rewrite a single load or store. Returns true if a
// rewrite happened.
bool tryRewriteOne(::mlir::Operation *op, bool topLevel,
                   ::mlir::OpBuilder &builder, RewriteCtx &ctx) {
  ::mlir::Value ptrArg;
  ::mlir::Type elemTy;
  bool isLoad = ::llvm::isa<::mlir::LLVM::LoadOp>(op);
  if (isLoad) {
    auto load = ::llvm::cast<::mlir::LLVM::LoadOp>(op);
    if (load.getVolatile_() ||
        load.getOrdering() != ::mlir::LLVM::AtomicOrdering::not_atomic)
      return false;
    ptrArg = load.getAddr();
    elemTy = load.getResult().getType();
  } else {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    if (store.getVolatile_() ||
        store.getOrdering() != ::mlir::LLVM::AtomicOrdering::not_atomic)
      return false;
    ptrArg = store.getAddr();
    elemTy = store.getValue().getType();
  }
  // Pointer-element loads/stores trip the streaming verifier; skip.
  if (::llvm::isa<::mlir::LLVM::LLVMPointerType>(elemTy))
    return false;
  auto resolved = resolvePointer(ptrArg, ctx.graph, topLevel, elemTy);
  if (!resolved)
    return false;
  if (!isSupportedBridgePointer(resolved->ptr, ctx.graph))
    return false;
  ::mlir::Value addressSource = resolved->intIndex;
  if (resolved->ordinalStream)
    addressSource = resolved->ordinalStream.getIv();
  if (addressSource && !canMaterializeIndex(addressSource.getType(), op))
    return false;
  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);
  ::mlir::Value addr;
  auto &addressCache =
      ctx.addressCache.try_emplace(op->getBlock()).first->second;
  BridgeKey addressKey{ptrArg, elemTy};
  if (auto it = addressCache.find(addressKey); it != addressCache.end()) {
    addr = it->second;
  } else {
    if (resolved->ordinalStream) {
      ::mlir::Value ordinal = resolved->ordinalStream.getIv();
      ordinal = getBiasedStreamOrdinal(builder, resolved->ordinalStream,
                                       ordinal, resolved->ordinalElementBias,
                                       ctx.ctrl, loc, op);
      if (!ordinal)
        return false;
      addr = getElementIndex(builder, ctx.graph, ctx.indexCastCache, ordinal,
                             resolved->byteToElementShift, ctx.ctrl, loc,
                             topLevel, op);
    } else if (resolved->intIndex) {
      ::mlir::Value intIndex = getLinearByteIndex(
          builder, resolved->intIndex, resolved->intIndexByteStride,
          resolved->intIndexByteBias, loc, op);
      if (!intIndex)
        return false;
      addr = getElementIndex(builder, ctx.graph, ctx.indexCastCache, intIndex,
                             resolved->byteToElementShift, ctx.ctrl, loc,
                             topLevel, op);
    } else {
      addr = getZeroIndex(builder, ctx.graph, ctx.zeroIdx, loc, topLevel, op);
    }
    if (!addr)
      return false;
    addressCache.try_emplace(addressKey, addr);
  }
  bool staticBinding = topLevel || isGraphPtrBlockArg(resolved->ptr, ctx.graph);
  ::mlir::Value mem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, resolved->ptr,
                      elemTy, loc, staticBinding, op);
  if (!mem)
    return false;
  if (isLoad) {
    auto load = ::llvm::cast<::mlir::LLVM::LoadOp>(op);
    auto newLoad = ::dataflow::LoadOp::create(
        builder, loc, /*data=*/elemTy, /*done=*/builder.getNoneType(),
        /*mem=*/mem, /*addr=*/addr, /*ctrl=*/ctx.ctrl);
    load.getResult().replaceAllUsesWith(newLoad.getData());
  } else {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    ::dataflow::StoreOp::create(
        builder, loc, /*done=*/builder.getNoneType(), /*mem=*/mem,
        /*addr=*/addr, /*data=*/store.getValue(), /*ctrl=*/ctx.ctrl);
  }
  op->erase();
  if (resolved->gepToErase)
    ctx.deadGeps.push_back(resolved->gepToErase);
  if (resolved->baseGepToErase)
    ctx.deadGeps.push_back(resolved->baseGepToErase);
  return true;
}

bool tryRewriteMemcpy(::mlir::Operation *op, bool topLevel,
                      ::mlir::OpBuilder &builder, RewriteCtx &ctx) {
  auto memcpy = ::llvm::dyn_cast<::mlir::LLVM::MemcpyOp>(op);
  if (!memcpy)
    return false;
  if (memcpy.getIsVolatile()) {
    memcpy.emitError("volatile llvm.intr.memcpy is unsupported");
    return false;
  }
  if (!topLevel)
    return tryRewriteNestedMemcpy(memcpy, builder, ctx);

  ::mlir::Type lenType = memcpy.getLen().getType();
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(lenType))
    return false;
  if (!canMaterializeIndex(memcpy.getLen().getType(), memcpy))
    return false;

  if (tryRewriteDirectMemcpy(memcpy, builder, ctx))
    return true;

  if (!::llvm::isa<::mlir::IntegerType>(lenType))
    return false;

  std::optional<MemcpyPtrResolution> src =
      resolveMemcpyPointer(memcpy.getSrc(), ctx.graph);
  std::optional<MemcpyPtrResolution> dst =
      resolveMemcpyPointer(memcpy.getDst(), ctx.graph);
  if (!src || !dst || src->outerStream != dst->outerStream)
    return false;
  if (!isSupportedBridgePointer(src->basePtr, ctx.graph) ||
      !isSupportedBridgePointer(dst->basePtr, ctx.graph))
    return false;

  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);

  ::mlir::Value zero = getDataflowConstant(builder, lenType, ctx.ctrl, 0, loc);
  ::mlir::Value one = getDataflowConstant(builder, lenType, ctx.ctrl, 1, loc);
  if (!zero || !one)
    return false;

  ::dataflow::StreamOp outer = src->outerStream;
  ::mlir::Value tripDelta = ::mlir::arith::SubIOp::create(
                                builder, loc, outer.getLimit(), outer.getInit())
                                .getResult();
  ::mlir::Value outerTrips =
      ::mlir::arith::DivSIOp::create(builder, loc, tripDelta, outer.getStep())
          .getResult();
  ::mlir::Value totalBytes =
      ::mlir::arith::MulIOp::create(builder, loc, outerTrips, memcpy.getLen())
          .getResult();

  auto copyStream = ::dataflow::StreamOp::create(
      builder, loc, lenType, builder.getI1Type(), /*init=*/zero,
      /*limit=*/totalBytes, /*step=*/one, ::dataflow::StreamStepKind::Add,
      ::mlir::arith::CmpIPredicate::slt);
  ::mlir::Value activeByte = copyStream.getIv();
  ::mlir::Value copyLenRaw =
      ::dataflow::InvariantOp::create(builder, loc, lenType,
                                      copyStream.getPhase(), memcpy.getLen())
          .getOutput();
  ::mlir::Value copyLen =
      projectStreamValue(builder, copyStream, copyLenRaw, loc);

  ::mlir::Value chunk =
      ::mlir::arith::DivSIOp::create(builder, loc, activeByte, copyLen)
          .getResult();
  ::mlir::Value offset =
      ::mlir::arith::RemSIOp::create(builder, loc, activeByte, copyLen)
          .getResult();
  ::mlir::Value srcAddr = activeByte;
  if (src->chunkStride != memcpy.getLen()) {
    ::mlir::Value srcStrideRaw =
        ::dataflow::InvariantOp::create(builder, loc, lenType,
                                        copyStream.getPhase(), src->chunkStride)
            .getOutput();
    ::mlir::Value srcStride =
        projectStreamValue(builder, copyStream, srcStrideRaw, loc);
    srcAddr = makeChunkedAddress(builder, chunk, srcStride, offset, loc);
  }
  ::mlir::Value dstAddr = activeByte;
  if (dst->chunkStride != memcpy.getLen()) {
    ::mlir::Value dstStrideRaw =
        ::dataflow::InvariantOp::create(builder, loc, lenType,
                                        copyStream.getPhase(), dst->chunkStride)
            .getOutput();
    ::mlir::Value dstStride =
        projectStreamValue(builder, copyStream, dstStrideRaw, loc);
    dstAddr = makeChunkedAddress(builder, chunk, dstStride, offset, loc);
  }

  ::mlir::Type byteTy = builder.getI8Type();
  ::mlir::Value srcMem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, src->basePtr, byteTy,
                      loc, topLevel, op);
  ::mlir::Value dstMem =
      getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, dst->basePtr, byteTy,
                      loc, topLevel, op);
  ::mlir::Value srcIndex =
      getIndexFromInt(builder, ctx.indexCastCache, srcAddr, loc, op);
  ::mlir::Value dstIndex =
      getIndexFromInt(builder, ctx.indexCastCache, dstAddr, loc, op);

  auto load = ::dataflow::LoadOp::create(
      builder, loc, /*data=*/byteTy, /*done=*/builder.getNoneType(),
      /*mem=*/srcMem, /*addr=*/srcIndex, /*ctrl=*/ctx.ctrl);
  ::dataflow::StoreOp::create(builder, loc, /*done=*/builder.getNoneType(),
                              /*mem=*/dstMem, /*addr=*/dstIndex,
                              /*data=*/load.getData(),
                              /*ctrl=*/load.getDone());
  op->erase();
  return true;
}

bool tryRewriteMemset(::mlir::Operation *op, bool topLevel,
                      ::mlir::OpBuilder &builder, RewriteCtx &ctx) {
  if (!isLLVMMemsetOp(op) || isVolatileMemIntrinsic(op))
    return false;
  if (op->getNumOperands() != 3)
    return false;

  ::mlir::Value dst = op->getOperand(0);
  ::mlir::Value byteValue = op->getOperand(1);
  ::mlir::Value byteCount = op->getOperand(2);
  if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(dst.getType()))
    return false;
  if (!isSupportedBridgePointer(dst, ctx.graph))
    return false;
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(byteCount.getType()))
    return false;
  if (!canMaterializeIndex(byteCount.getType(), op))
    return false;

  std::optional<::mlir::Type> elemTy =
      inferMemsetElementType(ctx.graph, dst, op);
  if (!elemTy || ::llvm::isa<::mlir::LLVM::LLVMPointerType>(*elemTy))
    return false;
  unsigned elementByteWidth = getElementByteWidth(*elemTy);
  if (elementByteWidth == 0)
    return false;
  if (elementByteWidth > 1 && !isZeroIntegerValue(byteValue))
    return false;
  if (!(elementByteWidth == 1 && byteValue.getType() == *elemTy) &&
      !getIntegerLikeAttr(builder, *elemTy, 0))
    return false;

  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);
  ::mlir::Value elementCount = getMemsetElementCount(
      builder, byteCount, elementByteWidth, ctx.ctrl, loc);
  if (!elementCount)
    return false;

  ::mlir::Value mem = getMemrefBridge(builder, ctx.graph, ctx.bridgeCache, dst,
                                      *elemTy, loc, topLevel, op);
  if (!mem)
    return false;

  auto makeFillValue = [&](::mlir::OpBuilder &b,
                           ::mlir::Location valueLoc) -> ::mlir::Value {
    if (elementByteWidth == 1 && byteValue.getType() == *elemTy)
      return byteValue;
    return getDataflowConstant(b, *elemTy, ctx.ctrl, 0, valueLoc);
  };

  if (!topLevel) {
    ::mlir::Value lower =
        getDataflowConstant(builder, builder.getIndexType(), ctx.ctrl, 0, loc);
    ::mlir::Value step =
        getDataflowConstant(builder, builder.getIndexType(), ctx.ctrl, 1, loc);
    ::mlir::Value upper =
        getIndexFromInt(builder, ctx.indexCastCache, elementCount, loc, op,
                        /*cacheResult=*/false);
    if (!lower || !step || !upper)
      return false;
    auto buildBody = [&](::mlir::OpBuilder &bodyBuilder,
                         ::mlir::Location bodyLoc, ::mlir::Value iv,
                         ::mlir::ValueRange) {
      ::mlir::Value fill = makeFillValue(bodyBuilder, bodyLoc);
      if (!fill)
        return;
      ::dataflow::StoreOp::create(bodyBuilder, bodyLoc,
                                  /*done=*/bodyBuilder.getNoneType(),
                                  /*mem=*/mem, /*addr=*/iv,
                                  /*data=*/fill, /*ctrl=*/ctx.ctrl);
      ::mlir::scf::YieldOp::create(bodyBuilder, bodyLoc);
    };
    ::mlir::scf::ForOp::create(builder, loc, lower, upper, step,
                               ::mlir::ValueRange{}, buildBody);
    op->erase();
    return true;
  }

  ::mlir::Value zero =
      getDataflowConstant(builder, byteCount.getType(), ctx.ctrl, 0, loc);
  ::mlir::Value one =
      getDataflowConstant(builder, byteCount.getType(), ctx.ctrl, 1, loc);
  if (!zero || !one)
    return false;
  auto fillStream = ::dataflow::StreamOp::create(
      builder, loc, byteCount.getType(), builder.getI1Type(), /*init=*/zero,
      /*limit=*/elementCount, /*step=*/one, ::dataflow::StreamStepKind::Add,
      ::mlir::arith::CmpIPredicate::slt);
  ::mlir::Value index =
      getIndexFromInt(builder, ctx.indexCastCache, fillStream.getIv(), loc, op);
  ::mlir::Value fill = makeFillValue(builder, loc);
  if (!index || !fill)
    return false;
  ::dataflow::StoreOp::create(builder, loc, /*done=*/builder.getNoneType(),
                              /*mem=*/mem, /*addr=*/index, /*data=*/fill,
                              /*ctrl=*/ctx.ctrl);
  op->erase();
  return true;
}

unsigned rewriteOneGraph(::dataflow::GraphOp graph,
                         ::mlir::OpBuilder &builder) {
  ::mlir::Value ctrl = getThreadCtrl(graph);
  if (!ctrl)
    return 0;

  RewriteCtx ctx;
  ctx.graph = graph;
  ctx.ctrl = ctrl;

  // Collect rewrite targets up front so the walk is independent of
  // mutations performed by tryRewriteOne.
  struct Target {
    ::mlir::Operation *op;
    bool topLevel;
  };
  ::llvm::SmallVector<Target, 16> targets;
  ::mlir::Block &entry = graph.getBody().front();
  graph.getBody().walk([&](::mlir::Operation *op) {
    if (::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                    ::mlir::LLVM::MemcpyOp>(op) ||
        isLLVMMemsetOp(op))
      targets.push_back({op, op->getBlock() == &entry});
    return ::mlir::WalkResult::advance();
  });

  unsigned rewrites = 0;
  for (auto &t : targets) {
    if (isLLVMMemsetOp(t.op)) {
      if (tryRewriteMemset(t.op, t.topLevel, builder, ctx))
        ++rewrites;
      continue;
    }
    if (::llvm::isa<::mlir::LLVM::MemcpyOp>(t.op)) {
      if (tryRewriteMemcpy(t.op, t.topLevel, builder, ctx))
        ++rewrites;
      continue;
    }
    if (tryRewriteOne(t.op, t.topLevel, builder, ctx))
      ++rewrites;
  }

  // Erase orphan geps (those whose only uses were the rewritten
  // load/store ops). Some may have other live uses -- skip those
  // silently.
  ::llvm::SmallPtrSet<::mlir::Operation *, 8> visitedDeadGeps;
  for (::mlir::Operation *gep : ctx.deadGeps) {
    if (visitedDeadGeps.contains(gep))
      continue;
    if (gep->use_empty()) {
      visitedDeadGeps.insert(gep);
      gep->erase();
    }
  }
  return rewrites;
}

::mlir::LogicalResult checkResidualMemoryEffects(::dataflow::GraphOp graph) {
  ::mlir::WalkResult result =
      graph.getBody().walk(
          [](::mlir::Operation *op) -> ::mlir::WalkResult {
            bool lacksCompletion =
                ::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                            ::mlir::LLVM::MemcpyOp>(op) ||
                isLLVMMemsetOp(op);
            if (!lacksCompletion)
              return ::mlir::WalkResult::advance();

            op->emitError()
                << "loom-lower-graph-memory: residual memory operation '"
                << op->getName().getStringRef()
                << "' has no explicit completion event";
            return ::mlir::WalkResult::interrupt();
          });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

::mlir::LogicalResult checkNormalizedMemoryEffects(::mlir::ModuleOp module) {
  ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
      ::mlir::cast<::mlir::ModuleOp>(module->clone()));
  ::mlir::OpBuilder builder(module.getContext());
  ::llvm::SmallVector<::dataflow::GraphOp, 8> graphs;
  scratch->walk([&](::dataflow::GraphOp graph) { graphs.push_back(graph); });
  for (auto graph : graphs) {
    if (!graph.isExternal())
      (void)rewriteOneGraph(graph, builder);
  }
  for (auto graph : graphs) {
    if (!graph.isExternal() &&
        ::mlir::failed(checkResidualMemoryEffects(graph)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

struct LowerGraphMemoryPass
    : public ::mlir::PassWrapper<LowerGraphMemoryPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphMemoryPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-graph-memory";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lower graph-local memory accesses and recursively flatten "
           "structured graph regions with per-alias-partition memory "
           "frontiers.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::memref::MemRefDialect,
                    ::mlir::scf::SCFDialect, ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    if (::mlir::failed(
            ::loom::lowering::checkGraphRegionLoweringPreconditions(module))) {
      signalPassFailure();
      return;
    }
    if (::mlir::failed(checkNormalizedMemoryEffects(module))) {
      signalPassFailure();
      return;
    }

    ::llvm::SmallVector<::dataflow::GraphOp, 8> graphs;
    module.walk([&](::dataflow::GraphOp graph) { graphs.push_back(graph); });

    for (::dataflow::GraphOp graph : graphs) {
      if (graph.isExternal())
        continue;
      (void)rewriteOneGraph(graph, builder);
    }

    for (::dataflow::GraphOp graph : graphs) {
      if (graph.isExternal())
        continue;
      if (::mlir::failed(::loom::lowering::lowerGraphRegions(graph))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerGraphMemoryPass() {
  return std::make_unique<LowerGraphMemoryPass>();
}

void registerLowerGraphMemoryPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerGraphMemoryPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
