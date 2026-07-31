// Tokenize residual `llvm.load` / `llvm.store` ops inside construction-local
// `dataflow.graph` bodies into `dataflow.load` / `dataflow.store`. LLVM pointer
// ABI roots are projected into canonical graph-owned memref ports; no pointer
// reinterpretation operation survives publication.
//
// Two recognition modes:
//   * Top-level (load/store sits directly in the graph entry block):
//     require the chain to terminate at a graph block argument so
//     the rewrite preserves the brief's "block-arg base" contract.
//        - gep %arg_ptr[%idx]   -> (arg_ptr, idx)
//        - dataflow.carry init=arg_ptr  -> (carry_result, null)
//        - graph block-arg !llvm.ptr   -> (block_arg, null)
//
//   * Nested (load/store sits inside an scf.for / scf.if region): the address
//     analysis must still resolve to the same graph memory root. Dynamic
//     pointer values never become independent graph memory capabilities.
//
// Loads whose pointer is none of the above (e.g. derived from
// `llvm.alloca` or `llvm.mlir.addressof` at the top level) remain untouched
// during recognition and are rejected by the residual gate. Raw LLVM memory
// operations do not provide the canonical dataflow completion contract.
// Pointer-element loads are also skipped: bridging !llvm.ptr-of-ptr to
// memref<?x!llvm.ptr> trips the streaming load verifier.

#include "Frontend/Lowering/Passes.h"

#include "Frontend/Lowering/GraphMemoryAddressing.h"
#include "GraphMemoryLowering.h"
#include "GraphRegionLowering.h"
#include "StreamOrdinal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using ::loom::lowering::LinearElementTerm;

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

struct ImportedViewKey {
  ::mlir::Value ptr;
  ::mlir::Type elem;
  bool operator==(const ImportedViewKey &o) const {
    return ptr == o.ptr && elem == o.elem;
  }
};

struct ImportedViewKeyInfo {
  static ImportedViewKey getEmptyKey() { return {{}, {}}; }
  static ImportedViewKey getTombstoneKey() {
    return {::mlir::Value::getFromOpaquePointer((void *)1),
            ::mlir::Type::getFromOpaquePointer((void *)1)};
  }
  static unsigned getHashValue(const ImportedViewKey &k) {
    return ::llvm::hash_combine(::mlir::hash_value(k.ptr),
                                ::mlir::hash_value(k.elem));
  }
  static bool isEqual(const ImportedViewKey &a, const ImportedViewKey &b) {
    return a == b;
  }
};

using ScopedValueCache =
    ::llvm::DenseMap<::mlir::Block *,
                     ::llvm::DenseMap<::mlir::Value, ::mlir::Value>>;
using ScopedBridgeCache = ::llvm::DenseMap<
    ::mlir::Block *,
    ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>>;

// Result of resolving one llvm.load / llvm.store address. Every GEP in a
// supported chain contributes either one typed linear term or constant bias;
// intermediate pointers never become independent memory capabilities.
struct AddrResolution {
  ::mlir::Value ptr;
  ::llvm::SmallVector<::loom::lowering::LinearElementTerm, 4>
      linearElementTerms;
  ::mlir::Value directElementIndex;
  ::dataflow::StreamOp ordinalStream;
  std::int64_t ordinalElementBias = 0;
  ::mlir::Type linearIndexType;
  std::int64_t linearElementBias = 0;
  ::llvm::SmallVector<::mlir::Operation *, 4> gepsToErase;
};

unsigned getElementByteWidth(::mlir::Type elemTy);

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

::dataflow::StreamOp getUnitStridePointerCarryStream(::dataflow::CarryOp carry,
                                                     ::dataflow::GraphOp graph,
                                                     ::mlir::Type elemTy,
                                                     unsigned indexBits) {
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
      !::loom::lowering::isZeroBasedUnitOrdinalStream(stream, indexBits))
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

std::optional<AddrResolution> resolvePointer(::mlir::Value loadStorePtr,
                                             ::dataflow::GraphOp graph,
                                             bool topLevel, ::mlir::Type elemTy,
                                             unsigned indexBits) {
  if (auto linear = ::loom::lowering::resolveLinearMemoryAddress(
          loadStorePtr, graph, elemTy, indexBits)) {
    AddrResolution resolution;
    resolution.ptr = linear->root;
    bool directElementIndex =
        linear->elementTerms.size() == 1 && linear->elementBias == 0 &&
        linear->elementTerms.front().scale == 1 &&
        linear->elementTerms.front().exactSignedDivideShift == 0;
    if (directElementIndex) {
      resolution.directElementIndex = linear->elementTerms.front().index;
    } else {
      resolution.linearElementTerms = std::move(linear->elementTerms);
      resolution.linearIndexType = linear->indexType;
      resolution.linearElementBias = linear->elementBias;
    }
    resolution.gepsToErase = std::move(linear->gepsLeafToRoot);
    return resolution;
  }
  if (topLevel) {
    if (auto gep = loadStorePtr.getDefiningOp<::mlir::LLVM::GEPOp>()) {
      ::mlir::Value base = gep.getBase();
      if (auto carry = getPointerCarry(base)) {
        if (::dataflow::StreamOp stream = getUnitStridePointerCarryStream(
                *carry, graph, elemTy, indexBits)) {
          std::optional<std::int64_t> bias =
              getSingleIndexElementStride(gep, elemTy);
          if (!bias)
            return std::nullopt;
          AddrResolution resolution;
          resolution.ptr = carry->getInit();
          resolution.ordinalStream = stream;
          resolution.ordinalElementBias = *bias;
          return resolution;
        }
      }
      return std::nullopt;
    }
    if (auto carry = loadStorePtr.getDefiningOp<::dataflow::CarryOp>()) {
      if (::dataflow::StreamOp stream = getUnitStridePointerCarryStream(
              carry, graph, elemTy, indexBits)) {
        AddrResolution resolution;
        resolution.ptr = carry.getInit();
        resolution.ordinalStream = stream;
        return resolution;
      }
      return std::nullopt;
    }
    if (auto carry = getGatedPointerCarry(loadStorePtr)) {
      if (::dataflow::StreamOp stream = getUnitStridePointerCarryStream(
              *carry, graph, elemTy, indexBits)) {
        AddrResolution resolution;
        resolution.ptr = carry->getInit();
        resolution.ordinalStream = stream;
        return resolution;
      }
      return std::nullopt;
    }
    if (isGraphPtrBlockArg(loadStorePtr, graph)) {
      AddrResolution resolution;
      resolution.ptr = loadStorePtr;
      return resolution;
    }
    return std::nullopt;
  }
  // MLIR may use an index-typed dynamic GEP operand before an LLVM DataLayout
  // exists. Accept only the exact one-element-step form rooted at the graph
  // memory formal; every other nested pointer remains unnormalized.
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
          ::llvm::isa<::mlir::IntegerType, ::mlir::IndexType>(idx.getType())) {
        AddrResolution resolution;
        resolution.ptr = gep.getBase();
        resolution.linearElementTerms.push_back({idx, 1, 0});
        resolution.linearIndexType = idx.getType();
        resolution.gepsToErase.push_back(gep.getOperation());
        return resolution;
      }
    }
  }
  return std::nullopt;
}

// A launch-owned imported view can only originate at an address-space-zero
// graph memory formal. Derived pointers are normalized into integer offsets;
// they never become additional capability roots.
bool isSupportedImportedViewRoot(::mlir::Value ptr, ::dataflow::GraphOp graph) {
  if (!isGraphPtrBlockArg(ptr, graph))
    return false;
  auto ptrTy = ::llvm::dyn_cast<::mlir::LLVM::LLVMPointerType>(ptr.getType());
  if (!ptrTy)
    return false;
  unsigned addressSpace = ptrTy.getAddressSpace();
  std::optional<::llvm::DataLayout> llvmDataLayout =
      getModuleLLVMDataLayout(graph);
  if (llvmDataLayout && llvmDataLayout->isNonIntegralAddressSpace(addressSpace))
    return false;
  // Canonical graph memrefs have the default memory space.
  return addressSpace == 0;
}

::mlir::Value getImportedMemrefView(
    ::dataflow::GraphOp graph,
    ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>
        &cache,
    ::mlir::Value ptr, ::mlir::Type elem, ::mlir::Location loc) {
  if (!isSupportedImportedViewRoot(ptr, graph))
    return {};
  ImportedViewKey key{ptr, elem};
  if (auto it = cache.find(key); it != cache.end())
    return it->second;
  auto memrefTy = ::mlir::MemRefType::get({::mlir::ShapedType::kDynamic}, elem);
  ::mlir::BlockArgument view =
      graph.getBody().front().addArgument(memrefTy, loc);
  cache.try_emplace(key, view);
  return view;
}

struct CanonicalMemoryPort {
  unsigned sourceOrdinal = 0;
  ::mlir::Type type;
  ::mlir::Value transientValue;
  ::mlir::DictionaryAttr attrs;
  std::vector<std::uint8_t> typeKey;
};

::mlir::LogicalResult normalizeGraphMemoryPorts(
    ::dataflow::GraphOp graph,
    const ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>
        &importedViews,
    ::llvm::SmallVectorImpl<unsigned> *sourceOrdinals) {
  ::llvm::ArrayRef<int32_t> segments = graph.getInputSegmentSizes();
  unsigned valueCount = static_cast<unsigned>(segments[0]);
  unsigned streamCount = static_cast<unsigned>(segments[1]);
  unsigned memoryCount = static_cast<unsigned>(segments[2]);
  unsigned firstMemoryInput = valueCount + streamCount;
  ::mlir::Block &entry = graph.getBody().front();

  ::llvm::SmallVector<::mlir::BlockArgument, 4> oldMemoryArgs;
  oldMemoryArgs.reserve(memoryCount);
  for (unsigned ordinal = 0; ordinal < memoryCount; ++ordinal)
    oldMemoryArgs.push_back(entry.getArgument(1 + firstMemoryInput + ordinal));

  ::mlir::Builder builder(graph.getContext());
  ::llvm::SmallVector<CanonicalMemoryPort, 4> ports;
  for (unsigned ordinal = 0; ordinal < memoryCount; ++ordinal) {
    ::mlir::BlockArgument source = oldMemoryArgs[ordinal];
    ::mlir::DictionaryAttr attrs =
        ::mlir::function_interface_impl::getArgAttrDict(
            graph, firstMemoryInput + ordinal);
    if (!attrs)
      attrs = builder.getDictionaryAttr({});

    auto appendPort = [&](::mlir::Type type,
                          ::mlir::Value transient) -> ::mlir::LogicalResult {
      ::llvm::Expected<::loom::CanonicalSemanticBytes> encoded =
          ::dataflow::encodeCanonicalType(type);
      if (!encoded)
        return graph.emitError("cannot encode canonical graph memory type ")
               << type << ": " << ::llvm::toString(encoded.takeError());
      ports.push_back({ordinal, type, transient, attrs,
                       std::vector<std::uint8_t>(encoded->bytes().begin(),
                                                 encoded->bytes().end())});
      return ::mlir::success();
    };

    if (::llvm::isa<::mlir::MemRefType, ::mlir::UnrankedMemRefType>(
            source.getType())) {
      if (::mlir::failed(appendPort(source.getType(), source)))
        return ::mlir::failure();
      continue;
    }
    if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(source.getType()))
      return graph.emitError("memory input #")
             << ordinal << " has unsupported transient type "
             << source.getType();

    for (const auto &entry : importedViews) {
      if (entry.first.ptr != source)
        continue;
      if (::mlir::failed(appendPort(entry.second.getType(), entry.second)))
        return ::mlir::failure();
    }
  }

  ::llvm::sort(ports, [](const CanonicalMemoryPort &lhs,
                         const CanonicalMemoryPort &rhs) {
    if (lhs.sourceOrdinal != rhs.sourceOrdinal)
      return lhs.sourceOrdinal < rhs.sourceOrdinal;
    return std::lexicographical_compare(lhs.typeKey.begin(), lhs.typeKey.end(),
                                        rhs.typeKey.begin(), rhs.typeKey.end());
  });

  // Every raw pointer use must have been absorbed into an integer address
  // function before the pointer formal can be removed.
  for (auto source : oldMemoryArgs)
    if (::llvm::isa<::mlir::LLVM::LLVMPointerType>(source.getType()) &&
        !source.use_empty())
      return graph.emitError("memory pointer input remains after graph memory "
                             "normalization: ")
             << source.getType();

  ::llvm::SmallVector<::mlir::DictionaryAttr, 8> argumentAttrs;
  argumentAttrs.reserve(firstMemoryInput + ports.size());
  for (unsigned index = 0; index < firstMemoryInput; ++index) {
    ::mlir::DictionaryAttr attrs =
        ::mlir::function_interface_impl::getArgAttrDict(graph, index);
    argumentAttrs.push_back(attrs ? attrs : builder.getDictionaryAttr({}));
  }

  for (auto [index, port] : ::llvm::enumerate(ports)) {
    auto arg = entry.insertArgument(1 + firstMemoryInput + index, port.type,
                                    graph.getLoc());
    port.transientValue.replaceAllUsesWith(arg);
    argumentAttrs.push_back(port.attrs);
  }

  ::llvm::BitVector erase(entry.getNumArguments());
  for (::mlir::BlockArgument arg : oldMemoryArgs)
    erase.set(arg.getArgNumber());
  for (const auto &view : importedViews)
    erase.set(::llvm::cast<::mlir::BlockArgument>(view.second).getArgNumber());
  entry.eraseArguments(erase);

  ::llvm::SmallVector<::mlir::Type, 8> inputTypes;
  inputTypes.reserve(firstMemoryInput + ports.size());
  ::llvm::ArrayRef<::mlir::Type> oldInputs =
      graph.getFunctionType().getInputs();
  inputTypes.append(oldInputs.begin(), oldInputs.begin() + firstMemoryInput);
  for (const CanonicalMemoryPort &port : ports)
    inputTypes.push_back(port.type);
  graph.setFunctionType(builder.getFunctionType(
      inputTypes, graph.getFunctionType().getResults()));
  ::llvm::SmallVector<int32_t, 3> normalizedSegments{
      static_cast<int32_t>(valueCount), static_cast<int32_t>(streamCount),
      static_cast<int32_t>(ports.size())};
  graph.setInputSegments(normalizedSegments);
  ::mlir::function_interface_impl::setAllArgAttrDicts(graph, argumentAttrs);

  if (sourceOrdinals) {
    sourceOrdinals->clear();
    for (const CanonicalMemoryPort &port : ports)
      sourceOrdinals->push_back(port.sourceOrdinal);
  }
  return ::mlir::success();
}

// `indexBits` is the canonical index width the pass boundary already
// resolved; this predicate never resolves it again.
bool canMaterializeIndex(::mlir::Type type, unsigned indexBits) {
  if (::llvm::isa<::mlir::IndexType>(type))
    return true;
  auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type);
  if (!intTy || !intTy.isSignless())
    return false;
  return intTy.getWidth() <= indexBits;
}

::mlir::Value getIndexCast(::mlir::OpBuilder &builder,
                           ::dataflow::GraphOp graph, ScopedValueCache &cache,
                           unsigned indexBits, ::mlir::Value iv,
                           ::mlir::Location loc, bool topLevel,
                           ::mlir::Operation *insertBeforeIfNested) {
  if (::llvm::isa<::mlir::IndexType>(iv.getType()))
    return iv;
  if (!canMaterializeIndex(iv.getType(), indexBits))
    return {};
  ::mlir::Value source = iv;
  auto &blockCache =
      cache.try_emplace(insertBeforeIfNested->getBlock()).first->second;
  if (auto it = blockCache.find(source); it != blockCache.end())
    return it->second;
  bool hoist = topLevel && ::llvm::isa<::mlir::BlockArgument>(iv);
  ::mlir::OpBuilder::InsertionGuard g(builder);
  if (hoist)
    builder.setInsertionPointToStart(&graph.getBody().front());
  else
    builder.setInsertionPoint(insertBeforeIfNested);
  if (auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(iv.getType());
      integer && integer.getWidth() < indexBits) {
    auto canonicalType = builder.getIntegerType(indexBits);
    iv = ::mlir::arith::ExtSIOp::create(builder, loc, canonicalType, iv)
             .getResult();
  }
  ::mlir::Value out = ::mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), iv)
                          .getResult();
  blockCache.try_emplace(source, out);
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

::mlir::Value getLinearElementIndex(::mlir::OpBuilder &builder,
                                    ::llvm::ArrayRef<LinearElementTerm> terms,
                                    ::mlir::Type indexType,
                                    std::int64_t elementBias,
                                    ::mlir::Location loc,
                                    ::mlir::Operation *insertBefore) {
  auto canonicalType = ::llvm::dyn_cast<::mlir::IntegerType>(indexType);
  if (!canonicalType || !canonicalType.isSignless())
    return {};
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(insertBefore);
  ::mlir::Value result;
  for (const LinearElementTerm &term : terms) {
    ::mlir::Value contribution = term.index;
    if (::llvm::isa<::mlir::IndexType>(contribution.getType())) {
      contribution = ::mlir::arith::IndexCastOp::create(
                         builder, loc, canonicalType, contribution)
                         .getResult();
    } else if (contribution.getType() != canonicalType) {
      auto source =
          ::llvm::dyn_cast<::mlir::IntegerType>(contribution.getType());
      if (!source || !source.isSignless() ||
          source.getWidth() >= canonicalType.getWidth())
        return {};
      contribution = ::mlir::arith::ExtSIOp::create(builder, loc, canonicalType,
                                                    contribution)
                         .getResult();
    }
    if (term.exactSignedDivideShift != 0) {
      auto shiftAttr =
          builder.getIntegerAttr(canonicalType, term.exactSignedDivideShift);
      ::mlir::Value shift = ::mlir::arith::ConstantOp::create(
                                builder, loc, canonicalType, shiftAttr)
                                .getResult();
      contribution =
          ::mlir::arith::ShRSIOp::create(builder, loc, contribution, shift)
              .getResult();
    }
    if (term.scale != 1) {
      ::mlir::TypedAttr scaleAttr =
          getIntegerLikeAttr(builder, canonicalType, term.scale);
      if (!scaleAttr)
        return {};
      ::mlir::Value scale = ::mlir::arith::ConstantOp::create(
                                builder, loc, canonicalType, scaleAttr)
                                .getResult();
      contribution =
          ::mlir::arith::MulIOp::create(builder, loc, contribution, scale)
              .getResult();
    }
    result = result ? ::mlir::arith::AddIOp::create(builder, loc, result,
                                                    contribution)
                          .getResult()
                    : contribution;
  }
  if (elementBias != 0) {
    ::mlir::TypedAttr biasAttr =
        getIntegerLikeAttr(builder, canonicalType, elementBias);
    if (!biasAttr)
      return {};
    ::mlir::Value bias =
        ::mlir::arith::ConstantOp::create(builder, loc, canonicalType, biasAttr)
            .getResult();
    result = result ? ::mlir::arith::AddIOp::create(builder, loc, result, bias)
                          .getResult()
                    : bias;
  }
  if (!result)
    return {};
  return ::mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), result)
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

// Per-graph rewrite state bundle, threaded through tryRewriteOne to
// keep the call sites concise.
struct RewriteCtx {
  ::dataflow::GraphOp graph;
  ::mlir::Value ctrl;
  // Resolved once at the pass boundary and read-only from here on.
  unsigned indexBits = 0;
  ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>
      importedViews;
  ScopedValueCache indexCastCache;
  ScopedBridgeCache addressCache;
  ::mlir::Value zeroIdx;
  ::llvm::SmallVector<::mlir::Operation *, 8> deadGeps;
};

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
  auto resolved =
      resolvePointer(ptrArg, ctx.graph, topLevel, elemTy, ctx.indexBits);
  if (!resolved)
    return false;
  if (!isSupportedImportedViewRoot(resolved->ptr, ctx.graph))
    return false;
  if (resolved->ordinalStream &&
      !canMaterializeIndex(resolved->ordinalStream.getIv().getType(),
                           ctx.indexBits))
    return false;
  for (const LinearElementTerm &term : resolved->linearElementTerms) {
    if (!canMaterializeIndex(term.index.getType(), ctx.indexBits))
      return false;
  }
  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);
  ::mlir::Value addr;
  auto &addressCache =
      ctx.addressCache.try_emplace(op->getBlock()).first->second;
  ImportedViewKey addressKey{ptrArg, elemTy};
  if (auto it = addressCache.find(addressKey); it != addressCache.end()) {
    addr = it->second;
  } else {
    if (resolved->directElementIndex) {
      addr = getIndexCast(builder, ctx.graph, ctx.indexCastCache, ctx.indexBits,
                          resolved->directElementIndex, loc, topLevel, op);
    } else if (resolved->ordinalStream) {
      ::mlir::Value ordinal = resolved->ordinalStream.getIv();
      ordinal = getBiasedStreamOrdinal(builder, resolved->ordinalStream,
                                       ordinal, resolved->ordinalElementBias,
                                       ctx.ctrl, loc, op);
      if (!ordinal)
        return false;
      addr = getIndexCast(builder, ctx.graph, ctx.indexCastCache, ctx.indexBits,
                          ordinal, loc, topLevel, op);
    } else if (!resolved->linearElementTerms.empty() ||
               resolved->linearElementBias != 0) {
      addr = getLinearElementIndex(builder, resolved->linearElementTerms,
                                   resolved->linearIndexType,
                                   resolved->linearElementBias, loc, op);
    } else {
      addr = getZeroIndex(builder, ctx.graph, ctx.zeroIdx, loc, topLevel, op);
    }
    if (!addr)
      return false;
    addressCache.try_emplace(addressKey, addr);
  }
  ::mlir::Value mem = getImportedMemrefView(ctx.graph, ctx.importedViews,
                                            resolved->ptr, elemTy, loc);
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
  ctx.deadGeps.append(resolved->gepsToErase);
  return true;
}

// The canonical index width of one graph, read at the exact scope that owns
// it. The pass boundary validates every graph before anything mutates, so a
// graph that cannot resolve one is simply left alone here.
std::optional<unsigned> getGraphIndexBits(::dataflow::GraphOp graph) {
  ::llvm::Expected<unsigned> bits = ::loom::getIndexBitWidth(graph);
  if (bits)
    return *bits;
  ::llvm::consumeError(bits.takeError());
  return std::nullopt;
}

unsigned sinkBranchSelectedLoads(::dataflow::GraphOp graph,
                                 ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::LLVM::LoadOp, 4> loads;
  graph.getBody().walk([&](::mlir::LLVM::LoadOp load) {
    if (!load.getVolatile_() &&
        load.getOrdering() == ::mlir::LLVM::AtomicOrdering::not_atomic)
      loads.push_back(load);
  });

  unsigned rewritten = 0;
  for (::mlir::LLVM::LoadOp load : loads) {
    auto selected = ::llvm::dyn_cast<::mlir::OpResult>(load.getAddr());
    if (!selected || !selected.hasOneUse())
      continue;
    if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(selected.getType()))
      continue;

    if (auto select =
            ::llvm::dyn_cast<::mlir::arith::SelectOp>(selected.getOwner())) {
      ::mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(load);
      auto cloneLoadAndYield = [&](::mlir::OpBuilder &bodyBuilder,
                                   ::mlir::Location loc,
                                   ::mlir::Value address) {
        ::mlir::IRMapping mapping;
        mapping.map(load.getAddr(), address);
        auto cloned = ::llvm::cast<::mlir::LLVM::LoadOp>(
            bodyBuilder.clone(*load.getOperation(), mapping));
        ::mlir::scf::YieldOp::create(bodyBuilder, loc, cloned.getResult());
      };
      auto thenBuilder = [&](::mlir::OpBuilder &bodyBuilder,
                             ::mlir::Location loc) {
        cloneLoadAndYield(bodyBuilder, loc, select.getTrueValue());
      };
      auto elseBuilder = [&](::mlir::OpBuilder &bodyBuilder,
                             ::mlir::Location loc) {
        cloneLoadAndYield(bodyBuilder, loc, select.getFalseValue());
      };
      auto branch = ::mlir::scf::IfOp::create(builder, load.getLoc(),
                                              select.getCondition(),
                                              thenBuilder, elseBuilder);
      load.getResult().replaceAllUsesWith(branch.getResult(0));
      load.erase();
      if (select->use_empty())
        select.erase();
      ++rewritten;
      continue;
    }

    auto branch = ::llvm::dyn_cast<::mlir::scf::IfOp>(selected.getOwner());
    if (!branch || branch.getElseRegion().empty())
      continue;

    unsigned resultIndex = selected.getResultNumber();
    ::mlir::scf::YieldOp thenYield = ::llvm::dyn_cast<::mlir::scf::YieldOp>(
        branch.getThenRegion().front().getTerminator());
    ::mlir::scf::YieldOp elseYield = ::llvm::dyn_cast<::mlir::scf::YieldOp>(
        branch.getElseRegion().front().getTerminator());
    if (!thenYield || !elseYield || resultIndex >= thenYield.getNumOperands() ||
        resultIndex >= elseYield.getNumOperands())
      continue;
    for (::mlir::scf::YieldOp yield : {thenYield, elseYield}) {
      ::mlir::IRMapping mapping;
      mapping.map(load.getAddr(), yield.getOperand(resultIndex));
      ::mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(yield);
      auto cloned = ::llvm::cast<::mlir::LLVM::LoadOp>(
          builder.clone(*load.getOperation(), mapping));
      yield.setOperand(resultIndex, cloned.getResult());
    }
    selected.setType(load.getResult().getType());
    load.getResult().replaceAllUsesWith(selected);
    load.erase();
    ++rewritten;
  }
  return rewritten;
}

::mlir::LogicalResult
checkGraphRegionLoweringPreconditionsAfterLoadSinking(::mlir::ModuleOp module) {
  ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
      ::mlir::cast<::mlir::ModuleOp>(module->clone()));
  ::mlir::OpBuilder builder(module.getContext());
  scratch->walk([&](::dataflow::GraphOp graph) {
    if (!graph.isExternal())
      (void)sinkBranchSelectedLoads(graph, builder);
  });
  return ::loom::lowering::checkGraphRegionLoweringPreconditions(*scratch);
}

::mlir::LogicalResult
rewriteOneGraph(::dataflow::GraphOp graph, ::mlir::OpBuilder &builder,
                ::llvm::SmallVectorImpl<unsigned> *sourceOrdinals = nullptr) {
  ::mlir::Value ctrl = getThreadCtrl(graph);
  if (!ctrl)
    return graph.emitError(
        "loom-lower-graph-memory: graph entry has no start token");
  std::optional<unsigned> indexBits = getGraphIndexBits(graph);
  if (!indexBits)
    return ::mlir::failure();

  RewriteCtx ctx;
  ctx.graph = graph;
  ctx.ctrl = ctrl;
  ctx.indexBits = *indexBits;

  (void)sinkBranchSelectedLoads(graph, builder);

  // Collect rewrite targets up front so the walk is independent of
  // mutations performed by tryRewriteOne.
  struct Target {
    ::mlir::Operation *op;
    bool topLevel;
  };
  ::llvm::SmallVector<Target, 16> targets;
  ::mlir::Block &entry = graph.getBody().front();
  graph.getBody().walk([&](::mlir::Operation *op) {
    if (::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp>(op))
      targets.push_back({op, op->getBlock() == &entry});
    return ::mlir::WalkResult::advance();
  });

  for (auto &t : targets) {
    (void)tryRewriteOne(t.op, t.topLevel, builder, ctx);
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
  return normalizeGraphMemoryPorts(graph, ctx.importedViews, sourceOrdinals);
}

::mlir::LogicalResult checkResidualMemoryEffects(::dataflow::GraphOp graph) {
  ::mlir::WalkResult result =
      graph.getBody().walk(
          [](::mlir::Operation *op) -> ::mlir::WalkResult {
            bool lacksCompletion =
                ::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                            ::mlir::LLVM::MemcpyOp, ::mlir::LLVM::MemmoveOp,
                            ::mlir::LLVM::MemsetOp>(op);
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

// One unusable index-width declaration fails the pass with its owner's reason,
// before any graph is rewritten.
::mlir::LogicalResult checkGraphIndexWidths(::mlir::ModuleOp module) {
  ::mlir::WalkResult result = module.walk([&](::dataflow::GraphOp graph) {
    if (graph.isExternal())
      return ::mlir::WalkResult::advance();
    ::llvm::Expected<unsigned> bits = ::loom::getIndexBitWidth(graph);
    if (bits)
      return ::mlir::WalkResult::advance();
    module.emitError("loom-lower-graph-memory: ")
        << ::llvm::toString(bits.takeError());
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
    if (!graph.isExternal() && ::mlir::failed(rewriteOneGraph(graph, builder)))
      return ::mlir::failure();
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
    if (::mlir::failed(::loom::lowering::lowerGraphMemory(getOperation())))
      signalPassFailure();
  }
};

} // namespace

namespace loom {
namespace lowering {

::mlir::LogicalResult lowerGraphMemory(
    ::mlir::ModuleOp module,
    ::llvm::SmallVectorImpl<GraphMemoryInputProjection> *projections) {
  ::mlir::OpBuilder builder(module.getContext());

  // Resolve each graph's canonical index width before mutation, then prove the
  // complete lowering on a scratch module. The production mutation below is
  // therefore failure-atomic at its caller's publication boundary.
  if (::mlir::failed(checkGraphIndexWidths(module)) ||
      ::mlir::failed(
          checkGraphRegionLoweringPreconditionsAfterLoadSinking(module)) ||
      ::mlir::failed(checkNormalizedMemoryEffects(module)))
    return ::mlir::failure();

  ::llvm::SmallVector<::dataflow::GraphOp, 8> graphs;
  module.walk([&](::dataflow::GraphOp graph) { graphs.push_back(graph); });

  if (projections)
    projections->clear();
  for (::dataflow::GraphOp graph : graphs) {
    if (graph.isExternal())
      continue;
    ::llvm::SmallVector<unsigned, 4> sourceOrdinals;
    if (::mlir::failed(rewriteOneGraph(
            graph, builder, projections ? &sourceOrdinals : nullptr)))
      return ::mlir::failure();
    if (projections)
      projections->push_back({graph, std::move(sourceOrdinals)});
  }

  for (::dataflow::GraphOp graph : graphs) {
    if (graph.isExternal())
      continue;
    std::optional<unsigned> indexBits = getGraphIndexBits(graph);
    if (!indexBits || ::mlir::failed(lowerGraphRegions(graph, *indexBits)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

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
