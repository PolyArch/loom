// Tokenize residual `llvm.load` / `llvm.store` ops inside construction-local
// `dataflow.graph` bodies into `dataflow.load` / `dataflow.store`. A selected
// root-relative access becomes capability-plus-index and loses its pointer
// arithmetic; an unmarked access retains the exact typed pointer expression.
// One graph memory input names the object-scoped service selected by the
// address root; the enclosing thread materializes that service before launch.

#include "Frontend/Lowering/GraphMemoryAddressing.h"
#include "Frontend/Lowering/Passes.h"

#include "GraphMemoryLowering.h"
#include "GraphRegionLowering.h"
#include "StreamOrdinal.h"

#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/IR/LoomDialect.h"

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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace {

bool isGraphPtrBlockArg(::mlir::Value v, ::dataflow::GraphOp graph) {
  auto blockArg = ::llvm::dyn_cast<::mlir::BlockArgument>(v);
  if (!blockArg || blockArg.getOwner() != &graph.getBody().front())
    return false;
  return ::llvm::isa<::mlir::LLVM::LLVMPointerType>(blockArg.getType());
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

::mlir::Value resolvePointerServiceRoot(::mlir::Value pointer,
                                        ::dataflow::GraphOp graph) {
  return ::loom::lowering::resolveMemoryServiceBoundaryRoot(
      pointer,
      [&](::mlir::Value value) { return isGraphPtrBlockArg(value, graph); });
}

::mlir::FailureOr<::mlir::Type> storageElementType(::mlir::Operation *access,
                                                   ::mlir::Type dataType) {
  if (auto vector = ::llvm::dyn_cast<::mlir::VectorType>(dataType)) {
    if (vector.isScalable() || vector.getRank() == 0 ||
        vector.getNumElements() == 0) {
      access->emitError(
          "LLVM vector memory access requires a fixed nonempty lane shape");
      return ::mlir::failure();
    }
    return vector.getElementType();
  }
  auto pointerType = ::llvm::dyn_cast<::mlir::LLVM::LLVMPointerType>(dataType);
  if (!pointerType)
    return dataType;
  ::llvm::Expected<::loom::PointerLayout> layout =
      ::loom::resolvePointerLayout(access, pointerType.getAddressSpace());
  if (!layout) {
    access->emitError() << ::llvm::toString(layout.takeError());
    return ::mlir::failure();
  }
  if (layout->kind != ::loom::PointerLayoutKind::StableIntegral) {
    access->emitError(
        "pointer payload requires a stable integral representation provider");
    return ::mlir::failure();
  }
  return ::mlir::IntegerType::get(access->getContext(),
                                  layout->representationBits);
}

::mlir::Value getImportedMemrefView(
    ::dataflow::GraphOp graph,
    ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>
        &cache,
    ::mlir::Value ptr, ::mlir::Type elem, ::mlir::Location loc) {
  if (!isGraphPtrBlockArg(ptr, graph))
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
  ::loom::lowering::GraphMemoryInputSource source;
  ::mlir::Type type;
  ::mlir::Value transientValue;
  ::mlir::DictionaryAttr attrs;
  std::vector<std::uint8_t> typeKey;
};

::mlir::LogicalResult normalizeGraphMemoryPorts(
    ::dataflow::GraphOp graph,
    const ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>
        &importedViews,
    ::llvm::SmallVectorImpl<::loom::lowering::GraphMemoryInputSource>
        *sources) {
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

    auto appendPort = [&](::loom::lowering::GraphMemoryInputSource source,
                          ::mlir::Type type,
                          ::mlir::Value transient) -> ::mlir::LogicalResult {
      ::llvm::Expected<::loom::CanonicalSemanticBytes> encoded =
          ::dataflow::encodeCanonicalType(type);
      if (!encoded)
        return graph.emitError("cannot encode canonical graph memory type ")
               << type << ": " << ::llvm::toString(encoded.takeError());
      ports.push_back({source, type, transient, attrs,
                       std::vector<std::uint8_t>(encoded->bytes().begin(),
                                                 encoded->bytes().end())});
      return ::mlir::success();
    };

    if (::llvm::isa<::mlir::MemRefType, ::mlir::UnrankedMemRefType>(
            source.getType())) {
      if (::mlir::failed(appendPort(
              {::loom::lowering::GraphMemoryInputSourceKind::ExistingMemory,
               ordinal},
              source.getType(), source)))
        return ::mlir::failure();
      continue;
    }
    return graph.emitError("memory input #")
           << ordinal
           << " is not an explicit memory capability: " << source.getType();
  }

  for (const auto &imported : importedViews) {
    auto pointer = ::llvm::dyn_cast<::mlir::BlockArgument>(imported.first.ptr);
    if (!pointer || pointer.getOwner() != &entry ||
        pointer.getArgNumber() == 0 || pointer.getArgNumber() > valueCount)
      return graph.emitError(
          "pointer-addressed service root is not a graph value input");
    ::mlir::DictionaryAttr attrs = builder.getDictionaryAttr({});
    ::llvm::Expected<::loom::CanonicalSemanticBytes> encoded =
        ::dataflow::encodeCanonicalType(imported.second.getType());
    if (!encoded)
      return graph.emitError("cannot encode pointer service memory type ")
             << imported.second.getType() << ": "
             << ::llvm::toString(encoded.takeError());
    ports.push_back(
        {{::loom::lowering::GraphMemoryInputSourceKind::PointerService,
          pointer.getArgNumber() - 1},
         imported.second.getType(),
         imported.second,
         attrs,
         std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end())});
  }

  ::llvm::sort(ports, [](const CanonicalMemoryPort &lhs,
                         const CanonicalMemoryPort &rhs) {
    if (lhs.source.kind != rhs.source.kind)
      return lhs.source.kind < rhs.source.kind;
    if (lhs.source.sourceOrdinal != rhs.source.sourceOrdinal)
      return lhs.source.sourceOrdinal < rhs.source.sourceOrdinal;
    return std::lexicographical_compare(lhs.typeKey.begin(), lhs.typeKey.end(),
                                        rhs.typeKey.begin(), rhs.typeKey.end());
  });

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

  if (sources) {
    sources->clear();
    for (const CanonicalMemoryPort &port : ports)
      sources->push_back(port.source);
  }
  return ::mlir::success();
}

// Per-graph rewrite state bundle, threaded through tryRewriteOne to
// keep the call sites concise.
struct RewriteCtx {
  ::dataflow::GraphOp graph;
  ::mlir::Value ctrl;
  unsigned indexBits = 0;
  ::llvm::DenseMap<ImportedViewKey, ::mlir::Value, ImportedViewKeyInfo>
      importedViews;
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> localAllocations;
};

std::optional<std::uint64_t>
constantIntegerValue(::mlir::Value value) {
  if (auto constant = value.getDefiningOp<::mlir::arith::ConstantOp>()) {
    if (auto integer =
            ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getValue()))
      return integer.getValue().getLimitedValue();
  }
  if (auto constant = value.getDefiningOp<::dataflow::ConstantOp>()) {
    if (auto integer =
            ::llvm::dyn_cast<::mlir::IntegerAttr>(constant.getConstValue()))
      return integer.getValue().getLimitedValue();
  }
  return std::nullopt;
}

std::optional<std::pair<std::uint64_t, ::mlir::Type>>
flattenStaticAllocaType(::mlir::Type type) {
  std::uint64_t elements = 1;
  while (auto array =
             ::llvm::dyn_cast<::mlir::LLVM::LLVMArrayType>(type)) {
    const std::uint64_t count = array.getNumElements();
    if (count == 0 ||
        elements > std::numeric_limits<std::uint64_t>::max() / count)
      return std::nullopt;
    elements *= count;
    type = array.getElementType();
  }
  if (!::llvm::isa<::mlir::IntegerType, ::mlir::Float16Type,
                   ::mlir::BFloat16Type, ::mlir::Float32Type,
                   ::mlir::Float64Type>(type))
    return std::nullopt;
  return std::make_pair(elements, type);
}

::mlir::LogicalResult materializeLocalAllocations(RewriteCtx &ctx,
                                                   ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::LLVM::AllocaOp, 8> allocas;
  ctx.graph.getBody().walk([&](::mlir::LLVM::AllocaOp alloca) {
    allocas.push_back(alloca);
  });
  if (allocas.empty())
    return ::mlir::success();

  ::mlir::Block &entry = ctx.graph.getBody().front();
  builder.setInsertionPointToStart(&entry);
  for (::mlir::LLVM::AllocaOp alloca : allocas) {
    const auto count = constantIntegerValue(alloca.getArraySize());
    const auto flattened = flattenStaticAllocaType(alloca.getElemType());
    if (!count || *count == 0 || !flattened ||
        *count > std::numeric_limits<std::uint64_t>::max() /
                     flattened->first)
      return alloca.emitOpError(
          "loom-lower-graph-memory: only static scalar LLVM alloca objects "
          "can be represented as graph-local memory");
    const std::uint64_t elements = *count * flattened->first;
    if (elements > static_cast<std::uint64_t>(
                       std::numeric_limits<int64_t>::max()))
      return alloca.emitOpError(
          "loom-lower-graph-memory: local alloca extent exceeds memref "
          "dimension range");
    auto memrefType = ::mlir::MemRefType::get(
        {static_cast<int64_t>(elements)}, flattened->second);
    auto allocation = ::mlir::memref::AllocOp::create(
        builder, alloca.getLoc(), memrefType, ::mlir::ValueRange{});
    ctx.localAllocations.try_emplace(alloca.getRes(), allocation.getMemref());
  }
  return ::mlir::success();
}

void ensurePointerValueServices(RewriteCtx &ctx) {
  ::llvm::DenseSet<::mlir::Value> representedPointers;
  for (const auto &view : ctx.importedViews)
    representedPointers.insert(view.first.ptr);

  const unsigned valueCount =
      static_cast<unsigned>(ctx.graph.getInputSegmentSizes()[0]);
  ::mlir::Block &entry = ctx.graph.getBody().front();
  ::mlir::Type byteType = ::mlir::IntegerType::get(ctx.graph.getContext(), 8);
  for (unsigned ordinal = 0; ordinal < valueCount; ++ordinal) {
    ::mlir::Value value = entry.getArgument(1 + ordinal);
    if (!::llvm::isa<::mlir::LLVM::LLVMPointerType>(value.getType()) ||
        representedPointers.contains(value))
      continue;
    (void)getImportedMemrefView(ctx.graph, ctx.importedViews, value, byteType,
                                ctx.graph.getLoc());
  }
}

// Attempt to rewrite a single load or store. Returns true if a
// rewrite happened.
::mlir::FailureOr<::mlir::Value>
materializeRootRelativeAddress(::mlir::Operation *access, ::mlir::Value pointer,
                               ::mlir::Type accessType,
                               ::mlir::OpBuilder &builder, unsigned indexBits) {
  auto resolved = ::loom::lowering::resolveLinearMemoryAddress(
      pointer, accessType, indexBits);
  if (!resolved || resolved->terms.size() != resolved->elementTerms.size()) {
    access->emitError(
        "loom-lower-graph-memory: selected root-relative address has no "
        "exact canonical-index projection");
    return ::mlir::failure();
  }

  ::mlir::Location loc = access->getLoc();
  auto integerType = builder.getIntegerType(indexBits);
  auto integerConstant = [&](std::int64_t value) -> ::mlir::Value {
    return ::mlir::arith::ConstantOp::create(
        builder, loc, integerType, builder.getIntegerAttr(integerType, value));
  };
  auto toCanonicalInteger = [&](::mlir::Value value) -> ::mlir::Value {
    if (::llvm::isa<::mlir::IndexType>(value.getType()))
      return ::mlir::arith::IndexCastOp::create(builder, loc, integerType,
                                                value);
    auto sourceType = ::llvm::dyn_cast<::mlir::IntegerType>(value.getType());
    if (!sourceType || !sourceType.isSignless() ||
        sourceType.getWidth() > indexBits)
      return {};
    if (sourceType.getWidth() < indexBits)
      return ::mlir::arith::ExtSIOp::create(builder, loc, integerType, value);
    return value;
  };

  ::mlir::Value result = integerConstant(resolved->elementBias);
  for (auto [byteTerm, elementTerm] :
       ::llvm::zip_equal(resolved->terms, resolved->elementTerms)) {
    (void)byteTerm;
    ::mlir::Value term = toCanonicalInteger(elementTerm.index);
    if (!term) {
      access->emitError(
          "loom-lower-graph-memory: root-relative address term exceeds the "
          "canonical index width");
      return ::mlir::failure();
    }
    if (elementTerm.exactSignedDivideShift != 0) {
      ::mlir::Value shift = integerConstant(elementTerm.exactSignedDivideShift);
      auto divide = ::mlir::arith::ShRSIOp::create(builder, loc, term, shift);
      divide.setIsExact(true);
      term = divide;
    }
    if (elementTerm.scale != 1) {
      ::mlir::Value scale = integerConstant(elementTerm.scale);
      auto multiply = ::mlir::arith::MulIOp::create(builder, loc, term, scale);
      multiply.setOverflowFlags(::mlir::arith::IntegerOverflowFlags::nsw);
      term = multiply;
    }
    auto add = ::mlir::arith::AddIOp::create(builder, loc, result, term);
    add.setOverflowFlags(::mlir::arith::IntegerOverflowFlags::nsw);
    result = add;
  }
  return ::mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), result)
      .getResult();
}

std::optional<::dataflow::AtomicOrdering>
convertAtomicOrdering(::mlir::LLVM::AtomicOrdering ordering) {
  switch (ordering) {
  case ::mlir::LLVM::AtomicOrdering::unordered:
    return ::dataflow::AtomicOrdering::Unordered;
  case ::mlir::LLVM::AtomicOrdering::monotonic:
    return ::dataflow::AtomicOrdering::Monotonic;
  case ::mlir::LLVM::AtomicOrdering::acquire:
    return ::dataflow::AtomicOrdering::Acquire;
  case ::mlir::LLVM::AtomicOrdering::release:
    return ::dataflow::AtomicOrdering::Release;
  case ::mlir::LLVM::AtomicOrdering::acq_rel:
    return ::dataflow::AtomicOrdering::AcqRel;
  case ::mlir::LLVM::AtomicOrdering::seq_cst:
    return ::dataflow::AtomicOrdering::SeqCst;
  case ::mlir::LLVM::AtomicOrdering::not_atomic:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<::dataflow::SyncScopeRefAttr>
convertSyncScope(::mlir::MLIRContext *context,
                 std::optional<::llvm::StringRef> syncscope) {
  if (!syncscope || syncscope->empty() || *syncscope == "system")
    return ::dataflow::SyncScopeRefAttr::get(
        context, ::dataflow::SyncScopeKind::System);
  if (*syncscope == "singlethread" || *syncscope == "single_thread")
    return ::dataflow::SyncScopeRefAttr::get(
        context, ::dataflow::SyncScopeKind::SingleThread);
  return std::nullopt;
}

std::optional<::dataflow::AtomicRmwKind>
convertAtomicRmwKind(::mlir::LLVM::AtomicBinOp kind) {
  switch (kind) {
  case ::mlir::LLVM::AtomicBinOp::xchg:
    return ::dataflow::AtomicRmwKind::Xchg;
  case ::mlir::LLVM::AtomicBinOp::add:
    return ::dataflow::AtomicRmwKind::Add;
  case ::mlir::LLVM::AtomicBinOp::sub:
    return ::dataflow::AtomicRmwKind::Sub;
  case ::mlir::LLVM::AtomicBinOp::_and:
    return ::dataflow::AtomicRmwKind::And;
  case ::mlir::LLVM::AtomicBinOp::nand:
    return ::dataflow::AtomicRmwKind::Nand;
  case ::mlir::LLVM::AtomicBinOp::_or:
    return ::dataflow::AtomicRmwKind::Or;
  case ::mlir::LLVM::AtomicBinOp::_xor:
    return ::dataflow::AtomicRmwKind::Xor;
  case ::mlir::LLVM::AtomicBinOp::max:
    return ::dataflow::AtomicRmwKind::Max;
  case ::mlir::LLVM::AtomicBinOp::min:
    return ::dataflow::AtomicRmwKind::Min;
  case ::mlir::LLVM::AtomicBinOp::umax:
    return ::dataflow::AtomicRmwKind::UMax;
  case ::mlir::LLVM::AtomicBinOp::umin:
    return ::dataflow::AtomicRmwKind::UMin;
  case ::mlir::LLVM::AtomicBinOp::fadd:
    return ::dataflow::AtomicRmwKind::FAdd;
  case ::mlir::LLVM::AtomicBinOp::fsub:
    return ::dataflow::AtomicRmwKind::FSub;
  case ::mlir::LLVM::AtomicBinOp::fmax:
    return ::dataflow::AtomicRmwKind::FMax;
  case ::mlir::LLVM::AtomicBinOp::fmin:
    return ::dataflow::AtomicRmwKind::FMin;
  case ::mlir::LLVM::AtomicBinOp::uinc_wrap:
    return ::dataflow::AtomicRmwKind::UIncWrap;
  case ::mlir::LLVM::AtomicBinOp::udec_wrap:
    return ::dataflow::AtomicRmwKind::UDecWrap;
  case ::mlir::LLVM::AtomicBinOp::usub_cond:
    return ::dataflow::AtomicRmwKind::USubCond;
  case ::mlir::LLVM::AtomicBinOp::usub_sat:
    return ::dataflow::AtomicRmwKind::USubSat;
  case ::mlir::LLVM::AtomicBinOp::fmaximum:
    return ::dataflow::AtomicRmwKind::FMaximum;
  case ::mlir::LLVM::AtomicBinOp::fminimum:
    return ::dataflow::AtomicRmwKind::FMinimum;
  case ::mlir::LLVM::AtomicBinOp::fmaximumnum:
    return ::dataflow::AtomicRmwKind::FMaximumNum;
  case ::mlir::LLVM::AtomicBinOp::fminimumnum:
    return ::dataflow::AtomicRmwKind::FMinimumNum;
  }
  return std::nullopt;
}

template <typename AtomicOp>
std::optional<::dataflow::AtomicAccessContractAttr>
makeAtomicAccessContract(AtomicOp op, ::mlir::MLIRContext *context,
                         ::mlir::Type dataType) {
  auto ordering = convertAtomicOrdering(op.getOrdering());
  auto scope = convertSyncScope(context, op.getSyncscope());
  auto alignment = op.getAlignment();
  if (!ordering || !scope || !alignment || *alignment == 0 ||
      !::llvm::isPowerOf2_64(*alignment)) {
    op.emitError(
        "loom-lower-graph-memory: atomic source requires a supported "
        "ordering/scope and an explicit power-of-two alignment");
    return std::nullopt;
  }
  std::optional<::dataflow::VectorAtomicGranularity> granularity;
  if (auto vector = ::llvm::dyn_cast<::mlir::VectorType>(dataType)) {
    if (vector.isScalable() || vector.getRank() == 0 ||
        vector.getNumElements() == 0) {
      op.emitError("loom-lower-graph-memory: scalable or empty atomic vector is "
                   "not representable");
      return std::nullopt;
    }
    granularity = ::dataflow::VectorAtomicGranularity::WholePayload;
  }
  return ::dataflow::AtomicAccessContractAttr::get(
      context, *ordering, *scope, *alignment, granularity,
      op.getVolatile_());
}

// Translate the LLVM memory spelling while keeping all ordering, scope,
// alignment, volatility, and atomic-action semantics in Dataflow contracts.
bool tryRewriteOne(::mlir::Operation *op, ::mlir::OpBuilder &builder,
                   RewriteCtx &ctx) {
  if (auto fence = ::llvm::dyn_cast<::mlir::LLVM::FenceOp>(op)) {
    auto ordering = convertAtomicOrdering(fence.getOrdering());
    auto scope = convertSyncScope(ctx.graph.getContext(), fence.getSyncscope());
    if (!ordering || !scope) {
      fence.emitError("loom-lower-graph-memory: unsupported LLVM fence "
                      "ordering or synchronization scope");
      return false;
    }
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(fence);
    auto contract = ::dataflow::FenceContractAttr::get(
        ctx.graph.getContext(), *ordering, *scope);
    ::dataflow::FenceOp::create(builder, fence.getLoc(), builder.getNoneType(),
                                ctx.ctrl, contract);
    fence.erase();
    return true;
  }

  ::mlir::Value ptrArg;
  ::mlir::Type elemTy;
  const bool isLoad = ::llvm::isa<::mlir::LLVM::LoadOp>(op);
  const bool isStore = ::llvm::isa<::mlir::LLVM::StoreOp>(op);
  const bool isRmw = ::llvm::isa<::mlir::LLVM::AtomicRMWOp>(op);
  const bool isCmpXchg = ::llvm::isa<::mlir::LLVM::AtomicCmpXchgOp>(op);
  if (isLoad) {
    auto load = ::llvm::cast<::mlir::LLVM::LoadOp>(op);
    ptrArg = load.getAddr();
    elemTy = load.getResult().getType();
  } else if (isStore) {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    ptrArg = store.getAddr();
    elemTy = store.getValue().getType();
  } else if (isRmw) {
    auto rmw = ::llvm::cast<::mlir::LLVM::AtomicRMWOp>(op);
    ptrArg = rmw.getPtr();
    elemTy = rmw.getVal().getType();
  } else if (isCmpXchg) {
    auto cmp = ::llvm::cast<::mlir::LLVM::AtomicCmpXchgOp>(op);
    ptrArg = cmp.getPtr();
    elemTy = cmp.getCmp().getType();
  } else {
    return false;
  }

  ::mlir::Value root = resolvePointerServiceRoot(ptrArg, ctx.graph);
  bool localAllocation = false;
  if (!root) {
    auto resolved = ::loom::lowering::resolveLinearMemoryAddress(
        ptrArg, elemTy, ctx.indexBits);
    if (!resolved)
      return false;
    root = resolved->root;
    localAllocation = ctx.localAllocations.contains(root);
  }
  if (!root || (localAllocation && !ctx.localAllocations.contains(root)))
    return false;
  ::mlir::FailureOr<::mlir::Type> storage = storageElementType(op, elemTy);
  if (::mlir::failed(storage))
    return false;
  ::mlir::Location loc = op->getLoc();
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  ::mlir::Value address = ptrArg;
  ::mlir::Attribute rootRelative =
      op->getAttr(::loom::rootRelativeAddressAttrName);
  if (rootRelative || localAllocation) {
    if (!::llvm::isa<::mlir::UnitAttr>(rootRelative)) {
      op->emitError("loom-lower-graph-memory: root-relative address marker is "
                    "malformed");
      return false;
    }
    auto projected = materializeRootRelativeAddress(op, ptrArg, elemTy, builder,
                                                    ctx.indexBits);
    if (::mlir::failed(projected))
      return false;
    address = *projected;
  }
  ::mlir::Value mem = localAllocation
                          ? ctx.localAllocations.lookup(root)
                          : getImportedMemrefView(ctx.graph, ctx.importedViews,
                                                  root, *storage, loc);
  if (!mem)
    return false;
  if (isLoad) {
    auto load = ::llvm::cast<::mlir::LLVM::LoadOp>(op);
    auto lowered = ::dataflow::LoadOp::create(
        builder, loc, elemTy, builder.getNoneType(), mem, address, ctx.ctrl);
    if (load.getOrdering() == ::mlir::LLVM::AtomicOrdering::not_atomic) {
      lowered.setContractAttr(::dataflow::PlainAccessContractAttr::get(
          ctx.graph.getContext(), load.getVolatile_()));
    } else if (auto contract = makeAtomicAccessContract(
                   load, ctx.graph.getContext(), elemTy)) {
      lowered.setContractAttr(*contract);
    } else {
      lowered.erase();
      return false;
    }
    load.getResult().replaceAllUsesWith(lowered.getData());
  } else if (isStore) {
    auto store = ::llvm::cast<::mlir::LLVM::StoreOp>(op);
    auto lowered = ::dataflow::StoreOp::create(
        builder, loc, builder.getNoneType(), mem, address, store.getValue(),
        ctx.ctrl);
    if (store.getOrdering() == ::mlir::LLVM::AtomicOrdering::not_atomic) {
      lowered.setContractAttr(::dataflow::PlainAccessContractAttr::get(
          ctx.graph.getContext(), store.getVolatile_()));
    } else if (auto contract = makeAtomicAccessContract(
                   store, ctx.graph.getContext(), store.getValue().getType())) {
      lowered.setContractAttr(*contract);
    } else {
      lowered.erase();
      return false;
    }
  } else if (isRmw) {
    auto rmw = ::llvm::cast<::mlir::LLVM::AtomicRMWOp>(op);
    auto kind = convertAtomicRmwKind(rmw.getBinOp());
    auto access = makeAtomicAccessContract(rmw, ctx.graph.getContext(), elemTy);
    if (!kind || !access)
      return false;
    auto contract = ::dataflow::AtomicRmwContractAttr::get(
        ctx.graph.getContext(), *kind, *access);
    auto lowered = ::dataflow::AtomicRmwOp::create(
        builder, loc, elemTy, builder.getNoneType(), mem, address, rmw.getVal(),
        ctx.ctrl, {}, contract);
    rmw.getRes().replaceAllUsesWith(lowered.getOld());
  } else {
    auto cmp = ::llvm::cast<::mlir::LLVM::AtomicCmpXchgOp>(op);
    auto successOrdering = convertAtomicOrdering(cmp.getSuccessOrdering());
    auto failureOrdering = convertAtomicOrdering(cmp.getFailureOrdering());
    auto scope = convertSyncScope(ctx.graph.getContext(), cmp.getSyncscope());
    auto alignment = cmp.getAlignment();
    if (!successOrdering || !failureOrdering || !scope || !alignment ||
        *alignment == 0 || !::llvm::isPowerOf2_64(*alignment)) {
      cmp.emitError(
          "loom-lower-graph-memory: compare-exchange requires supported "
          "orderings/scope and an explicit power-of-two alignment");
      return false;
    }
    auto contract = ::dataflow::CompareExchangeContractAttr::get(
        ctx.graph.getContext(), *successOrdering, *failureOrdering, *scope,
        *alignment, std::nullopt, cmp.getWeak(), cmp.getVolatile_());
    auto lowered = ::dataflow::CmpXchgOp::create(
        builder, loc, cmp.getCmp().getType(), builder.getI1Type(),
        builder.getNoneType(), mem, address, cmp.getCmp(), cmp.getVal(),
        ctx.ctrl, {}, contract);
    ::mlir::Value packed = ::mlir::LLVM::UndefOp::create(
        builder, loc, cmp.getRes().getType());
    packed = ::mlir::LLVM::InsertValueOp::create(
        builder, loc, packed, lowered.getOld(), ::llvm::ArrayRef<int64_t>{0});
    packed = ::mlir::LLVM::InsertValueOp::create(
        builder, loc, packed, lowered.getSuccess(),
        ::llvm::ArrayRef<int64_t>{1});
    cmp.getRes().replaceAllUsesWith(packed);
  }
  op->erase();
  return true;
}

void eraseDeadPointerAddressing(::dataflow::GraphOp graph) {
  bool changed = true;
  while (changed) {
    changed = false;
    ::llvm::SmallVector<::mlir::LLVM::GEPOp, 8> dead;
    graph.getBody().walk([&](::mlir::LLVM::GEPOp gep) {
      if (gep->use_empty())
        dead.push_back(gep);
    });
    for (::mlir::LLVM::GEPOp gep : dead) {
      gep.erase();
      changed = true;
    }
  }
}

void eraseDeadLocalAllocations(::dataflow::GraphOp graph) {
  ::llvm::SmallVector<::mlir::LLVM::AllocaOp, 8> dead;
  graph.getBody().walk([&](::mlir::LLVM::AllocaOp alloca) {
    if (alloca.getRes().use_empty())
      dead.push_back(alloca);
  });
  for (::mlir::LLVM::AllocaOp alloca : dead)
    alloca.erase();
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

::mlir::LogicalResult rewriteOneGraph(
    ::dataflow::GraphOp graph, ::mlir::OpBuilder &builder,
    ::llvm::SmallVectorImpl<::loom::lowering::GraphMemoryInputSource> *sources =
        nullptr) {
  ::mlir::Value ctrl = getThreadCtrl(graph);
  if (!ctrl)
    return graph.emitError(
        "loom-lower-graph-memory: graph entry has no start token");
  RewriteCtx ctx;
  ctx.graph = graph;
  ctx.ctrl = ctrl;
  auto indexBits = getGraphIndexBits(graph);
  if (!indexBits)
    return graph.emitError(
        "loom-lower-graph-memory: graph has no canonical index width");
  ctx.indexBits = *indexBits;

  if (::mlir::failed(materializeLocalAllocations(ctx, builder)))
    return ::mlir::failure();

  (void)sinkBranchSelectedLoads(graph, builder);

  // Collect rewrite targets up front so the walk is independent of
  // mutations performed by tryRewriteOne.
  ::llvm::SmallVector<::mlir::Operation *, 16> targets;
  graph.getBody().walk([&](::mlir::Operation *op) {
          if (::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                          ::mlir::LLVM::AtomicRMWOp,
                          ::mlir::LLVM::AtomicCmpXchgOp,
                          ::mlir::LLVM::FenceOp>(op))
      targets.push_back(op);
    return ::mlir::WalkResult::advance();
  });

  for (::mlir::Operation *target : targets)
    (void)tryRewriteOne(target, builder, ctx);
  // Root-relative accesses no longer consume their LLVM address expression.
  // Retire that pure pointer chain before region lowering decides which values
  // cross a structured control boundary.
  eraseDeadPointerAddressing(graph);
  eraseDeadLocalAllocations(graph);
  ensurePointerValueServices(ctx);
  return normalizeGraphMemoryPorts(graph, ctx.importedViews, sources);
}

::mlir::LogicalResult checkResidualMemoryEffects(::dataflow::GraphOp graph) {
  ::mlir::WalkResult result =
      graph.getBody().walk(
          [](::mlir::Operation *op) -> ::mlir::WalkResult {
            bool lacksCompletion =
                ::llvm::isa<::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                            ::mlir::LLVM::AtomicRMWOp,
                            ::mlir::LLVM::AtomicCmpXchgOp,
                            ::mlir::LLVM::FenceOp, ::mlir::LLVM::MemcpyOp,
                            ::mlir::LLVM::MemmoveOp, ::mlir::LLVM::MemsetOp,
                            ::mlir::LLVM::AllocaOp>(op);
            if (!lacksCompletion)
              return ::mlir::WalkResult::advance();

            op->emitError()
                << "loom-lower-graph-memory: residual memory operation '"
                << op->getName().getStringRef()
                << "' has no explicit completion event or local-memory "
                   "normalization";
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
    ::llvm::SmallVector<GraphMemoryInputSource, 4> sources;
    if (::mlir::failed(
            rewriteOneGraph(graph, builder, projections ? &sources : nullptr)))
      return ::mlir::failure();
    if (projections)
      projections->push_back({graph, std::move(sources)});
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
