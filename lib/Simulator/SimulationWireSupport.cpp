//===- SimulationWireSupport.cpp - shared wire context and codec ---------===//
//
// Launch-context recovery, typed-key order, and the semantic value, stream,
// and memory-byte codec shared by the workload and runtime-input families.
//
//===----------------------------------------------------------------------===//

#include "SimulationWireInternal.h"

#include "Common/IndexWidth.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

using namespace mlir;

namespace loom::sim::detail {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

//===----------------------------------------------------------------------===//
// Typed key order
//===----------------------------------------------------------------------===//

int compareIdentities(const ::loom::ArtifactIdentity &lhs,
                      const ::loom::ArtifactIdentity &rhs) {
  if (lhs.bytes() < rhs.bytes())
    return -1;
  if (lhs.bytes() > rhs.bytes())
    return 1;
  return 0;
}

static int compareU64(std::uint64_t lhs, std::uint64_t rhs) {
  if (lhs < rhs)
    return -1;
  if (lhs > rhs)
    return 1;
  return 0;
}

int compareRootKeys(const dataflow::LogicalMemoryRootRef &lhs,
                    const dataflow::LogicalMemoryRootRef &rhs) {
  if (int order = compareIdentities(lhs.artifact, rhs.artifact))
    return order;
  return compareU64(lhs.entity.value(), rhs.entity.value());
}

int compareRootOrViewKeys(const dataflow::LogicalMemoryRootOrViewRef &lhs,
                          const dataflow::LogicalMemoryRootOrViewRef &rhs) {
  const unsigned lhsKind =
      std::holds_alternative<dataflow::LogicalMemoryRootRef>(lhs) ? 0 : 1;
  const unsigned rhsKind =
      std::holds_alternative<dataflow::LogicalMemoryRootRef>(rhs) ? 0 : 1;
  if (lhsKind != rhsKind)
    return lhsKind < rhsKind ? -1 : 1;
  if (lhsKind == 0)
    return compareRootKeys(std::get<dataflow::LogicalMemoryRootRef>(lhs),
                           std::get<dataflow::LogicalMemoryRootRef>(rhs));
  const dataflow::LogicalMemoryViewRef &lhsView =
      std::get<dataflow::LogicalMemoryViewRef>(lhs);
  const dataflow::LogicalMemoryViewRef &rhsView =
      std::get<dataflow::LogicalMemoryViewRef>(rhs);
  if (int order = compareRootKeys(lhsView.root, rhsView.root))
    return order;
  return compareU64(lhsView.viewOrdinal, rhsView.viewOrdinal);
}

int compareObservableTargets(const SpatialMemoryObservableTarget &lhs,
                             const SpatialMemoryObservableTarget &rhs) {
  const unsigned lhsKind =
      std::holds_alternative<dataflow::LogicalMemoryRootOrViewRef>(lhs) ? 0 : 1;
  const unsigned rhsKind =
      std::holds_alternative<dataflow::LogicalMemoryRootOrViewRef>(rhs) ? 0 : 1;
  if (lhsKind != rhsKind)
    return lhsKind < rhsKind ? -1 : 1;
  if (lhsKind == 0)
    return compareRootOrViewKeys(
        std::get<dataflow::LogicalMemoryRootOrViewRef>(lhs),
        std::get<dataflow::LogicalMemoryRootOrViewRef>(rhs));
  return compareU64(std::get<MemoryExposureTarget>(lhs).memoryResultOrdinal,
                    std::get<MemoryExposureTarget>(rhs).memoryResultOrdinal);
}

//===----------------------------------------------------------------------===//
// Lane shapes
//===----------------------------------------------------------------------===//

static llvm::Expected<LaneShape> scalarLaneShape(Type type,
                                                 Operation *contextOp) {
  // A none-typed port carries pure signals: every token has zero lanes, so
  // the token count is free and the lane array is empty.
  if (isa<NoneType>(type))
    return LaneShape{0, 0, std::nullopt};
  if (auto integer = dyn_cast<IntegerType>(type))
    return LaneShape{1, integer.getWidth(), std::nullopt};
  if (isa<IndexType>(type)) {
    llvm::Expected<unsigned> width = loom::getIndexBitWidth(contextOp);
    if (!width)
      return width.takeError();
    return LaneShape{1, *width, std::nullopt};
  }
  if (auto floating = dyn_cast<FloatType>(type))
    return LaneShape{1, static_cast<std::uint32_t>(floating.getWidth()),
                     std::nullopt};
  if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type)) {
    llvm::Expected<PointerLayout> layout =
        resolvePointerLayout(contextOp, pointer.getAddressSpace());
    if (!layout)
      return layout.takeError();
    return LaneShape{1, layout->representationBits, *layout};
  }
  return invalid("simulation wire: value port type has no semantic lane "
                 "shape");
}

llvm::Expected<LaneShape> laneShapeOf(Type type, Operation *contextOp) {
  auto vector = dyn_cast<VectorType>(type);
  if (!vector)
    return scalarLaneShape(type, contextOp);
  if (vector.isScalable() || vector.getRank() == 0)
    return invalid(
        "simulation wire: scalable or rank-zero vectors have no lane shape");
  llvm::Expected<LaneShape> element =
      scalarLaneShape(vector.getElementType(), contextOp);
  if (!element)
    return element.takeError();
  std::uint64_t lanes = 1;
  for (std::int64_t dimension : vector.getShape()) {
    if (dimension <= 0 || lanes > std::numeric_limits<std::uint64_t>::max() /
                                      static_cast<std::uint64_t>(dimension))
      return invalid("simulation wire: vector lane count overflow");
    lanes *= static_cast<std::uint64_t>(dimension);
  }
  return LaneShape{lanes, element->laneBitWidth, std::nullopt};
}

//===----------------------------------------------------------------------===//
// Resolved launch context
//===----------------------------------------------------------------------===//

static unsigned threadRankOf(dataflow::ThreadOp thread) {
  Block &entry = thread.getBody().front();
  const unsigned functionInputs = thread.getFunctionType().getNumInputs();
  return entry.getNumArguments() - functionInputs - 1;
}

// Resolve one thread-level memory value bound into a graph launch to the
// imported logical-memory root it names, mirroring the admitted Dataflow
// composition forms on the public view surface: a thread formal, an admitted
// root-preserving view, or an earlier launch's memory-result exposure. A
// fresh allocation root is not imported and yields no binding target.
static llvm::Expected<std::optional<dataflow::LogicalMemoryRootRef>>
importedRootOfValue(
    Value value, dataflow::ThreadOp thread,
    const dataflow::RootedGraphLaunchRef &launch,
    const dataflow::CanonicalDataflowProgramView &view,
    const llvm::DenseMap<std::pair<Operation *, unsigned>,
                         dataflow::LogicalMemoryRootRef> &importedByFormal,
    const llvm::DenseMap<Operation *, dataflow::StaticGraphLaunchRef>
        &staticLaunchByOp,
    const llvm::DenseMap<Operation *, dataflow::LogicalMemoryRootRef>
        &serviceRootByOp,
    const llvm::DenseMap<Operation *, dataflow::LogicalMemoryRootRef>
        &freshRootByOp,
    const std::vector<dataflow::LogicalMemoryRootRef> &freshRoots) {
  while (true) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner()->getParentOp() != thread.getOperation())
        return invalid("simulation wire: launch memory binding escapes its "
                       "thread");
      auto found = importedByFormal.find(
          {thread.getOperation(), argument.getArgNumber()});
      if (found == importedByFormal.end())
        return invalid(
            "simulation wire: launch memory binding is not an imported root");
      return found->second;
    }
    Operation *definition = value.getDefiningOp();
    if (!definition)
      return invalid("simulation wire: unresolved launch memory binding");
    if (auto cast = dyn_cast<memref::CastOp>(definition)) {
      value = cast.getSource();
      continue;
    }
    if (auto found = serviceRootByOp.find(definition);
        found != serviceRootByOp.end())
      return found->second;
    if (auto earlier = dyn_cast<dataflow::GraphLaunchOp>(definition)) {
      auto found = staticLaunchByOp.find(earlier.getOperation());
      if (found == staticLaunchByOp.end())
        return invalid("simulation wire: launch memory result has no static "
                       "launch entity");
      std::optional<std::uint64_t> memoryResultOrdinal;
      std::uint64_t ordinal = 0;
      for (Value result : earlier.getMemoryResults()) {
        if (result == value) {
          memoryResultOrdinal = ordinal;
          break;
        }
        ++ordinal;
      }
      if (!memoryResultOrdinal)
        return invalid(
            "simulation wire: value result is not a memory capability");
      llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> exposure =
          view.resolveExposure(dataflow::MemoryExposureRef{
              dataflow::RootedGraphLaunchRef{launch.rootThreadLaunch,
                                             found->second},
              *memoryResultOrdinal});
      if (!exposure)
        return exposure.takeError();
      const dataflow::LogicalMemoryRootRef resolved =
          std::holds_alternative<dataflow::LogicalMemoryRootRef>(*exposure)
              ? std::get<dataflow::LogicalMemoryRootRef>(*exposure)
              : std::get<dataflow::LogicalMemoryViewRef>(*exposure).root;
      // A fresh allocation reached through an exposure chain is still not an
      // imported runtime object: it never receives a binding.
      if (std::binary_search(freshRoots.begin(), freshRoots.end(), resolved,
                             [](const auto &lhs, const auto &rhs) {
                               return compareRootKeys(lhs, rhs) < 0;
                             }))
        return std::nullopt;
      return resolved;
    }
    // A fresh allocation root is not an imported runtime object; it receives
    // no binding. Anything else was already rejected by Dataflow
    // finalization, so fail closed.
    if (freshRootByOp.contains(definition))
      return std::nullopt;
    return invalid("simulation wire: unresolved launch memory binding");
  }
}

llvm::Expected<ResolvedLaunchContext>
resolveLaunchContext(const dataflow::CanonicalDataflowProgramView &view,
                     const dataflow::RootedGraphLaunchRef &launch) {
  llvm::Expected<dataflow::GraphRef> graph = view.resolve(launch);
  if (!graph)
    return graph.takeError();
  ResolvedLaunchContext context{*graph};
  llvm::Expected<dataflow::CanonicalGraphView> graphView = view.resolve(*graph);
  if (!graphView)
    return graphView.takeError();
  context.graphOp = cast<dataflow::GraphOp>(graphView->op);
  llvm::Expected<dataflow::CanonicalRootThreadLaunchView> rootView =
      view.resolve(launch.rootThreadLaunch);
  if (!rootView)
    return rootView.takeError();
  context.thread = cast<dataflow::ThreadOp>(rootView->callee);
  context.rootLaunchOp = cast<dataflow::ThreadLaunchOp>(rootView->op);
  llvm::Expected<dataflow::CanonicalStaticGraphLaunchView> staticView =
      view.resolve(launch.staticGraphLaunch);
  if (!staticView)
    return staticView.takeError();
  context.graphLaunchOp = cast<dataflow::GraphLaunchOp>(staticView->op);

  llvm::ArrayRef<std::int32_t> inSeg = context.graphOp.getInputSegmentSizes();
  llvm::ArrayRef<std::int32_t> outSeg = context.graphOp.getResultSegmentSizes();
  context.numValueInputs = static_cast<std::uint64_t>(inSeg[0]);
  context.numStreamInputs = static_cast<std::uint64_t>(inSeg[1]);
  context.numValueResults = static_cast<std::uint64_t>(outSeg[0]);
  context.numStreamOutputs = context.graphLaunchOp.getStreamOutputs().size();
  context.threadRank = threadRankOf(context.thread);

  mlir::TypeRange inputs = context.graphOp.getFunctionType().getInputs();
  for (std::uint64_t i = 0; i < context.numValueInputs; ++i) {
    llvm::Expected<LaneShape> shape =
        laneShapeOf(inputs[i], context.graphOp.getOperation());
    if (!shape)
      return shape.takeError();
    context.valueInputShapes.push_back(*shape);
  }
  for (std::uint64_t k = 0; k < context.numStreamInputs; ++k) {
    llvm::Expected<LaneShape> shape = laneShapeOf(
        inputs[context.numValueInputs + k], context.graphOp.getOperation());
    if (!shape)
      return shape.takeError();
    context.streamInputShapes.push_back(*shape);
  }
  mlir::TypeRange results = context.graphOp.getFunctionType().getResults();
  for (std::uint64_t i = 0; i < context.numValueResults; ++i) {
    llvm::Expected<LaneShape> shape =
        laneShapeOf(results[i], context.graphOp.getOperation());
    if (!shape)
      return shape.takeError();
    context.valueResultShapes.push_back(*shape);
  }
  for (std::uint64_t k = 0; k < context.numStreamOutputs; ++k) {
    llvm::Expected<LaneShape> shape = laneShapeOf(
        results[context.numValueResults + k], context.graphOp.getOperation());
    if (!shape)
      return shape.takeError();
    context.streamOutputShapes.push_back(*shape);
  }

  // Enumerate the imported roots reachable through the graph memory-input
  // bindings, composing only public view resolutions.
  llvm::DenseMap<std::pair<Operation *, unsigned>,
                 dataflow::LogicalMemoryRootRef>
      importedByFormal;
  llvm::DenseMap<Operation *, dataflow::LogicalMemoryRootRef> serviceRootByOp;
  llvm::DenseMap<Operation *, dataflow::LogicalMemoryRootRef> freshRootByOp;
  std::vector<dataflow::LogicalMemoryRootRef> freshRoots;
  for (const dataflow::CanonicalLogicalMemoryRootView &root :
       view.logicalMemoryRoots()) {
    if (root.formalArgIndex) {
      importedByFormal.try_emplace(
          std::make_pair(root.op, *root.formalArgIndex), root.ref);
      continue;
    }
    if (isa<dataflow::MemoryServiceOp>(root.op)) {
      serviceRootByOp.try_emplace(root.op, root.ref);
      continue;
    }
    freshRootByOp.try_emplace(root.op, root.ref);
    freshRoots.push_back(root.ref);
  }
  std::sort(freshRoots.begin(), freshRoots.end(),
            [](const auto &lhs, const auto &rhs) {
              return compareRootKeys(lhs, rhs) < 0;
            });
  llvm::DenseMap<Operation *, dataflow::StaticGraphLaunchRef> staticLaunchByOp;
  for (const dataflow::CanonicalStaticGraphLaunchView &site :
       view.staticGraphLaunches())
    staticLaunchByOp.try_emplace(site.op, site.ref);

  std::vector<dataflow::LogicalMemoryRootRef> roots;
  for (Value binding : context.graphLaunchOp.getMemoryInputs()) {
    llvm::Expected<std::optional<dataflow::LogicalMemoryRootRef>> root =
        importedRootOfValue(binding, context.thread, launch, view,
                            importedByFormal, staticLaunchByOp, serviceRootByOp,
                            freshRootByOp, freshRoots);
    if (!root)
      return root.takeError();
    context.memoryInputRoots.push_back(*root);
    if (*root)
      roots.push_back(**root);
  }
  std::sort(roots.begin(), roots.end(), [](const auto &lhs, const auto &rhs) {
    return compareRootKeys(lhs, rhs) < 0;
  });
  roots.erase(std::unique(roots.begin(), roots.end(),
                          [](const auto &lhs, const auto &rhs) {
                            return compareRootKeys(lhs, rhs) == 0;
                          }),
              roots.end());
  context.importedRoots = std::move(roots);
  // Direct memory observables may also name a fresh allocation owned by the
  // called graph; roots of any other graph or thread are unreachable.
  context.observableRoots = context.importedRoots;
  for (const auto &[op, root] : freshRootByOp)
    if (op->getParentOfType<dataflow::GraphOp>() == context.graphOp)
      context.observableRoots.push_back(root);
  std::sort(context.observableRoots.begin(), context.observableRoots.end(),
            [](const auto &lhs, const auto &rhs) {
              return compareRootKeys(lhs, rhs) < 0;
            });
  return context;
}

//===----------------------------------------------------------------------===//
// Semantic value, stream, and memory-byte validation and codec
//===----------------------------------------------------------------------===//

llvm::Error validateValueSequence(const CanonicalValueSequence &sequence,
                                  const LaneShape &shape,
                                  const llvm::Twine &what,
                                  std::optional<std::uint64_t> objectCount) {
  if (shape.lanesPerToken == 0) {
    // None tokens: the token count is free and the lane array is empty.
    if (!sequence.lanes.empty())
      return invalid(what + ": none tokens carry no lanes");
    return llvm::Error::success();
  }
  if (sequence.lanes.size() % shape.lanesPerToken != 0 ||
      sequence.lanes.size() / shape.lanesPerToken != sequence.tokenCount)
    return invalid(what + ": lane count is not token count times lanes per "
                          "token");
  for (const SemanticLane &lane : sequence.lanes) {
    if (static_cast<std::uint32_t>(lane.state) >
        static_cast<std::uint32_t>(SemanticState::Undef))
      return invalid(what + ": lane state is out of domain");
    if (lane.state == SemanticState::Defined) {
      if (lane.bits.getBitWidth() != shape.laneBitWidth)
        return invalid(what + ": defined lane width does not match the "
                              "target type");
      if (shape.pointerLayout) {
        if (!lane.pointerTarget)
          return invalid(what + ": defined pointer lane has no object target");
        if (!objectCount)
          return invalid(what +
                         ": pointer lane requires the runtime object registry");
        if (lane.pointerTarget->objectOrdinal >= *objectCount)
          return invalid(what + ": pointer object ordinal is out of range");
        if (lane.pointerTarget->byteOffset.getBitWidth() !=
            shape.pointerLayout->addressBits)
          return invalid(what +
                         ": pointer byte-offset width does not match A(AS)");
      } else if (lane.pointerTarget) {
        return invalid(what + ": non-pointer lane carries a pointer target");
      }
      continue;
    }
    // A non-Defined lane carries no payload; any hidden bits would be
    // silently dropped by the encoder, so reject them instead.
    if (lane.bits.getBitWidth() != 1 || !lane.bits.isZero())
      return invalid(what + ": a non-defined lane carries hidden payload "
                            "bits");
    if (lane.pointerTarget)
      return invalid(what +
                     ": a non-defined pointer lane carries an object target");
  }
  return llvm::Error::success();
}

static void encodeBits(WireWriter &writer, const llvm::APInt &bits) {
  const unsigned width = bits.getBitWidth();
  const unsigned byteCount = (width + 7) / 8;
  for (unsigned index = 0; index < byteCount; ++index) {
    const unsigned low = (byteCount - 1 - index) * 8;
    const unsigned chunk = std::min(8u, width - low);
    writer.bytes(
        {static_cast<std::uint8_t>(bits.extractBitsAsZExtValue(chunk, low))});
  }
}

static llvm::Expected<llvm::APInt>
decodeBits(WireReader &reader, unsigned width, const llvm::Twine &what) {
  const unsigned byteCount = (width + 7) / 8;
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> raw = reader.bytes(byteCount);
  if (!raw)
    return raw.takeError();
  const unsigned padding = byteCount * 8 - width;
  if (padding > 0 && ((*raw)[0] >> (8 - padding)) != 0)
    return invalid(what + ": noncanonical padding bits");
  llvm::APInt bits(width, 0);
  for (std::uint8_t byte : *raw) {
    bits <<= 8;
    bits |= byte;
  }
  return bits;
}

void encodeValueSequence(WireWriter &writer,
                         const CanonicalValueSequence &sequence,
                         const LaneShape &shape) {
  writer.u64(sequence.tokenCount);
  writer.u64(sequence.lanes.size());
  for (const SemanticLane &lane : sequence.lanes) {
    writer.u32(static_cast<std::uint32_t>(lane.state));
    if (lane.state != SemanticState::Defined)
      continue;
    encodeBits(writer, lane.bits);
    if (shape.pointerLayout) {
      assert(lane.pointerTarget &&
             "validated defined pointer lane has no target");
      writer.u64(lane.pointerTarget->objectOrdinal);
      encodeBits(writer, lane.pointerTarget->byteOffset);
    }
  }
}

llvm::Expected<CanonicalValueSequence>
decodeValueSequence(WireReader &reader, const LaneShape &shape) {
  CanonicalValueSequence sequence;
  llvm::Expected<std::uint64_t> tokenCount = reader.u64();
  if (!tokenCount)
    return tokenCount.takeError();
  llvm::Expected<std::uint64_t> laneCount = reader.u64();
  if (!laneCount)
    return laneCount.takeError();
  if (shape.lanesPerToken == 0) {
    if (*laneCount != 0)
      return invalid("simulation wire: none tokens carry no lanes");
    sequence.tokenCount = *tokenCount;
    return sequence;
  }
  if (*laneCount % shape.lanesPerToken != 0 ||
      *laneCount / shape.lanesPerToken != *tokenCount)
    return invalid("simulation wire: lane count is not token count times "
                   "lanes per token");
  if (llvm::Error error = reader.guardCount(*laneCount, 4))
    return std::move(error);
  sequence.tokenCount = *tokenCount;
  sequence.lanes.reserve(*laneCount);
  for (std::uint64_t index = 0; index < *laneCount; ++index) {
    llvm::Expected<std::uint32_t> tag = reader.u32();
    if (!tag)
      return tag.takeError();
    if (*tag > static_cast<std::uint32_t>(SemanticState::Undef))
      return invalid("simulation wire: unknown lane state");
    const SemanticState state = static_cast<SemanticState>(*tag);
    if (state != SemanticState::Defined) {
      sequence.lanes.push_back(state == SemanticState::Poison
                                   ? SemanticLane::poison()
                                   : SemanticLane::undef());
      continue;
    }
    llvm::Expected<llvm::APInt> bits =
        decodeBits(reader, shape.laneBitWidth, "simulation wire: defined lane");
    if (!bits)
      return bits.takeError();
    if (!shape.pointerLayout) {
      sequence.lanes.push_back(SemanticLane::defined(std::move(*bits)));
      continue;
    }
    llvm::Expected<std::uint64_t> objectOrdinal = reader.u64();
    if (!objectOrdinal)
      return objectOrdinal.takeError();
    llvm::Expected<llvm::APInt> byteOffset =
        decodeBits(reader, shape.pointerLayout->addressBits,
                   "simulation wire: pointer byte offset");
    if (!byteOffset)
      return byteOffset.takeError();
    sequence.lanes.push_back(SemanticLane::definedPointer(
        std::move(*bits), *objectOrdinal, std::move(*byteOffset)));
  }
  return sequence;
}

void encodeStreamSequence(WireWriter &writer,
                          const CanonicalStreamSequence &sequence,
                          const LaneShape &shape) {
  encodeValueSequence(writer, sequence.values, shape);
  writer.u32(static_cast<std::uint32_t>(sequence.termination));
}

llvm::Expected<CanonicalStreamSequence>
decodeStreamSequence(WireReader &reader, const LaneShape &shape) {
  CanonicalStreamSequence sequence;
  llvm::Expected<CanonicalValueSequence> values =
      decodeValueSequence(reader, shape);
  if (!values)
    return values.takeError();
  sequence.values = std::move(*values);
  llvm::Expected<std::uint32_t> tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag > static_cast<std::uint32_t>(StreamTermination::OpenAfterLast))
    return invalid("simulation wire: unknown stream termination");
  sequence.termination = static_cast<StreamTermination>(*tag);
  return sequence;
}

void encodeMemoryObject(WireWriter &writer, const RuntimeMemoryObject &object) {
  writer.u64(object.initialBytes.size());
  writer.u64(object.initialBytes.size());
  for (const SemanticMemoryByte &byte : object.initialBytes) {
    writer.u32(static_cast<std::uint32_t>(byte.state));
    if (byte.state == SemanticState::Defined)
      writer.bytes({byte.value});
  }
  writer.u64(object.pointerValues.size());
  for (const RuntimeMemoryPointer &pointer : object.pointerValues) {
    writer.u64(pointer.storageByteOffset);
    writer.u32(pointer.addressSpace);
    writer.u64(pointer.target.objectOrdinal);
    encodeBits(writer, pointer.target.byteOffset);
  }
}

llvm::Expected<RuntimeMemoryObject> decodeMemoryObject(WireReader &reader,
                                                       Operation *scope) {
  RuntimeMemoryObject object;
  llvm::Expected<std::uint64_t> byteCount = reader.u64();
  if (!byteCount)
    return byteCount.takeError();
  llvm::Expected<std::uint64_t> arrayCount = reader.u64();
  if (!arrayCount)
    return arrayCount.takeError();
  if (*arrayCount != *byteCount)
    return invalid(
        "simulation wire: memory object byte count does not match its "
        "initial-byte array");
  if (llvm::Error error = reader.guardCount(*byteCount, 4))
    return std::move(error);
  object.initialBytes.reserve(*byteCount);
  for (std::uint64_t index = 0; index < *byteCount; ++index) {
    llvm::Expected<std::uint32_t> tag = reader.u32();
    if (!tag)
      return tag.takeError();
    if (*tag > static_cast<std::uint32_t>(SemanticState::Undef))
      return invalid("simulation wire: unknown memory-byte state");
    SemanticMemoryByte byte;
    byte.state = static_cast<SemanticState>(*tag);
    if (byte.state == SemanticState::Defined) {
      llvm::Expected<llvm::ArrayRef<std::uint8_t>> raw = reader.bytes(1);
      if (!raw)
        return raw.takeError();
      byte.value = (*raw)[0];
    }
    object.initialBytes.push_back(byte);
  }
  llvm::Expected<std::uint64_t> pointerCount = reader.u64();
  if (!pointerCount)
    return pointerCount.takeError();
  if (llvm::Error error = reader.guardCount(*pointerCount, 28))
    return std::move(error);
  object.pointerValues.reserve(*pointerCount);
  for (std::uint64_t index = 0; index < *pointerCount; ++index) {
    llvm::Expected<std::uint64_t> storageByteOffset = reader.u64();
    if (!storageByteOffset)
      return storageByteOffset.takeError();
    llvm::Expected<std::uint32_t> addressSpace = reader.u32();
    if (!addressSpace)
      return addressSpace.takeError();
    llvm::Expected<PointerLayout> layout =
        resolvePointerLayout(scope, *addressSpace);
    if (!layout)
      return layout.takeError();
    llvm::Expected<std::uint64_t> objectOrdinal = reader.u64();
    if (!objectOrdinal)
      return objectOrdinal.takeError();
    llvm::Expected<llvm::APInt> byteOffset =
        decodeBits(reader, layout->addressBits,
                   "simulation wire: memory pointer byte offset");
    if (!byteOffset)
      return byteOffset.takeError();
    object.pointerValues.push_back(RuntimeMemoryPointer{
        *storageByteOffset, *addressSpace,
        PointerTarget{*objectOrdinal, std::move(*byteOffset)}});
  }
  return object;
}

llvm::Expected<llvm::APInt>
decodeRuntimePointerRepresentation(const RuntimeMemoryObject &object,
                                   const RuntimeMemoryPointer &pointer,
                                   Operation *scope) {
  llvm::Expected<PointerLayout> layout =
      resolvePointerLayout(scope, pointer.addressSpace);
  if (!layout)
    return layout.takeError();
  if (layout->representationBits % 8 != 0)
    return invalid("simulation runtime input: pointer representation is not "
                   "byte-addressable");
  const std::uint64_t byteCount = layout->representationBits / 8;
  if (pointer.storageByteOffset > object.initialBytes.size() ||
      byteCount > object.initialBytes.size() - pointer.storageByteOffset)
    return invalid("simulation runtime input: stored pointer is out of range");
  llvm::Expected<llvm::DataLayout> dataLayout = resolveLLVMDataLayout(scope);
  if (!dataLayout)
    return dataLayout.takeError();
  llvm::APInt bits(layout->representationBits, 0);
  for (std::uint64_t index = 0; index < byteCount; ++index) {
    const SemanticMemoryByte &byte =
        object.initialBytes[pointer.storageByteOffset + index];
    if (byte.state != SemanticState::Defined)
      return invalid("simulation runtime input: stored pointer contains an "
                     "exceptional memory byte");
    const std::uint64_t semanticByte =
        dataLayout->isLittleEndian() ? index : byteCount - 1 - index;
    bits.insertBits(llvm::APInt(8, byte.value), semanticByte * 8);
  }
  return bits;
}

namespace {

llvm::Expected<llvm::APInt> checkedAddressAdd(const llvm::APInt &lhs,
                                              std::uint64_t rhs,
                                              const llvm::Twine &what) {
  llvm::APInt wide(64, rhs);
  if (wide.getActiveBits() > lhs.getBitWidth())
    return invalid(what + ": runtime object extent exceeds A(AS)");
  bool overflow = false;
  llvm::APInt sum = lhs.uadd_ov(wide.zextOrTrunc(lhs.getBitWidth()), overflow);
  if (overflow)
    return invalid(what + ": canonical object address space is exhausted");
  return sum;
}

llvm::Expected<llvm::APInt>
canonicalObjectBase(llvm::ArrayRef<RuntimeMemoryObject> objects,
                    std::uint64_t objectOrdinal, const PointerLayout &layout) {
  if (objectOrdinal >= objects.size())
    return invalid("simulation runtime input: pointer object ordinal is out "
                   "of range");

  // Zero remains distinct from every object. One address after each finite
  // object is reserved for its one-past pointer before the next base.
  llvm::APInt base(layout.addressBits, 1);
  for (std::uint64_t ordinal = 0; ordinal < objectOrdinal; ++ordinal) {
    llvm::Expected<llvm::APInt> afterObject = checkedAddressAdd(
        base, objects[ordinal].initialBytes.size(), "simulation runtime input");
    if (!afterObject)
      return afterObject.takeError();
    llvm::Expected<llvm::APInt> next =
        checkedAddressAdd(*afterObject, 1, "simulation runtime input");
    if (!next)
      return next.takeError();
    base = std::move(*next);
  }
  llvm::Expected<llvm::APInt> onePast =
      checkedAddressAdd(base, objects[objectOrdinal].initialBytes.size(),
                        "simulation runtime input");
  if (!onePast)
    return onePast.takeError();
  return base;
}

llvm::Expected<llvm::APInt>
canonicalPointerRepresentation(llvm::ArrayRef<RuntimeMemoryObject> objects,
                               const PointerTarget &target,
                               const PointerLayout &layout) {
  if (layout.kind != PointerLayoutKind::StableIntegral)
    return invalid("simulation runtime input: pointer format has no canonical "
                   "stable-integral codec");
  if (target.byteOffset.getBitWidth() != layout.addressBits)
    return invalid("simulation runtime input: pointer byte-offset width does "
                   "not match A(AS)");
  llvm::Expected<llvm::APInt> base =
      canonicalObjectBase(objects, target.objectOrdinal, layout);
  if (!base)
    return base.takeError();
  llvm::APInt low = *base + target.byteOffset;
  return low.zext(layout.representationBits);
}

llvm::Error writePointerRepresentation(RuntimeMemoryObject &object,
                                       const RuntimeMemoryPointer &pointer,
                                       const llvm::APInt &representation,
                                       Operation *scope) {
  llvm::Expected<llvm::DataLayout> dataLayout = resolveLLVMDataLayout(scope);
  if (!dataLayout)
    return dataLayout.takeError();
  const std::uint64_t byteCount = representation.getBitWidth() / 8;
  for (std::uint64_t index = 0; index < byteCount; ++index) {
    const std::uint64_t semanticByte =
        dataLayout->isLittleEndian() ? index : byteCount - 1 - index;
    object.initialBytes[pointer.storageByteOffset + index] = {
        SemanticState::Defined,
        static_cast<std::uint8_t>(
            representation.extractBitsAsZExtValue(8, semanticByte * 8))};
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error validateRuntimeMemoryObjectStructure(
    llvm::ArrayRef<RuntimeMemoryObject> objects, Operation *scope) {
  for (const RuntimeMemoryObject &object : objects) {
    if (object.initialBytes.empty())
      return invalid("simulation runtime input: empty memory object");
    for (const SemanticMemoryByte &byte : object.initialBytes) {
      if (static_cast<std::uint32_t>(byte.state) >
          static_cast<std::uint32_t>(SemanticState::Undef))
        return invalid(
            "simulation runtime input: memory-byte state is out of domain");
      if (byte.state != SemanticState::Defined && byte.value != 0)
        return invalid("simulation runtime input: a non-defined memory byte "
                       "carries a hidden value");
    }
    std::uint64_t previousEnd = 0;
    for (std::size_t index = 0; index < object.pointerValues.size(); ++index) {
      const RuntimeMemoryPointer &pointer = object.pointerValues[index];
      llvm::Expected<PointerLayout> layout =
          resolvePointerLayout(scope, pointer.addressSpace);
      if (!layout)
        return layout.takeError();
      if (layout->kind != PointerLayoutKind::StableIntegral)
        return invalid("simulation runtime input: stored pointer format has "
                       "no canonical stable-integral codec");
      if (pointer.target.objectOrdinal >= objects.size())
        return invalid("simulation runtime input: stored pointer object "
                       "ordinal is out of range");
      if (pointer.target.byteOffset.getBitWidth() != layout->addressBits)
        return invalid("simulation runtime input: stored pointer byte-offset "
                       "width does not match A(AS)");
      if (layout->representationBits % 8 != 0)
        return invalid("simulation runtime input: pointer representation is "
                       "not byte-addressable");
      const std::uint64_t byteCount = layout->representationBits / 8;
      if (pointer.storageByteOffset > object.initialBytes.size() ||
          byteCount > object.initialBytes.size() - pointer.storageByteOffset)
        return invalid(
            "simulation runtime input: stored pointer is out of range");
      if (index != 0 && pointer.storageByteOffset < previousEnd)
        return invalid("simulation runtime input: stored pointer table is not "
                       "sorted or contains overlapping records");
      previousEnd = pointer.storageByteOffset + byteCount;
      llvm::Expected<llvm::APInt> representation =
          decodeRuntimePointerRepresentation(object, pointer, scope);
      if (!representation)
        return representation.takeError();
    }
  }
  return llvm::Error::success();
}

llvm::Error canonicalizeRuntimeMemoryPointers(
    llvm::MutableArrayRef<RuntimeMemoryObject> objects, Operation *scope) {
  if (llvm::Error error = validateRuntimeMemoryObjectStructure(objects, scope))
    return error;
  for (RuntimeMemoryObject &object : objects) {
    for (const RuntimeMemoryPointer &pointer : object.pointerValues) {
      llvm::Expected<PointerLayout> layout =
          resolvePointerLayout(scope, pointer.addressSpace);
      if (!layout)
        return layout.takeError();
      llvm::Expected<llvm::APInt> representation =
          canonicalPointerRepresentation(objects, pointer.target, *layout);
      if (!representation)
        return representation.takeError();
      if (llvm::Error error = writePointerRepresentation(
              object, pointer, *representation, scope))
        return error;
    }
  }
  return llvm::Error::success();
}

llvm::Error canonicalizePointerValueSequence(
    CanonicalValueSequence &sequence, const LaneShape &shape,
    llvm::ArrayRef<RuntimeMemoryObject> objects, Operation *scope) {
  if (!shape.pointerLayout)
    return llvm::Error::success();
  for (SemanticLane &lane : sequence.lanes) {
    if (lane.state != SemanticState::Defined)
      continue;
    if (!lane.pointerTarget)
      return invalid("simulation runtime input: defined pointer lane has no "
                     "object target");
    llvm::Expected<llvm::APInt> representation = canonicalPointerRepresentation(
        objects, *lane.pointerTarget, *shape.pointerLayout);
    if (!representation)
      return representation.takeError();
    lane.bits = std::move(*representation);
  }
  return llvm::Error::success();
}

llvm::Error validateCanonicalPointerValueSequence(
    const CanonicalValueSequence &sequence, const LaneShape &shape,
    llvm::ArrayRef<RuntimeMemoryObject> objects, Operation *scope,
    const llvm::Twine &what) {
  if (!shape.pointerLayout)
    return llvm::Error::success();
  for (const SemanticLane &lane : sequence.lanes) {
    if (lane.state != SemanticState::Defined)
      continue;
    assert(lane.pointerTarget &&
           "validated defined pointer lane has no object target");
    llvm::Expected<llvm::APInt> expected = canonicalPointerRepresentation(
        objects, *lane.pointerTarget, *shape.pointerLayout);
    if (!expected)
      return expected.takeError();
    if (lane.bits != *expected)
      return invalid(what + ": pointer bits are not the canonical object "
                            "projection");
  }
  return llvm::Error::success();
}

llvm::Error
validateRuntimeMemoryObjects(llvm::ArrayRef<RuntimeMemoryObject> objects,
                             Operation *scope) {
  if (llvm::Error error = validateRuntimeMemoryObjectStructure(objects, scope))
    return error;
  for (const RuntimeMemoryObject &object : objects) {
    for (const RuntimeMemoryPointer &pointer : object.pointerValues) {
      llvm::Expected<PointerLayout> layout =
          resolvePointerLayout(scope, pointer.addressSpace);
      if (!layout)
        return layout.takeError();
      llvm::Expected<llvm::APInt> actual =
          decodeRuntimePointerRepresentation(object, pointer, scope);
      if (!actual)
        return actual.takeError();
      llvm::Expected<llvm::APInt> expected =
          canonicalPointerRepresentation(objects, pointer.target, *layout);
      if (!expected)
        return expected.takeError();
      if (*actual != *expected)
        return invalid("simulation runtime input: stored pointer bytes are not "
                       "the canonical object projection");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::DenseMap<std::uint64_t, std::uint64_t>>
deriveCanonicalObjectOrdinals(
    llvm::ArrayRef<RuntimeObjectBindingKey> bindings) {
  struct ObjectGroup {
    std::uint64_t authorObject = 0;
    std::vector<std::vector<std::uint8_t>> key;
  };

  llvm::DenseMap<std::uint64_t, std::size_t> groupIndex;
  std::vector<ObjectGroup> groups;
  for (const RuntimeObjectBindingKey &binding : bindings) {
    auto [position, inserted] =
        groupIndex.try_emplace(binding.authorObject, groups.size());
    if (inserted)
      groups.push_back(ObjectGroup{binding.authorObject, {}});
    groups[position->second].key.push_back(binding.targetAndOffset);
  }

  auto compareGroupKeys = [](const ObjectGroup &lhs, const ObjectGroup &rhs) {
    const std::size_t shared = std::min(lhs.key.size(), rhs.key.size());
    for (std::size_t index = 0; index < shared; ++index) {
      if (lhs.key[index] == rhs.key[index])
        continue;
      return std::lexicographical_compare(
                 lhs.key[index].begin(), lhs.key[index].end(),
                 rhs.key[index].begin(), rhs.key[index].end())
                 ? -1
                 : 1;
    }
    if (lhs.key.size() == rhs.key.size())
      return 0;
    return lhs.key.size() < rhs.key.size() ? -1 : 1;
  };

  std::vector<std::size_t> order(groups.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return compareGroupKeys(groups[lhs], groups[rhs]) < 0;
  });
  llvm::DenseMap<std::uint64_t, std::uint64_t> canonical;
  for (std::size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
    if (ordinal > 0 && compareGroupKeys(groups[order[ordinal - 1]],
                                        groups[order[ordinal]]) == 0)
      return invalid("simulation runtime input: two objects share one "
                     "canonical binding key");
    canonical[groups[order[ordinal]].authorObject] = ordinal;
  }
  return canonical;
}

} // namespace loom::sim::detail
