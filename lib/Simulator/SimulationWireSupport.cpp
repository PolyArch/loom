//===- SimulationWireSupport.cpp - shared wire context and codec ---------===//
//
// Launch-context recovery, typed-key order, and the semantic value, stream,
// and memory-byte codec shared by the workload and runtime-input families.
//
//===----------------------------------------------------------------------===//

#include "SimulationWireInternal.h"

#include "Common/IndexWidth.h"

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
    return LaneShape{0, 0};
  if (auto integer = dyn_cast<IntegerType>(type))
    return LaneShape{1, integer.getWidth()};
  if (isa<IndexType>(type)) {
    llvm::Expected<unsigned> width = loom::getIndexBitWidth(contextOp);
    if (!width)
      return width.takeError();
    return LaneShape{1, *width};
  }
  if (auto floating = dyn_cast<FloatType>(type))
    return LaneShape{1, static_cast<std::uint32_t>(floating.getWidth())};
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
  return LaneShape{lanes, element->laneBitWidth};
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
  llvm::DenseMap<Operation *, dataflow::LogicalMemoryRootRef> freshRootByOp;
  std::vector<dataflow::LogicalMemoryRootRef> freshRoots;
  for (const dataflow::CanonicalLogicalMemoryRootView &root :
       view.logicalMemoryRoots()) {
    if (root.formalArgIndex) {
      importedByFormal.try_emplace(
          std::make_pair(root.op, *root.formalArgIndex), root.ref);
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
                            importedByFormal, staticLaunchByOp, freshRootByOp,
                            freshRoots);
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
                                  const llvm::Twine &what) {
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
      continue;
    }
    // A non-Defined lane carries no payload; any hidden bits would be
    // silently dropped by the encoder, so reject them instead.
    if (lane.bits.getBitWidth() != 1 || !lane.bits.isZero())
      return invalid(what + ": a non-defined lane carries hidden payload "
                            "bits");
  }
  return llvm::Error::success();
}

void encodeValueSequence(WireWriter &writer,
                         const CanonicalValueSequence &sequence) {
  writer.u64(sequence.tokenCount);
  writer.u64(sequence.lanes.size());
  for (const SemanticLane &lane : sequence.lanes) {
    writer.u32(static_cast<std::uint32_t>(lane.state));
    if (lane.state != SemanticState::Defined)
      continue;
    const unsigned width = lane.bits.getBitWidth();
    const unsigned byteCount = (width + 7) / 8;
    for (unsigned index = 0; index < byteCount; ++index) {
      const unsigned low = (byteCount - 1 - index) * 8;
      const unsigned bits = std::min(8u, width - low);
      writer.bytes({static_cast<std::uint8_t>(
          lane.bits.extractBitsAsZExtValue(bits, low))});
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
  const unsigned byteCount = (shape.laneBitWidth + 7) / 8;
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
    llvm::Expected<llvm::ArrayRef<std::uint8_t>> raw = reader.bytes(byteCount);
    if (!raw)
      return raw.takeError();
    const unsigned padding = byteCount * 8 - shape.laneBitWidth;
    if (padding > 0 && ((*raw)[0] >> (8 - padding)) != 0)
      return invalid("simulation wire: noncanonical defined-lane padding bits");
    llvm::APInt bits(shape.laneBitWidth, 0);
    for (std::uint8_t byte : *raw) {
      bits <<= 8;
      bits |= byte;
    }
    sequence.lanes.push_back(SemanticLane::defined(std::move(bits)));
  }
  return sequence;
}

void encodeStreamSequence(WireWriter &writer,
                          const CanonicalStreamSequence &sequence) {
  encodeValueSequence(writer, sequence.values);
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
}

llvm::Expected<RuntimeMemoryObject> decodeMemoryObject(WireReader &reader) {
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
  return object;
}

llvm::Error
validateRuntimeMemoryObjects(llvm::ArrayRef<RuntimeMemoryObject> objects) {
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
