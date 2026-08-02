//===- DataflowStructuralView.cpp - closed structural references ---------===//
//
// Generation, validation, and resolution of the Dataflow-owned closed
// structural-reference catalog (DataflowStructuralRefs.h) on the imported
// CanonicalDataflowProgramView. Every inventory and derived relation is built
// once, from an already-computed canonical labeling: rooted launches from a
// grouped thread-owner inventory, the intra-graph software edge relation from
// SSA def-use, per-thread channel endpoint counts, the channel multicast
// relation from the one shared whole-program channel-topology owner, and the
// static memory composition (thread-formal and fresh-allocation roots,
// root-preserving views, and per-static-site launch exposures). Static transfer
// events remain exact aliases of their Dataflow-owned producer or consumer
// terminals and never receive a separate event identity. Hot queries take
// direct owner-slot and ordinal offsets and collision-free typed keys; no query
// walks MLIR or lossily packs an index.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowInterfaces.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowServiceSchema.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>

using namespace mlir;

namespace dataflow {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

bool isMemoryCapability(Type type) {
  return DataflowDialect::isMemoryCapabilityType(type) ||
         DataflowDialect::containsMemoryCapability(type);
}

// The viewed base of a root-preserving memory view op, or a null Value.
Value viewBase(Operation *op) {
  if (auto cast = dyn_cast<memref::CastOp>(op))
    return cast.getOperand();
  return Value();
}

// The number of value-typed thread body inputs, excluding channel handles and
// memory capabilities.
unsigned threadValueInputCount(ThreadOp thread) {
  unsigned count = 0;
  for (Type input : thread.getFunctionType().getInputs())
    if (!isa<ChannelType>(input) && !isMemoryCapability(input))
      ++count;
  return count;
}

// Producer key discriminators for the collision-free typed range maps.
enum GraphProducerKind {
  kActorResult = 0,
  kGraphStart = 1,
  kGraphValueInput = 2,
  kGraphStreamInput = 3,
};
enum ChannelProducerKind { kSend = 0, kStreamOutput = 1 };

// The canonical sort key of an intra-graph software-edge consumer endpoint.
std::tuple<int, std::uint64_t, std::uint64_t>
consumerKey(const CanonicalGraphConsumerEndpointRef &c) {
  return std::visit(
      Overloaded{
          [](const ActorTokenOperandRef &o) {
            return std::tuple<int, std::uint64_t, std::uint64_t>{
                0, o.actor.entity.value(), o.ordinal};
          },
          [](const GraphEgressTokenRef &e) {
            return std::visit(
                Overloaded{
                    [](const GraphValueOutputTokenRef &v) {
                      return std::tuple<int, std::uint64_t, std::uint64_t>{
                          1, v.graph.entity.value(), v.ordinal};
                    },
                    [](const GraphStreamOutputTokenRef &s) {
                      return std::tuple<int, std::uint64_t, std::uint64_t>{
                          2, s.graph.entity.value(), s.ordinal};
                    },
                    [](const GraphCompletionFrontierTokenRef &f) {
                      return std::tuple<int, std::uint64_t, std::uint64_t>{
                          3, f.graph.entity.value(), f.ordinal};
                    }},
                e);
          }},
      c);
}

// The canonical sort key of a channel consumer endpoint.
std::tuple<int, std::uint64_t, std::uint64_t, std::uint64_t>
channelConsumerKey(const ChannelConsumerRef &c) {
  if (const auto *s = std::get_if<GraphStreamInputConsumerRef>(&c))
    return {0, s->launch.rootThreadLaunch.entity.value(),
            s->launch.staticGraphLaunch.entity.value(), s->ordinal};
  const auto &r = std::get<ThreadChannelReceiveSiteRef>(c);
  return {1, r.launch.entity.value(), 0, r.ordinal};
}

// The block-argument index for a graph ingress ordinal.
llvm::Expected<unsigned> ingressArgumentIndex(GraphOp graph,
                                              const GraphIngressTokenRef &ref) {
  llvm::ArrayRef<int32_t> segments = graph.getInputSegmentSizes();
  return std::visit(
      Overloaded{
          [](const GraphStartTokenRef &) -> llvm::Expected<unsigned> {
            return 0u;
          },
          [&](const GraphValueInputTokenRef &v) -> llvm::Expected<unsigned> {
            if (v.ordinal >= static_cast<unsigned>(segments[0]))
              return invalid("canonical dataflow: graph value-input ordinal "
                             "out of range");
            return v.ordinal + 1;
          },
          [&](const GraphStreamInputTokenRef &s) -> llvm::Expected<unsigned> {
            if (s.ordinal >= static_cast<unsigned>(segments[1]))
              return invalid("canonical dataflow: graph stream-input ordinal "
                             "out of range");
            return static_cast<unsigned>(segments[0]) + s.ordinal + 1;
          }},
      ref);
}

} // namespace

//===----------------------------------------------------------------------===//
// One-pass closed structural inventory construction
//===----------------------------------------------------------------------===//

llvm::Error CanonicalDataflowProgramView::buildStructuralInventories(
    ModuleOp module, llvm::ArrayRef<Operation *> canonicalOperationOrder) {
  // (1) Compact thread-owner slots. Register every thread definition reached by
  // a root launch (as callee) or static graph launch (as parent). Rooted-launch
  // validation becomes an O(1) slot comparison; enumeration composes lazily.
  auto threadSlot = [&](Operation *threadDef) -> unsigned {
    auto it = threadSlotOf_.find(threadDef);
    if (it != threadSlotOf_.end())
      return it->second;
    unsigned slot = threadDefs_.size();
    threadDefs_.push_back(threadDef);
    threadSlotOf_[threadDef] = slot;
    return slot;
  };
  rootCalleeThreadSlot_.resize(rootThreadLaunches_.size());
  for (unsigned r = 0; r < rootThreadLaunches_.size(); ++r)
    rootCalleeThreadSlot_[r] = threadSlot(rootThreadLaunches_[r].callee);
  constexpr unsigned kNoThread = std::numeric_limits<unsigned>::max();
  staticOwnerThreadSlot_.assign(staticGraphLaunches_.size(), kNoThread);
  for (unsigned s = 0; s < staticGraphLaunches_.size(); ++s)
    if (Operation *owner =
            staticGraphLaunches_[s].op->getParentOfType<ThreadOp>())
      staticOwnerThreadSlot_[s] = threadSlot(owner);
  rootsByThreadSlot_.assign(threadDefs_.size(), {});
  staticsByThreadSlot_.assign(threadDefs_.size(), {});
  for (unsigned r = 0; r < rootThreadLaunches_.size(); ++r)
    rootsByThreadSlot_[rootCalleeThreadSlot_[r]].push_back(r);
  for (unsigned s = 0; s < staticGraphLaunches_.size(); ++s)
    if (staticOwnerThreadSlot_[s] != kNoThread)
      staticsByThreadSlot_[staticOwnerThreadSlot_[s]].push_back(s);

  // (2) Per-thread channel endpoint counts. One walk per thread definition
  // yields transient op-to-ordinal maps (for import) and the retained counts
  // (for hot ordinal-range validation); no endpoint pointer inventory is kept.
  threadSendCount_.assign(threadDefs_.size(), 0);
  threadReceiveCount_.assign(threadDefs_.size(), 0);
  llvm::DenseMap<Operation *, unsigned> sendOrdinalOf;
  llvm::DenseMap<Operation *, unsigned> receiveOrdinalOf;
  for (unsigned t = 0; t < threadDefs_.size(); ++t) {
    threadDefs_[t]->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<ChannelSendOp>(op))
        sendOrdinalOf[op] = threadSendCount_[t]++;
      else if (isa<ChannelReceiveOp>(op))
        receiveOrdinalOf[op] = threadReceiveCount_[t]++;
    });
  }

  // (3) Intra-graph software edge relation. Every token producer value's
  // complete, canonically sorted consumer set is one contiguous range addressed
  // by a collision-free typed producer key.
  auto recordConsumers = [&](Value value, GraphRef ownerGraph,
                             std::tuple<int, unsigned, unsigned> key) {
    if (isMemoryCapability(value.getType()))
      return;
    unsigned begin = graphEdgeConsumers_.size();
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (auto actorId = actorIdByOp_.find(owner);
          actorId != actorIdByOp_.end()) {
        graphEdgeConsumers_.push_back(ActorTokenOperandRef{
            ActorRef{identity_, ActorId(actorId->second)},
            static_cast<StructuralOrdinal>(use.getOperandNumber())});
        continue;
      }
      if (auto ret = dyn_cast<GraphReturnOp>(owner)) {
        unsigned number = use.getOperandNumber();
        unsigned values = ret.getValues().size();
        unsigned streams = values + ret.getStreams().size();
        unsigned memories = streams + ret.getMemories().size();
        if (number < values)
          graphEdgeConsumers_.push_back(
              GraphEgressTokenRef{GraphValueOutputTokenRef{
                  ownerGraph, static_cast<StructuralOrdinal>(number)}});
        else if (number < streams)
          graphEdgeConsumers_.push_back(
              GraphEgressTokenRef{GraphStreamOutputTokenRef{
                  ownerGraph,
                  static_cast<StructuralOrdinal>(number - values)}});
        else if (number >= memories)
          graphEdgeConsumers_.push_back(
              GraphEgressTokenRef{GraphCompletionFrontierTokenRef{
                  ownerGraph,
                  static_cast<StructuralOrdinal>(number - memories)}});
      }
    }
    if (graphEdgeConsumers_.size() == begin)
      return;
    std::sort(graphEdgeConsumers_.begin() + begin, graphEdgeConsumers_.end(),
              [](const CanonicalGraphConsumerEndpointRef &a,
                 const CanonicalGraphConsumerEndpointRef &b) {
                return consumerKey(a) < consumerKey(b);
              });
    graphEdgeRange_[key] = {begin, graphEdgeConsumers_.size() - begin};
  };
  for (unsigned as = 0; as < actors_.size(); ++as)
    for (unsigned i = 0; i < actors_[as].op->getNumResults(); ++i)
      recordConsumers(actors_[as].op->getResult(i), actors_[as].graph,
                      {kActorResult, as, i});
  for (unsigned gs = 0; gs < graphs_.size(); ++gs) {
    auto op = cast<GraphOp>(graphs_[gs].op);
    llvm::ArrayRef<int32_t> segments = op.getInputSegmentSizes();
    Block &body = op.getBody().front();
    if (body.getNumArguments() > 0)
      recordConsumers(body.getArgument(0), graphs_[gs].ref,
                      {kGraphStart, gs, 0});
    for (unsigned j = 0; j < static_cast<unsigned>(segments[0]) &&
                         1 + j < body.getNumArguments();
         ++j)
      recordConsumers(body.getArgument(1 + j), graphs_[gs].ref,
                      {kGraphValueInput, gs, j});
    for (unsigned k = 0;
         k < static_cast<unsigned>(segments[1]) &&
         1 + static_cast<unsigned>(segments[0]) + k < body.getNumArguments();
         ++k)
      recordConsumers(
          body.getArgument(1 + static_cast<unsigned>(segments[0]) + k),
          graphs_[gs].ref, {kGraphStreamInput, gs, k});
  }

  // (4) Channel multicast relation, via the one shared host-channel discovery
  // and relation owner. Each producer's canonically sorted multicast set is one
  // contiguous range keyed by a collision-free typed producer key; its sink
  // terminals share the range.
  if (llvm::Error error = forEachHostChannelRelation(
          module, [&](Value, const ChannelRelation &relation) -> llvm::Error {
            llvm::SmallVector<ChannelConsumerBinding, 4> consumers;
            for (const ChannelEndpointBinding &binding : relation.consumers) {
              auto consumerRootIt =
                  rootThreadLaunchIdByOp_.find(binding.rootLaunch);
              if (consumerRootIt == rootThreadLaunchIdByOp_.end())
                return invalid("canonical dataflow: channel consumer root is "
                               "not an entity");
              RootThreadLaunchRef rootRef{
                  identity_, RootThreadLaunchId(consumerRootIt->second)};
              if (binding.streamOrdinal) {
                auto staticIt = staticGraphLaunchIdByOp_.find(binding.site);
                if (staticIt == staticGraphLaunchIdByOp_.end())
                  return invalid("canonical dataflow: channel consumer graph "
                                 "launch is not an entity");
                RootedGraphLaunchRef rooted{
                    rootRef,
                    StaticGraphLaunchRef{
                        identity_, StaticGraphLaunchId(staticIt->second)}};
                consumers.push_back(
                    {ChannelConsumerRef{GraphStreamInputConsumerRef{
                         rooted, static_cast<StructuralOrdinal>(
                                     *binding.streamOrdinal)}},
                     binding.sourceMap});
              } else {
                auto ordinal = receiveOrdinalOf.find(binding.site);
                if (ordinal == receiveOrdinalOf.end())
                  return invalid("canonical dataflow: channel receive site has "
                                 "no canonical ordinal");
                consumers.push_back(
                    {ChannelConsumerRef{ThreadChannelReceiveSiteRef{
                         rootRef,
                         static_cast<StructuralOrdinal>(ordinal->second)}},
                     std::nullopt});
              }
            }
            if (consumers.empty())
              return invalid("canonical dataflow: channel producer has no "
                             "consumer");
            std::sort(consumers.begin(), consumers.end(),
                      [](const ChannelConsumerBinding &a,
                         const ChannelConsumerBinding &b) {
                        return channelConsumerKey(a.consumer) <
                               channelConsumerKey(b.consumer);
                      });

            const unsigned begin = channelBindings_.size();
            channelBindings_.insert(channelBindings_.end(), consumers.begin(),
                                    consumers.end());
            for (const ChannelConsumerBinding &binding : consumers)
              channelSinks_.push_back(
                  ChannelConsumerTerminalRef{binding.consumer});

            for (const ChannelEndpointBinding &producer : relation.producers) {
              auto rootIt = rootThreadLaunchIdByOp_.find(producer.rootLaunch);
              if (rootIt == rootThreadLaunchIdByOp_.end())
                return invalid("canonical dataflow: channel producer root is "
                               "not an entity");
              unsigned rootSlot = slotOfId_[rootIt->second];
              ChannelProducerKey producerKey;
              if (producer.streamOrdinal) {
                auto staticIt = staticGraphLaunchIdByOp_.find(producer.site);
                if (staticIt == staticGraphLaunchIdByOp_.end())
                  return invalid("canonical dataflow: channel producer graph "
                                 "launch is not an entity");
                producerKey = {
                    kStreamOutput, rootSlot,
                    static_cast<unsigned>(slotOfId_[staticIt->second]),
                    *producer.streamOrdinal};
              } else {
                auto ordinal = sendOrdinalOf.find(producer.site);
                if (ordinal == sendOrdinalOf.end())
                  return invalid(
                      "canonical dataflow: channel producer send site has no "
                      "canonical ordinal");
                producerKey = {kSend, rootSlot, 0, ordinal->second};
              }

              if (!channelRange_
                       .emplace(
                           producerKey,
                           std::pair<unsigned, unsigned>{
                               begin, static_cast<unsigned>(consumers.size())})
                       .second)
                return invalid("canonical dataflow: duplicate channel "
                               "producer terminal");
            }
            return llvm::Error::success();
          }))
    return error;

  // (5) Static memory composition, in canonical stored-program order. Roots
  // are thread memory formals, thread memory-service acquisitions, and fresh
  // graph allocations. A thread-body view is a global root-local view; a
  // graph-body view is resolved once per callee context at each actual static
  // launch site, contextual when it peels to a graph memory formal (so two
  // sites under two thread roots yield distinct root-local views) and global
  // when it peels to a graph-body allocation. Every admitted view is resolved
  // and every memory-result exposure is total; an unresolved relation fails
  // finalization. Root-launch occurrences never multiply this inventory.
  llvm::SmallVector<unsigned> viewCountByRoot(logicalMemoryRoots_.size(), 0);
  llvm::SmallVector<std::pair<std::size_t, LogicalMemoryViewRef>> flatViews;
  auto addRole = [&](LogicalMemoryRootOrViewRef role) -> unsigned {
    roleTable_.push_back(role);
    return roleTable_.size() - 1;
  };
  auto rootSlotOf = [&](unsigned roleIdx) -> std::size_t {
    const LogicalMemoryRootOrViewRef &role = roleTable_[roleIdx];
    if (const auto *r = std::get_if<LogicalMemoryRootRef>(&role))
      return slotOfId_[r->entity.value()];
    return slotOfId_[std::get<LogicalMemoryViewRef>(role).root.entity.value()];
  };
  auto makeViewRole = [&](std::size_t rootSlot) -> unsigned {
    unsigned ordinal = viewCountByRoot[rootSlot]++;
    LogicalMemoryViewRef view{logicalMemoryRoots_[rootSlot].ref, ordinal};
    flatViews.push_back({rootSlot, view});
    return addRole(LogicalMemoryRootOrViewRef{view});
  };
  auto rootRoleGlobal = [&](Value v, std::uint64_t id) -> unsigned {
    unsigned idx = addRole(LogicalMemoryRootOrViewRef{
        LogicalMemoryRootRef{identity_, LogicalMemoryRootId(id)}});
    roleIndexOf_[v] = idx;
    return idx;
  };

  // A thread-level value (a memory formal, a thread-body view, or an earlier
  // launch's memory result recorded below) resolves to one global role.
  std::function<llvm::Expected<unsigned>(Value)> resolveThreadValue =
      [&](Value v) -> llvm::Expected<unsigned> {
    if (auto it = roleIndexOf_.find(v); it != roleIndexOf_.end())
      return it->second;
    if (auto it = memoryRootIdByValue_.find(v);
        it != memoryRootIdByValue_.end())
      return rootRoleGlobal(v, it->second);
    if (Operation *def = v.getDefiningOp())
      if (Value base = viewBase(def)) {
        llvm::Expected<unsigned> baseIdx = resolveThreadValue(base);
        if (!baseIdx)
          return baseIdx.takeError();
        unsigned idx = makeViewRole(rootSlotOf(*baseIdx));
        roleIndexOf_[v] = idx;
        return idx;
      }
    return invalid("canonical dataflow: unresolved thread memory value role");
  };

  // (5a) Admitted graph-body view values indexed once per callee graph in
  // canonical order, so a repeated use resolves to one view.
  llvm::DenseMap<Operation *, llvm::SmallVector<Value>> graphBodyViews;
  for (Operation *op : canonicalOperationOrder)
    if (viewBase(op) && isMemoryCapability(op->getResult(0).getType()))
      if (GraphOp g = op->getParentOfType<GraphOp>())
        graphBodyViews[g.getOperation()].push_back(op->getResult(0));

  exposureByStaticSlot_.assign(staticGraphLaunches_.size(), {});
  for (unsigned s = 0; s < staticGraphLaunches_.size(); ++s) {
    auto ret = cast<GraphReturnOp>(
        cast<GraphOp>(
            llvm::cantFail(resolve(staticGraphLaunches_[s].callee)).op)
            .getBody()
            .front()
            .getTerminator());
    exposureByStaticSlot_[s].resize(ret.getMemories().size());
  }

  // Compose one static launch site: resolve every admitted graph-body view in
  // this callee context and every memory-result exposure, recording each launch
  // result's role for a later thread-body view or launch to consume. A per-site
  // value cache holds contextual formal and formal-rooted view roles;
  // allocation-rooted graph views keep one global role.
  auto composeSite = [&](GraphLaunchOp launch, unsigned s) -> llvm::Error {
    auto graph = cast<GraphOp>(
        llvm::cantFail(resolve(staticGraphLaunches_[s].callee)).op);
    llvm::ArrayRef<int32_t> inSeg = graph.getInputSegmentSizes();
    unsigned firstMemory =
        1 + static_cast<unsigned>(inSeg[0]) + static_cast<unsigned>(inSeg[1]);
    llvm::DenseMap<Value, unsigned> siteCache;
    std::function<llvm::Expected<unsigned>(Value)> resolveInContext =
        [&](Value m) -> llvm::Expected<unsigned> {
      if (auto it = roleIndexOf_.find(m); it != roleIndexOf_.end())
        return it->second;
      if (auto it = siteCache.find(m); it != siteCache.end())
        return it->second;
      if (auto it = memoryRootIdByValue_.find(m);
          it != memoryRootIdByValue_.end())
        return rootRoleGlobal(m, it->second);
      if (auto arg = dyn_cast<BlockArgument>(m)) {
        if (arg.getOwner()->getParentOp() != graph.getOperation())
          return invalid("canonical dataflow: memory value escapes its graph");
        if (arg.getArgNumber() < firstMemory)
          return invalid("canonical dataflow: memory formal expected");
        unsigned j = arg.getArgNumber() - firstMemory;
        if (j >= launch.getMemoryInputs().size())
          return invalid(
              "canonical dataflow: graph memory formal has no launch "
              "binding");
        Value actual = launch.getMemoryInputs()[j];
        llvm::Expected<unsigned> role = resolveThreadValue(actual);
        if (!role)
          return role.takeError();
        if (actual.getType() != arg.getType())
          return invalid(
              "canonical dataflow: graph memory binding type mismatch");
        siteCache[m] = *role;
        return *role;
      }
      if (Operation *def = m.getDefiningOp())
        if (Value base = viewBase(def)) {
          llvm::Expected<unsigned> baseRole = resolveInContext(base);
          if (!baseRole)
            return baseRole.takeError();
          // The base lives in the global role map (an allocation root or an
          // allocation-rooted view) or the per-site cache (a graph memory
          // formal or formal-rooted view); a view inherits that scope so two
          // sites keep distinct formal-rooted views.
          bool baseGlobal =
              roleIndexOf_.count(base) || memoryRootIdByValue_.count(base);
          unsigned idx = makeViewRole(rootSlotOf(*baseRole));
          if (baseGlobal)
            roleIndexOf_[m] = idx;
          else
            siteCache[m] = idx;
          return idx;
        }
      return invalid("canonical dataflow: unresolved graph memory value role");
    };
    for (unsigned ordinal = 0; ordinal < static_cast<unsigned>(inSeg[2]);
         ++ordinal)
      if (llvm::Expected<unsigned> idx = resolveInContext(
              graph.getBody().front().getArgument(firstMemory + ordinal));
          !idx)
        return idx.takeError();
    if (auto it = graphBodyViews.find(graph.getOperation());
        it != graphBodyViews.end())
      for (Value v : it->second)
        if (llvm::Expected<unsigned> idx = resolveInContext(v); !idx)
          return idx.takeError();
    auto ret = cast<GraphReturnOp>(graph.getBody().front().getTerminator());
    llvm::ArrayRef<int32_t> outSeg = graph.getResultSegmentSizes();
    unsigned resultBase =
        static_cast<unsigned>(outSeg[0]) + static_cast<unsigned>(outSeg[1]);
    for (unsigned r = 0; r < ret.getMemories().size(); ++r) {
      llvm::Expected<unsigned> role = resolveInContext(ret.getMemories()[r]);
      if (!role)
        return role.takeError();
      exposureByStaticSlot_[s][r] = *role;
      roleIndexOf_[launch->getResult(resultBase + r)] = *role;
    }
    return llvm::Error::success();
  };

  // (5b) Interleave, per canonical thread definition walked once in
  // stored-program order, thread-body views with StaticGraphLaunch sites: a
  // view after a launch consumes that launch's recorded memory result, and a
  // later launch consumes that view. Admitted views and exposures are total and
  // failure-atomic.
  unsigned composedSites = 0;
  llvm::DenseSet<Operation *> seenThreads;
  llvm::Error error = llvm::Error::success();
  for (Operation *top : canonicalOperationOrder) {
    if (!isa<ThreadOp>(top) || !seenThreads.insert(top).second)
      continue;
    top->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (error)
        return;
      if (auto launch = dyn_cast<GraphLaunchOp>(op)) {
        auto it = staticGraphLaunchIdByOp_.find(op);
        if (it == staticGraphLaunchIdByOp_.end())
          return;
        if (llvm::Error e = composeSite(launch, slotOfId_[it->second]))
          error = std::move(e);
        else
          ++composedSites;
      } else if (viewBase(op) &&
                 isMemoryCapability(op->getResult(0).getType()) &&
                 !op->getParentOfType<GraphOp>()) {
        if (llvm::Expected<unsigned> idx = resolveThreadValue(op->getResult(0));
            !idx)
          error = idx.takeError();
      }
    });
    if (error)
      break;
  }
  if (error)
    return error;
  if (composedSites != staticGraphLaunches_.size())
    return invalid(
        "canonical dataflow: a static graph launch is not reached in "
        "canonical stored-program order");

  // Flatten views to per-root contiguous ranges ordered by canonical ordinal.
  viewsByRootSlot_.assign(logicalMemoryRoots_.size(), {0u, 0u});
  std::sort(flatViews.begin(), flatViews.end(),
            [](const std::pair<std::size_t, LogicalMemoryViewRef> &a,
               const std::pair<std::size_t, LogicalMemoryViewRef> &b) {
              return std::make_pair(a.first, a.second.viewOrdinal) <
                     std::make_pair(b.first, b.second.viewOrdinal);
            });
  for (const auto &entry : flatViews) {
    if (viewsByRootSlot_[entry.first].second == 0)
      viewsByRootSlot_[entry.first].first = views_.size();
    ++viewsByRootSlot_[entry.first].second;
    views_.push_back(entry.second);
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Rooted graph launch
//===----------------------------------------------------------------------===//

void CanonicalDataflowProgramView::forEachRootedGraphLaunch(
    llvm::function_ref<void(RootedGraphLaunchRef)> callback) const {
  // Compose each root/static pair on demand from the grouped thread-owner
  // inventory. No roots-by-launches product is ever stored.
  for (unsigned t = 0; t < rootsByThreadSlot_.size(); ++t)
    for (unsigned rootSlot : rootsByThreadSlot_[t])
      for (unsigned staticSlot : staticsByThreadSlot_[t])
        callback({rootThreadLaunches_[rootSlot].ref,
                  staticGraphLaunches_[staticSlot].ref});
}

llvm::Expected<GraphRef>
CanonicalDataflowProgramView::resolve(RootedGraphLaunchRef ref) const {
  auto rootSlot = requireKind(ref.rootThreadLaunch.artifact,
                              ref.rootThreadLaunch.entity.value(),
                              CanonicalDataflowEntityKind::RootThreadLaunch);
  if (!rootSlot)
    return rootSlot.takeError();
  auto staticSlot = requireKind(ref.staticGraphLaunch.artifact,
                                ref.staticGraphLaunch.entity.value(),
                                CanonicalDataflowEntityKind::StaticGraphLaunch);
  if (!staticSlot)
    return staticSlot.takeError();
  if (rootCalleeThreadSlot_[*rootSlot] != staticOwnerThreadSlot_[*staticSlot])
    return invalid("canonical dataflow: rooted graph launch site does not "
                   "belong to the root launch's thread");
  return staticGraphLaunches_[*staticSlot].callee;
}

//===----------------------------------------------------------------------===//
// Token-plane endpoints and the intra-graph software edge relation
//===----------------------------------------------------------------------===//

llvm::Error CanonicalDataflowProgramView::validate(
    const CanonicalGraphProducerEndpointRef &endpoint) const {
  return std::visit(
      Overloaded{
          [&](const GraphIngressTokenRef &ingress) -> llvm::Error {
            const GraphRef &graph = std::visit(
                [](const auto &token) -> const GraphRef & {
                  return token.graph;
                },
                ingress);
            llvm::Expected<CanonicalGraphView> view = resolve(graph);
            if (!view)
              return view.takeError();
            return ingressArgumentIndex(cast<GraphOp>(view->op), ingress)
                .takeError();
          },
          [&](const ActorTokenResultRef &result) -> llvm::Error {
            llvm::Expected<CanonicalActorView> actor = resolve(result.actor);
            if (!actor)
              return actor.takeError();
            if (result.ordinal >= actor->op->getNumResults())
              return invalid("canonical dataflow: actor result ordinal out of "
                             "range");
            if (isMemoryCapability(
                    actor->op->getResult(result.ordinal).getType()))
              return invalid(
                  "canonical dataflow: a memory-capability result is "
                  "not a token endpoint");
            return llvm::Error::success();
          }},
      endpoint);
}

llvm::Error CanonicalDataflowProgramView::validate(
    const CanonicalGraphConsumerEndpointRef &endpoint) const {
  return std::visit(
      Overloaded{
          [&](const ActorTokenOperandRef &operand) -> llvm::Error {
            llvm::Expected<CanonicalActorView> actor = resolve(operand.actor);
            if (!actor)
              return actor.takeError();
            if (operand.ordinal >= actor->op->getNumOperands())
              return invalid("canonical dataflow: actor operand ordinal out of "
                             "range");
            if (isMemoryCapability(
                    actor->op->getOperand(operand.ordinal).getType()))
              return invalid("canonical dataflow: a memory-capability operand "
                             "is not a token endpoint");
            return llvm::Error::success();
          },
          [&](const GraphEgressTokenRef &egress) -> llvm::Error {
            const GraphRef &graph = std::visit(
                [](const auto &token) -> const GraphRef & {
                  return token.graph;
                },
                egress);
            llvm::Expected<CanonicalGraphView> view = resolve(graph);
            if (!view)
              return view.takeError();
            GraphOp op = cast<GraphOp>(view->op);
            auto ret =
                cast<GraphReturnOp>(op.getBody().front().getTerminator());
            llvm::ArrayRef<int32_t> segments = op.getResultSegmentSizes();
            return std::visit(
                Overloaded{
                    [&](const GraphValueOutputTokenRef &v) -> llvm::Error {
                      if (v.ordinal >= static_cast<unsigned>(segments[0]))
                        return invalid("canonical dataflow: graph value-output "
                                       "ordinal out of range");
                      return llvm::Error::success();
                    },
                    [&](const GraphStreamOutputTokenRef &s) -> llvm::Error {
                      if (s.ordinal >= static_cast<unsigned>(segments[1]))
                        return invalid(
                            "canonical dataflow: graph stream-output "
                            "ordinal out of range");
                      return llvm::Error::success();
                    },
                    [&](const GraphCompletionFrontierTokenRef &c)
                        -> llvm::Error {
                      if (c.ordinal >= ret.getComplete().size())
                        return invalid("canonical dataflow: graph completion "
                                       "frontier ordinal out of range");
                      return llvm::Error::success();
                    }},
                egress);
          }},
      endpoint);
}

llvm::Expected<llvm::ArrayRef<CanonicalGraphConsumerEndpointRef>>
CanonicalDataflowProgramView::graphConsumers(
    const CanonicalGraphProducerEndpointRef &producer) const {
  // Validate the producer, then return its prebuilt canonically sorted consumer
  // range addressed by the producer's collision-free owner-slot key.
  if (llvm::Error error = validate(producer))
    return std::move(error);
  std::tuple<int, unsigned, unsigned> key = std::visit(
      Overloaded{[&](const ActorTokenResultRef &r) {
                   return std::tuple<int, unsigned, unsigned>{
                       kActorResult,
                       static_cast<unsigned>(slotOfId_[r.actor.entity.value()]),
                       r.ordinal};
                 },
                 [&](const GraphIngressTokenRef &ingress) {
                   return std::visit(
                       Overloaded{[&](const GraphStartTokenRef &t) {
                                    return std::tuple<int, unsigned, unsigned>{
                                        kGraphStart,
                                        static_cast<unsigned>(
                                            slotOfId_[t.graph.entity.value()]),
                                        0};
                                  },
                                  [&](const GraphValueInputTokenRef &t) {
                                    return std::tuple<int, unsigned, unsigned>{
                                        kGraphValueInput,
                                        static_cast<unsigned>(
                                            slotOfId_[t.graph.entity.value()]),
                                        t.ordinal};
                                  },
                                  [&](const GraphStreamInputTokenRef &t) {
                                    return std::tuple<int, unsigned, unsigned>{
                                        kGraphStreamInput,
                                        static_cast<unsigned>(
                                            slotOfId_[t.graph.entity.value()]),
                                        t.ordinal};
                                  }},
                       ingress);
                 }},
      producer);
  auto range = graphEdgeRange_.find(key);
  if (range == graphEdgeRange_.end())
    return llvm::ArrayRef<CanonicalGraphConsumerEndpointRef>{};
  return llvm::ArrayRef<CanonicalGraphConsumerEndpointRef>(
      graphEdgeConsumers_.data() + range->second.first, range->second.second);
}

llvm::Error CanonicalDataflowProgramView::forEachGraphEdge(
    llvm::function_ref<llvm::Error(const CanonicalGraphProducerEndpointRef &,
                                   const CanonicalGraphConsumerEndpointRef &)>
        callback) const {
  for (const auto &[key, range] : graphEdgeRange_) {
    const auto [kind, slot, ordinal] = key;
    std::optional<CanonicalGraphProducerEndpointRef> producer;
    switch (kind) {
    case kActorResult:
      if (slot >= actors_.size())
        return invalid("canonical dataflow: graph-edge actor slot is invalid");
      producer = ActorTokenResultRef{actors_[slot].ref, ordinal};
      break;
    case kGraphStart:
      if (slot >= graphs_.size() || ordinal != 0)
        return invalid("canonical dataflow: graph-edge start slot is invalid");
      producer = GraphIngressTokenRef{GraphStartTokenRef{graphs_[slot].ref}};
      break;
    case kGraphValueInput:
      if (slot >= graphs_.size())
        return invalid(
            "canonical dataflow: graph-edge value-input slot is invalid");
      producer = GraphIngressTokenRef{
          GraphValueInputTokenRef{graphs_[slot].ref, ordinal}};
      break;
    case kGraphStreamInput:
      if (slot >= graphs_.size())
        return invalid(
            "canonical dataflow: graph-edge stream-input slot is invalid");
      producer = GraphIngressTokenRef{
          GraphStreamInputTokenRef{graphs_[slot].ref, ordinal}};
      break;
    default:
      return invalid("canonical dataflow: graph-edge producer kind is invalid");
    }
    const std::size_t begin = range.first;
    const std::size_t count = range.second;
    if (begin > graphEdgeConsumers_.size() ||
        count > graphEdgeConsumers_.size() - begin)
      return invalid("canonical dataflow: graph-edge range is invalid");
    for (const CanonicalGraphConsumerEndpointRef &consumer :
         llvm::ArrayRef(graphEdgeConsumers_).slice(begin, count))
      if (llvm::Error error = callback(*producer, consumer))
        return error;
  }
  return llvm::Error::success();
}

llvm::Expected<CanonicalGraphProducerEndpointRef>
CanonicalDataflowProgramView::graphProducer(
    const CanonicalGraphConsumerEndpointRef &consumer) const {
  Value value;
  GraphOp graph;
  std::optional<GraphRef> graphRef;
  if (const auto *operand = std::get_if<ActorTokenOperandRef>(&consumer)) {
    if (llvm::Error error =
            validate(CanonicalGraphConsumerEndpointRef{*operand}))
      return std::move(error);
    llvm::Expected<CanonicalActorView> actor = resolve(operand->actor);
    if (!actor)
      return actor.takeError();
    value = actor->op->getOperand(operand->ordinal);
    graph = cast<GraphOp>(llvm::cantFail(resolve(actor->graph)).op);
    graphRef = actor->graph;
  } else {
    const GraphEgressTokenRef &egress = std::get<GraphEgressTokenRef>(consumer);
    if (llvm::Error error = validate(CanonicalGraphConsumerEndpointRef{egress}))
      return std::move(error);
    graphRef = std::visit(
        [](const auto &t) -> const GraphRef & { return t.graph; }, egress);
    graph = cast<GraphOp>(llvm::cantFail(resolve(*graphRef)).op);
    auto ret = cast<GraphReturnOp>(graph.getBody().front().getTerminator());
    value =
        std::visit(Overloaded{[&](const GraphValueOutputTokenRef &v) {
                                return ret.getValues()[v.ordinal];
                              },
                              [&](const GraphStreamOutputTokenRef &s) {
                                return ret.getStreams()[s.ordinal];
                              },
                              [&](const GraphCompletionFrontierTokenRef &c) {
                                return ret.getComplete()[c.ordinal];
                              }},
                   egress);
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    unsigned index = arg.getArgNumber();
    if (index == 0)
      return CanonicalGraphProducerEndpointRef{
          GraphIngressTokenRef{GraphStartTokenRef{*graphRef}}};
    llvm::ArrayRef<int32_t> segments = graph.getInputSegmentSizes();
    unsigned input = index - 1;
    if (input < static_cast<unsigned>(segments[0]))
      return CanonicalGraphProducerEndpointRef{
          GraphIngressTokenRef{GraphValueInputTokenRef{*graphRef, input}}};
    return CanonicalGraphProducerEndpointRef{
        GraphIngressTokenRef{GraphStreamInputTokenRef{
            *graphRef, input - static_cast<unsigned>(segments[0])}}};
  }
  Operation *def = value.getDefiningOp();
  auto actorId = actorIdByOp_.find(def);
  if (actorId == actorIdByOp_.end())
    return invalid("canonical dataflow: token value has no canonical producer");
  return CanonicalGraphProducerEndpointRef{ActorTokenResultRef{
      ActorRef{identity_, ActorId(actorId->second)},
      static_cast<StructuralOrdinal>(cast<OpResult>(value).getResultNumber())}};
}

//===----------------------------------------------------------------------===//
// Boundary transfers
//===----------------------------------------------------------------------===//

llvm::Error CanonicalDataflowProgramView::validate(
    const RootThreadBoundaryTransferRef &transfer) const {
  return std::visit(
      Overloaded{[&](const RootThreadStartTransferRef &t) {
                   return resolve(t.launch).takeError();
                 },
                 [&](const RootThreadCompletionTransferRef &t) {
                   return resolve(t.launch).takeError();
                 },
                 [&](const RootThreadValueInputTransferRef &t) -> llvm::Error {
                   llvm::Expected<CanonicalRootThreadLaunchView> root =
                       resolve(t.launch);
                   if (!root)
                     return root.takeError();
                   if (t.ordinal >=
                       threadValueInputCount(cast<ThreadOp>(root->callee)))
                     return invalid(
                         "canonical dataflow: root-thread value-input "
                         "ordinal out of range");
                   return llvm::Error::success();
                 }},
      transfer);
}

llvm::Error CanonicalDataflowProgramView::validate(
    const GraphLaunchBoundaryTransferRef &transfer) const {
  auto boundedGraph =
      [&](RootedGraphLaunchRef launch) -> llvm::Expected<GraphOp> {
    llvm::Expected<GraphRef> graph = resolve(launch);
    if (!graph)
      return graph.takeError();
    return cast<GraphOp>(llvm::cantFail(resolve(*graph)).op);
  };
  return std::visit(
      Overloaded{
          [&](const GraphLaunchStartTransferRef &t) {
            return boundedGraph(t.launch).takeError();
          },
          [&](const GraphLaunchDoneTransferRef &t) {
            return boundedGraph(t.launch).takeError();
          },
          [&](const GraphLaunchValueInputTransferRef &t) -> llvm::Error {
            llvm::Expected<GraphOp> graph = boundedGraph(t.launch);
            if (!graph)
              return graph.takeError();
            if (t.ordinal >=
                static_cast<unsigned>(graph->getInputSegmentSizes()[0]))
              return invalid("canonical dataflow: graph-launch value-input "
                             "ordinal out of range");
            return llvm::Error::success();
          },
          [&](const GraphLaunchValueResultTransferRef &t) -> llvm::Error {
            llvm::Expected<GraphOp> graph = boundedGraph(t.launch);
            if (!graph)
              return graph.takeError();
            if (t.ordinal >=
                static_cast<unsigned>(graph->getResultSegmentSizes()[0]))
              return invalid("canonical dataflow: graph-launch value-result "
                             "ordinal out of range");
            return llvm::Error::success();
          }},
      transfer);
}

//===----------------------------------------------------------------------===//
// Channel relation and transfer terminals
//===----------------------------------------------------------------------===//

llvm::Expected<CanonicalDataflowProgramView::ChannelProducerKey>
CanonicalDataflowProgramView::channelProducerKey(
    const ChannelProducerRef &producer) const {
  if (const auto *out = std::get_if<GraphStreamOutputProducerRef>(&producer)) {
    if (llvm::Error error = resolve(out->launch).takeError())
      return std::move(error);
    unsigned rootSlot = slotOfId_[out->launch.rootThreadLaunch.entity.value()];
    unsigned staticSlot =
        slotOfId_[out->launch.staticGraphLaunch.entity.value()];
    auto graphLaunch = cast<GraphLaunchOp>(staticGraphLaunches_[staticSlot].op);
    if (out->ordinal >= graphLaunch.getStreamOutputs().size())
      return invalid("canonical dataflow: stream-output ordinal out of range");
    return ChannelProducerKey{kStreamOutput, rootSlot, staticSlot,
                              out->ordinal};
  }
  const auto &send = std::get<ThreadChannelSendSiteRef>(producer);
  llvm::Expected<CanonicalRootThreadLaunchView> root = resolve(send.launch);
  if (!root)
    return root.takeError();
  unsigned rootSlot = slotOfId_[send.launch.entity.value()];
  unsigned threadSlot = rootCalleeThreadSlot_[rootSlot];
  if (send.ordinal >= threadSendCount_[threadSlot])
    return invalid(
        "canonical dataflow: channel send-site ordinal out of range");
  return ChannelProducerKey{kSend, rootSlot, 0, send.ordinal};
}

llvm::Expected<llvm::ArrayRef<ChannelConsumerBinding>>
CanonicalDataflowProgramView::channelConsumers(
    const ChannelProducerRef &producer) const {
  llvm::Expected<ChannelProducerKey> key = channelProducerKey(producer);
  if (!key)
    return key.takeError();
  auto range = channelRange_.find(*key);
  if (range == channelRange_.end())
    return invalid("canonical dataflow: channel producer has no consumer");
  return llvm::ArrayRef<ChannelConsumerBinding>(
      channelBindings_.data() + range->second.first, range->second.second);
}

llvm::Error CanonicalDataflowProgramView::pairedSinks(
    const CanonicalProducerTerminalRef &producer,
    llvm::function_ref<void(const CanonicalSinkTerminalRef &)> callback) const {
  // A boundary source yields its one paired sink; a channel producer yields its
  // complete prebuilt multicast sink range. No caller scratch, no allocation.
  if (const auto *source =
          std::get_if<RootThreadBoundarySourceRef>(&producer)) {
    if (llvm::Error error = validate(source->transfer))
      return error;
    callback(
        CanonicalSinkTerminalRef{RootThreadBoundarySinkRef{source->transfer}});
    return llvm::Error::success();
  }
  if (const auto *source =
          std::get_if<GraphLaunchBoundarySourceRef>(&producer)) {
    if (llvm::Error error = validate(source->transfer))
      return error;
    callback(
        CanonicalSinkTerminalRef{GraphLaunchBoundarySinkRef{source->transfer}});
    return llvm::Error::success();
  }
  const auto &channel = std::get<ChannelProducerTerminalRef>(producer);
  llvm::Expected<ChannelProducerKey> key = channelProducerKey(channel.producer);
  if (!key)
    return key.takeError();
  auto range = channelRange_.find(*key);
  if (range == channelRange_.end())
    return invalid("canonical dataflow: channel producer has no consumer");
  for (unsigned i = 0; i < range->second.second; ++i)
    callback(channelSinks_[range->second.first + i]);
  return llvm::Error::success();
}

llvm::Error CanonicalDataflowProgramView::validate(
    const CanonicalProducerTerminalRef &terminal) const {
  return std::visit(
      Overloaded{[&](const RootThreadBoundarySourceRef &t) {
                   return validate(t.transfer);
                 },
                 [&](const GraphLaunchBoundarySourceRef &t) {
                   return validate(t.transfer);
                 },
                 [&](const ChannelProducerTerminalRef &t) {
                   return channelConsumers(t.producer).takeError();
                 }},
      terminal);
}

llvm::Error CanonicalDataflowProgramView::validate(
    const CanonicalSinkTerminalRef &terminal) const {
  return std::visit(
      Overloaded{
          [&](const RootThreadBoundarySinkRef &t) {
            return validate(t.transfer);
          },
          [&](const GraphLaunchBoundarySinkRef &t) {
            return validate(t.transfer);
          },
          [&](const ChannelConsumerTerminalRef &t) -> llvm::Error {
            // Reject a foreign-artifact or wrong-owner endpoint, then require
            // the endpoint ordinal to exist in its owner's closed receive-site
            // or stream-input inventory by a direct count check.
            return std::visit(
                Overloaded{
                    [&](const GraphStreamInputConsumerRef &c) -> llvm::Error {
                      llvm::Expected<GraphRef> graph = resolve(c.launch);
                      if (!graph)
                        return graph.takeError();
                      auto op =
                          cast<GraphOp>(llvm::cantFail(resolve(*graph)).op);
                      if (c.ordinal >=
                          static_cast<unsigned>(op.getInputSegmentSizes()[1]))
                        return invalid(
                            "canonical dataflow: channel stream-input "
                            "consumer ordinal out of range");
                      return llvm::Error::success();
                    },
                    [&](const ThreadChannelReceiveSiteRef &c) -> llvm::Error {
                      auto slot = requireKind(
                          c.launch.artifact, c.launch.entity.value(),
                          CanonicalDataflowEntityKind::RootThreadLaunch);
                      if (!slot)
                        return slot.takeError();
                      unsigned threadSlot = rootCalleeThreadSlot_[*slot];
                      if (c.ordinal >= threadReceiveCount_[threadSlot])
                        return invalid(
                            "canonical dataflow: channel receive-site "
                            "consumer ordinal out of range");
                      return llvm::Error::success();
                    }},
                t.consumer);
          }},
      terminal);
}

llvm::Error CanonicalDataflowProgramView::validate(
    const StaticTransferEventRef &event) const {
  return std::visit(Overloaded{[&](const ProducedTransferEventRef &e) {
                                 return validate(e.terminal);
                               },
                               [&](const ConsumedTransferEventRef &e) {
                                 return validate(e.terminal);
                               }},
                    event);
}

//===----------------------------------------------------------------------===//
// Memory plane
//===----------------------------------------------------------------------===//

llvm::Expected<LogicalMemoryRootOrViewRef>
CanonicalDataflowProgramView::roleOfValue(Value value) const {
  auto it = roleIndexOf_.find(value);
  if (it == roleIndexOf_.end())
    return invalid("canonical dataflow: value has no admitted memory role");
  return roleTable_[it->second];
}

llvm::Expected<llvm::ArrayRef<LogicalMemoryViewRef>>
CanonicalDataflowProgramView::views(LogicalMemoryRootRef root) const {
  auto slot = requireKind(root.artifact, root.entity.value(),
                          CanonicalDataflowEntityKind::LogicalMemoryRoot);
  if (!slot)
    return slot.takeError();
  std::pair<unsigned, unsigned> range = viewsByRootSlot_[*slot];
  if (range.second == 0)
    return llvm::ArrayRef<LogicalMemoryViewRef>{};
  return llvm::ArrayRef<LogicalMemoryViewRef>(views_.data() + range.first,
                                              range.second);
}

llvm::Expected<LogicalMemoryRootOrViewRef>
CanonicalDataflowProgramView::resolveExposure(MemoryExposureRef ref) const {
  // Validate the rooted launch, then take the exact precomputed static-site
  // exposure. A runtime root-launch occurrence does not change static
  // LogicalMemoryRoot identity, so the exposure is keyed by the static slot.
  if (llvm::Error error = resolve(ref.launch).takeError())
    return std::move(error);
  unsigned staticSlot = slotOfId_[ref.launch.staticGraphLaunch.entity.value()];
  const auto &exposures = exposureByStaticSlot_[staticSlot];
  if (ref.memoryResultOrdinal >= exposures.size())
    return invalid("canonical dataflow: memory exposure ordinal out of range");
  return roleTable_[exposures[ref.memoryResultOrdinal]];
}

//===----------------------------------------------------------------------===//
// Contextual actors, fence family, and service members
//===----------------------------------------------------------------------===//

llvm::Error
CanonicalDataflowProgramView::validate(ContextualActorRef ref) const {
  llvm::Expected<GraphRef> graph = resolve(ref.launch);
  if (!graph)
    return graph.takeError();
  llvm::Expected<CanonicalActorView> actor = resolve(ref.actor);
  if (!actor)
    return actor.takeError();
  if (actor->graph != *graph)
    return invalid("canonical dataflow: contextual actor does not belong to "
                   "the launched graph");
  return llvm::Error::success();
}

llvm::Expected<FenceActorFamilyRef>
CanonicalDataflowProgramView::asFenceFamily(ActorRef ref) const {
  llvm::Expected<CanonicalActorView> actor = resolve(ref);
  if (!actor)
    return actor.takeError();
  if (!isa<FenceOp>(actor->op))
    return invalid("canonical dataflow: actor is not a dataflow.fence");
  return FenceActorFamilyRef{ref};
}

llvm::Expected<ServiceMemberRef>
CanonicalDataflowProgramView::serviceMemberFor(ContextualActorRef ref) const {
  if (llvm::Error error = validate(ref))
    return std::move(error);
  llvm::Expected<CanonicalActorView> actor = resolve(ref.actor);
  if (!actor)
    return actor.takeError();
  if (isa<FenceOp>(actor->op))
    return ServiceMemberRef{FenceActorMemberRef{ref}};
  // Classify an addressed-memory member through the exact Dataflow service and
  // access schema: a member exists only for one of the addressed canonical
  // memory actors with a well-formed access view.
  if (llvm::Expected<semantics::CanonicalMemoryAccessView> access =
          semantics::getCanonicalMemoryAccessView(actor->op))
    return ServiceMemberRef{AddressedMemoryActorMemberRef{ref}};
  else
    llvm::consumeError(access.takeError());
  return invalid("canonical dataflow: actor is neither an addressed-memory nor "
                 "a fence service member");
}

llvm::Expected<ServiceMemberRef>
CanonicalDataflowProgramView::messageTransferMember(
    const CanonicalProducerTerminalRef &terminal) const {
  // MessageTransfer is the service member of every valid transfer obligation,
  // whether a boundary transfer or a channel multicast producer.
  if (llvm::Error error = validate(terminal))
    return std::move(error);
  return ServiceMemberRef{MessageTransferMemberRef{}};
}

} // namespace dataflow
