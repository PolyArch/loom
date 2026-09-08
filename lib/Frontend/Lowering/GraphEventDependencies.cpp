#include "GraphEventDependencies.h"

#include "Dataflow/IR/DataflowOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace {

/// Memo of one causal query. A result derived without meeting an event
/// still on the recursion stack is exact and cached; a false that only
/// followed a broken cycle is path-dependent and recomputed elsewhere.
struct CausalQuery final {
  ::mlir::Value prerequisite;
  ::llvm::DenseMap<::mlir::Value, bool> memo;
  ::llvm::DenseSet<::mlir::Value> inProgress;
};

struct CausalAnswer final {
  bool depends = false;
  bool exact = true;
};

CausalAnswer causallyDependsOnUncached(::mlir::Value event, CausalQuery &query);

CausalAnswer causallyDependsOn(::mlir::Value event,
                               CausalQuery &query) {
  if (event == query.prerequisite)
    return {true, true};
  if (!event)
    return {false, true};
  if (auto found = query.memo.find(event); found != query.memo.end())
    return {found->second, true};
  if (!query.inProgress.insert(event).second)
    return {false, false};
  const CausalAnswer answer = causallyDependsOnUncached(event, query);
  query.inProgress.erase(event);
  if (answer.depends || answer.exact)
    query.memo.try_emplace(event, answer.depends);
  return answer;
}

/// An event depends on the prerequisite through any sync input, through
/// every mux lane or carry side, and through the control chain of memory
/// actors, invariants, gates, and constants.
CausalAnswer causallyDependsOnUncached(::mlir::Value event,
                                       CausalQuery &query) {
  ::mlir::Operation *def = event.getDefiningOp();
  if (!def)
    return {false, true};
  const auto anyOf = [&](::mlir::ValueRange inputs) {
    CausalAnswer combined{false, true};
    for (::mlir::Value input : inputs) {
      const CausalAnswer answer = causallyDependsOn(input, query);
      if (answer.depends)
        return CausalAnswer{true, true};
      combined.exact &= answer.exact;
    }
    return combined;
  };
  const auto allOf = [&](::mlir::ValueRange inputs) {
    for (::mlir::Value input : inputs) {
      const CausalAnswer answer = causallyDependsOn(input, query);
      if (!answer.depends)
        return CausalAnswer{false, answer.exact};
    }
    return CausalAnswer{true, true};
  };
  const auto through = [&](::mlir::Value input) {
    return causallyDependsOn(input, query);
  };
  if (auto sync = ::llvm::dyn_cast<::dataflow::SyncOp>(def))
    return anyOf(sync.getInputs());
  if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(def))
    return event == load.getDone() ? through(load.getCtrl())
                                   : CausalAnswer{false, true};
  if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(def))
    return event == store.getDone() ? through(store.getCtrl())
                                    : CausalAnswer{false, true};
  if (auto rmw = ::llvm::dyn_cast<::dataflow::AtomicRmwOp>(def))
    return event == rmw.getDone() ? through(rmw.getCtrl())
                                  : CausalAnswer{false, true};
  if (auto cmp = ::llvm::dyn_cast<::dataflow::CmpXchgOp>(def))
    return event == cmp.getDone() ? through(cmp.getCtrl())
                                  : CausalAnswer{false, true};
  if (auto fence = ::llvm::dyn_cast<::dataflow::FenceOp>(def))
    return event == fence.getDone() ? through(fence.getCtrl())
                                    : CausalAnswer{false, true};
  if (auto demux = ::llvm::dyn_cast<::dataflow::DemuxOp>(def))
    return through(demux.getInput());
  if (auto mux = ::llvm::dyn_cast<::dataflow::MuxOp>(def))
    return allOf(mux.getInputs());
  if (auto carry = ::llvm::dyn_cast<::dataflow::CarryOp>(def)) {
    const ::llvm::SmallVector<::mlir::Value, 2> sides{carry.getInit(),
                                                       carry.getCarry()};
    return allOf(sides);
  }
  if (auto invariant = ::llvm::dyn_cast<::dataflow::InvariantOp>(def))
    return through(invariant.getInit());
  if (auto gate = ::llvm::dyn_cast<::dataflow::GateOp>(def))
    return through(gate.getBeforeValue());
  if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(def))
    return through(constant.getCtrl());
  return {false, true};
}

bool causallyDependsOn(::mlir::Value event,
                       ::mlir::Value prerequisite) {
  CausalQuery query;
  query.prerequisite = prerequisite;
  return causallyDependsOn(event, query).depends;
}

} // namespace

namespace loom::lowering {

::llvm::SmallVector<::mlir::Value, 4>
reduceEvents(::mlir::ValueRange inputs) {
  ::llvm::SmallVector<::mlir::Value, 4> unique;
  for (::mlir::Value input : inputs)
    if (input && !::llvm::is_contained(unique, input))
      unique.push_back(input);

  ::llvm::SmallVector<::mlir::Value, 4> reduced;
  for (unsigned i = 0; i < unique.size(); ++i) {
    bool covered = false;
    for (unsigned j = 0; j < unique.size(); ++j) {
      if (i != j && causallyDependsOn(unique[j], unique[i])) {
        covered = true;
        break;
      }
    }
    if (!covered)
      reduced.push_back(unique[i]);
  }
  return reduced;
}

} // namespace loom::lowering
