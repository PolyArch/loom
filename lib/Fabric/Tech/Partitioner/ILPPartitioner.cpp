#include "Fabric/Tech/Partitioner/ILPPartitioner.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"
#include "PartitionerCommon.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

#ifdef LOOM_HAS_ILP
#include "Highs.h"
#endif

namespace fabric {

namespace {

// Count the non-terminator ops in the graph body.
unsigned countOps(::dataflow::GraphOp graph) {
  unsigned n = 0;
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;
    ++n;
  }
  return n;
}

// Inspect the candidate cache: returns true if any op carries a multi-op
// template candidate. The simplified single-op MIP cannot model multi-op
// covering, so when this is true we delegate to greedy.
bool anyMultiOpCandidate(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                         const CandidateCache &cache) {
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;
    for (unsigned id : cache.templatesForOp(&op)) {
      if (lib.templates()[id].bodyOpCount > 1)
        return true;
    }
  }
  return false;
}

// Emit a module-level warning explaining why we are falling back to greedy.
PartitionResult fallbackToGreedy(::dataflow::GraphOp graph,
                                 const TemplateLibrary &lib,
                                 const ::loom::TechMapConfig &cfg,
                                 ::llvm::StringRef reason) {
  if (auto module =
          graph->getParentOfType<::mlir::ModuleOp>()) {
    module->emitWarning()
        << "loom-ilp-partitioner: " << reason
        << "; falling back to greedy partitioner";
  } else {
    graph->emitWarning()
        << "loom-ilp-partitioner: " << reason
        << "; falling back to greedy partitioner";
  }
  GreedyPartitioner greedy;
  return greedy.run(graph, lib, cfg);
}

} // namespace

#ifdef LOOM_HAS_ILP

PartitionResult ILPPartitioner::run(::dataflow::GraphOp graph,
                                    const TemplateLibrary &lib,
                                    const ::loom::TechMapConfig &cfg) {
  unsigned n = countOps(graph);
  if (n > kILPMaxOps) {
    return fallbackToGreedy(
        graph, lib, cfg,
        "graph has more than the supported ILP size (" +
            std::to_string(n) + " > " + std::to_string(kILPMaxOps) + " ops)");
  }

  // The single-op MIP can only model template candidates whose bodyOpCount
  // equals 1. Build the candidate cache and bail to greedy if any op has a
  // multi-op candidate.
  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);
  if (anyMultiOpCandidate(graph, lib, cache)) {
    return fallbackToGreedy(
        graph, lib, cfg,
        "multi-op template candidate detected (single-op MIP cannot cover it)");
  }

  // Collect ops in body program order for stable indexing into the MIP.
  ::llvm::SmallVector<::mlir::Operation *> ops;
  ops.reserve(n);
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;
    ops.push_back(&op);
  }

  // For each op, list its admissible single-op template ids (filtered by
  // root-op-name match and fabric op support, mirroring the greedy logic).
  ::llvm::SmallVector<::llvm::SmallVector<unsigned>> tplIdsPerOp(ops.size());
  unsigned totalAssigns = 0;
  for (unsigned i = 0; i < ops.size(); ++i) {
    if (!::fabric::isFabricOpSupported(ops[i]->getName().getStringRef()))
      continue;
    for (unsigned id : cache.templatesForOp(ops[i])) {
      const FuTemplate &tpl = lib.templates()[id];
      if (tpl.bodyOpCount != 1)
        continue;
      if (tpl.rootOpName != ops[i]->getName().getStringRef())
        continue;
      tplIdsPerOp[i].push_back(id);
      ++totalAssigns;
    }
  }

  // Variable layout in the MIP:
  //   columns [0 .. totalAssigns)            -> x[i,t]: op i bound to tpl t
  //   columns [totalAssigns .. totalAssigns+n) -> y[i] : op i at graph level
  //
  // For each x[i,t] we record (op_index, tpl_id) so we can decode the
  // solution back into Blocks.
  struct AssignVar {
    unsigned opIdx;
    unsigned tplId;
  };
  ::llvm::SmallVector<AssignVar> assigns;
  assigns.reserve(totalAssigns);
  ::llvm::SmallVector<::llvm::SmallVector<unsigned>> assignColsForOp(
      ops.size());
  for (unsigned i = 0; i < ops.size(); ++i) {
    for (unsigned id : tplIdsPerOp[i]) {
      unsigned col = static_cast<unsigned>(assigns.size());
      assigns.push_back({i, id});
      assignColsForOp[i].push_back(col);
    }
  }
  unsigned numCols = totalAssigns + static_cast<unsigned>(ops.size());

  // Bail out cleanly when the graph is empty: no MIP needed.
  if (numCols == 0) {
    PartitionResult result;
    return result;
  }

  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("log_to_console", false);
  highs.setOptionValue("threads", 1);

  // Variables: each is binary in [0,1], integer.
  // Objective:
  //   per-template-assign x[i,t]: cost alpha (block count contribution)
  //   per graph-level slot y[i] : cost alpha + 1 (graph-level slot is
  //                               strictly worse than any binding, so the
  //                               MIP prefers to cover ops with a template
  //                               whenever the candidate cache offers one)
  // cross_edges and density terms are not modeled in this initial single-op
  // MIP path; documented as a known limitation. The MIP therefore agrees
  // with greedy on small inputs that have only single-op templates: every
  // op gets bound when possible, otherwise it stays at graph level.
  std::vector<double> colCost(numCols, 0.0);
  std::vector<double> colLower(numCols, 0.0);
  std::vector<double> colUpper(numCols, 1.0);
  for (unsigned col = 0; col < totalAssigns; ++col)
    colCost[col] = cfg.alpha;
  for (unsigned i = 0; i < ops.size(); ++i)
    colCost[totalAssigns + i] = cfg.alpha + 1.0;

  highs.addCols(static_cast<HighsInt>(numCols), colCost.data(),
                colLower.data(), colUpper.data(),
                /*num_new_nz=*/0, /*starts=*/nullptr,
                /*indices=*/nullptr, /*values=*/nullptr);

  std::vector<HighsVarType> integrality(numCols, HighsVarType::kInteger);
  highs.changeColsIntegrality(0, static_cast<HighsInt>(numCols) - 1,
                              integrality.data());

  // Coverage constraint: for each op i, sum_t x[i,t] + y[i] = 1.
  for (unsigned i = 0; i < ops.size(); ++i) {
    std::vector<HighsInt> idx;
    std::vector<double> val;
    idx.reserve(assignColsForOp[i].size() + 1);
    val.reserve(assignColsForOp[i].size() + 1);
    for (unsigned col : assignColsForOp[i]) {
      idx.push_back(static_cast<HighsInt>(col));
      val.push_back(1.0);
    }
    idx.push_back(static_cast<HighsInt>(totalAssigns + i));
    val.push_back(1.0);
    highs.addRow(1.0, 1.0, static_cast<HighsInt>(idx.size()), idx.data(),
                 val.data());
  }

  HighsStatus status = highs.run();
  if (status != HighsStatus::kOk &&
      status != HighsStatus::kWarning) {
    return fallbackToGreedy(graph, lib, cfg, "HiGHS solver failed");
  }
  HighsModelStatus mstatus = highs.getModelStatus();
  if (mstatus != HighsModelStatus::kOptimal) {
    return fallbackToGreedy(graph, lib, cfg,
                            "HiGHS did not return an optimal solution");
  }

  const HighsSolution &sol = highs.getSolution();
  if (sol.col_value.size() != numCols) {
    return fallbackToGreedy(graph, lib, cfg,
                            "HiGHS solution had unexpected dimensions");
  }

  // Decode: pick the chosen template (or graph-level slot) for each op.
  // Threshold at 0.5 since variables are binary; HiGHS may return values
  // within a small numerical tolerance of {0,1}.
  PartitionResult result;
  result.blocks.reserve(ops.size());
  for (unsigned i = 0; i < ops.size(); ++i) {
    Block b;
    b.id = static_cast<unsigned>(result.blocks.size());
    b.ops.push_back(ops[i]);
    b.tpl = nullptr;
    for (unsigned col : assignColsForOp[i]) {
      if (sol.col_value[col] > 0.5) {
        b.tpl = &lib.templates()[assigns[col].tplId];
        break;
      }
    }
    result.blocks.push_back(std::move(b));
  }
  return result;
}

#else // !LOOM_HAS_ILP

PartitionResult ILPPartitioner::run(::dataflow::GraphOp graph,
                                    const TemplateLibrary &lib,
                                    const ::loom::TechMapConfig &cfg) {
  return fallbackToGreedy(graph, lib, cfg,
                          "ILP support not compiled in (LOOM_ENABLE_ILP=OFF)");
}

#endif // LOOM_HAS_ILP

} // namespace fabric
