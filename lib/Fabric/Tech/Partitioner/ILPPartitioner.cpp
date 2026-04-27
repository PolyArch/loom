#include "Fabric/Tech/Partitioner/ILPPartitioner.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"
#include "PartitionerCommon.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
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

// Read the LOOM_ILP_TIMEOUT_S env var; default to 30 seconds when unset
// or unparseable. A non-positive value is treated as "no time limit".
double readTimeoutSeconds() {
  const char *raw = std::getenv("LOOM_ILP_TIMEOUT_S");
  if (raw == nullptr || raw[0] == '\0')
    return 30.0;
  char *endp = nullptr;
  double v = std::strtod(raw, &endp);
  if (endp == raw)
    return 30.0;
  return v;
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

  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);

  // Collect ops in body program order for stable indexing into the MIP.
  ::llvm::SmallVector<::mlir::Operation *> ops;
  ops.reserve(n);
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;
    ops.push_back(&op);
  }

  // Reverse map: op pointer -> index in `ops`. Used to translate the VF2
  // candidate's op-pointer set into op indices for coverage constraints.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opIdx;
  opIdx.reserve(ops.size());
  for (unsigned i = 0; i < ops.size(); ++i)
    opIdx[ops[i]] = i;

  // Per-rootOpName max bodyOpCount across templates, used for the gamma
  // density term. Mirrors maxTemplateSizeByRoot in CostModel.cpp.
  ::llvm::StringMap<unsigned> maxByRoot;
  for (const FuTemplate &t : lib.templates()) {
    if (t.rootOpName.empty())
      continue;
    auto &slot = maxByRoot[t.rootOpName];
    if (t.bodyOpCount > slot)
      slot = t.bodyOpCount;
  }

  // Enumerate the MIP's x-variables: one per (op_i, template_t) such that
  //   * t.bodyOpCount >= 1,
  //   * t.rootOpName == op_i's op name,
  //   * collectMultiOpCandidate(op_i, t) returns a non-empty op set entirely
  //     contained in the partitionable set (i.e. every covered op is in
  //     `ops`).
  // We record the covered op-indices for downstream coverage / cross-edge
  // constraints.
  struct AssignVar {
    unsigned opIdx;                                 // root op index
    unsigned tplId;                                 // template id in lib
    ::llvm::SmallVector<unsigned, 4> coveredOps;    // indices into `ops`
    unsigned bodySize = 0;                          // K_t = template size
    unsigned maxRoot = 1;                           // M_t for normalization
  };
  ::llvm::SmallVector<AssignVar> assigns;
  // assignColsForOp[j] = list of x-variable column indices that cover op j.
  ::llvm::SmallVector<::llvm::SmallVector<unsigned, 4>> assignColsForOp(
      ops.size());

  for (unsigned i = 0; i < ops.size(); ++i) {
    if (!::fabric::isFabricOpSupported(ops[i]->getName().getStringRef()))
      continue;
    ::llvm::ArrayRef<unsigned> tplIds = cache.templatesForOp(ops[i]);
    for (unsigned id : tplIds) {
      const FuTemplate &tpl = lib.templates()[id];
      if (tpl.bodyOpCount == 0)
        continue;
      if (tpl.rootOpName != ops[i]->getName().getStringRef())
        continue;

      ::llvm::SmallVector<::mlir::Operation *> covered;
      if (tpl.bodyOpCount == 1) {
        covered.push_back(ops[i]);
      } else {
        covered = collectMultiOpCandidate(ops[i], tpl);
        if (covered.empty())
          continue;
      }

      AssignVar av;
      av.opIdx = i;
      av.tplId = id;
      av.bodySize = tpl.bodyOpCount;
      auto mit = maxByRoot.find(tpl.rootOpName);
      av.maxRoot = (mit == maxByRoot.end()) ? 1u : std::max(1u, mit->second);
      av.coveredOps.reserve(covered.size());
      bool allInPartition = true;
      for (::mlir::Operation *cop : covered) {
        auto oit = opIdx.find(cop);
        if (oit == opIdx.end()) {
          allInPartition = false;
          break;
        }
        av.coveredOps.push_back(oit->second);
      }
      if (!allInPartition)
        continue;

      unsigned col = static_cast<unsigned>(assigns.size());
      for (unsigned j : av.coveredOps)
        assignColsForOp[j].push_back(col);
      assigns.push_back(std::move(av));
    }
  }

  unsigned numAssign = static_cast<unsigned>(assigns.size());
  unsigned numOps = static_cast<unsigned>(ops.size());

  // Enumerate cross-edge variables: one e[j, k] per SSA def-use edge where
  // both endpoints are in the partitionable op set. We deduplicate so the
  // same (j, k) pair from multiple operand positions counts only once
  // structurally — the cost model's CostModel-style summation per operand
  // would over-count when the consumer reads the same producer twice; we
  // approximate the mean direction by summing distinct edges. This matches
  // the spirit of the gamma linearization (sum vs mean).
  struct Edge {
    unsigned consumer; // op index (j)
    unsigned producer; // op index (k)
  };
  ::llvm::SmallVector<Edge> edges;
  ::llvm::DenseSet<uint64_t> seen;
  for (unsigned j = 0; j < numOps; ++j) {
    ::mlir::Operation *u = ops[j];
    for (::mlir::Value v : u->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (def == nullptr)
        continue;
      auto it = opIdx.find(def);
      if (it == opIdx.end())
        continue;
      unsigned k = it->second;
      if (k == j)
        continue;
      uint64_t key = (static_cast<uint64_t>(j) << 32) | static_cast<uint64_t>(k);
      if (!seen.insert(key).second)
        continue;
      Edge e;
      e.consumer = j;
      e.producer = k;
      edges.push_back(e);
    }
  }
  unsigned numEdges = static_cast<unsigned>(edges.size());

  // Variable layout:
  //   [0 .. numAssign)                        -> x[i, t]
  //   [numAssign .. numAssign + numOps)       -> y[j]
  //   [numAssign + numOps .. numAssign + numOps + numEdges) -> e[j, k]
  unsigned yBase = numAssign;
  unsigned eBase = numAssign + numOps;
  unsigned numCols = eBase + numEdges;

  // Bail out cleanly when the graph is empty: no MIP needed.
  if (numCols == 0) {
    PartitionResult result;
    return result;
  }

  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("log_to_console", false);
  highs.setOptionValue("threads", 1);
  double timeoutS = readTimeoutSeconds();
  if (timeoutS > 0.0)
    highs.setOptionValue("time_limit", timeoutS);

  // Objective coefficients.
  //
  //   x[i, t] : alpha + gamma * (1 - K_t/M_t)   (per-block alpha cost plus
  //                                              per-block density deficit
  //                                              penalty -- see note below)
  //   y[j]    : alpha + 1.0                     (graph-level slot strictly
  //                                              worse than any binding so the
  //                                              optimizer covers ops when it
  //                                              can; matches the prior
  //                                              single-op formulation)
  //   e[j, k] : beta                            (cross-edge penalty)
  //
  // Density linearization. The cost model defines avg_density as the mean
  // of K_t/M_t across bound blocks; the MIP encodes the per-block density
  // *deficit* (1 - K_t/M_t), summed over all bound blocks. Two consequences:
  //
  //   * The mean is a non-linear ratio (numerator / |bound blocks|); the
  //     deficit linearization replaces it with a linear sum so the MIP stays
  //     in standard form.
  //   * Per block the deficit is non-negative: it is 0 when the chosen
  //     template fully utilizes the largest available template for that
  //     root, and 1 - K/M > 0 otherwise. Higher gamma therefore strictly
  //     prefers larger templates (covering more ops per block), matching
  //     the cost-model intent. Picking more blocks just to inflate
  //     `sum K/M` is not a useful escape because every additional block
  //     pays alpha and a non-negative deficit.
  std::vector<double> colCost(numCols, 0.0);
  std::vector<double> colLower(numCols, 0.0);
  std::vector<double> colUpper(numCols, 1.0);
  for (unsigned col = 0; col < numAssign; ++col) {
    const AssignVar &av = assigns[col];
    double density =
        static_cast<double>(av.bodySize) / static_cast<double>(av.maxRoot);
    double deficit = 1.0 - density;
    if (deficit < 0.0)
      deficit = 0.0;
    colCost[col] = cfg.alpha + cfg.gamma * deficit;
  }
  for (unsigned j = 0; j < numOps; ++j)
    colCost[yBase + j] = cfg.alpha + 1.0;
  for (unsigned ei = 0; ei < numEdges; ++ei)
    colCost[eBase + ei] = cfg.beta;

  highs.addCols(static_cast<HighsInt>(numCols), colCost.data(),
                colLower.data(), colUpper.data(),
                /*num_new_nz=*/0, /*starts=*/nullptr,
                /*indices=*/nullptr, /*values=*/nullptr);

  std::vector<HighsVarType> integrality(numCols, HighsVarType::kInteger);
  highs.changeColsIntegrality(0, static_cast<HighsInt>(numCols) - 1,
                              integrality.data());

  // Coverage constraint: for each op j,
  //   sum_{(i,t) covering j} x[i, t]  +  y[j]  =  1
  for (unsigned j = 0; j < numOps; ++j) {
    std::vector<HighsInt> idx;
    std::vector<double> val;
    idx.reserve(assignColsForOp[j].size() + 1);
    val.reserve(assignColsForOp[j].size() + 1);
    for (unsigned col : assignColsForOp[j]) {
      idx.push_back(static_cast<HighsInt>(col));
      val.push_back(1.0);
    }
    idx.push_back(static_cast<HighsInt>(yBase + j));
    val.push_back(1.0);
    highs.addRow(1.0, 1.0, static_cast<HighsInt>(idx.size()), idx.data(),
                 val.data());
  }

  // Cross-edge constraints. For each edge (j, k):
  //   e[j, k]  +  sum_{(i,t) covering BOTH j AND k} x[i, t]  >= 1
  //   e[j, k]  >=  y[j]
  //   e[j, k]  >=  y[k]
  // The first row encodes "if no x covers both j and k together, then they
  // are in different blocks"; the latter two encode "graph-level on either
  // side counts as a cross edge", matching the CostModel definition.
  for (unsigned ei = 0; ei < numEdges; ++ei) {
    const Edge &edge = edges[ei];
    // Compute the intersection of assignColsForOp[j] and assignColsForOp[k].
    // Both lists are appended in column-creation order which is also sorted
    // (we generate assigns in increasing op index, then in increasing
    // template id). For determinism iterate intersection via a small set.
    ::llvm::DenseSet<unsigned> setJ(assignColsForOp[edge.consumer].begin(),
                                    assignColsForOp[edge.consumer].end());
    std::vector<HighsInt> idx;
    std::vector<double> val;
    idx.reserve(8);
    val.reserve(8);
    idx.push_back(static_cast<HighsInt>(eBase + ei));
    val.push_back(1.0);
    for (unsigned col : assignColsForOp[edge.producer]) {
      if (setJ.contains(col)) {
        idx.push_back(static_cast<HighsInt>(col));
        val.push_back(1.0);
      }
    }
    // e[j,k] + sum >= 1  -->  HiGHS row range [1, +inf).
    highs.addRow(1.0, kHighsInf, static_cast<HighsInt>(idx.size()), idx.data(),
                 val.data());

    // e[j,k] - y[j] >= 0  -->  range [0, +inf).
    {
      std::vector<HighsInt> idx2 = {
          static_cast<HighsInt>(eBase + ei),
          static_cast<HighsInt>(yBase + edge.consumer)};
      std::vector<double> val2 = {1.0, -1.0};
      highs.addRow(0.0, kHighsInf, static_cast<HighsInt>(idx2.size()),
                   idx2.data(), val2.data());
    }
    // e[j,k] - y[k] >= 0.
    {
      std::vector<HighsInt> idx2 = {
          static_cast<HighsInt>(eBase + ei),
          static_cast<HighsInt>(yBase + edge.producer)};
      std::vector<double> val2 = {1.0, -1.0};
      highs.addRow(0.0, kHighsInf, static_cast<HighsInt>(idx2.size()),
                   idx2.data(), val2.data());
    }
  }

  HighsStatus status = highs.run();
  if (status != HighsStatus::kOk &&
      status != HighsStatus::kWarning) {
    return fallbackToGreedy(graph, lib, cfg, "HiGHS solver failed");
  }
  HighsModelStatus mstatus = highs.getModelStatus();
  if (mstatus == HighsModelStatus::kTimeLimit) {
    return fallbackToGreedy(graph, lib, cfg,
                            "HiGHS exceeded LOOM_ILP_TIMEOUT_S");
  }
  if (mstatus != HighsModelStatus::kOptimal) {
    return fallbackToGreedy(graph, lib, cfg,
                            "HiGHS did not return an optimal solution");
  }

  const HighsSolution &sol = highs.getSolution();
  if (sol.col_value.size() != numCols) {
    return fallbackToGreedy(graph, lib, cfg,
                            "HiGHS solution had unexpected dimensions");
  }

  // Decode: every op j ends up in exactly one bound block (when some x[i,t]
  // covering j is 1) or at the graph level (when y[j] is 1). Threshold at
  // 0.5 since variables are binary; HiGHS may return values within a small
  // numerical tolerance of {0,1}.
  //
  // Block creation order: program order of the root op `ops[i]`. This keeps
  // the materializer's textual output stable across template ids.
  ::llvm::SmallVector<unsigned> chosenAssignForOp(numOps, /*sentinel=*/UINT_MAX);
  for (unsigned col = 0; col < numAssign; ++col) {
    if (sol.col_value[col] <= 0.5)
      continue;
    const AssignVar &av = assigns[col];
    for (unsigned j : av.coveredOps)
      chosenAssignForOp[j] = col;
  }

  PartitionResult result;
  result.blocks.reserve(numOps);
  ::llvm::DenseSet<unsigned> emittedAssignCols;
  for (unsigned j = 0; j < numOps; ++j) {
    if (chosenAssignForOp[j] == UINT_MAX) {
      // Graph-level (y[j] is 1, or no covering x and y is forced).
      Block b;
      b.id = static_cast<unsigned>(result.blocks.size());
      b.ops.push_back(ops[j]);
      b.tpl = nullptr;
      result.blocks.push_back(std::move(b));
      continue;
    }
    unsigned col = chosenAssignForOp[j];
    if (!emittedAssignCols.insert(col).second)
      continue; // already emitted as part of an earlier root op
    const AssignVar &av = assigns[col];
    Block b;
    b.id = static_cast<unsigned>(result.blocks.size());
    b.ops.reserve(av.coveredOps.size());
    for (unsigned cidx : av.coveredOps)
      b.ops.push_back(ops[cidx]);
    b.tpl = &lib.templates()[av.tplId];
    result.blocks.push_back(std::move(b));
  }

  // Post-solve cycle repair.
  //
  // The MIP minimizes the cost above subject to coverage and cross-edge
  // constraints, but it has no direct acyclicity requirement. On graphs
  // with SSA feedback (e.g. dataflow.carry whose carry input is produced
  // by an op consuming carry's own result), the optimal MIP assignment can
  // bind both ops in the cycle into separate subgraphs that mutually
  // reference each other, which would emit IR violating AC-CORR-3.
  //
  // We detect any such multi-block SCC over the bound blocks and demote
  // members to graph level (tpl = nullptr) until the SCC is broken.
  // Demotion order: largest template id first (the cheaper template wins
  // and stays bound), tie-break by smallest first-op program position so
  // the same input always demotes the same op set.
  bool warned = false;
  while (partitionHasMultiBlockCycle(result)) {
    ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
    for (const Block &b : result.blocks)
      if (b.tpl != nullptr)
        for (::mlir::Operation *op : b.ops)
          opToBlock[op] = b.id;

    unsigned nb = 0;
    for (const Block &b : result.blocks)
      if (b.id + 1 > nb)
        nb = b.id + 1;
    ::llvm::SmallVector<PendingBlock> blocks(nb);
    for (const Block &b : result.blocks) {
      if (b.tpl == nullptr)
        continue;
      PendingBlock pb;
      pb.ops.append(b.ops.begin(), b.ops.end());
      pb.tpl = b.tpl;
      blocks[b.id] = std::move(pb);
    }
    auto blockEdges = collectBlockEdges(blocks, opToBlock);
    ReachMatrix verify;
    verify.rebuild(nb, blockEdges);

    ::llvm::SmallVector<unsigned> cycleMembers;
    for (unsigned i = 0; i < nb; ++i) {
      if (blocks[i].tpl == nullptr)
        continue;
      if (i >= verify.rows.size())
        continue;
      const ::llvm::BitVector &row_i = verify.rows[i];
      for (unsigned k = 0; k < nb; ++k) {
        if (k == i)
          continue;
        if (blocks[k].tpl == nullptr)
          continue;
        if (k >= row_i.size() || !row_i.test(k))
          continue;
        if (k >= verify.rows.size() || i >= verify.rows[k].size())
          continue;
        if (verify.rows[k].test(i)) {
          cycleMembers.push_back(i);
          break;
        }
      }
    }

    if (cycleMembers.empty())
      break; // defensive

    auto firstOpPos = [&](unsigned bi) -> unsigned {
      unsigned best = static_cast<unsigned>(ops.size());
      for (::mlir::Operation *op : blocks[bi].ops) {
        for (unsigned k = 0; k < ops.size(); ++k) {
          if (ops[k] == op) {
            if (k < best)
              best = k;
            break;
          }
        }
      }
      return best;
    };
    auto victimTplId = [&](unsigned bi) -> unsigned {
      const FuTemplate *tpl = blocks[bi].tpl;
      for (unsigned k = 0; k < lib.templates().size(); ++k)
        if (&lib.templates()[k] == tpl)
          return k;
      return 0u;
    };
    unsigned victim = cycleMembers.front();
    unsigned bestTplId = victimTplId(victim);
    unsigned bestPos = firstOpPos(victim);
    for (unsigned m : cycleMembers) {
      unsigned mTplId = victimTplId(m);
      unsigned mPos = firstOpPos(m);
      bool better = false;
      if (mTplId > bestTplId)
        better = true;
      else if (mTplId == bestTplId) {
        if (mPos < bestPos)
          better = true;
        else if (mPos == bestPos && m < victim)
          better = true;
      }
      if (better) {
        victim = m;
        bestTplId = mTplId;
        bestPos = mPos;
      }
    }

    for (Block &b : result.blocks) {
      if (b.id == victim) {
        b.tpl = nullptr;
        break;
      }
    }
    if (!warned) {
      if (auto module = graph->getParentOfType<::mlir::ModuleOp>())
        module->emitWarning()
            << "loom-ilp-partitioner: HiGHS solution induced a multi-block "
               "SSA cycle; demoting block(s) to graph level to satisfy "
               "AC-CORR-3";
      else
        graph->emitWarning()
            << "loom-ilp-partitioner: HiGHS solution induced a multi-block "
               "SSA cycle; demoting block(s) to graph level to satisfy "
               "AC-CORR-3";
      warned = true;
    }
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
