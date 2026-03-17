#include "fcc/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#define DEBUG_TYPE "fcc-mark-dfg-domain"

using namespace mlir;

namespace {

// Resource estimate for a candidate region, used for quick ADG feasibility.
struct ResourceEstimate {
  unsigned numArithOps = 0;
  unsigned numMemPorts = 0;
  unsigned numControlOps = 0;
  unsigned maxDataWidth = 0;
  unsigned estimatedPECount = 0;
  unsigned estimatedMemCount = 0;
  unsigned stateCarryCount = 0;
};

// Parameters for a DFG candidate variant.
struct DFGParams {
  unsigned unrollFactor = 1;
  unsigned parallelDegree = 1;
  std::string regionId;
};

// Represents a candidate region for DFG conversion.
struct DFGCandidate {
  enum Kind { InnerLoop, LoopNest, WholeFunction };

  Kind kind;
  Operation *regionRoot;
  func::FuncOp parentFunc;
  DFGParams params;
  ResourceEstimate resources;
  std::optional<bool> feasible;
  std::optional<double> mappingCost;

  std::string describe() const {
    std::string kindStr;
    switch (kind) {
    case InnerLoop:
      kindStr = "inner-loop";
      break;
    case LoopNest:
      kindStr = "loop-nest";
      break;
    case WholeFunction:
      kindStr = "whole-func";
      break;
    }
    return kindStr + ":" + params.regionId +
           " (unroll=" + std::to_string(params.unrollFactor) +
           ", PEs~" + std::to_string(resources.estimatedPECount) +
           ", mem~" + std::to_string(resources.estimatedMemCount) + ")";
  }
};

// ADG capacity summary for feasibility checking.
struct ADGCapacity {
  unsigned totalPEs = 0;
  unsigned totalFUs = 0;
  unsigned totalMemModules = 0;
  unsigned maxDataWidth = 64;

  bool isValid() const { return totalPEs > 0 || totalFUs > 0; }
};

static unsigned getComputeCapacity(const ADGCapacity &adg) {
  if (adg.totalFUs > 0)
    return adg.totalFUs;
  return adg.totalPEs;
}

// Count resources in an operation region for quick estimation.
static ResourceEstimate estimateResources(Operation *root) {
  ResourceEstimate est;
  llvm::DenseSet<Value> memrefsSeen;

  if (auto forOp = dyn_cast<scf::ForOp>(root))
    est.stateCarryCount = static_cast<unsigned>(forOp.getInitArgs().size());
  else if (auto whileOp = dyn_cast<scf::WhileOp>(root))
    est.stateCarryCount =
        std::max(static_cast<unsigned>(whileOp.getBeforeArguments().size()),
                 whileOp.getNumResults());

  root->walk([&](Operation *op) {
    // Arithmetic operations
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
            arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
            arith::ShRUIOp>(op)) {
      est.numArithOps++;
    }
    // Floating-point arithmetic
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(op)) {
      est.numArithOps++;
    }
    // Comparison
    if (isa<arith::CmpIOp, arith::CmpFOp>(op)) {
      est.numArithOps++;
    }
    // Math ops
    if (isa<math::SqrtOp, math::ExpOp, math::LogOp>(op)) {
      est.numArithOps++;
    }
    // Extension/truncation
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
      est.numArithOps++;
    }
    // Memory ports
    if (isa<memref::LoadOp>(op)) {
      est.numMemPorts++;
      auto loadOp = cast<memref::LoadOp>(op);
      memrefsSeen.insert(loadOp.getMemRef());
    }
    if (isa<memref::StoreOp>(op)) {
      est.numMemPorts++;
      auto storeOp = cast<memref::StoreOp>(op);
      memrefsSeen.insert(storeOp.getMemRef());
    }
    // Control ops (branches, select/mux)
    if (isa<arith::SelectOp>(op))
      est.numControlOps++;
    if (isa<scf::YieldOp>(op))
      est.numControlOps++;

    // Track data widths
    for (auto result : op->getResults()) {
      if (auto intTy = dyn_cast<IntegerType>(result.getType())) {
        est.maxDataWidth = std::max(est.maxDataWidth, intTy.getWidth());
      }
    }
  });

  // Each distinct memref is roughly one external memory interface
  est.estimatedMemCount = memrefsSeen.size();

  // PE count estimate: arith + control + some overhead for routing/merge
  unsigned overhead = (est.numArithOps + est.numControlOps > 0) ? 2 : 0;
  est.estimatedPECount = est.numArithOps + est.numControlOps + overhead;

  return est;
}

// Check if an operation contains unsupported constructs for DFG conversion.
static bool hasUnsupportedOps(Operation *root) {
  bool unsupported = false;
  root->walk([&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      // External calls (printf, etc.) are unsupported in DFG
      unsupported = true;
    }
  });
  return unsupported;
}

// Check if a loop is an inner (leaf) loop with no nested loops.
static bool isInnerLoop(Operation *op) {
  bool hasNestedLoop = false;
  op->walk([&](Operation *inner) {
    if (inner != op && isa<scf::ForOp, scf::WhileOp>(inner))
      hasNestedLoop = true;
  });
  return !hasNestedLoop;
}

// Enumerate DFG candidate regions from a function.
static void enumerateCandidates(func::FuncOp func,
                                SmallVectorImpl<DFGCandidate> &candidates) {
  if (func.isDeclaration() || func.getName() == "main")
    return;

  std::string funcName = func.getName().str();

  // Track whether the whole function is viable
  bool funcHasLoop = false;
  bool funcHasMemAccess = false;
  bool funcBodyUnsupported = hasUnsupportedOps(func);

  // Walk bottom-up: find inner loops first
  unsigned loopIdx = 0;
  func.walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::WhileOp>(op)) {
      funcHasLoop = true;

      // Check for memory access in the loop body
      bool loopHasMem = false;
      op->walk([&](Operation *inner) {
        if (isa<memref::LoadOp, memref::StoreOp>(inner))
          loopHasMem = true;
      });
      if (loopHasMem)
        funcHasMemAccess = true;

      if (hasUnsupportedOps(op))
        return;

      // Candidate: inner loop
      if (isInnerLoop(op)) {
        DFGCandidate c;
        c.kind = DFGCandidate::InnerLoop;
        c.regionRoot = op;
        c.parentFunc = func;
        c.params.unrollFactor = 1;
        c.params.parallelDegree = 1;
        c.params.regionId =
            funcName + "/loop_" + std::to_string(loopIdx);
        c.resources = estimateResources(op);
        candidates.push_back(c);
      } else {
        // Candidate: loop nest (contains sub-loops)
        DFGCandidate c;
        c.kind = DFGCandidate::LoopNest;
        c.regionRoot = op;
        c.parentFunc = func;
        c.params.unrollFactor = 1;
        c.params.parallelDegree = 1;
        c.params.regionId =
            funcName + "/nest_" + std::to_string(loopIdx);
        c.resources = estimateResources(op);
        candidates.push_back(c);
      }
      loopIdx++;
    }
  });

  // Candidate: whole function body (if it has loops, memory, no unsupported)
  if (funcHasLoop && funcHasMemAccess && !funcBodyUnsupported) {
    DFGCandidate c;
    c.kind = DFGCandidate::WholeFunction;
    c.regionRoot = func;
    c.parentFunc = func;
    c.params.unrollFactor = 1;
    c.params.parallelDegree = 1;
    c.params.regionId = funcName + "/body";
    c.resources = estimateResources(func);
    candidates.push_back(c);
  }
}

// Quick feasibility check against ADG capacity.
// Returns true if the candidate fits the ADG resources.
static bool checkFeasibility(DFGCandidate &candidate,
                             const ADGCapacity &adg) {
  if (!adg.isValid()) {
    // No ADG provided; assume feasible (compilation-only mode)
    candidate.feasible = true;
    return true;
  }

  const auto &r = candidate.resources;
  const unsigned computeCapacity = getComputeCapacity(adg);
  bool fits = true;
  if (computeCapacity > 0 && r.estimatedPECount > computeCapacity) {
    LLVM_DEBUG(llvm::dbgs()
               << "  INFEASIBLE: compute " << r.estimatedPECount
               << " > capacity " << computeCapacity << "\n");
    fits = false;
  }
  if (r.estimatedMemCount > adg.totalMemModules) {
    LLVM_DEBUG(llvm::dbgs()
               << "  INFEASIBLE: mem " << r.estimatedMemCount
               << " > modules " << adg.totalMemModules << "\n");
    fits = false;
  }
  if (r.maxDataWidth > adg.maxDataWidth) {
    LLVM_DEBUG(llvm::dbgs()
               << "  INFEASIBLE: width " << r.maxDataWidth
               << " > max " << adg.maxDataWidth << "\n");
    fits = false;
  }

  candidate.feasible = fits;
  return fits;
}

static int getKindRank(DFGCandidate::Kind kind) {
  switch (kind) {
  case DFGCandidate::InnerLoop:
    return 0;
  case DFGCandidate::LoopNest:
    return 1;
  case DFGCandidate::WholeFunction:
    return 2;
  }
  return 3;
}

// Select the cheapest feasible candidate. Smaller regions are preferred when
// they fit because they reduce mapper pressure and better match the current
// host-accel split contract.
static DFGCandidate *selectBest(SmallVectorImpl<DFGCandidate> &candidates) {
  bool hasFeasibleMemoryCandidate = false;
  for (auto &candidate : candidates) {
    if (!candidate.feasible.value_or(false))
      continue;
    if (candidate.resources.estimatedMemCount > 0) {
      hasFeasibleMemoryCandidate = true;
      break;
    }
  }

  DFGCandidate *best = nullptr;
  for (auto &candidate : candidates) {
    if (!candidate.feasible.value_or(false))
      continue;
    if (hasFeasibleMemoryCandidate &&
        candidate.resources.estimatedMemCount == 0)
      continue;
    if (!best) {
      best = &candidate;
      continue;
    }

    const auto &lhs = candidate.resources;
    const auto &rhs = best->resources;
    int lhsMemPriority =
        hasFeasibleMemoryCandidate ? -static_cast<int>(lhs.estimatedMemCount) : 0;
    int rhsMemPriority =
        hasFeasibleMemoryCandidate ? -static_cast<int>(rhs.estimatedMemCount) : 0;
    auto lhsKey = std::tuple(lhs.stateCarryCount, lhsMemPriority,
                             lhs.estimatedPECount,
                             lhs.numControlOps, lhs.maxDataWidth,
                             getKindRank(candidate.kind),
                             candidate.params.regionId);
    auto rhsKey = std::tuple(rhs.stateCarryCount, rhsMemPriority,
                             rhs.estimatedPECount,
                             rhs.numControlOps, rhs.maxDataWidth,
                             getKindRank(best->kind), best->params.regionId);
    if (lhsKey < rhsKey)
      best = &candidate;
  }
  return best;
}

struct MarkDFGDomainPass
    : public PassWrapper<MarkDFGDomainPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkDFGDomainPass)

  StringRef getArgument() const override { return "fcc-mark-dfg-domain"; }
  StringRef getDescription() const override {
    return "DFG domain exploration: enumerate candidates, check feasibility, "
           "select best region for DFG conversion";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto ctx = module.getContext();

    // Build ADG capacity from module attributes if available.
    // The pipeline sets these from the --adg flag before running this pass.
    ADGCapacity adg;
    if (auto attr = module->getAttrOfType<IntegerAttr>("fcc.adg_total_pes"))
      adg.totalPEs = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("fcc.adg_total_fus"))
      adg.totalFUs = attr.getInt();
    if (auto attr =
            module->getAttrOfType<IntegerAttr>("fcc.adg_total_mem_modules"))
      adg.totalMemModules = attr.getInt();
    if (auto attr =
            module->getAttrOfType<IntegerAttr>("fcc.adg_max_data_width"))
      adg.maxDataWidth = attr.getInt();

    // Enumerate candidates from all non-main functions
    SmallVector<DFGCandidate> candidates;
    module.walk([&](func::FuncOp func) {
      enumerateCandidates(func, candidates);
    });

    if (candidates.empty()) {
      llvm::outs() << "fcc: no DFG candidates found\n";
      return;
    }

    llvm::outs() << "fcc: enumerated " << candidates.size()
                 << " DFG candidate(s):\n";
    for (auto &c : candidates)
      llvm::outs() << "  - " << c.describe() << "\n";

    // Quick feasibility check
    for (auto &c : candidates)
      checkFeasibility(c, adg);

    unsigned feasibleCount = 0;
    for (auto &c : candidates)
      if (c.feasible.value_or(false))
        feasibleCount++;

    llvm::outs() << "fcc: " << feasibleCount << " of " << candidates.size()
                 << " candidate(s) feasible"
                 << (adg.isValid() ? "" : " (no ADG, all assumed feasible)")
                 << "\n";

    // Select the best candidate
    DFGCandidate *selected = selectBest(candidates);
    if (!selected) {
      llvm::outs() << "fcc: no feasible DFG candidate; skipping DFG "
                      "conversion\n";
      return;
    }

    llvm::outs() << "fcc: selected candidate: " << selected->describe()
                 << "\n";

    // Mark the parent function with fcc.dfg_candidate attribute.
    // For whole-function and loop candidates, the parent function
    // is the unit of SCF->DFG conversion.
    selected->parentFunc->setAttr("fcc.dfg_candidate", UnitAttr::get(ctx));

    // Store resource estimates as attributes for downstream passes
    selected->parentFunc->setAttr(
        "fcc.dfg_region_id",
        StringAttr::get(ctx, selected->params.regionId));
    selected->parentFunc->setAttr(
        "fcc.dfg_estimated_pes",
        IntegerAttr::get(IntegerType::get(ctx, 32),
                         selected->resources.estimatedPECount));
    selected->parentFunc->setAttr(
        "fcc.dfg_estimated_mem",
        IntegerAttr::get(IntegerType::get(ctx, 32),
                         selected->resources.estimatedMemCount));
    if (selected->regionRoot != selected->parentFunc.getOperation())
      selected->regionRoot->setAttr("fcc.selected_dfg_root",
                                    UnitAttr::get(ctx));
  }
};

} // namespace

std::unique_ptr<Pass> fcc::createMarkDFGDomainPass() {
  return std::make_unique<MarkDFGDomainPass>();
}
