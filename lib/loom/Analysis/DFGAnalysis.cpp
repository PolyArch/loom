//===-- DFGAnalysis.cpp - Level A: MLIR-level DFG analysis --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Level A analysis operates directly on handshake::FuncOp:
//   1. Loop structure detection via dataflow.stream tracing
//   2. Execution frequency estimation from trip count annotations
//   3. Attaches "loom.analysis" DictionaryAttr to each operation
//
//===----------------------------------------------------------------------===//

#include "loom/Analysis/DFGAnalysis.h"

#include "loom/Dialect/Dataflow/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <queue>
#include <regex>
#include <string>

namespace loom {
namespace analysis {

namespace {

//===----------------------------------------------------------------------===//
// Trip count extraction from annotations
//===----------------------------------------------------------------------===//

/// Parse trip count from a "loom.loop.tripcount typical=N" annotation string.
/// Returns -1 if not found.
int64_t parseTripCount(llvm::StringRef annotation) {
  // Match "loom.loop.tripcount typical=<number>"
  auto pos = annotation.find("loom.loop.tripcount");
  if (pos == llvm::StringRef::npos)
    return -1;
  auto typicalPos = annotation.find("typical=", pos);
  if (typicalPos == llvm::StringRef::npos)
    return -1;
  auto numStart = typicalPos + 8; // strlen("typical=")
  auto numEnd = numStart;
  while (numEnd < annotation.size() &&
         (annotation[numEnd] >= '0' && annotation[numEnd] <= '9'))
    ++numEnd;
  if (numEnd == numStart)
    return -1;
  int64_t val = 0;
  annotation.substr(numStart, numEnd - numStart).getAsInteger(10, val);
  return val;
}

/// Extract trip count from a dataflow.stream op by searching its annotations
/// and the annotations of the enclosing handshake.func.
int64_t getStreamTripCount(mlir::Operation *streamOp, int64_t defaultVal) {
  // Check stream op's own annotations.
  if (auto annots = streamOp->getAttrOfType<mlir::ArrayAttr>("loom.annotations")) {
    for (auto a : annots) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(a)) {
        int64_t tc = parseTripCount(strAttr.getValue());
        if (tc > 0)
          return tc;
      }
    }
  }
  // Check for loom.loop.tripcount as a direct attribute.
  if (auto tcAttr = streamOp->getAttrOfType<mlir::IntegerAttr>(
          "loom.loop.tripcount")) {
    return tcAttr.getInt();
  }
  return defaultVal;
}

//===----------------------------------------------------------------------===//
// Loop body detection
//===----------------------------------------------------------------------===//

/// Represents one loop nest level detected from a dataflow.stream.
struct LoopInfo {
  mlir::Operation *streamOp = nullptr;
  mlir::Operation *gateOp = nullptr;
  int64_t tripCount = 100;

  /// The set of operations that are inside this loop's body.
  llvm::DenseSet<mlir::Operation *> bodyOps;
};

/// Trace the willContinue output of a stream to find its gate consumer.
mlir::Operation *findGateForStream(mlir::Operation *streamOp) {
  // dataflow.stream has results: index (result 0), willContinue (result 1)
  if (streamOp->getNumResults() < 2)
    return nullptr;

  mlir::Value willContinue = streamOp->getResult(1);
  for (auto &use : willContinue.getUses()) {
    auto *user = use.getOwner();
    if (mlir::isa<loom::dataflow::GateOp>(user))
      return user;
  }
  return nullptr;
}

/// Build the loop body domain: ops reachable from gate's afterValue and
/// afterCond true paths, stopping at carry back-edges and cond_br false exits.
void buildLoopBody(mlir::Operation *gateOp,
                   llvm::DenseSet<mlir::Operation *> &bodyOps,
                   const llvm::DenseSet<mlir::Operation *> &allCarryOps) {
  // Gate results: afterValue (result 0), afterCond (result 1)
  if (gateOp->getNumResults() < 2)
    return;

  mlir::Value afterCond = gateOp->getResult(1);

  // BFS forward from all gate result consumers.
  std::queue<mlir::Operation *> worklist;
  llvm::DenseSet<mlir::Operation *> visited;

  auto enqueueUsers = [&](mlir::Value val) {
    for (auto &use : val.getUses()) {
      auto *user = use.getOwner();
      if (!visited.count(user)) {
        visited.insert(user);
        worklist.push(user);
      }
    }
  };

  // Seed from gate outputs.
  enqueueUsers(gateOp->getResult(0)); // afterValue
  enqueueUsers(afterCond);            // afterCond

  while (!worklist.empty()) {
    auto *op = worklist.front();
    worklist.pop();
    bodyOps.insert(op);

    // Stop at return ops.
    if (op->hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    // For carry ops: add to body but trace only from the output (result 0).
    // Do NOT trace further from carry's consumers to avoid leaking past
    // the back-edge. The carry output feeds the loop body; the carry
    // input b (operand 2) is where the back-edge enters.
    if (allCarryOps.count(op)) {
      // Trace from carry's output to reach loop body compute ops.
      if (op->getNumResults() > 0)
        enqueueUsers(op->getResult(0));
      continue;
    }

    // For cond_br ops that consume THIS loop's gate afterCond:
    // only trace from the TRUE output (result 0). The FALSE output
    // (result 1) is the loop exit and should NOT be part of the body.
    if (mlir::isa<circt::handshake::ConditionalBranchOp>(op)) {
      bool consumesThisGate = false;
      for (auto operand : op->getOperands()) {
        if (operand == afterCond) {
          consumesThisGate = true;
          break;
        }
      }
      if (consumesThisGate) {
        // Only trace from trueResult (result 0), skip falseResult (result 1).
        if (op->getNumResults() > 0)
          enqueueUsers(op->getResult(0));
        continue;
      }
    }

    // Continue traversal through this op's result users.
    for (auto result : op->getResults())
      enqueueUsers(result);
  }
}

//===----------------------------------------------------------------------===//
// Level A main logic
//===----------------------------------------------------------------------===//

/// Assign loop depth and execution frequency to all ops in a handshake.func.
void runLevelA(circt::handshake::FuncOp funcOp,
               const DFGAnalysisConfig &config) {
  auto &block = funcOp.getBody().front();
  mlir::Builder builder(funcOp.getContext());

  // Collect all stream and carry ops.
  llvm::SmallVector<mlir::Operation *, 4> streamOps;
  llvm::DenseSet<mlir::Operation *> allCarryOps;

  for (auto &op : block) {
    if (mlir::isa<loom::dataflow::StreamOp>(&op))
      streamOps.push_back(&op);
    if (mlir::isa<loom::dataflow::CarryOp>(&op))
      allCarryOps.insert(&op);
  }

  // Build loop info for each stream.
  llvm::SmallVector<LoopInfo, 4> loops;
  for (auto *streamOp : streamOps) {
    LoopInfo loop;
    loop.streamOp = streamOp;
    loop.tripCount = getStreamTripCount(streamOp, config.defaultTripCount);
    loop.gateOp = findGateForStream(streamOp);
    if (loop.gateOp) {
      buildLoopBody(loop.gateOp, loop.bodyOps, allCarryOps);
      // The stream, gate, carry, and invariant ops controlling this loop
      // are at this loop's depth level.
      loop.bodyOps.insert(streamOp);
      loop.bodyOps.insert(loop.gateOp);
    }
    loops.push_back(std::move(loop));
  }

  // Also add carry and invariant ops to the body of their controlling loop.
  // Carry and invariant ops consume the gate's afterCond, so they should
  // already be in the body. But ensure dataflow ops that consume stream
  // directly are also counted.
  for (auto &op : block) {
    if (mlir::isa<loom::dataflow::CarryOp>(&op) ||
        mlir::isa<loom::dataflow::InvariantOp>(&op)) {
      for (auto &loop : loops) {
        if (loop.bodyOps.count(&op))
          continue;
        // Check if this op consumes any value from the loop's gate.
        if (!loop.gateOp)
          continue;
        for (auto operand : op.getOperands()) {
          if (operand.getDefiningOp() == loop.gateOp) {
            loop.bodyOps.insert(&op);
            break;
          }
        }
      }
    }
  }

  // Compute per-op loop depth = number of enclosing streams.
  // Compute exec_freq = product of trip counts of enclosing streams.
  // Also track max values for normalization.
  int64_t maxExecFreq = 1;

  struct OpInfo {
    int32_t loopDepth = 0;
    int64_t execFreq = 1;
  };
  llvm::DenseMap<mlir::Operation *, OpInfo> opInfos;

  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    OpInfo info;
    for (auto &loop : loops) {
      if (loop.bodyOps.count(&op)) {
        ++info.loopDepth;
        info.execFreq *= loop.tripCount;
      }
    }
    maxExecFreq = std::max(maxExecFreq, info.execFreq);
    opInfos[&op] = info;
  }

  // Attach analysis attributes to each op.
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    auto it = opInfos.find(&op);
    if (it == opInfos.end())
      continue;
    auto &info = it->second;

    // Compute preliminary temporal score from Level A data.
    // Level B will refine this with recurrence and critical path info.
    std::string opName = op.getName().getStringRef().str();
    double tscore = 0.0;
    if (!isForcedSpatialOp(opName)) {
      double normFreq = 0.0;
      if (maxExecFreq > 1)
        normFreq = static_cast<double>(info.execFreq - 1) /
                   static_cast<double>(maxExecFreq - 1);
      tscore = config.w1 * (1.0 - normFreq) +
               config.w2 * 1.0 + // Not on critical path (unknown yet)
               config.w3 * (info.loopDepth == 0 ? 1.0 : 0.0) +
               config.w4 * 0.5;  // Not on recurrence (unknown yet)
      tscore = std::max(0.0, std::min(1.0, tscore));
    }

    llvm::SmallVector<mlir::NamedAttribute, 6> entries;
    entries.push_back(builder.getNamedAttr(
        "loop_depth", builder.getI32IntegerAttr(info.loopDepth)));
    entries.push_back(builder.getNamedAttr(
        "exec_freq", builder.getI64IntegerAttr(info.execFreq)));
    entries.push_back(builder.getNamedAttr(
        "on_recurrence", builder.getBoolAttr(false)));
    entries.push_back(builder.getNamedAttr(
        "recurrence_id", builder.getI32IntegerAttr(-1)));
    entries.push_back(builder.getNamedAttr(
        "on_critical_path", builder.getBoolAttr(false)));
    entries.push_back(builder.getNamedAttr(
        "temporal_score", builder.getF64FloatAttr(tscore)));

    auto dictAttr = builder.getDictionaryAttr(entries);
    op.setAttr("loom.analysis", dictAttr);
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void analyzeMLIR(circt::handshake::FuncOp funcOp,
                 const DFGAnalysisConfig &config) {
  runLevelA(funcOp, config);
}

//===----------------------------------------------------------------------===//
// MLIR attribute helpers
//===----------------------------------------------------------------------===//

mlir::DictionaryAttr getAnalysisDict(mlir::Operation *op) {
  return op->getAttrOfType<mlir::DictionaryAttr>("loom.analysis");
}

bool isForcedSpatialOp(llvm::StringRef opName) {
  // Dataflow ops: exclusive PE body semantics in fabric spec.
  if (opName.starts_with("dataflow."))
    return true;
  // Handshake control/memory ops: require dedicated PEs.
  if (opName.starts_with("handshake."))
    return true;
  // Constants fold into immediates and have no runtime execution cost.
  // They should not be temporal candidates.
  if (opName == "arith.constant")
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Dump utility
//===----------------------------------------------------------------------===//

void dumpAnalysisSummary(circt::handshake::FuncOp funcOp) {
  llvm::outs() << "=== DFG Analysis: " << funcOp.getName() << " ===\n";
  llvm::outs() << llvm::formatv(
      "{0,-30} {1,>5} {2,>8} {3,>5} {4,>5} {5,>5} {6,>8}\n",
      "Operation", "Depth", "ExecFreq", "Rec", "RecID", "CP", "TScore");
  llvm::outs() << std::string(72, '-') << "\n";

  for (auto &op : funcOp.getBody().front()) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    auto dict = getAnalysisDict(&op);
    if (!dict)
      continue;

    std::string name = op.getName().getStringRef().str();

    int32_t depth = 0;
    int64_t freq = 1;
    bool rec = false;
    int32_t recId = -1;
    bool cp = false;
    double tscore = 0.0;

    if (auto a = dict.getAs<mlir::IntegerAttr>("loop_depth"))
      depth = a.getInt();
    if (auto a = dict.getAs<mlir::IntegerAttr>("exec_freq"))
      freq = a.getInt();
    if (auto a = dict.getAs<mlir::BoolAttr>("on_recurrence"))
      rec = a.getValue();
    if (auto a = dict.getAs<mlir::IntegerAttr>("recurrence_id"))
      recId = a.getInt();
    if (auto a = dict.getAs<mlir::BoolAttr>("on_critical_path"))
      cp = a.getValue();
    if (auto a = dict.getAs<mlir::FloatAttr>("temporal_score"))
      tscore = a.getValueAsDouble();

    llvm::outs() << llvm::formatv(
        "{0,-30} {1,5} {2,8} {3,5} {4,5} {5,5} {6,8}\n",
        name, depth, freq, (rec ? "T" : "F"), recId,
        (cp ? "T" : "F"), llvm::formatv("{0:f3}", tscore));
  }
  llvm::outs() << "\n";
}

} // namespace analysis
} // namespace loom
