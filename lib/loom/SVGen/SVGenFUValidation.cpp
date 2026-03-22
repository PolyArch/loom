#include "SVGenInternal.h"

#include "loom/Dialect/Fabric/FabricOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace loom {
namespace svgen {

namespace {

/// Returns true if the given MLIR op name is a dataflow state-machine op.
static bool isDataflowOp(llvm::StringRef opName) {
  return opName == "dataflow.stream" || opName == "dataflow.gate" ||
         opName == "dataflow.carry" || opName == "dataflow.invariant";
}

/// Returns true if the given MLIR op name is a structural or terminator op
/// that does not contribute to intrinsic latency classification.
static bool isNonComputeOp(llvm::StringRef opName) {
  return opName == "fabric.yield" || opName == "fabric.mux";
}

/// Return the conservative minimum intrinsic latency for a single MLIR op.
/// Returns 0 for combinational ops, a positive value for multi-cycle ops,
/// and -1 for dataflow state-machine ops.
///
/// Returns -2 if the op is not recognized (caller should handle separately).
static int getOpIntrinsicLatency(llvm::StringRef opName) {
  // Combinational (latency 0): integer, logic, bitwise, comparison, casts,
  // FP sign ops, and handshake control ops.
  if (opName == "arith.addi" || opName == "arith.subi" ||
      opName == "arith.andi" || opName == "arith.ori" ||
      opName == "arith.xori" || opName == "arith.shli" ||
      opName == "arith.shrsi" || opName == "arith.shrui" ||
      opName == "arith.extsi" || opName == "arith.extui" ||
      opName == "arith.trunci" || opName == "arith.select" ||
      opName == "arith.index_cast" || opName == "arith.index_castui" ||
      opName == "arith.negf" || opName == "arith.cmpi" ||
      opName == "arith.cmpf" || opName == "math.absf" ||
      opName == "llvm.intr.bitreverse" ||
      opName == "handshake.cond_br" || opName == "handshake.constant" ||
      opName == "handshake.join" || opName == "handshake.load" ||
      opName == "handshake.store" || opName == "handshake.mux")
    return 0;

  // Integer multi-cycle.
  if (opName == "arith.muli")
    return 1;
  if (opName == "arith.divsi" || opName == "arith.divui" ||
      opName == "arith.remsi" || opName == "arith.remui")
    return 8;

  // Floating-point multi-cycle.
  if (opName == "arith.addf" || opName == "arith.subf" ||
      opName == "arith.mulf")
    return 3;
  if (opName == "arith.divf")
    return 10;
  if (opName == "arith.fptosi" || opName == "arith.fptoui" ||
      opName == "arith.sitofp" || opName == "arith.uitofp")
    return 2;

  // Math multi-cycle.
  if (opName == "math.fma")
    return 4;
  if (opName == "math.sqrt")
    return 10;

  // Dataflow state-machine ops.
  if (isDataflowOp(opName))
    return -1;

  // Unrecognized op.
  return -2;
}

/// Compute the critical-path intrinsic latency through the FU body's SSA DAG.
/// Also detects whether any dataflow op is present.
///
/// Sets `hasDataflowOp` to true if any dataflow op is found.
/// Sets `primaryOpName` to the op on the critical path with highest
/// contribution (the op whose arrival + latency determines the body latency).
/// Returns the body's critical-path intrinsic latency from block arguments
/// to yield operands.
///
/// Algorithm: single-pass forward propagation over the SSA DAG (ops in a
/// single MLIR block are already in topological order).
///   arrival[block_arg] = 0
///   arrival[op_result] = max over operands (arrival[operand] + producer_latency)
///     where producer_latency is the intrinsic latency of the op that defines
///     the operand (0 for block arguments).
///   body_latency = max over yield operands (arrival[operand] + producer_latency)
static unsigned
computeBodyIntrinsicLatency(loom::fabric::FunctionUnitOp fuOp,
                            bool &hasDataflowOp,
                            std::string &primaryOpName) {
  hasDataflowOp = false;
  primaryOpName.clear();

  auto &bodyBlock = fuOp.getBody().front();

  // Map from SSA value to its arrival time (cycle at which the value is ready).
  // Block arguments have arrival time 0 (they are inputs available at cycle 0).
  llvm::DenseMap<mlir::Value, unsigned> arrivalTime;

  // Block arguments: arrival = 0.
  for (mlir::Value arg : bodyBlock.getArguments())
    arrivalTime[arg] = 0;

  // Forward pass: propagate arrival times through the SSA DAG.
  // Ops in a single MLIR block are in SSA dominance (topological) order.
  for (auto &op : bodyBlock.getOperations()) {
    llvm::StringRef opName = op.getName().getStringRef();

    // Detect dataflow ops.
    if (isDataflowOp(opName)) {
      hasDataflowOp = true;
      primaryOpName = opName.str();
      // Dataflow ops are body-exclusive; their latency is not modeled as a
      // scalar value, so skip DAG propagation.
      continue;
    }

    // For the yield terminator, we do not assign arrival times to its
    // "results" (it has none); its operands are handled in the final step.
    if (opName == "fabric.yield")
      continue;

    // Determine this op's intrinsic latency.
    unsigned opLatency = 0;
    if (!isNonComputeOp(opName)) {
      int intrinsic = getOpIntrinsicLatency(opName);
      if (intrinsic > 0)
        opLatency = static_cast<unsigned>(intrinsic);
    }

    // Compute arrival time for this op's results:
    // arrival[result] = max(arrival[operand]) + opLatency
    // (all operands must be ready before the op can start, then it takes
    // opLatency cycles to produce its results).
    unsigned maxOperandArrival = 0;
    for (mlir::Value operand : op.getOperands()) {
      auto it = arrivalTime.find(operand);
      if (it != arrivalTime.end())
        maxOperandArrival = std::max(maxOperandArrival, it->second);
    }

    unsigned resultArrival = maxOperandArrival + opLatency;
    for (mlir::Value result : op.getResults())
      arrivalTime[result] = resultArrival;
  }

  // If this is a dataflow body, critical-path latency is not meaningful;
  // return 0 and let the caller handle the dataflow timing class.
  if (hasDataflowOp)
    return 0;

  // Final step: the body's intrinsic latency is the maximum arrival time
  // at any yield operand. Each yield operand's arrival time already includes
  // the producing op's latency (it was added when we set arrivalTime for
  // the producer's results).
  unsigned bodyLatency = 0;
  auto yieldOp = bodyBlock.getTerminator();
  for (mlir::Value operand : yieldOp->getOperands()) {
    auto it = arrivalTime.find(operand);
    unsigned arrival = (it != arrivalTime.end()) ? it->second : 0;
    if (arrival > bodyLatency) {
      bodyLatency = arrival;
      // Identify the critical-path bottleneck op for diagnostic messages.
      if (auto *defOp = operand.getDefiningOp())
        primaryOpName = defOp->getName().getStringRef().str();
    }
  }

  return bodyLatency;
}

} // namespace

unsigned computeFUIntrinsicLatency(loom::fabric::FunctionUnitOp fuOp) {
  bool hasDataflow = false;
  std::string primaryOp;
  return computeBodyIntrinsicLatency(fuOp, hasDataflow, primaryOp);
}

bool validateFUTimingConstraints(loom::fabric::FunctionUnitOp fuOp) {
  std::string fuName = fuOp.getSymName().str();

  // Gather declared latency and interval from the FU op attributes.
  int64_t declaredLatency = 0;
  if (auto lat = fuOp.getLatency())
    declaredLatency = *lat;

  int64_t declaredInterval = 1;
  if (auto intv = fuOp.getInterval())
    declaredInterval = *intv;

  // Analyze the body to determine intrinsic latency and timing class.
  bool hasDataflowOp = false;
  std::string primaryOpName;
  unsigned bodyIntrinsic =
      computeBodyIntrinsicLatency(fuOp, hasDataflowOp, primaryOpName);

  bool valid = true;

  // --- Latency validation ---

  if (hasDataflowOp) {
    // Dataflow FU: latency must be -1.
    if (declaredLatency != -1) {
      llvm::errs()
          << "gen-sv error: latency-violation: function_unit '" << fuName
          << "' contains dataflow op '" << primaryOpName
          << "' but declares latency=" << declaredLatency
          << "; dataflow FUs require latency=-1\n";
      valid = false;
    }
  } else {
    // Non-dataflow FU.
    if (declaredLatency == -1) {
      // latency=-1 is only legal for dataflow FUs.
      llvm::errs()
          << "gen-sv error: latency-violation: function_unit '" << fuName
          << "' declares latency=-1 but contains no dataflow op; "
             "only dataflow FUs may use latency=-1\n";
      valid = false;
    } else if (declaredLatency >= 0 &&
               static_cast<unsigned>(declaredLatency) < bodyIntrinsic) {
      // Declared latency is below the critical-path intrinsic minimum.
      llvm::errs()
          << "gen-sv error: latency-violation: function_unit '" << fuName
          << "' has latency=" << declaredLatency
          << " but critical-path intrinsic latency is " << bodyIntrinsic
          << " (bottleneck: '" << primaryOpName << "')\n";
      valid = false;
    }
  }

  // --- Interval validation ---

  if (hasDataflowOp) {
    // Dataflow FU: interval must be -1.
    if (declaredInterval != -1) {
      llvm::errs()
          << "gen-sv error: interval-violation: function_unit '" << fuName
          << "' contains dataflow op '" << primaryOpName
          << "' but declares interval=" << declaredInterval
          << "; dataflow FUs require interval=-1\n";
      valid = false;
    }
  } else {
    // Non-dataflow FU.
    if (declaredInterval == -1) {
      // interval=-1 is only legal for dataflow FUs.
      llvm::errs()
          << "gen-sv error: interval-violation: function_unit '" << fuName
          << "' declares interval=-1 but contains no dataflow op; "
             "only dataflow FUs may use interval=-1\n";
      valid = false;
    } else if (declaredInterval < 1) {
      // Non-dataflow interval must be >= 1.
      llvm::errs()
          << "gen-sv error: interval-violation: function_unit '" << fuName
          << "' declares interval=" << declaredInterval
          << "; non-dataflow FUs require interval >= 1\n";
      valid = false;
    }
  }

  return valid;
}

} // namespace svgen
} // namespace loom
