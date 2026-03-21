#include "SVGenInternal.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace fcc {
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
      opName == "llvm.bitreverse" ||
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

/// Scan the FU body to find the primary (highest-latency) non-terminator op.
/// Also detects whether any dataflow op is present.
///
/// Sets `hasDataflowOp` to true if any dataflow op is found.
/// Sets `primaryOpName` to the op with highest intrinsic latency.
/// Returns the body's critical-path intrinsic latency (max across all ops).
///
/// Note: This is a conservative single-level max across individual ops,
/// not a full DAG critical-path analysis. That is sufficient for the
/// current single-fire FU bodies and exclusive dataflow FU bodies.
static unsigned
computeBodyIntrinsicLatency(fcc::fabric::FunctionUnitOp fuOp,
                            bool &hasDataflowOp,
                            std::string &primaryOpName) {
  hasDataflowOp = false;
  primaryOpName.clear();
  unsigned maxLatency = 0;

  auto &bodyBlock = fuOp.getBody().front();
  for (auto &op : bodyBlock.getOperations()) {
    llvm::StringRef opName = op.getName().getStringRef();

    if (isNonComputeOp(opName))
      continue;

    if (isDataflowOp(opName)) {
      hasDataflowOp = true;
      primaryOpName = opName.str();
      continue;
    }

    int intrinsic = getOpIntrinsicLatency(opName);
    if (intrinsic < 0)
      continue;

    unsigned uIntrinsic = static_cast<unsigned>(intrinsic);
    if (uIntrinsic > maxLatency) {
      maxLatency = uIntrinsic;
      primaryOpName = opName.str();
    }
  }

  return maxLatency;
}

} // namespace

unsigned computeFUIntrinsicLatency(fcc::fabric::FunctionUnitOp fuOp) {
  bool hasDataflow = false;
  std::string primaryOp;
  return computeBodyIntrinsicLatency(fuOp, hasDataflow, primaryOp);
}

bool validateFUTimingConstraints(fcc::fabric::FunctionUnitOp fuOp) {
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
      // Declared latency is below the intrinsic minimum.
      llvm::errs()
          << "gen-sv error: latency-violation: function_unit '" << fuName
          << "' has latency=" << declaredLatency << " but operation '"
          << primaryOpName << "' requires minimum latency "
          << bodyIntrinsic << "\n";
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
} // namespace fcc
