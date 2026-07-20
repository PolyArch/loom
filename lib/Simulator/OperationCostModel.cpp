#include "Simulator/OperationCostModel.h"

#include <optional>
#include <system_error>

using namespace loom::sim;

namespace {

struct OperationCostEntry {
  const char *name;
  std::uint64_t baseScore;
  std::uint64_t repeatScore;
};

constexpr OperationCostEntry kOperationCosts[] = {
    {"arith.constant", 1, 1},
    {"arith.addf", 2, 2},
    {"arith.subf", 2, 2},
    {"arith.mulf", 3, 3},
    {"arith.divf", 12, 12},
    {"arith.addi", 1, 1},
    {"arith.subi", 1, 1},
    {"arith.muli", 3, 3},
    {"arith.andi", 1, 1},
    {"arith.ori", 1, 1},
    {"arith.xori", 1, 1},
    {"arith.shli", 1, 1},
    {"arith.shrsi", 1, 1},
    {"arith.shrui", 1, 1},
    {"arith.divsi", 8, 8},
    {"arith.divui", 8, 8},
    {"arith.remsi", 8, 8},
    {"arith.remui", 8, 8},
    {"arith.cmpi", 1, 1},
    {"arith.cmpf", 2, 2},
    {"arith.select", 1, 1},
    {"arith.index_cast", 1, 1},
    {"arith.index_castui", 1, 1},
    {"arith.extsi", 1, 1},
    {"arith.extui", 1, 1},
    {"arith.trunci", 1, 1},
    {"arith.sitofp", 3, 3},
    {"arith.uitofp", 3, 3},
    {"arith.fptosi", 3, 3},
    {"arith.fptoui", 3, 3},
    {"llvm.trunc", 1, 1},
    {"llvm.sext", 1, 1},
    {"llvm.zext", 1, 1},
    {"llvm.sitofp", 3, 3},
    {"llvm.uitofp", 3, 3},
    {"llvm.fptosi", 3, 3},
    {"llvm.fptoui", 3, 3},
    {"llvm.fneg", 1, 1},
    {"llvm.load", 4, 4},
    {"llvm.store", 4, 4},
    {"llvm.select", 1, 1},
    {"llvm.icmp", 1, 1},
    {"llvm.getelementptr", 1, 1},
    {"llvm.mlir.addressof", 1, 1},
    {"llvm.mlir.zero", 1, 1},
    {"ub.poison", 1, 1},
    {"llvm.intr.memcpy", 8, 8},
    {"llvm.intr.fshl", 1, 1},
    {"llvm.intr.bswap", 1, 1},
    {"llvm.intr.umin", 1, 1},
    {"llvm.intr.umax", 1, 1},
    {"llvm.intr.usub.sat", 1, 1},
    {"llvm.intr.smin", 1, 1},
    {"llvm.intr.smax", 1, 1},
    {"llvm.intr.ctlz", 1, 1},
    {"llvm.intr.fmuladd", 8, 8},
    {"llvm.intr.abs", 1, 1},
    {"llvm.intr.fabs", 1, 1},
    {"math.absf", 1, 1},
    {"math.absi", 1, 1},
    {"math.sin", 16, 16},
    {"math.cos", 16, 16},
    {"math.tan", 16, 16},
    {"math.sinh", 16, 16},
    {"math.cosh", 16, 16},
    {"math.tanh", 16, 16},
    {"math.exp", 12, 12},
    {"math.exp2", 12, 12},
    {"math.expm1", 12, 12},
    {"math.log", 12, 12},
    {"math.log2", 12, 12},
    {"math.log10", 12, 12},
    {"math.log1p", 12, 12},
    {"math.floor", 2, 2},
    {"math.ceil", 2, 2},
    {"math.round", 2, 2},
    {"math.trunc", 2, 2},
    {"math.roundeven", 2, 2},
    {"math.sqrt", 8, 8},
    {"math.rsqrt", 8, 8},
    {"math.erf", 16, 16},
    {"dataflow.stream", 1, 1},
    {"dataflow.carry", 1, 1},
    {"dataflow.invariant", 1, 1},
    {"dataflow.constant", 1, 1},
    {"dataflow.sync", 1, 1},
    {"dataflow.load", 4, 4},
    {"dataflow.store", 4, 4},
    {"dataflow.mux", 2, 2},
    {"dataflow.demux", 2, 2},
    {"dataflow.parallelize", 1, 1},
    {"dataflow.pack", 1, 1},
    {"dataflow.unpack", 1, 1},
    {"dataflow.serialize", 1, 1},
    {"dataflow.gate", 1, 1},
};

std::optional<OperationCost> lookupOperationCost(llvm::StringRef opName) {
  for (const OperationCostEntry &entry : kOperationCosts) {
    if (opName == entry.name)
      return OperationCost{entry.baseScore, entry.repeatScore};
  }
  return std::nullopt;
}

} // namespace

bool loom::sim::hasOperationCost(llvm::StringRef opName) {
  return lookupOperationCost(opName).has_value();
}

llvm::Expected<OperationCost>
loom::sim::estimateOperationCost(llvm::StringRef opName) {
  std::optional<OperationCost> cost = lookupOperationCost(opName);
  if (cost)
    return *cost;
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s has no operation cost model entry",
                                 opName.str().c_str());
}
