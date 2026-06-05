#include "Simulator/OperationSemantics.h"

#include <limits>
#include <optional>
#include <system_error>
#include <utility>

using namespace loom::sim;

namespace {

struct OperationCostEntry {
  const char *name;
  std::uint64_t latencyCycles;
  std::uint64_t reciprocalThroughput;
  bool isPrimitive;
  bool isMapped;
};

constexpr OperationCostEntry kOperationCosts[] = {
    {"arith.constant", 1, 1, false, false},
    {"arith.addf", 2, 2, true, true},
    {"arith.subf", 2, 2, true, true},
    {"arith.mulf", 3, 3, true, true},
    {"arith.addi", 1, 1, true, true},
    {"arith.muli", 3, 3, true, true},
    {"arith.andi", 1, 1, true, true},
    {"arith.ori", 1, 1, true, true},
    {"arith.shli", 1, 1, true, true},
    {"arith.shrui", 1, 1, true, true},
    {"arith.index_cast", 1, 1, true, true},
    {"llvm.zext", 1, 1, true, true},
    {"llvm.intr.fmuladd", 4, 4, true, true},
    {"llvm.intr.abs", 1, 1, true, true},
    {"dataflow.stream", 1, 1, false, true},
    {"dataflow.carry", 1, 1, false, true},
    {"dataflow.invariant", 1, 1, false, true},
    {"dataflow.constant", 1, 1, false, true},
    {"dataflow.sync", 1, 1, false, true},
    {"dataflow.load", 4, 4, false, true},
    {"dataflow.store", 4, 4, false, true},
    {"dataflow.mux", 1, 1, false, true},
    {"dataflow.demux", 1, 1, false, true},
    {"dataflow.gate", 1, 1, false, true},
};

const OperationCostEntry *lookupOperationCostEntry(llvm::StringRef opName) {
  for (const OperationCostEntry &entry : kOperationCosts) {
    if (opName == entry.name)
      return &entry;
  }
  return nullptr;
}

std::optional<OperationCost> lookupOperationCost(llvm::StringRef opName) {
  const OperationCostEntry *entry = lookupOperationCostEntry(opName);
  if (!entry)
    return std::nullopt;
  return OperationCost{entry->latencyCycles, entry->reciprocalThroughput};
}

llvm::Error requireArity(llvm::StringRef opName,
                         llvm::ArrayRef<PrimitiveValue> operands,
                         unsigned expected) {
  if (operands.size() == expected)
    return llvm::Error::success();
  return llvm::createStringError(
      std::errc::invalid_argument, "%s expects %u operands, got %u",
      opName.str().c_str(), expected, static_cast<unsigned>(operands.size()));
}

std::int64_t asInteger(const PrimitiveValue &value) {
  if (value.kind == PrimitiveValueKind::Bool)
    return value.boolValue ? 1 : 0;
  return value.intValue;
}

llvm::Expected<unsigned> asShiftAmount(llvm::StringRef opName,
                                       const PrimitiveValue &value) {
  std::int64_t raw = asInteger(value);
  if (raw < 0 || raw >= 64)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s shift amount must be in [0, 63], got %lld",
                                   opName.str().c_str(),
                                   static_cast<long long>(raw));
  return static_cast<unsigned>(raw);
}

double asFloat(const PrimitiveValue &value) {
  if (value.kind == PrimitiveValueKind::Float)
    return value.floatValue;
  return static_cast<double>(asInteger(value));
}

} // namespace

PrimitiveValue PrimitiveValue::none() { return PrimitiveValue{}; }

PrimitiveValue PrimitiveValue::integer(std::int64_t value) {
  PrimitiveValue result;
  result.kind = PrimitiveValueKind::Integer;
  result.intValue = value;
  return result;
}

PrimitiveValue PrimitiveValue::floating(double value) {
  PrimitiveValue result;
  result.kind = PrimitiveValueKind::Float;
  result.floatValue = value;
  return result;
}

PrimitiveValue PrimitiveValue::boolean(bool value) {
  PrimitiveValue result;
  result.kind = PrimitiveValueKind::Bool;
  result.boolValue = value;
  return result;
}

bool loom::sim::isSupportedPrimitiveOperation(llvm::StringRef opName) {
  const OperationCostEntry *entry = lookupOperationCostEntry(opName);
  return entry && entry->isPrimitive;
}

bool loom::sim::isSupportedMappedOperation(llvm::StringRef opName) {
  const OperationCostEntry *entry = lookupOperationCostEntry(opName);
  return entry && entry->isMapped;
}

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

llvm::Expected<PrimitiveValue>
loom::sim::evaluatePrimitiveOperation(llvm::StringRef opName,
                                      llvm::ArrayRef<PrimitiveValue> operands) {
  if (opName == "arith.addf") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::floating(asFloat(operands[0]) +
                                    asFloat(operands[1]));
  }
  if (opName == "arith.subf") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::floating(asFloat(operands[0]) -
                                    asFloat(operands[1]));
  }
  if (opName == "arith.mulf") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::floating(asFloat(operands[0]) *
                                    asFloat(operands[1]));
  }
  if (opName == "arith.addi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::integer(asInteger(operands[0]) +
                                   asInteger(operands[1]));
  }
  if (opName == "arith.muli") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::integer(asInteger(operands[0]) *
                                   asInteger(operands[1]));
  }
  if (opName == "arith.andi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::integer(asInteger(operands[0]) &
                                   asInteger(operands[1]));
  }
  if (opName == "arith.ori") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::integer(asInteger(operands[0]) |
                                   asInteger(operands[1]));
  }
  if (opName == "arith.shli") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    auto amountOrErr = asShiftAmount(opName, operands[1]);
    if (!amountOrErr)
      return amountOrErr.takeError();
    return PrimitiveValue::integer(static_cast<std::int64_t>(
        static_cast<std::uint64_t>(asInteger(operands[0])) << *amountOrErr));
  }
  if (opName == "arith.shrui") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    auto amountOrErr = asShiftAmount(opName, operands[1]);
    if (!amountOrErr)
      return amountOrErr.takeError();
    return PrimitiveValue::integer(static_cast<std::int64_t>(
        static_cast<std::uint64_t>(asInteger(operands[0])) >> *amountOrErr));
  }
  if (opName == "arith.index_cast") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return PrimitiveValue::integer(asInteger(operands[0]));
  }
  if (opName == "llvm.zext") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    std::int64_t value = asInteger(operands[0]);
    if (value < 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s requires a non-negative integer token", opName.str().c_str());
    return PrimitiveValue::integer(value);
  }
  if (opName == "llvm.intr.abs") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    std::int64_t value = asInteger(operands[0]);
    if (value == std::numeric_limits<std::int64_t>::min())
      return llvm::createStringError(
          std::errc::result_out_of_range,
          "%s cannot represent absolute value of int64 minimum",
          opName.str().c_str());
    return PrimitiveValue::integer(value < 0 ? -value : value);
  }
  if (opName == "llvm.intr.fmuladd") {
    if (llvm::Error arity = requireArity(opName, operands, 3))
      return std::move(arity);
    return PrimitiveValue::floating(
        asFloat(operands[0]) * asFloat(operands[1]) + asFloat(operands[2]));
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s is not supported by operation semantics",
                                 opName.str().c_str());
}
