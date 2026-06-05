#include "Simulator/OperationSemantics.h"

#include <limits>
#include <system_error>
#include <utility>

using namespace loom::sim;

namespace {

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
  return opName == "arith.addf" || opName == "arith.subf" ||
         opName == "arith.mulf" || opName == "arith.addi" ||
         opName == "arith.muli" || opName == "arith.andi" ||
         opName == "arith.ori" || opName == "arith.shli" ||
         opName == "arith.shrui" || opName == "arith.index_cast" ||
         opName == "llvm.intr.fmuladd" || opName == "llvm.intr.abs";
}

bool loom::sim::isSupportedMappedOperation(llvm::StringRef opName) {
  return isSupportedPrimitiveOperation(opName) || opName == "dataflow.stream" ||
         opName == "dataflow.carry" || opName == "dataflow.invariant" ||
         opName == "dataflow.constant" || opName == "dataflow.sync" ||
         opName == "dataflow.load" || opName == "dataflow.store" ||
         opName == "dataflow.mux" || opName == "dataflow.demux" ||
         opName == "dataflow.gate";
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
