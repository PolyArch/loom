#include "Simulator/OperationSemantics.h"

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
         opName == "arith.muli" || opName == "arith.index_cast" ||
         opName == "llvm.intr.fmuladd";
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
  if (opName == "arith.index_cast") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return PrimitiveValue::integer(asInteger(operands[0]));
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
