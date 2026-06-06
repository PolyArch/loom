#include "Simulator/OperationSemantics.h"

#include <cmath>
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
    {"arith.subi", 1, 1, true, true},
    {"arith.muli", 3, 3, true, true},
    {"arith.andi", 1, 1, true, true},
    {"arith.ori", 1, 1, true, true},
    {"arith.xori", 1, 1, true, true},
    {"arith.shli", 1, 1, true, true},
    {"arith.shrui", 1, 1, true, true},
    {"arith.remui", 8, 8, true, true},
    {"arith.cmpi", 1, 1, true, true},
    {"arith.cmpf", 2, 2, true, true},
    {"arith.select", 1, 1, true, true},
    {"arith.index_cast", 1, 1, true, true},
    {"arith.sitofp", 3, 3, true, true},
    {"arith.uitofp", 3, 3, true, true},
    {"arith.fptosi", 3, 3, true, true},
    {"arith.fptoui", 3, 3, true, true},
    {"llvm.trunc", 1, 1, true, true},
    {"llvm.zext", 1, 1, true, true},
    {"llvm.sitofp", 3, 3, true, true},
    {"llvm.uitofp", 3, 3, true, true},
    {"llvm.fptosi", 3, 3, true, true},
    {"llvm.fptoui", 3, 3, true, true},
    {"llvm.select", 1, 1, true, true},
    {"llvm.intr.fshl", 1, 1, true, true},
    {"llvm.intr.bswap", 1, 1, true, true},
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

unsigned normalizeBitWidth(unsigned bitWidth) {
  if (bitWidth == 0 || bitWidth > 64)
    return 64;
  return bitWidth;
}

std::uint64_t maskForBitWidth(unsigned bitWidth) {
  bitWidth = normalizeBitWidth(bitWidth);
  if (bitWidth == 64)
    return ~std::uint64_t{0};
  return (std::uint64_t{1} << bitWidth) - 1;
}

std::uint64_t toUnsignedBits(const PrimitiveValue &value, unsigned bitWidth) {
  return static_cast<std::uint64_t>(asInteger(value)) &
         maskForBitWidth(bitWidth);
}

std::int64_t fromUnsignedBits(std::uint64_t bits, unsigned bitWidth) {
  bitWidth = normalizeBitWidth(bitWidth);
  bits &= maskForBitWidth(bitWidth);
  if (bitWidth == 64)
    return static_cast<std::int64_t>(bits);
  const std::uint64_t signBit = std::uint64_t{1} << (bitWidth - 1);
  if ((bits & signBit) == 0)
    return static_cast<std::int64_t>(bits);
  return static_cast<std::int64_t>(bits | ~maskForBitWidth(bitWidth));
}

PrimitiveValue integerFromBits(std::uint64_t bits, unsigned bitWidth) {
  return PrimitiveValue::integer(fromUnsignedBits(bits, bitWidth));
}

PrimitiveValue integerFromSigned(std::int64_t value, unsigned bitWidth) {
  return integerFromBits(static_cast<std::uint64_t>(value), bitWidth);
}

llvm::Expected<PrimitiveValue>
byteSwapInteger(llvm::StringRef opName, const PrimitiveValue &value,
                unsigned bitWidth) {
  bitWidth = normalizeBitWidth(bitWidth);
  if (bitWidth % 8 != 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "%s result bit width must be a multiple of 8, got %u",
        opName.str().c_str(), bitWidth);
  const std::uint64_t input = toUnsignedBits(value, bitWidth);
  std::uint64_t output = 0;
  const unsigned bytes = bitWidth / 8;
  for (unsigned i = 0; i < bytes; ++i) {
    const std::uint64_t byte = (input >> (i * 8)) & 0xffu;
    output |= byte << ((bytes - 1 - i) * 8);
  }
  return integerFromBits(output, bitWidth);
}

bool compareInteger(llvm::StringRef predicate, const PrimitiveValue &lhs,
                    const PrimitiveValue &rhs, unsigned bitWidth) {
  if (predicate == "eq")
    return asInteger(lhs) == asInteger(rhs);
  if (predicate == "ne")
    return asInteger(lhs) != asInteger(rhs);
  if (predicate == "slt")
    return asInteger(lhs) < asInteger(rhs);
  if (predicate == "sle")
    return asInteger(lhs) <= asInteger(rhs);
  if (predicate == "sgt")
    return asInteger(lhs) > asInteger(rhs);
  if (predicate == "sge")
    return asInteger(lhs) >= asInteger(rhs);
  const std::uint64_t lhsBits = toUnsignedBits(lhs, bitWidth);
  const std::uint64_t rhsBits = toUnsignedBits(rhs, bitWidth);
  if (predicate == "ult")
    return lhsBits < rhsBits;
  if (predicate == "ule")
    return lhsBits <= rhsBits;
  if (predicate == "ugt")
    return lhsBits > rhsBits;
  if (predicate == "uge")
    return lhsBits >= rhsBits;
  return false;
}

bool compareFloat(llvm::StringRef predicate, const PrimitiveValue &lhs,
                  const PrimitiveValue &rhs) {
  const double lhsValue = asFloat(lhs);
  const double rhsValue = asFloat(rhs);
  const bool lhsNan = std::isnan(lhsValue);
  const bool rhsNan = std::isnan(rhsValue);
  const bool ordered = !lhsNan && !rhsNan;
  if (predicate == "false")
    return false;
  if (predicate == "true")
    return true;
  if (predicate == "ord")
    return ordered;
  if (predicate == "uno")
    return !ordered;
  if (predicate == "oeq")
    return ordered && lhsValue == rhsValue;
  if (predicate == "ogt")
    return ordered && lhsValue > rhsValue;
  if (predicate == "oge")
    return ordered && lhsValue >= rhsValue;
  if (predicate == "olt")
    return ordered && lhsValue < rhsValue;
  if (predicate == "ole")
    return ordered && lhsValue <= rhsValue;
  if (predicate == "one")
    return ordered && lhsValue != rhsValue;
  if (predicate == "ueq")
    return !ordered || lhsValue == rhsValue;
  if (predicate == "ugt")
    return !ordered || lhsValue > rhsValue;
  if (predicate == "uge")
    return !ordered || lhsValue >= rhsValue;
  if (predicate == "ult")
    return !ordered || lhsValue < rhsValue;
  if (predicate == "ule")
    return !ordered || lhsValue <= rhsValue;
  if (predicate == "une")
    return !ordered || lhsValue != rhsValue;
  return false;
}

llvm::Error requirePredicate(llvm::StringRef opName,
                             llvm::StringRef predicate) {
  if (!predicate.empty())
    return llvm::Error::success();
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s requires a predicate descriptor",
                                 opName.str().c_str());
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
  return evaluatePrimitiveOperation(PrimitiveOperationDescriptor{opName, "", 0},
                                    operands);
}

llvm::Expected<PrimitiveValue> loom::sim::evaluatePrimitiveOperation(
    const PrimitiveOperationDescriptor &descriptor,
    llvm::ArrayRef<PrimitiveValue> operands) {
  llvm::StringRef opName = descriptor.name;
  const unsigned bitWidth = normalizeBitWidth(descriptor.resultBitWidth);
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
    return integerFromSigned(asInteger(operands[0]) + asInteger(operands[1]),
                             bitWidth);
  }
  if (opName == "arith.subi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return integerFromSigned(asInteger(operands[0]) - asInteger(operands[1]),
                             bitWidth);
  }
  if (opName == "arith.muli") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return integerFromSigned(asInteger(operands[0]) * asInteger(operands[1]),
                             bitWidth);
  }
  if (opName == "arith.andi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return integerFromBits(toUnsignedBits(operands[0], bitWidth) &
                               toUnsignedBits(operands[1], bitWidth),
                           bitWidth);
  }
  if (opName == "arith.ori") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return integerFromBits(toUnsignedBits(operands[0], bitWidth) |
                               toUnsignedBits(operands[1], bitWidth),
                           bitWidth);
  }
  if (opName == "arith.xori") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return integerFromBits(toUnsignedBits(operands[0], bitWidth) ^
                               toUnsignedBits(operands[1], bitWidth),
                           bitWidth);
  }
  if (opName == "arith.shli") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    auto amountOrErr = asShiftAmount(opName, operands[1]);
    if (!amountOrErr)
      return amountOrErr.takeError();
    return integerFromBits(toUnsignedBits(operands[0], bitWidth)
                               << *amountOrErr,
                           bitWidth);
  }
  if (opName == "arith.shrui") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    auto amountOrErr = asShiftAmount(opName, operands[1]);
    if (!amountOrErr)
      return amountOrErr.takeError();
    return integerFromBits(toUnsignedBits(operands[0], bitWidth) >>
                               *amountOrErr,
                           bitWidth);
  }
  if (opName == "arith.remui") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    const std::uint64_t divisor = toUnsignedBits(operands[1], bitWidth);
    if (divisor == 0)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "%s divisor must be non-zero",
                                     opName.str().c_str());
    return integerFromBits(toUnsignedBits(operands[0], bitWidth) % divisor,
                           bitWidth);
  }
  if (opName == "arith.cmpi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    if (llvm::Error predicate =
            requirePredicate(opName, descriptor.predicate))
      return std::move(predicate);
    return PrimitiveValue::boolean(
        compareInteger(descriptor.predicate, operands[0], operands[1],
                       bitWidth));
  }
  if (opName == "arith.cmpf") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    if (llvm::Error predicate =
            requirePredicate(opName, descriptor.predicate))
      return std::move(predicate);
    return PrimitiveValue::boolean(
        compareFloat(descriptor.predicate, operands[0], operands[1]));
  }
  if (opName == "arith.select" || opName == "llvm.select") {
    if (llvm::Error arity = requireArity(opName, operands, 3))
      return std::move(arity);
    return operands[0].boolValue ? operands[1] : operands[2];
  }
  if (opName == "arith.index_cast") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return integerFromSigned(asInteger(operands[0]), bitWidth);
  }
  if (opName == "llvm.trunc") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return integerFromBits(toUnsignedBits(operands[0], bitWidth), bitWidth);
  }
  if (opName == "llvm.zext") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    const unsigned sourceBitWidth =
        normalizeBitWidth(descriptor.operandBitWidth);
    if (sourceBitWidth > bitWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s source bit width must not exceed result bit width",
          opName.str().c_str());
    return integerFromBits(toUnsignedBits(operands[0], sourceBitWidth),
                           bitWidth);
  }
  if (opName == "llvm.sitofp" || opName == "arith.sitofp") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return PrimitiveValue::floating(static_cast<double>(asInteger(operands[0])));
  }
  if (opName == "llvm.uitofp" || opName == "arith.uitofp") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    const unsigned sourceBitWidth =
        normalizeBitWidth(descriptor.operandBitWidth);
    return PrimitiveValue::floating(
        static_cast<double>(toUnsignedBits(operands[0], sourceBitWidth)));
  }
  if (opName == "llvm.fptosi" || opName == "arith.fptosi") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return integerFromSigned(static_cast<std::int64_t>(asFloat(operands[0])),
                             bitWidth);
  }
  if (opName == "llvm.fptoui" || opName == "arith.fptoui") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    const double value = asFloat(operands[0]);
    if (value < 0.0)
      return llvm::createStringError(
          std::errc::result_out_of_range,
          "%s cannot convert negative value to unsigned integer",
          opName.str().c_str());
    return integerFromBits(static_cast<std::uint64_t>(value), bitWidth);
  }
  if (opName == "llvm.intr.fshl") {
    if (llvm::Error arity = requireArity(opName, operands, 3))
      return std::move(arity);
    const unsigned amount =
        static_cast<unsigned>(toUnsignedBits(operands[2], bitWidth) % bitWidth);
    const std::uint64_t lhs = toUnsignedBits(operands[0], bitWidth);
    const std::uint64_t rhs = toUnsignedBits(operands[1], bitWidth);
    if (amount == 0)
      return integerFromBits(lhs, bitWidth);
    return integerFromBits((lhs << amount) | (rhs >> (bitWidth - amount)),
                           bitWidth);
  }
  if (opName == "llvm.intr.bswap") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return byteSwapInteger(opName, operands[0], bitWidth);
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
