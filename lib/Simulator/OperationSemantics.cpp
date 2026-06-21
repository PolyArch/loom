#include "Simulator/OperationSemantics.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <algorithm>
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
    {"arith.divf", 12, 12, true, true},
    {"arith.addi", 1, 1, true, true},
    {"arith.subi", 1, 1, true, true},
    {"arith.muli", 3, 3, true, true},
    {"arith.andi", 1, 1, true, true},
    {"arith.ori", 1, 1, true, true},
    {"arith.xori", 1, 1, true, true},
    {"arith.shli", 1, 1, true, true},
    {"arith.shrsi", 1, 1, true, true},
    {"arith.shrui", 1, 1, true, true},
    {"arith.divsi", 8, 8, true, true},
    {"arith.remsi", 8, 8, true, true},
    {"arith.remui", 8, 8, true, true},
    {"arith.cmpi", 1, 1, true, true},
    {"arith.cmpf", 2, 2, true, true},
    {"arith.select", 1, 1, true, true},
    {"arith.index_cast", 1, 1, true, true},
    {"arith.extsi", 1, 1, true, true},
    {"arith.trunci", 1, 1, true, true},
    {"arith.sitofp", 3, 3, true, true},
    {"arith.uitofp", 3, 3, true, true},
    {"arith.fptosi", 3, 3, true, true},
    {"arith.fptoui", 3, 3, true, true},
    {"llvm.trunc", 1, 1, true, true},
    {"llvm.sext", 1, 1, true, true},
    {"llvm.zext", 1, 1, true, true},
    {"llvm.sitofp", 3, 3, true, true},
    {"llvm.uitofp", 3, 3, true, true},
    {"llvm.fptosi", 3, 3, true, true},
    {"llvm.fptoui", 3, 3, true, true},
    {"llvm.load", 4, 4, false, true},
    {"llvm.select", 1, 1, true, true},
    {"llvm.getelementptr", 1, 1, false, false},
    {"llvm.intr.memcpy", 8, 8, false, false},
    {"llvm.intr.fshl", 1, 1, true, true},
    {"llvm.intr.bswap", 1, 1, true, true},
    {"llvm.intr.fmuladd", 8, 8, true, true},
    {"llvm.intr.abs", 1, 1, true, true},
    {"llvm.intr.fabs", 1, 1, true, true},
    {"llvm.arm.qadd16", 1, 1, true, true},
    {"llvm.arm.qsub8", 1, 1, true, true},
    {"llvm.arm.qsub16", 1, 1, true, true},
    {"math.absf", 1, 1, true, true},
    {"math.absi", 1, 1, true, true},
    {"math.sin", 16, 16, true, true},
    {"math.cos", 16, 16, true, true},
    {"math.tan", 16, 16, true, true},
    {"math.sinh", 16, 16, true, true},
    {"math.cosh", 16, 16, true, true},
    {"math.tanh", 16, 16, true, true},
    {"math.exp", 12, 12, true, true},
    {"math.exp2", 12, 12, true, true},
    {"math.expm1", 12, 12, true, true},
    {"math.log", 12, 12, true, true},
    {"math.log2", 12, 12, true, true},
    {"math.log10", 12, 12, true, true},
    {"math.log1p", 12, 12, true, true},
    {"math.floor", 2, 2, true, true},
    {"math.ceil", 2, 2, true, true},
    {"math.round", 2, 2, true, true},
    {"math.trunc", 2, 2, true, true},
    {"math.roundeven", 2, 2, true, true},
    {"math.sqrt", 8, 8, true, true},
    {"math.rsqrt", 8, 8, true, true},
    {"math.erf", 16, 16, true, true},
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

bool asBoolean(const PrimitiveValue &value) {
  if (value.kind == PrimitiveValueKind::Bool)
    return value.boolValue;
  return asInteger(value) != 0;
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

llvm::Expected<unsigned> checkedShiftAmount(llvm::StringRef opName,
                                            const PrimitiveValue &value,
                                            unsigned bitWidth) {
  auto amountOrErr = asShiftAmount(opName, value);
  if (!amountOrErr)
    return amountOrErr.takeError();
  bitWidth = normalizeBitWidth(bitWidth);
  if (*amountOrErr >= bitWidth)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "%s shift amount must be less than bit width %u, got %u",
        opName.str().c_str(), bitWidth, *amountOrErr);
  return *amountOrErr;
}

std::uint64_t maskForBitWidth(unsigned bitWidth) {
  bitWidth = normalizeBitWidth(bitWidth);
  if (bitWidth == 64)
    return ~std::uint64_t{0};
  return (std::uint64_t{1} << bitWidth) - 1;
}

std::uint64_t lowBitsMask(unsigned bitCount) {
  if (bitCount == 0)
    return 0;
  if (bitCount >= 64)
    return ~std::uint64_t{0};
  return (std::uint64_t{1} << bitCount) - 1;
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

std::int64_t signedMinForBitWidth(unsigned bitWidth) {
  bitWidth = normalizeBitWidth(bitWidth);
  if (bitWidth == 64)
    return std::numeric_limits<std::int64_t>::min();
  return -(std::int64_t{1} << (bitWidth - 1));
}

std::int64_t saturateSigned(std::int64_t value, unsigned bitWidth) {
  bitWidth = normalizeBitWidth(bitWidth);
  const std::int64_t min = signedMinForBitWidth(bitWidth);
  const std::int64_t max =
      bitWidth == 64 ? std::numeric_limits<std::int64_t>::max()
                     : ((std::int64_t{1} << (bitWidth - 1)) - 1);
  return std::min(std::max(value, min), max);
}

PrimitiveValue
packedSaturatingBinary(const PrimitiveValue &lhs, const PrimitiveValue &rhs,
                       unsigned bitWidth, unsigned laneWidth,
                       llvm::function_ref<std::int64_t(std::int64_t,
                                                       std::int64_t)>
                           combine) {
  std::uint64_t packed = 0;
  const std::uint64_t lhsBits = toUnsignedBits(lhs, bitWidth);
  const std::uint64_t rhsBits = toUnsignedBits(rhs, bitWidth);
  const std::uint64_t laneMask = maskForBitWidth(laneWidth);
  for (unsigned offset = 0; offset < bitWidth; offset += laneWidth) {
    const std::int64_t lhsLane =
        fromUnsignedBits((lhsBits >> offset) & laneMask, laneWidth);
    const std::int64_t rhsLane =
        fromUnsignedBits((rhsBits >> offset) & laneMask, laneWidth);
    const std::int64_t saturated =
        saturateSigned(combine(lhsLane, rhsLane), laneWidth);
    packed |= (static_cast<std::uint64_t>(saturated) & laneMask) << offset;
  }
  return integerFromBits(packed, bitWidth);
}

std::uint64_t arithmeticRightShiftBits(const PrimitiveValue &value,
                                       unsigned bitWidth, unsigned amount) {
  bitWidth = normalizeBitWidth(bitWidth);
  const std::uint64_t bits = toUnsignedBits(value, bitWidth);
  const std::uint64_t signBit = std::uint64_t{1} << (bitWidth - 1);
  if (amount == 0)
    return bits;
  if (amount >= bitWidth)
    return (bits & signBit) ? maskForBitWidth(bitWidth) : 0;
  std::uint64_t shifted = bits >> amount;
  if ((bits & signBit) == 0)
    return shifted;
  const unsigned keptBits = bitWidth - amount;
  const std::uint64_t lowMask =
      keptBits == 64 ? ~std::uint64_t{0}
                     : ((std::uint64_t{1} << keptBits) - 1);
  return shifted | (maskForBitWidth(bitWidth) & ~lowMask);
}

bool exactRightShiftWouldDiscardBits(const PrimitiveValue &value,
                                     unsigned bitWidth, unsigned amount) {
  return (toUnsignedBits(value, bitWidth) & lowBitsMask(amount)) != 0;
}

PrimitiveValue roundEven(double value) {
  const double lower = std::floor(value);
  const double fraction = value - lower;
  double rounded = lower;
  if (fraction < 0.5)
    rounded = lower;
  else if (fraction > 0.5)
    rounded = lower + 1.0;
  else
    rounded = std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
  if (rounded == 0.0)
    rounded = std::copysign(0.0, value);
  return PrimitiveValue::floating(rounded);
}

llvm::Expected<PrimitiveValue>
evaluateMathUnary(llvm::StringRef opName,
                  llvm::ArrayRef<PrimitiveValue> operands) {
  if (llvm::Error arity = requireArity(opName, operands, 1))
    return std::move(arity);
  const double value = asFloat(operands[0]);
  if (opName == "math.absf" || opName == "llvm.intr.fabs")
    return PrimitiveValue::floating(std::fabs(value));
  if (opName == "math.sin")
    return PrimitiveValue::floating(std::sin(value));
  if (opName == "math.cos")
    return PrimitiveValue::floating(std::cos(value));
  if (opName == "math.tan")
    return PrimitiveValue::floating(std::tan(value));
  if (opName == "math.sinh")
    return PrimitiveValue::floating(std::sinh(value));
  if (opName == "math.cosh")
    return PrimitiveValue::floating(std::cosh(value));
  if (opName == "math.tanh")
    return PrimitiveValue::floating(std::tanh(value));
  if (opName == "math.exp")
    return PrimitiveValue::floating(std::exp(value));
  if (opName == "math.exp2")
    return PrimitiveValue::floating(std::exp2(value));
  if (opName == "math.expm1")
    return PrimitiveValue::floating(std::expm1(value));
  if (opName == "math.log")
    return PrimitiveValue::floating(std::log(value));
  if (opName == "math.log2")
    return PrimitiveValue::floating(std::log2(value));
  if (opName == "math.log10")
    return PrimitiveValue::floating(std::log10(value));
  if (opName == "math.log1p")
    return PrimitiveValue::floating(std::log1p(value));
  if (opName == "math.floor")
    return PrimitiveValue::floating(std::floor(value));
  if (opName == "math.ceil")
    return PrimitiveValue::floating(std::ceil(value));
  if (opName == "math.round")
    return PrimitiveValue::floating(std::round(value));
  if (opName == "math.trunc")
    return PrimitiveValue::floating(std::trunc(value));
  if (opName == "math.roundeven")
    return roundEven(value);
  if (opName == "math.sqrt")
    return PrimitiveValue::floating(std::sqrt(value));
  if (opName == "math.rsqrt")
    return PrimitiveValue::floating(1.0 / std::sqrt(value));
  if (opName == "math.erf")
    return PrimitiveValue::floating(std::erf(value));
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s is not a supported unary math op",
                                 opName.str().c_str());
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
  return evaluatePrimitiveOperation(
      PrimitiveOperationDescriptor{opName.str(), "", 0}, operands);
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
  if (opName == "arith.divf") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    return PrimitiveValue::floating(asFloat(operands[0]) /
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
    auto amountOrErr = checkedShiftAmount(opName, operands[1], bitWidth);
    if (!amountOrErr)
      return amountOrErr.takeError();
    return integerFromBits(toUnsignedBits(operands[0], bitWidth)
                               << *amountOrErr,
                           bitWidth);
  }
  if (opName == "arith.shrui") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    auto amountOrErr = checkedShiftAmount(opName, operands[1], bitWidth);
    if (!amountOrErr)
      return amountOrErr.takeError();
    if (descriptor.isExact &&
        exactRightShiftWouldDiscardBits(operands[0], bitWidth, *amountOrErr))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s exact shift would discard non-zero bits",
          opName.str().c_str());
    return integerFromBits(toUnsignedBits(operands[0], bitWidth) >>
                               *amountOrErr,
                           bitWidth);
  }
  if (opName == "arith.shrsi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    auto amountOrErr = checkedShiftAmount(opName, operands[1], bitWidth);
    if (!amountOrErr)
      return amountOrErr.takeError();
    if (descriptor.isExact &&
        exactRightShiftWouldDiscardBits(operands[0], bitWidth, *amountOrErr))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s exact shift would discard non-zero bits",
          opName.str().c_str());
    return integerFromBits(
        arithmeticRightShiftBits(operands[0], bitWidth, *amountOrErr),
        bitWidth);
  }
  if (opName == "arith.divsi" || opName == "arith.remsi") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    const std::int64_t dividend =
        fromUnsignedBits(toUnsignedBits(operands[0], bitWidth), bitWidth);
    const std::int64_t divisor =
        fromUnsignedBits(toUnsignedBits(operands[1], bitWidth), bitWidth);
    if (divisor == 0)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "%s divisor must be non-zero",
                                     opName.str().c_str());
    if (opName == "arith.divsi" && dividend == signedMinForBitWidth(bitWidth) &&
        divisor == -1)
      return llvm::createStringError(std::errc::result_out_of_range,
                                     "%s signed overflow", opName.str().c_str());
    if (opName == "arith.divsi") {
      if (descriptor.isExact && dividend % divisor != 0)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "%s exact result would be poison", opName.str().c_str());
      return integerFromSigned(dividend / divisor, bitWidth);
    }
    if (dividend == signedMinForBitWidth(bitWidth) && divisor == -1)
      return integerFromSigned(0, bitWidth);
    return integerFromSigned(dividend % divisor, bitWidth);
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
    return asBoolean(operands[0]) ? operands[1] : operands[2];
  }
  if (opName == "arith.index_cast") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    return integerFromSigned(asInteger(operands[0]), bitWidth);
  }
  if (opName == "arith.extsi" || opName == "llvm.sext") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    const unsigned sourceBitWidth =
        normalizeBitWidth(descriptor.operandBitWidth);
    if (sourceBitWidth > bitWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s source bit width must not exceed result bit width",
          opName.str().c_str());
    return integerFromSigned(
        fromUnsignedBits(toUnsignedBits(operands[0], sourceBitWidth),
                         sourceBitWidth),
        bitWidth);
  }
  if (opName == "arith.trunci") {
    if (llvm::Error arity = requireArity(opName, operands, 1))
      return std::move(arity);
    const unsigned sourceBitWidth =
        normalizeBitWidth(descriptor.operandBitWidth);
    if (sourceBitWidth <= bitWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s source bit width must be wider than result bit width",
          opName.str().c_str());
    const std::uint64_t inputBits = toUnsignedBits(operands[0], sourceBitWidth);
    const unsigned truncatedBitCount = sourceBitWidth - bitWidth;
    const std::uint64_t truncatedBits = inputBits >> bitWidth;
    if (descriptor.noUnsignedWrap && truncatedBits != 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s overflow<nuw> result would be poison",
          opName.str().c_str());
    if (descriptor.noSignedWrap) {
      const bool resultSign = ((inputBits >> (bitWidth - 1)) & 1u) != 0;
      const std::uint64_t expectedTruncatedBits =
          resultSign ? lowBitsMask(truncatedBitCount) : 0;
      if (truncatedBits != expectedTruncatedBits)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "%s overflow<nsw> result would be poison",
            opName.str().c_str());
    }
    return integerFromBits(inputBits, bitWidth);
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
  if (opName == "llvm.intr.fabs")
    return evaluateMathUnary(opName, operands);
  if (opName == "llvm.arm.qadd16" || opName == "llvm.arm.qsub8" ||
      opName == "llvm.arm.qsub16") {
    if (llvm::Error arity = requireArity(opName, operands, 2))
      return std::move(arity);
    const unsigned laneWidth = opName == "llvm.arm.qsub8" ? 8 : 16;
    if (bitWidth == 0 || bitWidth % laneWidth != 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s result bit width must be a positive multiple of lane width %u",
          opName.str().c_str(), laneWidth);
    const bool isAdd = opName == "llvm.arm.qadd16";
    return packedSaturatingBinary(
        operands[0], operands[1], bitWidth, laneWidth,
        [isAdd](std::int64_t lhs, std::int64_t rhs) {
          return isAdd ? lhs + rhs : lhs - rhs;
        });
  }
  if (opName.starts_with("math.")) {
    if (opName == "math.absi") {
      if (llvm::Error arity = requireArity(opName, operands, 1))
        return std::move(arity);
      const std::int64_t value = asInteger(operands[0]);
      if (value == signedMinForBitWidth(bitWidth))
        return llvm::createStringError(
            std::errc::result_out_of_range,
            "%s cannot represent absolute value of signed minimum",
            opName.str().c_str());
      return integerFromSigned(value < 0 ? -value : value, bitWidth);
    }
    return evaluateMathUnary(opName, operands);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "%s is not supported by operation semantics",
                                 opName.str().c_str());
}
