#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Common/VectorWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

Token noneToken() { return Token{}; }

Token integerValueToken(std::int64_t value) {
  Token token;
  token.kind = TokenKind::Integer;
  token.scalarValue = static_cast<std::uint64_t>(value);
  return token;
}

Token floatValueToken(double value) {
  Token token;
  token.kind = TokenKind::Float;
  token.scalarValue = llvm::bit_cast<std::uint64_t>(value);
  return token;
}

Token boolValueToken(bool value) {
  Token token;
  token.kind = TokenKind::Bool;
  token.scalarValue = value ? 1 : 0;
  return token;
}

llvm::Expected<Token> exceptionalValueToken(PrimitiveValueState state,
                                            mlir::Type type) {
  if (state == PrimitiveValueState::Defined)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "exceptional token state is defined");
  Token token;
  token.valueState = state;
  if (mlir::isa<mlir::IndexType>(type))
    token.kind = TokenKind::Integer;
  else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    token.kind = integer.getWidth() == 1 ? TokenKind::Bool : TokenKind::Integer;
  else if (mlir::isa<mlir::FloatType>(type))
    token.kind = TokenKind::Float;
  else if (mlir::isa<mlir::LLVM::LLVMPointerType>(type))
    token.kind = TokenKind::Pointer;
  else if (mlir::isa<mlir::VectorType>(type))
    token.kind = TokenKind::Vector;
  else
    return llvm::createStringError(
        std::errc::invalid_argument,
        "type has no exceptional scalar or fixed-vector token state");
  return token;
}

static bool usesDoubleFloatText(mlir::FloatType type) {
  return llvm::APFloat::isRepresentableBy(type.getFloatSemantics(),
                                          llvm::APFloat::IEEEdouble());
}

static std::string typePrefix(mlir::Type type) {
  if (mlir::isa<mlir::NoneType>(type))
    return "none";
  if (mlir::isa<mlir::IndexType>(type))
    return "index";
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return llvm::formatv("i{0}", intType.getWidth()).str();
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isF16())
      return "f16";
    if (floatType.isF32())
      return "f32";
    if (floatType.isF64())
      return "f64";
  }
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return storage;
}

static std::string formatHexBitPattern(const llvm::APInt &bits,
                                       unsigned minimumDigits = 0) {
  llvm::SmallString<64> value;
  bits.toString(value, 16, /*Signed=*/false);
  std::string formatted = "0x";
  if (minimumDigits > value.size())
    formatted.append(minimumDigits - value.size(), '0');
  formatted.append(value.begin(), value.end());
  return formatted;
}

static unsigned hexDigitCount(unsigned bitWidth) {
  return bitWidth / 4 + (bitWidth % 4 != 0);
}

llvm::Expected<std::string> tokenToString(const Token &token, mlir::Type type,
                                          mlir::Operation *scope) {
  if (token.valueState == PrimitiveValueState::Poison)
    return typePrefix(type) + ":poison";
  if (token.valueState == PrimitiveValueState::Undef)
    return typePrefix(type) + ":undef";
  if (token.kind == TokenKind::None)
    return "none";
  if (token.kind == TokenKind::Bool)
    return typePrefix(type) + ":" + (boolToken(token) ? "true" : "false");
  if (token.kind == TokenKind::Integer) {
    if (mlir::isa<mlir::IndexType>(type)) {
      auto width = loom::getIndexBitWidth(scope);
      if (!width)
        return width.takeError();
      auto bits = indexTokenBitPattern(token, *width);
      if (!bits)
        return bits.takeError();
      llvm::SmallString<48> value;
      bits->toString(value, 10, /*Signed=*/true);
      return typePrefix(type) + ":" + value.str().str();
    }
    auto integer = mlir::cast<mlir::IntegerType>(type);
    if (integer.getWidth() <= 64)
      return typePrefix(type) + ":" +
             std::to_string(static_cast<std::int64_t>(token.scalarValue));
    auto bits = tokenBitPattern(token, integer);
    if (!bits)
      return bits.takeError();
    llvm::SmallString<64> value;
    bits->toString(value, 10, /*Signed=*/false);
    return typePrefix(type) + ":" + value.str().str();
  }
  if (token.kind == TokenKind::Vector) {
    auto vector = mlir::dyn_cast<mlir::VectorType>(type);
    if (vector && token.vectorLanePayload()) {
      auto lanes = vectorPrimitiveValues(token, vector, scope);
      if (!lanes)
        return lanes.takeError();
      std::string storage;
      llvm::raw_string_ostream os(storage);
      os << typePrefix(type) << ":[";
      for (auto [ordinal, lane] : llvm::enumerate(*lanes)) {
        if (ordinal != 0)
          os << ',';
        if (lane.state == PrimitiveValueState::Poison)
          os << "poison";
        else if (lane.state == PrimitiveValueState::Undef)
          os << "undef";
        else
          os << formatHexBitPattern(*lane.bits);
      }
      os << ']';
      return os.str();
    }
    auto bits = vector && mlir::isa<mlir::IndexType>(vector.getElementType())
                    ? vectorIndexTokenBitPattern(token, vector, scope)
                    : tokenBitPattern(token, type);
    if (!bits)
      return bits.takeError();
    return typePrefix(type) + ":" + formatHexBitPattern(*bits);
  }
  if (token.kind == TokenKind::Float) {
    auto floating = mlir::cast<mlir::FloatType>(type);
    if (!usesDoubleFloatText(floating)) {
      auto bits = tokenBitPattern(token, type);
      if (!bits)
        return bits.takeError();
      return typePrefix(type) + ":" +
             formatHexBitPattern(*bits, hexDigitCount(bits->getBitWidth()));
    }
  }
  if (token.kind == TokenKind::Pointer) {
    const PointerValue *pointer = token.pointerValue();
    if (!pointer)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "pointer token has no provenance");
    llvm::SmallString<48> offset;
    pointer->byteOffset.toString(offset, 10, /*Signed=*/true);
    return llvm::formatv(
               "{0}:as{1}:object{2}:{3}:{4}", typePrefix(type),
               pointer->addressSpace, pointer->objectOrdinal, offset,
               formatHexBitPattern(
                   pointer->representation,
                   hexDigitCount(pointer->representation.getBitWidth())))
        .str();
  }
  if (token.kind == TokenKind::MemoryCapability)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "memory capability token has no ordinary value spelling");
  double floatValue = llvm::bit_cast<double>(token.scalarValue);
  if (token.hasExactBitPattern()) {
    auto bits = tokenBitPattern(token, type);
    if (!bits)
      return bits.takeError();
    floatValue =
        llvm::APFloat(mlir::cast<mlir::FloatType>(type).getFloatSemantics(),
                      *bits)
            .convertToDouble();
  }
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << typePrefix(type) << ':';
  if (floatValue == 0.0 && std::signbit(floatValue))
    os << "-0";
  else if (std::floor(floatValue) == floatValue)
    os << static_cast<std::int64_t>(floatValue);
  else
    os << llvm::formatv("{0:f6}", floatValue);
  return os.str();
}

// A token stores its bit pattern in one APInt, whose width is an unsigned, so
// an exact type wider than that has no token representation here rather than a
// truncated one. The check is arithmetic: no value of that width is formed.
static llvm::Expected<unsigned> narrowTokenBitWidth(std::uint64_t width) {
  const auto narrowed = static_cast<unsigned>(width);
  if (narrowed != width)
    return llvm::createStringError(
        std::errc::value_too_large,
        "bit width %llu exceeds the token representation",
        static_cast<unsigned long long>(width));
  return narrowed;
}

llvm::Expected<unsigned> tokenTypeBitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 0)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "zero-width integer token type");
    return integer.getWidth();
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    auto width = dataflow::semantics::getFlattenedVectorBitWidth(vector);
    if (!width)
      return width.takeError();
    return narrowTokenBitWidth(*width);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "token type has no exact bit representation");
}

llvm::Expected<llvm::APInt> tokenBitPattern(const Token &token,
                                            mlir::Type type) {
  if (token.valueState != PrimitiveValueState::Defined)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "exceptional token has no defined bit pattern");
  auto width = tokenTypeBitWidth(type);
  if (!width)
    return width.takeError();
  if (token.hasExactBitPattern()) {
    if (token.exactBitWidth() != *width)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "token bit pattern width does not match its MLIR type");
    return token.exactBitPattern();
  }

  if (mlir::isa<mlir::IntegerType>(type)) {
    if (token.kind == TokenKind::Bool)
      return llvm::APInt(*width, token.scalarValue != 0 ? 1 : 0);
    if (token.kind == TokenKind::Integer)
      return llvm::APInt(*width, token.scalarValue,
                         /*isSigned=*/false, /*implicitTrunc=*/true);
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (token.kind != TokenKind::Float)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "floating-point token kind mismatch");
    llvm::APFloat value(llvm::bit_cast<double>(token.scalarValue));
    bool losesInfo = false;
    (void)value.convert(floating.getFloatSemantics(),
                        llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return value.bitcastToAPInt();
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "token kind has no representation for type");
}

llvm::Expected<Token> tokenFromBitPattern(const llvm::APInt &bits,
                                          mlir::Type type) {
  auto width = tokenTypeBitWidth(type);
  if (!width)
    return width.takeError();
  if (bits.getBitWidth() != *width)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "bit pattern width does not match destination MLIR type");

  Token token;
  token.setExactBitPattern(bits);
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 1) {
      token.kind = TokenKind::Bool;
      return token;
    }
    token.kind = TokenKind::Integer;
    if (integer.getWidth() < 64)
      token.scalarValue = bits.getZExtValue();
    else if (integer.getWidth() == 64)
      token.scalarValue = static_cast<std::uint64_t>(bits.getSExtValue());
    return token;
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    token.kind = TokenKind::Float;
    (void)floating;
    return token;
  }
  if (mlir::isa<mlir::VectorType>(type)) {
    token.kind = TokenKind::Vector;
    return token;
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported bit-pattern destination type");
}

llvm::Expected<Token> tokenFromTypedAttr(mlir::TypedAttr attr) {
  if (mlir::isa<mlir::NoneType>(attr.getType()))
    return noneToken();
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(intAttr.getType())) {
      auto token = tokenFromBitPattern(intAttr.getValue(), intType);
      if (!token)
        return token.takeError();
      if (intType.getWidth() <= 64)
        token->scalarValue =
            static_cast<std::uint64_t>(intAttr.getValue().getSExtValue());
      return *token;
    }
    return integerValueToken(intAttr.getValue().getSExtValue());
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return tokenFromBitPattern(floatAttr.getValue().bitcastToAPInt(),
                               floatAttr.getType());
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported dataflow.constant attribute");
}

static llvm::Expected<llvm::APInt>
parseIntegerBitPattern(llvm::StringRef raw, unsigned bitWidth,
                       llvm::StringRef argumentKind = "integer") {
  raw = raw.trim();
  if (bitWidth == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s token bit width must be nonzero",
                                   argumentKind.str().c_str());

  bool negative = false;
  if (!raw.empty() && (raw.front() == '-' || raw.front() == '+')) {
    negative = raw.front() == '-';
    raw = raw.drop_front();
  }
  if (raw.empty() ||
      !llvm::all_of(raw, [](char c) { return c >= '0' && c <= '9'; }))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s argument is not canonical base-10",
                                   argumentKind.str().c_str());

  llvm::APInt magnitude;
  if (raw.getAsInteger(10, magnitude))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s argument is not canonical base-10",
                                   argumentKind.str().c_str());

  bool fits = magnitude.getActiveBits() <= bitWidth;
  if (negative && magnitude.getActiveBits() == bitWidth)
    fits = magnitude.isPowerOf2();
  if (!fits)
    return llvm::createStringError(
        std::errc::result_out_of_range,
        "%s argument does not fit its declared bit width",
        argumentKind.str().c_str());

  llvm::APInt bits = magnitude.zextOrTrunc(bitWidth);
  if (negative)
    bits.negate();
  return bits;
}

static llvm::Expected<llvm::APInt>
parsePackedBitPattern(llvm::StringRef raw, unsigned bitWidth,
                      llvm::StringRef argumentKind) {
  raw = raw.trim();
  unsigned radix = 10;
  if (raw.consume_front("0x") || raw.consume_front("0X"))
    radix = 16;
  if (raw.empty() || raw.starts_with("+") || raw.starts_with("-"))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "%s argument is not an unsigned packed integer",
        argumentKind.str().c_str());

  llvm::APInt bits;
  if (raw.getAsInteger(radix, bits))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "%s argument is not an unsigned packed integer",
        argumentKind.str().c_str());
  if (bits.getActiveBits() > bitWidth)
    return llvm::createStringError(
        std::errc::result_out_of_range,
        "%s argument does not fit its declared bit width",
        argumentKind.str().c_str());
  return bits.zextOrTrunc(bitWidth);
}

static llvm::Expected<llvm::APInt>
parseExactFloatBitPattern(llvm::StringRef raw, mlir::FloatType type) {
  raw = raw.trim();
  llvm::StringRef payload = raw;
  const std::string typeName = typePrefix(type);
  if (!payload.consume_front("0x"))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "exact %s argument must use 0x-prefixed hexadecimal", typeName.c_str());

  auto width = tokenTypeBitWidth(type);
  if (!width)
    return width.takeError();
  const unsigned digits = hexDigitCount(*width);
  if (payload.size() != digits)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "exact %s argument requires %u hexadecimal digits", typeName.c_str(),
        digits);
  if (!llvm::all_of(payload, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
      }))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "exact %s argument is not canonical hexadecimal", typeName.c_str());
  return parsePackedBitPattern(raw, *width, "float");
}

static llvm::Expected<Token> parseNonIndexRuntimeToken(llvm::StringRef raw,
                                                       mlir::Type type) {
  raw = raw.trim();
  if (mlir::isa<mlir::NoneType>(type)) {
    if (raw == "none")
      return noneToken();
    return llvm::createStringError(std::errc::invalid_argument,
                                   "none argument expects value 'none'");
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() == 1) {
      if (raw == "true" || raw == "1")
        return boolValueToken(true);
      if (raw == "false" || raw == "0")
        return boolValueToken(false);
      return llvm::createStringError(std::errc::invalid_argument,
                                     "i1 argument expects true/false/0/1");
    }
    auto bits = parseIntegerBitPattern(raw, intType.getWidth());
    if (!bits)
      return bits.takeError();
    auto token = tokenFromBitPattern(*bits, type);
    if (!token)
      return token.takeError();
    if (intType.getWidth() <= 64 && raw.starts_with("-"))
      token->scalarValue = static_cast<std::uint64_t>(bits->getSExtValue());
    return *token;
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (!usesDoubleFloatText(floatType)) {
      auto bits = parseExactFloatBitPattern(raw, floatType);
      if (!bits)
        return bits.takeError();
      return tokenFromBitPattern(*bits, type);
    }
    double value = 0.0;
    if (raw.getAsDouble(value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "float argument is not parseable");
    return floatValueToken(value);
  }
  if (mlir::isa<mlir::VectorType>(type)) {
    auto width = tokenTypeBitWidth(type);
    if (!width)
      return width.takeError();
    auto bits = parsePackedBitPattern(raw, *width, "vector");
    if (!bits)
      return bits.takeError();
    return tokenFromBitPattern(*bits, type);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported runtime argument type");
}

// An `index` element carries no width in its MLIR type, so the token width uses
// the resolved index width instead of the semantic element width.
static llvm::Expected<unsigned>
indexVectorTokenBitWidth(mlir::VectorType type, mlir::Operation *scope) {
  auto elementWidth = loom::getIndexBitWidth(scope);
  if (!elementWidth)
    return elementWidth.takeError();
  auto width = loom::getFixedVectorBitWidth(type, *elementWidth);
  if (!width)
    return width.takeError();
  return narrowTokenBitWidth(*width);
}

llvm::Expected<unsigned> resolvedTokenTypeBitWidth(mlir::Type type,
                                                   mlir::Operation *scope) {
  if (mlir::isa<mlir::IndexType>(type))
    return loom::getIndexBitWidth(scope);
  if (auto pointer = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
    auto layout = loom::resolvePointerLayout(scope, pointer.getAddressSpace());
    if (!layout)
      return layout.takeError();
    return layout->representationBits;
  }
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    if (mlir::isa<mlir::IndexType>(vector.getElementType()))
      return indexVectorTokenBitWidth(vector, scope);
  return tokenTypeBitWidth(type);
}

static llvm::Expected<unsigned>
resolvedVectorLaneBitWidth(mlir::VectorType type, mlir::Operation *scope) {
  auto width = resolvedTokenTypeBitWidth(type.getElementType(), scope);
  if (!width)
    return width.takeError();
  if (*width == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "vector lane width is zero");
  return *width;
}

static llvm::Expected<unsigned>
resolvedVectorPayloadBitWidth(mlir::VectorType type, unsigned laneBitWidth) {
  const std::uint64_t laneCount = type.getNumElements();
  if (laneCount == 0 ||
      laneCount > std::numeric_limits<unsigned>::max() / laneBitWidth)
    return llvm::createStringError(
        std::errc::value_too_large,
        "fixed-vector token exceeds the simulator bit-width domain");
  return static_cast<unsigned>(laneCount) * laneBitWidth;
}

llvm::Expected<llvm::SmallVector<PrimitiveValue, 8>>
vectorPrimitiveValues(const Token &token, mlir::VectorType type,
                      mlir::Operation *scope) {
  if (type.isScalable())
    return llvm::createStringError(std::errc::not_supported,
                                   "scalable vector token is unsupported");
  const std::size_t laneCount = static_cast<std::size_t>(type.getNumElements());
  auto laneBitWidth = resolvedVectorLaneBitWidth(type, scope);
  if (!laneBitWidth)
    return laneBitWidth.takeError();
  auto payloadBitWidth = resolvedVectorPayloadBitWidth(type, *laneBitWidth);
  if (!payloadBitWidth)
    return payloadBitWidth.takeError();

  llvm::SmallVector<PrimitiveValue, 8> lanes;
  lanes.reserve(laneCount);
  if (token.valueState != PrimitiveValueState::Defined) {
    const PrimitiveValue exceptional =
        token.valueState == PrimitiveValueState::Poison
            ? PrimitiveValue::poison()
            : PrimitiveValue::undef();
    lanes.assign(laneCount, exceptional);
    return lanes;
  }
  if (token.kind != TokenKind::Vector)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "fixed-vector token has the wrong kind");

  if (const VectorLanePayload *payload = token.vectorLanePayload()) {
    if (payload->laneBitWidth != *laneBitWidth ||
        payload->laneStates.size() != laneCount ||
        payload->packedBits.getBitWidth() != *payloadBitWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mixed-lane payload does not match its fixed-vector type");
    for (auto [ordinal, state] : llvm::enumerate(payload->laneStates)) {
      if (state == PrimitiveValueState::Poison)
        lanes.push_back(PrimitiveValue::poison());
      else if (state == PrimitiveValueState::Undef)
        lanes.push_back(PrimitiveValue::undef());
      else
        lanes.push_back(PrimitiveValue::integer(payload->packedBits.extractBits(
            *laneBitWidth, static_cast<unsigned>(ordinal) * *laneBitWidth)));
    }
    return lanes;
  }

  auto bits = resolvedTokenBitPattern(token, type, scope);
  if (!bits)
    return bits.takeError();
  if (bits->getBitWidth() != *payloadBitWidth)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "fixed-vector bit pattern does not match its resolved lane width");
  for (std::size_t ordinal = 0; ordinal < laneCount; ++ordinal)
    lanes.push_back(PrimitiveValue::integer(bits->extractBits(
        *laneBitWidth, static_cast<unsigned>(ordinal) * *laneBitWidth)));
  return lanes;
}

llvm::Expected<Token>
tokenFromVectorPrimitiveValues(llvm::ArrayRef<PrimitiveValue> lanes,
                               mlir::VectorType type, mlir::Operation *scope) {
  if (type.isScalable() ||
      lanes.size() != static_cast<std::size_t>(type.getNumElements()))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector lane sequence does not match its fixed-vector type");
  auto laneBitWidth = resolvedVectorLaneBitWidth(type, scope);
  if (!laneBitWidth)
    return laneBitWidth.takeError();
  auto payloadBitWidth = resolvedVectorPayloadBitWidth(type, *laneBitWidth);
  if (!payloadBitWidth)
    return payloadBitWidth.takeError();

  llvm::APInt packedBits(*payloadBitWidth, 0);
  llvm::SmallVector<PrimitiveValueState, 4> states;
  states.reserve(lanes.size());
  bool allDefined = true;
  bool allPoison = true;
  bool allUndef = true;
  for (auto [ordinal, lane] : llvm::enumerate(lanes)) {
    states.push_back(lane.state);
    allDefined &= lane.state == PrimitiveValueState::Defined;
    allPoison &= lane.state == PrimitiveValueState::Poison;
    allUndef &= lane.state == PrimitiveValueState::Undef;
    if (lane.state != PrimitiveValueState::Defined)
      continue;
    if (!lane.bits || lane.bits->getBitWidth() != *laneBitWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "defined vector lane has the wrong resolved bit width");
    packedBits.insertBits(*lane.bits,
                          static_cast<unsigned>(ordinal) * *laneBitWidth);
  }
  if (allDefined)
    return tokenFromResolvedBitPattern(packedBits, type, scope);
  if (allPoison)
    return exceptionalValueToken(PrimitiveValueState::Poison, type);
  if (allUndef)
    return exceptionalValueToken(PrimitiveValueState::Undef, type);

  Token token;
  token.kind = TokenKind::Vector;
  token.setVectorLanePayload(VectorLanePayload{*laneBitWidth, std::move(states),
                                               std::move(packedBits)});
  return token;
}

llvm::Expected<llvm::APInt> vectorIndexTokenBitPattern(const Token &token,
                                                       mlir::VectorType type,
                                                       mlir::Operation *scope) {
  if (!mlir::isa<mlir::IndexType>(type.getElementType()))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector index token type must have 'index' elements");
  auto width = indexVectorTokenBitWidth(type, scope);
  if (!width)
    return width.takeError();
  if (!token.hasExactBitPattern() || token.exactBitWidth() != *width)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector index token bit pattern width does not match its MLIR type");
  return token.exactBitPattern();
}

llvm::Expected<llvm::APInt> indexTokenBitPattern(const Token &token,
                                                 unsigned width) {
  if (token.valueState != PrimitiveValueState::Defined)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "exceptional index token has no defined bit pattern");
  if (token.hasExactBitPattern()) {
    if (token.exactBitWidth() != width)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "index token bit pattern width does not match the resolved index "
          "width");
    return token.exactBitPattern();
  }
  if (token.kind != TokenKind::Integer)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "index token kind is not integer");
  // A token that carries only a host integer holds the complete semantic
  // value, so widening it is exact and narrowing it wraps like the declared
  // index type does.
  return llvm::APInt(64, token.scalarValue,
                     /*isSigned=*/true)
      .sextOrTrunc(width);
}

Token indexToken(const llvm::APInt &value) {
  Token token;
  token.kind = TokenKind::Integer;
  token.setExactBitPattern(value);
  if (value.getBitWidth() <= 64)
    token.scalarValue = static_cast<std::uint64_t>(value.getSExtValue());
  return token;
}

llvm::Expected<llvm::APInt> resolvedTokenBitPattern(const Token &token,
                                                    mlir::Type type,
                                                    mlir::Operation *scope) {
  if (mlir::isa<mlir::IndexType>(type)) {
    auto width = loom::getIndexBitWidth(scope);
    if (!width)
      return width.takeError();
    return indexTokenBitPattern(token, *width);
  }
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
    if (mlir::isa<mlir::IndexType>(vector.getElementType()))
      return vectorIndexTokenBitPattern(token, vector, scope);
  if (auto pointer = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
    auto layout = loom::resolvePointerLayout(scope, pointer.getAddressSpace());
    if (!layout)
      return layout.takeError();
    const PointerValue *value = token.pointerValue();
    if (token.kind != TokenKind::Pointer || !value)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "pointer token has no provenance");
    if (value->addressSpace != layout->addressSpace ||
        value->representation.getBitWidth() != layout->representationBits ||
        value->byteOffset.getBitWidth() != layout->addressBits)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "pointer token does not match the exact LLVM pointer layout");
    return value->representation;
  }
  return tokenBitPattern(token, type);
}

llvm::Expected<Token> tokenFromResolvedBitPattern(const llvm::APInt &bits,
                                                  mlir::Type type,
                                                  mlir::Operation *scope) {
  if (mlir::isa<mlir::IndexType>(type)) {
    auto width = loom::getIndexBitWidth(scope);
    if (!width)
      return width.takeError();
    if (bits.getBitWidth() != *width)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "bit pattern width does not match the resolved index width");
    return indexToken(bits);
  }
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    if (mlir::isa<mlir::IndexType>(vector.getElementType())) {
      auto width = indexVectorTokenBitWidth(vector, scope);
      if (!width)
        return width.takeError();
      if (bits.getBitWidth() != *width)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "bit pattern width does not match the resolved index-vector "
            "width");
      Token token;
      token.kind = TokenKind::Vector;
      token.setExactBitPattern(bits);
      return token;
    }
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "pointer provenance cannot be reconstructed from representation bits");
  return tokenFromBitPattern(bits, type);
}

llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type,
                                        mlir::Operation *scope) {
  raw = raw.trim();
  if (raw == "poison")
    return exceptionalValueToken(PrimitiveValueState::Poison, type);
  if (raw == "undef")
    return exceptionalValueToken(PrimitiveValueState::Undef, type);
  if (mlir::isa<mlir::IndexType>(type)) {
    auto width = loom::getIndexBitWidth(scope);
    if (!width)
      return width.takeError();
    auto bits = parseIntegerBitPattern(raw, *width, "index");
    if (!bits)
      return bits.takeError();
    return indexToken(*bits);
  }

  auto vector = mlir::dyn_cast<mlir::VectorType>(type);
  if (!vector || !mlir::isa<mlir::IndexType>(vector.getElementType()))
    return parseNonIndexRuntimeToken(raw, type);

  auto width = indexVectorTokenBitWidth(vector, scope);
  if (!width)
    return width.takeError();
  auto bits = parsePackedBitPattern(raw, *width, "vector");
  if (!bits)
    return bits.takeError();
  Token token;
  token.kind = TokenKind::Vector;
  token.setExactBitPattern(*bits);
  return token;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
