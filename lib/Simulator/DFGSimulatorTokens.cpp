#include "DFGSimulatorInternal.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <limits>
#include <string>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

Token noneToken() { return Token{}; }

Token integerValueToken(std::int64_t value) {
  Token token;
  token.kind = TokenKind::Integer;
  token.intValue = value;
  return token;
}

Token floatValueToken(double value) {
  Token token;
  token.kind = TokenKind::Float;
  token.floatValue = value;
  return token;
}

Token boolValueToken(bool value) {
  Token token;
  token.kind = TokenKind::Bool;
  token.boolValue = value;
  return token;
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

std::string tokenToString(const Token &token, mlir::Type type) {
  if (token.kind == TokenKind::None)
    return "none";
  if (token.kind == TokenKind::Bool)
    return typePrefix(type) + ":" + (token.boolValue ? "true" : "false");
  if (token.kind == TokenKind::Integer) {
    if (mlir::isa<mlir::IndexType>(type))
      return typePrefix(type) + ":" + std::to_string(token.intValue);
    auto integer = mlir::cast<mlir::IntegerType>(type);
    if (integer.getWidth() <= 64)
      return typePrefix(type) + ":" + std::to_string(token.intValue);
    llvm::APInt bits = llvm::cantFail(tokenBitPattern(token, integer));
    llvm::SmallString<64> value;
    bits.toString(value, 10, /*Signed=*/false);
    return typePrefix(type) + ":" + value.str().str();
  }
  if (token.kind == TokenKind::Vector) {
    llvm::APInt bits = llvm::cantFail(tokenBitPattern(token, type));
    llvm::SmallString<64> value;
    bits.toString(value, 16, /*Signed=*/false);
    return typePrefix(type) + ":0x" + value.str().str();
  }
  if (token.kind == TokenKind::Pointer)
    return typePrefix(type) + ":ptr+" +
           std::to_string(token.pointer.byteOffset);
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << typePrefix(type) << ':';
  if (token.floatValue == 0.0 && std::signbit(token.floatValue))
    os << "-0";
  else if (std::floor(token.floatValue) == token.floatValue)
    os << static_cast<std::int64_t>(token.floatValue);
  else
    os << llvm::formatv("{0:f6}", token.floatValue);
  return os.str();
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
    if (vector.getRank() != 1 || vector.isScalable())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "vector token type must be fixed-size and rank-1");
    auto elementWidth = tokenTypeBitWidth(vector.getElementType());
    if (!elementWidth)
      return elementWidth.takeError();
    const uint64_t lanes = vector.getShape().front();
    if (lanes > std::numeric_limits<unsigned>::max() / *elementWidth)
      return llvm::createStringError(std::errc::value_too_large,
                                     "vector token bit width is unsupported");
    return static_cast<unsigned>(lanes * *elementWidth);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "token type has no exact bit representation");
}

llvm::Expected<llvm::APInt> tokenBitPattern(const Token &token,
                                            mlir::Type type) {
  auto width = tokenTypeBitWidth(type);
  if (!width)
    return width.takeError();
  if (token.bitPattern) {
    if (token.bitPattern->getBitWidth() != *width)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "token bit pattern width does not match its MLIR type");
    return *token.bitPattern;
  }

  if (mlir::isa<mlir::IntegerType>(type)) {
    if (token.kind == TokenKind::Bool)
      return llvm::APInt(*width, token.boolValue ? 1 : 0);
    if (token.kind == TokenKind::Integer)
      return llvm::APInt(*width, static_cast<uint64_t>(token.intValue),
                         /*isSigned=*/false, /*implicitTrunc=*/true);
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (token.kind != TokenKind::Float)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "floating-point token kind mismatch");
    llvm::APFloat value(token.floatValue);
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
  token.bitPattern = bits;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 1) {
      token.kind = TokenKind::Bool;
      token.boolValue = bits.isOne();
      return token;
    }
    token.kind = TokenKind::Integer;
    if (integer.getWidth() < 64)
      token.intValue = static_cast<std::int64_t>(bits.getZExtValue());
    else if (integer.getWidth() == 64)
      token.intValue = bits.getSExtValue();
    return token;
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    token.kind = TokenKind::Float;
    token.floatValue =
        llvm::APFloat(floating.getFloatSemantics(), bits).convertToDouble();
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
        token->intValue = intAttr.getValue().getSExtValue();
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

static llvm::Expected<llvm::APInt> parseIntegerBitPattern(llvm::StringRef raw,
                                                          unsigned bitWidth) {
  raw = raw.trim();
  if (bitWidth == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "integer token bit width must be nonzero");

  bool negative = false;
  if (!raw.empty() && (raw.front() == '-' || raw.front() == '+')) {
    negative = raw.front() == '-';
    raw = raw.drop_front();
  }
  if (raw.empty() ||
      !llvm::all_of(raw, [](char c) { return c >= '0' && c <= '9'; }))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "integer argument is not canonical base-10");

  llvm::APInt magnitude;
  if (raw.getAsInteger(10, magnitude))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "integer argument is not canonical base-10");

  bool fits = magnitude.getActiveBits() <= bitWidth;
  if (negative && magnitude.getActiveBits() == bitWidth)
    fits = magnitude.isPowerOf2();
  if (!fits)
    return llvm::createStringError(
        std::errc::result_out_of_range,
        "integer argument does not fit its declared bit width");

  llvm::APInt bits = magnitude.zextOrTrunc(bitWidth);
  if (negative)
    bits.negate();
  return bits;
}

static llvm::Expected<llvm::APInt> parsePackedBitPattern(llvm::StringRef raw,
                                                         unsigned bitWidth) {
  raw = raw.trim();
  unsigned radix = 10;
  if (raw.consume_front("0x") || raw.consume_front("0X"))
    radix = 16;
  if (raw.empty() || raw.starts_with("+") || raw.starts_with("-"))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector argument is not an unsigned packed integer");

  llvm::APInt bits;
  if (raw.getAsInteger(radix, bits))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector argument is not an unsigned packed integer");
  if (bits.getActiveBits() > bitWidth)
    return llvm::createStringError(
        std::errc::result_out_of_range,
        "vector argument does not fit its declared bit width");
  return bits.zextOrTrunc(bitWidth);
}

llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type) {
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
      token->intValue = bits->getSExtValue();
    return *token;
  }
  if (mlir::isa<mlir::IndexType>(type)) {
    std::int64_t value = 0;
    if (raw.getAsInteger(10, value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "index argument is not base-10");
    return integerValueToken(value);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
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
    auto bits = parsePackedBitPattern(raw, *width);
    if (!bits)
      return bits.takeError();
    return tokenFromBitPattern(*bits, type);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported runtime argument type");
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
