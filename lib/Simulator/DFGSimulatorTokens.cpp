#include "DFGSimulatorInternal.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"

#include <limits>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

static llvm::Expected<unsigned> tokenTypeBitWidth(mlir::Type type) {
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
    if (integer.getWidth() <= 64)
      token.intValue = static_cast<std::int64_t>(bits.getZExtValue());
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

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
