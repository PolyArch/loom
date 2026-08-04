#include "Hardware/RTL/Transport.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

using namespace loom::hardware::rtl;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

unsigned integerWidth(mlir::Value value) {
  return mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
}

void lowBitAlignmentAndZeroWidthPayload() {
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> top = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(top->getBody());

  const mlir::Type bit = builder.getI1Type();
  const mlir::Type inputPayload = builder.getIntegerType(16);
  const mlir::Type inputTag = builder.getIntegerType(3);
  const mlir::Type outputPayload = builder.getIntegerType(8);
  const mlir::Type outputTag = builder.getIntegerType(5);
  llvm::SmallVector<circt::hw::PortInfo, 3> inputPorts;
  llvm::SmallVector<circt::hw::PortInfo, 3> outputPorts;
  auto port = [&](llvm::StringRef name, mlir::Type type,
                  circt::hw::ModulePort::Direction direction) {
    return circt::hw::PortInfo{{builder.getStringAttr(name), type, direction}};
  };
  inputPorts.push_back(
      port("valid", bit, circt::hw::ModulePort::Direction::Input));
  inputPorts.push_back(
      port("payload", inputPayload, circt::hw::ModulePort::Direction::Input));
  inputPorts.push_back(
      port("tag", inputTag, circt::hw::ModulePort::Direction::Input));
  outputPorts.push_back(
      port("out_valid", bit, circt::hw::ModulePort::Direction::Output));
  outputPorts.push_back(port("out_payload", outputPayload,
                             circt::hw::ModulePort::Direction::Output));
  outputPorts.push_back(
      port("out_tag", outputTag, circt::hw::ModulePort::Direction::Output));

  mlir::Value inputValid;
  mlir::Value adaptedValid;
  mlir::Value adaptedPayload;
  mlir::Value adaptedTag;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("adapt_tagged"),
      circt::hw::ModulePortInfo(inputPorts, outputPorts),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        inputValid = accessor.getInput("valid");
        ForwardTransportSignals result =
            take(__func__, adaptForwardTransportSignals(
                               bodyBuilder, location,
                               {::fabric::DataPathKind::BitsTag, 16, 3},
                               {::fabric::DataPathKind::BitsTag, 8, 5},
                               {inputValid, accessor.getInput("payload"),
                                accessor.getInput("tag")}));
        adaptedValid = result.valid;
        adaptedPayload = *result.payload;
        adaptedTag = *result.tag;
        accessor.setOutput("out_valid", result.valid);
        accessor.setOutput("out_payload", *result.payload);
        accessor.setOutput("out_tag", *result.tag);
      });

  require(__func__, adaptedValid == inputValid,
          "width adaptation changed the valid signal");
  auto payloadExtract = adaptedPayload.getDefiningOp<circt::comb::ExtractOp>();
  require(__func__,
          payloadExtract && payloadExtract.getLowBit() == 0 &&
              integerWidth(adaptedPayload) == 8,
          "payload truncation did not keep the low bits");
  auto tagConcat = adaptedTag.getDefiningOp<circt::comb::ConcatOp>();
  require(__func__, tagConcat && integerWidth(adaptedTag) == 5,
          "tag extension was not an explicit zero extension");
  auto highZeros =
      tagConcat.getInputs().front().getDefiningOp<circt::hw::ConstantOp>();
  require(__func__, highZeros && highZeros.getValue().isZero(),
          "tag extension did not place zeros in the high bits");

  mlir::Value zeroPayload;
  llvm::SmallVector<circt::hw::PortInfo, 2> zeroInputPorts{
      port("valid", bit, circt::hw::ModulePort::Direction::Input),
      port("tag", inputTag, circt::hw::ModulePort::Direction::Input)};
  llvm::SmallVector<circt::hw::PortInfo, 3> zeroOutputPorts{
      port("out_valid", bit, circt::hw::ModulePort::Direction::Output),
      port("out_payload", outputPayload,
           circt::hw::ModulePort::Direction::Output),
      port("out_tag", bit, circt::hw::ModulePort::Direction::Output)};
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("adapt_zero_payload"),
      circt::hw::ModulePortInfo(zeroInputPorts, zeroOutputPorts),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        ForwardTransportSignals result =
            take(__func__, adaptForwardTransportSignals(
                               bodyBuilder, location,
                               {::fabric::DataPathKind::BitsTag, 0, 3},
                               {::fabric::DataPathKind::BitsTag, 8, 1},
                               {accessor.getInput("valid"), std::nullopt,
                                accessor.getInput("tag")}));
        zeroPayload = *result.payload;
        accessor.setOutput("out_valid", result.valid);
        accessor.setOutput("out_payload", *result.payload);
        accessor.setOutput("out_tag", *result.tag);
      });
  auto payloadZero = zeroPayload.getDefiningOp<circt::hw::ConstantOp>();
  require(__func__,
          payloadZero && payloadZero.getValue().isZero() &&
              integerWidth(zeroPayload) == 8,
          "zero-width source payload did not become a destination-width zero");

  require(__func__, mlir::succeeded(mlir::verify(*top)),
          "transport adapter produced invalid CIRCT");
}

void malformedOrCrossKindAdaptationFails() {
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> top = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(top->getBody());
  mlir::Value valid =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(1, 1));
  mlir::Value payload =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(8, 0));

  expectErrorContains(
      __func__,
      adaptForwardTransportSignals(builder, location,
                                   {::fabric::DataPathKind::Bits, 8, 0},
                                   {::fabric::DataPathKind::BitsTag, 8, 2},
                                   {valid, payload, std::nullopt}),
      "different Fabric transport kinds");
  expectErrorContains(
      __func__,
      adaptForwardTransportSignals(builder, location,
                                   {::fabric::DataPathKind::BitsTag, 8, 0},
                                   {::fabric::DataPathKind::BitsTag, 8, 2},
                                   {valid, payload, std::nullopt}),
      "source type is malformed");
  expectErrorContains(
      __func__,
      adaptForwardTransportSignals(
          builder, location, {::fabric::DataPathKind::Bits, 16, 0},
          {::fabric::DataPathKind::Bits, 8, 0}, {valid, payload, std::nullopt}),
      "source payload signal");
  expectErrorContains(
      __func__,
      adaptForwardTransportSignals(builder, location,
                                   {::fabric::DataPathKind::Bits, 8, 0},
                                   {::fabric::DataPathKind::Bits, 8, 0},
                                   {mlir::Value{}, payload, std::nullopt}),
      "source valid signal");
}

} // namespace

int main() {
  lowBitAlignmentAndZeroWidthPayload();
  malformedOrCrossKindAdaptationFails();
  return 0;
}
