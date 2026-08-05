#include "Hardware/RTL/Transport.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
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

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(const char *test) : test_(test) {
    llvm::SmallString<128> path;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-transport-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      std::cerr << test_
                << ": unable to remove temporary directory: " << error.message()
                << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

unsigned integerWidth(mlir::Value value) {
  return mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
}

loom::fabric::FinalizedFabricRoot
makePointConnectionFabric(const char *test, const loom::ArtifactStore &store) {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
        module {
          fabric.module @point_connections(
              %arg: !fabric.bits_tag<16, 3>)
              -> !fabric.bits_tag<8, 9> {
            %wide = fabric.fifo %arg [max_depth = 2, bypassable = false]
                : !fabric.bits_tag<16, 3>
            %narrow = fabric.switch [temporal] %wide
                [{connectivity_table = ["1"], route_table_size = 1 : i32}]
                : (!fabric.bits_tag<16, 3> to !fabric.bits_tag<4, 7>)
                -> !fabric.bits_tag<4, 7>
            %extended = fabric.switch [temporal] %narrow
                [{connectivity_table = ["1"], route_table_size = 1 : i32}]
                : (!fabric.bits_tag<4, 7> to !fabric.bits_tag<8, 9>)
                -> !fabric.bits_tag<8, 9>
            fabric.yield %extended : !fabric.bits_tag<8, 9>
          }
        }
      )mlir",
                                              &context);
  require(test, static_cast<bool>(source),
          "unable to parse point-connection Fabric fixture");
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root),
          "point-connection fixture has no Module root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

const loom::fabric::FabricPointConnectionPayload &
findConnection(const char *test, const loom::fabric::FabricArtifactView &view,
               std::uint32_t sourcePayloadWidth, std::uint32_t sourceTagWidth,
               std::uint32_t destinationPayloadWidth,
               std::uint32_t destinationTagWidth) {
  std::string inventory;
  for (const loom::fabric::FabricPointConnectionPayload &connection :
       view.pointConnections()) {
    const auto source = view.transportEndpointDataPath(connection.source);
    const auto destination =
        view.transportEndpointDataPath(connection.destination);
    if (source && destination)
      inventory += " [" + std::to_string(source->payloadWidthBits) + "," +
                   std::to_string(source->tagWidthBits) + " -> " +
                   std::to_string(destination->payloadWidthBits) + "," +
                   std::to_string(destination->tagWidthBits) + "]";
    if (source && destination &&
        source->payloadWidthBits == sourcePayloadWidth &&
        source->tagWidthBits == sourceTagWidth &&
        destination->payloadWidthBits == destinationPayloadWidth &&
        destination->tagWidthBits == destinationTagWidth)
      return connection;
  }
  fail(test, "expected point connection is absent; inventory:" + inventory);
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

  const std::uint32_t excessiveWidth = mlir::IntegerType::kMaxWidth + 1u;
  mlir::Value tag =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(3, 0));
  auto expectCapacityFailure = [&](::fabric::DataPathType sourceType,
                                   ::fabric::DataPathType destinationType,
                                   ForwardTransportSignals signals,
                                   llvm::StringRef message) {
    const auto before =
        std::distance(top->getBody()->begin(), top->getBody()->end());
    expectErrorContains(
        __func__,
        adaptForwardTransportSignals(builder, location, sourceType,
                                     destinationType, std::move(signals)),
        message);
    require(__func__,
            std::distance(top->getBody()->begin(), top->getBody()->end()) ==
                before,
            "excessive transport width produced partial CIRCT");
  };
  expectCapacityFailure({::fabric::DataPathKind::Bits, excessiveWidth, 0},
                        {::fabric::DataPathKind::Bits, 8, 0},
                        {valid, payload, std::nullopt},
                        "source payload width exceeds CIRCT capacity");
  expectCapacityFailure({::fabric::DataPathKind::Bits, 8, 0},
                        {::fabric::DataPathKind::Bits, excessiveWidth, 0},
                        {valid, payload, std::nullopt},
                        "destination payload width exceeds CIRCT capacity");
  expectCapacityFailure({::fabric::DataPathKind::BitsTag, 8, excessiveWidth},
                        {::fabric::DataPathKind::BitsTag, 8, 3},
                        {valid, payload, tag},
                        "source tag width exceeds CIRCT capacity");
  expectCapacityFailure({::fabric::DataPathKind::BitsTag, 8, 3},
                        {::fabric::DataPathKind::BitsTag, 8, excessiveWidth},
                        {valid, payload, tag},
                        "destination tag width exceeds CIRCT capacity");
}

void exactFabricPointConnectionsOwnAdaptationTypes() {
  TemporaryDirectory directory(__func__);
  loom::ArtifactStore store(directory.path());
  loom::fabric::FinalizedFabricRoot fabric =
      makePointConnectionFabric(__func__, store);
  const loom::fabric::FabricArtifactView &view = fabric.view();
  require(__func__, view.pointConnections().size() == 2,
          "fixture did not produce two exact point connections");
  const auto &narrowing = findConnection(__func__, view, 16, 3, 4, 7);
  const auto &widening = findConnection(__func__, view, 4, 7, 8, 9);

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> top = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(top->getBody());

  mlir::Value valid =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(1, 1));
  mlir::Value payload16 =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(16, 0x35a7));
  mlir::Value tag3 =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(3, 5));
  ForwardTransportSignals narrowed =
      take(__func__,
           adaptFabricPointConnectionForwardSignals(
               builder, location, view, narrowing, {valid, payload16, tag3}));
  require(__func__, narrowed.valid == valid,
          "point-connection adaptation changed valid");
  auto payloadExtract =
      narrowed.payload->getDefiningOp<circt::comb::ExtractOp>();
  require(__func__,
          payloadExtract && payloadExtract.getLowBit() == 0 &&
              integerWidth(*narrowed.payload) == 4,
          "point-connection narrowing did not retain low payload bits");
  auto tagExtension = narrowed.tag->getDefiningOp<circt::comb::ConcatOp>();
  require(__func__, tagExtension && integerWidth(*narrowed.tag) == 7,
          "point-connection narrowing did not zero-extend its tag");
  auto tagZeros =
      tagExtension.getInputs().front().getDefiningOp<circt::hw::ConstantOp>();
  require(__func__, tagZeros && tagZeros.getValue().isZero(),
          "point-connection tag extension has nonzero high bits");

  ForwardTransportSignals widened =
      take(__func__, adaptFabricPointConnectionForwardSignals(
                         builder, location, view, widening,
                         {narrowed.valid, *narrowed.payload, *narrowed.tag}));
  auto payloadExtension =
      widened.payload->getDefiningOp<circt::comb::ConcatOp>();
  auto widerTagExtension = widened.tag->getDefiningOp<circt::comb::ConcatOp>();
  require(__func__, payloadExtension && integerWidth(*widened.payload) == 8,
          "point-connection widening did not zero-extend its payload");
  require(__func__, widerTagExtension && integerWidth(*widened.tag) == 9,
          "point-connection widening did not zero-extend its tag");
  require(__func__, mlir::succeeded(mlir::verify(*top)),
          "exact point-connection adaptation produced invalid CIRCT");

  loom::fabric::FabricPointConnectionPayload reversed{narrowing.destination,
                                                      narrowing.source};
  const auto beforeForeign =
      std::distance(top->getBody()->begin(), top->getBody()->end());
  expectErrorContains(
      __func__,
      adaptFabricPointConnectionForwardSignals(
          builder, location, view, reversed, {valid, payload16, tag3}),
      "point connection");
  require(__func__,
          std::distance(top->getBody()->begin(), top->getBody()->end()) ==
              beforeForeign,
          "absent point connection produced partial CIRCT");

  mlir::Value wrongPayload =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(8, 0));
  const auto beforeMalformed =
      std::distance(top->getBody()->begin(), top->getBody()->end());
  expectErrorContains(
      __func__,
      adaptFabricPointConnectionForwardSignals(
          builder, location, view, narrowing, {valid, wrongPayload, tag3}),
      "source payload signal");
  require(__func__,
          std::distance(top->getBody()->begin(), top->getBody()->end()) ==
              beforeMalformed,
          "malformed point-connection input produced partial CIRCT");

  loom::fabric::FinalizedFabricRoot reimported =
      take(__func__,
           loom::fabric::importEntireFabricRoot(fabric.reference(), store));
  require(__func__,
          reimported.view().pointConnections() == view.pointConnections(),
          "strict reimport changed point-connection identity or order");
  const auto &reimportedNarrowing =
      findConnection(__func__, reimported.view(), 16, 3, 4, 7);
  ForwardTransportSignals repeated =
      take(__func__, adaptFabricPointConnectionForwardSignals(
                         builder, location, reimported.view(),
                         reimportedNarrowing, {valid, payload16, tag3}));
  require(__func__,
          repeated.payload->getDefiningOp<circt::comb::ExtractOp>() &&
              integerWidth(*repeated.payload) == 4 &&
              repeated.tag->getDefiningOp<circt::comb::ConcatOp>() &&
              integerWidth(*repeated.tag) == 7,
          "strict reimport changed point-connection lowering");
}

} // namespace

int main() {
  lowBitAlignmentAndZeroWidthPayload();
  malformedOrCrossKindAdaptationFails();
  exactFabricPointConnectionsOwnAdaptationTypes();
  return 0;
}
