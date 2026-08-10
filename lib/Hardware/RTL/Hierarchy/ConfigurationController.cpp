#include "ConfigurationController.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

constexpr std::uint64_t axiOkay = 0;
constexpr std::uint64_t axiSlaveError = 2;
constexpr std::uint64_t axiDecodeError = 3;

circt::hw::PortInfo port(mlir::OpBuilder &builder, llvm::StringRef name,
                         mlir::Type type,
                         circt::hw::ModulePort::Direction direction) {
  return circt::hw::PortInfo{{builder.getStringAttr(name), type, direction}};
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

mlir::Value notValue(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Value value) {
  return circt::comb::createOrFoldNot(builder, location, value);
}

mlir::Value mux(mlir::OpBuilder &builder, mlir::Location location,
                mlir::Value condition, mlir::Value whenTrue,
                mlir::Value whenFalse) {
  return circt::comb::MuxOp::create(builder, location, condition, whenTrue,
                                    whenFalse, true);
}

mlir::Value equals(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value value, std::uint64_t expected) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, value,
      constant(builder, location, width, expected), true);
}

mlir::Value extract(mlir::OpBuilder &builder, mlir::Location location,
                    mlir::Value value, std::uint64_t low, unsigned width) {
  return circt::comb::ExtractOp::create(builder, location, value, low, width);
}

struct ConfigurationByteState final {
  circt::Backedge shadowNext;
  circt::Backedge activeNext;
  circt::Backedge coveredNext;
  mlir::Value shadow;
  mlir::Value active;
  mlir::Value covered;
};

struct ConfigurationUnitState final {
  const ConfigurationTransportUnitLayout *layout = nullptr;
  std::vector<ConfigurationByteState> bytes;
  mlir::Value complete;
  mlir::Value commitSuccess;
};

mlir::Value assembleWord(mlir::OpBuilder &builder, mlir::Location location,
                         llvm::ArrayRef<ConfigurationByteState> bytes,
                         std::uint64_t firstByte) {
  llvm::SmallVector<mlir::Value, 4> highToLow;
  for (unsigned slot = 4; slot != 0; --slot) {
    const std::uint64_t byte = firstByte + slot - 1;
    highToLow.push_back(byte < bytes.size()
                            ? bytes[static_cast<std::size_t>(byte)].active
                            : constant(builder, location, 8, 0));
  }
  return circt::comb::ConcatOp::create(builder, location, highToLow);
}

mlir::Value assemblePayload(mlir::OpBuilder &builder, mlir::Location location,
                            const ConfigurationUnitState &unit) {
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(unit.bytes.size());
  for (auto iterator = unit.bytes.rbegin(); iterator != unit.bytes.rend();
       ++iterator)
    highToLow.push_back(iterator->active);
  mlir::Value bytes =
      highToLow.size() == 1
          ? highToLow.front()
          : circt::comb::ConcatOp::create(builder, location, highToLow);
  if (unit.layout->payloadBitCount == unit.layout->payloadByteCount * 8)
    return bytes;
  return extract(builder, location, bytes, 0,
                 static_cast<unsigned>(unit.layout->payloadBitCount));
}

} // namespace

void appendAxiLiteConfigurationPorts(
    mlir::OpBuilder &builder,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs) {
  const auto input = circt::hw::ModulePort::Direction::Input;
  const auto output = circt::hw::ModulePort::Direction::Output;
  inputs.push_back(port(builder, "cfg_awaddr", builder.getI32Type(), input));
  inputs.push_back(port(builder, "cfg_awvalid", builder.getI1Type(), input));
  outputs.push_back(port(builder, "cfg_awready", builder.getI1Type(), output));
  inputs.push_back(port(builder, "cfg_wdata", builder.getI32Type(), input));
  inputs.push_back(port(builder, "cfg_wstrb", builder.getI4Type(), input));
  inputs.push_back(port(builder, "cfg_wvalid", builder.getI1Type(), input));
  outputs.push_back(port(builder, "cfg_wready", builder.getI1Type(), output));
  outputs.push_back(port(builder, "cfg_bresp", builder.getI2Type(), output));
  outputs.push_back(port(builder, "cfg_bvalid", builder.getI1Type(), output));
  inputs.push_back(port(builder, "cfg_bready", builder.getI1Type(), input));
  inputs.push_back(port(builder, "cfg_araddr", builder.getI32Type(), input));
  inputs.push_back(port(builder, "cfg_arvalid", builder.getI1Type(), input));
  outputs.push_back(port(builder, "cfg_arready", builder.getI1Type(), output));
  outputs.push_back(port(builder, "cfg_rdata", builder.getI32Type(), output));
  outputs.push_back(port(builder, "cfg_rresp", builder.getI2Type(), output));
  outputs.push_back(port(builder, "cfg_rvalid", builder.getI1Type(), output));
  inputs.push_back(port(builder, "cfg_rready", builder.getI1Type(), input));
}

llvm::Expected<ConfigurationControllerModule>
buildConfigurationControllerModule(
    mlir::OpBuilder &builder, mlir::Location location,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    const ClockResetPlan &clockReset) {
  for (const ConfigurationTransportUnitLayout &layout : transportLayout.units) {
    const ProgrammingUnit *unit =
        configurationAbi.findProgrammingUnit(layout.programmingUnit.unitId);
    if (!unit || unit->payloadBitCount != layout.payloadBitCount ||
        layout.payloadBitCount > mlir::IntegerType::kMaxWidth ||
        layout.payloadByteCount != layout.inactiveImage.size())
      return invalid("configuration transport unit disagrees with its ABI");
  }

  llvm::SmallVector<circt::hw::PortInfo, 24> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 24> outputs;
  inputs.push_back(port(builder, "clock",
                        circt::seq::ClockType::get(builder.getContext()),
                        circt::hw::ModulePort::Direction::Input));
  inputs.push_back(port(builder, "reset", builder.getI1Type(),
                        circt::hw::ModulePort::Direction::Input));
  appendAxiLiteConfigurationPorts(builder, inputs, outputs);
  for (auto [ordinal, layout] : llvm::enumerate(transportLayout.units))
    outputs.push_back(port(
        builder, configurationPortName(ordinal),
        builder.getIntegerType(static_cast<unsigned>(layout.payloadBitCount)),
        circt::hw::ModulePort::Direction::Output));

  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_configuration_controller"),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        mlir::Value clock = accessor.getInput("clock");
        mlir::Value reset = accessor.getInput("reset");

        circt::Backedge awHeldNext = backedges.get(bodyBuilder.getI1Type());
        circt::Backedge awAddressNext = backedges.get(bodyBuilder.getI32Type());
        circt::Backedge wHeldNext = backedges.get(bodyBuilder.getI1Type());
        circt::Backedge wDataNext = backedges.get(bodyBuilder.getI32Type());
        circt::Backedge wStrobeNext = backedges.get(bodyBuilder.getI4Type());
        circt::Backedge bValidNext = backedges.get(bodyBuilder.getI1Type());
        circt::Backedge bResponseNext = backedges.get(bodyBuilder.getI2Type());
        circt::Backedge rValidNext = backedges.get(bodyBuilder.getI1Type());
        circt::Backedge rDataNext = backedges.get(bodyBuilder.getI32Type());
        circt::Backedge rResponseNext = backedges.get(bodyBuilder.getI2Type());

        mlir::Value awHeld = createRegister(
            bodyBuilder, location, awHeldNext, clock, reset, llvm::APInt(1, 0),
            "cfg_aw_held", clockReset.asynchronousReset);
        mlir::Value awAddress = createRegister(
            bodyBuilder, location, awAddressNext, clock, reset,
            llvm::APInt(32, 0), "cfg_aw_address", clockReset.asynchronousReset);
        mlir::Value wHeld = createRegister(
            bodyBuilder, location, wHeldNext, clock, reset, llvm::APInt(1, 0),
            "cfg_w_held", clockReset.asynchronousReset);
        mlir::Value wData = createRegister(
            bodyBuilder, location, wDataNext, clock, reset, llvm::APInt(32, 0),
            "cfg_w_data", clockReset.asynchronousReset);
        mlir::Value wStrobe = createRegister(
            bodyBuilder, location, wStrobeNext, clock, reset, llvm::APInt(4, 0),
            "cfg_w_strobe", clockReset.asynchronousReset);
        mlir::Value bValid = createRegister(
            bodyBuilder, location, bValidNext, clock, reset, llvm::APInt(1, 0),
            "cfg_b_valid", clockReset.asynchronousReset);
        mlir::Value bResponse = createRegister(
            bodyBuilder, location, bResponseNext, clock, reset,
            llvm::APInt(2, 0), "cfg_b_response", clockReset.asynchronousReset);
        mlir::Value rValid = createRegister(
            bodyBuilder, location, rValidNext, clock, reset, llvm::APInt(1, 0),
            "cfg_r_valid", clockReset.asynchronousReset);
        mlir::Value rData = createRegister(
            bodyBuilder, location, rDataNext, clock, reset, llvm::APInt(32, 0),
            "cfg_r_data", clockReset.asynchronousReset);
        mlir::Value rResponse = createRegister(
            bodyBuilder, location, rResponseNext, clock, reset,
            llvm::APInt(2, 0), "cfg_r_response", clockReset.asynchronousReset);

        mlir::Value awReady =
            andValues(bodyBuilder, location,
                      {notValue(bodyBuilder, location, awHeld),
                       notValue(bodyBuilder, location, bValid)});
        mlir::Value wReady =
            andValues(bodyBuilder, location,
                      {notValue(bodyBuilder, location, wHeld),
                       notValue(bodyBuilder, location, bValid)});
        mlir::Value awAccept = andValues(
            bodyBuilder, location, {awReady, accessor.getInput("cfg_awvalid")});
        mlir::Value wAccept = andValues(
            bodyBuilder, location, {wReady, accessor.getInput("cfg_wvalid")});
        mlir::Value executeWrite =
            andValues(bodyBuilder, location,
                      {awHeld, wHeld, notValue(bodyBuilder, location, bValid)});

        std::vector<ConfigurationUnitState> units;
        units.reserve(transportLayout.units.size());
        for (auto [unitOrdinal, layout] :
             llvm::enumerate(transportLayout.units)) {
          ConfigurationUnitState state;
          state.layout = &layout;
          state.bytes.reserve(
              static_cast<std::size_t>(layout.payloadByteCount));
          for (std::uint64_t byte = 0; byte < layout.payloadByteCount; ++byte) {
            ConfigurationByteState item;
            item.shadowNext = backedges.get(bodyBuilder.getI8Type());
            item.activeNext = backedges.get(bodyBuilder.getI8Type());
            item.coveredNext = backedges.get(bodyBuilder.getI1Type());
            const llvm::APInt resetByte(8, layout.inactiveImage[byte]);
            const std::string prefix = "cfg_unit_" +
                                       std::to_string(unitOrdinal) + "_byte_" +
                                       std::to_string(byte);
            item.shadow = createRegister(
                bodyBuilder, location, item.shadowNext, clock, reset, resetByte,
                prefix + "_shadow", clockReset.asynchronousReset);
            item.active = createRegister(
                bodyBuilder, location, item.activeNext, clock, reset, resetByte,
                prefix + "_active", clockReset.asynchronousReset);
            item.covered =
                createRegister(bodyBuilder, location, item.coveredNext, clock,
                               reset, llvm::APInt(1, 0), prefix + "_covered",
                               clockReset.asynchronousReset);
            state.bytes.push_back(std::move(item));
          }
          llvm::SmallVector<mlir::Value> covered;
          for (const ConfigurationByteState &byte : state.bytes)
            covered.push_back(byte.covered);
          state.complete = andValues(bodyBuilder, location, covered);
          units.push_back(std::move(state));
        }

        mlir::Value writeResponse =
            constant(bodyBuilder, location, 2, axiDecodeError);
        for (ConfigurationUnitState &unit : units) {
          mlir::Value commandByte = extract(bodyBuilder, location, wData, 0, 8);
          mlir::Value commandValid =
              andValues(bodyBuilder, location,
                        {extract(bodyBuilder, location, wStrobe, 0, 1),
                         equals(bodyBuilder, location, commandByte, 1)});
          for (unsigned byte = 1; byte != 4; ++byte) {
            mlir::Value ignored =
                notValue(bodyBuilder, location,
                         extract(bodyBuilder, location, wStrobe, byte, 1));
            mlir::Value zeroByte =
                equals(bodyBuilder, location,
                       extract(bodyBuilder, location, wData, byte * 8, 8), 0);
            commandValid =
                andValues(bodyBuilder, location,
                          {commandValid, orValues(bodyBuilder, location,
                                                  {ignored, zeroByte})});
          }
          mlir::Value commitMatch = equals(bodyBuilder, location, awAddress,
                                           unit.layout->commitAddress);
          mlir::Value commitValid =
              andValues(bodyBuilder, location, {commandValid, unit.complete});
          unit.commitSuccess = andValues(
              bodyBuilder, location, {executeWrite, commitMatch, commitValid});
          mlir::Value commitResponse =
              mux(bodyBuilder, location, commitValid,
                  constant(bodyBuilder, location, 2, axiOkay),
                  constant(bodyBuilder, location, 2, axiSlaveError));
          writeResponse = mux(bodyBuilder, location, commitMatch,
                              commitResponse, writeResponse);

          mlir::Value statusMatch = equals(bodyBuilder, location, awAddress,
                                           unit.layout->statusAddress);
          writeResponse = mux(bodyBuilder, location, statusMatch,
                              constant(bodyBuilder, location, 2, axiSlaveError),
                              writeResponse);

          for (std::uint64_t word = 0; word < unit.layout->payloadWordCount;
               ++word) {
            const std::uint64_t address = unit.layout->baseAddress + word * 4;
            mlir::Value addressMatch =
                equals(bodyBuilder, location, awAddress, address);
            mlir::Value unusedZero = bitConstant(bodyBuilder, location, true);
            for (unsigned slot = 0; slot != 4; ++slot) {
              const std::uint64_t byteIndex = word * 4 + slot;
              if (byteIndex + 1 < unit.layout->payloadByteCount)
                continue;
              const unsigned usedBits =
                  byteIndex >= unit.layout->payloadByteCount
                      ? 0
                      : static_cast<unsigned>(unit.layout->payloadBitCount -
                                              byteIndex * 8);
              if (usedBits == 8)
                continue;
              const std::uint64_t invalidMask =
                  usedBits == 0 ? 0xffU : (0xffU << usedBits) & 0xffU;
              mlir::Value selected =
                  extract(bodyBuilder, location, wStrobe, slot, 1);
              mlir::Value byteValue =
                  extract(bodyBuilder, location, wData, slot * 8, 8);
              mlir::Value masked = circt::comb::AndOp::create(
                  bodyBuilder, location, byteValue,
                  constant(bodyBuilder, location, 8, invalidMask));
              mlir::Value validByte =
                  orValues(bodyBuilder, location,
                           {notValue(bodyBuilder, location, selected),
                            equals(bodyBuilder, location, masked, 0)});
              unusedZero =
                  andValues(bodyBuilder, location, {unusedZero, validByte});
            }
            mlir::Value payloadResponse =
                mux(bodyBuilder, location, unusedZero,
                    constant(bodyBuilder, location, 2, axiOkay),
                    constant(bodyBuilder, location, 2, axiSlaveError));
            writeResponse = mux(bodyBuilder, location, addressMatch,
                                payloadResponse, writeResponse);

            for (unsigned slot = 0; slot != 4; ++slot) {
              const std::uint64_t byteIndex = word * 4 + slot;
              if (byteIndex >= unit.bytes.size())
                continue;
              ConfigurationByteState &byte =
                  unit.bytes[static_cast<std::size_t>(byteIndex)];
              mlir::Value writeByte =
                  andValues(bodyBuilder, location,
                            {executeWrite, addressMatch, unusedZero,
                             extract(bodyBuilder, location, wStrobe, slot, 1)});
              byte.shadowNext.setValue(
                  mux(bodyBuilder, location, writeByte,
                      extract(bodyBuilder, location, wData, slot * 8, 8),
                      byte.shadow));
              byte.activeNext.setValue(mux(bodyBuilder, location,
                                           unit.commitSuccess, byte.shadow,
                                           byte.active));
              byte.coveredNext.setValue(mux(
                  bodyBuilder, location, unit.commitSuccess,
                  bitConstant(bodyBuilder, location, false),
                  orValues(bodyBuilder, location, {byte.covered, writeByte})));
            }
          }
        }

        awHeldNext.setValue(
            mux(bodyBuilder, location, executeWrite,
                bitConstant(bodyBuilder, location, false),
                orValues(bodyBuilder, location, {awHeld, awAccept})));
        awAddressNext.setValue(mux(bodyBuilder, location, awAccept,
                                   accessor.getInput("cfg_awaddr"), awAddress));
        wHeldNext.setValue(
            mux(bodyBuilder, location, executeWrite,
                bitConstant(bodyBuilder, location, false),
                orValues(bodyBuilder, location, {wHeld, wAccept})));
        wDataNext.setValue(mux(bodyBuilder, location, wAccept,
                               accessor.getInput("cfg_wdata"), wData));
        wStrobeNext.setValue(mux(bodyBuilder, location, wAccept,
                                 accessor.getInput("cfg_wstrb"), wStrobe));
        mlir::Value bRetained =
            andValues(bodyBuilder, location,
                      {bValid, notValue(bodyBuilder, location,
                                        accessor.getInput("cfg_bready"))});
        bValidNext.setValue(
            orValues(bodyBuilder, location, {executeWrite, bRetained}));
        bResponseNext.setValue(
            mux(bodyBuilder, location, executeWrite, writeResponse, bResponse));

        mlir::Value readData = constant(bodyBuilder, location, 32, 0);
        mlir::Value readResponse =
            constant(bodyBuilder, location, 2, axiDecodeError);
        mlir::Value arAddress = accessor.getInput("cfg_araddr");
        for (const ConfigurationUnitState &unit : units) {
          mlir::Value statusMatch = equals(bodyBuilder, location, arAddress,
                                           unit.layout->statusAddress);
          mlir::Value status = circt::comb::ConcatOp::create(
              bodyBuilder, location,
              llvm::SmallVector<mlir::Value>{
                  constant(bodyBuilder, location, 31, 0), unit.complete});
          readData = mux(bodyBuilder, location, statusMatch, status, readData);
          readResponse =
              mux(bodyBuilder, location, statusMatch,
                  constant(bodyBuilder, location, 2, axiOkay), readResponse);
          for (std::uint64_t word = 0; word < unit.layout->payloadWordCount;
               ++word) {
            const std::uint64_t address = unit.layout->baseAddress + word * 4;
            mlir::Value addressMatch =
                equals(bodyBuilder, location, arAddress, address);
            readData =
                mux(bodyBuilder, location, addressMatch,
                    assembleWord(bodyBuilder, location, unit.bytes, word * 4),
                    readData);
            readResponse =
                mux(bodyBuilder, location, addressMatch,
                    constant(bodyBuilder, location, 2, axiOkay), readResponse);
          }
        }
        mlir::Value arReady = notValue(bodyBuilder, location, rValid);
        mlir::Value readAccept = andValues(
            bodyBuilder, location, {arReady, accessor.getInput("cfg_arvalid")});
        mlir::Value rRetained =
            andValues(bodyBuilder, location,
                      {rValid, notValue(bodyBuilder, location,
                                        accessor.getInput("cfg_rready"))});
        rValidNext.setValue(
            orValues(bodyBuilder, location, {readAccept, rRetained}));
        rDataNext.setValue(
            mux(bodyBuilder, location, readAccept, readData, rData));
        rResponseNext.setValue(
            mux(bodyBuilder, location, readAccept, readResponse, rResponse));

        accessor.setOutput("cfg_awready", awReady);
        accessor.setOutput("cfg_wready", wReady);
        accessor.setOutput("cfg_bresp", bResponse);
        accessor.setOutput("cfg_bvalid", bValid);
        accessor.setOutput("cfg_arready", arReady);
        accessor.setOutput("cfg_rdata", rData);
        accessor.setOutput("cfg_rresp", rResponse);
        accessor.setOutput("cfg_rvalid", rValid);
        for (auto [ordinal, unit] : llvm::enumerate(units))
          accessor.setOutput(configurationPortName(ordinal),
                             assemblePayload(bodyBuilder, location, unit));
      });
  if (materializationError)
    return invalid(*materializationError);
  return ConfigurationControllerModule{module};
}

} // namespace loom::hardware::rtl::hierarchy
