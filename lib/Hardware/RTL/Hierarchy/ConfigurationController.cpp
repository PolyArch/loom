#include "ConfigurationController.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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

mlir::Value createStructuredRegister(mlir::OpBuilder &builder,
                                     mlir::Location location, mlir::Value next,
                                     mlir::Value clock, mlir::Value reset,
                                     mlir::Value resetValue,
                                     llvm::StringRef name,
                                     bool asynchronousReset) {
  if (asynchronousReset)
    return circt::seq::FirRegOp::create(
        builder, location, next, clock, builder.getStringAttr(name), reset,
        resetValue, circt::hw::InnerSymAttr{}, true);
  return circt::seq::CompRegOp::create(builder, location, next, clock, reset,
                                       resetValue, name);
}

mlir::Value createUnresetStructuredRegister(mlir::OpBuilder &builder,
                                            mlir::Location location,
                                            mlir::Value next, mlir::Value clock,
                                            llvm::StringRef name) {
  return circt::seq::CompRegOp::create(builder, location, next, clock, name);
}

std::uint32_t inactiveWord(const ConfigurationTransportUnitLayout &layout,
                           std::uint64_t word) {
  std::uint32_t result = 0;
  for (unsigned byte = 0; byte != 4; ++byte) {
    const std::uint64_t ordinal = word * 4 + byte;
    if (ordinal < layout.inactiveImage.size())
      result |= std::uint32_t(layout.inactiveImage[ordinal]) << (byte * 8);
  }
  return result;
}

mlir::Value inactiveWordAt(mlir::OpBuilder &builder, mlir::Location location,
                           const ConfigurationTransportUnitLayout &layout,
                           mlir::Value wordIndex) {
  mlir::Value result = constant(builder, location, 32, 0);
  for (std::uint64_t word = 0; word != layout.payloadWordCount; ++word) {
    const std::uint32_t value = inactiveWord(layout, word);
    if (value == 0)
      continue;
    result = mux(builder, location, equals(builder, location, wordIndex, word),
                 constant(builder, location, 32, value), result);
  }
  return result;
}

mlir::Value zeroArrayConstant(mlir::OpBuilder &builder, mlir::Location location,
                              circt::hw::ArrayType arrayType) {
  return circt::sv::VerbatimExprOp::create(builder, location, arrayType, "'0");
}

mlir::Value strobeMask(mlir::OpBuilder &builder, mlir::Location location,
                       mlir::Value strobe) {
  llvm::SmallVector<mlir::Value, 4> highToLow;
  for (unsigned lane = 4; lane != 0; --lane)
    highToLow.push_back(mux(builder, location,
                            extract(builder, location, strobe, lane - 1, 1),
                            constant(builder, location, 8, 0xff),
                            constant(builder, location, 8, 0)));
  return circt::comb::ConcatOp::create(builder, location, highToLow);
}

mlir::Value adaptUnsignedWidth(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value value,
                               unsigned width) {
  const unsigned current =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  if (current == width)
    return value;
  if (current > width)
    return extract(builder, location, value, 0, width);
  return circt::comb::ConcatOp::create(
      builder, location,
      llvm::ArrayRef<mlir::Value>{
          constant(builder, location, width - current, 0), value});
}

mlir::Value populationCount4(mlir::OpBuilder &builder, mlir::Location location,
                             mlir::Value value, unsigned width) {
  mlir::Value result = constant(builder, location, width, 0);
  for (unsigned bit = 0; bit != 4; ++bit)
    result = circt::comb::AddOp::create(
        builder, location, result,
        adaptUnsignedWidth(builder, location,
                           extract(builder, location, value, bit, 1), width),
        true);
  return result;
}

struct ConfigurationUnitState final {
  const ConfigurationTransportUnitLayout *layout = nullptr;
  circt::hw::ArrayType wordArrayType;
  circt::hw::ArrayType tagArrayType;
  circt::Backedge bankZeroNext;
  circt::Backedge bankOneNext;
  circt::Backedge tagsNext;
  circt::Backedge activeBankNext;
  circt::Backedge initializedNext;
  circt::Backedge coveredCountNext;
  mlir::Value bankZero;
  mlir::Value bankOne;
  mlir::Value tags;
  mlir::Value activeBank;
  mlir::Value initialized;
  mlir::Value coveredCount;
  std::vector<mlir::Value> activeWords;
  mlir::Value complete;
  mlir::Value commitSuccess;
};

mlir::Value activeWordAt(mlir::OpBuilder &builder, mlir::Location location,
                         const ConfigurationUnitState &unit,
                         mlir::Value wordIndex) {
  mlir::Value bankZeroWord = circt::hw::ArrayGetOp::create(
      builder, location, unit.bankZero, wordIndex);
  mlir::Value bankOneWord = circt::hw::ArrayGetOp::create(
      builder, location, unit.bankOne, wordIndex);
  return mux(builder, location, unit.activeBank, bankOneWord, bankZeroWord);
}

mlir::Value assembleBundle(mlir::OpBuilder &builder, mlir::Location location,
                           llvm::ArrayRef<ConfigurationUnitState> units,
                           const ConfigurationBundlePlan &configuration) {
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(configuration.words.size());
  for (const ConfigurationBundleWord &word :
       llvm::reverse(configuration.words)) {
    assert(word.key.transportUnitOrdinal < units.size() &&
           "configuration bundle references an absent transport unit");
    const ConfigurationUnitState &unit =
        units[word.key.transportUnitOrdinal];
    assert(word.key.wordOrdinal < unit.activeWords.size() &&
           "configuration bundle references an absent active word");
    mlir::Value selected = unit.activeWords[word.key.wordOrdinal];
    if (word.usedBitMask != std::numeric_limits<std::uint32_t>::max())
      selected = circt::comb::AndOp::create(
          builder, location, selected,
          constant(builder, location, 32, word.usedBitMask), true);
    highToLow.push_back(selected);
  }
  assert(!highToLow.empty() && "empty configuration bundle has no value");
  return circt::hw::ArrayCreateOp::create(builder, location, highToLow);
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

std::string
controllerConfigurationBundlePortName(std::size_t componentOrdinal) {
  return "configuration_bundle_" + std::to_string(componentOrdinal);
}

llvm::Expected<ConfigurationControllerModule>
buildConfigurationControllerModule(
    mlir::OpBuilder &builder, mlir::Location location,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    const ClockResetPlan &clockReset,
    llvm::ArrayRef<ConfigurationBundlePlan> componentConfigurations) {
  for (const ConfigurationTransportUnitLayout &layout : transportLayout.units) {
    const ProgrammingUnit *unit =
        configurationAbi.findProgrammingUnit(layout.programmingUnit.unitId);
    if (!unit || unit->payloadBitCount != layout.payloadBitCount ||
        layout.payloadBitCount == 0 ||
        layout.payloadBitCount > mlir::IntegerType::kMaxWidth ||
        layout.payloadByteCount != layout.inactiveImage.size() ||
        layout.payloadWordCount == 0 ||
        layout.payloadWordCount > mlir::IntegerType::kMaxWidth / 32)
      return invalid("configuration transport unit disagrees with its ABI");
  }
  auto fieldDecoders = prepareFieldDecoders(configurationAbi, transportLayout);
  if (!fieldDecoders)
    return fieldDecoders.takeError();
  auto allFields = deriveConfigurationBundlePlan(*fieldDecoders);
  if (!allFields)
    return allFields.takeError();
  for (const ConfigurationBundlePlan &configuration :
       componentConfigurations)
    for (const ConfigurationBundleWord &word : configuration.words) {
      const ConfigurationBundleWord *canonical = allFields->find(word.key);
      if (!canonical ||
          (canonical->usedBitMask & word.usedBitMask) != word.usedBitMask)
        return invalid(
            "component configuration bundle disagrees with its ABI word");
    }

  llvm::SmallVector<circt::hw::PortInfo, 24> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 24> outputs;
  inputs.push_back(port(builder, "clock",
                        circt::seq::ClockType::get(builder.getContext()),
                        circt::hw::ModulePort::Direction::Input));
  inputs.push_back(port(builder, "reset", builder.getI1Type(),
                        circt::hw::ModulePort::Direction::Input));
  appendAxiLiteConfigurationPorts(builder, inputs, outputs);
  for (auto [ordinal, configuration] :
       llvm::enumerate(componentConfigurations))
    if (!configuration.empty())
      outputs.push_back(port(
          builder, controllerConfigurationBundlePortName(ordinal),
          configurationBundleType(builder.getContext(), configuration),
          circt::hw::ModulePort::Direction::Output));

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
                        {commandValid,
                         orValues(bodyBuilder, location, {ignored, zeroByte})});
        }
        mlir::Value byteWriteMask = strobeMask(bodyBuilder, location, wStrobe);

        std::vector<ConfigurationUnitState> units;
        units.reserve(transportLayout.units.size());
        for (auto [unitOrdinal, layout] :
             llvm::enumerate(transportLayout.units)) {
          ConfigurationUnitState state;
          state.layout = &layout;
          state.wordArrayType = circt::hw::ArrayType::get(
              bodyBuilder.getI32Type(), layout.payloadWordCount);
          state.tagArrayType = circt::hw::ArrayType::get(
              bodyBuilder.getI4Type(), layout.payloadWordCount);
          state.bankZeroNext = backedges.get(state.wordArrayType);
          state.bankOneNext = backedges.get(state.wordArrayType);
          state.tagsNext = backedges.get(state.tagArrayType);
          state.activeBankNext = backedges.get(bodyBuilder.getI1Type());
          state.initializedNext = backedges.get(bodyBuilder.getI1Type());
          const unsigned coveredCountWidth =
              indexWidth(layout.payloadByteCount + 1);
          state.coveredCountNext =
              backedges.get(bodyBuilder.getIntegerType(coveredCountWidth));

          mlir::Value emptyTags =
              zeroArrayConstant(bodyBuilder, location, state.tagArrayType);
          const std::string prefix = "cfg_unit_" + std::to_string(unitOrdinal);
          state.bankZero = createUnresetStructuredRegister(
              bodyBuilder, location, state.bankZeroNext, clock,
              prefix + "_bank_0");
          state.bankOne = createUnresetStructuredRegister(
              bodyBuilder, location, state.bankOneNext, clock,
              prefix + "_bank_1");
          state.tags = createStructuredRegister(
              bodyBuilder, location, state.tagsNext, clock, reset, emptyTags,
              prefix + "_coverage_tags", clockReset.asynchronousReset);
          state.activeBank =
              createRegister(bodyBuilder, location, state.activeBankNext, clock,
                             reset, llvm::APInt(1, 0), prefix + "_active_bank",
                             clockReset.asynchronousReset);
          state.initialized = createRegister(
              bodyBuilder, location, state.initializedNext, clock, reset,
              llvm::APInt(1, 0), prefix + "_initialized",
              clockReset.asynchronousReset);
          state.coveredCount = createRegister(
              bodyBuilder, location, state.coveredCountNext, clock, reset,
              llvm::APInt(coveredCountWidth, 0), prefix + "_covered_count",
              clockReset.asynchronousReset);
          state.activeWords.reserve(layout.payloadWordCount);
          const unsigned wordIndexWidth = indexWidth(layout.payloadWordCount);
          for (std::uint64_t word = 0; word != layout.payloadWordCount;
               ++word) {
            mlir::Value storedWord = activeWordAt(
                bodyBuilder, location, state,
                constant(bodyBuilder, location, wordIndexWidth, word));
            state.activeWords.push_back(mux(
                bodyBuilder, location, state.initialized, storedWord,
                constant(bodyBuilder, location, 32,
                         inactiveWord(layout, word))));
          }
          state.complete = equals(bodyBuilder, location, state.coveredCount,
                                  layout.payloadByteCount);
          units.push_back(std::move(state));
        }

        mlir::Value writeResponse =
            constant(bodyBuilder, location, 2, axiDecodeError);
        for (ConfigurationUnitState &unit : units) {
          const auto &layout = *unit.layout;
          mlir::Value commitMatch =
              equals(bodyBuilder, location, awAddress, layout.commitAddress);
          mlir::Value commitValid =
              andValues(bodyBuilder, location, {commandValid, unit.complete});
          unit.commitSuccess = andValues(
              bodyBuilder, location, {executeWrite, commitMatch, commitValid});
          writeResponse =
              mux(bodyBuilder, location, commitMatch,
                  mux(bodyBuilder, location, commitValid,
                      constant(bodyBuilder, location, 2, axiOkay),
                      constant(bodyBuilder, location, 2, axiSlaveError)),
                  writeResponse);

          mlir::Value statusMatch =
              equals(bodyBuilder, location, awAddress, layout.statusAddress);
          writeResponse = mux(bodyBuilder, location, statusMatch,
                              constant(bodyBuilder, location, 2, axiSlaveError),
                              writeResponse);

          mlir::Value atOrAboveBase = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::uge, awAddress,
              constant(bodyBuilder, location, 32, layout.baseAddress), true);
          mlir::Value belowCommit = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ult, awAddress,
              constant(bodyBuilder, location, 32, layout.commitAddress), true);
          mlir::Value aligned =
              equals(bodyBuilder, location,
                     extract(bodyBuilder, location, awAddress, 0, 2), 0);
          mlir::Value payloadMatch = andValues(
              bodyBuilder, location, {atOrAboveBase, belowCommit, aligned});
          mlir::Value byteOffset = circt::comb::SubOp::create(
              bodyBuilder, location, awAddress,
              constant(bodyBuilder, location, 32, layout.baseAddress), true);
          const unsigned wordIndexWidth = indexWidth(layout.payloadWordCount);
          mlir::Value wordIndex =
              extract(bodyBuilder, location, byteOffset, 2, wordIndexWidth);
          mlir::Value safeWordIndex =
              mux(bodyBuilder, location, payloadMatch, wordIndex,
                  constant(bodyBuilder, location, wordIndexWidth, 0));
          mlir::Value lastWord = equals(bodyBuilder, location, safeWordIndex,
                                        layout.payloadWordCount - 1);

          const unsigned usedLastWordBits =
              static_cast<unsigned>(layout.payloadBitCount % 32);
          const std::uint32_t invalidLastWordMask =
              usedLastWordBits == 0
                  ? 0
                  : static_cast<std::uint32_t>(
                        ~((std::uint64_t{1} << usedLastWordBits) - 1));
          mlir::Value invalidMask =
              mux(bodyBuilder, location, lastWord,
                  constant(bodyBuilder, location, 32, invalidLastWordMask),
                  constant(bodyBuilder, location, 32, 0));
          mlir::Value invalidSelected = circt::comb::AndOp::create(
              bodyBuilder, location, wData, byteWriteMask, true);
          invalidSelected = circt::comb::AndOp::create(
              bodyBuilder, location, invalidSelected, invalidMask, true);
          mlir::Value unusedZero =
              equals(bodyBuilder, location, invalidSelected, 0);
          mlir::Value payloadResponse =
              mux(bodyBuilder, location, unusedZero,
                  constant(bodyBuilder, location, 2, axiOkay),
                  constant(bodyBuilder, location, 2, axiSlaveError));
          writeResponse = mux(bodyBuilder, location, payloadMatch,
                              payloadResponse, writeResponse);
          mlir::Value payloadWrite = andValues(
              bodyBuilder, location, {executeWrite, payloadMatch, unusedZero});

          mlir::Value inactiveWords =
              mux(bodyBuilder, location, unit.activeBank, unit.bankZero,
                  unit.bankOne);
          mlir::Value oldWord = circt::hw::ArrayGetOp::create(
              bodyBuilder, location, inactiveWords, safeWordIndex);
          mlir::Value retainedWord = circt::comb::AndOp::create(
              bodyBuilder, location, oldWord,
              notValue(bodyBuilder, location, byteWriteMask), true);
          mlir::Value selectedWord = circt::comb::AndOp::create(
              bodyBuilder, location, wData, byteWriteMask, true);
          mlir::Value updatedWord = circt::comb::OrOp::create(
              bodyBuilder, location, retainedWord, selectedWord, true);
          updatedWord = circt::comb::AndOp::create(
              bodyBuilder, location, updatedWord,
              notValue(bodyBuilder, location, invalidMask), true);
          mlir::Value updatedInactive = circt::hw::ArrayInjectOp::create(
              bodyBuilder, location, inactiveWords, safeWordIndex, updatedWord);
          mlir::Value writeBankZero =
              andValues(bodyBuilder, location, {payloadWrite, unit.activeBank});
          mlir::Value writeBankOne = andValues(
              bodyBuilder, location,
              {payloadWrite, notValue(bodyBuilder, location, unit.activeBank)});
          unit.bankZeroNext.setValue(mux(bodyBuilder, location, writeBankZero,
                                         updatedInactive, unit.bankZero));
          unit.bankOneNext.setValue(mux(bodyBuilder, location, writeBankOne,
                                        updatedInactive, unit.bankOne));

          const unsigned usedLastWordBytes =
              static_cast<unsigned>(layout.payloadByteCount % 4);
          const std::uint64_t lastLaneMask =
              usedLastWordBytes == 0 ? 0xf : (1U << usedLastWordBytes) - 1;
          mlir::Value validLanes =
              mux(bodyBuilder, location, lastWord,
                  constant(bodyBuilder, location, 4, lastLaneMask),
                  constant(bodyBuilder, location, 4, 0xf));
          mlir::Value selectedLanes = circt::comb::AndOp::create(
              bodyBuilder, location, wStrobe, validLanes, true);
          mlir::Value oldTags = circt::hw::ArrayGetOp::create(
              bodyBuilder, location, unit.tags, safeWordIndex);
          mlir::Value generation =
              notValue(bodyBuilder, location, unit.activeBank);
          mlir::Value coveredLanes =
              mux(bodyBuilder, location, generation, oldTags,
                  notValue(bodyBuilder, location, oldTags));
          mlir::Value newlyCovered = circt::comb::AndOp::create(
              bodyBuilder, location, selectedLanes,
              notValue(bodyBuilder, location, coveredLanes), true);
          mlir::Value tagsSet = circt::comb::OrOp::create(
              bodyBuilder, location, oldTags, selectedLanes, true);
          mlir::Value tagsCleared = circt::comb::AndOp::create(
              bodyBuilder, location, oldTags,
              notValue(bodyBuilder, location, selectedLanes), true);
          mlir::Value updatedTags =
              mux(bodyBuilder, location, generation, tagsSet, tagsCleared);
          mlir::Value injectedTags = circt::hw::ArrayInjectOp::create(
              bodyBuilder, location, unit.tags, safeWordIndex, updatedTags);
          unit.tagsNext.setValue(mux(bodyBuilder, location, payloadWrite,
                                     injectedTags, unit.tags));

          const unsigned countWidth =
              mlir::cast<mlir::IntegerType>(unit.coveredCount.getType())
                  .getWidth();
          mlir::Value countAfterWrite = circt::comb::AddOp::create(
              bodyBuilder, location, unit.coveredCount,
              populationCount4(bodyBuilder, location, newlyCovered, countWidth),
              true);
          countAfterWrite = mux(bodyBuilder, location, payloadWrite,
                                countAfterWrite, unit.coveredCount);
          unit.coveredCountNext.setValue(mux(
              bodyBuilder, location, unit.commitSuccess,
              constant(bodyBuilder, location, countWidth, 0), countAfterWrite));
          unit.activeBankNext.setValue(
              mux(bodyBuilder, location, unit.commitSuccess,
                  notValue(bodyBuilder, location, unit.activeBank),
                  unit.activeBank));
          unit.initializedNext.setValue(orValues(
              bodyBuilder, location, {unit.initialized, unit.commitSuccess}));
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
          const auto &layout = *unit.layout;
          mlir::Value statusMatch =
              equals(bodyBuilder, location, arAddress, layout.statusAddress);
          mlir::Value status = circt::comb::ConcatOp::create(
              bodyBuilder, location,
              llvm::SmallVector<mlir::Value>{
                  constant(bodyBuilder, location, 31, 0), unit.complete});
          readData = mux(bodyBuilder, location, statusMatch, status, readData);
          readResponse =
              mux(bodyBuilder, location, statusMatch,
                  constant(bodyBuilder, location, 2, axiOkay), readResponse);

          mlir::Value atOrAboveBase = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::uge, arAddress,
              constant(bodyBuilder, location, 32, layout.baseAddress), true);
          mlir::Value belowCommit = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ult, arAddress,
              constant(bodyBuilder, location, 32, layout.commitAddress), true);
          mlir::Value aligned =
              equals(bodyBuilder, location,
                     extract(bodyBuilder, location, arAddress, 0, 2), 0);
          mlir::Value payloadMatch = andValues(
              bodyBuilder, location, {atOrAboveBase, belowCommit, aligned});
          mlir::Value byteOffset = circt::comb::SubOp::create(
              bodyBuilder, location, arAddress,
              constant(bodyBuilder, location, 32, layout.baseAddress), true);
          mlir::Value wordIndex = extract(bodyBuilder, location, byteOffset, 2,
                                          indexWidth(layout.payloadWordCount));
          mlir::Value safeWordIndex =
              mux(bodyBuilder, location, payloadMatch, wordIndex,
                  constant(bodyBuilder, location,
                           indexWidth(layout.payloadWordCount), 0));
          mlir::Value storedWord =
              activeWordAt(bodyBuilder, location, unit, safeWordIndex);
          mlir::Value word =
              mux(bodyBuilder, location, unit.initialized, storedWord,
                  inactiveWordAt(bodyBuilder, location, layout, safeWordIndex));
          readData = mux(bodyBuilder, location, payloadMatch, word, readData);
          readResponse =
              mux(bodyBuilder, location, payloadMatch,
                  constant(bodyBuilder, location, 2, axiOkay), readResponse);
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
        for (auto [ordinal, configuration] :
             llvm::enumerate(componentConfigurations))
          if (!configuration.empty())
            accessor.setOutput(
                controllerConfigurationBundlePortName(ordinal),
                assembleBundle(bodyBuilder, location, units, configuration));
      });
  return ConfigurationControllerModule{module};
}

} // namespace loom::hardware::rtl::hierarchy
