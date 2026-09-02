#include "Support.h"

#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Hardware/RTL/CommonSkeleton.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_module_hierarchy_invalid: " + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::make_error<FabricStructuralLoweringUnsupportedError>(
      message.str());
}

std::string endpointKey(const fabric::FabricTransportEndpointRef &endpoint) {
  const std::vector<std::uint8_t> bytes =
      fabric::canonicalFabricBytes(endpoint);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

ConfigurationFieldKey configurationFieldKey(const FieldDecoderPlan &decoder) {
  return {decoder.transportUnitOrdinal, decoder.fieldOrdinal};
}

const ConfigurationBundleWord *
ConfigurationBundlePlan::find(ConfigurationWordKey key) const {
  const auto found = llvm::lower_bound(
      words, key, [](const ConfigurationBundleWord &word,
                     ConfigurationWordKey candidate) {
        return word.key < candidate;
      });
  return found != words.end() && found->key == key ? &*found : nullptr;
}

mlir::Type configurationBundleType(mlir::MLIRContext *context,
                                   const ConfigurationBundlePlan &plan) {
  assert(!plan.empty() && "empty configuration bundle has no type");
  return circt::hw::ArrayType::get(mlir::IntegerType::get(context, 32),
                                   plan.words.size());
}

llvm::Error verifyConfigurationBundlePort(
    circt::hw::HWModuleOp module, const ConfigurationBundlePlan &plan) {
  const auto ports = module.getPortList();
  const auto found = llvm::find_if(ports, [](const circt::hw::PortInfo &port) {
    return port.isInput() && port.getName() == configurationBundlePortName;
  });
  if (plan.empty())
    return found == ports.end()
               ? llvm::Error::success()
               : invalid("configuration-free module has a bundle port");
  if (found == ports.end())
    return invalid("configured module has no bundle port");
  if (found->type != configurationBundleType(module.getContext(), plan))
    return invalid("configured module bundle port type disagrees with its "
                   "occurrence plan");
  return llvm::Error::success();
}

llvm::Error verifyConfigurationValuePort(circt::hw::HWModuleOp module,
                                         const FieldDecoderPlan &decoder) {
  const auto ports = module.getPortList();
  const auto found = llvm::find_if(ports, [](const circt::hw::PortInfo &port) {
    return port.isInput() && port.getName() == configurationValuePortName;
  });
  if (found == ports.end())
    return invalid("decoded configuration module has no value port");
  const mlir::Type expected =
      mlir::IntegerType::get(module.getContext(), decoder.encodedBitCount);
  if (found->type != expected)
    return invalid("decoded configuration value port has the wrong type");
  return llvm::Error::success();
}

llvm::Expected<ConfigurationBundlePlan> deriveConfigurationBundlePlan(
    llvm::ArrayRef<FieldDecoderPlan> decoders,
    llvm::ArrayRef<ConfigurationBundlePlan> childBundles) {
  std::map<ConfigurationFieldKey, FieldDecoderPlan> fields;
  std::map<ConfigurationWordKey, std::uint32_t> words;
  const auto appendDecoder = [&](const FieldDecoderPlan &decoder)
      -> llvm::Error {
    if (decoder.encodedBitCount == 0 ||
        decoder.encodedBitCount > mlir::IntegerType::kMaxWidth)
      return unsupported(
          "configuration field width exceeds the CIRCT support envelope");
    auto [position, inserted] =
        fields.emplace(configurationFieldKey(decoder), decoder);
    if (!inserted &&
        (position->second.encodedBitCount != decoder.encodedBitCount ||
         position->second.destinationSlices != decoder.destinationSlices))
      return invalid("configuration field has inconsistent derived layouts");
    for (const DestinationSlice &slice : decoder.destinationSlices) {
      std::uint64_t bit = slice.destinationBitOffset;
      std::uint64_t remaining = slice.bitCount;
      while (remaining != 0) {
        const std::uint64_t wordOrdinal = bit / 32;
        const unsigned bitInWord = static_cast<unsigned>(bit % 32);
        const unsigned width = static_cast<unsigned>(
            std::min<std::uint64_t>(remaining, 32 - bitInWord));
        const std::uint32_t mask =
            width == 32
                ? std::numeric_limits<std::uint32_t>::max()
                : static_cast<std::uint32_t>(((std::uint64_t{1} << width) - 1)
                                             << bitInWord);
        words[{decoder.transportUnitOrdinal, wordOrdinal}] |= mask;
        bit += width;
        remaining -= width;
      }
    }
    return llvm::Error::success();
  };
  for (const FieldDecoderPlan &decoder : decoders)
    if (llvm::Error error = appendDecoder(decoder))
      return std::move(error);
  for (const ConfigurationBundlePlan &child : childBundles)
    for (const ConfigurationBundleWord &word : child.words)
      words[word.key] |= word.usedBitMask;

  ConfigurationBundlePlan result;
  result.words.reserve(words.size());
  for (const auto &[key, mask] : words) {
    if (mask == 0)
      return invalid("configuration bundle word has an empty used-bit mask");
    result.words.push_back(ConfigurationBundleWord{key, mask});
  }
  return result;
}

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyConfigurationField(fabric::SpatialCoreOccurrenceRef spatialCore,
                          const fabric::FabricSemanticConfigFieldRef &field) {
  auto target = fabric::FabricModulePhysicalTargetRef::create(field);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalConfigurationFieldRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

namespace {

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(const ConfigurationFieldEncoding &encoding,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout) {
  const ProgrammingUnit *owner = nullptr;
  std::size_t fieldOrdinal = 0;
  for (const ProgrammingUnit &unit : configurationAbi.programmingUnits())
    for (auto [ordinal, candidate] : llvm::enumerate(unit.fields))
      if (&candidate == &encoding) {
        if (owner)
          return invalid(
              "configuration field has duplicate programming owners");
        owner = &unit;
        fieldOrdinal = ordinal;
      }
  if (!owner)
    return invalid("configuration field has no programming owner");
  const ConfigurationTransportUnitLayout *transportUnit =
      transportLayout.find(owner->id);
  if (!transportUnit)
    return invalid("configuration field owner is absent from the local "
                   "configuration transport");
  const std::size_t transportUnitOrdinal =
      static_cast<std::size_t>(transportUnit - transportLayout.units.data());

  const ConfigurationEncodingRelation *relation =
      configurationAbi.findEncodingRelation(encoding);
  if (!relation)
    return invalid("configuration field names an unknown encoding relation");
  const std::uint64_t width = relation->encodedBitCount();
  if (width == 0 || width > mlir::IntegerType::kMaxWidth)
    return unsupported(
        "configuration field width exceeds the CIRCT support envelope");
  std::vector<DestinationSlice> destinationSlices = encoding.destinationSlices;
  llvm::sort(destinationSlices,
             [](const DestinationSlice &lhs, const DestinationSlice &rhs) {
               return std::tie(lhs.sourceBitOffset, lhs.destinationBitOffset,
                               lhs.bitCount) <
                      std::tie(rhs.sourceBitOffset, rhs.destinationBitOffset,
                               rhs.bitCount);
             });
  std::uint64_t sourceCursor = 0;
  for (const DestinationSlice &slice : destinationSlices) {
    if (slice.sourceBitOffset > width || slice.bitCount > width ||
        slice.sourceBitOffset + slice.bitCount > width ||
        slice.destinationBitOffset > owner->payloadBitCount ||
        slice.bitCount > owner->payloadBitCount ||
        slice.destinationBitOffset + slice.bitCount > owner->payloadBitCount)
      return invalid("configuration destination slice is out of range");
    if (slice.bitCount == 0 || slice.sourceBitOffset != sourceCursor)
      return invalid(
          "configuration destination slices do not exactly partition the "
          "field source");
    sourceCursor += slice.bitCount;
  }
  if (sourceCursor != width)
    return invalid("configuration destination slices do not cover the field");
  return FieldDecoderPlan{owner, transportUnitOrdinal, fieldOrdinal, width,
                          std::move(destinationSlices)};
}

} // namespace

llvm::Expected<std::vector<FieldDecoderPlan>>
prepareFieldDecoders(const ConfigurationABI &configurationAbi,
                     const ConfigurationTransportLayout &transportLayout) {
  std::vector<FieldDecoderPlan> result;
  for (const ConfigurationTransportUnitLayout &transportUnit :
       transportLayout.units) {
    const ProgrammingUnit *unit = configurationAbi.findProgrammingUnit(
        transportUnit.programmingUnit.unitId);
    if (!unit)
      return invalid("configuration transport names an absent ABI unit");
    result.reserve(result.size() + unit->fields.size());
    for (const ConfigurationFieldEncoding &field : unit->fields) {
      auto decoder =
          prepareFieldDecoder(field, configurationAbi, transportLayout);
      if (!decoder)
        return decoder.takeError();
      result.push_back(std::move(*decoder));
    }
  }
  return result;
}

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout) {
  return prepareFieldDecoder(spatialCore, field,
                             fabric::FabricStaticConfigurationResidency{},
                             configurationAbi, transportLayout);
}

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const fabric::FabricConfigurationResidency &residency,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout) {
  auto physical = qualifyConfigurationField(spatialCore, field);
  if (!physical)
    return physical.takeError();
  auto slot = fabric::qualifyFabricConfigurationSlot(*physical, residency);
  if (!slot)
    return slot.takeError();
  const ConfigurationFieldEncoding *encoding =
      configurationAbi.findField(*slot);
  if (!encoding)
    return invalid("configuration field is absent from ConfigurationABI: " +
                   fabric::printFabricRef(*physical));
  return prepareFieldDecoder(*encoding, configurationAbi, transportLayout);
}

llvm::Expected<std::pair<FieldDecoderPlan, const FiniteCodebookEncoding *>>
prepareFiniteField(fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricSemanticConfigFieldRef &field,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout) {
  return prepareFiniteField(spatialCore, field,
                            fabric::FabricStaticConfigurationResidency{},
                            configurationAbi, transportLayout);
}

llvm::Expected<std::pair<FieldDecoderPlan, const FiniteCodebookEncoding *>>
prepareFiniteField(fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricSemanticConfigFieldRef &field,
                   const fabric::FabricConfigurationResidency &residency,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout) {
  auto decoder = prepareFieldDecoder(spatialCore, field, residency,
                                     configurationAbi, transportLayout);
  if (!decoder)
    return decoder.takeError();
  auto physical = qualifyConfigurationField(spatialCore, field);
  if (!physical)
    return physical.takeError();
  auto slot = fabric::qualifyFabricConfigurationSlot(*physical, residency);
  if (!slot)
    return slot.takeError();
  const ConfigurationFieldEncoding *encoding =
      configurationAbi.findField(*slot);
  if (!encoding)
    return invalid("configuration field is absent from ConfigurationABI: " +
                   fabric::printFabricRef(*physical));
  const ConfigurationEncodingRelation *relation =
      configurationAbi.findEncodingRelation(*encoding);
  if (!relation)
    return invalid("configuration field names an unknown encoding relation");
  const auto *codebook =
      std::get_if<FiniteCodebookEncoding>(&relation->semanticEncoding);
  if (!codebook || codebook->encodedBitCount != decoder->encodedBitCount)
    return unsupported("configuration field requires a finite ABI codebook");
  return std::make_pair(std::move(*decoder), codebook);
}

llvm::Expected<llvm::APInt>
physicalCode(const FiniteCodebookEncoding &codebook,
             llvm::ArrayRef<std::uint8_t> semanticValue) {
  const FiniteCodebookEntry *entry = nullptr;
  for (const FiniteCodebookEntry &candidate : codebook.entries)
    if (llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
            .equals(semanticValue)) {
      if (entry)
        return invalid("finite codebook repeats one semantic value");
      entry = &candidate;
    }
  if (!entry)
    return invalid("finite codebook omits one semantic value");
  if (entry->physicalCode.size() <
      (codebook.encodedBitCount + std::uint64_t(7)) / std::uint64_t(8))
    return invalid("finite codebook physical code is truncated");
  llvm::APInt result(static_cast<unsigned>(codebook.encodedBitCount), 0);
  for (std::uint64_t bit = 0; bit < codebook.encodedBitCount; ++bit)
    if (((entry->physicalCode[static_cast<std::size_t>(bit / 8)] >> (bit % 8)) &
         1U) != 0)
      result.setBit(static_cast<unsigned>(bit));
  return result;
}

llvm::Expected<ClockResetPlan>
prepareClockReset(const fabric::FabricSystemRootView &system,
                  fabric::SpatialCoreOccurrenceRef spatialCore) {
  auto clockReference = system.effectiveHardwareDomain(
      spatialCore, fabric::FabricClockResetKind::Clock);
  if (!clockReference)
    return unsupported("hierarchy lowering requires exact Clock and Reset "
                       "domains");
  auto resetReference = system.effectiveHardwareDomain(
      spatialCore, fabric::FabricClockResetKind::Reset);
  if (!resetReference)
    return unsupported("hierarchy lowering requires exact Clock and Reset "
                       "domains");
  const fabric::HardwareDomainContractRecord *clock =
      system.hardwareDomainContract(*clockReference);
  const fabric::HardwareDomainContractRecord *reset =
      system.hardwareDomainContract(*resetReference);
  if (!clock || !reset)
    return invalid("effective Clock or Reset domain disappeared");
  if (!std::get_if<fabric::ClockDomainContractRecord>(&clock->contract()))
    return invalid("Clock domain carries a non-Clock contract");
  const auto *resetContract =
      std::get_if<fabric::ResetDomainContractRecord>(&reset->contract());
  if (!resetContract)
    return invalid("Reset domain carries a non-Reset contract");
  if (resetContract->initialState() != fabric::ResetInitialState::Asserted)
    return unsupported("hierarchy lowering requires initially asserted Reset");
  if (resetContract->releaseLatencyCycles() != 0)
    return unsupported("reset release latency requires a synchronizer");
  const bool asynchronous =
      resetContract->assertion() == fabric::ResetTiming::Asynchronous &&
      resetContract->deassertion() == fabric::ResetTiming::Asynchronous;
  const bool synchronous =
      resetContract->assertion() == fabric::ResetTiming::Synchronous &&
      resetContract->deassertion() == fabric::ResetTiming::Synchronous;
  if (!asynchronous && !synchronous)
    return unsupported("mixed Reset timing is unsupported");
  if (synchronous &&
      resetContract->synchronousTo() != fabric::ClockDomainRef(*clockReference))
    return invalid("synchronous Reset names a different Clock domain");
  return ClockResetPlan{asynchronous, resetContract->polarity() ==
                                          fabric::ResetPolarity::ActiveLow};
}

llvm::Expected<std::vector<EndpointPlan>>
deriveEndpointPlans(mlir::OpBuilder &builder,
                    const fabric::FabricArtifactView &fabric,
                    const fabric::FabricTransportEndpointOwnerRef &owner) {
  const std::uint64_t count = fabric.transportEndpointCount(owner);
  std::vector<EndpointPlan> result;
  result.reserve(static_cast<std::size_t>(count));
  std::uint64_t inputOrdinal = 0;
  std::uint64_t outputOrdinal = 0;
  for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal) {
    const fabric::FabricTransportEndpointRef endpoint{owner, ordinal};
    const auto direction = fabric.transportEndpointDirection(endpoint);
    const auto dataPath = fabric.transportEndpointDataPath(endpoint);
    if (!direction || !dataPath || !dataPath->isWellFormed())
      return invalid("Fabric endpoint has no complete transport contract");
    if (dataPath->payloadWidthBits > mlir::IntegerType::kMaxWidth ||
        dataPath->tagWidthBits > mlir::IntegerType::kMaxWidth)
      return unsupported("Fabric endpoint width exceeds CIRCT capacity");
    const std::uint64_t local = *direction == fabric::FabricPortDirection::Input
                                    ? inputOrdinal++
                                    : outputOrdinal++;
    const std::string prefix =
        (*direction == fabric::FabricPortDirection::Input ? "input_"
                                                          : "output_") +
        std::to_string(local);
    const auto forward = *direction == fabric::FabricPortDirection::Input
                             ? circt::hw::ModulePort::Direction::Input
                             : circt::hw::ModulePort::Direction::Output;
    const auto backward = *direction == fabric::FabricPortDirection::Input
                              ? circt::hw::ModulePort::Direction::Output
                              : circt::hw::ModulePort::Direction::Input;
    const auto port = [&](llvm::StringRef suffix, mlir::Type type,
                          circt::hw::ModulePort::Direction portDirection) {
      return circt::hw::PortInfo{
          {builder.getStringAttr(prefix + suffix.str()), type, portDirection}};
    };
    std::optional<circt::hw::PortInfo> data;
    if (dataPath->payloadWidthBits != 0)
      data = port("_data", builder.getIntegerType(dataPath->payloadWidthBits),
                  forward);
    std::optional<circt::hw::PortInfo> tag;
    if (dataPath->kind == ::fabric::DataPathKind::BitsTag)
      tag =
          port("_tag", builder.getIntegerType(dataPath->tagWidthBits), forward);
    result.push_back({endpoint, *direction, local, *dataPath, std::move(data),
                      std::move(tag),
                      port("_valid", builder.getI1Type(), forward),
                      port("_ready", builder.getI1Type(), backward)});
  }
  return result;
}

void appendEndpointPorts(llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
                         llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
                         const EndpointPlan &endpoint) {
  const auto append = [&](const circt::hw::PortInfo &port) {
    (port.isOutput() ? outputs : inputs).push_back(port);
  };
  if (endpoint.data)
    append(*endpoint.data);
  if (endpoint.tag)
    append(*endpoint.tag);
  append(endpoint.valid);
  append(endpoint.ready);
}

void appendClockResetAndConfigurationPorts(
    mlir::OpBuilder &builder, const ConfigurationBundlePlan &configuration,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs) {
  inputs.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("clock"),
                           circt::seq::ClockType::get(builder.getContext()),
                           circt::hw::ModulePort::Direction::Input}});
  inputs.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("reset"), builder.getI1Type(),
                           circt::hw::ModulePort::Direction::Input}});
  if (!configuration.empty())
    inputs.push_back(circt::hw::PortInfo{
        {builder.getStringAttr(configurationBundlePortName),
         configurationBundleType(builder.getContext(), configuration),
         circt::hw::ModulePort::Direction::Input}});
}

llvm::Expected<mlir::Value> projectConfigurationBundle(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value parentValue,
    const ConfigurationBundlePlan &parent,
    const ConfigurationBundlePlan &child) {
  if (child.empty())
    return invalid("empty configuration bundle has no structural value");
  if (!parentValue || parent.empty() ||
      parentValue.getType() !=
          configurationBundleType(builder.getContext(), parent))
    return invalid("parent configuration bundle has the wrong type");
  if (parent.words == child.words)
    return parentValue;

  const unsigned parentIndexWidth =
      std::max(1U, llvm::Log2_64_Ceil(parent.words.size()));
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(child.words.size());
  for (const ConfigurationBundleWord &childWord :
       llvm::reverse(child.words)) {
    const ConfigurationBundleWord *parentWord = parent.find(childWord.key);
    if (!parentWord ||
        (parentWord->usedBitMask & childWord.usedBitMask) !=
            childWord.usedBitMask)
      return invalid("child configuration word is absent from its parent "
                     "bundle");
    const std::size_t parentOrdinal =
        static_cast<std::size_t>(parentWord - parent.words.data());
    mlir::Value word = circt::hw::ArrayGetOp::create(
        builder, location, parentValue,
        circt::hw::ConstantOp::create(
            builder, location,
            llvm::APInt(parentIndexWidth, parentOrdinal)));
    if (childWord.usedBitMask != std::numeric_limits<std::uint32_t>::max())
      word = circt::comb::AndOp::create(
          builder, location, word,
          circt::hw::ConstantOp::create(
              builder, location, llvm::APInt(32, childWord.usedBitMask)),
          true);
    highToLow.push_back(word);
  }
  return circt::hw::ArrayCreateOp::create(builder, location, highToLow);
}

llvm::Error addConfigurationInstanceInput(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor,
    const ConfigurationBundlePlan &parent,
    const ConfigurationBundlePlan &child, circt::hw::HWModuleOp childModule,
    std::map<std::string, mlir::Value> &inputs) {
  if (llvm::Error error = verifyConfigurationBundlePort(childModule, child))
    return error;
  if (child.empty())
    return llvm::Error::success();
  auto projected = projectConfigurationBundle(
      builder, location, accessor.getInput(configurationBundlePortName), parent,
      child);
  if (!projected)
    return projected.takeError();
  if (!inputs.emplace(configurationBundlePortName.str(), *projected).second)
    return invalid("configuration bundle instance input is duplicated");
  return llvm::Error::success();
}

ConfigurationBundleSignals configurationBundleSignals(
    circt::hw::HWModulePortAccessor &accessor,
    const ConfigurationBundlePlan &configuration) {
  assert(!configuration.empty() &&
         "empty configuration bundle has no structural signals");
  return ConfigurationBundleSignals{
      &configuration, accessor.getInput(configurationBundlePortName),
      std::vector<mlir::Value>(configuration.words.size())};
}

unsigned indexWidth(std::uint64_t count) {
  return std::max(1U, llvm::Log2_64_Ceil(std::max<std::uint64_t>(count, 1)));
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(1, value));
}

mlir::Value andValues(mlir::OpBuilder &builder, mlir::Location location,
                      llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return bitConstant(builder, location, true);
  std::vector<mlir::Value> level(values.begin(), values.end());
  while (level.size() != 1) {
    std::vector<mlir::Value> next;
    next.reserve((level.size() + 1) / 2);
    for (std::size_t index = 0; index < level.size(); index += 2) {
      if (index + 1 == level.size())
        next.push_back(level[index]);
      else
        next.push_back(circt::comb::AndOp::create(
            builder, location, level[index], level[index + 1]));
    }
    level = std::move(next);
  }
  return level.front();
}

mlir::Value orValues(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return bitConstant(builder, location, false);
  std::vector<mlir::Value> level(values.begin(), values.end());
  while (level.size() != 1) {
    std::vector<mlir::Value> next;
    next.reserve((level.size() + 1) / 2);
    for (std::size_t index = 0; index < level.size(); index += 2) {
      if (index + 1 == level.size())
        next.push_back(level[index]);
      else
        next.push_back(circt::comb::OrOp::create(
            builder, location, level[index], level[index + 1]));
    }
    level = std::move(next);
  }
  return level.front();
}

mlir::Value decodeFieldSignal(mlir::OpBuilder &builder, mlir::Location location,
                              ConfigurationBundleSignals &configuration,
                              const FieldDecoderPlan &decoder) {
  assert(configuration.plan && configuration.bundle &&
         "configuration bundle signals are incomplete");
  const unsigned indexWidth =
      std::max(1U, llvm::Log2_64_Ceil(configuration.plan->words.size()));
  const auto readWord = [&](std::uint64_t wordOrdinal) -> mlir::Value {
    const ConfigurationBundleWord *word = configuration.plan->find(
        {decoder.transportUnitOrdinal, wordOrdinal});
    assert(word && "configuration decoder word is absent from its bundle");
    const std::size_t ordinal =
        static_cast<std::size_t>(word - configuration.plan->words.data());
    if (!configuration.cachedWords[ordinal])
      configuration.cachedWords[ordinal] = circt::hw::ArrayGetOp::create(
          builder, location, configuration.bundle,
          circt::hw::ConstantOp::create(builder, location,
                                        llvm::APInt(indexWidth, ordinal)));
    return configuration.cachedWords[ordinal];
  };

  llvm::SmallVector<mlir::Value> decodedSlices;
  decodedSlices.reserve(decoder.destinationSlices.size());
  for (const DestinationSlice &slice : decoder.destinationSlices) {
    llvm::SmallVector<mlir::Value> lowToHigh;
    std::uint64_t bit = slice.destinationBitOffset;
    std::uint64_t remaining = slice.bitCount;
    while (remaining != 0) {
      const std::uint64_t wordOrdinal = bit / 32;
      const unsigned bitInWord = static_cast<unsigned>(bit % 32);
      const unsigned width = static_cast<unsigned>(
          std::min<std::uint64_t>(remaining, 32 - bitInWord));
      lowToHigh.push_back(circt::comb::ExtractOp::create(
          builder, location, readWord(wordOrdinal), bitInWord, width));
      bit += width;
      remaining -= width;
    }
    if (lowToHigh.size() == 1)
      decodedSlices.push_back(lowToHigh.front());
    else
      decodedSlices.push_back(circt::comb::ConcatOp::create(
          builder, location,
          llvm::SmallVector<mlir::Value>(llvm::reverse(lowToHigh))));
  }
  if (decodedSlices.size() == 1)
    return decodedSlices.front();
  return circt::comb::ConcatOp::create(
      builder, location,
      llvm::SmallVector<mlir::Value>(llvm::reverse(decodedSlices)));
}

mlir::Value matchesCode(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value field, const llvm::APInt &code) {
  mlir::Value constant = circt::hw::ConstantOp::create(builder, location, code);
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, field, constant, true);
}

mlir::Value selectedBit(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value field, std::uint64_t bit) {
  return circt::comb::ExtractOp::create(builder, location, field, bit, 1);
}

llvm::Expected<ForwardTransportSignals>
adaptForward(mlir::OpBuilder &builder, mlir::Location location,
             const EndpointPlan &source, const EndpointPlan &destination,
             const ChannelSignals &signals) {
  return adaptForwardTransportSignals(
      builder, location, source.dataPath, destination.dataPath,
      ForwardTransportSignals{signals.valid, signals.data, signals.tag});
}

mlir::Value createRegister(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value next, mlir::Value clock,
                           mlir::Value reset, const llvm::APInt &resetValue,
                           llvm::StringRef name, bool asynchronousReset) {
  mlir::Value resetConstant =
      circt::hw::ConstantOp::create(builder, location, resetValue);
  if (asynchronousReset)
    return circt::seq::FirRegOp::create(
        builder, location, next, clock, builder.getStringAttr(name), reset,
        resetConstant, circt::hw::InnerSymAttr{}, true);
  return circt::seq::CompRegOp::create(builder, location, next, clock, reset,
                                       resetConstant, name);
}

llvm::Expected<std::map<std::string, mlir::Value>>
instantiateModule(mlir::OpBuilder &builder, mlir::Location location,
                  circt::hw::HWModuleOp module, llvm::StringRef instanceName,
                  const std::map<std::string, mlir::Value> &inputs) {
  const auto describeType = [](mlir::Type type) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    stream << type;
    return result;
  };
  llvm::SmallVector<mlir::Value> operands;
  for (const circt::hw::PortInfo &port : module.getPortList()) {
    if (port.isOutput())
      continue;
    const auto found = inputs.find(port.getName().str());
    if (found == inputs.end())
      return invalid("module '" + module.getSymName().str() + "' instance '" +
                     instanceName.str() + "' input '" + port.getName().str() +
                     "' has no structural signal");
    if (found->second.getType() != port.type)
      return invalid("module '" + module.getSymName().str() + "' instance '" +
                     instanceName.str() + "' input '" + port.getName().str() +
                     "' expects " + describeType(port.type) + ", received " +
                     describeType(found->second.getType()));
    operands.push_back(found->second);
  }
  auto instance = circt::hw::InstanceOp::create(
      builder, location, module.getOperation(), instanceName, operands);
  std::map<std::string, mlir::Value> outputs;
  unsigned ordinal = 0;
  for (const circt::hw::PortInfo &port : module.getPortList()) {
    if (!port.isOutput())
      continue;
    if (!outputs.emplace(port.getName().str(), instance.getResult(ordinal++))
             .second)
      return invalid("module output name is duplicated");
  }
  return outputs;
}

llvm::Expected<mlir::Operation *>
findCanonicalEntityOperation(const fabric::FabricArtifactView &fabric,
                             fabric::FabricEntityId id) {
  const mlir::Operation *canonical = fabric.canonicalOperation();
  if (!canonical)
    return invalid("Fabric import has no canonical MLIR cache");
  mlir::Operation *found = nullptr;
  std::uint64_t matchCount = 0;
  const_cast<mlir::Operation *>(canonical)->walk([&](mlir::Operation *op) {
    auto entity =
        op->getAttrOfType<::fabric::EntityIdAttr>(::fabric::kEntityIdAttrName);
    if (!entity || entity.getId() != id)
      return;
    ++matchCount;
    if (matchCount == 1)
      found = op;
  });
  if (matchCount != 1)
    return invalid("Fabric entity has no unique canonical MLIR operation");
  return found;
}

llvm::Expected<mlir::Operation *>
findCanonicalFuNodeOperation(const fabric::FabricArtifactView &fabric,
                             fabric::FabricFuOccurrenceNodeRef node) {
  auto operation = findCanonicalEntityOperation(fabric, node.fu.id());
  if (!operation)
    return operation.takeError();
  auto fu = mlir::dyn_cast<::fabric::FuOp>(*operation);
  if (!fu)
    return invalid("FU occurrence entity does not name fabric.fu");
  std::uint64_t ordinal = 0;
  for (mlir::Operation &candidate : fu.getBody().front()) {
    const bool isNode =
        mlir::isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
            candidate);
    if (!isNode)
      continue;
    if (ordinal++ != node.ordinal)
      continue;
    const auto expectedKind = fabric.fuNodeKind(
        fabric::FabricInventoryOwnerRef::of(node.fu), node.ordinal);
    if (!expectedKind || *expectedKind != node.node)
      return invalid("canonical FU node kind disagrees with Fabric identity");
    return &candidate;
  }
  return invalid("canonical FU node ordinal is absent");
}

} // namespace loom::hardware::rtl::hierarchy
