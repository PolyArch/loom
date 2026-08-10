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

#include <climits>
#include <set>
#include <utility>

namespace loom::hardware::rtl::hierarchy {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_module_hierarchy_invalid: " + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::make_error<FabricStructuralLoweringUnsupportedError>(
      message.str());
}

std::string configurationPortName(std::size_t transportUnitOrdinal) {
  return "configuration_" + std::to_string(transportUnitOrdinal);
}

std::string endpointKey(const fabric::FabricTransportEndpointRef &endpoint) {
  const std::vector<std::uint8_t> bytes =
      fabric::canonicalFabricBytes(endpoint);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
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
  for (const ProgrammingUnit &unit : configurationAbi.programmingUnits())
    for (const ConfigurationFieldEncoding &candidate : unit.fields)
      if (&candidate == &encoding) {
        if (owner)
          return invalid(
              "configuration field has duplicate programming owners");
        owner = &unit;
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

  const std::uint64_t width = encoding.encodedBitCount();
  if (width == 0 || width > mlir::IntegerType::kMaxWidth)
    return unsupported(
        "configuration field width exceeds the CIRCT support envelope");
  std::vector<std::uint64_t> destinationBits(static_cast<std::size_t>(width),
                                             UINT64_MAX);
  for (const DestinationSlice &slice : encoding.destinationSlices) {
    if (slice.sourceBitOffset > width || slice.bitCount > width ||
        slice.sourceBitOffset + slice.bitCount > width ||
        slice.destinationBitOffset > owner->payloadBitCount ||
        slice.bitCount > owner->payloadBitCount ||
        slice.destinationBitOffset + slice.bitCount > owner->payloadBitCount)
      return invalid("configuration destination slice is out of range");
    for (std::uint64_t bit = 0; bit < slice.bitCount; ++bit) {
      const std::size_t source =
          static_cast<std::size_t>(slice.sourceBitOffset + bit);
      if (destinationBits[source] != UINT64_MAX)
        return invalid(
            "configuration destination slices overlap one source bit");
      destinationBits[source] = slice.destinationBitOffset + bit;
    }
  }
  if (llvm::is_contained(destinationBits, UINT64_MAX))
    return invalid("configuration destination slices do not cover the field");
  return FieldDecoderPlan{owner, transportUnitOrdinal, width,
                          std::move(destinationBits)};
}

} // namespace

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout) {
  auto physical = qualifyConfigurationField(spatialCore, field);
  if (!physical)
    return physical.takeError();
  auto slot = fabric::qualifyFabricConfigurationSlot(
      *physical, fabric::FabricStaticConfigurationResidency{});
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
  auto decoder = prepareFieldDecoder(spatialCore, field, configurationAbi,
                                     transportLayout);
  if (!decoder)
    return decoder.takeError();
  auto physical = qualifyConfigurationField(spatialCore, field);
  if (!physical)
    return physical.takeError();
  auto slot = fabric::qualifyFabricConfigurationSlot(
      *physical, fabric::FabricStaticConfigurationResidency{});
  if (!slot)
    return slot.takeError();
  const ConfigurationFieldEncoding *encoding =
      configurationAbi.findField(*slot);
  if (!encoding)
    return invalid("configuration field is absent from ConfigurationABI: " +
                   fabric::printFabricRef(*physical));
  const auto *codebook =
      std::get_if<FiniteCodebookEncoding>(&encoding->semanticEncoding);
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
  const fabric::FabricInventoryOwnerRef owner =
      fabric::FabricInventoryOwnerRef::of(spatialCore);
  const fabric::HardwareDomainContractRecord *clock = nullptr;
  const fabric::HardwareDomainContractRecord *reset = nullptr;
  std::optional<fabric::ClockDomainRef> clockReference;
  for (fabric::HardwareDomainRef domain : system.hardwareDomains()) {
    const fabric::HardwareDomainContractRecord *contract =
        system.hardwareDomainContract(domain);
    if (!contract || !llvm::is_contained(contract->members(), owner))
      continue;
    if (contract->kind() == fabric::FabricHardwareDomainKind::Clock) {
      if (clock)
        return invalid("SpatialCore belongs to multiple Clock domains");
      clock = contract;
      clockReference = fabric::ClockDomainRef(domain);
    } else if (contract->kind() == fabric::FabricHardwareDomainKind::Reset) {
      if (reset)
        return invalid("SpatialCore belongs to multiple Reset domains");
      reset = contract;
    }
  }
  if (!clock || !reset || !clockReference)
    return unsupported("hierarchy lowering requires exact Clock and Reset "
                       "domains");
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
  if (synchronous && resetContract->synchronousTo() != clockReference)
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
    mlir::OpBuilder &builder, const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs) {
  inputs.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("clock"),
                           circt::seq::ClockType::get(builder.getContext()),
                           circt::hw::ModulePort::Direction::Input}});
  inputs.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("reset"), builder.getI1Type(),
                           circt::hw::ModulePort::Direction::Input}});
  for (auto [ordinal, transportUnit] : llvm::enumerate(transportLayout.units)) {
    const ProgrammingUnit *unit = configurationAbi.findProgrammingUnit(
        transportUnit.programmingUnit.unitId);
    assert(unit && "transport layout must reference an ABI unit");
    inputs.push_back(circt::hw::PortInfo{
        {builder.getStringAttr(configurationPortName(ordinal)),
         builder.getIntegerType(static_cast<unsigned>(unit->payloadBitCount)),
         circt::hw::ModulePort::Direction::Input}});
  }
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(1, value));
}

mlir::Value andValues(mlir::OpBuilder &builder, mlir::Location location,
                      llvm::ArrayRef<mlir::Value> values) {
  mlir::Value result = bitConstant(builder, location, true);
  for (mlir::Value value : values)
    result = circt::comb::AndOp::create(builder, location, result, value);
  return result;
}

mlir::Value orValues(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> values) {
  mlir::Value result = bitConstant(builder, location, false);
  for (mlir::Value value : values)
    result = circt::comb::OrOp::create(builder, location, result, value);
  return result;
}

mlir::Value decodeFieldSignal(mlir::OpBuilder &builder, mlir::Location location,
                              circt::hw::HWModulePortAccessor &accessor,
                              const FieldDecoderPlan &decoder) {
  mlir::Value payload =
      accessor.getInput(configurationPortName(decoder.transportUnitOrdinal));
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(static_cast<std::size_t>(decoder.encodedBitCount));
  for (std::uint64_t source = decoder.encodedBitCount; source > 0; --source)
    highToLow.push_back(circt::comb::ExtractOp::create(
        builder, location, payload,
        decoder.destinationBits[static_cast<std::size_t>(source - 1)], 1));
  if (highToLow.size() == 1)
    return highToLow.front();
  return circt::comb::ConcatOp::create(builder, location, highToLow);
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
  llvm::SmallVector<mlir::Value> operands;
  for (const circt::hw::PortInfo &port : module.getPortList()) {
    if (port.isOutput())
      continue;
    const auto found = inputs.find(port.getName().str());
    if (found == inputs.end())
      return invalid("module input '" + port.getName().str() +
                     "' has no structural signal");
    if (found->second.getType() != port.type)
      return invalid("module input '" + port.getName().str() +
                     "' has the wrong structural type");
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
