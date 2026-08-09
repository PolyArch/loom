#include "Hardware/RTL/OperationLeaf.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_operation_leaf_invalid: " + message);
}

circt::hw::PortInfo port(mlir::OpBuilder &builder, llvm::StringRef name,
                         unsigned width,
                         circt::hw::ModulePort::Direction direction) {
  return circt::hw::PortInfo{
      {builder.getStringAttr(name), builder.getIntegerType(width), direction}};
}

bool samePort(const circt::hw::PortInfo &lhs, const circt::hw::PortInfo &rhs) {
  return lhs.getName() == rhs.getName() && lhs.type == rhs.type &&
         lhs.dir == rhs.dir;
}

struct PhysicalPortInventory final {
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
};

llvm::Expected<PhysicalPortInventory> derivePhysicalPortInventory(
    const fabric::ResolvedFabricOpCapabilityView &capability) {
  PhysicalPortInventory result;
  std::set<std::pair<fabric::FabricPortDirection, fabric::FabricOrdinal>> seen;
  for (const fabric::ResolvedFabricOpPhysicalPortView &physical :
       capability.physicalPorts) {
    if (physical.reference.node != capability.occurrence)
      return invalid("physical port belongs to a different Fabric operation");
    if (!seen.emplace(physical.reference.direction, physical.reference.ordinal)
             .second)
      return invalid("physical port reference is duplicated");
    if (physical.reference.direction == fabric::FabricPortDirection::Input)
      result.inputs.push_back(&physical);
    else if (physical.reference.direction ==
             fabric::FabricPortDirection::Output)
      result.outputs.push_back(&physical);
    else
      return invalid("physical port has an unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(result.inputs, byOrdinal);
  llvm::sort(result.outputs, byOrdinal);
  const auto dense =
      [](llvm::ArrayRef<const fabric::ResolvedFabricOpPhysicalPortView *>
             ports) {
        return llvm::all_of(llvm::enumerate(ports), [](const auto &entry) {
          return entry.value()->reference.ordinal == entry.index();
        });
      };
  if (!dense(result.inputs) || !dense(result.outputs))
    return invalid("physical port ordinals are not dense");
  return result;
}

llvm::Error
requireControlShape(const fabric::ResolvedFabricOpCapabilityView &capability,
                    const PhysicalPortInventory &ports,
                    ::dataflow::OperationSchemaId schema,
                    std::size_t inputCount, std::size_t outputCount) {
  if (capability.enabledOperationSchemas.size() != 1 ||
      capability.enabledOperationSchemas.front() != schema)
    return invalid(
        ::fabric::implementationFamilyKeyword(capability.implementationFamily) +
        " does not expose its exact registered schema");
  if (ports.inputs.size() != inputCount || ports.outputs.size() != outputCount)
    return invalid(
        ::fabric::implementationFamilyKeyword(capability.implementationFamily) +
        " physical port inventory is incomplete");
  auto cases = ::dataflow::semantics::projectActorHandshakeCases(
      schema, static_cast<std::uint32_t>(inputCount),
      static_cast<std::uint32_t>(outputCount));
  if (!cases)
    return cases.takeError();
  if (cases->empty())
    return invalid("control/stream schema has no transition shape");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FabricOperationLeafInterface> deriveFabricOperationLeafInterface(
    const fabric::ResolvedFabricOpCapabilityView &capability) {
  using ::dataflow::OperationSchemaId;
  using ::fabric::ImplementationFamilyId;

  auto ports = derivePhysicalPortInventory(capability);
  if (!ports)
    return ports.takeError();

  FabricOperationLeafProtocol protocol;
  OperationSchemaId schema;
  std::size_t inputCount = 0;
  std::size_t outputCount = 0;
  switch (capability.implementationFamily) {
  case ImplementationFamilyId::FixedVectorParallelize:
    protocol = FabricOperationLeafProtocol::OrderedCardinalityToken;
    schema = OperationSchemaId::DataflowParallelize;
    inputCount = 2;
    outputCount = 3;
    break;
  case ImplementationFamilyId::FixedVectorSerialize:
    protocol = FabricOperationLeafProtocol::OrderedCardinalityToken;
    schema = OperationSchemaId::DataflowSerialize;
    inputCount = 3;
    outputCount = 2;
    break;
  case ImplementationFamilyId::TokenConstant:
    protocol = FabricOperationLeafProtocol::ElasticToken;
    schema = OperationSchemaId::DataflowConstant;
    inputCount = 1;
    outputCount = 1;
    break;
  case ImplementationFamilyId::TokenSync:
    protocol = FabricOperationLeafProtocol::ElasticToken;
    schema = OperationSchemaId::DataflowSync;
    inputCount = ports->inputs.size();
    outputCount = ports->outputs.size();
    if (inputCount == 0 || inputCount != outputCount)
      return invalid("TokenSync requires equal nonempty physical lane images");
    break;
  case ImplementationFamilyId::TokenMux:
    protocol = FabricOperationLeafProtocol::ElasticToken;
    schema = OperationSchemaId::DataflowMux;
    inputCount = ports->inputs.size();
    outputCount = 1;
    if (inputCount < 3)
      return invalid("TokenMux physical input inventory is incomplete");
    break;
  case ImplementationFamilyId::TokenDemux:
    protocol = FabricOperationLeafProtocol::ElasticToken;
    schema = OperationSchemaId::DataflowDemux;
    inputCount = 2;
    outputCount = ports->outputs.size();
    if (outputCount < 2)
      return invalid("TokenDemux physical output inventory is incomplete");
    break;
  case ImplementationFamilyId::LoopStream:
    protocol = FabricOperationLeafProtocol::ManagedToken;
    schema = OperationSchemaId::DataflowStream;
    inputCount = 3;
    outputCount = 2;
    break;
  case ImplementationFamilyId::LoopCarry:
    protocol = FabricOperationLeafProtocol::TransparentToken;
    schema = OperationSchemaId::DataflowCarry;
    inputCount = 3;
    outputCount = 1;
    break;
  case ImplementationFamilyId::LoopInvariant:
    protocol = FabricOperationLeafProtocol::TransparentToken;
    schema = OperationSchemaId::DataflowInvariant;
    inputCount = 2;
    outputCount = 1;
    break;
  case ImplementationFamilyId::LoopGate:
    protocol = FabricOperationLeafProtocol::TransparentToken;
    schema = OperationSchemaId::DataflowGate;
    inputCount = 2;
    outputCount = 2;
    break;
  default:
    return FabricOperationLeafInterface{};
  }

  if (llvm::Error error = requireControlShape(capability, *ports, schema,
                                              inputCount, outputCount))
    return std::move(error);
  return FabricOperationLeafInterface{protocol};
}

llvm::APInt encodeLoopCarryOperationLeafState(
    ::dataflow::semantics::CarrySemanticState state) {
  switch (state) {
  case ::dataflow::semantics::CarrySemanticState::Initial:
    return llvm::APInt(1, 0);
  case ::dataflow::semantics::CarrySemanticState::Running:
    return llvm::APInt(1, 1);
  }
  llvm_unreachable("unknown carry semantic state");
}

llvm::Expected<std::optional<TransparentLoopOperationLeafStateLayout>>
deriveTransparentLoopOperationLeafStateLayout(
    const fabric::ResolvedFabricOpCapabilityView &capability) {
  using ::fabric::ImplementationFamilyId;
  switch (capability.implementationFamily) {
  case ImplementationFamilyId::LoopCarry:
  case ImplementationFamilyId::LoopGate:
    return TransparentLoopOperationLeafStateLayout{};
  case ImplementationFamilyId::LoopInvariant:
    break;
  default:
    return std::nullopt;
  }

  const fabric::ResolvedFabricOpPhysicalPortView *payloadInput = nullptr;
  const fabric::ResolvedFabricOpPhysicalPortView *payloadOutput = nullptr;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    const bool isInput =
        port.reference.direction == fabric::FabricPortDirection::Input &&
        port.reference.ordinal == 1;
    const bool isOutput =
        port.reference.direction == fabric::FabricPortDirection::Output &&
        port.reference.ordinal == 0;
    if (isInput) {
      if (payloadInput)
        return invalid("loop invariant state layout repeats its payload input");
      payloadInput = &port;
    }
    if (isOutput) {
      if (payloadOutput)
        return invalid(
            "loop invariant state layout repeats its payload output");
      payloadOutput = &port;
    }
  }
  if (!payloadInput || !payloadOutput)
    return invalid("loop invariant state layout is missing a payload port");

  const unsigned payloadWidth =
      std::min(payloadInput->payloadWidthBits, payloadOutput->payloadWidthBits);
  if (payloadWidth >= mlir::IntegerType::kMaxWidth)
    return invalid(
        "transparent loop state width exceeds the CIRCT integer limit");
  return TransparentLoopOperationLeafStateLayout{payloadWidth};
}

namespace {

llvm::Error appendStateField(FabricOperationLeafStateLayout &layout,
                             FabricOperationLeafStateFieldKind kind,
                             unsigned width) {
  if (width == 0)
    return invalid("operation state field has zero width");
  if (width > mlir::IntegerType::kMaxWidth ||
      layout.bitCount > mlir::IntegerType::kMaxWidth - width)
    return invalid("operation state width exceeds the CIRCT integer limit");
  layout.fields.push_back({kind, layout.bitCount, width});
  layout.bitCount += width;
  return llvm::Error::success();
}

unsigned maximumLoopStreamWidth(const ::fabric::LoopStreamParams &parameters,
                                const PhysicalPortInventory &ports) {
  unsigned semanticWidth = 0;
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    if (parameters.integerWidths.contains(width))
      semanticWidth = std::max(semanticWidth, ::fabric::getBitWidth(width));
  if (ports.inputs.size() != 3 || ports.outputs.size() != 2)
    return 0;
  return std::min({semanticWidth, ports.inputs[0]->payloadWidthBits,
                   ports.inputs[1]->payloadWidthBits,
                   ports.inputs[2]->payloadWidthBits,
                   ports.outputs[0]->payloadWidthBits});
}

unsigned minimumAdapterElementWidth(
    const ::fabric::FixedVectorAdapterParams &parameters) {
  unsigned result = std::numeric_limits<unsigned>::max();
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    if (parameters.integerElementWidths.contains(width))
      result = std::min(result, ::fabric::getBitWidth(width));
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain)
    if (parameters.floatElementFormats.contains(format))
      result = std::min(result, ::fabric::getBitWidth(format));
  return result == std::numeric_limits<unsigned>::max() ? 0 : result;
}

llvm::Expected<FabricOperationLeafStateLayout> deriveAdapterStateLayout(
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const PhysicalPortInventory &ports, bool parallelize) {
  const auto *parameters = std::get_if<::fabric::FixedVectorAdapterParams>(
      &capability.parameterizedCapability);
  if (!parameters)
    return invalid("fixed-vector adapter has the wrong parameter schema");
  const unsigned elementWidth = minimumAdapterElementWidth(*parameters);
  if (elementWidth == 0)
    return invalid("fixed-vector adapter has no supported element width");
  const unsigned physicalValueWidth = parallelize
                                          ? ports.outputs[0]->payloadWidthBits
                                          : ports.inputs[0]->payloadWidthBits;
  const unsigned physicalMaskWidth = parallelize
                                         ? ports.outputs[1]->payloadWidthBits
                                         : ports.inputs[1]->payloadWidthBits;
  const unsigned valueWidth =
      std::min(parameters->maxPayloadBits, physicalValueWidth);
  const unsigned maskWidth =
      std::min(physicalMaskWidth, valueWidth / elementWidth);
  if (valueWidth == 0 || maskWidth == 0)
    return invalid("fixed-vector adapter state carrier is empty");

  FabricOperationLeafStateLayout result;
  if (llvm::Error error = appendStateField(
          result, FabricOperationLeafStateFieldKind::BufferedValue, valueWidth))
    return std::move(error);
  if (llvm::Error error = appendStateField(
          result, FabricOperationLeafStateFieldKind::BufferedMask, maskWidth))
    return std::move(error);
  return result;
}

} // namespace

llvm::Expected<std::optional<FabricOperationLeafStateLayout>>
deriveFabricOperationLeafStateLayout(
    const fabric::ResolvedFabricOpCapabilityView &capability) {
  using ::fabric::ImplementationFamilyId;
  auto interface = deriveFabricOperationLeafInterface(capability);
  if (!interface)
    return interface.takeError();
  auto ports = derivePhysicalPortInventory(capability);
  if (!ports)
    return ports.takeError();

  FabricOperationLeafStateLayout result;
  switch (capability.implementationFamily) {
  case ImplementationFamilyId::LoopCarry:
  case ImplementationFamilyId::LoopGate:
    if (llvm::Error error = appendStateField(
            result, FabricOperationLeafStateFieldKind::Mode, 1))
      return std::move(error);
    return std::optional<FabricOperationLeafStateLayout>{std::move(result)};
  case ImplementationFamilyId::LoopInvariant: {
    auto transparent =
        deriveTransparentLoopOperationLeafStateLayout(capability);
    if (!transparent)
      return transparent.takeError();
    if (!*transparent)
      return invalid("LoopInvariant has no transparent state layout");
    if (llvm::Error error = appendStateField(
            result, FabricOperationLeafStateFieldKind::Mode, 1))
      return std::move(error);
    if ((*transparent)->payloadWidthBits != 0)
      if (llvm::Error error = appendStateField(
              result, FabricOperationLeafStateFieldKind::RetainedValue,
              (*transparent)->payloadWidthBits))
        return std::move(error);
    return std::optional<FabricOperationLeafStateLayout>{std::move(result)};
  }
  case ImplementationFamilyId::LoopStream: {
    const auto *parameters = std::get_if<::fabric::LoopStreamParams>(
        &capability.parameterizedCapability);
    if (!parameters)
      return invalid("LoopStream has the wrong parameter schema");
    const unsigned valueWidth = maximumLoopStreamWidth(*parameters, *ports);
    if (valueWidth == 0)
      return invalid("LoopStream has no reachable state carrier width");
    if (llvm::Error error = appendStateField(
            result, FabricOperationLeafStateFieldKind::Mode, 1))
      return std::move(error);
    for (FabricOperationLeafStateFieldKind kind :
         {FabricOperationLeafStateFieldKind::Current,
          FabricOperationLeafStateFieldKind::Limit,
          FabricOperationLeafStateFieldKind::Step})
      if (llvm::Error error = appendStateField(result, kind, valueWidth))
        return std::move(error);
    return std::optional<FabricOperationLeafStateLayout>{std::move(result)};
  }
  case ImplementationFamilyId::FixedVectorParallelize: {
    auto layout = deriveAdapterStateLayout(capability, *ports, true);
    if (!layout)
      return layout.takeError();
    return std::optional<FabricOperationLeafStateLayout>{std::move(*layout)};
  }
  case ImplementationFamilyId::FixedVectorSerialize: {
    auto layout = deriveAdapterStateLayout(capability, *ports, false);
    if (!layout)
      return layout.takeError();
    return std::optional<FabricOperationLeafStateLayout>{std::move(*layout)};
  }
  default:
    return std::nullopt;
  }
}

llvm::Expected<std::vector<circt::hw::PortInfo>> deriveFabricOperationLeafPorts(
    mlir::OpBuilder &builder,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi) {
  auto physicalPorts = derivePhysicalPortInventory(capability);
  if (!physicalPorts)
    return physicalPorts.takeError();
  const auto &inputs = physicalPorts->inputs;
  const auto &outputs = physicalPorts->outputs;

  std::vector<fabric::FabricSemanticConfigFieldRef> configurationFields =
      capability.configurationFieldSchema;
  llvm::sort(configurationFields, [](const auto &lhs, const auto &rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  if (std::adjacent_find(configurationFields.begin(),
                         configurationFields.end()) !=
      configurationFields.end())
    return invalid("configuration field reference is duplicated");

  auto stateLayout = deriveFabricOperationLeafStateLayout(capability);
  if (!stateLayout)
    return stateLayout.takeError();
  auto interface = deriveFabricOperationLeafInterface(capability);
  if (!interface)
    return interface.takeError();
  const bool stateTransform = stateLayout->has_value();
  const bool tokenHandshake = interface->hasTokenHandshake();
  const bool orderedProduction = interface->hasOrderedProductionGroups();
  std::vector<circt::hw::PortInfo> result;
  result.reserve(inputs.size() + configurationFields.size() + outputs.size() +
                 (tokenHandshake ? 2 * (inputs.size() + outputs.size()) : 0) +
                 (stateTransform ? 3 : 0) + (orderedProduction ? 1 : 0));
  for (const auto *input : inputs) {
    if (input->payloadWidthBits == 0)
      continue;
    if (input->payloadWidthBits > mlir::IntegerType::kMaxWidth)
      return invalid("physical input width exceeds the CIRCT integer limit");
    result.push_back(
        port(builder, "data_input_" + std::to_string(input->reference.ordinal),
             input->payloadWidthBits, circt::hw::ModulePort::Direction::Input));
  }
  if (tokenHandshake) {
    for (const auto *input : inputs)
      result.push_back(port(
          builder, "valid_input_" + std::to_string(input->reference.ordinal), 1,
          circt::hw::ModulePort::Direction::Input));
    for (const auto *output : outputs)
      result.push_back(port(
          builder, "ready_output_" + std::to_string(output->reference.ordinal),
          1, circt::hw::ModulePort::Direction::Input));
  }
  if (stateTransform) {
    const unsigned stateWidth = stateLayout->value().encodedBitCount();
    result.push_back(port(builder, "state_current", stateWidth,
                          circt::hw::ModulePort::Direction::Input));
  }
  for (const fabric::FabricSemanticConfigFieldRef &field :
       configurationFields) {
    const ConfigurationFieldEncoding *encoding =
        configurationAbi.findOperationField(occurrence, field.ordinal);
    if (!encoding)
      return invalid("configuration field is absent from ConfigurationABI");
    const std::uint64_t width = encoding->encodedBitCount();
    if (width == 0)
      return invalid("configuration field has zero encoded width");
    if (width > mlir::IntegerType::kMaxWidth)
      return invalid("configuration field width exceeds the CIRCT integer "
                     "limit");
    result.push_back(port(builder, "config_" + std::to_string(field.ordinal),
                          static_cast<unsigned>(width),
                          circt::hw::ModulePort::Direction::Input));
  }
  if (tokenHandshake)
    for (const auto *input : inputs)
      result.push_back(port(
          builder, "ready_input_" + std::to_string(input->reference.ordinal), 1,
          circt::hw::ModulePort::Direction::Output));
  for (const auto *output : outputs) {
    if (output->payloadWidthBits == 0)
      continue;
    if (output->payloadWidthBits > mlir::IntegerType::kMaxWidth)
      return invalid("physical output width exceeds the CIRCT integer limit");
    result.push_back(port(
        builder, "data_output_" + std::to_string(output->reference.ordinal),
        output->payloadWidthBits, circt::hw::ModulePort::Direction::Output));
  }
  if (tokenHandshake) {
    for (const auto *output : outputs)
      result.push_back(port(
          builder, "valid_output_" + std::to_string(output->reference.ordinal),
          1, circt::hw::ModulePort::Direction::Output));
  }
  if (orderedProduction)
    result.push_back(port(builder, "final_production", 1,
                          circt::hw::ModulePort::Direction::Output));
  if (stateTransform) {
    const unsigned stateWidth = stateLayout->value().encodedBitCount();
    result.push_back(port(builder, "state_next", stateWidth,
                          circt::hw::ModulePort::Direction::Output));
    result.push_back(port(builder, "state_write", 1,
                          circt::hw::ModulePort::Direction::Output));
  }
  return result;
}

llvm::Error verifyFabricOperationLeafPorts(
    circt::hw::HWModuleGeneratedOp leaf,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi) {
  if (!leaf)
    return invalid("operation leaf is absent");
  mlir::OpBuilder builder(leaf.getContext());
  auto expected = deriveFabricOperationLeafPorts(builder, occurrence,
                                                 capability, configurationAbi);
  if (!expected)
    return expected.takeError();
  const auto actual = leaf.getPortList();
  if (actual.size() != expected->size())
    return invalid("leaf port count does not match its derived contract");
  for (auto [index, expectedPort] : llvm::enumerate(*expected))
    if (!samePort(actual[index], expectedPort))
      return invalid("leaf port " + llvm::Twine(index) +
                     " does not match its derived contract");
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
