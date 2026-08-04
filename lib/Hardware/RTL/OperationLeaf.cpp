#include "Hardware/RTL/OperationLeaf.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
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

} // namespace

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

llvm::Expected<std::vector<circt::hw::PortInfo>> deriveFabricOperationLeafPorts(
    mlir::OpBuilder &builder,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi) {
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  std::set<std::pair<fabric::FabricPortDirection, fabric::FabricOrdinal>> seen;
  for (const fabric::ResolvedFabricOpPhysicalPortView &physical :
       capability.physicalPorts) {
    if (physical.reference.node != capability.occurrence)
      return invalid("physical port belongs to a different Fabric operation");
    if (!seen.emplace(physical.reference.direction, physical.reference.ordinal)
             .second)
      return invalid("physical port reference is duplicated");
    if (physical.reference.direction == fabric::FabricPortDirection::Input)
      inputs.push_back(&physical);
    else if (physical.reference.direction ==
             fabric::FabricPortDirection::Output)
      outputs.push_back(&physical);
    else
      return invalid("physical port has an unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);

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

  auto stateLayout = deriveTransparentLoopOperationLeafStateLayout(capability);
  if (!stateLayout)
    return stateLayout.takeError();
  const bool stateTransform = stateLayout->has_value();
  std::vector<circt::hw::PortInfo> result;
  result.reserve(
      inputs.size() + configurationFields.size() + outputs.size() +
      (stateTransform ? 2 * (inputs.size() + outputs.size()) + 3 : 0));
  for (const auto *input : inputs) {
    if (input->payloadWidthBits == 0)
      continue;
    if (input->payloadWidthBits > mlir::IntegerType::kMaxWidth)
      return invalid("physical input width exceeds the CIRCT integer limit");
    result.push_back(
        port(builder, "data_input_" + std::to_string(input->reference.ordinal),
             input->payloadWidthBits, circt::hw::ModulePort::Direction::Input));
  }
  if (stateTransform) {
    for (const auto *input : inputs)
      result.push_back(port(
          builder, "valid_input_" + std::to_string(input->reference.ordinal), 1,
          circt::hw::ModulePort::Direction::Input));
    for (const auto *output : outputs)
      result.push_back(port(
          builder, "ready_output_" + std::to_string(output->reference.ordinal),
          1, circt::hw::ModulePort::Direction::Input));
    const unsigned stateWidth = stateLayout->value().encodedBitCount();
    result.push_back(port(builder, "state_current", stateWidth,
                          circt::hw::ModulePort::Direction::Input));
  }
  for (const fabric::FabricSemanticConfigFieldRef &field :
       configurationFields) {
    if (field.owner.catalog() !=
        fabric::FabricInventoryOwnerRef::of(capability.occurrence))
      return invalid("configuration field belongs to a different operation");
    const ConfigurationFieldEncoding *encoding =
        configurationAbi.findField(field);
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
  if (stateTransform)
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
  if (stateTransform) {
    for (const auto *output : outputs)
      result.push_back(port(
          builder, "valid_output_" + std::to_string(output->reference.ordinal),
          1, circt::hw::ModulePort::Direction::Output));
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
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi) {
  if (!leaf)
    return invalid("operation leaf is absent");
  mlir::OpBuilder builder(leaf.getContext());
  auto expected =
      deriveFabricOperationLeafPorts(builder, capability, configurationAbi);
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
