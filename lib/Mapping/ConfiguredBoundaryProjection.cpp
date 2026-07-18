#include "VerifierInternal.h"

#include <cstddef>
#include <map>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;

llvm::Expected<ValidatedConfiguredBoundaryIndex>
loom::mapping::detail::buildValidatedConfiguredBoundaryIndex(
    const EncodingDescriptor &encoding) {
  ValidatedConfiguredBoundaryIndex index;
  index.inputs.reserve(encoding.inputs.size());
  std::unordered_map<std::uint32_t, std::size_t> inputByFuPort;
  for (const ConfiguredInputDescriptor &input : encoding.inputs) {
    const std::size_t denseIndex = index.inputs.size();
    if (!inputByFuPort.emplace(input.fuPort, denseIndex).second)
      return mappingError(
          MappingErrorCode::InvalidConfiguredFunction,
          "encoding has a duplicate configured FU input boundary");
    index.inputs.push_back({PortDirection::Input, input.fuPort, input.port});
  }

  index.operations.reserve(encoding.operations.size());
  for (const ConfiguredFabricOpDescriptor &operation : encoding.operations) {
    ValidatedConfiguredBoundaryOperation projected{operation.operation, {}};
    projected.inputOperands.reserve(operation.operands.size());
    for (const ConfiguredValue &operand : operation.operands) {
      const auto *input = std::get_if<FuInputValue>(&operand);
      if (!input) {
        projected.inputOperands.push_back(std::nullopt);
        continue;
      }
      auto denseIndex = inputByFuPort.find(input->index);
      if (denseIndex == inputByFuPort.end())
        return mappingError(
            MappingErrorCode::InvalidConfiguredFunction,
            "configured operation uses an unknown FU input boundary");
      projected.inputOperands.push_back(denseIndex->second);
    }
    index.operations.push_back(std::move(projected));
  }

  index.outputs.reserve(encoding.outputs.size());
  for (const ConfiguredOutputDescriptor &output : encoding.outputs) {
    ValidatedConfiguredBoundaryPort port{PortDirection::Output, output.fuPort,
                                         output.port};
    if (const auto *input = std::get_if<FuInputValue>(&output.value)) {
      auto denseIndex = inputByFuPort.find(input->index);
      if (denseIndex == inputByFuPort.end())
        return mappingError(
            MappingErrorCode::InvalidConfiguredFunction,
            "configured output uses an unknown FU input boundary");
      index.outputs.push_back({std::move(port), denseIndex->second});
    } else {
      index.outputs.push_back(
          {std::move(port), std::get<FabricOpResultValue>(output.value)});
    }
  }
  return index;
}

std::vector<ValidatedConfiguredBoundaryPort>
loom::mapping::detail::deriveActiveConfiguredBoundaryPorts(
    const ValidatedConfiguredBoundaryIndex &index,
    const std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *>
        &actorToOp,
    const std::map<std::uint64_t, PairedLaneProjection>
        &actorToLaneProjections) {
  std::unordered_map<std::uint64_t, std::vector<bool>> selectedByOperation;
  for (const auto &entry : actorToLaneProjections) {
    const ConfiguredFabricOpDescriptor &operation = *actorToOp.at(entry.first);
    std::vector<bool> selected(operation.operands.size(), false);
    for (std::uint32_t lane : entry.second.laneIndices)
      selected[lane] = true;
    selectedByOperation.emplace(operation.operation.value(),
                                std::move(selected));
  }

  std::vector<std::size_t> totalInputUses(index.inputs.size(), 0);
  std::vector<std::size_t> inactivePairedInputUses(index.inputs.size(), 0);
  for (const ValidatedConfiguredBoundaryOperation &operation :
       index.operations) {
    auto selected = selectedByOperation.find(operation.operation.value());
    for (std::size_t lane = 0; lane < operation.inputOperands.size(); ++lane) {
      if (!operation.inputOperands[lane])
        continue;
      const std::size_t input = *operation.inputOperands[lane];
      ++totalInputUses[input];
      if (selected != selectedByOperation.end() && !selected->second[lane])
        ++inactivePairedInputUses[input];
    }
  }

  std::vector<bool> activeOutputs(index.outputs.size(), true);
  for (std::size_t outputIndex = 0; outputIndex < index.outputs.size();
       ++outputIndex) {
    const ValidatedConfiguredBoundaryOutput &output =
        index.outputs[outputIndex];
    if (const auto *input = std::get_if<std::size_t>(&output.source)) {
      ++totalInputUses[*input];
      continue;
    }
    const FabricOpResultValue &result =
        std::get<FabricOpResultValue>(output.source);
    auto selected = selectedByOperation.find(result.operation.value());
    if (selected != selectedByOperation.end() &&
        !selected->second[result.index])
      activeOutputs[outputIndex] = false;
  }

  using PortKey = std::pair<PortDirection, std::uint32_t>;
  std::map<PortKey, PortDescriptor> active;
  for (std::size_t input = 0; input < index.inputs.size(); ++input) {
    if (totalInputUses[input] != 0 &&
        totalInputUses[input] == inactivePairedInputUses[input])
      continue;
    const ValidatedConfiguredBoundaryPort &port = index.inputs[input];
    active.emplace(PortKey{port.direction, port.fuPort}, port.descriptor);
  }
  for (std::size_t output = 0; output < index.outputs.size(); ++output) {
    if (!activeOutputs[output])
      continue;
    const ValidatedConfiguredBoundaryPort &port = index.outputs[output].port;
    active.emplace(PortKey{port.direction, port.fuPort}, port.descriptor);
  }

  std::vector<ValidatedConfiguredBoundaryPort> projection;
  projection.reserve(active.size());
  for (const auto &entry : active)
    projection.push_back({entry.first.first, entry.first.second, entry.second});
  return projection;
}
