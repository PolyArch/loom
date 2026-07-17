#include "VerifierInternal.h"

#include <cstddef>
#include <map>
#include <utility>
#include <variant>

using namespace loom::mapping;
using namespace loom::mapping::detail;

std::vector<ValidatedConfiguredBoundaryPort>
loom::mapping::detail::deriveActiveConfiguredBoundaryPorts(
    const EncodingDescriptor &encoding,
    const std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *>
        &actorToOp,
    const std::map<std::uint64_t, PairedLaneProjection>
        &actorToLaneProjections) {
  using PortKey = std::pair<PortDirection, std::uint32_t>;
  std::map<PortKey, PortDescriptor> active;
  auto activateInput = [&](std::uint32_t fuPort) {
    for (const ConfiguredInputDescriptor &input : encoding.inputs)
      if (input.fuPort == fuPort)
        active.emplace(PortKey{PortDirection::Input, fuPort}, input.port);
  };

  if (actorToLaneProjections.empty()) {
    for (const ConfiguredInputDescriptor &input : encoding.inputs)
      active.emplace(PortKey{PortDirection::Input, input.fuPort}, input.port);
    for (const ConfiguredOutputDescriptor &output : encoding.outputs)
      active.emplace(PortKey{PortDirection::Output, output.fuPort},
                     output.port);
  } else {
    for (const auto &entry : actorToOp) {
      const ConfiguredFabricOpDescriptor &configured = *entry.second;
      auto selectedLanes = actorToLaneProjections.find(entry.first);
      if (selectedLanes == actorToLaneProjections.end()) {
        for (const ConfiguredValue &operand : configured.operands)
          if (const auto *input = std::get_if<FuInputValue>(&operand))
            activateInput(input->index);
        for (std::size_t result = 0; result < configured.outputPorts.size();
             ++result) {
          const ConfiguredValue value = FabricOpResultValue{
              configured.operation, static_cast<std::uint32_t>(result)};
          for (const ConfiguredOutputDescriptor &output : encoding.outputs)
            if (output.value == value)
              active.emplace(PortKey{PortDirection::Output, output.fuPort},
                             output.port);
        }
        continue;
      }

      for (std::uint32_t lane : selectedLanes->second.laneIndices) {
        if (const auto *input =
                std::get_if<FuInputValue>(&configured.operands[lane]))
          activateInput(input->index);
        const ConfiguredValue value =
            FabricOpResultValue{configured.operation, lane};
        for (const ConfiguredOutputDescriptor &output : encoding.outputs)
          if (output.value == value)
            active.emplace(PortKey{PortDirection::Output, output.fuPort},
                           output.port);
      }
    }
  }

  std::vector<ValidatedConfiguredBoundaryPort> projection;
  projection.reserve(active.size());
  for (const auto &entry : active)
    projection.push_back({entry.first.first, entry.first.second, entry.second});
  return projection;
}
