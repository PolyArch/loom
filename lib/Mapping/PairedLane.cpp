#include "VerifierInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;

llvm::Expected<ValidatedPairedLaneCapability>
loom::mapping::detail::buildValidatedPairedLaneCapability(
    const FabricOpDescriptor &operation) {
  const std::size_t laneCount = operation.pairedLanes.size();
  if (laneCount == 0 || laneCount != operation.inputPorts.size() ||
      laneCount != operation.outputPorts.size())
    return mappingError(
        MappingErrorCode::InvalidConfiguredFunction,
        "paired-lane capability must cover the complete physical signature");

  constexpr std::uint32_t invalidLane =
      std::numeric_limits<std::uint32_t>::max();
  ValidatedPairedLaneCapability validated{
      std::vector<std::uint32_t>(laneCount, invalidLane),
      std::vector<std::uint32_t>(laneCount, invalidLane)};
  std::vector<bool> maskBits(laneCount, false);
  for (auto [laneIndex, lane] : llvm::enumerate(operation.pairedLanes)) {
    if (lane.inputPort >= operation.inputPorts.size() ||
        lane.outputPort >= operation.outputPorts.size() ||
        lane.maskBit >= laneCount ||
        validated.laneByInputPort[lane.inputPort] != invalidLane ||
        validated.laneByOutputPort[lane.outputPort] != invalidLane ||
        maskBits[lane.maskBit])
      return mappingError(
          MappingErrorCode::InvalidConfiguredFunction,
          "paired-lane capability has invalid endpoints or mask positions");
    validated.laneByInputPort[lane.inputPort] =
        static_cast<std::uint32_t>(laneIndex);
    validated.laneByOutputPort[lane.outputPort] =
        static_cast<std::uint32_t>(laneIndex);
    maskBits[lane.maskBit] = true;
  }
  return validated;
}

bool loom::mapping::detail::validPairedConfiguredPorts(
    const ConfiguredFabricOpDescriptor &configured,
    const FabricOpDescriptor &operation) {
  if (configured.inputPorts.size() != operation.pairedLanes.size() ||
      configured.outputPorts.size() != operation.pairedLanes.size())
    return false;
  for (auto [laneIndex, lane] : llvm::enumerate(operation.pairedLanes)) {
    const PortDescriptor &physicalInput = operation.inputPorts[lane.inputPort];
    const PortDescriptor &configuredInput = configured.inputPorts[laneIndex];
    const PortDescriptor &physicalOutput =
        operation.outputPorts[lane.outputPort];
    const PortDescriptor &configuredOutput = configured.outputPorts[laneIndex];
    if (physicalInput.kind != configuredInput.kind ||
        physicalInput.role != configuredInput.role ||
        physicalOutput.kind != configuredOutput.kind ||
        physicalOutput.role != configuredOutput.role)
      return false;
  }
  return true;
}

llvm::Expected<PairedLaneProjection>
loom::mapping::detail::validateAndProjectPairedLaneSelection(
    const ArtifactIdentity &fabricIdentity, const FabricOpDescriptor &operation,
    const ValidatedPairedLaneCapability &capability,
    const ActorToFabricOp &correspondence) {
  if (correspondence.fabricOp.artifact != fabricIdentity ||
      correspondence.fabricOp.entity != operation.id)
    return mappingError(
        MappingErrorCode::ConfiguredFunctionMismatch,
        "paired-lane correspondence does not name the supplied fabric.op");
  PairedLaneProjection projection;
  projection.bitmask.assign(operation.pairedLanes.size(), '0');
  std::vector<bool> selectedLanes(operation.pairedLanes.size(), false);
  for (const PairedLaneSelection &selection : correspondence.laneSelections) {
    if (selection.inputPort >= capability.laneByInputPort.size() ||
        selection.outputPort >= capability.laneByOutputPort.size() ||
        capability.laneByInputPort[selection.inputPort] !=
            capability.laneByOutputPort[selection.outputPort])
      return mappingError(
          MappingErrorCode::ConfiguredFunctionMismatch,
          "input and output maps do not select the same declared lane");
    const std::uint32_t laneIndex =
        capability.laneByInputPort[selection.inputPort];
    if (selectedLanes[laneIndex])
      return mappingError(MappingErrorCode::ConfiguredFunctionMismatch,
                          "paired-lane correspondence repeats a lane");
    selectedLanes[laneIndex] = true;
    projection.laneIndices.push_back(laneIndex);
    projection.bitmask[operation.pairedLanes[laneIndex].maskBit] = '1';
  }
  return projection;
}
