#include "VerifierInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>

using namespace loom::mapping;
using namespace loom::mapping::detail;

namespace {
struct ValidatedPairedLaneCapability {
  std::map<std::uint32_t, std::uint32_t> laneByInputPort;
  std::map<std::uint32_t, std::uint32_t> laneByOutputPort;
};

llvm::Expected<ValidatedPairedLaneCapability>
buildValidatedPairedLaneCapability(const FabricOpDescriptor &operation) {
  const std::size_t laneCount = operation.pairedLanes.size();
  if (laneCount == 0 || laneCount != operation.inputPorts.size() ||
      laneCount != operation.outputPorts.size())
    return mappingError(
        MappingErrorCode::InvalidConfiguredFunction,
        "paired-lane capability must cover the complete physical signature");

  ValidatedPairedLaneCapability validated;
  std::set<std::uint32_t> maskBits;
  for (auto [laneIndex, lane] : llvm::enumerate(operation.pairedLanes)) {
    if (lane.inputPort >= operation.inputPorts.size() ||
        lane.outputPort >= operation.outputPorts.size() ||
        lane.maskBit >= laneCount ||
        !validated.laneByInputPort
             .emplace(lane.inputPort, static_cast<std::uint32_t>(laneIndex))
             .second ||
        !validated.laneByOutputPort
             .emplace(lane.outputPort, static_cast<std::uint32_t>(laneIndex))
             .second ||
        !maskBits.insert(lane.maskBit).second)
      return mappingError(
          MappingErrorCode::InvalidConfiguredFunction,
          "paired-lane capability has invalid endpoints or mask positions");
  }
  return validated;
}
} // namespace

llvm::Error loom::mapping::detail::validatePairedLaneCapability(
    const FabricOpDescriptor &operation) {
  auto capability = buildValidatedPairedLaneCapability(operation);
  if (!capability)
    return capability.takeError();
  return llvm::Error::success();
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
    const ActorToFabricOp &correspondence) {
  if (correspondence.fabricOp.artifact != fabricIdentity ||
      correspondence.fabricOp.entity != operation.id)
    return mappingError(
        MappingErrorCode::ConfiguredFunctionMismatch,
        "paired-lane correspondence does not name the supplied fabric.op");
  auto capability = buildValidatedPairedLaneCapability(operation);
  if (!capability)
    return capability.takeError();

  PairedLaneProjection projection;
  projection.bitmask.assign(operation.pairedLanes.size(), '0');
  std::set<std::uint32_t> selectedLanes;
  for (const PairedLaneSelection &selection : correspondence.laneSelections) {
    auto inputLane = capability->laneByInputPort.find(selection.inputPort);
    auto outputLane = capability->laneByOutputPort.find(selection.outputPort);
    if (inputLane == capability->laneByInputPort.end() ||
        outputLane == capability->laneByOutputPort.end() ||
        inputLane->second != outputLane->second)
      return mappingError(
          MappingErrorCode::ConfiguredFunctionMismatch,
          "input and output maps do not select the same declared lane");
    const std::uint32_t laneIndex = inputLane->second;
    if (!selectedLanes.insert(laneIndex).second)
      return mappingError(MappingErrorCode::ConfiguredFunctionMismatch,
                          "paired-lane correspondence repeats a lane");
    projection.laneIndices.push_back(laneIndex);
    projection.bitmask[operation.pairedLanes[laneIndex].maskBit] = '1';
  }
  return projection;
}
