#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <utility>
#include <variant>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_invocation_invalid: " + message);
}

llvm::Expected<std::uint32_t>
transportBitCount(SpatialSimulationValueShape shape) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint32_t>::max() / shape.laneBitWidth)
    return invalid("boundary shape exceeds the invocation wire domain");
  return static_cast<std::uint32_t>(shape.lanesPerToken) * shape.laneBitWidth;
}

llvm::APInt
unpackLittleEndianBits(const runtime::SpatialInvocationValue &value) {
  llvm::APInt bits(value.bitCount, 0);
  for (std::size_t byte = 0; byte != value.littleEndianBits.size(); ++byte) {
    const std::uint64_t base = byte * 8;
    for (unsigned bit = 0; bit != 8 && base + bit < value.bitCount; ++bit)
      if ((value.littleEndianBits[byte] & (1U << bit)) != 0)
        bits.setBit(static_cast<unsigned>(base + bit));
  }
  return bits;
}

std::vector<std::uint8_t> packLittleEndianBits(const llvm::APInt &bits) {
  const std::size_t byteCount = (bits.getBitWidth() + 7) / 8;
  std::vector<std::uint8_t> bytes;
  bytes.reserve(byteCount);
  for (std::size_t byte = 0; byte != byteCount; ++byte) {
    const unsigned offset = static_cast<unsigned>(byte * 8);
    const unsigned width = std::min<unsigned>(8, bits.getBitWidth() - offset);
    bytes.push_back(
        static_cast<std::uint8_t>(bits.extractBitsAsZExtValue(width, offset)));
  }
  return bytes;
}

llvm::Error
validateInvocationOwner(const runtime::SpatialInvocationWire &wire,
                        const ImportedSpatialSimulationWorkload &workload,
                        const SpatialSimulationWorkload &spatial) {
  std::string wireDiagnostic;
  if (!runtime::validateSpatialInvocationWire(wire, wireDiagnostic))
    return invalid(wireDiagnostic);
  if (wire.canonicalDataflowIdentity != workload.dataflow.identity().bytes())
    return invalid("invocation names a foreign Canonical Dataflow owner");
  if (wire.rootThreadLaunchEntity !=
          spatial.launchRef.rootThreadLaunch.entity.value() ||
      wire.graphLaunchEntity !=
          spatial.launchRef.staticGraphLaunch.entity.value())
    return invalid("invocation names a foreign rooted graph launch");
  if (wire.denseCoordinates != spatial.denseCoordinates)
    return invalid("invocation dense coordinates differ from the workload");
  return llvm::Error::success();
}

llvm::Error
validateResultDestinations(const runtime::SpatialInvocationWire &wire,
                           const SpatialSimulationWorkload &workload,
                           const SpatialSimulationBoundaryShapes &shapes) {
  if (wire.results.size() != workload.observableContract.valueResults.size())
    return invalid("result destinations are not total over observable values");
  struct AddressInterval final {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<AddressInterval> intervals;
  intervals.reserve(wire.results.size());
  for (std::size_t ordinal = 0; ordinal != wire.results.size(); ++ordinal) {
    const std::uint64_t valueResult =
        workload.observableContract.valueResults[ordinal];
    if (valueResult >= shapes.valueResults.size())
      return invalid("observable value result is outside its boundary shape");
    auto bitCount = transportBitCount(shapes.valueResults[valueResult]);
    if (!bitCount)
      return bitCount.takeError();
    const runtime::SpatialInvocationResultDestination &destination =
        wire.results[ordinal];
    if (destination.bitCount != *bitCount || destination.address == 0)
      return invalid("result destination has the wrong shape or null address");
    const std::uint64_t byteCount =
        (static_cast<std::uint64_t>(destination.bitCount) + 7) / 8;
    if (byteCount >
        std::numeric_limits<std::uint64_t>::max() - destination.address)
      return invalid("result destination address range overflows");
    intervals.push_back({destination.address, destination.address + byteCount});
  }
  llvm::sort(intervals,
             [](const AddressInterval &lhs, const AddressInterval &rhs) {
               return lhs.begin < rhs.begin;
             });
  for (std::size_t ordinal = 1; ordinal != intervals.size(); ++ordinal)
    if (intervals[ordinal - 1].end > intervals[ordinal].begin)
      return invalid("result destination address ranges overlap");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CanonicalSimulationRuntimeInput>
materializeSpatialInvocationRuntimeInput(
    const ImportedSpatialSimulationWorkload &workload,
    const runtime::SpatialInvocationWire &wire) {
  const SpatialSimulationWorkload *spatial = workload.workload.spatial();
  if (!spatial)
    return invalid("workload lost its Spatial payload");
  if (llvm::Error error = validateInvocationOwner(wire, workload, *spatial))
    return std::move(error);
  auto view = workload.dataflow.view();
  if (!view)
    return view.takeError();
  auto shapes =
      projectSpatialSimulationBoundaryShapes(*view, spatial->launchRef);
  if (!shapes)
    return shapes.takeError();
  auto memoryInputs = view->graphMemoryInputs(spatial->launchRef);
  if (!memoryInputs)
    return memoryInputs.takeError();
  if (!shapes->streamInputs.empty() || !memoryInputs->empty() ||
      !spatial->observableContract.streamOutputs.empty() ||
      !spatial->observableContract.memories.empty())
    return invalid("dynamic invocation currently admits value-only graphs");
  if (wire.values.size() != spatial->valueInputPlan.size() ||
      wire.values.size() != shapes->valueInputs.size())
    return invalid("invocation values are not total over graph value inputs");

  SpatialSimulationRuntimeInputDraft draft{workload.workload.identity()};
  draft.runtimeValues.reserve(wire.values.size());
  for (std::size_t ordinal = 0; ordinal != wire.values.size(); ++ordinal) {
    if (!std::holds_alternative<RuntimeValueInput>(
            spatial->valueInputPlan[ordinal]))
      return invalid("dynamic invocation cannot replace a fixed value input");
    auto bitCount = transportBitCount(shapes->valueInputs[ordinal]);
    if (!bitCount)
      return bitCount.takeError();
    if (wire.values[ordinal].bitCount != *bitCount)
      return invalid("invocation value width differs from the graph input");
    auto lanes = unpackDefinedSpatialSimulationToken(
        unpackLittleEndianBits(wire.values[ordinal]),
        shapes->valueInputs[ordinal]);
    if (!lanes)
      return lanes.takeError();
    draft.runtimeValues.push_back(
        {ordinal, CanonicalValueSequence{1, std::move(*lanes)}});
  }
  if (llvm::Error error = validateResultDestinations(wire, *spatial, *shapes))
    return std::move(error);
  return finalizeSimulationRuntimeInput(draft, workload.workload, *view);
}

llvm::Expected<ImportedSpatialSimulationInputs>
materializeSpatialInvocationInputs(ImportedSpatialSimulationWorkload workload,
                                   const runtime::SpatialInvocationWire &wire) {
  auto runtimeInput = materializeSpatialInvocationRuntimeInput(workload, wire);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return ImportedSpatialSimulationInputs{std::move(workload.dataflow),
                                         std::move(workload.workload),
                                         std::move(*runtimeInput)};
}

namespace {

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectResultWrites(const runtime::SpatialInvocationWire &wire,
                    const dataflow::CanonicalDataflowArtifact &dataflow,
                    const CanonicalSimulationWorkload &workload,
                    const SpatialFunctionalObservations &observations) {
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return invalid("workload lost its Spatial payload");
  auto view = dataflow.view();
  if (!view)
    return view.takeError();
  auto shapes =
      projectSpatialSimulationBoundaryShapes(*view, spatial->launchRef);
  if (!shapes)
    return shapes.takeError();
  if (llvm::Error error = validateResultDestinations(wire, *spatial, *shapes))
    return std::move(error);
  if (observations.valueResults.size() != wire.results.size())
    return invalid("execution did not return every invocation value result");

  std::vector<SpatialInvocationMemoryWrite> writes;
  writes.reserve(wire.results.size());
  for (std::size_t ordinal = 0; ordinal != wire.results.size(); ++ordinal) {
    const auto *published =
        std::get_if<PublishedValueResult>(&observations.valueResults[ordinal]);
    if (!published)
      return invalid("an invocation value result was not published");
    const std::uint64_t resultOrdinal =
        spatial->observableContract.valueResults[ordinal];
    auto packed = packDefinedSpatialSimulationToken(
        published->value, shapes->valueResults[resultOrdinal], 0);
    if (!packed)
      return packed.takeError();
    writes.push_back(
        {wire.results[ordinal].address, packLittleEndianBits(*packed)});
  }
  return writes;
}

} // namespace

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationInputs &inputs,
    const SpatialFunctionalObservations &observations) {
  return projectResultWrites(wire, inputs.dataflow, inputs.workload,
                             observations);
}

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationWorkload &workload,
    const SpatialFunctionalObservations &observations) {
  return projectResultWrites(wire, workload.dataflow, workload.workload,
                             observations);
}

} // namespace loom::sim
