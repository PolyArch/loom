#include "SimulationExecutionInternal.h"

#include "Common/ArtifactStore.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic{'L', 'S', 'E', '1'};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_engine_boundary_result_invalid: " + message);
}

llvm::Error validateResult(const SpatialEngineBoundaryResult &result,
                           const detail::SpatialExecutionContext &context) {
  if (std::holds_alternative<HaltedExecution>(result.terminal))
    return invalid("Halted requires a Request-owned terminal witness");
  if (llvm::Error error = detail::validateSpatialProgressObservations(
          result.progressObservations, result.terminal))
    return error;
  if (llvm::Error error = detail::validateSpatialFunctionalObservations(
          result.functionalObservations, result.terminal, context))
    return error;
  return detail::validateActorActivitySummaries(
      result.activitySummaries, result.terminal, result.progressObservations,
      context);
}

void encodeTerminal(detail::WireWriter &writer,
                    const ExecutionTerminal &terminal) {
  writer.u32(std::holds_alternative<RetiredExecution>(terminal) ? 0 : 1);
}

llvm::Expected<ExecutionTerminal> decodeTerminal(detail::WireReader &reader) {
  auto tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0)
    return ExecutionTerminal{RetiredExecution{}};
  if (*tag == 1)
    return ExecutionTerminal{StoppedByLimitExecution{}};
  return invalid("unknown terminal discriminant");
}

std::vector<std::uint8_t>
encodeCanonical(const SpatialEngineBoundaryResult &result,
                const detail::SpatialExecutionContext &context) {
  std::vector<std::uint8_t> bytes(kMagic.begin(), kMagic.end());
  detail::WireWriter writer;
  encodeTerminal(writer, result.terminal);
  detail::encodeSpatialFunctionalObservations(
      writer, result.functionalObservations, context);
  detail::encodeSpatialProgressObservations(writer,
                                            result.progressObservations);
  detail::encodeActorActivitySummaries(writer, result.activitySummaries);
  std::vector<std::uint8_t> payload = writer.take();
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

llvm::Expected<SpatialEngineBoundaryResult>
decodeCanonical(llvm::ArrayRef<std::uint8_t> bytes,
                const detail::SpatialExecutionContext &context) {
  if (bytes.size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
    return invalid("wrong or truncated wire identity");
  detail::WireReader reader(bytes.drop_front(kMagic.size()));
  auto terminal = decodeTerminal(reader);
  if (!terminal)
    return terminal.takeError();
  auto functional =
      detail::decodeSpatialFunctionalObservations(reader, context);
  if (!functional)
    return functional.takeError();
  auto progress = detail::decodeSpatialProgressObservations(reader);
  if (!progress)
    return progress.takeError();
  auto activities = detail::decodeActorActivitySummaries(reader);
  if (!activities)
    return activities.takeError();
  if (!reader.atEnd())
    return invalid("trailing bytes");
  SpatialEngineBoundaryResult result{
      std::move(*terminal), std::move(*functional), std::move(*progress),
      std::move(*activities)};
  if (llvm::Error error = validateResult(result, context))
    return std::move(error);
  if (llvm::ArrayRef<std::uint8_t>(encodeCanonical(result, context)) != bytes)
    return invalid("wire bytes are not canonical");
  return result;
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
encodeSpatialEngineBoundaryResult(const SpatialEngineBoundaryResult &result,
                                  const ArtifactRootReference &workload,
                                  const ArtifactRootReference &runtimeInput,
                                  const ArtifactStore &store) {
  auto context =
      detail::resolveSpatialEngineResultContext(workload, runtimeInput, store);
  if (!context)
    return context.takeError();
  if (llvm::Error error = validateResult(result, *context))
    return std::move(error);
  return encodeCanonical(result, *context);
}

llvm::Expected<std::vector<std::uint8_t>> encodeSpatialEngineBoundaryResult(
    const SpatialEngineBoundaryResult &result,
    const ImportedSpatialSimulationInputs &inputs) {
  auto context = detail::resolveSpatialEngineResultContext(inputs);
  if (!context)
    return context.takeError();
  if (llvm::Error error = validateResult(result, *context))
    return std::move(error);
  return encodeCanonical(result, *context);
}

llvm::Expected<std::vector<std::uint8_t>> encodeSpatialEngineBoundaryResult(
    const SpatialEngineBoundaryResult &result,
    const ImportedSpatialSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  auto context =
      detail::resolveSpatialEngineResultContext(workload, runtimeInput);
  if (!context)
    return context.takeError();
  if (llvm::Error error = validateResult(result, *context))
    return std::move(error);
  return encodeCanonical(result, *context);
}

llvm::Expected<SpatialEngineBoundaryResult> decodeSpatialEngineBoundaryResult(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ArtifactStore &store) {
  auto context =
      detail::resolveSpatialEngineResultContext(workload, runtimeInput, store);
  if (!context)
    return context.takeError();
  return decodeCanonical(bytes, *context);
}

llvm::Expected<SpatialEngineBoundaryResult> decodeSpatialEngineBoundaryResult(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ImportedSpatialSimulationInputs &inputs) {
  auto context = detail::resolveSpatialEngineResultContext(inputs);
  if (!context)
    return context.takeError();
  return decodeCanonical(bytes, *context);
}

llvm::Expected<SpatialEngineBoundaryResult> decodeSpatialEngineBoundaryResult(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ImportedSpatialSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  auto context =
      detail::resolveSpatialEngineResultContext(workload, runtimeInput);
  if (!context)
    return context.takeError();
  return decodeCanonical(bytes, *context);
}

} // namespace loom::sim
