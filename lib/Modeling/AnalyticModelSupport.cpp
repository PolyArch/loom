#include "AnalyticModelSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Evaluation/NumericValue.h"
#include "Evaluation/OwnerValue.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models::detail {
namespace {

// The model identity pins these low-fidelity assumptions. They are not
// measured Fabric timing or physical implementation facts.
constexpr std::uint64_t kInstructionLeafEstimatePicoseconds = 1000;
constexpr std::uint64_t kSpatialPressureEstimatePicoseconds = 250;

struct EmptyConfigView {};

struct ActorDemand final {
  mlir::Operation *representative = nullptr;
  std::uint64_t count = 0;
};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.static_pressure.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyConfigView{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyConfigView>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "static pressure model config has the "
                                   "wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "static pressure model config must be empty");
  return OwnerValue::get(EmptyConfigView{});
}

} // namespace

const ResolvedModelConfigViewContract &emptyStaticPressureConfigView() {
  static const ResolvedModelConfigViewContract contract{
      configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};
  return contract;
}

llvm::Expected<CaseArtifactResolution>
resolveSingleSubjectFabricCase(const ArtifactRootReference &subject,
                               const ArtifactRootReference &fabricReference,
                               const ArtifactStore &artifactStore) {
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      resolved(&artifactRootReferenceLess);

  std::function<llvm::Error(const ArtifactRootReference &)> visit =
      [&](const ArtifactRootReference &current) -> llvm::Error {
    if (resolved.count(current) != 0)
      return llvm::Error::success();
    auto root = fabric::importEntireFabricRoot(current, artifactStore);
    if (!root)
      return root.takeError();
    std::vector<ArtifactRootReference> closure;
    for (const fabric::FabricDirectDependency &dependency :
         root->directDependencies()) {
      if (llvm::Error error = visit(dependency.root))
        return error;
      closure.push_back(dependency.root);
      const auto &nested = resolved.find(dependency.root)->second;
      closure.insert(closure.end(), nested.begin(), nested.end());
    }
    std::sort(closure.begin(), closure.end(), artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    resolved.emplace(current, std::move(closure));
    return llvm::Error::success();
  };

  if (llvm::Error error = visit(fabricReference))
    return std::move(error);
  std::vector<CaseArtifactResolution::Entry> entries;
  entries.reserve(resolved.size() + 1);
  for (auto &[artifact, closure] : resolved)
    entries.push_back({artifact, std::move(closure)});
  entries.push_back({subject, {}});
  return CaseArtifactResolution::get(std::move(entries));
}

llvm::Expected<MetricResult>
staticPressureRuntimeMetric(std::uint64_t instructionLeaves,
                            std::uint64_t spatialPressure) {
  const std::optional<std::uint64_t> instructionTime = llvm::checkedMulUnsigned(
      instructionLeaves, kInstructionLeafEstimatePicoseconds);
  const std::optional<std::uint64_t> spatialTime = llvm::checkedMulUnsigned(
      spatialPressure, kSpatialPressureEstimatePicoseconds);
  if (!instructionTime || !spatialTime)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "static_pressure_model_overflow: pressure product");
  const std::optional<std::uint64_t> total =
      llvm::checkedAddUnsigned(*instructionTime, *spatialTime);
  if (!total || *total > static_cast<std::uint64_t>(
                             std::numeric_limits<std::int64_t>::max()))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "static_pressure_model_overflow: Runtime estimate");
  auto decimal = DecimalValue::get(static_cast<std::int64_t>(*total), -12);
  if (!decimal)
    return decimal.takeError();
  return MetricResult{
      UncertaintyKind::Unknown, PointObservation{MetricValue{*decimal}}, {}};
}

llvm::Expected<std::optional<std::uint64_t>> canonicalDataflowStaticPressure(
    const ::dataflow::CanonicalDataflowProgramView &program,
    const fabric::FinalizedFabricRoot &fabricRoot) {
  std::map<std::vector<std::uint8_t>, ActorDemand> actorDemands;
  for (const dataflow::CanonicalActorView &actor : program.actors()) {
    auto projection =
        dataflow::projectRegisteredActorSchemaProjectionBytes(actor.op);
    if (!projection)
      return projection.takeError();
    std::vector<std::uint8_t> key(projection->bytes().begin(),
                                  projection->bytes().end());
    ActorDemand &demand = actorDemands[key];
    if (!demand.representative)
      demand.representative = actor.op;
    ++demand.count;
  }

  frontend::FabricCapabilityIndex capabilities(fabricRoot.view());
  std::uint64_t spatialPressure = 0;
  for (const auto &[key, demand] : actorDemands) {
    (void)key;
    llvm::Expected<std::uint64_t> capacity =
        dataflow::isCanonicalDataflowActor(
            demand.representative, dataflow::CanonicalDataflowActorKind::Memory)
            ? capabilities.admittingMemoryResourceCount(demand.representative)
            : capabilities.admittingOperationResourceCount(
                  demand.representative);
    if (!capacity)
      return capacity.takeError();
    if (*capacity == 0)
      return std::optional<std::uint64_t>{};
    const std::uint64_t pressure =
        demand.count / *capacity + (demand.count % *capacity != 0 ? 1 : 0);
    spatialPressure = std::max(spatialPressure, pressure);
  }
  return std::optional<std::uint64_t>(spatialPressure);
}

} // namespace loom::evaluation::models::detail
