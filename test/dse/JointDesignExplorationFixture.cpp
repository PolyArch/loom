#include "JointDesignExplorationFixture.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/LoomOps.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

namespace loom::dse::joint_test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint design exploration anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::string key(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

} // namespace

TemporaryDirectory::TemporaryDirectory() {
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("loom-joint-design", path_))
    fail("cannot create test directory: " + error.message());
}

TemporaryDirectory::~TemporaryDirectory() {
  llvm::sys::fs::remove_directories(path_);
}

llvm::StringRef TemporaryDirectory::path() const { return path_; }

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry
      .insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
              mlir::DLTIDialect, mlir::func::FuncDialect, loom::LoomDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  std::int32_t constant) {
  const std::string source = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) iv (%i: index) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant )mlir" +
                             std::to_string(constant) + R"mlir( : i32
    %extent = arith.constant 4 : index
    %thread = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    return
  }
}

)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

ArtifactRootReference
publishApplicationWorkload(const dataflow::CanonicalDataflowArtifact &artifact,
                           const ArtifactStore &store) {
  auto view = take(artifact.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("application fixture does not have one rooted graph launch");
  dataflow::RootedGraphLaunchRef launch{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
  sim::SpatialSimulationWorkload draft{launch};
  auto logicalDomain =
      take(view.projectRootThreadLogicalDomain(launch.rootThreadLaunch));
  draft.denseCoordinates.assign(logicalDomain.coordinateRank, 0);
  auto shapes = take(sim::projectSpatialSimulationBoundaryShapes(view, launch));
  draft.valueInputPlan.assign(shapes.valueInputs.size(),
                              sim::RuntimeValueInput{});
  auto workload = take(sim::finalizeSimulationWorkload(draft, view));
  return take(sim::publishSimulationWorkload(workload, store));
}

ArtifactRootReference
publishApplicationRuntimeInput(const ArtifactRootReference &workload,
                               std::int32_t value, const ArtifactStore &store) {
  auto imported = take(sim::importSpatialSimulationWorkload(workload, store));
  auto view = take(imported.dataflow.view());
  const auto *spatial = imported.workload.spatial();
  if (!spatial)
    fail("application fixture workload is not Spatial");
  sim::SpatialSimulationRuntimeInputDraft draft{imported.workload.identity()};
  for (auto [ordinal, source] : llvm::enumerate(spatial->valueInputPlan))
    if (std::holds_alternative<sim::RuntimeValueInput>(source))
      draft.runtimeValues.push_back(
          {static_cast<std::uint64_t>(ordinal),
           {1, {sim::SemanticLane::defined(llvm::APInt(32, value))}}});
  auto runtime =
      take(sim::finalizeSimulationRuntimeInput(draft, imported.workload, view));
  return take(sim::publishSimulationRuntimeInput(runtime, store));
}

evaluation::models::FpaFeatureView
projectFpaFeatures(const ArtifactRootReference &dataflow,
                   const ArtifactRootReference &system,
                   const ResolvedConfig &config, const ArtifactStore &artifacts,
                   const BlobStore &blobs) {
  auto prepared =
      take(evaluation::models::prepareCanonicalDataflowFabricEvaluation(
          dataflow, system, config, artifacts, blobs));
  const evaluation::EvaluationModelDescriptor *descriptor =
      prepared.request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    fail("FPA feature fixture lost its model descriptor");
  auto evaluationCase = take(evaluation::EvaluationCase::get(
      descriptor->caseSignature, prepared.request.subjectBindings(),
      prepared.request.workload(), prepared.request.runtimeInput(),
      prepared.request.baseConditions(), prepared.resolution, artifacts,
      blobs));
  auto projected = take(evaluation::projectModelFeatures(
      evaluation::models::fpaModelParameterContractRef(), evaluationCase,
      prepared.resolution, artifacts, blobs));
  const auto *features = projected.getIf<evaluation::models::FpaFeatureView>();
  if (!features)
    fail("FPA contract returned a foreign feature view");
  return *features;
}

std::vector<fabric::FabricModuleEntityCorrespondence>
identityModuleEntityCorrespondence(const fabric::FabricArtifactView &module) {
  std::vector<fabric::FabricModuleEntityCorrespondence> result;
  const auto append = [&](auto occurrences, fabric::FabricEntityKind kind) {
    for (std::uint64_t ordinal = 0; ordinal != occurrences.size(); ++ordinal) {
      const auto occurrence = occurrences[ordinal];
      result.push_back(
          {{kind, occurrence.id(), ordinal}, {kind, occurrence.id(), ordinal}});
    }
  };
  append(module.peOccurrences(), fabric::FabricEntityKind::FabricPeOccurrence);
  append(module.fuOccurrences(), fabric::FabricEntityKind::FabricFuOccurrence);
  append(module.memoryOccurrences(),
         fabric::FabricEntityKind::FabricMemoryOccurrence);
  append(module.switchOccurrences(),
         fabric::FabricEntityKind::FabricSwitchOccurrence);
  append(module.fifoOccurrences(),
         fabric::FabricEntityKind::FabricFifoOccurrence);
  append(module.boundaryOccurrences(),
         fabric::FabricEntityKind::FabricBoundaryOccurrence);
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.source.kind, lhs.source.occurrenceOrdinal) <
           std::tie(rhs.source.kind, rhs.source.occurrenceOrdinal);
  });
  return result;
}

bool everyCoreIsUsed(const ArtifactRootReference &systemReference,
                     llvm::ArrayRef<ArtifactRootReference> mappings,
                     const ArtifactStore &store) {
  auto systemArtifact =
      take(fabric::importEntireFabricRoot(systemReference, store));
  auto system = take(fabric::requireSystemRoot(systemArtifact.view()));
  std::set<std::string> used;
  for (const ArtifactRootReference &reference : mappings) {
    auto mapping = take(mapping::importSystemMapping(reference, store));
    ArtifactRootReference dataflowReference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        mapping.view().dataflowIdentity()};
    auto dataflowArtifact =
        take(dataflow::importCanonicalDataflow(dataflowReference, store));
    auto dataflowView = take(dataflowArtifact.view());
    auto projection = take(mapping::projectSystemExecutionContexts(
        dataflowView, mapping.view().executionBindings()));
    for (const auto &domain : projection.instructionDomains)
      used.insert(key(fabric::canonicalFabricBytes(domain.context.accCore)));
  }
  return llvm::all_of(
      system.artifact().accCoreOccurrences(),
      [&](fabric::AccCoreOccurrenceRef core) {
        return used.count(key(fabric::canonicalFabricBytes(core))) != 0;
      });
}

} // namespace loom::dse::joint_test
