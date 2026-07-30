#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/Lowering/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <map>
#include <set>

namespace loom::lowering {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "canonical_dataflow_lowering_invalid: " +
                                     message);
}

} // namespace

llvm::Error
lowerStructuredModuleInPlace(mlir::ModuleOp module,
                             CanonicalDataflowLoweringOptions options) {
  if (!module)
    return invalid("missing Structured Program module");

  registerLoweringPasses();
  mlir::PassManager pipeline(module.getContext());
  pipeline.enableVerifier(options.verifyEach);
  if (options.applyPassManagerCommandLineOptions &&
      failed(mlir::applyPassManagerCLOptions(pipeline)))
    return invalid("cannot apply pass-manager command-line options");
  buildLoweringPipeline(pipeline);
  if (failed(pipeline.run(module)))
    return invalid("mechanical SCF-to-Dataflow lowering failed");
  if (llvm::Error error = dataflow::validateFinalizedProgram(module))
    return error;
  return llvm::Error::success();
}

llvm::Expected<dataflow::CanonicalDataflowArtifact>
lowerStructuredModuleToCanonicalDataflow(
    mlir::ModuleOp module, CanonicalDataflowLoweringOptions options) {
  if (!module)
    return invalid("missing Structured Program module");
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(module->clone()));
  if (llvm::Error error = lowerStructuredModuleInPlace(clone.get(), options))
    return std::move(error);
  return dataflow::finalizeCanonicalDataflow(clone.get());
}

llvm::Expected<ProjectedCanonicalDataflow>
lowerStructuredProgramToCanonicalDataflowWithProjection(
    const frontend::StructuredProgramCandidate &candidate,
    CanonicalDataflowLoweringOptions options) {
  auto candidateView = candidate.view();
  if (!candidateView)
    return candidateView.takeError();

  std::vector<frontend::StructuredEntityRef> spatialRegions;
  std::vector<mlir::Operation *> spatialOperations;
  for (const frontend::StructuredEntity &entity :
       candidateView->entities(frontend::StructuredEntityKind::Operation)) {
    if (!llvm::isa_and_nonnull<::loom::SpatialRegionOp>(entity.operation))
      continue;
    spatialRegions.push_back(entity.reference);
    spatialOperations.push_back(entity.operation);
  }
  bool hasPreexistingGraphLaunch = false;
  candidate.module().walk([&](mlir::Operation *operation) {
    hasPreexistingGraphLaunch |= llvm::isa<dataflow::GraphLaunchOp>(operation);
  });
  if (hasPreexistingGraphLaunch)
    return invalid("Structured Program contains a preexisting graph launch");

  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(candidate.module()->clone(cloneMapping)));
  std::vector<std::string> projectionNames;
  projectionNames.reserve(spatialOperations.size());
  std::set<std::string> reservedNames;
  for (auto [ordinal, sourceSpatial] : llvm::enumerate(spatialOperations)) {
    auto spatial = llvm::dyn_cast_or_null<::loom::SpatialRegionOp>(
        cloneMapping.lookupOrNull(sourceSpatial));
    if (!spatial)
      return invalid("Spatial owner was not cloned into the lowering input");
    std::string name =
        (llvm::Twine("__loom_spatial_projection_") + llvm::Twine(ordinal))
            .str();
    while (reservedNames.count(name) ||
           mlir::SymbolTable::lookupSymbolIn(clone.get(), name))
      name.push_back('_');
    reservedNames.insert(name);
    // The construction-local graph name carries only this lowering
    // transaction's correspondence. Canonical Dataflow private-name
    // normalization removes it from artifact identity.
    spatial->setAttr("graph_name",
                     mlir::StringAttr::get(clone->getContext(), name));
    projectionNames.push_back(std::move(name));
  }
  if (llvm::Error error = lowerStructuredModuleInPlace(clone.get(), options))
    return std::move(error);

  std::map<std::string, mlir::Operation *> graphLaunchesByCallee;
  clone->walk([&](dataflow::GraphLaunchOp launch) {
    const std::string callee = launch.getCallee().str();
    if (reservedNames.count(callee))
      graphLaunchesByCallee.emplace(callee, launch.getOperation());
  });
  if (graphLaunchesByCallee.size() != spatialRegions.size())
    return invalid("Spatial-to-graph lowering correspondence is not total");
  std::vector<mlir::Operation *> graphLaunches;
  graphLaunches.reserve(projectionNames.size());
  for (const std::string &name : projectionNames) {
    auto found = graphLaunchesByCallee.find(name);
    if (found == graphLaunchesByCallee.end())
      return invalid("Spatial-to-graph lowering correspondence is absent");
    graphLaunches.push_back(found->second);
  }

  auto finalized =
      dataflow::finalizeCanonicalDataflowWithTrackedStaticGraphLaunches(
          clone.get(), graphLaunches);
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedStaticGraphLaunches.size() != spatialRegions.size())
    return invalid("canonical graph launch projection changed cardinality");
  auto ownedArtifact = dataflow::importCanonicalDataflow(
      finalized->artifact.identity(), finalized->artifact.canonicalBytes());
  if (!ownedArtifact)
    return ownedArtifact.takeError();
  auto view = ownedArtifact->view();
  if (!view)
    return view.takeError();

  std::vector<StructuredSpatialGraphProjection> projections;
  projections.reserve(spatialRegions.size());
  for (auto [region, staticLaunch] :
       llvm::zip_equal(spatialRegions, finalized->trackedStaticGraphLaunches)) {
    if (auto resolved = view->resolve(staticLaunch); !resolved)
      return resolved.takeError();
    projections.push_back({region, staticLaunch});
  }
  return ProjectedCanonicalDataflow{std::move(*ownedArtifact),
                                    std::move(projections)};
}

llvm::Expected<dataflow::CanonicalDataflowArtifact>
lowerStructuredProgramToCanonicalDataflow(
    const frontend::StructuredProgramCandidate &candidate,
    CanonicalDataflowLoweringOptions options) {
  auto projected = lowerStructuredProgramToCanonicalDataflowWithProjection(
      candidate, options);
  if (!projected)
    return projected.takeError();
  return std::move(projected->artifact);
}

} // namespace loom::lowering
