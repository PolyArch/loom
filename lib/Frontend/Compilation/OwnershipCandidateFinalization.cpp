#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace loom::frontend {
namespace {

llvm::Error reject(SpatialOwnershipCandidateRejectionKind kind,
                   const llvm::Twine &message) {
  return llvm::make_error<SpatialOwnershipCandidateRejection>(kind,
                                                              message.str());
}

std::string typeSpelling(mlir::FunctionType type) {
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  stream << type;
  return spelling;
}

llvm::Error requireExactFabricCapabilities(
    const dataflow::CanonicalDataflowArtifact &program,
    const fabric::FinalizedFabricRoot &fabric) {
  auto view = program.view();
  if (!view)
    return view.takeError();
  if (view->graphs().empty() || view->actors().empty())
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  "materialized candidate has no SpatialCore workload");

  FabricCapabilityIndex capabilities(fabric.view());
  auto miss = capabilities.firstInadmissibleActor(program);
  if (!miss)
    return miss.takeError();
  if (!*miss)
    return llvm::Error::success();
  const llvm::StringRef resource =
      (*miss)->actorKind == dataflow::CanonicalDataflowActorKind::Memory
          ? "memory resource"
          : "operation resource";
  return reject(SpatialOwnershipCandidateRejectionKind::ExactFabricInadmissible,
                "exact Fabric admits no " + resource + " for actor " +
                    dataflow::operationSchemaSpelling((*miss)->schema) +
                    " with type " + typeSpelling((*miss)->type));
}

} // namespace

llvm::Expected<MaterializedOwnershipCandidate>
finalizeSpatialOwnershipCandidate(
    MaterializedStructuredOwnershipCandidate candidate,
    const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &loweringOptions) {
  auto projected =
      lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
          candidate.structuredProgram, loweringOptions);
  if (!projected)
    return projected.takeError();
  if (llvm::Error error =
          requireExactFabricCapabilities(projected->artifact, fabric))
    return std::move(error);
  return MaterializedOwnershipCandidate{
      std::move(candidate.structuredProgram),
      std::move(projected->artifact),
      std::move(projected->spatialGraphs),
      std::move(candidate.ownedSpatialRegion),
      std::move(candidate.blockActivityLineage),
      std::move(candidate.sourceProvenance)};
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeSpatialOwnershipDecision(
    const StructuredProgramCandidate &parent,
    const SpatialOwnershipScope &scope,
    const SpatialOwnershipDecisionPoint &decision,
    const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &lowering,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto structured = materializeStructuredSpatialOwnershipDecision(
      parent, scope, decision, sourceProvenance);
  if (!structured)
    return structured.takeError();
  return finalizeSpatialOwnershipCandidate(std::move(*structured), fabric,
                                           lowering);
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeSpatialOwnership(const StructuredProgramCandidate &parent,
                            const StructuredEntityRef &selection,
                            const fabric::FinalizedFabricRoot &fabric,
                            const SpatialOwnershipOptions &options) {
  return materializeSpatialOwnershipDecision(
      parent, {selection},
      {options.addressProjection, options.forallOwnershipShape,
       options.directCallSpecializationShape, options.directCallInlining},
      fabric, options.lowering);
}

} // namespace loom::frontend
