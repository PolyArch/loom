#ifndef LOOM_TEST_MAPPING_TECHMAPPINGARTIFACTTESTSUPPORT_H
#define LOOM_TEST_MAPPING_TECHMAPPINGARTIFACTTESTSUPPORT_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

namespace loom::test::tech_mapping_artifact {

mlir::MLIRContext makeComputeBoundaryContext();

dataflow::CanonicalDataflowArtifact
buildComputeBoundaryDataflow(mlir::MLIRContext &context);

std::string computeBoundaryMappingText(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::fabric::FabricArtifactView &fabric, bool includeBoundaries);

mlir::OwningOpRef<mlir::ModuleOp> parseTechMapping(mlir::MLIRContext &context,
                                                   llvm::StringRef text);

std::string spatialConstraintMappingText(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const loom::mapping::TechMappingView &techMapping,
    const loom::fabric::FabricArtifactView &fabric, llvm::StringRef clauses);

void artifactRoundTripAndReferenceValidation();
void computeBoundaryClosure();
void spatialCandidateWorkflow(llvm::StringRef testCase);

} // namespace loom::test::tech_mapping_artifact

#endif // LOOM_TEST_MAPPING_TECHMAPPINGARTIFACTTESTSUPPORT_H
