#pragma once

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingOps.h"
#include "PnR/System/SystemCandidateState.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;

namespace adg {
class FinalizedFabricDesign;
}

namespace fabric {
class FinalizedFabricRoot;
}

namespace pnr::test {

mlir::DenseI8ArrayAttr bytesAttr(mlir::MLIRContext *context,
                                 llvm::ArrayRef<std::uint8_t> bytes);

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute);

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes);

CanonicalSemanticBytes rawSystemBytes(::mapping::SystemOp root);

std::size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle);

::mapping::SystemPresburgerCellAttr
withFirstCoordinateLowerBound(::mapping::SystemPresburgerCellAttr cell,
                              std::int64_t lowerBound);

adg::FinalizedFabricDesign buildHeterogeneousSystem(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context, bool extraSupportsRead = true,
    bool routeExtraMemoryThroughTransform = false);

void verifyFinalizedSystemMappingWorkflow(
    const SystemCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const fabric::FabricSystemRootView &fabric,
    const mapping::SystemMappingConstraintSetView &emptyConstraints,
    ArtifactStore &store, mlir::MLIRContext &context,
    std::size_t expectedServiceCount);

} // namespace pnr::test
} // namespace loom
