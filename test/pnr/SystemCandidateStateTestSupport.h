#pragma once

#include "Common/Artifact.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>

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

} // namespace pnr::test
} // namespace loom
