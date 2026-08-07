#pragma once

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

adg::FinalizedFabricDesign buildHeterogeneousSystem(
    ArtifactStore &store, const fabric::FinalizedFabricRoot &baselineSystem,
    const fabric::FinalizedFabricRoot &primaryModule,
    const fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context, bool extraSupportsRead = true);

} // namespace pnr::test
} // namespace loom
