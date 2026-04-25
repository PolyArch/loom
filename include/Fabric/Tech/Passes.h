#ifndef FABRIC_TECH_PASSES_H
#define FABRIC_TECH_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace fabric {

// Walk every fabric.fu in the module and append, immediately after the FU,
// one dataflow.subgraph per supported software configuration. Each emitted
// subgraph carries an `loom.from_fu_config` string attribute summarizing
// the configuration that produced it.
std::unique_ptr<::mlir::Pass> createEnumerateFuSubgraphsPass();

// For each dataflow.subgraph annotated with `loom.is_pattern`, try to find
// a fabric.fu in the module that can implement the pattern. On success the
// pattern is annotated with `loom.matched_fu` (FuOp identifier) and
// `loom.match_config` (configuration description); otherwise the pattern
// is tagged `loom.unmatched`.
std::unique_ptr<::mlir::Pass> createMapSubgraphToFusPass();

void registerFabricTechPasses();

} // namespace fabric

#endif // FABRIC_TECH_PASSES_H
