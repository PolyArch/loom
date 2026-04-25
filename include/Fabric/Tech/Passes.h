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

void registerFabricTechPasses();

} // namespace fabric

#endif // FABRIC_TECH_PASSES_H
