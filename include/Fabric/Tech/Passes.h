#ifndef FABRIC_TECH_PASSES_H
#define FABRIC_TECH_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace fabric {

// Emit one non-authoritative legacy display adapter per explicit FU semantic
// encoding. Each derived dataflow.subgraph carries its
// `loom.from_fu_encoding` array index; Mapping decisions never consume or
// persist this projection.
std::unique_ptr<::mlir::Pass> createEnumerateFuSubgraphsPass();

// Synthesize one fabric.fu per `loom.synth_group` value out of the input
// dataflow.subgraphs in the module. Each successfully synthesized FU is
// wrapped in a freshly created `func.func` named `@fu_<sanitized(group)>`
// and appended to the module; failed groups have their input func.funcs
// annotated with `loom.synth_failed = "<reason>"`. `configPath` selects a
// YAML/TOML SynthConfig file (empty == built-in defaults). `failAsError`
// escalates per-group warnings to errors (and signals pass failure).
// `dumpStats` emits one canonical `synth-stat` remark per group.
//
// Lives in MLIRFabricTechSynthesizer (separate library); use
// `registerFabricTechSynthesizerPasses` to expose it in a driver.
std::unique_ptr<::mlir::Pass>
createGeneralizeSubgraphsToFuPass(std::string configPath = "",
                                  bool failAsError = false,
                                  bool dumpStats = false);

void registerFabricTechPasses();
void registerFabricTechSynthesizerPasses();

} // namespace fabric

#endif // FABRIC_TECH_PASSES_H
