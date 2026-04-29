#ifndef FABRIC_TECH_PASSES_H
#define FABRIC_TECH_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

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

// Partition each dataflow.graph body in the module into dataflow.subgraphs
// using the algorithm and weights specified by the (optional) tech-mapping
// config file. An empty `configPath` selects the built-in defaults.
std::unique_ptr<::mlir::Pass>
createPartitionGraphPass(std::string configPath = "");

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
