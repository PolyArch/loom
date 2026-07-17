#ifndef FABRIC_TECH_PASSES_H
#define FABRIC_TECH_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace fabric {

// Synthesize one fabric.fu per `loom.synth_group` value out of the
// single-block configured functions in the module. Each successfully
// synthesized FU is wrapped in a freshly created `fabric.module` named
// `@fu_<sanitized(group)>`
// and appended to the module; failed groups have their input func.funcs
// annotated with `loom.synth_failed = "<reason>"`. `configPath` selects a
// YAML/TOML SynthConfig file (empty == built-in defaults). `failAsError`
// escalates per-group warnings to errors (and signals pass failure).
// `dumpStats` emits one canonical `synth-stat` remark per group.
//
// This Fabric Tech utility pass is not part of canonical program IR.
std::unique_ptr<::mlir::Pass>
createSynthesizeConfiguredFunctionsPass(std::string configPath = "",
                                        bool failAsError = false,
                                        bool dumpStats = false);

void registerConfiguredFunctionSynthesisPass();

} // namespace fabric

#endif // FABRIC_TECH_PASSES_H
