#ifndef FCC_SVGEN_SVGENINTERNAL_H
#define FCC_SVGEN_SVGENINTERNAL_H

#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace fcc {
namespace svgen {

/// Per-module config layout information used to generate config_ctrl
/// parameters (slice offsets and word counts).
struct ModuleConfigSlice {
  std::string moduleName;
  unsigned bitCount = 0;
  unsigned wordCount = 0;
  unsigned wordOffset = 0;
};

/// Compute the config layout for all modules inside a fabric.module.
/// Returns slices in block-walk order matching ConfigGen.
std::vector<ModuleConfigSlice>
computeConfigLayout(fcc::fabric::ModuleOp fabricMod);

/// Compute the critical-path intrinsic latency for an FU body by analyzing
/// the SSA DAG from block arguments to yield operands. Returns the minimum
/// cycle count from input acceptance to result availability inside the FU
/// body itself. For dataflow FUs this returns 0 (their timing is not
/// latency-modeled).
unsigned computeFUIntrinsicLatency(fcc::fabric::FunctionUnitOp fuOp);

/// Validate the declared latency and interval of an FU against its body
/// operations. Emits "gen-sv error: latency-violation:" or
/// "gen-sv error: interval-violation:" diagnostics to llvm::errs() and
/// returns false if any constraint is violated.
bool validateFUTimingConstraints(fcc::fabric::FunctionUnitOp fuOp);

/// Generate a FU body SV module. Returns the generated SV module name,
/// or an empty string if code generation failed (e.g. unsupported op).
std::string generateFUBody(fcc::fabric::FunctionUnitOp fuOp,
                           llvm::raw_ostream &os,
                           SVModuleRegistry &registry,
                           llvm::StringRef fpIpProfile);

/// Generate a spatial PE wrapper. Returns the generated SV module name.
std::string generateSpatialPE(fcc::fabric::SpatialPEOp peOp,
                               llvm::raw_ostream &os,
                               SVModuleRegistry &registry,
                               llvm::StringRef fpIpProfile);

/// Generate a temporal PE wrapper. Returns the generated SV module name.
std::string generateTemporalPE(fcc::fabric::TemporalPEOp peOp,
                                llvm::raw_ostream &os,
                                SVModuleRegistry &registry,
                                llvm::StringRef fpIpProfile);

/// Generate the fabric_top module.
void generateTopModule(fcc::fabric::ModuleOp fabricMod,
                       llvm::raw_ostream &os,
                       SVModuleRegistry &registry,
                       const llvm::DenseMap<mlir::Operation *, std::string>
                           &peModuleNames);

} // namespace svgen
} // namespace fcc

#endif // FCC_SVGEN_SVGENINTERNAL_H
