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

/// Generate a FU body SV module. Returns the generated SV module name.
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
