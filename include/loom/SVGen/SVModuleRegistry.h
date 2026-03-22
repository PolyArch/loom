#ifndef LOOM_SVGEN_SVMODULEREGISTRY_H
#define LOOM_SVGEN_SVMODULEREGISTRY_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <set>
#include <string>
#include <vector>

namespace loom {
namespace svgen {

/// Tracks which pre-written SystemVerilog files from src/rtl/design/ are
/// needed for a given fabric design. Only modules actually referenced by the
/// MLIR are included in the final output.
class SVModuleRegistry {
public:
  /// Register that a pre-written SV module file is needed.
  /// @param category  Sub-directory under design/ (e.g. "common", "fabric",
  ///                  "fabric/spatial_sw", "arith", "math").
  /// @param filename  Filename within that category (e.g.
  ///                  "fabric_spatial_sw.sv").
  void requireModule(llvm::StringRef category, llvm::StringRef filename);

  /// Register a pre-written arithmetic/dialect op module given its MLIR
  /// op name. Maps dialect-qualified names like "arith.addi" to the
  /// corresponding SV file path (e.g. "arith/fu_op_addi.sv").
  /// Returns false if the op is an unsupported Tier 3 transcendental op
  /// and no FP IP profile is set.
  bool requireArithOp(llvm::StringRef mlirOpName,
                      llvm::StringRef fpIpProfile);

  /// Return the set of all required relative file paths (relative to
  /// src/rtl/design/), in a deterministic order suitable for filelist.f.
  std::vector<std::string> getRequiredFiles() const;

  /// Check whether a given MLIR op name maps to a known SV file.
  static bool isKnownOp(llvm::StringRef mlirOpName);

  /// Get the SV module name for a given MLIR op name
  /// (e.g. "arith.addi" -> "fu_op_addi").
  static std::string getSVModuleName(llvm::StringRef mlirOpName);

  /// Get the relative SV file path for a given MLIR op name
  /// (e.g. "arith.addi" -> "arith/fu_op_addi.sv").
  static std::string getSVFilePath(llvm::StringRef mlirOpName);

  /// Returns true if the given MLIR op name is a Tier 3 transcendental
  /// FP op that requires --fp-ip-profile to generate.
  static bool isTier3TranscendentalOp(llvm::StringRef mlirOpName);

private:
  /// Set of required relative file paths (category/filename).
  std::set<std::string> requiredFiles_;

  /// Always-needed common infrastructure files.
  void requireCommonInfrastructure();
  bool commonRequired_ = false;
};

} // namespace svgen
} // namespace loom

#endif // LOOM_SVGEN_SVMODULEREGISTRY_H
