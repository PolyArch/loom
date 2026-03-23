#ifndef LOOM_SVGEN_MULTICORESVGEN_H
#define LOOM_SVGEN_MULTICORESVGEN_H

#include "loom/Mapper/ConfigGen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace svgen {

/// Per-core compilation result consumed by multi-core SVGen.
/// This mirrors the CoreResult structure from the system compiler and will
/// be replaced by a direct dependency once that module is integrated.
struct MultiCoreCoreDesc {
  /// Instance name (unique across the system).
  std::string coreInstanceName;

  /// Core type name (shared across instances of the same type).
  std::string coreType;

  /// ADG MLIR module describing this core's fabric.
  mlir::ModuleOp adgModule;

  /// Aggregate configuration blob (all kernels combined).
  std::vector<uint8_t> aggregateConfigBlob;

  /// Per-hardware-node configuration slices.
  std::vector<ConfigGen::ConfigSlice> configSlices;
};

/// System-level compilation result consumed by multi-core SVGen.
struct MultiCoreCompilationDesc {
  bool success = false;

  /// Per-core descriptions.
  std::vector<MultiCoreCoreDesc> coreDescs;
};

/// Options controlling multi-core system-level SystemVerilog generation.
struct MultiCoreSVGenOptions {
  /// Output directory for generated RTL collateral.
  std::string outputDir;

  /// Path to the pre-written src/rtl/ directory in the source tree.
  std::string rtlSourceDir;

  /// Optional floating-point IP profile name.
  std::string fpIpProfile;

  /// Mesh dimensions for the NoC.
  unsigned meshRows = 1;
  unsigned meshCols = 1;

  /// NoC parameters.
  unsigned nocFlitWidth = 32;
  unsigned nocNumVC = 2;
  unsigned nocBufferDepth = 4;

  /// Memory hierarchy parameters.
  unsigned l2SizeBytes = 262144;
  unsigned l2NumBanks = 4;
  unsigned spmSizePerCore = 65536;
};

/// Result of multi-core system-level SVGen.
struct MultiCoreSVGenResult {
  bool success = false;

  /// Path to the generated system top module file.
  std::string systemTopFile;

  /// Path to the generated system filelist file.
  std::string systemFilelistFile;

  /// Per-core filelist paths.
  std::vector<std::string> perCoreFilelists;

  /// All generated files (absolute paths).
  std::vector<std::string> allGeneratedFiles;
};

/// Generate multi-core system-level SystemVerilog collateral.
///
/// This produces:
///   1. Per-core RTL via existing single-core SVGen
///   2. System top module (tapestry_system_top.sv) instantiating cores,
///      NoC mesh, and memory hierarchy
///   3. System filelist (system_filelist.f) for compilation
///
/// \param compilation  The multi-core compilation description with per-core
///                     ADG modules and config data.
/// \param options      Multi-core SVGen configuration.
/// \param ctx          MLIR context for per-core SVGen.
/// \returns            Result with file paths and success status.
MultiCoreSVGenResult
generateMultiCoreSV(const MultiCoreCompilationDesc &compilation,
                    const MultiCoreSVGenOptions &options,
                    mlir::MLIRContext *ctx);

} // namespace svgen
} // namespace loom

#endif // LOOM_SVGEN_MULTICORESVGEN_H
