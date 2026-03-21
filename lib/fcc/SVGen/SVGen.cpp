#include "SVGenInternal.h"

#include "fcc/SVGen/SVGen.h"
#include "fcc/SVGen/SVEmitter.h"
#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace fcc {
namespace svgen {

namespace {

/// Copy a single file from srcPath to dstPath.
static bool copyFile(llvm::StringRef srcPath, llvm::StringRef dstPath) {
  auto ec = llvm::sys::fs::copy_file(srcPath, dstPath);
  if (ec) {
    llvm::errs() << "svgen: cannot copy " << srcPath << " -> " << dstPath
                 << ": " << ec.message() << "\n";
    return false;
  }
  return true;
}

/// Ensure a directory exists, creating it if needed.
static bool ensureDir(llvm::StringRef dirPath) {
  auto ec = llvm::sys::fs::create_directories(dirPath);
  if (ec) {
    llvm::errs() << "svgen: cannot create directory " << dirPath << ": "
                 << ec.message() << "\n";
    return false;
  }
  return true;
}

/// Write a string to a file.
static bool writeFile(llvm::StringRef path, llvm::StringRef content) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "svgen: cannot write " << path << ": " << ec.message()
                 << "\n";
    return false;
  }
  os << content;
  return true;
}

/// Collect FU body ops from a FunctionUnitOp and register needed SV files.
static bool collectFUDeps(fcc::fabric::FunctionUnitOp fuOp,
                          SVModuleRegistry &registry,
                          llvm::StringRef fpIpProfile) {
  auto &bodyBlock = fuOp.getBody().front();
  for (auto &op : bodyBlock.getOperations()) {
    llvm::StringRef opName = op.getName().getStringRef();

    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;

    if (mlir::isa<fcc::fabric::MuxOp>(op)) {
      registry.requireModule("fabric", "fabric_mux.sv");
      continue;
    }

    // Check for Tier 3 transcendental FP ops first.
    if (SVModuleRegistry::isTier3TranscendentalOp(opName) &&
        fpIpProfile.empty()) {
      llvm::errs() << "gen-sv error: unsupported-op: transcendental FP op '"
                   << opName
                   << "' requires --fp-ip-profile; no portable "
                      "synthesizable implementation available.\n";
      return false;
    }
    if (SVModuleRegistry::isKnownOp(opName)) {
      if (!registry.requireArithOp(opName, fpIpProfile)) {
        llvm::errs() << "gen-sv error: unsupported-op: operation '" << opName
                     << "' could not be registered for RTL generation\n";
        return false;
      }
      continue;
    }

    // Unknown ops: report error if they're non-terminator body operations.
    // Yield and block arguments are handled separately, but actual compute
    // ops that SVGen doesn't know how to lower must be rejected.
    llvm::errs() << "gen-sv error: unsupported-op: operation '" << opName
                 << "' inside function_unit '" << fuOp.getSymName()
                 << "' has no known RTL implementation\n";
    return false;
  }
  return true;
}

/// Register pre-written SV module dependencies for a fabric.module body op.
static bool registerOpDeps(mlir::Operation &op, SVModuleRegistry &registry,
                           llvm::StringRef fpIpProfile) {
  if (auto spatialSw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
    registry.requireModule("fabric/spatial_sw", "fabric_spatial_sw.sv");
    registry.requireModule("fabric/spatial_sw", "fabric_spatial_sw_core.sv");
    registry.requireModule("fabric/spatial_sw", "fabric_spatial_sw_decomp.sv");
    return true;
  }
  if (auto temporalSw = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
    registry.requireModule("fabric/temporal_sw", "fabric_temporal_sw.sv");
    registry.requireModule("fabric/temporal_sw",
                           "fabric_temporal_sw_slot_match.sv");
    registry.requireModule("fabric/temporal_sw",
                           "fabric_temporal_sw_arbiter.sv");
    return true;
  }
  if (mlir::isa<fcc::fabric::AddTagOp>(op)) {
    registry.requireModule("fabric", "fabric_add_tag.sv");
    return true;
  }
  if (mlir::isa<fcc::fabric::DelTagOp>(op)) {
    registry.requireModule("fabric", "fabric_del_tag.sv");
    return true;
  }
  if (mlir::isa<fcc::fabric::MapTagOp>(op)) {
    registry.requireModule("fabric", "fabric_map_tag.sv");
    return true;
  }
  if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
    registry.requireModule("fabric", "fabric_fifo.sv");
    return true;
  }
  if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
    registry.requireModule("fabric/memory", "fabric_memory.sv");
    registry.requireModule("fabric/memory", "fabric_memory_lsq.sv");
    registry.requireModule("fabric/memory", "fabric_memory_sram.sv");
    return true;
  }
  if (auto extMem = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
    registry.requireModule("fabric/extmemory", "fabric_extmemory.sv");
    registry.requireModule("fabric/extmemory", "fabric_extmemory_req.sv");
    registry.requireModule("fabric/extmemory", "fabric_extmemory_resp.sv");
    registry.requireModule("common", "fabric_axi_pkg.sv");
    return true;
  }
  if (auto spatialPE = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
    registry.requireModule("fabric/spatial_pe",
                           "fabric_spatial_pe_mux.sv");
    registry.requireModule("fabric/spatial_pe",
                           "fabric_spatial_pe_demux.sv");
    registry.requireModule("fabric/spatial_pe",
                           "fabric_spatial_pe_fu_slot.sv");
    // Recurse into PE body for FU ops.
    for (auto &bodyOp : spatialPE.getBody().front().getOperations()) {
      if (auto fuOp =
              mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(bodyOp)) {
        if (!collectFUDeps(fuOp, registry, fpIpProfile))
          return false;
      }
    }
    return true;
  }
  if (auto temporalPE = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
    // Recurse into PE body for FU ops.
    for (auto &bodyOp : temporalPE.getBody().front().getOperations()) {
      if (auto fuOp =
              mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(bodyOp)) {
        if (!collectFUDeps(fuOp, registry, fpIpProfile))
          return false;
      }
    }
    return true;
  }
  // fabric.instance, fabric.yield, etc.: no direct deps.
  return true;
}

} // namespace

bool generateSV(mlir::ModuleOp adgModule, mlir::MLIRContext *ctx,
                const SVGenOptions &options) {
  // Find the fabric.module in the top-level MLIR module.
  fcc::fabric::ModuleOp fabricMod;
  adgModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod) {
    llvm::errs() << "svgen: no fabric.module found in input MLIR\n";
    return false;
  }

  llvm::outs() << "svgen: generating SystemVerilog for fabric.module @"
               << fabricMod.getSymName() << "\n";

  SVModuleRegistry registry;

  // Always need the config controller.
  registry.requireModule("fabric", "fabric_config_ctrl.sv");

  // --- Pass 1: Collect definitions and dependency set ---
  auto &body = fabricMod.getBody().front();
  for (auto &op : body.getOperations()) {
    if (!registerOpDeps(op, registry, options.fpIpProfile))
      return false;
  }

  // Also walk for FU definitions referenced by fabric.instance ops.
  adgModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    collectFUDeps(fuOp, registry, options.fpIpProfile);
  });

  // --- Pass 2: Generate/Copy ---

  // Create output directory structure.
  std::string outRtlDir = options.outputDir + "/rtl";
  std::string outDesignDir = outRtlDir + "/design";
  std::string outGenDir = outRtlDir + "/generated";
  if (!ensureDir(outDesignDir) || !ensureDir(outGenDir))
    return false;

  // Copy pre-written SV files.
  auto requiredFiles = registry.getRequiredFiles();
  for (const auto &relPath : requiredFiles) {
    std::string srcPath = options.rtlSourceDir + "/design/" + relPath;
    std::string dstPath = outDesignDir + "/" + relPath;

    // Ensure subdirectory exists.
    llvm::SmallString<256> dstDir(dstPath);
    llvm::sys::path::remove_filename(dstDir);
    if (!ensureDir(dstDir))
      return false;

    if (!copyFile(srcPath, dstPath))
      return false;
  }

  llvm::outs() << "svgen: copied " << requiredFiles.size()
               << " pre-written SV files\n";

  // Generate FU body SV files.
  llvm::DenseMap<mlir::Operation *, std::string> peModuleNames;
  std::vector<std::string> generatedFiles;

  // Collect unique FU definitions.
  llvm::StringMap<fcc::fabric::FunctionUnitOp> fuDefs;
  adgModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    fuDefs[fuOp.getSymName()] = fuOp;
  });

  for (auto &entry : fuDefs) {
    auto fuOp = entry.second;
    std::string fuName = SVEmitter::sanitizeName(fuOp.getSymName());
    std::string fileName = "fu_" + fuName + ".sv";
    std::string filePath = outGenDir + "/" + fileName;

    std::error_code ec;
    llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "svgen: cannot write " << filePath << ": "
                   << ec.message() << "\n";
      return false;
    }

    std::string modName =
        generateFUBody(fuOp, os, registry, options.fpIpProfile);
    if (modName.empty()) {
      llvm::errs() << "svgen: failed to generate FU body for "
                   << fuName << "\n";
      return false;
    }
    generatedFiles.push_back("generated/" + fileName);

    llvm::outs() << "svgen: generated FU body: " << fileName << "\n";
  }

  // Generate PE wrapper SV files.
  for (auto &op : body.getOperations()) {
    if (auto spatialPE = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      std::string peName = SVEmitter::sanitizeName(
          spatialPE.getSymName().value_or("spatial_pe"));
      std::string fileName = "pe_" + peName + ".sv";
      std::string filePath = outGenDir + "/" + fileName;

      std::error_code ec;
      llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::errs() << "svgen: cannot write " << filePath << ": "
                     << ec.message() << "\n";
        return false;
      }

      std::string modName =
          generateSpatialPE(spatialPE, os, registry, options.fpIpProfile);
      peModuleNames[&op] = modName;
      generatedFiles.push_back("generated/" + fileName);

      llvm::outs() << "svgen: generated spatial PE: " << fileName << "\n";
    }

    if (auto temporalPE = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      std::string peName = SVEmitter::sanitizeName(
          temporalPE.getSymName().value_or("temporal_pe"));
      std::string fileName = "pe_" + peName + ".sv";
      std::string filePath = outGenDir + "/" + fileName;

      std::error_code ec;
      llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::errs() << "svgen: cannot write " << filePath << ": "
                     << ec.message() << "\n";
        return false;
      }

      std::string modName =
          generateTemporalPE(temporalPE, os, registry, options.fpIpProfile);
      peModuleNames[&op] = modName;
      generatedFiles.push_back("generated/" + fileName);

      llvm::outs() << "svgen: generated temporal PE: " << fileName << "\n";
    }
  }

  // Generate top module.
  {
    std::string topName = SVEmitter::sanitizeName(
        fabricMod.getSymName().str());
    std::string fileName = "fabric_top_" + topName + ".sv";
    std::string filePath = outGenDir + "/" + fileName;

    std::error_code ec;
    llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "svgen: cannot write " << filePath << ": "
                   << ec.message() << "\n";
      return false;
    }

    generateTopModule(fabricMod, os, registry, peModuleNames);
    generatedFiles.push_back("generated/" + fileName);

    llvm::outs() << "svgen: generated top module: " << fileName << "\n";
  }

  // Write filelist.f in compilation order.
  {
    std::string filelistPath = outRtlDir + "/filelist.f";
    std::string content;
    llvm::raw_string_ostream flOs(content);

    flOs << "// Auto-generated filelist for fabric RTL\n";
    flOs << "// Compilation order: packages -> common -> pre-written leaves "
            "-> generated FU/PE -> top\n";
    flOs << "\n";

    // Pre-written files first (already in compilation order from registry).
    for (const auto &relPath : requiredFiles)
      flOs << "design/" << relPath << "\n";

    flOs << "\n";

    // Generated files.
    for (const auto &genFile : generatedFiles)
      flOs << genFile << "\n";

    if (!writeFile(filelistPath, content))
      return false;

    llvm::outs() << "svgen: wrote filelist.f with "
                 << (requiredFiles.size() + generatedFiles.size())
                 << " entries\n";
  }

  llvm::outs() << "svgen: generation complete -> " << outRtlDir << "/\n";
  return true;
}

} // namespace svgen
} // namespace fcc
