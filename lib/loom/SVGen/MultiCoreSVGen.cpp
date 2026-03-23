#include "loom/SVGen/MultiCoreSVGen.h"
#include "loom/SVGen/SVGen.h"
#include "loom/SVGen/SVEmitter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace loom {
namespace svgen {

namespace {

/// Ensure a directory exists, creating it if needed.
static bool ensureDir(llvm::StringRef dirPath) {
  auto ec = llvm::sys::fs::create_directories(dirPath);
  if (ec) {
    llvm::errs() << "multi-core-svgen: cannot create directory " << dirPath
                 << ": " << ec.message() << "\n";
    return false;
  }
  return true;
}

/// Write a string to a file.
static bool writeFile(llvm::StringRef path, llvm::StringRef content) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "multi-core-svgen: cannot write " << path << ": "
                 << ec.message() << "\n";
    return false;
  }
  os << content;
  return true;
}

/// Emit the system top module ports.
static void emitSystemTopPorts(SVEmitter &emitter,
                               const MultiCoreCompilationDesc &compilation,
                               const MultiCoreSVGenOptions &options) {
  unsigned numCores = compilation.coreDescs.size();

  std::vector<SVPort> ports;

  // Clock and reset.
  ports.push_back({SVPortDir::Input, "logic", "clk"});
  ports.push_back({SVPortDir::Input, "logic", "rst_n"});

  // Global configuration interface.
  ports.push_back({SVPortDir::Input, "logic", "cfg_valid"});
  ports.push_back({SVPortDir::Input, "logic [31:0]", "cfg_addr"});
  ports.push_back({SVPortDir::Input, "logic [31:0]", "cfg_wdata"});
  ports.push_back({SVPortDir::Output, "logic", "cfg_ready"});

  // Per-core external memory AXI interfaces (simplified).
  for (unsigned i = 0; i < numCores; ++i) {
    std::string prefix = "core" + std::to_string(i) + "_axi_";
    ports.push_back({SVPortDir::Output, "logic", prefix + "arvalid"});
    ports.push_back({SVPortDir::Input, "logic", prefix + "arready"});
    ports.push_back({SVPortDir::Output, "logic [31:0]", prefix + "araddr"});
    ports.push_back({SVPortDir::Output, "logic", prefix + "awvalid"});
    ports.push_back({SVPortDir::Input, "logic", prefix + "awready"});
    ports.push_back({SVPortDir::Output, "logic [31:0]", prefix + "awaddr"});
    ports.push_back({SVPortDir::Output, "logic [31:0]", prefix + "wdata"});
    ports.push_back({SVPortDir::Output, "logic", prefix + "wvalid"});
    ports.push_back({SVPortDir::Input, "logic", prefix + "wready"});
    ports.push_back({SVPortDir::Input, "logic [31:0]", prefix + "rdata"});
    ports.push_back({SVPortDir::Input, "logic", prefix + "rvalid"});
    ports.push_back({SVPortDir::Output, "logic", prefix + "rready"});
  }

  // System status.
  ports.push_back({SVPortDir::Output, "logic", "system_idle"});

  emitter.emitModuleHeader("tapestry_system_top", {}, ports);
}

/// Emit localparams for system configuration constants.
static void emitSystemParams(SVEmitter &emitter,
                             const MultiCoreCompilationDesc &compilation,
                             const MultiCoreSVGenOptions &options) {
  unsigned numCores = compilation.coreDescs.size();

  emitter.emitBlankLine();
  emitter.emitComment("System configuration parameters");
  emitter.emitLocalParam("integer", "NUM_CORES", std::to_string(numCores));
  emitter.emitLocalParam("integer", "MESH_ROWS",
                         std::to_string(options.meshRows));
  emitter.emitLocalParam("integer", "MESH_COLS",
                         std::to_string(options.meshCols));
  emitter.emitLocalParam("integer", "NOC_FLIT_WIDTH",
                         std::to_string(options.nocFlitWidth));
  emitter.emitLocalParam("integer", "NOC_NUM_VC",
                         std::to_string(options.nocNumVC));
  emitter.emitLocalParam("integer", "NOC_BUFFER_DEPTH",
                         std::to_string(options.nocBufferDepth));
  emitter.emitLocalParam("integer", "L2_SIZE_BYTES",
                         std::to_string(options.l2SizeBytes));
  emitter.emitLocalParam("integer", "L2_NUM_BANKS",
                         std::to_string(options.l2NumBanks));
  emitter.emitLocalParam("integer", "SPM_SIZE_PER_CORE",
                         std::to_string(options.spmSizePerCore));
}

/// Emit interconnect wires between cores and NoC.
static void emitInterconnectWires(SVEmitter &emitter,
                                  const MultiCoreCompilationDesc &compilation,
                                  const MultiCoreSVGenOptions &options) {
  unsigned numCores = compilation.coreDescs.size();

  emitter.emitBlankLine();
  emitter.emitComment("Per-core configuration demux wires");
  for (unsigned i = 0; i < numCores; ++i) {
    std::string idx = std::to_string(i);
    emitter.emitWire("logic", "cfg_valid_core_" + idx);
    emitter.emitWire("logic [31:0]", "cfg_addr_core_" + idx);
    emitter.emitWire("logic [31:0]", "cfg_wdata_core_" + idx);
    emitter.emitWire("logic", "cfg_ready_core_" + idx);
  }

  emitter.emitBlankLine();
  emitter.emitComment("Per-core idle signals");
  for (unsigned i = 0; i < numCores; ++i) {
    emitter.emitWire("logic", "core_idle_" + std::to_string(i));
  }

  emitter.emitBlankLine();
  emitter.emitComment("NoC local port wires (core-to-router)");
  unsigned flitW = options.nocFlitWidth;
  std::string flitRange = SVEmitter::bitRange(flitW);
  for (unsigned i = 0; i < numCores; ++i) {
    std::string idx = std::to_string(i);
    emitter.emitWire("logic" + flitRange, "noc_core_tx_flit_" + idx);
    emitter.emitWire("logic", "noc_core_tx_valid_" + idx);
    emitter.emitWire("logic", "noc_core_tx_ready_" + idx);
    emitter.emitWire("logic" + flitRange, "noc_core_rx_flit_" + idx);
    emitter.emitWire("logic", "noc_core_rx_valid_" + idx);
    emitter.emitWire("logic", "noc_core_rx_ready_" + idx);
  }

  emitter.emitBlankLine();
  emitter.emitComment("Memory hierarchy wires");
  for (unsigned i = 0; i < numCores; ++i) {
    std::string idx = std::to_string(i);
    emitter.emitWire("logic", "mem_req_valid_" + idx);
    emitter.emitWire("logic", "mem_req_ready_" + idx);
    emitter.emitWire("logic [31:0]", "mem_req_addr_" + idx);
    emitter.emitWire("logic [31:0]", "mem_req_wdata_" + idx);
    emitter.emitWire("logic", "mem_req_wen_" + idx);
    emitter.emitWire("logic", "mem_resp_valid_" + idx);
    emitter.emitWire("logic [31:0]", "mem_resp_rdata_" + idx);
  }
}

/// Emit configuration address demux logic.
static void emitConfigDemux(SVEmitter &emitter,
                            const MultiCoreCompilationDesc &compilation) {
  unsigned numCores = compilation.coreDescs.size();

  emitter.emitBlankLine();
  emitter.emitComment(
      "Configuration address demux: route config transactions "
      "to the target core based on upper address bits");

  // Upper bits [31:24] select the core, lower bits [23:0] are the
  // core-local address.
  for (unsigned i = 0; i < numCores; ++i) {
    std::string idx = std::to_string(i);
    emitter.emitAssign("cfg_valid_core_" + idx,
                       "cfg_valid && (cfg_addr[31:24] == 8'd" + idx + ")");
    emitter.emitAssign("cfg_addr_core_" + idx, "{8'b0, cfg_addr[23:0]}");
    emitter.emitAssign("cfg_wdata_core_" + idx, "cfg_wdata");
  }

  // OR-reduce ready signals.
  std::string readyExpr;
  for (unsigned i = 0; i < numCores; ++i) {
    if (i > 0)
      readyExpr += " | ";
    readyExpr += "cfg_ready_core_" + std::to_string(i);
  }
  emitter.emitAssign("cfg_ready", readyExpr);
}

/// Emit per-core fabric_top instances.
static void emitCoreInstances(SVEmitter &emitter,
                              const MultiCoreCompilationDesc &compilation,
                              const MultiCoreSVGenOptions &options) {
  emitter.emitBlankLine();
  emitter.emitComment("Per-core CGRA fabric instances");

  for (unsigned i = 0; i < compilation.coreDescs.size(); ++i) {
    const auto &core = compilation.coreDescs[i];
    std::string instName = SVEmitter::sanitizeName(core.coreInstanceName);
    std::string moduleName =
        "fabric_top_" + SVEmitter::sanitizeName(core.coreType);
    std::string idx = std::to_string(i);

    emitter.emitBlankLine();
    emitter.emitComment("Core instance: " + core.coreInstanceName +
                        " (type: " + core.coreType + ")");

    std::vector<SVConnection> conns;
    conns.push_back({"clk", "clk"});
    conns.push_back({"rst_n", "rst_n"});

    // Config interface.
    conns.push_back({"cfg_valid", "cfg_valid_core_" + idx});
    conns.push_back({"cfg_addr", "cfg_addr_core_" + idx});
    conns.push_back({"cfg_wdata", "cfg_wdata_core_" + idx});
    conns.push_back({"cfg_ready", "cfg_ready_core_" + idx});

    // NoC local port.
    conns.push_back({"noc_tx_flit", "noc_core_tx_flit_" + idx});
    conns.push_back({"noc_tx_valid", "noc_core_tx_valid_" + idx});
    conns.push_back({"noc_tx_ready", "noc_core_tx_ready_" + idx});
    conns.push_back({"noc_rx_flit", "noc_core_rx_flit_" + idx});
    conns.push_back({"noc_rx_valid", "noc_core_rx_valid_" + idx});
    conns.push_back({"noc_rx_ready", "noc_core_rx_ready_" + idx});

    // Memory interface.
    conns.push_back({"mem_req_valid", "mem_req_valid_" + idx});
    conns.push_back({"mem_req_ready", "mem_req_ready_" + idx});
    conns.push_back({"mem_req_addr", "mem_req_addr_" + idx});
    conns.push_back({"mem_req_wdata", "mem_req_wdata_" + idx});
    conns.push_back({"mem_req_wen", "mem_req_wen_" + idx});
    conns.push_back({"mem_resp_valid", "mem_resp_valid_" + idx});
    conns.push_back({"mem_resp_rdata", "mem_resp_rdata_" + idx});

    // AXI passthrough.
    conns.push_back({"axi_arvalid", "core" + idx + "_axi_arvalid"});
    conns.push_back({"axi_arready", "core" + idx + "_axi_arready"});
    conns.push_back({"axi_araddr", "core" + idx + "_axi_araddr"});
    conns.push_back({"axi_awvalid", "core" + idx + "_axi_awvalid"});
    conns.push_back({"axi_awready", "core" + idx + "_axi_awready"});
    conns.push_back({"axi_awaddr", "core" + idx + "_axi_awaddr"});
    conns.push_back({"axi_wdata", "core" + idx + "_axi_wdata"});
    conns.push_back({"axi_wvalid", "core" + idx + "_axi_wvalid"});
    conns.push_back({"axi_wready", "core" + idx + "_axi_wready"});
    conns.push_back({"axi_rdata", "core" + idx + "_axi_rdata"});
    conns.push_back({"axi_rvalid", "core" + idx + "_axi_rvalid"});
    conns.push_back({"axi_rready", "core" + idx + "_axi_rready"});

    // Idle output.
    conns.push_back({"idle", "core_idle_" + idx});

    emitter.emitInstance(moduleName, "u_" + instName, {}, conns);
  }
}

/// Emit the NoC mesh instantiation.
static void emitNoCMesh(SVEmitter &emitter,
                        const MultiCoreCompilationDesc &compilation,
                        const MultiCoreSVGenOptions &options) {
  unsigned numCores = compilation.coreDescs.size();

  emitter.emitBlankLine();
  emitter.emitComment("NoC mesh instantiation");

  // Declare aggregate wire arrays for the NoC mesh local ports.
  unsigned flitW = options.nocFlitWidth;
  std::string flitRange = SVEmitter::bitRange(flitW);
  emitter.emitRaw("logic" + flitRange + " noc_local_tx_flit [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic noc_local_tx_valid [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic noc_local_tx_ready [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic" + flitRange + " noc_local_rx_flit [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic noc_local_rx_valid [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic noc_local_rx_ready [0:" +
                  std::to_string(numCores - 1) + "];\n");

  emitter.emitBlankLine();
  emitter.emitComment("Wire per-core NoC signals to aggregate arrays");

  for (unsigned i = 0; i < numCores; ++i) {
    std::string idx = std::to_string(i);
    emitter.emitAssign("noc_local_tx_flit[" + idx + "]",
                       "noc_core_tx_flit_" + idx);
    emitter.emitAssign("noc_local_tx_valid[" + idx + "]",
                       "noc_core_tx_valid_" + idx);
    emitter.emitAssign("noc_core_tx_ready_" + idx,
                       "noc_local_tx_ready[" + idx + "]");
    emitter.emitAssign("noc_core_rx_flit_" + idx,
                       "noc_local_rx_flit[" + idx + "]");
    emitter.emitAssign("noc_core_rx_valid_" + idx,
                       "noc_local_rx_valid[" + idx + "]");
    emitter.emitAssign("noc_local_rx_ready[" + idx + "]",
                       "noc_core_rx_ready_" + idx);
  }

  emitter.emitBlankLine();

  // Instantiate the NoC mesh top module.
  std::vector<std::string> nocParams;
  nocParams.push_back(std::to_string(options.meshRows));
  nocParams.push_back(std::to_string(options.meshCols));
  nocParams.push_back(std::to_string(options.nocFlitWidth));
  nocParams.push_back(std::to_string(options.nocNumVC));
  nocParams.push_back(std::to_string(options.nocBufferDepth));

  std::vector<SVConnection> nocConns;
  nocConns.push_back({"clk", "clk"});
  nocConns.push_back({"rst_n", "rst_n"});
  nocConns.push_back({"local_tx_flit", "noc_local_tx_flit"});
  nocConns.push_back({"local_tx_valid", "noc_local_tx_valid"});
  nocConns.push_back({"local_tx_ready", "noc_local_tx_ready"});
  nocConns.push_back({"local_rx_flit", "noc_local_rx_flit"});
  nocConns.push_back({"local_rx_valid", "noc_local_rx_valid"});
  nocConns.push_back({"local_rx_ready", "noc_local_rx_ready"});

  emitter.emitInstance("noc_mesh_top", "u_noc_mesh", nocParams, nocConns);
}

/// Emit the memory hierarchy instantiation.
static void emitMemoryHierarchy(SVEmitter &emitter,
                                const MultiCoreCompilationDesc &compilation,
                                const MultiCoreSVGenOptions &options) {
  unsigned numCores = compilation.coreDescs.size();

  emitter.emitBlankLine();
  emitter.emitComment("Memory hierarchy instantiation");

  // Declare aggregate wire arrays for memory ports.
  emitter.emitRaw("logic mem_req_valid_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic mem_req_ready_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic [31:0] mem_req_addr_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic [31:0] mem_req_wdata_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic mem_req_wen_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic mem_resp_valid_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");
  emitter.emitRaw("logic [31:0] mem_resp_rdata_arr [0:" +
                  std::to_string(numCores - 1) + "];\n");

  emitter.emitBlankLine();
  emitter.emitComment("Wire per-core memory signals to aggregate arrays");

  for (unsigned i = 0; i < numCores; ++i) {
    std::string idx = std::to_string(i);
    emitter.emitAssign("mem_req_valid_arr[" + idx + "]",
                       "mem_req_valid_" + idx);
    emitter.emitAssign("mem_req_ready_" + idx,
                       "mem_req_ready_arr[" + idx + "]");
    emitter.emitAssign("mem_req_addr_arr[" + idx + "]",
                       "mem_req_addr_" + idx);
    emitter.emitAssign("mem_req_wdata_arr[" + idx + "]",
                       "mem_req_wdata_" + idx);
    emitter.emitAssign("mem_req_wen_arr[" + idx + "]",
                       "mem_req_wen_" + idx);
    emitter.emitAssign("mem_resp_valid_" + idx,
                       "mem_resp_valid_arr[" + idx + "]");
    emitter.emitAssign("mem_resp_rdata_" + idx,
                       "mem_resp_rdata_arr[" + idx + "]");
  }

  emitter.emitBlankLine();

  // Instantiate the memory hierarchy top module.
  std::vector<std::string> memParams;
  memParams.push_back(std::to_string(numCores));
  memParams.push_back(std::to_string(options.l2SizeBytes));
  memParams.push_back(std::to_string(options.l2NumBanks));
  memParams.push_back(std::to_string(options.spmSizePerCore));

  std::vector<SVConnection> memConns;
  memConns.push_back({"clk", "clk"});
  memConns.push_back({"rst_n", "rst_n"});
  memConns.push_back({"req_valid", "mem_req_valid_arr"});
  memConns.push_back({"req_ready", "mem_req_ready_arr"});
  memConns.push_back({"req_addr", "mem_req_addr_arr"});
  memConns.push_back({"req_wdata", "mem_req_wdata_arr"});
  memConns.push_back({"req_wen", "mem_req_wen_arr"});
  memConns.push_back({"resp_valid", "mem_resp_valid_arr"});
  memConns.push_back({"resp_rdata", "mem_resp_rdata_arr"});

  emitter.emitInstance("tapestry_mem_top", "u_mem_hierarchy", memParams,
                       memConns);
}

/// Emit system-level idle logic.
static void emitSystemIdle(SVEmitter &emitter,
                           const MultiCoreCompilationDesc &compilation) {
  unsigned numCores = compilation.coreDescs.size();

  emitter.emitBlankLine();
  emitter.emitComment("System idle: all cores must be idle");
  std::string idleExpr;
  for (unsigned i = 0; i < numCores; ++i) {
    if (i > 0)
      idleExpr += " & ";
    idleExpr += "core_idle_" + std::to_string(i);
  }
  emitter.emitAssign("system_idle", idleExpr);
}

/// Emit the complete system top module.
static bool emitSystemTop(const std::string &path,
                          const MultiCoreCompilationDesc &compilation,
                          const MultiCoreSVGenOptions &options) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "multi-core-svgen: cannot write " << path << ": "
                 << ec.message() << "\n";
    return false;
  }

  SVEmitter emitter(os);
  emitter.emitFileHeader("tapestry_system_top");

  emitSystemTopPorts(emitter, compilation, options);
  emitSystemParams(emitter, compilation, options);
  emitInterconnectWires(emitter, compilation, options);
  emitConfigDemux(emitter, compilation);
  emitCoreInstances(emitter, compilation, options);
  emitNoCMesh(emitter, compilation, options);
  emitMemoryHierarchy(emitter, compilation, options);
  emitSystemIdle(emitter, compilation);

  emitter.emitBlankLine();
  emitter.emitModuleFooter("tapestry_system_top");

  return true;
}

/// Generate the system filelist that includes all sub-filelists and system
/// RTL files.
static bool emitSystemFilelist(const std::string &path,
                               const std::vector<std::string> &perCoreFilelists,
                               const std::string &rtlSourceDir,
                               const std::string &outputDir,
                               const MultiCoreSVGenOptions &options) {
  std::string content;
  llvm::raw_string_ostream flOs(content);

  flOs << "// System-level filelist for multi-core Tapestry RTL\n";
  flOs << "// Compilation order: packages -> NoC -> memory -> per-core -> "
          "system top\n";
  flOs << "\n";

  // NoC RTL files (from pre-written RTL source).
  flOs << "// NoC RTL\n";
  flOs << rtlSourceDir << "/design/noc/noc_pkg.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_router.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_input_port.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_output_port.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_crossbar.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_vc_allocator.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_sw_allocator.sv\n";
  flOs << rtlSourceDir << "/design/noc/noc_mesh_top.sv\n";
  flOs << "\n";

  // Memory controller RTL files (from pre-written RTL source).
  flOs << "// Memory hierarchy RTL\n";
  flOs << rtlSourceDir << "/design/memory_ctrl/mem_ctrl_pkg.sv\n";
  flOs << rtlSourceDir << "/design/memory_ctrl/tapestry_spm.sv\n";
  flOs << rtlSourceDir << "/design/memory_ctrl/tapestry_l2_bank.sv\n";
  flOs << rtlSourceDir << "/design/memory_ctrl/tapestry_l2_arbiter.sv\n";
  flOs << rtlSourceDir << "/design/memory_ctrl/tapestry_dma_engine.sv\n";
  flOs << rtlSourceDir << "/design/memory_ctrl/tapestry_mem_top.sv\n";
  flOs << "\n";

  // Per-core filelists (included via -f).
  flOs << "// Per-core fabric filelists\n";
  for (const auto &fl : perCoreFilelists) {
    flOs << "-f " << fl << "\n";
  }
  flOs << "\n";

  // System top module.
  flOs << "// System top\n";
  flOs << outputDir << "/tapestry_system_top.sv\n";

  return writeFile(path, content);
}

} // namespace

MultiCoreSVGenResult
generateMultiCoreSV(const MultiCoreCompilationDesc &compilation,
                    const MultiCoreSVGenOptions &options,
                    mlir::MLIRContext *ctx) {
  MultiCoreSVGenResult result;
  result.success = false;

  llvm::outs() << "multi-core-svgen: generating system-level RTL for "
               << compilation.coreDescs.size() << " cores\n";

  if (compilation.coreDescs.empty()) {
    llvm::errs() << "multi-core-svgen: no core descriptions in compilation\n";
    return result;
  }

  // Ensure output directory exists.
  if (!ensureDir(options.outputDir))
    return result;

  // Generate per-core RTL using existing single-core SVGen.
  for (const auto &coreDesc : compilation.coreDescs) {
    std::string coreOutputDir =
        options.outputDir + "/" +
        SVEmitter::sanitizeName(coreDesc.coreInstanceName);

    if (!ensureDir(coreOutputDir))
      return result;

    SVGenOptions coreOpts;
    coreOpts.rtlSourceDir = options.rtlSourceDir;
    coreOpts.outputDir = coreOutputDir;
    coreOpts.fpIpProfile = options.fpIpProfile;

    llvm::outs() << "multi-core-svgen: generating core RTL for "
                 << coreDesc.coreInstanceName << " -> " << coreOutputDir
                 << "\n";

    if (!generateSV(coreDesc.adgModule, ctx, coreOpts)) {
      llvm::errs() << "multi-core-svgen: failed to generate RTL for core "
                   << coreDesc.coreInstanceName << "\n";
      return result;
    }

    std::string coreFilelist = coreOutputDir + "/rtl/filelist.f";
    result.perCoreFilelists.push_back(coreFilelist);
    result.allGeneratedFiles.push_back(coreFilelist);
  }

  // Generate system top module.
  result.systemTopFile = options.outputDir + "/tapestry_system_top.sv";
  if (!emitSystemTop(result.systemTopFile, compilation, options))
    return result;

  result.allGeneratedFiles.push_back(result.systemTopFile);
  llvm::outs() << "multi-core-svgen: generated system top module -> "
               << result.systemTopFile << "\n";

  // Generate system filelist.
  result.systemFilelistFile = options.outputDir + "/system_filelist.f";
  if (!emitSystemFilelist(result.systemFilelistFile, result.perCoreFilelists,
                          options.rtlSourceDir, options.outputDir, options))
    return result;

  result.allGeneratedFiles.push_back(result.systemFilelistFile);
  llvm::outs() << "multi-core-svgen: generated system filelist -> "
               << result.systemFilelistFile << "\n";

  result.success = true;
  llvm::outs() << "multi-core-svgen: generation complete ("
               << result.allGeneratedFiles.size() << " files)\n";

  return result;
}

} // namespace svgen
} // namespace loom
