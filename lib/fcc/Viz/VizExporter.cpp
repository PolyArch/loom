#include "VizExporterInternal.h"

#include "VizAssets.h"

namespace fcc {
namespace viz_detail {

// ---- String escaping helpers ----

std::string jsonEsc(llvm::StringRef s) {
  std::string r;
  r.reserve(s.size() + 4);
  for (char c : s) {
    if (c == '"') r += "\\\"";
    else if (c == '\\') r += "\\\\";
    else if (c == '\n') r += "\\n";
    else r += c;
  }
  return r;
}

std::string htmlEsc(llvm::StringRef s) {
  std::string r;
  for (char c : s) {
    if (c == '<') r += "&lt;";
    else if (c == '>') r += "&gt;";
    else if (c == '&') r += "&amp;";
    else if (c == '"') r += "&quot;";
    else r += c;
  }
  return r;
}

std::string scriptSafe(const std::string &s) {
  std::string r;
  r.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '/') {
      r += "<\\/";
      ++i;
    } else {
      r += s[i];
    }
  }
  return r;
}

std::string printType(mlir::Type type) {
  if (!type) return "";
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}

// ---- DFG metadata helpers ----

std::string dfgEdgeType(mlir::Type type) {
  if (!type) return "data";
  if (mlir::isa<mlir::MemRefType>(type)) return "memref";
  if (mlir::isa<mlir::NoneType>(type)) return "control";
  return "data";
}

std::string dfgOperandName(mlir::Operation *op, unsigned idx) {
  if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op))
    return load.getOperandName(idx);
  if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op))
    return store.getOperandName(idx);
  if (auto memory = mlir::dyn_cast<circt::handshake::MemoryOp>(op))
    return memory.getOperandName(idx);
  if (auto ext = mlir::dyn_cast<circt::handshake::ExternalMemoryOp>(op))
    return ext.getOperandName(idx);
  if (auto mux = mlir::dyn_cast<circt::handshake::MuxOp>(op))
    return mux.getOperandName(idx);
  if (auto cbr = mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op))
    return cbr.getOperandName(idx);
  if (auto constant = mlir::dyn_cast<circt::handshake::ConstantOp>(op))
    return constant.getOperandName(idx);
  return ("I" + std::to_string(idx));
}

std::string dfgResultName(mlir::Operation *op, unsigned idx) {
  if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op))
    return load.getResultName(idx);
  if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op))
    return store.getResultName(idx);
  if (auto memory = mlir::dyn_cast<circt::handshake::MemoryOp>(op))
    return memory.getResultName(idx);
  if (auto ext = mlir::dyn_cast<circt::handshake::ExternalMemoryOp>(op))
    return ext.getResultName(idx);
  if (auto cbr = mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op))
    return cbr.getResultName(idx);
  if (auto ctrlMerge =
          mlir::dyn_cast<circt::handshake::ControlMergeOp>(op))
    return ctrlMerge.getResultName(idx);
  return ("O" + std::to_string(idx));
}

// ---- Name/title helpers ----

std::string getADGName(mlir::ModuleOp topModule) {
  if (!topModule) return "";
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod) return "";
  return fabricMod.getSymName().str();
}

std::string getDFGName(mlir::ModuleOp topModule) {
  if (!topModule) return "";
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });
  if (!funcOp) return "";
  return funcOp.getName().str();
}

std::string makeVizTitle(mlir::ModuleOp adgModule,
                         mlir::ModuleOp dfgModule,
                         bool hasMapping) {
  std::string adgName = getADGName(adgModule);
  std::string dfgName = getDFGName(dfgModule);
  if (!dfgName.empty() && !adgName.empty())
    return dfgName + (hasMapping ? " on " : " and ") + adgName;
  if (!dfgName.empty()) return dfgName;
  if (!adgName.empty()) return adgName;
  return hasMapping ? "fcc viz (mapped)" : "fcc viz";
}

std::string loadVizSidecarJson(mlir::ModuleOp adgModule,
                                llvm::StringRef adgSourcePath) {
  if (!adgModule)
    return "null";

  fcc::fabric::ModuleOp fabricMod;
  adgModule->walk([&](fcc::fabric::ModuleOp mod) {
    if (!fabricMod)
      fabricMod = mod;
  });
  if (!fabricMod)
    return "null";

  auto pathAttr = fabricMod->getAttrOfType<mlir::StringAttr>("viz_file");
  if (!pathAttr)
    return "null";

  llvm::SmallString<256> resolved(pathAttr.getValue());
  if (!llvm::sys::path::is_absolute(resolved) && !adgSourcePath.empty()) {
    llvm::SmallString<256> baseDir(adgSourcePath);
    llvm::sys::path::remove_filename(baseDir);
    llvm::sys::path::append(baseDir, resolved);
    resolved = baseDir;
  }

  auto buffer = llvm::MemoryBuffer::getFile(resolved);
  if (!buffer) {
    llvm::errs() << "fcc viz: warning: cannot open ADG viz sidecar "
                 << resolved << "\n";
    return "null";
  }
  return (*buffer)->getBuffer().str();
}

std::string loadSiblingSimulationTraceJson(llvm::StringRef mapJsonPath) {
  if (mapJsonPath.empty())
    return "null";
  llvm::SmallString<256> tracePath(mapJsonPath);
  if (!tracePath.ends_with(".map.json"))
    return "null";
  tracePath.resize(tracePath.size() - llvm::StringRef(".map.json").size());
  tracePath += ".sim.trace";
  auto buffer = llvm::MemoryBuffer::getFile(tracePath);
  if (!buffer)
    return "null";
  return (*buffer)->getBuffer().str();
}

// ---- JSON utility ----

unsigned getJsonUnsigned(const llvm::json::Object &obj,
                         llvm::StringRef key,
                         unsigned defaultValue) {
  if (auto v = obj.getInteger(key); v && *v >= 0)
    return static_cast<unsigned>(*v);
  return defaultValue;
}

} // namespace viz_detail

// ---- Public API ----

mlir::LogicalResult exportVizOnly(const std::string &outputPath,
                                  mlir::ModuleOp adgModule,
                                  mlir::ModuleOp dfgModule,
                                  const std::string &adgSourcePath,
                                  VizLayoutMode layoutMode,
                                  mlir::MLIRContext *ctx) {
  using namespace viz_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  std::string title = makeVizTitle(adgModule, dfgModule, false);

  // Serialize data
  std::string adgJson;
  if (adgModule) {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJson(ss, adgModule, ctx);
  } else {
    adgJson = "null";
  }

  std::string dfgJson;
  if (dfgModule) {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJson(ss, dfgModule);
  } else {
    dfgJson = "null";
  }
  std::string adgLayoutJson =
      selectVizLayoutJson(adgModule, adgSourcePath, adgJson, layoutMode);

  // Emit HTML
  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEsc(title) << "</title>\n"
      << "  <style>\n" << viz::RENDERER_CSS << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEsc(title) << "</span>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  // Embedded data
  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const ADG_LAYOUT_DATA = " << scriptSafe(adgLayoutJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n"
      << "const MAPPING_DATA = null;\n"
      << "const SIM_TRACE_DATA = null;\n"
      << "</script>\n\n";

  // Bundled D3 for self-contained local viewing.
  out << "<script>\n" << viz::D3_MIN_JS << "\n</script>\n\n";

  // Renderer JS
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return mlir::success();
}

mlir::LogicalResult exportVizWithMapping(const std::string &outputPath,
                                         mlir::ModuleOp adgModule,
                                         mlir::ModuleOp dfgModule,
                                         const std::string &mapJsonPath,
                                         const std::string &adgSourcePath,
                                         VizLayoutMode layoutMode,
                                         mlir::MLIRContext *ctx) {
  using namespace viz_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  // Read mapping JSON
  std::string mapJson = "null";
  auto mapBuf = llvm::MemoryBuffer::getFile(mapJsonPath);
  if (mapBuf)
    mapJson = (*mapBuf)->getBuffer().str();

  std::string adgJson;
  if (adgModule) {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJson(ss, adgModule, ctx);
  } else {
    adgJson = "null";
  }

  std::string dfgJson;
  if (dfgModule) {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJson(ss, dfgModule);
  } else {
    dfgJson = "null";
  }
  std::string adgLayoutJson =
      selectVizLayoutJson(adgModule, adgSourcePath, adgJson, layoutMode);

  std::string title = makeVizTitle(adgModule, dfgModule, true);
  std::string simTraceJson = loadSiblingSimulationTraceJson(mapJsonPath);

  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEsc(title) << "</title>\n"
      << "  <style>\n" << viz::RENDERER_CSS << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEsc(title) << "</span>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const ADG_LAYOUT_DATA = " << scriptSafe(adgLayoutJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n\n"
      << "const MAPPING_DATA = " << scriptSafe(mapJson) << ";\n"
      << "const SIM_TRACE_DATA = " << scriptSafe(simTraceJson) << ";\n"
      << "</script>\n\n";

  out << "<script>\n" << viz::D3_MIN_JS << "\n</script>\n\n";
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";
  out << "</body>\n</html>\n";

  return mlir::success();
}

} // namespace fcc
