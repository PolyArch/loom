#include "fcc/Viz/VizExporter.h"

#include "VizAssets.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace fcc {

// ---- helpers ----

static std::string jsonEsc(llvm::StringRef s) {
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

static std::string htmlEsc(llvm::StringRef s) {
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

static std::string scriptSafe(const std::string &s) {
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

// ---- Serialize fabric.module to JSON ----

static void writeADGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule,
                          mlir::MLIRContext *ctx) {
  // Find fabric.module
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });

  os << "{\n";

  if (!fabricMod) {
    os << "  \"module\": null\n}";
    return;
  }

  // Module signature
  auto fnType = fabricMod.getFunctionType();
  os << "  \"name\": \"" << jsonEsc(fabricMod.getSymName().str())
     << "\",\n";
  os << "  \"numInputs\": " << fnType.getNumInputs()
     << ", \"numOutputs\": " << fnType.getNumResults() << ",\n";

  // Collect PE/SW definitions from both top-level module and fabric.module.
  // Definitions may be outside fabric.module (referenced by instances inside).
  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefMap;
  llvm::StringMap<fcc::fabric::SpatialSwOp> swDefMap;
  topModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    auto name = peOp.getSymName();
    if (name) peDefMap[*name] = peOp;
  });
  topModule->walk([&](fcc::fabric::SpatialSwOp swOp) {
    auto name = swOp.getSymName();
    if (name) swDefMap[*name] = swOp;
  });

  // Helper: emit FU details for a PE definition.
  // Extracts full SSA connectivity: inputEdges (arg->op), edges (op->op),
  // outputEdges (op/arg->yield output).
  auto emitPEFUs = [&](fcc::fabric::SpatialPEOp peOp) {
    os << ", \"fus\": [";
    bool firstFU = true;
    auto &peBody = peOp.getBody().front();
    for (auto &innerOp : peBody.getOperations()) {
      auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(innerOp);
      if (!fuOp) continue;

      if (!firstFU) os << ", ";
      firstFU = false;

      auto fuFnType = fuOp.getFunctionType();
      os << "{\"name\": \"" << jsonEsc(fuOp.getSymName().str()) << "\"";
      os << ", \"numIn\": " << fuFnType.getNumInputs();
      os << ", \"numOut\": " << fuFnType.getNumResults();

      // Map Value -> index: block args are negative (-1 - argIdx), ops are 0+
      llvm::DenseMap<mlir::Value, int> valToIdx;
      for (auto arg : fuOp.getBody().front().getArguments())
        valToIdx[arg] = -1 - static_cast<int>(arg.getArgNumber());

      llvm::SmallVector<std::pair<int,int>, 4> dagEdges;    // op -> op
      llvm::SmallVector<std::pair<int,int>, 4> inputEdges;  // argIdx -> opIdx
      llvm::SmallVector<std::pair<int,int>, 4> outputEdges; // valIdx -> yieldIdx

      os << ", \"ops\": [";
      bool firstOp = true;
      int opIdx = 0;

      for (auto &bodyOp : fuOp.getBody().front().getOperations()) {
        if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(bodyOp)) {
          // Track yield operands -> output connections
          for (unsigned yi = 0; yi < yieldOp.getNumOperands(); ++yi) {
            auto it = valToIdx.find(yieldOp.getOperand(yi));
            if (it != valToIdx.end())
              outputEdges.push_back({it->second, static_cast<int>(yi)});
          }
          continue;
        }

        if (!firstOp) os << ", ";
        firstOp = false;
        os << "\"" << jsonEsc(bodyOp.getName().getStringRef().str()) << "\"";

        // Track operand sources
        for (auto operand : bodyOp.getOperands()) {
          auto it = valToIdx.find(operand);
          if (it != valToIdx.end()) {
            if (it->second >= 0)
              dagEdges.push_back({it->second, opIdx});  // op -> op
            else
              inputEdges.push_back({-(it->second + 1), opIdx}); // arg -> op
          }
        }
        for (auto result : bodyOp.getResults())
          valToIdx[result] = opIdx;
        opIdx++;
      }
      os << "]";

      // op-to-op edges
      if (!dagEdges.empty()) {
        os << ", \"edges\": [";
        for (size_t k = 0; k < dagEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << dagEdges[k].first << ", " << dagEdges[k].second << "]";
        }
        os << "]";
      }

      // input arg -> op edges
      if (!inputEdges.empty()) {
        os << ", \"inputEdges\": [";
        for (size_t k = 0; k < inputEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << inputEdges[k].first << ", " << inputEdges[k].second << "]";
        }
        os << "]";
      }

      // op/arg -> yield output edges
      if (!outputEdges.empty()) {
        os << ", \"outputEdges\": [";
        for (size_t k = 0; k < outputEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << outputEdges[k].first << ", " << outputEdges[k].second << "]";
        }
        os << "]";
      }

      // Generate DOT: full FU internal DAG with I/O port nodes.
      // Graphviz handles ALL layout including port-to-op connections.
      if (opIdx > 0) {
        os << ", \"dot\": \"digraph FU {\\n";
        os << "  rankdir=TB;\\n";
        os << "  bgcolor=\\\"transparent\\\";\\n";
        os << "  node [style=filled, fontsize=9, "
           << "fontname=\\\"monospace\\\", fontcolor=\\\"#c8d6e5\\\"];\\n";
        os << "  edge [color=\\\"#6a8faf\\\", penwidth=1.2, "
           << "arrowsize=0.6];\\n";
        // Input port nodes
        for (unsigned ai = 0; ai < fuFnType.getNumInputs(); ++ai) {
          os << "  in" << ai
             << " [label=\\\"I" << ai << "\\\", shape=square, "
             << "width=0.2, height=0.2, fontsize=7, "
             << "fillcolor=\\\"#2a5f5f\\\", color=\\\"#4ecdc4\\\"];\\n";
        }
        // Op nodes
        {
          int oi2 = 0;
          for (auto &bodyOp2 : fuOp.getBody().front().getOperations()) {
            if (mlir::isa<fcc::fabric::YieldOp>(bodyOp2)) continue;
            std::string on = bodyOp2.getName().getStringRef().str();
            bool isMux = (on.find("static_mux") != std::string::npos);
            std::string displayName = on;
            auto dotPos = on.find('.');
            if (dotPos != std::string::npos)
              displayName = on.substr(0, dotPos) + "\\n" + on.substr(dotPos + 1);
            os << "  op" << oi2
               << " [label=\\\"" << jsonEsc(displayName) << "\\\", ";
            if (isMux)
              os << "shape=invtrapezium, fillcolor=\\\"#3a3520\\\", "
                 << "color=\\\"#ffd166\\\", fontcolor=\\\"#ffd166\\\"];\\n";
            else
              os << "shape=ellipse, fillcolor=\\\"#1a3050\\\", "
                 << "color=\\\"#5dade2\\\"];\\n";
            oi2++;
          }
        }
        // Output port nodes
        for (unsigned yi = 0; yi < fuFnType.getNumResults(); ++yi) {
          os << "  out" << yi
             << " [label=\\\"O" << yi << "\\\", shape=square, "
             << "width=0.2, height=0.2, fontsize=7, "
             << "fillcolor=\\\"#5f2a1a\\\", color=\\\"#ff6b35\\\"];\\n";
        }
        // Rank constraints
        os << "  { rank=source; ";
        for (unsigned ai = 0; ai < fuFnType.getNumInputs(); ++ai)
          os << "in" << ai << "; ";
        os << "}\\n";
        os << "  { rank=sink; ";
        for (unsigned yi = 0; yi < fuFnType.getNumResults(); ++yi)
          os << "out" << yi << "; ";
        os << "}\\n";
        // Input -> op edges
        for (auto &ie : inputEdges)
          os << "  in" << ie.first << " -> op" << ie.second
             << " [color=\\\"#4ecdc4\\\"];\\n";
        // Op -> op edges
        for (auto &de : dagEdges)
          os << "  op" << de.first << " -> op" << de.second << ";\\n";
        // Op/arg -> output edges
        for (auto &oe : outputEdges) {
          if (oe.first >= 0)
            os << "  op" << oe.first << " -> out" << oe.second
               << " [color=\\\"#ff6b35\\\"];\\n";
          else
            os << "  in" << (-(oe.first + 1)) << " -> out" << oe.second
               << " [style=dashed, color=\\\"#888888\\\"];\\n";
        }
        os << "}\\n\"";
      }

      os << "}";
    }
    os << "]";
  };

  os << "  \"components\": [\n";
  bool first = true;
  auto &body = fabricMod.getBody().front();

  for (auto &op : body.getOperations()) {
    // spatial_pe definitions inside fabric.module (inline, not via instance)
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      auto peFnType = peOp.getFunctionType();
      os << "    {\"kind\": \"spatial_pe\", \"name\": \""
         << jsonEsc(peOp.getSymName().value_or("pe").str()) << "\"";
      os << ", \"numInputs\": " << peFnType.getNumInputs();
      os << ", \"numOutputs\": " << peFnType.getNumResults();
      emitPEFUs(peOp);
      os << "}";
    }

    // spatial_sw definitions
    if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      auto swFnType = swOp.getFunctionType();
      os << "    {\"kind\": \"spatial_sw\", \"name\": \""
         << jsonEsc(swOp.getSymName().value_or("sw").str()) << "\"";
      os << ", \"numInputs\": " << swFnType.getNumInputs();
      os << ", \"numOutputs\": " << swFnType.getNumResults();
      os << "}";
    }

    // instances - resolve to PE/SW definitions for FU details
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      if (!first) os << ",\n";
      first = false;

      auto moduleName = instOp.getModule();
      auto peIt = peDefMap.find(moduleName);
      auto swIt = swDefMap.find(moduleName);

      if (peIt != peDefMap.end()) {
        // Instance of a spatial_pe - emit as PE with FU details
        auto peOp = peIt->second;
        auto peFnType = peOp.getFunctionType();
        os << "    {\"kind\": \"spatial_pe\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("pe").str()) << "\"";
        os << ", \"defName\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << peFnType.getNumInputs();
        os << ", \"numOutputs\": " << peFnType.getNumResults();
        emitPEFUs(peOp);
        os << "}";
      } else if (swIt != swDefMap.end()) {
        // Instance of a spatial_sw
        auto swOp = swIt->second;
        auto swFnType = swOp.getFunctionType();
        os << "    {\"kind\": \"spatial_sw\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("sw").str()) << "\"";
        os << ", \"numInputs\": " << swFnType.getNumInputs();
        os << ", \"numOutputs\": " << swFnType.getNumResults();
        os << "}";
      } else {
        // Generic instance
        os << "    {\"kind\": \"instance\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("inst").str()) << "\"";
        os << ", \"module\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << instOp.getNumOperands();
        os << ", \"numOutputs\": " << instOp.getNumResults();
        os << "}";
      }
    }
  }

  os << "\n  ],\n";

  // Connections: trace SSA (two-pass for graph-region circular references).
  os << "  \"connections\": [\n";
  bool firstConn = true;

  llvm::DenseMap<mlir::Value, int> blockArgIdx;
  for (auto arg : body.getArguments())
    blockArgIdx[arg] = static_cast<int>(arg.getArgNumber());

  // Pass 1: collect ALL instance results first (handles forward references)
  struct InstResult { std::string name; unsigned idx; };
  llvm::DenseMap<mlir::Value, InstResult> instResultMap;
  for (auto &op : body.getOperations()) {
    auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op);
    if (!instOp) continue;
    std::string instName = instOp.getSymName().value_or("inst").str();
    for (unsigned i = 0; i < instOp.getNumResults(); ++i)
      instResultMap[instOp.getResult(i)] = {instName, i};
  }

  // Pass 2: trace all operand connections
  for (auto &op : body.getOperations()) {
    auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op);
    if (!instOp) continue;
    std::string instName = instOp.getSymName().value_or("inst").str();

    for (unsigned i = 0; i < instOp.getNumOperands(); ++i) {
      auto operand = instOp.getOperand(i);
      // Module input -> instance
      auto argIt = blockArgIdx.find(operand);
      if (argIt != blockArgIdx.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"module_in\", \"fromIdx\": " << argIt->second
           << ", \"to\": \"" << jsonEsc(instName) << "\", \"toIdx\": " << i
           << "}";
      }
      // Instance -> instance (now works for circular refs too)
      auto irIt = instResultMap.find(operand);
      if (irIt != instResultMap.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"" << jsonEsc(irIt->second.name)
           << "\", \"fromIdx\": " << irIt->second.idx
           << ", \"to\": \"" << jsonEsc(instName) << "\", \"toIdx\": " << i
           << "}";
      }
    }
  }

  // Yield: instance results -> module outputs
  auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(body.getTerminator());
  if (yieldOp) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
      auto ir = instResultMap.find(yieldOp->getOperand(i));
      if (ir != instResultMap.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"" << jsonEsc(ir->second.name)
           << "\", \"fromIdx\": " << ir->second.idx
           << ", \"to\": \"module_out\", \"toIdx\": " << i << "}";
      }
    }
  }

  os << "\n  ]\n}";
}

// ---- Serialize handshake.func DFG to DOT + JSON ----

static void writeDFGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule) {
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });

  os << "{\n";
  if (!funcOp) {
    os << "  \"dot\": null, \"nodes\": [], \"edges\": []\n}";
    return;
  }

  // Build DOT
  std::string dot;
  llvm::raw_string_ostream ds(dot);
  ds << "digraph DFG {\\n";
  ds << "  rankdir=TB;\\n";
  ds << "  node [style=filled, fontsize=11];\\n";
  ds << "  edge [color=\\\"#555555\\\"];\\n\\n";

  auto &body = funcOp.getBody().front();
  llvm::DenseMap<mlir::Value, std::pair<int, unsigned>> valToPort;

  // Nodes
  int nodeIdx = 0;
  std::string srcRank, sinkRank;

  // Block arguments as inputs
  for (auto arg : body.getArguments()) {
    std::string nid = "n" + std::to_string(nodeIdx);
    std::string label = "arg" + std::to_string(arg.getArgNumber());
    ds << "  \\\"" << nid << "\\\" [label=\\\"" << label
       << "\\\", shape=invtriangle, fillcolor=\\\"#90caf9\\\"];\\n";
    srcRank += "\\\"" + nid + "\\\"; ";
    valToPort[arg] = {nodeIdx, 0};
    nodeIdx++;
  }

  // Operations
  for (auto &op : body.getOperations()) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      // return/end node
      std::string nid = "n" + std::to_string(nodeIdx);
      ds << "  \\\"" << nid << "\\\" [label=\\\"return\\\", shape=triangle, "
            "fillcolor=\\\"#ef9a9a\\\"];\\n";
      sinkRank += "\\\"" + nid + "\\\"; ";
      // edges from operands
      for (auto operand : op.getOperands()) {
        auto it = valToPort.find(operand);
        if (it != valToPort.end()) {
          ds << "  \\\"n" << it->second.first << "\\\" -> \\\"" << nid
             << "\\\";\\n";
        }
      }
      nodeIdx++;
      continue;
    }

    std::string nid = "n" + std::to_string(nodeIdx);
    std::string opName = op.getName().getStringRef().str();

    // Split "dialect.op" into "dialect\nop" for two-line label
    std::string dfgDisplayName = opName;
    auto dfgDotPos = opName.find('.');
    if (dfgDotPos != std::string::npos)
      dfgDisplayName = opName.substr(0, dfgDotPos) + "\\n" + opName.substr(dfgDotPos + 1);

    // Color by dialect
    std::string color = "#e0e0e0";
    if (opName.find("arith.") == 0) color = "#bbdefb";
    else if (opName.find("handshake.") == 0) color = "#fff9c4";
    else if (opName.find("dataflow.") == 0) color = "#c8e6c9";

    ds << "  \\\"" << nid << "\\\" [label=\\\"" << dfgDisplayName
       << "\\\", shape=ellipse, fillcolor=\\\"" << color << "\\\"];\\n";

    // Edges from operands
    for (auto operand : op.getOperands()) {
      auto it = valToPort.find(operand);
      if (it != valToPort.end()) {
        ds << "  \\\"n" << it->second.first << "\\\" -> \\\"" << nid
           << "\\\";\\n";
      }
    }

    // Map results
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      valToPort[op.getResult(i)] = {nodeIdx, i};

    nodeIdx++;
  }

  if (!srcRank.empty())
    ds << "\\n  { rank=source; " << srcRank << "}\\n";
  if (!sinkRank.empty())
    ds << "  { rank=sink; " << sinkRank << "}\\n";
  ds << "}\\n";

  os << "  \"dot\": \"" << dot << "\"\n}";
}

// ---- Public API ----

mlir::LogicalResult exportVizOnly(const std::string &outputPath,
                                  mlir::ModuleOp adgModule,
                                  mlir::ModuleOp dfgModule,
                                  mlir::MLIRContext *ctx) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  std::string title = "fcc viz";

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
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  // Embedded data
  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n"
      << "const MAPPING_DATA = null;\n"
      << "</script>\n\n";

  // D3.js + Graphviz WASM from CDN
  out << "<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n";
  out << "<script src=\"https://unpkg.com/@viz-js/"
         "viz@3.2.4/lib/viz-standalone.js\"></script>\n\n";

  // Renderer JS
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return mlir::success();
}

mlir::LogicalResult exportVizWithMapping(const std::string &outputPath,
                                         mlir::ModuleOp adgModule,
                                         mlir::ModuleOp dfgModule,
                                         const std::string &mapJsonPath,
                                         mlir::MLIRContext *ctx) {
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

  std::string title = "fcc viz (mapped)";

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
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n\n"
      << "const MAPPING_DATA = " << scriptSafe(mapJson) << ";\n"
      << "</script>\n\n";

  out << "<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n";
  out << "<script src=\"https://unpkg.com/@viz-js/"
         "viz@3.2.4/lib/viz-standalone.js\"></script>\n\n";
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";
  out << "</body>\n</html>\n";

  return mlir::success();
}

} // namespace fcc
