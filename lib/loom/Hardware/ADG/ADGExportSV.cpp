//===-- ADGExportSV.cpp - SystemVerilog export implementation --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements ADGBuilder::exportSV() and the internal generateSV() method.
// Generates a self-contained SystemVerilog design directory from the ADG.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGBuilderImpl.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Helper: get data width in bits for a Type
//===----------------------------------------------------------------------===//

static unsigned getDataWidthBits(const Type &t) {
  switch (t.getKind()) {
  case Type::I1:    return 1;
  case Type::I8:    return 8;
  case Type::I16:   return 16;
  case Type::I32:   return 32;
  case Type::I64:   return 64;
  case Type::IN:    return t.getWidth();
  case Type::BF16:  return 16;
  case Type::F16:   return 16;
  case Type::F32:   return 32;
  case Type::F64:   return 64;
  case Type::Index: return 64;
  case Type::None:  return 0;
  case Type::Tagged:
    return getDataWidthBits(t.getValueType());
  }
  return 32;
}

static unsigned getTagWidthBits(const Type &t) {
  if (!t.isTagged())
    return 0;
  return getDataWidthBits(t.getTagType());
}

//===----------------------------------------------------------------------===//
// Helper: write string to file
//===----------------------------------------------------------------------===//

static void writeFile(const std::string &path, const std::string &content) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot write output file: " << path << "\n"
                 << ec.message() << "\n";
    std::exit(1);
  }
  out << content;
  out.flush();
}

//===----------------------------------------------------------------------===//
// Helper: copy a template file from LOOM_SV_TEMPLATE_DIR
//===----------------------------------------------------------------------===//

static void copyTemplateFile(const std::string &srcDir,
                             const std::string &relPath,
                             const std::string &dstDir) {
  llvm::SmallString<256> srcPath(srcDir);
  llvm::sys::path::append(srcPath, relPath);

  llvm::SmallString<256> dstPath(dstDir);
  llvm::sys::path::append(dstPath, llvm::sys::path::filename(relPath));

  std::error_code ec = llvm::sys::fs::copy_file(srcPath, dstPath);
  if (ec) {
    llvm::errs() << "error: cannot copy template file: " << srcPath << " -> "
                 << dstPath << "\n"
                 << ec.message() << "\n";
    std::exit(1);
  }
}

//===----------------------------------------------------------------------===//
// SV module kind name
//===----------------------------------------------------------------------===//

static const char *svModuleName(ModuleKind kind) {
  switch (kind) {
  case ModuleKind::PE:            return "fabric_pe";
  case ModuleKind::ConstantPE:    return "fabric_pe_constant";
  case ModuleKind::LoadPE:        return "fabric_pe_load";
  case ModuleKind::StorePE:       return "fabric_pe_store";
  case ModuleKind::Switch:        return "fabric_switch";
  case ModuleKind::TemporalPE:    return "fabric_temporal_pe";
  case ModuleKind::TemporalSwitch:return "fabric_temporal_sw";
  case ModuleKind::Memory:        return "fabric_memory";
  case ModuleKind::ExtMemory:     return "fabric_extmemory";
  case ModuleKind::AddTag:        return "fabric_add_tag";
  case ModuleKind::MapTag:        return "fabric_map_tag";
  case ModuleKind::DelTag:        return "fabric_del_tag";
  case ModuleKind::Fifo:          return "fabric_fifo";
  }
  return "fabric_unknown";
}

//===----------------------------------------------------------------------===//
// Helper: compute NUM_CONNECTED for a switch definition
//===----------------------------------------------------------------------===//

static unsigned getNumConnected(const SwitchDef &def) {
  if (def.connectivity.empty())
    return def.numIn * def.numOut;
  unsigned count = 0;
  for (const auto &row : def.connectivity)
    for (bool v : row)
      if (v) ++count;
  return count;
}

static unsigned getNumConnected(const TemporalSwitchDef &def) {
  if (def.connectivity.empty())
    return def.numIn * def.numOut;
  unsigned count = 0;
  for (const auto &row : def.connectivity)
    for (bool v : row)
      if (v) ++count;
  return count;
}

//===----------------------------------------------------------------------===//
// Helper: read a file into a string
//===----------------------------------------------------------------------===//

static std::string readFile(const std::string &path) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    llvm::errs() << "error: cannot read file: " << path << "\n";
    std::exit(1);
  }
  return (*bufOrErr)->getBuffer().str();
}

//===----------------------------------------------------------------------===//
// PE helpers: operation latency and body generation
//===----------------------------------------------------------------------===//

/// Return the typical latency (in cycles) for a single operation name.
static unsigned getOpLatency(const std::string &opName) {
  // Bitwise / extension / truncation: 0 cycles
  if (opName == "arith.andi" || opName == "arith.ori" ||
      opName == "arith.xori" || opName == "arith.shli" ||
      opName == "arith.shrsi" || opName == "arith.shrui" ||
      opName == "arith.trunci" || opName == "arith.extsi" ||
      opName == "arith.extui" || opName == "arith.index_cast" ||
      opName == "arith.index_castui" || opName == "llvm.intr.bitreverse")
    return 0;
  // Integer add/sub/cmp/select: 1 cycle
  if (opName == "arith.addi" || opName == "arith.subi" ||
      opName == "arith.cmpi" || opName == "arith.select")
    return 1;
  // Integer mul: 3 cycles
  if (opName == "arith.muli")
    return 3;
  // Integer div/rem: 6 cycles
  if (opName == "arith.divsi" || opName == "arith.divui" ||
      opName == "arith.remsi" || opName == "arith.remui")
    return 6;
  // FP add/sub/cmp/neg: 4 cycles
  if (opName == "arith.addf" || opName == "arith.subf" ||
      opName == "arith.cmpf" || opName == "arith.negf" ||
      opName == "math.absf")
    return 4;
  // FP mul/fma: 5 cycles
  if (opName == "arith.mulf" || opName == "math.fma")
    return 5;
  // FP conversion: 4 cycles
  if (opName == "arith.sitofp" || opName == "arith.uitofp" ||
      opName == "arith.fptosi" || opName == "arith.fptoui")
    return 4;
  // FP div/sqrt: 10 cycles
  if (opName == "arith.divf" || opName == "math.sqrt")
    return 10;
  // FP transcendental: 12 cycles
  if (opName == "math.sin" || opName == "math.cos" ||
      opName == "math.exp" || opName == "math.log2")
    return 12;
  // Default: 1 cycle
  return 1;
}

/// Convert a dialect.op name to the SV module name (e.g., "arith.addi" -> "arith_addi").
static std::string opToSVModule(const std::string &opName) {
  // Normalize LLVM intrinsic names: strip the ".intr" segment
  // e.g., "llvm.intr.bitreverse" -> "llvm_bitreverse"
  if (opName.find("llvm.intr.") == 0) {
    std::string result = "llvm_" + opName.substr(10);
    for (char &c : result)
      if (c == '.')
        c = '_';
    return result;
  }
  std::string result = opName;
  for (char &c : result)
    if (c == '.')
      c = '_';
  return result;
}

/// Return true if the op is a width-conversion operation.
static bool isConversionOp(const std::string &opName) {
  return opName == "arith.extsi" || opName == "arith.extui" ||
         opName == "arith.trunci";
}

/// Return true if the op is a compare operation (1-bit result).
static bool isCompareOp(const std::string &opName) {
  return opName == "arith.cmpi" || opName == "arith.cmpf";
}

/// Parse a simple MLIR SSA line: "%res = dialect.op %a, %b : type"
/// Returns {result, opName, {operands}, typeAnnotation} or empty result on
/// failure.
struct MLIRStmt {
  std::string result; // e.g., "%t0"
  std::string opName; // e.g., "arith.muli"
  std::vector<std::string> operands; // e.g., {"%arg0", "%arg1"}
  std::string typeAnnotation; // e.g., "i16 to i32" (text after ':')
};

static MLIRStmt parseMLIRLine(const std::string &line) {
  MLIRStmt stmt;
  auto eqPos = line.find('=');
  if (eqPos == std::string::npos)
    return stmt;

  // Extract result name (trim whitespace)
  std::string lhs = line.substr(0, eqPos);
  auto pctPos = lhs.find('%');
  if (pctPos == std::string::npos)
    return stmt;
  auto endRes = lhs.find_first_of(" \t", pctPos);
  stmt.result = lhs.substr(pctPos, endRes - pctPos);

  // Extract RHS: "dialect.op %a, %b : type"
  std::string rhs = line.substr(eqPos + 1);
  // Trim leading whitespace
  auto rhsStart = rhs.find_first_not_of(" \t");
  if (rhsStart == std::string::npos)
    return stmt;
  rhs = rhs.substr(rhsStart);

  // Extract op name (first token)
  auto opEnd = rhs.find_first_of(" \t");
  if (opEnd == std::string::npos) {
    stmt.opName = rhs;
    return stmt;
  }
  stmt.opName = rhs.substr(0, opEnd);

  // Extract operands between op name and ':'
  auto colonPos = rhs.find(':');
  std::string opSection = rhs.substr(opEnd, colonPos != std::string::npos
                                                 ? colonPos - opEnd
                                                 : std::string::npos);
  // Find all %name tokens
  size_t pos = 0;
  while ((pos = opSection.find('%', pos)) != std::string::npos) {
    auto end = opSection.find_first_of(" ,:\t\n)", pos);
    stmt.operands.push_back(opSection.substr(pos, end - pos));
    pos = (end != std::string::npos) ? end + 1 : opSection.size();
  }

  // Capture type annotation (text after ':')
  if (colonPos != std::string::npos) {
    std::string ta = rhs.substr(colonPos + 1);
    auto taStart = ta.find_first_not_of(" \t");
    if (taStart != std::string::npos) {
      auto taEnd = ta.find_last_not_of(" \t\n\r");
      stmt.typeAnnotation = ta.substr(taStart, taEnd - taStart + 1);
    }
  }

  return stmt;
}

/// Parse width from an MLIR integer type string like "i32", "i16", "index".
/// Returns 0 if not a recognized integer type.
static unsigned parseMLIRTypeWidth(const std::string &typeStr) {
  auto s = typeStr.find_first_not_of(" \t");
  if (s == std::string::npos) return 0;
  std::string t = typeStr.substr(s);
  auto e = t.find_first_of(" \t,)");
  if (e != std::string::npos) t = t.substr(0, e);
  if (t == "index") return 32;
  if (t.size() > 1 && (t[0] == 'i' || t[0] == 'f')) {
    unsigned w = 0;
    for (size_t i = 1; i < t.size(); ++i) {
      if (!std::isdigit(t[i])) return 0;
      w = w * 10 + (t[i] - '0');
    }
    return w;
  }
  return 0;
}

/// Parse conversion op type annotation: "i16 to i32" -> {16, 32}.
/// Returns {0, 0} if parsing fails.
static std::pair<unsigned, unsigned>
parseConversionWidths(const std::string &typeAnnotation) {
  // Format: "i16 to i32" or "i32 to i16"
  auto toPos = typeAnnotation.find(" to ");
  if (toPos == std::string::npos) {
    // Try without spaces: "i16to i32" is unlikely but handle "i16 -> i32"
    toPos = typeAnnotation.find("->");
    if (toPos != std::string::npos) {
      unsigned inW = parseMLIRTypeWidth(typeAnnotation.substr(0, toPos));
      unsigned outW = parseMLIRTypeWidth(typeAnnotation.substr(toPos + 2));
      return {inW, outW};
    }
    return {0, 0};
  }
  unsigned inW = parseMLIRTypeWidth(typeAnnotation.substr(0, toPos));
  unsigned outW = parseMLIRTypeWidth(typeAnnotation.substr(toPos + 4));
  return {inW, outW};
}

/// Extract all operation names from a bodyMLIR string.
static std::vector<std::string> extractBodyMLIROps(const std::string &bodyMLIR) {
  std::vector<std::string> ops;
  std::string body = bodyMLIR;
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos)
      body = body.substr(closePos + 2);
  }
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    auto s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    line = line.substr(s);
    if (line.find("fabric.yield") == 0) continue;
    MLIRStmt stmt = parseMLIRLine(line);
    if (!stmt.opName.empty())
      ops.push_back(stmt.opName);
  }
  return ops;
}

/// Compute the critical-path latency through a bodyMLIR DAG.
/// Uses a simple approach: sum of all operation latencies along the
/// longest dependency chain.
static unsigned computeBodyMLIRLatency(const std::string &bodyMLIR) {
  std::string body = bodyMLIR;
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos)
      body = body.substr(closePos + 2);
  }

  // Parse all stmts and build a dependency graph
  std::vector<MLIRStmt> stmts;
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    auto s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    line = line.substr(s);
    if (line.find("fabric.yield") == 0) continue;
    MLIRStmt stmt = parseMLIRLine(line);
    if (!stmt.opName.empty())
      stmts.push_back(stmt);
  }

  // Map SSA result -> index in stmts
  std::map<std::string, unsigned> resultToIdx;
  for (unsigned i = 0; i < stmts.size(); ++i)
    resultToIdx[stmts[i].result] = i;

  // Compute longest path to each node
  std::vector<unsigned> longest(stmts.size(), 0);
  for (unsigned i = 0; i < stmts.size(); ++i) {
    unsigned maxPred = 0;
    for (const auto &op : stmts[i].operands) {
      auto it = resultToIdx.find(op);
      if (it != resultToIdx.end())
        maxPred = std::max(maxPred, longest[it->second]);
    }
    longest[i] = maxPred + getOpLatency(stmts[i].opName);
  }

  unsigned maxLatency = 0;
  for (unsigned v : longest)
    maxLatency = std::max(maxLatency, v);
  return maxLatency;
}

/// Generate SV body from bodyMLIR containing multiple operations.
static std::string genMultiOpBodySV(const PEDef &def) {
  std::string body = def.bodyMLIR;

  // Strip ^bb0(...): header if present
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos) {
      body = body.substr(closePos + 2);
      if (!body.empty() && body[0] == '\n')
        body = body.substr(1);
    }
  }

  // Rename user args to %argN (same as transformBodyMLIR)
  for (size_t i = 0; i < def.inputPorts.size(); ++i) {
    std::string argName = "%arg" + std::to_string(i);
    // Already in %argN form after transformBodyMLIR
    (void)argName;
  }

  // Parse lines into statements
  std::vector<MLIRStmt> stmts;
  std::vector<std::string> yieldOperands;
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    // Trim whitespace
    auto s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    line = line.substr(s);

    // Handle fabric.yield
    if (line.find("fabric.yield") == 0) {
      // Extract yield operands
      auto colonPos = line.find(':');
      std::string opSection = line.substr(12,
          colonPos != std::string::npos ? colonPos - 12 : std::string::npos);
      size_t pos = 0;
      while ((pos = opSection.find('%', pos)) != std::string::npos) {
        auto end = opSection.find_first_of(" ,:\t\n", pos);
        yieldOperands.push_back(opSection.substr(pos, end - pos));
        pos = (end != std::string::npos) ? end + 1 : opSection.size();
      }
      continue;
    }

    MLIRStmt stmt = parseMLIRLine(line);
    if (!stmt.opName.empty())
      stmts.push_back(stmt);
  }

  if (stmts.empty())
    return "  // empty multi-op body\n";

  // Map SSA names to SV wire names
  // %argN -> in_value[N]
  // %result -> body_tN (intermediate wire)
  std::map<std::string, std::string> ssaToSV;
  for (size_t i = 0; i < def.inputPorts.size(); ++i) {
    ssaToSV["%arg" + std::to_string(i)] = "in_value[" + std::to_string(i) + "]";
  }

  std::ostringstream os;
  unsigned wireIdx = 0;

  for (const auto &stmt : stmts) {
    std::string svModule = opToSVModule(stmt.opName);
    std::string wireName = "body_t" + std::to_string(wireIdx);
    ssaToSV[stmt.result] = wireName;

    if (isConversionOp(stmt.opName)) {
      // Parse IN_WIDTH and OUT_WIDTH from the type annotation.
      // Falls back to DATA_WIDTH if type parsing fails.
      auto [inW, outW] = parseConversionWidths(stmt.typeAnnotation);
      os << "  logic [SAFE_DW-1:0] " << wireName << ";\n";
      if (inW > 0 && outW > 0) {
        os << "  " << svModule << " #(.IN_WIDTH(" << inW << "), .OUT_WIDTH("
           << outW << ")) u_op" << wireIdx << " (\n";
        os << "    .a(" << ssaToSV[stmt.operands[0]] << "[" << (inW - 1)
           << ":0]),\n";
        os << "    .result(" << wireName << "[" << (outW - 1) << ":0])\n";
        os << "  );\n";
        // Zero-fill upper bits when OUT_WIDTH < DATA_WIDTH
        os << "  generate\n";
        os << "    if (DATA_WIDTH > " << outW << ") begin : g_conv_pad_"
           << wireIdx << "\n";
        os << "      assign " << wireName << "[DATA_WIDTH-1:" << outW
           << "] = '0;\n";
        os << "    end\n";
        os << "  endgenerate\n";
      } else {
        os << "  " << svModule << " #(.IN_WIDTH(DATA_WIDTH), .OUT_WIDTH("
           << "DATA_WIDTH)) u_op" << wireIdx << " (\n";
        os << "    .a(" << ssaToSV[stmt.operands[0]] << "),\n";
        os << "    .result(" << wireName << ")\n";
        os << "  );\n";
      }
    } else if (isCompareOp(stmt.opName)) {
      os << "  logic " << wireName << ";\n";
      os << "  " << svModule << " #(.WIDTH(DATA_WIDTH), .PREDICATE(0)) u_op"
         << wireIdx << " (\n";
      os << "    .a(" << ssaToSV[stmt.operands[0]] << "),\n";
      if (stmt.operands.size() > 1)
        os << "    .b(" << ssaToSV[stmt.operands[1]] << "),\n";
      os << "    .result(" << wireName << ")\n";
      os << "  );\n";
    } else {
      os << "  logic [SAFE_DW-1:0] " << wireName << ";\n";
      os << "  " << svModule << " #(.WIDTH(DATA_WIDTH)) u_op" << wireIdx << " (\n";

      if (stmt.opName == "arith.select" && stmt.operands.size() >= 3) {
        os << "    .condition(" << ssaToSV[stmt.operands[0]] << "[0]),\n";
        os << "    .a(" << ssaToSV[stmt.operands[1]] << "),\n";
        os << "    .b(" << ssaToSV[stmt.operands[2]] << "),\n";
      } else if (stmt.opName == "math.fma" && stmt.operands.size() >= 3) {
        os << "    .a(" << ssaToSV[stmt.operands[0]] << "),\n";
        os << "    .b(" << ssaToSV[stmt.operands[1]] << "),\n";
        os << "    .c(" << ssaToSV[stmt.operands[2]] << "),\n";
      } else if (stmt.operands.size() >= 2) {
        os << "    .a(" << ssaToSV[stmt.operands[0]] << "),\n";
        os << "    .b(" << ssaToSV[stmt.operands[1]] << "),\n";
      } else if (stmt.operands.size() >= 1) {
        os << "    .a(" << ssaToSV[stmt.operands[0]] << "),\n";
      }

      os << "    .result(" << wireName << ")\n";
      os << "  );\n";
    }
    os << "\n";
    ++wireIdx;
  }

  // Assign body_result from yield operands or last wire
  if (!yieldOperands.empty()) {
    for (size_t i = 0; i < yieldOperands.size(); ++i) {
      auto it = ssaToSV.find(yieldOperands[i]);
      std::string src = (it != ssaToSV.end()) ? it->second : "'0";
      if (isCompareOp(stmts.back().opName) && i == 0) {
        os << "  assign body_result[" << i << "] = {{(SAFE_DW-1){1'b0}}, " << src << "};\n";
      } else {
        os << "  assign body_result[" << i << "] = " << src << ";\n";
      }
    }
  } else if (!stmts.empty()) {
    os << "  assign body_result[0] = " << ssaToSV[stmts.back().result] << ";\n";
  }

  return os.str();
}

/// Generate the PE body SV text for a PEDef.
/// Returns the text to insert between BEGIN/END PE BODY markers.
static std::string genPEBodySV(const PEDef &def) {
  if (def.singleOp.empty()) {
    if (!def.bodyMLIR.empty())
      return genMultiOpBodySV(def);
    return "  // empty PE body\n";
  }

  std::string svModule = opToSVModule(def.singleOp);
  unsigned numIn = def.inputPorts.size();

  std::ostringstream os;

  if (isConversionOp(def.singleOp)) {
    // Conversion ops use IN_WIDTH and OUT_WIDTH parameters
    unsigned inW = numIn > 0 ? getDataWidthBits(def.inputPorts[0]) : 32;
    unsigned outW = def.outputPorts.size() > 0
                        ? getDataWidthBits(def.outputPorts[0])
                        : 32;
    if (inW == 0) inW = 1;
    if (outW == 0) outW = 1;
    os << "  " << svModule << " #(.IN_WIDTH(" << inW << "), .OUT_WIDTH("
       << outW << ")) u_body (\n";
    os << "    .a(in_value[0][" << (inW - 1) << ":0]),\n";
    os << "    .result(body_result[0][" << (outW - 1) << ":0])\n";
    os << "  );\n";
    // Zero-fill upper bits if OUT_WIDTH < DATA_WIDTH
    os << "  generate\n";
    os << "    if (DATA_WIDTH > " << outW << ") begin : g_conv_pad\n";
    os << "      assign body_result[0][DATA_WIDTH-1:" << outW << "] = '0;\n";
    os << "    end\n";
    os << "  endgenerate\n";
  } else if (isCompareOp(def.singleOp)) {
    // Compare ops have a PREDICATE parameter and 1-bit output
    os << "  logic cmp_result;\n";
    os << "  " << svModule
       << " #(.WIDTH(DATA_WIDTH), .PREDICATE(0)) u_body (\n";
    // Map input ports
    os << "    .a(in_value[0]),\n";
    os << "    .b(in_value[1]),\n";
    os << "    .result(cmp_result)\n";
    os << "  );\n";
    // Zero-extend 1-bit result to DATA_WIDTH
    os << "  assign body_result[0] = {{(DATA_WIDTH-1){1'b0}}, cmp_result};\n";
  } else {
    // Standard ops: use WIDTH parameter
    os << "  " << svModule << " #(.WIDTH(DATA_WIDTH)) u_body (\n";

    // Map input ports: a, b, c, or special names for specific ops
    if (def.singleOp == "arith.select") {
      // select: condition (i1), a, b
      os << "    .condition(in_value[0][0]),\n";
      os << "    .a(in_value[1]),\n";
      os << "    .b(in_value[2]),\n";
    } else if (def.singleOp == "math.fma") {
      os << "    .a(in_value[0]),\n";
      os << "    .b(in_value[1]),\n";
      os << "    .c(in_value[2]),\n";
    } else if (numIn >= 2) {
      os << "    .a(in_value[0]),\n";
      os << "    .b(in_value[1]),\n";
    } else {
      os << "    .a(in_value[0]),\n";
    }

    os << "    .result(body_result[0])\n";
    os << "  );\n";
  }

  return os.str();
}

/// Generate the temporal PE body SV text for a TemporalPEDef.
/// Instantiates per-FU customized fabric_pe modules and muxes results by fu_sel.
static std::string genTemporalPEBodySV(const TemporalPEDef &def,
                                        const std::vector<PEDef> &peDefs,
                                        const std::string &instName) {
  unsigned numFU = def.fuPEDefIndices.size();
  if (numFU == 0)
    return "  // no FU types defined\n";

  unsigned numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();

  // FU_SEL_BITS mirrors the localparam in fabric_temporal_pe.sv
  unsigned fuSelBits = 0;
  if (numFU > 1) {
    unsigned v = numFU - 1;
    while (v > 0) { fuSelBits++; v >>= 1; }
  }

  std::ostringstream os;

  // Extract fu_sel from the matched instruction
  if (fuSelBits > 0) {
    os << "  logic [FU_SEL_BITS-1:0] fu_sel;\n";
    os << "  assign fu_sel = cfg_data[matched_insn * INSN_WIDTH + "
       << "NUM_INPUTS * REG_BITS + NUM_OUTPUTS * RESULT_WIDTH +: FU_SEL_BITS];\n";
    os << "\n";
  }

  // Per-FU result wires, output data wires, and valid signals
  for (unsigned f = 0; f < numFU; ++f) {
    os << "  logic [NUM_OUTPUTS-1:0][SAFE_DW-1:0] fu" << f << "_result;\n";
    os << "  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] fu" << f << "_out_data;\n";
    os << "  logic [NUM_OUTPUTS-1:0] fu" << f << "_out_valid;\n";
  }
  os << "\n";

  // Instantiate each FU as a customized fabric_pe module (TAG_WIDTH=0).
  // Each FU uses its own latency from getOpLatency().
  for (unsigned f = 0; f < numFU; ++f) {
    const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
    std::string fuModName = instName + "_fu" + std::to_string(f) + "_pe";
    unsigned fuNumIn = fuDef.inputPorts.size();
    unsigned fuNumOut = fuDef.outputPorts.size();
    unsigned fuLatency = fuDef.singleOp.empty()
                             ? computeBodyMLIRLatency(fuDef.bodyMLIR)
                             : getOpLatency(fuDef.singleOp);

    os << "  // FU " << f << ": " << (fuDef.singleOp.empty() ? "passthrough" : fuDef.singleOp)
       << " (latency=" << fuLatency << ")\n";
    os << "  " << fuModName << " #(\n";
    os << "    .NUM_INPUTS(" << fuNumIn << "),\n";
    os << "    .NUM_OUTPUTS(" << fuNumOut << "),\n";
    os << "    .DATA_WIDTH(DATA_WIDTH),\n";
    os << "    .TAG_WIDTH(0),\n";
    os << "    .LATENCY_TYP(" << fuLatency << ")\n";
    os << "  ) u_fu" << f << " (\n";
    os << "    .clk(clk),\n";
    os << "    .rst_n(rst_n),\n";
    os << "    .in_valid({" << fuNumIn << "{fu_launch}}),\n";
    os << "    .in_ready(),\n";
    os << "    .in_data({";
    // Map input ports from fu_operands (reverse order for packed array)
    for (unsigned i = fuNumIn; i > 0; --i) {
      if (i < fuNumIn) os << ", ";
      os << "fu_operands[" << (i - 1) << "]";
    }
    os << "}),\n";
    os << "    .out_valid(fu" << f << "_out_valid),\n";
    os << "    .out_ready({" << fuNumOut << "{fire}}),\n";
    os << "    .out_data(fu" << f << "_out_data),\n";
    os << "    .cfg_data('0)\n";
    os << "  );\n";
    // Extract data portion from FU output (TAG_WIDTH=0 so out_data is just data)
    for (unsigned o = 0; o < fuNumOut; ++o) {
      os << "  assign fu" << f << "_result[" << o << "] = fu" << f
         << "_out_data[" << o << "][SAFE_DW-1:0];\n";
    }
    for (unsigned o = fuNumOut; o < numOut; ++o) {
      os << "  assign fu" << f << "_result[" << o << "] = '0;\n";
    }
    os << "\n";
  }

  // Mux body_result and body_valid based on fu_sel
  if (numFU == 1) {
    for (unsigned o = 0; o < numOut; ++o)
      os << "  assign body_result[" << o << "] = fu0_result[" << o << "];\n";
    os << "  assign body_valid = fu0_out_valid[0];\n";
  } else {
    os << "  always_comb begin : fu_mux\n";
    os << "    integer iter_var0;\n";
    os << "    body_valid = 1'b0;\n";
    os << "    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; "
       << "iter_var0 = iter_var0 + 1) begin : per_out\n";
    os << "      body_result[iter_var0] = '0;\n";
    os << "    end\n";
    os << "    case (fu_sel)\n";
    for (unsigned f = 0; f < numFU; ++f) {
      os << "      " << fuSelBits << "'d" << f << ": begin : fu" << f << "\n";
      os << "        body_valid = fu" << f << "_out_valid[0];\n";
      os << "        for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; "
         << "iter_var0 = iter_var0 + 1) begin : assign_out\n";
      os << "          body_result[iter_var0] = fu" << f
         << "_result[iter_var0];\n";
      os << "        end\n";
      os << "      end\n";
    }
    os << "      default: begin : fu_default\n";
    os << "        for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; "
       << "iter_var0 = iter_var0 + 1) begin : zero_out\n";
    os << "          body_result[iter_var0] = '0;\n";
    os << "        end\n";
    os << "      end\n";
    os << "    endcase\n";
    os << "  end\n";
  }

  return os.str();
}

/// Fill the temporal PE template with the generated FU body.
static std::string fillTemporalPETemplate(const std::string &templateDir,
                                           const TemporalPEDef &def,
                                           const std::vector<PEDef> &peDefs,
                                           const std::string &instName) {
  llvm::SmallString<256> tmplPath(templateDir);
  llvm::sys::path::append(tmplPath, "Fabric", "fabric_temporal_pe.sv");
  std::string tmpl = readFile(tmplPath.str().str());

  std::string body = genTemporalPEBodySV(def, peDefs, instName);

  const std::string beginMarker = "  // ===== BEGIN PE BODY =====";
  const std::string endMarker = "  // ===== END PE BODY =====";

  auto beginPos = tmpl.find(beginMarker);
  auto endPos = tmpl.find(endMarker);

  if (beginPos == std::string::npos || endPos == std::string::npos) {
    llvm::errs() << "error: temporal PE template missing body markers\n";
    std::exit(1);
  }

  auto afterBegin = tmpl.find('\n', beginPos);
  std::string result;
  result += tmpl.substr(0, afterBegin + 1);
  result += body;
  result += tmpl.substr(endPos);

  return result;
}

/// Generate PE instance parameters.
static std::string genPEParams(const PEDef &def) {
  unsigned numIn = def.inputPorts.size();
  unsigned numOut = def.outputPorts.size();
  // DATA_WIDTH = max across all input and output port data widths.
  // This is essential for conversion ops (e.g., extsi i16->i32) where
  // input and output widths differ; body_result must be wide enough.
  unsigned dw = 0;
  for (unsigned i = 0; i < numIn; ++i)
    dw = std::max(dw, getDataWidthBits(def.inputPorts[i]));
  for (unsigned i = 0; i < numOut; ++i)
    dw = std::max(dw, getDataWidthBits(def.outputPorts[i]));
  if (dw == 0)
    dw = 32;
  unsigned tw = numIn > 0 ? getTagWidthBits(def.inputPorts[0]) : 0;
  unsigned latency = def.singleOp.empty() && !def.bodyMLIR.empty()
                         ? computeBodyMLIRLatency(def.bodyMLIR)
                         : getOpLatency(def.singleOp);

  std::ostringstream os;
  os << "    .NUM_INPUTS(" << numIn << "),\n";
  os << "    .NUM_OUTPUTS(" << numOut << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .LATENCY_TYP(" << latency << ")";
  return os.str();
}

/// Read the fabric_pe.sv template and replace the body markers with generated body.
/// Returns the complete customized module text.
static std::string fillPETemplate(const std::string &templateDir,
                                  const PEDef &def) {
  llvm::SmallString<256> tmplPath(templateDir);
  llvm::sys::path::append(tmplPath, "Fabric", "fabric_pe.sv");
  std::string tmpl = readFile(tmplPath.str().str());

  std::string body = genPEBodySV(def);

  // Find and replace the body region
  const std::string beginMarker = "  // ===== BEGIN PE BODY =====";
  const std::string endMarker = "  // ===== END PE BODY =====";

  auto beginPos = tmpl.find(beginMarker);
  auto endPos = tmpl.find(endMarker);

  if (beginPos == std::string::npos || endPos == std::string::npos) {
    llvm::errs() << "error: PE template missing body markers\n";
    std::exit(1);
  }

  // Replace from after beginMarker line to before endMarker line
  auto afterBegin = tmpl.find('\n', beginPos);
  std::string result;
  result += tmpl.substr(0, afterBegin + 1);
  result += body;
  result += tmpl.substr(endPos);

  return result;
}

//===----------------------------------------------------------------------===//
// Generate constant PE instance parameters
//===----------------------------------------------------------------------===//

static std::string genConstantPEParams(const ConstantPEDef &def) {
  unsigned dw = getDataWidthBits(def.outputType);
  unsigned tw = getTagWidthBits(def.outputType);
  if (dw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate load PE instance parameters
//===----------------------------------------------------------------------===//

static std::string genLoadPEParams(const LoadPEDef &def) {
  unsigned dw = getDataWidthBits(def.dataType);
  if (dw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << def.tagWidth << "),\n";
  os << "    .HW_TYPE(" << (def.hwType == HardwareType::TagTransparent ? 1 : 0)
     << "),\n";
  os << "    .QUEUE_DEPTH(" << def.queueDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate store PE instance parameters
//===----------------------------------------------------------------------===//

static std::string genStorePEParams(const StorePEDef &def) {
  unsigned dw = getDataWidthBits(def.dataType);
  if (dw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << def.tagWidth << "),\n";
  os << "    .HW_TYPE(" << (def.hwType == HardwareType::TagTransparent ? 1 : 0)
     << "),\n";
  os << "    .QUEUE_DEPTH(" << def.queueDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate temporal switch instance parameters
//===----------------------------------------------------------------------===//

static std::string genTemporalSwitchParams(const TemporalSwitchDef &def) {
  unsigned dw = getDataWidthBits(def.interfaceType);
  unsigned tw = getTagWidthBits(def.interfaceType);
  if (dw + tw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .NUM_INPUTS(" << def.numIn << "),\n";
  os << "    .NUM_OUTPUTS(" << def.numOut << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .NUM_ROUTE_TABLE(" << def.numRouteTable << ")";

  // Connectivity matrix
  if (!def.connectivity.empty()) {
    unsigned total = def.numOut * def.numIn;
    os << ",\n    .CONNECTIVITY(" << total << "'b";
    for (int o = static_cast<int>(def.numOut) - 1; o >= 0; --o) {
      for (int i = static_cast<int>(def.numIn) - 1; i >= 0; --i) {
        if (static_cast<unsigned>(o) < def.connectivity.size() &&
            static_cast<unsigned>(i) < def.connectivity[o].size())
          os << (def.connectivity[o][i] ? "1" : "0");
        else
          os << "1";
      }
    }
    os << ")";
  }

  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate temporal PE instance parameters
//===----------------------------------------------------------------------===//

static std::string genTemporalPEParams(const TemporalPEDef &def,
                                       const std::vector<PEDef> &peDefs) {
  unsigned dw = getDataWidthBits(def.interfaceType);
  unsigned tw = getTagWidthBits(def.interfaceType);
  if (dw == 0)
    dw = 1;
  unsigned numIn = 1, numOut = 1;
  if (!def.fuPEDefIndices.empty()) {
    numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
    numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
  }
  std::ostringstream os;
  os << "    .NUM_INPUTS(" << numIn << "),\n";
  os << "    .NUM_OUTPUTS(" << numOut << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .NUM_FU_TYPES(" << def.fuPEDefIndices.size() << "),\n";
  os << "    .NUM_REGISTERS(" << def.numRegisters << "),\n";
  os << "    .NUM_INSTRUCTIONS(" << def.numInstructions << "),\n";
  os << "    .REG_FIFO_DEPTH(" << def.regFifoDepth << "),\n";
  os << "    .SHARE_MODE_B(" << (def.shareModeB ? 1 : 0) << "),\n";
  os << "    .OPERAND_BUFFER_SIZE(" << def.shareBufferSize << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate memory instance parameters
//===----------------------------------------------------------------------===//

static std::string genMemoryParams(const MemoryDef &def) {
  unsigned dw = getDataWidthBits(def.shape.getElemType());
  if (dw == 0)
    dw = 1;
  // Determine tag width from tagging rules
  unsigned tw = 0;
  if (def.ldCount > 1 || def.stCount > 1) {
    unsigned maxCount = std::max(def.ldCount, def.stCount);
    unsigned tagBits = 1;
    while ((1u << tagBits) < maxCount)
      ++tagBits;
    tw = tagBits;
  }
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .LD_COUNT(" << def.ldCount << "),\n";
  os << "    .ST_COUNT(" << def.stCount << "),\n";
  os << "    .LSQ_DEPTH(" << def.lsqDepth << "),\n";
  os << "    .IS_PRIVATE(" << (def.isPrivate ? 1 : 0) << "),\n";
  // Use shape size as MEM_DEPTH
  unsigned memDepth = def.shape.isDynamic() ? 64 : def.shape.getSize();
  if (memDepth == 0)
    memDepth = 64;
  os << "    .MEM_DEPTH(" << memDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate external memory instance parameters
//===----------------------------------------------------------------------===//

static std::string genExtMemoryParams(const ExtMemoryDef &def) {
  unsigned dw = getDataWidthBits(def.shape.getElemType());
  if (dw == 0)
    dw = 1;
  unsigned tw = 0;
  if (def.ldCount > 1 || def.stCount > 1) {
    unsigned maxCount = std::max(def.ldCount, def.stCount);
    unsigned tagBits = 1;
    while ((1u << tagBits) < maxCount)
      ++tagBits;
    tw = tagBits;
  }
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .LD_COUNT(" << def.ldCount << "),\n";
  os << "    .ST_COUNT(" << def.stCount << "),\n";
  os << "    .LSQ_DEPTH(" << def.lsqDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate switch instance parameters
//===----------------------------------------------------------------------===//

static std::string genSwitchParams(const SwitchDef &def) {
  unsigned dw = getDataWidthBits(def.portType);
  unsigned tw = getTagWidthBits(def.portType);
  // Control-token types (Type::none) have zero data+tag width.
  // Use DATA_WIDTH=1 as the minimum SV representation so the
  // hardware compiles; the data bits are unused for control tokens.
  if (dw + tw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .NUM_INPUTS(" << def.numIn << "),\n";
  os << "    .NUM_OUTPUTS(" << def.numOut << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << ")";

  // Connectivity matrix
  if (!def.connectivity.empty()) {
    unsigned total = def.numOut * def.numIn;
    os << ",\n    .CONNECTIVITY(" << total << "'b";
    // Emit MSB first (highest output, highest input)
    for (int o = static_cast<int>(def.numOut) - 1; o >= 0; --o) {
      for (int i = static_cast<int>(def.numIn) - 1; i >= 0; --i) {
        if (static_cast<unsigned>(o) < def.connectivity.size() &&
            static_cast<unsigned>(i) < def.connectivity[o].size())
          os << (def.connectivity[o][i] ? "1" : "0");
        else
          os << "1";
      }
    }
    os << ")";
  }

  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate FIFO instance parameters
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Generate add_tag instance parameters
//===----------------------------------------------------------------------===//

static std::string genAddTagParams(const AddTagDef &def) {
  unsigned dw = getDataWidthBits(def.valueType);
  unsigned tw = getDataWidthBits(def.tagType);
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate del_tag instance parameters
//===----------------------------------------------------------------------===//

static std::string genDelTagParams(const DelTagDef &def) {
  unsigned dw = getDataWidthBits(def.inputType.getValueType());
  unsigned tw = getDataWidthBits(def.inputType.getTagType());
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate map_tag instance parameters
//===----------------------------------------------------------------------===//

static std::string genMapTagParams(const MapTagDef &def) {
  unsigned dw = getDataWidthBits(def.valueType);
  unsigned itw = getDataWidthBits(def.inputTagType);
  unsigned otw = getDataWidthBits(def.outputTagType);
  std::ostringstream os;
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .IN_TAG_WIDTH(" << itw << "),\n";
  os << "    .OUT_TAG_WIDTH(" << otw << "),\n";
  os << "    .TABLE_SIZE(" << def.tableSize << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate FIFO instance parameters
//===----------------------------------------------------------------------===//

static std::string genFifoParams(const FifoDef &def) {
  unsigned dw = getDataWidthBits(def.elementType);
  unsigned tw = getTagWidthBits(def.elementType);
  // Control-token types (Type::none) have zero data+tag width.
  // Use DATA_WIDTH=1 as the minimum SV representation so the
  // hardware compiles; the data bits are unused for control tokens.
  if (dw + tw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .DEPTH(" << def.depth << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .BYPASSABLE(" << (def.bypassable ? 1 : 0) << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// ADGBuilder::Impl::generateSV
//===----------------------------------------------------------------------===//

static bool hasSVTemplate(ModuleKind kind) {
  return kind == ModuleKind::Switch || kind == ModuleKind::Fifo ||
         kind == ModuleKind::AddTag || kind == ModuleKind::DelTag ||
         kind == ModuleKind::MapTag || kind == ModuleKind::PE ||
         kind == ModuleKind::ConstantPE || kind == ModuleKind::LoadPE ||
         kind == ModuleKind::StorePE || kind == ModuleKind::TemporalSwitch ||
         kind == ModuleKind::TemporalPE || kind == ModuleKind::Memory ||
         kind == ModuleKind::ExtMemory;
}

/// Returns true for module kinds that expose error_valid/error_code ports.
static bool hasErrorOutput(ModuleKind kind) {
  return kind == ModuleKind::Switch || kind == ModuleKind::TemporalSwitch ||
         kind == ModuleKind::TemporalPE || kind == ModuleKind::MapTag ||
         kind == ModuleKind::Memory || kind == ModuleKind::ExtMemory;
}

//===----------------------------------------------------------------------===//
// Helper: validate a name as a legal SystemVerilog identifier
//===----------------------------------------------------------------------===//

static bool isSVKeyword(const std::string &name) {
  // IEEE 1800-2017 Table B.1 - all reserved keywords.
  static const std::set<std::string> keywords = {
      "accept_on",    "alias",         "always",        "always_comb",
      "always_ff",    "always_latch",  "and",           "assert",
      "assign",       "assume",        "automatic",     "before",
      "begin",        "bind",          "bins",          "binsof",
      "bit",          "break",         "buf",           "bufif0",
      "bufif1",       "byte",          "case",          "casex",
      "casez",        "cell",          "chandle",       "checker",
      "class",        "clocking",      "cmos",          "config",
      "const",        "constraint",    "context",       "continue",
      "cover",        "covergroup",    "coverpoint",    "cross",
      "deassign",     "default",       "defparam",      "design",
      "disable",      "dist",          "do",            "edge",
      "else",         "end",           "endcase",       "endchecker",
      "endclass",     "endclocking",   "endconfig",     "endfunction",
      "endgenerate",  "endgroup",      "endinterface",  "endmodule",
      "endpackage",   "endprimitive",  "endprogram",    "endproperty",
      "endsequence",  "endspecify",    "endtable",      "endtask",
      "enum",         "event",         "eventually",    "expect",
      "export",       "extends",       "extern",        "final",
      "first_match",  "for",           "force",         "foreach",
      "forever",      "fork",          "forkjoin",      "function",
      "generate",     "genvar",        "global",        "highz0",
      "highz1",       "if",            "iff",           "ifnone",
      "ignore_bins",  "illegal_bins",  "implements",    "implies",
      "import",       "incdir",        "include",       "initial",
      "inout",        "input",         "inside",        "instance",
      "int",          "integer",       "interconnect",  "interface",
      "intersect",    "join",          "join_any",      "join_none",
      "large",        "let",           "liblist",       "library",
      "local",        "localparam",    "logic",         "longint",
      "macromodule",  "matches",       "medium",        "modport",
      "module",       "nand",          "negedge",       "nettype",
      "new",          "nexttime",      "nmos",          "nor",
      "noshowcancelled","not",         "notif0",        "notif1",
      "null",         "or",            "output",        "package",
      "packed",       "parameter",     "pmos",          "posedge",
      "primitive",    "priority",      "program",       "property",
      "protected",    "pull0",         "pull1",         "pulldown",
      "pullup",       "pulsestyle_ondetect","pulsestyle_onevent","pure",
      "rand",         "randc",         "randcase",      "randsequence",
      "rcmos",        "real",          "realtime",      "ref",
      "reg",          "reject_on",     "release",       "repeat",
      "restrict",     "return",        "rnmos",         "rpmos",
      "rtran",        "rtranif0",      "rtranif1",      "s_always",
      "s_eventually", "s_nexttime",    "s_until",       "s_until_with",
      "scalared",     "sequence",      "shortint",      "shortreal",
      "showcancelled","signed",        "small",         "soft",
      "solve",        "specify",       "specparam",     "static",
      "string",       "strong",        "strong0",       "strong1",
      "struct",       "super",         "supply0",       "supply1",
      "sync_accept_on","sync_reject_on","table",        "tagged",
      "task",         "this",          "throughout",    "time",
      "timeprecision","timeunit",      "tran",          "tranif0",
      "tranif1",      "tri",           "tri0",          "tri1",
      "triand",       "trior",         "trireg",        "type",
      "typedef",      "union",         "unique",        "unique0",
      "unsigned",     "until",         "until_with",    "untyped",
      "use",          "uwire",         "var",           "vectored",
      "virtual",      "void",          "wait",          "wait_order",
      "wand",         "weak",          "weak0",         "weak1",
      "while",        "wildcard",      "wire",          "with",
      "within",       "wor",           "xnor",          "xor",
  };
  return keywords.count(name) != 0;
}

static bool isValidSVIdentifier(const std::string &name) {
  if (name.empty())
    return false;
  // Must start with letter or underscore
  if (!std::isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_')
    return false;
  for (char c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      return false;
  }
  return !isSVKeyword(name);
}

void ADGBuilder::Impl::generateSV(const std::string &directory) const {
  // Reject unsupported module kinds before producing any output
  for (const auto &inst : instances) {
    if (!hasSVTemplate(inst.kind)) {
      llvm::errs() << "error: exportSV does not support module kind '"
                   << svModuleName(inst.kind) << "' (instance '" << inst.name
                   << "')\n";
      std::exit(1);
    }
  }

  // Validate module name (emitted as <moduleName>_top)
  if (!isValidSVIdentifier(moduleName)) {
    llvm::errs() << "error: exportSV: module name '" << moduleName
                 << "' is not a valid SystemVerilog identifier\n";
    std::exit(1);
  }

  // Reject memref ports (not supported in SV export)
  for (const auto &p : ports) {
    if (p.isMemref) {
      llvm::errs() << "error: exportSV does not support memref port '"
                   << p.name << "'\n";
      std::exit(1);
    }
  }

  // Determine once whether any instance produces error signals
  bool hasErrorPorts = false;
  for (const auto &inst : instances) {
    if (hasErrorOutput(inst.kind)) {
      hasErrorPorts = true;
      break;
    }
  }

  // Validate instance names: must be valid SV identifiers and unique
  {
    std::set<std::string> seenNames;
    for (const auto &inst : instances) {
      if (!isValidSVIdentifier(inst.name)) {
        llvm::errs() << "error: exportSV: instance name '" << inst.name
                     << "' is not a valid SystemVerilog identifier\n";
        std::exit(1);
      }
      if (!seenNames.insert(inst.name).second) {
        llvm::errs() << "error: exportSV: duplicate instance name '"
                     << inst.name << "'\n";
        std::exit(1);
      }
    }
  }

  // Validate port names: must be valid SV identifiers, unique, and not
  // collide with generated aggregated port names (error_valid/error_code).
  {
    std::set<std::string> seenPorts;
    if (hasErrorPorts)
      seenPorts.insert("error");
    for (const auto &p : ports) {
      if (!isValidSVIdentifier(p.name)) {
        llvm::errs() << "error: exportSV: port name '" << p.name
                     << "' is not a valid SystemVerilog identifier\n";
        std::exit(1);
      }
      if (!seenPorts.insert(p.name).second) {
        llvm::errs() << "error: exportSV: port name '" << p.name
                     << "' collides with another port or reserved name\n";
        std::exit(1);
      }
    }
  }

  // Collect instance-derived internal signal names (wires + config ports).
  static const char *streamSuffixes[] = {"_valid", "_ready", "_data"};
  std::set<std::string> instanceDerived;
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = getInstanceInputCount(i);
    unsigned numOut = getInstanceOutputCount(i);
    for (unsigned p = 0; p < numIn; ++p) {
      std::string base = inst.name + "_in" + std::to_string(p);
      for (const char *sfx : streamSuffixes)
        instanceDerived.insert(base + sfx);
    }
    for (unsigned p = 0; p < numOut; ++p) {
      std::string base = inst.name + "_out" + std::to_string(p);
      for (const char *sfx : streamSuffixes)
        instanceDerived.insert(base + sfx);
    }
    if (hasErrorOutput(inst.kind)) {
      instanceDerived.insert(inst.name + "_error_valid");
      instanceDerived.insert(inst.name + "_error_code");
    }
    if (inst.kind == ModuleKind::Switch) {
      instanceDerived.insert(inst.name + "_cfg_route_table");
    } else if (inst.kind == ModuleKind::Fifo) {
      const auto &def = fifoDefs[inst.defIdx];
      if (def.bypassable)
        instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::AddTag) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::MapTag) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::PE) {
      const auto &def = peDefs[inst.defIdx];
      unsigned tw = def.inputPorts.size() > 0
                        ? getTagWidthBits(def.inputPorts[0])
                        : 0;
      if (tw > 0)
        instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::ConstantPE) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::LoadPE) {
      const auto &def = loadPEDefs[inst.defIdx];
      if (def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::StorePE) {
      const auto &def = storePEDefs[inst.defIdx];
      if (def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::TemporalSwitch) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::TemporalPE) {
      instanceDerived.insert(inst.name + "_cfg_data");
    }
    // Memory and ExtMemory have no config ports (CONFIG_WIDTH = 0)
  }

  // Check port suffixed names against instance-derived signals
  for (const auto &p : ports) {
    for (const char *sfx : streamSuffixes) {
      if (instanceDerived.count(p.name + sfx)) {
        llvm::errs() << "error: exportSV: port '" << p.name
                     << "' generates signal '" << p.name << sfx
                     << "' that collides with an internal/config signal\n";
        std::exit(1);
      }
    }
  }

  // Check instance names against top-level identifiers (fixed ports,
  // port-derived signals, error aggregation ports, and instance-derived names).
  {
    std::set<std::string> topLevel({"clk", "rst_n"});
    if (hasErrorPorts) {
      topLevel.insert("error_valid");
      topLevel.insert("error_code");
    }
    for (const auto &p : ports) {
      for (const char *sfx : streamSuffixes)
        topLevel.insert(p.name + sfx);
    }
    topLevel.insert(instanceDerived.begin(), instanceDerived.end());
    for (const auto &inst : instances) {
      if (topLevel.count(inst.name)) {
        llvm::errs() << "error: exportSV: instance name '" << inst.name
                     << "' collides with a generated top-level identifier\n";
        std::exit(1);
      }
    }
  }

  // Create output directories
  llvm::SmallString<256> libDir(directory);
  llvm::sys::path::append(libDir, "lib");

  llvm::sys::fs::create_directories(directory);
  llvm::sys::fs::create_directories(libDir);

  // Copy template files from LOOM_SV_TEMPLATE_DIR
#ifdef LOOM_SV_TEMPLATE_DIR
  const std::string templateDir = LOOM_SV_TEMPLATE_DIR;
#else
  const std::string templateDir = "";
#endif

  // Track which operation dialects are used (for copying operation SV files)
  std::set<std::string> usedDialects;

  if (!templateDir.empty()) {
    // Copy Common/ files
    copyTemplateFile(templateDir, "Common/fabric_common.svh", libDir.str().str());
    // Copy Fabric/ files
    copyTemplateFile(templateDir, "Fabric/fabric_fifo.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_switch.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_add_tag.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_del_tag.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_map_tag.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_pe_constant.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_pe_load.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_pe_store.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_temporal_sw.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_temporal_pe.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_memory.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_extmemory.sv", libDir.str().str());

    // For PE instances: generate body-filled customized modules
    for (const auto &inst : instances) {
      if (inst.kind == ModuleKind::PE) {
        const auto &def = peDefs[inst.defIdx];
        std::string customized = fillPETemplate(templateDir, def);

        // Replace module name to make it unique per instance
        std::string origName = "module fabric_pe";
        std::string newName = "module " + inst.name + "_pe";
        auto pos = customized.find(origName);
        if (pos != std::string::npos)
          customized.replace(pos, origName.size(), newName);

        llvm::SmallString<256> pePath(libDir);
        llvm::sys::path::append(pePath, inst.name + "_pe.sv");
        writeFile(pePath.str().str(), customized);

        // Track the dialect for operation module copying
        if (!def.singleOp.empty()) {
          auto dotPos = def.singleOp.find('.');
          if (dotPos != std::string::npos)
            usedDialects.insert(def.singleOp.substr(0, dotPos));
        }
        if (!def.bodyMLIR.empty()) {
          for (const auto &op : extractBodyMLIROps(def.bodyMLIR)) {
            auto dotPos = op.find('.');
            if (dotPos != std::string::npos)
              usedDialects.insert(op.substr(0, dotPos));
          }
        }
      }
    }

    // For TemporalPE instances: generate per-FU fabric_pe modules and
    // body-filled customized temporal_pe modules
    for (const auto &inst : instances) {
      if (inst.kind == ModuleKind::TemporalPE) {
        const auto &def = temporalPEDefs[inst.defIdx];

        // Generate customized fabric_pe modules for each FU type
        for (unsigned f = 0; f < def.fuPEDefIndices.size(); ++f) {
          const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
          std::string fuModName = inst.name + "_fu" + std::to_string(f) + "_pe";
          std::string fuCustomized = fillPETemplate(templateDir, fuDef);

          // Replace module name to match FU module name
          std::string origPeName = "module fabric_pe";
          std::string newPeName = "module " + fuModName;
          auto pePos = fuCustomized.find(origPeName);
          if (pePos != std::string::npos)
            fuCustomized.replace(pePos, origPeName.size(), newPeName);

          llvm::SmallString<256> fuPath(libDir);
          llvm::sys::path::append(fuPath, fuModName + ".sv");
          writeFile(fuPath.str().str(), fuCustomized);
        }

        // Generate the temporal PE module with FU instantiations
        std::string customized =
            fillTemporalPETemplate(templateDir, def, peDefs, inst.name);

        // Replace module name to make it unique per instance
        std::string origName = "module fabric_temporal_pe";
        std::string newName = "module " + inst.name + "_temporal_pe";
        auto pos = customized.find(origName);
        if (pos != std::string::npos)
          customized.replace(pos, origName.size(), newName);

        llvm::SmallString<256> tpePath(libDir);
        llvm::sys::path::append(tpePath, inst.name + "_temporal_pe.sv");
        writeFile(tpePath.str().str(), customized);

        // Track dialects used by FU types (singleOp and bodyMLIR)
        for (unsigned idx : def.fuPEDefIndices) {
          const auto &fuDef = peDefs[idx];
          if (!fuDef.singleOp.empty()) {
            auto dotPos = fuDef.singleOp.find('.');
            if (dotPos != std::string::npos)
              usedDialects.insert(fuDef.singleOp.substr(0, dotPos));
          }
          if (!fuDef.bodyMLIR.empty()) {
            for (const auto &op : extractBodyMLIROps(fuDef.bodyMLIR)) {
              auto dotPos = op.find('.');
              if (dotPos != std::string::npos)
                usedDialects.insert(op.substr(0, dotPos));
            }
          }
        }
      }
    }

    // Copy operation module SV files for used dialects
    // Map dialect names to directory names
    static const std::map<std::string, std::string> dialectDirs = {
        {"arith", "Arith"}, {"math", "Math"}, {"llvm", "LLVM"}};

    for (const auto &dialect : usedDialects) {
      auto it = dialectDirs.find(dialect);
      if (it == dialectDirs.end())
        continue;

      // Collect all operation names used by PEs and TemporalPE FU types
      std::set<std::string> opsForDialect;
      auto addOpsFromDef = [&](const PEDef &def) {
        if (!def.singleOp.empty()) {
          auto dotPos = def.singleOp.find('.');
          if (dotPos != std::string::npos && def.singleOp.substr(0, dotPos) == dialect)
            opsForDialect.insert(def.singleOp);
        }
        if (!def.bodyMLIR.empty()) {
          for (const auto &op : extractBodyMLIROps(def.bodyMLIR)) {
            auto dotPos = op.find('.');
            if (dotPos != std::string::npos && op.substr(0, dotPos) == dialect)
              opsForDialect.insert(op);
          }
        }
      };
      for (const auto &inst : instances) {
        if (inst.kind == ModuleKind::PE) {
          addOpsFromDef(peDefs[inst.defIdx]);
        } else if (inst.kind == ModuleKind::TemporalPE) {
          const auto &tpeDef = temporalPEDefs[inst.defIdx];
          for (unsigned idx : tpeDef.fuPEDefIndices)
            addOpsFromDef(peDefs[idx]);
        }
      }
      for (const auto &op : opsForDialect) {
        std::string svFile = opToSVModule(op) + ".sv";
        copyTemplateFile(templateDir, it->second + "/" + svFile,
                         libDir.str().str());
      }
    }
  }

  // ---- Generate top-level module ----
  std::ostringstream top;
  top << "//===-- " << moduleName << "_top.sv - Generated top-level module ---===//\n";
  top << "//\n";
  top << "// Generated by Loom ADG Builder. Do not edit.\n";
  top << "//\n";
  top << "//===----------------------------------------------------------------------===//\n\n";

  // Module port declarations
  top << "module " << moduleName << "_top (\n";
  top << "    input  logic clk,\n";

  // Collect stream ports
  std::vector<const ModulePort *> streamInputs, streamOutputs;
  for (const auto &p : ports) {
    if (p.isInput)
      streamInputs.push_back(&p);
    else
      streamOutputs.push_back(&p);
  }

  // Collect instance config ports (switch route tables, bypassable FIFO cfg)
  struct InstCfgPort {
    std::string portName; // top-level port name
    unsigned width;
  };
  std::vector<InstCfgPort> instCfgPorts;
  for (const auto &inst : instances) {
    if (inst.kind == ModuleKind::Switch) {
      const auto &def = switchDefs[inst.defIdx];
      instCfgPorts.push_back({inst.name + "_cfg_route_table",
                              getNumConnected(def)});
    } else if (inst.kind == ModuleKind::Fifo) {
      const auto &def = fifoDefs[inst.defIdx];
      if (def.bypassable)
        instCfgPorts.push_back({inst.name + "_cfg_data", 1});
    } else if (inst.kind == ModuleKind::AddTag) {
      const auto &def = addTagDefs[inst.defIdx];
      unsigned tw = getDataWidthBits(def.tagType);
      instCfgPorts.push_back({inst.name + "_cfg_data", tw});
    } else if (inst.kind == ModuleKind::MapTag) {
      const auto &def = mapTagDefs[inst.defIdx];
      unsigned itw = getDataWidthBits(def.inputTagType);
      unsigned otw = getDataWidthBits(def.outputTagType);
      unsigned entryWidth = 1 + itw + otw;
      instCfgPorts.push_back(
          {inst.name + "_cfg_data", def.tableSize * entryWidth});
    } else if (inst.kind == ModuleKind::PE) {
      const auto &def = peDefs[inst.defIdx];
      unsigned tw = def.inputPorts.size() > 0
                        ? getTagWidthBits(def.inputPorts[0])
                        : 0;
      if (tw > 0) {
        unsigned cfgBits = def.outputPorts.size() * tw;
        instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
      }
    } else if (inst.kind == ModuleKind::ConstantPE) {
      const auto &def = constantPEDefs[inst.defIdx];
      unsigned dw = getDataWidthBits(def.outputType);
      unsigned tw = getTagWidthBits(def.outputType);
      unsigned cfgBits = (tw > 0) ? dw + tw : dw;
      instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
    } else if (inst.kind == ModuleKind::LoadPE) {
      const auto &def = loadPEDefs[inst.defIdx];
      if (def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", def.tagWidth});
    } else if (inst.kind == ModuleKind::StorePE) {
      const auto &def = storePEDefs[inst.defIdx];
      if (def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", def.tagWidth});
    } else if (inst.kind == ModuleKind::TemporalSwitch) {
      const auto &def = temporalSwitchDefs[inst.defIdx];
      unsigned tw = getTagWidthBits(def.interfaceType);
      unsigned numConn = getNumConnected(def);
      unsigned entryWidth = 1 + tw + numConn;
      instCfgPorts.push_back(
          {inst.name + "_cfg_data", def.numRouteTable * entryWidth});
    } else if (inst.kind == ModuleKind::TemporalPE) {
      const auto &def = temporalPEDefs[inst.defIdx];
      unsigned tw = getTagWidthBits(def.interfaceType);
      unsigned numIn = 1, numOut = 1;
      if (!def.fuPEDefIndices.empty()) {
        numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
        numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
      }
      unsigned fuTypes = def.fuPEDefIndices.size();
      // log2Ceil helper
      auto log2Ceil = [](unsigned v) -> unsigned {
        if (v <= 1) return 0;
        unsigned bits = 0;
        v--;
        while (v > 0) { bits++; v >>= 1; }
        return bits;
      };
      unsigned regBits = (def.numRegisters > 0) ?
          (1 + log2Ceil(std::max(def.numRegisters, 2u))) : 0;
      unsigned fuSelBits = (fuTypes > 1) ? log2Ceil(fuTypes) : 0;
      unsigned resBits = regBits;
      unsigned resultWidth = resBits + tw;
      unsigned insnWidth = 1 + tw + fuSelBits + numIn * regBits + numOut * resultWidth;
      unsigned cfgBits = def.numInstructions * insnWidth;
      if (cfgBits > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
    }
    // Memory and ExtMemory have CONFIG_WIDTH = 0, no config ports
  }
  bool hasCfgPorts = !instCfgPorts.empty();

  bool hasMorePorts = !streamInputs.empty() || !streamOutputs.empty() ||
                      hasCfgPorts || hasErrorPorts;
  top << "    input  logic rst_n" << (hasMorePorts ? "," : "") << "\n";

  for (size_t i = 0; i < streamInputs.size(); ++i) {
    const auto *p = streamInputs[i];
    unsigned w = getDataWidthBits(p->type) + getTagWidthBits(p->type);
    bool last = (i + 1 == streamInputs.size()) && streamOutputs.empty() &&
                !hasCfgPorts && !hasErrorPorts;
    top << "    input  logic " << p->name << "_valid,\n";
    top << "    output logic " << p->name << "_ready,\n";
    top << "    input  logic " << (w > 1 ? "[" + std::to_string(w-1) + ":0] " : "")
        << p->name << "_data" << (last ? "" : ",") << "\n";
  }

  for (size_t i = 0; i < streamOutputs.size(); ++i) {
    const auto *p = streamOutputs[i];
    unsigned w = getDataWidthBits(p->type) + getTagWidthBits(p->type);
    bool last = (i + 1 == streamOutputs.size()) && !hasCfgPorts &&
                !hasErrorPorts;
    top << "    output logic " << p->name << "_valid,\n";
    top << "    input  logic " << p->name << "_ready,\n";
    top << "    output logic " << (w > 1 ? "[" + std::to_string(w-1) + ":0] " : "")
        << p->name << "_data" << (last ? "" : ",") << "\n";
  }

  // Per-instance config input ports
  for (size_t i = 0; i < instCfgPorts.size(); ++i) {
    const auto &cp = instCfgPorts[i];
    bool last = (i + 1 == instCfgPorts.size()) && !hasErrorPorts;
    top << "    input  logic "
        << (cp.width > 1 ? "[" + std::to_string(cp.width - 1) + ":0] " : "")
        << cp.portName << (last ? "" : ",") << "\n";
  }

  if (hasErrorPorts) {
    top << "    output logic        error_valid,\n";
    top << "    output logic [15:0] error_code\n";
  }
  top << ");\n\n";

  // Wire declarations for internal connections
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = getInstanceInputCount(i);
    unsigned numOut = getInstanceOutputCount(i);

    // For PE instances, all data wires use the PE's PAYLOAD_WIDTH (= max
    // data width across all ports + tag width) so that the packed-array port
    // connection has uniform element width.  This matters for conversion ops
    // (e.g. arith.extsi i16->i32) where individual port types differ.
    unsigned pePayloadWidth = 0;
    if (inst.kind == ModuleKind::PE) {
      const auto &def = peDefs[inst.defIdx];
      unsigned dw = 0;
      for (const auto &pt : def.inputPorts)
        dw = std::max(dw, getDataWidthBits(pt));
      for (const auto &pt : def.outputPorts)
        dw = std::max(dw, getDataWidthBits(pt));
      if (dw == 0)
        dw = 32;
      unsigned tw = !def.inputPorts.empty()
                        ? getTagWidthBits(def.inputPorts[0])
                        : 0;
      pePayloadWidth = dw + tw;
    }

    for (unsigned p = 0; p < numIn; ++p) {
      unsigned w;
      if (pePayloadWidth > 0)
        w = pePayloadWidth;
      else {
        Type pt = getInstanceInputType(i, p);
        w = getDataWidthBits(pt) + getTagWidthBits(pt);
      }
      top << "  logic " << inst.name << "_in" << p << "_valid;\n";
      top << "  logic " << inst.name << "_in" << p << "_ready;\n";
      if (w > 1)
        top << "  logic [" << (w-1) << ":0] " << inst.name << "_in" << p << "_data;\n";
      else
        top << "  logic " << inst.name << "_in" << p << "_data;\n";
    }
    for (unsigned p = 0; p < numOut; ++p) {
      unsigned w;
      if (pePayloadWidth > 0)
        w = pePayloadWidth;
      else {
        Type pt = getInstanceOutputType(i, p);
        w = getDataWidthBits(pt) + getTagWidthBits(pt);
      }
      top << "  logic " << inst.name << "_out" << p << "_valid;\n";
      top << "  logic " << inst.name << "_out" << p << "_ready;\n";
      if (w > 1)
        top << "  logic [" << (w-1) << ":0] " << inst.name << "_out" << p << "_data;\n";
      else
        top << "  logic " << inst.name << "_out" << p << "_data;\n";
    }

    // Per-instance error signals for modules that support them
    if (hasErrorOutput(inst.kind)) {
      top << "  logic " << inst.name << "_error_valid;\n";
      top << "  logic [15:0] " << inst.name << "_error_code;\n";
    }
    top << "\n";
  }

  // Instantiate components
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    top << "  // Instance: " << inst.name << " (" << svModuleName(inst.kind) << ")\n";

    switch (inst.kind) {
    case ModuleKind::Switch: {
      const auto &def = switchDefs[inst.defIdx];
      top << "  fabric_switch #(\n" << genSwitchParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      // Port connections
      top << "    .in_valid({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .cfg_route_table(" << inst.name << "_cfg_route_table),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::Fifo: {
      const auto &def = fifoDefs[inst.defIdx];
      top << "  fabric_fifo #(\n" << genFifoParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      if (def.bypassable)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::AddTag: {
      const auto &def = addTagDefs[inst.defIdx];
      top << "  fabric_add_tag #(\n" << genAddTagParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::DelTag: {
      const auto &def = delTagDefs[inst.defIdx];
      top << "  fabric_del_tag #(\n" << genDelTagParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::MapTag: {
      const auto &def = mapTagDefs[inst.defIdx];
      top << "  fabric_map_tag #(\n" << genMapTagParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::PE: {
      const auto &def = peDefs[inst.defIdx];
      unsigned numIn = def.inputPorts.size();
      unsigned numOut = def.outputPorts.size();
      top << "  " << inst.name << "_pe #(\n" << genPEParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      // Input port connections (packed array)
      top << "    .in_valid({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      // Config: output tags for tagged PE, '0 for native
      unsigned tw = numIn > 0 ? getTagWidthBits(def.inputPorts[0]) : 0;
      if (tw > 0 && instCfgPorts.size() > 0)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::ConstantPE: {
      const auto &def = constantPEDefs[inst.defIdx];
      top << "  fabric_pe_constant #(\n" << genConstantPEParams(def)
          << "\n  ) " << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::LoadPE: {
      const auto &def = loadPEDefs[inst.defIdx];
      top << "  fabric_pe_load #(\n" << genLoadPEParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in0_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in0_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in0_data(" << inst.name << "_in0_data),\n";
      top << "    .in1_valid(" << inst.name << "_in1_valid),\n";
      top << "    .in1_ready(" << inst.name << "_in1_ready),\n";
      top << "    .in1_data(" << inst.name << "_in1_data),\n";
      top << "    .in2_valid(" << inst.name << "_in2_valid),\n";
      top << "    .in2_ready(" << inst.name << "_in2_ready),\n";
      top << "    .in2_data(" << inst.name << "_in2_data),\n";
      top << "    .out0_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out0_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out0_data(" << inst.name << "_out0_data),\n";
      top << "    .out1_valid(" << inst.name << "_out1_valid),\n";
      top << "    .out1_ready(" << inst.name << "_out1_ready),\n";
      top << "    .out1_data(" << inst.name << "_out1_data),\n";
      unsigned tw = def.tagWidth;
      if (def.hwType == HardwareType::TagOverwrite && tw > 0)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::StorePE: {
      const auto &def = storePEDefs[inst.defIdx];
      top << "  fabric_pe_store #(\n" << genStorePEParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in0_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in0_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in0_data(" << inst.name << "_in0_data),\n";
      top << "    .in1_valid(" << inst.name << "_in1_valid),\n";
      top << "    .in1_ready(" << inst.name << "_in1_ready),\n";
      top << "    .in1_data(" << inst.name << "_in1_data),\n";
      top << "    .in2_valid(" << inst.name << "_in2_valid),\n";
      top << "    .in2_ready(" << inst.name << "_in2_ready),\n";
      top << "    .in2_data(" << inst.name << "_in2_data),\n";
      top << "    .out0_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out0_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out0_data(" << inst.name << "_out0_data),\n";
      top << "    .out1_valid(" << inst.name << "_out1_valid),\n";
      top << "    .out1_ready(" << inst.name << "_out1_ready),\n";
      top << "    .out1_data(" << inst.name << "_out1_data),\n";
      unsigned tw = def.tagWidth;
      if (def.hwType == HardwareType::TagOverwrite && tw > 0)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::TemporalSwitch: {
      const auto &def = temporalSwitchDefs[inst.defIdx];
      top << "  fabric_temporal_sw #(\n" << genTemporalSwitchParams(def)
          << "\n  ) " << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::TemporalPE: {
      const auto &def = temporalPEDefs[inst.defIdx];
      unsigned numIn = 1, numOut = 1;
      if (!def.fuPEDefIndices.empty()) {
        numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
        numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
      }
      top << "  " << inst.name << "_temporal_pe #(\n" << genTemporalPEParams(def, peDefs)
          << "\n  ) " << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::Memory: {
      const auto &def = memoryDefs[inst.defIdx];
      unsigned numIn = getInstanceInputCount(i);
      unsigned numOut = getInstanceOutputCount(i);
      top << "  fabric_memory #(\n" << genMemoryParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::ExtMemory: {
      const auto &def = extMemoryDefs[inst.defIdx];
      unsigned numIn = getInstanceInputCount(i);
      unsigned numOut = getInstanceOutputCount(i);
      top << "  fabric_extmemory #(\n" << genExtMemoryParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    default:
      llvm_unreachable("unsupported module kind (should be caught earlier)");
    }
  }

  // Collect ready sources per output port for fanout aggregation.
  // Key: "<inst_name>_out<port>" or "%<module_port_name>" -> list of ready signals.
  std::map<std::string, std::vector<std::string>> readySources;

  // Wire connections: module inputs to instances
  for (const auto &conn : inputConns) {
    const auto &port = ports[conn.portIdx];
    const auto &inst = instances[conn.instIdx];
    std::string sinkReady = inst.name + "_in" + std::to_string(conn.dstPort) + "_ready";
    top << "  assign " << inst.name << "_in" << conn.dstPort
        << "_valid = " << port.name << "_valid;\n";
    top << "  assign " << inst.name << "_in" << conn.dstPort
        << "_data = " << port.name << "_data;\n";
    readySources["%" + port.name].push_back(sinkReady);
  }
  // Emit aggregated ready for module input ports; drive unconnected to 0
  for (const auto *p : streamInputs) {
    std::string key = "%" + p->name;
    auto it = readySources.find(key);
    if (it == readySources.end()) {
      top << "  assign " << p->name << "_ready = 1'b0;\n";
    } else if (it->second.size() == 1) {
      top << "  assign " << p->name << "_ready = " << it->second[0] << ";\n";
    } else {
      top << "  assign " << p->name << "_ready = " << it->second[0];
      for (size_t s = 1; s < it->second.size(); ++s)
        top << " & " << it->second[s];
      top << ";\n";
    }
  }
  readySources.clear();

  // Wire connections: instances to module outputs
  std::set<unsigned> assignedOutputPorts;
  for (const auto &conn : outputConns) {
    const auto &inst = instances[conn.instIdx];
    const auto &port = ports[conn.portIdx];
    if (!assignedOutputPorts.insert(conn.portIdx).second) {
      llvm::errs() << "error: exportSV: module output '" << port.name
                   << "' has multiple source connections\n";
      std::exit(1);
    }
    std::string srcKey = inst.name + "_out" + std::to_string(conn.srcPort);
    top << "  assign " << port.name << "_valid = " << srcKey << "_valid;\n";
    top << "  assign " << port.name << "_data = " << srcKey << "_data;\n";
    readySources[srcKey].push_back(port.name + "_ready");
  }

  // Wire connections: internal (instance to instance)
  for (const auto &conn : internalConns) {
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    std::string srcKey = srcInst.name + "_out" + std::to_string(conn.srcPort);
    top << "  assign " << dstInst.name << "_in" << conn.dstPort
        << "_valid = " << srcKey << "_valid;\n";
    top << "  assign " << dstInst.name << "_in" << conn.dstPort
        << "_data = " << srcKey << "_data;\n";
    readySources[srcKey].push_back(
        dstInst.name + "_in" + std::to_string(conn.dstPort) + "_ready");
  }

  // Emit aggregated ready for instance output ports
  for (const auto &[srcKey, sources] : readySources) {
    if (sources.size() == 1) {
      top << "  assign " << srcKey << "_ready = " << sources[0] << ";\n";
    } else {
      top << "  assign " << srcKey << "_ready = " << sources[0];
      for (size_t s = 1; s < sources.size(); ++s)
        top << " & " << sources[s];
      top << ";\n";
    }
  }

  // Error aggregation: latch first error until reset
  if (hasErrorPorts) {
    top << "\n  // Error latch: captures first error, held until reset\n";
    top << "  always_ff @(posedge clk or negedge rst_n) begin\n";
    top << "    if (!rst_n) begin\n";
    top << "      error_valid <= 1'b0;\n";
    top << "      error_code  <= 16'd0;\n";
    top << "    end else if (!error_valid) begin\n";
    // Only capture when no error latched yet
    bool first = true;
    for (const auto &inst : instances) {
      if (hasErrorOutput(inst.kind)) {
        top << "      " << (first ? "if" : "else if") << " ("
            << inst.name << "_error_valid) begin\n";
        top << "        error_valid <= 1'b1;\n";
        top << "        error_code  <= " << inst.name << "_error_code;\n";
        top << "      end\n";
        first = false;
      }
    }
    top << "    end\n";
    top << "  end\n";
  }

  top << "\nendmodule\n";

  // Write top-level module
  llvm::SmallString<256> topPath(directory);
  llvm::sys::path::append(topPath, moduleName + "_top.sv");
  writeFile(topPath.str().str(), top.str());
}

//===----------------------------------------------------------------------===//
// ADGBuilder::exportSV
//===----------------------------------------------------------------------===//

void ADGBuilder::exportSV(const std::string &directory) {
  auto validation = validateADG();
  if (!validation.success) {
    llvm::errs() << "error: ADG validation failed with "
                 << validation.errors.size() << " error(s):\n";
    for (const auto &err : validation.errors) {
      llvm::errs() << "  [" << err.code << "] " << err.message;
      if (!err.location.empty())
        llvm::errs() << " (at " << err.location << ")";
      llvm::errs() << "\n";
    }
    std::exit(1);
  }

  impl_->generateSV(directory);
}

} // namespace adg
} // namespace loom
