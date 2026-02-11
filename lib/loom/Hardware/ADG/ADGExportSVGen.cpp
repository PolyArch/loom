//===-- ADGExportSVGen.cpp - SV export generation functions ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Module generation and parameter computation for the SystemVerilog export.
//
//===----------------------------------------------------------------------===//

#include "ADGExportSVInternal.h"

#include "loom/Hardware/ADG/ADGBuilderImpl.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// SV module kind name
//===----------------------------------------------------------------------===//

const char *svModuleName(ModuleKind kind) {
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

unsigned getNumConnected(const SwitchDef &def) {
  if (def.connectivity.empty())
    return def.numIn * def.numOut;
  unsigned count = 0;
  for (const auto &row : def.connectivity)
    for (bool v : row)
      if (v) ++count;
  return count;
}

unsigned getNumConnected(const TemporalSwitchDef &def) {
  if (def.connectivity.empty())
    return def.numIn * def.numOut;
  unsigned count = 0;
  for (const auto &row : def.connectivity)
    for (bool v : row)
      if (v) ++count;
  return count;
}

//===----------------------------------------------------------------------===//
// PE body generation (static -- only called within this file)
//===----------------------------------------------------------------------===//

/// Generate SV body from bodyMLIR containing multiple operations.
static std::string genMultiOpBodySV(const PEDef &def) {
  std::string body = def.bodyMLIR;

  // Strip ^bb0(...): header if present, extracting named block args
  // so they can be renamed to %argN in the body text.
  std::vector<std::string> blockArgNames;
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos) {
      std::string argList = body.substr(bbPos + 5, closePos - (bbPos + 5));
      // Parse "%name: type, %name2: type2, ..."
      std::istringstream argStream(argList);
      std::string token;
      while (std::getline(argStream, token, ',')) {
        auto pct = token.find('%');
        if (pct != std::string::npos) {
          auto colon = token.find(':', pct);
          if (colon != std::string::npos)
            blockArgNames.push_back(token.substr(pct, colon - pct));
          else
            blockArgNames.push_back(
                token.substr(pct, token.find_first_of(" \t\n", pct) - pct));
        }
      }
      body = body.substr(closePos + 2);
      if (!body.empty() && body[0] == '\n')
        body = body.substr(1);
    }
  }

  // Rename user block args to %argN in the body text via a two-phase
  // approach to avoid SSA collisions (e.g. body already contains %arg0).
  // Phase 1: rename original names to unique temporaries.
  // Phase 2: rename temporaries to final %argN names.
  //
  // Helper: whole-token replacement (next char must not be alnum/_).
  auto replaceWholeToken = [](std::string &s, const std::string &from,
                              const std::string &to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      size_t endPos = pos + from.size();
      if (endPos < s.size() &&
          (std::isalnum(s[endPos]) || s[endPos] == '_')) {
        pos = endPos;
        continue;
      }
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  };

  // Phase 1: original names -> unique temporaries (%__loom_tmp_N)
  std::vector<std::pair<std::string, std::string>> phase1;
  for (size_t i = 0; i < blockArgNames.size(); ++i)
    phase1.push_back({blockArgNames[i],
                      "%__loom_tmp_" + std::to_string(i)});
  // Sort by name length descending to avoid partial replacement
  std::sort(phase1.begin(), phase1.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });
  for (const auto &[from, to] : phase1) {
    if (from == to)
      continue;
    replaceWholeToken(body, from, to);
  }

  // Phase 2: temporaries -> final %argN names
  for (size_t i = 0; i < blockArgNames.size(); ++i) {
    std::string tmp = "%__loom_tmp_" + std::to_string(i);
    std::string final_ = "%arg" + std::to_string(i);
    if (tmp != final_)
      replaceWholeToken(body, tmp, final_);
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
  for (size_t i = 0; i < def.inputPorts.size(); ++i)
    ssaToSV["%arg" + std::to_string(i)] =
        "in_value[" + std::to_string(i) + "]";

  // Track SSA use counts for fork logic
  std::map<std::string, unsigned> ssaUseCount;
  for (const auto &stmt : stmts)
    for (const auto &op : stmt.operands)
      ssaUseCount[op]++;
  for (const auto &yo : yieldOperands)
    ssaUseCount[yo]++;

  // Maps: SSA name -> valid/ready SV wire names
  std::map<std::string, std::string> ssaToValid;
  std::map<std::string, std::string> ssaToBase;
  std::map<std::string, unsigned> ssaNextUse;

  for (size_t i = 0; i < def.inputPorts.size(); ++i) {
    std::string ssa = "%arg" + std::to_string(i);
    ssaToBase[ssa] = "in_value_" + std::to_string(i);
    ssaToValid[ssa] = "in_value_" + std::to_string(i) + "_valid";
  }

  // Helper: get the ready wire for the next use of an SSA value.
  // For single-use values, returns "<base>_ready".
  // For multi-use values, returns "<base>_fork_ready_<N>" and increments N.
  auto getUseReady = [&](const std::string &ssa) -> std::string {
    unsigned uses = ssaUseCount.count(ssa) ? ssaUseCount[ssa] : 1;
    if (uses > 1) {
      unsigned idx = ssaNextUse[ssa]++;
      return ssaToBase[ssa] + "_fork_ready_" + std::to_string(idx);
    }
    return ssaToBase[ssa] + "_ready";
  };

  // Helper: get the valid wire for the next use of an SSA value.
  // For single-use values, returns the source valid wire.
  // For multi-use values, returns the eager fork valid wire (source valid
  // gated by all sibling readies).  Must be called before getUseReady
  // for the same use so they share the same index.
  auto getUseValid = [&](const std::string &ssa) -> std::string {
    unsigned uses = ssaUseCount.count(ssa) ? ssaUseCount[ssa] : 1;
    if (uses > 1) {
      unsigned idx = ssaNextUse.count(ssa) ? ssaNextUse[ssa] : 0;
      return ssaToBase[ssa] + "_fork_valid_" + std::to_string(idx);
    }
    return ssaToValid.count(ssa) ? ssaToValid[ssa] : "1'b1";
  };

  std::ostringstream os;

  // Emit eager fork wires for multi-use input ports.
  // in_value_N_valid and in_value_N_ready are declared in the PE skeleton.
  auto emitForkWires = [&](const std::string &base, const std::string &ssa) {
    unsigned uses = ssaUseCount.count(ssa) ? ssaUseCount[ssa] : 0;
    if (uses > 1) {
      // Fork ready wires
      for (unsigned u = 0; u < uses; ++u)
        os << "  logic " << base << "_fork_ready_" << u << ";\n";
      os << "  assign " << base << "_ready = ";
      for (unsigned u = 0; u < uses; ++u) {
        if (u > 0) os << " & ";
        os << base << "_fork_ready_" << u;
      }
      os << ";\n";
      // Fork valid wires (eager fork pattern)
      for (unsigned u = 0; u < uses; ++u) {
        os << "  logic " << base << "_fork_valid_" << u << ";\n";
        os << "  assign " << base << "_fork_valid_" << u << " = "
           << base << "_valid";
        for (unsigned j = 0; j < uses; ++j) {
          if (j != u)
            os << " & " << base << "_fork_ready_" << j;
        }
        os << ";\n";
      }
    }
    // For single-use: in_value_N_ready driven by operation module port
  };

  for (size_t i = 0; i < def.inputPorts.size(); ++i) {
    std::string ssa = "%arg" + std::to_string(i);
    std::string base = "in_value_" + std::to_string(i);
    emitForkWires(base, ssa);
  }
  os << "\n";

  unsigned wireIdx = 0;

  for (const auto &stmt : stmts) {
    std::string svModule = opToSVModule(stmt.opName);
    std::string wireName = "body_t" + std::to_string(wireIdx);
    ssaToSV[stmt.result] = wireName;
    ssaToBase[stmt.result] = wireName;
    ssaToValid[stmt.result] = wireName + "_valid";

    // Declare data, valid, ready wires for this intermediate
    if (isCompareOp(stmt.opName))
      os << "  logic " << wireName << ";\n";
    else
      os << "  logic [SAFE_DW-1:0] " << wireName << ";\n";
    os << "  logic " << wireName << "_valid;\n";
    os << "  logic " << wireName << "_ready;\n";
    emitForkWires(wireName, stmt.result);

    if (isConversionOp(stmt.opName)) {
      auto [inW, outW] = parseConversionWidths(stmt.typeAnnotation);
      std::string aV = getUseValid(stmt.operands[0]);
      std::string aR = getUseReady(stmt.operands[0]);
      if (inW > 0 && outW > 0) {
        os << "  " << svModule << " #(.IN_WIDTH(" << inW << "), .OUT_WIDTH("
           << outW << ")) u_op" << wireIdx << " (\n";
        os << "    .a_valid(" << aV << "),\n";
        os << "    .a_ready(" << aR << "),\n";
        os << "    .a_data(" << ssaToSV[stmt.operands[0]] << "[" << (inW - 1)
           << ":0]),\n";
        os << "    .result_valid(" << wireName << "_valid),\n";
        os << "    .result_ready(" << wireName << "_ready),\n";
        os << "    .result_data(" << wireName << "[" << (outW - 1) << ":0])\n";
        os << "  );\n";
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
        os << "    .a_valid(" << aV << "),\n";
        os << "    .a_ready(" << aR << "),\n";
        os << "    .a_data(" << ssaToSV[stmt.operands[0]] << "),\n";
        os << "    .result_valid(" << wireName << "_valid),\n";
        os << "    .result_ready(" << wireName << "_ready),\n";
        os << "    .result_data(" << wireName << ")\n";
        os << "  );\n";
      }
    } else if (isCompareOp(stmt.opName)) {
      int predVal = resolveComparePredicate(stmt.opName, stmt.predicate);
      unsigned cmpW = parseMLIRTypeWidth(stmt.typeAnnotation);
      std::string aV = getUseValid(stmt.operands[0]);
      std::string aR = getUseReady(stmt.operands[0]);
      if (cmpW > 0) {
        os << "  " << svModule << " #(.WIDTH(" << cmpW << "), .PREDICATE("
           << predVal << ")) u_op" << wireIdx << " (\n";
        os << "    .a_valid(" << aV << "),\n";
        os << "    .a_ready(" << aR << "),\n";
        os << "    .a_data(" << ssaToSV[stmt.operands[0]] << "[" << (cmpW - 1)
           << ":0]),\n";
        if (stmt.operands.size() > 1) {
          std::string bV = getUseValid(stmt.operands[1]);
          std::string bR = getUseReady(stmt.operands[1]);
          os << "    .b_valid(" << bV << "),\n";
          os << "    .b_ready(" << bR << "),\n";
          os << "    .b_data(" << ssaToSV[stmt.operands[1]] << "[" << (cmpW - 1)
             << ":0]),\n";
        }
      } else {
        os << "  " << svModule << " #(.WIDTH(DATA_WIDTH), .PREDICATE("
           << predVal << ")) u_op" << wireIdx << " (\n";
        os << "    .a_valid(" << aV << "),\n";
        os << "    .a_ready(" << aR << "),\n";
        os << "    .a_data(" << ssaToSV[stmt.operands[0]] << "),\n";
        if (stmt.operands.size() > 1) {
          std::string bV = getUseValid(stmt.operands[1]);
          std::string bR = getUseReady(stmt.operands[1]);
          os << "    .b_valid(" << bV << "),\n";
          os << "    .b_ready(" << bR << "),\n";
          os << "    .b_data(" << ssaToSV[stmt.operands[1]] << "),\n";
        }
      }
      os << "    .result_valid(" << wireName << "_valid),\n";
      os << "    .result_ready(" << wireName << "_ready),\n";
      os << "    .result_data(" << wireName << ")\n";
      os << "  );\n";
    } else {
      unsigned opW = parseMLIRTypeWidth(stmt.typeAnnotation);
      if (stmt.opName == "arith.select") {
        auto commaPos = stmt.typeAnnotation.find(',');
        if (commaPos != std::string::npos)
          opW = parseMLIRTypeWidth(stmt.typeAnnotation.substr(commaPos + 1));
      }
      bool useNarrow = opW > 0;
      if (useNarrow)
        os << "  " << svModule << " #(.WIDTH(" << opW << ")) u_op" << wireIdx
           << " (\n";
      else
        os << "  " << svModule << " #(.WIDTH(DATA_WIDTH)) u_op" << wireIdx
           << " (\n";

      // Helper: emit a handshake operand with valid/ready/data.
      auto emitOp = [&](const std::string &port,
                         const std::string &operandSSA) {
        std::string opV = getUseValid(operandSSA);
        std::string opR = getUseReady(operandSSA);
        os << "    ." << port << "_valid(" << opV << "),\n";
        os << "    ." << port << "_ready(" << opR << "),\n";
        if (useNarrow)
          os << "    ." << port << "_data(" << ssaToSV[operandSSA] << "["
             << (opW - 1) << ":0]),\n";
        else
          os << "    ." << port << "_data(" << ssaToSV[operandSSA] << "),\n";
      };

      if (stmt.opName == "arith.select" && stmt.operands.size() >= 3) {
        std::string condV = getUseValid(stmt.operands[0]);
        std::string condR = getUseReady(stmt.operands[0]);
        os << "    .condition_valid(" << condV << "),\n";
        os << "    .condition_ready(" << condR << "),\n";
        os << "    .condition_data(" << ssaToSV[stmt.operands[0]] << "[0]),\n";
        emitOp("a", stmt.operands[1]);
        emitOp("b", stmt.operands[2]);
      } else if (stmt.opName == "math.fma" && stmt.operands.size() >= 3) {
        emitOp("a", stmt.operands[0]);
        emitOp("b", stmt.operands[1]);
        emitOp("c", stmt.operands[2]);
      } else if (stmt.operands.size() >= 2) {
        emitOp("a", stmt.operands[0]);
        emitOp("b", stmt.operands[1]);
      } else if (stmt.operands.size() >= 1) {
        emitOp("a", stmt.operands[0]);
      }

      if (useNarrow) {
        os << "    .result_valid(" << wireName << "_valid),\n";
        os << "    .result_ready(" << wireName << "_ready),\n";
        os << "    .result_data(" << wireName << "[" << (opW - 1) << ":0])\n";
        os << "  );\n";
        os << "  generate\n";
        os << "    if (DATA_WIDTH > " << opW << ") begin : g_op_pad_"
           << wireIdx << "\n";
        os << "      assign " << wireName << "[DATA_WIDTH-1:" << opW
           << "] = '0;\n";
        os << "    end\n";
        os << "  endgenerate\n";
      } else {
        os << "    .result_valid(" << wireName << "_valid),\n";
        os << "    .result_ready(" << wireName << "_ready),\n";
        os << "    .result_data(" << wireName << ")\n";
        os << "  );\n";
      }
    }
    os << "\n";
    ++wireIdx;
  }

  // Assign body_result and body_valid from yield operands
  std::vector<std::string> yieldValidWires;
  if (!yieldOperands.empty()) {
    for (size_t i = 0; i < yieldOperands.size(); ++i) {
      auto it = ssaToSV.find(yieldOperands[i]);
      std::string src = (it != ssaToSV.end()) ? it->second : "'0";
      // Check if this yield operand comes from a compare op
      bool isYieldCmp = false;
      for (const auto &st : stmts) {
        if (st.result == yieldOperands[i] && isCompareOp(st.opName)) {
          isYieldCmp = true;
          break;
        }
      }
      if (isYieldCmp)
        os << "  assign body_result[" << i
           << "] = {{(SAFE_DW-1){1'b0}}, " << src << "};\n";
      else
        os << "  assign body_result[" << i << "] = " << src << ";\n";

      std::string yv = getUseValid(yieldOperands[i]);
      yieldValidWires.push_back(yv);
      std::string yr = getUseReady(yieldOperands[i]);
      os << "  assign " << yr << " = body_ready;\n";
    }
  } else if (!stmts.empty()) {
    os << "  assign body_result[0] = " << ssaToSV[stmts.back().result]
       << ";\n";
    yieldValidWires.push_back(getUseValid(stmts.back().result));
    std::string yr = getUseReady(stmts.back().result);
    os << "  assign " << yr << " = body_ready;\n";
  }

  // body_valid = AND of all yield valid signals
  os << "  assign body_valid = ";
  if (yieldValidWires.empty()) {
    os << "1'b0";
  } else {
    for (size_t i = 0; i < yieldValidWires.size(); ++i) {
      if (i > 0) os << " & ";
      os << yieldValidWires[i];
    }
  }
  os << ";\n";

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
  os << "  logic u_body_result_valid;\n";

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
    os << "    .a_valid(in_value_0_valid),\n";
    os << "    .a_ready(in_value_0_ready),\n";
    os << "    .a_data(in_value[0][" << (inW - 1) << ":0]),\n";
    os << "    .result_valid(u_body_result_valid),\n";
    os << "    .result_ready(body_ready),\n";
    os << "    .result_data(body_result[0][" << (outW - 1) << ":0])\n";
    os << "  );\n";
    // Zero-fill upper bits if OUT_WIDTH < DATA_WIDTH
    os << "  generate\n";
    os << "    if (DATA_WIDTH > " << outW << ") begin : g_conv_pad\n";
    os << "      assign body_result[0][DATA_WIDTH-1:" << outW << "] = '0;\n";
    os << "    end\n";
    os << "  endgenerate\n";
  } else if (isCompareOp(def.singleOp)) {
    // Compare ops have a PREDICATE parameter and 1-bit output.
    // Clamp to valid range: cmpi 0-9, cmpf 0-15.
    int maxPred = (def.singleOp == "arith.cmpf") ? 15 : 9;
    int pred = def.comparePredicate;
    if (pred < 0 || pred > maxPred) pred = 0;
    os << "  logic cmp_result;\n";
    os << "  " << svModule
       << " #(.WIDTH(DATA_WIDTH), .PREDICATE(" << pred
       << ")) u_body (\n";
    os << "    .a_valid(in_value_0_valid),\n";
    os << "    .a_ready(in_value_0_ready),\n";
    os << "    .a_data(in_value[0]),\n";
    os << "    .b_valid(in_value_1_valid),\n";
    os << "    .b_ready(in_value_1_ready),\n";
    os << "    .b_data(in_value[1]),\n";
    os << "    .result_valid(u_body_result_valid),\n";
    os << "    .result_ready(body_ready),\n";
    os << "    .result_data(cmp_result)\n";
    os << "  );\n";
    // Zero-extend 1-bit result to DATA_WIDTH
    os << "  assign body_result[0] = {{(DATA_WIDTH-1){1'b0}}, cmp_result};\n";
  } else if (def.singleOp == "dataflow.invariant") {
    os << "  dataflow_invariant #(.WIDTH(DATA_WIDTH)) u_body (\n";
    os << "    .clk(clk),\n";
    os << "    .rst_n(rst_n),\n";
    os << "    .d_valid(in_value_0_valid),\n";
    os << "    .d_ready(in_value_0_ready),\n";
    os << "    .d_data(in_value[0][0]),\n";
    os << "    .a_valid(in_value_1_valid),\n";
    os << "    .a_ready(in_value_1_ready),\n";
    os << "    .a_data(in_value[1]),\n";
    os << "    .o_valid(u_body_result_valid),\n";
    os << "    .o_ready(body_ready),\n";
    os << "    .o_data(body_result[0])\n";
    os << "  );\n";
  } else if (def.singleOp == "dataflow.carry") {
    os << "  dataflow_carry #(.WIDTH(DATA_WIDTH)) u_body (\n";
    os << "    .clk(clk),\n";
    os << "    .rst_n(rst_n),\n";
    os << "    .d_valid(in_value_0_valid),\n";
    os << "    .d_ready(in_value_0_ready),\n";
    os << "    .d_data(in_value[0][0]),\n";
    os << "    .a_valid(in_value_1_valid),\n";
    os << "    .a_ready(in_value_1_ready),\n";
    os << "    .a_data(in_value[1]),\n";
    os << "    .b_valid(in_value_2_valid),\n";
    os << "    .b_ready(in_value_2_ready),\n";
    os << "    .b_data(in_value[2]),\n";
    os << "    .o_valid(u_body_result_valid),\n";
    os << "    .o_ready(body_ready),\n";
    os << "    .o_data(body_result[0])\n";
    os << "  );\n";
  } else if (def.singleOp == "dataflow.gate") {
    os << "  logic gate_av_valid;\n";
    os << "  logic gate_ac_valid;\n";
    os << "  dataflow_gate #(.WIDTH(DATA_WIDTH)) u_body (\n";
    os << "    .clk(clk),\n";
    os << "    .rst_n(rst_n),\n";
    os << "    .bv_valid(in_value_0_valid),\n";
    os << "    .bv_ready(in_value_0_ready),\n";
    os << "    .bv_data(in_value[0]),\n";
    os << "    .bc_valid(in_value_1_valid),\n";
    os << "    .bc_ready(in_value_1_ready),\n";
    os << "    .bc_data(in_value[1][0]),\n";
    os << "    .av_valid(gate_av_valid),\n";
    os << "    .av_ready(body_ready),\n";
    os << "    .av_data(body_result[0]),\n";
    os << "    .ac_valid(gate_ac_valid),\n";
    os << "    .ac_ready(body_ready),\n";
    os << "    .ac_data(body_result[1][0])\n";
    os << "  );\n";
    os << "  generate\n";
    os << "    if (DATA_WIDTH > 1) begin : g_gate_cond_pad\n";
    os << "      assign body_result[1][DATA_WIDTH-1:1] = '0;\n";
    os << "    end\n";
    os << "  endgenerate\n";
    os << "  assign body_valid = gate_av_valid & gate_ac_valid;\n";
    return os.str();
  } else if (def.singleOp == "dataflow.stream") {
    unsigned tw = def.inputPorts.empty() ? 0 : getTagWidthBits(def.inputPorts[0]);
    unsigned tagCfgBits = (tw > 0) ? def.outputPorts.size() * tw : 0;
    os << "  localparam int STREAM_CFG_LSB = " << tagCfgBits << ";\n";
    os << "  logic stream_index_valid;\n";
    os << "  logic stream_cont_valid;\n";
    os << "  logic stream_error_valid;\n";
    os << "  logic [15:0] stream_error_code;\n";
    os << "  dataflow_stream #(.WIDTH(DATA_WIDTH), .STEP_OP("
       << def.streamStepOp << ")) u_body (\n";
    os << "    .clk(clk),\n";
    os << "    .rst_n(rst_n),\n";
    os << "    .start_valid(in_value_0_valid),\n";
    os << "    .start_ready(in_value_0_ready),\n";
    os << "    .start_data(in_value[0]),\n";
    os << "    .step_valid(in_value_1_valid),\n";
    os << "    .step_ready(in_value_1_ready),\n";
    os << "    .step_data(in_value[1]),\n";
    os << "    .bound_valid(in_value_2_valid),\n";
    os << "    .bound_ready(in_value_2_ready),\n";
    os << "    .bound_data(in_value[2]),\n";
    os << "    .index_valid(stream_index_valid),\n";
    os << "    .index_ready(body_ready),\n";
    os << "    .index_data(body_result[0]),\n";
    os << "    .cont_valid(stream_cont_valid),\n";
    os << "    .cont_ready(body_ready),\n";
    os << "    .cont_data(body_result[1][0]),\n";
    os << "    .cfg_cont_cond_sel(cfg_data[STREAM_CFG_LSB +: 5]),\n";
    os << "    .error_valid(stream_error_valid),\n";
    os << "    .error_code(stream_error_code)\n";
    os << "  );\n";
    os << "  generate\n";
    os << "    if (DATA_WIDTH > 1) begin : g_stream_cond_pad\n";
    os << "      assign body_result[1][DATA_WIDTH-1:1] = '0;\n";
    os << "    end\n";
    os << "  endgenerate\n";
    os << "  assign body_valid = stream_index_valid & stream_cont_valid;\n";
    return os.str();
  } else {
    // Standard ops: use WIDTH parameter
    os << "  " << svModule << " #(.WIDTH(DATA_WIDTH)) u_body (\n";

    // Map input ports with proper valid/ready handshake
    if (def.singleOp == "arith.select") {
      // select: condition (i1), a, b
      os << "    .condition_valid(in_value_0_valid),\n";
      os << "    .condition_ready(in_value_0_ready),\n";
      os << "    .condition_data(in_value[0][0]),\n";
      os << "    .a_valid(in_value_1_valid),\n";
      os << "    .a_ready(in_value_1_ready),\n";
      os << "    .a_data(in_value[1]),\n";
      os << "    .b_valid(in_value_2_valid),\n";
      os << "    .b_ready(in_value_2_ready),\n";
      os << "    .b_data(in_value[2]),\n";
    } else if (def.singleOp == "math.fma") {
      os << "    .a_valid(in_value_0_valid),\n";
      os << "    .a_ready(in_value_0_ready),\n";
      os << "    .a_data(in_value[0]),\n";
      os << "    .b_valid(in_value_1_valid),\n";
      os << "    .b_ready(in_value_1_ready),\n";
      os << "    .b_data(in_value[1]),\n";
      os << "    .c_valid(in_value_2_valid),\n";
      os << "    .c_ready(in_value_2_ready),\n";
      os << "    .c_data(in_value[2]),\n";
    } else if (numIn >= 2) {
      os << "    .a_valid(in_value_0_valid),\n";
      os << "    .a_ready(in_value_0_ready),\n";
      os << "    .a_data(in_value[0]),\n";
      os << "    .b_valid(in_value_1_valid),\n";
      os << "    .b_ready(in_value_1_ready),\n";
      os << "    .b_data(in_value[1]),\n";
    } else {
      os << "    .a_valid(in_value_0_valid),\n";
      os << "    .a_ready(in_value_0_ready),\n";
      os << "    .a_data(in_value[0]),\n";
    }

    os << "    .result_valid(u_body_result_valid),\n";
    os << "    .result_ready(body_ready),\n";
    os << "    .result_data(body_result[0])\n";
    os << "  );\n";
  }

  os << "  assign body_valid = u_body_result_valid;\n";
  return os.str();
}

/// Generate the temporal PE body SV text for a TemporalPEDef.
/// Instantiates per-FU customized fabric_pe modules and muxes results by fu_sel.
static std::string genTemporalPEBodySV(const TemporalPEDef &def,
                                        const std::vector<PEDef> &peDefs,
                                        const std::string &instName,
                                        unsigned tpeDW) {
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

  // Compute per-FU effective DATA_WIDTH: max of temporal PE interface width
  // and the FU's own required width (port widths + internal body widths).
  std::vector<unsigned> fuEffDW(numFU);
  for (unsigned f = 0; f < numFU; ++f) {
    const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
    unsigned dw = tpeDW;
    for (const auto &pt : fuDef.inputPorts)
      dw = std::max(dw, getDataWidthBits(pt));
    for (const auto &pt : fuDef.outputPorts)
      dw = std::max(dw, getDataWidthBits(pt));
    if (!fuDef.bodyMLIR.empty())
      dw = std::max(dw, computeBodyMLIRMaxWidth(fuDef.bodyMLIR));
    fuEffDW[f] = dw;
  }

  std::ostringstream os;

  // Extract fu_sel from the committed instruction (latched for multi-cycle FU)
  if (fuSelBits > 0) {
    os << "  logic [FU_SEL_BITS-1:0] fu_sel;\n";
    os << "  assign fu_sel = cfg_data[commit_insn * INSN_WIDTH + "
       << "INSN_FU_SEL_LSB +: FU_SEL_BITS];\n";
    os << "\n";
  }

  // Per-FU result wires and per-port output wires.
  // fu_result is temporal-PE-width (SAFE_DW) for the mux output.
  // Per-port wires match genFullPESV per-port interface (TAG_WIDTH=0).
  for (unsigned f = 0; f < numFU; ++f) {
    const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
    unsigned fuNumOut = fuDef.outputPorts.size();
    os << "  logic [NUM_OUTPUTS-1:0][SAFE_DW-1:0] fu" << f << "_result;\n";
    for (unsigned o = 0; o < fuNumOut; ++o) {
      // Port width matches genFullPESV output port: outDW[o] (tw=0)
      unsigned ow = getDataWidthBits(fuDef.outputPorts[o]);
      if (ow == 0) ow = 1;
      os << "  logic fu" << f << "_out" << o << "_valid;\n";
      os << "  logic [" << (ow - 1) << ":0] fu" << f << "_out" << o << "_data;\n";
    }
  }
  os << "\n";

  // Instantiate each FU as a customized fabric_pe module (TAG_WIDTH=0).
  // Each FU uses its configured latency from the PEDef.
  // Connections use per-port interface matching genFullPESV output.
  for (unsigned f = 0; f < numFU; ++f) {
    const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
    std::string fuModName = instName + "_fu" + std::to_string(f) + "_pe";
    unsigned fuNumIn = fuDef.inputPorts.size();
    unsigned fuNumOut = fuDef.outputPorts.size();
    unsigned fuLatency =
        static_cast<unsigned>(std::max<int16_t>(fuDef.latTyp, 0));

    unsigned fw = fuEffDW[f];

    os << "  // FU " << f << ": " << (fuDef.singleOp.empty() ? "passthrough" : fuDef.singleOp)
       << " (latency=" << fuLatency << ", DATA_WIDTH=" << fw << ")\n";
    os << "  " << fuModName << " u_fu" << f << " (\n";
    os << "    .clk(clk),\n";
    os << "    .rst_n(rst_n),\n";
    // Per-port input connections: FU PE port width = getDataWidthBits(port)
    // fu_operands[i] is tpeDW bits wide; truncate or zero-pad to match FU port
    for (unsigned i = 0; i < fuNumIn; ++i) {
      unsigned portW = getDataWidthBits(fuDef.inputPorts[i]);
      if (portW == 0) portW = 1;
      os << "    .in" << i << "_valid(fu_launch),\n";
      os << "    .in" << i << "_ready(),\n";
      os << "    .in" << i << "_data(";
      if (portW < tpeDW) {
        os << "fu_operands[" << i << "][" << (portW - 1) << ":0]";
      } else if (portW > tpeDW) {
        os << "{" << (portW - tpeDW) << "'b0, fu_operands[" << i << "]}";
      } else {
        os << "fu_operands[" << i << "]";
      }
      os << "),\n";
    }
    // Per-port output connections
    for (unsigned o = 0; o < fuNumOut; ++o) {
      os << "    .out" << o << "_valid(fu" << f << "_out" << o << "_valid),\n";
      os << "    .out" << o << "_ready(all_out_ready),\n";
      os << "    .out" << o << "_data(fu" << f << "_out" << o << "_data),\n";
    }
    os << "    .cfg_data('0)\n";
    os << "  );\n";
    // Extract data portion from FU output (TAG_WIDTH=0 so out_data is just data).
    for (unsigned o = 0; o < fuNumOut; ++o) {
      unsigned ow = getDataWidthBits(fuDef.outputPorts[o]);
      if (ow == 0) ow = 1;
      if (ow >= tpeDW) {
        os << "  assign fu" << f << "_result[" << o << "] = fu" << f
           << "_out" << o << "_data[SAFE_DW-1:0];\n";
      } else {
        os << "  assign fu" << f << "_result[" << o << "] = {"
           << (tpeDW - ow) << "'b0, fu" << f << "_out" << o << "_data};\n";
      }
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
    os << "  assign body_valid = fu0_out0_valid;\n";
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
      os << "        body_valid = fu" << f << "_out0_valid;\n";
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

//===----------------------------------------------------------------------===//
// Full module generation
//===----------------------------------------------------------------------===//

/// Generate a fully code-generated temporal PE module with per-port data widths.
/// External ports get per-position max widths across FUs; internal logic keeps
/// uniform DATA_WIDTH with boundary adapters for width conversion.
std::string genFullTemporalPESV(const std::string &templateDir,
                                        const TemporalPEDef &def,
                                        const std::vector<PEDef> &peDefs,
                                        const std::string &instName) {
  // Read the template for core logic extraction
  llvm::SmallString<256> tmplPath(templateDir);
  llvm::sys::path::append(tmplPath, "Fabric", "fabric_temporal_pe.sv");
  std::string tmpl = readFile(tmplPath.str().str());

  // Basic parameters
  unsigned numFU = def.fuPEDefIndices.size();
  unsigned numIn = 1, numOut = 1;
  if (!def.fuPEDefIndices.empty()) {
    numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
    numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
  }

  unsigned interfaceDW = getDataWidthBits(def.interfaceType);
  unsigned tagWidth = getTagWidthBits(def.interfaceType);
  if (interfaceDW == 0) interfaceDW = 1;

  // Compute per-port data widths: max across all FUs at each position
  std::vector<unsigned> inDW(numIn, interfaceDW);
  std::vector<unsigned> outDW(numOut, interfaceDW);
  if (numFU > 0) {
    for (unsigned i = 0; i < numIn; ++i) inDW[i] = 0;
    for (unsigned o = 0; o < numOut; ++o) outDW[o] = 0;
    for (unsigned f = 0; f < numFU; ++f) {
      const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
      for (unsigned i = 0; i < numIn && i < fuDef.inputPorts.size(); ++i)
        inDW[i] = std::max(inDW[i], getDataWidthBits(fuDef.inputPorts[i]));
      for (unsigned o = 0; o < numOut && o < fuDef.outputPorts.size(); ++o)
        outDW[o] = std::max(outDW[o], getDataWidthBits(fuDef.outputPorts[o]));
    }
    for (unsigned i = 0; i < numIn; ++i)
      if (inDW[i] == 0) inDW[i] = 1;
    for (unsigned o = 0; o < numOut; ++o)
      if (outDW[o] == 0) outDW[o] = 1;
  }

  // DATA_WIDTH must cover all FU port widths to prevent data truncation
  // in internal packed arrays (operand buffers, register FIFOs, etc.)
  unsigned dataWidth = interfaceDW;
  for (unsigned w : inDW) dataWidth = std::max(dataWidth, w);
  for (unsigned w : outDW) dataWidth = std::max(dataWidth, w);
  if (dataWidth == 0) dataWidth = 1;
  unsigned payloadWidth = dataWidth + tagWidth;

  // Instruction memory layout parameters (log2Ceil mirrors $clog2)
  auto log2Ceil = [](unsigned v) -> unsigned {
    if (v <= 1) return 0;
    unsigned bits = 0;
    v--;
    while (v > 0) { bits++; v >>= 1; }
    return bits;
  };
  unsigned numRegisters = def.numRegisters;
  unsigned numInstructions = def.numInstructions;
  unsigned regBits = (numRegisters > 0)
      ? (1 + log2Ceil(std::max(numRegisters, 2u))) : 0;
  unsigned fuSelBits = (numFU > 1) ? log2Ceil(numFU) : 0;
  unsigned resBits = regBits;
  unsigned resultWidth = resBits + tagWidth;
  unsigned insnWidth = 1 + tagWidth + fuSelBits
      + numIn * regBits + numOut * resultWidth;
  unsigned configWidth = numInstructions * insnWidth;
  unsigned cfgPortWidth = configWidth > 0 ? configWidth : 1;

  std::ostringstream os;

  // ---- File header ----
  os << "`include \"fabric_common.svh\"\n\n";

  // ---- Module declaration with per-port signals ----
  os << "module " << instName << "_temporal_pe (\n";
  os << "    input  logic                       clk,\n";
  os << "    input  logic                       rst_n,\n";
  for (unsigned i = 0; i < numIn; ++i) {
    unsigned pw = inDW[i] + tagWidth;
    if (pw == 0) pw = 1;
    os << "    input  logic                       in" << i << "_valid,\n";
    os << "    output logic                       in" << i << "_ready,\n";
    os << "    input  logic "
       << (pw > 1 ? "[" + std::to_string(pw - 1) + ":0] " : "")
       << "in" << i << "_data,\n";
  }
  for (unsigned o = 0; o < numOut; ++o) {
    unsigned pw = outDW[o] + tagWidth;
    if (pw == 0) pw = 1;
    os << "    output logic                       out" << o << "_valid,\n";
    os << "    input  logic                       out" << o << "_ready,\n";
    os << "    output logic "
       << (pw > 1 ? "[" + std::to_string(pw - 1) + ":0] " : "")
       << "out" << o << "_data,\n";
  }
  os << "    input  logic "
     << (cfgPortWidth > 1 ? "[" + std::to_string(cfgPortWidth - 1) + ":0] "
                          : "")
     << "cfg_data,\n";
  os << "    output logic                       error_valid,\n";
  os << "    output logic [15:0]                error_code\n";
  os << ");\n\n";

  // ---- Localparams: base params hardcoded, derived use SV expressions ----
  os << "  localparam int NUM_INPUTS             = " << numIn << ";\n";
  os << "  localparam int NUM_OUTPUTS            = " << numOut << ";\n";
  os << "  localparam int DATA_WIDTH             = " << dataWidth << ";\n";
  os << "  localparam int TAG_WIDTH              = " << tagWidth << ";\n";
  os << "  localparam int NUM_FU_TYPES           = " << numFU << ";\n";
  os << "  localparam int NUM_REGISTERS          = " << numRegisters << ";\n";
  os << "  localparam int NUM_INSTRUCTIONS       = " << numInstructions << ";\n";
  os << "  localparam int REG_FIFO_DEPTH         = " << def.regFifoDepth << ";\n";
  os << "  localparam int SHARED_OPERAND_BUFFER  = "
     << (def.sharedOperandBuffer ? 1 : 0) << ";\n";
  os << "  localparam int OPERAND_BUFFER_SIZE    = "
     << def.shareBufferSize << ";\n";
  os << "  localparam int PAYLOAD_WIDTH    = DATA_WIDTH + TAG_WIDTH;\n";
  os << "  localparam int SAFE_PW          = (PAYLOAD_WIDTH > 0) "
     << "? PAYLOAD_WIDTH : 1;\n";
  os << "  localparam int SAFE_DW          = (DATA_WIDTH > 0) "
     << "? DATA_WIDTH : 1;\n";
  os << "  localparam int REG_BITS         = (NUM_REGISTERS > 0) ? "
     << "(1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2)) : 0;\n";
  os << "  localparam int FU_SEL_BITS      = (NUM_FU_TYPES > 1) ? "
     << "$clog2(NUM_FU_TYPES) : 0;\n";
  os << "  localparam int RES_BITS         = (NUM_REGISTERS > 0) ? "
     << "(1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2)) : 0;\n";
  os << "  localparam int RESULT_WIDTH     = RES_BITS + TAG_WIDTH;\n";
  os << "  localparam int INSN_WIDTH       = 1 + TAG_WIDTH + FU_SEL_BITS "
     << "+ NUM_INPUTS * REG_BITS + NUM_OUTPUTS * RESULT_WIDTH;\n";
  os << "  localparam int CONFIG_WIDTH     = NUM_INSTRUCTIONS * INSN_WIDTH;\n";
  os << "\n";

  // ---- Internal packed-array signals ----
  os << "  logic [NUM_INPUTS-1:0]                       in_valid;\n";
  os << "  logic [NUM_INPUTS-1:0]                       in_ready;\n";
  os << "  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]          in_data;\n";
  os << "  logic [NUM_OUTPUTS-1:0]                      out_valid;\n";
  os << "  logic [NUM_OUTPUTS-1:0]                      out_ready;\n";
  os << "  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0]         out_data;\n";
  os << "\n";

  // ---- Input boundary adaptation: per-port -> internal packed ----
  os << "  assign in_valid = {";
  for (int i = numIn - 1; i >= 0; --i) {
    os << "in" << i << "_valid";
    if (i > 0) os << ", ";
  }
  os << "};\n";
  for (unsigned i = 0; i < numIn; ++i) {
    if (inDW[i] == dataWidth) {
      os << "  assign in_data[" << i << "] = in" << i << "_data;\n";
    } else {
      unsigned inPW = inDW[i] + tagWidth;
      os << "  assign in_data[" << i << "] = {in" << i
         << "_data[" << (inPW - 1) << ":" << inDW[i] << "], "
         << (dataWidth - inDW[i]) << "'b0, in" << i
         << "_data[" << (inDW[i] - 1) << ":0]};\n";
    }
  }
  for (unsigned i = 0; i < numIn; ++i) {
    os << "  assign in" << i << "_ready = in_ready[" << i << "];\n";
  }
  os << "\n";

  // ---- Extract core logic from template using section markers ----
  const std::string endParamCheck = "  // ===== END PARAM CHECK =====";
  const std::string beginPEBody = "  // ===== BEGIN PE BODY =====";
  const std::string endPEBody = "  // ===== END PE BODY =====";

  auto posEndPC = tmpl.find(endParamCheck);
  auto posBeginPB = tmpl.find(beginPEBody);
  auto posEndPB = tmpl.find(endPEBody);
  auto posEndModule = tmpl.rfind("endmodule");

  if (posEndPC == std::string::npos || posBeginPB == std::string::npos ||
      posEndPB == std::string::npos || posEndModule == std::string::npos) {
    llvm::errs() << "error: temporal PE template missing required markers\n";
    std::exit(1);
  }

  // Core logic section 1: instruction unpacking, input extraction,
  // operand buffer, matching, fire logic, FU launch, fu_operands
  auto afterEndPC = tmpl.find('\n', posEndPC);
  os << tmpl.substr(afterEndPC + 1, posBeginPB - afterEndPC - 1);

  // PE body (generated FU instantiations)
  os << "  // ===== BEGIN PE BODY =====\n";
  os << genTemporalPEBodySV(def, peDefs, instName, dataWidth);
  os << "  // ===== END PE BODY =====\n";

  // Core logic section 2: output assembly, in_ready, register FIFOs,
  // operand buffer update, operand merge, error detection
  auto afterEndPB = tmpl.find('\n', posEndPB);
  os << tmpl.substr(afterEndPB + 1, posEndModule - afterEndPB - 1);

  // ---- Output boundary adaptation: internal packed -> per-port ----
  for (unsigned o = 0; o < numOut; ++o) {
    os << "  assign out" << o << "_valid = out_valid[" << o << "];\n";
    if (outDW[o] == dataWidth) {
      os << "  assign out" << o << "_data = out_data[" << o << "];\n";
    } else {
      os << "  assign out" << o << "_data = {out_data[" << o << "]["
         << (payloadWidth - 1) << ":" << dataWidth << "], out_data[" << o
         << "][" << (outDW[o] - 1) << ":0]};\n";
    }
  }
  os << "  assign out_ready = {";
  for (int o = numOut - 1; o >= 0; --o) {
    os << "out" << o << "_ready";
    if (o > 0) os << ", ";
  }
  os << "};\n\n";

  os << "endmodule\n";
  return os.str();
}

/// Generate a fully code-generated PE module with per-port data widths.
/// Each input and output port gets its own exact semantic data width.
std::string genFullPESV(const PEDef &def) {
  unsigned numIn = def.inputPorts.size();
  unsigned numOut = def.outputPorts.size();
  unsigned tw = numIn > 0 ? getTagWidthBits(def.inputPorts[0]) : 0;
  unsigned latency =
      static_cast<unsigned>(std::max<int16_t>(def.latTyp, 0));
  unsigned interval =
      static_cast<unsigned>(std::max<int16_t>(def.intTyp, 1));

  // Per-port data widths
  std::vector<unsigned> inDW(numIn), outDW(numOut);
  for (unsigned i = 0; i < numIn; ++i) {
    inDW[i] = getDataWidthBits(def.inputPorts[i]);
    if (inDW[i] == 0) inDW[i] = 1;
  }
  for (unsigned i = 0; i < numOut; ++i) {
    outDW[i] = getDataWidthBits(def.outputPorts[i]);
    if (outDW[i] == 0) outDW[i] = 1;
  }

  // DATA_WIDTH for body computation (max across all ports + body intermediates)
  unsigned dw = 0;
  for (unsigned w : inDW) dw = std::max(dw, w);
  for (unsigned w : outDW) dw = std::max(dw, w);
  if (!def.bodyMLIR.empty())
    dw = std::max(dw, computeBodyMLIRMaxWidth(def.bodyMLIR));
  if (dw == 0) dw = 32;

  std::ostringstream os;
  os << "`include \"fabric_common.svh\"\n\n";
  // Module declaration with per-port signals
  os << "module " << def.name << "_pe (\n";
  os << "    input  logic clk,\n";
  os << "    input  logic rst_n,\n";
  for (unsigned i = 0; i < numIn; ++i) {
    unsigned pw = inDW[i] + tw;
    os << "    input  logic in" << i << "_valid,\n";
    os << "    output logic in" << i << "_ready,\n";
    os << "    input  logic [" << (pw - 1) << ":0] in" << i << "_data,\n";
  }
  for (unsigned i = 0; i < numOut; ++i) {
    unsigned pw = outDW[i] + tw;
    os << "    output logic out" << i << "_valid,\n";
    os << "    input  logic out" << i << "_ready,\n";
    os << "    output logic [" << (pw - 1) << ":0] out" << i << "_data,\n";
  }
  // Config port for output tags and optional dataflow.stream condition selector.
  unsigned tagCfgBits = (tw > 0) ? numOut * tw : 0;
  unsigned streamCfgBits = (def.singleOp == "dataflow.stream") ? 5 : 0;
  unsigned totalCfgBits = tagCfgBits + streamCfgBits;
  if (totalCfgBits > 0) {
    os << "    input  logic [" << (totalCfgBits - 1) << ":0] cfg_data\n";
  } else {
    os << "    input  logic [0:0] cfg_data\n";
  }
  os << ");\n\n";

  // Localparams used by body generation
  os << "  localparam int DATA_WIDTH = " << dw << ";\n";
  os << "  localparam int SAFE_DW = " << dw << ";\n";
  os << "  localparam int NUM_INPUTS = " << numIn << ";\n";
  os << "  localparam int NUM_OUTPUTS = " << numOut << ";\n\n";

  // Tag stripping: extract value from each input
  for (unsigned i = 0; i < numIn; ++i) {
    os << "  logic [" << (dw - 1) << ":0] in_value_" << i << ";\n";
    if (tw > 0)
      os << "  assign in_value_" << i << " = {"
         << (dw - inDW[i]) << "'b0, in" << i << "_data["
         << (inDW[i] - 1) << ":0]};\n";
    else if (inDW[i] < dw)
      os << "  assign in_value_" << i << " = {"
         << (dw - inDW[i]) << "'b0, in" << i << "_data["
         << (inDW[i] - 1) << ":0]};\n";
    else
      os << "  assign in_value_" << i << " = in" << i << "_data;\n";
  }
  // Also provide in_value array for body compatibility
  os << "  logic [" << (numIn - 1) << ":0][" << (dw - 1) << ":0] in_value;\n";
  for (unsigned i = 0; i < numIn; ++i)
    os << "  assign in_value[" << i << "] = in_value_" << i << ";\n";
  os << "\n";

  // Body input handshake wires: valid driven by skeleton, ready driven by body
  for (unsigned i = 0; i < numIn; ++i) {
    os << "  logic in_value_" << i << "_valid;\n";
    os << "  logic in_value_" << i << "_ready;\n";
  }
  os << "\n";

  // All inputs valid (used by II counter)
  os << "  logic all_in_valid;\n";
  os << "  assign all_in_valid = ";
  for (unsigned i = 0; i < numIn; ++i) {
    if (i > 0) os << " & ";
    os << "in" << i << "_valid";
  }
  os << ";\n\n";

  // Pipeline ready
  os << "  logic pipeline_ready;\n";

  // Initiation interval gating
  os << "  logic ii_allow;\n";
  os << "  logic fire;\n";
  os << "  logic body_valid;\n";
  os << "  logic body_ready;\n\n";
  if (interval > 1) {
    unsigned iiCtrW = 0;
    unsigned v = interval - 1;
    while (v > 0) { iiCtrW++; v >>= 1; }
    os << "  logic [" << (iiCtrW - 1) << ":0] ii_ctr;\n";
    os << "  assign fire = body_valid && pipeline_ready;\n";
    os << "  always_ff @(posedge clk or negedge rst_n) begin : ii_counter\n";
    os << "    if (!rst_n) begin : reset\n";
    os << "      ii_ctr <= '0;\n";
    os << "    end else if (fire) begin : reload\n";
    os << "      ii_ctr <= " << iiCtrW << "'d" << (interval - 1) << ";\n";
    os << "    end else if (ii_ctr != '0 && pipeline_ready) begin : tick\n";
    os << "      ii_ctr <= ii_ctr - " << iiCtrW << "'d1;\n";
    os << "    end\n";
    os << "  end\n";
    os << "  assign ii_allow = (ii_ctr == '0);\n\n";
  } else {
    os << "  assign ii_allow = 1'b1;\n";
    os << "  assign fire = body_valid && pipeline_ready;\n\n";
  }

  // Body input valid: gated by ii_allow
  for (unsigned i = 0; i < numIn; ++i)
    os << "  assign in_value_" << i << "_valid = in" << i
       << "_valid && ii_allow;\n";
  os << "\n";

  // Input ready: driven by body's internal v/r graph, gated by ii_allow
  for (unsigned i = 0; i < numIn; ++i)
    os << "  assign in" << i << "_ready = in_value_" << i
       << "_ready && ii_allow;\n";
  os << "\n";

  // Body result array (body_valid/body_ready declared above with fire)
  os << "  logic [" << (numOut - 1) << ":0][" << (dw - 1)
     << ":0] body_result;\n\n";

  // Body generation
  os << genPEBodySV(def);
  os << "\n";

  // Latency pipeline
  os << "  logic all_out_ready;\n";
  os << "  assign all_out_ready = ";
  for (unsigned i = 0; i < numOut; ++i) {
    if (i > 0) os << " & ";
    os << "out" << i << "_ready";
  }
  os << ";\n";
  os << "  assign pipeline_ready = all_out_ready;\n";
  os << "  assign body_ready = pipeline_ready;\n\n";

  if (latency > 0) {
    os << "  logic [" << (latency - 1) << ":0] sr_valid;\n";
    os << "  logic [" << (latency - 1) << ":0][" << (numOut - 1) << ":0]["
       << (dw - 1) << ":0] sr_data;\n\n";
    os << "  always_ff @(posedge clk or negedge rst_n) begin : shift_reg\n";
    os << "    if (!rst_n) begin : reset\n";
    os << "      integer iter_var0;\n";
    os << "      for (iter_var0 = 0; iter_var0 < " << latency
       << "; iter_var0 = iter_var0 + 1) begin : clr\n";
    os << "        sr_valid[iter_var0] <= 1'b0;\n";
    os << "      end\n";
    os << "    end else if (all_out_ready) begin : advance\n";
    os << "      integer iter_var0;\n";
    os << "      sr_valid[0] <= body_valid;\n";
    os << "      sr_data[0]  <= body_result;\n";
    os << "      for (iter_var0 = 1; iter_var0 < " << latency
       << "; iter_var0 = iter_var0 + 1) begin : shift\n";
    os << "        sr_valid[iter_var0] <= sr_valid[iter_var0 - 1];\n";
    os << "        sr_data[iter_var0]  <= sr_data[iter_var0 - 1];\n";
    os << "      end\n";
    os << "    end\n";
    os << "  end\n\n";

    os << "  logic last_valid;\n";
    os << "  logic [" << (numOut - 1) << ":0][" << (dw - 1)
       << ":0] last_data;\n";
    os << "  assign last_valid = sr_valid[" << (latency - 1) << "];\n";
    os << "  assign last_data  = sr_data[" << (latency - 1) << "];\n\n";

    for (unsigned o = 0; o < numOut; ++o) {
      if (tw > 0) {
        os << "  assign out" << o << "_data = {cfg_data["
           << (o * tw + tw - 1) << ":" << (o * tw) << "], last_data["
           << o << "][" << (outDW[o] - 1) << ":0]};\n";
      } else {
        os << "  assign out" << o << "_data = last_data[" << o << "]["
           << (outDW[o] - 1) << ":0];\n";
      }
      os << "  assign out" << o << "_valid = last_valid;\n";
    }
  } else {
    // Zero latency: combinational passthrough
    for (unsigned o = 0; o < numOut; ++o) {
      if (tw > 0) {
        os << "  assign out" << o << "_data = {cfg_data["
           << (o * tw + tw - 1) << ":" << (o * tw) << "], body_result["
           << o << "][" << (outDW[o] - 1) << ":0]};\n";
      } else {
        os << "  assign out" << o << "_data = body_result[" << o << "]["
           << (outDW[o] - 1) << ":0];\n";
      }
      os << "  assign out" << o << "_valid = body_valid;\n";
    }
  }

  os << "\nendmodule\n";
  return os.str();
}

/// Generate a complete PE module using genFullPESV().
/// Returns the complete customized module text.
std::string fillPETemplate(const std::string &templateDir,
                                  const PEDef &def) {
  // Use full code generation with per-port widths
  return genFullPESV(def);
}

//===----------------------------------------------------------------------===//
// Generate constant PE instance parameters
//===----------------------------------------------------------------------===//

std::string genConstantPEParams(const ConstantPEDef &def) {
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

unsigned ceilLog2(unsigned v) {
  if (v <= 1) return 1;
  unsigned r = 0;
  unsigned vv = v - 1;
  while (vv > 0) { ++r; vv >>= 1; }
  return r;
}

std::string genLoadPEParams(const LoadPEDef &def) {
  unsigned ew = getDataWidthBits(def.dataType);
  if (ew == 0)
    ew = 1;
  std::ostringstream os;
  os << "    .ELEM_WIDTH(" << ew << "),\n";
  os << "    .ADDR_WIDTH(" << DEFAULT_ADDR_WIDTH << "),\n";
  os << "    .TAG_WIDTH(" << def.tagWidth << "),\n";
  os << "    .HW_TYPE(" << (def.hwType == HardwareType::TagTransparent ? 1 : 0)
     << "),\n";
  os << "    .QUEUE_DEPTH(" << def.queueDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate store PE instance parameters
//===----------------------------------------------------------------------===//

std::string genStorePEParams(const StorePEDef &def) {
  unsigned ew = getDataWidthBits(def.dataType);
  if (ew == 0)
    ew = 1;
  std::ostringstream os;
  os << "    .ELEM_WIDTH(" << ew << "),\n";
  os << "    .ADDR_WIDTH(" << DEFAULT_ADDR_WIDTH << "),\n";
  os << "    .TAG_WIDTH(" << def.tagWidth << "),\n";
  os << "    .HW_TYPE(" << (def.hwType == HardwareType::TagTransparent ? 1 : 0)
     << "),\n";
  os << "    .QUEUE_DEPTH(" << def.queueDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate temporal switch instance parameters
//===----------------------------------------------------------------------===//

std::string genTemporalSwitchParams(const TemporalSwitchDef &def) {
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

//===----------------------------------------------------------------------===//
// Generate memory instance parameters
//===----------------------------------------------------------------------===//

std::string genMemoryParams(const MemoryDef &def) {
  unsigned memDepth = def.shape.isDynamic() ? 64 : def.shape.getSize();
  if (memDepth == 0)
    memDepth = 64;
  unsigned ew = getDataWidthBits(def.shape.getElemType());
  if (ew == 0)
    ew = 1;
  unsigned tw = 0;
  if (def.ldCount > 1 || def.stCount > 1) {
    unsigned maxCount = std::max(def.ldCount, def.stCount);
    unsigned tagBits = 1;
    while ((1u << tagBits) < maxCount)
      ++tagBits;
    tw = tagBits;
  }
  std::ostringstream os;
  os << "    .ADDR_WIDTH(" << DEFAULT_ADDR_WIDTH << "),\n";
  os << "    .ELEM_WIDTH(" << ew << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .LD_COUNT(" << def.ldCount << "),\n";
  os << "    .ST_COUNT(" << def.stCount << "),\n";
  os << "    .LSQ_DEPTH(" << def.lsqDepth << "),\n";
  os << "    .IS_PRIVATE(" << (def.isPrivate ? 1 : 0) << "),\n";
  os << "    .MEM_DEPTH(" << memDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate external memory instance parameters
//===----------------------------------------------------------------------===//

std::string genExtMemoryParams(const ExtMemoryDef &def) {
  unsigned ew = getDataWidthBits(def.shape.getElemType());
  if (ew == 0)
    ew = 1;
  unsigned tw = 0;
  if (def.ldCount > 1 || def.stCount > 1) {
    unsigned maxCount = std::max(def.ldCount, def.stCount);
    unsigned tagBits = 1;
    while ((1u << tagBits) < maxCount)
      ++tagBits;
    tw = tagBits;
  }
  std::ostringstream os;
  os << "    .ADDR_WIDTH(" << DEFAULT_ADDR_WIDTH << "),\n";
  os << "    .ELEM_WIDTH(" << ew << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .LD_COUNT(" << def.ldCount << "),\n";
  os << "    .ST_COUNT(" << def.stCount << "),\n";
  os << "    .LSQ_DEPTH(" << def.lsqDepth << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Compute hardware payload width for memory modules
//===----------------------------------------------------------------------===//
// Returns DATA_WIDTH + TAG_WIDTH matching the packed-array port element width
// used in fabric_memory.sv / fabric_extmemory.sv.

// getMemoryPayloadWidth / getExtMemoryPayloadWidth removed:
// Memory/ExtMemory now use per-port semantic widths from
// getInstanceInputType/getInstanceOutputType, not a uniform payload width.

//===----------------------------------------------------------------------===//
// Generate switch instance parameters
//===----------------------------------------------------------------------===//

std::string genSwitchParams(const SwitchDef &def) {
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

std::string genAddTagParams(const AddTagDef &def) {
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

std::string genDelTagParams(const DelTagDef &def) {
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

std::string genMapTagParams(const MapTagDef &def) {
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

std::string genFifoParams(const FifoDef &def) {
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
// ADGBuilder::Impl::generateSV helpers
//===----------------------------------------------------------------------===//

bool hasSVTemplate(ModuleKind kind) {
  return kind == ModuleKind::Switch || kind == ModuleKind::Fifo ||
         kind == ModuleKind::AddTag || kind == ModuleKind::DelTag ||
         kind == ModuleKind::MapTag || kind == ModuleKind::PE ||
         kind == ModuleKind::ConstantPE || kind == ModuleKind::LoadPE ||
         kind == ModuleKind::StorePE || kind == ModuleKind::TemporalSwitch ||
         kind == ModuleKind::TemporalPE || kind == ModuleKind::Memory ||
         kind == ModuleKind::ExtMemory;
}

/// Returns true for module kinds that expose error_valid/error_code ports.
bool hasErrorOutput(ModuleKind kind) {
  return kind == ModuleKind::Switch || kind == ModuleKind::TemporalSwitch ||
         kind == ModuleKind::TemporalPE || kind == ModuleKind::MapTag ||
         kind == ModuleKind::Memory || kind == ModuleKind::ExtMemory;
}

} // namespace adg
} // namespace loom
