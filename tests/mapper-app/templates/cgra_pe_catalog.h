//===-- cgra_pe_catalog.h - Shared PE catalog for CGRA generators --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Shared header that registers the complete PE catalog using the ADGBuilder
// C++ API and provides a LatticeConnector for distributing PE connections
// across lattice mesh switches via round-robin port allocation.
//
//===----------------------------------------------------------------------===//

#ifndef CGRA_PE_CATALOG_H
#define CGRA_PE_CATALOG_H

#include <loom/adg.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace loom::adg;

//===----------------------------------------------------------------------===//
// Type plane helpers
//===----------------------------------------------------------------------===//

enum TypePlane { TP_I32, TP_F32, TP_INDEX, TP_I1, TP_NONE, TP_I64, TP_I16,
                 TP_COUNT };

inline Type typePlaneToType(TypePlane tp) {
  switch (tp) {
  case TP_I32:   return Type::i32();
  case TP_F32:   return Type::f32();
  case TP_INDEX: return Type::index();
  case TP_I1:    return Type::i1();
  case TP_NONE:  return Type::none();
  case TP_I64:   return Type::i64();
  case TP_I16:   return Type::i16();
  default:       return Type::i32();
  }
}

inline TypePlane typeNameToPlane(const std::string &name) {
  if (name == "i32")   return TP_I32;
  if (name == "f32")   return TP_F32;
  if (name == "index") return TP_INDEX;
  if (name == "i1")    return TP_I1;
  if (name == "none")  return TP_NONE;
  if (name == "i64")   return TP_I64;
  if (name == "i16")   return TP_I16;
  assert(false && "unknown type name");
  return TP_I32;
}

//===----------------------------------------------------------------------===//
// Width plane helpers (width-merged routing planes)
//===----------------------------------------------------------------------===//

enum WidthPlane {
  WP_1, WP_NONE, WP_8, WP_16, WP_32, WP_64, WP_INDEX, WP_COUNT
};

inline Type widthPlaneToType(WidthPlane wp) {
  switch (wp) {
  case WP_1:     return Type::bits(1);
  case WP_NONE:  return Type::none();
  case WP_8:     return Type::bits(8);
  case WP_16:    return Type::bits(16);
  case WP_32:    return Type::bits(32);
  case WP_64:    return Type::bits(64);
  case WP_INDEX: return Type::index();
  default:       return Type::bits(32);
  }
}

inline WidthPlane typeNameToWidthPlane(const std::string &name) {
  if (name == "i1")    return WP_1;
  if (name == "none")  return WP_NONE;
  if (name == "i8")    return WP_8;
  if (name == "i16")   return WP_16;
  if (name == "i32")   return WP_32;
  if (name == "f32")   return WP_32;  // width merge: 32-bit
  if (name == "i64")   return WP_64;
  if (name == "f64")   return WP_64;  // width merge: 64-bit
  if (name == "f16")   return WP_16;  // width merge: 16-bit
  if (name == "bf16")  return WP_16;
  if (name == "index") return WP_INDEX;
  assert(false && "unknown type name for width plane");
  return WP_32;
}

/// Compute near-square grid dimensions for a given PE count.
/// Returns {rows, cols} where rows*cols >= peCount.
inline std::pair<int, int> computePEGridDims(int peCount) {
  if (peCount <= 0) return {0, 0};
  int cols = static_cast<int>(
      std::ceil(std::sqrt(static_cast<double>(peCount))));
  int rows = (peCount + cols - 1) / cols;
  return {rows, cols};
}

/// Generate anti-diagonal boundary switch order for input ports
/// (top + left edge of the switch grid).
inline std::vector<std::pair<int, int>>
inputBoundaryOrder(int swRows, int swCols) {
  std::vector<std::pair<int, int>> order;
  int maxDiag = std::max(swRows, swCols);
  for (int d = 0; d < maxDiag; d++) {
    if (d < swCols) order.push_back({0, d});
    if (d > 0 && d < swRows) order.push_back({d, 0});
  }
  // Expand inward if needed
  for (int d = 2; d < swRows + swCols; d++) {
    for (int r = 1; r < swRows; r++) {
      int c = d - r;
      if (c >= 1 && c < swCols)
        order.push_back({r, c});
    }
  }
  return order;
}

/// Generate anti-diagonal boundary switch order for output ports
/// (bottom + right edge of the switch grid).
inline std::vector<std::pair<int, int>>
outputBoundaryOrder(int swRows, int swCols) {
  std::vector<std::pair<int, int>> order;
  int maxDiag = std::max(swRows, swCols);
  for (int d = 0; d < maxDiag; d++) {
    int c = swCols - 1 - d;
    if (c >= 0) order.push_back({swRows - 1, c});
    int r = swRows - 1 - d;
    if (d > 0 && r >= 0) order.push_back({r, swCols - 1});
  }
  // Expand inward if needed
  for (int d = 2; d < swRows + swCols; d++) {
    for (int r = swRows - 2; r >= 0; r--) {
      int c = swCols - 1 - (d - (swRows - 1 - r));
      if (c >= 0 && c < swCols - 1)
        order.push_back({r, c});
    }
  }
  return order;
}

//===----------------------------------------------------------------------===//
// PE definition metadata
//===----------------------------------------------------------------------===//

struct PEDef {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::string bodyMLIR;
  int16_t latencyHW;
};

/// Variant handle that holds either a PEHandle or a generic ModuleHandle.
/// We use PEHandle for all body-MLIR PEs since they are all created via
/// newPE().
struct PETemplateHandle {
  PEHandle handle;
};

struct PECatalogEntry {
  PEDef def;
  PETemplateHandle tmpl;
};

//===----------------------------------------------------------------------===//
// LatticeConnector - round-robin port allocation across lattice switches
//===----------------------------------------------------------------------===//

/// Distributes PE and module-I/O connections across switches in a lattice mesh.
/// Ports 0-3 are reserved for inter-switch connections; ports 4..(portCount-1)
/// are available for PE and I/O connections.
struct LatticeConnector {
  ADGBuilder &builder;
  LatticeMeshResult &lattice;
  int swRows, swCols;
  int portsPerSw;
  std::vector<int> nextInPort;   // flat [r * swCols + c], start at 4
  std::vector<int> nextOutPort;  // flat [r * swCols + c], start at 4
  int rrIn;   // round-robin cursor for input allocation
  int rrOut;  // round-robin cursor for output allocation

  LatticeConnector(ADGBuilder &b, LatticeMeshResult &lat, int portCount)
      : builder(b), lattice(lat),
        swRows(lat.peRows + 1), swCols(lat.peCols + 1),
        portsPerSw(portCount),
        nextInPort(swRows * swCols, 4),
        nextOutPort(swRows * swCols, 4),
        rrIn(0), rrOut(0) {}

  int totalSwitches() const { return swRows * swCols; }

  /// Find the next switch with a free input port, starting from rrIn.
  int findFreeInSwitch() {
    int total = totalSwitches();
    for (int i = 0; i < total; ++i) {
      int idx = (rrIn + i) % total;
      if (nextInPort[idx] < portsPerSw)
        return idx;
    }
    assert(false && "no free input port on any switch");
    return -1;
  }

  /// Find the next switch with a free output port, starting from rrOut.
  int findFreeOutSwitch() {
    int total = totalSwitches();
    for (int i = 0; i < total; ++i) {
      int idx = (rrOut + i) % total;
      if (nextOutPort[idx] < portsPerSw)
        return idx;
    }
    assert(false && "no free output port on any switch");
    return -1;
  }

  InstanceHandle swAt(int flatIdx) {
    int r = flatIdx / swCols;
    int c = flatIdx % swCols;
    return lattice.swGrid[r][c];
  }

  /// Connect a PE input port: allocate a switch output port, wire switch->PE.
  void feedPEInput(InstanceHandle pe, unsigned pePort) {
    int idx = findFreeOutSwitch();
    int port = nextOutPort[idx]++;
    builder.connectPorts(swAt(idx), port, pe, pePort);
    rrOut = (idx + 1) % totalSwitches();
  }

  /// Connect a PE output port: allocate a switch input port, wire PE->switch.
  void drainPEOutput(InstanceHandle pe, unsigned pePort) {
    int idx = findFreeInSwitch();
    int port = nextInPort[idx]++;
    builder.connectPorts(pe, pePort, swAt(idx), port);
    rrIn = (idx + 1) % totalSwitches();
  }

  /// Connect a module input to a switch input port.
  void feedModuleInput(PortHandle mPort) {
    int idx = findFreeInSwitch();
    int port = nextInPort[idx]++;
    builder.connectToModuleInput(mPort, swAt(idx), port);
    rrIn = (idx + 1) % totalSwitches();
  }

  /// Connect a switch output port to a module output.
  void drainModuleOutput(PortHandle mPort) {
    int idx = findFreeOutSwitch();
    int port = nextOutPort[idx]++;
    builder.connectToModuleOutput(swAt(idx), port, mPort);
    rrOut = (idx + 1) % totalSwitches();
  }
};

//===----------------------------------------------------------------------===//
// PE catalog registration
//===----------------------------------------------------------------------===//

/// Build the complete PE catalog, returning a map from PE name to catalog entry.
inline std::map<std::string, PECatalogEntry>
registerAllPEs(ADGBuilder &builder) {
  std::map<std::string, PECatalogEntry> catalog;

  // Helper lambda to register a PE with bodyMLIR.
  auto reg = [&](const std::string &name,
                 const std::vector<std::string> &inTypes,
                 const std::vector<std::string> &outTypes,
                 const std::string &body,
                 int16_t latHW) {
    // Build Type vectors
    std::vector<Type> inT, outT;
    for (auto &t : inTypes) {
      if (t == "i32")        inT.push_back(Type::i32());
      else if (t == "f32")   inT.push_back(Type::f32());
      else if (t == "i64")   inT.push_back(Type::i64());
      else if (t == "i16")   inT.push_back(Type::i16());
      else if (t == "i1")    inT.push_back(Type::i1());
      else if (t == "index") inT.push_back(Type::index());
      else if (t == "none")  inT.push_back(Type::none());
    }
    for (auto &t : outTypes) {
      if (t == "i32")        outT.push_back(Type::i32());
      else if (t == "f32")   outT.push_back(Type::f32());
      else if (t == "i64")   outT.push_back(Type::i64());
      else if (t == "i16")   outT.push_back(Type::i16());
      else if (t == "i1")    outT.push_back(Type::i1());
      else if (t == "index") outT.push_back(Type::index());
      else if (t == "none")  outT.push_back(Type::none());
    }

    // Build the full MLIR body with header and yield.
    // The body already uses %a0, %a1, etc. We build a ^bb0 header and append
    // fabric.yield.
    std::string fullBody;
    // Build ^bb0 header
    fullBody += "^bb0(";
    for (size_t i = 0; i < inTypes.size(); ++i) {
      if (i > 0) fullBody += ", ";
      fullBody += "%a" + std::to_string(i) + ": " + inTypes[i];
    }
    fullBody += "):\n";
    fullBody += "  " + body + "\n";

    // Build fabric.yield
    if (outTypes.empty()) {
      fullBody += "  fabric.yield\n";
    } else {
      // Extract the result variable names from the body LHS
      std::string firstLine = body;
      // Find position of '='
      auto eqPos = firstLine.find('=');
      if (eqPos != std::string::npos) {
        std::string lhs = firstLine.substr(0, eqPos);
        // Trim trailing spaces
        while (!lhs.empty() && lhs.back() == ' ') lhs.pop_back();
        std::string typeStr;
        for (size_t i = 0; i < outTypes.size(); ++i) {
          if (i > 0) typeStr += ", ";
          typeStr += outTypes[i];
        }
        fullBody += "  fabric.yield " + lhs + " : " + typeStr + "\n";
      } else {
        // No results (e.g. sink operations)
        fullBody += "  fabric.yield\n";
      }
    }

    auto pe = builder.newPE(name)
        .setLatency(1, 1, latHW)
        .setInterval(1, 1, 1)
        .setInputPorts(inT)
        .setOutputPorts(outT)
        .setBodyMLIR(fullBody);

    PEDef def{name, inTypes, outTypes, body, latHW};
    catalog[name] = PECatalogEntry{def, PETemplateHandle{pe}};
  };

  // --- Constants ---
  reg("pe_const_i32", {"none"}, {"i32"},
      "%r = handshake.constant %a0 {value = 0 : i32} : i32", 1);
  reg("pe_const_f32", {"none"}, {"f32"},
      "%r = handshake.constant %a0 {value = 0.000000e+00 : f32} : f32", 1);
  reg("pe_const_i64", {"none"}, {"i64"},
      "%r = handshake.constant %a0 {value = 0 : i64} : i64", 1);
  reg("pe_const_index", {"none"}, {"index"},
      "%r = handshake.constant %a0 {value = 0 : index} : index", 1);
  reg("pe_const_i1", {"none"}, {"i1"},
      "%r = handshake.constant %a0 {value = true} : i1", 1);
  reg("pe_const_i16", {"none"}, {"i16"},
      "%r = handshake.constant %a0 {value = 0 : i16} : i16", 1);

  // --- Joins ---
  reg("pe_join1", {"none"}, {"none"},
      "%r = handshake.join %a0 : none", 1);
  reg("pe_join2", {"none", "none"}, {"none"},
      "%r = handshake.join %a0, %a1 : none, none", 1);
  reg("pe_join3", {"none", "none", "none"}, {"none"},
      "%r = handshake.join %a0, %a1, %a2 : none, none, none", 1);
  reg("pe_join4", {"none", "none", "none", "none"}, {"none"},
      "%r = handshake.join %a0, %a1, %a2, %a3 : none, none, none, none", 1);
  reg("pe_join5", {"none", "none", "none", "none", "none"}, {"none"},
      "%r = handshake.join %a0, %a1, %a2, %a3, %a4 "
      ": none, none, none, none, none", 1);
  reg("pe_join6",
      {"none", "none", "none", "none", "none", "none"}, {"none"},
      "%r = handshake.join %a0, %a1, %a2, %a3, %a4, %a5 "
      ": none, none, none, none, none, none", 1);
  reg("pe_join_i1", {"i1"}, {"none"},
      "%r = handshake.join %a0 : i1", 1);

  // --- Cond branch ---
  reg("pe_cond_br_none", {"i1", "none"}, {"none", "none"},
      "%t, %f = handshake.cond_br %a0, %a1 : none", 1);
  reg("pe_cond_br_i32", {"i1", "i32"}, {"i32", "i32"},
      "%t, %f = handshake.cond_br %a0, %a1 : i32", 1);
  reg("pe_cond_br_f32", {"i1", "f32"}, {"f32", "f32"},
      "%t, %f = handshake.cond_br %a0, %a1 : f32", 1);
  reg("pe_cond_br_index", {"i1", "index"}, {"index", "index"},
      "%t, %f = handshake.cond_br %a0, %a1 : index", 1);

  // --- Mux ---
  reg("pe_mux_i32", {"index", "i32", "i32"}, {"i32"},
      "%r = handshake.mux %a0 [%a1, %a2] : index, i32", 1);
  reg("pe_mux_f32", {"index", "f32", "f32"}, {"f32"},
      "%r = handshake.mux %a0 [%a1, %a2] : index, f32", 1);
  reg("pe_mux_none", {"index", "none", "none"}, {"none"},
      "%r = handshake.mux %a0 [%a1, %a2] : index, none", 1);
  reg("pe_mux_index", {"index", "index", "index"}, {"index"},
      "%r = handshake.mux %a0 [%a1, %a2] : index, index", 1);
  reg("pe_mux_i64", {"index", "i64", "i64"}, {"i64"},
      "%r = handshake.mux %a0 [%a1, %a2] : index, i64", 1);

  // --- Integer arithmetic (i32) ---
  reg("pe_addi", {"i32", "i32"}, {"i32"},
      "%r = arith.addi %a0, %a1 : i32", 1);
  reg("pe_subi", {"i32", "i32"}, {"i32"},
      "%r = arith.subi %a0, %a1 : i32", 1);
  reg("pe_muli", {"i32", "i32"}, {"i32"},
      "%r = arith.muli %a0, %a1 : i32", 2);
  reg("pe_divui", {"i32", "i32"}, {"i32"},
      "%r = arith.divui %a0, %a1 : i32", 10);
  reg("pe_divsi", {"i32", "i32"}, {"i32"},
      "%r = arith.divsi %a0, %a1 : i32", 10);
  reg("pe_remui", {"i32", "i32"}, {"i32"},
      "%r = arith.remui %a0, %a1 : i32", 10);
  reg("pe_remsi", {"i32", "i32"}, {"i32"},
      "%r = arith.remsi %a0, %a1 : i32", 10);

  // --- Integer arithmetic (i64) ---
  reg("pe_addi_i64", {"i64", "i64"}, {"i64"},
      "%r = arith.addi %a0, %a1 : i64", 1);
  reg("pe_subi_i64", {"i64", "i64"}, {"i64"},
      "%r = arith.subi %a0, %a1 : i64", 1);
  reg("pe_muli_i64", {"i64", "i64"}, {"i64"},
      "%r = arith.muli %a0, %a1 : i64", 2);
  reg("pe_cmpi_i64", {"i64", "i64"}, {"i1"},
      "%r = arith.cmpi ult, %a0, %a1 : i64", 1);
  reg("pe_shli_i64", {"i64", "i64"}, {"i64"},
      "%r = arith.shli %a0, %a1 : i64", 1);
  reg("pe_remui_i64", {"i64", "i64"}, {"i64"},
      "%r = arith.remui %a0, %a1 : i64", 10);

  // --- Integer arithmetic (index) ---
  reg("pe_addi_index", {"index", "index"}, {"index"},
      "%r = arith.addi %a0, %a1 : index", 1);
  reg("pe_subi_index", {"index", "index"}, {"index"},
      "%r = arith.subi %a0, %a1 : index", 1);
  reg("pe_divui_index", {"index", "index"}, {"index"},
      "%r = arith.divui %a0, %a1 : index", 10);
  reg("pe_divsi_index", {"index", "index"}, {"index"},
      "%r = arith.divsi %a0, %a1 : index", 10);
  reg("pe_remui_index", {"index", "index"}, {"index"},
      "%r = arith.remui %a0, %a1 : index", 10);
  reg("pe_muli_index", {"index", "index"}, {"index"},
      "%r = arith.muli %a0, %a1 : index", 2);

  // --- Bitwise (i32) ---
  reg("pe_andi", {"i32", "i32"}, {"i32"},
      "%r = arith.andi %a0, %a1 : i32", 1);
  reg("pe_ori", {"i32", "i32"}, {"i32"},
      "%r = arith.ori %a0, %a1 : i32", 1);
  reg("pe_xori", {"i32", "i32"}, {"i32"},
      "%r = arith.xori %a0, %a1 : i32", 1);
  reg("pe_shli", {"i32", "i32"}, {"i32"},
      "%r = arith.shli %a0, %a1 : i32", 1);
  reg("pe_shrui", {"i32", "i32"}, {"i32"},
      "%r = arith.shrui %a0, %a1 : i32", 1);
  reg("pe_shrsi", {"i32", "i32"}, {"i32"},
      "%r = arith.shrsi %a0, %a1 : i32", 1);

  // --- Bitwise (i1) ---
  reg("pe_xori_i1", {"i1", "i1"}, {"i1"},
      "%r = arith.xori %a0, %a1 : i1", 1);

  // --- Float arithmetic ---
  reg("pe_addf", {"f32", "f32"}, {"f32"},
      "%r = arith.addf %a0, %a1 : f32", 3);
  reg("pe_subf", {"f32", "f32"}, {"f32"},
      "%r = arith.subf %a0, %a1 : f32", 3);
  reg("pe_mulf", {"f32", "f32"}, {"f32"},
      "%r = arith.mulf %a0, %a1 : f32", 3);
  reg("pe_divf", {"f32", "f32"}, {"f32"},
      "%r = arith.divf %a0, %a1 : f32", 10);
  reg("pe_fma", {"f32", "f32", "f32"}, {"f32"},
      "%r = math.fma %a0, %a1, %a2 : f32", 3);

  // --- Float unary ---
  reg("pe_negf", {"f32"}, {"f32"},
      "%r = arith.negf %a0 : f32", 1);
  reg("pe_absf", {"f32"}, {"f32"},
      "%r = math.absf %a0 : f32", 1);
  reg("pe_sin", {"f32"}, {"f32"},
      "%r = math.sin %a0 : f32", 10);
  reg("pe_cos", {"f32"}, {"f32"},
      "%r = math.cos %a0 : f32", 10);
  reg("pe_exp", {"f32"}, {"f32"},
      "%r = math.exp %a0 : f32", 10);
  reg("pe_sqrt", {"f32"}, {"f32"},
      "%r = math.sqrt %a0 : f32", 10);
  reg("pe_log2", {"f32"}, {"f32"},
      "%r = math.log2 %a0 : f32", 10);

  // --- Compare ---
  reg("pe_cmpi", {"i32", "i32"}, {"i1"},
      "%r = arith.cmpi ult, %a0, %a1 : i32", 1);
  reg("pe_cmpf", {"f32", "f32"}, {"i1"},
      "%r = arith.cmpf ult, %a0, %a1 : f32", 1);

  // --- Select ---
  reg("pe_select", {"i1", "i32", "i32"}, {"i32"},
      "%r = arith.select %a0, %a1, %a2 : i32", 1);
  reg("pe_select_index", {"i1", "index", "index"}, {"index"},
      "%r = arith.select %a0, %a1, %a2 : index", 1);
  reg("pe_select_f32", {"i1", "f32", "f32"}, {"f32"},
      "%r = arith.select %a0, %a1, %a2 : f32", 1);

  // --- Type cast ---
  reg("pe_index_cast_i32", {"i32"}, {"index"},
      "%r = arith.index_cast %a0 : i32 to index", 1);
  reg("pe_index_cast_i64", {"i64"}, {"index"},
      "%r = arith.index_cast %a0 : i64 to index", 1);
  reg("pe_index_cast_to_i32", {"index"}, {"i32"},
      "%r = arith.index_cast %a0 : index to i32", 1);
  reg("pe_index_cast_to_i64", {"index"}, {"i64"},
      "%r = arith.index_cast %a0 : index to i64", 1);
  reg("pe_index_castui", {"i32"}, {"index"},
      "%r = arith.index_castui %a0 : i32 to index", 1);
  reg("pe_extui", {"i32"}, {"i64"},
      "%r = arith.extui %a0 : i32 to i64", 1);
  reg("pe_trunci", {"i64"}, {"i32"},
      "%r = arith.trunci %a0 : i64 to i32", 1);
  reg("pe_extui_i1", {"i1"}, {"i32"},
      "%r = arith.extui %a0 : i1 to i32", 1);
  reg("pe_trunci_to_i1", {"i32"}, {"i1"},
      "%r = arith.trunci %a0 : i32 to i1", 1);
  reg("pe_extui_i16", {"i16"}, {"i32"},
      "%r = arith.extui %a0 : i16 to i32", 1);
  reg("pe_trunci_to_i16", {"i64"}, {"i16"},
      "%r = arith.trunci %a0 : i64 to i16", 1);
  reg("pe_remui_i16", {"i16", "i16"}, {"i16"},
      "%r = arith.remui %a0, %a1 : i16", 10);
  reg("pe_uitofp", {"i32"}, {"f32"},
      "%r = arith.uitofp %a0 : i32 to f32", 3);
  reg("pe_uitofp_i16", {"i16"}, {"f32"},
      "%r = arith.uitofp %a0 : i16 to f32", 3);
  reg("pe_sitofp", {"i32"}, {"f32"},
      "%r = arith.sitofp %a0 : i32 to f32", 3);
  reg("pe_fptoui", {"f32"}, {"i32"},
      "%r = arith.fptoui %a0 : f32 to i32", 3);

  // --- Dataflow ---
  reg("pe_stream", {"index", "index", "index"}, {"index", "i1"},
      "%idx, %wc = dataflow.stream %a0, %a1, %a2", 1);
  reg("pe_gate", {"i32", "i1"}, {"i32", "i1"},
      "%av, %ac = dataflow.gate %a0, %a1 : i32, i1 -> i32, i1", 1);
  reg("pe_gate_f32", {"f32", "i1"}, {"f32", "i1"},
      "%av, %ac = dataflow.gate %a0, %a1 : f32, i1 -> f32, i1", 1);
  reg("pe_gate_index", {"index", "i1"}, {"index", "i1"},
      "%av, %ac = dataflow.gate %a0, %a1 : index, i1 -> index, i1", 1);
  reg("pe_carry", {"i1", "i32", "i32"}, {"i32"},
      "%o = dataflow.carry %a0, %a1, %a2 : i1, i32, i32 -> i32", 1);
  reg("pe_carry_f32", {"i1", "f32", "f32"}, {"f32"},
      "%o = dataflow.carry %a0, %a1, %a2 : i1, f32, f32 -> f32", 1);
  reg("pe_carry_none", {"i1", "none", "none"}, {"none"},
      "%o = dataflow.carry %a0, %a1, %a2 : i1, none, none -> none", 1);
  reg("pe_carry_index", {"i1", "index", "index"}, {"index"},
      "%o = dataflow.carry %a0, %a1, %a2 : i1, index, index -> index", 1);
  reg("pe_invariant", {"i1", "i32"}, {"i32"},
      "%o = dataflow.invariant %a0, %a1 : i1, i32 -> i32", 1);
  reg("pe_invariant_i1", {"i1", "i1"}, {"i1"},
      "%o = dataflow.invariant %a0, %a1 : i1, i1 -> i1", 1);
  reg("pe_invariant_none", {"i1", "none"}, {"none"},
      "%o = dataflow.invariant %a0, %a1 : i1, none -> none", 1);
  reg("pe_invariant_f32", {"i1", "f32"}, {"f32"},
      "%o = dataflow.invariant %a0, %a1 : i1, f32 -> f32", 1);
  reg("pe_invariant_index", {"i1", "index"}, {"index"},
      "%o = dataflow.invariant %a0, %a1 : i1, index -> index", 1);
  reg("pe_invariant_i64", {"i1", "i64"}, {"i64"},
      "%o = dataflow.invariant %a0, %a1 : i1, i64 -> i64", 1);

  // --- Memory ---
  reg("pe_load", {"index", "i32", "none"}, {"i32", "index"},
      "%ld_d, %ld_a = handshake.load [%a0] %a1, %a2 : index, i32", 1);
  reg("pe_load_f32", {"index", "f32", "none"}, {"f32", "index"},
      "%ld_d, %ld_a = handshake.load [%a0] %a1, %a2 : index, f32", 1);
  reg("pe_store", {"index", "i32", "none"}, {"i32", "index"},
      "%st_d, %st_a = handshake.store [%a0] %a1, %a2 : index, i32", 1);
  reg("pe_store_f32", {"index", "f32", "none"}, {"f32", "index"},
      "%st_d, %st_a = handshake.store [%a0] %a1, %a2 : index, f32", 1);

  // --- Sink ---
  reg("pe_sink_i1", {"i1"}, {},
      "handshake.sink %a0 : i1", 1);
  reg("pe_sink_none", {"none"}, {},
      "handshake.sink %a0 : none", 1);
  reg("pe_sink_i32", {"i32"}, {},
      "handshake.sink %a0 : i32", 1);
  reg("pe_sink_index", {"index"}, {},
      "handshake.sink %a0 : index", 1);
  reg("pe_sink_f32", {"f32"}, {},
      "handshake.sink %a0 : f32", 1);

  return catalog;
}

//===----------------------------------------------------------------------===//
// Lattice sizing helper
//===----------------------------------------------------------------------===//

/// Calculate lattice dimensions for a type plane given the number of PE-facing
/// connections needed. Each switch provides (portsPerSw - 4) usable ports.
/// Returns {peRows, peCols} for the lattice mesh. Minimum is 1x1.
inline std::pair<int, int> computeLatticeDims(int totalConns, int portsPerSw) {
  int usablePerSw = portsPerSw - 4;
  if (usablePerSw <= 0) usablePerSw = 1;
  if (totalConns <= 0) return {1, 1};
  // We need enough switches. swCount = (peRows+1)*(peCols+1).
  // Target: swCount * usablePerSw >= totalConns.
  int neededSwitches = (totalConns + usablePerSw - 1) / usablePerSw;
  // swCount = (peRows+1)^2 for square lattice.
  // (peRows+1) = ceil(sqrt(neededSwitches))
  int side = static_cast<int>(std::ceil(std::sqrt(
      static_cast<double>(neededSwitches))));
  if (side < 2) side = 2;
  int peRows = side - 1;
  int peCols = side - 1;
  if (peRows < 1) peRows = 1;
  if (peCols < 1) peCols = 1;
  return {peRows, peCols};
}

//===----------------------------------------------------------------------===//
// Connection counting helper
//===----------------------------------------------------------------------===//

/// Count how many input and output connections each type plane needs.
/// Returns a pair of arrays: [inConns per plane, outConns per plane].
inline void countPlaneConnections(
    const std::map<std::string, int> &instances,
    const std::map<std::string, PECatalogEntry> &catalog,
    int planeConns[TP_COUNT]) {
  for (int i = 0; i < TP_COUNT; ++i) planeConns[i] = 0;

  for (auto &kv : instances) {
    auto it = catalog.find(kv.first);
    if (it == catalog.end()) continue;
    auto &def = it->second.def;
    int count = kv.second;
    for (auto &t : def.inputTypes)
      planeConns[typeNameToPlane(t)] += count;
    for (auto &t : def.outputTypes)
      planeConns[typeNameToPlane(t)] += count;
  }
}

//===----------------------------------------------------------------------===//
// Memory connection helpers
//===----------------------------------------------------------------------===//

/// Compute the minimum tag type width for multi-port memory.
inline Type computeMemTagType(unsigned count) {
  unsigned tw = 1;
  while ((1u << tw) < count)
    tw++;
  return Type::iN(tw);
}

/// ExtMemory/Memory port layout info (presence-based).
struct MemPortLayout {
  int numIn;
  int numOut;
  bool isTagged;
  // Input port indices (relative to port 0 for Memory, port 1 for ExtMemory)
  int ldAddrPort;  // -1 if absent
  int stAddrPort;  // -1 if absent
  int stDataPort;  // -1 if absent
  // Output port indices
  int ldDataPort;  // -1 if absent
  int ldDonePort;  // -1 if absent
  int stDonePort;  // -1 if absent
};

/// Compute the presence-based port layout for ExtMemory.
inline MemPortLayout extMemPortLayout(int ldCount, int stCount) {
  MemPortLayout layout{};
  layout.isTagged = (ldCount > 1 || stCount > 1);
  layout.ldAddrPort = -1;
  layout.stAddrPort = -1;
  layout.stDataPort = -1;
  layout.ldDataPort = -1;
  layout.ldDonePort = -1;
  layout.stDonePort = -1;

  // Inputs: [memref(0)] [ld_addr?] [st_addr? st_data?]
  int inPort = 1; // port 0 is memref
  if (ldCount > 0) { layout.ldAddrPort = inPort++; }
  if (stCount > 0) { layout.stAddrPort = inPort++; layout.stDataPort = inPort++; }
  layout.numIn = inPort;

  // Outputs: [ld_data? ld_done?] [st_done?]
  int outPort = 0;
  if (ldCount > 0) { layout.ldDataPort = outPort++; layout.ldDonePort = outPort++; }
  if (stCount > 0) { layout.stDonePort = outPort++; }
  layout.numOut = outPort;

  return layout;
}

/// Compute the presence-based port layout for private Memory.
inline MemPortLayout privMemPortLayout(int ldCount, int stCount) {
  MemPortLayout layout{};
  layout.isTagged = (ldCount > 1 || stCount > 1);
  layout.ldAddrPort = -1;
  layout.stAddrPort = -1;
  layout.stDataPort = -1;
  layout.ldDataPort = -1;
  layout.ldDonePort = -1;
  layout.stDonePort = -1;

  // Inputs: [ld_addr?] [st_addr? st_data?]
  int inPort = 0;
  if (ldCount > 0) { layout.ldAddrPort = inPort++; }
  if (stCount > 0) { layout.stAddrPort = inPort++; layout.stDataPort = inPort++; }
  layout.numIn = inPort;

  // Outputs: [ld_data? ld_done?] [st_done?]
  int outPort = 0;
  if (ldCount > 0) { layout.ldDataPort = outPort++; layout.ldDonePort = outPort++; }
  if (stCount > 0) { layout.stDonePort = outPort++; }
  layout.numOut = outPort;

  return layout;
}

/// Connect an ExtMemory instance to lattice connectors for native ports, or to
/// direct module I/O for tagged ports. The memref input (port 0) must already
/// be connected before calling this function.
///
/// For native (untagged) ports, data/address ports connect through the
/// appropriate type-plane lattice connectors. For tagged ports (when
/// ldCount > 1 or stCount > 1), ports are connected as direct module I/O.
inline void connectExtMem(ADGBuilder &builder,
                          InstanceHandle inst,
                          const std::string &name,
                          const std::string &elemType,
                          int ldCount, int stCount,
                          std::vector<LatticeConnector *> &planeConn) {
  auto layout = extMemPortLayout(ldCount, stCount);
  TypePlane dataTp = typeNameToPlane(elemType);

  if (!layout.isTagged) {
    // Native ports: connect through lattice
    if (layout.ldAddrPort >= 0)
      planeConn[TP_INDEX]->feedPEInput(inst, layout.ldAddrPort);
    if (layout.stAddrPort >= 0)
      planeConn[TP_INDEX]->feedPEInput(inst, layout.stAddrPort);
    if (layout.stDataPort >= 0)
      planeConn[dataTp]->feedPEInput(inst, layout.stDataPort);

    if (layout.ldDataPort >= 0)
      planeConn[dataTp]->drainPEOutput(inst, layout.ldDataPort);
    if (layout.ldDonePort >= 0)
      planeConn[TP_NONE]->drainPEOutput(inst, layout.ldDonePort);
    if (layout.stDonePort >= 0)
      planeConn[TP_NONE]->drainPEOutput(inst, layout.stDonePort);
  } else {
    // Tagged ports: connect as direct module I/O
    unsigned maxCount = std::max((unsigned)ldCount, (unsigned)stCount);
    Type tagType = computeMemTagType(maxCount);
    Type taggedIndex = Type::tagged(Type::index(), tagType);
    Type taggedData = Type::tagged(typePlaneToType(dataTp), tagType);
    Type taggedNone = Type::tagged(Type::none(), tagType);

    if (layout.ldAddrPort >= 0) {
      auto p = builder.addModuleInput(name + "_ld_addr", taggedIndex);
      builder.connectToModuleInput(p, inst, layout.ldAddrPort);
    }
    if (layout.stAddrPort >= 0) {
      auto p = builder.addModuleInput(name + "_st_addr", taggedIndex);
      builder.connectToModuleInput(p, inst, layout.stAddrPort);
    }
    if (layout.stDataPort >= 0) {
      auto p = builder.addModuleInput(name + "_st_data", taggedData);
      builder.connectToModuleInput(p, inst, layout.stDataPort);
    }
    if (layout.ldDataPort >= 0) {
      auto p = builder.addModuleOutput(name + "_ld_data", taggedData);
      builder.connectToModuleOutput(inst, layout.ldDataPort, p);
    }
    if (layout.ldDonePort >= 0) {
      auto p = builder.addModuleOutput(name + "_ld_done", taggedNone);
      builder.connectToModuleOutput(inst, layout.ldDonePort, p);
    }
    if (layout.stDonePort >= 0) {
      auto p = builder.addModuleOutput(name + "_st_done", taggedNone);
      builder.connectToModuleOutput(inst, layout.stDonePort, p);
    }
  }
}

/// Connect a private Memory instance to lattice connectors for native ports,
/// or to direct module I/O for tagged ports.
inline void connectPrivMem(ADGBuilder &builder,
                           InstanceHandle inst,
                           const std::string &name,
                           const std::string &elemType,
                           int ldCount, int stCount,
                           std::vector<LatticeConnector *> &planeConn) {
  auto layout = privMemPortLayout(ldCount, stCount);
  TypePlane dataTp = typeNameToPlane(elemType);

  if (!layout.isTagged) {
    if (layout.ldAddrPort >= 0)
      planeConn[TP_INDEX]->feedPEInput(inst, layout.ldAddrPort);
    if (layout.stAddrPort >= 0)
      planeConn[TP_INDEX]->feedPEInput(inst, layout.stAddrPort);
    if (layout.stDataPort >= 0)
      planeConn[dataTp]->feedPEInput(inst, layout.stDataPort);

    if (layout.ldDataPort >= 0)
      planeConn[dataTp]->drainPEOutput(inst, layout.ldDataPort);
    if (layout.ldDonePort >= 0)
      planeConn[TP_NONE]->drainPEOutput(inst, layout.ldDonePort);
    if (layout.stDonePort >= 0)
      planeConn[TP_NONE]->drainPEOutput(inst, layout.stDonePort);
  } else {
    unsigned maxCount = std::max((unsigned)ldCount, (unsigned)stCount);
    Type tagType = computeMemTagType(maxCount);
    Type taggedIndex = Type::tagged(Type::index(), tagType);
    Type taggedData = Type::tagged(typePlaneToType(dataTp), tagType);
    Type taggedNone = Type::tagged(Type::none(), tagType);

    if (layout.ldAddrPort >= 0) {
      auto p = builder.addModuleInput(name + "_ld_addr", taggedIndex);
      builder.connectToModuleInput(p, inst, layout.ldAddrPort);
    }
    if (layout.stAddrPort >= 0) {
      auto p = builder.addModuleInput(name + "_st_addr", taggedIndex);
      builder.connectToModuleInput(p, inst, layout.stAddrPort);
    }
    if (layout.stDataPort >= 0) {
      auto p = builder.addModuleInput(name + "_st_data", taggedData);
      builder.connectToModuleInput(p, inst, layout.stDataPort);
    }
    if (layout.ldDataPort >= 0) {
      auto p = builder.addModuleOutput(name + "_ld_data", taggedData);
      builder.connectToModuleOutput(inst, layout.ldDataPort, p);
    }
    if (layout.ldDonePort >= 0) {
      auto p = builder.addModuleOutput(name + "_ld_done", taggedNone);
      builder.connectToModuleOutput(inst, layout.ldDonePort, p);
    }
    if (layout.stDonePort >= 0) {
      auto p = builder.addModuleOutput(name + "_st_done", taggedNone);
      builder.connectToModuleOutput(inst, layout.stDonePort, p);
    }
  }
}

/// Count plane connections for extmemory/privmem configs (native ports only).
/// Tagged ports connect as direct module I/O and don't go through the lattice.
inline void countMemPlaneConns(const std::string &elemType,
                               int ldCount, int stCount,
                               bool isExtMem,
                               int planeConns[TP_COUNT]) {
  bool isTagged = (ldCount > 1 || stCount > 1);
  if (isTagged) return; // tagged ports bypass lattice

  TypePlane dataTp = typeNameToPlane(elemType);
  if (ldCount > 0) planeConns[TP_INDEX]++;     // ld_addr
  if (stCount > 0) {
    planeConns[TP_INDEX]++;                     // st_addr
    planeConns[dataTp]++;                       // st_data
  }
  if (ldCount > 0) {
    planeConns[dataTp]++;                       // ld_data
    planeConns[TP_NONE]++;                      // ld_done
  }
  if (stCount > 0) planeConns[TP_NONE]++;       // st_done
}

#endif // CGRA_PE_CATALOG_H
