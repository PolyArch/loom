//===-- ADGBuilder.cpp - ADG Builder implementation ---------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"
#include "fcc/ADG/ADGVerifier.h"

#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <set>
#include <sstream>

namespace fcc {
namespace adg {

//===----------------------------------------------------------------------===//
// Internal Data Structures
//===----------------------------------------------------------------------===//

struct FUDef {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<std::string> ops;
  unsigned latency = 1;
  unsigned interval = 1;
};

struct PEDef {
  std::string name;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  unsigned bitsWidth = 32;
  std::vector<unsigned> fuIndices;
};

struct SWDef {
  std::string name;
  std::vector<unsigned> inputWidths;
  std::vector<unsigned> outputWidths;
  std::vector<std::vector<bool>> connectivity;
  int decomposableBits = -1;
};

struct ExtMemDef {
  std::string name;
  unsigned ldPorts = 1;
  unsigned stPorts = 1;
  unsigned lsqDepth = 0;
};

struct FIFODef {
  std::string name;
  unsigned depth = 2;
  unsigned bitsWidth = 32;
};

enum class InstanceKind { PE, SW, ExtMem, FIFO };

struct InstanceDef {
  InstanceKind kind;
  unsigned defIdx;
  std::string name;
};

struct Connection {
  unsigned srcInst;
  unsigned srcPort;
  unsigned dstInst;
  unsigned dstPort;
};

struct MemrefInput {
  std::string name;
  std::string typeStr;
};

struct MemrefConnection {
  unsigned memrefIdx;
  unsigned extMemInstIdx;
};

struct ScalarInput {
  std::string name;
  unsigned bitsWidth;
};

struct ScalarOutput {
  std::string name;
  unsigned bitsWidth;
};

struct ScalarToInstanceConn {
  unsigned scalarIdx;
  unsigned dstInst;
  unsigned dstPort;
};

struct InstanceToScalarConn {
  unsigned srcInst;
  unsigned srcPort;
  unsigned scalarOutputIdx;
};

struct VizPlacement {
  double centerX = 0.0;
  double centerY = 0.0;
  int gridRow = -1;
  int gridCol = -1;
};

//===----------------------------------------------------------------------===//
// Builder Implementation
//===----------------------------------------------------------------------===//

struct ADGBuilder::Impl {
  std::string moduleName;

  std::vector<FUDef> fuDefs;
  std::vector<PEDef> peDefs;
  std::vector<SWDef> swDefs;
  std::vector<ExtMemDef> extMemDefs;
  std::vector<FIFODef> fifoDefs;

  std::vector<InstanceDef> instances;
  std::vector<Connection> connections;

  std::vector<MemrefInput> memrefInputs;
  std::vector<MemrefConnection> memrefConnections;

  std::vector<ScalarInput> scalarInputs;
  std::vector<ScalarOutput> scalarOutputs;

  std::vector<ScalarToInstanceConn> scalarToInstConns;
  std::vector<InstanceToScalarConn> instToScalarConns;
  std::map<unsigned, VizPlacement> vizPlacements;

  std::string generateMLIR(llvm::StringRef vizFileName) const;
  std::string generateVizJson() const;
  bool validate(std::string &errMsg) const;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static std::string bitsType(unsigned width) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

/// Generate FU body with the actual MLIR operation inside.
/// Each FU contains the real op it implements, so the body IS the
/// hardware operation graph.
static void emitFUBody(std::ostringstream &os, const FUDef &fu,
                       const std::string &indent) {
  if (fu.ops.empty() || fu.outputTypes.empty()) {
    os << indent << "  fabric.yield\n";
    return;
  }

  // Multi-op FU: generate an internal DAG with mux output selection.
  // For FUs with 2+ ops, chain them and add a fabric.mux at the output.
  if (fu.ops.size() >= 2) {
    // Generate the first op (e.g. arith.muli %a, %b -> %d)
    const std::string &op0 = fu.ops[0];
    os << indent << "  %d = " << op0 << " %arg0, %arg1 : "
       << fu.outputTypes[0] << "\n";

    // Generate the second op using the first result (e.g. arith.addi %d, %c)
    const std::string &op1 = fu.ops[1];
    // The second op uses the first result plus the next available input.
    unsigned nextArg = 2;
    if (nextArg < fu.inputTypes.size()) {
      os << indent << "  %e = " << op1 << " %d, %arg" << nextArg << " : "
         << fu.outputTypes[0] << "\n";
    } else {
      os << indent << "  %e = " << op1 << " %d, %arg1 : "
         << fu.outputTypes[0] << "\n";
    }

    // Generate fabric.mux to select between the two results.
    // Format: fabric.mux %d, %e {sel = 0 : i64} : T, T -> T
    os << indent << "  %g = fabric.mux"
       << " %d, %e"
       << " {sel = 0 : i64, discard = false, disconnect = false}"
       << " : " << fu.outputTypes[0] << ", " << fu.outputTypes[0]
       << " -> " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %g : " << fu.outputTypes[0] << "\n";
    return;
  }

  const std::string &op = fu.ops[0];

  // --- dataflow dialect ops ---

  // dataflow.stream: (index, index, index) -> (index, i1)
  if (op == "dataflow.stream") {
    os << indent << "  %0, %1 = dataflow.stream %arg0, %arg1, %arg2"
       << " {step_op = \"" << "+=" << "\", cont_cond = \"" << "<" << "\"}"
       << " : (index, index, index) -> (index, i1)\n";
    os << indent << "  fabric.yield %0, %1 : index, i1\n";
    return;
  }

  // dataflow.gate: (T, i1) -> (T, i1)
  if (op == "dataflow.gate") {
    os << indent << "  %0, %1 = dataflow.gate %arg0, %arg1 : "
       << fu.inputTypes[0] << ", i1 -> " << fu.outputTypes[0] << ", i1\n";
    os << indent << "  fabric.yield %0, %1 : "
       << fu.outputTypes[0] << ", i1\n";
    return;
  }

  // dataflow.carry: (i1, T, T) -> (T)
  if (op == "dataflow.carry") {
    os << indent << "  %0 = dataflow.carry %arg0, %arg1, %arg2 : "
       << "i1, " << fu.inputTypes[1] << ", " << fu.inputTypes[2]
       << " -> " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // dataflow.invariant: (i1, T) -> (T)
  if (op == "dataflow.invariant") {
    os << indent << "  %0 = dataflow.invariant %arg0, %arg1 : "
       << "i1, " << fu.inputTypes[1] << " -> " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // --- handshake dialect ops ---

  // handshake.load: (addr_indices..., data, ctrl) -> (data, addr_indices...)
  // Assembly: %d, %a = handshake.load [%addr] %data, %ctrl : index, i32
  if (op == "handshake.load") {
    os << indent << "  %0, %1 = handshake.load [%arg0] %arg1, %arg2 : "
       << fu.inputTypes[0] << ", " << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0, %1 : "
       << fu.outputTypes[0] << ", " << fu.outputTypes[1] << "\n";
    return;
  }

  // handshake.store: (addr_indices..., data, ctrl) -> (data, addr_indices...)
  // Assembly: %d, %a = handshake.store [%addr] %data, %ctrl : index, i32
  if (op == "handshake.store") {
    os << indent << "  %0, %1 = handshake.store [%arg0] %arg1, %arg2 : "
       << fu.inputTypes[0] << ", " << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0, %1 : "
       << fu.outputTypes[0] << ", " << fu.outputTypes[1] << "\n";
    return;
  }

  // handshake.constant: (none) -> (T)
  // Assembly: %c = handshake.constant %ctrl {value = 0 : T} : T
  if (op == "handshake.constant") {
    os << indent << "  %0 = handshake.constant %arg0 {value = 0 : "
       << fu.outputTypes[0] << "} : " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // handshake.cond_br: (i1, T) -> (T, T)
  // Assembly: %t, %f = handshake.cond_br %cond, %data : T
  if (op == "handshake.cond_br") {
    os << indent << "  %0, %1 = handshake.cond_br %arg0, %arg1 : "
       << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0, %1 : "
       << fu.outputTypes[0] << ", " << fu.outputTypes[1] << "\n";
    return;
  }

  // handshake.mux: (index, T, T, ...) -> (T)
  // Assembly: %r = handshake.mux %sel [%d0, %d1] : index, T
  if (op == "handshake.mux") {
    os << indent << "  %0 = handshake.mux %arg0 [%arg1, %arg2] : "
       << fu.inputTypes[0] << ", " << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // handshake.join: (T0, T1, ...) -> (none)
  // Assembly: %r = handshake.join %a, %b[, %c, ...] : T0, T1[, T2, ...]
  if (op == "handshake.join") {
    os << indent << "  %0 = handshake.join %arg0";
    for (size_t j = 1; j < fu.inputTypes.size(); ++j)
      os << ", %arg" << j;
    os << " : " << fu.inputTypes[0];
    for (size_t j = 1; j < fu.inputTypes.size(); ++j)
      os << ", " << fu.inputTypes[j];
    os << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // --- arith/math dialect ops ---

  // Compare ops
  if (op == "arith.cmpi") {
    os << indent << "  %0 = arith.cmpi eq, %arg0, %arg1 : "
       << fu.inputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }
  if (op == "arith.cmpf") {
    os << indent << "  %0 = arith.cmpf oeq, %arg0, %arg1 : "
       << fu.inputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // arith.select: (%cond: i1, %a: T, %b: T) -> (T)
  if (op == "arith.select") {
    os << indent << "  %0 = arith.select %arg0, %arg1, %arg2 : "
       << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // Extension/truncation ops (1 input, 1 output, different types)
  if (op == "arith.extui" || op == "arith.extsi" ||
      op == "arith.trunci" || op == "arith.fptosi" ||
      op == "arith.fptoui" || op == "arith.sitofp" ||
      op == "arith.uitofp") {
    os << indent << "  %0 = " << op << " %arg0 : "
       << fu.inputTypes[0] << " to " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // arith.index_cast / index_castui
  if (op == "arith.index_cast" || op == "arith.index_castui") {
    os << indent << "  %0 = " << op << " %arg0 : "
       << fu.inputTypes[0] << " to " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // Unary ops (negf, absf, cos, sin, exp, log2, sqrt)
  if (op == "arith.negf" || op == "math.absf" || op == "math.cos" ||
      op == "math.sin" || op == "math.exp" || op == "math.log2" ||
      op == "math.sqrt") {
    os << indent << "  %0 = " << op << " %arg0 : "
       << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // math.fma: 3-input
  if (op == "math.fma") {
    os << indent << "  %0 = math.fma %arg0, %arg1, %arg2 : "
       << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  // Default: standard binary ops (addi, subi, muli, andi, ori, xori,
  // addf, subf, mulf, divf, divsi, divui, remsi, remui, shli, shrsi, shrui)
  os << indent << "  %0 = " << op << " %arg0, %arg1 : "
     << fu.outputTypes[0] << "\n";
  os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
}

/// Get the number of output ports for an instance.
static unsigned getInstanceOutputCount(const std::vector<InstanceDef> &instances,
                                       const std::vector<PEDef> &peDefs,
                                       const std::vector<SWDef> &swDefs,
                                       const std::vector<ExtMemDef> &extMemDefs,
                                       unsigned instIdx) {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case InstanceKind::PE:
    return peDefs[inst.defIdx].numOutputs;
  case InstanceKind::SW:
    return swDefs[inst.defIdx].outputWidths.size();
  case InstanceKind::ExtMem: {
    const auto &mem = extMemDefs[inst.defIdx];
    return mem.ldPorts + mem.stPorts + mem.ldPorts;
  }
  case InstanceKind::FIFO:
    return 1;
  }
  return 0;
}

static unsigned getInstanceInputWidth(const std::vector<InstanceDef> &instances,
                                      const std::vector<PEDef> &peDefs,
                                      const std::vector<SWDef> &swDefs,
                                      const std::vector<FIFODef> &fifoDefs,
                                      unsigned instIdx, unsigned portIdx) {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case InstanceKind::PE:
    return peDefs[inst.defIdx].bitsWidth;
  case InstanceKind::SW:
    return swDefs[inst.defIdx].inputWidths[portIdx];
  case InstanceKind::FIFO:
    return fifoDefs[inst.defIdx].bitsWidth;
  case InstanceKind::ExtMem:
    return 32;
  }
  return 32;
}

static unsigned getInstanceOutputWidth(const std::vector<InstanceDef> &instances,
                                       const std::vector<PEDef> &peDefs,
                                       const std::vector<SWDef> &swDefs,
                                       const std::vector<FIFODef> &fifoDefs,
                                       unsigned instIdx, unsigned portIdx) {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case InstanceKind::PE:
    return peDefs[inst.defIdx].bitsWidth;
  case InstanceKind::SW:
    return swDefs[inst.defIdx].outputWidths[portIdx];
  case InstanceKind::FIFO:
    return fifoDefs[inst.defIdx].bitsWidth;
  case InstanceKind::ExtMem:
    return 32;
  }
  return 32;
}

//===----------------------------------------------------------------------===//
// Full MLIR generation (instance-based)
//===----------------------------------------------------------------------===//

std::string ADGBuilder::Impl::generateMLIR(llvm::StringRef vizFileName) const {
  std::ostringstream os;

  // Top-level module wrapper required for parseSourceFile<ModuleOp>.
  os << "module {\n";

  // Module header: memref inputs, then scalar inputs
  os << "fabric.module @" << moduleName << "(";
  unsigned argIdx = 0;
  for (size_t i = 0; i < memrefInputs.size(); ++i) {
    if (argIdx > 0) os << ", ";
    os << "%mem" << i << ": " << memrefInputs[i].typeStr;
    argIdx++;
  }
  for (size_t i = 0; i < scalarInputs.size(); ++i) {
    if (argIdx > 0) os << ", ";
    os << "%scalar" << i << ": " << bitsType(scalarInputs[i].bitsWidth);
    argIdx++;
  }
  os << ") -> (";
  for (size_t i = 0; i < scalarOutputs.size(); ++i) {
    if (i > 0) os << ", ";
    os << bitsType(scalarOutputs[i].bitsWidth);
  }
  os << ")";
  if (!vizFileName.empty())
    os << " attributes {viz_file = \"" << vizFileName.str() << "\"}";
  os << " {\n";

  // Collect which PE and SW definitions are actually used.
  std::set<unsigned> usedPEDefs, usedSWDefs, usedExtMemDefs, usedFIFODefs;
  for (const auto &inst : instances) {
    switch (inst.kind) {
    case InstanceKind::PE: usedPEDefs.insert(inst.defIdx); break;
    case InstanceKind::SW: usedSWDefs.insert(inst.defIdx); break;
    case InstanceKind::ExtMem: usedExtMemDefs.insert(inst.defIdx); break;
    case InstanceKind::FIFO: usedFIFODefs.insert(inst.defIdx); break;
    }
  }

  // Emit PE type definitions (once per unique PE type).
  for (unsigned peIdx : usedPEDefs) {
    const auto &pe = peDefs[peIdx];
    std::string bt = bitsType(pe.bitsWidth);

    os << "  fabric.spatial_pe @" << pe.name << "(";
    for (unsigned p = 0; p < pe.numInputs; ++p) {
      if (p > 0) os << ", ";
      os << "%in" << p << ": " << bt;
    }
    os << ") -> (";
    for (unsigned p = 0; p < pe.numOutputs; ++p) {
      if (p > 0) os << ", ";
      os << bt;
    }
    os << ") {\n";

    // Emit function units
    for (unsigned fuIdx : pe.fuIndices) {
      const auto &fu = fuDefs[fuIdx];
      os << "    fabric.function_unit @" << fu.name << "(";
      for (size_t j = 0; j < fu.inputTypes.size(); ++j) {
        if (j > 0) os << ", ";
        os << "%arg" << j << ": " << fu.inputTypes[j];
      }
      os << ")";
      if (!fu.outputTypes.empty()) {
        os << " -> (";
        for (size_t j = 0; j < fu.outputTypes.size(); ++j) {
          if (j > 0) os << ", ";
          os << fu.outputTypes[j];
        }
        os << ")";
      }
      // Emit hw params in square brackets
      os << " [latency = " << fu.latency
         << ", interval = " << fu.interval << "]";

      os << " {\n";
      emitFUBody(os, fu, "    ");
      os << "    }\n";
    }

    // PE yield is just a terminator; actual I/O routing is defined by the
    // switch network connectivity, not by SSA values.
    os << "    fabric.yield\n";
    os << "  }\n";
  }

  // Emit SW type definitions (once per unique SW type).
  for (unsigned swIdx : usedSWDefs) {
    const auto &sw = swDefs[swIdx];
    os << "  fabric.spatial_sw @" << sw.name;

    // Emit hw params in square brackets
    bool hasHw = false;
    auto startHw = [&]() {
      if (!hasHw) os << " [";
      else os << ", ";
      hasHw = true;
    };

    if (!sw.connectivity.empty()) {
      startHw();
      unsigned numO = sw.outputWidths.size();
      unsigned numI = sw.inputWidths.size();
      os << "connectivity_table = [";
      for (unsigned o = 0; o < numO; ++o) {
        if (o > 0) os << ", ";
        os << "\"";
        for (unsigned ii = 0; ii < numI; ++ii)
          os << (sw.connectivity[o][ii] ? "1" : "0");
        os << "\"";
      }
      os << "]";
    }

    if (sw.decomposableBits >= 0) {
      startHw();
      os << "decomposable_bits = "
         << sw.decomposableBits << " : i64";
    }

    if (hasHw) os << "]";

    // Function type
    os << " : (";
    for (size_t p = 0; p < sw.inputWidths.size(); ++p) {
      if (p > 0) os << ", ";
      os << bitsType(sw.inputWidths[p]);
    }
    os << ") -> (";
    for (size_t p = 0; p < sw.outputWidths.size(); ++p) {
      if (p > 0) os << ", ";
      os << bitsType(sw.outputWidths[p]);
    }
    os << ")\n";
  }

  // Build connection graph: for each instance, which connections feed
  // into it (organized by destination port).
  // Map: dstInst -> (dstPort -> (srcInst, srcPort))
  std::map<unsigned, std::map<unsigned, std::pair<unsigned, unsigned>>>
      incomingConns;
  for (const auto &conn : connections) {
    incomingConns[conn.dstInst][conn.dstPort] = {conn.srcInst, conn.srcPort};
  }

  // Build scalar-to-instance connection map.
  // Map: (dstInst, dstPort) -> scalarIdx
  std::map<unsigned, std::map<unsigned, unsigned>> scalarInputConns;
  for (const auto &sc : scalarToInstConns) {
    scalarInputConns[sc.dstInst][sc.dstPort] = sc.scalarIdx;
  }

  // Build instance-to-scalar output map.
  // Map: scalarOutputIdx -> (srcInst, srcPort)
  std::map<unsigned, std::pair<unsigned, unsigned>> scalarOutputConns;
  for (const auto &ic : instToScalarConns) {
    scalarOutputConns[ic.scalarOutputIdx] = {ic.srcInst, ic.srcPort};
  }

  // Emit instances using fabric.instance for PEs and SWs,
  // and inline ops for ExtMem and FIFO.
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];

    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &pe = peDefs[inst.defIdx];
      std::string bt = bitsType(pe.bitsWidth);
      unsigned numOut = pe.numOutputs;

      // Emit: %vi:N = fabric.instance @pe_type(%inputs...) {sym_name = "..."}
      //         : (types...) -> (types...)
      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1) os << ":" << numOut;
      }
      os << " = fabric.instance @" << pe.name << "(";

      // Gather input operands from connections
      for (unsigned p = 0; p < pe.numInputs; ++p) {
        if (p > 0) os << ", ";
        auto scIt = scalarInputConns.find(i);
        if (scIt != scalarInputConns.end()) {
          auto scPit = scIt->second.find(p);
          if (scPit != scIt->second.end()) {
            os << "%scalar" << scPit->second;
            continue;
          }
        }
        auto it = incomingConns.find(i);
        if (it != incomingConns.end()) {
          auto pit = it->second.find(p);
          if (pit != it->second.end()) {
            unsigned srcInst = pit->second.first;
            unsigned srcPort = pit->second.second;
            unsigned srcOutCount = getInstanceOutputCount(instances, peDefs, swDefs, extMemDefs, srcInst);
            os << "%v" << srcInst;
            if (srcOutCount > 1) os << "#" << srcPort;
            continue;
          }
        }
        llvm::report_fatal_error(
            "ADGBuilder attempted to emit a spatial_pe with an unconnected "
            "input port; fix the ADG description instead of relying on "
            "implicit self-loops");
      }
      os << ") {sym_name = \"" << inst.name << "\"}";

      // Type signature
      os << " : (";
      for (unsigned p = 0; p < pe.numInputs; ++p) {
        if (p > 0) os << ", ";
        os << bt;
      }
      os << ") -> (";
      for (unsigned p = 0; p < numOut; ++p) {
        if (p > 0) os << ", ";
        os << bt;
      }
      os << ")\n";
      break;
    }
    case InstanceKind::SW: {
      const auto &sw = swDefs[inst.defIdx];
      unsigned numIn = sw.inputWidths.size();
      unsigned numOut = sw.outputWidths.size();

      // Emit as inline spatial_sw with operands from connections.
      // spatial_sw takes SSA inputs and produces SSA outputs.
      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1) os << ":" << numOut;
      }
      os << " = fabric.instance @" << sw.name << "(";

      for (unsigned p = 0; p < numIn; ++p) {
        if (p > 0) os << ", ";
        // Check for scalar input connection first
        auto scIt = scalarInputConns.find(i);
        if (scIt != scalarInputConns.end()) {
          auto scPit = scIt->second.find(p);
          if (scPit != scIt->second.end()) {
            os << "%scalar" << scPit->second;
            continue;
          }
        }
        auto it = incomingConns.find(i);
        if (it != incomingConns.end()) {
          auto pit = it->second.find(p);
          if (pit != it->second.end()) {
            unsigned srcInst = pit->second.first;
            unsigned srcPort = pit->second.second;
            unsigned srcOutCount = getInstanceOutputCount(instances, peDefs, swDefs, extMemDefs, srcInst);
            os << "%v" << srcInst;
            if (srcOutCount > 1) os << "#" << srcPort;
            continue;
          }
        }
        llvm::report_fatal_error(
            "ADGBuilder attempted to emit a spatial_sw with an unconnected "
            "input port; fix the ADG description instead of relying on "
            "implicit self-loops");
      }
      os << ") {sym_name = \"" << inst.name << "\"}";

      os << " : (";
      for (unsigned p = 0; p < numIn; ++p) {
        if (p > 0) os << ", ";
        os << bitsType(sw.inputWidths[p]);
      }
      os << ") -> (";
      for (unsigned p = 0; p < numOut; ++p) {
        if (p > 0) os << ", ";
        os << bitsType(sw.outputWidths[p]);
      }
      os << ")\n";
      break;
    }
    case InstanceKind::ExtMem: {
      const auto &mem = extMemDefs[inst.defIdx];
      unsigned numDataInputs = mem.ldPorts + mem.stPorts * 2;
      unsigned numOut = mem.ldPorts + mem.ldPorts + mem.stPorts;
      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1) os << ":" << numOut;
      }
      os << " = fabric.extmemory @" << inst.name;
      os << " [ldCount = " << mem.ldPorts
         << ", stCount = " << mem.stPorts
         << ", lsqDepth = " << mem.lsqDepth
         << ", memrefType = memref<?xi32>]";
      os << " (";

      bool firstOperand = true;
      auto emitOperandSep = [&]() {
        if (!firstOperand)
          os << ", ";
        firstOperand = false;
      };

      bool emittedMemref = false;
      for (const auto &mc : memrefConnections) {
        if (mc.extMemInstIdx != i)
          continue;
        emitOperandSep();
        os << "%mem" << mc.memrefIdx;
        emittedMemref = true;
        break;
      }
      if (!emittedMemref) {
        emitOperandSep();
        os << "%mem0";
      }

      auto emitConnectedOperand = [&](unsigned portIdx) {
        emitOperandSep();

        auto scIt = scalarInputConns.find(i);
        if (scIt != scalarInputConns.end()) {
          auto scPit = scIt->second.find(portIdx);
          if (scPit != scIt->second.end()) {
            os << "%scalar" << scPit->second;
            return;
          }
        }

        auto it = incomingConns.find(i);
        if (it != incomingConns.end()) {
          auto pit = it->second.find(portIdx);
          if (pit != it->second.end()) {
            unsigned srcInst = pit->second.first;
            unsigned srcPort = pit->second.second;
            unsigned srcOutCount = getInstanceOutputCount(
                instances, peDefs, swDefs, extMemDefs, srcInst);
            os << "%v" << srcInst;
            if (srcOutCount > 1)
              os << "#" << srcPort;
            return;
          }
        }

        os << "%mem0";
      };

      for (unsigned p = 0; p < numDataInputs; ++p)
        emitConnectedOperand(1 + p);
      os << ")";

      auto getExtMemInputWidth = [&](unsigned portIdx) {
        auto scIt = scalarInputConns.find(i);
        if (scIt != scalarInputConns.end()) {
          auto scPit = scIt->second.find(portIdx);
          if (scPit != scIt->second.end())
            return scalarInputs[scPit->second].bitsWidth;
        }
        auto it = incomingConns.find(i);
        if (it != incomingConns.end()) {
          auto pit = it->second.find(portIdx);
          if (pit != it->second.end()) {
            return getInstanceOutputWidth(instances, peDefs, swDefs, fifoDefs,
                                          pit->second.first, pit->second.second);
          }
        }
        return 32u;
      };

      auto getExtMemOutputWidth = [&](unsigned portIdx) {
        for (const auto &conn : connections) {
          if (conn.srcInst != i || conn.srcPort != portIdx)
            continue;
          return getInstanceInputWidth(instances, peDefs, swDefs, fifoDefs,
                                       conn.dstInst, conn.dstPort);
        }
        for (const auto &ic : instToScalarConns) {
          if (ic.srcInst == i && ic.srcPort == portIdx)
            return scalarOutputs[ic.scalarOutputIdx].bitsWidth;
        }
        return 32u;
      };

      os << " : (memref<?xi32>";
      for (unsigned l = 0; l < mem.ldPorts; ++l)
        os << ", " << bitsType(getExtMemInputWidth(1 + l));
      for (unsigned s = 0; s < mem.stPorts; ++s)
        os << ", " << bitsType(getExtMemInputWidth(1 + mem.ldPorts + s));
      for (unsigned s = 0; s < mem.stPorts; ++s)
        os << ", "
           << bitsType(getExtMemInputWidth(1 + mem.ldPorts + mem.stPorts + s));
      os << ") -> (";
      bool first = true;
      for (unsigned l = 0; l < mem.ldPorts; ++l) {
        if (!first) os << ", ";
        first = false;
        os << bitsType(getExtMemOutputWidth(l));
      }
      for (unsigned l = 0; l < mem.ldPorts; ++l) {
        if (!first) os << ", ";
        first = false;
        os << bitsType(getExtMemOutputWidth(mem.ldPorts + l));
      }
      for (unsigned s = 0; s < mem.stPorts; ++s) {
        if (!first) os << ", ";
        first = false;
        os << bitsType(getExtMemOutputWidth(mem.ldPorts * 2 + s));
      }
      os << ")\n";
      break;
    }
    case InstanceKind::FIFO: {
      // FIFOs are not emitted in the ADG MLIR because the fabric.fifo op
      // does not take SSA operands. FIFO connections are represented as
      // direct switch-to-switch connections instead.
      // Skip FIFO instances entirely.
      break;
    }
    }
  }

  // Yield scalar outputs using explicit connections when available.
  os << "  fabric.yield";
  if (!scalarOutputs.empty()) {
    os << " ";
    for (size_t i = 0; i < scalarOutputs.size(); ++i) {
      if (i > 0) os << ", ";
      auto soIt = scalarOutputConns.find(i);
      if (soIt != scalarOutputConns.end()) {
        unsigned srcInst = soIt->second.first;
        unsigned srcPort = soIt->second.second;
        unsigned srcOutCount = getInstanceOutputCount(instances, peDefs, swDefs, extMemDefs, srcInst);
        os << "%v" << srcInst;
        if (srcOutCount > 1) os << "#" << srcPort;
      } else {
        // Fallback: find the first SW instance to use as source.
        unsigned firstSwInst = UINT_MAX;
        for (size_t j = 0; j < instances.size(); ++j) {
          if (instances[j].kind == InstanceKind::SW) {
            firstSwInst = j;
            break;
          }
        }
        if (firstSwInst != UINT_MAX) {
          unsigned srcOutCount = getInstanceOutputCount(instances, peDefs, swDefs, extMemDefs, firstSwInst);
          os << "%v" << firstSwInst;
          if (srcOutCount > 1) os << "#0";
        } else {
          os << "%scalar0";
        }
      }
    }
    os << " : ";
    for (size_t i = 0; i < scalarOutputs.size(); ++i) {
      if (i > 0) os << ", ";
      os << bitsType(scalarOutputs[i].bitsWidth);
    }
  }
  os << "\n";
  os << "}\n";

  // Close top-level module wrapper.
  os << "}\n";

  return os.str();
}

std::string ADGBuilder::Impl::generateVizJson() const {
  struct BoxInfo {
    double centerX = 0.0;
    double centerY = 0.0;
    double width = 0.0;
    double height = 0.0;
    unsigned numInputs = 0;
    unsigned numOutputs = 0;
    bool valid = false;
  };
  struct RoutePt {
    double x = 0.0;
    double y = 0.0;
  };

  auto computePEWidth = [&](const PEDef &peDef) {
    const double approxFuBoxW = 140.0;
    const double approxFuGap = 12.0;
    const double approxPEPadX = 60.0;
    return std::max(200.0,
                    peDef.fuIndices.size() * approxFuBoxW +
                        std::max(0.0, static_cast<double>(peDef.fuIndices.size()) - 1.0) *
                            approxFuGap +
                        approxPEPadX);
  };
  auto computePEHeight = [&](const PEDef &) {
    return 200.0;
  };
  constexpr double kSwitchPortPitch = 24.0;
  constexpr double kSwitchMinSide = 84.0;
  auto buildPortSideCounts = [&](unsigned count, unsigned sideCount) {
    std::array<unsigned, 4> counts = {0, 0, 0, 0};
    for (unsigned idx = 0; idx < count; ++idx)
      counts[idx % sideCount] += 1;
    return counts;
  };
  auto buildPEPortSideCounts = [&](unsigned count) {
    std::array<unsigned, 2> counts = {0, 0};
    for (unsigned idx = 0; idx < count; ++idx)
      counts[idx % 2] += 1;
    return counts;
  };
  auto computeSWSide = [&](const SWDef &swDef) {
    std::array<unsigned, 4> inCounts =
        buildPortSideCounts(swDef.inputWidths.size(), 2);
    std::array<unsigned, 4> outCounts =
        buildPortSideCounts(swDef.outputWidths.size(), 2);
    unsigned maxSideSlots = 0;
    maxSideSlots = std::max(maxSideSlots, inCounts[0]);
    maxSideSlots = std::max(maxSideSlots, inCounts[1]);
    maxSideSlots = std::max(maxSideSlots, outCounts[0]);
    maxSideSlots = std::max(maxSideSlots, outCounts[1]);
    return std::max(kSwitchMinSide,
                    32.0 + (static_cast<double>(std::max(1U, maxSideSlots)) + 1.0) *
                        kSwitchPortPitch);
  };
  auto computeBoxInfo = [&](unsigned instIdx) -> BoxInfo {
    BoxInfo info;
    auto it = vizPlacements.find(instIdx);
    if (it == vizPlacements.end())
      return info;
    info.centerX = it->second.centerX;
    info.centerY = it->second.centerY;
    const auto &inst = instances[instIdx];
    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &peDef = peDefs[inst.defIdx];
      info.width = computePEWidth(peDef);
      info.height = computePEHeight(peDef);
      info.numInputs = peDef.numInputs;
      info.numOutputs = peDef.numOutputs;
      break;
    }
    case InstanceKind::SW: {
      const auto &swDef = swDefs[inst.defIdx];
      double side = computeSWSide(swDef);
      info.width = side;
      info.height = side;
      info.numInputs = swDef.inputWidths.size();
      info.numOutputs = swDef.outputWidths.size();
      break;
    }
    case InstanceKind::ExtMem: {
      const auto &memDef = extMemDefs[inst.defIdx];
      info.width = 170.0;
      info.height = 80.0;
      info.numInputs = 1 + memDef.ldPorts + memDef.stPorts * 2;
      info.numOutputs = memDef.ldPorts + memDef.stPorts + memDef.ldPorts;
      break;
    }
    case InstanceKind::FIFO: {
      info.width = 100.0;
      info.height = 56.0;
      info.numInputs = 1;
      info.numOutputs = 1;
      break;
    }
    }
    info.valid = true;
    return info;
  };
  struct ModuleBounds {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    bool valid = false;
  };
  auto computeModuleBounds = [&]() -> ModuleBounds {
    ModuleBounds bounds;
    bool haveContent = false;
    double actualMinX = 0.0;
    double actualMinY = 0.0;
    double actualMaxX = 0.0;
    double actualMaxY = 0.0;
    for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
      BoxInfo box = computeBoxInfo(static_cast<unsigned>(instIdx));
      if (!box.valid)
        continue;
      double boxMinX = box.centerX - box.width / 2.0;
      double boxMinY = box.centerY - box.height / 2.0;
      double boxMaxX = box.centerX + box.width / 2.0;
      double boxMaxY = box.centerY + box.height / 2.0;
      if (!haveContent) {
        actualMinX = boxMinX;
        actualMinY = boxMinY;
        actualMaxX = boxMaxX;
        actualMaxY = boxMaxY;
        haveContent = true;
      } else {
        actualMinX = std::min(actualMinX, boxMinX);
        actualMinY = std::min(actualMinY, boxMinY);
        actualMaxX = std::max(actualMaxX, boxMaxX);
        actualMaxY = std::max(actualMaxY, boxMaxY);
      }
    }
    if (!haveContent)
      return bounds;
    double contentW = actualMaxX - actualMinX;
    double contentH = actualMaxY - actualMinY;
    double contentArea = contentW * contentH;
    double margin = std::max(60.0, std::round(std::sqrt(contentArea / 4.0)));
    bounds.x = actualMinX - margin;
    bounds.y = actualMinY - margin - 28.0;
    bounds.w = contentW + margin * 2.0;
    bounds.h = contentH + margin * 2.0 + 28.0;
    bounds.valid = true;
    return bounds;
  };
  auto computeInputPortPos = [&](const BoxInfo &box, const InstanceDef &inst,
                                 unsigned portIdx) -> RoutePt {
    RoutePt pt;
    if (inst.kind == InstanceKind::PE) {
      const auto &peDef = peDefs[inst.defIdx];
      std::array<unsigned, 2> sideCounts =
          buildPEPortSideCounts(peDef.numInputs);
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(sideCounts[sideIdx] + 1);
      if (sideIdx == 0) {
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY - box.height / 2.0;
      } else {
        pt.x = box.centerX - box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
      }
    } else if (inst.kind == InstanceKind::SW) {
      const auto &swDef = swDefs[inst.defIdx];
      std::array<unsigned, 4> inCounts =
          buildPortSideCounts(swDef.inputWidths.size(), 2);
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      unsigned slotCount = inCounts[sideIdx];
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(slotCount + 1);
      switch (sideIdx) {
      case 0:
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY - box.height / 2.0;
        break;
      default:
        pt.x = box.centerX - box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
        break;
      }
    } else {
      pt.x = box.centerX - box.width / 2.0;
      pt.y = box.centerY - box.height / 2.0 + 16.0 +
             (box.height - 32.0) * (static_cast<double>(portIdx + 1) /
                                    static_cast<double>(box.numInputs + 1));
    }
    return pt;
  };
  auto computeOutputPortPos = [&](const BoxInfo &box, const InstanceDef &inst,
                                  unsigned portIdx) -> RoutePt {
    RoutePt pt;
    if (inst.kind == InstanceKind::PE) {
      const auto &peDef = peDefs[inst.defIdx];
      std::array<unsigned, 2> sideCounts =
          buildPEPortSideCounts(peDef.numOutputs);
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(sideCounts[sideIdx] + 1);
      if (sideIdx == 0) {
        pt.x = box.centerX + box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
      } else {
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY + box.height / 2.0;
      }
    } else if (inst.kind == InstanceKind::SW) {
      const auto &swDef = swDefs[inst.defIdx];
      std::array<unsigned, 4> outCounts =
          buildPortSideCounts(swDef.outputWidths.size(), 2);
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      unsigned slotCount = outCounts[sideIdx];
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(slotCount + 1);
      switch (sideIdx) {
      case 0:
        pt.x = box.centerX + box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
        break;
      default:
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY + box.height / 2.0;
        break;
      }
    } else {
      pt.x = box.centerX + box.width / 2.0;
      pt.y = box.centerY - box.height / 2.0 + 16.0 +
             (box.height - 32.0) * (static_cast<double>(portIdx + 1) /
                                    static_cast<double>(box.numOutputs + 1));
    }
    return pt;
  };
  ModuleBounds moduleBounds = computeModuleBounds();
  auto computeModuleInputPortPos = [&](unsigned portIdx) -> RoutePt {
    RoutePt pt;
    pt.x = moduleBounds.x + moduleBounds.w *
           (static_cast<double>(portIdx + 1) /
            static_cast<double>(scalarInputs.size() + 1));
    pt.y = moduleBounds.y;
    return pt;
  };
  auto computeModuleOutputPortPos = [&](unsigned portIdx) -> RoutePt {
    RoutePt pt;
    pt.x = moduleBounds.x + moduleBounds.w *
           (static_cast<double>(portIdx + 1) /
            static_cast<double>(scalarOutputs.size() + 1));
    pt.y = moduleBounds.y + moduleBounds.h;
    return pt;
  };
  auto routeModuleInputConnection = [&](unsigned scalarIdx, unsigned dstInstIdx,
                                        unsigned dstPortIdx)
      -> std::vector<RoutePt> {
    if (!moduleBounds.valid)
      return {};
    BoxInfo dstBox = computeBoxInfo(dstInstIdx);
    if (!dstBox.valid)
      return {};
    const auto &dstInst = instances[dstInstIdx];
    RoutePt srcPort = computeModuleInputPortPos(scalarIdx);
    RoutePt dstPort = computeInputPortPos(dstBox, dstInst, dstPortIdx);
    const int signedLane = static_cast<int>(scalarIdx % 5) - 2;
    const double laneOffset = static_cast<double>(signedLane) * 7.0;
    const double entryY = moduleBounds.y + 42.0 + std::abs(laneOffset);
    const double dstApproachX = dstPort.x - (24.0 + std::abs(laneOffset));
    std::vector<RoutePt> pts;
    pts.push_back({srcPort.x, entryY});
    if (std::abs(srcPort.x - dstApproachX) > 0.5)
      pts.push_back({dstApproachX, entryY});
    if (std::abs(entryY - dstPort.y) > 0.5)
      pts.push_back({dstApproachX, dstPort.y});
    return pts;
  };
  auto routeModuleOutputConnection = [&](unsigned srcInstIdx, unsigned srcPortIdx,
                                         unsigned scalarOutIdx)
      -> std::vector<RoutePt> {
    if (!moduleBounds.valid)
      return {};
    BoxInfo srcBox = computeBoxInfo(srcInstIdx);
    if (!srcBox.valid)
      return {};
    const auto &srcInst = instances[srcInstIdx];
    RoutePt srcPort = computeOutputPortPos(srcBox, srcInst, srcPortIdx);
    RoutePt dstPort = computeModuleOutputPortPos(scalarOutIdx);
    const int signedLane = static_cast<int>(scalarOutIdx % 5) - 2;
    const double laneOffset = static_cast<double>(signedLane) * 7.0;
    const double exitX = srcPort.x + (24.0 + std::abs(laneOffset));
    const double corridorY = moduleBounds.y + moduleBounds.h - 42.0 -
                             std::abs(laneOffset);
    std::vector<RoutePt> pts;
    pts.push_back({exitX, srcPort.y});
    if (std::abs(srcPort.y - corridorY) > 0.5)
      pts.push_back({exitX, corridorY});
    if (std::abs(exitX - dstPort.x) > 0.5)
      pts.push_back({dstPort.x, corridorY});
    return pts;
  };
  auto routeConnection = [&](const Connection &conn,
                             unsigned routeOrdinal) -> std::vector<RoutePt> {
    BoxInfo srcBox = computeBoxInfo(conn.srcInst);
    BoxInfo dstBox = computeBoxInfo(conn.dstInst);
    if (!srcBox.valid || !dstBox.valid)
      return {};

    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    RoutePt srcPort = computeOutputPortPos(srcBox, srcInst, conn.srcPort);
    RoutePt dstPort = computeInputPortPos(dstBox, dstInst, conn.dstPort);

    const int signedLane = static_cast<int>(routeOrdinal % 5) - 2;
    const double laneOffset = static_cast<double>(signedLane) * 7.0;
    const double margin = 22.0 + std::abs(laneOffset) * 0.5;
    const double srcRight = srcBox.centerX + srcBox.width / 2.0;
    const double dstLeft = dstBox.centerX - dstBox.width / 2.0;
    const double srcTop = srcBox.centerY - srcBox.height / 2.0;
    const double srcBottom = srcBox.centerY + srcBox.height / 2.0;
    const double dstTop = dstBox.centerY - dstBox.height / 2.0;
    const double dstBottom = dstBox.centerY + dstBox.height / 2.0;

    std::vector<RoutePt> pts;
    if (srcBox.centerX + 1.0 < dstBox.centerX) {
      double corridorX = (srcRight + dstLeft) / 2.0 + laneOffset;
      pts.push_back({corridorX, srcPort.y});
      if (std::abs(srcPort.y - dstPort.y) > 0.5)
        pts.push_back({corridorX, dstPort.y});
      return pts;
    }

    double srcExitX = srcRight + margin;
    double dstEntryX = dstLeft - margin;
    bool routeAbove = srcBox.centerY <= dstBox.centerY;
    double corridorY = routeAbove
      ? std::min(srcTop, dstTop) - margin - std::abs(laneOffset)
      : std::max(srcBottom, dstBottom) + margin + std::abs(laneOffset);
    pts.push_back({srcExitX, srcPort.y});
    pts.push_back({srcExitX, corridorY});
    pts.push_back({dstEntryX, corridorY});
    pts.push_back({dstEntryX, dstPort.y});
    return pts;
  };
  std::map<unsigned, std::map<unsigned, std::pair<unsigned, unsigned>>>
      incomingConns;
  for (const auto &conn : connections)
    incomingConns[conn.dstInst][conn.dstPort] = {conn.srcInst, conn.srcPort};

  std::map<unsigned, std::map<unsigned, unsigned>> scalarInputConns;
  for (const auto &sc : scalarToInstConns)
    scalarInputConns[sc.dstInst][sc.dstPort] = sc.scalarIdx;

  std::ostringstream os;
  os << "{\n"
     << "  \"version\": 1,\n"
     << "  \"components\": [\n";

  bool first = true;
  for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
    auto it = vizPlacements.find(static_cast<unsigned>(instIdx));
    if (it == vizPlacements.end())
      continue;

    if (!first)
      os << ",\n";
    first = false;

    const auto &inst = instances[instIdx];
    const auto &placement = it->second;
    const char *kindName = "instance";
    switch (inst.kind) {
    case InstanceKind::PE:
      kindName = "spatial_pe";
      break;
    case InstanceKind::SW:
      kindName = "spatial_sw";
      break;
    case InstanceKind::ExtMem:
      kindName = "extmemory";
      break;
    case InstanceKind::FIFO:
      kindName = "fifo";
      break;
    }

    os << "    {\"name\": \"" << inst.name << "\""
       << ", \"kind\": \"" << kindName << "\""
       << ", \"center_x\": " << placement.centerX
       << ", \"center_y\": " << placement.centerY;
    if (placement.gridRow >= 0)
      os << ", \"grid_row\": " << placement.gridRow;
    if (placement.gridCol >= 0)
      os << ", \"grid_col\": " << placement.gridCol;
    os << "}";
  }

  os << "\n  ],\n"
     << "  \"routes\": [\n";

  bool firstRoute = true;
  std::map<std::pair<unsigned, unsigned>, unsigned> nextPairOrdinal;
  auto emitRouteRecord = [&](llvm::StringRef fromName, unsigned fromPort,
                             llvm::StringRef toName, unsigned toPort,
                             const std::vector<RoutePt> &pts) {
    if (!firstRoute)
      os << ",\n";
    firstRoute = false;
    os << "    {\"from\": \"" << fromName.str() << "\""
       << ", \"from_port\": " << fromPort
       << ", \"to\": \"" << toName.str() << "\""
       << ", \"to_port\": " << toPort
       << ", \"points\": [";
    for (size_t ptIdx = 0; ptIdx < pts.size(); ++ptIdx) {
      if (ptIdx > 0)
        os << ", ";
      os << "{\"x\": " << pts[ptIdx].x
         << ", \"y\": " << pts[ptIdx].y << "}";
    }
    os << "]}";
  };
  for (size_t connIdx = 0; connIdx < connections.size(); ++connIdx) {
    const auto &conn = connections[connIdx];
    if (vizPlacements.find(conn.srcInst) == vizPlacements.end() ||
        vizPlacements.find(conn.dstInst) == vizPlacements.end())
      continue;
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    auto pairKey = std::make_pair(std::min(conn.srcInst, conn.dstInst),
                                  std::max(conn.srcInst, conn.dstInst));
    unsigned pairOrdinal = nextPairOrdinal[pairKey]++;
    std::vector<RoutePt> pts = routeConnection(conn, pairOrdinal);

    emitRouteRecord(srcInst.name, conn.srcPort, dstInst.name, conn.dstPort, pts);
  }
  for (const auto &sc : scalarToInstConns) {
    if (vizPlacements.find(sc.dstInst) == vizPlacements.end())
      continue;
    std::vector<RoutePt> pts =
        routeModuleInputConnection(sc.scalarIdx, sc.dstInst, sc.dstPort);
    emitRouteRecord("module_in", sc.scalarIdx, instances[sc.dstInst].name,
                    sc.dstPort, pts);
  }
  for (const auto &ic : instToScalarConns) {
    if (vizPlacements.find(ic.srcInst) == vizPlacements.end())
      continue;
    std::vector<RoutePt> pts =
        routeModuleOutputConnection(ic.srcInst, ic.srcPort, ic.scalarOutputIdx);
    emitRouteRecord(instances[ic.srcInst].name, ic.srcPort, "module_out",
                    ic.scalarOutputIdx, pts);
  }
  os << "\n  ]\n"
     << "}\n";
  return os.str();
}

//===----------------------------------------------------------------------===//
// ADGBuilder Public API
//===----------------------------------------------------------------------===//

ADGBuilder::ADGBuilder(const std::string &moduleName)
    : impl_(std::make_unique<Impl>()) {
  impl_->moduleName = moduleName;
}

ADGBuilder::~ADGBuilder() = default;

FUHandle ADGBuilder::defineFU(const std::string &name,
                              const std::vector<std::string> &inputTypes,
                              const std::vector<std::string> &outputTypes,
                              const std::vector<std::string> &ops,
                              unsigned latency, unsigned interval) {
  unsigned id = impl_->fuDefs.size();
  impl_->fuDefs.push_back({name, inputTypes, outputTypes, ops,
                            latency, interval});
  return {id};
}

PEHandle ADGBuilder::defineSpatialPE(const std::string &name,
                                     unsigned numInputs, unsigned numOutputs,
                                     unsigned bitsWidth,
                                     const std::vector<FUHandle> &fus) {
  unsigned id = impl_->peDefs.size();
  PEDef pe;
  pe.name = name;
  pe.numInputs = numInputs;
  pe.numOutputs = numOutputs;
  pe.bitsWidth = bitsWidth;
  for (const auto &fu : fus)
    pe.fuIndices.push_back(fu.id);
  impl_->peDefs.push_back(std::move(pe));
  return {id};
}

SWHandle ADGBuilder::defineSpatialSW(const std::string &name,
                                     const std::vector<unsigned> &inputWidths,
                                     const std::vector<unsigned> &outputWidths,
                                     const std::vector<std::vector<bool>> &conn,
                                     int decomposableBits) {
  unsigned id = impl_->swDefs.size();
  impl_->swDefs.push_back({name, inputWidths, outputWidths, conn,
                            decomposableBits});
  return {id};
}

ExtMemHandle ADGBuilder::defineExtMemory(const std::string &name,
                                         unsigned ldPorts, unsigned stPorts,
                                         unsigned lsqDepth) {
  unsigned id = impl_->extMemDefs.size();
  impl_->extMemDefs.push_back({name, ldPorts, stPorts, lsqDepth});
  return {id};
}

FIFOHandle ADGBuilder::defineFIFO(const std::string &name,
                                  unsigned depth, unsigned bitsWidth) {
  unsigned id = impl_->fifoDefs.size();
  impl_->fifoDefs.push_back({name, depth, bitsWidth});
  return {id};
}

InstanceHandle ADGBuilder::instantiatePE(PEHandle pe,
                                         const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::PE, pe.id, instanceName});
  return {id};
}

InstanceHandle ADGBuilder::instantiateSW(SWHandle sw,
                                         const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::SW, sw.id, instanceName});
  return {id};
}

InstanceHandle ADGBuilder::instantiateExtMem(ExtMemHandle mem,
                                             const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::ExtMem, mem.id, instanceName});
  return {id};
}

InstanceHandle ADGBuilder::instantiateFIFO(FIFOHandle fifo,
                                           const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::FIFO, fifo.id, instanceName});
  return {id};
}

void ADGBuilder::connect(InstanceHandle src, unsigned srcPort,
                         InstanceHandle dst, unsigned dstPort) {
  impl_->connections.push_back({src.id, srcPort, dst.id, dstPort});
}

MeshResult ADGBuilder::buildMesh(unsigned rows, unsigned cols,
                                 PEHandle pe, SWHandle sw) {
  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows, std::vector<InstanceHandle>(cols));

  // Instantiate switches first (they come before PEs since PEs connect
  // to their local switch). In the graph region, order does not matter
  // for SSA dominance.
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string swName = "sw_" + std::to_string(r) + "_" + std::to_string(c);
      result.swGrid[r][c] = instantiateSW(sw, swName);
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string peName = "pe_" + std::to_string(r) + "_" + std::to_string(c);
      result.peGrid[r][c] = instantiatePE(pe, peName);
    }
  }

  // Wire PE <-> local SW
  const auto &peDef = impl_->peDefs[pe.id];
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      // PE outputs -> SW inputs (ports 0..numOutputs-1)
      for (unsigned p = 0; p < peDef.numOutputs; ++p)
        connect(result.peGrid[r][c], p, result.swGrid[r][c], p);
      // SW outputs -> PE inputs (ports 0..numInputs-1)
      for (unsigned p = 0; p < peDef.numInputs; ++p)
        connect(result.swGrid[r][c], p, result.peGrid[r][c], p);

      // Inter-switch NSEW connections — TORUS topology
      // Boundary wraps around to opposite edge (no dangling ports).
      unsigned peOut = peDef.numOutputs;
      unsigned peIn = peDef.numInputs;
      // North: wrap to row (rows-1) if at row 0
      unsigned northR = (r > 0) ? r - 1 : rows - 1;
      connect(result.swGrid[r][c], peIn + 0,
              result.swGrid[northR][c], peOut + 1);
      // South: wrap to row 0 if at row (rows-1)
      unsigned southR = (r + 1 < rows) ? r + 1 : 0;
      connect(result.swGrid[r][c], peIn + 1,
              result.swGrid[southR][c], peOut + 0);
      // East: wrap to col 0 if at col (cols-1)
      unsigned eastC = (c + 1 < cols) ? c + 1 : 0;
      connect(result.swGrid[r][c], peIn + 2,
              result.swGrid[r][eastC], peOut + 3);
      // West: wrap to col (cols-1) if at col 0
      unsigned westC = (c > 0) ? c - 1 : cols - 1;
      connect(result.swGrid[r][c], peIn + 3,
              result.swGrid[r][westC], peOut + 2);
    }
  }

  return result;
}

MeshResult ADGBuilder::buildChessMesh(unsigned rows, unsigned cols, PEHandle pe,
                                      int decomposableBits,
                                      unsigned topLeftExtraInputs,
                                      unsigned bottomRightExtraOutputs) {
  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows + 1, std::vector<InstanceHandle>(cols + 1));

  const auto &peDef = impl_->peDefs[pe.id];
  assert(peDef.numInputs >= 4 &&
         "buildChessMesh expects at least four PE input ports");
  assert(peDef.numOutputs >= 4 &&
         "buildChessMesh expects at least four PE output ports");

  std::map<std::pair<unsigned, unsigned>, SWHandle> switchTemplateCache;
  auto makeSwitchTemplate = [&](unsigned numInputs,
                                unsigned numOutputs) -> SWHandle {
    auto key = std::make_pair(numInputs, numOutputs);
    auto it = switchTemplateCache.find(key);
    if (it != switchTemplateCache.end())
      return it->second;

    std::vector<unsigned> inputWidths(numInputs, peDef.bitsWidth);
    std::vector<unsigned> outputWidths(numOutputs, peDef.bitsWidth);
    std::vector<std::vector<bool>> fullCrossbar(
        numOutputs, std::vector<bool>(numInputs, true));
    std::string name = "__chess_sw_" + std::to_string(numInputs) + "x" +
                       std::to_string(numOutputs) + "_" +
                       std::to_string(impl_->swDefs.size());
    auto handle = defineSpatialSW(name, inputWidths, outputWidths, fullCrossbar,
                                  decomposableBits);
    switchTemplateCache[key] = handle;
    return handle;
  };

  auto switchDegree = [&](unsigned sr, unsigned sc) -> unsigned {
    unsigned degree = 0;
    if (sr > 0)
      degree++;
    if (sr + 1 < rows + 1)
      degree++;
    if (sc > 0)
      degree++;
    if (sc + 1 < cols + 1)
      degree++;

    if (sr > 0 && sc > 0)
      degree++;
    if (sr > 0 && sc < cols)
      degree++;
    if (sr < rows && sc > 0)
      degree++;
    if (sr < rows && sc < cols)
      degree++;
    return degree;
  };

  const unsigned maxSwitchDegree = 8;
  const double maxSwitchBox = std::max(80.0, maxSwitchDegree * 30.0 + 30.0);
  const double approxFuBoxW = 132.0;
  const double approxFuGap = 12.0;
  const double approxPEPadX = 40.0;
  const double approxPEBoxW =
      std::max(200.0,
               peDef.fuIndices.size() * approxFuBoxW +
                   std::max(0.0, static_cast<double>(peDef.fuIndices.size()) - 1.0) *
                       approxFuGap +
                   approxPEPadX);
  const double approxPEBoxH = 200.0;
  const double componentGap = 40.0;
  const double switchStepX =
      std::max(520.0, maxSwitchBox + approxPEBoxW + componentGap);
  const double switchStepY =
      std::max(520.0, maxSwitchBox + approxPEBoxH + componentGap);
  const double originX = maxSwitchBox / 2.0 + 80.0;
  const double originY = maxSwitchBox / 2.0 + 80.0;

  for (unsigned sr = 0; sr <= rows; ++sr) {
    for (unsigned sc = 0; sc <= cols; ++sc) {
      unsigned degree = switchDegree(sr, sc);
      unsigned numInputs = degree;
      unsigned numOutputs = degree;
      if (sr == 0 && sc == 0)
        numInputs += topLeftExtraInputs;
      if (sr == rows && sc == cols)
        numOutputs += bottomRightExtraOutputs;
      SWHandle swHandle = makeSwitchTemplate(numInputs, numOutputs);
      std::string swName = "sw_" + std::to_string(sr) + "_" +
                           std::to_string(sc);
      auto inst = instantiateSW(swHandle, swName);
      result.swGrid[sr][sc] = inst;
      setInstanceVizPosition(inst, originX + sc * switchStepX,
                             originY + sr * switchStepY, sr, sc);
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string peName = "pe_" + std::to_string(r) + "_" +
                           std::to_string(c);
      auto inst = instantiatePE(pe, peName);
      result.peGrid[r][c] = inst;
      setInstanceVizPosition(inst, originX + (c + 0.5) * switchStepX,
                             originY + (r + 0.5) * switchStepY, r, c);
    }
  }

  std::map<std::pair<unsigned, unsigned>, unsigned> nextSwitchSlot;
  auto allocSwitchSlot = [&](unsigned sr, unsigned sc) -> unsigned {
    auto key = std::make_pair(sr, sc);
    unsigned slot = nextSwitchSlot[key];
    nextSwitchSlot[key] = slot + 1;
    return slot;
  };

  auto connectSwitchBidirectional = [&](unsigned sr0, unsigned sc0,
                                        unsigned sr1, unsigned sc1) {
    unsigned slot0 = allocSwitchSlot(sr0, sc0);
    unsigned slot1 = allocSwitchSlot(sr1, sc1);
    auto sw0 = result.swGrid[sr0][sc0];
    auto sw1 = result.swGrid[sr1][sc1];
    connect(sw0, slot0, sw1, slot1);
    connect(sw1, slot1, sw0, slot0);
  };

  for (unsigned sr = 0; sr <= rows; ++sr) {
    for (unsigned sc = 0; sc + 1 <= cols; ++sc) {
      if (sc + 1 <= cols)
        connectSwitchBidirectional(sr, sc, sr, sc + 1);
    }
  }
  for (unsigned sr = 0; sr + 1 <= rows; ++sr) {
    for (unsigned sc = 0; sc <= cols; ++sc) {
      if (sr + 1 <= rows)
        connectSwitchBidirectional(sr, sc, sr + 1, sc);
    }
  }

  auto connectPEToSwitch = [&](unsigned r, unsigned c, unsigned peSlot,
                               unsigned sr, unsigned sc) {
    unsigned swSlot = allocSwitchSlot(sr, sc);
    auto peInst = result.peGrid[r][c];
    auto swInst = result.swGrid[sr][sc];
    connect(peInst, peSlot, swInst, swSlot);
    connect(swInst, swSlot, peInst, peSlot);
  };

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      connectPEToSwitch(r, c, 0, r, c);
      connectPEToSwitch(r, c, 1, r, c + 1);
      connectPEToSwitch(r, c, 2, r + 1, c);
      connectPEToSwitch(r, c, 3, r + 1, c + 1);
    }
  }

  return result;
}

unsigned ADGBuilder::addMemrefInput(const std::string &name,
                                    const std::string &memrefTypeStr) {
  unsigned idx = impl_->memrefInputs.size();
  impl_->memrefInputs.push_back({name, memrefTypeStr});
  return idx;
}

void ADGBuilder::connectMemrefToExtMem(unsigned memrefIdx,
                                       InstanceHandle extMemInst) {
  impl_->memrefConnections.push_back({memrefIdx, extMemInst.id});
}

unsigned ADGBuilder::addScalarInput(const std::string &name,
                                    unsigned bitsWidth) {
  unsigned idx = impl_->scalarInputs.size();
  impl_->scalarInputs.push_back({name, bitsWidth});
  return idx;
}

unsigned ADGBuilder::addScalarOutput(const std::string &name,
                                     unsigned bitsWidth) {
  unsigned idx = impl_->scalarOutputs.size();
  impl_->scalarOutputs.push_back({name, bitsWidth});
  return idx;
}

void ADGBuilder::connectScalarInputToInstance(unsigned scalarIdx,
                                               InstanceHandle dst,
                                               unsigned dstPort) {
  impl_->scalarToInstConns.push_back({scalarIdx, dst.id, dstPort});
}

void ADGBuilder::connectInstanceToScalarOutput(InstanceHandle src,
                                              unsigned srcPort,
                                              unsigned scalarOutputIdx) {
  impl_->instToScalarConns.push_back({src.id, srcPort, scalarOutputIdx});
}

void ADGBuilder::associateExtMemWithSW(InstanceHandle extMem,
                                       InstanceHandle sw,
                                       unsigned swInputPortBase,
                                       unsigned swOutputPortBase) {
  // Create real SSA connections: ExtMem outputs -> SW input ports.
  // ExtMem outputs: [ldData_0..L, stDone_0..S, ldDone_0..L]
  // These feed into SW input ports starting at swInputPortBase.
  // This direction creates visible SSA edges: the SW instance will
  // reference %v<extmem>#<port> as operands.
  const auto &extMemInst = impl_->instances[extMem.id];
  assert(extMemInst.kind == InstanceKind::ExtMem);
  const auto &mem = impl_->extMemDefs[extMemInst.defIdx];

  unsigned numExtMemOutputs = mem.ldPorts + mem.stPorts + mem.ldPorts;

  for (unsigned p = 0; p < numExtMemOutputs; ++p) {
    impl_->connections.push_back(
        {extMem.id, p, sw.id, swInputPortBase + p});
  }

  // Reverse direction: SW output ports -> ExtMem input ports.
  // These are emitted as real SSA operands on the inline fabric.extmemory op.
  unsigned numExtMemDataInputs = mem.stPorts * 2 + mem.ldPorts;
  for (unsigned p = 0; p < numExtMemDataInputs; ++p) {
    impl_->connections.push_back(
        {sw.id, swOutputPortBase + p, extMem.id, 1 + p});
  }
}

void ADGBuilder::setInstanceVizPosition(InstanceHandle inst, double centerX,
                                        double centerY, int gridRow,
                                        int gridCol) {
  impl_->vizPlacements[inst.id] = {centerX, centerY, gridRow, gridCol};
}

bool ADGBuilder::Impl::validate(std::string &errMsg) const {
  bool valid = true;
  std::ostringstream errs;

  // Build connection maps
  std::map<unsigned, std::set<unsigned>> connectedInputs;
  std::map<unsigned, std::set<unsigned>> connectedOutputs;
  for (const auto &conn : connections) {
    connectedInputs[conn.dstInst].insert(conn.dstPort);
    connectedOutputs[conn.srcInst].insert(conn.srcPort);
  }

  // Track scalar connections
  for (const auto &sc : scalarToInstConns) {
    connectedInputs[sc.dstInst].insert(sc.dstPort);
  }
  for (const auto &ic : instToScalarConns) {
    connectedOutputs[ic.srcInst].insert(ic.srcPort);
  }
  for (const auto &mc : memrefConnections) {
    connectedInputs[mc.extMemInstIdx].insert(0);
  }

  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = 0, numOut = 0;

    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &pe = peDefs[inst.defIdx];
      numIn = pe.numInputs;
      numOut = pe.numOutputs;
      break;
    }
    case InstanceKind::SW: {
      const auto &sw = swDefs[inst.defIdx];
      numIn = sw.inputWidths.size();
      numOut = sw.outputWidths.size();
      break;
    }
    case InstanceKind::ExtMem: {
      const auto &mem = extMemDefs[inst.defIdx];
      numIn = 1; // memref
      for (unsigned s = 0; s < mem.stPorts; ++s) numIn += 2;
      for (unsigned l = 0; l < mem.ldPorts; ++l) numIn += 1;
      numOut = mem.ldPorts + mem.stPorts + mem.ldPorts;
      // ExtMem port 0 (memref) is connected via memrefConnections, skip
      break;
    }
    case InstanceKind::FIFO:
      numIn = 1;
      numOut = 1;
      break;
    }

    // Check inputs
    for (unsigned p = 0; p < numIn; ++p) {
      if (connectedInputs[i].find(p) == connectedInputs[i].end()) {
        errs << "  dangling input: " << inst.name << " port " << p << "\n";
        valid = false;
      }
    }

    // Check outputs
    for (unsigned p = 0; p < numOut; ++p) {
      if (connectedOutputs[i].find(p) == connectedOutputs[i].end()) {
        errs << "  dangling output: " << inst.name << " port " << p << "\n";
        valid = false;
      }
    }
  }

  errMsg = errs.str();
  return valid;
}

void ADGBuilder::exportMLIR(const std::string &path) {
  // Run validation
  std::string valErr;
  if (!impl_->validate(valErr)) {
    llvm::report_fatal_error(llvm::Twine("ADGBuilder validation failed:\n") +
                             valErr);
  }

  std::string vizFileName;
  std::string vizJsonText;
  if (!impl_->vizPlacements.empty()) {
    llvm::SmallString<256> vizPath(path);
    llvm::sys::path::replace_extension(vizPath, "viz.json");
    vizFileName = std::string(llvm::sys::path::filename(vizPath));
    vizJsonText = impl_->generateVizJson();
  }

  std::string mlirText = impl_->generateMLIR(vizFileName);

  mlir::MLIRContext context;
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    llvm::errs() << "\n";
    return mlir::success();
  });

  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<fcc::dataflow::DataflowDialect>();
  context.getOrLoadDialect<fcc::fabric::FabricDialect>();
  context.getOrLoadDialect<circt::handshake::HandshakeDialect>();

  // Verify the generated MLIR by parsing + verifying.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse generated MLIR\n";
    llvm::errs() << "--- generated MLIR ---\n" << mlirText << "---\n";
    std::exit(1);
  }

  if (failed(mlir::verify(*module))) {
    llvm::errs() << "error: generated MLIR failed verification\n";
    llvm::errs() << "--- generated MLIR ---\n" << mlirText << "---\n";
    std::exit(1);
  }

  if (failed(fcc::verifyFabricModule(*module))) {
    llvm::errs() << "error: generated ADG failed fabric.module verification\n";
    llvm::errs() << "--- generated MLIR ---\n" << mlirText << "---\n";
    std::exit(1);
  }

  // Write the raw generated text directly. The round-trip printer reformats
  // integer attributes with `: i64` suffixes that cause parse failures in
  // contexts where additional dialects (e.g., LLVM) are loaded.
  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot write output file: " << path << "\n";
    llvm::errs() << ec.message() << "\n";
    std::exit(1);
  }

  output << mlirText;
  output.flush();

  if (!vizJsonText.empty()) {
    llvm::SmallString<256> vizPath(path);
    llvm::sys::path::replace_extension(vizPath, "viz.json");
    std::error_code vizEc;
    llvm::raw_fd_ostream vizOut(vizPath, vizEc, llvm::sys::fs::OF_Text);
    if (vizEc) {
      llvm::errs() << "error: cannot write viz sidecar: "
                   << vizPath << "\n";
      llvm::errs() << vizEc.message() << "\n";
      std::exit(1);
    }
    vizOut << vizJsonText;
    vizOut.flush();
  }
}

} // namespace adg
} // namespace fcc
