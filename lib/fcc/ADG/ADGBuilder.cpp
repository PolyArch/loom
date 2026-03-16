//===-- ADGBuilder.cpp - ADG Builder implementation ---------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

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
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

  struct ExtMemSWAssoc {
    unsigned extMemInstIdx;
    unsigned swInstIdx;
  };
  std::vector<ExtMemSWAssoc> extMemSWAssociations;

  std::string generateMLIR() const;
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

  // Multi-op FU: generate an internal DAG with static_mux output selection.
  // For FUs with 2+ ops, chain them and add a fabric.static_mux at the output.
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

    // Generate fabric.static_mux to select between the two results.
    // Format: fabric.static_mux %d, %e {sel = 0 : i64} : T, T -> T
    os << indent << "  %g = fabric.static_mux"
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

//===----------------------------------------------------------------------===//
// Full MLIR generation (instance-based)
//===----------------------------------------------------------------------===//

std::string ADGBuilder::Impl::generateMLIR() const {
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
  os << ") {\n";

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
        // Self-loop unconnected input from own output.
        // Graph region allows this circular reference.
        unsigned selfPort = p < numOut ? p : 0;
        os << "%v" << i;
        if (numOut > 1) os << "#" << selfPort;
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
        // Self-loop unconnected input: feed from own output port.
        // Graph region allows this circular reference.
        unsigned selfPort = p < numOut ? p : 0;
        os << "%v" << i;
        if (numOut > 1) os << "#" << selfPort;
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
      unsigned numOut = mem.ldPorts + mem.stPorts + mem.ldPorts;
      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1) os << ":" << numOut;
      }
      os << " = fabric.extmemory @" << inst.name;
      os << " [ldCount = " << mem.ldPorts
         << ", stCount = " << mem.stPorts
         << ", lsqDepth = " << mem.lsqDepth
         << ", memref_type = memref<?xi32>]";
      // Emit attributes dict with memref_arg_index and connected_sw.
      {
        bool hasAttrs = false;
        auto startAttrs = [&]() {
          if (!hasAttrs) os << " attributes {";
          else os << ", ";
          hasAttrs = true;
        };

        for (const auto &mc : memrefConnections) {
          if (mc.extMemInstIdx == i) {
            startAttrs();
            os << "memref_arg_index = " << mc.memrefIdx << " : i32";
            break;
          }
        }

        // connected_sw attribute records which switches this extmem is
        // adjacent to. ExtMem outputs also feed into these switches as
        // real SSA operands (visible in the SW instance operand lists).
        std::set<unsigned> connectedSWInsts;
        for (const auto &assoc : extMemSWAssociations) {
          if (assoc.extMemInstIdx == i)
            connectedSWInsts.insert(assoc.swInstIdx);
        }
        if (!connectedSWInsts.empty()) {
          startAttrs();
          os << "connected_sw = [";
          bool first2 = true;
          for (unsigned swInst : connectedSWInsts) {
            if (!first2) os << ", ";
            first2 = false;
            os << "\"" << instances[swInst].name << "\"";
          }
          os << "]";
        }

        if (hasAttrs) os << "}";
      }

      // Function type: all data ports use uniform 32-bit width for
      // compatibility with the switch network port widths.
      os << " : (memref<?xi32>";
      for (unsigned s = 0; s < mem.stPorts; ++s)
        os << ", !fabric.bits<32>, !fabric.bits<32>";
      for (unsigned l = 0; l < mem.ldPorts; ++l)
        os << ", !fabric.bits<32>";
      os << ") -> (";
      bool first = true;
      for (unsigned l = 0; l < mem.ldPorts; ++l) {
        if (!first) os << ", "; first = false;
        os << "!fabric.bits<32>";
      }
      for (unsigned s = 0; s < mem.stPorts; ++s) {
        if (!first) os << ", "; first = false;
        os << "!fabric.bits<32>";
      }
      for (unsigned l = 0; l < mem.ldPorts; ++l) {
        if (!first) os << ", "; first = false;
        os << "!fabric.bits<32>";
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

      // Inter-switch NSEW connections
      // SW output ports [numInputs..numInputs+3] are NSEW outputs
      // SW input ports [numOutputs..numOutputs+3] are NSEW inputs
      unsigned peOut = peDef.numOutputs;
      unsigned peIn = peDef.numInputs;
      // North: this SW -> north neighbor SW
      if (r > 0)
        connect(result.swGrid[r][c], peIn + 0,
                result.swGrid[r-1][c], peOut + 1);
      // South
      if (r + 1 < rows)
        connect(result.swGrid[r][c], peIn + 1,
                result.swGrid[r+1][c], peOut + 0);
      // East
      if (c + 1 < cols)
        connect(result.swGrid[r][c], peIn + 2,
                result.swGrid[r][c+1], peOut + 3);
      // West
      if (c > 0)
        connect(result.swGrid[r][c], peIn + 3,
                result.swGrid[r][c-1], peOut + 2);
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
  // Store a structural association for metadata (connected_sw attribute).
  impl_->extMemSWAssociations.push_back({extMem.id, sw.id});

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
  // Recorded in the connection graph for analysis. The extmemory MLIR
  // op does not take SSA data operands (parser limitation), so this
  // direction is also captured by the connected_sw attribute metadata.
  unsigned numExtMemDataInputs = mem.stPorts * 2 + mem.ldPorts;
  for (unsigned p = 0; p < numExtMemDataInputs; ++p) {
    impl_->connections.push_back(
        {sw.id, swOutputPortBase + p, extMem.id, 1 + p});
  }
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

    // Check inputs (for ExtMem, skip port 0 which is memref)
    unsigned startPort = (inst.kind == InstanceKind::ExtMem) ? 1 : 0;
    for (unsigned p = startPort; p < numIn; ++p) {
      if (connectedInputs[i].find(p) == connectedInputs[i].end()) {
        // Boundary PE/SW ports use self-loops (allowed), but FIFO and
        // ExtMem data ports should be connected.
        if (inst.kind == InstanceKind::FIFO) {
          errs << "  dangling input: " << inst.name << " port " << p << "\n";
          valid = false;
        }
      }
    }

    // Check outputs: at least one consumer for FIFO/ExtMem
    if (inst.kind == InstanceKind::FIFO) {
      bool hasConsumer = false;
      for (unsigned p = 0; p < numOut; ++p) {
        if (connectedOutputs[i].find(p) != connectedOutputs[i].end())
          hasConsumer = true;
      }
      if (!hasConsumer) {
        errs << "  dangling output: " << inst.name
             << " has no consumers\n";
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
    llvm::errs() << "warning: ADG validation found issues:\n" << valErr;
  }

  std::string mlirText = impl_->generateMLIR();

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
}

} // namespace adg
} // namespace fcc
