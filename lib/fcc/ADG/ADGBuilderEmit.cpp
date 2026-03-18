//===-- ADGBuilderEmit.cpp - ADG Builder export paths ------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilderDetail.h"
#include "fcc/ADG/ADGVerifier.h"

#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "fcc/Dialect/Fabric/FabricDialect.h"

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
#include <sstream>

namespace fcc {
namespace adg {

using namespace detail;

std::string ADGBuilder::Impl::generateMLIR(llvm::StringRef vizFileName) const {
  std::ostringstream os;

  os << "module {\n";

  os << "fabric.module @" << moduleName << "(";
  unsigned argIdx = 0;
  for (size_t i = 0; i < memrefInputs.size(); ++i) {
    if (argIdx > 0)
      os << ", ";
    os << "%mem" << i << ": " << memrefInputs[i].typeStr;
    argIdx++;
  }
  for (size_t i = 0; i < scalarInputs.size(); ++i) {
    if (argIdx > 0)
      os << ", ";
    os << "%scalar" << i << ": " << scalarInputs[i].typeStr;
    argIdx++;
  }
  os << ") -> (";
  for (size_t i = 0; i < scalarOutputs.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << scalarOutputs[i].typeStr;
  }
  os << ")";
  if (!vizFileName.empty())
    os << " attributes {viz_file = \"" << vizFileName.str() << "\"}";
  os << " {\n";

  std::set<unsigned> usedPEDefs;
  std::set<unsigned> usedSWDefs;
  std::set<unsigned> usedMemoryDefs;
  std::set<unsigned> usedExtMemDefs;
  std::set<unsigned> usedFIFODefs;
  for (const auto &inst : instances) {
    switch (inst.kind) {
    case InstanceKind::PE:
      usedPEDefs.insert(inst.defIdx);
      break;
    case InstanceKind::SW:
      usedSWDefs.insert(inst.defIdx);
      break;
    case InstanceKind::Memory:
      usedMemoryDefs.insert(inst.defIdx);
      break;
    case InstanceKind::ExtMem:
      usedExtMemDefs.insert(inst.defIdx);
      break;
    case InstanceKind::FIFO:
      usedFIFODefs.insert(inst.defIdx);
      break;
    case InstanceKind::AddTag:
    case InstanceKind::MapTag:
    case InstanceKind::DelTag:
      break;
    }
  }

  for (unsigned peIdx : usedPEDefs) {
    const auto &pe = peDefs[peIdx];
    os << "  fabric." << (pe.temporal ? "temporal_pe" : "spatial_pe") << " @"
       << pe.name << "(";
    for (size_t p = 0; p < pe.inputTypes.size(); ++p) {
      if (p > 0)
        os << ", ";
      os << "%in" << p << ": " << pe.inputTypes[p];
    }
    os << ") -> (";
    for (size_t p = 0; p < pe.outputTypes.size(); ++p) {
      if (p > 0)
        os << ", ";
      os << pe.outputTypes[p];
    }
    os << ")";
    if (pe.temporal) {
      os << " [num_register = " << pe.numRegister << " : i64"
         << ", num_instruction = " << pe.numInstruction << " : i64"
         << ", reg_fifo_depth = " << pe.regFifoDepth << " : i64";
      if (pe.enableShareOperandBuffer)
        os << ", enable_share_operand_buffer = true";
      if (pe.operandBufferSize)
        os << ", operand_buffer_size = " << *pe.operandBufferSize << " : i64";
      os << "]";
    }
    os << " {\n";

    for (unsigned fuIdx : pe.fuIndices) {
      const auto &fu = fuDefs[fuIdx];
      os << "    fabric.function_unit @" << fu.name << "(";
      for (size_t j = 0; j < fu.inputTypes.size(); ++j) {
        if (j > 0)
          os << ", ";
        os << "%arg" << j << ": " << fu.inputTypes[j];
      }
      os << ")";
      if (!fu.outputTypes.empty()) {
        os << " -> (";
        for (size_t j = 0; j < fu.outputTypes.size(); ++j) {
          if (j > 0)
            os << ", ";
          os << fu.outputTypes[j];
        }
        os << ")";
      }
      os << " [latency = " << fu.latency << ", interval = " << fu.interval
         << "]";
      os << " {\n";
      emitFUBody(os, fu, "    ");
      os << "    }\n";
    }

    os << "    fabric.yield\n";
    os << "  }\n";
  }

  for (unsigned swIdx : usedSWDefs) {
    const auto &sw = swDefs[swIdx];
    os << "  fabric." << (sw.temporal ? "temporal_sw" : "spatial_sw") << " @"
       << sw.name;

    bool hasHw = false;
    auto startHw = [&]() {
      if (!hasHw)
        os << " [";
      else
        os << ", ";
      hasHw = true;
    };

    if (!sw.connectivity.empty()) {
      startHw();
      unsigned numO = sw.outputTypes.size();
      unsigned numI = sw.inputTypes.size();
      os << "connectivity_table = [";
      for (unsigned o = 0; o < numO; ++o) {
        if (o > 0)
          os << ", ";
        os << "\"";
        for (unsigned ii = 0; ii < numI; ++ii)
          os << (sw.connectivity[o][ii] ? "1" : "0");
        os << "\"";
      }
      os << "]";
    }

    if (sw.decomposableBits >= 0) {
      startHw();
      os << "decomposable_bits = " << sw.decomposableBits << " : i64";
    }
    if (sw.temporal) {
      startHw();
      os << "num_route_table = " << sw.numRouteTable << " : i64";
    }

    if (hasHw)
      os << "]";

    os << " : (";
    for (size_t p = 0; p < sw.inputTypes.size(); ++p) {
      if (p > 0)
        os << ", ";
      os << sw.inputTypes[p];
    }
    os << ") -> (";
    for (size_t p = 0; p < sw.outputTypes.size(); ++p) {
      if (p > 0)
        os << ", ";
      os << sw.outputTypes[p];
    }
    os << ")\n";
  }

  for (unsigned memIdx : usedMemoryDefs) {
    const auto &mem = memoryDefs[memIdx];
    os << "  fabric.memory @" << mem.name << " [ldCount = " << mem.ldPorts
       << ", stCount = " << mem.stPorts << ", lsqDepth = " << mem.lsqDepth
       << ", memrefType = " << mem.memrefType;
    if (!mem.isPrivate)
      os << ", is_private";
    os << "] : (";
    bool first = true;
    for (unsigned l = 0; l < mem.ldPorts; ++l) {
      if (!first)
        os << ", ";
      first = false;
      os << bitsType(64);
    }
    for (unsigned s = 0; s < mem.stPorts; ++s) {
      if (!first)
        os << ", ";
      first = false;
      os << bitsType(64);
    }
    for (unsigned s = 0; s < mem.stPorts; ++s) {
      if (!first)
        os << ", ";
      first = false;
      os << bitsType(64);
    }
    os << ") -> (";
    first = true;
    for (unsigned l = 0; l < mem.ldPorts; ++l) {
      if (!first)
        os << ", ";
      first = false;
      os << bitsType(64);
    }
    for (unsigned l = 0; l < mem.ldPorts; ++l) {
      if (!first)
        os << ", ";
      first = false;
      os << bitsType(64);
    }
    for (unsigned s = 0; s < mem.stPorts; ++s) {
      if (!first)
        os << ", ";
      first = false;
      os << bitsType(64);
    }
    if (!mem.isPrivate) {
      if (!first)
        os << ", ";
      os << mem.memrefType;
    }
    os << ")\n";
  }

  std::map<unsigned, std::map<unsigned, std::pair<unsigned, unsigned>>>
      incomingConns;
  for (const auto &conn : connections)
    incomingConns[conn.dstInst][conn.dstPort] = {conn.srcInst, conn.srcPort};

  std::map<unsigned, std::map<unsigned, unsigned>> scalarInputConns;
  for (const auto &sc : scalarToInstConns)
    scalarInputConns[sc.dstInst][sc.dstPort] = sc.scalarIdx;

  std::map<unsigned, std::pair<unsigned, unsigned>> scalarOutputConns;
  for (const auto &ic : instToScalarConns)
    scalarOutputConns[ic.scalarOutputIdx] = {ic.srcInst, ic.srcPort};

  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];

    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &pe = peDefs[inst.defIdx];
      unsigned numOut = pe.outputTypes.size();

      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1)
          os << ":" << numOut;
      }
      os << " = fabric.instance @" << pe.name << "(";

      for (size_t p = 0; p < pe.inputTypes.size(); ++p) {
        if (p > 0)
          os << ", ";
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
            unsigned srcOutCount =
                getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                       extMemDefs, srcInst);
            os << "%v" << srcInst;
            if (srcOutCount > 1)
              os << "#" << srcPort;
            continue;
          }
        }
        llvm::report_fatal_error(
            "ADGBuilder attempted to emit a spatial_pe with an unconnected "
            "input port; fix the ADG description instead of relying on "
            "implicit self-loops");
      }
      os << ") {sym_name = \"" << inst.name << "\"}";

      os << " : (";
      for (size_t p = 0; p < pe.inputTypes.size(); ++p) {
        if (p > 0)
          os << ", ";
        os << pe.inputTypes[p];
      }
      os << ") -> (";
      for (size_t p = 0; p < pe.outputTypes.size(); ++p) {
        if (p > 0)
          os << ", ";
        os << pe.outputTypes[p];
      }
      os << ")\n";
      break;
    }
    case InstanceKind::SW: {
      const auto &sw = swDefs[inst.defIdx];
      unsigned numIn = sw.inputTypes.size();
      unsigned numOut = sw.outputTypes.size();

      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1)
          os << ":" << numOut;
      }
      os << " = fabric.instance @" << sw.name << "(";

      for (unsigned p = 0; p < numIn; ++p) {
        if (p > 0)
          os << ", ";
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
            unsigned srcOutCount =
                getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                       extMemDefs, srcInst);
            os << "%v" << srcInst;
            if (srcOutCount > 1)
              os << "#" << srcPort;
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
        if (p > 0)
          os << ", ";
        os << sw.inputTypes[p];
      }
      os << ") -> (";
      for (unsigned p = 0; p < numOut; ++p) {
        if (p > 0)
          os << ", ";
        os << sw.outputTypes[p];
      }
      os << ")\n";
      break;
    }
    case InstanceKind::Memory: {
      const auto &mem = memoryDefs[inst.defIdx];
      const unsigned numDataInputs = mem.ldPorts + mem.stPorts * 2;
      const unsigned numOut =
          mem.ldPorts + mem.ldPorts + mem.stPorts + (mem.isPrivate ? 0 : 1);
      if (numOut > 0) {
        os << "  %v" << i;
        if (numOut > 1)
          os << ":" << numOut;
      }
      os << " = fabric.memory @" << inst.name << " [ldCount = " << mem.ldPorts
         << ", stCount = " << mem.stPorts << ", lsqDepth = " << mem.lsqDepth
         << ", memrefType = " << mem.memrefType << ", numRegion = 1";
      if (!mem.isPrivate)
        os << ", is_private";
      os << "] (";

      bool firstOperand = true;
      auto emitOperandSep = [&]() {
        if (!firstOperand)
          os << ", ";
        firstOperand = false;
      };
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
            unsigned srcOutCount =
                getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                       extMemDefs, srcInst);
            os << "%v" << srcInst;
            if (srcOutCount > 1)
              os << "#" << srcPort;
            return;
          }
        }

        llvm::report_fatal_error(
            "ADGBuilder attempted to emit a fabric.memory with an "
            "unconnected input port");
      };
      for (unsigned p = 0; p < numDataInputs; ++p)
        emitConnectedOperand(p);
      os << ")";

      os << " : (";
      for (unsigned p = 0; p < numDataInputs; ++p) {
        if (p > 0)
          os << ", ";
        os << getInstanceInputType(instances, peDefs, swDefs, memoryDefs,
                                   addTagDefs, mapTagDefs, delTagDefs,
                                   fifoDefs, i, p);
      }
      os << ") -> (";
      bool first = true;
      for (unsigned p = 0; p < numOut; ++p) {
        if (!first)
          os << ", ";
        first = false;
        os << getInstanceOutputType(instances, peDefs, swDefs, memoryDefs,
                                    extMemDefs, addTagDefs, mapTagDefs,
                                    delTagDefs, fifoDefs, i, p);
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
        if (numOut > 1)
          os << ":" << numOut;
      }
      os << " = fabric.extmemory @" << inst.name;
      os << " [ldCount = " << mem.ldPorts << ", stCount = " << mem.stPorts
         << ", lsqDepth = " << mem.lsqDepth << ", memrefType = "
         << mem.memrefType << "]";
      os << " (";

      bool firstOperand = true;
      auto emitOperandSep = [&]() {
        if (!firstOperand)
          os << ", ";
        firstOperand = false;
      };

      bool emittedMemref = false;
      for (const auto &mc : memrefConnections) {
        if (mc.instIdx != i)
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
            unsigned srcOutCount =
                getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                       extMemDefs, srcInst);
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

      auto getExtMemInputType = [&](unsigned portIdx) {
        auto scIt = scalarInputConns.find(i);
        if (scIt != scalarInputConns.end()) {
          auto scPit = scIt->second.find(portIdx);
          if (scPit != scIt->second.end())
            return scalarInputs[scPit->second].typeStr;
        }
        auto it = incomingConns.find(i);
        if (it != incomingConns.end()) {
          auto pit = it->second.find(portIdx);
          if (pit != it->second.end()) {
            return getInstanceOutputType(instances, peDefs, swDefs, memoryDefs,
                                         extMemDefs, addTagDefs, mapTagDefs,
                                         delTagDefs, fifoDefs, pit->second.first,
                                         pit->second.second);
          }
        }
        return bitsType(64);
      };

      auto getExtMemOutputType = [&](unsigned portIdx) {
        for (const auto &conn : connections) {
          if (conn.srcInst != i || conn.srcPort != portIdx)
            continue;
          return getInstanceInputType(instances, peDefs, swDefs, memoryDefs,
                                      addTagDefs, mapTagDefs, delTagDefs,
                                      fifoDefs, conn.dstInst, conn.dstPort);
        }
        for (const auto &ic : instToScalarConns) {
          if (ic.srcInst == i && ic.srcPort == portIdx)
            return scalarOutputs[ic.scalarOutputIdx].typeStr;
        }
        return bitsType(64);
      };

      os << " : (" << mem.memrefType;
      for (unsigned l = 0; l < mem.ldPorts; ++l)
        os << ", " << getExtMemInputType(1 + l);
      for (unsigned s = 0; s < mem.stPorts; ++s)
        os << ", " << getExtMemInputType(1 + mem.ldPorts + s);
      for (unsigned s = 0; s < mem.stPorts; ++s)
        os << ", " << getExtMemInputType(1 + mem.ldPorts + mem.stPorts + s);
      os << ") -> (";
      bool first = true;
      for (unsigned l = 0; l < mem.ldPorts; ++l) {
        if (!first)
          os << ", ";
        first = false;
        os << getExtMemOutputType(l);
      }
      for (unsigned l = 0; l < mem.ldPorts; ++l) {
        if (!first)
          os << ", ";
        first = false;
        os << getExtMemOutputType(mem.ldPorts + l);
      }
      for (unsigned s = 0; s < mem.stPorts; ++s) {
        if (!first)
          os << ", ";
        first = false;
        os << getExtMemOutputType(mem.ldPorts * 2 + s);
      }
      os << ")\n";
      break;
    }
    case InstanceKind::FIFO:
      break;
    case InstanceKind::AddTag: {
      const auto &node = addTagDefs[inst.defIdx];
      os << "  %v" << i << " = fabric.add_tag ";
      auto scIt = scalarInputConns.find(i);
      if (scIt != scalarInputConns.end()) {
        auto scPit = scIt->second.find(0);
        if (scPit != scIt->second.end()) {
          os << "%scalar" << scPit->second;
        } else {
          llvm::report_fatal_error(
              "ADGBuilder attempted to emit fabric.add_tag with an "
              "unconnected input port");
        }
      } else {
        auto it = incomingConns.find(i);
        if (it == incomingConns.end() || it->second.find(0) == it->second.end())
          llvm::report_fatal_error(
              "ADGBuilder attempted to emit fabric.add_tag with an "
              "unconnected input port");
        unsigned srcInst = it->second.at(0).first;
        unsigned srcPort = it->second.at(0).second;
        unsigned srcOutCount =
            getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                   extMemDefs, srcInst);
        os << "%v" << srcInst;
        if (srcOutCount > 1)
          os << "#" << srcPort;
      }
      os << " {tag = " << node.tag << " : i64}"
         << " : " << node.inputType << " -> " << node.outputType << "\n";
      break;
    }
    case InstanceKind::MapTag: {
      const auto &node = mapTagDefs[inst.defIdx];
      os << "  %v" << i << " = fabric.map_tag ";
      auto scIt = scalarInputConns.find(i);
      if (scIt != scalarInputConns.end()) {
        auto scPit = scIt->second.find(0);
        if (scPit != scIt->second.end()) {
          os << "%scalar" << scPit->second;
        } else {
          llvm::report_fatal_error(
              "ADGBuilder attempted to emit fabric.map_tag with an "
              "unconnected input port");
        }
      } else {
        auto it = incomingConns.find(i);
        if (it == incomingConns.end() || it->second.find(0) == it->second.end())
          llvm::report_fatal_error(
              "ADGBuilder attempted to emit fabric.map_tag with an "
              "unconnected input port");
        unsigned srcInst = it->second.at(0).first;
        unsigned srcPort = it->second.at(0).second;
        unsigned srcOutCount =
            getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                   extMemDefs, srcInst);
        os << "%v" << srcInst;
        if (srcOutCount > 1)
          os << "#" << srcPort;
      }
      os << " [table_size = " << node.table.size()
         << " : i64] attributes {table = [";
      for (size_t entryIdx = 0; entryIdx < node.table.size(); ++entryIdx) {
        if (entryIdx > 0)
          os << ", ";
        const auto &entry = node.table[entryIdx];
        os << "[" << (entry.valid ? "1" : "0") << " : i64, " << entry.srcTag
           << " : i64, " << entry.dstTag << " : i64]";
      }
      os << "]}"
         << " : " << node.inputType << " -> " << node.outputType << "\n";
      break;
    }
    case InstanceKind::DelTag: {
      const auto &node = delTagDefs[inst.defIdx];
      os << "  %v" << i << " = fabric.del_tag ";
      auto scIt = scalarInputConns.find(i);
      if (scIt != scalarInputConns.end()) {
        auto scPit = scIt->second.find(0);
        if (scPit != scIt->second.end()) {
          os << "%scalar" << scPit->second;
        } else {
          llvm::report_fatal_error(
              "ADGBuilder attempted to emit fabric.del_tag with an "
              "unconnected input port");
        }
      } else {
        auto it = incomingConns.find(i);
        if (it == incomingConns.end() || it->second.find(0) == it->second.end())
          llvm::report_fatal_error(
              "ADGBuilder attempted to emit fabric.del_tag with an "
              "unconnected input port");
        unsigned srcInst = it->second.at(0).first;
        unsigned srcPort = it->second.at(0).second;
        unsigned srcOutCount =
            getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                   extMemDefs, srcInst);
        os << "%v" << srcInst;
        if (srcOutCount > 1)
          os << "#" << srcPort;
      }
      os << " : " << node.inputType << " -> " << node.outputType << "\n";
      break;
    }
    }
  }

  os << "  fabric.yield";
  if (!scalarOutputs.empty()) {
    os << " ";
    for (size_t i = 0; i < scalarOutputs.size(); ++i) {
      if (i > 0)
        os << ", ";
      auto soIt = scalarOutputConns.find(i);
      if (soIt != scalarOutputConns.end()) {
        unsigned srcInst = soIt->second.first;
        unsigned srcPort = soIt->second.second;
        unsigned srcOutCount =
            getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                   extMemDefs, srcInst);
        os << "%v" << srcInst;
        if (srcOutCount > 1)
          os << "#" << srcPort;
      } else {
        unsigned firstSwInst = UINT_MAX;
        for (size_t j = 0; j < instances.size(); ++j) {
          if (instances[j].kind == InstanceKind::SW) {
            firstSwInst = j;
            break;
          }
        }
        if (firstSwInst != UINT_MAX) {
          unsigned srcOutCount =
              getInstanceOutputCount(instances, peDefs, swDefs, memoryDefs,
                                     extMemDefs, firstSwInst);
          os << "%v" << firstSwInst;
          if (srcOutCount > 1)
            os << "#0";
        } else {
          os << "%scalar0";
        }
      }
    }
    os << " : ";
    for (size_t i = 0; i < scalarOutputs.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << scalarOutputs[i].typeStr;
    }
  }
  os << "\n";
  os << "}\n";
  os << "}\n";

  return os.str();
}

// Generate visualization sidecar JSON with explicit component positions and
// pre-routed module-level edges.
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
                        std::max(0.0,
                                 static_cast<double>(peDef.fuIndices.size()) -
                                     1.0) *
                            approxFuGap +
                        approxPEPadX);
  };
  auto computePEHeight = [&](const PEDef &) { return 200.0; };
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
        buildPortSideCounts(swDef.inputTypes.size(), 2);
    std::array<unsigned, 4> outCounts =
        buildPortSideCounts(swDef.outputTypes.size(), 2);
    unsigned maxSideSlots = 0;
    maxSideSlots = std::max(maxSideSlots, inCounts[0]);
    maxSideSlots = std::max(maxSideSlots, inCounts[1]);
    maxSideSlots = std::max(maxSideSlots, outCounts[0]);
    maxSideSlots = std::max(maxSideSlots, outCounts[1]);
    return std::max(kSwitchMinSide,
                    32.0 +
                        (static_cast<double>(std::max(1U, maxSideSlots)) + 1.0) *
                            kSwitchPortPitch);
  };

  auto estimateBoxInfo = [&](unsigned instIdx) -> BoxInfo {
    BoxInfo info;
    const auto &inst = instances[instIdx];
    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &peDef = peDefs[inst.defIdx];
      info.width = computePEWidth(peDef);
      info.height = computePEHeight(peDef);
      info.numInputs = peDef.inputTypes.size();
      info.numOutputs = peDef.outputTypes.size();
      break;
    }
    case InstanceKind::SW: {
      const auto &swDef = swDefs[inst.defIdx];
      double side = computeSWSide(swDef);
      info.width = side;
      info.height = side;
      info.numInputs = swDef.inputTypes.size();
      info.numOutputs = swDef.outputTypes.size();
      break;
    }
    case InstanceKind::Memory: {
      const auto &memDef = memoryDefs[inst.defIdx];
      info.width = 170.0;
      info.height = 80.0;
      info.numInputs = memDef.ldPorts + memDef.stPorts * 2;
      info.numOutputs =
          memDef.ldPorts + memDef.stPorts + memDef.ldPorts + (memDef.isPrivate ? 0 : 1);
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
    case InstanceKind::FIFO:
      info.width = 100.0;
      info.height = 56.0;
      info.numInputs = 1;
      info.numOutputs = 1;
      break;
    case InstanceKind::AddTag:
    case InstanceKind::MapTag:
    case InstanceKind::DelTag:
      info.width = 92.0;
      info.height = 52.0;
      info.numInputs = 1;
      info.numOutputs = 1;
      break;
    }
    info.valid = true;
    return info;
  };

  auto computeEffectivePlacements = [&]() {
    std::map<unsigned, VizPlacement> placements = vizPlacements;

    double placedMinX = 0.0;
    double placedMaxY = 0.0;
    bool havePlaced = false;
    for (const auto &[instIdx, placement] : placements) {
      BoxInfo info = estimateBoxInfo(instIdx);
      if (!info.valid)
        continue;
      double boxMinX = placement.centerX - info.width / 2.0;
      double boxMaxY = placement.centerY + info.height / 2.0;
      if (!havePlaced) {
        placedMinX = boxMinX;
        placedMaxY = boxMaxY;
        havePlaced = true;
      } else {
        placedMinX = std::min(placedMinX, boxMinX);
        placedMaxY = std::max(placedMaxY, boxMaxY);
      }
    }

    constexpr double kAutoGapX = 88.0;
    constexpr double kAutoGapY = 108.0;
    constexpr double kAutoWrapWidth = 3600.0;
    double startX = havePlaced ? placedMinX : 120.0;
    double cursorX = startX;
    double cursorY = havePlaced ? placedMaxY + 160.0 : 120.0;
    double rowHeight = 0.0;
    int packedRow = havePlaced ? 1000 : 0;
    int packedCol = 0;
    for (unsigned instIdx = 0; instIdx < instances.size(); ++instIdx) {
      if (placements.count(instIdx))
        continue;
      BoxInfo info = estimateBoxInfo(instIdx);
      if (!info.valid)
        continue;
      if (cursorX > startX &&
          cursorX + info.width > startX + kAutoWrapWidth) {
        cursorX = startX;
        cursorY += rowHeight + kAutoGapY;
        rowHeight = 0.0;
        ++packedRow;
        packedCol = 0;
      }
      placements[instIdx] = {cursorX + info.width / 2.0,
                             cursorY + info.height / 2.0, packedRow,
                             packedCol++};
      cursorX += info.width + kAutoGapX;
      rowHeight = std::max(rowHeight, info.height);
    }
    return placements;
  };

  std::map<unsigned, VizPlacement> placements = computeEffectivePlacements();

  auto computeBoxInfo = [&](unsigned instIdx) -> BoxInfo {
    BoxInfo info = estimateBoxInfo(instIdx);
    if (!info.valid)
      return info;
    auto it = placements.find(instIdx);
    if (it == placements.end()) {
      info.valid = false;
      return info;
    }
    info.centerX = it->second.centerX;
    info.centerY = it->second.centerY;
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
          buildPEPortSideCounts(peDef.inputTypes.size());
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
          buildPortSideCounts(swDef.inputTypes.size(), 2);
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
          buildPEPortSideCounts(peDef.outputTypes.size());
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
          buildPortSideCounts(swDef.outputTypes.size(), 2);
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
                                        unsigned dstPortIdx) -> std::vector<RoutePt> {
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
                                         unsigned scalarOutIdx) -> std::vector<RoutePt> {
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
    const double corridorY =
        moduleBounds.y + moduleBounds.h - 42.0 - std::abs(laneOffset);
    std::vector<RoutePt> pts;
    pts.push_back({exitX, srcPort.y});
    if (std::abs(srcPort.y - corridorY) > 0.5)
      pts.push_back({exitX, corridorY});
    if (std::abs(exitX - dstPort.x) > 0.5)
      pts.push_back({dstPort.x, corridorY});
    return pts;
  };

  auto routeConnection = [&](const Connection &conn, unsigned routeOrdinal)
      -> std::vector<RoutePt> {
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
    double corridorY =
        routeAbove ? std::min(srcTop, dstTop) - margin - std::abs(laneOffset)
                   : std::max(srcBottom, dstBottom) + margin + std::abs(laneOffset);
    pts.push_back({srcExitX, srcPort.y});
    pts.push_back({srcExitX, corridorY});
    pts.push_back({dstEntryX, corridorY});
    pts.push_back({dstEntryX, dstPort.y});
    return pts;
  };

  std::ostringstream os;
  os << "{\n"
     << "  \"version\": 1,\n"
     << "  \"components\": [\n";

  bool first = true;
  for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
    auto it = placements.find(static_cast<unsigned>(instIdx));
    if (it == placements.end())
      continue;

    if (!first)
      os << ",\n";
    first = false;

    const auto &inst = instances[instIdx];
    const auto &placement = it->second;
    const char *kindName = "instance";
    switch (inst.kind) {
    case InstanceKind::PE:
      kindName = peDefs[inst.defIdx].temporal ? "temporal_pe" : "spatial_pe";
      break;
    case InstanceKind::SW:
      kindName = swDefs[inst.defIdx].temporal ? "temporal_sw" : "spatial_sw";
      break;
    case InstanceKind::Memory:
      kindName = "memory";
      break;
    case InstanceKind::ExtMem:
      kindName = "extmemory";
      break;
    case InstanceKind::FIFO:
      kindName = "fifo";
      break;
    case InstanceKind::AddTag:
      kindName = "add_tag";
      break;
    case InstanceKind::MapTag:
      kindName = "map_tag";
      break;
    case InstanceKind::DelTag:
      kindName = "del_tag";
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
       << ", \"from_port\": " << fromPort << ", \"to\": \"" << toName.str()
       << "\"" << ", \"to_port\": " << toPort << ", \"points\": [";
    for (size_t ptIdx = 0; ptIdx < pts.size(); ++ptIdx) {
      if (ptIdx > 0)
        os << ", ";
      os << "{\"x\": " << pts[ptIdx].x << ", \"y\": " << pts[ptIdx].y
         << "}";
    }
    os << "]}";
  };
  for (size_t connIdx = 0; connIdx < connections.size(); ++connIdx) {
    const auto &conn = connections[connIdx];
    if (placements.find(conn.srcInst) == placements.end() ||
        placements.find(conn.dstInst) == placements.end())
      continue;
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    auto pairKey =
        std::make_pair(std::min(conn.srcInst, conn.dstInst),
                       std::max(conn.srcInst, conn.dstInst));
    unsigned pairOrdinal = nextPairOrdinal[pairKey]++;
    std::vector<RoutePt> pts = routeConnection(conn, pairOrdinal);

    emitRouteRecord(srcInst.name, conn.srcPort, dstInst.name, conn.dstPort,
                    pts);
  }
  for (const auto &sc : scalarToInstConns) {
    if (placements.find(sc.dstInst) == placements.end())
      continue;
    std::vector<RoutePt> pts =
        routeModuleInputConnection(sc.scalarIdx, sc.dstInst, sc.dstPort);
    emitRouteRecord("module_in", sc.scalarIdx, instances[sc.dstInst].name,
                    sc.dstPort, pts);
  }
  for (const auto &ic : instToScalarConns) {
    if (placements.find(ic.srcInst) == placements.end())
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

void ADGBuilder::exportMLIR(const std::string &path) {
  std::string valErr;
  if (!impl_->validate(valErr)) {
    llvm::report_fatal_error(llvm::Twine("ADGBuilder validation failed:\n") +
                             valErr);
  }

  std::string vizFileName;
  std::string vizJsonText;
  llvm::SmallString<256> vizPath(path);
  llvm::sys::path::replace_extension(vizPath, "viz.json");
  vizFileName = std::string(llvm::sys::path::filename(vizPath));
  vizJsonText = impl_->generateVizJson();

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
      llvm::errs() << "error: cannot write viz sidecar: " << vizPath << "\n";
      llvm::errs() << vizEc.message() << "\n";
      std::exit(1);
    }
    vizOut << vizJsonText;
    vizOut.flush();
  }
}

} // namespace adg
} // namespace fcc
