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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

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
