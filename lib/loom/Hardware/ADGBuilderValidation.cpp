//===-- ADGBuilderValidation.cpp - ADG Builder validation ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADGBuilderImpl.h"

#include <functional>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Validation Helpers
//===----------------------------------------------------------------------===//

/// Validate connectivity table shape and completeness for a switch-like
/// component (used by both switch and temporal_switch validation).
static void validateConnectivityTable(
    const std::vector<std::vector<bool>> &connectivity, unsigned numIn,
    unsigned numOut, const std::string &codePrefix, const std::string &loc,
    std::function<void(const std::string &, const std::string &,
                       const std::string &)>
        addError) {
  if (connectivity.empty())
    return;
  if (connectivity.size() != numOut)
    addError(codePrefix + "_TABLE_SHAPE",
             "connectivity_table row count != num_outputs", loc);
  for (size_t r = 0; r < connectivity.size(); ++r) {
    if (connectivity[r].size() != numIn)
      addError(codePrefix + "_TABLE_SHAPE",
               "connectivity_table column count != num_inputs", loc);
    bool hasOne = false;
    for (bool v : connectivity[r])
      if (v) hasOne = true;
    if (!hasOne)
      addError(codePrefix + "_ROW_EMPTY",
               "connectivity row " + std::to_string(r) + " has no 1", loc);
  }
  for (unsigned c = 0; c < numIn; ++c) {
    bool hasOne = false;
    for (unsigned r = 0; r < connectivity.size(); ++r)
      if (c < connectivity[r].size() && connectivity[r][c]) hasOne = true;
    if (!hasOne)
      addError(codePrefix + "_COL_EMPTY",
               "connectivity column " + std::to_string(c) + " has no 1", loc);
  }
}

/// Validate memory port rules shared by both memory and external memory.
static void validateMemoryPorts(
    unsigned ldCount, unsigned stCount, unsigned lsqDepth,
    const std::string &loc,
    std::function<void(const std::string &, const std::string &,
                       const std::string &)>
        addError) {
  if (ldCount == 0 && stCount == 0)
    addError("COMP_MEMORY_PORTS_EMPTY", "ldCount and stCount are both 0", loc);
  if (stCount > 0 && lsqDepth < 1)
    addError("COMP_MEMORY_LSQ_MIN", "lsqDepth must be >= 1 when stCount > 0",
             loc);
  if (stCount == 0 && lsqDepth > 0)
    addError("COMP_MEMORY_LSQ_WITHOUT_STORE",
             "lsqDepth must be 0 when stCount == 0", loc);
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

ValidationResult ADGBuilder::validateADG() {
  ValidationResult result;

  auto addError = [&](const std::string &code, const std::string &msg,
                      const std::string &loc = "") {
    result.errors.push_back({code, msg, loc});
    result.success = false;
  };

  // Validate switch definitions.
  for (size_t i = 0; i < impl_->switchDefs.size(); ++i) {
    const auto &sw = impl_->switchDefs[i];
    std::string loc = "switch @" + sw.name;
    if (sw.numIn < 1 || sw.numOut < 1)
      addError("COMP_SWITCH_PORT_ZERO",
               "switch must have at least 1 input and 1 output", loc);
    if (sw.numIn > 32 || sw.numOut > 32)
      addError("COMP_SWITCH_PORT_LIMIT",
               "switch has more than 32 inputs or outputs", loc);
    validateConnectivityTable(sw.connectivity, sw.numIn, sw.numOut,
                              "COMP_SWITCH", loc, addError);
  }

  // Validate temporal switch definitions.
  for (size_t i = 0; i < impl_->temporalSwitchDefs.size(); ++i) {
    const auto &ts = impl_->temporalSwitchDefs[i];
    std::string loc = "temporal_sw @" + ts.name;
    if (ts.numIn < 1 || ts.numOut < 1)
      addError("COMP_TEMPORAL_SW_PORT_ZERO",
               "temporal switch must have at least 1 input and 1 output", loc);
    if (ts.numIn > 32 || ts.numOut > 32)
      addError("COMP_TEMPORAL_SW_PORT_LIMIT",
               "temporal switch has more than 32 inputs or outputs", loc);
    if (!ts.interfaceType.isTagged())
      addError("COMP_TEMPORAL_SW_INTERFACE_NOT_TAGGED",
               "temporal switch interface type must be tagged", loc);
    if (ts.numRouteTable < 1)
      addError("COMP_TEMPORAL_SW_NUM_ROUTE_TABLE",
               "num_route_table must be >= 1", loc);
    validateConnectivityTable(ts.connectivity, ts.numIn, ts.numOut,
                              "COMP_TEMPORAL_SW", loc, addError);
  }

  // Validate temporal PE definitions.
  for (size_t i = 0; i < impl_->temporalPEDefs.size(); ++i) {
    const auto &tp = impl_->temporalPEDefs[i];
    std::string loc = "temporal_pe @" + tp.name;
    if (!tp.interfaceType.isTagged())
      addError("COMP_TEMPORAL_PE_INTERFACE_NOT_TAGGED",
               "temporal PE interface type must be tagged", loc);
    if (tp.numInstructions < 1)
      addError("COMP_TEMPORAL_PE_NUM_INSTRUCTION",
               "num_instruction must be >= 1", loc);
    if (tp.numRegisters > 0 && tp.regFifoDepth == 0)
      addError("COMP_TEMPORAL_PE_REG_FIFO_DEPTH",
               "reg_fifo_depth must be > 0 when num_register > 0", loc);
    if (tp.numRegisters == 0 && tp.regFifoDepth > 0)
      addError("COMP_TEMPORAL_PE_REG_FIFO_DEPTH",
               "reg_fifo_depth must be 0 when num_register == 0", loc);
    if (tp.fuPEDefIndices.empty())
      addError("COMP_TEMPORAL_PE_EMPTY_BODY",
               "temporal PE has no FU definitions", loc);
    // Validate each FU definition referenced by this temporal PE.
    unsigned expectedIn = 0, expectedOut = 0;
    for (size_t fi = 0; fi < tp.fuPEDefIndices.size(); ++fi) {
      unsigned fuIdx = tp.fuPEDefIndices[fi];
      std::string fuLoc = loc + " FU[" + std::to_string(fi) + "]";
      if (fuIdx >= impl_->peDefs.size()) {
        addError("COMP_TEMPORAL_PE_FU_INVALID",
                 "FU index out of range", fuLoc);
        continue;
      }
      const auto &fu = impl_->peDefs[fuIdx];
      // FUs must use native interface (not tagged).
      if (fu.interface == InterfaceCategory::Tagged)
        addError("COMP_TEMPORAL_PE_TAGGED_FU",
                 "temporal PE FU must use native interface", fuLoc);
      bool fuHasTagged = false;
      for (const auto &t : fu.inputPorts)
        if (t.isTagged()) fuHasTagged = true;
      for (const auto &t : fu.outputPorts)
        if (t.isTagged()) fuHasTagged = true;
      if (fuHasTagged)
        addError("COMP_TEMPORAL_PE_TAGGED_FU",
                 "temporal PE FU must not have tagged ports", fuLoc);
      // All FUs must have the same port arity.
      unsigned nIn = fu.inputPorts.size();
      unsigned nOut = fu.outputPorts.size();
      if (fi == 0) {
        expectedIn = nIn;
        expectedOut = nOut;
      } else {
        if (nIn != expectedIn || nOut != expectedOut)
          addError("COMP_TEMPORAL_PE_FU_ARITY",
                   "FU port count mismatch with first FU ("
                   + std::to_string(nIn) + " in, " + std::to_string(nOut)
                   + " out vs " + std::to_string(expectedIn) + " in, "
                   + std::to_string(expectedOut) + " out)", fuLoc);
      }
    }
    if (!tp.shareModeB && tp.shareBufferSize > 0)
      addError("COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE",
               "operand_buffer_size set without enable_share_operand_buffer",
               loc);
    if (tp.shareModeB && tp.shareBufferSize == 0)
      addError("COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING",
               "operand_buffer_size missing with share_operand_buffer", loc);
    if (tp.shareModeB && tp.shareBufferSize > 8192)
      addError("COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE",
               "operand_buffer_size out of range [1, 8192]", loc);
  }

  // Validate memory definitions.
  for (size_t i = 0; i < impl_->memoryDefs.size(); ++i) {
    const auto &mem = impl_->memoryDefs[i];
    std::string loc = "memory @" + mem.name;
    validateMemoryPorts(mem.ldCount, mem.stCount, mem.lsqDepth, loc, addError);
    if (mem.shape.isDynamic())
      addError("COMP_MEMORY_STATIC_REQUIRED",
               "fabric.memory requires static memref shape", loc);
  }

  // Validate external memory definitions.
  for (size_t i = 0; i < impl_->extMemoryDefs.size(); ++i) {
    const auto &em = impl_->extMemoryDefs[i];
    std::string loc = "extmemory @" + em.name;
    validateMemoryPorts(em.ldCount, em.stCount, em.lsqDepth, loc, addError);
  }

  // Validate map_tag definitions.
  for (size_t i = 0; i < impl_->mapTagDefs.size(); ++i) {
    const auto &mt = impl_->mapTagDefs[i];
    std::string loc = "map_tag @" + mt.name;
    if (mt.tableSize < 1 || mt.tableSize > 256)
      addError("COMP_MAP_TAG_TABLE_SIZE",
               "table_size out of range [1, 256]", loc);
  }

  // Validate tag types for tag operations.
  // Accept both Type::iN(w) and canonical aliases (i1, i8, i16) with valid widths.
  auto getIntWidth = [](Type t) -> int {
    switch (t.getKind()) {
    case Type::I1:  return 1;
    case Type::I8:  return 8;
    case Type::I16: return 16;
    case Type::IN:  return (int)t.getWidth();
    default:        return -1; // not an integer type
    }
  };
  auto validateTagType = [&](Type tagType, const std::string &loc) {
    int w = getIntWidth(tagType);
    if (w < 0)
      addError("COMP_TAG_WIDTH_RANGE",
               "tag type must be an integer type", loc);
    else if (w < 1 || w > 16)
      addError("COMP_TAG_WIDTH_RANGE",
               "tag width outside [1, 16]", loc);
  };

  // Validate FIFO definitions.
  for (size_t i = 0; i < impl_->fifoDefs.size(); ++i) {
    const auto &fifo = impl_->fifoDefs[i];
    std::string loc = "fifo @" + fifo.name;
    if (fifo.depth < 1)
      addError("COMP_FIFO_DEPTH_ZERO", "depth must be >= 1", loc);
    // Type must be native or tagged. Accept Type::IN when width matches
    // a native integer width (1, 8, 16, 32, 64) since Type::iN(32) and
    // Type::i32() are semantically equivalent.
    auto kind = fifo.elementType.getKind();
    bool validType = kind == Type::I1 || kind == Type::I8 ||
                     kind == Type::I16 || kind == Type::I32 ||
                     kind == Type::I64 || kind == Type::BF16 ||
                     kind == Type::F16 || kind == Type::F32 ||
                     kind == Type::F64 || kind == Type::Index ||
                     kind == Type::None || kind == Type::Tagged;
    if (!validType && kind == Type::IN) {
      unsigned w = fifo.elementType.getWidth();
      validType = (w == 1 || w == 8 || w == 16 || w == 32 || w == 64);
    }
    if (!validType)
      addError("COMP_FIFO_INVALID_TYPE",
               "type must be a native type or tagged; got " +
                   fifo.elementType.toMLIR(),
               loc);
    // Validate tagged element: both payload and tag must be valid.
    if (kind == Type::Tagged) {
      validateTagType(fifo.elementType.getTagType(), loc);
      // Payload (value) type must be a native type.
      auto vk = fifo.elementType.getValueType().getKind();
      bool validPayload = vk == Type::I1 || vk == Type::I8 ||
                          vk == Type::I16 || vk == Type::I32 ||
                          vk == Type::I64 || vk == Type::BF16 ||
                          vk == Type::F16 || vk == Type::F32 ||
                          vk == Type::F64 || vk == Type::Index ||
                          vk == Type::None;
      if (!validPayload && vk == Type::IN) {
        unsigned w = fifo.elementType.getValueType().getWidth();
        validPayload = (w == 1 || w == 8 || w == 16 || w == 32 || w == 64);
      }
      if (!validPayload)
        addError("COMP_FIFO_INVALID_TYPE",
                 "tagged payload type must be a native type; got " +
                     fifo.elementType.toMLIR(),
                 loc);
    }
  }

  for (size_t i = 0; i < impl_->addTagDefs.size(); ++i) {
    const auto &at = impl_->addTagDefs[i];
    std::string loc = "add_tag @" + at.name;
    validateTagType(at.tagType, loc);
  }
  for (size_t i = 0; i < impl_->mapTagDefs.size(); ++i) {
    const auto &mt = impl_->mapTagDefs[i];
    std::string loc = "map_tag @" + mt.name;
    validateTagType(mt.inputTagType, loc + " (inputTagType)");
    validateTagType(mt.outputTagType, loc + " (outputTagType)");
  }
  for (size_t i = 0; i < impl_->delTagDefs.size(); ++i) {
    const auto &dt = impl_->delTagDefs[i];
    std::string loc = "del_tag @" + dt.name;
    if (dt.inputType.isTagged())
      validateTagType(dt.inputType.getTagType(), loc);
  }

  // Helper: validate tag width on any tagged type encountered in definitions.
  auto validateTaggedType = [&](Type t, const std::string &loc) {
    if (t.isTagged())
      validateTagType(t.getTagType(), loc);
  };

  // Validate PE definitions.
  for (size_t i = 0; i < impl_->peDefs.size(); ++i) {
    const auto &pe = impl_->peDefs[i];
    std::string loc = "pe @" + pe.name;
    if (pe.inputPorts.empty())
      addError("COMP_PE_EMPTY_BODY", "PE has no input ports", loc);
    if (pe.outputPorts.empty())
      addError("COMP_PE_EMPTY_BODY", "PE has no output ports", loc);
    if (pe.bodyMLIR.empty() && pe.singleOp.empty())
      addError("COMP_PE_EMPTY_BODY", "PE has no body or operation", loc);
    // Check for mixed interface (some tagged, some not).
    bool hasTagged = false, hasNative = false;
    for (const auto &t : pe.inputPorts)
      (t.isTagged() ? hasTagged : hasNative) = true;
    for (const auto &t : pe.outputPorts)
      (t.isTagged() ? hasTagged : hasNative) = true;
    if (hasTagged && hasNative)
      addError("COMP_PE_MIXED_INTERFACE",
               "PE has mixed native and tagged ports", loc);
    if (pe.interface == InterfaceCategory::Tagged && !hasTagged)
      addError("COMP_PE_TAGGED_INTERFACE_NATIVE_PORTS",
               "PE has Tagged interface but all ports are native", loc);
    if (pe.interface == InterfaceCategory::Native && hasTagged)
      addError("COMP_PE_NATIVE_INTERFACE_TAGGED_PORTS",
               "PE has Native interface but has tagged ports", loc);
    // Validate tag widths on all tagged port types.
    for (const auto &t : pe.inputPorts)
      validateTaggedType(t, loc + " (input port)");
    for (const auto &t : pe.outputPorts)
      validateTaggedType(t, loc + " (output port)");
  }

  // Validate load PE definitions.
  for (size_t i = 0; i < impl_->loadPEDefs.size(); ++i) {
    const auto &lp = impl_->loadPEDefs[i];
    std::string loc = "load_pe @" + lp.name;
    if (lp.interface == InterfaceCategory::Tagged) {
      int w = getIntWidth(Type::iN(lp.tagWidth));
      if (w < 1 || w > 16)
        addError("COMP_TAG_WIDTH_RANGE",
                 "tag width outside [1, 16]", loc);
    }
    if (lp.hwType == HardwareType::TagTransparent &&
        lp.interface != InterfaceCategory::Tagged)
      addError("COMP_LOADPE_TRANSPARENT_NATIVE",
               "TagTransparent hardware type requires Tagged interface", loc);
    if (lp.hwType == HardwareType::TagTransparent &&
        lp.interface == InterfaceCategory::Tagged && lp.queueDepth < 1)
      addError("COMP_LOADPE_TRANSPARENT_QUEUE_DEPTH",
               "TagTransparent load PE requires queueDepth >= 1", loc);
  }

  // Validate store PE definitions.
  for (size_t i = 0; i < impl_->storePEDefs.size(); ++i) {
    const auto &sp = impl_->storePEDefs[i];
    std::string loc = "store_pe @" + sp.name;
    if (sp.interface == InterfaceCategory::Tagged) {
      int w = getIntWidth(Type::iN(sp.tagWidth));
      if (w < 1 || w > 16)
        addError("COMP_TAG_WIDTH_RANGE",
                 "tag width outside [1, 16]", loc);
    }
    if (sp.hwType == HardwareType::TagTransparent &&
        sp.interface != InterfaceCategory::Tagged)
      addError("COMP_STOREPE_TRANSPARENT_NATIVE",
               "TagTransparent hardware type requires Tagged interface", loc);
    if (sp.hwType == HardwareType::TagTransparent &&
        sp.interface == InterfaceCategory::Tagged && sp.queueDepth < 1)
      addError("COMP_STOREPE_TRANSPARENT_QUEUE_DEPTH",
               "TagTransparent store PE requires queueDepth >= 1", loc);
  }

  // Validate tag widths on temporal PE/SW interface types.
  for (size_t i = 0; i < impl_->temporalPEDefs.size(); ++i) {
    const auto &tp = impl_->temporalPEDefs[i];
    validateTaggedType(tp.interfaceType,
                       "temporal_pe @" + tp.name + " (interface)");
  }
  for (size_t i = 0; i < impl_->temporalSwitchDefs.size(); ++i) {
    const auto &ts = impl_->temporalSwitchDefs[i];
    validateTaggedType(ts.interfaceType,
                       "temporal_sw @" + ts.name + " (interface)");
  }

  // Validate tag widths on module port types.
  for (size_t i = 0; i < impl_->ports.size(); ++i) {
    const auto &p = impl_->ports[i];
    if (!p.isMemref)
      validateTaggedType(p.type, "module port %" + p.name);
  }

  // Check for empty module body.
  if (impl_->instances.empty())
    addError("COMP_MODULE_EMPTY_BODY",
             "module has no instances", "module @" + impl_->moduleName);

  // Graph-level validation: check type compatibility of all connections.
  for (const auto &conn : impl_->internalConns) {
    if (conn.srcInst >= impl_->instances.size() ||
        conn.dstInst >= impl_->instances.size())
      continue;
    auto srcType = impl_->getInstanceOutputPortType(conn.srcInst, conn.srcPort);
    auto dstType = impl_->getInstanceInputPortType(conn.dstInst, conn.dstPort);
    if (!srcType.matches(dstType)) {
      std::string loc =
          impl_->instances[conn.srcInst].name + ":" +
          std::to_string(conn.srcPort) + " -> " +
          impl_->instances[conn.dstInst].name + ":" +
          std::to_string(conn.dstPort);
      addError("COMP_TYPE_MISMATCH",
               "type mismatch: " + srcType.toMLIR() + " vs " + dstType.toMLIR(),
               loc);
    }
  }

  // Check module-input-to-instance type compatibility.
  for (const auto &conn : impl_->inputConns) {
    if (conn.portIdx >= impl_->ports.size() ||
        conn.instIdx >= impl_->instances.size())
      continue;
    const auto &port = impl_->ports[conn.portIdx];
    PortType srcType = port.isMemref ? PortType::memref(port.memrefType)
                                     : PortType::scalar(port.type);
    auto dstType = impl_->getInstanceInputPortType(conn.instIdx, conn.dstPort);
    if (!srcType.matches(dstType)) {
      std::string loc =
          "%" + port.name + " -> " +
          impl_->instances[conn.instIdx].name + ":" +
          std::to_string(conn.dstPort);
      addError("COMP_TYPE_MISMATCH",
               "type mismatch: " + srcType.toMLIR() + " vs " + dstType.toMLIR(),
               loc);
    }
  }

  // Check instance-to-module-output type compatibility.
  for (const auto &conn : impl_->outputConns) {
    if (conn.instIdx >= impl_->instances.size() ||
        conn.portIdx >= impl_->ports.size())
      continue;
    auto srcType = impl_->getInstanceOutputPortType(conn.instIdx, conn.srcPort);
    const auto &port = impl_->ports[conn.portIdx];
    PortType dstType = port.isMemref ? PortType::memref(port.memrefType)
                                     : PortType::scalar(port.type);
    if (!srcType.matches(dstType)) {
      std::string loc =
          impl_->instances[conn.instIdx].name + ":" +
          std::to_string(conn.srcPort) + " -> %" + port.name;
      addError("COMP_TYPE_MISMATCH",
               "type mismatch: " + srcType.toMLIR() + " vs " + dstType.toMLIR(),
               loc);
    }
  }

  // Check that every module output port has a source connection.
  for (unsigned i = 0; i < impl_->ports.size(); ++i) {
    if (impl_->ports[i].isInput) continue;
    bool found = false;
    for (const auto &conn : impl_->outputConns) {
      if (conn.portIdx == i) { found = true; break; }
    }
    if (!found)
      addError("COMP_OUTPUT_UNCONNECTED",
               "module output %" + impl_->ports[i].name + " has no source",
               "module @" + impl_->moduleName);
  }

  // One-driver-per-input-port check and unconnected input port detection.
  // Both checks share the same iteration over input connections.
  for (unsigned instIdx = 0; instIdx < impl_->instances.size(); ++instIdx) {
    unsigned numIn = impl_->getInstanceInputCount(instIdx);
    std::vector<unsigned> driverCount(numIn, 0);

    for (const auto &conn : impl_->inputConns) {
      if (conn.instIdx == instIdx && conn.dstPort >= 0 &&
          (unsigned)conn.dstPort < numIn)
        driverCount[conn.dstPort]++;
    }
    for (const auto &conn : impl_->internalConns) {
      if (conn.dstInst == instIdx && conn.dstPort >= 0 &&
          (unsigned)conn.dstPort < numIn)
        driverCount[conn.dstPort]++;
    }

    for (unsigned p = 0; p < numIn; ++p) {
      std::string loc =
          impl_->instances[instIdx].name + ":" + std::to_string(p);
      if (driverCount[p] > 1)
        addError("COMP_MULTI_DRIVER",
                 "input port has " + std::to_string(driverCount[p]) +
                     " drivers (expected at most 1)",
                 loc);
      if (driverCount[p] == 0)
        addError("COMP_INPUT_UNCONNECTED",
                 "instance input port is not connected", loc);
    }
  }

  // Dangling instance output port detection: all output ports must be connected
  // to at least one destination (spec: connectivity completeness).
  for (unsigned instIdx = 0; instIdx < impl_->instances.size(); ++instIdx) {
    unsigned numOut = impl_->getInstanceOutputCount(instIdx);
    std::vector<bool> used(numOut, false);

    for (const auto &conn : impl_->internalConns) {
      if (conn.srcInst == instIdx && conn.srcPort >= 0 &&
          (unsigned)conn.srcPort < numOut)
        used[conn.srcPort] = true;
    }
    for (const auto &conn : impl_->outputConns) {
      if (conn.instIdx == instIdx && conn.srcPort >= 0 &&
          (unsigned)conn.srcPort < numOut)
        used[conn.srcPort] = true;
    }

    for (unsigned p = 0; p < numOut; ++p) {
      if (!used[p]) {
        std::string loc = impl_->instances[instIdx].name + ":" +
                          std::to_string(p) + " (output)";
        addError("COMP_OUTPUT_DANGLING",
                 "instance output port is not connected to any consumer", loc);
      }
    }
  }

  // Combinational loop detection using a port-level graph. Each node is
  // (combinational_instance, output_port). Edges respect switch connectivity
  // tables: an internal connection srcInst.srcPort -> dstInst.dstPort creates
  // edges from (srcInst, srcPort) to (dstInst, j) for each output j that
  // dstPort can drive per the connectivity table.
  {
    unsigned numInst = impl_->instances.size();

    auto isCombinational = [&](unsigned idx) -> bool {
      switch (impl_->instances[idx].kind) {
      case ModuleKind::Switch:
      case ModuleKind::TemporalSwitch:
      case ModuleKind::AddTag:
      case ModuleKind::MapTag:
      case ModuleKind::DelTag:
        return true;
      default:
        return false;
      }
    };

    auto getOutputCount = [&](unsigned idx) -> unsigned {
      const auto &inst = impl_->instances[idx];
      switch (inst.kind) {
      case ModuleKind::Switch:
        return impl_->switchDefs[inst.defIdx].numOut;
      case ModuleKind::TemporalSwitch:
        return impl_->temporalSwitchDefs[inst.defIdx].numOut;
      case ModuleKind::AddTag:
      case ModuleKind::MapTag:
      case ModuleKind::DelTag:
        return 1;
      default:
        return 0;
      }
    };

    // Check if input port inPort of instance idx can drive output port outPort
    // according to the connectivity table.
    auto canInputDriveOutput =
        [&](unsigned idx, int inPort, unsigned outPort) -> bool {
      const auto &inst = impl_->instances[idx];
      if (inst.kind == ModuleKind::Switch) {
        const auto &def = impl_->switchDefs[inst.defIdx];
        if (def.connectivity.empty())
          return true; // full connectivity
        if (outPort >= def.connectivity.size())
          return false;
        if (inPort < 0 ||
            (unsigned)inPort >= def.connectivity[outPort].size())
          return false;
        return def.connectivity[outPort][inPort];
      }
      if (inst.kind == ModuleKind::TemporalSwitch) {
        const auto &def = impl_->temporalSwitchDefs[inst.defIdx];
        if (def.connectivity.empty())
          return true;
        if (outPort >= def.connectivity.size())
          return false;
        if (inPort < 0 ||
            (unsigned)inPort >= def.connectivity[outPort].size())
          return false;
        return def.connectivity[outPort][inPort];
      }
      // AddTag, MapTag, DelTag: single input/output, always connected.
      return true;
    };

    // Build port-level graph: assign a node to each output port of
    // combinational instances.
    std::vector<unsigned> portBase(numInst, 0);
    std::vector<unsigned> portCount(numInst, 0);
    unsigned totalPorts = 0;
    for (unsigned i = 0; i < numInst; ++i) {
      if (isCombinational(i)) {
        portBase[i] = totalPorts;
        portCount[i] = getOutputCount(i);
        totalPorts += portCount[i];
      }
    }

    if (totalPorts > 0) {
      std::vector<std::vector<unsigned>> adj(totalPorts);
      for (const auto &conn : impl_->internalConns) {
        if (conn.srcInst >= numInst || conn.dstInst >= numInst)
          continue;
        if (!isCombinational(conn.srcInst) || !isCombinational(conn.dstInst))
          continue;
        if (conn.srcPort < 0 ||
            (unsigned)conn.srcPort >= portCount[conn.srcInst])
          continue;
        unsigned srcNode = portBase[conn.srcInst] + (unsigned)conn.srcPort;
        for (unsigned j = 0; j < portCount[conn.dstInst]; ++j) {
          if (canInputDriveOutput(conn.dstInst, conn.dstPort, j))
            adj[srcNode].push_back(portBase[conn.dstInst] + j);
        }
      }

      // DFS cycle detection (0 = unvisited, 1 = in-stack, 2 = done).
      std::vector<int> color(totalPorts, 0);
      bool hasCycle = false;
      std::function<void(unsigned)> dfs = [&](unsigned u) {
        color[u] = 1;
        for (unsigned v : adj[u]) {
          if (color[v] == 1) {
            hasCycle = true;
            return;
          }
          if (color[v] == 0) {
            dfs(v);
            if (hasCycle)
              return;
          }
        }
        color[u] = 2;
      };

      for (unsigned i = 0; i < totalPorts && !hasCycle; ++i) {
        if (color[i] == 0)
          dfs(i);
      }

      if (hasCycle)
        addError("COMP_ADG_COMBINATIONAL_LOOP",
                 "connection graph contains a combinational loop (all elements "
                 "are zero-delay)",
                 "module @" + impl_->moduleName);
    }
  }

  return result;
}

} // namespace adg
} // namespace loom
