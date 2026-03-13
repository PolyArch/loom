//===-- SimModule.cpp - Module factory --------------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimModule.h"
#include "loom/Simulator/SimFifo.h"
#include "loom/Simulator/SimMemory.h"
#include "loom/Simulator/SimPE.h"
#include "loom/Simulator/SimSwitch.h"
#include "loom/Simulator/SimTagOps.h"
#include "loom/Simulator/SimTemporalPE.h"
#include "loom/Simulator/SimTemporalSW.h"

namespace loom {
namespace sim {

namespace {

int64_t getIntAttr(
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::string &name, int64_t dflt = 0) {
  for (auto &[k, v] : intAttrs) {
    if (k == name)
      return v;
  }
  return dflt;
}

std::string getStrAttr(
    const std::vector<std::pair<std::string, std::string>> &strAttrs,
    const std::string &name, const std::string &dflt = "") {
  for (auto &[k, v] : strAttrs) {
    if (k == name)
      return v;
  }
  return dflt;
}

bool hasAttr(
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::string &name) {
  for (auto &[k, v] : intAttrs) {
    if (k == name)
      return true;
  }
  return false;
}

const std::vector<int8_t> *getArrayAttr(
    const std::vector<std::pair<std::string, std::vector<int8_t>>> &arrayAttrs,
    const std::string &name) {
  for (auto &[k, v] : arrayAttrs) {
    if (k == name)
      return &v;
  }
  return nullptr;
}

/// Decode a flat connectivity_table array (output-major, nOut*nIn) into a
/// vector<bool>. Returns a fully-connected table if the attribute is absent
/// or the size doesn't match.
std::vector<bool> decodeConnTable(
    const std::vector<std::pair<std::string, std::vector<int8_t>>> &arrayAttrs,
    unsigned nOut, unsigned nIn) {
  const auto *arr = getArrayAttr(arrayAttrs, "connectivity_table");
  if (arr && arr->size() == static_cast<size_t>(nOut) * nIn) {
    std::vector<bool> conn(nOut * nIn);
    for (size_t i = 0; i < arr->size(); ++i)
      conn[i] = ((*arr)[i] != 0);
    return conn;
  }
  // Default: fully connected.
  return std::vector<bool>(nOut * nIn, true);
}

/// Ceiling of log2 for positive integers. Returns 0 for x <= 1.
unsigned log2Ceil(unsigned x) {
  if (x <= 1)
    return 0;
  unsigned r = 0;
  unsigned v = x - 1;
  while (v > 0) {
    v >>= 1;
    ++r;
  }
  return r;
}

} // namespace

unsigned computeConfigWidth(
    const std::string &opName, unsigned numInputs, unsigned numOutputs,
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::vector<std::pair<std::string, std::string>> &strAttrs,
    const std::vector<std::pair<std::string, std::vector<int8_t>>>
        &arrayAttrs) {

  if (opName == "fabric.switch") {
    unsigned nIn =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_inputs", numInputs));
    unsigned nOut =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_outputs", numOutputs));
    auto conn = decodeConnTable(arrayAttrs, nOut, nIn);
    unsigned K = 0;
    for (bool c : conn)
      if (c)
        ++K;
    return K;
  }

  if (opName == "fabric.fifo") {
    bool bypassable = hasAttr(intAttrs, "bypassable");
    return bypassable ? 1 : 0;
  }

  if (opName == "fabric.add_tag")
    return static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));

  if (opName == "fabric.map_tag") {
    unsigned inTW =
        static_cast<unsigned>(getIntAttr(intAttrs, "in_tag_width", 4));
    unsigned outTW =
        static_cast<unsigned>(getIntAttr(intAttrs, "out_tag_width", 4));
    unsigned tableSize =
        static_cast<unsigned>(getIntAttr(intAttrs, "table_size", 4));
    return tableSize * (1 + inTW + outTW);
  }

  if (opName == "fabric.del_tag")
    return 0;

  if (opName == "fabric.memory" || opName == "fabric.extmemory") {
    // Per spec-fabric-mem.md: CONFIG_WIDTH = numRegion * (1 + 2*TW + AW).
    unsigned numRegion =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_region", 1));
    unsigned tagWidth =
        static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned addrWidth =
        static_cast<unsigned>(getIntAttr(intAttrs, "addr_width", 32));
    return numRegion * (1 + 2 * tagWidth + addrWidth);
  }

  if (opName == "fabric.temporal_sw") {
    unsigned tagWidth =
        static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned numRouteTable =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_route_table", 4));
    unsigned nIn =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_inputs", numInputs));
    unsigned nOut =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_outputs", numOutputs));
    auto conn = decodeConnTable(arrayAttrs, nOut, nIn);
    unsigned K = 0;
    for (bool c : conn)
      if (c)
        ++K;
    unsigned slotWidth = 1 + tagWidth + K;
    return numRouteTable * slotWidth;
  }

  if (opName == "fabric.temporal_pe") {
    unsigned tagWidth =
        static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned numInsns =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_instructions", 4));
    unsigned numRegs =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_registers", 0));
    unsigned numFuTypes =
        static_cast<unsigned>(getIntAttr(intAttrs, "num_fu_types", 1));

    unsigned O = log2Ceil(numFuTypes);
    unsigned opcW = (numRegs > 0) ? (1 + log2Ceil(numRegs)) : 0;
    unsigned resW = opcW + tagWidth;
    unsigned insnWidth =
        1 + tagWidth + O + numInputs * opcW + numOutputs * resW;

    // Per-FU cmp ops contribute 4 bits each. The per-FU cmp count is not
    // available from ADG graph attributes at this level; default to 0.
    // If FU cmp config is needed, the ADG should provide num_cmp_ops_per_fu.
    unsigned cmpBits = 0;
    return cmpBits + numInsns * insnWidth;
  }

  if (opName == "fabric.pe") {
    std::string bodyOp = getStrAttr(strAttrs, "body_op", "arith.addi");
    std::string resClass = getStrAttr(strAttrs, "resource_class", "compute");
    bool isTagged = hasAttr(intAttrs, "output_tag") ||
                    getIntAttr(intAttrs, "is_tagged", 0) != 0;
    unsigned tagWidth =
        static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned dataWidth =
        static_cast<unsigned>(getIntAttr(intAttrs, "data_width", 32));

    bool isCmp = (bodyOp == "arith.cmpi" || bodyOp == "arith.cmpf");
    unsigned numCmpOps = isCmp ? 1 : 0;

    // Constant PE.
    if (resClass == "constant") {
      if (isTagged && numOutputs == 1)
        return dataWidth + tagWidth;
      return dataWidth;
    }

    // Load/Store PE.
    if (resClass == "load" || resClass == "store") {
      std::string tagMode = getStrAttr(strAttrs, "tag_mode", "native");
      if (tagMode == "tag_overwrite" && isTagged)
        return tagWidth;
      return 0;
    }

    // dataflow.stream PE.
    if (bodyOp == "dataflow.stream") {
      if (isTagged)
        return numOutputs * tagWidth + 5;
      return 5;
    }

    // General compute (including dataflow.carry, invariant, gate).
    if (isTagged)
      return numOutputs * tagWidth + 4 * numCmpOps;
    return 4 * numCmpOps;
  }

  return 0;
}

std::unique_ptr<SimModule> createSimModule(
    uint32_t hwNodeId, const std::string &name, const std::string &opName,
    unsigned numInputs, unsigned numOutputs,
    const std::vector<std::pair<std::string, int64_t>> &intAttrs,
    const std::vector<std::pair<std::string, std::string>> &strAttrs,
    const std::vector<std::pair<std::string, std::vector<int8_t>>>
        &arrayAttrs) {

  std::unique_ptr<SimModule> mod;

  if (opName == "fabric.switch") {
    // Build connectivity table from attributes.
    unsigned nIn = static_cast<unsigned>(getIntAttr(intAttrs, "num_inputs", numInputs));
    unsigned nOut = static_cast<unsigned>(getIntAttr(intAttrs, "num_outputs", numOutputs));
    std::vector<bool> conn = decodeConnTable(arrayAttrs, nOut, nIn);
    mod = std::make_unique<SimSwitch>(nIn, nOut, conn);
  } else if (opName == "fabric.fifo") {
    unsigned depth = static_cast<unsigned>(getIntAttr(intAttrs, "depth", 2));
    bool bypassable = hasAttr(intAttrs, "bypassable");
    mod = std::make_unique<SimFifo>(depth, bypassable);
  } else if (opName == "fabric.add_tag") {
    unsigned tagWidth = static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    mod = std::make_unique<SimAddTag>(tagWidth);
  } else if (opName == "fabric.map_tag") {
    unsigned inTW = static_cast<unsigned>(getIntAttr(intAttrs, "in_tag_width", 4));
    unsigned outTW = static_cast<unsigned>(getIntAttr(intAttrs, "out_tag_width", 4));
    unsigned tableSize = static_cast<unsigned>(getIntAttr(intAttrs, "table_size", 4));
    mod = std::make_unique<SimMapTag>(inTW, outTW, tableSize);
  } else if (opName == "fabric.del_tag") {
    mod = std::make_unique<SimDelTag>();
  } else if (opName == "fabric.pe") {
    std::string bodyOp = getStrAttr(strAttrs, "body_op", "arith.addi");
    std::string resClass = getStrAttr(strAttrs, "resource_class", "compute");
    bool isTagged = hasAttr(intAttrs, "output_tag") ||
                    getIntAttr(intAttrs, "is_tagged", 0) != 0;
    unsigned tagWidth = static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned dataWidth = static_cast<unsigned>(getIntAttr(intAttrs, "data_width", 32));

    SimPE::BodyType bodyType = SimPE::BodyType::Compute;
    SimPE::TagMode tagMode = SimPE::TagMode::Native;

    if (resClass == "constant") {
      bodyType = SimPE::BodyType::Constant;
    } else if (resClass == "load") {
      bodyType = SimPE::BodyType::Load;
      std::string tm = getStrAttr(strAttrs, "tag_mode", "native");
      if (tm == "tag_overwrite")
        tagMode = SimPE::TagMode::TagOverwrite;
      else if (tm == "tag_transparent")
        tagMode = SimPE::TagMode::TagTransparent;
    } else if (resClass == "store") {
      bodyType = SimPE::BodyType::Store;
      std::string tm = getStrAttr(strAttrs, "tag_mode", "native");
      if (tm == "tag_overwrite")
        tagMode = SimPE::TagMode::TagOverwrite;
      else if (tm == "tag_transparent")
        tagMode = SimPE::TagMode::TagTransparent;
    } else if (bodyOp == "dataflow.stream") {
      bodyType = SimPE::BodyType::StreamCont;
    } else if (bodyOp == "dataflow.carry") {
      bodyType = SimPE::BodyType::Carry;
    } else if (bodyOp == "dataflow.invariant") {
      bodyType = SimPE::BodyType::Invariant;
    } else if (bodyOp == "dataflow.gate") {
      bodyType = SimPE::BodyType::Gate;
    }

    auto pe = std::make_unique<SimPE>(bodyType, numInputs, numOutputs, isTagged,
                                      tagWidth, dataWidth, bodyOp, tagMode);
    // Plumb step_op from ADG attribute into the stream PE.
    if (bodyType == SimPE::BodyType::StreamCont) {
      std::string stepOp = getStrAttr(strAttrs, "step_op", "+=");
      pe->setStepOp(stepOp);
    }
    mod = std::move(pe);
  }
  else if (opName == "fabric.temporal_pe") {
    unsigned tagWidth = static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned numInsns = static_cast<unsigned>(getIntAttr(intAttrs, "num_instructions", 4));
    unsigned numRegs = static_cast<unsigned>(getIntAttr(intAttrs, "num_registers", 0));
    unsigned regFifoDepth = static_cast<unsigned>(getIntAttr(intAttrs, "reg_fifo_depth", 4));
    unsigned numFuTypes = static_cast<unsigned>(getIntAttr(intAttrs, "num_fu_types", 1));
    unsigned valueWidth = static_cast<unsigned>(getIntAttr(intAttrs, "data_width", 32));
    bool sharedBuf = getIntAttr(intAttrs, "shared_operand_buffer", 0) != 0;
    unsigned bufSize = static_cast<unsigned>(getIntAttr(intAttrs, "operand_buffer_size", numInsns));
    auto tpe = std::make_unique<SimTemporalPE>(numInputs, numOutputs, tagWidth,
                                                numInsns, numRegs, regFifoDepth,
                                                numFuTypes, valueWidth, sharedBuf,
                                                bufSize);
    // Set per-FU body ops from ADG attributes (fu_body_op_N) if available.
    for (unsigned f = 0; f < numFuTypes; ++f) {
      std::string key = "fu_body_op_" + std::to_string(f);
      std::string fuOp = getStrAttr(strAttrs, key, "arith.addi");
      std::string dwKey = "fu_data_width_" + std::to_string(f);
      unsigned fuDw = static_cast<unsigned>(
          getIntAttr(intAttrs, dwKey, static_cast<int64_t>(valueWidth)));
      tpe->setFUDescriptor(f, fuOp, fuDw);
    }
    mod = std::move(tpe);
  } else if (opName == "fabric.temporal_sw") {
    unsigned tagWidth = static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned numRouteTable = static_cast<unsigned>(getIntAttr(intAttrs, "num_route_table", 4));
    unsigned nIn = static_cast<unsigned>(getIntAttr(intAttrs, "num_inputs", numInputs));
    unsigned nOut = static_cast<unsigned>(getIntAttr(intAttrs, "num_outputs", numOutputs));
    std::vector<bool> conn = decodeConnTable(arrayAttrs, nOut, nIn);
    mod = std::make_unique<SimTemporalSW>(nIn, nOut, tagWidth, numRouteTable, conn);
  } else if (opName == "fabric.memory" || opName == "fabric.extmemory") {
    bool isExternal = (opName == "fabric.extmemory");
    unsigned ldCount = static_cast<unsigned>(getIntAttr(intAttrs, "ld_count", 1));
    unsigned stCount = static_cast<unsigned>(getIntAttr(intAttrs, "st_count", 1));
    unsigned dataWidth = static_cast<unsigned>(getIntAttr(intAttrs, "data_width", 32));
    unsigned tagWidth = static_cast<unsigned>(getIntAttr(intAttrs, "tag_width", 4));
    unsigned addrWidth = static_cast<unsigned>(getIntAttr(intAttrs, "addr_width", 32));
    unsigned numRegion = static_cast<unsigned>(getIntAttr(intAttrs, "num_region", 1));
    // Use ext_latency from node attributes if present; otherwise 0.
    // Zero-latency simplifies handshake: load response is combinational,
    // avoiding pipeline-induced deadlocks in the dataflow graph.
    uint32_t extLatency = static_cast<uint32_t>(getIntAttr(intAttrs, "ext_latency", 0));
    mod = std::make_unique<SimMemory>(isExternal, ldCount, stCount, dataWidth,
                                       tagWidth, addrWidth, numRegion,
                                       extLatency);
  }

  if (mod) {
    mod->hwNodeId = hwNodeId;
    mod->name = name;
    mod->opName = opName;
  }

  return mod;
}

} // namespace sim
} // namespace loom
