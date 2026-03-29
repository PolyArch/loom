//===- FabricSystemOps.cpp - System-level op implementations ----*- C++ -*-===//
//
// Verifiers for RouterOp, SharedMemOp, and NoCLinkOp.
//
//===----------------------------------------------------------------------===//

#include "FabricOpsInternal.h"

using namespace mlir;
using namespace loom::fabric;

//===----------------------------------------------------------------------===//
// RouterOp
//===----------------------------------------------------------------------===//

LogicalResult RouterOp::verify() {
  int64_t numPorts = getNumPorts();
  if (numPorts <= 0)
    return emitOpError("num_ports must be positive, got ")
           << numPorts;
  if (numPorts > 64)
    return emitOpError("num_ports exceeds maximum (64), got ")
           << numPorts;

  int64_t vc = getVirtualChannels();
  if (vc <= 0)
    return emitOpError("virtual_channels must be positive, got ")
           << vc;

  int64_t bufDepth = getBufferDepth();
  if (bufDepth <= 0)
    return emitOpError("buffer_depth must be positive, got ")
           << bufDepth;

  int64_t pipelineStages = getPipelineStages();
  if (pipelineStages < 0)
    return emitOpError("pipeline_stages must be non-negative, got ")
           << pipelineStages;

  int64_t flitWidth = getFlitWidthBits();
  if (flitWidth <= 0)
    return emitOpError("flit_width_bits must be positive, got ")
           << flitWidth;

  llvm::StringRef routingStrategy = getRoutingStrategy();
  if (routingStrategy != "xy_dor" && routingStrategy != "yx_dor" &&
      routingStrategy != "adaptive")
    return emitOpError("routing_strategy must be \"xy_dor\", \"yx_dor\", or "
                       "\"adaptive\", got \"")
           << routingStrategy << "\"";

  llvm::StringRef topologyRole = getTopologyRole();
  if (topologyRole != "mesh" && topologyRole != "ring" &&
      topologyRole != "hierarchical")
    return emitOpError("topology_role must be \"mesh\", \"ring\", or "
                       "\"hierarchical\", got \"")
           << topologyRole << "\"";

  return success();
}

//===----------------------------------------------------------------------===//
// SharedMemOp
//===----------------------------------------------------------------------===//

LogicalResult SharedMemOp::verify() {
  int64_t sizeBytes = getSizeBytes();
  if (sizeBytes <= 0)
    return emitOpError("size_bytes must be positive, got ")
           << sizeBytes;

  int64_t widthBytes = getWidthBytes();
  if (widthBytes <= 0)
    return emitOpError("width_bytes must be positive, got ")
           << widthBytes;

  int64_t numBanks = getNumBanks();
  if (numBanks <= 0)
    return emitOpError("num_banks must be positive, got ")
           << numBanks;

  llvm::StringRef memType = getMemType();
  if (memType != "l2_cache" && memType != "external_dram")
    return emitOpError("mem_type must be \"l2_cache\" or \"external_dram\", "
                       "got \"")
           << memType << "\"";

  int64_t portCount = getPortCount();
  if (portCount <= 0)
    return emitOpError("port_count must be positive, got ")
           << portCount;

  return success();
}

//===----------------------------------------------------------------------===//
// NoCLinkOp
//===----------------------------------------------------------------------===//

LogicalResult NoCLinkOp::verify() {
  int64_t sourcePort = getSourcePort();
  if (sourcePort < 0)
    return emitOpError("source_port must be non-negative, got ")
           << sourcePort;

  int64_t destPort = getDestPort();
  if (destPort < 0)
    return emitOpError("dest_port must be non-negative, got ")
           << destPort;

  int64_t widthBits = getWidthBits();
  if (widthBits <= 0)
    return emitOpError("width_bits must be positive, got ")
           << widthBits;

  int64_t latencyCycles = getLatencyCycles();
  if (latencyCycles < 0)
    return emitOpError("latency_cycles must be non-negative, got ")
           << latencyCycles;

  int64_t bw = getBandwidth();
  if (bw <= 0)
    return emitOpError("bandwidth must be positive, got ")
           << bw;

  return success();
}
