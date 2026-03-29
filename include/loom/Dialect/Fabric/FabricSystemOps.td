#ifndef LOOM_DIALECT_FABRIC_FABRICSYSTEMOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICSYSTEMOPS_TD

include "loom/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

// fabric.router - NoC router node
def Fabric_RouterOp : Fabric_Op<"router", [Symbol]> {
  let summary = "NoC router node";
  let description = [{
    Configurable NoC router with ports, virtual channels, and buffer depth.
    Represents a router in the system-level NoC topology. Each router has
    a specified number of ports (typically 5 for mesh: N/S/E/W + local),
    virtual channels for flow control, and configurable pipeline depth.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$num_ports,
    I64Attr:$virtual_channels,
    I64Attr:$buffer_depth,
    I64Attr:$pipeline_stages,
    I64Attr:$flit_width_bits,
    StrAttr:$routing_strategy,
    StrAttr:$topology_role
  );
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
  let hasVerifier = 1;
}

// fabric.shared_mem - Shared memory bank (L2 cache or external memory)
def Fabric_SharedMemOp : Fabric_Op<"shared_mem", [Symbol]> {
  let summary = "Shared memory bank (L2 cache or external memory)";
  let description = [{
    Represents a shared memory resource at the system level. This includes
    L2 cache banks and external DRAM interfaces. Distinct from per-core
    fabric.memory (SPM) and fabric.extmemory.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$size_bytes,
    I64Attr:$width_bytes,
    I64Attr:$num_banks,
    StrAttr:$mem_type,
    I64Attr:$port_count
  );
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
  let hasVerifier = 1;
}

// fabric.noc_link - Directed connection between two system components
def Fabric_NoCLinkOp : Fabric_Op<"noc_link", []> {
  let summary = "Directed connection between two system components";
  let description = [{
    Represents a directed link in the NoC topology connecting two components
    (cores, routers, or shared memory). Each link has source/destination
    identifiers with port indices, a data width, and a latency.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$source,
    I64Attr:$source_port,
    FlatSymbolRefAttr:$dest,
    I64Attr:$dest_port,
    I64Attr:$width_bits,
    I64Attr:$latency_cycles,
    I64Attr:$bandwidth
  );
  let assemblyFormat = [{
    attr-dict
  }];
  let hasVerifier = 1;
}

#endif
