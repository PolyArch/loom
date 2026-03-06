//===-- fabric_common.svh - Fabric common definitions ---------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Common definitions shared across all Fabric SystemVerilog modules.
// Includes the streaming interface, error code constants, and assertion guard.
//
//===----------------------------------------------------------------------===//

`ifndef FABRIC_COMMON_SVH
`define FABRIC_COMMON_SVH

//===----------------------------------------------------------------------===//
// Streaming Interface
//===----------------------------------------------------------------------===//

interface fabric_stream #(
    parameter int WIDTH = 32
);
  logic             valid;
  logic             ready;
  logic [WIDTH-1:0] data;

  modport source(output valid, input  ready, output data);
  modport sink  (input  valid, output ready, input  data);
endinterface

//===----------------------------------------------------------------------===//
// Address Bit Width
//===----------------------------------------------------------------------===//
// Centralized address bit-width for index-type ports in the fabric.
//
// IMPORTANT: When changing this value, also update ADDR_BIT_WIDTH in
// include/loom/Hardware/Common/FabricConstants.h to match.
// The spec test check-loom-spec-addrwidth enforces consistency.
`define FABRIC_ADDR_BIT_WIDTH 57

//===----------------------------------------------------------------------===//
// Assertion Guard
//===----------------------------------------------------------------------===//

// Define FABRIC_ASSERTIONS_ON to enable inline SVA. Simulators should define
// this; synthesis tools should leave it undefined.
// Usage: `ifdef FABRIC_ASSERTIONS_ON ... `endif

//===----------------------------------------------------------------------===//
// Hardware Error Codes
//===----------------------------------------------------------------------===//
// CFG_ and RT_ error codes are defined in fabric_error.svh (single source of
// truth for hardware error codes). CPL_ codes are in FabricError.h (C++).
// The normative specification is docs/spec-fabric-error.md.

`include "fabric_error.svh"

`endif // FABRIC_COMMON_SVH
