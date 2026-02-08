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
// Assertion Guard
//===----------------------------------------------------------------------===//

// Define FABRIC_ASSERTIONS_ON to enable inline SVA. Simulators should define
// this; synthesis tools should leave it undefined.
// Usage: `ifdef FABRIC_ASSERTIONS_ON ... `endif

//===----------------------------------------------------------------------===//
// Hardware Error Codes (from spec-fabric-error.md)
//===----------------------------------------------------------------------===//

// CFG_ errors: runtime configuration errors
localparam logic [15:0] FABRIC_OK                         = 16'd0;
localparam logic [15:0] CFG_SWITCH_ROUTE_MULTI_OUT        = 16'd1;
localparam logic [15:0] CFG_SWITCH_ROUTE_MULTI_IN         = 16'd2;
localparam logic [15:0] CFG_TEMPORAL_SW_ROUTE_MULTI_OUT   = 16'd3;
localparam logic [15:0] CFG_TEMPORAL_SW_ROUTE_MULTI_IN    = 16'd4;
localparam logic [15:0] CFG_TEMPORAL_SW_DUP_TAG           = 16'd5;
localparam logic [15:0] CFG_TEMPORAL_PE_DUP_TAG           = 16'd6;
localparam logic [15:0] CFG_TEMPORAL_PE_ILLEGAL_REG       = 16'd7;
localparam logic [15:0] CFG_TEMPORAL_PE_REG_TAG_NONZERO   = 16'd8;
localparam logic [15:0] CFG_MAP_TAG_DUP_TAG               = 16'd9;
localparam logic [15:0] CFG_PE_STREAM_CONT_COND_ONEHOT    = 16'd10;

// RT_ errors: runtime execution errors
localparam logic [15:0] RT_TEMPORAL_PE_NO_MATCH           = 16'd256;
localparam logic [15:0] RT_TEMPORAL_SW_NO_MATCH           = 16'd257;
localparam logic [15:0] RT_MAP_TAG_NO_MATCH               = 16'd258;
localparam logic [15:0] RT_DATAFLOW_STREAM_ZERO_STEP      = 16'd259;
localparam logic [15:0] RT_MEMORY_TAG_OOB                 = 16'd260;
localparam logic [15:0] RT_MEMORY_STORE_DEADLOCK          = 16'd261;
localparam logic [15:0] RT_SWITCH_UNROUTED_INPUT          = 16'd262;
localparam logic [15:0] RT_TEMPORAL_SW_UNROUTED_INPUT     = 16'd263;

`endif // FABRIC_COMMON_SVH
