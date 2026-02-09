//===-- fabric_del_tag.sv - Tag removal module ----------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Strips the tag portion from tagged input data, passing only the value.
// Combinational: valid/ready pass through with zero latency.
// No configuration ports (CONFIG_WIDTH = 0).
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_del_tag #(
    parameter int DATA_WIDTH = 32,
    parameter int TAG_WIDTH  = 4,
    localparam int IN_PW     = (DATA_WIDTH + TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : 1,
    localparam int OUT_PW    = (DATA_WIDTH > 0) ? DATA_WIDTH : 1
) (
    input  logic                clk,
    input  logic                rst_n,

    // Streaming input (tagged)
    input  logic                in_valid,
    output logic                in_ready,
    input  logic [IN_PW-1:0]   in_data,

    // Streaming output (untagged)
    output logic                out_valid,
    input  logic                out_ready,
    output logic [OUT_PW-1:0]   out_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation (COMP_ errors)
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (TAG_WIDTH < 1)
      $fatal(1, "COMP_DEL_TAG_TAG_WIDTH: TAG_WIDTH must be >= 1");
    if (DATA_WIDTH < 1)
      $fatal(1, "COMP_DEL_TAG_DATA_WIDTH: DATA_WIDTH must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Combinational pass-through with tag strip
  // -----------------------------------------------------------------------
  assign out_valid = in_valid;
  assign in_ready  = out_ready;
  assign out_data  = in_data[DATA_WIDTH-1:0];

endmodule
