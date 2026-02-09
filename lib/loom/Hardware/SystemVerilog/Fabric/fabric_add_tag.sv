//===-- fabric_add_tag.sv - Tag attachment module --------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Prepends a configured tag value to the input data stream.
// Combinational: valid/ready pass through with zero latency.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_add_tag #(
    parameter int DATA_WIDTH = 32,
    parameter int TAG_WIDTH  = 4,
    localparam int IN_PW     = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    localparam int OUT_PW    = (DATA_WIDTH + TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : 1,
    localparam int CONFIG_WIDTH = TAG_WIDTH
) (
    input  logic                clk,
    input  logic                rst_n,

    // Streaming input (untagged)
    input  logic                in_valid,
    output logic                in_ready,
    input  logic [IN_PW-1:0]   in_data,

    // Streaming output (tagged)
    output logic                out_valid,
    input  logic                out_ready,
    output logic [OUT_PW-1:0]   out_data,

    // Configuration: tag value to prepend
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation (COMP_ errors)
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (TAG_WIDTH < 1)
      $fatal(1, "COMP_ADD_TAG_TAG_WIDTH: TAG_WIDTH must be >= 1");
    if (DATA_WIDTH < 0)
      $fatal(1, "COMP_ADD_TAG_DATA_WIDTH: DATA_WIDTH must be >= 0");
  end

  // -----------------------------------------------------------------------
  // Combinational pass-through with tag prepend
  // -----------------------------------------------------------------------
  assign out_valid = in_valid;
  assign in_ready  = out_ready;
  assign out_data  = {cfg_data[TAG_WIDTH-1:0], in_data[DATA_WIDTH-1:0]};

endmodule
