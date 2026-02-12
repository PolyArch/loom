//===-- fabric_pe_constant.sv - Constant PE module -------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Emits a configured constant value on each control token.
// Output = cfg_data[DATA_WIDTH-1:0]. If tagged, output_tag from cfg_data MSBs.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_pe_constant #(
    parameter int DATA_WIDTH = 32,
    parameter int TAG_WIDTH  = 0,
    localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW       = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    // Config: constant value + optional output tag
    localparam int CONFIG_WIDTH  = (TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : DATA_WIDTH
) (
    input  logic                clk,
    input  logic                rst_n,

    // Control input (trigger)
    input  logic                in_valid,
    output logic                in_ready,
    input  logic [SAFE_PW-1:0]  in_data,

    // Constant output
    output logic                out_valid,
    input  logic                out_ready,
    output logic [SAFE_PW-1:0]  out_data,

    // Configuration: constant value [DATA_WIDTH-1:0], output_tag [CONFIG_WIDTH-1:DATA_WIDTH]
    input  logic [CONFIG_WIDTH-1:0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (DATA_WIDTH < 1)
      $fatal(1, "CPL_PE_CONSTANT_DATA_WIDTH: DATA_WIDTH must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Output logic: fire on control token
  // -----------------------------------------------------------------------
  assign out_valid = in_valid;
  assign in_ready  = out_ready;

  generate
    if (TAG_WIDTH > 0) begin : g_tagged
      logic [TAG_WIDTH-1:0]  output_tag;
      logic [DATA_WIDTH-1:0] const_value;
      assign const_value = cfg_data[DATA_WIDTH-1:0];
      assign output_tag  = cfg_data[CONFIG_WIDTH-1:DATA_WIDTH];
      assign out_data    = {output_tag, const_value};
    end else begin : g_native
      assign out_data = cfg_data[DATA_WIDTH-1:0];
    end
  endgenerate

endmodule
