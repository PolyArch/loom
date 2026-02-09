//===-- fabric_pe_store.sv - Store PE module -------------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Memory store adapter PE. Synchronizes address + data + control,
// forwards address and data to memory.
//
// TagOverwrite mode: synchronize addr+data+ctrl, attach output_tag.
// TagTransparent mode: tag-match addr+data+ctrl, forward unchanged.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_pe_store #(
    parameter int DATA_WIDTH  = 32,
    parameter int TAG_WIDTH   = 0,
    parameter int HW_TYPE     = 0,   // 0=TagOverwrite, 1=TagTransparent
    parameter int QUEUE_DEPTH = 4,
    localparam int ADDR_PW    = (DATA_WIDTH + TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : 1,
    localparam int DATA_PW    = (DATA_WIDTH + TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : 1,
    localparam int SAFE_DW    = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    // Done signal width: tagged<none, iK> = TAG_WIDTH bits, else 1 bit
    localparam int DONE_PW    = (TAG_WIDTH > 0) ? TAG_WIDTH : 1,
    // Config: output_tag for TagOverwrite+tagged, else 0
    localparam int CONFIG_WIDTH = (HW_TYPE == 0 && TAG_WIDTH > 0) ? TAG_WIDTH : 0
) (
    input  logic               clk,
    input  logic               rst_n,

    // Input 0: address from compute
    input  logic               in0_valid,
    output logic               in0_ready,
    input  logic [ADDR_PW-1:0] in0_data,

    // Input 1: data from compute
    input  logic               in1_valid,
    output logic               in1_ready,
    input  logic [DATA_PW-1:0] in1_data,

    // Input 2: control token
    input  logic               in2_valid,
    output logic               in2_ready,
    input  logic [ADDR_PW-1:0] in2_data,

    // Output 0: address to memory (index type, tagged when TAG_WIDTH > 0)
    output logic               out0_valid,
    input  logic               out0_ready,
    output logic [ADDR_PW-1:0] out0_data,

    // Output 1: done signal (tagged<none, iK> when tagged, else 1-bit)
    output logic               out1_valid,
    input  logic               out1_ready,
    output logic [DONE_PW-1:0] out1_data,

    // Configuration
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (DATA_WIDTH < 1)
      $fatal(1, "COMP_PE_STORE_DATA_WIDTH: DATA_WIDTH must be >= 1");
    if (HW_TYPE != 0 && HW_TYPE != 1)
      $fatal(1, "COMP_PE_LOADSTORE_TAG_MODE: HW_TYPE must be 0 or 1");
    if (HW_TYPE == 1 && TAG_WIDTH == 0)
      $fatal(1, "COMP_PE_LOADSTORE_TAG_WIDTH: TagTransparent requires TAG_WIDTH > 0");
  end

  // -----------------------------------------------------------------------
  // Synchronize addr + data + ctrl, forward to memory
  // -----------------------------------------------------------------------
  generate
    if (HW_TYPE == 0) begin : g_overwrite
      // Synchronize addr (in0) + data (in1) + ctrl (in2)
      logic all_valid;
      assign all_valid = in0_valid && in1_valid && in2_valid;

      logic [SAFE_DW-1:0] addr_value;
      assign addr_value = in0_data[DATA_WIDTH-1:0];

      logic both_out_ready;
      assign both_out_ready = out0_ready && out1_ready;

      logic fire;
      assign fire = all_valid && both_out_ready;

      // out0: address to memory (with tag if tagged)
      assign out0_valid = all_valid && out1_ready;
      // out1: done signal (with tag if tagged)
      assign out1_valid = all_valid && out0_ready;

      if (TAG_WIDTH > 0) begin : g_tag_out
        logic [TAG_WIDTH-1:0] output_tag;
        assign output_tag = cfg_data[TAG_WIDTH-1:0];
        assign out0_data  = {output_tag, addr_value};
        assign out1_data  = output_tag;
      end else begin : g_no_tag_out
        assign out0_data  = addr_value;
        assign out1_data  = 1'b0;
      end

      assign in0_ready = fire;
      assign in1_ready = fire;
      assign in2_ready = fire;
    end else begin : g_transparent
      // TagTransparent: tag-match addr+data+ctrl
      logic [TAG_WIDTH-1:0] addr_tag, data_tag, ctrl_tag;
      assign addr_tag = in0_data[DATA_WIDTH +: TAG_WIDTH];
      assign data_tag = in1_data[DATA_WIDTH +: TAG_WIDTH];
      assign ctrl_tag = in2_data[TAG_WIDTH-1:0];

      logic tags_match;
      assign tags_match = (addr_tag == data_tag) && (addr_tag == ctrl_tag);

      logic all_valid;
      assign all_valid = in0_valid && in1_valid && in2_valid && tags_match;

      logic [SAFE_DW-1:0] addr_value;
      assign addr_value = in0_data[DATA_WIDTH-1:0];

      logic both_out_ready;
      assign both_out_ready = out0_ready && out1_ready;

      logic fire;
      assign fire = all_valid && both_out_ready;

      // out0: address to memory (with tag)
      assign out0_valid = all_valid && out1_ready;
      assign out0_data  = {addr_tag, addr_value};
      // out1: done signal (with tag)
      assign out1_valid = all_valid && out0_ready;
      assign out1_data  = addr_tag;

      assign in0_ready = fire;
      assign in1_ready = fire;
      assign in2_ready = fire;
    end
  endgenerate

endmodule
