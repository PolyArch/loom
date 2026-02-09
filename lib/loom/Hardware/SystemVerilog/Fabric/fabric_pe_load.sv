//===-- fabric_pe_load.sv - Load PE module ---------------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Memory load adapter PE. Synchronizes address + control, forwards address
// to memory, returns memory data to compute output.
//
// TagOverwrite mode: synchronize addr+ctrl, forward addr, attach output_tag.
// TagTransparent mode: tag-match addr+ctrl, queue management.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_pe_load #(
    parameter int DATA_WIDTH  = 32,
    parameter int TAG_WIDTH   = 0,
    parameter int HW_TYPE     = 0,   // 0=TagOverwrite, 1=TagTransparent
    parameter int QUEUE_DEPTH = 4,
    localparam int ADDR_PW    = (DATA_WIDTH + TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : 1,
    localparam int DATA_PW    = (DATA_WIDTH + TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : 1,
    localparam int SAFE_DW    = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    // Config: output_tag for TagOverwrite+tagged, else 0
    localparam int CONFIG_WIDTH = (HW_TYPE == 0 && TAG_WIDTH > 0) ? TAG_WIDTH : 0
) (
    input  logic               clk,
    input  logic               rst_n,

    // Input 0: address from compute (index type, possibly tagged)
    input  logic               in0_valid,
    output logic               in0_ready,
    input  logic [ADDR_PW-1:0] in0_data,

    // Input 1: data from memory (does NOT participate in synchronization)
    input  logic               in1_valid,
    output logic               in1_ready,
    input  logic [SAFE_DW-1:0] in1_data,

    // Input 2: control token (none type, possibly tagged)
    input  logic               in2_valid,
    output logic               in2_ready,
    input  logic [ADDR_PW-1:0] in2_data,

    // Output 0: data to compute (dataType, possibly tagged)
    output logic               out0_valid,
    input  logic               out0_ready,
    output logic [DATA_PW-1:0] out0_data,

    // Output 1: address to memory (index type, untagged)
    output logic               out1_valid,
    input  logic               out1_ready,
    output logic [SAFE_DW-1:0] out1_data,

    // Configuration
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (DATA_WIDTH < 1)
      $fatal(1, "COMP_PE_LOAD_DATA_WIDTH: DATA_WIDTH must be >= 1");
    if (HW_TYPE != 0 && HW_TYPE != 1)
      $fatal(1, "COMP_PE_LOADSTORE_TAG_MODE: HW_TYPE must be 0 or 1");
    if (HW_TYPE == 1 && TAG_WIDTH == 0)
      $fatal(1, "COMP_PE_LOADSTORE_TAG_WIDTH: TagTransparent requires TAG_WIDTH > 0");
  end

  // -----------------------------------------------------------------------
  // TagOverwrite mode: synchronize addr + ctrl, forward addr to memory
  // -----------------------------------------------------------------------
  generate
    if (HW_TYPE == 0) begin : g_overwrite
      // Synchronize addr (in0) + ctrl (in2); in1 = data from memory
      logic sync_valid;
      assign sync_valid = in0_valid && in2_valid;

      // Extract address value (strip tag if present)
      logic [SAFE_DW-1:0] addr_value;
      assign addr_value = in0_data[DATA_WIDTH-1:0];

      // Forward address to memory (out1)
      assign out1_valid = sync_valid;
      assign out1_data  = addr_value;

      logic fire;
      assign fire = sync_valid && out1_ready;
      assign in0_ready = fire;
      assign in2_ready = fire;

      // Forward memory data (in1) to compute (out0), attaching output_tag
      if (TAG_WIDTH > 0) begin : g_tag_attach
        logic [TAG_WIDTH-1:0] output_tag;
        assign output_tag = cfg_data[TAG_WIDTH-1:0];
        assign out0_data  = {output_tag, in1_data};
      end else begin : g_no_tag
        assign out0_data = in1_data;
      end
      assign out0_valid = in1_valid;
      assign in1_ready  = out0_ready;
    end else begin : g_transparent
      // TagTransparent: tag-match addr (in0) + ctrl (in2), forward unchanged
      logic [TAG_WIDTH-1:0] addr_tag, ctrl_tag;
      assign addr_tag = in0_data[DATA_WIDTH +: TAG_WIDTH];
      assign ctrl_tag = in2_data[TAG_WIDTH-1:0]; // ctrl is tagged<none, iK>

      logic tags_match;
      assign tags_match = (addr_tag == ctrl_tag);

      logic sync_valid;
      assign sync_valid = in0_valid && in2_valid && tags_match;

      logic [SAFE_DW-1:0] addr_value;
      assign addr_value = in0_data[DATA_WIDTH-1:0];

      // Forward address to memory (out1)
      assign out1_valid = sync_valid;
      assign out1_data  = addr_value;

      logic fire;
      assign fire = sync_valid && out1_ready;
      assign in0_ready = fire;
      assign in2_ready = fire;

      // Forward memory data (in1) with original tag to compute (out0)
      assign out0_data  = {addr_tag, in1_data};
      assign out0_valid = in1_valid;
      assign in1_ready  = out0_ready;
    end
  endgenerate

endmodule
