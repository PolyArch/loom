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
    parameter int ELEM_WIDTH  = 32,
    parameter int ADDR_WIDTH  = 64,
    parameter int TAG_WIDTH   = 0,
    parameter int HW_TYPE     = 0,   // 0=TagOverwrite, 1=TagTransparent
    parameter int QUEUE_DEPTH = 4,
    localparam int ADDR_PW    = (ADDR_WIDTH + TAG_WIDTH > 0) ? ADDR_WIDTH + TAG_WIDTH : 1,
    localparam int ELEM_PW    = (ELEM_WIDTH + TAG_WIDTH > 0) ? ELEM_WIDTH + TAG_WIDTH : 1,
    localparam int SAFE_AW    = (ADDR_WIDTH > 0) ? ADDR_WIDTH : 1,
    localparam int SAFE_EW    = (ELEM_WIDTH > 0) ? ELEM_WIDTH : 1,
    // Ctrl port width: TagTransparent = TAG_WIDTH, else 1 (none token)
    localparam int CTRL_PW    = (HW_TYPE == 1 && TAG_WIDTH > 0) ? TAG_WIDTH : 1,
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
    input  logic [ELEM_PW-1:0] in1_data,

    // Input 2: control token (none=1bit for TagOverwrite, tagged<none>=TAG_WIDTH for TagTransparent)
    input  logic               in2_valid,
    output logic               in2_ready,
    input  logic [CTRL_PW-1:0] in2_data,

    // Output 0: address to memory (index type, tagged when TAG_WIDTH > 0)
    output logic               out0_valid,
    input  logic               out0_ready,
    output logic [ADDR_PW-1:0] out0_data,

    // Output 1: data to compute (elemType, possibly tagged)
    output logic               out1_valid,
    input  logic               out1_ready,
    output logic [ELEM_PW-1:0] out1_data,

    // Configuration
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (ELEM_WIDTH < 1)
      $fatal(1, "COMP_PE_LOAD_ELEM_WIDTH: ELEM_WIDTH must be >= 1");
    if (ADDR_WIDTH < 1)
      $fatal(1, "COMP_PE_LOAD_ADDR_WIDTH: ADDR_WIDTH must be >= 1");
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
      logic [SAFE_AW-1:0] addr_value;
      assign addr_value = in0_data[ADDR_WIDTH-1:0];

      // Forward address to memory (out0), with tag if tagged
      assign out0_valid = sync_valid;
      if (TAG_WIDTH > 0) begin : g_addr_tag
        logic [TAG_WIDTH-1:0] output_tag;
        assign output_tag = cfg_data[TAG_WIDTH-1:0];
        assign out0_data  = {output_tag, addr_value};
      end else begin : g_addr_no_tag
        assign out0_data  = addr_value;
      end

      logic fire;
      assign fire = sync_valid && out0_ready;
      assign in0_ready = fire;
      assign in2_ready = fire;

      // Forward memory data (in1) to compute (out1), attaching output_tag
      if (TAG_WIDTH > 0) begin : g_tag_attach
        assign out1_data = {g_addr_tag.output_tag, in1_data[ELEM_WIDTH-1:0]};
      end else begin : g_no_tag
        assign out1_data = in1_data;
      end
      assign out1_valid = in1_valid;
      assign in1_ready  = out1_ready;
    end else begin : g_transparent
      // TagTransparent: tag-match addr (in0) + ctrl (in2), forward unchanged
      logic [TAG_WIDTH-1:0] addr_tag, ctrl_tag;
      assign addr_tag = in0_data[ADDR_WIDTH +: TAG_WIDTH];
      assign ctrl_tag = in2_data[TAG_WIDTH-1:0]; // ctrl is tagged<none, iK>

      logic tags_match;
      assign tags_match = (addr_tag == ctrl_tag);

      logic sync_valid;
      assign sync_valid = in0_valid && in2_valid && tags_match;

      logic [SAFE_AW-1:0] addr_value;
      assign addr_value = in0_data[ADDR_WIDTH-1:0];

      // Forward address to memory (out0), with tag
      assign out0_valid = sync_valid;
      assign out0_data  = {addr_tag, addr_value};

      logic fire;
      assign fire = sync_valid && out0_ready;
      assign in0_ready = fire;
      assign in2_ready = fire;

      // Forward memory data (in1) with original tag to compute (out1)
      assign out1_data  = {addr_tag, in1_data[ELEM_WIDTH-1:0]};
      assign out1_valid = in1_valid;
      assign in1_ready  = out1_ready;
    end
  endgenerate

endmodule
