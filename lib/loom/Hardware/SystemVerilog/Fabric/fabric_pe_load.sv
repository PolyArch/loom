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
    if (HW_TYPE == 1 && QUEUE_DEPTH < 1)
      $fatal(1, "COMP_PE_LOADSTORE_TAG_MODE: TagTransparent requires QUEUE_DEPTH >= 1");
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
      // TagTransparent: queue addr and ctrl arrivals, then match by tag.
      localparam int SAFE_QD = (QUEUE_DEPTH > 0) ? QUEUE_DEPTH : 1;
      localparam int Q_IDX_W = $clog2(SAFE_QD > 1 ? SAFE_QD : 2);

      logic [SAFE_QD-1:0]                   addr_q_valid;
      logic [SAFE_QD-1:0][TAG_WIDTH-1:0]   addr_q_tag;
      logic [SAFE_QD-1:0][SAFE_AW-1:0]     addr_q_value;

      logic [SAFE_QD-1:0]                   ctrl_q_valid;
      logic [SAFE_QD-1:0][TAG_WIDTH-1:0]   ctrl_q_tag;

      logic addr_free_found;
      logic [Q_IDX_W-1:0] addr_free_idx;
      logic ctrl_free_found;
      logic [Q_IDX_W-1:0] ctrl_free_idx;

      logic match_found;
      logic [Q_IDX_W-1:0] match_addr_idx;
      logic [Q_IDX_W-1:0] match_ctrl_idx;
      logic [TAG_WIDTH-1:0] match_tag;

      logic [TAG_WIDTH-1:0] in0_addr_tag;
      logic [SAFE_AW-1:0]   in0_addr_value;
      logic [TAG_WIDTH-1:0] in2_ctrl_tag;
      assign in0_addr_tag   = in0_data[ADDR_WIDTH +: TAG_WIDTH];
      assign in0_addr_value = in0_data[ADDR_WIDTH-1:0];
      assign in2_ctrl_tag   = in2_data[TAG_WIDTH-1:0];

      always_comb begin : addr_free_find
        integer iter_var0;
        addr_free_found = 1'b0;
        addr_free_idx = '0;
        for (iter_var0 = 0; iter_var0 < SAFE_QD; iter_var0 = iter_var0 + 1) begin : scan
          if (!addr_free_found && !addr_q_valid[iter_var0]) begin : hit
            addr_free_found = 1'b1;
            addr_free_idx = Q_IDX_W'(iter_var0);
          end
        end
      end

      always_comb begin : ctrl_free_find
        integer iter_var0;
        ctrl_free_found = 1'b0;
        ctrl_free_idx = '0;
        for (iter_var0 = 0; iter_var0 < SAFE_QD; iter_var0 = iter_var0 + 1) begin : scan
          if (!ctrl_free_found && !ctrl_q_valid[iter_var0]) begin : hit
            ctrl_free_found = 1'b1;
            ctrl_free_idx = Q_IDX_W'(iter_var0);
          end
        end
      end

      always_comb begin : match_find
        integer iter_var0;
        integer iter_var1;
        match_found = 1'b0;
        match_addr_idx = '0;
        match_ctrl_idx = '0;
        match_tag = '0;
        for (iter_var0 = 0; iter_var0 < SAFE_QD; iter_var0 = iter_var0 + 1) begin : scan_addr
          if (addr_q_valid[iter_var0] && !match_found) begin : scan_ctrl_block
            for (iter_var1 = 0; iter_var1 < SAFE_QD; iter_var1 = iter_var1 + 1) begin : scan_ctrl
              if (ctrl_q_valid[iter_var1] && !match_found &&
                  addr_q_tag[iter_var0] == ctrl_q_tag[iter_var1]) begin : hit
                match_found = 1'b1;
                match_addr_idx = Q_IDX_W'(iter_var0);
                match_ctrl_idx = Q_IDX_W'(iter_var1);
                match_tag = addr_q_tag[iter_var0];
              end
            end
          end
        end
      end

      logic addr_push;
      logic ctrl_push;
      logic match_fire;
      assign addr_push = in0_valid && in0_ready;
      assign ctrl_push = in2_valid && in2_ready;
      assign match_fire = match_found && out0_ready;

      assign in0_ready = addr_free_found;
      assign in2_ready = ctrl_free_found;

      // out0 emits the matched request tag+address.
      assign out0_valid = match_found;
      assign out0_data  = {match_tag, addr_q_value[match_addr_idx]};

      // out1 forwards memory response token unchanged in transparent mode.
      assign out1_valid = in1_valid;
      assign in1_ready  = out1_ready;
      assign out1_data  = in1_data;

      always_ff @(posedge clk or negedge rst_n) begin : queue_state
        integer iter_var0;
        if (!rst_n) begin : reset
          for (iter_var0 = 0; iter_var0 < SAFE_QD; iter_var0 = iter_var0 + 1) begin : clr
            addr_q_valid[iter_var0] <= 1'b0;
            addr_q_tag[iter_var0] <= '0;
            addr_q_value[iter_var0] <= '0;
            ctrl_q_valid[iter_var0] <= 1'b0;
            ctrl_q_tag[iter_var0] <= '0;
          end
        end else begin : tick
          if (match_fire) begin : pop_match
            addr_q_valid[match_addr_idx] <= 1'b0;
            ctrl_q_valid[match_ctrl_idx] <= 1'b0;
          end

          if (addr_push) begin : push_addr
            addr_q_valid[addr_free_idx] <= 1'b1;
            addr_q_tag[addr_free_idx] <= in0_addr_tag;
            addr_q_value[addr_free_idx] <= in0_addr_value;
          end

          if (ctrl_push) begin : push_ctrl
            ctrl_q_valid[ctrl_free_idx] <= 1'b1;
            ctrl_q_tag[ctrl_free_idx] <= in2_ctrl_tag;
          end
        end
      end
    end
  endgenerate

endmodule
