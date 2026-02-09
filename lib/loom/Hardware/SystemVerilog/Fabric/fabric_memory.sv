//===-- fabric_memory.sv - On-chip memory module ---------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// On-chip scratchpad memory with configurable load/store ports and optional
// load-store queue for ordering. Supports tagged ports for multi-port
// disambiguation.
//
// Port layout (inputs):
//   [ld_addr * LD_COUNT] [st_addr * ST_COUNT] [st_data * ST_COUNT]
// Port layout (outputs):
//   [memref? (if !IS_PRIVATE)] [ld_data * LD_COUNT] [ld_done] [st_done?]
//
// Errors:
//   RT_MEMORY_TAG_OOB         - Tagged request uses tag >= port count
//   RT_MEMORY_STORE_DEADLOCK  - Store pairing timeout
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_memory #(
    parameter int DATA_WIDTH    = 32,
    parameter int TAG_WIDTH     = 0,
    parameter int LD_COUNT      = 1,
    parameter int ST_COUNT      = 0,
    parameter int LSQ_DEPTH     = 0,
    parameter int IS_PRIVATE    = 1,
    parameter int MEM_DEPTH     = 64,
    localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW       = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    localparam int SAFE_DW       = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    localparam int ADDR_WIDTH    = $clog2(MEM_DEPTH > 1 ? MEM_DEPTH : 2),
    localparam int NUM_INPUTS    = LD_COUNT + 2 * ST_COUNT,
    localparam int NUM_OUTPUTS   = (IS_PRIVATE ? 0 : 1) + LD_COUNT + 1 + (ST_COUNT > 0 ? 1 : 0),
    localparam int CONFIG_WIDTH  = 0
) (
    input  logic               clk,
    input  logic               rst_n,

    // Streaming inputs: [ld_addr * LD_COUNT, st_addr * ST_COUNT, st_data * ST_COUNT]
    input  logic [NUM_INPUTS-1:0]               in_valid,
    output logic [NUM_INPUTS-1:0]               in_ready,
    input  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]  in_data,

    // Streaming outputs: [memref?] [ld_data * LD_COUNT] [ld_done] [st_done?]
    output logic [NUM_OUTPUTS-1:0]              out_valid,
    input  logic [NUM_OUTPUTS-1:0]              out_ready,
    output logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data,

    // Error output
    output logic                                error_valid,
    output logic [15:0]                         error_code
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (LD_COUNT == 0 && ST_COUNT == 0)
      $fatal(1, "COMP_MEMORY_PORTS_EMPTY: must have at least 1 load or store port");
    if (LSQ_DEPTH > 0 && ST_COUNT == 0)
      $fatal(1, "COMP_MEMORY_LSQ_WITHOUT_STORE: lsqDepth requires stCount > 0");
    if (ST_COUNT > 0 && LSQ_DEPTH < 1)
      $fatal(1, "COMP_MEMORY_LSQ_MIN: lsqDepth must be >= 1 when stCount > 0");
    if (DATA_WIDTH < 1)
      $fatal(1, "COMP_MEMORY_DATA_WIDTH: must be >= 1");
    if (MEM_DEPTH < 1)
      $fatal(1, "COMP_MEMORY_MEM_DEPTH: must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Internal memory array
  // -----------------------------------------------------------------------
  logic [SAFE_DW-1:0] mem [0:MEM_DEPTH-1];

  // -----------------------------------------------------------------------
  // Load path: read from memory
  // -----------------------------------------------------------------------
  generate
    genvar gli;
    for (gli = 0; gli < LD_COUNT; gli++) begin : g_load
      localparam int IN_IDX  = gli;
      localparam int OUT_IDX = IS_PRIVATE ? 0 : 1; // skip memref if non-private
      localparam int DATA_OUT = OUT_IDX + gli;
      localparam int DONE_OUT = OUT_IDX + LD_COUNT; // lddone

      logic [ADDR_WIDTH-1:0] ld_addr;
      assign ld_addr = in_data[IN_IDX][ADDR_WIDTH-1:0];

      logic ld_fire;
      assign ld_fire = in_valid[IN_IDX] && out_ready[DATA_OUT] && out_ready[DONE_OUT];

      assign in_ready[IN_IDX] = ld_fire;
      assign out_valid[DATA_OUT] = in_valid[IN_IDX];

      if (TAG_WIDTH > 0) begin : g_tagged_ld
        logic [TAG_WIDTH-1:0] ld_tag;
        assign ld_tag = in_data[IN_IDX][DATA_WIDTH +: TAG_WIDTH];
        assign out_data[DATA_OUT] = {ld_tag, mem[ld_addr]};
      end else begin : g_untagged_ld
        assign out_data[DATA_OUT] = mem[ld_addr];
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Load done signal (carries tag when LD_COUNT > 1 and TAG_WIDTH > 0)
  // -----------------------------------------------------------------------
  generate
    if (LD_COUNT > 0) begin : g_lddone
      localparam int DONE_IDX = (IS_PRIVATE ? 0 : 1) + LD_COUNT;
      logic any_ld_valid;
      logic [SAFE_PW-1:0] lddone_data;
      if (TAG_WIDTH > 0 && LD_COUNT > 1) begin : g_tagged_lddone
        always_comb begin : lddone_logic
          integer iter_var0;
          any_ld_valid = 1'b0;
          lddone_data  = '0;
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0 = iter_var0 + 1) begin : chk
            if (in_valid[iter_var0]) begin : fire
              any_ld_valid = 1'b1;
              lddone_data[DATA_WIDTH +: TAG_WIDTH] = in_data[iter_var0][DATA_WIDTH +: TAG_WIDTH];
            end
          end
        end
      end else begin : g_untagged_lddone
        always_comb begin : lddone_logic
          integer iter_var0;
          any_ld_valid = 1'b0;
          lddone_data  = '0;
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0 = iter_var0 + 1) begin : chk
            if (in_valid[iter_var0]) begin : fire
              any_ld_valid = 1'b1;
            end
          end
        end
      end
      assign out_valid[DONE_IDX] = any_ld_valid;
      assign out_data[DONE_IDX]  = lddone_data;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Store path: FIFO-based address/data pairing per store port
  // -----------------------------------------------------------------------
  // Each store port has an independent address FIFO and data FIFO (depth =
  // LSQ_DEPTH). Address and data can arrive at different times. When both
  // FIFOs are non-empty, the head entries pair and the store fires.
  logic [ST_COUNT > 0 ? ST_COUNT-1 : 0 : 0] st_paired;

  generate
    if (ST_COUNT > 0) begin : g_store
      localparam int SAFE_LSQ = (LSQ_DEPTH > 0) ? LSQ_DEPTH : 1;
      localparam int Q_IDX_W  = $clog2(SAFE_LSQ > 1 ? SAFE_LSQ : 2);

      genvar gsi;
      for (gsi = 0; gsi < ST_COUNT; gsi++) begin : g_st_port
        localparam int ADDR_IN = LD_COUNT + gsi;
        localparam int DATA_IN = LD_COUNT + ST_COUNT + gsi;
        localparam int DONE_IDX = (IS_PRIVATE ? 0 : 1) + LD_COUNT + 1;

        // Address FIFO
        logic [ADDR_WIDTH-1:0] addr_q [0:SAFE_LSQ-1];
        logic [Q_IDX_W:0] addr_cnt;
        logic [Q_IDX_W-1:0] addr_wr_ptr, addr_rd_ptr;
        logic addr_full, addr_empty;
        assign addr_full  = (addr_cnt == SAFE_LSQ[Q_IDX_W:0]);
        assign addr_empty = (addr_cnt == '0);

        // Data FIFO
        logic [SAFE_DW-1:0] data_q [0:SAFE_LSQ-1];
        logic [Q_IDX_W:0] data_cnt;
        logic [Q_IDX_W-1:0] data_wr_ptr, data_rd_ptr;
        logic data_full, data_empty;
        assign data_full  = (data_cnt == SAFE_LSQ[Q_IDX_W:0]);
        assign data_empty = (data_cnt == '0);

        // Enqueue: accept address/data independently when FIFO not full
        assign in_ready[ADDR_IN] = !addr_full;
        assign in_ready[DATA_IN] = !data_full;

        logic addr_enq, data_enq;
        assign addr_enq = in_valid[ADDR_IN] && !addr_full;
        assign data_enq = in_valid[DATA_IN] && !data_full;

        // Dequeue: paired store fires when both FIFOs non-empty and done ready
        logic pair_fire;
        assign pair_fire = !addr_empty && !data_empty && out_ready[DONE_IDX];
        assign st_paired[gsi] = pair_fire;

        always_ff @(posedge clk or negedge rst_n) begin : addr_fifo
          if (!rst_n) begin : reset
            addr_cnt    <= '0;
            addr_wr_ptr <= '0;
            addr_rd_ptr <= '0;
          end else begin : tick
            if (addr_enq && !pair_fire) begin : enq_only
              addr_q[addr_wr_ptr] <= in_data[ADDR_IN][ADDR_WIDTH-1:0];
              addr_wr_ptr <= (addr_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (addr_wr_ptr + Q_IDX_W'(1));
              addr_cnt    <= addr_cnt + (Q_IDX_W+1)'(1);
            end else if (!addr_enq && pair_fire) begin : deq_only
              addr_rd_ptr <= (addr_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (addr_rd_ptr + Q_IDX_W'(1));
              addr_cnt    <= addr_cnt - (Q_IDX_W+1)'(1);
            end else if (addr_enq && pair_fire) begin : enq_deq
              addr_q[addr_wr_ptr] <= in_data[ADDR_IN][ADDR_WIDTH-1:0];
              addr_wr_ptr <= (addr_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (addr_wr_ptr + Q_IDX_W'(1));
              addr_rd_ptr <= (addr_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (addr_rd_ptr + Q_IDX_W'(1));
            end
          end
        end

        always_ff @(posedge clk or negedge rst_n) begin : data_fifo
          if (!rst_n) begin : reset
            data_cnt    <= '0;
            data_wr_ptr <= '0;
            data_rd_ptr <= '0;
          end else begin : tick
            if (data_enq && !pair_fire) begin : enq_only
              data_q[data_wr_ptr] <= in_data[DATA_IN][DATA_WIDTH-1:0];
              data_wr_ptr <= (data_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_wr_ptr + Q_IDX_W'(1));
              data_cnt    <= data_cnt + (Q_IDX_W+1)'(1);
            end else if (!data_enq && pair_fire) begin : deq_only
              data_rd_ptr <= (data_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_rd_ptr + Q_IDX_W'(1));
              data_cnt    <= data_cnt - (Q_IDX_W+1)'(1);
            end else if (data_enq && pair_fire) begin : enq_deq
              data_q[data_wr_ptr] <= in_data[DATA_IN][DATA_WIDTH-1:0];
              data_wr_ptr <= (data_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_wr_ptr + Q_IDX_W'(1));
              data_rd_ptr <= (data_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_rd_ptr + Q_IDX_W'(1));
            end
          end
        end

        // Write to memory on paired fire
        always_ff @(posedge clk) begin : write_mem
          if (pair_fire) begin : do_write
            mem[addr_q[addr_rd_ptr]] <= data_q[data_rd_ptr];
          end
        end
      end

      // Store done signal: fires when any store port has a paired completion.
      // When ST_COUNT > 1 && TAG_WIDTH > 0, stdone carries the port tag.
      begin : g_stdone
        localparam int DONE_IDX = (IS_PRIVATE ? 0 : 1) + LD_COUNT + 1;
        logic any_st_fire;
        logic [SAFE_PW-1:0] stdone_data;
        if (TAG_WIDTH > 0 && ST_COUNT > 1) begin : g_tagged_stdone
          always_comb begin : stdone_logic
            integer iter_var0;
            any_st_fire = 1'b0;
            stdone_data = '0;
            for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : chk
              if (st_paired[iter_var0]) begin : fire
                any_st_fire = 1'b1;
                stdone_data[DATA_WIDTH +: TAG_WIDTH] = TAG_WIDTH'(iter_var0);
              end
            end
          end
        end else begin : g_untagged_stdone
          always_comb begin : stdone_logic
            integer iter_var0;
            any_st_fire = 1'b0;
            stdone_data = '0;
            for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : chk
              if (st_paired[iter_var0]) begin : fire
                any_st_fire = 1'b1;
              end
            end
          end
        end
        assign out_valid[DONE_IDX] = any_st_fire;
        assign out_data[DONE_IDX]  = stdone_data;
      end
    end else begin : g_no_store
      assign st_paired = 1'b0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Non-private memref output (placeholder, always valid with base addr 0)
  // -----------------------------------------------------------------------
  generate
    if (!IS_PRIVATE) begin : g_memref
      assign out_valid[0] = 1'b1;
      assign out_data[0]  = '0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Store deadlock timeout counter (per store port)
  // Deadlock: one FIFO has entries but the other is empty with no progress.
  // -----------------------------------------------------------------------
  localparam int DEADLOCK_TIMEOUT = 65535;
  logic [ST_COUNT > 0 ? ST_COUNT-1 : 0 : 0] st_deadlock_hit;

  generate
    if (ST_COUNT > 0) begin : g_deadlock
      genvar gdi;
      for (gdi = 0; gdi < ST_COUNT; gdi++) begin : g_dl_cnt
        logic [15:0] dl_counter;
        logic        addr_waiting, data_waiting;
        // Deadlock condition: one FIFO has data but the other is empty
        assign addr_waiting = !g_store.g_st_port[gdi].addr_empty && g_store.g_st_port[gdi].data_empty;
        assign data_waiting = g_store.g_st_port[gdi].addr_empty && !g_store.g_st_port[gdi].data_empty;

        always_ff @(posedge clk or negedge rst_n) begin : dl_cnt
          if (!rst_n) begin : reset
            dl_counter <= 16'd0;
          end else begin : tick
            if (addr_waiting || data_waiting) begin : waiting
              if (dl_counter < DEADLOCK_TIMEOUT[15:0]) begin : inc
                dl_counter <= dl_counter + 16'd1;
              end
            end else begin : clear
              dl_counter <= 16'd0;
            end
          end
        end

        assign st_deadlock_hit[gdi] = (dl_counter == DEADLOCK_TIMEOUT[15:0]);
      end
    end else begin : g_no_deadlock
      assign st_deadlock_hit = 1'b0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------
  logic        err_tag_oob;
  logic        err_deadlock;

  // RT_MEMORY_TAG_OOB: tag >= count (generate-time guard for TAG_WIDTH=0)
  generate
    if (TAG_WIDTH > 0) begin : g_tag_oob_chk
      always_comb begin : tag_oob_logic
        integer iter_var0;
        err_tag_oob = 1'b0;
        if (LD_COUNT > 1) begin : ld_tag_chk
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0 = iter_var0 + 1) begin : per_ld
            if (in_valid[iter_var0] &&
                ({{(32-TAG_WIDTH){1'b0}}, in_data[iter_var0][DATA_WIDTH +: TAG_WIDTH]} >= 32'(LD_COUNT))) begin : oob
              err_tag_oob = 1'b1;
            end
          end
        end
        if (ST_COUNT > 1) begin : st_tag_chk
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : per_st_addr
            if (in_valid[LD_COUNT + iter_var0] &&
                ({{(32-TAG_WIDTH){1'b0}}, in_data[LD_COUNT + iter_var0][DATA_WIDTH +: TAG_WIDTH]} >= 32'(ST_COUNT))) begin : oob
              err_tag_oob = 1'b1;
            end
          end
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : per_st_data
            if (in_valid[LD_COUNT + ST_COUNT + iter_var0] &&
                ({{(32-TAG_WIDTH){1'b0}}, in_data[LD_COUNT + ST_COUNT + iter_var0][DATA_WIDTH +: TAG_WIDTH]} >= 32'(ST_COUNT))) begin : oob
              err_tag_oob = 1'b1;
            end
          end
        end
      end
    end else begin : g_no_tag_oob
      assign err_tag_oob = 1'b0;
    end
  endgenerate

  // RT_MEMORY_STORE_DEADLOCK: store pairing timeout
  always_comb begin : deadlock_check
    integer iter_var0;
    err_deadlock = 1'b0;
    for (iter_var0 = 0; iter_var0 < (ST_COUNT > 0 ? ST_COUNT : 1); iter_var0 = iter_var0 + 1) begin : dl_chk
      if (st_deadlock_hit[iter_var0]) begin : deadlock
        err_deadlock = 1'b1;
      end
    end
  end

  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_encode
    err_detect    = 1'b0;
    err_code_comb = 16'd0;
    if (err_tag_oob) begin : e_oob
      err_detect    = 1'b1;
      err_code_comb = RT_MEMORY_TAG_OOB;
    end else if (err_deadlock) begin : e_dl
      err_detect    = 1'b1;
      err_code_comb = RT_MEMORY_STORE_DEADLOCK;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin : error_latch
    if (!rst_n) begin : reset
      error_valid <= 1'b0;
      error_code  <= 16'd0;
    end else if (!error_valid && err_detect) begin : capture
      error_valid <= 1'b1;
      error_code  <= err_code_comb;
    end
  end

endmodule
