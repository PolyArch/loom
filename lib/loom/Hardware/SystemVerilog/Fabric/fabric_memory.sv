//===-- fabric_memory.sv - On-chip memory module ---------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// On-chip scratchpad memory with singular port groups per category and optional
// load-store queue for ordering. Supports tagged ports for multi-port
// disambiguation. LD_COUNT/ST_COUNT determine TAG_WIDTH for multiplexing,
// not physical port count.
//
// Port groups (singular, presence-based on LD_COUNT/ST_COUNT):
//   Inputs:  ld_addr (if LD_COUNT>0), st_addr+st_data (if ST_COUNT>0)
//   Outputs: memref (if !IS_PRIVATE), ld_data+ld_done (if LD_COUNT>0),
//            st_done (if ST_COUNT>0)
//
// Errors:
//   RT_MEMORY_TAG_OOB         - Tagged request uses tag >= port count
//   RT_MEMORY_STORE_DEADLOCK  - Store pairing timeout
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_memory #(
    parameter int ADDR_WIDTH       = 64,
    parameter int ELEM_WIDTH       = 32,
    parameter int TAG_WIDTH        = 0,
    parameter int LD_COUNT         = 1,
    parameter int ST_COUNT         = 0,
    parameter int LSQ_DEPTH        = 0,
    parameter int IS_PRIVATE       = 1,
    parameter int MEM_DEPTH        = 64,
    parameter int DEADLOCK_TIMEOUT = 65535,
    localparam int ADDR_PW   = (ADDR_WIDTH + TAG_WIDTH > 0) ? ADDR_WIDTH + TAG_WIDTH : 1,
    localparam int ELEM_PW   = (ELEM_WIDTH + TAG_WIDTH > 0) ? ELEM_WIDTH + TAG_WIDTH : 1,
    localparam int DONE_PW   = (TAG_WIDTH > 0) ? TAG_WIDTH : 1,
    localparam int MEM_AW    = $clog2(MEM_DEPTH > 1 ? MEM_DEPTH : 2),
    localparam int SAFE_EW   = (ELEM_WIDTH > 0) ? ELEM_WIDTH : 1,
    localparam int SAFE_AW   = (ADDR_WIDTH > 0) ? ADDR_WIDTH : 1,
    localparam int SAFE_TW   = (TAG_WIDTH > 0) ? TAG_WIDTH : 1,
    localparam int CONFIG_WIDTH = 0
) (
    input  logic               clk,
    input  logic               rst_n,

    // Load address input (used when LD_COUNT > 0)
    input  logic               ld_addr_valid,
    output logic               ld_addr_ready,
    input  logic [ADDR_PW-1:0] ld_addr_data,

    // Store address input (used when ST_COUNT > 0)
    input  logic               st_addr_valid,
    output logic               st_addr_ready,
    input  logic [ADDR_PW-1:0] st_addr_data,

    // Store data input (used when ST_COUNT > 0)
    input  logic               st_data_valid,
    output logic               st_data_ready,
    input  logic [ELEM_PW-1:0] st_data_data,

    // Load data output (used when LD_COUNT > 0)
    output logic               ld_data_valid,
    input  logic               ld_data_ready,
    output logic [ELEM_PW-1:0] ld_data_data,

    // Load done output (used when LD_COUNT > 0)
    output logic               ld_done_valid,
    input  logic               ld_done_ready,
    output logic [DONE_PW-1:0] ld_done_data,

    // Store done output (used when ST_COUNT > 0)
    output logic               st_done_valid,
    input  logic               st_done_ready,
    output logic [DONE_PW-1:0] st_done_data,

    // Memref output (non-private only)
    output logic               memref_valid,
    input  logic               memref_ready,

    // Error output
    output logic               error_valid,
    output logic [15:0]        error_code
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
    if (ELEM_WIDTH < 1)
      $fatal(1, "COMP_MEMORY_ELEM_WIDTH: must be >= 1");
    if (MEM_DEPTH < 1)
      $fatal(1, "COMP_MEMORY_MEM_DEPTH: must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Internal memory array
  // -----------------------------------------------------------------------
  logic [SAFE_EW-1:0] mem [0:MEM_DEPTH-1];

  // -----------------------------------------------------------------------
  // Tag range checks used by ready/enqueue and error reporting
  // -----------------------------------------------------------------------
  logic ld_tag_oob_req;
  logic st_addr_tag_oob_req;
  logic st_data_tag_oob_req;

  generate
    if (TAG_WIDTH > 0) begin : g_req_tag_chk
      if (LD_COUNT > 1) begin : g_ld_req_tag_chk
        assign ld_tag_oob_req =
            ({{(32-TAG_WIDTH){1'b0}}, ld_addr_data[ADDR_WIDTH +: TAG_WIDTH]} >= 32'(LD_COUNT));
      end else begin : g_ld_req_tag_ok
        assign ld_tag_oob_req = 1'b0;
      end
      if (ST_COUNT > 1) begin : g_st_req_tag_chk
        assign st_addr_tag_oob_req =
            ({{(32-TAG_WIDTH){1'b0}}, st_addr_data[ADDR_WIDTH +: TAG_WIDTH]} >= 32'(ST_COUNT));
        assign st_data_tag_oob_req =
            ({{(32-TAG_WIDTH){1'b0}}, st_data_data[ELEM_WIDTH +: TAG_WIDTH]} >= 32'(ST_COUNT));
      end else begin : g_st_req_tag_ok
        assign st_addr_tag_oob_req = 1'b0;
        assign st_data_tag_oob_req = 1'b0;
      end
    end else begin : g_no_req_tag_chk
      assign ld_tag_oob_req = 1'b0;
      assign st_addr_tag_oob_req = 1'b0;
      assign st_data_tag_oob_req = 1'b0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Load path: read from memory (only active when LD_COUNT > 0)
  // -----------------------------------------------------------------------
  generate
    if (LD_COUNT > 0) begin : g_load
      logic [MEM_AW-1:0] ld_addr_val;
      assign ld_addr_val = ld_addr_data[MEM_AW-1:0];

      // lddata/lddone must represent one aligned completion event.
      logic ld_req_ok;
      assign ld_req_ok = !ld_tag_oob_req;

      assign ld_addr_ready = ld_req_ok && ld_data_ready && ld_done_ready;
      assign ld_data_valid = ld_addr_valid && ld_req_ok && ld_done_ready;
      assign ld_done_valid = ld_addr_valid && ld_req_ok && ld_data_ready;

      if (TAG_WIDTH > 0) begin : g_tagged_ld
        logic [TAG_WIDTH-1:0] ld_tag;
        assign ld_tag = ld_addr_data[ADDR_WIDTH +: TAG_WIDTH];
        assign ld_data_data = {ld_tag, mem[ld_addr_val]};
        assign ld_done_data = ld_tag[DONE_PW-1:0];
      end else begin : g_untagged_ld
        assign ld_data_data = mem[ld_addr_val];
        assign ld_done_data = 1'b0;
      end
    end else begin : g_no_load
      assign ld_addr_ready = 1'b0;
      assign ld_data_valid = 1'b0;
      assign ld_data_data  = '0;
      assign ld_done_valid = 1'b0;
      assign ld_done_data  = '0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Store path: true per-tag FIFO-based address/data pairing
  // (only active when ST_COUNT > 0)
  // -----------------------------------------------------------------------
  localparam int SAFE_STC = (ST_COUNT > 0) ? ST_COUNT : 1;
  logic [SAFE_STC-1:0] st_paired;

  generate
    if (ST_COUNT > 0) begin : g_store
      localparam int SAFE_LSQ = (LSQ_DEPTH > 0) ? LSQ_DEPTH : 1;
      localparam int Q_IDX_W  = $clog2(SAFE_LSQ > 1 ? SAFE_LSQ : 2);

      // Per-tag FIFOs: ST_COUNT tags (tag range [0, ST_COUNT-1])
      logic [ST_COUNT-1:0]                    tag_addr_full;
      logic [ST_COUNT-1:0]                    tag_data_full;
      logic [ST_COUNT-1:0]                    tag_addr_enq;
      logic [ST_COUNT-1:0][MEM_AW-1:0]       tag_addr_enq_val;
      logic [ST_COUNT-1:0]                    tag_data_enq;
      logic [ST_COUNT-1:0][SAFE_EW-1:0]      tag_data_enq_val;

      // Flat write-export arrays
      logic [ST_COUNT-1:0]                    tag_write_en;
      logic [ST_COUNT-1:0][MEM_AW-1:0]       tag_write_addr;
      logic [ST_COUNT-1:0][SAFE_EW-1:0]      tag_write_data;
      logic [ST_COUNT-1:0]                    pair_req;
      logic [ST_COUNT-1:0]                    pair_grant;
      logic                                   pair_sel_valid;
      logic [SAFE_TW-1:0]                     pair_sel_tag;

      genvar gti;
      for (gti = 0; gti < ST_COUNT; gti++) begin : g_tag_fifo
        // Address FIFO for this tag
        logic [MEM_AW-1:0] addr_q [0:SAFE_LSQ-1];
        logic [Q_IDX_W:0] addr_cnt;
        logic [Q_IDX_W-1:0] addr_wr_ptr, addr_rd_ptr;
        logic addr_full, addr_empty;
        assign addr_full  = (addr_cnt == SAFE_LSQ[Q_IDX_W:0]);
        assign addr_empty = (addr_cnt == '0);

        // Data FIFO for this tag
        logic [SAFE_EW-1:0] data_q [0:SAFE_LSQ-1];
        logic [Q_IDX_W:0] data_cnt;
        logic [Q_IDX_W-1:0] data_wr_ptr, data_rd_ptr;
        logic data_full, data_empty;
        assign data_full  = (data_cnt == SAFE_LSQ[Q_IDX_W:0]);
        assign data_empty = (data_cnt == '0);

        // One paired store is selected globally each cycle.
        logic pair_fire;
        assign pair_req[gti] = !addr_empty && !data_empty;
        assign pair_fire = pair_grant[gti] && st_done_ready;
        assign st_paired[gti] = pair_fire;

        // Export to flat arrays
        assign tag_addr_full[gti] = addr_full;
        assign tag_data_full[gti] = data_full;

        // Per-tag enqueue signals driven from flat arrays
        logic addr_enq;
        logic [MEM_AW-1:0] addr_enq_val;
        logic data_enq;
        logic [SAFE_EW-1:0] data_enq_val;
        assign addr_enq     = tag_addr_enq[gti];
        assign addr_enq_val = tag_addr_enq_val[gti];
        assign data_enq     = tag_data_enq[gti];
        assign data_enq_val = tag_data_enq_val[gti];

        always_ff @(posedge clk or negedge rst_n) begin : addr_fifo
          if (!rst_n) begin : reset
            addr_cnt    <= '0;
            addr_wr_ptr <= '0;
            addr_rd_ptr <= '0;
          end else begin : tick
            if (addr_enq && !pair_fire) begin : enq_only
              addr_q[addr_wr_ptr] <= addr_enq_val;
              addr_wr_ptr <= (addr_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (addr_wr_ptr + Q_IDX_W'(1));
              addr_cnt    <= addr_cnt + (Q_IDX_W+1)'(1);
            end else if (!addr_enq && pair_fire) begin : deq_only
              addr_rd_ptr <= (addr_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (addr_rd_ptr + Q_IDX_W'(1));
              addr_cnt    <= addr_cnt - (Q_IDX_W+1)'(1);
            end else if (addr_enq && pair_fire) begin : enq_deq
              addr_q[addr_wr_ptr] <= addr_enq_val;
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
              data_q[data_wr_ptr] <= data_enq_val;
              data_wr_ptr <= (data_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_wr_ptr + Q_IDX_W'(1));
              data_cnt    <= data_cnt + (Q_IDX_W+1)'(1);
            end else if (!data_enq && pair_fire) begin : deq_only
              data_rd_ptr <= (data_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_rd_ptr + Q_IDX_W'(1));
              data_cnt    <= data_cnt - (Q_IDX_W+1)'(1);
            end else if (data_enq && pair_fire) begin : enq_deq
              data_q[data_wr_ptr] <= data_enq_val;
              data_wr_ptr <= (data_wr_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_wr_ptr + Q_IDX_W'(1));
              data_rd_ptr <= (data_rd_ptr == Q_IDX_W'(SAFE_LSQ - 1)) ? '0 : (data_rd_ptr + Q_IDX_W'(1));
            end
          end
        end

        // Export write signals
        assign tag_write_en[gti]   = pair_fire;
        assign tag_write_addr[gti] = addr_q[addr_rd_ptr];
        assign tag_write_data[gti] = data_q[data_rd_ptr];
      end

      always_comb begin : pair_select
        integer iter_var0;
        logic grant_taken;
        pair_grant = '0;
        grant_taken = 1'b0;
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : pick
          if (!grant_taken && pair_req[iter_var0]) begin : grant
            pair_grant[iter_var0] = 1'b1;
            grant_taken = 1'b1;
          end
        end
      end

      assign pair_sel_valid = |pair_grant;
      if (TAG_WIDTH > 0 && ST_COUNT > 1) begin : g_pair_tag
        always_comb begin : pair_tag_encode
          integer iter_var0;
          pair_sel_tag = '0;
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : encode
            if (pair_grant[iter_var0]) begin : hit
              pair_sel_tag = SAFE_TW'(iter_var0);
            end
          end
        end
      end else begin : g_pair_tag_zero
        assign pair_sel_tag = '0;
      end

      // Single always_ff to write memory
      always_ff @(posedge clk) begin : write_mem
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : per_tag
          if (tag_write_en[iter_var0]) begin : do_write
            mem[tag_write_addr[iter_var0]] <= tag_write_data[iter_var0];
          end
        end
      end

      // st_addr ready: depends only on FIFO full and addr data (tag routing),
      // NOT on st_addr_valid.  Kept in its own always_comb so Verilator sees
      // no valid-to-ready path (avoids false UNOPTFLAT through store PE
      // split-ready cross-dependency).
      always_comb begin : st_addr_ready_gen
        st_addr_ready = 1'b0;
        begin : check_addr
          automatic int target_tag = (TAG_WIDTH > 0 && ST_COUNT > 1)
            ? int'(st_addr_data[ADDR_PW-1 -: SAFE_TW]) : 0;
          if (target_tag >= ST_COUNT) begin : clamp
            target_tag = 0;
          end
          if (!st_addr_tag_oob_req && !tag_addr_full[target_tag]) begin : grant
            st_addr_ready = 1'b1;
          end
        end
      end

      // st_addr enqueue: reads st_addr_valid to gate FIFO writes.
      always_comb begin : st_addr_enq_gen
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : clr_enq
          tag_addr_enq[iter_var0] = 1'b0;
          tag_addr_enq_val[iter_var0] = '0;
        end
        begin : route_addr
          automatic int target_tag = (TAG_WIDTH > 0 && ST_COUNT > 1)
            ? int'(st_addr_data[ADDR_PW-1 -: SAFE_TW]) : 0;
          if (target_tag >= ST_COUNT) begin : clamp
            target_tag = 0;
          end
          if (!st_addr_tag_oob_req && !tag_addr_full[target_tag] && st_addr_valid) begin : enq
            tag_addr_enq[target_tag] = 1'b1;
            tag_addr_enq_val[target_tag] = st_addr_data[MEM_AW-1:0];
          end
        end
      end

      // st_data ready: depends only on FIFO full and data (tag routing),
      // NOT on st_data_valid.
      always_comb begin : st_data_ready_gen
        st_data_ready = 1'b0;
        begin : check_data
          automatic int target_tag = (TAG_WIDTH > 0 && ST_COUNT > 1)
            ? int'(st_data_data[ELEM_PW-1 -: SAFE_TW]) : 0;
          if (target_tag >= ST_COUNT) begin : clamp
            target_tag = 0;
          end
          if (!st_data_tag_oob_req && !tag_data_full[target_tag]) begin : grant
            st_data_ready = 1'b1;
          end
        end
      end

      // st_data enqueue: reads st_data_valid to gate FIFO writes.
      always_comb begin : st_data_enq_gen
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : clr_enq
          tag_data_enq[iter_var0] = 1'b0;
          tag_data_enq_val[iter_var0] = '0;
        end
        begin : route_data
          automatic int target_tag = (TAG_WIDTH > 0 && ST_COUNT > 1)
            ? int'(st_data_data[ELEM_PW-1 -: SAFE_TW]) : 0;
          if (target_tag >= ST_COUNT) begin : clamp
            target_tag = 0;
          end
          if (!st_data_tag_oob_req && !tag_data_full[target_tag] && st_data_valid) begin : enq
            tag_data_enq[target_tag] = 1'b1;
            tag_data_enq_val[target_tag] = st_data_data[ELEM_WIDTH-1:0];
          end
        end
      end

      // Store done signal
      assign st_done_valid = pair_sel_valid;
      assign st_done_data  = DONE_PW'(pair_sel_tag);
    end else begin : g_no_store
      assign st_paired    = 1'b0;
      assign st_addr_ready = 1'b0;
      assign st_data_ready = 1'b0;
      assign st_done_valid = 1'b0;
      assign st_done_data  = '0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Non-private memref output (placeholder, always valid)
  // -----------------------------------------------------------------------
  generate
    if (!IS_PRIVATE) begin : g_memref
      assign memref_valid = 1'b1;
    end else begin : g_no_memref
      assign memref_valid = 1'b0;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Store deadlock timeout counter
  // -----------------------------------------------------------------------
  logic [ST_COUNT > 0 ? ST_COUNT-1 : 0 : 0] st_deadlock_hit;

  generate
    if (ST_COUNT > 0) begin : g_deadlock
      genvar gdi;
      for (gdi = 0; gdi < ST_COUNT; gdi++) begin : g_dl_cnt
        logic [15:0] dl_counter;
        logic        addr_waiting, data_waiting;
        assign addr_waiting = !g_store.g_tag_fifo[gdi].addr_empty && g_store.g_tag_fifo[gdi].data_empty;
        assign data_waiting = g_store.g_tag_fifo[gdi].addr_empty && !g_store.g_tag_fifo[gdi].data_empty;

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

  // RT_MEMORY_TAG_OOB
  generate
    if (TAG_WIDTH > 0) begin : g_tag_oob_chk
      always_comb begin : tag_oob_logic
        err_tag_oob = 1'b0;
        if (ld_addr_valid && ld_tag_oob_req) begin : ld_tag_chk
          err_tag_oob = 1'b1;
        end
        if (st_addr_valid && st_addr_tag_oob_req) begin : st_addr_tag_chk
          err_tag_oob = 1'b1;
        end
        if (st_data_valid && st_data_tag_oob_req) begin : st_data_tag_chk
          err_tag_oob = 1'b1;
        end
      end
    end else begin : g_no_tag_oob
      assign err_tag_oob = 1'b0;
    end
  endgenerate

  // RT_MEMORY_STORE_DEADLOCK
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
