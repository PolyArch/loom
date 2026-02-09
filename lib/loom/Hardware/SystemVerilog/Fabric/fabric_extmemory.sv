//===-- fabric_extmemory.sv - External memory interface --------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// External memory interface module. Binds to a module-level memref input and
// provides load/store ports with optional tagging for multi-port dispatch.
//
// Port layout (inputs):
//   [memref_binding] [ld_addr * LD_COUNT] [st_addr * ST_COUNT] [st_data * ST_COUNT]
// Port layout (outputs):
//   [ld_data * LD_COUNT] [ld_done] [st_done?]
//
// Errors:
//   RT_MEMORY_TAG_OOB         - Tagged request uses tag >= port count
//   RT_MEMORY_STORE_DEADLOCK  - Store pairing timeout
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_extmemory #(
    parameter int DATA_WIDTH    = 32,
    parameter int TAG_WIDTH     = 0,
    parameter int LD_COUNT      = 1,
    parameter int ST_COUNT      = 0,
    parameter int LSQ_DEPTH     = 0,
    localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW       = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    localparam int SAFE_DW       = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    // First input is memref binding
    localparam int NUM_INPUTS    = 1 + LD_COUNT + 2 * ST_COUNT,
    localparam int NUM_OUTPUTS   = LD_COUNT + 1 + (ST_COUNT > 0 ? 1 : 0),
    localparam int CONFIG_WIDTH  = 0
) (
    input  logic               clk,
    input  logic               rst_n,

    // Streaming inputs: [memref, ld_addr * LD_COUNT, st_addr * ST_COUNT, st_data * ST_COUNT]
    input  logic [NUM_INPUTS-1:0]               in_valid,
    output logic [NUM_INPUTS-1:0]               in_ready,
    input  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]  in_data,

    // Streaming outputs: [ld_data * LD_COUNT] [ld_done] [st_done?]
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
  end

  // -----------------------------------------------------------------------
  // Memref binding input (in0) - pass-through, always ready
  // -----------------------------------------------------------------------
  assign in_ready[0] = 1'b1;

  // -----------------------------------------------------------------------
  // Behavioral memory for simulation (external memory modeled internally)
  // -----------------------------------------------------------------------
  localparam int EXT_MEM_DEPTH = 256;
  localparam int EXT_ADDR_WIDTH = $clog2(EXT_MEM_DEPTH > 1 ? EXT_MEM_DEPTH : 2);
  logic [SAFE_DW-1:0] ext_mem [0:EXT_MEM_DEPTH-1];

  // -----------------------------------------------------------------------
  // Load path: read from behavioral memory
  // -----------------------------------------------------------------------
  generate
    genvar gli;
    for (gli = 0; gli < LD_COUNT; gli++) begin : g_load
      localparam int IN_IDX  = 1 + gli; // skip memref
      localparam int DATA_OUT = gli;
      localparam int DONE_OUT = LD_COUNT;

      logic [EXT_ADDR_WIDTH-1:0] ld_addr;
      assign ld_addr = in_data[IN_IDX][EXT_ADDR_WIDTH-1:0];

      assign out_valid[DATA_OUT] = in_valid[IN_IDX];
      if (TAG_WIDTH > 0) begin : g_tagged_ld
        logic [TAG_WIDTH-1:0] ld_tag;
        assign ld_tag = in_data[IN_IDX][DATA_WIDTH +: TAG_WIDTH];
        assign out_data[DATA_OUT] = {ld_tag, ext_mem[ld_addr]};
      end else begin : g_untagged_ld
        assign out_data[DATA_OUT] = ext_mem[ld_addr];
      end
      assign in_ready[IN_IDX] = out_ready[DATA_OUT] && out_ready[DONE_OUT];
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Load done signal (carries tag when LD_COUNT > 1)
  // -----------------------------------------------------------------------
  generate
    if (LD_COUNT > 0) begin : g_lddone
      localparam int DONE_IDX = LD_COUNT;
      logic any_ld_valid;
      logic [SAFE_PW-1:0] lddone_data;
      always_comb begin : lddone_logic
        integer iter_var0;
        any_ld_valid = 1'b0;
        lddone_data  = '0;
        for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0 = iter_var0 + 1) begin : chk
          if (in_valid[1 + iter_var0]) begin : fire
            any_ld_valid = 1'b1;
            if (TAG_WIDTH > 0 && LD_COUNT > 1) begin : tag_fwd
              lddone_data[DATA_WIDTH +: TAG_WIDTH] = in_data[1 + iter_var0][DATA_WIDTH +: TAG_WIDTH];
            end
          end
        end
      end
      assign out_valid[DONE_IDX] = any_ld_valid;
      assign out_data[DONE_IDX]  = lddone_data;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Store path
  // -----------------------------------------------------------------------
  generate
    if (ST_COUNT > 0) begin : g_store
      genvar gsi;
      for (gsi = 0; gsi < ST_COUNT; gsi++) begin : g_st_port
        localparam int ADDR_IN = 1 + LD_COUNT + gsi;
        localparam int DATA_IN = 1 + LD_COUNT + ST_COUNT + gsi;
        localparam int DONE_IDX = LD_COUNT + 1;

        logic st_sync;
        assign st_sync = in_valid[ADDR_IN] && in_valid[DATA_IN];

        logic st_fire;
        assign st_fire = st_sync && out_ready[DONE_IDX];

        assign in_ready[ADDR_IN] = st_fire;
        assign in_ready[DATA_IN] = st_fire;

        logic [EXT_ADDR_WIDTH-1:0] st_addr;
        logic [SAFE_DW-1:0]        st_data_val;
        assign st_addr     = in_data[ADDR_IN][EXT_ADDR_WIDTH-1:0];
        assign st_data_val = in_data[DATA_IN][DATA_WIDTH-1:0];

        always_ff @(posedge clk) begin : write_mem
          if (st_fire) begin : do_write
            ext_mem[st_addr] <= st_data_val;
          end
        end
      end

      // Store done signal (carries tag when ST_COUNT > 1)
      begin : g_stdone
        localparam int DONE_IDX = LD_COUNT + 1;
        logic any_st_sync;
        logic [SAFE_PW-1:0] stdone_data;
        always_comb begin : stdone_logic
          integer iter_var0;
          any_st_sync = 1'b0;
          stdone_data = '0;
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : chk
            if (in_valid[1 + LD_COUNT + iter_var0] && in_valid[1 + LD_COUNT + ST_COUNT + iter_var0]) begin : fire
              any_st_sync = 1'b1;
              if (TAG_WIDTH > 0 && ST_COUNT > 1) begin : tag_fwd
                stdone_data[DATA_WIDTH +: TAG_WIDTH] = in_data[1 + LD_COUNT + iter_var0][DATA_WIDTH +: TAG_WIDTH];
              end
            end
          end
        end
        assign out_valid[DONE_IDX] = any_st_sync;
        assign out_data[DONE_IDX]  = stdone_data;
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Store deadlock timeout counter (per store port)
  // -----------------------------------------------------------------------
  localparam int DEADLOCK_TIMEOUT = 65535;
  logic [ST_COUNT > 0 ? ST_COUNT-1 : 0 : 0] st_deadlock_hit;

  generate
    if (ST_COUNT > 0) begin : g_deadlock
      genvar gdi;
      for (gdi = 0; gdi < ST_COUNT; gdi++) begin : g_dl_cnt
        localparam int ADDR_IN = 1 + LD_COUNT + gdi;
        localparam int DATA_IN = 1 + LD_COUNT + ST_COUNT + gdi;

        logic [15:0] dl_counter;
        logic        addr_only, data_only;
        assign addr_only = in_valid[ADDR_IN] && !in_valid[DATA_IN];
        assign data_only = !in_valid[ADDR_IN] && in_valid[DATA_IN];

        always_ff @(posedge clk or negedge rst_n) begin : dl_cnt
          if (!rst_n) begin : reset
            dl_counter <= 16'd0;
          end else begin : tick
            if (addr_only || data_only) begin : waiting
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
  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_check
    integer iter_var0;
    err_detect    = 1'b0;
    err_code_comb = 16'd0;

    // RT_MEMORY_TAG_OOB: tag >= count (only when TAG_WIDTH > 0 and multi-port)
    if (TAG_WIDTH > 0) begin : tag_oob_chk
      // Check load ports: tag must be < LD_COUNT (only if LD_COUNT > 1)
      if (LD_COUNT > 1) begin : ld_tag_chk
        for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0 = iter_var0 + 1) begin : per_ld
          if (in_valid[1 + iter_var0] &&
              ({{(32-TAG_WIDTH){1'b0}}, in_data[1 + iter_var0][DATA_WIDTH +: TAG_WIDTH]} >= 32'(LD_COUNT))) begin : oob
            err_detect    = 1'b1;
            err_code_comb = RT_MEMORY_TAG_OOB;
          end
        end
      end
      // Check store ports: tag must be < ST_COUNT (only if ST_COUNT > 1)
      if (ST_COUNT > 1) begin : st_tag_chk
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : per_st_addr
          if (in_valid[1 + LD_COUNT + iter_var0] &&
              ({{(32-TAG_WIDTH){1'b0}}, in_data[1 + LD_COUNT + iter_var0][DATA_WIDTH +: TAG_WIDTH]} >= 32'(ST_COUNT))) begin : oob
            err_detect    = 1'b1;
            err_code_comb = RT_MEMORY_TAG_OOB;
          end
        end
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : per_st_data
          if (in_valid[1 + LD_COUNT + ST_COUNT + iter_var0] &&
              ({{(32-TAG_WIDTH){1'b0}}, in_data[1 + LD_COUNT + ST_COUNT + iter_var0][DATA_WIDTH +: TAG_WIDTH]} >= 32'(ST_COUNT))) begin : oob
            err_detect    = 1'b1;
            err_code_comb = RT_MEMORY_TAG_OOB;
          end
        end
      end
    end

    // RT_MEMORY_STORE_DEADLOCK: store pairing timeout
    for (iter_var0 = 0; iter_var0 < (ST_COUNT > 0 ? ST_COUNT : 1); iter_var0 = iter_var0 + 1) begin : dl_chk
      if (st_deadlock_hit[iter_var0]) begin : deadlock
        err_detect    = 1'b1;
        err_code_comb = RT_MEMORY_STORE_DEADLOCK;
      end
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
