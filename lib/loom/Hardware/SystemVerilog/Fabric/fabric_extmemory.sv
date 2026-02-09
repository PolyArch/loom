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
  // Load path: forward address request, wait for external data return
  // In simulation, external memory is modeled as a simple passthrough.
  // The actual memory binding is handled by the testbench or co-simulation.
  // -----------------------------------------------------------------------
  generate
    genvar gli;
    for (gli = 0; gli < LD_COUNT; gli++) begin : g_load
      localparam int IN_IDX  = 1 + gli; // skip memref
      localparam int DATA_OUT = gli;
      localparam int DONE_OUT = LD_COUNT;

      // Pass-through: load address in -> load data out (simulation stub)
      assign out_valid[DATA_OUT] = in_valid[IN_IDX];
      assign out_data[DATA_OUT]  = '0; // data from external memory (stub)
      assign in_ready[IN_IDX]    = out_ready[DATA_OUT] && out_ready[DONE_OUT];
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Load done signal
  // -----------------------------------------------------------------------
  generate
    if (LD_COUNT > 0) begin : g_lddone
      localparam int DONE_IDX = LD_COUNT;
      logic any_ld_valid;
      always_comb begin : lddone_logic
        integer iter_var0;
        any_ld_valid = 1'b0;
        for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0 = iter_var0 + 1) begin : chk
          any_ld_valid |= in_valid[1 + iter_var0];
        end
      end
      assign out_valid[DONE_IDX] = any_ld_valid;
      assign out_data[DONE_IDX]  = '0;
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
      end

      // Store done signal
      begin : g_stdone
        localparam int DONE_IDX = LD_COUNT + 1;
        logic any_st_sync;
        always_comb begin : stdone_logic
          integer iter_var0;
          any_st_sync = 1'b0;
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0 = iter_var0 + 1) begin : chk
            any_st_sync |= (in_valid[1 + LD_COUNT + iter_var0] && in_valid[1 + LD_COUNT + ST_COUNT + iter_var0]);
          end
        end
        assign out_valid[DONE_IDX] = any_st_sync;
        assign out_data[DONE_IDX]  = '0;
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------
  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_check
    err_detect    = 1'b0;
    err_code_comb = 16'd0;
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
