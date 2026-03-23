// tapestry_spm.sv -- Per-core scratchpad memory with DMA port.
//
// Dual-access SRAM: one core port (priority) and one DMA port.
// Core access always wins on conflict (same-cycle access to same
// resource).  Single-cycle read latency for the core port.
//
// The SRAM is behaviorally described and maps to SRAM macros via
// synthesis compile directives.

module tapestry_spm #(
  parameter int unsigned SPM_SIZE_BYTES = 4096,
  parameter int unsigned DATA_WIDTH     = 32,
  parameter int unsigned ADDR_WIDTH     = 12   // log2(SPM_SIZE_BYTES)
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Core access port (priority) ---
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [ADDR_WIDTH-1:0]  core_addr,
  input  logic [DATA_WIDTH-1:0]  core_wdata,
  input  logic                   core_wr_en,
  input  logic                   core_req_valid,
  output logic                   core_req_ready,
  output logic [DATA_WIDTH-1:0]  core_rdata,
  output logic                   core_rdata_valid,

  /* verilator lint_on UNUSEDSIGNAL */

  // --- DMA access port (lower priority) ---
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [ADDR_WIDTH-1:0]  dma_addr,
  input  logic [DATA_WIDTH-1:0]  dma_wdata,
  input  logic                   dma_wr_en,
  input  logic                   dma_req_valid,
  output logic                   dma_req_ready,
  /* verilator lint_on UNUSEDSIGNAL */
  output logic [DATA_WIDTH-1:0]  dma_rdata,
  output logic                   dma_rdata_valid
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned NUM_WORDS = SPM_SIZE_BYTES / (DATA_WIDTH / 8);
  localparam int unsigned WORD_AW   = $clog2(NUM_WORDS > 1 ? NUM_WORDS : 2);

  // ---------------------------------------------------------------
  // Storage array
  // ---------------------------------------------------------------
  (* ram_style = "auto" *)
  logic [DATA_WIDTH-1:0] mem [0:NUM_WORDS-1];

  // ---------------------------------------------------------------
  // Priority arbitration: core always has priority over DMA
  // ---------------------------------------------------------------

  // Core port is always ready.
  assign core_req_ready = 1'b1;

  // DMA port is ready only when core is not accessing.
  assign dma_req_ready = ~core_req_valid;

  // ---------------------------------------------------------------
  // Word-addressed index from byte address
  // ---------------------------------------------------------------
  logic [WORD_AW-1:0] core_word_addr;
  logic [WORD_AW-1:0] dma_word_addr;

  assign core_word_addr = core_addr[ADDR_WIDTH-1 -: WORD_AW];
  assign dma_word_addr  = dma_addr[ADDR_WIDTH-1 -: WORD_AW];

  // ---------------------------------------------------------------
  // Actual access select signals
  // ---------------------------------------------------------------
  logic core_access;
  logic dma_access;

  assign core_access = core_req_valid;
  assign dma_access  = dma_req_valid & ~core_req_valid;

  // ---------------------------------------------------------------
  // Write logic
  // ---------------------------------------------------------------
  always_ff @(posedge clk) begin : spm_write
    if (core_access && core_wr_en) begin : core_write
      mem[core_word_addr] <= core_wdata;
    end : core_write
    else if (dma_access && dma_wr_en) begin : dma_write
      mem[dma_word_addr] <= dma_wdata;
    end : dma_write
  end : spm_write

  // ---------------------------------------------------------------
  // Read logic -- single-cycle read latency (synchronous read)
  // ---------------------------------------------------------------
  logic [DATA_WIDTH-1:0] core_rdata_reg;
  logic                  core_rvalid_reg;
  logic [DATA_WIDTH-1:0] dma_rdata_reg;
  logic                  dma_rvalid_reg;

  always_ff @(posedge clk or negedge rst_n) begin : spm_read
    if (!rst_n) begin : spm_read_reset
      core_rdata_reg  <= '0;
      core_rvalid_reg <= 1'b0;
      dma_rdata_reg   <= '0;
      dma_rvalid_reg  <= 1'b0;
    end : spm_read_reset
    else begin : spm_read_active
      // Core read
      core_rvalid_reg <= core_access & ~core_wr_en;
      if (core_access && !core_wr_en) begin : core_read_capture
        core_rdata_reg <= mem[core_word_addr];
      end : core_read_capture

      // DMA read
      dma_rvalid_reg <= dma_access & ~dma_wr_en;
      if (dma_access && !dma_wr_en) begin : dma_read_capture
        dma_rdata_reg <= mem[dma_word_addr];
      end : dma_read_capture
    end : spm_read_active
  end : spm_read

  assign core_rdata       = core_rdata_reg;
  assign core_rdata_valid = core_rvalid_reg;
  assign dma_rdata        = dma_rdata_reg;
  assign dma_rdata_valid  = dma_rvalid_reg;

endmodule : tapestry_spm
