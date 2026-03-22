// fabric_memory.sv -- On-chip scratchpad memory top module.
//
// Provides tagged load/store access to an on-chip SRAM scratchpad.
// Multiple logical lanes (identified by tag) share a single physical
// SRAM through the internal load-store queue (fabric_memory_lsq).
//
// Parameters:
//   LD_COUNT         -- Number of logical load lanes (0 = no loads)
//   ST_COUNT         -- Number of logical store lanes (0 = no stores)
//   DATA_WIDTH       -- Data payload width in bits
//   TAG_WIDTH        -- Tag width for lane identification (0 = untagged)
//   LSQ_DEPTH        -- Depth of load-store queue (reserved for future)
//   NUM_REGION       -- Number of region table entries
//   IS_PRIVATE       -- When 0, expose AXI-MM slave for external access
//   SPAD_SIZE_BYTES  -- Scratchpad size in bytes
//   ADDR_WIDTH       -- Address width for AXI slave port
//
// Config: region table (NUM_REGION entries, each 5 words:
//   valid, start_lane, end_lane, addr_offset, elem_size_log2)
//
// When !IS_PRIVATE: an AXI-MM slave port allows external agents to
// read/write the scratchpad contents.

module fabric_memory #(
  /* verilator lint_off UNUSEDPARAM */
  parameter int unsigned LD_COUNT        = 1,
  parameter int unsigned ST_COUNT        = 1,
  /* verilator lint_on UNUSEDPARAM */
  parameter int unsigned DATA_WIDTH      = 32,
  parameter int unsigned TAG_WIDTH       = 0,
  /* verilator lint_off UNUSEDPARAM */
  parameter int unsigned LSQ_DEPTH       = 4,
  /* verilator lint_on UNUSEDPARAM */
  parameter int unsigned NUM_REGION      = 1,
  parameter bit          IS_PRIVATE      = 1'b1,
  parameter int unsigned SPAD_SIZE_BYTES = 4096,
  parameter int unsigned ADDR_WIDTH      = 32,
  // Derived: effective tag width (at least 1 bit to avoid zero-width ports)
  parameter int unsigned TAG_W           = (TAG_WIDTH > 0) ? TAG_WIDTH : 1
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Config port (word-serial) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready,

  // --- Load address input ---
  input  logic                    load_addr_valid,
  output logic                    load_addr_ready,
  input  logic [DATA_WIDTH-1:0]  load_addr_data,
  input  logic [TAG_W-1:0]       load_addr_tag,

  // --- Store address input ---
  input  logic                    store_addr_valid,
  output logic                    store_addr_ready,
  input  logic [DATA_WIDTH-1:0]  store_addr_data,
  input  logic [TAG_W-1:0]       store_addr_tag,

  // --- Store data input ---
  input  logic                    store_data_valid,
  output logic                    store_data_ready,
  input  logic [DATA_WIDTH-1:0]  store_data_data,
  input  logic [TAG_W-1:0]       store_data_tag,

  // --- Load data output ---
  output logic                    load_data_valid,
  input  logic                    load_data_ready,
  output logic [DATA_WIDTH-1:0]  load_data_data,
  output logic [TAG_W-1:0]       load_data_tag,

  // --- Load done output ---
  output logic                    load_done_valid,
  input  logic                    load_done_ready,
  output logic [TAG_W-1:0]       load_done_tag,

  // --- Store done output ---
  output logic                    store_done_valid,
  input  logic                    store_done_ready,
  output logic [TAG_W-1:0]       store_done_tag,

  // --- AXI-MM slave port (active only when !IS_PRIVATE) ---
  // When IS_PRIVATE=1, these are tied off and unused.
  /* verilator lint_off UNUSEDSIGNAL */
  // Write address channel
  input  logic [ADDR_WIDTH-1:0]  s_axi_awaddr,
  input  logic                    s_axi_awvalid,
  output logic                    s_axi_awready,
  // Write data channel
  input  logic [DATA_WIDTH-1:0]  s_axi_wdata,
  input  logic [(DATA_WIDTH/8)-1:0] s_axi_wstrb,
  input  logic                    s_axi_wvalid,
  output logic                    s_axi_wready,
  // Write response channel
  output logic [1:0]              s_axi_bresp,
  output logic                    s_axi_bvalid,
  input  logic                    s_axi_bready,
  // Read address channel
  input  logic [ADDR_WIDTH-1:0]  s_axi_araddr,
  input  logic                    s_axi_arvalid,
  output logic                    s_axi_arready,
  // Read data channel
  output logic [DATA_WIDTH-1:0]  s_axi_rdata,
  output logic [1:0]              s_axi_rresp,
  output logic                    s_axi_rvalid,
  input  logic                    s_axi_rready
  /* verilator lint_on UNUSEDSIGNAL */
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned SRAM_WORD_BYTES     = DATA_WIDTH / 8;
  localparam int unsigned SRAM_WORD_BYTE_LOG2 = $clog2(SRAM_WORD_BYTES > 0 ? SRAM_WORD_BYTES : 1);
  localparam int unsigned SRAM_DEPTH          = SPAD_SIZE_BYTES / (SRAM_WORD_BYTES > 0 ? SRAM_WORD_BYTES : 1);
  localparam int unsigned SRAM_ADDR_W         = $clog2(SRAM_DEPTH > 1 ? SRAM_DEPTH : 2);

  // ---------------------------------------------------------------
  // SRAM instance
  // ---------------------------------------------------------------
  logic                    sram_wr_en;
  logic [SRAM_ADDR_W-1:0] sram_wr_addr;
  logic [DATA_WIDTH-1:0]  sram_wr_data;
  logic                    sram_rd_en;
  logic [SRAM_ADDR_W-1:0] sram_rd_addr;
  logic [DATA_WIDTH-1:0]  sram_rd_data;
  logic                    sram_rd_valid;

  // Internal LSQ SRAM signals (before AXI arbitration).
  logic                    lsq_sram_wr_en;
  logic [SRAM_ADDR_W-1:0] lsq_sram_wr_addr;
  logic [DATA_WIDTH-1:0]  lsq_sram_wr_data;
  logic                    lsq_sram_rd_en;
  logic [SRAM_ADDR_W-1:0] lsq_sram_rd_addr;

  fabric_memory_sram #(
    .DEPTH (SRAM_DEPTH),
    .WIDTH (DATA_WIDTH)
  ) u_sram (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (sram_wr_en),
    .wr_addr (sram_wr_addr),
    .wr_data (sram_wr_data),
    .rd_en   (sram_rd_en),
    .rd_addr (sram_rd_addr),
    .rd_data (sram_rd_data),
    .rd_valid(sram_rd_valid)
  );

  // ---------------------------------------------------------------
  // Load-Store Queue instance
  // ---------------------------------------------------------------
  fabric_memory_lsq #(
    .DATA_WIDTH      (DATA_WIDTH),
    .TAG_WIDTH       (TAG_WIDTH),
    .NUM_REGION      (NUM_REGION),
    .SPAD_SIZE_BYTES (SPAD_SIZE_BYTES)
  ) u_lsq (
    .clk            (clk),
    .rst_n          (rst_n),
    .cfg_valid      (cfg_valid),
    .cfg_wdata      (cfg_wdata),
    .cfg_ready      (cfg_ready),
    .ld_addr_valid  (load_addr_valid),
    .ld_addr_ready  (load_addr_ready),
    .ld_addr_data   (load_addr_data),
    .ld_addr_tag    (load_addr_tag),
    .st_addr_valid  (store_addr_valid),
    .st_addr_ready  (store_addr_ready),
    .st_addr_data   (store_addr_data),
    .st_addr_tag    (store_addr_tag),
    .st_data_valid  (store_data_valid),
    .st_data_ready  (store_data_ready),
    .st_data_data   (store_data_data),
    .st_data_tag    (store_data_tag),
    .ld_data_valid  (load_data_valid),
    .ld_data_ready  (load_data_ready),
    .ld_data_data   (load_data_data),
    .ld_data_tag    (load_data_tag),
    .ld_done_valid  (load_done_valid),
    .ld_done_ready  (load_done_ready),
    .ld_done_tag    (load_done_tag),
    .st_done_valid  (store_done_valid),
    .st_done_ready  (store_done_ready),
    .st_done_tag    (store_done_tag),
    .sram_wr_en     (lsq_sram_wr_en),
    .sram_wr_addr   (lsq_sram_wr_addr),
    .sram_wr_data   (lsq_sram_wr_data),
    .sram_rd_en     (lsq_sram_rd_en),
    .sram_rd_addr   (lsq_sram_rd_addr),
    .sram_rd_data   (sram_rd_data),
    .sram_rd_valid  (sram_rd_valid)
  );

  // ---------------------------------------------------------------
  // AXI slave port for external access (when !IS_PRIVATE)
  // ---------------------------------------------------------------
  generate
    if (!IS_PRIVATE) begin : gen_axi_slave

      // AXI slave state machine.
      // Simple single-beat read/write (no burst support -- single
      // word transactions only for scratchpad debug/init access).
      logic                    axi_wr_pending;
      logic [SRAM_ADDR_W-1:0] axi_wr_addr;
      logic [DATA_WIDTH-1:0]  axi_wr_data;
      logic                    axi_rd_pending;
      logic [SRAM_ADDR_W-1:0] axi_rd_addr;
      logic                    axi_rd_resp_valid;
      logic [DATA_WIDTH-1:0]  axi_rd_resp_data;
      logic                    axi_wr_resp_valid;

      // Accept AW/W together for simplicity.
      assign s_axi_awready = !axi_wr_pending && !axi_wr_resp_valid;
      assign s_axi_wready  = !axi_wr_pending && !axi_wr_resp_valid;
      assign s_axi_arready = !axi_rd_pending && !axi_rd_resp_valid;

      // Write response.
      assign s_axi_bvalid = axi_wr_resp_valid;
      assign s_axi_bresp  = 2'b00; // OKAY

      // Read response.
      assign s_axi_rvalid = axi_rd_resp_valid;
      assign s_axi_rdata  = axi_rd_resp_data;
      assign s_axi_rresp  = 2'b00; // OKAY

      always_ff @(posedge clk or negedge rst_n) begin : axi_slave_fsm
        if (!rst_n) begin : axi_slave_reset
          axi_wr_pending    <= 1'b0;
          axi_wr_addr       <= '0;
          axi_wr_data       <= '0;
          axi_rd_pending    <= 1'b0;
          axi_rd_addr       <= '0;
          axi_rd_resp_valid <= 1'b0;
          axi_rd_resp_data  <= '0;
          axi_wr_resp_valid <= 1'b0;
        end : axi_slave_reset
        else begin : axi_slave_op

          // -- Accept write request --
          if (s_axi_awvalid && s_axi_awready &&
              s_axi_wvalid && s_axi_wready) begin : axi_wr_accept
            axi_wr_pending <= 1'b1;
            axi_wr_addr    <= s_axi_awaddr[SRAM_WORD_BYTE_LOG2 +: SRAM_ADDR_W];
            axi_wr_data    <= s_axi_wdata;
          end : axi_wr_accept

          // -- Execute write (1-cycle) --
          if (axi_wr_pending) begin : axi_wr_exec
            axi_wr_pending    <= 1'b0;
            axi_wr_resp_valid <= 1'b1;
          end : axi_wr_exec

          // -- Write response handshake --
          if (axi_wr_resp_valid && s_axi_bready) begin : axi_wr_resp_drain
            axi_wr_resp_valid <= 1'b0;
          end : axi_wr_resp_drain

          // -- Accept read request --
          if (s_axi_arvalid && s_axi_arready) begin : axi_rd_accept
            axi_rd_pending <= 1'b1;
            axi_rd_addr    <= s_axi_araddr[SRAM_WORD_BYTE_LOG2 +: SRAM_ADDR_W];
          end : axi_rd_accept

          // -- Read response arrives from SRAM --
          if (axi_rd_pending && sram_rd_valid) begin : axi_rd_complete
            axi_rd_pending    <= 1'b0;
            axi_rd_resp_valid <= 1'b1;
            axi_rd_resp_data  <= sram_rd_data;
          end : axi_rd_complete

          // -- Read response handshake --
          if (axi_rd_resp_valid && s_axi_rready) begin : axi_rd_resp_drain
            axi_rd_resp_valid <= 1'b0;
          end : axi_rd_resp_drain

        end : axi_slave_op
      end : axi_slave_fsm

      // SRAM mux: LSQ has priority; AXI accesses when LSQ is idle.
      // In practice, AXI access should only happen during quiescent
      // mode (no active dataflow), so contention is not expected.
      always_comb begin : sram_mux
        if (lsq_sram_wr_en || lsq_sram_rd_en) begin : sram_mux_lsq
          sram_wr_en   = lsq_sram_wr_en;
          sram_wr_addr = lsq_sram_wr_addr;
          sram_wr_data = lsq_sram_wr_data;
          sram_rd_en   = lsq_sram_rd_en;
          sram_rd_addr = lsq_sram_rd_addr;
        end : sram_mux_lsq
        else begin : sram_mux_axi
          sram_wr_en   = axi_wr_pending;
          sram_wr_addr = axi_wr_addr;
          sram_wr_data = axi_wr_data;
          sram_rd_en   = axi_rd_pending;
          sram_rd_addr = axi_rd_addr;
        end : sram_mux_axi
      end : sram_mux

    end : gen_axi_slave
    else begin : gen_private

      // No AXI slave -- SRAM connected directly to LSQ.
      assign sram_wr_en   = lsq_sram_wr_en;
      assign sram_wr_addr = lsq_sram_wr_addr;
      assign sram_wr_data = lsq_sram_wr_data;
      assign sram_rd_en   = lsq_sram_rd_en;
      assign sram_rd_addr = lsq_sram_rd_addr;

      // Tie off AXI slave outputs.
      assign s_axi_awready = 1'b0;
      assign s_axi_wready  = 1'b0;
      assign s_axi_bresp   = 2'b00;
      assign s_axi_bvalid  = 1'b0;
      assign s_axi_arready = 1'b0;
      assign s_axi_rdata   = '0;
      assign s_axi_rresp   = 2'b00;
      assign s_axi_rvalid  = 1'b0;

    end : gen_private
  endgenerate

endmodule : fabric_memory
