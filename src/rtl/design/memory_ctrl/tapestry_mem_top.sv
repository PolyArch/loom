// tapestry_mem_top.sv -- Top-level memory hierarchy wrapper.
//
// Instantiates per-core SPMs, per-core DMA engines, and the L2
// controller with banks.  Provides external DRAM AXI-MM master
// interface.

/* verilator lint_off UNUSEDPARAM */
module tapestry_mem_top
  import mem_ctrl_pkg::*;
  import fabric_axi_pkg::*;
#(
  parameter int unsigned NUM_CORES        = 4,
  parameter int unsigned SPM_SIZE_BYTES   = 4096,
  parameter int unsigned L2_SIZE_BYTES    = 262144,  // 256KB total
  parameter int unsigned L2_NUM_BANKS     = 4,
  parameter int unsigned DATA_WIDTH       = 32,
  parameter int unsigned INTERLEAVE_BYTES = 64,
  parameter int unsigned ACCESS_LATENCY   = 4,
  parameter int unsigned DMA_MAX_INFLIGHT = 4,
  parameter int unsigned AXI_ADDR_WIDTH   = 32,
  parameter int unsigned AXI_ID_WIDTH     = 4,
  // Derived parameters
  parameter int unsigned SPM_ADDR_WIDTH   = $clog2(SPM_SIZE_BYTES),
  parameter int unsigned BANK_SIZE_BYTES  = L2_SIZE_BYTES / L2_NUM_BANKS,
  parameter int unsigned BANK_ADDR_WIDTH  = $clog2(BANK_SIZE_BYTES)
)(
  input  logic                     clk,
  input  logic                     rst_n,

  // --- Per-core SPM interfaces ---
  input  logic [SPM_ADDR_WIDTH-1:0]  core_spm_addr      [NUM_CORES],
  input  logic [DATA_WIDTH-1:0]      core_spm_wdata     [NUM_CORES],
  input  logic                       core_spm_wr_en     [NUM_CORES],
  input  logic                       core_spm_req_valid [NUM_CORES],
  output logic                       core_spm_req_ready [NUM_CORES],
  output logic [DATA_WIDTH-1:0]      core_spm_rdata     [NUM_CORES],
  output logic                       core_spm_rdata_valid [NUM_CORES],

  // --- Per-core DMA command interfaces ---
  input  dma_cmd_t                   core_dma_cmd       [NUM_CORES],
  input  logic                       core_dma_cmd_valid [NUM_CORES],
  output logic                       core_dma_cmd_ready [NUM_CORES],
  output logic                       core_dma_busy      [NUM_CORES],
  output logic                       core_dma_done      [NUM_CORES],
  output logic [15:0]                core_dma_bytes_xferred [NUM_CORES],

  // --- L2 direct access interfaces (from NoC) ---
  input  mem_req_t                   l2_req             [NUM_CORES],
  input  logic                       l2_req_valid       [NUM_CORES],
  output logic                       l2_req_ready       [NUM_CORES],
  output mem_resp_t                  l2_resp            [NUM_CORES],
  output logic                       l2_resp_valid      [NUM_CORES],
  input  logic                       l2_resp_ready      [NUM_CORES],

  // --- External DRAM AXI-MM Master Interface ---
  // AXI inputs are currently unused (DRAM path tied off for initial bringup)
  /* verilator lint_off UNUSEDSIGNAL */
  // Write address channel
  output logic [AXI_ID_WIDTH-1:0]    m_axi_awid,
  output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
  output logic [7:0]                 m_axi_awlen,
  output logic [2:0]                 m_axi_awsize,
  output logic [1:0]                 m_axi_awburst,
  output logic                       m_axi_awvalid,
  input  logic                       m_axi_awready,

  // Write data channel
  output logic [DATA_WIDTH-1:0]      m_axi_wdata,
  output logic [(DATA_WIDTH/8)-1:0]  m_axi_wstrb,
  output logic                       m_axi_wlast,
  output logic                       m_axi_wvalid,
  input  logic                       m_axi_wready,

  // Write response channel
  input  logic [AXI_ID_WIDTH-1:0]    m_axi_bid,
  input  logic [1:0]                 m_axi_bresp,
  input  logic                       m_axi_bvalid,
  output logic                       m_axi_bready,

  // Read address channel
  output logic [AXI_ID_WIDTH-1:0]    m_axi_arid,
  output logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
  output logic [7:0]                 m_axi_arlen,
  output logic [2:0]                 m_axi_arsize,
  output logic [1:0]                 m_axi_arburst,
  output logic                       m_axi_arvalid,
  input  logic                       m_axi_arready,

  // Read data channel
  input  logic [AXI_ID_WIDTH-1:0]    m_axi_rid,
  input  logic [DATA_WIDTH-1:0]      m_axi_rdata,
  input  logic [1:0]                 m_axi_rresp,
  input  logic                       m_axi_rlast,
  input  logic                       m_axi_rvalid,
  output logic                       m_axi_rready
  /* verilator lint_on UNUSEDSIGNAL */
);

  // ---------------------------------------------------------------
  // Internal signals: DMA <-> SPM
  // ---------------------------------------------------------------
  logic [31:0]           dma_spm_addr      [NUM_CORES];
  logic [31:0]           dma_spm_wdata     [NUM_CORES];
  logic                  dma_spm_wr_en     [NUM_CORES];
  logic                  dma_spm_req_valid [NUM_CORES];
  logic                  dma_spm_req_ready [NUM_CORES];
  logic [31:0]           dma_spm_rdata     [NUM_CORES];
  logic                  dma_spm_rdata_valid [NUM_CORES];

  // Internal signals: DMA <-> L2 (external port of DMA)
  mem_req_t              dma_ext_req       [NUM_CORES];
  logic                  dma_ext_req_valid [NUM_CORES];
  logic                  dma_ext_req_ready [NUM_CORES];
  mem_resp_t             dma_ext_resp      [NUM_CORES];
  logic                  dma_ext_resp_valid [NUM_CORES];
  logic                  dma_ext_resp_ready [NUM_CORES];

  // ---------------------------------------------------------------
  // Combined L2 request/response (merge direct + DMA)
  // For simplicity, direct L2 access takes priority over DMA L2 access
  // ---------------------------------------------------------------
  mem_req_t              l2_merged_req       [NUM_CORES];
  logic                  l2_merged_req_valid [NUM_CORES];
  logic                  l2_merged_req_ready [NUM_CORES];
  mem_resp_t             l2_merged_resp       [NUM_CORES];
  logic                  l2_merged_resp_valid [NUM_CORES];
  logic                  l2_merged_resp_ready [NUM_CORES];

  // ---------------------------------------------------------------
  // Per-core SPM instantiation
  // ---------------------------------------------------------------
  generate
    genvar gi;
    for (gi = 0; gi < NUM_CORES; gi = gi + 1) begin : gen_spm
      tapestry_spm #(
        .SPM_SIZE_BYTES (SPM_SIZE_BYTES),
        .DATA_WIDTH     (DATA_WIDTH),
        .ADDR_WIDTH     (SPM_ADDR_WIDTH)
      ) u_spm (
        .clk             (clk),
        .rst_n           (rst_n),
        .core_addr       (core_spm_addr[gi]),
        .core_wdata      (core_spm_wdata[gi]),
        .core_wr_en      (core_spm_wr_en[gi]),
        .core_req_valid  (core_spm_req_valid[gi]),
        .core_req_ready  (core_spm_req_ready[gi]),
        .core_rdata      (core_spm_rdata[gi]),
        .core_rdata_valid(core_spm_rdata_valid[gi]),
        .dma_addr        (dma_spm_addr[gi][SPM_ADDR_WIDTH-1:0]),
        .dma_wdata       (dma_spm_wdata[gi][DATA_WIDTH-1:0]),
        .dma_wr_en       (dma_spm_wr_en[gi]),
        .dma_req_valid   (dma_spm_req_valid[gi]),
        .dma_req_ready   (dma_spm_req_ready[gi]),
        .dma_rdata       (dma_spm_rdata[gi][DATA_WIDTH-1:0]),
        .dma_rdata_valid (dma_spm_rdata_valid[gi])
      );
    end : gen_spm
  endgenerate

  // ---------------------------------------------------------------
  // Per-core DMA engine instantiation
  // ---------------------------------------------------------------
  generate
    genvar gd;
    for (gd = 0; gd < NUM_CORES; gd = gd + 1) begin : gen_dma
      tapestry_dma_engine #(
        .MAX_INFLIGHT (DMA_MAX_INFLIGHT),
        .DATA_WIDTH   (DATA_WIDTH)
      ) u_dma (
        .clk              (clk),
        .rst_n            (rst_n),
        .cmd              (core_dma_cmd[gd]),
        .cmd_valid        (core_dma_cmd_valid[gd]),
        .cmd_ready        (core_dma_cmd_ready[gd]),
        .busy             (core_dma_busy[gd]),
        .done             (core_dma_done[gd]),
        .bytes_transferred(core_dma_bytes_xferred[gd]),
        .spm_addr         (dma_spm_addr[gd]),
        .spm_wdata        (dma_spm_wdata[gd]),
        .spm_wr_en        (dma_spm_wr_en[gd]),
        .spm_req_valid    (dma_spm_req_valid[gd]),
        .spm_req_ready    (dma_spm_req_ready[gd]),
        .spm_rdata        (dma_spm_rdata[gd]),
        .spm_rdata_valid  (dma_spm_rdata_valid[gd]),
        .ext_req          (dma_ext_req[gd]),
        .ext_req_valid    (dma_ext_req_valid[gd]),
        .ext_req_ready    (dma_ext_req_ready[gd]),
        .ext_resp         (dma_ext_resp[gd]),
        .ext_resp_valid   (dma_ext_resp_valid[gd]),
        .ext_resp_ready   (dma_ext_resp_ready[gd])
      );
    end : gen_dma
  endgenerate

  // ---------------------------------------------------------------
  // L2 request merging: direct L2 access has priority over DMA
  // ---------------------------------------------------------------
  always_comb begin : l2_merge
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_CORES; iter_var0 = iter_var0 + 1) begin : merge_loop
      if (l2_req_valid[iter_var0]) begin : direct_priority
        // Direct L2 access from NoC takes priority
        l2_merged_req[iter_var0]       = l2_req[iter_var0];
        l2_merged_req_valid[iter_var0] = 1'b1;
        l2_req_ready[iter_var0]        = l2_merged_req_ready[iter_var0];
        dma_ext_req_ready[iter_var0]   = 1'b0;
      end : direct_priority
      else begin : dma_access
        // DMA L2 access when no direct access
        l2_merged_req[iter_var0]       = dma_ext_req[iter_var0];
        l2_merged_req_valid[iter_var0] = dma_ext_req_valid[iter_var0];
        l2_req_ready[iter_var0]        = 1'b0;
        dma_ext_req_ready[iter_var0]   = l2_merged_req_ready[iter_var0];
      end : dma_access

      // Response routing: check core_id to determine if response
      // goes to direct port or DMA port. For simplicity, route to
      // both and let the active requester consume it.
      l2_resp[iter_var0]            = l2_merged_resp[iter_var0];
      l2_resp_valid[iter_var0]      = l2_merged_resp_valid[iter_var0];
      dma_ext_resp[iter_var0]       = l2_merged_resp[iter_var0];
      dma_ext_resp_valid[iter_var0] = l2_merged_resp_valid[iter_var0];
      l2_merged_resp_ready[iter_var0] = l2_resp_ready[iter_var0] |
                                         dma_ext_resp_ready[iter_var0];
    end : merge_loop
  end : l2_merge

  // ---------------------------------------------------------------
  // L2 controller instantiation
  // ---------------------------------------------------------------
  tapestry_l2_ctrl #(
    .NUM_BANKS        (L2_NUM_BANKS),
    .BANK_SIZE_BYTES  (BANK_SIZE_BYTES),
    .DATA_WIDTH       (DATA_WIDTH),
    .INTERLEAVE_BYTES (INTERLEAVE_BYTES),
    .NUM_CORES        (NUM_CORES),
    .BANK_ADDR_WIDTH  (BANK_ADDR_WIDTH),
    .ACCESS_LATENCY   (ACCESS_LATENCY)
  ) u_l2_ctrl (
    .clk             (clk),
    .rst_n           (rst_n),
    .core_req        (l2_merged_req),
    .core_req_valid  (l2_merged_req_valid),
    .core_req_ready  (l2_merged_req_ready),
    .core_resp       (l2_merged_resp),
    .core_resp_valid (l2_merged_resp_valid),
    .core_resp_ready (l2_merged_resp_ready)
  );

  // ---------------------------------------------------------------
  // External DRAM AXI-MM interface
  //
  // For now, tie off the AXI interface with default (idle) values.
  // Full DRAM controller integration will be done when the external
  // memory path is connected.
  // ---------------------------------------------------------------
  assign m_axi_awid    = '0;
  assign m_axi_awaddr  = '0;
  assign m_axi_awlen   = '0;
  assign m_axi_awsize  = axsize_from_bytes(DATA_WIDTH / 8);
  assign m_axi_awburst = 2'b01;  // INCR
  assign m_axi_awvalid = 1'b0;

  assign m_axi_wdata   = '0;
  assign m_axi_wstrb   = '0;
  assign m_axi_wlast   = 1'b0;
  assign m_axi_wvalid  = 1'b0;

  assign m_axi_bready  = 1'b1;

  assign m_axi_arid    = '0;
  assign m_axi_araddr  = '0;
  assign m_axi_arlen   = '0;
  assign m_axi_arsize  = axsize_from_bytes(DATA_WIDTH / 8);
  assign m_axi_arburst = 2'b01;  // INCR
  assign m_axi_arvalid = 1'b0;

  assign m_axi_rready  = 1'b1;

endmodule : tapestry_mem_top
/* verilator lint_on UNUSEDPARAM */
