// fabric_extmemory.sv -- External memory top module.
//
// Provides tagged load/store access to external memory through an
// AXI4 master interface.  The request/response path is split into
// two sub-modules:
//   - fabric_extmemory_req:  converts internal requests to AXI AR/AW/W
//   - fabric_extmemory_resp: converts AXI R/B to internal responses
//
// Parameters match fabric_memory for port compatibility.  The key
// difference is that fabric_extmemory uses an AXI master (not SRAM)
// and the region addr_offset is expected to be patched by the host
// runtime before launch (mapper emits addr_offset=0 for extmemory).
//
// Config: same region table as fabric_memory (NUM_REGION entries,
// each 5 words: valid, start_lane, end_lane, addr_offset, elem_size_log2).

module fabric_extmemory #(
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
  parameter int unsigned ADDR_WIDTH      = 32,
  parameter int unsigned AXI_ID_WIDTH    = 4,
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

  // --- AXI-MM Master: Read Address Channel (AR) ---
  output logic [AXI_ID_WIDTH-1:0] m_axi_arid,
  output logic [ADDR_WIDTH-1:0]  m_axi_araddr,
  output logic [7:0]              m_axi_arlen,
  output logic [2:0]              m_axi_arsize,
  output logic [1:0]              m_axi_arburst,
  output logic                    m_axi_arvalid,
  input  logic                    m_axi_arready,

  // --- AXI-MM Master: Read Data Channel (R) ---
  input  logic [AXI_ID_WIDTH-1:0] m_axi_rid,
  input  logic [DATA_WIDTH-1:0]  m_axi_rdata,
  input  logic [1:0]              m_axi_rresp,
  input  logic                    m_axi_rlast,
  input  logic                    m_axi_rvalid,
  output logic                    m_axi_rready,

  // --- AXI-MM Master: Write Address Channel (AW) ---
  output logic [AXI_ID_WIDTH-1:0] m_axi_awid,
  output logic [ADDR_WIDTH-1:0]  m_axi_awaddr,
  output logic [7:0]              m_axi_awlen,
  output logic [2:0]              m_axi_awsize,
  output logic [1:0]              m_axi_awburst,
  output logic                    m_axi_awvalid,
  input  logic                    m_axi_awready,

  // --- AXI-MM Master: Write Data Channel (W) ---
  output logic [DATA_WIDTH-1:0]  m_axi_wdata,
  output logic [(DATA_WIDTH/8)-1:0] m_axi_wstrb,
  output logic                    m_axi_wlast,
  output logic                    m_axi_wvalid,
  input  logic                    m_axi_wready,

  // --- AXI-MM Master: Write Response Channel (B) ---
  input  logic [AXI_ID_WIDTH-1:0] m_axi_bid,
  input  logic [1:0]              m_axi_bresp,
  input  logic                    m_axi_bvalid,
  output logic                    m_axi_bready
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned MAX_LANES = (1 << TAG_W);

  // ---------------------------------------------------------------
  // Per-lane outstanding request tracking
  // ---------------------------------------------------------------
  logic [MAX_LANES-1:0] ld_outstanding;
  logic [MAX_LANES-1:0] st_outstanding;

  // Issue/completion signals from sub-modules.
  logic             req_ld_issued;
  logic [TAG_W-1:0] req_ld_issued_lane;
  logic             req_st_issued;
  logic [TAG_W-1:0] req_st_issued_lane;
  logic             resp_ld_completed;
  logic [TAG_W-1:0] resp_ld_completed_lane;
  logic             resp_st_completed;
  logic [TAG_W-1:0] resp_st_completed_lane;

  always_ff @(posedge clk or negedge rst_n) begin : outstanding_update
    if (!rst_n) begin : outstanding_reset
      ld_outstanding <= '0;
      st_outstanding <= '0;
    end : outstanding_reset
    else begin : outstanding_op

      // Load tracking.
      if (req_ld_issued && !resp_ld_completed) begin : ld_set
        ld_outstanding[req_ld_issued_lane] <= 1'b1;
      end : ld_set
      else if (!req_ld_issued && resp_ld_completed) begin : ld_clear
        ld_outstanding[resp_ld_completed_lane] <= 1'b0;
      end : ld_clear
      else if (req_ld_issued && resp_ld_completed) begin : ld_both
        // Simultaneous issue and completion (different lanes).
        ld_outstanding[req_ld_issued_lane]      <= 1'b1;
        ld_outstanding[resp_ld_completed_lane]  <= 1'b0;
      end : ld_both

      // Store tracking.
      if (req_st_issued && !resp_st_completed) begin : st_set
        st_outstanding[req_st_issued_lane] <= 1'b1;
      end : st_set
      else if (!req_st_issued && resp_st_completed) begin : st_clear
        st_outstanding[resp_st_completed_lane] <= 1'b0;
      end : st_clear
      else if (req_st_issued && resp_st_completed) begin : st_both
        st_outstanding[req_st_issued_lane]      <= 1'b1;
        st_outstanding[resp_st_completed_lane]  <= 1'b0;
      end : st_both

    end : outstanding_op
  end : outstanding_update

  // ---------------------------------------------------------------
  // Request generator
  // ---------------------------------------------------------------
  fabric_extmemory_req #(
    .DATA_WIDTH   (DATA_WIDTH),
    .TAG_WIDTH    (TAG_WIDTH),
    .NUM_REGION   (NUM_REGION),
    .ADDR_WIDTH   (ADDR_WIDTH),
    .AXI_ID_WIDTH (AXI_ID_WIDTH)
  ) u_req (
    .clk             (clk),
    .rst_n           (rst_n),
    .cfg_valid       (cfg_valid),
    .cfg_wdata       (cfg_wdata),
    .cfg_ready       (cfg_ready),
    .ld_addr_valid   (load_addr_valid),
    .ld_addr_ready   (load_addr_ready),
    .ld_addr_data    (load_addr_data),
    .ld_addr_tag     (load_addr_tag),
    .st_addr_valid   (store_addr_valid),
    .st_addr_ready   (store_addr_ready),
    .st_addr_data    (store_addr_data),
    .st_addr_tag     (store_addr_tag),
    .st_data_valid   (store_data_valid),
    .st_data_ready   (store_data_ready),
    .st_data_data    (store_data_data),
    .st_data_tag     (store_data_tag),
    .ld_outstanding  (ld_outstanding),
    .st_outstanding  (st_outstanding),
    .m_axi_arid      (m_axi_arid),
    .m_axi_araddr    (m_axi_araddr),
    .m_axi_arlen     (m_axi_arlen),
    .m_axi_arsize    (m_axi_arsize),
    .m_axi_arburst   (m_axi_arburst),
    .m_axi_arvalid   (m_axi_arvalid),
    .m_axi_arready   (m_axi_arready),
    .m_axi_awid      (m_axi_awid),
    .m_axi_awaddr    (m_axi_awaddr),
    .m_axi_awlen     (m_axi_awlen),
    .m_axi_awsize    (m_axi_awsize),
    .m_axi_awburst   (m_axi_awburst),
    .m_axi_awvalid   (m_axi_awvalid),
    .m_axi_awready   (m_axi_awready),
    .m_axi_wdata     (m_axi_wdata),
    .m_axi_wstrb     (m_axi_wstrb),
    .m_axi_wlast     (m_axi_wlast),
    .m_axi_wvalid    (m_axi_wvalid),
    .m_axi_wready    (m_axi_wready),
    .ld_issued       (req_ld_issued),
    .ld_issued_lane  (req_ld_issued_lane),
    .st_issued       (req_st_issued),
    .st_issued_lane  (req_st_issued_lane)
  );

  // ---------------------------------------------------------------
  // Response handler
  // ---------------------------------------------------------------
  fabric_extmemory_resp #(
    .DATA_WIDTH   (DATA_WIDTH),
    .TAG_WIDTH    (TAG_WIDTH),
    .AXI_ID_WIDTH (AXI_ID_WIDTH)
  ) u_resp (
    .clk              (clk),
    .rst_n            (rst_n),
    .m_axi_rid        (m_axi_rid),
    .m_axi_rdata      (m_axi_rdata),
    .m_axi_rresp      (m_axi_rresp),
    .m_axi_rlast      (m_axi_rlast),
    .m_axi_rvalid     (m_axi_rvalid),
    .m_axi_rready     (m_axi_rready),
    .m_axi_bid        (m_axi_bid),
    .m_axi_bresp      (m_axi_bresp),
    .m_axi_bvalid     (m_axi_bvalid),
    .m_axi_bready     (m_axi_bready),
    .ld_data_valid    (load_data_valid),
    .ld_data_ready    (load_data_ready),
    .ld_data_data     (load_data_data),
    .ld_data_tag      (load_data_tag),
    .ld_done_valid    (load_done_valid),
    .ld_done_ready    (load_done_ready),
    .ld_done_tag      (load_done_tag),
    .st_done_valid    (store_done_valid),
    .st_done_ready    (store_done_ready),
    .st_done_tag      (store_done_tag),
    .ld_completed     (resp_ld_completed),
    .ld_completed_lane(resp_ld_completed_lane),
    .st_completed     (resp_st_completed),
    .st_completed_lane(resp_st_completed_lane)
  );

endmodule : fabric_extmemory
