// fabric_extmemory_req.sv -- AXI request generator for external memory.
//
// Converts internal load and store requests into AXI4 AR/AW/W
// transactions.  Region-based address offset is applied using the
// configured region table.  elem_size_log2 is converted to AXI AxSIZE.
//
// One outstanding request per lane is enforced externally by the
// top-level module.  This module issues one AXI transaction per cycle.
// Loads and stores are arbitrated by a simple round-robin toggle.

module fabric_extmemory_req #(
  parameter int unsigned DATA_WIDTH  = 32,
  parameter int unsigned TAG_WIDTH   = 4,
  parameter int unsigned NUM_REGION  = 4,
  parameter int unsigned ADDR_WIDTH  = 32,
  parameter int unsigned AXI_ID_WIDTH = 4,
  // Derived parameters (used in port declarations)
  parameter int unsigned TAG_W      = (TAG_WIDTH > 0) ? TAG_WIDTH : 1,
  parameter int unsigned MAX_LANES  = (1 << TAG_W)
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Config port (word-serial, 5 * NUM_REGION words) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready,

  // --- Load address input ---
  input  logic                    ld_addr_valid,
  output logic                    ld_addr_ready,
  input  logic [DATA_WIDTH-1:0]  ld_addr_data,
  input  logic [TAG_W-1:0]       ld_addr_tag,

  // --- Store address input ---
  input  logic                    st_addr_valid,
  output logic                    st_addr_ready,
  input  logic [DATA_WIDTH-1:0]  st_addr_data,
  input  logic [TAG_W-1:0]       st_addr_tag,

  // --- Store data input ---
  input  logic                    st_data_valid,
  output logic                    st_data_ready,
  input  logic [DATA_WIDTH-1:0]  st_data_data,
  input  logic [TAG_W-1:0]       st_data_tag,

  // --- Per-lane outstanding tracking (from top) ---
  input  logic [MAX_LANES-1:0]   ld_outstanding,
  input  logic [MAX_LANES-1:0]   st_outstanding,

  // --- AXI Read Address Channel (AR) ---
  output logic [AXI_ID_WIDTH-1:0] m_axi_arid,
  output logic [ADDR_WIDTH-1:0]  m_axi_araddr,
  output logic [7:0]              m_axi_arlen,
  output logic [2:0]              m_axi_arsize,
  output logic [1:0]              m_axi_arburst,
  output logic                    m_axi_arvalid,
  input  logic                    m_axi_arready,

  // --- AXI Write Address Channel (AW) ---
  output logic [AXI_ID_WIDTH-1:0] m_axi_awid,
  output logic [ADDR_WIDTH-1:0]  m_axi_awaddr,
  output logic [7:0]              m_axi_awlen,
  output logic [2:0]              m_axi_awsize,
  output logic [1:0]              m_axi_awburst,
  output logic                    m_axi_awvalid,
  input  logic                    m_axi_awready,

  // --- AXI Write Data Channel (W) ---
  output logic [DATA_WIDTH-1:0]  m_axi_wdata,
  output logic [(DATA_WIDTH/8)-1:0] m_axi_wstrb,
  output logic                    m_axi_wlast,
  output logic                    m_axi_wvalid,
  input  logic                    m_axi_wready,

  // --- Issue notification to top for outstanding tracking ---
  output logic                    ld_issued,
  output logic [TAG_W-1:0]       ld_issued_lane,
  output logic                    st_issued,
  output logic [TAG_W-1:0]       st_issued_lane
);

  // ---------------------------------------------------------------
  // Region table (loaded via config bus)
  // ---------------------------------------------------------------
  logic        region_valid     [0:NUM_REGION-1];
  logic [31:0] region_start_lane[0:NUM_REGION-1];
  logic [31:0] region_end_lane  [0:NUM_REGION-1];
  logic [31:0] region_addr_off  [0:NUM_REGION-1];
  logic [31:0] region_elem_log2 [0:NUM_REGION-1];

  localparam int unsigned CFG_WORD_COUNT = 5 * NUM_REGION;
  localparam int unsigned CFG_CNT_W = $clog2(CFG_WORD_COUNT > 1 ? CFG_WORD_COUNT + 1 : 2);
  logic [CFG_CNT_W-1:0] cfg_word_idx;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin : cfg_load
    integer iter_var0;
    if (!rst_n) begin : cfg_reset
      cfg_word_idx <= '0;
      for (iter_var0 = 0; iter_var0 < NUM_REGION; iter_var0 = iter_var0 + 1) begin : cfg_reset_loop
        region_valid[iter_var0]      <= 1'b0;
        region_start_lane[iter_var0] <= '0;
        region_end_lane[iter_var0]   <= '0;
        region_addr_off[iter_var0]   <= '0;
        region_elem_log2[iter_var0]  <= '0;
      end : cfg_reset_loop
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        automatic int unsigned region_idx;
        automatic int unsigned field_idx;
        region_idx = int'(cfg_word_idx) / 5;
        field_idx  = int'(cfg_word_idx) % 5;
        if (region_idx < NUM_REGION) begin : cfg_store_field
          case (field_idx)
            0: region_valid[region_idx]      <= cfg_wdata[0];
            1: region_start_lane[region_idx] <= cfg_wdata;
            2: region_end_lane[region_idx]   <= cfg_wdata;
            3: region_addr_off[region_idx]   <= cfg_wdata;
            4: region_elem_log2[region_idx]  <= cfg_wdata;
            default: begin : cfg_field_unused
            end : cfg_field_unused
          endcase
        end : cfg_store_field
        cfg_word_idx <= cfg_word_idx + 1'b1;
      end : cfg_capture
    end : cfg_update
  end : cfg_load

  // ---------------------------------------------------------------
  // Region lookup function
  // ---------------------------------------------------------------
  typedef struct packed {
    logic        found;
    logic [ADDR_WIDTH-1:0] byte_addr;
    logic [2:0]  axsize;
  } resolved_t;

  function automatic resolved_t resolve_region(
    input logic [TAG_W-1:0]      lane,
    input logic [DATA_WIDTH-1:0] logical_addr
  );
    resolved_t result;
    integer iter_var0;
    result.found     = 1'b0;
    result.byte_addr = '0;
    result.axsize    = 3'b010; // default 4 bytes
    for (iter_var0 = 0; iter_var0 < NUM_REGION; iter_var0 = iter_var0 + 1) begin : resolve_scan
      if (!result.found &&
          region_valid[iter_var0] &&
          ({1'b0, lane} >= region_start_lane[iter_var0][TAG_W:0]) &&
          ({1'b0, lane} <  region_end_lane[iter_var0][TAG_W:0])) begin : resolve_hit
        result.found     = 1'b1;
        result.byte_addr = region_addr_off[iter_var0][ADDR_WIDTH-1:0] +
                           (logical_addr[ADDR_WIDTH-1:0] << region_elem_log2[iter_var0][2:0]);
        result.axsize    = region_elem_log2[iter_var0][2:0];
      end : resolve_hit
    end : resolve_scan
    return result;
  endfunction : resolve_region

  // ---------------------------------------------------------------
  // Latched store halves
  // ---------------------------------------------------------------
  logic                   st_addr_latched [0:MAX_LANES-1];
  logic [DATA_WIDTH-1:0]  st_addr_value   [0:MAX_LANES-1];
  logic                   st_data_latched [0:MAX_LANES-1];
  logic [DATA_WIDTH-1:0]  st_data_value   [0:MAX_LANES-1];

  // ---------------------------------------------------------------
  // Load resolution
  // ---------------------------------------------------------------
  resolved_t ld_resolved;
  logic      ld_can_issue;

  always_comb begin : ld_resolve
    ld_resolved  = resolve_region(ld_addr_tag, ld_addr_data);
    ld_can_issue = ld_addr_valid &&
                   ld_resolved.found &&
                   !ld_outstanding[ld_addr_tag];
  end : ld_resolve

  // ---------------------------------------------------------------
  // Store capture eligibility
  // ---------------------------------------------------------------
  logic st_addr_can_capture;
  logic st_data_can_capture;

  always_comb begin : st_capture_check
    resolved_t region_tmp;
    region_tmp = resolve_region(st_addr_tag, st_addr_data);
    st_addr_can_capture = st_addr_valid &&
                          !st_addr_latched[st_addr_tag] &&
                          region_tmp.found;
    st_data_can_capture = st_data_valid &&
                          !st_data_latched[st_data_tag];
  end : st_capture_check

  // ---------------------------------------------------------------
  // Store issue selection
  // ---------------------------------------------------------------
  logic             st_can_issue;
  logic [TAG_W-1:0] st_issue_lane;
  /* verilator lint_off UNUSEDSIGNAL */
  resolved_t        st_issue_resolved;
  /* verilator lint_on UNUSEDSIGNAL */

  always_comb begin : st_issue_select
    integer iter_var0;
    st_can_issue      = 1'b0;
    st_issue_lane     = '0;
    st_issue_resolved = '0;
    for (iter_var0 = 0; iter_var0 < MAX_LANES; iter_var0 = iter_var0 + 1) begin : st_scan
      if (!st_can_issue &&
          st_addr_latched[iter_var0] &&
          st_data_latched[iter_var0] &&
          !st_outstanding[iter_var0]) begin : st_scan_hit
        automatic resolved_t r;
        r = resolve_region(iter_var0[TAG_W-1:0], st_addr_value[iter_var0]);
        if (r.found) begin : st_scan_found
          st_can_issue      = 1'b1;
          st_issue_lane     = iter_var0[TAG_W-1:0];
          st_issue_resolved = r;
        end : st_scan_found
      end : st_scan_hit
    end : st_scan
  end : st_issue_select

  // ---------------------------------------------------------------
  // Arbitration toggle
  // ---------------------------------------------------------------
  logic load_priority;
  logic do_issue_load;
  logic do_issue_store;

  always_ff @(posedge clk or negedge rst_n) begin : arb_update
    if (!rst_n) begin : arb_reset
      load_priority <= 1'b1;
    end : arb_reset
    else begin : arb_toggle
      if (do_issue_load || do_issue_store) begin : arb_flip
        load_priority <= ~load_priority;
      end : arb_flip
    end : arb_toggle
  end : arb_update

  always_comb begin : arb_decision
    do_issue_load  = 1'b0;
    do_issue_store = 1'b0;
    if (ld_can_issue && st_can_issue) begin : arb_both
      if (load_priority) begin : arb_load_first
        do_issue_load = 1'b1;
      end : arb_load_first
      else begin : arb_store_first
        do_issue_store = 1'b1;
      end : arb_store_first
    end : arb_both
    else if (ld_can_issue) begin : arb_load_only
      do_issue_load = 1'b1;
    end : arb_load_only
    else if (st_can_issue) begin : arb_store_only
      do_issue_store = 1'b1;
    end : arb_store_only
  end : arb_decision

  // ---------------------------------------------------------------
  // Input ready
  // ---------------------------------------------------------------
  assign ld_addr_ready = do_issue_load && m_axi_arready;
  assign st_addr_ready = st_addr_can_capture;
  assign st_data_ready = st_data_can_capture;

  // ---------------------------------------------------------------
  // AXI AR channel (load request)
  // ---------------------------------------------------------------
  assign m_axi_arvalid = do_issue_load;
  assign m_axi_araddr  = ld_resolved.byte_addr;
  assign m_axi_arsize  = ld_resolved.axsize;
  assign m_axi_arlen   = 8'h00; // single beat
  assign m_axi_arburst = 2'b01; // INCR
  assign m_axi_arid    = {{(AXI_ID_WIDTH - TAG_W){1'b0}}, ld_addr_tag};

  // ---------------------------------------------------------------
  // AXI AW/W channels (store request)
  // ---------------------------------------------------------------
  assign m_axi_awvalid = do_issue_store;
  assign m_axi_awaddr  = st_issue_resolved.byte_addr;
  assign m_axi_awsize  = st_issue_resolved.axsize;
  assign m_axi_awlen   = 8'h00; // single beat
  assign m_axi_awburst = 2'b01; // INCR
  assign m_axi_awid    = {{(AXI_ID_WIDTH - TAG_W){1'b0}}, st_issue_lane};

  assign m_axi_wvalid  = do_issue_store;
  assign m_axi_wdata   = st_data_value[st_issue_lane];
  assign m_axi_wstrb   = {(DATA_WIDTH/8){1'b1}}; // full strobe
  assign m_axi_wlast   = 1'b1; // single beat

  // ---------------------------------------------------------------
  // Issue notifications
  // ---------------------------------------------------------------
  assign ld_issued      = do_issue_load && m_axi_arready;
  assign ld_issued_lane = ld_addr_tag;
  assign st_issued      = do_issue_store && m_axi_awready && m_axi_wready;
  assign st_issued_lane = st_issue_lane;

  // ---------------------------------------------------------------
  // Latched store state
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : st_latch_update
    integer iter_var0;
    if (!rst_n) begin : st_latch_reset
      for (iter_var0 = 0; iter_var0 < MAX_LANES; iter_var0 = iter_var0 + 1) begin : st_latch_reset_loop
        st_addr_latched[iter_var0] <= 1'b0;
        st_addr_value[iter_var0]   <= '0;
        st_data_latched[iter_var0] <= 1'b0;
        st_data_value[iter_var0]   <= '0;
      end : st_latch_reset_loop
    end : st_latch_reset
    else begin : st_latch_op
      // Capture store address.
      if (st_addr_valid && st_addr_ready) begin : st_latch_addr_cap
        st_addr_latched[st_addr_tag] <= 1'b1;
        st_addr_value[st_addr_tag]   <= st_addr_data;
      end : st_latch_addr_cap
      // Capture store data.
      if (st_data_valid && st_data_ready) begin : st_latch_data_cap
        st_data_latched[st_data_tag] <= 1'b1;
        st_data_value[st_data_tag]   <= st_data_data;
      end : st_latch_data_cap
      // Clear on issue.
      if (st_issued) begin : st_latch_clear
        st_addr_latched[st_issued_lane] <= 1'b0;
        st_data_latched[st_issued_lane] <= 1'b0;
      end : st_latch_clear
    end : st_latch_op
  end : st_latch_update

endmodule : fabric_extmemory_req
