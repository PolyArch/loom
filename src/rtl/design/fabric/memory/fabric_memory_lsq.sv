// fabric_memory_lsq.sv -- Load-store queue for on-chip scratchpad memory.
//
// Accepts load and store requests on handshake ports, performs region
// lookup by tag to compute SRAM addresses, and arbitrates SRAM access
// between loads and stores.  Per-lane outstanding request tracking
// ensures at most one in-flight request per lane.
//
// Region table is configured via word-serial config bus.  Each region
// occupies 5 words: valid, start_lane, end_lane, addr_offset,
// elem_size_log2.  Tag-to-region matching: a request tag must fall
// within [start_lane, end_lane) of a valid region entry.
//
// Address computation:
//   byte_addr = addr_offset + (input_addr << elem_size_log2)
//   sram_word_addr = byte_addr >> SRAM_WORD_BYTE_LOG2
//
// The LSQ issues one SRAM operation per cycle (load or store),
// selected by round-robin arbitration when both are pending.

module fabric_memory_lsq #(
  parameter int unsigned DATA_WIDTH       = 32,
  parameter int unsigned TAG_WIDTH        = 4,
  parameter int unsigned NUM_REGION       = 4,
  parameter int unsigned SPAD_SIZE_BYTES  = 4096
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

  // --- Load data output ---
  output logic                    ld_data_valid,
  input  logic                    ld_data_ready,
  output logic [DATA_WIDTH-1:0]  ld_data_data,
  output logic [TAG_W-1:0]       ld_data_tag,

  // --- Load done output ---
  output logic                    ld_done_valid,
  input  logic                    ld_done_ready,
  output logic [TAG_W-1:0]       ld_done_tag,

  // --- Store done output ---
  output logic                    st_done_valid,
  input  logic                    st_done_ready,
  output logic [TAG_W-1:0]       st_done_tag,

  // --- SRAM interface ---
  output logic                    sram_wr_en,
  output logic [SRAM_ADDR_W-1:0] sram_wr_addr,
  output logic [DATA_WIDTH-1:0]  sram_wr_data,
  output logic                    sram_rd_en,
  output logic [SRAM_ADDR_W-1:0] sram_rd_addr,
  input  logic [DATA_WIDTH-1:0]  sram_rd_data,
  input  logic                    sram_rd_valid
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
  localparam int unsigned SRAM_WORD_BYTES    = DATA_WIDTH / 8;
  localparam int unsigned SRAM_WORD_BYTE_LOG2 = $clog2(SRAM_WORD_BYTES > 0 ? SRAM_WORD_BYTES : 1);
  localparam int unsigned SRAM_DEPTH         = SPAD_SIZE_BYTES / (SRAM_WORD_BYTES > 0 ? SRAM_WORD_BYTES : 1);
  localparam int unsigned SRAM_ADDR_W        = $clog2(SRAM_DEPTH > 1 ? SRAM_DEPTH : 2);
  localparam int unsigned MAX_LANES          = (1 << TAG_W);

  // ---------------------------------------------------------------
  // Region table (loaded via config bus)
  // ---------------------------------------------------------------
  logic        region_valid     [0:NUM_REGION-1];
  logic [31:0] region_start_lane[0:NUM_REGION-1];
  logic [31:0] region_end_lane  [0:NUM_REGION-1];
  logic [31:0] region_addr_off  [0:NUM_REGION-1];
  logic [31:0] region_elem_log2 [0:NUM_REGION-1];

  // Config word counter (5 words per region).
  localparam int unsigned CFG_WORD_COUNT = 5 * NUM_REGION;
  localparam int unsigned CFG_CNT_W = $clog2(CFG_WORD_COUNT > 1 ? CFG_WORD_COUNT + 1 : 2);
  logic [CFG_CNT_W-1:0] cfg_word_idx;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin : cfg_load
    integer iter_var0;
    if (!rst_n) begin : cfg_reset
      cfg_word_idx <= '0;
      for (iter_var0 = 0; iter_var0 < NUM_REGION; iter_var0 = iter_var0 + 1) begin : cfg_reset_region
        region_valid[iter_var0]      <= 1'b0;
        region_start_lane[iter_var0] <= '0;
        region_end_lane[iter_var0]   <= '0;
        region_addr_off[iter_var0]   <= '0;
        region_elem_log2[iter_var0]  <= '0;
      end : cfg_reset_region
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        // Compute which region and which field within the region.
        // Fields per region: [0]=valid, [1]=start_lane, [2]=end_lane,
        //                    [3]=addr_offset, [4]=elem_size_log2
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
  // Region lookup: find first valid region matching a given lane
  // ---------------------------------------------------------------
  // Resolved region output.
  typedef struct packed {
    logic        found;
    logic [31:0] byte_addr;
  } resolved_t;

  function automatic resolved_t resolve_region(
    input logic [TAG_W-1:0]      lane,
    input logic [DATA_WIDTH-1:0] logical_addr
  );
    resolved_t result;
    integer iter_var0;
    result.found     = 1'b0;
    result.byte_addr = '0;
    for (iter_var0 = 0; iter_var0 < NUM_REGION; iter_var0 = iter_var0 + 1) begin : resolve_scan
      if (!result.found &&
          region_valid[iter_var0] &&
          ({1'b0, lane} >= region_start_lane[iter_var0][TAG_W:0]) &&
          ({1'b0, lane} <  region_end_lane[iter_var0][TAG_W:0])) begin : resolve_hit
        result.found     = 1'b1;
        result.byte_addr = region_addr_off[iter_var0] +
                           (logical_addr << region_elem_log2[iter_var0][2:0]);
      end : resolve_hit
    end : resolve_scan
    return result;
  endfunction : resolve_region

  // ---------------------------------------------------------------
  // Per-lane outstanding tracking
  // ---------------------------------------------------------------
  logic [MAX_LANES-1:0] ld_outstanding;
  logic [MAX_LANES-1:0] st_outstanding;

  // Latched store halves (addr and data captured independently).
  logic                   st_addr_latched [0:MAX_LANES-1];
  logic [DATA_WIDTH-1:0]  st_addr_value   [0:MAX_LANES-1];
  logic                   st_data_latched [0:MAX_LANES-1];
  logic [DATA_WIDTH-1:0]  st_data_value   [0:MAX_LANES-1];

  // ---------------------------------------------------------------
  // Load request resolution
  // ---------------------------------------------------------------
  resolved_t ld_resolved;
  logic      ld_can_issue;

  always_comb begin : ld_resolve_logic
    ld_resolved = resolve_region(ld_addr_tag, ld_addr_data);
    ld_can_issue = ld_addr_valid &&
                   ld_resolved.found &&
                   !ld_outstanding[ld_addr_tag];
  end : ld_resolve_logic

  // ---------------------------------------------------------------
  // Store address/data capture logic
  // ---------------------------------------------------------------
  logic st_addr_can_capture;
  logic st_data_can_capture;

  always_comb begin : st_capture_logic
    st_addr_can_capture = st_addr_valid &&
                          !st_addr_latched[st_addr_tag] &&
                          resolve_region(st_addr_tag, st_addr_data).found;
    st_data_can_capture = st_data_valid &&
                          !st_data_latched[st_data_tag];
  end : st_capture_logic

  // ---------------------------------------------------------------
  // Store issue selection: find a lane with both addr and data latched
  // that has no outstanding request.
  // ---------------------------------------------------------------
  logic             st_can_issue;
  logic [TAG_W-1:0] st_issue_lane;
  /* verilator lint_off UNUSEDSIGNAL */
  resolved_t        st_resolved;
  /* verilator lint_on UNUSEDSIGNAL */

  always_comb begin : st_issue_select
    integer iter_var0;
    st_can_issue  = 1'b0;
    st_issue_lane = '0;
    st_resolved   = '0;
    for (iter_var0 = 0; iter_var0 < MAX_LANES; iter_var0 = iter_var0 + 1) begin : st_scan
      if (!st_can_issue &&
          st_addr_latched[iter_var0] &&
          st_data_latched[iter_var0] &&
          !st_outstanding[iter_var0]) begin : st_scan_hit
        automatic resolved_t r;
        r = resolve_region(iter_var0[TAG_W-1:0], st_addr_value[iter_var0]);
        if (r.found) begin : st_scan_resolved
          st_can_issue  = 1'b1;
          st_issue_lane = iter_var0[TAG_W-1:0];
          st_resolved   = r;
        end : st_scan_resolved
      end : st_scan_hit
    end : st_scan
  end : st_issue_select

  // ---------------------------------------------------------------
  // Arbitration: one SRAM operation per cycle (load or store)
  // Round-robin priority toggles each cycle an operation issues.
  // ---------------------------------------------------------------
  logic load_priority;

  always_ff @(posedge clk or negedge rst_n) begin : arb_priority_update
    if (!rst_n) begin : arb_reset
      load_priority <= 1'b1;
    end : arb_reset
    else begin : arb_toggle
      if (issue_load || issue_store) begin : arb_flip
        load_priority <= ~load_priority;
      end : arb_flip
    end : arb_toggle
  end : arb_priority_update

  logic issue_load;
  logic issue_store;

  always_comb begin : arb_decision
    issue_load  = 1'b0;
    issue_store = 1'b0;
    if (ld_can_issue && st_can_issue) begin : arb_both
      if (load_priority) begin : arb_pick_load
        issue_load = 1'b1;
      end : arb_pick_load
      else begin : arb_pick_store
        issue_store = 1'b1;
      end : arb_pick_store
    end : arb_both
    else if (ld_can_issue) begin : arb_load_only
      issue_load = 1'b1;
    end : arb_load_only
    else if (st_can_issue) begin : arb_store_only
      issue_store = 1'b1;
    end : arb_store_only
  end : arb_decision

  // ---------------------------------------------------------------
  // Input ready signals
  // ---------------------------------------------------------------
  assign ld_addr_ready = issue_load;
  assign st_addr_ready = st_addr_can_capture;
  assign st_data_ready = st_data_can_capture;

  // ---------------------------------------------------------------
  // SRAM interface drive
  // ---------------------------------------------------------------
  always_comb begin : sram_drive
    sram_wr_en   = 1'b0;
    sram_wr_addr = '0;
    sram_wr_data = '0;
    sram_rd_en   = 1'b0;
    sram_rd_addr = '0;
    if (issue_load) begin : sram_issue_read
      sram_rd_en   = 1'b1;
      sram_rd_addr = ld_resolved.byte_addr[SRAM_WORD_BYTE_LOG2 +: SRAM_ADDR_W];
    end : sram_issue_read
    if (issue_store) begin : sram_issue_write
      sram_wr_en   = 1'b1;
      sram_wr_addr = st_resolved.byte_addr[SRAM_WORD_BYTE_LOG2 +: SRAM_ADDR_W];
      sram_wr_data = st_data_value[st_issue_lane];
    end : sram_issue_write
  end : sram_drive

  // ---------------------------------------------------------------
  // Response pipeline registers
  // ---------------------------------------------------------------
  // Load response: SRAM has 1-cycle read latency.  Capture issued
  // load tag to pair with returning read data.
  logic             ld_resp_pending;
  logic [TAG_W-1:0] ld_resp_tag;

  // Store response: immediate (write completes same cycle).
  logic             st_resp_pending;
  logic [TAG_W-1:0] st_resp_tag;

  // Output holding registers (backpressure support).
  logic             ld_data_out_valid;
  logic [DATA_WIDTH-1:0] ld_data_out_data;
  logic [TAG_W-1:0] ld_data_out_tag;
  logic             ld_done_out_valid;
  logic [TAG_W-1:0] ld_done_out_tag;
  logic             st_done_out_valid;
  logic [TAG_W-1:0] st_done_out_tag;

  // ---------------------------------------------------------------
  // Sequential state update
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : state_update
    integer iter_var0;
    if (!rst_n) begin : state_reset
      ld_outstanding   <= '0;
      st_outstanding   <= '0;
      ld_resp_pending  <= 1'b0;
      ld_resp_tag      <= '0;
      st_resp_pending  <= 1'b0;
      st_resp_tag      <= '0;
      ld_data_out_valid <= 1'b0;
      ld_data_out_data  <= '0;
      ld_data_out_tag   <= '0;
      ld_done_out_valid <= 1'b0;
      ld_done_out_tag   <= '0;
      st_done_out_valid <= 1'b0;
      st_done_out_tag   <= '0;
      for (iter_var0 = 0; iter_var0 < MAX_LANES; iter_var0 = iter_var0 + 1) begin : state_reset_latch
        st_addr_latched[iter_var0] <= 1'b0;
        st_addr_value[iter_var0]   <= '0;
        st_data_latched[iter_var0] <= 1'b0;
        st_data_value[iter_var0]   <= '0;
      end : state_reset_latch
    end : state_reset
    else begin : state_op

      // -- Load issue: mark lane outstanding, begin SRAM read --
      if (issue_load) begin : state_ld_issue
        ld_outstanding[ld_addr_tag] <= 1'b1;
        ld_resp_pending             <= 1'b1;
        ld_resp_tag                 <= ld_addr_tag;
      end : state_ld_issue

      // -- Store addr/data capture --
      if (st_addr_valid && st_addr_ready) begin : state_st_addr_cap
        st_addr_latched[st_addr_tag] <= 1'b1;
        st_addr_value[st_addr_tag]   <= st_addr_data;
      end : state_st_addr_cap
      if (st_data_valid && st_data_ready) begin : state_st_data_cap
        st_data_latched[st_data_tag] <= 1'b1;
        st_data_value[st_data_tag]   <= st_data_data;
      end : state_st_data_cap

      // -- Store issue: mark lane outstanding, clear latches --
      if (issue_store) begin : state_st_issue
        st_outstanding[st_issue_lane]   <= 1'b1;
        st_addr_latched[st_issue_lane]  <= 1'b0;
        st_data_latched[st_issue_lane]  <= 1'b0;
        st_resp_pending                 <= 1'b1;
        st_resp_tag                     <= st_issue_lane;
      end : state_st_issue

      // -- Load response: SRAM read data returns --
      if (ld_resp_pending && sram_rd_valid) begin : state_ld_resp
        if (!ld_data_out_valid || (ld_data_valid && ld_data_ready)) begin : state_ld_resp_accept
          ld_data_out_valid <= 1'b1;
          ld_data_out_data  <= sram_rd_data;
          ld_data_out_tag   <= ld_resp_tag;
          ld_resp_pending   <= 1'b0;
          ld_outstanding[ld_resp_tag] <= 1'b0;
        end : state_ld_resp_accept
        // Load done fires alongside load data.
        if (!ld_done_out_valid || (ld_done_valid && ld_done_ready)) begin : state_ld_done_accept
          ld_done_out_valid <= 1'b1;
          ld_done_out_tag   <= ld_resp_tag;
        end : state_ld_done_accept
      end : state_ld_resp

      // -- Store response: immediate completion --
      if (st_resp_pending) begin : state_st_resp
        if (!st_done_out_valid || (st_done_valid && st_done_ready)) begin : state_st_done_accept
          st_done_out_valid <= 1'b1;
          st_done_out_tag   <= st_resp_tag;
          st_resp_pending   <= 1'b0;
          st_outstanding[st_resp_tag] <= 1'b0;
        end : state_st_done_accept
      end : state_st_resp

      // -- Drain output holding registers on transfer --
      if (ld_data_out_valid && ld_data_ready) begin : state_ld_data_drain
        if (!(ld_resp_pending && sram_rd_valid)) begin : state_ld_data_clear
          ld_data_out_valid <= 1'b0;
        end : state_ld_data_clear
      end : state_ld_data_drain
      if (ld_done_out_valid && ld_done_ready) begin : state_ld_done_drain
        if (!(ld_resp_pending && sram_rd_valid)) begin : state_ld_done_clear
          ld_done_out_valid <= 1'b0;
        end : state_ld_done_clear
      end : state_ld_done_drain
      if (st_done_out_valid && st_done_ready) begin : state_st_done_drain
        if (!st_resp_pending) begin : state_st_done_clear
          st_done_out_valid <= 1'b0;
        end : state_st_done_clear
      end : state_st_done_drain

    end : state_op
  end : state_update

  // ---------------------------------------------------------------
  // Output port drive
  // ---------------------------------------------------------------
  assign ld_data_valid = ld_data_out_valid;
  assign ld_data_data  = ld_data_out_data;
  assign ld_data_tag   = ld_data_out_tag;
  assign ld_done_valid = ld_done_out_valid;
  assign ld_done_tag   = ld_done_out_tag;
  assign st_done_valid = st_done_out_valid;
  assign st_done_tag   = st_done_out_tag;

endmodule : fabric_memory_lsq
